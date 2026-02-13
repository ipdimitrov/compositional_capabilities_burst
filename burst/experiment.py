"""
Unified Burst Schedule Experiment
==================================
Question: does the *temporal arrangement* of training data affect how
quickly a model can unlearn that data?

All schedules see the SAME total number of target (B) samples.
Only WHEN those samples appear during training differs.

Schedules tested:
  uniform       - B mixed uniformly throughout (p_target per step)
  end_block     - all B concentrated at the end
  mid_block     - all B concentrated in the middle
  early_block   - all B concentrated at the start
  end_mixed     - B only in second half, mixed with A at 2*p rate
  bookend       - half B at start, half at end
  spread_K3     - B split into 3 evenly-spaced burst windows
  spread_K5     - B split into 5 evenly-spaced burst windows

Undo phase: train on shuffled-label data to actively disrupt learned
bijection mappings.  Main metric is undo AUC (lower = faster unlearning).

Paper reference: Ramesh et al. 2023 (arXiv:2311.12997)
  Original: 2L/120d/1H, batch=16, ~20k steps, 10 alphabets, depth 5,
            3 base bijections, 50 train compositions, 100k docs.
  Ours:     6L/384d/6H, batch=128, 15k+5k steps, 3 seeds, same data config
            but with 100k docs (matching paper) and 10k eval docs.
"""
import sys, os, time, pickle, json, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf
from tqdm import tqdm

from synthetic.init import set_seed
from net.nanogpt import nanoGPT
from net.runner import configure_optimizers, update_cosine_warmup_lr
from burst.data import build_function_pool, tag_tasks, generate_pool, BurstDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── configuration ──────────────────────────────────────────────────────
# Paper-exact config: arXiv:2311.12997 Appendix A.1
# Architecture: 12L/120d/12H (paper spec, not 2L which was their small variant)
# N=4 base bijections → 1024 possible compositions (4^5)
# 100 total compositions: 90 background (A) + 10 target (B)
# Batch size 512, AdamW β=(0.9,0.95), weight_decay=1e-3, grad_clip=1.0
# 100K training samples, cosine LR schedule 3e-4 → 6e-5
CFG = {
    # seeds - reduced to 1 for <1hr runtime
    "seed_base": 42,
    "n_seeds": 1,

    # data (paper exact: 10 alphabets, seq_len 6, depth 5, N=4 bijections, 100 compositions)
    "n_alphabets": 10,
    "seq_len": 6,
    "depth": 5,
    "n_functions": 4,  # N=4 base bijections (was 3)
    "n_train_compositions": 100,  # 100 total compositions (was 50)
    "n_target": 10,  # 10 target (B) tasks, 90 background (A) (was 5)
    "ndocuments": 100_000,
    "neval_documents": 10_000,

    # model (paper exact: 12L/120d/12H, ~1.5M params)
    "n_layer": 12,  # 12 layers (was 6)
    "n_embd": 120,  # 120 embedding dim (was 384)
    "n_head": 12,  # 12 heads (was 6)
    "vocab_size": 512,
    "context_size": 50,

    # optimizer (paper exact from Appendix A.1)
    "lr": 3e-4,
    "weight_decay": 1e-3,  # 1e-3 (was 0.1)
    "beta1": 0.9,
    "beta2": 0.95,  # 0.95 (paper spec)
    "grad_clip": 1.0,
    "warmup_iters": 200,
    "min_lr": 6e-5,  # 6e-5 (was 6e-6)

    # training (reduced for ~2hr total runtime: 4 schedules × ~30min each)
    "batch_size": 8192,  # 8192 for RTX 5090 32GB
    "total_steps": 5_000,  # 50% reduction for faster runtime
    "p_target": 0.10,
    "undo_steps": 1_500,  # 50% reduction (proportional to total_steps)
    "eval_every": 200,
    "unlearn_threshold": 0.25,
}

SCHEDULES = [
    "early_block", "mid_block",
]


# ── schedule logic ─────────────────────────────────────────────────────
def n_target_for_step(step, total_steps, schedule, p, batch_size):
    T = total_steps
    burst_len = max(int(p * T), 1)

    if schedule == "uniform":
        return int(np.random.binomial(batch_size, p))

    if schedule == "end_block":
        return batch_size if step >= T - burst_len else 0

    if schedule == "mid_block":
        mid = T // 2
        return batch_size if mid - burst_len // 2 <= step < mid + (burst_len - burst_len // 2) else 0

    if schedule == "early_block":
        return batch_size if step < burst_len else 0

    if schedule == "end_mixed":
        half = T // 2
        return int(np.random.binomial(batch_size, 2 * p)) if step >= half else 0

    if schedule == "bookend":
        window = max(burst_len // 2, 1)
        return batch_size if (step < window or step >= T - window) else 0

    if schedule == "spread_K3":
        K, w = 3, max(burst_len // 3, 1)
        cycle = T // K
        return batch_size if (step % cycle) >= cycle - w else 0

    if schedule == "spread_K5":
        K, w = 5, max(burst_len // 5, 1)
        cycle = T // K
        return batch_size if (step % cycle) >= cycle - w else 0

    if schedule == "undo":
        return 0

    raise ValueError(f"Unknown schedule: {schedule}")


# ── data ───────────────────────────────────────────────────────────────
def build_data(seed):
    from burst.config import BurstExperimentConfig

    cfg = BurstExperimentConfig(
        seed=seed,
        n_alphabets=CFG["n_alphabets"],
        seq_len=CFG["seq_len"],
        depth=CFG["depth"],
        n_functions=CFG["n_functions"],
        n_train_compositions=CFG["n_train_compositions"],
        ndocuments=CFG["ndocuments"],
        neval_documents=CFG["neval_documents"],
    )
    set_seed(seed)
    syn, composed_functions, info = build_function_pool(cfg)
    target_ids, bg_ids, fn_lookup = tag_tasks(
        info, composed_functions, n_target=CFG["n_target"]
    )

    n_docs = max(CFG["ndocuments"] // max(len(bg_ids), 1), 500)
    target_pool = generate_pool(syn, target_ids, fn_lookup, n_docs)
    bg_pool = generate_pool(syn, bg_ids, fn_lookup, n_docs)

    eval_target = generate_pool(syn, target_ids, fn_lookup, CFG["neval_documents"])
    eval_bg = generate_pool(
        syn, bg_ids[:5], fn_lookup, CFG["neval_documents"] // 5
    )
    eval_target_flat = np.concatenate(list(eval_target.values()))
    eval_bg_flat = np.concatenate(list(eval_bg.values()))
    eval_docs = {"target": eval_target_flat, "background": eval_bg_flat}

    sp_idx = syn.token_idx[" "]
    space_pos = int(np.where(eval_docs["target"][0] == sp_idx)[0][-1])

    task_examples = _collect_task_examples(
        syn, fn_lookup, target_ids[:2], bg_ids[:2]
    )
    return target_pool, bg_pool, eval_docs, space_pos, syn, task_examples


def _collect_task_examples(syn, fn_lookup, target_ids, bg_ids):
    examples = []
    for tid, is_bg in [(t, False) for t in target_ids] + [(t, True) for t in bg_ids]:
        fn_tuple = fn_lookup[tid]
        tok = syn.sample_token()
        outs = syn.stepbystep_outputs(tok, fn_tuple[2])
        ex = {
            "task_id": [int(x) for x in tid],
            "depth": int(sum(1 for x in tid if int(x) != 0)),
            "input_tokens": [syn.token[int(t)] for t in tok],
            "outputs": [[syn.token[int(t)] for t in o] for o in outs],
            "is_identity_steps": [bool(int(x) == 0) for x in tid],
        }
        if is_bg:
            ex["is_background"] = True
        examples.append(ex)
    return examples


# ── model / optimiser ──────────────────────────────────────────────────
def make_model():
    net_cfg = OmegaConf.create({
        "compile": False,
        "vocab_size": CFG["vocab_size"],
        "context_size": CFG["context_size"],
        "n_layer": CFG["n_layer"],
        "n_head": CFG["n_head"],
        "n_embd": CFG["n_embd"],
        "dropout": 0.0,
        "bias": False,
        "mlp": True,
    })
    net = nanoGPT(net_cfg)
    net.to(DEVICE)
    return net


def make_optimizer(net):
    optim_cfg = OmegaConf.create({
        "learning_rate": CFG["lr"],
        "weight_decay": CFG["weight_decay"],
        "beta1": CFG["beta1"],
        "beta2": CFG["beta2"],
        "grad_clip": CFG["grad_clip"],
        "decay_lr": True,
        "warmup_iters": CFG["warmup_iters"],
        "min_lr": CFG["min_lr"],
    })
    return configure_optimizers(net, optim_cfg), optim_cfg


# ── helpers ────────────────────────────────────────────────────────────
def eval_accuracy(net, docs_BL, space_pos):
    net.eval()
    ds = BurstDataset(docs_BL)
    loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for dat, tgt in loader:
            dat, tgt = dat.to(DEVICE), tgt.to(DEVICE)
            logits = net(dat)[:, space_pos:]
            preds = logits.argmax(-1)
            correct += (preds == tgt[:, space_pos:]).float().sum().item()
            total += tgt[:, space_pos:].numel()
    net.train()
    return correct / max(total, 1)


def snapshot_weights(net):
    return OrderedDict(
        (n, p.detach().cpu().clone()) for n, p in net.named_parameters()
    )


def weight_deltas(w0, w1):
    return {n: (w1[n] - w0[n]).norm().item() for n in w0}


def sample_batch(target_pool, bg_pool, n_target, batch_size):
    t_ids = list(target_pool.keys())
    b_ids = list(bg_pool.keys())
    n_bg = batch_size - n_target
    parts = []
    for _ in range(n_target):
        tid = t_ids[np.random.randint(len(t_ids))]
        parts.append(target_pool[tid][np.random.randint(len(target_pool[tid]))])
    for _ in range(n_bg):
        bid = b_ids[np.random.randint(len(b_ids))]
        parts.append(bg_pool[bid][np.random.randint(len(bg_pool[bid]))])
    batch = np.array(parts)
    return batch[np.random.permutation(len(batch))]


# ── single run ─────────────────────────────────────────────────────────
def run_one(schedule, seed, target_pool, bg_pool, eval_docs, space_pos):
    set_seed(seed)
    net = make_model()
    optimizer, optim_cfg = make_optimizer(net)

    # Mixed precision training (bf16)
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE == "cuda")
    use_amp = DEVICE == "cuda"

    T = CFG["total_steps"]
    U = CFG["undo_steps"]
    total_lr_steps = T + U
    bs = CFG["batch_size"]
    p = CFG["p_target"]
    ev = CFG["eval_every"]

    log = {
        "step": [], "loss": [], "acc_target": [], "acc_background": [],
        "phase": [], "n_target_in_batch": [], "weight_deltas": [],
    }

    w0 = snapshot_weights(net)
    net.train()
    it = 0

    # ── TRAIN phase ────────────────────────────────────────────────────
    pbar = tqdm(range(T), desc=f"  Training [{schedule}]", unit="step", ncols=100)
    for s in pbar:
        nt = n_target_for_step(s, T, schedule, p, bs)
        batch = sample_batch(target_pool, bg_pool, nt, bs)
        dat = torch.from_numpy(batch).long().to(DEVICE)
        inp, tgt = dat[:, :-1], dat[:, 1:]
        
        if s == 0:
            print(f"  Training on device: {inp.device}", flush=True)

        it, lr = update_cosine_warmup_lr(it, optim_cfg, optimizer, total_lr_steps)
        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
            logits = net(inp)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)
            )
        
        scaler.scale(loss).backward()
        if CFG["grad_clip"] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), CFG["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.2e}"})

        if s % ev == 0 or s == T - 1:
            log["step"].append(it)
            log["loss"].append(loss.item())
            log["acc_target"].append(
                eval_accuracy(net, eval_docs["target"], space_pos)
            )
            log["acc_background"].append(
                eval_accuracy(net, eval_docs["background"], space_pos)
            )
            log["phase"].append("train")
            log["n_target_in_batch"].append(nt)
            log["weight_deltas"].append(weight_deltas(w0, snapshot_weights(net)))
            net.train()

    w_train_end = snapshot_weights(net)
    pbar.close()

    # ── UNDO phase (shuffled-label training) ───────────────────────────
    pbar = tqdm(range(U), desc=f"  Unlearning [{schedule}]", unit="step", ncols=100)
    for s in pbar:
        batch = sample_batch(target_pool, bg_pool, 0, bs)
        dat = torch.from_numpy(batch).long().to(DEVICE)
        inp, tgt = dat[:, :-1], dat[:, 1:]

        tgt_shuffled = tgt.clone()
        for b_idx in range(tgt_shuffled.shape[0]):
            out_part = tgt_shuffled[b_idx, space_pos:]
            tgt_shuffled[b_idx, space_pos:] = out_part[torch.randperm(out_part.shape[0])]

        it, lr = update_cosine_warmup_lr(it, optim_cfg, optimizer, total_lr_steps)
        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
            logits = net(inp)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt_shuffled.reshape(-1)
            )
        
        scaler.scale(loss).backward()
        if CFG["grad_clip"] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), CFG["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.2e}"})

        if s % ev == 0 or s == U - 1:
            log["step"].append(it)
            log["loss"].append(loss.item())
            acc_t = eval_accuracy(net, eval_docs["target"], space_pos)
            log["acc_target"].append(acc_t)
            log["acc_background"].append(
                eval_accuracy(net, eval_docs["background"], space_pos)
            )
            log["phase"].append("undo")
            log["n_target_in_batch"].append(0)
            log["weight_deltas"].append(
                weight_deltas(w_train_end, snapshot_weights(net))
            )
            net.train()
    
    pbar.close()

    # ── compute summary metrics ────────────────────────────────────────
    threshold = CFG["unlearn_threshold"]
    train_end_acc = None
    unlearn_step = None
    undo_accs, undo_steps_list = [], []

    for i, ph in enumerate(log["phase"]):
        if ph == "train":
            train_end_acc = log["acc_target"][i]
        if ph == "undo":
            undo_accs.append(log["acc_target"][i])
            undo_steps_list.append(log["step"][i] - T)
            if unlearn_step is None and log["acc_target"][i] < threshold:
                unlearn_step = log["step"][i] - T

    undo_end_acc = undo_accs[-1] if undo_accs else train_end_acc
    undo_auc = float(np.trapz(undo_accs, undo_steps_list)) if len(undo_accs) > 1 else 0.0

    mlp_undo, attn_undo = 0.0, 0.0
    undo_wds = [wd for i, wd in enumerate(log["weight_deltas"]) if log["phase"][i] == "undo"]
    if undo_wds:
        last = undo_wds[-1]
        mlp_undo = sum(v for k, v in last.items() if "mlp" in k)
        attn_undo = sum(v for k, v in last.items() if "attn" in k)

    return {
        "schedule": schedule,
        "seed": seed,
        "log": log,
        "train_end_acc": train_end_acc,
        "undo_end_acc": undo_end_acc,
        "undo_auc": undo_auc,
        "unlearn_step": unlearn_step,
        "mlp_undo_delta": mlp_undo,
        "attn_undo_delta": attn_undo,
        "config": dict(CFG),
    }


# ── orchestrator ───────────────────────────────────────────────────────
def make_run_dir():
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = Path("data") / f"burst_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, tuple):
            return list(obj)
        return super().default(obj)


def run_all():
    run_dir = make_run_dir()
    print(f"Output: {run_dir}", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    seed_base = CFG["seed_base"]
    n_seeds = CFG["n_seeds"]

    target_pool, bg_pool, eval_docs, space_pos, syn, task_examples = build_data(
        seed_base
    )
    
    print(f"A:B split = {len(bg_pool)}:{len(target_pool)} (background:target tasks)", flush=True)
    print(f"Total compositions: {len(bg_pool) + len(target_pool)}", flush=True)

    with open(run_dir / "task_examples.json", "w") as f:
        json.dump(task_examples, f, indent=2, cls=NpEncoder)
    with open(run_dir / "config.json", "w") as f:
        json.dump(CFG, f, indent=2)

    all_results = []
    total = len(SCHEDULES) * n_seeds
    done = 0

    for sched in SCHEDULES:
        for si in range(n_seeds):
            seed = seed_base + si
            done += 1
            print(f"\n[{done}/{total}] {sched} seed={seed}", flush=True)
            t0 = time.time()
            result = run_one(
                sched, seed, target_pool, bg_pool, eval_docs, space_pos
            )
            elapsed = time.time() - t0
            print(
                f"  train_end={result['train_end_acc']:.4f}  "
                f"undo_end={result['undo_end_acc']:.4f}  "
                f"unlearn_step={result['unlearn_step']}  "
                f"auc={result['undo_auc']:.0f}  "
                f"mlp={result['mlp_undo_delta']:.3f}  "
                f"({elapsed:.0f}s)",
                flush=True,
            )
            all_results.append(result)

            with open(run_dir / f"{sched}_seed{seed}.pkl", "wb") as f:
                pickle.dump(result, f)

    with open(run_dir / "all_results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    print(f"\nAll done. Results in {run_dir}", flush=True)
    return run_dir, all_results, task_examples


if __name__ == "__main__":
    run_dir, all_results, task_examples = run_all()
    print(f"\nTo generate plots and report:")
    print(f"  python burst/plot_and_report.py {run_dir}")
