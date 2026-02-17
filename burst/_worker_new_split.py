"""Worker for depth-3 pure-bijection burst experiment.

Launched as a subprocess by experiment_new_split.py.
Each worker trains one model on one schedule and saves results.
"""
import sys, os, argparse, pickle, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict
from pathlib import Path
from omegaconf import OmegaConf

from synthetic.init import set_seed
from net.nanogpt import nanoGPT
from net.runner import configure_optimizers, update_cosine_warmup_lr
from burst.data import BurstDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_KEYS = ["acc_A_base", "acc_A_comp", "acc_A_heldout", "acc_B_comp", "acc_B_heldout"]

MIXED_SCHEDULES = {
    "end_mixed_50":  0.50,
    "end_mixed_75b": 0.75,
    "end_mixed_25b": 0.25,
}


def n_target_for_step(step, total_steps, schedule, p, batch_size):
    T = total_steps
    burst_len = max(int(p * T), 1)

    if schedule == "uniform":
        return int(np.random.binomial(batch_size, p))

    if schedule == "end_block":
        return batch_size if step >= T - burst_len else 0

    if schedule == "mid_block":
        mid = T // 2
        half = burst_len // 2
        return batch_size if mid - half <= step < mid + (burst_len - half) else 0

    if schedule in MIXED_SCHEDULES:
        frac = MIXED_SCHEDULES[schedule]
        window = min(int(burst_len / frac), T)
        return int(round(batch_size * frac)) if step >= T - window else 0

    if schedule == "ramp_up":
        max_frac = 0.20
        ramp_len = min(int(2 * burst_len / max_frac), T)
        if step >= T - ramp_len:
            progress = (step - (T - ramp_len)) / max(ramp_len - 1, 1)
            return int(round(batch_size * progress * max_frac))
        return 0

    if schedule == "undo":
        return 0

    raise ValueError(f"Unknown schedule: {schedule}")


def sample_batch(target_pool, bg_pool, n_target, batch_size):
    t_ids = list(target_pool.keys())
    b_ids = list(bg_pool.keys())
    parts = []
    for _ in range(n_target):
        tid = t_ids[np.random.randint(len(t_ids))]
        parts.append(target_pool[tid][np.random.randint(len(target_pool[tid]))])
    for _ in range(batch_size - n_target):
        bid = b_ids[np.random.randint(len(b_ids))]
        parts.append(bg_pool[bid][np.random.randint(len(bg_pool[bid]))])
    return np.array(parts)[np.random.permutation(batch_size)]


@torch.no_grad()
def eval_free_gen(net, docs_BL, prompt_len: int):
    if docs_BL.shape[0] <= 1 and docs_BL.sum() == 0:
        return 0.0
    net.eval()
    loader = torch.utils.data.DataLoader(
        BurstDataset(docs_BL), batch_size=256, shuffle=False)
    correct, total = 0, 0
    for dat, tgt in loader:
        dat, tgt = dat.to(DEVICE), tgt.to(DEVICE)
        inp = dat[:, :prompt_len]
        for _ in range(dat.shape[1] - prompt_len):
            nxt = net(inp)[:, -1, :].argmax(-1, keepdim=True)
            inp = torch.cat([inp, nxt], dim=1)
        gen = inp[:, prompt_len:]
        ref = tgt[:, prompt_len - 1:]
        ml = min(gen.shape[1], ref.shape[1])
        last6 = max(0, ml - 6)
        correct += (gen[:, last6:ml] == ref[:, last6:ml]).float().sum().item()
        total += ref[:, last6:ml].numel()
    net.train()
    return correct / max(total, 1)


def run(job, shared_data_path, run_dir, progress_dir):
    label, schedule, seed, cfg = job["label"], job["schedule"], job["seed"], job["cfg"]

    progress_file = Path(progress_dir) / f"{label}.txt"
    progress_file.write_text("0")

    with open(shared_data_path, "rb") as f:
        target_pool, bg_pool, eval_docs, prompt_len, _ = pickle.load(f)

    set_seed(seed)
    net = nanoGPT(OmegaConf.create({
        "compile": False, "vocab_size": cfg["vocab_size"],
        "context_size": cfg["context_size"],
        "n_layer": cfg["n_layer"], "n_head": cfg["n_head"],
        "n_embd": cfg["n_embd"], "dropout": 0.0, "bias": False, "mlp": True,
    })).to(DEVICE)

    optim_cfg = OmegaConf.create({
        "learning_rate": cfg["lr"], "weight_decay": cfg["weight_decay"],
        "beta1": cfg["beta1"], "beta2": cfg["beta2"],
        "grad_clip": cfg["grad_clip"], "decay_lr": True,
        "warmup_iters": cfg["warmup_iters"], "min_lr": cfg["min_lr"],
    })
    optimizer = configure_optimizers(net, optim_cfg)
    scaler = torch.amp.GradScaler('cuda', enabled=DEVICE == "cuda")

    T, U = cfg["total_steps"], cfg["undo_steps"]
    bs, p, ev = cfg["batch_size"], cfg["p_target"], cfg["eval_every"]

    log = {"step": [], "loss": [], "phase": []}
    for k in EVAL_KEYS:
        log[k] = []

    def do_eval(phase, loss_val):
        log["step"].append(it)
        log["loss"].append(loss_val)
        log["phase"].append(phase)
        for k in EVAL_KEYS:
            log[k].append(eval_free_gen(net, eval_docs[k.replace("acc_", "")], prompt_len))
        net.train()

    def train_step(batch_np):
        nonlocal it
        dat = torch.from_numpy(batch_np).long().to(DEVICE)
        inp, tgt = dat[:, :-1], dat[:, 1:]
        it, _ = update_cosine_warmup_lr(it, optim_cfg, optimizer, T + U)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            logits = net(inp)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        scaler.scale(loss).backward()
        if cfg["grad_clip"] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        return loss.item()

    net.train()
    it = 0

    for s in range(T):
        nt = n_target_for_step(s, T, schedule, p, bs)
        loss_val = train_step(sample_batch(target_pool, bg_pool, nt, bs))
        if s % ev == 0 or s == T - 1:
            do_eval("train", loss_val)
        if (s + 1) % 50 == 0:
            progress_file.write_text(str(s + 1))

    for s in range(U):
        loss_val = train_step(sample_batch(target_pool, bg_pool, 0, bs))
        if s % ev == 0 or s == U - 1:
            do_eval("undo", loss_val)
        if (s + 1) % 50 == 0:
            progress_file.write_text(str(T + s + 1))

    progress_file.write_text(str(T + U))

    undo_accs = [log["acc_B_comp"][i] for i, ph in enumerate(log["phase"]) if ph == "undo"]
    undo_steps = [log["step"][i] - T for i, ph in enumerate(log["phase"]) if ph == "undo"]
    train_accs = [log["acc_B_comp"][i] for i, ph in enumerate(log["phase"]) if ph == "train"]

    peak_B = train_accs[-1] if train_accs else 0
    undo_end_B = undo_accs[-1] if undo_accs else peak_B
    undo_auc = float(np.trapz(undo_accs, undo_steps)) if len(undo_accs) > 1 else 0.0

    half_life = U
    if peak_B > 1e-6:
        for acc_val, us in zip(undo_accs, undo_steps):
            if acc_val <= peak_B * 0.5:
                half_life = us
                break

    dropoff_abs = peak_B - undo_end_B
    dropoff_pct = (dropoff_abs / peak_B * 100) if peak_B > 1e-6 else 0.0

    result = {
        "schedule": schedule, "seed": seed,
        "n_b_seen": job.get("n_b_seen", 0),
        "label": label, "log": log, "config": dict(cfg),
        "train_end_B_comp": peak_B, "undo_end_B_comp": undo_end_B,
        "dropoff_abs": dropoff_abs, "dropoff_pct": dropoff_pct,
        "half_life": half_life, "undo_auc": undo_auc,
    }
    for k in EVAL_KEYS:
        for phase in ("train", "undo"):
            vals = [log[k][i] for i, ph in enumerate(log["phase"]) if ph == phase]
            result[f"{phase}_end_{k}"] = vals[-1] if vals else 0

    with open(Path(run_dir) / f"{label}.pkl", "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--progress-dir", required=True)
    args = parser.parse_args()

    with open(args.job_path, "rb") as f:
        job = pickle.load(f)
    run(job, args.data_path, args.run_dir, args.progress_dir)
