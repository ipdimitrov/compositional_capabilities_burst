"""Single-job worker process for parallel experiments."""
import sys, os, argparse, pickle, json, math, time
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
        return int(np.random.binomial(batch_size, 2 * p)) if step >= T // 2 else 0
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
    if schedule == "early_block_2x":
        return batch_size if step < 2 * burst_len else 0
    if schedule == "late_ramp":
        frac = step / max(T - 1, 1)
        p_step = p * 2 * frac
        return int(np.random.binomial(batch_size, min(p_step, 1.0)))
    if schedule == "cyclic":
        K, w = 4, max(burst_len // 4, 1)
        cycle = T // K
        return batch_size if (step % cycle) < w else 0
    if schedule == "front_heavy":
        return int(np.random.binomial(batch_size, 2 * p)) if step < T // 2 else 0
    if schedule == "undo":
        return 0
    raise ValueError(schedule)


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
    batch = np.array(parts)
    return batch[np.random.permutation(len(batch))]


def eval_accuracy(net, docs_BL, space_pos):
    net.eval()
    ds = BurstDataset(docs_BL)
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False)
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
    return OrderedDict((n, p.detach().cpu().clone()) for n, p in net.named_parameters())


def weight_deltas(w0, w1):
    return {n: (w1[n] - w0[n]).norm().item() for n in w0}


def run(job, shared_data_path, run_dir, progress_dir):
    label = job["label"]
    schedule = job["schedule"]
    seed = job["seed"]
    cfg = job["cfg"]

    progress_file = Path(progress_dir) / f"{label}.txt"
    progress_file.write_text("0")

    with open(shared_data_path, "rb") as f:
        target_pool, bg_pool, eval_docs, space_pos = pickle.load(f)

    set_seed(seed)
    net_cfg = OmegaConf.create({
        "compile": False, "vocab_size": cfg["vocab_size"], "context_size": cfg["context_size"],
        "n_layer": cfg["n_layer"], "n_head": cfg["n_head"], "n_embd": cfg["n_embd"],
        "dropout": 0.0, "bias": False, "mlp": True,
    })
    net = nanoGPT(net_cfg).to(DEVICE)
    optim_cfg = OmegaConf.create({
        "learning_rate": cfg["lr"], "weight_decay": cfg["weight_decay"],
        "beta1": cfg["beta1"], "beta2": cfg["beta2"], "grad_clip": cfg["grad_clip"],
        "decay_lr": True, "warmup_iters": cfg["warmup_iters"], "min_lr": cfg["min_lr"],
    })
    optimizer = configure_optimizers(net, optim_cfg)

    scaler = torch.amp.GradScaler('cuda', enabled=DEVICE == "cuda")
    use_amp = DEVICE == "cuda"

    T, U = cfg["total_steps"], cfg["undo_steps"]
    total_lr_steps = T + U
    bs, p, ev = cfg["batch_size"], cfg["p_target"], cfg["eval_every"]

    log = {"step": [], "loss": [], "acc_target": [], "acc_background": [],
           "phase": [], "n_target_in_batch": [], "weight_deltas": []}

    w0 = snapshot_weights(net)
    net.train()
    it = 0
    steps_done = 0

    for s in range(T):
        nt = n_target_for_step(s, T, schedule, p, bs)
        batch = sample_batch(target_pool, bg_pool, nt, bs)
        dat = torch.from_numpy(batch).long().to(DEVICE)
        inp, tgt = dat[:, :-1], dat[:, 1:]

        it, lr = update_cosine_warmup_lr(it, optim_cfg, optimizer, total_lr_steps)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
            logits = net(inp)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        scaler.scale(loss).backward()
        if cfg["grad_clip"] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        steps_done += 1
        if steps_done % 100 == 0:
            progress_file.write_text(str(steps_done))

        if s % ev == 0 or s == T - 1:
            log["step"].append(it)
            log["loss"].append(loss.item())
            log["acc_target"].append(eval_accuracy(net, eval_docs["target"], space_pos))
            log["acc_background"].append(eval_accuracy(net, eval_docs["background"], space_pos))
            log["phase"].append("train")
            log["n_target_in_batch"].append(nt)
            log["weight_deltas"].append(weight_deltas(w0, snapshot_weights(net)))
            net.train()

    w_train_end = snapshot_weights(net)

    # Passive forgetting: train on background (A) data only with CORRECT labels.
    # No shuffling, no adversarial intervention - just neglect of B data.
    for s in range(U):
        batch = sample_batch(target_pool, bg_pool, 0, bs)
        dat = torch.from_numpy(batch).long().to(DEVICE)
        inp, tgt = dat[:, :-1], dat[:, 1:]

        it, lr = update_cosine_warmup_lr(it, optim_cfg, optimizer, total_lr_steps)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
            logits = net(inp)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        scaler.scale(loss).backward()
        if cfg["grad_clip"] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        steps_done += 1
        if steps_done % 100 == 0:
            progress_file.write_text(str(steps_done))

        if s % ev == 0 or s == U - 1:
            log["step"].append(it)
            log["loss"].append(loss.item())
            acc_t = eval_accuracy(net, eval_docs["target"], space_pos)
            log["acc_target"].append(acc_t)
            log["acc_background"].append(eval_accuracy(net, eval_docs["background"], space_pos))
            log["phase"].append("undo")
            log["n_target_in_batch"].append(0)
            log["weight_deltas"].append(weight_deltas(w_train_end, snapshot_weights(net)))
            net.train()

    progress_file.write_text(str(T + U))

    threshold = cfg["unlearn_threshold"]
    train_end_acc, unlearn_step = None, None
    undo_accs, undo_steps_list = [], []
    for i, ph in enumerate(log["phase"]):
        if ph == "train": train_end_acc = log["acc_target"][i]
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

    result = {
        "schedule": schedule, "seed": seed, "log": log,
        "train_end_acc": train_end_acc, "undo_end_acc": undo_end_acc,
        "undo_auc": undo_auc, "unlearn_step": unlearn_step,
        "mlp_undo_delta": mlp_undo, "attn_undo_delta": attn_undo,
        "config": dict(cfg), "label": label,
    }
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
