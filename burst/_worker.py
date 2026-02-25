"""Worker for pure-bijection burst experiment.

Launched as a subprocess by experiment.py.
Each worker trains one model on one schedule, saves checkpoints for
post-hoc grad-sim, and writes training metrics.
"""
import sys, os, argparse, pickle, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
from pathlib import Path

import csv
from omegaconf import OmegaConf

from synthetic.init import set_seed
from net.nanogpt import nanoGPT
from net.runner import configure_optimizers, update_cosine_warmup_lr
from burst.data import BurstDataset
from burst.config import (
    EVAL_KEYS, MIXED_FRACTIONS, UNIFORM_SCHEDULE,
    PHASE_FOUNDATION, PHASE_BURST, PHASE_REVERSION,
    TrainConfig, reversion_life_key,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_EVERY = 50
PAIRWISE_STEP_NAMES = {"begin", "mid_burst", "end_burst", "mid_reversion", "end_reversion"}


def n_target_for_step(step, total_steps, schedule, p, batch_size):
    T = total_steps
    burst_len = max(int(p * T), 1)

    if schedule == UNIFORM_SCHEDULE:
        return int(np.random.binomial(batch_size, p))

    if schedule == "burst_100":
        return batch_size if step >= T - burst_len else 0

    if schedule == "mid_block":
        mid = T // 2
        half = burst_len // 2
        return batch_size if mid - half <= step < mid + (burst_len - half) else 0

    if schedule in MIXED_FRACTIONS:
        frac = MIXED_FRACTIONS[schedule]
        window = min(int(burst_len / frac), T)
        return int(round(batch_size * frac)) if step >= T - window else 0

    if schedule == "ramp_up":
        max_frac = 0.20
        ramp_len = min(int(2 * burst_len / max_frac), T)
        if step >= T - ramp_len:
            progress = (step - (T - ramp_len)) / max(ramp_len - 1, 1)
            return int(round(batch_size * progress * max_frac))
        return 0

    if schedule == "reversion_only":
        return 0

    raise ValueError(f"Unknown schedule: {schedule}")


def sample_batch(target_pool, bg_pool, n_target, batch_size):
    t_ids = list(target_pool.keys())
    b_ids = list(bg_pool.keys())
    parts = []
    sampled_tasks = []

    def _sample_from(pool, ids, n):
        if n == 0:
            return
        per = n // len(ids)
        rem = n % len(ids)
        for i, tid in enumerate(ids):
            k = per + (1 if i < rem else 0)
            if k > 0:
                idx = np.random.randint(len(pool[tid]), size=k)
                parts.append(pool[tid][idx])
                sampled_tasks.extend([tid] * k)

    _sample_from(target_pool, t_ids, n_target)
    _sample_from(bg_pool, b_ids, batch_size - n_target)

    perm = np.random.permutation(batch_size)
    return np.concatenate(parts)[perm], [sampled_tasks[i] for i in perm]


@torch.no_grad()
def eval_free_gen(net, docs_BL, prompt_len: int):
    if docs_BL.shape[0] <= 1 and docs_BL.sum() == 0:
        return 0.0
    net.eval()
    loader = torch.utils.data.DataLoader(
        BurstDataset(docs_BL), batch_size=256, shuffle=False,
        pin_memory=(DEVICE == "cuda"))
    correct_t = torch.zeros(1, device=DEVICE)
    total = 0
    for dat, tgt in loader:
        dat, tgt = dat.to(DEVICE, non_blocking=True), tgt.to(DEVICE, non_blocking=True)
        inp = dat[:, :prompt_len]
        for _ in range(dat.shape[1] - prompt_len):
            nxt = net(inp)[:, -1, :].argmax(-1, keepdim=True)
            inp = torch.cat([inp, nxt], dim=1)
        gen = inp[:, prompt_len:]
        ref = tgt[:, prompt_len - 1:]
        ml = min(gen.shape[1], ref.shape[1])
        last6 = max(0, ml - 6)
        correct_t += (gen[:, last6:ml] == ref[:, last6:ml]).float().sum()
        total += ref[:, last6:ml].numel()
    net.train()
    return correct_t.item() / max(total, 1)


def checkpoint_steps(T: int, U: int) -> dict[int, str]:
    """Return {global_step: phase} for all steps that need a checkpoint."""
    steps = {}
    for s in range(T):
        if s % CHECKPOINT_EVERY == 0 or s == T - 1:
            steps[s] = PHASE_BURST
    for s in range(U):
        if s % CHECKPOINT_EVERY == 0 or s == U - 1:
            steps[T + s] = PHASE_REVERSION

    pairwise = {0: PHASE_BURST, T // 2: PHASE_BURST, T - 1: PHASE_BURST,
                T + U // 2: PHASE_REVERSION, T + U - 1: PHASE_REVERSION}
    steps.update(pairwise)
    return steps


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

    T, U = cfg["total_steps"], cfg["reversion_steps"]
    bs, p, ev = cfg["batch_size"], cfg["p_target"], cfg["eval_every"]

    log = {"step": [], "loss": [], "phase": []}
    for k in EVAL_KEYS:
        log[k] = []

    ckpt_dir = Path(run_dir) / "checkpoints" / label
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_steps = checkpoint_steps(T, U)

    def do_eval(phase, loss_val):
        log["step"].append(it)
        log["loss"].append(loss_val)
        log["phase"].append(phase)
        for k in EVAL_KEYS:
            log[k].append(eval_free_gen(net, eval_docs[k.removeprefix("acc_")], prompt_len))
        net.train()

    def train_step(batch_np, tasks_sampled):
        nonlocal it
        dat = torch.as_tensor(batch_np, dtype=torch.long, device=DEVICE)
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

    task_counts_foundation = Counter()
    task_counts_burst = Counter()
    task_counts_reversion = Counter()

    burst_len = max(int(p * T), 1)
    if schedule == UNIFORM_SCHEDULE:
        foundation_end = T
    elif schedule == "burst_100":
        foundation_end = T - burst_len
    elif schedule in MIXED_FRACTIONS:
        frac = MIXED_FRACTIONS[schedule]
        window = min(int(burst_len / frac), T)
        foundation_end = T - window
    else:
        foundation_end = T // 2

    for s in range(T):
        nt = n_target_for_step(s, T, schedule, p, bs)
        batch_np, sampled_tasks = sample_batch(target_pool, bg_pool, nt, bs)

        if s < foundation_end:
            task_counts_foundation.update(sampled_tasks)
        else:
            task_counts_burst.update(sampled_tasks)

        loss_val = train_step(batch_np, sampled_tasks)
        if s % ev == 0 or s == T - 1:
            do_eval(PHASE_BURST, loss_val)
        if s in ckpt_steps:
            torch.save(net.state_dict(), ckpt_dir / f"step_{s}.pt")
        if (s + 1) % 50 == 0:
            progress_file.write_text(str(s + 1))

    for s in range(U):
        batch_np, sampled_tasks = sample_batch(target_pool, bg_pool, 0, bs)
        task_counts_reversion.update(sampled_tasks)

        loss_val = train_step(batch_np, sampled_tasks)
        global_s = T + s
        if s % ev == 0 or s == U - 1:
            do_eval(PHASE_REVERSION, loss_val)
        if global_s in ckpt_steps:
            torch.save(net.state_dict(), ckpt_dir / f"step_{global_s}.pt")
        if (s + 1) % 50 == 0:
            progress_file.write_text(str(T + s + 1))

    progress_file.write_text(str(T + U))

    stats_dir = Path(run_dir) / "task_distributions"
    stats_dir.mkdir(exist_ok=True)

    def save_task_distribution_stats(phase_name, counter_data):
        csv_path = stats_dir / f"{label}_{phase_name}.csv"

        rows = []
        for task, count in counter_data.items():
            task_type = task[0]
            fn_vals = task[1:]

            row = {
                "schedule": schedule,
                "seed": seed,
                "phase": phase_name,
                "task_type": task_type,
                "composition": "_".join(str(f) for f in fn_vals),
                "count": count,
            }
            for i, fv in enumerate(fn_vals):
                row[f"f{len(fn_vals) - i}"] = fv

            rows.append(row)

        if rows:
            fieldnames = ["schedule", "seed", "phase", "task_type", "composition", "count"]
            depth = len(task[1:]) if counter_data else 3
            fieldnames += [f"f{d}" for d in range(depth, 0, -1)]
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)

    if task_counts_foundation:
        save_task_distribution_stats(PHASE_FOUNDATION, task_counts_foundation)
    if task_counts_burst:
        save_task_distribution_stats(PHASE_BURST, task_counts_burst)
    if task_counts_reversion:
        save_task_distribution_stats(PHASE_REVERSION, task_counts_reversion)

    reversion_accs = [log["acc_burst"][i] for i, ph in enumerate(log["phase"]) if ph == PHASE_REVERSION]
    reversion_steps = [log["step"][i] - T for i, ph in enumerate(log["phase"]) if ph == PHASE_REVERSION]
    burst_accs = [log["acc_burst"][i] for i, ph in enumerate(log["phase"]) if ph == PHASE_BURST]

    peak_burst = max(burst_accs) if burst_accs else 0
    reversion_end_burst = reversion_accs[-1] if reversion_accs else peak_burst
    reversion_auc = float(np.trapz(reversion_accs, reversion_steps)) if len(reversion_accs) > 1 else 0.0

    thresholds = TrainConfig().reversion_thresholds
    life_times: dict[str, int] = {}
    if peak_burst > 1e-6:
        remaining = {t: True for t in thresholds}
        for acc_val, us in zip(reversion_accs, reversion_steps):
            for t in list(remaining):
                if acc_val <= peak_burst * t:
                    life_times[reversion_life_key(t)] = us
                    del remaining[t]
            if not remaining:
                break
    for t in thresholds:
        k = reversion_life_key(t)
        if k not in life_times:
            life_times[k] = U

    dropoff_abs = peak_burst - reversion_end_burst
    dropoff_pct = (dropoff_abs / peak_burst * 100) if peak_burst > 1e-6 else 0.0

    result = {
        "schedule": schedule, "seed": seed,
        "label": label, "log": log, "config": dict(cfg),
        "peak_burst": peak_burst, "reversion_end_burst": reversion_end_burst,
        "dropoff_abs": dropoff_abs, "dropoff_pct": dropoff_pct,
        "reversion_auc": reversion_auc,
        **life_times,
    }
    for k in EVAL_KEYS:
        for phase in (PHASE_BURST, PHASE_REVERSION):
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
