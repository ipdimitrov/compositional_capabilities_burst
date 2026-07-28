"""Worker for pure-bijection burst experiment.

Launched as a subprocess by experiment.py.
Each worker trains one model on one schedule, saves checkpoints for
post-hoc grad-sim, and writes training metrics.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import torch

from burst.config import (
    ACC_BURST,
    CHECKPOINT_EVERY,
    CLASS_BURST,
    CLASS_OTHER,
    EVAL_BATCH_SIZE,
    EVAL_KEYS,
    GRAD_NORM_EPS,
    LOSS_BURST,
    LOSS_OTHER,
    PHASE_BURST,
    PHASE_REVERSION,
    TrainConfig,
    reversion_life_key,
)
from burst.core.data import BurstDataset
from burst.core.train_utils import (
    DEVICE,
    cross_entropy_logits_BTV_targets_BT,
    make_net_bare,
    make_optim_cfg,
    make_scaler,
    n_target_for_step,
    sample_batch,
)
from burst.rng import seed_all
from net.runner import configure_optimizers, reset_optimizer_state, update_phase_lr

if TYPE_CHECKING:
    from burst.types import WorkerJob
    from net.nanogpt import nanoGPT


@torch.no_grad()
def eval_free_gen(
    net: nanoGPT, docs_BL: np.ndarray, prompt_len: int,
    eval_start: int, eval_end: int,
) -> float:
    """Evaluate free-generation accuracy on tokens [eval_start, eval_end)."""
    if docs_BL.shape[0] == 0:
        return 0.0
    net.eval()
    loader = torch.utils.data.DataLoader(
        BurstDataset(docs_BL), batch_size=EVAL_BATCH_SIZE, shuffle=False, pin_memory=(DEVICE == "cuda")
    )
    correct_t = torch.zeros(1, device=DEVICE)
    total = 0
    n_new = docs_BL.shape[1] - prompt_len
    for dat, tgt in loader:
        dat_d, tgt_d = dat.to(DEVICE, non_blocking=True), tgt.to(DEVICE, non_blocking=True)
        full = net.generate(dat_d[:, :prompt_len], n_new)
        gen = full[:, eval_start:eval_end]
        ref = tgt_d[:, eval_start - 1 : eval_end - 1]
        correct_t += (gen == ref).float().sum()
        total += ref.numel()
    net.train()
    return correct_t.item() / max(total, 1)


@torch.no_grad()
def eval_loss(net: nanoGPT, docs_BL: np.ndarray) -> float:
    """Evaluate average cross-entropy loss on a document set."""
    if docs_BL.shape[0] == 0:
        return float("nan")
    net.eval()
    loader = torch.utils.data.DataLoader(
        BurstDataset(docs_BL), batch_size=EVAL_BATCH_SIZE, shuffle=False, pin_memory=(DEVICE == "cuda")
    )
    total_loss = 0.0
    n_batches = 0
    for inputs_BT, targets_BT in loader:
        inp_d = inputs_BT.to(DEVICE, non_blocking=True)
        tgt_d = targets_BT.to(DEVICE, non_blocking=True)
        logits_BTV = net(inp_d)
        loss = cross_entropy_logits_BTV_targets_BT(logits_BTV, tgt_d)
        total_loss += loss.item()
        n_batches += 1
    net.train()
    return total_loss / max(n_batches, 1)


def checkpoint_steps(T: int, U: int) -> dict[int, str]:
    """Return {global_step: phase} for all steps that need a checkpoint.

    T = total_steps (burst), U = reversion_steps.
    Global steps: [0, T) = burst, [T, T+U) = reversion.
    """
    steps = {}
    for s in range(T):
        if s % CHECKPOINT_EVERY == 0 or s == T - 1:
            steps[s] = PHASE_BURST
    for s in range(U):
        gs = T + s
        if s % CHECKPOINT_EVERY == 0 or s == U - 1:
            steps[gs] = PHASE_REVERSION

    pairwise = {
        0: PHASE_BURST,
        T // 2: PHASE_BURST,
        T - 1: PHASE_BURST,
        T + U // 2: PHASE_REVERSION,
        T + U - 1: PHASE_REVERSION,
    }
    steps.update(pairwise)
    return steps


def save_task_distribution(  # noqa: PLR0913
    stats_dir: Path,
    label: str,
    schedule: str,
    seed: int,
    phase_name: str,
    counter_data: Counter,
) -> None:
    """Write per-task sample counts to a CSV file."""
    if not counter_data:
        return
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

    if not rows:
        return
    fieldnames = ["schedule", "seed", "phase", "task_type", "composition", "count"]
    depth = len(next(iter(counter_data))[1:])
    fieldnames += [f"f{d}" for d in range(depth, 0, -1)]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def compute_reversion_metrics(
    log: dict, P: int, T: int, U: int
) -> dict[str, object]:
    """Extract reversion summary metrics from the training log."""
    burst_end_step = P + T
    reversion_accs = [
        log[ACC_BURST][i] for i, ph in enumerate(log["phase"]) if ph == PHASE_REVERSION
    ]
    reversion_steps_log = [
        log["step"][i] for i, ph in enumerate(log["phase"]) if ph == PHASE_REVERSION
    ]
    reversion_steps_rel = [s - burst_end_step for s in reversion_steps_log]
    burst_accs = [log[ACC_BURST][i] for i, ph in enumerate(log["phase"]) if ph == PHASE_BURST]

    peak_burst = max(burst_accs) if burst_accs else 0
    reversion_end_burst = reversion_accs[-1] if reversion_accs else peak_burst
    reversion_auc = (
        float(np.trapezoid(reversion_accs, reversion_steps_rel))
        if len(reversion_accs) > 1
        else 0.0
    )

    thresholds = TrainConfig().reversion_thresholds
    life_times: dict[str, int] = {}
    if peak_burst > GRAD_NORM_EPS:
        remaining = dict.fromkeys(thresholds, True)
        for acc_val, us in zip(reversion_accs, reversion_steps_rel, strict=True):
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
    dropoff_pct = (dropoff_abs / peak_burst * 100) if peak_burst > GRAD_NORM_EPS else 0.0

    return {
        "burst_end_step": burst_end_step,
        "peak_burst": peak_burst,
        "reversion_end_burst": reversion_end_burst,
        "dropoff_abs": dropoff_abs,
        "dropoff_pct": dropoff_pct,
        "reversion_auc": reversion_auc,
        **life_times,
    }


def run(job: WorkerJob, shared_data_path: str, run_dir: str, progress_dir: str) -> None:  # noqa: C901, PLR0912, PLR0915
    """Train one model on one schedule and write results to disk."""
    label, schedule, seed, cfg = job["label"], job["schedule"], job["seed"], job["cfg"]
    deterministic = bool(job["deterministic"])
    seed_all(seed, deterministic=deterministic)
    pretrain_ckpt = job.get("pretrain_ckpt")
    pretrain_log_path = job.get("pretrain_log_path")

    progress_file = Path(progress_dir) / f"{label}.txt"
    progress_file.write_text("0")

    with Path(shared_data_path).open("rb") as f:
        target_pool, bg_pool, eval_docs, prompt_len, eval_start, eval_end = pickle.load(f)  # noqa: S301

    pretrain_log = None
    if pretrain_log_path and Path(pretrain_log_path).exists():
        with Path(pretrain_log_path).open("rb") as f:
            pretrain_log = pickle.load(f)  # noqa: S301

    seed_all(seed, deterministic=deterministic)
    net = make_net_bare(cfg)
    if pretrain_ckpt and Path(pretrain_ckpt).exists():
        net.load_state_dict(torch.load(pretrain_ckpt, map_location=DEVICE, weights_only=True))
    if DEVICE == "cuda":
        net = torch.compile(net)
    raw_net = getattr(net, "_orig_mod", net)

    optimizer = configure_optimizers(net, make_optim_cfg(cfg))
    scaler = make_scaler()

    T, U = cfg["total_steps"], cfg["reversion_steps"]
    P = cfg["pre_burst_steps"]
    bs, p, ev = cfg["batch_size"], cfg["p_target"], cfg["eval_every"]
    lr_max = cfg["lr"]
    warmup_steps = cfg["warmup_iters"]
    lr_pe = cfg["lr_pretrain_end_frac"]
    lr_be = cfg["lr_burst_end_frac"]
    lr_re = cfg["lr_reversion_end_frac"]

    log = {"step": [], "loss": [], LOSS_OTHER: [], LOSS_BURST: [], "phase": []}
    for k in EVAL_KEYS:
        log[k] = []

    if pretrain_log and pretrain_log.get("step"):
        for key, vals in log.items():
            vals.extend(pretrain_log[key])

    ckpt_dir = Path(run_dir) / "checkpoints" / label
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_steps = checkpoint_steps(T, U)

    def do_eval(global_step: int, phase: str, loss_val: float) -> None:
        log["step"].append(global_step)
        log["loss"].append(loss_val)
        log["phase"].append(phase)
        for k in EVAL_KEYS:
            log[k].append(eval_free_gen(net, eval_docs[k.removeprefix("acc_")],
            prompt_len, eval_start, eval_end))
        log[LOSS_OTHER].append(eval_loss(net, eval_docs[CLASS_OTHER]))
        log[LOSS_BURST].append(eval_loss(net, eval_docs[CLASS_BURST]))
        net.train()

    max_micro_bs = 512

    def do_train_step(batch_np: np.ndarray, global_step: int) -> float:
        tokens_BL = torch.as_tensor(batch_np, dtype=torch.long, device=DEVICE)
        inputs_BT, targets_BT = tokens_BL[:, :-1], tokens_BL[:, 1:]
        update_phase_lr(
            global_step, optimizer, warmup_steps, P, T, U, lr_max, lr_pe, lr_be, lr_re,
        )
        optimizer.zero_grad(set_to_none=True)
        n = inputs_BT.size(0)
        n_accum = (n + max_micro_bs - 1) // max_micro_bs
        total_loss = 0.0
        for i in range(n_accum):
            s, e = i * max_micro_bs, min((i + 1) * max_micro_bs, n)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                logits_BTV = net(inputs_BT[s:e])
                loss = cross_entropy_logits_BTV_targets_BT(logits_BTV, targets_BT[s:e])
                loss = loss / n_accum
            scaler.scale(loss).backward()
            total_loss += loss.item()
        if cfg["grad_clip"] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        return total_loss

    net.train()
    t_ids = list(target_pool.keys())
    b_ids = list(bg_pool.keys())
    task_counts_burst = Counter()
    task_counts_reversion = Counter()

    for s in range(T):
        nt = n_target_for_step(s, T, schedule, p, bs)
        batch_np, sampled_tasks = sample_batch(target_pool, bg_pool, nt, bs, t_ids, b_ids)
        task_counts_burst.update(sampled_tasks)
        gs = P + s
        loss_val = do_train_step(batch_np, gs + 1)
        if s % ev == 0 or s == T - 1:
            do_eval(gs, PHASE_BURST, loss_val)
        if s in ckpt_steps:
            torch.save(raw_net.state_dict(), ckpt_dir / f"step_{s}.pt")
        if (s + 1) % 50 == 0:
            progress_file.write_text(str(s + 1))

    reset_optimizer_state(optimizer)

    for s in range(U):
        batch_np, sampled_tasks = sample_batch(target_pool, bg_pool, 0, bs, t_ids, b_ids)
        task_counts_reversion.update(sampled_tasks)
        gs = P + T + s
        loss_val = do_train_step(batch_np, gs + 1)
        if s % ev == 0 or s == U - 1:
            do_eval(gs, PHASE_REVERSION, loss_val)
        local_gs = T + s
        if local_gs in ckpt_steps:
            torch.save(raw_net.state_dict(), ckpt_dir / f"step_{local_gs}.pt")
        if (s + 1) % 50 == 0:
            progress_file.write_text(str(T + s + 1))

    progress_file.write_text(str(T + U))

    stats_dir = Path(run_dir) / "task_distributions"
    stats_dir.mkdir(exist_ok=True)
    save_task_distribution(
        stats_dir, label, schedule, seed, PHASE_BURST, task_counts_burst,
    )
    save_task_distribution(
        stats_dir, label, schedule, seed, PHASE_REVERSION, task_counts_reversion,
    )

    metrics = compute_reversion_metrics(log, P, T, U)

    result = {
        "schedule": schedule,
        "seed": seed,
        "label": label,
        "log": log,
        "config": dict(cfg),
        "pre_burst_steps": P,
        **metrics,
    }
    for k in EVAL_KEYS:
        for phase in (PHASE_BURST, PHASE_REVERSION):
            vals = [log[k][i] for i, ph in enumerate(log["phase"]) if ph == phase]
            result[f"{phase}_end_{k}"] = vals[-1] if vals else 0

    with (Path(run_dir) / f"{label}.pkl").open("wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--progress-dir", required=True)
    args = parser.parse_args()

    with Path(args.job_path).open("rb") as f:
        job = pickle.load(f)  # noqa: S301
    run(job, args.data_path, args.run_dir, args.progress_dir)
