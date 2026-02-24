"""Worker for pure-bijection burst experiment.

Launched as a subprocess by experiment.py.
Each worker trains one model on one schedule and saves results.
"""
import sys, os, argparse, pickle, math, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
from pathlib import Path
from omegaconf import OmegaConf
import csv

from synthetic.init import set_seed
from net.nanogpt import nanoGPT
from net.runner import configure_optimizers, update_cosine_warmup_lr
from burst.data import BurstDataset
from burst.config import (
    EVAL_KEYS, MIXED_FRACTIONS, UNIFORM_SCHEDULE,
    PHASE_FOUNDATION, PHASE_BURST, PHASE_REVERSION,
    ExperimentConfig,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRAD_SIM_EVERY = ExperimentConfig().grad_sim_every


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


def _flat_grad(net) -> torch.Tensor:
    """Concatenate all parameter gradients into a single flat vector."""
    grads = [p.grad.detach().view(-1) for p in net.parameters() if p.grad is not None]
    return torch.cat(grads) if grads else torch.zeros(1, device=DEVICE)


def _grad_vec_for_docs(net, docs_np: np.ndarray, n_samples: int = 64) -> torch.Tensor:
    """Compute gradient vector for a sample of docs without modifying optimizer state."""
    n = min(n_samples, docs_np.shape[0])
    idx = np.random.choice(docs_np.shape[0], n, replace=False)
    dat = torch.as_tensor(docs_np[idx], dtype=torch.long, device=DEVICE)
    inp, tgt = dat[:, :-1], dat[:, 1:]
    # Use float32 directly to avoid scaler state entanglement
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        logits = net(inp)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
    loss.backward()
    return _flat_grad(net).float()


def compute_grad_cosine_sim(net, docs_burst_BL, docs_other_BL,
                            n_samples: int = 2048) -> dict:
    """Cosine similarity between burst-class and other-class gradient vectors."""
    net.train()
    net.zero_grad(set_to_none=True)
    g_burst = _grad_vec_for_docs(net, docs_burst_BL, n_samples=n_samples)
    net.zero_grad(set_to_none=True)
    g_other = _grad_vec_for_docs(net, docs_other_BL, n_samples=n_samples)
    cos_sim = F.cosine_similarity(g_burst.unsqueeze(0), g_other.unsqueeze(0)).item()
    net.zero_grad(set_to_none=True)
    return {"burst_vs_other": cos_sim}


def compute_pairwise_grad_sim(net, task_docs: dict,
                               burst_tasks: list, other_tasks: list,
                               n_samples: int = 2048) -> dict:
    """Pairwise cosine similarity between all task gradient vectors.

    task_docs: {task_tuple: docs_np}
    burst_tasks: list of burst task tuples (b* at pos 2)
    other_tasks: list of other task tuples

    Returns 'matrix' (list of lists), 'labels', 'n_burst', 'n_other'.
    """
    net.train()

    b_tasks = burst_tasks[:5]
    o_tasks = other_tasks[:5]
    all_tasks = b_tasks + o_tasks
    labels = [f"B{i+1}" for i in range(len(b_tasks))] + [f"O{i+1}" for i in range(len(o_tasks))]

    grad_vecs = []
    for task in all_tasks:
        if task in task_docs and task_docs[task].shape[0] > 0:
            net.zero_grad(set_to_none=True)
            grad_vecs.append(_grad_vec_for_docs(net, task_docs[task], n_samples=n_samples))
        else:
            grad_vecs.append(None)

    n = len(all_tasks)
    matrix = np.eye(n)
    valid = [(i, v) for i, v in enumerate(grad_vecs) if v is not None]
    if len(valid) >= 2:
        G = torch.stack([v for _, v in valid])
        G_norm = F.normalize(G, dim=1)
        sim = (G_norm @ G_norm.T).cpu().numpy()
        for ri, (i, _) in enumerate(valid):
            for rj, (j, _) in enumerate(valid):
                matrix[i, j] = sim[ri, rj]

    net.zero_grad(set_to_none=True)
    return {"matrix": matrix.tolist(), "labels": labels,
            "n_burst": len(b_tasks), "n_other": len(o_tasks)}


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
    gs_bs = cfg.get("grad_sim_batch_size", 2048)

    log = {"step": [], "loss": [], "phase": []}
    for k in EVAL_KEYS:
        log[k] = []

    # Gradient cosine similarity logs
    grad_sim_log = {"step": [], "phase": [], "burst_vs_other": []}
    pairwise_snapshots = []  # list of {step, phase, matrix, labels, n_burst, n_other}

    # Build flat arrays for grad sim computation
    burst_docs_all = np.concatenate(list(target_pool.values())) if target_pool else None
    other_docs_all = np.concatenate(list(bg_pool.values())) if bg_pool else None

    def do_eval(phase, loss_val):
        log["step"].append(it)
        log["loss"].append(loss_val)
        log["phase"].append(phase)
        for k in EVAL_KEYS:
            log[k].append(eval_free_gen(net, eval_docs[k.removeprefix("acc_")], prompt_len))
        net.train()

    def do_grad_sim(phase):
        if burst_docs_all is None or other_docs_all is None:
            return
        sim = compute_grad_cosine_sim(net, burst_docs_all, other_docs_all,
                                      n_samples=gs_bs)
        grad_sim_log["step"].append(it)
        grad_sim_log["phase"].append(phase)
        grad_sim_log["burst_vs_other"].append(sim["burst_vs_other"])
        net.train()

    def do_pairwise_snap(phase):
        task_docs = {**target_pool, **bg_pool}
        burst_tasks = list(target_pool.keys())
        other_tasks = list(bg_pool.keys())
        snap = compute_pairwise_grad_sim(net, task_docs, burst_tasks, other_tasks,
                                         n_samples=gs_bs)
        snap["step"] = it
        snap["phase"] = phase
        pairwise_snapshots.append(snap)
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

    # Pairwise snapshot steps: beginning, mid-burst, end-burst, mid-reversion, end-reversion
    pairwise_steps = {0, T // 2, T - 1, T + U // 2, T + U - 1}

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
        if s % GRAD_SIM_EVERY == 0 or s == T - 1:
            do_grad_sim(PHASE_BURST)
        if s in pairwise_steps:
            do_pairwise_snap(PHASE_BURST)
        if (s + 1) % 50 == 0:
            progress_file.write_text(str(s + 1))

    for s in range(U):
        batch_np, sampled_tasks = sample_batch(target_pool, bg_pool, 0, bs)
        task_counts_reversion.update(sampled_tasks)

        loss_val = train_step(batch_np, sampled_tasks)
        global_s = T + s
        if s % ev == 0 or s == U - 1:
            do_eval(PHASE_REVERSION, loss_val)
        if s % GRAD_SIM_EVERY == 0 or s == U - 1:
            do_grad_sim(PHASE_REVERSION)
        if global_s in pairwise_steps:
            do_pairwise_snap(PHASE_REVERSION)
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

    # t1/4: first reversion step where burst class acc drops to 25% of peak
    quarter_life = U
    half_life = U
    if peak_burst > 1e-6:
        for acc_val, us in zip(reversion_accs, reversion_steps):
            if half_life == U and acc_val <= peak_burst * 0.5:
                half_life = us
            if acc_val <= peak_burst * 0.25:
                quarter_life = us
                break

    dropoff_abs = peak_burst - reversion_end_burst
    dropoff_pct = (dropoff_abs / peak_burst * 100) if peak_burst > 1e-6 else 0.0

    gs_dir = Path(run_dir) / "grad_cosine_sim"
    gs_dir.mkdir(exist_ok=True)
    gs_record = {
        "schedule": schedule, "seed": seed, "label": label,
        "grad_sim_batch_size": gs_bs,
        "grad_sim_log": grad_sim_log,
        "pairwise_snapshots": [
            {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()}
            for s in pairwise_snapshots
        ],
    }
    with open(gs_dir / f"{label}.json", "w") as f:
        json.dump(gs_record, f)

    result = {
        "schedule": schedule, "seed": seed,
        "label": label, "log": log, "config": dict(cfg),
        "peak_burst": peak_burst, "reversion_end_burst": reversion_end_burst,
        "dropoff_abs": dropoff_abs, "dropoff_pct": dropoff_pct,
        "quarter_life": quarter_life, "reversion_auc": reversion_auc,
        "grad_sim_log": grad_sim_log,
        "pairwise_snapshots": pairwise_snapshots,
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
