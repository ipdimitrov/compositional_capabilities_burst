"""Ten new post-hoc mechanistic metrics for burstiness runs.

Complements burst/deep_analysis.py (which covers ADL, gradient interference,
EMA interpolation probe, critical sharpness, weight delta rank).

New metrics implemented here:

From checkpoints:
  1. Task Vector Transfer       — does τ = θ_post − θ_pre transfer to a fresh model?
  2. Forgetting Trajectory Dim  — PCA dimensionality of the reversion weight path
  3. Relearning Efficiency       — burst accuracy recovery after 50 fine-tune steps
  4. Linear Mode Connectivity    — loss barrier on the straight path peak→reverted
  5. Pruning Robustness          — burst accuracy vs magnitude-based weight sparsity

From existing data (no checkpoints needed):
  6. Pairwise Gradient Separation  — BURST vs ALL_OTHER cosine sim across 5 key steps
  7. Forgetting Speed Decomposition — initial slope / plateau / AUC from training curves
  8. Per-Layer Interference Localisation — which layer has most negative cosine sim?
  9. Gradient Interference Temporal Dynamics — reversion-phase re-alignment trajectory
 10. Burst Position Comparison    — cross-run meta-analysis (pos1 / pos2 / pos3)

Usage:
    uv run python burst/new_metrics.py \\
        data/burst_d3_pos1_<tag> data/burst_d3_pos2_<tag> data/burst_d3_pos3_<tag> \\
        --existing-results data/deep_analysis_combined/results.pkl \\
        --out-dir data/new_metrics_combined \\
        --n-seeds 3

Flags:
    --existing-results PATH   Path to deep_analysis_combined/results.pkl (required)
    --out-dir PATH            Output directory (default: data/new_metrics_combined)
    --n-seeds N               Seeds per schedule for checkpoint-based metrics (default: 3)
    --n-prune-levels N        Sparsity levels for pruning robustness (default: 10)
    --relearn-steps N         Fine-tune steps for relearning efficiency (default: 50)

Dimension key:
    B: batch_size
    L: sequence_length (doc_len)
    N: n_embd (model dimension)
    K: n_layers + 1 (embedding + each transformer block output)
    T: token positions (L - 1)
    V: vocab_size
    P: number of parameters (flattened)
    C: number of PCA components
"""
import sys, os, argparse, pickle, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from omegaconf import OmegaConf

from net.nanogpt import nanoGPT
from net.runner import configure_optimizers, update_cosine_warmup_lr
from burst.train_utils import load_net
from burst.config import (
    PHASE_BURST, PHASE_REVERSION, SCHEDULE_ORDER, SCHED_COLORS,
    parse_run_config,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCHEDULES_ORDERED = SCHEDULE_ORDER
SCHEDULE_COLORS = SCHED_COLORS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_params(net: nanoGPT) -> torch.Tensor:
    return torch.cat([p.detach().float().cpu().view(-1) for p in net.parameters()])


@torch.no_grad()
def _free_gen_acc(net: nanoGPT, docs_BL: np.ndarray, prompt_len: int) -> float:
    net.eval()
    docs_t = torch.as_tensor(docs_BL, dtype=torch.long, device=DEVICE)
    B, L = docs_t.shape
    target_B6 = docs_t[:, -6:]
    generated = net.generate(docs_t[:, :prompt_len], L - prompt_len)
    return (generated[:, -6:] == target_B6).all(dim=1).float().mean().item()


def _sched_order(s: str) -> int:
    try:
        return SCHEDULES_ORDERED.index(s)
    except ValueError:
        return 99


def _color(s: str) -> str:
    return SCHEDULE_COLORS.get(s, "#888888")


def _ckpt_files(ckpt_dir: Path) -> dict[int, Path]:
    return {int(p.stem.split("_")[1]): p for p in ckpt_dir.glob("step_*.pt")}


# ---------------------------------------------------------------------------
# Metric 1: Task Vector Transfer
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_task_vector_transfer(
    ckpt_root: Path,
    all_results: list[dict],
    burst_docs_BL: np.ndarray,
    prompt_len: int,
    n_seeds: int = 3,
) -> dict:
    """Transfer τ = θ_post_burst − θ_pre_burst to a different seed's pre-burst model.

    For a shallow wrapper (burst_100): τ encodes the burst capability as a
    modular add-on — adding it to any model should grant burst accuracy.
    For deep learning (burst_10): τ is entangled with the specific training
    trajectory and won't transfer cleanly.

    Returns per-schedule mean transfer accuracy.
    """
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)
    results = {}

    for sched in schedules:
        sched_results = jobs_by_schedule[sched]
        transfer_accs = []
        seeds_done = 0

        for i, r_src in enumerate(sched_results):
            if seeds_done >= n_seeds:
                break
            label_src = r_src["label"]
            ckpt_dir_src = ckpt_root / label_src
            if not ckpt_dir_src.exists():
                continue
            files_src = _ckpt_files(ckpt_dir_src)
            if not files_src:
                continue

            available = sorted(files_src.keys())
            T = r_src["config"]["total_steps"]
            pre_step = available[0]
            peak_step = min(available, key=lambda x: abs(x - (T - 1)))

            # Find a different seed to transfer to
            r_tgt = next(
                (r for j, r in enumerate(sched_results) if j != i
                 and (ckpt_root / r["label"]).exists()),
                None,
            )
            if r_tgt is None:
                continue

            label_tgt = r_tgt["label"]
            ckpt_dir_tgt = ckpt_root / label_tgt
            files_tgt = _ckpt_files(ckpt_dir_tgt)
            if not files_tgt:
                continue
            pre_step_tgt = min(files_tgt.keys())

            cfg = r_src["config"]
            # Load state dicts directly without full model instantiation for τ computation
            peak_sd = {k: v.float() for k, v in torch.load(
                str(files_src[peak_step]), map_location="cpu", weights_only=True).items()}
            pre_sd = {k: v.float() for k, v in torch.load(
                str(files_src[pre_step]), map_location="cpu", weights_only=True).items()}
            tgt_sd = {k: v.float() for k, v in torch.load(
                str(files_tgt[pre_step_tgt]), map_location="cpu", weights_only=True).items()}

            # τ = θ_peak − θ_pre (source)
            tau = {k: peak_sd[k] - pre_sd[k] for k in peak_sd}

            # Apply τ to target pre-burst model and evaluate
            transferred_sd = {k: tgt_sd[k] + tau[k] for k in tgt_sd}
            net_pre_tgt = load_net(cfg, str(files_tgt[pre_step_tgt]))
            net_pre_tgt.load_state_dict(transferred_sd)

            n = min(256, burst_docs_BL.shape[0])
            idx = np.random.choice(burst_docs_BL.shape[0], n, replace=False)
            acc = _free_gen_acc(net_pre_tgt, burst_docs_BL[idx], prompt_len)
            transfer_accs.append(acc)
            seeds_done += 1
            print(f"  {label_src} → {label_tgt}: transfer_acc={acc:.3f}", flush=True)

        results[sched] = {
            "transfer_accs": transfer_accs,
            "mean_transfer_acc": float(np.mean(transfer_accs)) if transfer_accs else float("nan"),
        }

    return results


# ---------------------------------------------------------------------------
# Metric 2: Forgetting Trajectory Dimensionality
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_forgetting_trajectory_dim(
    ckpt_root: Path,
    all_results: list[dict],
    n_seeds: int = 3,
    variance_threshold: float = 0.95,
    max_ckpts: int = 15,
) -> dict:
    """PCA dimensionality of the weight trajectory during reversion.

    Low dimensionality = the model is undoing a single direction (wrapper).
    High dimensionality = complex multi-directional restructuring (deep).

    Subsamples up to max_ckpts reversion checkpoints evenly to keep runtime
    tractable (102 checkpoints → 15 sampled).

    Returns per-schedule mean effective dimensionality.
    """
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)
    results = {}

    for sched in schedules:
        sched_results = jobs_by_schedule[sched]
        dims = []
        seeds_done = 0

        for r in sched_results:
            if seeds_done >= n_seeds:
                break
            label = r["label"]
            ckpt_dir = ckpt_root / label
            if not ckpt_dir.exists():
                continue
            files = _ckpt_files(ckpt_dir)
            if not files:
                continue

            T = r["config"]["total_steps"]
            rev_steps_all = sorted(s for s in files if s >= T)
            if len(rev_steps_all) < 3:
                continue

            # Subsample evenly to keep runtime tractable
            if len(rev_steps_all) > max_ckpts:
                indices = np.linspace(0, len(rev_steps_all) - 1, max_ckpts, dtype=int)
                rev_steps = [rev_steps_all[i] for i in indices]
            else:
                rev_steps = rev_steps_all

            # Collect flattened weight vectors during reversion
            # Load state_dict directly (no model instantiation needed)
            weight_vecs = []
            for step in rev_steps:
                sd = torch.load(str(files[step]), map_location="cpu", weights_only=True)
                flat = torch.cat([v.float().view(-1) for v in sd.values()])
                weight_vecs.append(flat.numpy())

            W = np.stack(weight_vecs)  # (n_steps, n_params)
            W_centered = W - W.mean(axis=0, keepdims=True)

            # PCA via SVD on the trajectory matrix (n_steps × n_params)
            # Economy SVD — n_steps << n_params
            try:
                _, sv, _ = np.linalg.svd(W_centered, full_matrices=False)
                var_explained = np.cumsum(sv ** 2) / (sv ** 2).sum()
                dim = int(np.searchsorted(var_explained, variance_threshold)) + 1
            except np.linalg.LinAlgError:
                dim = len(rev_steps)

            dims.append(dim)
            seeds_done += 1
            print(f"  {label}: trajectory_dim={dim} (from {len(rev_steps)} ckpts)", flush=True)

        results[sched] = {
            "dims": dims,
            "mean_dim": float(np.mean(dims)) if dims else float("nan"),
        }

    return results


# ---------------------------------------------------------------------------
# Metric 3: Relearning Efficiency
# ---------------------------------------------------------------------------

def compute_relearning_efficiency(
    ckpt_root: Path,
    all_results: list[dict],
    burst_docs_BL: np.ndarray,
    other_docs_BL: np.ndarray,
    prompt_len: int,
    n_seeds: int = 3,
    relearn_steps: int = 50,
) -> dict:
    """Re-expose burst data after full reversion; measure recovery speed.

    Shallow (burst_100): pathway is suppressed not destroyed — fast reacquisition.
    Deep (burst_10): capability genuinely restructured — slower recovery.

    Returns per-schedule mean relearning AUC (higher = faster relearning).
    """
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)
    results = {}

    for sched in schedules:
        sched_results = jobs_by_schedule[sched]
        all_accs: list[list[float]] = []
        seeds_done = 0

        for r in sched_results:
            if seeds_done >= n_seeds:
                break
            label = r["label"]
            ckpt_dir = ckpt_root / label
            if not ckpt_dir.exists():
                continue
            files = _ckpt_files(ckpt_dir)
            if not files:
                continue

            rev_step = max(files.keys())
            cfg = r["config"]
            net = load_net(cfg, str(files[rev_step]))

            optim_cfg = OmegaConf.create({
                "learning_rate": cfg["lr"] * 0.3,
                "weight_decay": cfg["weight_decay"],
                "beta1": cfg["beta1"], "beta2": cfg["beta2"],
                "grad_clip": cfg["grad_clip"], "decay_lr": False,
                "warmup_iters": 0, "min_lr": cfg["lr"] * 0.3,
            })
            optimizer = configure_optimizers(net, optim_cfg)
            scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")

            n = min(256, burst_docs_BL.shape[0])
            idx = np.random.choice(burst_docs_BL.shape[0], n, replace=False)
            docs_fine = burst_docs_BL[idx]

            accs = []
            net.train()
            it = 0
            for step in range(relearn_steps):
                batch_idx = np.random.choice(n, min(cfg["batch_size"], n), replace=True)
                batch = torch.as_tensor(docs_fine[batch_idx], dtype=torch.long, device=DEVICE)
                inp, tgt = batch[:, :-1], batch[:, 1:]
                it, _ = update_cosine_warmup_lr(it, optim_cfg, optimizer, relearn_steps)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                    logits = net(inp)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()

                if step % 5 == 0 or step == relearn_steps - 1:
                    acc = _free_gen_acc(net, docs_fine, prompt_len)
                    accs.append((step, acc))

            all_accs.append(accs)
            seeds_done += 1
            final_acc = accs[-1][1] if accs else 0
            print(f"  {label}: relearn_final_acc={final_acc:.3f}", flush=True)

        if all_accs:
            steps_common = [a[0] for a in all_accs[0]]
            mean_accs = [float(np.mean([run[i][1] for run in all_accs])) for i in range(len(steps_common))]
            _trapz = getattr(np, "trapezoid", np.trapz)
            auc = float(_trapz(mean_accs, steps_common)) / max(steps_common[-1], 1) if len(steps_common) > 1 else 0.0
        else:
            steps_common, mean_accs, auc = [], [], float("nan")

        results[sched] = {
            "steps": steps_common,
            "mean_accs": mean_accs,
            "auc": auc,
        }

    return results


# ---------------------------------------------------------------------------
# Metric 4: Linear Mode Connectivity
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_linear_mode_connectivity(
    ckpt_root: Path,
    all_results: list[dict],
    burst_docs_BL: np.ndarray,
    prompt_len: int,
    n_seeds: int = 3,
    n_alphas: int = 11,
) -> dict:
    """Loss barrier on the linear path from θ_peak_burst to θ_post_reversion.

    Low barrier = model slid back along a ridge (shallow, reversible wrapper).
    High barrier = genuinely different solution found (deep, robust learning).

    Returns per-schedule mean barrier height and path loss curves.
    """
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)
    alphas = np.linspace(0, 1, n_alphas).tolist()
    results = {}

    n = min(256, burst_docs_BL.shape[0])
    idx = np.random.choice(burst_docs_BL.shape[0], n, replace=False)
    docs = burst_docs_BL[idx]
    docs_t = torch.as_tensor(docs, dtype=torch.long, device=DEVICE)
    inp_t, tgt_t = docs_t[:, :-1], docs_t[:, 1:]

    for sched in schedules:
        sched_results = jobs_by_schedule[sched]
        all_loss_curves: list[list[float]] = []
        seeds_done = 0

        for r in sched_results:
            if seeds_done >= n_seeds:
                break
            label = r["label"]
            ckpt_dir = ckpt_root / label
            if not ckpt_dir.exists():
                continue
            files = _ckpt_files(ckpt_dir)
            if not files:
                continue

            T = r["config"]["total_steps"]
            available = sorted(files.keys())
            peak_step = min(available, key=lambda x: abs(x - (T - 1)))
            rev_step = max(available)

            cfg = r["config"]
            peak_sd = {k: v.float() for k, v in torch.load(
                str(files[peak_step]), map_location="cpu", weights_only=True).items()}
            rev_sd = {k: v.float() for k, v in torch.load(
                str(files[rev_step]), map_location="cpu", weights_only=True).items()}

            net_interp = load_net(cfg, str(files[peak_step]))
            loss_curve = []
            for alpha in alphas:
                interp_sd = {k: (1 - alpha) * peak_sd[k] + alpha * rev_sd[k] for k in peak_sd}
                net_interp.load_state_dict(interp_sd)
                net_interp.eval()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                    logits = net_interp(inp_t).float()
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_t.reshape(-1)).item()
                loss_curve.append(loss)

            all_loss_curves.append(loss_curve)
            seeds_done += 1

            endpoint_avg = (loss_curve[0] + loss_curve[-1]) / 2
            barrier = max(loss_curve) - endpoint_avg
            print(f"  {label}: LMC barrier={barrier:.4f}", flush=True)

        if all_loss_curves:
            mean_curve = [float(np.mean([c[i] for c in all_loss_curves])) for i in range(n_alphas)]
            endpoint_avg = (mean_curve[0] + mean_curve[-1]) / 2
            barrier = float(max(mean_curve) - endpoint_avg)
        else:
            mean_curve = [float("nan")] * n_alphas
            barrier = float("nan")

        results[sched] = {
            "alphas": alphas,
            "mean_loss_curve": mean_curve,
            "barrier": barrier,
        }

    return results


# ---------------------------------------------------------------------------
# Metric 5: Pruning Robustness
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_pruning_robustness(
    ckpt_root: Path,
    all_results: list[dict],
    burst_docs_BL: np.ndarray,
    prompt_len: int,
    n_seeds: int = 3,
    n_prune_levels: int = 10,
) -> dict:
    """Burst accuracy vs magnitude-based weight sparsity at peak burst.

    Shallow wrapper: accuracy drops sharply even at low sparsity — the
    capability is concentrated in a few high-magnitude weights.
    Deep representation: accuracy degrades gracefully — distributed storage.

    Returns per-schedule accuracy-vs-sparsity curves.
    """
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)
    sparsities = np.linspace(0, 0.9, n_prune_levels).tolist()
    results = {}

    n = min(256, burst_docs_BL.shape[0])
    idx = np.random.choice(burst_docs_BL.shape[0], n, replace=False)
    docs = burst_docs_BL[idx]

    for sched in schedules:
        sched_results = jobs_by_schedule[sched]
        all_acc_curves: list[list[float]] = []
        seeds_done = 0

        for r in sched_results:
            if seeds_done >= n_seeds:
                break
            label = r["label"]
            ckpt_dir = ckpt_root / label
            if not ckpt_dir.exists():
                continue
            files = _ckpt_files(ckpt_dir)
            if not files:
                continue

            T = r["config"]["total_steps"]
            available = sorted(files.keys())
            peak_step = min(available, key=lambda x: abs(x - (T - 1)))
            cfg = r["config"]

            base_sd = torch.load(str(files[peak_step]), map_location=DEVICE, weights_only=True)
            all_weights = torch.cat([v.view(-1).abs() for v in base_sd.values()])
            net = load_net(cfg, str(files[peak_step]))

            acc_curve = []
            for sparsity in sparsities:
                if sparsity > 0:
                    threshold = torch.quantile(all_weights, sparsity)
                    pruned_sd = {k: v * (v.abs() >= threshold).to(v.dtype) for k, v in base_sd.items()}
                    net.load_state_dict(pruned_sd)
                acc = _free_gen_acc(net, docs, prompt_len)
                acc_curve.append(acc)

            all_acc_curves.append(acc_curve)
            seeds_done += 1
            print(f"  {label}: pruning acc@0%={acc_curve[0]:.3f}, acc@90%={acc_curve[-1]:.3f}", flush=True)

        if all_acc_curves:
            mean_curve = [float(np.mean([c[i] for c in all_acc_curves])) for i in range(n_prune_levels)]
            # Area under pruning curve (higher = more robust = deeper)
            _trapz = getattr(np, "trapezoid", np.trapz)
            robustness_auc = float(_trapz(mean_curve, sparsities))
        else:
            mean_curve = [float("nan")] * n_prune_levels
            robustness_auc = float("nan")

        results[sched] = {
            "sparsities": sparsities,
            "mean_accs": mean_curve,
            "robustness_auc": robustness_auc,
        }

    return results


# ---------------------------------------------------------------------------
# Metric 6: Pairwise Gradient Separation
# ---------------------------------------------------------------------------

def compute_pairwise_grad_separation(all_results: list[dict]) -> dict:
    """Extract BURST vs ALL_OTHER gradient cosine similarity from pairwise_snapshots.

    The pairwise_snapshots contain a 6×6 matrix at 5 key steps:
    labels = [BURST, O_F1, O_F2, O_F3, ALL_OTHER, ALL_DATA]

    We extract the BURST↔ALL_OTHER entry (index [0, 4]) at each step.
    Low cosine sim (near -1) = burst gradients are maximally opposed to other
    gradients = the model is learning a conflicting representation.

    Returns per-schedule mean separation at each key step.
    """
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)
    results = {}

    for sched in schedules:
        sched_results = jobs_by_schedule[sched]
        step_sims: dict[int, list[float]] = {}

        for r in sched_results:
            snapshots = r.get("pairwise_snapshots", [])
            for snap in snapshots:
                step = snap["step"]
                matrix = snap["matrix"]
                labels = snap["labels"]
                try:
                    burst_idx = labels.index("BURST")
                    other_idx = labels.index("ALL_OTHER")
                    sim = matrix[burst_idx][other_idx]
                    step_sims.setdefault(step, []).append(sim)
                except (ValueError, IndexError):
                    pass

        steps_sorted = sorted(step_sims.keys())
        results[sched] = {
            "steps": steps_sorted,
            "mean_sims": [float(np.mean(step_sims[s])) for s in steps_sorted],
            "std_sims": [float(np.std(step_sims[s])) for s in steps_sorted],
        }

    return results


# ---------------------------------------------------------------------------
# Metric 7: Forgetting Speed Decomposition
# ---------------------------------------------------------------------------

def compute_forgetting_speed_decomposition(all_results: list[dict]) -> dict:
    """Decompose reversion into initial drop rate, plateau level, and total AUC.

    Three sub-metrics:
    - initial_slope: rate of burst accuracy drop in first 50 reversion steps
    - plateau_acc: mean burst accuracy in the last 20% of reversion
    - reversion_auc: total area under burst accuracy curve during reversion

    Reveals whether burstiness affects *speed* of initial forgetting vs
    *depth* of final forgetting.
    """
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)
    results = {}

    for sched in schedules:
        sched_results = jobs_by_schedule[sched]
        initial_slopes, plateau_accs, rev_aucs, peak_bursts = [], [], [], []

        for r in sched_results:
            log = r["log"]
            steps = log["step"]
            accs = log["acc_burst"]
            phases = log["phase"]
            T = r["config"]["total_steps"]

            rev_steps = [s - T for s, p in zip(steps, phases) if p == PHASE_REVERSION]
            rev_accs = [a for a, p in zip(accs, phases) if p == PHASE_REVERSION]

            if len(rev_accs) < 2:
                continue

            peak = r.get("peak_burst", max(
                a for a, p in zip(accs, phases) if p == PHASE_BURST
            ) if any(p == PHASE_BURST for p in phases) else 0)
            peak_bursts.append(peak)

            # Initial slope: linear fit over first 50 reversion steps
            early_mask = [s <= 50 for s in rev_steps]
            early_steps = [s for s, m in zip(rev_steps, early_mask) if m]
            early_accs = [a for a, m in zip(rev_accs, early_mask) if m]
            if len(early_steps) >= 2:
                slope = float(np.polyfit(early_steps, early_accs, 1)[0])
            else:
                slope = float("nan")
            initial_slopes.append(slope)

            # Plateau: mean of last 20% of reversion steps
            cutoff = int(len(rev_accs) * 0.8)
            plateau = float(np.mean(rev_accs[cutoff:])) if rev_accs[cutoff:] else float("nan")
            plateau_accs.append(plateau)

            rev_aucs.append(r.get("reversion_auc", float("nan")))

        results[sched] = {
            "mean_initial_slope": float(np.nanmean(initial_slopes)) if initial_slopes else float("nan"),
            "mean_plateau_acc": float(np.nanmean(plateau_accs)) if plateau_accs else float("nan"),
            "mean_reversion_auc": float(np.nanmean(rev_aucs)) if rev_aucs else float("nan"),
            "mean_peak_burst": float(np.nanmean(peak_bursts)) if peak_bursts else float("nan"),
            "initial_slopes": initial_slopes,
            "plateau_accs": plateau_accs,
        }

    return results


# ---------------------------------------------------------------------------
# Metric 8: Per-Layer Interference Localisation
# ---------------------------------------------------------------------------

def compute_layer_interference_localisation(all_results: list[dict]) -> dict:
    """Find which layer has the most negative gradient cosine sim during burst.

    burst_100 (pure burst batches): interference is zero by construction —
    all gradients point the same way. The per-layer signal shows where the
    model is most 'committed' to the burst direction.
    burst_10 (mixed batches): interference is non-zero — the most negative
    layer is where the conflict is sharpest.

    Returns per-schedule: most-conflicted layer, min cosine sim, and full
    per-layer mean interference profile.
    """
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)
    results = {}

    for sched in schedules:
        sched_results = jobs_by_schedule[sched]
        layer_sims: dict[str, list[float]] = {}
        layer_names = []

        for r in sched_results:
            gsl = r.get("grad_sim_log", {})
            per_layer = gsl.get("per_layer", {})
            phases = gsl.get("phase", [])
            if not layer_names and gsl.get("layer_names"):
                layer_names = gsl["layer_names"]

            for ln, vals in per_layer.items():
                burst_vals = [v for v, p in zip(vals, phases) if p == PHASE_BURST]
                if burst_vals:
                    layer_sims.setdefault(ln, []).append(float(np.mean(burst_vals)))

        if not layer_sims:
            results[sched] = {}
            continue

        mean_per_layer = {ln: float(np.mean(vs)) for ln, vs in layer_sims.items()}
        most_conflicted = min(mean_per_layer, key=mean_per_layer.get)
        min_sim = mean_per_layer[most_conflicted]

        results[sched] = {
            "mean_per_layer": mean_per_layer,
            "most_conflicted_layer": most_conflicted,
            "min_cosine_sim": min_sim,
            "layer_names": layer_names or list(mean_per_layer.keys()),
        }

    return results


# ---------------------------------------------------------------------------
# Metric 9: Gradient Interference Temporal Dynamics (reversion re-alignment)
# ---------------------------------------------------------------------------

def compute_grad_interference_temporal(all_results: list[dict]) -> dict:
    """Track burst_vs_other cosine similarity during reversion phase.

    After the burst ends, how quickly do burst and other-class gradients
    re-align (cosine sim → 1)?

    burst_100: sharp re-alignment — the wrapper is quickly suppressed.
    burst_10: slower, smoother re-alignment — the entangled representation
    takes longer to restructure.

    Returns per-schedule mean trajectory and re-alignment speed metric.
    """
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)
    results = {}

    for sched in schedules:
        sched_results = jobs_by_schedule[sched]
        step_sims: dict[int, list[float]] = {}

        for r in sched_results:
            gsl = r.get("grad_sim_log", {})
            steps = gsl.get("step", [])
            sims = gsl.get("burst_vs_other", [])
            phases = gsl.get("phase", [])
            T = r["config"]["total_steps"]

            for s, sim, ph in zip(steps, sims, phases):
                if ph == PHASE_REVERSION:
                    rel_step = s - T
                    step_sims.setdefault(rel_step, []).append(sim)

        if not step_sims:
            results[sched] = {"steps": [], "mean_sims": [], "realign_speed": float("nan")}
            continue

        steps_sorted = sorted(step_sims.keys())
        mean_sims = [float(np.mean(step_sims[s])) for s in steps_sorted]

        # Re-alignment speed: steps until cosine sim first exceeds 0.1
        # (threshold of 0.5 is too high — reversion sims rarely exceed 0.5)
        realign_step = next((s for s, sim in zip(steps_sorted, mean_sims) if sim > 0.1),
                            steps_sorted[-1] if steps_sorted else float("nan"))

        # Also compute mean sim in last 20% of reversion (plateau alignment)
        cutoff = int(len(mean_sims) * 0.8)
        plateau_sim = float(np.mean(mean_sims[cutoff:])) if mean_sims[cutoff:] else float("nan")

        results[sched] = {
            "steps": steps_sorted,
            "mean_sims": mean_sims,
            "realign_speed": float(realign_step),
            "plateau_sim": plateau_sim,
        }

    return results


# ---------------------------------------------------------------------------
# Metric 11: Gradient Projection (OGD-style interference decomposition)
# ---------------------------------------------------------------------------

_PROJ_KEYS = ("interference_magnitude", "useful_learning",
              "interference_ratio", "burst_norm", "other_norm")


def compute_grad_projection_metrics(all_results: list[dict]) -> dict:
    """Aggregate gradient projection time-series from grad_sim_log.

    For each schedule, collects per-step:
      interference_magnitude: ||g_burst^parallel||  (absolute interference)
      useful_learning:        ||g_burst^perp||       (orthogonal learning signal)
      interference_ratio:     ||g_parallel|| / ||g_burst||  (= |cos α|)

    Returns per-schedule mean ± std trajectories over seeds, split by phase.
    """
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    results = {}
    for sched, sched_results in sorted(jobs_by_schedule.items(), key=lambda x: _sched_order(x[0])):
        step_data: dict[int, dict[str, list[float]]] = {}
        step_phases: dict[int, str] = {}

        for r in sched_results:
            gsl = r.get("grad_sim_log", {})
            proj = gsl.get("grad_projection", {})
            steps = gsl.get("step", [])
            if not proj or not steps:
                continue
            for i, (step, phase) in enumerate(zip(steps, gsl.get("phase", []))):
                step_data.setdefault(step, {k: [] for k in _PROJ_KEYS})
                step_phases[step] = phase
                for k in _PROJ_KEYS:
                    vals = proj.get(k, [])
                    if i < len(vals) and not np.isnan(vals[i]):
                        step_data[step][k].append(vals[i])

        if not step_data:
            results[sched] = {}
            continue

        steps_sorted = sorted(step_data)
        out, out_std = {k: [] for k in _PROJ_KEYS}, {f"{k}_std": [] for k in _PROJ_KEYS}
        for step in steps_sorted:
            for k in _PROJ_KEYS:
                vals = step_data[step][k]
                out[k].append(float(np.mean(vals)) if vals else float("nan"))
                out_std[f"{k}_std"].append(float(np.std(vals)) if len(vals) > 1 else 0.0)

        results[sched] = {
            "steps": steps_sorted,
            "phases": [step_phases[s] for s in steps_sorted],
            **out, **out_std,
        }

    return results


# ---------------------------------------------------------------------------
# Metric 10: Burst Position Comparison (cross-run meta-analysis)
# ---------------------------------------------------------------------------

def compute_burst_position_comparison(existing_analyses: list[dict]) -> dict:
    """Compare key metrics across burst positions (pos1, pos2, pos3).

    Uses the already-computed deep_analysis results to show whether the
    position of the burst function in the composition chain affects:
    - forgetting speed (reversion_auc)
    - learning depth (EMA cliff alpha, weight delta rank)
    - sharpness
    - ADL readability

    Returns per-position per-schedule summary.
    """
    results = {}

    for analysis in existing_analyses:
        run_name = analysis["run_name"]
        burst_pos = analysis.get("burst_pos", "?")
        sm = analysis.get("summary_metrics", {})
        tv = analysis.get("task_vectors", {})
        sh = analysis.get("sharpness", {})
        wdr = analysis.get("weight_delta_rank", {})
        adl = analysis.get("adl", {})

        schedules = sorted(sm.keys(), key=_sched_order)
        pos_data = {}

        for sched in schedules:
            # ADL readability at peak burst
            adl_sched = adl.get(sched, {})
            peak_steps = [s for s in adl_sched if s <= 499]
            adl_readability = (adl_sched[max(peak_steps)]["mean_readability"]
                               if peak_steps else float("nan"))
            adl_acc_drop = (adl_sched[max(peak_steps)]["mean_max_acc_drop"]
                            if peak_steps else float("nan"))

            pos_data[sched] = {
                "reversion_auc": sm[sched].get("mean_reversion_auc", float("nan")),
                "peak_burst": sm[sched].get("mean_peak_burst", float("nan")),
                "ema_cliff_alpha": tv.get(sched, {}).get("mean_cliff_alpha", float("nan")),
                "ema_auc": tv.get(sched, {}).get("mean_auc", float("nan")),
                "sharpness": sh.get(sched, {}).get("mean", float("nan")),
                "weight_delta_rank": wdr.get(sched, {}).get("total_rank", float("nan")),
                "adl_readability": adl_readability,
                "adl_acc_drop": adl_acc_drop,
            }

        results[run_name] = {
            "burst_pos": burst_pos,
            "schedules": schedules,
            "per_schedule": pos_data,
        }

    return results


# ---------------------------------------------------------------------------
# Dashboard generation
# ---------------------------------------------------------------------------

from burst.plot_utils import save_png as _save_png


_METRIC_DESCRIPTIONS = {
    "task_vector_transfer": {
        "title": "Task Vector Transfer Accuracy",
        "what": (
            "Computes τ = θ_post_burst − θ_pre_burst for each model, then adds τ to a "
            "different seed's pre-burst model and measures burst accuracy on that new model. "
            "This tests whether the burst capability is a modular, transferable add-on."
        ),
        "high": "High transfer accuracy → the burst capability is stored as a modular wrapper (τ encodes it cleanly).",
        "low": "Low transfer accuracy → the capability is entangled with the specific training trajectory and cannot be transplanted.",
        "limitations": (
            "Transfer may fail for reasons other than depth (e.g., different random initialisation). "
            "Works best when comparing extreme schedules (burst_100 vs burst_10)."
        ),
    },
    "forgetting_trajectory_dim": {
        "title": "Forgetting Trajectory Dimensionality",
        "what": (
            "Collects the flattened weight vector at every reversion checkpoint, then runs PCA "
            "on the trajectory matrix. The number of principal components needed to explain 95% "
            "of variance in the trajectory = dimensionality of the forgetting path."
        ),
        "high": "High dimensionality → the model is restructuring weights in many directions simultaneously (deep, complex forgetting).",
        "low": "Low dimensionality (e.g., 1–2) → the model is simply undoing a single direction (shallow wrapper removal).",
        "limitations": (
            "Sensitive to checkpoint frequency. Very small models may have low dimensionality by default. "
            "Compare relative differences across schedules rather than absolute values."
        ),
    },
    "relearning_efficiency": {
        "title": "Relearning Efficiency After Full Reversion",
        "what": (
            "After full reversion, re-exposes the model to burst data for 50 fine-tuning steps "
            "and measures the burst accuracy recovery curve. AUC of this curve is the relearning score."
        ),
        "high": "High AUC → fast reacquisition of burst capability (the pathway was suppressed, not destroyed — shallow).",
        "low": "Low AUC → slow recovery (the capability was genuinely restructured — deep).",
        "limitations": (
            "Uses a reduced learning rate (0.3× original). Results may vary with LR choice. "
            "Only 50 steps — very fast learners may saturate before differences emerge."
        ),
    },
    "linear_mode_connectivity": {
        "title": "Linear Mode Connectivity: Loss Barrier",
        "what": (
            "Interpolates linearly between θ_peak_burst and θ_post_reversion and evaluates "
            "burst-class cross-entropy loss at each interpolation point. The 'barrier' is the "
            "maximum loss minus the average of the two endpoint losses."
        ),
        "high": "High barrier → the two models are in different loss basins (deep learning found a genuinely different solution).",
        "low": "Low barrier → the path between peak and reverted is smooth (shallow wrapper — model just slid back along a ridge).",
        "limitations": (
            "Loss barrier is measured on burst data only. A high barrier could also indicate "
            "that the reverted model is in a bad basin for burst data (expected), not necessarily "
            "that the representation is deep. Compare with EMA interpolation accuracy curves."
        ),
    },
    "pruning_robustness": {
        "title": "Pruning Robustness: Accuracy vs Sparsity",
        "what": (
            "At peak burst accuracy, prunes k% of weights by magnitude (sets smallest-magnitude "
            "weights to zero) and measures burst accuracy at each sparsity level."
        ),
        "high": "Robust to pruning (accuracy stays high at high sparsity) → capability is distributed across many weights (deep).",
        "low": "Fragile to pruning (accuracy drops fast at low sparsity) → capability is concentrated in a few high-magnitude weights (shallow wrapper).",
        "limitations": (
            "Magnitude-based pruning is a crude proxy. Structured pruning or Fisher-information-based "
            "pruning would be more principled but much more expensive."
        ),
    },
    "pairwise_grad_separation": {
        "title": "Pairwise Gradient Separation: BURST vs ALL_OTHER",
        "what": (
            "Extracts the cosine similarity between the BURST task gradient and the ALL_OTHER "
            "task gradient from the pre-computed pairwise gradient snapshots at 5 key training steps "
            "(step 0, mid-burst, end-burst, mid-reversion, end-reversion)."
        ),
        "high": "High cosine sim (near 1) → burst and other gradients are aligned (no conflict — model learns both simultaneously).",
        "low": "Low cosine sim (near -1) → burst and other gradients are opposed (strong conflict — burst learning interferes with other-class knowledge).",
        "limitations": (
            "Only 5 snapshots per run. The pairwise matrix uses a single representative batch "
            "per task group, so estimates are noisy. Use as a directional indicator."
        ),
    },
    "forgetting_speed_decomposition": {
        "title": "Forgetting Speed Decomposition",
        "what": (
            "Decomposes the reversion phase into three sub-metrics: "
            "(1) initial slope — rate of burst accuracy drop in the first 50 reversion steps; "
            "(2) plateau accuracy — mean burst accuracy in the last 20% of reversion; "
            "(3) reversion AUC — total area under the burst accuracy curve."
        ),
        "high": "High initial slope magnitude → fast initial forgetting. High plateau → capability partially retained at end.",
        "low": "Low initial slope → slow forgetting onset. Low plateau → capability fully lost by end of reversion.",
        "limitations": (
            "Slope estimate is noisy with only ~2–4 data points in the first 50 steps. "
            "Plateau is computed from the last 20% of logged steps, not wall-clock time."
        ),
    },
    "layer_interference_localisation": {
        "title": "Per-Layer Gradient Interference Localisation",
        "what": (
            "For each layer, computes the mean cosine similarity between burst and other-class "
            "gradients during the burst phase. Identifies which layer has the most negative "
            "cosine similarity (most conflicted) for each schedule."
        ),
        "high": "High (near 1) cosine sim at a layer → burst and other gradients agree at that layer (shared representation).",
        "low": "Low (near -1) cosine sim → strong conflict at that layer (burst is overwriting other-class knowledge there).",
        "limitations": (
            "burst_100 has zero interference by construction (pure burst batches), so all values "
            "will be near 1. The metric is most informative for mixed schedules (burst_10 to burst_75)."
        ),
    },
    "grad_interference_temporal": {
        "title": "Gradient Re-Alignment During Reversion",
        "what": (
            "Tracks the burst_vs_other gradient cosine similarity during the reversion phase. "
            "After the burst ends, how quickly do burst and other-class gradients re-align? "
            "Re-alignment speed = number of reversion steps until cosine sim first exceeds 0.5."
        ),
        "high": "Fast re-alignment (low re-alignment step) → the burst representation is quickly suppressed (shallow).",
        "low": "Slow re-alignment (high re-alignment step) → the burst representation persists and conflicts with other-class learning for longer (deep).",
        "limitations": (
            "The 0.1 threshold is empirically chosen based on observed sim ranges. "
            "Re-alignment speed is sensitive to the initial cosine similarity at the start of reversion. "
            "For burst_100 (pure burst batches), the starting sim is most negative, making re-alignment slower."
        ),
    },
    "grad_projection": {
        "title": "Gradient Projection: Interference vs Useful Learning",
        "what": (
            "Decomposes the burst gradient g_burst into a component parallel to g_other "
            "(interference: g_parallel = projection onto g_other direction) and an orthogonal "
            "residual (useful learning: g_perp = g_burst - g_parallel). "
            "Tracks all three quantities over training steps with error bars across seeds."
        ),
        "high": (
            "High interference_magnitude → burst steps strongly affect other-class parameters. "
            "High useful_learning → burst steps move model in directions orthogonal to other-class concerns."
        ),
        "low": (
            "Low interference_ratio → most of the burst gradient is orthogonal to other-class gradients "
            "(minimal conflict). Diluted schedules should show lower interference_ratio."
        ),
        "limitations": (
            "Projection is computed on aggregate (pooled) gradients, not per-example. "
            "The interference_ratio equals |cos(α)| so it is bounded in [0, 1]. "
            "A ratio near 0.5 is expected at random initialisation."
        ),
    },
    "burst_position_comparison": {
        "title": "Burst Position Effect on Learning Depth",
        "what": (
            "Cross-run meta-analysis comparing the 5 existing deep-analysis metrics "
            "(reversion AUC, EMA cliff alpha, sharpness, weight delta rank, ADL readability) "
            "across burst positions 1, 2, and 3 in the composition chain. "
            "Position 1 = burst function applied first; position 3 = applied last."
        ),
        "high": "Higher reversion AUC at a given position → capability is more robust at that position in the chain.",
        "low": "Lower reversion AUC → faster forgetting (shallower learning) at that position.",
        "limitations": (
            "Only 3 positions available. The effect of position may interact with depth (n_a=3). "
            "Differences may reflect task difficulty rather than learning depth per se."
        ),
    },
}


def make_dashboard(new_results: dict, existing_analyses: list[dict], out_dir: Path) -> None:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    all_figs: list[tuple[str, str, go.Figure]] = []

    def _add_fig(key: str, fig: go.Figure) -> None:
        desc = _METRIC_DESCRIPTIONS.get(key, {})
        title = desc.get("title", key)
        all_figs.append((key, title, fig))
        _save_png(fig, str(charts_dir / f"{key}.png"))

    # ------------------------------------------------------------------
    # Chart 1: Task Vector Transfer
    # ------------------------------------------------------------------
    for analysis in existing_analyses:
        run_name = analysis["run_name"]
        tvt = new_results.get("task_vector_transfer", {}).get(run_name, {})
        if not tvt:
            continue
        schedules = sorted(tvt.keys(), key=_sched_order)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=schedules,
            y=[tvt[s]["mean_transfer_acc"] for s in schedules],
            marker_color=[_color(s) for s in schedules],
            name=run_name,
        ))
        fig.update_layout(
            title=f"Task Vector Transfer Accuracy — {run_name}",
            xaxis_title="Schedule",
            yaxis_title="Burst Accuracy After Transfer",
            template="plotly_white", height=500,
        )
        _add_fig("task_vector_transfer", fig)

    # ------------------------------------------------------------------
    # Chart 2: Forgetting Trajectory Dimensionality
    # ------------------------------------------------------------------
    for analysis in existing_analyses:
        run_name = analysis["run_name"]
        ftd = new_results.get("forgetting_trajectory_dim", {}).get(run_name, {})
        if not ftd:
            continue
        schedules = sorted(ftd.keys(), key=_sched_order)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=schedules,
            y=[ftd[s]["mean_dim"] for s in schedules],
            marker_color=[_color(s) for s in schedules],
            name=run_name,
        ))
        fig.update_layout(
            title=f"Forgetting Trajectory Dimensionality — {run_name}<br>"
                  "<sup>PCA components to explain 95% variance of reversion weight path</sup>",
            xaxis_title="Schedule",
            yaxis_title="Effective Dimensionality (95% variance)",
            template="plotly_white", height=500,
        )
        _add_fig("forgetting_trajectory_dim", fig)

    # ------------------------------------------------------------------
    # Chart 3: Relearning Efficiency
    # ------------------------------------------------------------------
    for analysis in existing_analyses:
        run_name = analysis["run_name"]
        rle = new_results.get("relearning_efficiency", {}).get(run_name, {})
        if not rle:
            continue
        schedules = sorted(rle.keys(), key=_sched_order)
        fig = go.Figure()
        for sched in schedules:
            d = rle[sched]
            if not d["steps"]:
                continue
            fig.add_trace(go.Scatter(
                x=d["steps"], y=d["mean_accs"],
                name=sched,
                line=dict(color=_color(sched), width=2),
                mode="lines+markers",
            ))
        fig.update_layout(
            title=f"Relearning Efficiency After Full Reversion — {run_name}<br>"
                  "<sup>Burst accuracy recovery during 50 fine-tune steps on burst data</sup>",
            xaxis_title="Relearning Step",
            yaxis_title="Burst Accuracy",
            legend_title="Schedule",
            template="plotly_white", height=500,
        )
        _add_fig("relearning_efficiency", fig)

    # Relearning AUC bar chart
    for analysis in existing_analyses:
        run_name = analysis["run_name"]
        rle = new_results.get("relearning_efficiency", {}).get(run_name, {})
        if not rle:
            continue
        schedules = sorted(rle.keys(), key=_sched_order)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=schedules,
            y=[rle[s]["auc"] for s in schedules],
            marker_color=[_color(s) for s in schedules],
            name=run_name,
        ))
        fig.update_layout(
            title=f"Relearning AUC — {run_name}<br>"
                  "<sup>Higher = faster reacquisition = shallower original learning</sup>",
            xaxis_title="Schedule",
            yaxis_title="Relearning AUC (normalised)",
            template="plotly_white", height=500,
        )
        _add_fig("relearning_auc", fig)

    # ------------------------------------------------------------------
    # Chart 4: Linear Mode Connectivity (Peak→Reverted removed — use unified_analysis for pre→peak LMC)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Chart 5: Pruning Robustness
    # ------------------------------------------------------------------
    for analysis in existing_analyses:
        run_name = analysis["run_name"]
        pr = new_results.get("pruning_robustness", {}).get(run_name, {})
        if not pr:
            continue
        schedules = sorted(pr.keys(), key=_sched_order)
        fig = go.Figure()
        for sched in schedules:
            d = pr[sched]
            fig.add_trace(go.Scatter(
                x=[s * 100 for s in d["sparsities"]], y=d["mean_accs"],
                name=sched,
                line=dict(color=_color(sched), width=2),
                mode="lines+markers",
            ))
        fig.update_layout(
            title=f"Pruning Robustness — {run_name}<br>"
                  "<sup>Burst accuracy vs weight sparsity at peak burst. "
                  "Robust = deep; fragile = shallow wrapper.</sup>",
            xaxis_title="Weight Sparsity (%)",
            yaxis_title="Burst Accuracy",
            yaxis_range=[0, 1],
            legend_title="Schedule",
            template="plotly_white", height=500,
        )
        _add_fig("pruning_robustness", fig)

    # ------------------------------------------------------------------
    # Chart 6: Pairwise Gradient Separation
    # ------------------------------------------------------------------
    for analysis in existing_analyses:
        run_name = analysis["run_name"]
        pgs = new_results.get("pairwise_grad_separation", {}).get(run_name, {})
        if not pgs:
            continue
        schedules = sorted(pgs.keys(), key=_sched_order)
        fig = go.Figure()
        for sched in schedules:
            d = pgs[sched]
            if not d["steps"]:
                continue
            fig.add_trace(go.Scatter(
                x=d["steps"], y=d["mean_sims"],
                name=sched,
                line=dict(color=_color(sched), width=2),
                mode="lines+markers",
            ))
        fig.add_vline(x=499, line_dash="dash", line_color="gray",
                      annotation_text="burst→reversion")
        fig.update_layout(
            title=f"Pairwise Gradient Separation: BURST vs ALL_OTHER — {run_name}<br>"
                  "<sup>Cosine similarity between burst and other-class gradients at 5 key steps. "
                  "Low = strong conflict.</sup>",
            xaxis_title="Training Step",
            yaxis_title="Cosine Similarity (BURST vs ALL_OTHER)",
            legend_title="Schedule",
            template="plotly_white", height=500,
        )
        _add_fig("pairwise_grad_separation", fig)

    # ------------------------------------------------------------------
    # Chart 7: Forgetting Speed Decomposition
    # ------------------------------------------------------------------
    for analysis in existing_analyses:
        run_name = analysis["run_name"]
        fsd = new_results.get("forgetting_speed_decomposition", {}).get(run_name, {})
        if not fsd:
            continue
        schedules = sorted(fsd.keys(), key=_sched_order)

        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=["Initial Drop Rate", "Plateau Accuracy", "Reversion AUC"])
        colors = [_color(s) for s in schedules]

        fig.add_trace(go.Bar(
            x=schedules,
            y=[fsd[s]["mean_initial_slope"] for s in schedules],
            marker_color=colors, showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=schedules,
            y=[fsd[s]["mean_plateau_acc"] for s in schedules],
            marker_color=colors, showlegend=False,
        ), row=1, col=2)
        fig.add_trace(go.Bar(
            x=schedules,
            y=[fsd[s]["mean_reversion_auc"] for s in schedules],
            marker_color=colors, showlegend=False,
        ), row=1, col=3)

        fig.update_layout(
            title=f"Forgetting Speed Decomposition — {run_name}<br>"
                  "<sup>Initial slope (first 50 steps), plateau (last 20%), total AUC</sup>",
            template="plotly_white", height=500,
        )
        _add_fig("forgetting_speed_decomposition", fig)

    # ------------------------------------------------------------------
    # Chart 8: Per-Layer Interference Localisation
    # ------------------------------------------------------------------
    for analysis in existing_analyses:
        run_name = analysis["run_name"]
        pli = new_results.get("layer_interference_localisation", {}).get(run_name, {})
        if not pli:
            continue
        schedules = sorted(pli.keys(), key=_sched_order)
        sample = next((pli[s] for s in schedules if pli.get(s)), None)
        if not sample:
            continue
        layer_names = sample.get("layer_names", [])

        z = []
        for sched in schedules:
            d = pli.get(sched, {})
            row = [d.get("mean_per_layer", {}).get(ln, float("nan")) for ln in layer_names]
            z.append(row)

        fig = go.Figure(go.Heatmap(
            z=z, x=layer_names, y=schedules,
            colorscale="RdBu", zmid=0,
            colorbar=dict(title="Cosine Sim"),
        ))
        fig.update_layout(
            title=f"Per-Layer Gradient Interference (Burst Phase) — {run_name}<br>"
                  "<sup>Mean cosine sim between burst and other-class gradients per layer. "
                  "Blue = conflict; Red = agreement.</sup>",
            xaxis_title="Layer",
            yaxis_title="Schedule",
            template="plotly_white", height=500,
        )
        _add_fig("layer_interference_localisation", fig)

    # ------------------------------------------------------------------
    # Chart 9: Gradient Interference Temporal Dynamics
    # ------------------------------------------------------------------
    for analysis in existing_analyses:
        run_name = analysis["run_name"]
        git = new_results.get("grad_interference_temporal", {}).get(run_name, {})
        if not git:
            continue
        schedules = sorted(git.keys(), key=_sched_order)
        fig = go.Figure()
        for sched in schedules:
            d = git[sched]
            if not d["steps"]:
                continue
            fig.add_trace(go.Scatter(
                x=d["steps"], y=d["mean_sims"],
                name=sched,
                line=dict(color=_color(sched), width=2),
                mode="lines",
            ))
        fig.add_hline(y=0.5, line_dash="dot", line_color="gray",
                      annotation_text="re-alignment threshold")
        fig.update_layout(
            title=f"Gradient Re-Alignment During Reversion — {run_name}<br>"
                  "<sup>Burst vs other-class gradient cosine similarity during reversion phase. "
                  "Fast rise = shallow (wrapper quickly suppressed).</sup>",
            xaxis_title="Reversion Step",
            yaxis_title="Cosine Similarity (burst vs other)",
            legend_title="Schedule",
            template="plotly_white", height=500,
        )
        _add_fig("grad_interference_temporal", fig)

    # Re-alignment speed bar chart
    for analysis in existing_analyses:
        run_name = analysis["run_name"]
        git = new_results.get("grad_interference_temporal", {}).get(run_name, {})
        if not git:
            continue
        schedules = sorted(git.keys(), key=_sched_order)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=schedules,
            y=[git[s]["realign_speed"] for s in schedules],
            marker_color=[_color(s) for s in schedules],
            name=run_name,
        ))
        fig.update_layout(
            title=f"Gradient Re-Alignment Speed — {run_name}<br>"
                  "<sup>Steps until burst-vs-other cosine sim exceeds 0.5. "
                  "Lower = faster re-alignment = shallower.</sup>",
            xaxis_title="Schedule",
            yaxis_title="Re-Alignment Step",
            template="plotly_white", height=500,
        )
        _add_fig("realign_speed", fig)

    # ------------------------------------------------------------------
    # Chart 11: Gradient Projection — interference magnitude, useful learning,
    #           and interference ratio over time with error bands
    # ------------------------------------------------------------------

    def _proj_timeseries_fig(gp, schedules, metric_key, title, yaxis_label, hline=None):
        """Line + shaded-band figure for one projection metric across schedules."""
        fig = go.Figure()
        for sched in schedules:
            d = gp[sched]
            if not d.get("steps"):
                continue
            steps = d["steps"]
            y = d.get(metric_key, [])
            y_std = d.get(f"{metric_key}_std", [0.0] * len(steps))
            color = _color(sched)
            fig.add_trace(go.Scatter(
                x=steps, y=y, name=sched,
                line=dict(color=color, width=2), mode="lines",
            ))
            fig.add_trace(go.Scatter(
                x=steps + steps[::-1],
                y=[v + e for v, e in zip(y, y_std)] + [v - e for v, e in zip(y[::-1], y_std[::-1])],
                fill="toself", fillcolor=color, opacity=0.15,
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))
        if hline is not None:
            fig.add_hline(y=hline, line_dash="dot", line_color="gray",
                          annotation_text="random baseline")
        fig.update_layout(
            title=title, xaxis_title="Training Step", yaxis_title=yaxis_label,
            legend_title="Schedule", template="plotly_white", height=500,
        )
        return fig

    for analysis in existing_analyses:
        run_name = analysis["run_name"]
        gp = new_results.get("grad_projection", {}).get(run_name, {})
        if not gp:
            continue
        schedules = sorted(gp.keys(), key=_sched_order)

        _add_fig("grad_interference_magnitude", _proj_timeseries_fig(
            gp, schedules, "interference_magnitude",
            f"Gradient Interference Magnitude — {run_name}<br>"
            "<sup>||g_burst^parallel||: projection of burst gradient onto other-class direction. "
            "Higher = more damage to other-class parameters per step.</sup>",
            "Interference Magnitude ||g_parallel||",
        ))
        _add_fig("grad_interference_ratio", _proj_timeseries_fig(
            gp, schedules, "interference_ratio",
            f"Gradient Interference Ratio — {run_name}<br>"
            "<sup>||g_parallel|| / ||g_burst|| = |cos α|. "
            "Fraction of burst gradient interfering with other-class learning. Range [0,1].</sup>",
            "Interference Ratio (= |cos α|)", hline=0.5,
        ))
        _add_fig("grad_useful_learning", _proj_timeseries_fig(
            gp, schedules, "useful_learning",
            f"Gradient Useful Learning — {run_name}<br>"
            "<sup>||g_burst^perp||: orthogonal component — safe learning that does not conflict.</sup>",
            "Useful Learning ||g_perp||",
        ))

        # Bar: mean interference ratio during burst phase with error bars
        sched_labels, mean_ratios, std_ratios = [], [], []
        for sched in schedules:
            d = gp[sched]
            burst_ratios = [r for r, p in zip(d.get("interference_ratio", []), d.get("phases", []))
                            if p == PHASE_BURST and not np.isnan(r)]
            if burst_ratios:
                sched_labels.append(sched)
                mean_ratios.append(float(np.mean(burst_ratios)))
                std_ratios.append(float(np.std(burst_ratios)))

        fig_bar = go.Figure(go.Bar(
            x=sched_labels, y=mean_ratios,
            error_y=dict(type="data", array=std_ratios, visible=True),
            marker_color=[_color(s) for s in sched_labels],
        ))
        fig_bar.update_layout(
            title=f"Mean Gradient Interference Ratio During Burst — {run_name}<br>"
                  "<sup>Mean |cos α| across burst-phase steps. "
                  "Lower = burst gradient more orthogonal to other-class gradients.</sup>",
            xaxis_title="Schedule", yaxis_title="Mean Interference Ratio",
            template="plotly_white", height=500,
        )
        _add_fig("grad_interference_ratio_bar", fig_bar)

    # ------------------------------------------------------------------
    # Chart 10: Burst Position Comparison
    # ------------------------------------------------------------------
    bpc = new_results.get("burst_position_comparison", {})
    if bpc:
        metrics_to_compare = [
            ("reversion_auc", "Reversion AUC (lower = faster forgetting)"),
            ("ema_cliff_alpha", "EMA Cliff Alpha (higher = shallower)"),
            ("sharpness", "Critical Sharpness"),
            ("adl_readability", "ADL Readability"),
        ]
        for metric_key, metric_label in metrics_to_compare:
            fig = go.Figure()
            run_names = sorted(bpc.keys())
            for run_name in run_names:
                pos_data = bpc[run_name]
                schedules = sorted(pos_data["per_schedule"].keys(), key=_sched_order)
                burst_pcts = [int(s.replace("burst_", "")) for s in schedules]
                vals = [pos_data["per_schedule"][s].get(metric_key, float("nan"))
                        for s in schedules]
                burst_pos = pos_data["burst_pos"]
                fig.add_trace(go.Scatter(
                    x=burst_pcts, y=vals,
                    name=f"pos{burst_pos}",
                    mode="lines+markers",
                    line=dict(width=2),
                ))
            fig.update_layout(
                title=f"Burst Position Effect: {metric_label}<br>"
                      "<sup>Compares pos1 / pos2 / pos3 across all schedules</sup>",
                xaxis_title="Burst % (burstiness level)",
                yaxis_title=metric_label,
                legend_title="Burst Position",
                template="plotly_white", height=500,
            )
            _add_fig(f"burst_position_{metric_key}", fig)

    # ------------------------------------------------------------------
    # Summary: all new metrics vs burstiness (first run only)
    # ------------------------------------------------------------------
    if existing_analyses:
        analysis = existing_analyses[0]
        run_name = analysis["run_name"]
        sm = analysis.get("summary_metrics", {})
        schedules = sorted(sm.keys(), key=_sched_order)
        burst_pcts = [int(s.replace("burst_", "")) for s in schedules]

        def _get_scalar(metric_dict, sched, key):
            d = metric_dict.get(run_name, {}).get(sched, {})
            return d.get(key, float("nan")) if isinstance(d, dict) else float("nan")

        fig_summary = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                "Task Vector Transfer",
                "Forgetting Trajectory Dim",
                "Relearning AUC",
                "LMC Barrier",
                "Pruning Robustness AUC",
                "Grad Re-Alignment Speed",
            ],
        )
        colors = [_color(s) for s in schedules]

        def _add_scatter_summary(fig, row, col, y_vals):
            fig.add_trace(go.Scatter(
                x=burst_pcts, y=y_vals,
                mode="markers+lines",
                marker=dict(color=colors, size=10),
                line=dict(color="gray", width=1, dash="dot"),
                showlegend=False,
            ), row=row, col=col)

        _add_scatter_summary(fig_summary, 1, 1,
            [_get_scalar(new_results.get("task_vector_transfer", {}), s, "mean_transfer_acc") for s in schedules])
        _add_scatter_summary(fig_summary, 1, 2,
            [_get_scalar(new_results.get("forgetting_trajectory_dim", {}), s, "mean_dim") for s in schedules])
        _add_scatter_summary(fig_summary, 1, 3,
            [_get_scalar(new_results.get("relearning_efficiency", {}), s, "auc") for s in schedules])
        _add_scatter_summary(fig_summary, 2, 1,
            [_get_scalar(new_results.get("linear_mode_connectivity", {}), s, "barrier") for s in schedules])
        _add_scatter_summary(fig_summary, 2, 2,
            [_get_scalar(new_results.get("pruning_robustness", {}), s, "robustness_auc") for s in schedules])
        _add_scatter_summary(fig_summary, 2, 3,
            [_get_scalar(new_results.get("grad_interference_temporal", {}), s, "realign_speed") for s in schedules])

        fig_summary.update_xaxes(title_text="Burst %")
        fig_summary.update_layout(
            title=f"Summary: All New Metrics vs Burstiness — {run_name}",
            template="plotly_white", height=800,
        )
        _add_fig("summary_new_metrics", fig_summary)

    # ------------------------------------------------------------------
    # Assemble HTML dashboard
    # ------------------------------------------------------------------
    html_parts = ["""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Burstiness New Metrics Dashboard</title>
<style>
  body { font-family: Arial, sans-serif; margin: 20px; background: #f0f2f5; }
  h1 { color: #1a1a2e; font-size: 1.8em; }
  h2 { color: #16213e; margin-top: 40px; font-size: 1.3em; }
  .chart-container {
    background: white; border-radius: 10px; padding: 20px;
    margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  }
  .metric-info {
    background: #f8f9ff; border-left: 4px solid #4a90d9;
    padding: 12px 16px; margin: 8px 0 16px 0;
    border-radius: 0 6px 6px 0; font-size: 0.9em; color: #333;
  }
  .metric-info .what { margin-bottom: 8px; }
  .metric-info .interp { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 8px; }
  .metric-info .interp span { flex: 1; min-width: 200px; }
  .metric-info .high { color: #1a7a4a; }
  .metric-info .low { color: #c0392b; }
  .metric-info .limits { color: #777; font-style: italic; font-size: 0.88em; }
  .toc { background: white; border-radius: 10px; padding: 20px; margin: 20px 0;
         box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
  .toc a { display: block; margin: 4px 0; color: #1565c0; text-decoration: none; }
  .toc a:hover { text-decoration: underline; }
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 12px;
    font-size: 0.75em; font-weight: bold; margin-left: 8px;
    background: #e3f2fd; color: #1565c0;
  }
  .badge.no-ckpt { background: #e8f5e9; color: #2e7d32; }
</style>
</head>
<body>
<h1>Burstiness: 10 New Mechanistic Metrics</h1>
<p style="color:#555; max-width:900px;">
  Post-hoc analysis of three burst experiment runs (pos1, pos2, pos3).
  These 10 metrics complement the 5 already in the deep analysis dashboard
  (ADL, gradient interference, EMA interpolation, critical sharpness, weight delta rank).
  Together they build a mechanistic picture of <em>why</em> burstiness causes shallow learning
  and fast forgetting.
</p>
<div class="toc">
  <strong>Contents:</strong>
"""]

    for i, (key, title, _) in enumerate(all_figs):
        desc = _METRIC_DESCRIPTIONS.get(key, {})
        needs_ckpt = key in {
            "task_vector_transfer", "forgetting_trajectory_dim",
            "relearning_efficiency", "relearning_auc",
            "linear_mode_connectivity", "lmc_barrier",
            "pruning_robustness",
        }
        badge = '<span class="badge">needs checkpoints</span>' if needs_ckpt else '<span class="badge no-ckpt">from existing data</span>'
        anchor = f"chart_{i}"
        html_parts.append(
            f'  <a href="#{anchor}">{i+1}. {title}{badge}</a>\n'
        )

    html_parts.append("</div>\n")

    for i, (key, title, fig) in enumerate(all_figs):
        desc = _METRIC_DESCRIPTIONS.get(key, {})
        anchor = f"chart_{i}"
        html_parts.append(f'<div class="chart-container" id="{anchor}">\n')
        html_parts.append(f'<h2>{i+1}. {title}</h2>\n')

        if desc:
            html_parts.append('<div class="metric-info">\n')
            if desc.get("what"):
                html_parts.append(f'<div class="what"><strong>What this measures:</strong> {desc["what"]}</div>\n')
            if desc.get("high") or desc.get("low"):
                html_parts.append('<div class="interp">\n')
                if desc.get("high"):
                    html_parts.append(f'<span class="high"><strong>↑ High:</strong> {desc["high"]}</span>\n')
                if desc.get("low"):
                    html_parts.append(f'<span class="low"><strong>↓ Low:</strong> {desc["low"]}</span>\n')
                html_parts.append('</div>\n')
            if desc.get("limitations"):
                html_parts.append(f'<div class="limits"><strong>Limitations:</strong> {desc["limitations"]}</div>\n')
            html_parts.append('</div>\n')

        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=(i == 0)))
        html_parts.append("</div>\n")

    html_parts.append("</body></html>")

    html_path = out_dir / "dashboard.html"
    with open(html_path, "w") as f:
        f.write("".join(html_parts))
    print(f"\nDashboard saved: {html_path}", flush=True)
    print(f"Charts saved: {charts_dir}", flush=True)

    from burst.plot_utils import write_text_report
    write_text_report(all_figs, out_dir / "dashboard.txt",
                      dashboard_title="Burstiness: 10 New Mechanistic Metrics",
                      descriptions=_METRIC_DESCRIPTIONS)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def analyse_run(
    run_dir: Path,
    existing_analysis: dict,
    n_seeds: int = 3,
    n_prune_levels: int = 10,
    relearn_steps: int = 50,
) -> dict:
    """Run all 10 new metrics on a single run directory."""
    print(f"\n{'='*60}", flush=True)
    print(f"Analysing: {run_dir.name}", flush=True)
    print(f"{'='*60}", flush=True)

    from burst.train_utils import resolve_run_paths
    cfg_path, logs_dir, _ = resolve_run_paths(run_dir)
    with open(cfg_path) as f:
        run_cfg = json.load(f)

    rc = parse_run_config(run_cfg)
    base_cfg = rc["base_cfg"]

    with open(logs_dir / "_data.pkl", "rb") as f:
        target_pool, bg_pool, _, _, _ = pickle.load(f)

    other_docs_BL = np.concatenate(list(bg_pool.values()))
    burst_docs_BL = np.concatenate(list(target_pool.values()))
    prompt_len = run_cfg["task_info"]["prompt_len"]

    with open(logs_dir / "all_results.pkl", "rb") as f:
        all_results = pickle.load(f)

    ckpt_root = logs_dir / "checkpoints"
    run_name = run_dir.name

    result = {"run_name": run_name, "burst_pos": rc["burst_pos"]}

    # Metrics from existing data (no checkpoints needed)
    print("\n[6/10] Pairwise gradient separation...", flush=True)
    result["pairwise_grad_separation"] = compute_pairwise_grad_separation(all_results)

    print("\n[7/10] Forgetting speed decomposition...", flush=True)
    result["forgetting_speed_decomposition"] = compute_forgetting_speed_decomposition(all_results)

    print("\n[8/10] Per-layer interference localisation...", flush=True)
    result["layer_interference_localisation"] = compute_layer_interference_localisation(all_results)

    print("\n[9/10] Gradient interference temporal dynamics...", flush=True)
    result["grad_interference_temporal"] = compute_grad_interference_temporal(all_results)

    print("\n[11/11] Gradient projection (OGD-style interference decomposition)...", flush=True)
    result["grad_projection"] = compute_grad_projection_metrics(all_results)

    print("\n[10/10] Burst position comparison (uses existing results)...", flush=True)
    # This is computed globally after all runs are processed

    if not ckpt_root.exists():
        print(f"  No checkpoints directory — skipping checkpoint-based metrics.", flush=True)
        return result

    print("\n[1/10] Task vector transfer...", flush=True)
    result["task_vector_transfer"] = compute_task_vector_transfer(
        ckpt_root, all_results, burst_docs_BL, prompt_len, n_seeds=n_seeds)

    print("\n[2/10] Forgetting trajectory dimensionality...", flush=True)
    result["forgetting_trajectory_dim"] = compute_forgetting_trajectory_dim(
        ckpt_root, all_results, n_seeds=n_seeds)

    # DISABLED — re-enable by uncommenting the block below
    # print("\n[3/10] Relearning efficiency...", flush=True)
    # result["relearning_efficiency"] = compute_relearning_efficiency(
    #     ckpt_root, all_results, burst_docs_BL, other_docs_BL, prompt_len,
    #     n_seeds=n_seeds, relearn_steps=relearn_steps)

    print("\n[4/10] Linear mode connectivity...", flush=True)
    result["linear_mode_connectivity"] = compute_linear_mode_connectivity(
        ckpt_root, all_results, burst_docs_BL, prompt_len, n_seeds=n_seeds)

    print("\n[5/10] Pruning robustness...", flush=True)
    result["pruning_robustness"] = compute_pruning_robustness(
        ckpt_root, all_results, burst_docs_BL, prompt_len,
        n_seeds=n_seeds, n_prune_levels=n_prune_levels)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="10 new post-hoc mechanistic metrics for burstiness runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--existing-results", type=Path, required=True,
                        help="Path to deep_analysis_combined/results.pkl")
    parser.add_argument("--out-dir", type=Path, default=Path("data/new_metrics_combined"))
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-prune-levels", type=int, default=10)
    parser.add_argument("--relearn-steps", type=int, default=50)
    args = parser.parse_args()

    with open(args.existing_results, "rb") as f:
        existing_analyses = pickle.load(f)

    existing_by_name = {a["run_name"]: a for a in existing_analyses}

    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_run_results = []
    for run_dir in args.run_dirs:
        run_dir = Path(run_dir)
        existing = existing_by_name.get(run_dir.name, {})
        t0 = time.time()
        r = analyse_run(
            run_dir, existing,
            n_seeds=args.n_seeds,
            n_prune_levels=args.n_prune_levels,
            relearn_steps=args.relearn_steps,
        )
        per_run_results.append(r)
        print(f"  Completed {run_dir.name} in {time.time() - t0:.1f}s", flush=True)

    # Metric 10: burst position comparison (global, uses existing_analyses)
    print("\n[10/10] Burst position comparison...", flush=True)
    bpc = compute_burst_position_comparison(existing_analyses)
    for r in per_run_results:
        r["burst_position_comparison"] = bpc

    # Restructure for dashboard: per-metric, per-run-name
    new_results: dict[str, dict] = {}
    for r in per_run_results:
        run_name = r["run_name"]
        for metric_key in [
            "task_vector_transfer", "forgetting_trajectory_dim",
            "relearning_efficiency", "linear_mode_connectivity",
            "pruning_robustness", "pairwise_grad_separation",
            "forgetting_speed_decomposition", "layer_interference_localisation",
            "grad_interference_temporal", "grad_projection",
        ]:
            if metric_key in r:
                new_results.setdefault(metric_key, {})[run_name] = r[metric_key]
    new_results["burst_position_comparison"] = bpc

    results_path = args.out_dir / "results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump({"per_run": per_run_results, "by_metric": new_results}, f)
    print(f"\nResults saved: {results_path}", flush=True)

    print("\nGenerating dashboard...", flush=True)
    make_dashboard(new_results, existing_analyses, args.out_dir)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
