r"""Basin geometry metrics for burstiness runs.

Implements three metrics from Kim et al. (2025) "Rethinking Safety in LLM
Fine-tuning: An Optimization Perspective", adapted to measure whether bursty
training produces shallower (narrower) parameter basins than mixed training.

Metrics:
  1. Gaussian noise robustness  — add N(0, sigma^2) noise to weights at peak burst,
                                   measure burst vs other accuracy at each sigma.
                                   Bursty models should lose burst accuracy at
                                   smaller sigma than mixed models (narrower basin).

  2. Weight drift vs forgetting  — compute ||theta_peak - theta_pre_burst||_2 and
                                   correlate with reversion_auc across seeds
                                   and schedules.

  3. Loss surface visualisation  — Li et al. (2018) filter-normalised 2D slice
                                   of the loss landscape at peak burst, evaluated
                                   on burst vs other prompts separately.
                                   Bursty → narrow/steep for burst prompts.

Usage:
    uv run python burst/basin_metrics.py data/burst_d3_pos3_<tag> \\
        [data/burst_d3_pos1_<tag> ...] \\
        --out-dir data/basin_metrics_combined \\
        --n-seeds 3

Dimension key:
    B: batch_size
    L: sequence_length (doc_len)
    N: n_embd (model dimension)
    V: vocab_size
    P: n_params (total parameters, flattened)
    S: number of noise sigma levels
    A: grid resolution for loss surface (n_alpha x n_alpha)
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch

from burst.config import (
    parse_run_config,
)
from burst.core.train_utils import DEVICE, load_net, resolve_run_paths
from burst.dev._shared import (
    ckpt_files as _ckpt_files,
)
from burst.dev._shared import (
    cross_entropy_loss as _cross_entropy_loss,
)
from burst.dev._shared import (
    free_gen_acc as _free_gen_acc,
)
from burst.dev._shared import (
    sched_color as _color,
)
from burst.dev._shared import (
    sched_order as _sched_order,
)

_rng = np.random.default_rng()

NOISE_SIGMAS: list[float] = [0.0, 0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.010, 0.015, 0.020]
NOISE_SIGMA_THRESHOLD: float = 0.004
SURFACE_GRID: int = 15
SURFACE_RANGE: float = 0.02
_MIN_DIM_FOR_FILTER_NORM: int = 2
_NORM_EPS: float = 1e-10
_MIN_SAMPLES_FOR_CORR: int = 2


def _filter_normalise(
    direction: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Filter-normalise a random direction to match the scale of reference weights.

    For each parameter tensor, normalise each filter (row for 2D, full tensor
    for 1D) so its norm equals the corresponding filter norm in the reference.
    This makes the perturbation scale-invariant across layers.
    """
    normed = {}
    for name, d in direction.items():
        ref = reference[name].float()
        d_f = d.float()
        if d_f.dim() >= _MIN_DIM_FOR_FILTER_NORM:
            # Treat first dimension as filter dimension
            d_norms = d_f.view(d_f.shape[0], -1).norm(dim=1, keepdim=True)
            ref_norms = ref.view(ref.shape[0], -1).norm(dim=1, keepdim=True)
            scale = ref_norms / (d_norms + 1e-10)
            normed[name] = (d_f.view(d_f.shape[0], -1) * scale).view(d_f.shape)
        else:
            d_norm = d_f.norm()
            ref_norm = ref.norm()
            normed[name] = d_f * (ref_norm / (d_norm + 1e-10))
    return normed


# ---------------------------------------------------------------------------
# Metric 1+5: Gaussian Noise Robustness
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_noise_robustness(
    ckpt_root: Path,
    all_results: list[dict],
    burst_docs_BL: np.ndarray,
    other_docs_BL: np.ndarray,
    prompt_len: int,
    n_seeds: int = 3,
    sigmas: list[float] = NOISE_SIGMAS,
    n_eval_docs: int = 256,
) -> dict:
    """Measure burst and other accuracy under Gaussian weight noise at peak burst.

    For each schedule and seed, loads the peak-burst checkpoint, adds
    N(0, sigma^2I) noise to all parameters, and evaluates both burst and other
    accuracy. Repeats across sigma levels.

    Bursty models (burst_100) should lose burst accuracy at smaller sigma than
    mixed models (burst_25), mirroring Kim et al. Figure 5.

    Returns per-schedule mean accuracy curves for burst and other classes.
    """
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)

    n_burst = min(n_eval_docs, burst_docs_BL.shape[0])
    n_other = min(n_eval_docs, other_docs_BL.shape[0])
    burst_idx = _rng.choice(burst_docs_BL.shape[0], n_burst, replace=False)
    other_idx = _rng.choice(other_docs_BL.shape[0], n_other, replace=False)
    burst_eval = burst_docs_BL[burst_idx]
    other_eval = other_docs_BL[other_idx]

    results = {}

    for sched in schedules:
        sched_results = jobs_by_schedule[sched]
        # burst_acc_curves[seed_i] = list of accs, one per sigma
        burst_acc_curves: list[list[float]] = []
        other_acc_curves: list[list[float]] = []
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

            base_sd = {
                k: v.float()
                for k, v in torch.load(
                    str(files[peak_step]), map_location="cpu", weights_only=True
                ).items()
            }

            net = load_net(cfg, str(files[peak_step]))

            burst_accs: list[float] = []
            other_accs: list[float] = []

            for sigma in sigmas:
                if sigma == 0.0:
                    net.load_state_dict({k: v.to(DEVICE) for k, v in base_sd.items()})
                else:
                    noisy_sd = {
                        k: (v + torch.randn_like(v) * sigma).to(DEVICE) for k, v in base_sd.items()
                    }
                    net.load_state_dict(noisy_sd)

                burst_accs.append(_free_gen_acc(net, burst_eval, prompt_len))
                other_accs.append(_free_gen_acc(net, other_eval, prompt_len))

            burst_acc_curves.append(burst_accs)
            other_acc_curves.append(other_accs)
            seeds_done += 1

        if burst_acc_curves:
            mean_burst = [
                float(np.mean([c[i] for c in burst_acc_curves])) for i in range(len(sigmas))
            ]
            mean_other = [
                float(np.mean([c[i] for c in other_acc_curves])) for i in range(len(sigmas))
            ]
            std_burst = [
                float(np.std([c[i] for c in burst_acc_curves])) for i in range(len(sigmas))
            ]
            std_other = [
                float(np.std([c[i] for c in other_acc_curves])) for i in range(len(sigmas))
            ]
        else:
            mean_burst = mean_other = [float("nan")] * len(sigmas)
            std_burst = std_other = [float("nan")] * len(sigmas)

        results[sched] = {
            "sigmas": sigmas,
            "mean_burst_accs": mean_burst,
            "mean_other_accs": mean_other,
            "std_burst_accs": std_burst,
            "std_other_accs": std_other,
            "burst_acc_curves": burst_acc_curves,
            "other_acc_curves": other_acc_curves,
        }

    return results


# ---------------------------------------------------------------------------
# Metric 1b: Directed Noise Robustness (along burst weight-delta)
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_directed_noise_robustness(
    ckpt_root: Path,
    all_results: list[dict],
    burst_docs_BL: np.ndarray,
    _other_docs_BL: np.ndarray,
    prompt_len: int,
    n_seeds: int = 3,
    epsilons: list[float] | None = None,
    n_eval_docs: int = 256,
) -> dict:
    """Directed noise along the burst weight-delta τ = theta_peak - theta_pre.

    For each epsilon, evaluates:
      theta' = theta_peak + ε * τ/||τ||   (undo direction)
      theta' = theta_peak + ε * r/||r||   (random orthogonal direction)

    If learning is shallow, undo-direction noise kills accuracy faster than
    random-direction noise. The ratio is the "narrowness" measure.
    """
    if epsilons is None:
        epsilons = [0.0, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05]

    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)

    n_burst = min(n_eval_docs, burst_docs_BL.shape[0])
    burst_idx = _rng.choice(burst_docs_BL.shape[0], n_burst, replace=False)
    burst_eval = burst_docs_BL[burst_idx]

    results = {}

    for sched in schedules:
        sched_results = jobs_by_schedule[sched]
        undo_curves: list[list[float]] = []
        random_curves: list[list[float]] = []
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
            pre_step = available[0]
            peak_step = min(available, key=lambda x: abs(x - (T - 1)))
            cfg = r["config"]

            sd_pre = {
                k: v.float()
                for k, v in torch.load(
                    str(files[pre_step]), map_location="cpu", weights_only=True
                ).items()
            }
            sd_peak = {
                k: v.float()
                for k, v in torch.load(
                    str(files[peak_step]), map_location="cpu", weights_only=True
                ).items()
            }

            tau_flat = torch.cat([(sd_peak[k] - sd_pre[k]).view(-1) for k in sd_peak])
            tau_norm = tau_flat.norm()
            if tau_norm < 1e-10:
                continue
            tau_unit = tau_flat / tau_norm

            rand_dir = torch.randn_like(tau_flat)
            rand_dir -= (rand_dir @ tau_unit) * tau_unit
            rand_norm = rand_dir.norm()
            if rand_norm < 1e-10:
                continue
            rand_unit = rand_dir / rand_norm

            net = load_net(cfg, str(files[peak_step]))

            undo_accs: list[float] = []
            rand_accs: list[float] = []

            for eps in epsilons:
                if eps == 0.0:
                    acc = _free_gen_acc(net, burst_eval, prompt_len)
                    undo_accs.append(acc)
                    rand_accs.append(acc)
                    continue

                sd_undo = {}
                sd_rand = {}
                offset = 0
                for k in sd_peak:
                    numel = sd_peak[k].numel()
                    t_chunk = tau_unit[offset : offset + numel].view(sd_peak[k].shape)
                    r_chunk = rand_unit[offset : offset + numel].view(sd_peak[k].shape)
                    sd_undo[k] = (sd_peak[k] + eps * t_chunk).to(DEVICE)
                    sd_rand[k] = (sd_peak[k] + eps * r_chunk).to(DEVICE)
                    offset += numel

                net.load_state_dict(sd_undo)
                undo_accs.append(_free_gen_acc(net, burst_eval, prompt_len))

                net.load_state_dict(sd_rand)
                rand_accs.append(_free_gen_acc(net, burst_eval, prompt_len))

            undo_curves.append(undo_accs)
            random_curves.append(rand_accs)
            seeds_done += 1

        if undo_curves:
            mean_undo = [float(np.mean([c[i] for c in undo_curves])) for i in range(len(epsilons))]
            mean_rand = [
                float(np.mean([c[i] for c in random_curves])) for i in range(len(epsilons))
            ]
            narrowness = []
            for u, r in zip(mean_undo, mean_rand, strict=True):
                base = mean_undo[0] if mean_undo[0] > 0 else 1.0
                drop_u = base - u
                drop_r = base - r
                narrowness.append(drop_u / (drop_r + 1e-10) if drop_r > 1e-10 else 0.0)
        else:
            mean_undo = mean_rand = [float("nan")] * len(epsilons)
            narrowness = [float("nan")] * len(epsilons)

        results[sched] = {
            "epsilons": epsilons,
            "mean_undo_accs": mean_undo,
            "mean_random_accs": mean_rand,
            "narrowness_ratio": narrowness,
            "undo_curves": undo_curves,
            "random_curves": random_curves,
        }

    return results


# ---------------------------------------------------------------------------
# Metric 2: Weight Drift vs Forgetting Correlation
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_weight_drift_correlation(
    ckpt_root: Path,
    all_results: list[dict],
    n_seeds: int | None = None,
) -> dict:
    """Correlate ||theta_peak - theta_pre_burst||_2 with reversion_auc.

    For each (schedule, seed), loads the earliest and peak-burst checkpoints,
    computes the L2 weight drift, and pairs it with the reversion_auc from
    the training log.

    Returns:
        per_schedule: dict mapping schedule → list of (drift, reversion_auc) pairs
        correlation:  Pearson r across all (schedule, seed) pairs
        scatter_data: flat lists of drift and auc for scatter plotting

    """
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)

    all_drifts: list[float] = []
    all_aucs: list[float] = []
    all_labels: list[str] = []
    per_schedule: dict[str, dict] = {}

    for sched in schedules:
        sched_results = jobs_by_schedule[sched]
        drifts, aucs = [], []
        seeds_done = 0

        for r in sched_results:
            if n_seeds is not None and seeds_done >= n_seeds:
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
            pre_step = available[0]
            peak_step = min(available, key=lambda x: abs(x - (T - 1)))

            pre_sd = {
                k: v.float()
                for k, v in torch.load(
                    str(files[pre_step]), map_location="cpu", weights_only=True
                ).items()
            }
            peak_sd = {
                k: v.float()
                for k, v in torch.load(
                    str(files[peak_step]), map_location="cpu", weights_only=True
                ).items()
            }

            drift = float(sum((peak_sd[k] - pre_sd[k]).norm().item() ** 2 for k in peak_sd) ** 0.5)

            auc = r.get("reversion_auc", float("nan"))
            if not np.isnan(auc):
                drifts.append(drift)
                aucs.append(auc)
                all_drifts.append(drift)
                all_aucs.append(auc)
                all_labels.append(sched)
                seeds_done += 1

        per_schedule[sched] = {"drifts": drifts, "aucs": aucs}

    r_val = float(np.corrcoef(all_drifts, all_aucs)[0, 1]) if len(all_drifts) >= 2 else float("nan")

    return {
        "per_schedule": per_schedule,
        "correlation_r": r_val,
        "scatter_drifts": all_drifts,
        "scatter_aucs": all_aucs,
        "scatter_labels": all_labels,
    }


# ---------------------------------------------------------------------------
# Metric 3: Loss Surface Visualisation (Li et al. 2018)
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_loss_surface(
    ckpt_root: Path,
    all_results: list[dict],
    burst_docs_BL: np.ndarray,
    other_docs_BL: np.ndarray,
    n_seeds: int = 2,
    grid_size: int = SURFACE_GRID,
    surface_range: float = SURFACE_RANGE,
    n_eval_docs: int = 128,
) -> dict:
    """2D filter-normalised loss surface at peak burst.

    For each (schedule, seed):
      1. Load peak-burst checkpoint theta₀.
      2. Sample two random filter-normalised directions δ₁, δ_2.
      3. Evaluate loss on burst and other prompts at theta₀ + alpha·δ₁ + β·δ_2
         for (alpha, β) on a grid_size x grid_size grid in [-range, +range]².
      4. Return the 2D loss arrays.

    Bursty models should show a narrower/steeper surface for burst prompts
    than mixed models.

    Returns per-schedule mean surface arrays and sharpness metrics.
    """
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)

    alphas = np.linspace(-surface_range, surface_range, grid_size)
    betas = np.linspace(-surface_range, surface_range, grid_size)

    n_burst = min(n_eval_docs, burst_docs_BL.shape[0])
    n_other = min(n_eval_docs, other_docs_BL.shape[0])
    burst_idx = _rng.choice(burst_docs_BL.shape[0], n_burst, replace=False)
    other_idx = _rng.choice(other_docs_BL.shape[0], n_other, replace=False)
    burst_eval = burst_docs_BL[burst_idx]
    other_eval = other_docs_BL[other_idx]

    results = {}

    for sched in schedules:
        sched_results = jobs_by_schedule[sched]
        burst_surfaces: list[np.ndarray] = []
        other_surfaces: list[np.ndarray] = []
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

            base_sd = {
                k: v.float().cpu()
                for k, v in torch.load(
                    str(files[peak_step]), map_location="cpu", weights_only=True
                ).items()
            }

            # Sample two random directions and filter-normalise
            dir1_raw = {k: torch.randn_like(v) for k, v in base_sd.items()}
            dir2_raw = {k: torch.randn_like(v) for k, v in base_sd.items()}
            dir1 = _filter_normalise(dir1_raw, base_sd)
            dir2 = _filter_normalise(dir2_raw, base_sd)

            net = load_net(cfg, str(files[peak_step]))

            burst_surface = np.zeros((grid_size, grid_size))
            other_surface = np.zeros((grid_size, grid_size))

            for i, alpha in enumerate(alphas):
                for j, beta in enumerate(betas):
                    perturbed_sd = {
                        k: (base_sd[k] + alpha * dir1[k] + beta * dir2[k]).to(DEVICE)
                        for k in base_sd
                    }
                    net.load_state_dict(perturbed_sd)
                    burst_surface[i, j] = _cross_entropy_loss(net, burst_eval)
                    other_surface[i, j] = _cross_entropy_loss(net, other_eval)

            burst_surfaces.append(burst_surface)
            other_surfaces.append(other_surface)
            seeds_done += 1

            # Sharpness: max loss - centre loss (at alpha=β=0)
            centre_i = grid_size // 2
            burst_sharpness = float(burst_surface.max() - burst_surface[centre_i, centre_i])
            other_sharpness = float(other_surface.max() - other_surface[centre_i, centre_i])

        centre_i = grid_size // 2
        if burst_surfaces:
            mean_burst_surface = np.mean(burst_surfaces, axis=0)
            mean_other_surface = np.mean(other_surfaces, axis=0)
            burst_sharpness = float(
                mean_burst_surface.max() - mean_burst_surface[centre_i, centre_i]
            )
            other_sharpness = float(
                mean_other_surface.max() - mean_other_surface[centre_i, centre_i]
            )
            per_seed_burst_sharpness = [
                float(s.max() - s[centre_i, centre_i]) for s in burst_surfaces
            ]
            per_seed_other_sharpness = [
                float(s.max() - s[centre_i, centre_i]) for s in other_surfaces
            ]
        else:
            mean_burst_surface = np.full((grid_size, grid_size), float("nan"))
            mean_other_surface = np.full((grid_size, grid_size), float("nan"))
            burst_sharpness = other_sharpness = float("nan")
            per_seed_burst_sharpness = []
            per_seed_other_sharpness = []

        results[sched] = {
            "alphas": alphas.tolist(),
            "betas": betas.tolist(),
            "mean_burst_surface": mean_burst_surface.tolist(),
            "mean_other_surface": mean_other_surface.tolist(),
            "burst_sharpness": burst_sharpness,
            "other_sharpness": other_sharpness,
            "per_seed_burst_sharpness": per_seed_burst_sharpness,
            "per_seed_other_sharpness": per_seed_other_sharpness,
        }

    return results


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


def make_dashboard(results: dict, out_dir: Path) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from burst.dev.plot_utils import save_png as _save_png

    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    all_figs: list[tuple[str, go.Figure]] = []

    def _add(key: str, fig: go.Figure) -> None:
        all_figs.append((key, fig))
        _save_png(fig, str(charts_dir / f"{key}.png"))

    for run_name, run_data in results.items():
        # ------------------------------------------------------------------
        # Metric 1+5: Noise Robustness
        # ------------------------------------------------------------------
        nr = run_data.get("noise_robustness", {})
        if nr:
            schedules = sorted(nr.keys(), key=_sched_order)

            # Burst accuracy vs sigma
            fig = go.Figure()
            fig_burst_delta = go.Figure()
            for sched in schedules:
                d = nr[sched]
                diff = [b - o for b, o in zip(d["mean_burst_accs"], d["mean_other_accs"], strict=True)]  # noqa: E501
                diff_std = [
                    float(np.sqrt(sb**2 + so**2))
                    for sb, so in zip(d["std_burst_accs"], d["std_other_accs"], strict=True)
                ]
                fig.add_trace(
                    go.Scatter(
                        x=d["sigmas"],
                        y=d["mean_burst_accs"],
                        name=sched,
                        line={"color": _color(sched), "width": 2},
                        mode="lines+markers",
                        error_y={"array": d["std_burst_accs"], "visible": True, "thickness": 1},
                    )
                )
                fig_burst_delta.add_trace(
                    go.Scatter(
                        x=d["sigmas"],
                        y=diff,
                        name=sched,
                        line={"color": _color(sched), "width": 2},
                        mode="lines+markers",
                        error_y={"array": diff_std, "visible": True, "thickness": 1},
                    )
                )
            fig.add_vline(
                x=0.004,
                line_dash="dash",
                line_color="gray",
                annotation_text="sigma=0.004 (Kim et al. safety threshold)",
            )
            fig.update_layout(
                title=f"Burst Accuracy Under Gaussian Weight Noise — {run_name}",
                xaxis_title="Noise sigma",
                yaxis_title="Burst Accuracy",
                legend_title="Schedule",
                template="plotly_white",
                height=500,
            )
            _add(f"noise_burst_{run_name}", fig)
            fig_burst_delta.add_vline(
                x=0.004, line_dash="dash", line_color="gray", annotation_text="sigma=0.004"
            )
            fig_burst_delta.update_layout(
                title=f"Noise Δ (Burst - Other) Under Gaussian Weight Noise — {run_name}",
                xaxis_title="Noise sigma",
                yaxis_title="Δ(Burst - Other)",
                legend_title="Schedule",
                template="plotly_white",
                height=500,
            )
            _add(f"noise_burst_delta_{run_name}", fig_burst_delta)

            # Other accuracy vs sigma
            fig = go.Figure()
            fig_other_delta = go.Figure()
            for sched in schedules:
                d = nr[sched]
                diff = [b - o for b, o in zip(d["mean_burst_accs"], d["mean_other_accs"], strict=True)]  # noqa: E501
                diff_std = [
                    float(np.sqrt(sb**2 + so**2))
                    for sb, so in zip(d["std_burst_accs"], d["std_other_accs"], strict=True)
                ]
                fig.add_trace(
                    go.Scatter(
                        x=d["sigmas"],
                        y=d["mean_other_accs"],
                        name=sched,
                        line={"color": _color(sched), "width": 2},
                        mode="lines+markers",
                        error_y={"array": d["std_other_accs"], "visible": True, "thickness": 1},
                    )
                )
                fig_other_delta.add_trace(
                    go.Scatter(
                        x=d["sigmas"],
                        y=diff,
                        name=sched,
                        line={"color": _color(sched), "width": 2},
                        mode="lines+markers",
                        error_y={"array": diff_std, "visible": True, "thickness": 1},
                    )
                )
            fig.add_vline(x=0.004, line_dash="dash", line_color="gray", annotation_text="sigma=0.004")  # noqa: E501
            fig.update_layout(
                title=f"Other-Class Accuracy Under Gaussian Weight Noise — {run_name}",
                xaxis_title="Noise sigma",
                yaxis_title="Other-Class Accuracy",
                legend_title="Schedule",
                template="plotly_white",
                height=500,
            )
            _add(f"noise_other_{run_name}", fig)
            fig_other_delta.add_vline(
                x=0.004, line_dash="dash", line_color="gray", annotation_text="sigma=0.004"
            )
            fig_other_delta.update_layout(
                title=f"Noise Δ (Burst - Other) [Other-Class View] — {run_name}",
                xaxis_title="Noise sigma",
                yaxis_title="Δ(Burst - Other)",
                legend_title="Schedule",
                template="plotly_white",
                height=500,
            )
            _add(f"noise_other_delta_{run_name}", fig_other_delta)

            # Differential sensitivity: burst drop / other drop at sigma=0.004
            sigma_idx = NOISE_SIGMAS.index(0.004) if 0.004 in NOISE_SIGMAS else -1
            if sigma_idx >= 0:
                burst_drops = [
                    nr[s]["mean_burst_accs"][0] - nr[s]["mean_burst_accs"][sigma_idx]
                    for s in schedules
                ]
                other_drops = [
                    nr[s]["mean_other_accs"][0] - nr[s]["mean_other_accs"][sigma_idx]
                    for s in schedules
                ]
                diff_drops = [b - o for b, o in zip(burst_drops, other_drops, strict=True)]
                colors = [_color(s) for s in schedules]
                fig_diff_burst = go.Figure(
                    go.Bar(x=schedules, y=burst_drops, marker_color=colors, showlegend=False)
                )
                fig_diff_burst.update_layout(
                    title=f"Noise Sensitivity (Burst Drop) at sigma=0.004 — {run_name}<br>"
                    "<sup>Larger drop = narrower burst basin.</sup>",
                    xaxis_title="Schedule",
                    yaxis_title="Burst Accuracy Drop",
                    template="plotly_white",
                    height=500,
                )
                _add(f"noise_differential_burst_{run_name}", fig_diff_burst)
                fig_diff_other = go.Figure(
                    go.Bar(x=schedules, y=other_drops, marker_color=colors, showlegend=False)
                )
                fig_diff_other.update_layout(
                    title=f"Noise Sensitivity (Other Drop) at sigma=0.004 — {run_name}",
                    xaxis_title="Schedule",
                    yaxis_title="Other Accuracy Drop",
                    template="plotly_white",
                    height=500,
                )
                _add(f"noise_differential_other_{run_name}", fig_diff_other)
                fig_diff_delta = go.Figure(
                    go.Bar(x=schedules, y=diff_drops, marker_color=colors, showlegend=False)
                )
                fig_diff_delta.update_layout(
                    title=f"Noise Sensitivity Δ (Burst - Other Drop) at sigma=0.004 — {run_name}<br>"  # noqa: E501
                    "<sup>Positive = burst basin narrower than other-class basin.</sup>",
                    xaxis_title="Schedule",
                    yaxis_title="Δ(Burst - Other) Drop",
                    template="plotly_white",
                    height=500,
                )
                _add(f"noise_differential_delta_{run_name}", fig_diff_delta)

        # ------------------------------------------------------------------
        # Metric 1b: Directed Noise Robustness
        # ------------------------------------------------------------------
        dn = run_data.get("directed_noise", {})
        if dn:
            schedules = sorted(dn.keys(), key=_sched_order)
            fig = make_subplots(
                rows=1, cols=2, subplot_titles=["Undo vs Random Direction", "Narrowness Ratio"]
            )
            for sched in schedules:
                d = dn[sched]
                fig.add_trace(
                    go.Scatter(
                        x=d["epsilons"],
                        y=d["mean_undo_accs"],
                        name=f"{sched} undo",
                        line={"color": _color(sched), "width": 2},
                        mode="lines+markers",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=d["epsilons"],
                        y=d["mean_random_accs"],
                        name=f"{sched} random",
                        line={"color": _color(sched), "width": 2, "dash": "dot"},
                        mode="lines+markers",
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=d["epsilons"],
                        y=d["narrowness_ratio"],
                        name=sched,
                        line={"color": _color(sched), "width": 2},
                        mode="lines+markers",
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )
            fig.update_layout(
                title=f"Directed Noise: Undo vs Random — {run_name}<br>"
                "<sup>Solid: undo direction. Dotted: random orthogonal. "
                "Ratio > 1 = narrow/shallow basin.</sup>",
                template="plotly_white",
                height=500,
            )
            fig.update_xaxes(title_text="ε", row=1, col=1)
            fig.update_xaxes(title_text="ε", row=1, col=2)
            fig.update_yaxes(title_text="Burst Accuracy", row=1, col=1)
            fig.update_yaxes(title_text="Narrowness (undo_drop / rand_drop)", row=1, col=2)
            _add(f"directed_noise_{run_name}", fig)

        # ------------------------------------------------------------------
        # Metric 2: Weight Drift Correlation
        # ------------------------------------------------------------------
        wd = run_data.get("weight_drift", {})
        if wd:
            schedules = sorted(wd["per_schedule"].keys(), key=_sched_order)

            # Scatter: drift vs reversion_auc, coloured by schedule
            fig = go.Figure()
            for sched in schedules:
                d = wd["per_schedule"][sched]
                if not d["drifts"]:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=d["drifts"],
                        y=d["aucs"],
                        name=sched,
                        mode="markers",
                        marker={"color": _color(sched), "size": 10},
                    )
                )
            r_val = wd.get("correlation_r", float("nan"))
            fig.update_layout(
                title=f"Weight Drift vs Reversion AUC — {run_name}<br>"
                f"<sup>Pearson r = {r_val:.3f}. Larger drift → more forgetting (higher AUC)?</sup>",
                xaxis_title="||theta_peak - theta_pre_burst||_2 (weight drift)",
                yaxis_title="Reversion AUC (lower = faster forgetting)",
                legend_title="Schedule",
                template="plotly_white",
                height=500,
            )
            _add(f"weight_drift_scatter_{run_name}", fig)

            # Mean drift per schedule bar chart
            mean_drifts = [
                float(np.mean(wd["per_schedule"][s]["drifts"]))
                if wd["per_schedule"][s]["drifts"]
                else float("nan")
                for s in schedules
            ]
            fig = go.Figure(
                go.Bar(
                    x=schedules,
                    y=mean_drifts,
                    marker_color=[_color(s) for s in schedules],
                )
            )
            fig.update_layout(
                title=f"Mean Weight Drift per Schedule — {run_name}<br>"
                "<sup>||theta_peak - theta_pre_burst||_2 averaged over seeds</sup>",
                xaxis_title="Schedule",
                yaxis_title="Mean Weight Drift (L2 norm)",
                template="plotly_white",
                height=500,
            )
            _add(f"weight_drift_bar_{run_name}", fig)

        # ------------------------------------------------------------------
        # Metric 3: Loss Surface
        # ------------------------------------------------------------------
        ls = run_data.get("loss_surface", {})
        if ls:
            schedules = sorted(ls.keys(), key=_sched_order)

            # Sharpness comparison: burst vs other per schedule (separate figures with CI)
            colors = [_color(s) for s in schedules]

            def _ci95(vals: Any) -> Any:  # noqa: ANN401
                if len(vals) > 1:
                    return 1.96 * float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                return 0.0

            burst_sharpness = [ls[s]["burst_sharpness"] for s in schedules]
            other_sharpness = [ls[s]["other_sharpness"] for s in schedules]
            burst_ci = [_ci95(ls[s].get("per_seed_burst_sharpness", [])) for s in schedules]
            other_ci = [_ci95(ls[s].get("per_seed_other_sharpness", [])) for s in schedules]
            diff_sharpness = [b - o for b, o in zip(burst_sharpness, other_sharpness, strict=True)]
            diff_ci = [float(np.sqrt(bc**2 + oc**2)) for bc, oc in zip(burst_ci, other_ci, strict=True)]  # noqa: E501

            fig_sharp_burst = go.Figure(
                go.Bar(
                    x=schedules,
                    y=burst_sharpness,
                    marker_color=colors,
                    showlegend=False,
                    error_y={"type": "data", "array": burst_ci, "visible": True},
                )
            )
            fig_sharp_burst.update_layout(
                title=f"Loss Surface Sharpness (Burst) at Peak Burst — {run_name}<br>"
                "<sup>max(loss) - centre(loss) over ±{:.3f} perturbation range. "
                "Higher = narrower basin. Error bars = 95% CI across seeds.</sup>".format(
                    SURFACE_RANGE
                ),
                xaxis_title="Schedule",
                yaxis_title="Sharpness",
                template="plotly_white",
                height=500,
            )
            _add(f"loss_surface_sharpness_burst_{run_name}", fig_sharp_burst)
            fig_sharp_other = go.Figure(
                go.Bar(
                    x=schedules,
                    y=other_sharpness,
                    marker_color=colors,
                    showlegend=False,
                    error_y={"type": "data", "array": other_ci, "visible": True},
                )
            )
            fig_sharp_other.update_layout(
                title=f"Loss Surface Sharpness (Other) at Peak Burst — {run_name}<br>"
                "<sup>max(loss) - centre(loss) over ±{:.3f} perturbation range. "
                "Error bars = 95% CI across seeds.</sup>".format(SURFACE_RANGE),
                xaxis_title="Schedule",
                yaxis_title="Sharpness",
                template="plotly_white",
                height=500,
            )
            _add(f"loss_surface_sharpness_other_{run_name}", fig_sharp_other)
            fig_sharp_delta = go.Figure(
                go.Bar(
                    x=schedules,
                    y=diff_sharpness,
                    marker_color=colors,
                    showlegend=False,
                    error_y={"type": "data", "array": diff_ci, "visible": True},
                )
            )
            fig_sharp_delta.update_layout(
                title=f"Loss Surface Sharpness Δ (Burst - Other) at Peak Burst — {run_name}<br>"
                "<sup>Positive = burst basin narrower than other-class basin. "
                "Error bars = 95% CI across seeds.</sup>",
                xaxis_title="Schedule",
                yaxis_title="Δ Sharpness",
                template="plotly_white",
                height=500,
            )
            _add(f"loss_surface_sharpness_delta_{run_name}", fig_sharp_delta)

            # 2D heatmaps for all schedules
            extreme_scheds = list(ls.keys())
            for sched in extreme_scheds:
                d = ls[sched]
                alphas = d["alphas"]
                betas = d["betas"]

                burst_arr = np.array(d["mean_burst_surface"])
                other_arr = np.array(d["mean_other_surface"])
                burst_span = float(np.nanmax(burst_arr) - np.nanmin(burst_arr))
                other_span = float(np.nanmax(other_arr) - np.nanmin(other_arr))
                shared_span = max(burst_span, other_span)

                burst_mid = (float(np.nanmax(burst_arr)) + float(np.nanmin(burst_arr))) / 2
                other_mid = (float(np.nanmax(other_arr)) + float(np.nanmin(other_arr))) / 2
                burst_zmin = burst_mid - shared_span / 2
                burst_zmax = burst_mid + shared_span / 2
                other_zmin = other_mid - shared_span / 2
                other_zmax = other_mid + shared_span / 2

                fig = make_subplots(
                    rows=1,
                    cols=2,
                    subplot_titles=["Burst Prompts", "Other Prompts"],
                )
                fig.add_trace(
                    go.Heatmap(
                        z=d["mean_burst_surface"],
                        x=alphas,
                        y=betas,
                        colorscale="Viridis",
                        zmin=burst_zmin,
                        zmax=burst_zmax,
                        colorbar={"title": "CE Loss", "x": 0.45},
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Heatmap(
                        z=d["mean_other_surface"],
                        x=alphas,
                        y=betas,
                        colorscale="Viridis",
                        zmin=other_zmin,
                        zmax=other_zmax,
                        colorbar={"title": "CE Loss", "x": 1.0},
                    ),
                    row=1,
                    col=2,
                )
                fig.update_layout(
                    title=f"Loss Surface at Peak Burst — {run_name} / {sched}<br>"
                    "<sup>Filter-normalised 2D slice (Li et al. 2018). "
                    "Narrow = fragile basin. Both subplots share the same colour span.</sup>",
                    template="plotly_white",
                    height=500,
                )
                _add(f"loss_surface_2d_{run_name}_{sched}", fig)

    # ------------------------------------------------------------------
    # HTML dashboard
    # ------------------------------------------------------------------
    html_parts = [
        """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Basin Geometry Metrics</title>
<style>
  body { font-family: Arial, sans-serif; margin: 20px; background: #f0f2f5; }
  h1 { color: #1a1a2e; font-size: 1.8em; }
  h2 { color: #16213e; margin-top: 40px; font-size: 1.3em; }
  .chart-container {
    background: white; border-radius: 10px; padding: 20px;
    margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  }
</style>
</head>
<body>
<h1>Basin Geometry Metrics (Kim et al. 2025 adaptation)</h1>
<p style="color:#555; max-width:900px;">
  Three metrics adapted from Kim et al. (2025) to test whether bursty training
  produces narrower parameter basins for the burst capability:
  (1) Gaussian noise robustness, (2) weight drift vs forgetting correlation,
  (3) filter-normalised loss surface visualisation.
</p>
"""
    ]

    for i, (key, fig) in enumerate(all_figs):
        html_parts.append('<div class="chart-container">\n')
        html_parts.append(f"<h2>{i + 1}. {key.replace('_', ' ').title()}</h2>\n")
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=(i == 0)))
        html_parts.append("</div>\n")

    html_parts.append("</body></html>")

    html_path = out_dir / "dashboard.html"
    with html_path.open("w") as f:
        f.write("".join(html_parts))

    from burst.dev.plot_utils import write_text_report

    write_text_report(
        all_figs,
        out_dir / "dashboard.txt",
        dashboard_title="Basin Geometry Metrics (Kim et al. 2025)",
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def analyse_run(
    run_dir: Path,
    n_seeds: int = 3,
    noise_sigmas: list[float] = NOISE_SIGMAS,
    surface_grid: int = SURFACE_GRID,
    surface_range: float = SURFACE_RANGE,
    skip_surface: bool = False,
) -> dict:
    """Run all basin metrics on a single run directory."""
    cfg_path, logs_dir, _ = resolve_run_paths(run_dir)
    with cfg_path.open() as f:
        run_cfg = json.load(f)

    parse_run_config(run_cfg)

    with logs_dir / "_data.pkl".open("rb") as f:
        target_pool, bg_pool, _, _, _ = pickle.load(f)

    burst_docs_BL = np.concatenate(list(target_pool.values()))
    other_docs_BL = np.concatenate(list(bg_pool.values()))
    prompt_len = run_cfg["task_info"]["prompt_len"]

    with logs_dir / "all_results.pkl".open("rb") as f:
        all_results = pickle.load(f)

    ckpt_root = logs_dir / "checkpoints"
    result: dict = {}

    if not ckpt_root.exists():
        return result

    result["noise_robustness"] = compute_noise_robustness(
        ckpt_root,
        all_results,
        burst_docs_BL,
        other_docs_BL,
        prompt_len,
        n_seeds=n_seeds,
        sigmas=noise_sigmas,
    )

    result["directed_noise"] = compute_directed_noise_robustness(
        ckpt_root,
        all_results,
        burst_docs_BL,
        other_docs_BL,
        prompt_len,
        n_seeds=n_seeds,
    )

    result["weight_drift"] = compute_weight_drift_correlation(
        ckpt_root,
        all_results,
        n_seeds=None,
    )

    if not skip_surface:
        result["loss_surface"] = compute_loss_surface(
            ckpt_root,
            all_results,
            burst_docs_BL,
            other_docs_BL,
            n_seeds=n_seeds,
            grid_size=surface_grid,
            surface_range=surface_range,
        )
    else:
        pass

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Basin geometry metrics: noise robustness, weight drift, loss surface.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("data/basin_metrics_combined"))
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--surface-grid", type=int, default=SURFACE_GRID)
    parser.add_argument("--surface-range", type=float, default=SURFACE_RANGE)
    parser.add_argument(
        "--skip-surface", action="store_true", help="Skip the expensive loss surface computation"
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_run_results: dict[str, dict] = {}
    for run_dir in args.run_dirs:
        run_dir = Path(run_dir)
        r = analyse_run(
            run_dir,
            n_seeds=args.n_seeds,
            surface_grid=args.surface_grid,
            surface_range=args.surface_range,
            skip_surface=args.skip_surface,
        )
        all_run_results[run_dir.name] = r

    results_path = args.out_dir / "results.pkl"
    with results_path.open("wb") as f:
        pickle.dump(all_run_results, f)

    make_dashboard(all_run_results, args.out_dir)


if __name__ == "__main__":
    main()
