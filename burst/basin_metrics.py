"""Basin geometry metrics for burstiness runs.

Implements three metrics from Kim et al. (2025) "Rethinking Safety in LLM
Fine-tuning: An Optimization Perspective", adapted to measure whether bursty
training produces shallower (narrower) parameter basins than mixed training.

Metrics:
  1. Gaussian noise robustness  — add N(0, σ²) noise to weights at peak burst,
                                   measure burst vs other accuracy at each σ.
                                   Bursty models should lose burst accuracy at
                                   smaller σ than mixed models (narrower basin).

  2. Weight drift vs forgetting  — compute ||θ_peak − θ_pre_burst||₂ and
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
    A: grid resolution for loss surface (n_alpha × n_alpha)
"""
import sys, os, argparse, pickle, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from omegaconf import OmegaConf

from net.nanogpt import nanoGPT
from burst.train_utils import load_net, resolve_run_paths
from burst.config import (
    PHASE_BURST, PHASE_REVERSION, SCHEDULE_ORDER, SCHED_COLORS,
    parse_run_config,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Noise sigma levels: 0 → 0.02, matching Kim et al. range around σ=0.004
NOISE_SIGMAS: list[float] = [0.0, 0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.010, 0.015, 0.020]

# Loss surface grid resolution (n_alpha × n_alpha perturbation grid)
SURFACE_GRID: int = 15

# Range of perturbation magnitude for loss surface (±SURFACE_RANGE)
SURFACE_RANGE: float = 0.02


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sched_order(s: str) -> int:
    try:
        return SCHEDULE_ORDER.index(s)
    except ValueError:
        return 99


def _color(s: str) -> str:
    return SCHED_COLORS.get(s, "#888888")


def _ckpt_files(ckpt_dir: Path) -> dict[int, Path]:
    return {int(p.stem.split("_")[1]): p for p in ckpt_dir.glob("step_*.pt")}


def _flat_params(net: nanoGPT) -> torch.Tensor:
    return torch.cat([p.detach().float().cpu().view(-1) for p in net.parameters()])


@torch.no_grad()
def _free_gen_acc(net: nanoGPT, docs_BL: np.ndarray, prompt_len: int) -> float:
    if docs_BL.shape[0] == 0:
        return 0.0
    net.eval()
    docs_t = torch.as_tensor(docs_BL, dtype=torch.long, device=DEVICE)
    B, L = docs_t.shape
    target_B6 = docs_t[:, -6:]
    generated = net.generate(docs_t[:, :prompt_len], L - prompt_len)
    return (generated[:, -6:] == target_B6).all(dim=1).float().mean().item()


@torch.no_grad()
def _cross_entropy_loss(net: nanoGPT, docs_BL: np.ndarray) -> float:
    if docs_BL.shape[0] == 0:
        return float("nan")
    net.eval()
    n = min(256, docs_BL.shape[0])
    idx = np.random.choice(docs_BL.shape[0], n, replace=False)
    dat = torch.as_tensor(docs_BL[idx], dtype=torch.long, device=DEVICE)
    inp, tgt = dat[:, :-1], dat[:, 1:]
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        logits = net(inp).float()
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)).item()


def _filter_normalise(direction: dict[str, torch.Tensor],
                      reference: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Filter-normalise a random direction to match the scale of reference weights.

    For each parameter tensor, normalise each filter (row for 2D, full tensor
    for 1D) so its norm equals the corresponding filter norm in the reference.
    This makes the perturbation scale-invariant across layers.
    """
    normed = {}
    for name, d in direction.items():
        ref = reference[name].float()
        d_f = d.float()
        if d_f.dim() >= 2:
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
    N(0, σ²I) noise to all parameters, and evaluates both burst and other
    accuracy. Repeats across sigma levels.

    Bursty models (burst_100) should lose burst accuracy at smaller σ than
    mixed models (burst_25), mirroring Kim et al. Figure 5.

    Returns per-schedule mean accuracy curves for burst and other classes.
    """
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)

    n_burst = min(n_eval_docs, burst_docs_BL.shape[0])
    n_other = min(n_eval_docs, other_docs_BL.shape[0])
    burst_idx = np.random.choice(burst_docs_BL.shape[0], n_burst, replace=False)
    other_idx = np.random.choice(other_docs_BL.shape[0], n_other, replace=False)
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

            base_sd = {k: v.float() for k, v in torch.load(
                str(files[peak_step]), map_location="cpu", weights_only=True).items()}

            net = load_net(cfg, str(files[peak_step]))

            burst_accs: list[float] = []
            other_accs: list[float] = []

            for sigma in sigmas:
                if sigma == 0.0:
                    net.load_state_dict({k: v.to(DEVICE) for k, v in base_sd.items()})
                else:
                    noisy_sd = {
                        k: (v + torch.randn_like(v) * sigma).to(DEVICE)
                        for k, v in base_sd.items()
                    }
                    net.load_state_dict(noisy_sd)

                burst_accs.append(_free_gen_acc(net, burst_eval, prompt_len))
                other_accs.append(_free_gen_acc(net, other_eval, prompt_len))

            burst_acc_curves.append(burst_accs)
            other_acc_curves.append(other_accs)
            seeds_done += 1
            print(f"  {label}: burst@σ=0={burst_accs[0]:.3f}, "
                  f"burst@σ=0.004={burst_accs[sigmas.index(0.004)] if 0.004 in sigmas else '?':.3f}",
                  flush=True)

        if burst_acc_curves:
            mean_burst = [float(np.mean([c[i] for c in burst_acc_curves]))
                          for i in range(len(sigmas))]
            mean_other = [float(np.mean([c[i] for c in other_acc_curves]))
                          for i in range(len(sigmas))]
            std_burst = [float(np.std([c[i] for c in burst_acc_curves]))
                         for i in range(len(sigmas))]
            std_other = [float(np.std([c[i] for c in other_acc_curves]))
                         for i in range(len(sigmas))]
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
# Metric 2: Weight Drift vs Forgetting Correlation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_weight_drift_correlation(
    ckpt_root: Path,
    all_results: list[dict],
    n_seeds: int | None = None,
) -> dict:
    """Correlate ||θ_peak − θ_pre_burst||₂ with reversion_auc.

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

            pre_sd = {k: v.float() for k, v in torch.load(
                str(files[pre_step]), map_location="cpu", weights_only=True).items()}
            peak_sd = {k: v.float() for k, v in torch.load(
                str(files[peak_step]), map_location="cpu", weights_only=True).items()}

            drift = float(sum(
                (peak_sd[k] - pre_sd[k]).norm().item() ** 2
                for k in peak_sd
            ) ** 0.5)

            auc = r.get("reversion_auc", float("nan"))
            if not np.isnan(auc):
                drifts.append(drift)
                aucs.append(auc)
                all_drifts.append(drift)
                all_aucs.append(auc)
                all_labels.append(sched)
                seeds_done += 1
                print(f"  {label}: drift={drift:.4f}, reversion_auc={auc:.4f}", flush=True)

        per_schedule[sched] = {"drifts": drifts, "aucs": aucs}

    if len(all_drifts) >= 2:
        r_val = float(np.corrcoef(all_drifts, all_aucs)[0, 1])
    else:
        r_val = float("nan")

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
      1. Load peak-burst checkpoint θ₀.
      2. Sample two random filter-normalised directions δ₁, δ₂.
      3. Evaluate loss on burst and other prompts at θ₀ + α·δ₁ + β·δ₂
         for (α, β) on a grid_size × grid_size grid in [-range, +range]².
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
    burst_idx = np.random.choice(burst_docs_BL.shape[0], n_burst, replace=False)
    other_idx = np.random.choice(other_docs_BL.shape[0], n_other, replace=False)
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

            base_sd = {k: v.float().cpu() for k, v in torch.load(
                str(files[peak_step]), map_location="cpu", weights_only=True).items()}

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

            # Sharpness: max loss - centre loss (at α=β=0)
            centre_i = grid_size // 2
            burst_sharpness = float(burst_surface.max() - burst_surface[centre_i, centre_i])
            other_sharpness = float(other_surface.max() - other_surface[centre_i, centre_i])
            print(f"  {label}: burst_sharpness={burst_sharpness:.4f}, "
                  f"other_sharpness={other_sharpness:.4f}", flush=True)

        if burst_surfaces:
            mean_burst_surface = np.mean(burst_surfaces, axis=0)
            mean_other_surface = np.mean(other_surfaces, axis=0)
            centre_i = grid_size // 2
            burst_sharpness = float(mean_burst_surface.max() - mean_burst_surface[centre_i, centre_i])
            other_sharpness = float(mean_other_surface.max() - mean_other_surface[centre_i, centre_i])
        else:
            mean_burst_surface = np.full((grid_size, grid_size), float("nan"))
            mean_other_surface = np.full((grid_size, grid_size), float("nan"))
            burst_sharpness = other_sharpness = float("nan")

        results[sched] = {
            "alphas": alphas.tolist(),
            "betas": betas.tolist(),
            "mean_burst_surface": mean_burst_surface.tolist(),
            "mean_other_surface": mean_other_surface.tolist(),
            "burst_sharpness": burst_sharpness,
            "other_sharpness": other_sharpness,
        }

    return results


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def make_dashboard(results: dict, out_dir: Path) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from burst.plot_utils import save_png as _save_png

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
            for sched in schedules:
                d = nr[sched]
                fig.add_trace(go.Scatter(
                    x=d["sigmas"], y=d["mean_burst_accs"],
                    name=sched,
                    line=dict(color=_color(sched), width=2),
                    mode="lines+markers",
                    error_y=dict(array=d["std_burst_accs"], visible=True, thickness=1),
                ))
            fig.add_vline(x=0.004, line_dash="dash", line_color="gray",
                          annotation_text="σ=0.004 (Kim et al. safety threshold)")
            fig.update_layout(
                title=f"Burst Accuracy Under Gaussian Weight Noise — {run_name}<br>"
                      "<sup>Bursty models should lose burst accuracy at smaller σ (narrower basin)</sup>",
                xaxis_title="Noise σ",
                yaxis_title="Burst Accuracy",
                legend_title="Schedule",
                template="plotly_white", height=500,
            )
            _add(f"noise_burst_{run_name}", fig)

            # Other accuracy vs sigma (should be robust across all schedules)
            fig = go.Figure()
            for sched in schedules:
                d = nr[sched]
                fig.add_trace(go.Scatter(
                    x=d["sigmas"], y=d["mean_other_accs"],
                    name=sched,
                    line=dict(color=_color(sched), width=2),
                    mode="lines+markers",
                ))
            fig.add_vline(x=0.004, line_dash="dash", line_color="gray",
                          annotation_text="σ=0.004")
            fig.update_layout(
                title=f"Other-Class Accuracy Under Gaussian Weight Noise — {run_name}<br>"
                      "<sup>Should be more robust than burst accuracy (wider basin)</sup>",
                xaxis_title="Noise σ",
                yaxis_title="Other-Class Accuracy",
                legend_title="Schedule",
                template="plotly_white", height=500,
            )
            _add(f"noise_other_{run_name}", fig)

            # Differential sensitivity: burst drop / other drop at σ=0.004
            sigma_idx = NOISE_SIGMAS.index(0.004) if 0.004 in NOISE_SIGMAS else -1
            if sigma_idx >= 0:
                burst_drops = [nr[s]["mean_burst_accs"][0] - nr[s]["mean_burst_accs"][sigma_idx]
                               for s in schedules]
                other_drops = [nr[s]["mean_other_accs"][0] - nr[s]["mean_other_accs"][sigma_idx]
                               for s in schedules]
                fig = make_subplots(rows=1, cols=2,
                                    subplot_titles=["Burst Accuracy Drop at σ=0.004",
                                                    "Other Accuracy Drop at σ=0.004"])
                colors = [_color(s) for s in schedules]
                fig.add_trace(go.Bar(x=schedules, y=burst_drops,
                                     marker_color=colors, showlegend=False), row=1, col=1)
                fig.add_trace(go.Bar(x=schedules, y=other_drops,
                                     marker_color=colors, showlegend=False), row=1, col=2)
                fig.update_layout(
                    title=f"Differential Noise Sensitivity at σ=0.004 — {run_name}<br>"
                          "<sup>Larger burst drop = narrower burst basin (shallower learning)</sup>",
                    template="plotly_white", height=500,
                )
                _add(f"noise_differential_{run_name}", fig)

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
                fig.add_trace(go.Scatter(
                    x=d["drifts"], y=d["aucs"],
                    name=sched,
                    mode="markers",
                    marker=dict(color=_color(sched), size=10),
                ))
            r_val = wd.get("correlation_r", float("nan"))
            fig.update_layout(
                title=f"Weight Drift vs Reversion AUC — {run_name}<br>"
                      f"<sup>Pearson r = {r_val:.3f}. Larger drift → more forgetting (higher AUC)?</sup>",
                xaxis_title="||θ_peak − θ_pre_burst||₂ (weight drift)",
                yaxis_title="Reversion AUC (lower = faster forgetting)",
                legend_title="Schedule",
                template="plotly_white", height=500,
            )
            _add(f"weight_drift_scatter_{run_name}", fig)

            # Mean drift per schedule bar chart
            mean_drifts = [float(np.mean(wd["per_schedule"][s]["drifts"]))
                           if wd["per_schedule"][s]["drifts"] else float("nan")
                           for s in schedules]
            fig = go.Figure(go.Bar(
                x=schedules, y=mean_drifts,
                marker_color=[_color(s) for s in schedules],
            ))
            fig.update_layout(
                title=f"Mean Weight Drift per Schedule — {run_name}<br>"
                      "<sup>||θ_peak − θ_pre_burst||₂ averaged over seeds</sup>",
                xaxis_title="Schedule",
                yaxis_title="Mean Weight Drift (L2 norm)",
                template="plotly_white", height=500,
            )
            _add(f"weight_drift_bar_{run_name}", fig)

        # ------------------------------------------------------------------
        # Metric 3: Loss Surface
        # ------------------------------------------------------------------
        ls = run_data.get("loss_surface", {})
        if ls:
            schedules = sorted(ls.keys(), key=_sched_order)

            # Sharpness comparison: burst vs other per schedule
            burst_sharpness = [ls[s]["burst_sharpness"] for s in schedules]
            other_sharpness = [ls[s]["other_sharpness"] for s in schedules]
            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=["Burst Prompt Sharpness",
                                                "Other Prompt Sharpness"])
            colors = [_color(s) for s in schedules]
            fig.add_trace(go.Bar(x=schedules, y=burst_sharpness,
                                 marker_color=colors, showlegend=False), row=1, col=1)
            fig.add_trace(go.Bar(x=schedules, y=other_sharpness,
                                 marker_color=colors, showlegend=False), row=1, col=2)
            fig.update_layout(
                title=f"Loss Surface Sharpness at Peak Burst — {run_name}<br>"
                      "<sup>max(loss) − centre(loss) over ±{:.3f} perturbation range. "
                      "Higher = narrower basin.</sup>".format(SURFACE_RANGE),
                template="plotly_white", height=500,
            )
            _add(f"loss_surface_sharpness_{run_name}", fig)

            # 2D heatmaps for extreme schedules (burst_100 vs burst_25)
            extreme_scheds = [s for s in ["burst_100", "burst_25"] if s in ls]
            for sched in extreme_scheds:
                d = ls[sched]
                alphas = d["alphas"]
                betas = d["betas"]

                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=["Burst Prompts", "Other Prompts"],
                )
                fig.add_trace(go.Heatmap(
                    z=d["mean_burst_surface"],
                    x=alphas, y=betas,
                    colorscale="Viridis",
                    colorbar=dict(title="CE Loss", x=0.45),
                ), row=1, col=1)
                fig.add_trace(go.Heatmap(
                    z=d["mean_other_surface"],
                    x=alphas, y=betas,
                    colorscale="Viridis",
                    colorbar=dict(title="CE Loss", x=1.0),
                ), row=1, col=2)
                fig.update_layout(
                    title=f"Loss Surface at Peak Burst — {run_name} / {sched}<br>"
                          "<sup>Filter-normalised 2D slice (Li et al. 2018). "
                          "Narrow = fragile basin.</sup>",
                    template="plotly_white", height=500,
                )
                _add(f"loss_surface_2d_{run_name}_{sched}", fig)

    # ------------------------------------------------------------------
    # HTML dashboard
    # ------------------------------------------------------------------
    html_parts = ["""<!DOCTYPE html>
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
"""]

    for i, (key, fig) in enumerate(all_figs):
        html_parts.append(f'<div class="chart-container">\n')
        html_parts.append(f'<h2>{i+1}. {key.replace("_", " ").title()}</h2>\n')
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=(i == 0)))
        html_parts.append("</div>\n")

    html_parts.append("</body></html>")

    html_path = out_dir / "dashboard.html"
    with open(html_path, "w") as f:
        f.write("".join(html_parts))
    print(f"\nDashboard saved: {html_path}", flush=True)


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
    print(f"\n{'='*60}", flush=True)
    print(f"Analysing: {run_dir.name}", flush=True)
    print(f"{'='*60}", flush=True)

    cfg_path, logs_dir, _ = resolve_run_paths(run_dir)
    with open(cfg_path) as f:
        run_cfg = json.load(f)

    rc = parse_run_config(run_cfg)

    with open(logs_dir / "_data.pkl", "rb") as f:
        target_pool, bg_pool, _, _, _ = pickle.load(f)

    burst_docs_BL = np.concatenate(list(target_pool.values()))
    other_docs_BL = np.concatenate(list(bg_pool.values()))
    prompt_len = run_cfg["task_info"]["prompt_len"]

    with open(logs_dir / "all_results.pkl", "rb") as f:
        all_results = pickle.load(f)

    ckpt_root = logs_dir / "checkpoints"
    result: dict = {}

    if not ckpt_root.exists():
        print("  No checkpoints directory — all three metrics require checkpoints.", flush=True)
        return result

    print("\n[1/3] Gaussian noise robustness...", flush=True)
    result["noise_robustness"] = compute_noise_robustness(
        ckpt_root, all_results, burst_docs_BL, other_docs_BL, prompt_len,
        n_seeds=n_seeds, sigmas=noise_sigmas,
    )

    print("\n[2/3] Weight drift vs forgetting correlation...", flush=True)
    result["weight_drift"] = compute_weight_drift_correlation(
        ckpt_root, all_results, n_seeds=None,
    )

    if not skip_surface:
        print("\n[3/3] Loss surface visualisation...", flush=True)
        result["loss_surface"] = compute_loss_surface(
            ckpt_root, all_results, burst_docs_BL, other_docs_BL,
            n_seeds=n_seeds, grid_size=surface_grid, surface_range=surface_range,
        )
    else:
        print("\n[3/3] Loss surface skipped (--skip-surface).", flush=True)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Basin geometry metrics: noise robustness, weight drift, loss surface.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("data/basin_metrics_combined"))
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--surface-grid", type=int, default=SURFACE_GRID)
    parser.add_argument("--surface-range", type=float, default=SURFACE_RANGE)
    parser.add_argument("--skip-surface", action="store_true",
                        help="Skip the expensive loss surface computation")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_run_results: dict[str, dict] = {}
    for run_dir in args.run_dirs:
        run_dir = Path(run_dir)
        t0 = time.time()
        r = analyse_run(
            run_dir,
            n_seeds=args.n_seeds,
            surface_grid=args.surface_grid,
            surface_range=args.surface_range,
            skip_surface=args.skip_surface,
        )
        all_run_results[run_dir.name] = r
        print(f"  Completed {run_dir.name} in {time.time() - t0:.1f}s", flush=True)

    results_path = args.out_dir / "results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(all_run_results, f)
    print(f"\nResults saved: {results_path}", flush=True)

    print("\nGenerating dashboard...", flush=True)
    make_dashboard(all_run_results, args.out_dir)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
