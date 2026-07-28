"""New analysis metrics for burst experiments.

Computes:
  1. Layerwise weight difference throughout training (vs pre-burst checkpoint)
  2. Per-layer activations during training for burst and other data
  3. Loss basin with N random directions — magnitude and variance charts
  4. Weight norm hypothesis test (does more burst = higher weight norm?)
  5. Sharpness on all burst settings
  6. Gradient norm (L1, L2, Linf) over time and per layer, correlated with cosim
  7. Grad rank investigation and re-plot

Usage:
    python burst/new_analysis.py <run_dir> [--n-seeds 3] [--basin-runs 50]

Dimension key:
    B: batch_size
    L: sequence_length (doc_len)
    N: n_embd (model dimension)
    V: vocab_size
    P: n_params (total parameters, flattened)
"""
import sys, os, argparse, pickle, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from net.nanogpt import nanoGPT
from burst.train_utils import load_net, resolve_run_paths
from burst.config import (
    SCHEDULE_ORDER, SCHED_COLORS, SCHED_DISPLAY, parse_run_config,
    ordered_schedules,
)
from burst.grad_sim import _layer_groups

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _sched_order(s: str) -> int:
    try:
        return SCHEDULE_ORDER.index(s)
    except ValueError:
        return 99


def _color(s: str) -> str:
    return SCHED_COLORS.get(s, "#888888")


def _label(s: str) -> str:
    return SCHED_DISPLAY.get(s, s)


def _ckpt_files(ckpt_dir: Path) -> dict[int, Path]:
    return {int(p.stem.split("_")[1]): p for p in ckpt_dir.glob("step_*.pt")}


# ---------------------------------------------------------------------------
# 1. Layerwise weight difference throughout training
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_layerwise_weight_diff(
    ckpt_root: Path,
    all_results: list[dict],
    n_seeds: int = 3,
) -> dict:
    """For each schedule/seed, compute per-layer ||W_step - W_pre_burst|| at each checkpoint."""
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)
    results = {}

    for sched in schedules:
        sched_results = jobs_by_schedule[sched]
        all_seed_data = []
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

            cfg = r["config"]
            available = sorted(files.keys())
            pre_step = available[0]
            pre_sd = {k: v.float().cpu() for k, v in torch.load(
                str(files[pre_step]), map_location="cpu", weights_only=True).items()}

            layer_groups = None
            steps_data = []
            for step in available:
                sd = {k: v.float().cpu() for k, v in torch.load(
                    str(files[step]), map_location="cpu", weights_only=True).items()}

                if layer_groups is None:
                    net_tmp = load_net(cfg, str(files[step]))
                    layer_groups = _layer_groups(net_tmp)
                    del net_tmp

                per_layer = {}
                for name, pnames in layer_groups:
                    diff_norm = 0.0
                    for pn in pnames:
                        if pn in sd and pn in pre_sd:
                            diff_norm += (sd[pn] - pre_sd[pn]).norm().item() ** 2
                    per_layer[name] = float(diff_norm ** 0.5)

                total_diff = float(sum(
                    (sd[k] - pre_sd[k]).norm().item() ** 2 for k in sd
                ) ** 0.5)

                steps_data.append({
                    "step": step,
                    "per_layer": per_layer,
                    "total_diff": total_diff,
                })

            all_seed_data.append(steps_data)
            seeds_done += 1

        results[sched] = all_seed_data

    return results


# ---------------------------------------------------------------------------
# 2. Per-layer activations during training
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_layerwise_activations(
    ckpt_root: Path,
    all_results: list[dict],
    burst_docs_BL: np.ndarray,
    other_docs_BL: np.ndarray,
    n_seeds: int = 3,
    n_eval: int = 128,
) -> dict:
    """Compute per-layer activation norms for burst and other data at each checkpoint."""
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)

    n_burst = min(n_eval, burst_docs_BL.shape[0])
    n_other = min(n_eval, other_docs_BL.shape[0])
    burst_idx = np.random.choice(burst_docs_BL.shape[0], n_burst, replace=False)
    other_idx = np.random.choice(other_docs_BL.shape[0], n_other, replace=False)
    burst_eval = burst_docs_BL[burst_idx]
    other_eval = other_docs_BL[other_idx]

    results = {}

    for sched in schedules:
        sched_results = jobs_by_schedule[sched]
        all_seed_data = []
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

            cfg = r["config"]
            available = sorted(files.keys())

            steps_data = []
            for step in available:
                net = load_net(cfg, str(files[step]))
                net.eval()

                layer_norms_burst = {}
                layer_norms_other = {}

                hooks = []
                activations = {}

                def make_hook(name):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            out = output[0]
                        else:
                            out = output
                        activations[name] = out.detach().float().norm(dim=-1).mean().item()
                    return hook_fn

                for i, block in enumerate(net.transformer.h):
                    hooks.append(block.register_forward_hook(make_hook(f"L{i}")))

                burst_t = torch.as_tensor(burst_eval, dtype=torch.long, device=DEVICE)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                    net(burst_t[:, :-1])
                layer_norms_burst = dict(activations)
                activations.clear()

                other_t = torch.as_tensor(other_eval, dtype=torch.long, device=DEVICE)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                    net(other_t[:, :-1])
                layer_norms_other = dict(activations)

                for h in hooks:
                    h.remove()
                del net

                steps_data.append({
                    "step": step,
                    "burst_norms": layer_norms_burst,
                    "other_norms": layer_norms_other,
                })

            all_seed_data.append(steps_data)
            seeds_done += 1

        results[sched] = all_seed_data

    return results


# ---------------------------------------------------------------------------
# 3. Loss basin with N random directions
# ---------------------------------------------------------------------------

@torch.no_grad()
def _cross_entropy_loss(net, docs_BL: np.ndarray) -> float:
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


@torch.no_grad()
def compute_loss_basin_random_directions(
    ckpt_root: Path,
    all_results: list[dict],
    burst_docs_BL: np.ndarray,
    other_docs_BL: np.ndarray,
    n_seeds: int = 2,
    n_directions: int = 50,
    n_points: int = 8,
    max_epsilon: float = 0.03,
) -> dict:
    """For each schedule, sample n_directions random directions and evaluate loss along each.

    Returns per-direction loss profiles so we can plot magnitude and variance.
    Half the default resolution (n_points=8 instead of 15).
    """
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)
    epsilons = np.linspace(-max_epsilon, max_epsilon, n_points)

    n_burst = min(128, burst_docs_BL.shape[0])
    n_other = min(128, other_docs_BL.shape[0])
    burst_idx = np.random.choice(burst_docs_BL.shape[0], n_burst, replace=False)
    other_idx = np.random.choice(other_docs_BL.shape[0], n_other, replace=False)
    burst_eval = burst_docs_BL[burst_idx]
    other_eval = other_docs_BL[other_idx]

    results = {}

    for sched in schedules:
        sched_results = jobs_by_schedule[sched]
        all_direction_losses_burst = []
        all_direction_losses_other = []
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

            net = load_net(cfg, str(files[peak_step]))

            for d_idx in range(n_directions):
                direction = {k: torch.randn_like(v) for k, v in base_sd.items()}
                dir_flat = torch.cat([v.view(-1) for v in direction.values()])
                dir_norm = dir_flat.norm()
                if dir_norm < 1e-10:
                    continue
                direction = {k: v / dir_norm for k, v in direction.items()}

                burst_losses = []
                other_losses = []

                for eps in epsilons:
                    perturbed = {
                        k: (base_sd[k] + eps * direction[k]).to(DEVICE)
                        for k in base_sd
                    }
                    net.load_state_dict(perturbed)
                    burst_losses.append(_cross_entropy_loss(net, burst_eval))
                    other_losses.append(_cross_entropy_loss(net, other_eval))

                all_direction_losses_burst.append(burst_losses)
                all_direction_losses_other.append(other_losses)

            seeds_done += 1
            print(f"  {label}: {n_directions} directions done", flush=True)
            del net

        results[sched] = {
            "epsilons": epsilons.tolist(),
            "burst_losses": all_direction_losses_burst,
            "other_losses": all_direction_losses_other,
        }

    return results


# ---------------------------------------------------------------------------
# 4. Weight norm hypothesis
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_weight_norms(
    ckpt_root: Path,
    all_results: list[dict],
    n_seeds: int | None = None,
) -> dict:
    """Compute total weight norm at peak burst for each schedule/seed."""
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)
    results = {}

    for sched in schedules:
        sched_results = jobs_by_schedule[sched]
        norms = []
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
            peak_step = min(available, key=lambda x: abs(x - (T - 1)))

            sd = torch.load(str(files[peak_step]), map_location="cpu", weights_only=True)
            total_norm = float(sum(v.float().norm().item() ** 2 for v in sd.values()) ** 0.5)
            norms.append(total_norm)
            seeds_done += 1

        results[sched] = norms

    return results


# ---------------------------------------------------------------------------
# 5. Gradient norms (L1, L2, Linf) over time and per layer
# ---------------------------------------------------------------------------

def compute_grad_norms_from_gradsim(run_dir: Path) -> dict:
    """Extract gradient norm data from existing grad_sim JSON files.

    The grad_sim worker already computes per-layer gradient vectors.
    We load the grad_projection data which has burst_norm and other_norm (L2).
    We also load per-layer grad_norm_ratio data.
    """
    gs_dir = run_dir / "results" / "grad_cosine_sim"
    if not gs_dir.exists():
        gs_dir = run_dir / "grad_cosine_sim"
    if not gs_dir.exists():
        return {}

    records = []
    for fp in sorted(gs_dir.glob("*.json")):
        with open(fp) as f:
            records.append(json.load(f))

    return {"records": records}


# ---------------------------------------------------------------------------
# 6. Grad rank investigation
# ---------------------------------------------------------------------------

def investigate_grad_rank(run_dir: Path) -> dict:
    """Load grad_rank data from grad_sim records and check for issues."""
    gs_dir = run_dir / "results" / "grad_cosine_sim"
    if not gs_dir.exists():
        gs_dir = run_dir / "grad_cosine_sim"
    if not gs_dir.exists():
        return {"error": "no grad_sim data"}

    records = []
    for fp in sorted(gs_dir.glob("*.json")):
        with open(fp) as f:
            records.append(json.load(f))

    issues = []
    for rec in records:
        gsl = rec.get("grad_sim_log", {})
        rank_data = gsl.get("grad_rank", {})
        if not rank_data:
            issues.append(f"{rec.get('label', '?')}: no grad_rank data")
            continue
        for layer, vals in rank_data.items():
            nan_count = sum(1 for v in vals if v != v)
            if nan_count > 0:
                issues.append(f"{rec.get('label', '?')} {layer}: {nan_count}/{len(vals)} NaN values")

    return {"records": records, "issues": issues}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _style(ax, xl="", yl="", t=""):
    ax.set_xlabel(xl, fontsize=11, fontweight="bold")
    ax.set_ylabel(yl, fontsize=11, fontweight="bold")
    if t:
        ax.set_title(t, fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.15, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_layerwise_weight_diff(data: dict, out_dir: Path, P: int = 0):
    out_dir.mkdir(parents=True, exist_ok=True)
    schedules = sorted(data.keys(), key=_sched_order)

    fig, ax = plt.subplots(figsize=(14, 7))
    for sched in schedules:
        seed_data = data[sched]
        if not seed_data:
            continue
        steps_ref = [d["step"] for d in seed_data[0]]
        total_diffs = np.array([[d["total_diff"] for d in sd] for sd in seed_data])
        m = total_diffs.mean(axis=0)
        n_s = len(total_diffs)
        ci = 1.96 * total_diffs.std(axis=0) / np.sqrt(n_s) if n_s > 1 else total_diffs.std(axis=0)
        ax.plot(steps_ref, m, color=_color(sched), lw=2, label=_label(sched))
        ax.fill_between(steps_ref, m - ci, m + ci, color=_color(sched), alpha=0.15)

    _style(ax, "Step", "||W_step - W_pre||₂",
           "Total Weight Difference vs Pre-Burst (mean ± 95% CI)")
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "weight_diff_total.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    for sched in schedules:
        seed_data = data[sched]
        if not seed_data:
            continue
        steps_ref = [d["step"] for d in seed_data[0]]
        layers = list(seed_data[0][0]["per_layer"].keys())

        fig, ax = plt.subplots(figsize=(14, 7))
        cmap = plt.get_cmap("tab20")
        for li, layer in enumerate(layers):
            vals = np.array([[d["per_layer"][layer] for d in sd] for sd in seed_data])
            m = vals.mean(axis=0)
            ax.plot(steps_ref, m, color=cmap(li / max(len(layers) - 1, 1)),
                    lw=1.5, label=layer)

        _style(ax, "Step", "||W_layer - W_pre_layer||₂",
               f"{_label(sched)}: Per-Layer Weight Difference")
        ax.legend(fontsize=7, loc="best", ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / f"weight_diff_layers_{sched}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_layerwise_activations(data: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    schedules = sorted(data.keys(), key=_sched_order)

    for data_type, data_key in [("burst", "burst_norms"), ("other", "other_norms")]:
        fig, ax = plt.subplots(figsize=(14, 7))
        for sched in schedules:
            seed_data = data[sched]
            if not seed_data:
                continue
            steps_ref = [d["step"] for d in seed_data[0]]
            all_means = []
            for sd in seed_data:
                mean_per_step = [np.mean(list(d[data_key].values())) for d in sd]
                all_means.append(mean_per_step)
            arr = np.array(all_means)
            m = arr.mean(axis=0)
            n_s = len(arr)
            ci = 1.96 * arr.std(axis=0) / np.sqrt(n_s) if n_s > 1 else arr.std(axis=0)
            ax.plot(steps_ref, m, color=_color(sched), lw=2, label=_label(sched))
            ax.fill_between(steps_ref, m - ci, m + ci, color=_color(sched), alpha=0.15)

        _style(ax, "Step", "Mean Activation Norm",
               f"Mean Activation Norm ({data_type} data) Over Training")
        ax.legend(fontsize=9, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"activation_norm_{data_type}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    for sched in schedules:
        seed_data = data[sched]
        if not seed_data:
            continue
        steps_ref = [d["step"] for d in seed_data[0]]
        layers = list(seed_data[0][0]["burst_norms"].keys())

        for data_type, data_key in [("burst", "burst_norms"), ("other", "other_norms")]:
            fig, ax = plt.subplots(figsize=(14, 7))
            cmap = plt.get_cmap("tab20")
            for li, layer in enumerate(layers):
                vals = np.array([[d[data_key].get(layer, 0) for d in sd] for sd in seed_data])
                m = vals.mean(axis=0)
                ax.plot(steps_ref, m, color=cmap(li / max(len(layers) - 1, 1)),
                        lw=1.5, label=layer)

            _style(ax, "Step", "Activation Norm",
                   f"{_label(sched)}: Per-Layer Activation Norm ({data_type})")
            ax.legend(fontsize=7, loc="best", ncol=2)
            fig.tight_layout()
            fig.savefig(out_dir / f"activation_layers_{sched}_{data_type}.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)


def plot_loss_basin(data: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    schedules = sorted(data.keys(), key=_sched_order)

    for loss_type in ["burst", "other"]:
        key = f"{loss_type}_losses"

        fig_mag, ax_mag = plt.subplots(figsize=(14, 7))
        fig_var, ax_var = plt.subplots(figsize=(14, 7))

        for sched in schedules:
            d = data[sched]
            if not d[key]:
                continue
            epsilons = d["epsilons"]
            losses_arr = np.array(d[key])
            mean_per_eps = losses_arr.mean(axis=0)
            var_per_eps = losses_arr.var(axis=0)

            ax_mag.plot(epsilons, mean_per_eps, color=_color(sched), lw=2, label=_label(sched))
            ax_var.plot(epsilons, var_per_eps, color=_color(sched), lw=2, label=_label(sched))

        _style(ax_mag, "ε (perturbation)", "Mean Loss",
               f"Loss Basin: Mean Loss ({loss_type}) Across {data[schedules[0]].get('n_directions', '?')} Random Directions")
        ax_mag.legend(fontsize=9, loc="best")
        fig_mag.tight_layout()
        fig_mag.savefig(out_dir / f"basin_magnitude_{loss_type}.png", dpi=150, bbox_inches="tight")
        plt.close(fig_mag)

        _style(ax_var, "ε (perturbation)", "Variance of Loss",
               f"Loss Basin: Variance ({loss_type}) Across Random Directions")
        ax_var.legend(fontsize=9, loc="best")
        fig_var.tight_layout()
        fig_var.savefig(out_dir / f"basin_variance_{loss_type}.png", dpi=150, bbox_inches="tight")
        plt.close(fig_var)


def plot_weight_norms(data: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    schedules = sorted(data.keys(), key=_sched_order)

    fig, ax = plt.subplots(figsize=(12, 7))
    xs = np.arange(len(schedules))
    means = [np.mean(data[s]) if data[s] else 0 for s in schedules]
    cis = [1.96 * np.std(data[s]) / np.sqrt(len(data[s])) if len(data[s]) > 1 else 0
           for s in schedules]
    colors = [_color(s) for s in schedules]

    ax.bar(xs, means, yerr=cis, color=colors, edgecolor="black", lw=0.8,
           capsize=5, alpha=0.85)
    for i, vals in enumerate(schedules):
        jit = np.random.default_rng(42).uniform(-0.12, 0.12, len(data[vals]))
        ax.scatter(np.full(len(data[vals]), i) + jit, data[vals],
                   color="black", s=30, zorder=5, alpha=0.5, edgecolor="white", lw=0.5)

    ax.set_xticks(xs)
    ax.set_xticklabels([_label(s) for s in schedules], fontsize=9, rotation=30, ha="right")
    _style(ax, "", "||W||₂ (total weight norm at peak burst)",
           "Weight Norm at Peak Burst by Schedule\n(Hypothesis: more burst → higher norm)")
    fig.tight_layout()
    fig.savefig(out_dir / "weight_norm_hypothesis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    burst_pcts = []
    norm_vals = []
    for s in schedules:
        try:
            pct = int(s.split("_")[1])
        except (IndexError, ValueError):
            continue
        for n in data[s]:
            burst_pcts.append(pct)
            norm_vals.append(n)

    if len(burst_pcts) > 2:
        fig, ax = plt.subplots(figsize=(10, 7))
        for s in schedules:
            try:
                pct = int(s.split("_")[1])
            except (IndexError, ValueError):
                continue
            for n in data[s]:
                ax.scatter(pct, n, color=_color(s), s=60, edgecolor="black", lw=0.5)

        corr = np.corrcoef(burst_pcts, norm_vals)[0, 1]
        z = np.polyfit(burst_pcts, norm_vals, 1)
        xline = np.linspace(min(burst_pcts), max(burst_pcts), 100)
        ax.plot(xline, np.polyval(z, xline), "k--", lw=1.5, alpha=0.5)
        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top")

        _style(ax, "Burst %", "||W||₂",
               "Weight Norm vs Burst Percentage\n(each dot = one seed)")
        fig.tight_layout()
        fig.savefig(out_dir / "weight_norm_vs_burst_pct.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_grad_norms_and_cosim(gs_records: list, out_dir: Path, P: int = 0):
    """Plot gradient norms (L1, L2, Linf) over time and per layer, correlated with cosim."""
    out_dir.mkdir(parents=True, exist_ok=True)

    gs_groups = defaultdict(list)
    for r in gs_records:
        gs_groups[r["schedule"]].append(r)

    schedules = sorted(gs_groups.keys(), key=_sched_order)

    for norm_type, norm_label in [("burst_norm", "||g_burst||₂"),
                                   ("other_norm", "||g_other||₂"),
                                   ("burst_l1", "||g_burst||₁"),
                                   ("other_l1", "||g_other||₁"),
                                   ("burst_linf", "||g_burst||∞"),
                                   ("other_linf", "||g_other||∞")]:
        fig, ax = plt.subplots(figsize=(14, 7))
        for sched in schedules:
            runs = gs_groups[sched]
            all_steps = []
            all_vals = []
            for r in runs:
                gsl = r.get("grad_sim_log", {})
                proj = gsl.get("grad_projection", r.get("grad_projection_log", {}))
                if not proj or norm_type not in proj:
                    continue
                steps = gsl.get("step", [])
                vals = proj[norm_type]
                if len(steps) == len(vals) and len(steps) > 0:
                    all_steps.append(np.array(steps))
                    all_vals.append(np.array(vals))

            if not all_steps:
                continue
            steps_ref = all_steps[0]
            interp_vals = [np.interp(steps_ref, s, v) for s, v in zip(all_steps, all_vals)]
            arr = np.array(interp_vals)
            m = arr.mean(axis=0)
            n_s = len(arr)
            ci = 1.96 * arr.std(axis=0) / np.sqrt(n_s) if n_s > 1 else arr.std(axis=0)
            ax.plot(steps_ref, m, color=_color(sched), lw=2, label=_label(sched))
            ax.fill_between(steps_ref, m - ci, m + ci, color=_color(sched), alpha=0.15)

        _style(ax, "Step", norm_label,
               f"Gradient Norm ({norm_label}) Over Training")
        ax.legend(fontsize=9, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"grad_norm_{norm_type}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    for sched in schedules:
        runs = gs_groups[sched]
        for r in runs:
            gsl = r.get("grad_sim_log", {})
            proj = gsl.get("grad_projection", r.get("grad_projection_log", {}))
            cosims = gsl.get("burst_vs_other", [])
            burst_norms = proj.get("burst_norm", [])
            if not cosims or not burst_norms:
                continue
            n = min(len(cosims), len(burst_norms))
            ax.scatter(cosims[:n], burst_norms[:n], color=_color(sched),
                       s=15, alpha=0.4, edgecolor="none")

    _style(ax, "Cosine Similarity (burst vs other)", "||g_burst||₂",
           "Gradient Norm vs Cosine Similarity\n(each dot = one step × seed)")
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=_color(s),
                          markersize=8, label=_label(s)) for s in schedules]
    ax.legend(handles=handles, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "grad_norm_vs_cosim.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 7))
    for sched in schedules:
        runs = gs_groups[sched]
        all_steps = []
        all_cosims = []
        for r in runs:
            gsl = r.get("grad_sim_log", {})
            steps = gsl.get("step", [])
            cosims = gsl.get("burst_vs_other", [])
            if len(steps) == len(cosims) and len(steps) > 0:
                all_steps.append(np.array(steps))
                all_cosims.append(np.array(cosims))

        if not all_steps:
            continue
        steps_ref = all_steps[0]
        interp_vals = [np.interp(steps_ref, s, v) for s, v in zip(all_steps, all_cosims)]
        arr = np.array(interp_vals)
        m = arr.mean(axis=0)
        n_s = len(arr)
        ci = 1.96 * arr.std(axis=0) / np.sqrt(n_s) if n_s > 1 else arr.std(axis=0)
        ax.plot(steps_ref, m, color=_color(sched), lw=2, label=_label(sched))
        ax.fill_between(steps_ref, m - ci, m + ci, color=_color(sched), alpha=0.15)

    ax.axhline(0, color="gray", ls=":", lw=1, alpha=0.5)
    _style(ax, "Step", "Cosine Similarity",
           "Gradient Cosine Similarity Over Time (every checkpoint)")
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "cosim_over_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    for sched in schedules:
        runs = gs_groups[sched]
        layer_data = defaultdict(list)
        steps_ref = None
        for r in runs:
            gsl = r.get("grad_sim_log", {})
            per_layer = gsl.get("per_layer", {})
            steps = gsl.get("step", [])
            if not per_layer or not steps:
                continue
            if steps_ref is None:
                steps_ref = np.array(steps)
            for layer, vals in per_layer.items():
                if len(vals) == len(steps):
                    layer_data[layer].append(np.interp(steps_ref, steps, vals))

        if not layer_data or steps_ref is None:
            continue

        fig, ax = plt.subplots(figsize=(14, 7))
        cmap = plt.get_cmap("tab20")
        layers = sorted(layer_data.keys())
        for li, layer in enumerate(layers):
            arr = np.array(layer_data[layer])
            m = arr.mean(axis=0)
            ax.plot(steps_ref, m, color=cmap(li / max(len(layers) - 1, 1)),
                    lw=1.5, label=layer)

        ax.axhline(0, color="gray", ls=":", lw=1, alpha=0.5)
        _style(ax, "Step", "Cosine Similarity",
               f"{_label(sched)}: Per-Layer Cosine Similarity Over Time")
        ax.legend(fontsize=7, loc="best", ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / f"cosim_per_layer_{sched}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_grad_rank(gs_records: list, out_dir: Path):
    """Re-plot grad rank from existing data, investigating issues."""
    out_dir.mkdir(parents=True, exist_ok=True)

    gs_groups = defaultdict(list)
    for r in gs_records:
        gs_groups[r["schedule"]].append(r)

    schedules = sorted(gs_groups.keys(), key=_sched_order)

    for sched in schedules:
        runs = gs_groups[sched]
        layer_data = defaultdict(list)
        steps_ref = None

        for r in runs:
            gsl = r.get("grad_sim_log", {})
            rank_data = gsl.get("grad_rank", {})
            steps = gsl.get("step", [])
            if not rank_data or not steps:
                continue
            if steps_ref is None:
                steps_ref = np.array(steps)
            for layer, vals in rank_data.items():
                clean_vals = [v if v == v else 0.0 for v in vals]
                if len(clean_vals) == len(steps):
                    layer_data[layer].append(np.interp(steps_ref, steps, clean_vals))

        if not layer_data or steps_ref is None:
            continue

        fig, ax = plt.subplots(figsize=(14, 7))
        cmap = plt.get_cmap("tab20")
        layers = sorted(layer_data.keys())
        for li, layer in enumerate(layers):
            arr = np.array(layer_data[layer])
            valid_mask = arr > 0
            m = np.where(valid_mask.any(axis=0), np.nanmean(np.where(valid_mask, arr, np.nan), axis=0), 0)
            ax.plot(steps_ref, m, color=cmap(li / max(len(layers) - 1, 1)),
                    lw=1.5, label=layer)

        _style(ax, "Step", "Effective Rank",
               f"{_label(sched)}: Gradient Effective Rank Per Layer\n(NaN values replaced with 0)")
        ax.legend(fontsize=7, loc="best", ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / f"grad_rank_{sched}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 7))
    for sched in schedules:
        runs = gs_groups[sched]
        all_steps = []
        all_mean_ranks = []
        for r in runs:
            gsl = r.get("grad_sim_log", {})
            rank_data = gsl.get("grad_rank", {})
            steps = gsl.get("step", [])
            if not rank_data or not steps:
                continue
            per_step_means = []
            for si in range(len(steps)):
                vals = [rank_data[layer][si] for layer in rank_data
                        if si < len(rank_data[layer]) and rank_data[layer][si] == rank_data[layer][si]]
                per_step_means.append(np.mean(vals) if vals else 0)
            all_steps.append(np.array(steps))
            all_mean_ranks.append(np.array(per_step_means))

        if not all_steps:
            continue
        steps_ref = all_steps[0]
        interp_vals = [np.interp(steps_ref, s, v) for s, v in zip(all_steps, all_mean_ranks)]
        arr = np.array(interp_vals)
        m = arr.mean(axis=0)
        ax.plot(steps_ref, m, color=_color(sched), lw=2, label=_label(sched))

    _style(ax, "Step", "Mean Effective Rank",
           "Mean Gradient Effective Rank Over Training")
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "grad_rank_mean_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sharpness(loss_surface_data: dict, out_dir: Path):
    """Plot sharpness bars for all burst settings."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not loss_surface_data:
        return

    schedules = sorted(loss_surface_data.keys(), key=_sched_order)
    for loss_type in ["burst", "other"]:
        sharpness_key = f"{loss_type}_sharpness"
        vals = [loss_surface_data[s].get(sharpness_key, 0) for s in schedules]
        per_seed_key = f"per_seed_{loss_type}_sharpness"
        cis = []
        for s in schedules:
            ps = loss_surface_data[s].get(per_seed_key, [])
            if len(ps) > 1:
                cis.append(1.96 * np.std(ps) / np.sqrt(len(ps)))
            else:
                cis.append(0)

        fig, ax = plt.subplots(figsize=(12, 7))
        xs = np.arange(len(schedules))
        colors = [_color(s) for s in schedules]
        ax.bar(xs, vals, yerr=cis, color=colors, edgecolor="black", lw=0.8,
               capsize=5, alpha=0.85)
        ax.set_xticks(xs)
        ax.set_xticklabels([_label(s) for s in schedules], fontsize=9, rotation=30, ha="right")
        _style(ax, "", "Sharpness (max - centre loss)",
               f"Loss Surface Sharpness ({loss_type}) at Peak Burst")
        fig.tight_layout()
        fig.savefig(out_dir / f"sharpness_{loss_type}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--basin-runs", type=int, default=50)
    parser.add_argument("--basin-points", type=int, default=8)
    args = parser.parse_args()

    run_dir = args.run_dir
    cfg_path, logs_dir, results_dir = resolve_run_paths(run_dir)
    with open(cfg_path) as f:
        run_cfg = json.load(f)

    rc = parse_run_config(run_cfg)

    with open(logs_dir / "_data.pkl", "rb") as f:
        target_pool, bg_pool, _, _, _ = pickle.load(f)

    burst_docs_BL = np.concatenate(list(target_pool.values()))
    other_docs_BL = np.concatenate(list(bg_pool.values()))

    with open(logs_dir / "all_results.pkl", "rb") as f:
        all_results = pickle.load(f)

    ckpt_root = logs_dir / "checkpoints"
    out_dir = results_dir / "new_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    P = rc["base_cfg"].get("pre_burst_steps", 0)

    t_total_start = time.time()

    print("\n[1/8] Layerwise weight difference...", flush=True)
    t0 = time.time()
    wd_data = compute_layerwise_weight_diff(ckpt_root, all_results, n_seeds=args.n_seeds)
    plot_layerwise_weight_diff(wd_data, out_dir / "weight_diff", P=P)
    print(f"  Done in {time.time() - t0:.1f}s", flush=True)

    print("\n[2/8] Per-layer activations...", flush=True)
    t0 = time.time()
    act_data = compute_layerwise_activations(
        ckpt_root, all_results, burst_docs_BL, other_docs_BL, n_seeds=args.n_seeds)
    plot_layerwise_activations(act_data, out_dir / "activations")
    print(f"  Done in {time.time() - t0:.1f}s", flush=True)

    print(f"\n[3/8] Loss basin ({args.basin_runs} directions, {args.basin_points} points)...", flush=True)
    t0 = time.time()
    basin_data = compute_loss_basin_random_directions(
        ckpt_root, all_results, burst_docs_BL, other_docs_BL,
        n_seeds=min(args.n_seeds, 2), n_directions=args.basin_runs,
        n_points=args.basin_points)
    plot_loss_basin(basin_data, out_dir / "loss_basin")
    basin_time = time.time() - t0
    print(f"  Done in {basin_time:.1f}s", flush=True)

    print("\n[4/8] Weight norm hypothesis...", flush=True)
    t0 = time.time()
    wn_data = compute_weight_norms(ckpt_root, all_results, n_seeds=args.n_seeds)
    plot_weight_norms(wn_data, out_dir / "weight_norms")
    print(f"  Done in {time.time() - t0:.1f}s", flush=True)

    print("\n[5/8] Sharpness (from basin_metrics)...", flush=True)
    t0 = time.time()
    try:
        from burst.basin_metrics import analyse_run as bm_analyse
        bm_result = bm_analyse(run_dir, n_seeds=args.n_seeds, skip_surface=False)
        ls_data = bm_result.get("loss_surface", {})
        plot_sharpness(ls_data, out_dir / "sharpness")
    except Exception as e:
        print(f"  WARNING: sharpness failed: {e}", flush=True)
        ls_data = {}
    print(f"  Done in {time.time() - t0:.1f}s", flush=True)

    print("\n[6/8] Gradient norms and cosim...", flush=True)
    t0 = time.time()
    from burst.pres_charts import load_grad_sim_data
    gs_records = load_grad_sim_data(run_dir)
    if gs_records:
        plot_grad_norms_and_cosim(gs_records, out_dir / "grad_norms", P=P)
    print(f"  Done in {time.time() - t0:.1f}s", flush=True)

    print("\n[7/8] Grad rank investigation...", flush=True)
    t0 = time.time()
    rank_info = investigate_grad_rank(run_dir)
    if rank_info.get("issues"):
        print(f"  Issues found: {rank_info['issues'][:5]}", flush=True)
    if gs_records:
        plot_grad_rank(gs_records, out_dir / "grad_rank")
    print(f"  Done in {time.time() - t0:.1f}s", flush=True)

    print("\n[8/8] Saving results...", flush=True)
    summary = {
        "basin_time_seconds": basin_time,
        "total_time_seconds": time.time() - t_total_start,
        "basin_runs": args.basin_runs,
        "basin_points": args.basin_points,
        "n_seeds": args.n_seeds,
        "weight_norm_data": {s: v for s, v in wn_data.items()},
        "grad_rank_issues": rank_info.get("issues", []),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    total_time = time.time() - t_total_start
    print(f"\nAll done in {total_time:.1f}s ({total_time/60:.1f} min)", flush=True)
    print(f"Basin took {basin_time:.1f}s for {args.basin_runs} directions", flush=True)

    if basin_time < 300:
        suggested = min(10000, max(1000, int(args.basin_runs * 300 / max(basin_time, 1))))
        print(f"Basin was fast — consider scaling to {suggested} directions", flush=True)

    print(f"\nResults saved to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
