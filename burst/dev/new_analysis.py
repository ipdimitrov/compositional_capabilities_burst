"""New analysis metrics for burst experiments.

Computes:
  1. Layerwise weight difference throughout training (vs pre-burst checkpoint)
  2. Per-layer activations during training for burst and other data
  3. Loss basin with N random directions -- magnitude and variance charts
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

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable

mpl.use("Agg")

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from burst.config import (  # noqa: E402
    parse_run_config,
)
from burst.core.metrics.gradients import _layer_groups  # noqa: E402
from burst.core.train_utils import DEVICE, load_net, resolve_run_paths  # noqa: E402
from burst.dev._shared import cross_entropy_loss as _cross_entropy_loss_shared  # noqa: E402

_rng = np.random.default_rng()
_DIR_NORM_EPS = 1e-10
_BASIN_FAST_THRESHOLD = 300
_MIN_SEEDS_FOR_SCATTER = 2


from burst.dev._shared import ckpt_files as _ckpt_files  # noqa: E402
from burst.dev._shared import sched_color as _color  # noqa: E402
from burst.dev._shared import sched_label as _label  # noqa: E402
from burst.dev._shared import sched_order as _sched_order  # noqa: E402


def _is_nan(v: float) -> bool:
    """Check if a value is NaN via self-comparison."""
    return math.isnan(v)


# ---------------------------------------------------------------------------
# 1. Layerwise weight difference throughout training
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_layerwise_weight_diff(  # noqa: C901
    ckpt_root: Path,
    all_results: list[dict],
    n_seeds: int = 3,
) -> dict:
    """Compute per-layer ||W_step - W_pre_burst|| at each checkpoint for each schedule/seed."""
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
            pre_sd = {
                k: v.float().cpu()
                for k, v in torch.load(
                    str(files[pre_step]), map_location="cpu", weights_only=True
                ).items()
            }

            layer_groups = None
            steps_data = []
            for step in available:
                sd = {
                    k: v.float().cpu()
                    for k, v in torch.load(
                        str(files[step]), map_location="cpu", weights_only=True
                    ).items()
                }

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
                    per_layer[name] = float(diff_norm**0.5)

                total_diff = float(
                    sum((sd[k] - pre_sd[k]).norm().item() ** 2 for k in sd) ** 0.5
                )

                steps_data.append(
                    {
                        "step": step,
                        "per_layer": per_layer,
                        "total_diff": total_diff,
                    }
                )

            all_seed_data.append(steps_data)
            seeds_done += 1

        results[sched] = all_seed_data

    return results


# ---------------------------------------------------------------------------
# 2. Per-layer activations during training
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_layerwise_activations(  # noqa: C901, PLR0913, PLR0915
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
    burst_idx = _rng.choice(burst_docs_BL.shape[0], n_burst, replace=False)
    other_idx = _rng.choice(other_docs_BL.shape[0], n_other, replace=False)
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

                hooks = []
                activations: dict[str, float] = {}

                def make_hook(
                    name: str, _acts: dict[str, float] = activations
                ) -> Callable[..., None]:
                    """Create forward hook capturing activation norm."""
                    def hook_fn(
                        _module: nn.Module, _inp: tuple, output: torch.Tensor | tuple,
                    ) -> None:
                        out = output[0] if isinstance(output, tuple) else output
                        _acts[name] = (
                            out.detach().float().norm(dim=-1).mean().item()
                        )

                    return hook_fn

                for i, block in enumerate(net.transformer.h):
                    hooks.append(
                        block.register_forward_hook(make_hook(f"L{i}"))
                    )

                burst_t = torch.as_tensor(
                    burst_eval, dtype=torch.long, device=DEVICE
                )
                with torch.amp.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"
                ):
                    net(burst_t[:, :-1])
                layer_norms_burst = dict(activations)
                activations.clear()

                other_t = torch.as_tensor(
                    other_eval, dtype=torch.long, device=DEVICE
                )
                with torch.amp.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"
                ):
                    net(other_t[:, :-1])
                layer_norms_other = dict(activations)

                for h in hooks:
                    h.remove()
                del net

                steps_data.append(
                    {
                        "step": step,
                        "burst_norms": layer_norms_burst,
                        "other_norms": layer_norms_other,
                    }
                )

            all_seed_data.append(steps_data)
            seeds_done += 1

        results[sched] = all_seed_data

    return results


# ---------------------------------------------------------------------------
# 3. Loss basin with N random directions
# ---------------------------------------------------------------------------


_cross_entropy_loss = _cross_entropy_loss_shared


@torch.no_grad()
def compute_loss_basin_random_directions(  # noqa: PLR0913
    ckpt_root: Path,
    all_results: list[dict],
    burst_docs_BL: np.ndarray,
    other_docs_BL: np.ndarray,
    n_seeds: int = 2,
    n_directions: int = 50,
    n_points: int = 8,
    max_epsilon: float = 1.0,
) -> dict:
    """Sample n_directions random directions and evaluate loss along each.

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
    burst_idx = _rng.choice(burst_docs_BL.shape[0], n_burst, replace=False)
    other_idx = _rng.choice(other_docs_BL.shape[0], n_other, replace=False)
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

            base_sd = {
                k: v.float().cpu()
                for k, v in torch.load(
                    str(files[peak_step]),
                    map_location="cpu",
                    weights_only=True,
                ).items()
            }

            net = load_net(cfg, str(files[peak_step]))

            for _d_idx in range(n_directions):
                direction = {
                    k: torch.randn_like(v) for k, v in base_sd.items()
                }
                dir_flat = torch.cat(
                    [v.view(-1) for v in direction.values()]
                )
                dir_norm = dir_flat.norm()
                if dir_norm < _DIR_NORM_EPS:
                    continue
                direction = {
                    k: v / dir_norm for k, v in direction.items()
                }

                burst_losses = []
                other_losses = []

                for eps in epsilons:
                    perturbed = {
                        k: (base_sd[k] + eps * direction[k]).to(DEVICE)
                        for k in base_sd
                    }
                    net.load_state_dict(perturbed)
                    burst_losses.append(
                        _cross_entropy_loss(net, burst_eval)
                    )
                    other_losses.append(
                        _cross_entropy_loss(net, other_eval)
                    )

                all_direction_losses_burst.append(burst_losses)
                all_direction_losses_other.append(other_losses)

            seeds_done += 1
            logger.debug("%s: %d directions done", label, n_directions)
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

            sd = torch.load(
                str(files[peak_step]),
                map_location="cpu",
                weights_only=True,
            )
            total_norm = float(
                sum(v.float().norm().item() ** 2 for v in sd.values()) ** 0.5
            )
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
        with fp.open() as f:
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
        with fp.open() as f:
            records.append(json.load(f))

    issues = []
    for rec in records:
        gsl = rec.get("grad_sim_log", {})
        rank_data = gsl.get("grad_rank", {})
        if not rank_data:
            issues.append(f"{rec.get('label', '?')}: no grad_rank data")
            continue
        for layer, vals in rank_data.items():
            nan_count = sum(1 for v in vals if _is_nan(v))
            if nan_count > 0:
                issues.append(
                    f"{rec.get('label', '?')} {layer}: "
                    f"{nan_count}/{len(vals)} NaN values"
                )

    return {"records": records, "issues": issues}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _style(
    ax: mpl.axes.Axes, xl: str = "", yl: str = "", t: str = "",
) -> None:
    ax.set_xlabel(xl, fontsize=11, fontweight="bold")
    ax.set_ylabel(yl, fontsize=11, fontweight="bold")
    if t:
        ax.set_title(t, fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(labelsize=9)
    ax.grid(visible=True, alpha=0.15, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_layerwise_weight_diff(
    data: dict, out_dir: Path, _P: int = 0,
) -> None:
    """Plot total and per-layer weight differences over training."""
    out_dir.mkdir(parents=True, exist_ok=True)
    schedules = sorted(data.keys(), key=_sched_order)

    fig, ax = plt.subplots(figsize=(14, 7))
    for sched in schedules:
        seed_data = data[sched]
        if not seed_data:
            continue
        steps_ref = [d["step"] for d in seed_data[0]]
        total_diffs = np.array(
            [[d["total_diff"] for d in sd] for sd in seed_data]
        )
        m = total_diffs.mean(axis=0)
        n_s = len(total_diffs)
        ci = (
            1.96 * total_diffs.std(axis=0) / np.sqrt(n_s)
            if n_s > 1
            else total_diffs.std(axis=0)
        )
        ax.plot(steps_ref, m, color=_color(sched), lw=2, label=_label(sched))
        ax.fill_between(
            steps_ref, m - ci, m + ci, color=_color(sched), alpha=0.15
        )

    _style(
        ax,
        "Step",
        "||W_step - W_pre||_2",
        "Total Weight Difference vs Pre-Burst (mean +/- 95% CI)",
    )
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(
        out_dir / "weight_diff_total.png", dpi=150, bbox_inches="tight"
    )
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
            vals = np.array(
                [[d["per_layer"][layer] for d in sd] for sd in seed_data]
            )
            m = vals.mean(axis=0)
            ax.plot(
                steps_ref,
                m,
                color=cmap(li / max(len(layers) - 1, 1)),
                lw=1.5,
                label=layer,
            )

        _style(
            ax,
            "Step",
            "||W_layer - W_pre_layer||_2",
            f"{_label(sched)}: Per-Layer Weight Difference",
        )
        ax.legend(fontsize=7, loc="best", ncol=2)
        fig.tight_layout()
        fig.savefig(
            out_dir / f"weight_diff_layers_{sched}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_layerwise_activations(data: dict, out_dir: Path) -> None:
    """Plot per-layer activation norms over training."""
    out_dir.mkdir(parents=True, exist_ok=True)
    schedules = sorted(data.keys(), key=_sched_order)

    for data_type, data_key in [
        ("burst", "burst_norms"),
        ("other", "other_norms"),
    ]:
        fig, ax = plt.subplots(figsize=(14, 7))
        for sched in schedules:
            seed_data = data[sched]
            if not seed_data:
                continue
            steps_ref = [d["step"] for d in seed_data[0]]
            all_means = []
            for sd in seed_data:
                mean_per_step = [
                    np.mean(list(d[data_key].values())) for d in sd
                ]
                all_means.append(mean_per_step)
            arr = np.array(all_means)
            m = arr.mean(axis=0)
            n_s = len(arr)
            ci = (
                1.96 * arr.std(axis=0) / np.sqrt(n_s)
                if n_s > 1
                else arr.std(axis=0)
            )
            ax.plot(
                steps_ref, m, color=_color(sched), lw=2, label=_label(sched)
            )
            ax.fill_between(
                steps_ref, m - ci, m + ci, color=_color(sched), alpha=0.15
            )

        _style(
            ax,
            "Step",
            "Mean Activation Norm",
            f"Mean Activation Norm ({data_type} data) Over Training",
        )
        ax.legend(fontsize=9, loc="best")
        fig.tight_layout()
        fig.savefig(
            out_dir / f"activation_norm_{data_type}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    for sched in schedules:
        seed_data = data[sched]
        if not seed_data:
            continue
        steps_ref = [d["step"] for d in seed_data[0]]
        layers = list(seed_data[0][0]["burst_norms"].keys())

        for data_type, data_key in [
            ("burst", "burst_norms"),
            ("other", "other_norms"),
        ]:
            fig, ax = plt.subplots(figsize=(14, 7))
            cmap = plt.get_cmap("tab20")
            for li, layer in enumerate(layers):
                vals = np.array(
                    [
                        [d[data_key].get(layer, 0) for d in sd]
                        for sd in seed_data
                    ]
                )
                m = vals.mean(axis=0)
                ax.plot(
                    steps_ref,
                    m,
                    color=cmap(li / max(len(layers) - 1, 1)),
                    lw=1.5,
                    label=layer,
                )

            _style(
                ax,
                "Step",
                "Activation Norm",
                f"{_label(sched)}: Per-Layer Activation Norm ({data_type})",
            )
            ax.legend(fontsize=7, loc="best", ncol=2)
            fig.tight_layout()
            fig.savefig(
                out_dir / f"activation_layers_{sched}_{data_type}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)


def plot_loss_basin(data: dict, out_dir: Path) -> None:
    """Plot loss basin magnitude and variance charts."""
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

            ax_mag.plot(
                epsilons,
                mean_per_eps,
                color=_color(sched),
                lw=2,
                label=_label(sched),
            )
            ax_var.plot(
                epsilons,
                var_per_eps,
                color=_color(sched),
                lw=2,
                label=_label(sched),
            )

        _style(
            ax_mag,
            "eps (perturbation)",
            "Mean Loss",
            (
                f"Loss Basin: Mean Loss ({loss_type}) Across "
                f"{data[schedules[0]].get('n_directions', '?')}"
                " Random Directions"
            ),
        )
        ax_mag.legend(fontsize=9, loc="best")
        fig_mag.tight_layout()
        fig_mag.savefig(
            out_dir / f"basin_magnitude_{loss_type}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig_mag)

        _style(
            ax_var,
            "eps (perturbation)",
            "Variance of Loss",
            f"Loss Basin: Variance ({loss_type}) Across Random Directions",
        )
        ax_var.legend(fontsize=9, loc="best")
        fig_var.tight_layout()
        fig_var.savefig(
            out_dir / f"basin_variance_{loss_type}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig_var)


def plot_loss_basin_per_schedule(data: dict, out_dir: Path) -> None:
    """Plot per-schedule chart: burst variance vs other variance."""
    out_dir.mkdir(parents=True, exist_ok=True)
    schedules = sorted(data.keys(), key=_sched_order)

    for sched in schedules:
        d = data[sched]
        epsilons = d["epsilons"]
        burst_arr = (
            np.array(d["burst_losses"]) if d["burst_losses"] else None
        )
        other_arr = (
            np.array(d["other_losses"]) if d["other_losses"] else None
        )
        if burst_arr is None and other_arr is None:
            continue

        fig, ax = plt.subplots(figsize=(14, 7))
        if burst_arr is not None:
            ax.plot(
                epsilons,
                burst_arr.var(axis=0),
                color="#E91E63",
                lw=2,
                label="Burst data",
            )
        if other_arr is not None:
            ax.plot(
                epsilons,
                other_arr.var(axis=0),
                color="#2196F3",
                lw=2,
                label="Other data",
            )

        _style(
            ax,
            "eps (perturbation)",
            "Variance of Loss",
            f"{_label(sched)}: Variance Across Directions (burst vs other)",
        )
        ax.legend(fontsize=11, loc="best")
        fig.tight_layout()
        fig.savefig(
            out_dir / f"basin_variance_compare_{sched}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_weight_norms(data: dict, out_dir: Path) -> None:
    """Plot weight norm bar chart and scatter vs burst percentage."""
    out_dir.mkdir(parents=True, exist_ok=True)
    schedules = sorted(data.keys(), key=_sched_order)

    fig, ax = plt.subplots(figsize=(12, 7))
    xs = np.arange(len(schedules))
    means = [np.mean(data[s]) if data[s] else 0 for s in schedules]
    cis = [
        1.96 * np.std(data[s]) / np.sqrt(len(data[s]))
        if len(data[s]) > 1
        else 0
        for s in schedules
    ]
    colors = [_color(s) for s in schedules]

    ax.bar(
        xs,
        means,
        yerr=cis,
        color=colors,
        edgecolor="black",
        lw=0.8,
        capsize=5,
        alpha=0.85,
    )
    for i, vals in enumerate(schedules):
        jit = np.random.default_rng(42).uniform(
            -0.12, 0.12, len(data[vals])
        )
        ax.scatter(
            np.full(len(data[vals]), i) + jit,
            data[vals],
            color="black",
            s=30,
            zorder=5,
            alpha=0.5,
            edgecolor="white",
            lw=0.5,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(
        [_label(s) for s in schedules], fontsize=9, rotation=30, ha="right"
    )
    _style(
        ax,
        "",
        "||W||_2 (total weight norm at peak burst)",
        "Weight Norm at Peak Burst by Schedule\n"
        "(Hypothesis: more burst -> higher norm)",
    )
    fig.tight_layout()
    fig.savefig(
        out_dir / "weight_norm_hypothesis.png", dpi=150, bbox_inches="tight"
    )
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

    if len(burst_pcts) > _MIN_SEEDS_FOR_SCATTER:
        fig, ax = plt.subplots(figsize=(10, 7))
        for s in schedules:
            try:
                pct = int(s.split("_")[1])
            except (IndexError, ValueError):
                continue
            for n in data[s]:
                ax.scatter(
                    pct, n, color=_color(s), s=60, edgecolor="black", lw=0.5
                )

        corr = np.corrcoef(burst_pcts, norm_vals)[0, 1]
        z = np.polyfit(burst_pcts, norm_vals, 1)
        xline = np.linspace(min(burst_pcts), max(burst_pcts), 100)
        ax.plot(xline, np.polyval(z, xline), "k--", lw=1.5, alpha=0.5)
        ax.text(
            0.05,
            0.95,
            f"r = {corr:.3f}",
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            va="top",
        )

        _style(
            ax,
            "Burst %",
            "||W||_2",
            "Weight Norm vs Burst Percentage\n(each dot = one seed)",
        )
        fig.tight_layout()
        fig.savefig(
            out_dir / "weight_norm_vs_burst_pct.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_grad_norms_and_cosim(  # noqa: C901, PLR0912, PLR0915
    gs_records: list, out_dir: Path, _P: int = 0,
) -> None:
    """Plot gradient norms (L1, L2, Linf) over time and per layer, correlated with cosim."""
    out_dir.mkdir(parents=True, exist_ok=True)

    gs_groups: dict[str, list] = defaultdict(list)
    for r in gs_records:
        gs_groups[r["schedule"]].append(r)

    schedules = sorted(gs_groups.keys(), key=_sched_order)

    for norm_type, norm_label in [
        ("burst_norm", "||g_burst||_2"),
        ("other_norm", "||g_other||_2"),
        ("burst_l1", "||g_burst||_1"),
        ("other_l1", "||g_other||_1"),
        ("burst_linf", "||g_burst||_inf"),
        ("other_linf", "||g_other||_inf"),
    ]:
        fig, ax = plt.subplots(figsize=(14, 7))
        for sched in schedules:
            runs = gs_groups[sched]
            all_steps = []
            all_vals = []
            for r in runs:
                gsl = r.get("grad_sim_log", {})
                proj = gsl.get(
                    "grad_projection", r.get("grad_projection_log", {})
                )
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
            interp_vals = [
                np.interp(steps_ref, s, v)
                for s, v in zip(all_steps, all_vals, strict=True)
            ]
            arr = np.array(interp_vals)
            m = arr.mean(axis=0)
            n_s = len(arr)
            ci = (
                1.96 * arr.std(axis=0) / np.sqrt(n_s)
                if n_s > 1
                else arr.std(axis=0)
            )
            ax.plot(
                steps_ref, m, color=_color(sched), lw=2, label=_label(sched)
            )
            ax.fill_between(
                steps_ref, m - ci, m + ci, color=_color(sched), alpha=0.15
            )

        _style(
            ax,
            "Step",
            norm_label,
            f"Gradient Norm ({norm_label}) Over Training",
        )
        ax.legend(fontsize=9, loc="best")
        fig.tight_layout()
        fig.savefig(
            out_dir / f"grad_norm_{norm_type}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    for sched in schedules:
        runs = gs_groups[sched]
        for r in runs:
            gsl = r.get("grad_sim_log", {})
            proj = gsl.get(
                "grad_projection", r.get("grad_projection_log", {})
            )
            cosims = gsl.get("burst_vs_other", [])
            burst_norms = proj.get("burst_norm", [])
            if not cosims or not burst_norms:
                continue
            n = min(len(cosims), len(burst_norms))
            ax.scatter(
                cosims[:n],
                burst_norms[:n],
                color=_color(sched),
                s=15,
                alpha=0.4,
                edgecolor="none",
            )

    _style(
        ax,
        "Cosine Similarity (burst vs other)",
        "||g_burst||_2",
        "Gradient Norm vs Cosine Similarity\n(each dot = one step x seed)",
    )
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=_color(s),
            markersize=8,
            label=_label(s),
        )
        for s in schedules
    ]
    ax.legend(handles=handles, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(
        out_dir / "grad_norm_vs_cosim.png", dpi=150, bbox_inches="tight"
    )
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
        interp_vals = [
            np.interp(steps_ref, s, v)
            for s, v in zip(all_steps, all_cosims, strict=True)
        ]
        arr = np.array(interp_vals)
        m = arr.mean(axis=0)
        n_s = len(arr)
        ci = (
            1.96 * arr.std(axis=0) / np.sqrt(n_s)
            if n_s > 1
            else arr.std(axis=0)
        )
        ax.plot(steps_ref, m, color=_color(sched), lw=2, label=_label(sched))
        ax.fill_between(
            steps_ref, m - ci, m + ci, color=_color(sched), alpha=0.15
        )

    ax.axhline(0, color="gray", ls=":", lw=1, alpha=0.5)
    _style(
        ax,
        "Step",
        "Cosine Similarity",
        "Gradient Cosine Similarity Over Time (every checkpoint)",
    )
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(
        out_dir / "cosim_over_time.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    for sched in schedules:
        runs = gs_groups[sched]
        layer_data: dict[str, list] = defaultdict(list)
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
                    layer_data[layer].append(
                        np.interp(steps_ref, steps, vals)
                    )

        if not layer_data or steps_ref is None:
            continue

        fig, ax = plt.subplots(figsize=(14, 7))
        cmap = plt.get_cmap("tab20")
        layers = sorted(layer_data.keys())
        for li, layer in enumerate(layers):
            arr = np.array(layer_data[layer])
            m = arr.mean(axis=0)
            ax.plot(
                steps_ref,
                m,
                color=cmap(li / max(len(layers) - 1, 1)),
                lw=1.5,
                label=layer,
            )

        ax.axhline(0, color="gray", ls=":", lw=1, alpha=0.5)
        _style(
            ax,
            "Step",
            "Cosine Similarity",
            f"{_label(sched)}: Per-Layer Cosine Similarity Over Time",
        )
        ax.legend(fontsize=7, loc="best", ncol=2)
        fig.tight_layout()
        fig.savefig(
            out_dir / f"cosim_per_layer_{sched}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_grad_rank(  # noqa: C901, PLR0912, PLR0915
    gs_records: list, out_dir: Path,
) -> None:
    """Re-plot grad rank from existing data, investigating issues."""
    out_dir.mkdir(parents=True, exist_ok=True)

    gs_groups: dict[str, list] = defaultdict(list)
    for r in gs_records:
        gs_groups[r["schedule"]].append(r)

    schedules = sorted(gs_groups.keys(), key=_sched_order)

    for sched in schedules:
        runs = gs_groups[sched]
        layer_data: dict[str, list] = defaultdict(list)
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
                clean_vals = [
                    v if not _is_nan(v) else 0.0 for v in vals
                ]
                if len(clean_vals) == len(steps):
                    layer_data[layer].append(
                        np.interp(steps_ref, steps, clean_vals)
                    )

        if not layer_data or steps_ref is None:
            continue

        fig, ax = plt.subplots(figsize=(14, 7))
        cmap = plt.get_cmap("tab20")
        layers = sorted(layer_data.keys())
        for li, layer in enumerate(layers):
            arr = np.array(layer_data[layer])
            valid_mask = arr > 0
            m = np.where(
                valid_mask.any(axis=0),
                np.nanmean(np.where(valid_mask, arr, np.nan), axis=0),
                0,
            )
            ax.plot(
                steps_ref,
                m,
                color=cmap(li / max(len(layers) - 1, 1)),
                lw=1.5,
                label=layer,
            )

        _style(
            ax,
            "Step",
            "Effective Rank",
            f"{_label(sched)}: Gradient Effective Rank Per Layer\n"
            "(NaN values replaced with 0)",
        )
        ax.legend(fontsize=7, loc="best", ncol=2)
        fig.tight_layout()
        fig.savefig(
            out_dir / f"grad_rank_{sched}.png", dpi=150, bbox_inches="tight"
        )
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
                vals = [
                    rank_data[layer][si]
                    for layer in rank_data
                    if si < len(rank_data[layer])
                    and not _is_nan(rank_data[layer][si])
                ]
                per_step_means.append(np.mean(vals) if vals else 0)
            all_steps.append(np.array(steps))
            all_mean_ranks.append(np.array(per_step_means))

        if not all_steps:
            continue
        steps_ref = all_steps[0]
        interp_vals = [
            np.interp(steps_ref, s, v)
            for s, v in zip(all_steps, all_mean_ranks, strict=True)
        ]
        arr = np.array(interp_vals)
        m = arr.mean(axis=0)
        ax.plot(steps_ref, m, color=_color(sched), lw=2, label=_label(sched))

    _style(
        ax,
        "Step",
        "Mean Effective Rank",
        "Mean Gradient Effective Rank Over Training",
    )
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(
        out_dir / "grad_rank_mean_all.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def plot_sharpness(loss_surface_data: dict, out_dir: Path) -> None:
    """Plot sharpness bars for all burst settings."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not loss_surface_data:
        return

    schedules = sorted(loss_surface_data.keys(), key=_sched_order)
    for loss_type in ["burst", "other"]:
        sharpness_key = f"{loss_type}_sharpness"
        vals = [
            loss_surface_data[s].get(sharpness_key, 0) for s in schedules
        ]
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
        ax.bar(
            xs,
            vals,
            yerr=cis,
            color=colors,
            edgecolor="black",
            lw=0.8,
            capsize=5,
            alpha=0.85,
        )
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [_label(s) for s in schedules],
            fontsize=9,
            rotation=30,
            ha="right",
        )
        _style(
            ax,
            "",
            "Sharpness (max - centre loss)",
            f"Loss Surface Sharpness ({loss_type}) at Peak Burst",
        )
        fig.tight_layout()
        fig.savefig(
            out_dir / f"sharpness_{loss_type}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: PLR0915
    """Run the full new-analysis pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--basin-runs", type=int, default=50)
    parser.add_argument("--basin-points", type=int, default=5)
    parser.add_argument("--basin-max-epsilon", type=float, default=1.0)
    parser.add_argument(
        "--only-basin-sharpness",
        action="store_true",
        help="Only run loss basin + sharpness (skip all other stages)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    cfg_path, logs_dir, results_dir = resolve_run_paths(run_dir)
    with cfg_path.open() as f:
        run_cfg = json.load(f)

    rc = parse_run_config(run_cfg)

    with (logs_dir / "_data.pkl").open("rb") as f:
        target_pool, bg_pool, _, _, _ = pickle.load(f)  # noqa: S301

    burst_docs_BL = np.concatenate(list(target_pool.values()))
    other_docs_BL = np.concatenate(list(bg_pool.values()))

    with (logs_dir / "all_results.pkl").open("rb") as f:
        all_results = pickle.load(f)  # noqa: S301

    ckpt_root = logs_dir / "checkpoints"
    out_dir = results_dir / "new_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    P = rc["base_cfg"].get("pre_burst_steps", 0)

    t_total_start = time.time()
    basin_time = 0.0

    if not args.only_basin_sharpness:
        logger.info("[1/8] Layerwise weight difference...")
        t0 = time.time()
        wd_data = compute_layerwise_weight_diff(
            ckpt_root, all_results, n_seeds=args.n_seeds
        )
        plot_layerwise_weight_diff(wd_data, out_dir / "weight_diff", _P=P)
        logger.info("  Done in %.1fs", time.time() - t0)

        logger.info("[2/8] Per-layer activations...")
        t0 = time.time()
        act_data = compute_layerwise_activations(
            ckpt_root,
            all_results,
            burst_docs_BL,
            other_docs_BL,
            n_seeds=args.n_seeds,
        )
        plot_layerwise_activations(act_data, out_dir / "activations")
        logger.info("  Done in %.1fs", time.time() - t0)

    logger.info(
        "[3/8] Loss basin (%d directions, %d points)...",
        args.basin_runs,
        args.basin_points,
    )
    t0 = time.time()
    basin_data = compute_loss_basin_random_directions(
        ckpt_root,
        all_results,
        burst_docs_BL,
        other_docs_BL,
        n_seeds=args.n_seeds,
        n_directions=args.basin_runs,
        n_points=args.basin_points,
        max_epsilon=args.basin_max_epsilon,
    )
    plot_loss_basin(basin_data, out_dir / "loss_basin")
    plot_loss_basin_per_schedule(basin_data, out_dir / "loss_basin")
    basin_time = time.time() - t0
    logger.info("  Done in %.1fs", basin_time)

    if not args.only_basin_sharpness:
        logger.info("[4/8] Weight norm hypothesis...")
        t0 = time.time()
        wn_data = compute_weight_norms(
            ckpt_root, all_results, n_seeds=args.n_seeds
        )
        plot_weight_norms(wn_data, out_dir / "weight_norms")
        logger.info("  Done in %.1fs", time.time() - t0)

    logger.info("[5/8] Sharpness (from basin_metrics)...")
    t0 = time.time()
    try:
        from burst.dev.basin_metrics import analyse_run as bm_analyse  # noqa: PLC0415

        bm_result = bm_analyse(
            run_dir, n_seeds=args.n_seeds, skip_surface=False
        )
        ls_data = bm_result.get("loss_surface", {})
        plot_sharpness(ls_data, out_dir / "sharpness")
    except Exception:  # noqa: BLE001
        ls_data = {}
    logger.info("  Done in %.1fs", time.time() - t0)

    if not args.only_basin_sharpness:
        logger.info("[6/8] Gradient norms and cosim...")
        t0 = time.time()
        from burst.dev.pres_charts import load_grad_sim_data  # noqa: PLC0415

        gs_records = load_grad_sim_data(run_dir)
        if gs_records:
            plot_grad_norms_and_cosim(
                gs_records, out_dir / "grad_norms", _P=P
            )
        logger.info("  Done in %.1fs", time.time() - t0)

        logger.info("[7/8] Grad rank investigation...")
        t0 = time.time()
        rank_info = investigate_grad_rank(run_dir)
        if rank_info.get("issues"):
            logger.info("  Issues found: %s", rank_info["issues"][:5])
        if gs_records:
            plot_grad_rank(gs_records, out_dir / "grad_rank")
        logger.info("  Done in %.1fs", time.time() - t0)

    logger.info("[8/8] Saving results...")
    summary = {
        "basin_time_seconds": basin_time,
        "total_time_seconds": time.time() - t_total_start,
        "basin_runs": args.basin_runs,
        "basin_points": args.basin_points,
        "n_seeds": args.n_seeds,
        "only_basin_sharpness": args.only_basin_sharpness,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    total_time = time.time() - t_total_start
    logger.info(
        "All done in %.1fs (%.1f min)", total_time, total_time / 60
    )
    logger.info(
        "Basin took %.1fs for %d directions", basin_time, args.basin_runs
    )

    if basin_time < _BASIN_FAST_THRESHOLD:
        suggested = min(
            10000,
            max(
                1000,
                int(
                    args.basin_runs
                    * _BASIN_FAST_THRESHOLD
                    / max(basin_time, 1)
                ),
            ),
        )
        logger.info(
            "Basin was fast -- consider scaling to %d directions", suggested
        )

    logger.info("Results saved to: %s", out_dir)


if __name__ == "__main__":
    main()
