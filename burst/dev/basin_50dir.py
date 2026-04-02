"""50-direction loss basin analysis at peak burst.

For each (schedule, seed), loads the peak-burst checkpoint, samples 50
filter-normalised random directions (Li et al. 2018), and evaluates
cross-entropy loss at 8 epsilon points along each direction.

Produces:
  - Mean loss curves (burst / other) vs epsilon
  - Variance curves across directions
  - Sharpness bar chart: burst vs other per schedule (log scale)

Usage:
    python burst/basin_50dir.py data/20260321-133424_burst_d3_pos3_constant_steps
    python burst/basin_50dir.py <run_dir> --n-directions 100 --n-points 12
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

mpl.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from burst.config import (
    CLASS_BURST,
    CLASS_OTHER,
    parse_run_config,
)
from burst.core.train_utils import load_net, resolve_run_paths

logger = logging.getLogger(__name__)

from burst.core.train_utils import DEVICE
from burst.dev._shared import cross_entropy_loss as _cross_entropy_loss

_rng = np.random.default_rng()

N_DIRECTIONS = 50
N_POINTS = 8
MAX_EPSILON = 0.02
N_EVAL_DOCS = 128
_MIN_FILTER_DIM = 2


from burst.dev._shared import sched_color as _color
from burst.dev._shared import sched_label as _label
from burst.dev._shared import sched_order as _sched_order


def _filter_normalise(
    direction: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    normed = {}
    for name, d in direction.items():
        ref = reference[name].float()
        d_f = d.float()
        if d_f.dim() >= _MIN_FILTER_DIM:
            d_norms = d_f.view(d_f.shape[0], -1).norm(dim=1, keepdim=True)
            ref_norms = ref.view(ref.shape[0], -1).norm(dim=1, keepdim=True)
            scale = ref_norms / (d_norms + 1e-10)
            normed[name] = (d_f.view(d_f.shape[0], -1) * scale).view(d_f.shape)
        else:
            d_norm = d_f.norm()
            ref_norm = ref.norm()
            normed[name] = d_f * (ref_norm / (d_norm + 1e-10))
    return normed


@torch.no_grad()
def compute_one_seed(
    cfg: dict,
    ckpt_path: str,
    burst_eval: np.ndarray,
    other_eval: np.ndarray,
    n_directions: int,
    epsilons: np.ndarray,
) -> tuple[list[list[float]], list[list[float]]]:
    """Run filter-normalised perturbations and return (burst_losses, other_losses)."""
    base_sd = {
        k: v.float().cpu()
        for k, v in torch.load(
            ckpt_path, map_location="cpu", weights_only=True
        ).items()
    }
    net = load_net(cfg, ckpt_path)

    burst_all: list[list[float]] = []
    other_all: list[list[float]] = []

    for _ in range(n_directions):
        direction = _filter_normalise(
            {k: torch.randn_like(v) for k, v in base_sd.items()},
            base_sd,
        )

        burst_row: list[float] = []
        other_row: list[float] = []
        for eps in epsilons:
            perturbed = {
                k: (base_sd[k] + eps * direction[k]).to(DEVICE)
                for k in base_sd
            }
            net.load_state_dict(perturbed)
            burst_row.append(_cross_entropy_loss(net, burst_eval))
            other_row.append(_cross_entropy_loss(net, other_eval))

        burst_all.append(burst_row)
        other_all.append(other_row)

    del net
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return burst_all, other_all


def _ckpt_files(ckpt_dir: Path) -> dict[int, Path]:
    return {
        int(p.stem.split("_")[1]): p for p in ckpt_dir.glob("step_*.pt")
    }


def _style(
    ax: plt.Axes, xl: str = "", yl: str = "", t: str = ""
) -> None:
    ax.set_xlabel(xl, fontsize=11, fontweight="bold")
    ax.set_ylabel(yl, fontsize=11, fontweight="bold")
    if t:
        ax.set_title(t, fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(labelsize=9)
    ax.grid(visible=True, alpha=0.15, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_mean_and_variance(
    results: dict, out_dir: Path, n_directions: int
) -> None:
    """Plot mean and variance loss curves across random directions."""
    schedules = sorted(results.keys(), key=_sched_order)

    for loss_type in [CLASS_BURST, CLASS_OTHER]:
        key = f"{loss_type}_losses"

        fig_mag, ax_mag = plt.subplots(figsize=(14, 7))
        fig_var, ax_var = plt.subplots(figsize=(14, 7))

        for sched in schedules:
            d = results[sched]
            if not d[key]:
                continue
            epsilons = d["epsilons"]
            arr = np.array(d[key])
            ax_mag.plot(
                epsilons, arr.mean(axis=0),
                color=_color(sched), lw=2, label=_label(sched),
            )
            ax_var.plot(
                epsilons, arr.var(axis=0),
                color=_color(sched), lw=2, label=_label(sched),
            )

        title_suffix = (
            f"({loss_type}) Across {n_directions} Random Directions"
        )
        _style(
            ax_mag, "e (perturbation)", "Mean Loss",
            f"Loss Basin: Mean Loss {title_suffix}",
        )
        ax_mag.legend(fontsize=9, loc="best")
        fig_mag.tight_layout()
        fig_mag.savefig(
            out_dir / f"basin_mean_{loss_type}.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig_mag)

        _style(
            ax_var, "e (perturbation)", "Variance of Loss",
            f"Loss Basin: Variance {title_suffix}",
        )
        ax_var.legend(fontsize=9, loc="best")
        fig_var.tight_layout()
        fig_var.savefig(
            out_dir / f"basin_var_{loss_type}.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig_var)


def plot_sharpness(
    results: dict, out_dir: Path, n_directions: int
) -> None:
    """Plot grouped bar chart of basin sharpness burst vs other.

    Sharpness per (schedule, seed, direction) = max(loss along eps)
    minus loss at center. Average over directions per seed, then show
    mean +/- 95% CI across seeds.
    """
    schedules = sorted(results.keys(), key=_sched_order)
    epsilons = np.array(results[schedules[0]]["epsilons"])
    center_idx = len(epsilons) // 2

    burst_means, burst_cis = [], []
    other_means, other_cis = [], []

    for sched in schedules:
        d = results[sched]
        burst_arr = np.array(d["burst_losses"])
        other_arr = np.array(d["other_losses"])

        burst_sharp = burst_arr.max(axis=1) - burst_arr[:, center_idx]
        other_sharp = other_arr.max(axis=1) - other_arr[:, center_idx]

        n_seeds = d.get("n_seeds", 1)
        n_per_seed = len(burst_sharp) // max(n_seeds, 1)

        if n_seeds > 1 and n_per_seed > 0:
            seed_burst = [
                burst_sharp[i * n_per_seed : (i + 1) * n_per_seed].mean()
                for i in range(n_seeds)
            ]
            seed_other = [
                other_sharp[i * n_per_seed : (i + 1) * n_per_seed].mean()
                for i in range(n_seeds)
            ]
            burst_means.append(np.mean(seed_burst))
            other_means.append(np.mean(seed_other))
            burst_cis.append(
                1.96 * np.std(seed_burst) / np.sqrt(n_seeds)
            )
            other_cis.append(
                1.96 * np.std(seed_other) / np.sqrt(n_seeds)
            )
        else:
            burst_means.append(float(burst_sharp.mean()))
            other_means.append(float(other_sharp.mean()))
            burst_cis.append(0.0)
            other_cis.append(0.0)

    xs = np.arange(len(schedules))
    w = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(
        xs - w / 2, burst_means, w,
        yerr=burst_cis, label="Burst", color="#E91E63",
        edgecolor="black", lw=0.6, capsize=4, alpha=0.85,
    )
    ax.bar(
        xs + w / 2, other_means, w,
        yerr=other_cis, label="Other", color="#2196F3",
        edgecolor="black", lw=0.6, capsize=4, alpha=0.85,
    )
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [_label(s) for s in schedules],
        fontsize=9, rotation=30, ha="right",
    )
    _style(
        ax, "",
        "Sharpness (max - center loss, log scale)",
        f"Basin Sharpness: Burst vs Other"
        f" ({n_directions} directions, mean +/- 95% CI)",
    )
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(
        out_dir / "sharpness_comparison.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    """Run 50-direction filter-normalised loss basin analysis."""
    parser = argparse.ArgumentParser(
        description=(
            "50-direction filter-normalised loss basin analysis"
            " at peak burst."
        ),
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--n-directions", type=int, default=N_DIRECTIONS,
    )
    parser.add_argument("--n-points", type=int, default=N_POINTS)
    parser.add_argument(
        "--max-epsilon", type=float, default=MAX_EPSILON,
    )
    parser.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args()

    run_dir = args.run_dir
    cfg_path, logs_dir, results_dir = resolve_run_paths(run_dir)

    with cfg_path.open() as f:
        run_cfg = json.load(f)
    parse_run_config(run_cfg)

    with (logs_dir / "_data.pkl").open("rb") as f:
        target_pool, bg_pool, _, _, _ = pickle.load(f)

    burst_docs = np.concatenate(list(target_pool.values()))
    other_docs = np.concatenate(list(bg_pool.values()))

    n_burst = min(N_EVAL_DOCS, burst_docs.shape[0])
    n_other = min(N_EVAL_DOCS, other_docs.shape[0])
    rng = np.random.RandomState(42)
    burst_eval = burst_docs[
        rng.choice(burst_docs.shape[0], n_burst, replace=False)
    ]
    other_eval = other_docs[
        rng.choice(other_docs.shape[0], n_other, replace=False)
    ]

    with (logs_dir / "all_results.pkl").open("rb") as f:
        all_results = pickle.load(f)

    ckpt_root = logs_dir / "checkpoints"
    assert ckpt_root.exists(), f"No checkpoints at {ckpt_root}"

    epsilons = np.linspace(
        -args.max_epsilon, args.max_epsilon, args.n_points,
    )

    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)

    work_items: list[tuple[str, int, str, dict]] = []
    for sched in schedules:
        for r in jobs_by_schedule[sched]:
            ckpt_dir = ckpt_root / r["label"]
            if not ckpt_dir.exists():
                continue
            files = _ckpt_files(ckpt_dir)
            if not files:
                continue
            T = r["config"]["total_steps"]
            available = sorted(files.keys())
            peak_step = min(
                available, key=lambda x, t=T: abs(x - (t - 1))
            )
            work_items.append(
                (sched, r["seed"], str(files[peak_step]), r["config"])
            )

    logger.info("Run: %s", run_dir.name)
    logger.info("Schedules: %d, work items: %d", len(schedules), len(work_items))
    logger.info(
        "Directions: %d, points: %d, epsilon: +/-%.4f",
        args.n_directions, args.n_points, args.max_epsilon,
    )
    logger.info("Workers: %d", args.max_workers)
    logger.info("Eval docs: %d burst, %d other", n_burst, n_other)

    per_item_fwd = args.n_directions * args.n_points * 2
    total_fwd = len(work_items) * per_item_fwd
    est_seconds = total_fwd * 0.003
    logger.info(
        "Estimated: %d forward passes, ~%.0fs (%.1f min)",
        total_fwd, est_seconds, est_seconds / 60,
    )

    t0 = time.time()

    raw_results: dict[str, dict] = {
        s: {CLASS_BURST: [], CLASS_OTHER: [], "seeds": []} for s in schedules
    }

    if args.max_workers <= 1:
        for i, (sched, seed, ckpt_path, cfg) in enumerate(work_items):
            burst_losses, other_losses = compute_one_seed(
                cfg, ckpt_path, burst_eval, other_eval,
                args.n_directions, epsilons,
            )
            raw_results[sched][CLASS_BURST].extend(burst_losses)
            raw_results[sched][CLASS_OTHER].extend(other_losses)
            raw_results[sched]["seeds"].append(seed)
            elapsed = time.time() - t0
            logger.debug(
                "[%d/%d] %s s%d done (%.1fs)", i + 1, len(work_items), sched, seed, elapsed,
            )
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.max_workers, mp_context=ctx,
        ) as pool:
            futures = {}
            for sched, seed, ckpt_path, cfg in work_items:
                fut = pool.submit(
                    compute_one_seed, cfg, ckpt_path,
                    burst_eval, other_eval,
                    args.n_directions, epsilons,
                )
                futures[fut] = (sched, seed)

            for done_count, fut in enumerate(
                as_completed(futures), 1
            ):
                sched, seed = futures[fut]
                burst_losses, other_losses = fut.result()
                raw_results[sched][CLASS_BURST].extend(burst_losses)
                raw_results[sched][CLASS_OTHER].extend(other_losses)
                raw_results[sched]["seeds"].append(seed)
                elapsed = time.time() - t0
                logger.debug(
                    "[%d/%d] %s s%d done (%.1fs)",
                    done_count, len(work_items), sched, seed, elapsed,
                )

    compute_time = time.time() - t0
    logger.info("Compute done in %.1fs (%.1f min)", compute_time, compute_time / 60)

    final: dict[str, dict] = {}
    for sched in schedules:
        rd = raw_results[sched]
        final[sched] = {
            "epsilons": epsilons.tolist(),
            "burst_losses": rd[CLASS_BURST],
            "other_losses": rd[CLASS_OTHER],
            "n_seeds": len(rd["seeds"]),
            "n_directions": args.n_directions,
        }

    out_dir = results_dir / "basin_50dir"
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "results.pkl").open("wb") as f:
        pickle.dump(final, f)

    logger.info("Plotting...")
    plot_mean_and_variance(final, out_dir, args.n_directions)
    plot_sharpness(final, out_dir, args.n_directions)

    total_time = time.time() - t0
    logger.info("All done in %.1fs (%.1f min)", total_time, total_time / 60)
    logger.info("Results: %s", out_dir)


if __name__ == "__main__":
    main()
