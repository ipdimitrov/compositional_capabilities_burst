"""Plot probe heatmaps for Other-vs-Burst representation analysis.

Reads probe results from probe.py and generates:
  1. Per-model layer x token heatmaps at key checkpoints
  2. Training dynamics curves (probe accuracy over training steps)
  3. Cross-model comparison heatmaps (e.g. end_block vs end_mixed_25b)
  4. Mean-pooled probe accuracy over time per schedule

Usage:
    python burst/plot_probes.py data/burst_d<depth>_<run_tag>
"""

from __future__ import annotations

import json
import pickle
import sys
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt

from burst.config import SCHED_COLORS, SCHEDULE_ORDER, ordered_schedules, sched_sort_key

_TEXT_CONTRAST_THRESHOLD = 0.75
_STEP_TOLERANCE = 30


def _load_steps_from_config(run_dir: Path) -> tuple[int, int] | None:
    from burst.core.train_utils import resolve_run_paths  # noqa: PLC0415

    cfg_path, _, _ = resolve_run_paths(run_dir)
    if not cfg_path.exists():
        return None
    with cfg_path.open() as f:
        cfg = json.load(f)
    bcfg = cfg.get("base_cfg", {})
    ts = bcfg.get("total_steps")
    us = bcfg.get("reversion_steps")
    if ts is not None and us is not None:
        return int(ts), int(us)
    return None


def load_probe_results(run_dir: Path) -> tuple[list[dict], dict]:
    """Load probe results from run directory."""
    probe_dir = run_dir / "probes"
    if not probe_dir.exists():
        sys.exit(1)

    all_path = probe_dir / "all_probes.pkl"
    if not all_path.exists():
        individual = sorted(probe_dir.glob("*_probe.pkl"))
        if individual:
            results = []
            for p in individual:
                with p.open("rb") as f:
                    results.append(pickle.load(f))  # noqa: S301
            with all_path.open("wb") as f:
                pickle.dump(results, f)
        else:
            sys.exit(1)
    else:
        with all_path.open("rb") as f:
            results = pickle.load(f)  # noqa: S301

    steps_from_cfg = _load_steps_from_config(run_dir)

    meta_path = probe_dir / "probe_meta.json"
    if not meta_path.exists():
        r0 = results[0]
        fallback_ts = steps_from_cfg[0] if steps_from_cfg else 400
        fallback_us = steps_from_cfg[1] if steps_from_cfg else 400
        first_acc = next(iter(r0["probes"].values()))["train_acc_KT"]
        meta = {
            "checkpoint_steps": r0.get("checkpoint_steps", sorted(r0["probes"].keys())),
            "token_labels": r0.get(
                "token_labels",
                [f"t{i}" for i in range(first_acc.shape[1])],
            ),
            "n_layers": r0.get("n_layers", first_acc.shape[0] - 1),
            "total_steps": r0.get("total_steps", fallback_ts),
            "reversion_steps": r0.get("reversion_steps", fallback_us),
        }
    else:
        with meta_path.open() as f:
            meta = json.load(f)

    if steps_from_cfg:
        meta["total_steps"] = steps_from_cfg[0]
        meta["reversion_steps"] = steps_from_cfg[1]

    return results, meta


def _layer_labels(n_layers: int) -> list[str]:
    return ["emb"] + [f"L{i}" for i in range(n_layers)]


def plot_heatmap(  # noqa: PLR0913
    acc_KT: np.ndarray,
    token_labels: list[str],
    layer_labels: list[str],
    title: str,
    save_path: Path,
    vmin: float = 0.4,
    vmax: float = 1.0,
) -> None:
    """Render layer x token probe accuracy heatmap."""
    K, T = acc_KT.shape
    fig, ax = plt.subplots(figsize=(max(14, T * 0.5), max(4, K * 0.6)))
    im = ax.imshow(
        acc_KT, aspect="auto", cmap="Blues", vmin=vmin, vmax=vmax, interpolation="nearest"
    )

    ax.set_xticks(range(T))
    ax.set_xticklabels(token_labels[:T], rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(K))
    ax.set_yticklabels(layer_labels[:K], fontsize=8)
    ax.set_xlabel("Token Position", fontsize=10)
    ax.set_ylabel("Layer", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")

    for k in range(K):
        for t in range(T):
            val = acc_KT[k, t]
            color = "white" if val > _TEXT_CONTRAST_THRESHOLD else "black"
            ax.text(t, k, f"{val:.2f}", ha="center", va="center", fontsize=5, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Probe Accuracy (Other vs Burst)", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_diff_heatmap(  # noqa: PLR0913
    acc_other_KT: np.ndarray,
    acc_burst_KT: np.ndarray,
    token_labels: list[str],
    layer_labels: list[str],
    title: str,
    label_other: str,
    label_burst: str,
    save_path: Path,
) -> None:
    """Render difference heatmap between two schedule probe accuracies."""
    diff_KT = acc_other_KT - acc_burst_KT
    K, T = diff_KT.shape
    vmax = max(abs(diff_KT.min()), abs(diff_KT.max()), 0.1)

    fig, ax = plt.subplots(figsize=(max(14, T * 0.5), max(4, K * 0.6)))
    im = ax.imshow(
        diff_KT, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest"
    )

    ax.set_xticks(range(T))
    ax.set_xticklabels(token_labels[:T], rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(K))
    ax.set_yticklabels(layer_labels[:K], fontsize=8)
    ax.set_xlabel("Token Position", fontsize=10)
    ax.set_ylabel("Layer", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")

    for k in range(K):
        for t in range(T):
            val = diff_KT[k, t]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(t, k, f"{val:+.2f}", ha="center", va="center", fontsize=5, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(f"Delta accuracy ({label_other} - {label_burst})", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_dynamics(  # noqa: C901
    result: dict,
    token_labels: list[str],
    layer_labels: list[str],
    save_path: Path,
) -> None:
    """Plot probe accuracy over training steps for selected (layer, token) combos."""
    probes = result["probes"]
    total_steps = result["total_steps"]
    n_layers = result["n_layers"]

    interesting_set = set()
    for lbl in token_labels:
        if lbl.startswith("F") or lbl == "sp1" or lbl.endswith("_0"):
            interesting_set.add(lbl)
    last_sp = [label for label in token_labels if label.startswith("sp")]
    if last_sp:
        interesting_set.add(last_sp[-1])
    interesting_tokens = [(i, lbl) for i, lbl in enumerate(token_labels) if lbl in interesting_set]
    if not interesting_tokens:
        interesting_tokens = [(i, token_labels[i]) for i in range(min(5, len(token_labels)))]

    interesting_layers = [0, n_layers // 2, n_layers]

    steps_sorted = sorted(probes.keys())
    fig, axes = plt.subplots(
        len(interesting_layers), 1, figsize=(14, 4 * len(interesting_layers)), sharex=True
    )
    if len(interesting_layers) == 1:
        axes = [axes]

    fig.suptitle(
        f"Training Dynamics -- {result['label']}\nProbe accuracy (Other vs Burst) over training",
        fontsize=13,
        fontweight="bold",
    )

    for ax_idx, layer_k in enumerate(interesting_layers):
        ax = axes[ax_idx]
        for tok_idx, tok_lbl in interesting_tokens:
            accs = []
            valid_steps = []
            for step in steps_sorted:
                acc_KT = probes[step]["train_acc_KT"]
                if layer_k < acc_KT.shape[0] and tok_idx < acc_KT.shape[1]:
                    accs.append(acc_KT[layer_k, tok_idx])
                    valid_steps.append(step)
            if accs:
                ax.plot(valid_steps, accs, marker=".", markersize=3, lw=1.5, label=tok_lbl)

        ax.axvline(total_steps, color="gray", ls="--", alpha=0.5, lw=2)
        ax.axhline(0.5, color="gray", ls=":", alpha=0.3)
        ax.set_ylabel(f"{layer_labels[layer_k]}\nAccuracy", fontsize=9)
        ax.set_ylim(0.35, 1.05)
        ax.legend(fontsize=7, ncol=4, loc="upper left")
        ax.grid(visible=True, alpha=0.2)

    axes[-1].set_xlabel("Global Step", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mean_dynamics_by_schedule(
    results: list[dict],
    save_path: Path,
) -> None:
    """Mean-pooled probe accuracy over time, one line per schedule."""
    sched_data = {}
    for r in results:
        sched = r["schedule"]
        if sched not in sched_data:
            sched_data[sched] = []
        sched_data[sched].append(r)

    total_steps = results[0]["total_steps"]

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    fig.suptitle(
        "Mean Probe Accuracy Over Training (averaged across layers & tokens)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_title("5-fold CV (train compositions)", fontsize=11)

    for sched in ordered_schedules(sched_data.keys()):
        runs = sched_data[sched]
        all_steps = set()
        for r in runs:
            all_steps.update(r["probes"].keys())
        steps_sorted = sorted(all_steps)

        per_seed_curves = []
        for r in runs:
            curve = []
            for step in steps_sorted:
                if step in r["probes"]:
                    curve.append(r["probes"][step]["train_acc_KT"].mean())
                else:
                    curve.append(np.nan)
            per_seed_curves.append(curve)

        arr = np.array(per_seed_curves)
        mean_vals = np.nanmean(arr, axis=0)
        std_vals = np.nanstd(arr, axis=0)
        n = np.sum(~np.isnan(arr), axis=0)
        ci = np.where(n > 1, 1.96 * std_vals / np.sqrt(n), std_vals)

        c = SCHED_COLORS.get(sched, "gray")
        ax.plot(steps_sorted, mean_vals, color=c, lw=2.5, label=sched)
        ax.fill_between(steps_sorted, mean_vals - ci, mean_vals + ci, color=c, alpha=0.2)

    ax.axvline(total_steps, color="gray", ls="--", alpha=0.5, lw=2)
    ax.axhline(0.5, color="gray", ls=":", alpha=0.3)
    ax.set_ylabel("Mean Probe Accuracy", fontsize=10)
    ax.set_ylim(0.35, 1.05)
    ax.legend(fontsize=9, loc="best")
    ax.grid(visible=True, alpha=0.2)
    ax.set_xlabel("Global Step", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_layer_depth_dynamics(
    results: list[dict],
    save_path: Path,
) -> None:
    """Per-layer mean probe accuracy over time, one subplot per schedule."""
    sched_data = {}
    for r in results:
        sched = r["schedule"]
        if sched not in sched_data:
            sched_data[sched] = []
        sched_data[sched].append(r)

    n_layers = results[0]["n_layers"]
    total_steps = results[0]["total_steps"]
    layer_labels = _layer_labels(n_layers)
    K = n_layers + 1

    n_scheds = len(sched_data)
    fig, axes = plt.subplots(n_scheds, 1, figsize=(14, 4 * n_scheds), sharex=True)
    if n_scheds == 1:
        axes = [axes]

    fig.suptitle(
        "Per-Layer Probe Accuracy Over Training\n(mean across token positions & seeds)",
        fontsize=14,
        fontweight="bold",
    )

    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, K))

    for ax_idx, sched in enumerate(ordered_schedules(sched_data.keys())):
        ax = axes[ax_idx]
        runs = sched_data[sched]

        all_steps = set()
        for r in runs:
            all_steps.update(r["probes"].keys())
        steps_sorted = sorted(all_steps)

        for k in range(K):
            per_seed = []
            for r in runs:
                curve = []
                for step in steps_sorted:
                    if step in r["probes"]:
                        curve.append(r["probes"][step]["train_acc_KT"][k, :].mean())
                    else:
                        curve.append(np.nan)
                per_seed.append(curve)

            arr = np.array(per_seed)
            mean_vals = np.nanmean(arr, axis=0)
            ax.plot(steps_sorted, mean_vals, color=cmap[k], lw=2, label=layer_labels[k])

        ax.axvline(total_steps, color="gray", ls="--", alpha=0.5, lw=2)
        ax.axhline(0.5, color="gray", ls=":", alpha=0.3)
        ax.set_ylabel("Probe Acc", fontsize=9)
        ax.set_title(sched, fontsize=10, fontweight="bold")
        ax.set_ylim(0.35, 1.05)
        ax.legend(fontsize=7, ncol=K, loc="upper left")
        ax.grid(visible=True, alpha=0.2)

    axes[-1].set_xlabel("Global Step", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_layer_schedule_heatmap(
    results: list[dict],
    step: int,
    save_path: Path,
    acc_key: str = "train_acc_KT",
) -> None:
    """Layer x Schedule heatmap of mean probe accuracy at a given step.

    Rows: L0 ... L{n-1}  (transformer blocks, excludes embedding)
    Columns: schedules in SCHEDULE_ORDER
    Values: mean across token positions and seeds
    """
    n_layers = results[0]["n_layers"]
    layer_labels = [f"L{i}" for i in range(n_layers)]

    sched_set = {r["schedule"] for r in results}
    col_scheds = [s for s in SCHEDULE_ORDER if s in sched_set]
    if not col_scheds:
        col_scheds = sorted(sched_set)

    grid = np.full((n_layers, len(col_scheds)), np.nan)

    for ci, sched in enumerate(col_scheds):
        seed_means = []
        for r in results:
            if r["schedule"] != sched:
                continue
            closest = min(r["probes"].keys(), key=lambda s, _st=step: abs(s - _st))
            if abs(closest - step) > _STEP_TOLERANCE:
                continue
            acc_KT = r["probes"][closest][acc_key]
            per_layer = acc_KT[1:, :].mean(axis=1)
            seed_means.append(per_layer)
        if seed_means:
            grid[:, ci] = np.mean(seed_means, axis=0)

    fig, ax = plt.subplots(
        figsize=(max(6, len(col_scheds) * 1.4), max(3, n_layers * 0.6))
    )
    im = ax.imshow(grid, aspect="auto", cmap="Blues", vmin=0.4, vmax=1.0, interpolation="nearest")

    ax.set_xticks(range(len(col_scheds)))
    ax.set_xticklabels(col_scheds, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(layer_labels, fontsize=10)
    ax.set_xlabel("Schedule", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_title(
        f"Layer x Schedule -- mean probe accuracy at step {step}",
        fontsize=13,
        fontweight="bold",
    )

    for row in range(n_layers):
        for col in range(len(col_scheds)):
            val = grid[row, col]
            if np.isnan(val):
                continue
            color = "white" if val > _TEXT_CONTRAST_THRESHOLD else "black"
            ax.text(col, row, f"{val:.3f}", ha="center", va="center", fontsize=10, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
    cbar.set_label("Mean Probe Accuracy", fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _pick_representative_seed(results: list[dict], schedule: str) -> dict | None:
    """Pick the first seed for a given schedule."""
    for r in results:
        if r["schedule"] == schedule:
            return r
    return None


def _mean_acc_at_step(
    results: list[dict], schedule: str, step: int, key: str,
) -> np.ndarray | None:
    """Average probe accuracy across seeds for a schedule at a given step."""
    arrs = [
        r["probes"][step][key]
        for r in results
        if r["schedule"] == schedule and step in r["probes"]
    ]
    if not arrs:
        return None
    return np.mean(arrs, axis=0)


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Generate all probe visualisation plots for a run directory."""
    if len(sys.argv) < 2:  # noqa: PLR2004
        data_dir = Path("data")
        burst_dirs = sorted([d for d in data_dir.glob("burst_d*") if d.is_dir()])
        if not burst_dirs:
            sys.exit(1)
        run_dir = burst_dirs[-1]
    else:
        run_dir = Path(sys.argv[1])

    results, meta = load_probe_results(run_dir)
    token_labels = meta["token_labels"]
    n_layers = meta["n_layers"]
    total_steps = meta["total_steps"]
    reversion_steps = meta["reversion_steps"]
    layer_labels = _layer_labels(n_layers)

    plots_dir = run_dir / "probes" / "plots"
    plots_dir.mkdir(exist_ok=True)

    schedules = ordered_schedules({r["schedule"] for r in results})
    key_steps = [
        0,
        total_steps // 2,
        total_steps,
        total_steps + reversion_steps // 2,
        total_steps + reversion_steps,
    ]

    for r in results:
        label = r["label"]
        idx = sched_sort_key(r["schedule"])
        probes = r["probes"]
        for step in key_steps:
            closest = min(probes.keys(), key=lambda s, _st=step: abs(s - _st))
            if abs(closest - step) > _STEP_TOLERANCE:
                continue
            phase = "train" if closest <= total_steps else "reversion"
            acc_KT = probes[closest]["train_acc_KT"]
            plot_heatmap(
                acc_KT,
                token_labels,
                layer_labels,
                f"{label} -- step {closest} ({phase})\n5-fold CV probe accuracy (Other vs Burst)",
                plots_dir / f"{idx:02d}_heatmap_{label}_step{closest}.png",
            )

    for r in results:
        idx = sched_sort_key(r["schedule"])
        plot_training_dynamics(
            r, token_labels, layer_labels, plots_dir / f"{idx:02d}_dynamics_{r['label']}.png"
        )

    plot_mean_dynamics_by_schedule(results, plots_dir / "mean_dynamics_by_schedule.png")

    plot_layer_depth_dynamics(results, plots_dir / "layer_depth_dynamics.png")

    comparison_dir = plots_dir / "comparisons"
    comparison_dir.mkdir(exist_ok=True)

    for step in [total_steps, total_steps + reversion_steps]:
        phase_label = "end_train" if step == total_steps else "end_reversion"

        for s1, s2 in combinations(schedules, 2):
            acc_s1 = _mean_acc_at_step(results, s1, step, "train_acc_KT")
            acc_s2 = _mean_acc_at_step(results, s2, step, "train_acc_KT")
            if acc_s1 is None or acc_s2 is None:
                closest_s1 = min(
                    [s for r in results if r["schedule"] == s1 for s in r["probes"]],
                    key=lambda s, _st=step: abs(s - _st),
                    default=None,
                )
                closest_s2 = min(
                    [s for r in results if r["schedule"] == s2 for s in r["probes"]],
                    key=lambda s, _st=step: abs(s - _st),
                    default=None,
                )
                if closest_s1 is not None and abs(closest_s1 - step) <= _STEP_TOLERANCE:
                    acc_s1 = _mean_acc_at_step(results, s1, closest_s1, "train_acc_KT")
                if closest_s2 is not None and abs(closest_s2 - step) <= _STEP_TOLERANCE:
                    acc_s2 = _mean_acc_at_step(results, s2, closest_s2, "train_acc_KT")
            if acc_s1 is None or acc_s2 is None:
                continue

            plot_diff_heatmap(
                acc_s1,
                acc_s2,
                token_labels,
                layer_labels,
                f"{s1} vs {s2} -- step {step} ({phase_label})\n"
                f"Delta probe accuracy (positive = {s1} higher)",
                s1,
                s2,
                comparison_dir / f"diff_{s1}_vs_{s2}_{phase_label}.png",
            )

        for sched in schedules:
            si = sched_sort_key(sched)
            acc_train_end = _mean_acc_at_step(results, sched, total_steps, "train_acc_KT")
            acc_reversion_end = _mean_acc_at_step(
                results, sched, total_steps + reversion_steps, "train_acc_KT"
            )
            if acc_train_end is not None and acc_reversion_end is not None:
                plot_diff_heatmap(
                    acc_train_end,
                    acc_reversion_end,
                    token_labels,
                    layer_labels,
                    f"{sched} -- end_train vs end_reversion\n"
                    "Delta probe accuracy "
                    "(positive = more Other/Burst distinction at end of training)",
                    "end_train",
                    "end_reversion",
                    comparison_dir / f"{si:02d}_diff_{sched}_train_vs_reversion.png",
                )

    sched_dir = plots_dir / "by_schedule"
    sched_dir.mkdir(exist_ok=True)

    for sched in schedules:
        si = sched_sort_key(sched)
        for step in key_steps:
            acc = _mean_acc_at_step(results, sched, step, "train_acc_KT")
            matched_step = step
            if acc is None:
                closest = min(
                    [s for r in results if r["schedule"] == sched for s in r["probes"]],
                    key=lambda s, _st=step: abs(s - _st),
                    default=None,
                )
                if closest is not None and abs(closest - step) <= _STEP_TOLERANCE:
                    acc = _mean_acc_at_step(results, sched, closest, "train_acc_KT")
                    matched_step = closest
            if acc is None:
                continue
            phase = "train" if matched_step <= total_steps else "reversion"
            plot_heatmap(
                acc,
                token_labels,
                layer_labels,
                f"{sched} (mean across seeds) -- step {matched_step} ({phase})\n"
                f"5-fold CV probe accuracy (Other vs Burst)",
                sched_dir / f"{si:02d}_heatmap_{sched}_step{matched_step}.png",
            )

    final_step = total_steps + reversion_steps
    plot_layer_schedule_heatmap(
        results, final_step, plots_dir / f"layer_schedule_heatmap_step{final_step}.png"
    )


if __name__ == "__main__":
    main()
