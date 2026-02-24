"""Chart generation for the hypothesis-driven presentation."""
import sys, os, math, pickle, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from burst._worker import n_target_for_step
from burst.train_utils import compute_lr_schedule as _compute_lr
from burst.config import (
    SCHED_COLORS as PALETTE, SCHED_DISPLAY as SCHED_SHORT,
    SCHEDULE_ORDER, ordered_schedules as _ordered,
)


def _group(results):
    g = defaultdict(list)
    for r in results:
        g[r["schedule"]].append(r)
    return g


def _style(ax, xl="", yl="", t=""):
    ax.set_xlabel(xl, fontsize=13, fontweight="bold")
    ax.set_ylabel(yl, fontsize=13, fontweight="bold")
    if t:
        ax.set_title(t, fontsize=15, fontweight="bold", pad=12)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.15, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def schedule_bars(pdir, results, cfg):
    bcfg = cfg.get("base_cfg", cfg)
    T, U, bs, p = bcfg["total_steps"], bcfg["reversion_steps"], bcfg["batch_size"], bcfg["p_target"]
    total = T + U
    scheds = _ordered(set(r["schedule"] for r in results))
    n = len(scheds)
    fig, axes = plt.subplots(n, 1, figsize=(14, 1.8 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for i, sched in enumerate(scheds):
        ax = axes[i]
        fracs = np.zeros(total)
        for s in range(T):
            np.random.seed(107 * 10000 + s)
            fracs[s] = n_target_for_step(s, T, sched, p, bs) / bs
        ax.fill_between(range(total), fracs, color=PALETTE[sched], alpha=0.7)
        ax.axvline(T, color="black", lw=2, ls="--")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, total)
        ax.set_ylabel(SCHED_SHORT[sched], fontsize=9, fontweight="bold",
                       rotation=0, labelpad=120, ha="left", va="center")
        ax.set_yticks([0, 0.5, 1.0])
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel("Global Step", fontsize=12, fontweight="bold")
    axes[0].set_title("Training Schedules: Fraction of Burst Data per Step",
                      fontsize=14, fontweight="bold", pad=10)
    axes[0].annotate("FOUNDATION + BURST", xy=(T * 0.5, 1.15), fontsize=11, color="gray",
                     fontweight="bold", ha="center", annotation_clip=False)
    axes[0].annotate("REVERSION (Other Classes only)", xy=(T + U * 0.5, 1.15), fontsize=11,
                     color="gray", fontweight="bold", ha="center", annotation_clip=False)
    fig.tight_layout(rect=[0.15, 0, 1, 0.97])
    p_ = pdir / "schedule_bars.png"
    fig.savefig(p_, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p_


def overlay(pdir, results, cfg, key, yl, title, fname, loc="center left", groups=None):
    bcfg = cfg.get("base_cfg", cfg)
    T, U = bcfg["total_steps"], bcfg["reversion_steps"]
    if groups is None:
        groups = _group(results)
    fig, ax = plt.subplots(figsize=(14, 7))
    for sched in _ordered(groups.keys()):
        runs = groups[sched]
        steps = np.array(runs[0]["log"]["step"])
        vals = np.array([np.array(r["log"][key]) for r in runs])
        m = np.mean(vals, axis=0)
        n_s = len(runs)
        ci = 1.96 * np.std(vals, axis=0) / np.sqrt(n_s) if n_s > 1 else np.std(vals, axis=0)
        ax.plot(steps, m, color=PALETTE[sched], lw=2.5, label=SCHED_SHORT[sched])
        ax.fill_between(steps, m - ci, m + ci, color=PALETTE[sched], alpha=0.15)
    ax.axvline(T, color="black", ls="--", lw=2, alpha=0.6)
    ax.text(T * 0.5, -0.12, "FOUNDATION+BURST", ha="center", fontsize=12, color="gray",
            fontweight="bold", transform=ax.get_xaxis_transform())
    ax.text(T + U * 0.5, -0.12, "REVERSION", ha="center", fontsize=12, color="gray",
            fontweight="bold", transform=ax.get_xaxis_transform())
    ax.set_xlim(0, T + U)
    ax.set_ylim(-0.05, 1.05)
    _style(ax, "Step", yl, title)
    ax.legend(fontsize=11, loc=loc, framealpha=0.9, edgecolor="gray")
    fig.tight_layout()
    p_ = pdir / fname
    fig.savefig(p_, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p_


def bar_chart(pdir, results, cfg, metric, yl, title, fname, fmt_dec=0, groups=None):
    bcfg = cfg.get("base_cfg", cfg)
    U = bcfg["reversion_steps"]
    if groups is None:
        groups = _group(results)
    scheds = _ordered(groups.keys())
    n = len(scheds)
    fig, ax = plt.subplots(figsize=(12, 7))
    xs = np.arange(n)
    means, cis, all_v = [], [], []
    for sched in scheds:
        vals = np.array([r[metric] for r in groups[sched]])
        m = vals.mean()
        ci = 1.96 * vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else vals.std()
        means.append(m)
        cis.append(ci)
        all_v.append(vals)
    colors = [PALETTE[s] for s in scheds]
    ax.bar(xs, means, yerr=cis, color=colors, edgecolor="black", lw=0.8,
           capsize=6, error_kw={"lw": 2, "capthick": 2}, width=0.6, alpha=0.85)
    for i, vals in enumerate(all_v):
        jit = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), i) + jit, vals,
                   color="black", s=40, zorder=5, alpha=0.6, edgecolor="white", lw=0.5)
    for i, (m, ci) in enumerate(zip(means, cis)):
        lbl = f"{m:.{fmt_dec}f}" if m < U or fmt_dec > 0 else f">{U}"
        ax.text(i, m + ci + max(means) * 0.02, lbl, ha="center", fontsize=12, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([SCHED_SHORT[s] for s in scheds], fontsize=10, fontweight="bold")
    _style(ax, "", yl, title)
    if metric == "quarter_life":
        ax.axhline(U, color="gray", ls=":", alpha=0.5, lw=1.5)
    fig.tight_layout()
    p_ = pdir / fname
    fig.savefig(p_, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p_


def auc_diff(pdir, results, cfg, groups=None):
    if groups is None:
        groups = _group(results)
    scheds = _ordered(groups.keys())
    n = len(scheds)
    mean_aucs = {s: np.mean([r["reversion_auc"] for r in groups[s]])
                 for s in scheds}
    grid = np.zeros((n, n))
    for i, sa in enumerate(scheds):
        for j, sb in enumerate(scheds):
            if i != j:
                base = mean_aucs[sb]
                grid[i, j] = ((mean_aucs[sa] - base) / abs(base) * 100) if abs(base) > 1e-9 else 0.0
    fig, ax = plt.subplots(figsize=(9, 8))
    vmax = max(abs(grid.min()), abs(grid.max()), 1)
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    labels = [SCHED_SHORT[s] for s in scheds]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Baseline (denominator)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Compared (numerator)", fontsize=12, fontweight="bold")
    ax.set_title("Pairwise Reversion AUC Difference (%)\n(row - col) / |col| x 100",
                 fontsize=14, fontweight="bold")
    for i in range(n):
        for j in range(n):
            c = "white" if abs(grid[i, j]) > vmax * 0.55 else "black"
            ax.text(j, i, f"{grid[i, j]:+.1f}%", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=c)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03, label="% difference")
    fig.tight_layout()
    p_ = pdir / "auc_diff_heatmap.png"
    fig.savefig(p_, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p_


def lr_schedule(pdir, cfg):
    bcfg = cfg.get("base_cfg", cfg)
    T, U = bcfg["total_steps"], bcfg["reversion_steps"]
    total = T + U
    steps, lrs = _compute_lr(bcfg)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(steps, lrs, color="#1565C0", lw=2.5)
    ax.axvline(T, color="black", lw=2, ls="--")
    ax.set_xlim(0, total)
    _style(ax, "Global Step", "Learning Rate",
           "Learning Rate Schedule (cosine decay with linear warmup)")
    ax.text(T * 0.5, ax.get_ylim()[1] * 0.92, "TRAIN", ha="center", fontsize=11,
            color="gray", fontweight="bold")
    ax.text(T + U * 0.5, ax.get_ylim()[1] * 0.92, "REVERSION", ha="center", fontsize=11,
            color="gray", fontweight="bold")
    fig.tight_layout()
    p_ = pdir / "lr_schedule.png"
    fig.savefig(p_, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p_


def reversion_zoom(pdir, results, cfg, groups=None):
    bcfg = cfg.get("base_cfg", cfg)
    T, U = bcfg["total_steps"], bcfg["reversion_steps"]
    if groups is None:
        groups = _group(results)
    fig, ax = plt.subplots(figsize=(14, 7))
    burst_log_key = "acc_burst"
    for sched in _ordered(groups.keys()):
        runs = groups[sched]
        steps = np.array(runs[0]["log"]["step"])
        vals = np.array([np.array(r["log"][burst_log_key]) for r in runs])
        mask = steps >= T
        reversion_steps_arr = steps[mask] - T
        uv = vals[:, mask]
        m = np.mean(uv, axis=0)
        n_s = len(runs)
        ci = 1.96 * np.std(uv, axis=0) / np.sqrt(n_s) if n_s > 1 else np.std(uv, axis=0)
        ax.plot(reversion_steps_arr, m, color=PALETTE[sched], lw=2.5, label=SCHED_SHORT[sched])
        ax.fill_between(reversion_steps_arr, m - ci, m + ci, color=PALETTE[sched], alpha=0.15)
    ax.axhline(0.25, color="gray", ls=":", alpha=0.5, lw=1.5)
    ax.text(U * 0.95, 0.27, "25% threshold", fontsize=9, color="gray", ha="right")
    ax.set_xlim(0, U)
    ax.set_ylim(-0.05, 1.05)
    _style(ax, "Reversion Steps (after Burst Class removal)", "Burst Class Accuracy",
           "Forgetting Dynamics: Burst Class Accuracy During Reversion\n(mean +/- 95% CI, n=5 seeds)")
    ax.legend(fontsize=11, loc="upper right", framealpha=0.9, edgecolor="gray")
    fig.tight_layout()
    p_ = pdir / "reversion_zoom.png"
    fig.savefig(p_, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p_


def summary_table(pdir, results, cfg, groups=None):
    bcfg = cfg.get("base_cfg", cfg)
    U = bcfg["reversion_steps"]
    if groups is None:
        groups = _group(results)
    scheds = _ordered(groups.keys())
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")
    cols = ["Schedule", "Peak Burst\n(mean +/- CI)", "Quarter-life\n(mean +/- CI)",
            "Reversion AUC\n(mean +/- CI)", "Other Classes Acc\n(mean +/- CI)"]
    rows = []
    for sched in scheds:
        runs = groups[sched]
        def fmt(vals, d=3):
            m = vals.mean()
            ci = 1.96 * vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else vals.std()
            return f"{m:.{d}f} +/- {ci:.{d}f}" if d > 0 else f"{m:.0f} +/- {ci:.0f}"
        rows.append([
            SCHED_SHORT[sched],
            fmt(np.array([r["peak_burst"] for r in runs]), 3),
            fmt(np.array([r.get("quarter_life", U) for r in runs]), 0),
            fmt(np.array([r["reversion_auc"] for r in runs]), 0),
            fmt(np.array([r["log"]["acc_other"][-1] for r in runs]), 3),
        ])
    table = ax.table(cellText=rows, colLabels=cols, loc="center",
                     cellLoc="center", colColours=["#E0E0E0"] * len(cols))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold", fontsize=10)
            cell.set_facecolor("#E0E0E0")
        elif col == 0:
            cell.set_facecolor(PALETTE[scheds[row - 1]] + "25")
            cell.set_text_props(fontweight="bold", fontsize=9)
        cell.set_edgecolor("#CCCCCC")
    ax.set_title(f"Summary Statistics (n={len(groups[scheds[0]])} seeds per schedule)",
                 fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    p_ = pdir / "summary_table.png"
    fig.savefig(p_, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p_


def per_sched(pdir, results, cfg, groups=None):
    bcfg = cfg.get("base_cfg", cfg)
    T, U = bcfg["total_steps"], bcfg["reversion_steps"]
    if groups is None:
        groups = _group(results)
    paths = []
    for sched in _ordered(groups.keys()):
        runs = groups[sched]
        steps = np.array(runs[0]["log"]["step"])
        fig, ax = plt.subplots(figsize=(14, 6))
        for k, (c, lbl) in [("acc_other", ("#1565C0", "Other Classes")),
                              ("acc_burst", ("#D32F2F", "Burst Class"))]:
            log_key = k
            vals = np.array([np.array(r["log"][log_key]) for r in runs])
            m = np.mean(vals, axis=0)
            n_s = len(runs)
            ci = 1.96 * np.std(vals, axis=0) / np.sqrt(n_s) if n_s > 1 else np.std(vals, axis=0)
            ax.plot(steps, m, color=c, lw=2.5, label=lbl)
            ax.fill_between(steps, m - ci, m + ci, color=c, alpha=0.15)
        ax.axvline(T, color="black", ls="--", lw=2, alpha=0.6)
        ax.set_xlim(0, T + U)
        ax.set_ylim(-0.05, 1.05)
        n_s = len(runs)
        _style(ax, "Step", "Accuracy (free generation)",
               f"{SCHED_SHORT[sched]}: Other Classes vs Burst Class (mean +/- 95% CI, n={n_s})")
        ax.legend(fontsize=12, loc="center left", framealpha=0.9)
        fig.tight_layout()
        p_ = pdir / f"per_sched_{sched}.png"
        fig.savefig(p_, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(p_)
    return paths


def grad_cosine_sim_overlay(pdir, results, cfg, groups=None):
    """Plot burst-vs-other gradient cosine similarity over training steps, per schedule."""
    bcfg = cfg.get("base_cfg", cfg)
    T, U = bcfg["total_steps"], bcfg["reversion_steps"]
    if groups is None:
        groups = _group(results)

    has_data = any("grad_sim_log" in r and r["grad_sim_log"]["step"] for r in results)
    if not has_data:
        return None

    fig, ax = plt.subplots(figsize=(14, 7))
    for sched in _ordered(groups.keys()):
        runs = [r for r in groups[sched] if "grad_sim_log" in r and r["grad_sim_log"]["step"]]
        if not runs:
            continue
        steps_list = [np.array(r["grad_sim_log"]["step"]) for r in runs]
        vals_list = [np.array(r["grad_sim_log"]["burst_vs_other"]) for r in runs]
        steps_ref = steps_list[0]
        # Interpolate all runs to the same step grid
        interp_vals = []
        for s, v in zip(steps_list, vals_list):
            if len(s) > 1:
                interp_vals.append(np.interp(steps_ref, s, v))
        if not interp_vals:
            continue
        vals_arr = np.array(interp_vals)
        m = np.mean(vals_arr, axis=0)
        n_s = len(interp_vals)
        ci = 1.96 * np.std(vals_arr, axis=0) / np.sqrt(n_s) if n_s > 1 else np.std(vals_arr, axis=0)
        ax.plot(steps_ref, m, color=PALETTE[sched], lw=2, label=SCHED_SHORT[sched])
        ax.fill_between(steps_ref, m - ci, m + ci, color=PALETTE[sched], alpha=0.12)

    ax.axvline(T, color="black", ls="--", lw=2, alpha=0.6)
    ax.axhline(0, color="gray", ls=":", lw=1, alpha=0.5)
    ax.text(T * 0.5, ax.get_ylim()[0] * 0.9 if ax.get_ylim()[0] < 0 else -0.12,
            "FOUNDATION+BURST", ha="center", fontsize=11, color="gray",
            fontweight="bold", transform=ax.get_xaxis_transform())
    ax.text(T + U * 0.5, -0.12, "REVERSION", ha="center", fontsize=11, color="gray",
            fontweight="bold", transform=ax.get_xaxis_transform())
    ax.set_xlim(0, T + U)
    _style(ax, "Step", "Cosine Similarity",
           "Gradient Cosine Similarity: Burst Class vs Other Classes\n(mean +/- 95% CI)")
    ax.legend(fontsize=10, loc="best", framealpha=0.9, edgecolor="gray")
    fig.tight_layout()
    p_ = pdir / "grad_cosine_burst_vs_other.png"
    fig.savefig(p_, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p_


def grad_cosine_sim_by_schedule(pdir, results, cfg, groups=None):
    """Bar chart: mean burst-vs-other cosine similarity at end of burst phase, per schedule."""
    bcfg = cfg.get("base_cfg", cfg)
    T = bcfg["total_steps"]
    if groups is None:
        groups = _group(results)

    has_data = any("grad_sim_log" in r and r["grad_sim_log"]["step"] for r in results)
    if not has_data:
        return None

    scheds = _ordered(groups.keys())
    means, cis = [], []
    for sched in scheds:
        runs = [r for r in groups[sched] if "grad_sim_log" in r and r["grad_sim_log"]["step"]]
        end_vals = []
        for r in runs:
            steps = np.array(r["grad_sim_log"]["step"])
            sims = np.array(r["grad_sim_log"]["burst_vs_other"])
            burst_mask = steps <= T
            if burst_mask.any():
                end_vals.append(sims[burst_mask][-1])
        if end_vals:
            arr = np.array(end_vals)
            means.append(arr.mean())
            cis.append(1.96 * arr.std() / np.sqrt(len(arr)) if len(arr) > 1 else arr.std())
        else:
            means.append(0.0)
            cis.append(0.0)

    fig, ax = plt.subplots(figsize=(12, 6))
    xs = np.arange(len(scheds))
    colors = [PALETTE[s] for s in scheds]
    ax.bar(xs, means, yerr=cis, color=colors, edgecolor="black", lw=0.8,
           capsize=6, error_kw={"lw": 2, "capthick": 2}, width=0.6, alpha=0.85)
    ax.axhline(0, color="gray", ls=":", lw=1.5, alpha=0.6)
    for i, (m, ci) in enumerate(zip(means, cis)):
        ax.text(i, m + ci + 0.01, f"{m:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([SCHED_SHORT[s] for s in scheds], fontsize=10, fontweight="bold")
    _style(ax, "", "Cosine Similarity (end of burst phase)",
           "Gradient Cosine Similarity at End of Burst Phase\nBurst Class vs Other Classes (mean +/- 95% CI)")
    fig.tight_layout()
    p_ = pdir / "grad_cosine_end_burst_bars.png"
    fig.savefig(p_, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p_


def pairwise_grad_cosine_heatmap(pdir, results, cfg):
    """Heatmap of pairwise gradient cosine similarity between all task types.

    Shows snapshots at key training steps, averaged across seeds and schedules.
    Focuses on burst_pos=2 setting: burst tasks (b* at pos 2) vs other-class compositions.
    """
    has_data = any("pairwise_snapshots" in r and r["pairwise_snapshots"] for r in results)
    if not has_data:
        return []

    snaps_by_step = defaultdict(list)
    for r in results:
        if "pairwise_snapshots" not in r:
            continue
        for snap in r["pairwise_snapshots"]:
            snaps_by_step[snap["step"]].append(snap)

    all_steps = sorted(snaps_by_step.keys())

    paths = []
    for target_step in all_steps:
        snaps_at_step = snaps_by_step[target_step]
        if not snaps_at_step:
            continue

        labels = snaps_at_step[0]["labels"]
        n = len(labels)
        matrices = [np.array(s["matrix"]) for s in snaps_at_step if len(s["matrix"]) == n]
        if not matrices:
            continue
        mean_matrix = np.mean(matrices, axis=0)

        n_burst = snaps_at_step[0]["n_burst"]
        phase = snaps_at_step[0]["phase"]

        fig, ax = plt.subplots(figsize=(max(6, n * 0.9), max(5, n * 0.8)))
        vmax = 1.0
        vmin = -1.0
        im = ax.imshow(mean_matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax, interpolation="nearest")

        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=10, fontweight="bold")

        # Draw separator between burst and other tasks
        if 0 < n_burst < n:
            sep = n_burst - 0.5
            ax.axhline(sep, color="black", lw=2)
            ax.axvline(sep, color="black", lw=2)
            ax.text(n_burst / 2 - 0.5, -0.7, "Burst Tasks", ha="center",
                    fontsize=9, color="#D32F2F", fontweight="bold")
            ax.text(n_burst + (n - n_burst) / 2 - 0.5, -0.7, "Other Tasks", ha="center",
                    fontsize=9, color="#1565C0", fontweight="bold")

        for i in range(n):
            for j in range(n):
                val = mean_matrix[i, j]
                color = "white" if abs(val) > 0.55 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)

        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03, label="Cosine Similarity")
        ax.set_title(
            f"Pairwise Gradient Cosine Similarity — Step {target_step} ({phase})\n"
            f"(B=Burst tasks with b* at pos 2, O=Other-class tasks, avg over {len(matrices)} runs)",
            fontsize=12, fontweight="bold")
        fig.tight_layout()
        p_ = pdir / f"pairwise_grad_cosine_step{target_step}.png"
        fig.savefig(p_, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(p_)

    return paths


def pairwise_grad_cosine_evolution(pdir, results, cfg):
    """Line plot showing how mean within-group and cross-group cosine similarity evolves."""
    bcfg = cfg.get("base_cfg", cfg)
    T, U = bcfg["total_steps"], bcfg["reversion_steps"]

    has_data = any("pairwise_snapshots" in r and r["pairwise_snapshots"] for r in results)
    if not has_data:
        return None

    # For each result, extract per-step mean similarities for:
    # burst-burst, other-other, burst-other
    def _extract_means(snaps):
        steps, bb_vals, oo_vals, bo_vals = [], [], [], []
        for snap in snaps:
            mat = np.array(snap["matrix"])
            n_b = snap["n_burst"]
            n_o = snap["n_other"]
            n = n_b + n_o
            if mat.shape[0] != n:
                continue

            bb_block = mat[:n_b, :n_b]
            n_o = n - n_b
            oo_block = mat[n_b:, n_b:]
            bo_block = mat[:n_b, n_b:]

            bb_mask = ~np.eye(n_b, dtype=bool)
            oo_mask = ~np.eye(n_o, dtype=bool)

            steps.append(snap["step"])
            bb_vals.append(bb_block[bb_mask].mean() if n_b > 1 else 0.0)
            oo_vals.append(oo_block[oo_mask].mean() if n_o > 1 else 0.0)
            bo_vals.append(bo_block.mean() if bo_block.size > 0 else 0.0)
        return np.array(steps), np.array(bb_vals), np.array(oo_vals), np.array(bo_vals)

    all_steps_set = sorted(set(
        snap["step"]
        for r in results if "pairwise_snapshots" in r
        for snap in r["pairwise_snapshots"]
    ))
    if not all_steps_set:
        return None

    steps_ref = np.array(all_steps_set)
    bb_all, oo_all, bo_all = [], [], []
    for r in results:
        if "pairwise_snapshots" not in r or not r["pairwise_snapshots"]:
            continue
        s, bb, oo, bo = _extract_means(r["pairwise_snapshots"])
        if len(s) < 2:
            continue
        bb_all.append(np.interp(steps_ref, s, bb))
        oo_all.append(np.interp(steps_ref, s, oo))
        bo_all.append(np.interp(steps_ref, s, bo))

    if not bb_all:
        return None

    fig, ax = plt.subplots(figsize=(14, 7))
    for arr, color, label in [
        (np.array(bb_all), "#D32F2F", "Burst–Burst (within-group)"),
        (np.array(oo_all), "#1565C0", "Other–Other (within-group)"),
        (np.array(bo_all), "#FF6F00", "Burst–Other (cross-group)"),
    ]:
        m = np.mean(arr, axis=0)
        n_s = len(arr)
        ci = 1.96 * np.std(arr, axis=0) / np.sqrt(n_s) if n_s > 1 else np.std(arr, axis=0)
        ax.plot(steps_ref, m, color=color, lw=2.5, label=label, marker="o", markersize=5)
        ax.fill_between(steps_ref, m - ci, m + ci, color=color, alpha=0.15)

    ax.axvline(T, color="black", ls="--", lw=2, alpha=0.6)
    ax.axhline(0, color="gray", ls=":", lw=1, alpha=0.4)
    ax.text(T * 0.5, -0.12, "FOUNDATION+BURST", ha="center", fontsize=11, color="gray",
            fontweight="bold", transform=ax.get_xaxis_transform())
    ax.text(T + U * 0.5, -0.12, "REVERSION", ha="center", fontsize=11, color="gray",
            fontweight="bold", transform=ax.get_xaxis_transform())
    ax.set_xlim(steps_ref[0], steps_ref[-1])
    _style(ax, "Step", "Mean Cosine Similarity",
           "Pairwise Gradient Cosine Similarity Evolution\n"
           "Burst Tasks (b* at pos 2) vs Other-Class Tasks (mean +/- 95% CI, all seeds/schedules)")
    ax.legend(fontsize=12, loc="best", framealpha=0.9, edgecolor="gray")
    fig.tight_layout()
    p_ = pdir / "pairwise_grad_cosine_evolution.png"
    fig.savefig(p_, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p_


def generate_all(run_dir, results, cfg):
    pdir = Path(run_dir) / "presentation"
    pdir.mkdir(exist_ok=True)
    cp = {}
    ns = len(set(r["seed"] for r in results))
    gr = _group(results)

    burst_key = "acc_burst"
    other_key = "acc_other"
    auc_metric = "reversion_auc"
    peak_metric = "peak_burst"

    print("  Schedule bars...")
    cp["schedule_bars"] = schedule_bars(pdir, results, cfg)
    print("  Burst class overlay...")
    cp["overlay_burst"] = overlay(pdir, results, cfg, burst_key,
                              "Burst Class Accuracy (free generation)",
                              f"Burst Class Accuracy Over Foundation+Burst & Reversion\n(mean +/- 95% CI, n={ns} seeds)",
                              "overlay_burst.png", groups=gr)
    print("  Other classes overlay...")
    cp["overlay_other"] = overlay(pdir, results, cfg, other_key,
                              "Other Classes Accuracy (free generation)",
                              f"Other Classes Accuracy Over Foundation+Burst & Reversion\n(mean +/- 95% CI, n={ns} seeds)",
                              "overlay_other.png", loc="lower right", groups=gr)
    print("  Reversion AUC bars...")
    cp["auc_bars"] = bar_chart(pdir, results, cfg, auc_metric,
                               "Reversion AUC (higher = slower forgetting)",
                               "Reversion AUC by Schedule\n(mean +/- 95% CI, individual seeds shown)",
                               "auc_bars.png", groups=gr)
    print("  Quarter-life bars...")
    cp["ql_bars"] = bar_chart(pdir, results, cfg, "quarter_life",
                              "Quarter-life (reversion steps to 25% of peak)",
                              "Quarter-life by Schedule\n(mean +/- 95% CI, individual seeds shown)",
                              "quarterlife_bars.png", groups=gr)
    print("  Peak burst bars...")
    cp["peak_bars"] = bar_chart(pdir, results, cfg, peak_metric,
                                "Peak Burst Class Accuracy at End of Training",
                                "Peak Burst Class Accuracy by Schedule\n(mean +/- 95% CI, individual seeds shown)",
                                "peak_b_bars.png", fmt_dec=3, groups=gr)
    print("  AUC diff heatmap...")
    cp["auc_diff"] = auc_diff(pdir, results, cfg, groups=gr)
    print("  LR schedule...")
    cp["lr"] = lr_schedule(pdir, cfg)
    print("  Reversion zoom...")
    cp["reversion_zoom"] = reversion_zoom(pdir, results, cfg, groups=gr)
    print("  Summary table...")
    cp["summary_table"] = summary_table(pdir, results, cfg, groups=gr)
    print("  Per-schedule overlays...")
    cp["per_sched"] = per_sched(pdir, results, cfg, groups=gr)

    print("  Gradient cosine similarity overlay...")
    cp["grad_cosine_overlay"] = grad_cosine_sim_overlay(pdir, results, cfg, groups=gr)
    print("  Gradient cosine similarity bars...")
    cp["grad_cosine_bars"] = grad_cosine_sim_by_schedule(pdir, results, cfg, groups=gr)
    print("  Pairwise gradient cosine heatmaps...")
    cp["pairwise_heatmaps"] = pairwise_grad_cosine_heatmap(pdir, results, cfg)
    print("  Pairwise gradient cosine evolution...")
    cp["pairwise_evolution"] = pairwise_grad_cosine_evolution(pdir, results, cfg)
    return cp
