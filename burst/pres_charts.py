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

PALETTE = {
    "end_block": "#D32F2F", "end_mixed_75b": "#E64A19",
    "end_mixed_50b": "#F57C00", "end_mixed_25b": "#00897B",
    "uniform": "#1565C0",
}
SCHED_SHORT = {
    "end_block": "End Block (100% B)", "end_mixed_75b": "End Mixed (75% B)",
    "end_mixed_50b": "End Mixed (50% B)", "end_mixed_25b": "End Mixed (25% B)",
    "uniform": "Uniform (~10% B)",
}
SCHEDULE_ORDER = ["end_block", "end_mixed_75b", "end_mixed_50b", "end_mixed_25b", "uniform"]


def _ordered(scheds):
    return [s for s in SCHEDULE_ORDER if s in scheds]


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
    T, U, bs, p = bcfg["total_steps"], bcfg["undo_steps"], bcfg["batch_size"], bcfg["p_target"]
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
    axes[0].set_title("Training Schedules: Fraction of B Data per Step",
                      fontsize=14, fontweight="bold", pad=10)
    axes[0].annotate("TRAIN", xy=(T * 0.5, 1.15), fontsize=11, color="gray",
                     fontweight="bold", ha="center", annotation_clip=False)
    axes[0].annotate("UNDO (A only)", xy=(T + U * 0.5, 1.15), fontsize=11, color="gray",
                     fontweight="bold", ha="center", annotation_clip=False)
    fig.tight_layout(rect=[0.15, 0, 1, 0.97])
    p_ = pdir / "schedule_bars.png"
    fig.savefig(p_, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p_


def overlay(pdir, results, cfg, key, yl, title, fname, loc="center left"):
    bcfg = cfg.get("base_cfg", cfg)
    T, U = bcfg["total_steps"], bcfg["undo_steps"]
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
    ax.text(T * 0.5, -0.12, "TRAIN", ha="center", fontsize=12, color="gray",
            fontweight="bold", transform=ax.get_xaxis_transform())
    ax.text(T + U * 0.5, -0.12, "UNDO", ha="center", fontsize=12, color="gray",
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


def bar_chart(pdir, results, cfg, metric, yl, title, fname, fmt_dec=0):
    bcfg = cfg.get("base_cfg", cfg)
    U = bcfg.get("undo_steps", 500)
    groups = _group(results)
    scheds = _ordered(groups.keys())
    n = len(scheds)
    fig, ax = plt.subplots(figsize=(12, 7))
    xs = np.arange(n)
    means, cis, all_v = [], [], []
    for sched in scheds:
        vals = np.array([r.get(metric, 0) for r in groups[sched]])
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


def auc_diff(pdir, results, cfg):
    groups = _group(results)
    scheds = _ordered(groups.keys())
    n = len(scheds)
    mean_aucs = {s: np.mean([r.get("undo_auc", 0) for r in groups[s]]) for s in scheds}
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
    ax.set_title("Pairwise Undo AUC Difference (%)\n(row - col) / |col| x 100",
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
    T, U = bcfg["total_steps"], bcfg["undo_steps"]
    total = T + U
    lr_max, lr_min, warmup = bcfg["lr"], bcfg["min_lr"], bcfg["warmup_iters"]
    steps = np.arange(1, total + 1)
    lrs = np.zeros(total)
    for i, s in enumerate(steps):
        if s < warmup:
            lrs[i] = lr_max * s / warmup
        else:
            decay = (s - warmup) / (total - warmup)
            lrs[i] = lr_min + 0.5 * (1.0 + math.cos(math.pi * decay)) * (lr_max - lr_min)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(steps, lrs, color="#1565C0", lw=2.5)
    ax.axvline(T, color="black", lw=2, ls="--")
    ax.set_xlim(0, total)
    _style(ax, "Global Step", "Learning Rate",
           "Learning Rate Schedule (cosine decay with linear warmup)")
    ax.text(T * 0.5, ax.get_ylim()[1] * 0.92, "TRAIN", ha="center", fontsize=11,
            color="gray", fontweight="bold")
    ax.text(T + U * 0.5, ax.get_ylim()[1] * 0.92, "UNDO", ha="center", fontsize=11,
            color="gray", fontweight="bold")
    fig.tight_layout()
    p_ = pdir / "lr_schedule.png"
    fig.savefig(p_, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p_


def undo_zoom(pdir, results, cfg):
    bcfg = cfg.get("base_cfg", cfg)
    T, U = bcfg["total_steps"], bcfg["undo_steps"]
    groups = _group(results)
    fig, ax = plt.subplots(figsize=(14, 7))
    for sched in _ordered(groups.keys()):
        runs = groups[sched]
        steps = np.array(runs[0]["log"]["step"])
        vals = np.array([np.array(r["log"]["acc_B_comp"]) for r in runs])
        mask = steps >= T
        us = steps[mask] - T
        uv = vals[:, mask]
        m = np.mean(uv, axis=0)
        n_s = len(runs)
        ci = 1.96 * np.std(uv, axis=0) / np.sqrt(n_s) if n_s > 1 else np.std(uv, axis=0)
        ax.plot(us, m, color=PALETTE[sched], lw=2.5, label=SCHED_SHORT[sched])
        ax.fill_between(us, m - ci, m + ci, color=PALETTE[sched], alpha=0.15)
    ax.axhline(0.25, color="gray", ls=":", alpha=0.5, lw=1.5)
    ax.text(U * 0.95, 0.27, "25% threshold", fontsize=9, color="gray", ha="right")
    ax.set_xlim(0, U)
    ax.set_ylim(-0.05, 1.05)
    _style(ax, "Undo Steps (after B removal)", "B Comp Accuracy",
           "Forgetting Dynamics: B Accuracy During Undo Phase\n(mean +/- 95% CI, n=5 seeds)")
    ax.legend(fontsize=11, loc="upper right", framealpha=0.9, edgecolor="gray")
    fig.tight_layout()
    p_ = pdir / "undo_zoom.png"
    fig.savefig(p_, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p_


def summary_table(pdir, results, cfg):
    bcfg = cfg.get("base_cfg", cfg)
    U = bcfg["undo_steps"]
    groups = _group(results)
    scheds = _ordered(groups.keys())
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")
    cols = ["Schedule", "Peak B\n(mean +/- CI)", "Quarter-life\n(mean +/- CI)",
            "Undo AUC\n(mean +/- CI)", "A Acc End\n(mean +/- CI)"]
    rows = []
    for sched in scheds:
        runs = groups[sched]
        def fmt(vals, d=3):
            m = vals.mean()
            ci = 1.96 * vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else vals.std()
            return f"{m:.{d}f} +/- {ci:.{d}f}" if d > 0 else f"{m:.0f} +/- {ci:.0f}"
        rows.append([
            SCHED_SHORT[sched],
            fmt(np.array([r.get("train_end_B_comp", 0) for r in runs]), 3),
            fmt(np.array([r.get("quarter_life", U) for r in runs]), 0),
            fmt(np.array([r.get("undo_auc", 0) for r in runs]), 0),
            fmt(np.array([r["log"]["acc_A_comp"][-1] for r in runs]), 3),
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
    ax.set_title("Summary Statistics (n=5 seeds per schedule)",
                 fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    p_ = pdir / "summary_table.png"
    fig.savefig(p_, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p_


def per_sched(pdir, results, cfg):
    bcfg = cfg.get("base_cfg", cfg)
    T, U = bcfg["total_steps"], bcfg["undo_steps"]
    groups = _group(results)
    paths = []
    for sched in _ordered(groups.keys()):
        runs = groups[sched]
        steps = np.array(runs[0]["log"]["step"])
        fig, ax = plt.subplots(figsize=(14, 6))
        for k, (c, lbl) in [("acc_A_comp", ("#1565C0", "A comp (background)")),
                              ("acc_B_comp", ("#D32F2F", "B comp (novel)"))]:
            vals = np.array([np.array(r["log"][k]) for r in runs])
            m = np.mean(vals, axis=0)
            n_s = len(runs)
            ci = 1.96 * np.std(vals, axis=0) / np.sqrt(n_s) if n_s > 1 else np.std(vals, axis=0)
            ax.plot(steps, m, color=c, lw=2.5, label=lbl)
            ax.fill_between(steps, m - ci, m + ci, color=c, alpha=0.15)
        ax.axvline(T, color="black", ls="--", lw=2, alpha=0.6)
        ax.set_xlim(0, T + U)
        ax.set_ylim(-0.05, 1.05)
        _style(ax, "Step", "Accuracy (free generation)",
               f"{SCHED_SHORT[sched]}: A vs B Accuracy (mean +/- 95% CI, n=5)")
        ax.legend(fontsize=12, loc="center left", framealpha=0.9)
        fig.tight_layout()
        p_ = pdir / f"per_sched_{sched}.png"
        fig.savefig(p_, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(p_)
    return paths


def generate_all(run_dir, results, cfg):
    pdir = Path(run_dir) / "presentation"
    pdir.mkdir(exist_ok=True)
    cp = {}
    print("  Schedule bars...")
    cp["schedule_bars"] = schedule_bars(pdir, results, cfg)
    print("  B comp overlay...")
    cp["overlay_b"] = overlay(pdir, results, cfg, "acc_B_comp",
                              "B Comp Accuracy (free generation)",
                              "Novel Function (B) Accuracy Over Training & Undo\n(mean +/- 95% CI, n=5 seeds)",
                              "overlay_b_comp.png")
    print("  A comp overlay...")
    cp["overlay_a"] = overlay(pdir, results, cfg, "acc_A_comp",
                              "A Comp Accuracy (free generation)",
                              "Background Knowledge (A) Accuracy Over Training & Undo\n(mean +/- 95% CI, n=5 seeds)",
                              "overlay_a_comp.png", loc="lower right")
    print("  AUC bars...")
    cp["auc_bars"] = bar_chart(pdir, results, cfg, "undo_auc",
                               "Undo AUC (higher = slower forgetting)",
                               "Undo AUC by Schedule\n(mean +/- 95% CI, individual seeds shown)",
                               "auc_bars.png")
    print("  Quarter-life bars...")
    cp["ql_bars"] = bar_chart(pdir, results, cfg, "quarter_life",
                              "Quarter-life (undo steps to 25% of peak)",
                              "Quarter-life by Schedule\n(mean +/- 95% CI, individual seeds shown)",
                              "quarterlife_bars.png")
    print("  Peak B bars...")
    cp["peak_bars"] = bar_chart(pdir, results, cfg, "train_end_B_comp",
                                "Peak B Accuracy at End of Training",
                                "Peak Novel Function (B) Accuracy by Schedule\n(mean +/- 95% CI, individual seeds shown)",
                                "peak_b_bars.png", fmt_dec=3)
    print("  AUC diff heatmap...")
    cp["auc_diff"] = auc_diff(pdir, results, cfg)
    print("  LR schedule...")
    cp["lr"] = lr_schedule(pdir, cfg)
    print("  Undo zoom...")
    cp["undo_zoom"] = undo_zoom(pdir, results, cfg)
    print("  Summary table...")
    cp["summary_table"] = summary_table(pdir, results, cfg)
    print("  Per-schedule overlays...")
    cp["per_sched"] = per_sched(pdir, results, cfg)
    return cp
