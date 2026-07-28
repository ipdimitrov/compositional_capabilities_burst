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
    TrainConfig, reversion_life_key, reversion_life_label,
    burst_steps_for_mode as _burst_T_mode, MODE_CURRENT,
)


def load_grad_sim_data(run_dir) -> list[dict]:
    """Load all grad cosine sim records from the dedicated folder.

    Falls back to extracting from all_results.pkl if the folder doesn't exist.
    """
    rd = Path(run_dir)
    records = []
    for gs_dir in [rd / "results" / "grad_cosine_sim", rd / "grad_cosine_sim"]:
        if gs_dir.is_dir():
            for fp in sorted(gs_dir.glob("*.json")):
                with open(fp) as f:
                    records.append(json.load(f))
            if records:
                return records

    for pkl in [rd / "logs" / "all_results.pkl", rd / "all_results.pkl"]:
        if pkl.exists():
            with open(pkl, "rb") as f:
                results = pickle.load(f)
            for r in results:
                if "grad_sim_log" in r and r["grad_sim_log"]["step"]:
                    records.append({
                        "schedule": r["schedule"], "seed": r["seed"],
                        "label": r.get("label", ""),
                        "grad_sim_log": r["grad_sim_log"],
                        "pairwise_snapshots": r.get("pairwise_snapshots", []),
                    })
            if records:
                return records
    return records


def _group_gs(records):
    g = defaultdict(list)
    for r in records:
        g[r["schedule"]].append(r)
    return g


def _group(results):
    g = defaultdict(list)
    for r in results:
        g[r["schedule"]].append(r)
    return g


def _sched_T(groups: dict) -> dict[str, int]:
    """Per-schedule burst length T from the first run's config."""
    return {s: runs[0]["config"]["total_steps"] for s, runs in groups.items()}


def _T_for(sched: str, bcfg: dict) -> int:
    """Burst length T for a schedule, using base_steps from bcfg."""
    mode = bcfg.get("_burst_mode", MODE_CURRENT)
    return _burst_T_mode(sched, mode, bcfg["total_steps"])


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
    bcfg = {**cfg.get("base_cfg", cfg), "_burst_mode": cfg.get("burst_mode", MODE_CURRENT)}
    P = bcfg.get("pre_burst_steps", 0)
    U, bs, p = bcfg["reversion_steps"], bcfg["batch_size"], bcfg["p_target"]
    groups = _group(results)
    Ts = _sched_T(groups)
    scheds = _ordered(groups.keys())
    n = len(scheds)
    max_total = max(P + Ts[s] + U for s in scheds)
    fig, axes = plt.subplots(n, 1, figsize=(14, 1.8 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for i, sched in enumerate(scheds):
        ax = axes[i]
        T_s = Ts[sched]
        total_s = P + T_s + U
        fracs = np.zeros(total_s)
        for s in range(T_s):
            np.random.seed(107 * 10000 + s)
            fracs[P + s] = n_target_for_step(s, T_s, sched, p, bs) / bs
        ax.fill_between(range(total_s), fracs, color=PALETTE[sched], alpha=0.7)
        if P > 0:
            ax.axvline(P, color="black", lw=2, ls="--")
        ax.axvline(P + T_s, color="black", lw=2, ls="--")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, max_total)
        ax.set_ylabel(SCHED_SHORT[sched], fontsize=9, fontweight="bold",
                       rotation=0, labelpad=120, ha="left", va="center")
        ax.set_yticks([0, 0.5, 1.0])
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel("Step", fontsize=12, fontweight="bold")
    axes[0].set_title("Training Schedules: Fraction of Burst Data per Step",
                      fontsize=14, fontweight="bold", pad=10)
    fig.tight_layout(rect=[0.15, 0, 1, 0.97])
    p_ = pdir / "schedule_bars.png"
    fig.savefig(p_, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p_


def overlay(pdir, results, cfg, key, yl, title, fname, loc="center left",
            groups=None, align="absolute", clamp_01=None):
    bcfg = {**cfg.get("base_cfg", cfg), "_burst_mode": cfg.get("burst_mode", MODE_CURRENT)}
    P = bcfg.get("pre_burst_steps", 0)
    U = bcfg["reversion_steps"]
    if groups is None:
        groups = _group(results)
    Ts = _sched_T(groups)
    fig, ax = plt.subplots(figsize=(14, 7))
    all_vals_flat = []
    for sched in _ordered(groups.keys()):
        runs = groups[sched]
        T_s = Ts[sched]
        burst_end_s = P + T_s
        steps = np.array(runs[0]["log"]["step"])
        try:
            vals = np.array([np.array(r["log"][key]) for r in runs])
        except (KeyError, ValueError):
            continue
        m = np.mean(vals, axis=0)
        n_s = len(runs)
        ci = 1.96 * np.std(vals, axis=0) / np.sqrt(n_s) if n_s > 1 else np.std(vals, axis=0)
        if align == "end":
            x = steps - burst_end_s
        else:
            x = steps
        ax.plot(x, m, color=PALETTE[sched], lw=2.5, label=SCHED_SHORT[sched])
        ax.fill_between(x, m - ci, m + ci, color=PALETTE[sched], alpha=0.15)
        all_vals_flat.extend(m.tolist())

    T_max = max(Ts.values())
    burst_end_max = P + T_max
    total = P + T_max + U
    if align == "end":
        ax.axvline(0, color="black", ls="--", lw=2, alpha=0.6)
        if P > 0:
            ax.text(-burst_end_max + P * 0.5, -0.12, "PRE", ha="center", fontsize=10, color="gray",
                    fontweight="bold", transform=ax.get_xaxis_transform())
        ax.text(-T_max * 0.3, -0.12, "SPECIAL", ha="center", fontsize=12, color="gray",
                fontweight="bold", transform=ax.get_xaxis_transform())
        ax.text(U * 0.5, -0.12, "ALL-BUT-SPECIAL", ha="center", fontsize=12, color="gray",
                fontweight="bold", transform=ax.get_xaxis_transform())
        ax.set_xlim(-burst_end_max, U)
        xl = "Steps from Burst End"
    else:
        if P > 0:
            ax.axvline(P, color="black", ls="--", lw=2, alpha=0.6)
            ax.text(P * 0.5, -0.12, "ALL-BUT-SPECIAL", ha="center", fontsize=10, color="gray",
                    fontweight="bold", transform=ax.get_xaxis_transform())
        ax.text(P + T_max * 0.5, -0.12, "SPECIAL", ha="center", fontsize=12, color="gray",
                fontweight="bold", transform=ax.get_xaxis_transform())
        ax.text(burst_end_max + U * 0.5, -0.12, "ALL-BUT-SPECIAL", ha="center", fontsize=12, color="gray",
                fontweight="bold", transform=ax.get_xaxis_transform())
        ax.set_xlim(0, total)
        xl = "Steps from Burst Start" if align == "start" else "Step"

    is_acc = key.startswith("acc_") or clamp_01 is True
    if is_acc:
        ax.set_ylim(-0.05, 1.05)
    _style(ax, xl, yl, title)
    ax.legend(fontsize=11, loc=loc, framealpha=0.9, edgecolor="gray")
    fig.tight_layout()
    p_ = pdir / fname
    fig.savefig(p_, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p_


def bar_chart(pdir, results, cfg, metric, yl, title, fname, fmt_dec=0, groups=None):
    bcfg = {**cfg.get("base_cfg", cfg), "_burst_mode": cfg.get("burst_mode", MODE_CURRENT)}
    U = bcfg["reversion_steps"]
    if groups is None:
        groups = _group(results)
    if not any(metric in r for r in results):
        print(f"    [skip] no results contain metric '{metric}'")
        return None
    scheds = _ordered(groups.keys())
    n = len(scheds)
    fig, ax = plt.subplots(figsize=(12, 7))
    xs = np.arange(n)
    means, cis, all_v = [], [], []
    for sched in scheds:
        vals = np.array([r[metric] for r in groups[sched] if metric in r])
        if len(vals) == 0:
            vals = np.array([0.0])
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
    if metric.startswith("life_"):
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
    bcfg = {**cfg.get("base_cfg", cfg), "_burst_mode": cfg.get("burst_mode", MODE_CURRENT)}
    P = bcfg.get("pre_burst_steps", 0)
    U = bcfg["reversion_steps"]
    schedules = cfg.get("schedules", list(SCHEDULE_ORDER))

    fig, ax = plt.subplots(figsize=(14, 5))
    for sched in _ordered(schedules):
        T_s = _T_for(sched, bcfg)
        steps, lrs = _compute_lr(bcfg, pretrain_steps=P, burst_steps=T_s)
        ax.plot(steps, lrs, color=PALETTE.get(sched, "#1565C0"), lw=2.5,
                label=SCHED_SHORT.get(sched, sched), alpha=0.85)

    T_ref = _T_for(schedules[0], bcfg)
    ax.axvline(P, color="black", lw=1.5, ls="--", alpha=0.6)
    ylim = ax.get_ylim()
    ax.text(P * 0.5, ylim[1] * 0.92, "ALL-BUT-SPECIAL", ha="center", fontsize=10,
            color="gray", fontweight="bold")
    ax.text(P + T_ref * 0.5, ylim[1] * 0.92, "SPECIAL", ha="center", fontsize=11,
            color="gray", fontweight="bold")
    ax.text(P + T_ref + U * 0.5, ylim[1] * 0.92, "ALL-BUT-SPECIAL", ha="center", fontsize=10,
            color="gray", fontweight="bold")

    _style(ax, "Step", "Learning Rate",
           "Learning Rate Schedule (three-phase cosine)")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    fig.tight_layout()
    p_ = pdir / "lr_schedule.png"
    fig.savefig(p_, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p_


def reversion_zoom(pdir, results, cfg, groups=None, fname="reversion_zoom.png"):
    bcfg = {**cfg.get("base_cfg", cfg), "_burst_mode": cfg.get("burst_mode", MODE_CURRENT)}
    P = bcfg.get("pre_burst_steps", 0)
    U = bcfg["reversion_steps"]
    if groups is None:
        groups = _group(results)
    Ts = _sched_T(groups)
    fig, ax = plt.subplots(figsize=(14, 7))
    burst_log_key = "acc_burst"
    for sched in _ordered(groups.keys()):
        runs = groups[sched]
        T_s = Ts[sched]
        burst_end_s = P + T_s
        steps = np.array(runs[0]["log"]["step"])
        vals = np.array([np.array(r["log"][burst_log_key]) for r in runs])
        mask = steps >= burst_end_s
        reversion_steps_arr = steps[mask] - burst_end_s
        uv = vals[:, mask]
        m = np.mean(uv, axis=0)
        n_s = len(runs)
        ci = 1.96 * np.std(uv, axis=0) / np.sqrt(n_s) if n_s > 1 else np.std(uv, axis=0)
        ax.plot(reversion_steps_arr, m, color=PALETTE[sched], lw=2.5, label=SCHED_SHORT[sched])
        ax.fill_between(reversion_steps_arr, m - ci, m + ci, color=PALETTE[sched], alpha=0.15)
    thresholds = TrainConfig().reversion_thresholds
    for t in thresholds:
        ax.axhline(t, color="gray", ls=":", alpha=0.35, lw=1)
        ax.text(U * 0.95, t + 0.015, f"{int(t*100)}%", fontsize=7, color="gray", ha="right")
    ax.set_xlim(0, U)
    ax.set_ylim(-0.05, 1.05)
    ns = len(next(iter(groups.values())))
    _style(ax, "Reversion Steps (after Special Class removal)", "Special Class Accuracy",
           f"Forgetting Dynamics: Special Class Accuracy During Reversion\n(mean +/- 95% CI, n={ns} seeds)")
    ax.legend(fontsize=11, loc="upper right", framealpha=0.9, edgecolor="gray")
    fig.tight_layout()
    p_ = pdir / fname
    fig.savefig(p_, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p_


def summary_table(pdir, results, cfg, groups=None):
    bcfg = {**cfg.get("base_cfg", cfg), "_burst_mode": cfg.get("burst_mode", MODE_CURRENT)}
    U = bcfg["reversion_steps"]
    if groups is None:
        groups = _group(results)
    scheds = _ordered(groups.keys())
    thresholds = TrainConfig().reversion_thresholds
    fig_w = max(14, 6 + 2.5 * len(thresholds))
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    ax.axis("off")
    cols = ["Schedule", "Peak Special\n(mean +/- CI)"]
    for t in thresholds:
        cols.append(f"{reversion_life_label(t)}\n(mean +/- CI)")
    cols += ["Reversion AUC\n(mean +/- CI)", "Other Classes Acc\n(mean +/- CI)"]
    rows = []
    for sched in scheds:
        runs = groups[sched]
        def fmt(vals, d=3):
            m = vals.mean()
            ci = 1.96 * vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else vals.std()
            return f"{m:.{d}f} +/- {ci:.{d}f}" if d > 0 else f"{m:.0f} +/- {ci:.0f}"
        row = [
            SCHED_SHORT[sched],
            fmt(np.array([r["peak_burst"] for r in runs]), 3),
        ]
        for t in thresholds:
            key = reversion_life_key(t)
            row.append(fmt(np.array([r.get(key, U) for r in runs]), 0))
        row += [
            fmt(np.array([r["reversion_auc"] for r in runs]), 0),
            fmt(np.array([r["log"]["acc_other"][-1] for r in runs]), 3),
        ]
        rows.append(row)
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


def per_sched(pdir, results, cfg, groups=None, align="absolute"):
    bcfg = {**cfg.get("base_cfg", cfg), "_burst_mode": cfg.get("burst_mode", MODE_CURRENT)}
    P = bcfg.get("pre_burst_steps", 0)
    U = bcfg["reversion_steps"]
    if groups is None:
        groups = _group(results)
    Ts = _sched_T(groups)
    paths = []
    for sched in _ordered(groups.keys()):
        runs = groups[sched]
        T_s = Ts[sched]
        burst_end_s = P + T_s
        steps = np.array(runs[0]["log"]["step"])
        fig, ax = plt.subplots(figsize=(14, 6))
        for k, (c, lbl) in [("acc_other", ("#1565C0", "Other Classes")),
                              ("acc_burst", ("#D32F2F", "Special Class"))]:
            vals = np.array([np.array(r["log"][k]) for r in runs])
            m = np.mean(vals, axis=0)
            n_s = len(runs)
            ci = 1.96 * np.std(vals, axis=0) / np.sqrt(n_s) if n_s > 1 else np.std(vals, axis=0)
            if align == "end":
                xp = steps - burst_end_s
            else:
                xp = steps
            ax.plot(xp, m, color=c, lw=2.5, label=lbl)
            ax.fill_between(xp, m - ci, m + ci, color=c, alpha=0.15)
        if align == "end":
            if P > 0:
                ax.axvline(-burst_end_s, color="black", ls="--", lw=2, alpha=0.6)
            ax.axvline(-T_s, color="black", ls="--", lw=2, alpha=0.6)
            ax.axvline(0, color="black", ls="--", lw=2, alpha=0.6)
            ax.set_xlim(-burst_end_s, U)
        else:
            if P > 0:
                ax.axvline(P, color="black", ls="--", lw=2, alpha=0.6)
            ax.axvline(burst_end_s, color="black", ls="--", lw=2, alpha=0.6)
            ax.set_xlim(0, P + T_s + U)
        ax.set_ylim(-0.05, 1.05)
        n_s = len(runs)
        _style(ax, "Step", "Accuracy (free generation)",
               f"{SCHED_SHORT[sched]}: Other Classes vs Special Class (mean +/- 95% CI, n={n_s})")
        ax.legend(fontsize=12, loc="center left", framealpha=0.9)
        fig.tight_layout()
        suffix = f"_{align}" if align != "absolute" else ""
        p_ = pdir / f"per_sched_{sched}{suffix}.png"
        fig.savefig(p_, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(p_)
    return paths


def _interp_gs(gs_groups, scheds, key="burst_vs_other"):
    """Interpolate grad sim traces to a common step grid, per schedule.

    Returns {sched: (steps_ref, vals_arr)} where vals_arr is (n_seeds, n_steps).
    """
    out = {}
    for sched in scheds:
        runs = [r for r in gs_groups[sched]
                if r["grad_sim_log"]["step"]]
        if not runs:
            continue
        steps_list = [np.array(r["grad_sim_log"]["step"]) for r in runs]
        vals_list = [np.array(r["grad_sim_log"][key]) for r in runs]
        steps_ref = steps_list[0]
        interp_vals = []
        for s, v in zip(steps_list, vals_list):
            if len(s) > 1:
                interp_vals.append(np.interp(steps_ref, s, v))
        if interp_vals:
            out[sched] = (steps_ref, np.array(interp_vals))
    return out


def grad_cosine_sim_overlay(pdir, cfg, gs_records):
    bcfg = {**cfg.get("base_cfg", cfg), "_burst_mode": cfg.get("burst_mode", MODE_CURRENT)}
    U = bcfg["reversion_steps"]
    gs_groups = _group_gs(gs_records)
    scheds = _ordered(gs_groups.keys())
    interp = _interp_gs(gs_groups, scheds)
    if not interp:
        return None

    fig, ax = plt.subplots(figsize=(14, 9))
    for sched in scheds:
        if sched not in interp:
            continue
        steps_ref, vals_arr = interp[sched]
        m = np.mean(vals_arr, axis=0)
        n_s = len(vals_arr)
        ci = 1.96 * np.std(vals_arr, axis=0) / np.sqrt(n_s) if n_s > 1 else np.std(vals_arr, axis=0)
        ax.plot(steps_ref, m, color=PALETTE[sched], lw=2, label=SCHED_SHORT[sched])
        ax.fill_between(steps_ref, m - ci, m + ci, color=PALETTE[sched], alpha=0.12)

    T_max = max(_T_for(s, bcfg) for s in scheds)
    ax.axvline(T_max, color="black", ls="--", lw=2, alpha=0.6)
    ax.axhline(0, color="gray", ls=":", lw=1, alpha=0.5)
    ax.text(T_max * 0.5, -0.12, "SPECIAL", ha="center", fontsize=11, color="gray",
            fontweight="bold", transform=ax.get_xaxis_transform())
    ax.text(T_max + U * 0.5, -0.12, "ALL-BUT-SPECIAL", ha="center", fontsize=11, color="gray",
            fontweight="bold", transform=ax.get_xaxis_transform())
    ax.set_xlim(0, T_max + U)
    _style(ax, "Step", "Cosine Similarity",
           "Gradient Cosine Similarity: Special Class vs Other Classes\n(mean +/- 95% CI)")
    ax.legend(fontsize=10, loc="best", framealpha=0.9, edgecolor="gray")
    fig.tight_layout()
    p_ = pdir / "grad_cosine_burst_vs_other.png"
    fig.savefig(p_, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p_


def grad_cosine_sim_by_schedule(pdir, cfg, gs_records):
    bcfg = {**cfg.get("base_cfg", cfg), "_burst_mode": cfg.get("burst_mode", MODE_CURRENT)}
    gs_groups = _group_gs(gs_records)
    scheds = _ordered(gs_groups.keys())
    if not scheds:
        return None

    means, cis, all_v = [], [], []
    for sched in scheds:
        T_s = _T_for(sched, bcfg)
        runs = [r for r in gs_groups[sched] if r["grad_sim_log"]["step"]]
        end_vals = []
        for r in runs:
            steps = np.array(r["grad_sim_log"]["step"])
            sims = np.array(r["grad_sim_log"]["burst_vs_other"])
            burst_mask = steps <= T_s
            if burst_mask.any():
                end_vals.append(sims[burst_mask][-1])
        arr = np.array(end_vals) if end_vals else np.array([0.0])
        means.append(arr.mean())
        cis.append(1.96 * arr.std() / np.sqrt(len(arr)) if len(arr) > 1 else arr.std())
        all_v.append(arr)

    fig, ax = plt.subplots(figsize=(12, 7))
    xs = np.arange(len(scheds))
    colors = [PALETTE[s] for s in scheds]
    ax.bar(xs, means, yerr=cis, color=colors, edgecolor="black", lw=0.8,
           capsize=6, error_kw={"lw": 2, "capthick": 2}, width=0.6, alpha=0.85)
    for i, vals in enumerate(all_v):
        jit = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), i) + jit, vals,
                   color="black", s=40, zorder=5, alpha=0.6, edgecolor="white", lw=0.5)
    ax.axhline(0, color="gray", ls=":", lw=1.5, alpha=0.6)
    for i, (m, ci) in enumerate(zip(means, cis)):
        offset = 0.02 if m >= 0 else -0.02
        va = "bottom" if m >= 0 else "top"
        y = (m + ci + offset) if m >= 0 else (m - ci + offset)
        ax.text(i, y, f"{m:.3f}", ha="center", va=va, fontsize=11, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([SCHED_SHORT[s] for s in scheds], fontsize=10, fontweight="bold")
    _style(ax, "", "Cosine Similarity (end of burst phase)",
           "Gradient Cosine Similarity at End of Burst Phase\nSpecial Class vs Other Classes (mean +/- 95% CI)")
    fig.tight_layout()
    p_ = pdir / "grad_cosine_end_burst_bars.png"
    fig.savefig(p_, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p_


def grad_cosine_per_seed(pdir, cfg, gs_records):
    """Individual seed traces for each schedule — shows variance."""
    bcfg = {**cfg.get("base_cfg", cfg), "_burst_mode": cfg.get("burst_mode", MODE_CURRENT)}
    U = bcfg["reversion_steps"]
    gs_groups = _group_gs(gs_records)
    scheds = _ordered(gs_groups.keys())
    if not scheds:
        return []

    paths = []
    for sched in scheds:
        T_s = _T_for(sched, bcfg)
        runs = [r for r in gs_groups[sched] if r["grad_sim_log"]["step"]]
        if not runs:
            continue
        fig, ax = plt.subplots(figsize=(14, 7))
        for r in runs:
            steps = np.array(r["grad_sim_log"]["step"])
            vals = np.array(r["grad_sim_log"]["burst_vs_other"])
            ax.plot(steps, vals, lw=1.2, alpha=0.6, label=f"seed {r['seed']}")
        ax.axvline(T_s, color="black", ls="--", lw=2, alpha=0.6)
        ax.axhline(0, color="gray", ls=":", lw=1, alpha=0.5)
        ax.set_xlim(0, T_s + U)
        _style(ax, "Step", "Cosine Similarity",
               f"{SCHED_SHORT[sched]}: Gradient Cosine Similarity per Seed")
        ax.legend(fontsize=9, loc="best", framealpha=0.9)
        fig.tight_layout()
        p_ = pdir / f"grad_cosine_per_seed_{sched}.png"
        fig.savefig(p_, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(p_)
    return paths


def grad_cosine_rate_of_change(pdir, cfg, gs_records):
    """Derivative of cosine similarity over time — shows where alignment shifts fastest."""
    bcfg = {**cfg.get("base_cfg", cfg), "_burst_mode": cfg.get("burst_mode", MODE_CURRENT)}
    U = bcfg["reversion_steps"]
    gs_groups = _group_gs(gs_records)
    scheds = _ordered(gs_groups.keys())
    interp = _interp_gs(gs_groups, scheds)
    if not interp:
        return None

    fig, ax = plt.subplots(figsize=(14, 7))
    for sched in scheds:
        if sched not in interp:
            continue
        steps_ref, vals_arr = interp[sched]
        m = np.mean(vals_arr, axis=0)
        if len(steps_ref) < 3:
            continue
        dt = np.diff(steps_ref)
        dm = np.diff(m)
        rate = dm / np.maximum(dt, 1)
        mid_steps = (steps_ref[:-1] + steps_ref[1:]) / 2
        ax.plot(mid_steps, rate, color=PALETTE[sched], lw=2, label=SCHED_SHORT[sched])

    T_max = max(_T_for(s, bcfg) for s in scheds)
    ax.axvline(T_max, color="black", ls="--", lw=2, alpha=0.6)
    ax.axhline(0, color="gray", ls=":", lw=1, alpha=0.5)
    ax.set_xlim(0, T_max + U)
    _style(ax, "Step", "d(Cosine Similarity)/d(Step)",
           "Rate of Change of Gradient Cosine Similarity\nSpecial vs Other Classes")
    ax.legend(fontsize=10, loc="best", framealpha=0.9, edgecolor="gray")
    fig.tight_layout()
    p_ = pdir / "grad_cosine_rate_of_change.png"
    fig.savefig(p_, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p_


def grad_cosine_phase_comparison(pdir, cfg, gs_records):
    """Grouped bar chart: mean cosine sim during burst phase vs reversion phase, per schedule."""
    bcfg = {**cfg.get("base_cfg", cfg), "_burst_mode": cfg.get("burst_mode", MODE_CURRENT)}
    gs_groups = _group_gs(gs_records)
    scheds = _ordered(gs_groups.keys())
    if not scheds:
        return None

    burst_means, burst_cis = [], []
    rev_means, rev_cis = [], []
    for sched in scheds:
        T_s = _T_for(sched, bcfg)
        runs = [r for r in gs_groups[sched] if r["grad_sim_log"]["step"]]
        b_vals, r_vals = [], []
        for r in runs:
            steps = np.array(r["grad_sim_log"]["step"])
            sims = np.array(r["grad_sim_log"]["burst_vs_other"])
            b_mask = steps <= T_s
            r_mask = steps > T_s
            if b_mask.any():
                b_vals.append(sims[b_mask].mean())
            if r_mask.any():
                r_vals.append(sims[r_mask].mean())
        for vals, ms, cs in [(b_vals, burst_means, burst_cis),
                              (r_vals, rev_means, rev_cis)]:
            if vals:
                arr = np.array(vals)
                ms.append(arr.mean())
                cs.append(1.96 * arr.std() / np.sqrt(len(arr)) if len(arr) > 1 else arr.std())
            else:
                ms.append(0.0)
                cs.append(0.0)

    fig, ax = plt.subplots(figsize=(14, 7))
    xs = np.arange(len(scheds))
    w = 0.35
    ax.bar(xs - w / 2, burst_means, w, yerr=burst_cis, color="#D32F2F", alpha=0.8,
           edgecolor="black", lw=0.6, capsize=4, label="Burst Phase (mean)")
    ax.bar(xs + w / 2, rev_means, w, yerr=rev_cis, color="#1565C0", alpha=0.8,
           edgecolor="black", lw=0.6, capsize=4, label="Reversion Phase (mean)")
    ax.axhline(0, color="gray", ls=":", lw=1.5, alpha=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([SCHED_SHORT[s] for s in scheds], fontsize=10, fontweight="bold")
    _style(ax, "", "Mean Cosine Similarity",
           "Gradient Cosine Similarity: Burst Phase vs Reversion Phase\n(mean +/- 95% CI)")
    ax.legend(fontsize=11, framealpha=0.9)
    fig.tight_layout()
    p_ = pdir / "grad_cosine_phase_comparison.png"
    fig.savefig(p_, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p_


def grad_cosine_vs_auc_scatter(pdir, cfg, gs_records, results):
    """Scatter: end-of-burst cosine similarity vs reversion AUC, one dot per seed x schedule."""
    bcfg = {**cfg.get("base_cfg", cfg), "_burst_mode": cfg.get("burst_mode", MODE_CURRENT)}
    if not gs_records or not results:
        return None

    auc_lookup = {(r["schedule"], r["seed"]): r.get("reversion_auc", 0) for r in results}
    xs, ys, cs, labels = [], [], [], []
    for rec in gs_records:
        T_rec = _T_for(rec["schedule"], bcfg)
        steps = np.array(rec["grad_sim_log"]["step"])
        sims = np.array(rec["grad_sim_log"]["burst_vs_other"])
        burst_mask = steps <= T_rec
        if not burst_mask.any():
            continue
        end_sim = sims[burst_mask][-1]
        auc = auc_lookup.get((rec["schedule"], rec["seed"]))
        if auc is None:
            continue
        xs.append(end_sim)
        ys.append(auc)
        cs.append(PALETTE.get(rec["schedule"], "gray"))
        labels.append(rec["schedule"])

    if len(xs) < 2:
        return None

    fig, ax = plt.subplots(figsize=(10, 8))
    for x, y, c in zip(xs, ys, cs):
        ax.scatter(x, y, color=c, s=60, edgecolor="black", lw=0.5, zorder=3)

    seen = set()
    for x, y, c, lbl in zip(xs, ys, cs, labels):
        if lbl not in seen:
            ax.scatter([], [], color=c, s=60, edgecolor="black", lw=0.5,
                       label=SCHED_SHORT.get(lbl, lbl))
            seen.add(lbl)

    xs_arr, ys_arr = np.array(xs), np.array(ys)
    if len(xs_arr) > 2:
        corr = np.corrcoef(xs_arr, ys_arr)[0, 1]
        z = np.polyfit(xs_arr, ys_arr, 1)
        xline = np.linspace(xs_arr.min(), xs_arr.max(), 100)
        ax.plot(xline, np.polyval(z, xline), "k--", lw=1.5, alpha=0.5)
        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top")

    _style(ax, "Cosine Similarity (end of burst phase)", "Reversion AUC",
           "Gradient Alignment vs Forgetting Resistance\n(each dot = one seed x schedule)")
    ax.legend(fontsize=9, loc="best", framealpha=0.9, edgecolor="gray")
    fig.tight_layout()
    p_ = pdir / "grad_cosine_vs_auc_scatter.png"
    fig.savefig(p_, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p_


def grad_cosine_mean_over_phases_bars(pdir, cfg, gs_records):
    """Stacked-style bar: mean cosine sim at start, mid-burst, end-burst, mid-reversion, end-reversion."""
    bcfg = {**cfg.get("base_cfg", cfg), "_burst_mode": cfg.get("burst_mode", MODE_CURRENT)}
    U = bcfg["reversion_steps"]
    gs_groups = _group_gs(gs_records)
    scheds = _ordered(gs_groups.keys())
    if not scheds:
        return None

    cp_colors = ["#4CAF50", "#FF9800", "#D32F2F", "#2196F3", "#9C27B0"]
    cp_names = ["Start", "Mid-Burst", "End-Burst", "Mid-Rev", "End-Rev"]

    fig, ax = plt.subplots(figsize=(14, 7))
    n_cp = len(cp_names)
    xs = np.arange(len(scheds))
    w = 0.8 / n_cp

    for ci, cp_name in enumerate(cp_names):
        means = []
        for sched in scheds:
            T_s = _T_for(sched, bcfg)
            checkpoints = [
                (0, T_s // 4),
                (T_s // 4, 3 * T_s // 4),
                (3 * T_s // 4, T_s),
                (T_s, T_s + U // 2),
                (T_s + U // 2, T_s + U),
            ]
            lo, hi = checkpoints[ci]
            runs = [r for r in gs_groups[sched] if r["grad_sim_log"]["step"]]
            vals = []
            for r in runs:
                steps = np.array(r["grad_sim_log"]["step"])
                sims = np.array(r["grad_sim_log"]["burst_vs_other"])
                mask = (steps >= lo) & (steps < hi)
                if mask.any():
                    vals.append(sims[mask].mean())
            means.append(np.mean(vals) if vals else 0.0)
        ax.bar(xs + ci * w - 0.4 + w / 2, means, w, color=cp_colors[ci],
               alpha=0.85, edgecolor="black", lw=0.4, label=cp_names[ci])

    ax.axhline(0, color="gray", ls=":", lw=1.5, alpha=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([SCHED_SHORT[s] for s in scheds], fontsize=10, fontweight="bold")
    _style(ax, "", "Mean Cosine Similarity",
           "Gradient Cosine Similarity Across Training Phases\n(mean per phase window)")
    ax.legend(fontsize=10, framealpha=0.9, edgecolor="gray", ncol=n_cp)
    fig.tight_layout()
    p_ = pdir / "grad_cosine_phase_bars.png"
    fig.savefig(p_, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p_


def _draw_pairwise_heatmap(ax, mean_matrix, labels, n_burst, n_other_sub,
                           show_error, std_matrix):
    n = len(labels)
    im = ax.imshow(mean_matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0, interpolation="nearest")

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=9, fontweight="bold", rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9, fontweight="bold")

    seps = []
    if n_burst > 0:
        seps.append(n_burst - 0.5)
    if n_other_sub > 0:
        seps.append(n_burst + n_other_sub - 0.5)
    if n > n_burst + n_other_sub + 1:
        seps.append(n - 1 - 0.5)
    for sep in seps:
        ax.axhline(sep, color="black", lw=1.5)
        ax.axvline(sep, color="black", lw=1.5)

    fontsize = 8 if show_error else 9
    for i in range(n):
        for j in range(n):
            val = mean_matrix[i, j]
            txt_color = "white" if abs(val) > 0.55 else "black"
            if show_error and std_matrix is not None:
                txt = f"{val:.2f}\n+/-{std_matrix[i,j]:.2f}"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=6, fontweight="bold", color=txt_color)
            else:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=fontsize, fontweight="bold", color=txt_color)
    return im


def pairwise_grad_cosine_heatmap(pdir, cfg, gs_records):
    gs_groups = _group_gs(gs_records)
    scheds = _ordered(gs_groups.keys())

    paths = []
    for sched in scheds:
        records = gs_groups[sched]
        snaps_by_step: dict[int, list] = defaultdict(list)
        for r in records:
            if "pairwise_snapshots" not in r:
                continue
            for snap in r["pairwise_snapshots"]:
                snaps_by_step[snap["step"]].append(snap)

        if not snaps_by_step:
            continue

        for target_step in sorted(snaps_by_step.keys()):
            snaps_at_step = snaps_by_step[target_step]
            if not snaps_at_step:
                continue

            labels = snaps_at_step[0]["labels"]
            n = len(labels)
            matrices = [np.array(s["matrix"]) for s in snaps_at_step
                        if len(s["matrix"]) == n]
            if not matrices:
                continue
            stacked = np.array(matrices)
            mean_matrix = np.mean(stacked, axis=0)
            std_matrix = np.std(stacked, axis=0) if len(matrices) > 1 else np.zeros_like(mean_matrix)

            ref_snap = snaps_at_step[0]
            new_fmt = _is_new_format(ref_snap)
            if new_fmt:
                n_burst = ref_snap["n_burst"]
                n_other_sub = ref_snap["n_other_sub"]
            else:
                n_burst = ref_snap.get("n_burst", n // 2)
                n_other_sub = n - n_burst
            phase = ref_snap["phase"]
            n_seeds = len(matrices)
            sched_label = SCHED_SHORT.get(sched, sched)

            for show_error in [False, True]:
                suffix = "err" if show_error else "mean"
                fig_w = max(7, n * 1.1) if show_error else max(6, n * 0.9)
                fig_h = max(6, n * 1.0) if show_error else max(5, n * 0.8)
                fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                im = _draw_pairwise_heatmap(
                    ax, mean_matrix, labels, n_burst, n_other_sub,
                    show_error=show_error, std_matrix=std_matrix)
                fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03, label="Cosine Similarity")
                err_note = " (mean +/- std)" if show_error else ""
                ax.set_title(
                    f"Pairwise Grad Cosine Sim -- {sched_label} -- Step {target_step} ({phase}){err_note}\n"
                    f"(avg over {n_seeds} seeds)",
                    fontsize=11, fontweight="bold")
                fig.tight_layout()
                p_ = pdir / f"pw_heatmap_{sched}_step{target_step}_{suffix}.png"
                fig.savefig(p_, dpi=200, bbox_inches="tight")
                plt.close(fig)
                paths.append(p_)

    return paths


def _is_new_format(snap: dict) -> bool:
    return "n_other_sub" in snap


def _extract_pairwise_metrics(snap: dict) -> dict[str, float]:
    """Extract scalar summary metrics from a pairwise snapshot matrix.

    New format [BURST, O_F1..O_Fn, ALL_OTHER, ALL_DATA]:
      - burst_vs_other_sub: mean of BURST row across O_F1..O_Fn columns
      - other_sub_within:   mean off-diagonal of the O_F* block
      - burst_vs_all_other: BURST vs ALL_OTHER cell
      - burst_vs_all_data:  BURST vs ALL_DATA cell

    Old format [B1..B5, O1..O5] (backwards compat):
      - burst_vs_other_sub: mean of burst-other cross-block
      - other_sub_within:   mean off-diagonal of other-other block
    """
    mat = np.array(snap["matrix"])
    n = mat.shape[0]
    if n < 2:
        return {}

    if _is_new_format(snap):
        n_b = snap["n_burst"]
        n_os = snap["n_other_sub"]
        burst_idx = 0
        of_start, of_end = n_b, n_b + n_os
        all_other_idx = of_end if of_end < n else None
        all_data_idx = of_end + 1 if of_end + 1 < n else None

        metrics: dict[str, float] = {}
        if n_os > 0:
            metrics["burst_vs_other_sub"] = float(mat[burst_idx, of_start:of_end].mean())
            of_block = mat[of_start:of_end, of_start:of_end]
            if n_os > 1:
                mask = ~np.eye(n_os, dtype=bool)
                metrics["other_sub_within"] = float(of_block[mask].mean())
            else:
                metrics["other_sub_within"] = 1.0
        if all_other_idx is not None:
            metrics["burst_vs_all_other"] = float(mat[burst_idx, all_other_idx])
        if all_data_idx is not None:
            metrics["burst_vs_all_data"] = float(mat[burst_idx, all_data_idx])
        return metrics

    n_b = snap.get("n_burst", n // 2)
    n_o = n - n_b
    bo_block = mat[:n_b, n_b:]
    oo_block = mat[n_b:, n_b:]
    metrics = {
        "burst_vs_other_sub": float(bo_block.mean()),
    }
    if n_o > 1:
        oo_mask = ~np.eye(n_o, dtype=bool)
        metrics["other_sub_within"] = float(oo_block[oo_mask].mean())
    return metrics


PAIRWISE_METRIC_LABELS = {
    "burst_vs_other_sub": "BURST vs O_F* (mean)",
    "other_sub_within": "O_F* within-group (off-diag mean)",
    "burst_vs_all_other": "BURST vs ALL_OTHER",
    "burst_vs_all_data": "BURST vs ALL_DATA",
}

PAIRWISE_METRIC_COLORS = {
    "burst_vs_other_sub": "#FF6F00",
    "other_sub_within": "#1565C0",
    "burst_vs_all_other": "#D32F2F",
    "burst_vs_all_data": "#7B1FA2",
}


def _collect_pairwise_series(gs_records, scheds_grouped):
    """Build {sched: {metric: (steps_ref, vals_array_SxT)}} from pairwise snapshots."""
    result = {}
    for sched, records in scheds_grouped.items():
        all_steps_set = sorted(set(
            snap["step"]
            for r in records if "pairwise_snapshots" in r
            for snap in r["pairwise_snapshots"]
        ))
        if not all_steps_set:
            continue
        steps_ref = np.array(all_steps_set)

        per_seed: dict[str, list[np.ndarray]] = defaultdict(list)
        for r in records:
            if "pairwise_snapshots" not in r or not r["pairwise_snapshots"]:
                continue
            seed_steps, seed_metrics = [], defaultdict(list)
            for snap in sorted(r["pairwise_snapshots"], key=lambda s: s["step"]):
                m = _extract_pairwise_metrics(snap)
                if not m:
                    continue
                seed_steps.append(snap["step"])
                for k, v in m.items():
                    seed_metrics[k].append(v)
            if len(seed_steps) < 2:
                continue
            s_arr = np.array(seed_steps)
            for k, vals in seed_metrics.items():
                per_seed[k].append(np.interp(steps_ref, s_arr, np.array(vals)))

        if not per_seed:
            continue
        result[sched] = {
            "_steps": steps_ref,
            **{k: np.array(v) for k, v in per_seed.items()},
        }
    return result


def pairwise_grad_cosine_evolution_by_metric(pdir, cfg, gs_records):
    """One plot per metric, one line per schedule, error bars across seeds."""
    bcfg = {**cfg.get("base_cfg", cfg), "_burst_mode": cfg.get("burst_mode", MODE_CURRENT)}
    U = bcfg["reversion_steps"]

    gs_groups = _group_gs(gs_records)
    scheds = _ordered(gs_groups.keys())
    series = _collect_pairwise_series(gs_records, gs_groups)
    if not series:
        return []

    T_max = max(_T_for(s, bcfg) for s in scheds)
    metrics = list(PAIRWISE_METRIC_LABELS.keys())
    paths = []
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(14, 7))
        any_data = False
        for sched in scheds:
            if sched not in series or metric not in series[sched]:
                continue
            steps_ref = series[sched]["_steps"]
            vals_arr = series[sched][metric]
            m = np.mean(vals_arr, axis=0)
            n_s = len(vals_arr)
            ci = 1.96 * np.std(vals_arr, axis=0) / np.sqrt(n_s) if n_s > 1 else np.std(vals_arr, axis=0)
            c = PALETTE.get(sched, "gray")
            ax.plot(steps_ref, m, color=c, lw=2, label=SCHED_SHORT.get(sched, sched),
                    marker="o", markersize=4)
            ax.fill_between(steps_ref, m - ci, m + ci, color=c, alpha=0.12)
            any_data = True

        if not any_data:
            plt.close(fig)
            continue

        ax.axvline(T_max, color="black", ls="--", lw=2, alpha=0.6)
        ax.axhline(0, color="gray", ls=":", lw=1, alpha=0.4)
        ax.text(T_max * 0.5, -0.12, "BURST", ha="center", fontsize=11, color="gray",
                fontweight="bold", transform=ax.get_xaxis_transform())
        ax.text(T_max + U * 0.5, -0.12, "ALL-BUT-SPECIAL", ha="center", fontsize=11, color="gray",
                fontweight="bold", transform=ax.get_xaxis_transform())
        ax.set_xlim(0, T_max + U)
        _style(ax, "Step", "Cosine Similarity",
               f"{PAIRWISE_METRIC_LABELS[metric]}\n(mean +/- 95% CI per schedule)")
        ax.legend(fontsize=9, loc="best", framealpha=0.9, edgecolor="gray")
        fig.tight_layout()
        p_ = pdir / f"pw_evo_{metric}.png"
        fig.savefig(p_, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(p_)

    return paths


def pairwise_grad_cosine_evolution_per_schedule(pdir, cfg, gs_records):
    """One subplot per schedule, each showing all metric lines over time."""
    bcfg = {**cfg.get("base_cfg", cfg), "_burst_mode": cfg.get("burst_mode", MODE_CURRENT)}
    U = bcfg["reversion_steps"]

    gs_groups = _group_gs(gs_records)
    scheds = _ordered(gs_groups.keys())
    series = _collect_pairwise_series(gs_records, gs_groups)
    if not series:
        return None

    active_scheds = [s for s in scheds if s in series]
    if not active_scheds:
        return None

    metrics = list(PAIRWISE_METRIC_LABELS.keys())
    n_cols = min(3, len(active_scheds))
    n_rows = math.ceil(len(active_scheds) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 4.5 * n_rows), squeeze=False)

    for idx, sched in enumerate(active_scheds):
        T_s = _T_for(sched, bcfg)
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        steps_ref = series[sched]["_steps"]

        for metric in metrics:
            if metric not in series[sched]:
                continue
            vals_arr = series[sched][metric]
            m = np.mean(vals_arr, axis=0)
            n_s = len(vals_arr)
            ci = 1.96 * np.std(vals_arr, axis=0) / np.sqrt(n_s) if n_s > 1 else np.std(vals_arr, axis=0)
            c = PAIRWISE_METRIC_COLORS.get(metric, "gray")
            ax.plot(steps_ref, m, color=c, lw=2, label=PAIRWISE_METRIC_LABELS[metric],
                    marker="o", markersize=3)
            ax.fill_between(steps_ref, m - ci, m + ci, color=c, alpha=0.12)

        ax.axvline(T_s, color="black", ls="--", lw=1.5, alpha=0.5)
        ax.axhline(0, color="gray", ls=":", lw=1, alpha=0.4)
        ax.set_xlim(0, T_s + U)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(SCHED_SHORT.get(sched, sched), fontsize=10, fontweight="bold")
        ax.set_xlabel("Step", fontsize=9)
        ax.set_ylabel("Cosine Sim", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.15, lw=0.5)
        if idx == 0:
            ax.legend(fontsize=6, loc="best", framealpha=0.9)

    for idx in range(len(active_scheds), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle("Pairwise Grad Cosine Metrics per Schedule\n(mean +/- 95% CI across seeds)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p_ = pdir / "pw_evo_per_schedule.png"
    fig.savefig(p_, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p_


def _load_probe_data(run_dir):
    """Load probe results if available. Returns (results, meta) or (None, None)."""
    rd = Path(run_dir)
    probe_dir = rd / "probes"
    if not probe_dir.exists():
        probe_dir = rd / "results" / "probes"
    all_path = probe_dir / "all_probes.pkl"
    if not all_path.exists():
        return None, None
    with open(all_path, "rb") as f:
        results = pickle.load(f)
    meta_path = probe_dir / "probe_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        r0 = results[0]
        meta = {
            "checkpoint_steps": r0.get("checkpoint_steps", sorted(r0["probes"].keys())),
            "token_labels": r0.get("token_labels", []),
            "n_layers": r0.get("n_layers", 6),
            "total_steps": r0.get("total_steps", 500),
            "reversion_steps": r0.get("reversion_steps", 500),
        }
    return results, meta


def probe_heatmap_aggregated(pdir, probe_results, meta, cfg):
    """Seed-aggregated probe heatmaps at key steps, one per schedule."""
    if not probe_results:
        return []

    bcfg = {**cfg.get("base_cfg", cfg), "_burst_mode": cfg.get("burst_mode", MODE_CURRENT)}
    T = meta["total_steps"]
    U = meta["reversion_steps"]
    n_layers = meta["n_layers"]
    token_labels = meta["token_labels"]
    layer_labels = ["emb"] + [f"L{i}" for i in range(n_layers)]

    sched_data = defaultdict(list)
    for r in probe_results:
        sched_data[r["schedule"]].append(r)

    key_steps = [T // 2, T, T + U // 2, T + U]
    paths = []

    for sched in _ordered(sched_data.keys()):
        runs = sched_data[sched]
        for target_step in key_steps:
            arrs = []
            actual_step = target_step
            for r in runs:
                closest = min(r["probes"].keys(), key=lambda s: abs(s - target_step))
                if abs(closest - target_step) <= 30:
                    arrs.append(r["probes"][closest]["train_acc_KT"])
                    actual_step = closest
            if not arrs:
                continue

            mean_KT = np.mean(arrs, axis=0)
            K, Tpos = mean_KT.shape
            phase = "train" if actual_step <= T else "reversion"
            n_s = len(arrs)

            fig, ax = plt.subplots(figsize=(max(14, Tpos * 0.5), max(4, K * 0.6)))
            im = ax.imshow(mean_KT, aspect="auto", cmap="Blues", vmin=0.4, vmax=1.0,
                           interpolation="nearest")
            ax.set_xticks(range(Tpos))
            ax.set_xticklabels(token_labels[:Tpos], rotation=60, ha="right", fontsize=7)
            ax.set_yticks(range(K))
            ax.set_yticklabels(layer_labels[:K], fontsize=8)
            ax.set_xlabel("Token Position", fontsize=10)
            ax.set_ylabel("Layer", fontsize=10)
            ax.set_title(f"{SCHED_SHORT.get(sched, sched)} — step {actual_step} ({phase})\n"
                         f"Probe accuracy (Other vs Special), mean over {n_s} seeds",
                         fontsize=12, fontweight="bold")
            for k in range(K):
                for t in range(Tpos):
                    val = mean_KT[k, t]
                    color = "white" if val > 0.75 else "black"
                    ax.text(t, k, f"{val:.2f}", ha="center", va="center",
                            fontsize=5, color=color)
            cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label("Probe Accuracy (Other vs Special)", fontsize=9)
            fig.tight_layout()
            p_ = pdir / f"probe_heatmap_{sched}_step{actual_step}.png"
            fig.savefig(p_, dpi=150, bbox_inches="tight")
            plt.close(fig)
            paths.append(p_)

    return paths


def probe_dynamics_aggregated(pdir, probe_results, meta, cfg):
    """Mean probe accuracy over training, aggregated across seeds with 95% CI."""
    if not probe_results:
        return None

    bcfg = {**cfg.get("base_cfg", cfg), "_burst_mode": cfg.get("burst_mode", MODE_CURRENT)}
    T = meta["total_steps"]
    n_layers = meta["n_layers"]

    sched_data = defaultdict(list)
    for r in probe_results:
        sched_data[r["schedule"]].append(r)

    fig, ax = plt.subplots(figsize=(14, 7))
    for sched in _ordered(sched_data.keys()):
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
        n_s = np.sum(~np.isnan(arr), axis=0)
        std_vals = np.nanstd(arr, axis=0)
        ci = np.where(n_s > 1, 1.96 * std_vals / np.sqrt(n_s), std_vals)

        c = PALETTE.get(sched, "gray")
        ax.plot(steps_sorted, mean_vals, color=c, lw=2.5, label=SCHED_SHORT.get(sched, sched))
        ax.fill_between(steps_sorted, mean_vals - ci, mean_vals + ci, color=c, alpha=0.15)

    ax.axvline(T, color="black", ls="--", lw=2, alpha=0.6)
    ax.axhline(0.5, color="gray", ls=":", alpha=0.3)
    ns = len(set(r["seed"] for r in probe_results))
    _style(ax, "Step", "Mean Probe Accuracy (all layers & tokens)",
           f"Probe Accuracy Over Training (Other vs Special)\n(mean +/- 95% CI, n={ns} seeds)")
    ax.set_ylim(0.35, 1.05)
    ax.legend(fontsize=10, loc="best", framealpha=0.9, edgecolor="gray")
    fig.tight_layout()
    p_ = pdir / "probe_dynamics_aggregated.png"
    fig.savefig(p_, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p_


def probe_layer_schedule_heatmap(pdir, probe_results, meta, cfg):
    """Layer x Schedule heatmap at end of training + end of reversion."""
    if not probe_results:
        return []

    T = meta["total_steps"]
    U = meta["reversion_steps"]
    n_layers = meta["n_layers"]
    layer_labels = [f"L{i}" for i in range(n_layers)]

    sched_set = set(r["schedule"] for r in probe_results)
    col_scheds = [s for s in SCHEDULE_ORDER if s in sched_set] or sorted(sched_set)

    paths = []
    for target_step, phase_label in [(T, "end_train"), (T + U, "end_reversion")]:
        grid = np.full((n_layers, len(col_scheds)), np.nan)
        ci_grid = np.full((n_layers, len(col_scheds)), np.nan)

        for ci_idx, sched in enumerate(col_scheds):
            seed_means = []
            for r in probe_results:
                if r["schedule"] != sched:
                    continue
                closest = min(r["probes"].keys(), key=lambda s: abs(s - target_step))
                if abs(closest - target_step) > 30:
                    continue
                acc_KT = r["probes"][closest]["train_acc_KT"]
                seed_means.append(acc_KT[1:, :].mean(axis=1))
            if seed_means:
                arr = np.array(seed_means)
                grid[:, ci_idx] = arr.mean(axis=0)
                n_s = len(arr)
                if n_s > 1:
                    ci_grid[:, ci_idx] = 1.96 * arr.std(axis=0) / np.sqrt(n_s)

        fig, ax = plt.subplots(figsize=(max(6, len(col_scheds) * 1.4), max(3, n_layers * 0.6)))
        im = ax.imshow(grid, aspect="auto", cmap="Blues", vmin=0.4, vmax=1.0,
                       interpolation="nearest")
        ax.set_xticks(range(len(col_scheds)))
        ax.set_xticklabels([SCHED_SHORT.get(s, s) for s in col_scheds],
                           rotation=30, ha="right", fontsize=10)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels(layer_labels, fontsize=10)
        ax.set_xlabel("Schedule", fontsize=11)
        ax.set_ylabel("Layer", fontsize=11)
        ns = len(set(r["seed"] for r in probe_results))
        ax.set_title(f"Layer x Schedule — probe accuracy at {phase_label}\n"
                     f"(mean across tokens & {ns} seeds)",
                     fontsize=13, fontweight="bold")
        for row in range(n_layers):
            for col in range(len(col_scheds)):
                val = grid[row, col]
                if np.isnan(val):
                    continue
                ci_val = ci_grid[row, col]
                txt = f"{val:.3f}"
                if not np.isnan(ci_val):
                    txt += f"\n+/-{ci_val:.3f}"
                color = "white" if val > 0.75 else "black"
                ax.text(col, row, txt, ha="center", va="center", fontsize=8, color=color)
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03, label="Mean Probe Accuracy")
        fig.tight_layout()
        p_ = pdir / f"probe_layer_schedule_{phase_label}.png"
        fig.savefig(p_, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(p_)

    return paths


def _load_per_layer_data(gs_records) -> tuple[list[str], dict]:
    """Extract per-layer cossim data from gs_records.

    Returns (layer_names, {sched: {layer: (steps_arr, vals_SxT)}}).
    """
    layer_names: list[str] = []
    for r in gs_records:
        names = r.get("layer_names") or r.get("grad_sim_log", {}).get("layer_names", [])
        if names:
            layer_names = names
            break

    gs_groups = _group_gs(gs_records)
    out: dict[str, dict[str, tuple]] = {}

    for sched, records in gs_groups.items():
        layer_data: dict[str, tuple] = {}
        for layer in layer_names:
            runs_with_layer = [
                r for r in records
                if r["grad_sim_log"].get("per_layer", {}).get(layer)
            ]
            if not runs_with_layer:
                continue
            steps_list = [np.array(r["grad_sim_log"]["step"]) for r in runs_with_layer]
            vals_list = [np.array(r["grad_sim_log"]["per_layer"][layer])
                         for r in runs_with_layer]
            steps_ref = steps_list[0]
            interp_vals = []
            for s, v in zip(steps_list, vals_list):
                if len(s) > 1 and len(v) == len(s):
                    interp_vals.append(np.interp(steps_ref, s, v))
            if interp_vals:
                layer_data[layer] = (steps_ref, np.array(interp_vals))
        if layer_data:
            out[sched] = layer_data

    return layer_names, out


def grad_cosine_per_layer_overlay(pdir, cfg, gs_records):
    """One chart per schedule: all layers overlaid as lines over time."""
    bcfg = {**cfg.get("base_cfg", cfg), "_burst_mode": cfg.get("burst_mode", MODE_CURRENT)}
    U = bcfg["reversion_steps"]
    layer_names, sched_layer_data = _load_per_layer_data(gs_records)
    if not layer_names or not sched_layer_data:
        return []

    cmap = plt.get_cmap("tab20")
    layer_colors = {ln: cmap(i / max(len(layer_names) - 1, 1))
                    for i, ln in enumerate(layer_names)}

    paths = []
    for sched, layer_data in sched_layer_data.items():
        T_s = _T_for(sched, bcfg)
        fig, ax = plt.subplots(figsize=(14, 7))
        for layer in layer_names:
            if layer not in layer_data:
                continue
            steps_ref, vals_arr = layer_data[layer]
            m = np.mean(vals_arr, axis=0)
            n_s = len(vals_arr)
            ci = 1.96 * np.std(vals_arr, axis=0) / np.sqrt(n_s) if n_s > 1 else np.std(vals_arr, axis=0)
            c = layer_colors[layer]
            ax.plot(steps_ref, m, color=c, lw=1.8, label=layer)
            ax.fill_between(steps_ref, m - ci, m + ci, color=c, alpha=0.1)
        ax.axvline(T_s, color="black", ls="--", lw=2, alpha=0.6)
        ax.axhline(0, color="gray", ls=":", lw=1, alpha=0.5)
        ax.set_xlim(0, T_s + U)
        sched_label = SCHED_SHORT.get(sched, sched)
        _style(ax, "Step", "Cosine Similarity",
               f"{sched_label}: Per-Layer Gradient Cosine Similarity\n(Special vs Other, mean +/- 95% CI)")
        ax.legend(fontsize=8, loc="best", framealpha=0.9, ncol=2)
        fig.tight_layout()
        p_ = pdir / f"layer_cossim_overlay_{sched}.png"
        fig.savefig(p_, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(p_)
    return paths


def grad_cosine_per_layer_all_scheds(pdir, cfg, gs_records):
    """One chart per layer: all schedules overlaid — easy cross-schedule comparison."""
    bcfg = {**cfg.get("base_cfg", cfg), "_burst_mode": cfg.get("burst_mode", MODE_CURRENT)}
    U = bcfg["reversion_steps"]
    layer_names, sched_layer_data = _load_per_layer_data(gs_records)
    if not layer_names or not sched_layer_data:
        return []

    scheds = _ordered(sched_layer_data.keys())
    T_max = max(_T_for(s, bcfg) for s in scheds)
    paths = []
    for layer in layer_names:
        fig, ax = plt.subplots(figsize=(14, 6))
        for sched in scheds:
            if sched not in sched_layer_data or layer not in sched_layer_data[sched]:
                continue
            steps_ref, vals_arr = sched_layer_data[sched][layer]
            m = np.mean(vals_arr, axis=0)
            n_s = len(vals_arr)
            ci = 1.96 * np.std(vals_arr, axis=0) / np.sqrt(n_s) if n_s > 1 else np.std(vals_arr, axis=0)
            c = PALETTE.get(sched, "gray")
            ax.plot(steps_ref, m, color=c, lw=2, label=SCHED_SHORT.get(sched, sched))
            ax.fill_between(steps_ref, m - ci, m + ci, color=c, alpha=0.12)
        ax.axvline(T_max, color="black", ls="--", lw=2, alpha=0.6)
        ax.axhline(0, color="gray", ls=":", lw=1, alpha=0.5)
        ax.set_xlim(0, T_max + U)
        _style(ax, "Step", "Cosine Similarity",
               f"Layer {layer}: Gradient Cosine Similarity — All Schedules\n(Special vs Other, mean +/- 95% CI)")
        ax.legend(fontsize=10, loc="best", framealpha=0.9)
        fig.tight_layout()
        p_ = pdir / f"layer_cossim_all_scheds_{layer}.png"
        fig.savefig(p_, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(p_)
    return paths


def grad_cosine_layer_step_heatmap(pdir, cfg, gs_records):
    """Heatmap: rows=layers, cols=steps, one chart per schedule.

    Shows how cossim evolves over time for every layer simultaneously.
    """
    bcfg = {**cfg.get("base_cfg", cfg), "_burst_mode": cfg.get("burst_mode", MODE_CURRENT)}
    layer_names, sched_layer_data = _load_per_layer_data(gs_records)
    if not layer_names or not sched_layer_data:
        return []

    paths = []
    for sched, layer_data in sched_layer_data.items():
        T_s = _T_for(sched, bcfg)
        layers_present = [ln for ln in layer_names if ln in layer_data]
        if not layers_present:
            continue
        steps_ref = layer_data[layers_present[0]][0]
        n_layers = len(layers_present)
        n_steps = len(steps_ref)

        grid = np.full((n_layers, n_steps), np.nan)
        for ri, layer in enumerate(layers_present):
            steps_l, vals_arr = layer_data[layer]
            m = np.mean(vals_arr, axis=0)
            grid[ri] = np.interp(steps_ref, steps_l, m)

        fig_w = max(14, n_steps * 0.15)
        fig_h = max(4, n_layers * 0.55)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0,
                       interpolation="nearest")

        burst_col = np.searchsorted(steps_ref, T_s)
        if 0 < burst_col < n_steps:
            ax.axvline(burst_col - 0.5, color="black", lw=2, alpha=0.8)

        tick_stride = max(1, n_steps // 12)
        tick_idxs = list(range(0, n_steps, tick_stride))
        ax.set_xticks(tick_idxs)
        ax.set_xticklabels([str(int(steps_ref[i])) for i in tick_idxs],
                           rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels(layers_present, fontsize=9)
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel("Layer", fontsize=11)
        sched_label = SCHED_SHORT.get(sched, sched)
        ax.set_title(f"{sched_label}: Per-Layer Gradient Cosine Similarity Over Time\n"
                     f"(Special vs Other, mean across seeds)",
                     fontsize=12, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="Cosine Similarity")
        fig.tight_layout()
        p_ = pdir / f"layer_cossim_heatmap_{sched}.png"
        fig.savefig(p_, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(p_)
    return paths


def grad_cosine_layer_schedule_heatmap(pdir, cfg, gs_records):
    """Heatmap: rows=layers, cols=schedules, at key training phases.

    Best for comparing which layers differ most across schedules.
    """
    bcfg = {**cfg.get("base_cfg", cfg), "_burst_mode": cfg.get("burst_mode", MODE_CURRENT)}
    U = bcfg["reversion_steps"]
    layer_names, sched_layer_data = _load_per_layer_data(gs_records)
    if not layer_names or not sched_layer_data:
        return []

    scheds = _ordered(sched_layer_data.keys())
    phase_labels = ["End-Burst", "End-Rev"]

    paths = []
    for phase_label in phase_labels:
        grid = np.full((len(layer_names), len(scheds)), np.nan)
        for ci_idx, sched in enumerate(scheds):
            if sched not in sched_layer_data:
                continue
            T_s = _T_for(sched, bcfg)
            if phase_label == "End-Burst":
                lo, hi = int(3 * T_s / 4), T_s
            else:
                lo, hi = T_s + int(U / 2), T_s + U
            for ri, layer in enumerate(layer_names):
                if layer not in sched_layer_data[sched]:
                    continue
                steps_ref, vals_arr = sched_layer_data[sched][layer]
                mask = (steps_ref >= lo) & (steps_ref < hi)
                if mask.any():
                    grid[ri, ci_idx] = np.mean(vals_arr[:, mask])

        fig_w = max(6, len(scheds) * 1.3)
        fig_h = max(4, len(layer_names) * 0.55)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0,
                       interpolation="nearest")
        ax.set_xticks(range(len(scheds)))
        ax.set_xticklabels([SCHED_SHORT.get(s, s) for s in scheds],
                           rotation=30, ha="right", fontsize=10)
        ax.set_yticks(range(len(layer_names)))
        ax.set_yticklabels(layer_names, fontsize=9)
        ax.set_xlabel("Schedule", fontsize=11)
        ax.set_ylabel("Layer", fontsize=11)
        ax.set_title(f"Layer x Schedule: Gradient Cosine Similarity ({phase_label})\n"
                     f"(Special vs Other, mean across seeds & steps in window)",
                     fontsize=12, fontweight="bold")
        for ri in range(len(layer_names)):
            for ci_idx in range(len(scheds)):
                val = grid[ri, ci_idx]
                if not np.isnan(val):
                    txt_color = "white" if abs(val) > 0.55 else "black"
                    ax.text(ci_idx, ri, f"{val:.2f}", ha="center", va="center",
                            fontsize=8, fontweight="bold", color=txt_color)
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="Cosine Similarity")
        fig.tight_layout()
        p_ = pdir / f"layer_cossim_layer_sched_{phase_label.lower().replace('-', '_')}.png"
        fig.savefig(p_, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(p_)
    return paths


def grad_cosine_layer_change_heatmap(pdir, cfg, gs_records):
    """Heatmap: rows=layers, cols=steps, showing rate-of-change of cossim.

    Highlights where and when gradient alignment shifts fastest per layer.
    """
    bcfg = {**cfg.get("base_cfg", cfg), "_burst_mode": cfg.get("burst_mode", MODE_CURRENT)}
    layer_names, sched_layer_data = _load_per_layer_data(gs_records)
    if not layer_names or not sched_layer_data:
        return []

    paths = []
    for sched, layer_data in sched_layer_data.items():
        T_s = _T_for(sched, bcfg)
        layers_present = [ln for ln in layer_names if ln in layer_data]
        if not layers_present:
            continue
        steps_ref = layer_data[layers_present[0]][0]
        if len(steps_ref) < 3:
            continue
        n_layers = len(layers_present)
        n_steps = len(steps_ref) - 1

        rate_grid = np.full((n_layers, n_steps), np.nan)
        for ri, layer in enumerate(layers_present):
            steps_l, vals_arr = layer_data[layer]
            m = np.mean(vals_arr, axis=0)
            interp_m = np.interp(steps_ref, steps_l, m)
            dt = np.diff(steps_ref)
            rate_grid[ri] = np.diff(interp_m) / np.maximum(dt, 1)

        mid_steps = (steps_ref[:-1] + steps_ref[1:]) / 2
        vmax = np.nanpercentile(np.abs(rate_grid), 95)
        vmax = max(vmax, 1e-6)

        fig_w = max(14, n_steps * 0.15)
        fig_h = max(4, n_layers * 0.55)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(rate_grid, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       interpolation="nearest")

        burst_col = np.searchsorted(mid_steps, T_s)
        if 0 < burst_col < n_steps:
            ax.axvline(burst_col - 0.5, color="black", lw=2, alpha=0.8)

        tick_stride = max(1, n_steps // 12)
        tick_idxs = list(range(0, n_steps, tick_stride))
        ax.set_xticks(tick_idxs)
        ax.set_xticklabels([str(int(mid_steps[i])) for i in tick_idxs],
                           rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels(layers_present, fontsize=9)
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel("Layer", fontsize=11)
        sched_label = SCHED_SHORT.get(sched, sched)
        ax.set_title(f"{sched_label}: Rate of Change of Per-Layer Cosine Similarity\n"
                     f"(d(cossim)/d(step), mean across seeds)",
                     fontsize=12, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="d(cossim)/d(step)")
        fig.tight_layout()
        p_ = pdir / f"layer_cossim_change_{sched}.png"
        fig.savefig(p_, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(p_)
    return paths


def grad_cosine_layer_end_burst_bars(pdir, cfg, gs_records):
    """Grouped bar chart: one group per layer, bars per schedule — end-of-burst snapshot."""
    bcfg = {**cfg.get("base_cfg", cfg), "_burst_mode": cfg.get("burst_mode", MODE_CURRENT)}
    layer_names, sched_layer_data = _load_per_layer_data(gs_records)
    if not layer_names or not sched_layer_data:
        return None

    scheds = _ordered(sched_layer_data.keys())
    n_layers = len(layer_names)
    n_scheds = len(scheds)

    means_LS = np.full((n_layers, n_scheds), np.nan)
    for ci_idx, sched in enumerate(scheds):
        if sched not in sched_layer_data:
            continue
        T_s = _T_for(sched, bcfg)
        for ri, layer in enumerate(layer_names):
            if layer not in sched_layer_data[sched]:
                continue
            steps_ref, vals_arr = sched_layer_data[sched][layer]
            burst_mask = steps_ref <= T_s
            if burst_mask.any():
                means_LS[ri, ci_idx] = np.mean(vals_arr[:, burst_mask][:, -1])

    fig_w = max(14, n_layers * 0.9)
    fig, ax = plt.subplots(figsize=(fig_w, 7))
    w = 0.8 / n_scheds
    xs = np.arange(n_layers)
    for ci_idx, sched in enumerate(scheds):
        vals = means_LS[:, ci_idx]
        c = PALETTE.get(sched, "gray")
        ax.bar(xs + ci_idx * w - 0.4 + w / 2, vals, w,
               color=c, alpha=0.85, edgecolor="black", lw=0.4,
               label=SCHED_SHORT.get(sched, sched))
    ax.axhline(0, color="gray", ls=":", lw=1.5, alpha=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(layer_names, fontsize=9, fontweight="bold", rotation=30, ha="right")
    _style(ax, "Layer", "Cosine Similarity (end of burst)",
           "Per-Layer Gradient Cosine Similarity at End of Burst\n(Special vs Other, mean across seeds)")
    ax.legend(fontsize=9, loc="best", framealpha=0.9, ncol=min(n_scheds, 4))
    fig.tight_layout()
    p_ = pdir / "layer_cossim_end_burst_bars.png"
    fig.savefig(p_, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p_


def load_adl_data(run_dir) -> list[dict]:
    """Load ADL records from the adl/ folder."""
    rd = Path(run_dir)
    for adl_dir in [rd / "results" / "adl", rd / "adl"]:
        if adl_dir.is_dir():
            records = []
            for fp in sorted(adl_dir.glob("*.json")):
                with open(fp) as f:
                    records.append(json.load(f))
            if records:
                return records
    return []


def _group_adl(records):
    g = defaultdict(list)
    for r in records:
        g[r["schedule"]].append(r)
    return g


def adl_delta_norm_overlay(pdir, cfg, adl_records):
    """Mean delta norm (summed over layers) over training steps, one line per schedule."""
    if not adl_records:
        return None
    bcfg = {**cfg.get("base_cfg", cfg), "_burst_mode": cfg.get("burst_mode", MODE_CURRENT)}
    U = bcfg["reversion_steps"]
    groups = _group_adl(adl_records)
    scheds = _ordered(groups.keys())
    T_max = max(_T_for(s, bcfg) for s in scheds)

    fig, ax = plt.subplots(figsize=(14, 6))
    for sched in scheds:
        runs = groups[sched]
        steps_list = [np.array(r["adl_log"]["step"]) for r in runs]
        norm_list = [np.sum(r["adl_log"]["delta_norm_K"], axis=-1) for r in runs]
        steps_ref = steps_list[0]
        interp_vals = [np.interp(steps_ref, s, v) for s, v in zip(steps_list, norm_list)
                       if len(s) > 1]
        if not interp_vals:
            continue
        arr = np.array(interp_vals)
        m = arr.mean(axis=0)
        n_s = len(arr)
        ci = 1.96 * arr.std(axis=0) / np.sqrt(n_s) if n_s > 1 else arr.std(axis=0)
        ax.plot(steps_ref, m, color=PALETTE[sched], lw=2.5, label=SCHED_SHORT[sched])
        ax.fill_between(steps_ref, m - ci, m + ci, color=PALETTE[sched], alpha=0.15)
    ax.axvline(T_max, color="black", ls="--", lw=2, alpha=0.6)
    ax.set_xlim(0, T_max + U)
    _style(ax, "Step", "||delta|| (sum over layers)",
           "ADL: Activation Bias Magnitude Over Training\n(mean +/- 95% CI)")
    ax.legend(fontsize=11, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    p_ = pdir / "adl_delta_norm.png"
    fig.savefig(p_, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p_


def adl_readability_overlay(pdir, cfg, adl_records):
    """Mean readability (averaged over layers and token positions) over steps."""
    if not adl_records:
        return None
    bcfg = {**cfg.get("base_cfg", cfg), "_burst_mode": cfg.get("burst_mode", MODE_CURRENT)}
    U = bcfg["reversion_steps"]
    groups = _group_adl(adl_records)
    scheds = _ordered(groups.keys())
    T_max = max(_T_for(s, bcfg) for s in scheds)

    fig, ax = plt.subplots(figsize=(14, 6))
    for sched in scheds:
        runs = groups[sched]
        steps_list = [np.array(r["adl_log"]["step"]) for r in runs]
        read_list = [np.mean(r["adl_log"]["readability_KT"], axis=(-1, -2))
                     for r in runs]
        steps_ref = steps_list[0]
        interp_vals = [np.interp(steps_ref, s, v) for s, v in zip(steps_list, read_list)
                       if len(s) > 1]
        if not interp_vals:
            continue
        arr = np.array(interp_vals)
        m = arr.mean(axis=0)
        n_s = len(arr)
        ci = 1.96 * arr.std(axis=0) / np.sqrt(n_s) if n_s > 1 else arr.std(axis=0)
        ax.plot(steps_ref, m, color=PALETTE[sched], lw=2.5, label=SCHED_SHORT[sched])
        ax.fill_between(steps_ref, m - ci, m + ci, color=PALETTE[sched], alpha=0.15)
    ax.axvline(T_max, color="black", ls="--", lw=2, alpha=0.6)
    ax.set_xlim(0, T_max + U)
    ax.set_ylim(-0.02, None)
    _style(ax, "Step", "Burst-token readability (frac. top-10)",
           "ADL: Logit Lens Readability of Activation Bias\n(mean +/- 95% CI)")
    ax.legend(fontsize=11, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    p_ = pdir / "adl_readability.png"
    fig.savefig(p_, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p_


def adl_causal_ablation_overlay(pdir, cfg, adl_records):
    """Mean accuracy drop from ablating the delta direction, over steps."""
    if not adl_records:
        return None
    bcfg = {**cfg.get("base_cfg", cfg), "_burst_mode": cfg.get("burst_mode", MODE_CURRENT)}
    U = bcfg["reversion_steps"]
    groups = _group_adl(adl_records)
    scheds = _ordered(groups.keys())
    T_max = max(_T_for(s, bcfg) for s in scheds)

    fig, ax = plt.subplots(figsize=(14, 6))
    for sched in scheds:
        runs = groups[sched]
        steps_list = [np.array(r["adl_log"]["step"]) for r in runs]
        drop_list = [np.mean(r["adl_log"]["acc_drop_K"], axis=-1) for r in runs]
        steps_ref = steps_list[0]
        interp_vals = [np.interp(steps_ref, s, v) for s, v in zip(steps_list, drop_list)
                       if len(s) > 1]
        if not interp_vals:
            continue
        arr = np.array(interp_vals)
        m = arr.mean(axis=0)
        n_s = len(arr)
        ci = 1.96 * arr.std(axis=0) / np.sqrt(n_s) if n_s > 1 else arr.std(axis=0)
        ax.plot(steps_ref, m, color=PALETTE[sched], lw=2.5, label=SCHED_SHORT[sched])
        ax.fill_between(steps_ref, m - ci, m + ci, color=PALETTE[sched], alpha=0.15)
    ax.axvline(T_max, color="black", ls="--", lw=2, alpha=0.6)
    ax.axhline(0, color="gray", ls=":", lw=1, alpha=0.5)
    ax.set_xlim(0, T_max + U)
    _style(ax, "Step", "Accuracy drop (baseline - ablated)",
           "ADL: Causal Ablation — Accuracy Drop When delta Projected Out\n(mean +/- 95% CI)")
    ax.legend(fontsize=11, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    p_ = pdir / "adl_causal_ablation.png"
    fig.savefig(p_, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p_


def adl_end_burst_bars(pdir, cfg, adl_records):
    """Bar chart: readability and ablation drop at end-of-burst, one bar per schedule."""
    if not adl_records:
        return None
    bcfg = {**cfg.get("base_cfg", cfg), "_burst_mode": cfg.get("burst_mode", MODE_CURRENT)}
    groups = _group_adl(adl_records)
    scheds = _ordered(groups.keys())

    readability_means, readability_cis = [], []
    ablation_means, ablation_cis = [], []

    for sched in scheds:
        T_s = _T_for(sched, bcfg)
        runs = groups[sched]
        r_vals, a_vals = [], []
        for r in runs:
            steps = np.array(r["adl_log"]["step"])
            burst_mask = steps <= T_s
            if not burst_mask.any():
                continue
            last_burst_idx = np.where(burst_mask)[0][-1]
            r_vals.append(np.mean(r["adl_log"]["readability_KT"][last_burst_idx]))
            a_vals.append(np.mean(r["adl_log"]["acc_drop_K"][last_burst_idx]))
        if not r_vals:
            readability_means.append(0.0)
            readability_cis.append(0.0)
            ablation_means.append(0.0)
            ablation_cis.append(0.0)
        else:
            rv = np.array(r_vals)
            av = np.array(a_vals)
            readability_means.append(rv.mean())
            readability_cis.append(1.96 * rv.std() / np.sqrt(len(rv)) if len(rv) > 1 else 0.0)
            ablation_means.append(av.mean())
            ablation_cis.append(1.96 * av.std() / np.sqrt(len(av)) if len(av) > 1 else 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    xs = np.arange(len(scheds))
    colors = [PALETTE[s] for s in scheds]
    labels = [SCHED_SHORT[s] for s in scheds]

    axes[0].bar(xs, readability_means, yerr=readability_cis, color=colors,
                edgecolor="black", lw=0.8, capsize=5, alpha=0.85)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
    _style(axes[0], "", "Burst-token readability (frac. top-10)",
           "ADL Readability at End of Burst")

    axes[1].bar(xs, ablation_means, yerr=ablation_cis, color=colors,
                edgecolor="black", lw=0.8, capsize=5, alpha=0.85)
    axes[1].axhline(0, color="gray", ls=":", lw=1.5, alpha=0.6)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
    _style(axes[1], "", "Accuracy drop",
           "Causal Ablation Drop at End of Burst")

    fig.suptitle("ADL End-of-Burst Summary (mean +/- 95% CI)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    p_ = pdir / "adl_end_burst_bars.png"
    fig.savefig(p_, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p_


def adl_readability_vs_auc(pdir, cfg, adl_records, results):
    """Scatter: end-of-burst ADL readability vs reversion AUC (one dot per seed x schedule)."""
    if not adl_records:
        return None
    bcfg = {**cfg.get("base_cfg", cfg), "_burst_mode": cfg.get("burst_mode", MODE_CURRENT)}
    groups = _group_adl(adl_records)
    res_by_label = {r["label"]: r for r in results}

    fig, ax = plt.subplots(figsize=(9, 7))
    for sched in _ordered(groups.keys()):
        T_s = _T_for(sched, bcfg)
        runs = groups[sched]
        for r in runs:
            steps = np.array(r["adl_log"]["step"])
            burst_mask = steps <= T_s
            if not burst_mask.any():
                continue
            last_idx = np.where(burst_mask)[0][-1]
            readability = float(np.mean(r["adl_log"]["readability_KT"][last_idx]))
            parent_label = r["label"]
            if parent_label not in res_by_label:
                continue
            auc = res_by_label[parent_label].get("reversion_auc", None)
            if auc is None:
                continue
            ax.scatter(readability, auc, color=PALETTE[sched], s=60, alpha=0.75,
                       edgecolor="white", lw=0.5, label=SCHED_SHORT[sched]
                       if sched not in [s for s in groups if s < sched] else "")

    handles, labels_leg = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels_leg):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), fontsize=10, loc="best", framealpha=0.9)
    _style(ax, "ADL Readability (end of burst)", "Reversion AUC",
           "ADL Readability vs Forgetting Resistance\n(each dot = one seed x schedule)")
    fig.tight_layout()
    p_ = pdir / "adl_readability_vs_auc.png"
    fig.savefig(p_, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p_


def generate_all(run_dir, results, cfg):
    pdir = Path(run_dir) / "presentation"
    pdir.mkdir(exist_ok=True)
    cp = {}
    ns = len(set(r["seed"] for r in results))
    gr = _group(results)

    has_training_data = any(r.get("log", {}).get("step") for r in results)

    if has_training_data:
        burst_key = "acc_burst"
        other_key = "acc_other"
        auc_metric = "reversion_auc"
        peak_metric = "peak_burst"

        print("  Schedule bars...")
        cp["schedule_bars"] = schedule_bars(pdir, results, cfg)
        for al, al_suffix in [("absolute", ""), ("start", "_aligned_start"), ("end", "_aligned_end")]:
            print(f"  Special class overlay ({al})...")
            cp[f"overlay_burst{al_suffix}"] = overlay(pdir, results, cfg, burst_key,
                                      "Special Class Accuracy (free generation)",
                                      f"Special Class Accuracy\n(mean +/- 95% CI, n={ns} seeds)",
                                      f"overlay_burst{al_suffix}.png", groups=gr, align=al)
            print(f"  Other classes overlay ({al})...")
            cp[f"overlay_other{al_suffix}"] = overlay(pdir, results, cfg, other_key,
                                      "Other Classes Accuracy (free generation)",
                                      f"Other Classes Accuracy\n(mean +/- 95% CI, n={ns} seeds)",
                                      f"overlay_other{al_suffix}.png", loc="lower right", groups=gr, align=al)
            print(f"  Training loss overlay ({al})...")
            cp[f"overlay_loss{al_suffix}"] = overlay(pdir, results, cfg, "loss",
                                      "Training Loss",
                                      f"Training Loss\n(mean +/- 95% CI, n={ns} seeds)",
                                      f"overlay_loss{al_suffix}.png", loc="upper right", groups=gr, align=al)
        print("  Reversion AUC bars...")
        cp["auc_bars"] = bar_chart(pdir, results, cfg, auc_metric,
                                   "Reversion AUC (higher = slower forgetting)",
                                   "Reversion AUC by Schedule\n(mean +/- 95% CI, individual seeds shown)",
                                   "auc_bars.png", groups=gr)
        thresholds = TrainConfig().reversion_thresholds
        cp["life_bars"] = {}
        for t in thresholds:
            key = reversion_life_key(t)
            label = reversion_life_label(t)
            pct = int(t * 100)
            print(f"  {label} bars...")
            chart = bar_chart(
                pdir, results, cfg, key,
                f"{label} (reversion steps to {pct}% of peak)",
                f"{label} by Schedule\n(mean +/- 95% CI, individual seeds shown)",
                f"life_{pct}_bars.png", groups=gr,
            )
            if chart is not None:
                cp["life_bars"][t] = chart
        print("  Peak burst bars...")
        cp["peak_bars"] = bar_chart(pdir, results, cfg, peak_metric,
                                    "Peak Special Class Accuracy at End of Training",
                                    "Peak Special Class Accuracy by Schedule\n(mean +/- 95% CI, individual seeds shown)",
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
        cp["per_sched"] = per_sched(pdir, results, cfg, groups=gr, align="absolute")
        print("  Per-schedule overlays (aligned start)...")
        cp["per_sched_start"] = per_sched(pdir, results, cfg, groups=gr, align="start")
        print("  Per-schedule overlays (aligned end)...")
        cp["per_sched_end"] = per_sched(pdir, results, cfg, groups=gr, align="end")
    else:
        print("  (skipping training-data charts — no all_results.pkl)")

    gs_records = load_grad_sim_data(run_dir)
    gs_dir = pdir / "grad_cosine_sim"
    gs_dir.mkdir(exist_ok=True)

    print("  Gradient cosine similarity overlay...")
    cp["grad_cosine_overlay"] = grad_cosine_sim_overlay(gs_dir, cfg, gs_records)
    print("  Gradient cosine similarity bars (end of burst)...")
    cp["grad_cosine_bars"] = grad_cosine_sim_by_schedule(gs_dir, cfg, gs_records)
    print("  Gradient cosine per-seed traces...")
    cp["grad_cosine_per_seed"] = grad_cosine_per_seed(gs_dir, cfg, gs_records)
    print("  Gradient cosine rate of change...")
    cp["grad_cosine_rate"] = grad_cosine_rate_of_change(gs_dir, cfg, gs_records)
    print("  Gradient cosine phase comparison...")
    cp["grad_cosine_phase"] = grad_cosine_phase_comparison(gs_dir, cfg, gs_records)
    print("  Gradient cosine vs AUC scatter...")
    cp["grad_cosine_vs_auc"] = grad_cosine_vs_auc_scatter(gs_dir, cfg, gs_records, results)
    print("  Gradient cosine phase bars...")
    cp["grad_cosine_phase_bars"] = grad_cosine_mean_over_phases_bars(gs_dir, cfg, gs_records)
    print("  Pairwise gradient cosine heatmaps...")
    cp["pairwise_heatmaps"] = pairwise_grad_cosine_heatmap(gs_dir, cfg, gs_records)
    print("  Pairwise gradient cosine evolution (by metric)...")
    cp["pairwise_evo_by_metric"] = pairwise_grad_cosine_evolution_by_metric(gs_dir, cfg, gs_records)
    print("  Pairwise gradient cosine evolution (per schedule)...")
    cp["pairwise_evo_per_schedule"] = pairwise_grad_cosine_evolution_per_schedule(gs_dir, cfg, gs_records)

    layer_gs_dir = pdir / "grad_cosine_sim" / "per_layer"
    layer_gs_dir.mkdir(exist_ok=True)
    print("  Per-layer cossim overlay (per schedule)...")
    cp["layer_cossim_overlay"] = grad_cosine_per_layer_overlay(layer_gs_dir, cfg, gs_records)
    print("  Per-layer cossim all-schedules (per layer)...")
    cp["layer_cossim_all_scheds"] = grad_cosine_per_layer_all_scheds(layer_gs_dir, cfg, gs_records)
    print("  Per-layer cossim heatmap (layer x step)...")
    cp["layer_cossim_heatmap"] = grad_cosine_layer_step_heatmap(layer_gs_dir, cfg, gs_records)
    print("  Per-layer cossim layer x schedule heatmap...")
    cp["layer_cossim_layer_sched"] = grad_cosine_layer_schedule_heatmap(layer_gs_dir, cfg, gs_records)
    print("  Per-layer cossim rate-of-change heatmap...")
    cp["layer_cossim_change"] = grad_cosine_layer_change_heatmap(layer_gs_dir, cfg, gs_records)
    print("  Per-layer cossim end-of-burst bars...")
    cp["layer_cossim_end_burst_bars"] = grad_cosine_layer_end_burst_bars(layer_gs_dir, cfg, gs_records)

    probe_data, probe_meta = _load_probe_data(run_dir)
    if probe_data:
        probe_dir = pdir / "probes"
        probe_dir.mkdir(exist_ok=True)
        print("  Probe heatmaps (aggregated)...")
        cp["probe_heatmaps"] = probe_heatmap_aggregated(probe_dir, probe_data, probe_meta, cfg)
        print("  Probe dynamics (aggregated)...")
        cp["probe_dynamics"] = probe_dynamics_aggregated(probe_dir, probe_data, probe_meta, cfg)
        print("  Probe layer x schedule heatmap...")
        cp["probe_layer_schedule"] = probe_layer_schedule_heatmap(probe_dir, probe_data, probe_meta, cfg)
    else:
        cp["probe_heatmaps"] = []
        cp["probe_dynamics"] = None
        cp["probe_layer_schedule"] = []

    adl_records = load_adl_data(run_dir)
    if adl_records:
        adl_dir = pdir / "adl"
        adl_dir.mkdir(exist_ok=True)
        print("  ADL delta norm overlay...")
        cp["adl_delta_norm"] = adl_delta_norm_overlay(adl_dir, cfg, adl_records)
        print("  ADL readability overlay...")
        cp["adl_readability"] = adl_readability_overlay(adl_dir, cfg, adl_records)
        print("  ADL causal ablation overlay...")
        cp["adl_causal_ablation"] = adl_causal_ablation_overlay(adl_dir, cfg, adl_records)
        print("  ADL end-of-burst bars...")
        cp["adl_end_burst_bars"] = adl_end_burst_bars(adl_dir, cfg, adl_records)
        print("  ADL readability vs AUC scatter...")
        cp["adl_readability_vs_auc"] = adl_readability_vs_auc(adl_dir, cfg, adl_records, results)
    else:
        cp["adl_delta_norm"] = None
        cp["adl_readability"] = None
        cp["adl_causal_ablation"] = None
        cp["adl_end_burst_bars"] = None
        cp["adl_readability_vs_auc"] = None

    return cp
