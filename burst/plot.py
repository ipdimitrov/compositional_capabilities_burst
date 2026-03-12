"""Plot + PDF report for burst experiment.

Usage: python burst/plot.py data/burst_d<depth>_<run_tag>
"""
import sys, os, pickle, json, math, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, Counter
from burst._worker import n_target_for_step
from burst.train_utils import load_results, compute_lr_schedule
from burst.config import (
    EVAL_KEYS, CURVE_STYLE, SCHED_COLORS, SCHEDULE_ORDER,
    PHASE_PRE_BURST, PHASE_BURST, PHASE_REVERSION,
    ordered_schedules, sched_sort_key,
    TrainConfig, reversion_life_key, reversion_life_label,
    parse_run_config, burst_steps_for_mode, BURST_BASE_STEPS, MODE_CURRENT,
)

def _bar_label(ax, x, text):
    ax.text(x, 0.5, text, ha="center", va="center", fontsize=5,
            color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.45, lw=0))


def _schedule_bar(ax, T, U, sched, p, bs, seed, P=0):
    total = P + T + U
    fracs = np.zeros(total)
    saved_rng_state = np.random.get_state()
    for s in range(T):
        np.random.seed(seed * 10000 + s)
        fracs[P + s] = n_target_for_step(s, T, sched, p, bs) / bs
    np.random.set_state(saved_rng_state)
    ax.imshow(fracs.reshape(1, -1), aspect="auto", cmap="Blues",
              extent=[0, total, 0, 1], vmin=0, vmax=1)
    ax.set_yticks([])
    ax.set_ylabel("Burst frac", fontsize=7)
    ax.set_xlim(0, total)
    if P > 0:
        ax.axvline(P, color="black", lw=2)
    ax.axvline(P + T, color="black", lw=2)

    if P > 0:
        _bar_label(ax, P / 2, "Other: 100% | Special: 0%")

    if sched == "burst_10":
        _bar_label(ax, P + T / 2, f"Other: ~{(1-p)*100:.0f}% | Special: ~{p*100:.0f}% (random)")
        _bar_label(ax, P + T + U / 2, "Other: 100% | Special: 0%")
        return

    if sched == "ramp_up":
        burst_len = max(int(p * T), 1)
        ramp_len = min(int(2 * burst_len / 0.20), T)
        ramp_start = T - ramp_len
        if ramp_start > 0:
            _bar_label(ax, P + ramp_start / 2, "Other: 100% | Special: 0%")
        _bar_label(ax, P + (ramp_start + T) / 2, "Special: 0% -> 20% (ramp)")
        _bar_label(ax, P + T + U / 2, "Other: 100% | Special: 0%")
        return

    burst_fracs = fracs[P:P + T + U]
    burst_total = T + U
    regions, cur_val, start = [], burst_fracs[0], 0
    for i in range(1, burst_total):
        if abs(burst_fracs[i] - cur_val) > 0.01:
            regions.append((start, i, cur_val))
            cur_val, start = burst_fracs[i], i
    regions.append((start, burst_total, cur_val))

    merged = []
    for s, e, v in regions:
        if merged and abs(merged[-1][2] - v) < 0.01:
            merged[-1] = (merged[-1][0], e, v)
        else:
            merged.append((s, e, v))

    for s, e, v in merged:
        if (e - s) < burst_total * 0.03:
            continue
        b_pct = v * 100
        txt = (f"Other: {100-b_pct:.0f}% | Special: 0%" if b_pct < 0.5
               else f"Other: {100-b_pct:.0f}% | Special: {b_pct:.0f}%")
        _bar_label(ax, P + (s + e) / 2, txt)


def plot_lr_schedule(cfg, plots_dir, schedules=None, burst_mode=MODE_CURRENT):
    P = cfg.get("pre_burst_steps", 0)
    U = cfg["reversion_steps"]
    warmup = cfg["warmup_iters"]

    if schedules is None:
        schedules = list(SCHEDULE_ORDER)

    fig, ax = plt.subplots(figsize=(14, 5))
    for sched in ordered_schedules(schedules):
        T_s = burst_steps_for_mode(sched, burst_mode, BURST_BASE_STEPS)
        steps, lrs = compute_lr_schedule(cfg, pretrain_steps=P, burst_steps=T_s)
        color = SCHED_COLORS.get(sched, "#1565C0")
        ax.plot(steps, lrs, color=color, lw=2, label=sched, alpha=0.85)

    T_ref = burst_steps_for_mode(schedules[0], burst_mode, BURST_BASE_STEPS)
    ax.axvline(P, color="black", lw=1.5, ls="--", alpha=0.6)
    ax.axvline(warmup, color="gray", lw=1, ls=":", alpha=0.4)

    ylim = ax.get_ylim()
    ax.text(P * 0.5, ylim[1] * 0.95, "ALL-BUT-SPECIAL", ha="center", fontsize=8, color="gray")
    ax.text(P + T_ref * 0.5, ylim[1] * 0.95, "SPECIAL", ha="center", fontsize=9, color="gray")
    ax.text(P + T_ref + U * 0.5, ylim[1] * 0.95, "ALL-BUT-SPECIAL", ha="center", fontsize=8, color="gray")

    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule (three-phase cosine)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(plots_dir / "lr_schedule.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_run(result, plots_dir, run_cfg):
    log, sched, seed, cfg = result["log"], result["schedule"], result["seed"], result["config"]
    steps = np.array(log["step"])
    loss = np.array(log["loss"])
    T, U = cfg["total_steps"], cfg["reversion_steps"]
    P = result.get("pre_burst_steps", 0)
    bs, p = cfg["batch_size"], cfg["p_target"]
    burst_end = P + T

    _depth = run_cfg["depth"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 4, 2]})
    fig.suptitle(f"{sched}  seed={seed}  (depth-{_depth} bijection chain)",
                 fontsize=13, fontweight="bold")

    _schedule_bar(axes[0], T, U, sched, p, bs, seed, P=P)
    axes[0].set_title("Schedule (Special fraction per step)", fontsize=9)

    total = P + T + U
    ax = axes[1]
    for k, sty in CURVE_STYLE.items():
        vals = np.array(log.get(k, [0.0] * len(steps)))
        ax.plot(steps, vals, color=sty["color"], ls=sty["ls"], lw=1.5, label=sty["label"])
    if P > 0:
        ax.axvline(P, color="gray", ls="--", alpha=0.5)
    ax.axvline(burst_end, color="gray", ls="--", alpha=0.5)
    ax.set_xlim(0, total)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Free-gen Accuracy (last 6 tok)")
    ax.legend(fontsize=5, loc="lower left", ncol=2)
    ax.grid(True, alpha=0.2)

    peak = result["peak_burst"]
    drop = result.get("dropoff_abs", 0)
    drop_pct = result.get("dropoff_pct", 0)
    thresholds = TrainConfig().reversion_thresholds
    first_key = reversion_life_key(thresholds[0])
    first_val = result.get(first_key, U)
    first_str = f"{first_val:.0f}" if first_val < U else f">{U}"
    ax.text(burst_end + U * 0.5, 0.95,
            f"peak={peak:.3f}  {reversion_life_label(thresholds[0])}={first_str}  drop={drop:.3f}({drop_pct:.0f}%)",
            ha="center", fontsize=7, color="#D32F2F", fontweight="bold",
            transform=ax.get_xaxis_transform())
    for t in thresholds:
        lk = reversion_life_key(t)
        lv = result.get(lk, U)
        if lv < U:
            ax.axvline(burst_end + lv, color="#D32F2F", ls="--", lw=1, alpha=0.4)
            ax.axhline(peak * t, color="#D32F2F", ls=":", lw=0.8, alpha=0.3)

    ax = axes[2]
    ax.plot(steps, loss, color="#333", lw=1, label="loss")
    if P > 0:
        ax.axvline(P, color="gray", ls="--", alpha=0.5)
    ax.axvline(burst_end, color="gray", ls="--", alpha=0.5)
    ax.set_xlim(0, total)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Step")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    if P > 0:
        axes[1].text(P * 0.5, -0.04, "ALL-BUT-SPECIAL", ha="center", fontsize=6,
                     color="gray", transform=axes[1].get_xaxis_transform())
    axes[1].text(P + T * 0.5, -0.04, "SPECIAL", ha="center", fontsize=7,
                 color="gray", transform=axes[1].get_xaxis_transform())
    axes[1].text(burst_end + U * 0.5, -0.04, "ALL-BUT-SPECIAL", ha="center", fontsize=6,
                 color="gray", transform=axes[1].get_xaxis_transform())

    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    idx = sched_sort_key(sched)
    fname = f"{idx:02d}_run_{sched}_s{seed}.png"
    fig.savefig(plots_dir / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_summary_bars(results, plots_dir, cfg):
    bcfg = cfg.get("base_cfg", cfg)
    total_steps = bcfg["total_steps"]
    reversion_steps = bcfg["reversion_steps"]

    thresholds = TrainConfig().reversion_thresholds
    first_t = thresholds[0]
    first_key = reversion_life_key(first_t)
    first_label = reversion_life_label(first_t)

    ordered = sorted(results, key=lambda r: sched_sort_key(r["schedule"]))
    scheds = [r["schedule"] for r in ordered]
    peaks = [r["peak_burst"] for r in ordered]
    life_vals = [r.get(first_key, reversion_steps) for r in ordered]
    aucs = [r["reversion_auc"] for r in ordered]
    colors = [SCHED_COLORS.get(s, "gray") for s in scheds]
    xs = np.arange(len(scheds))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Peak Special Class Accuracy + {first_label} + AUC by Schedule",
                 fontsize=14, fontweight="bold")

    titles = [f"Peak Special Class Accuracy at step {total_steps}",
              f"{first_label} (lower = faster forgetting)",
              "Reversion AUC (lower = faster forgetting)"]
    ylabels = ["Peak Special Class accuracy", f"{first_label} (reversion steps)", "Reversion AUC"]
    data = [peaks, life_vals, aucs]

    for ax, vals, title, ylabel in zip(axes, data, titles, ylabels):
        bars = ax.bar(xs, vals, color=colors, edgecolor="black", lw=0.5)
        for b, v in zip(bars, vals):
            lbl = f"{v:.3f}" if max(vals) <= 1.5 else (f"{v:.0f}" if v < reversion_steps else f">{reversion_steps}")
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(vals) * 0.01,
                    lbl, ha="center", fontsize=7, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels(scheds, fontsize=8, rotation=25, ha="right")
        ax.grid(True, alpha=0.2, axis="y")

    axes[1].axhline(reversion_steps, color="gray", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(plots_dir / "summary_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _build_sched_groups(results):
    sched_groups = {}
    for r in results:
        s = r["schedule"]
        sched_groups.setdefault(s, []).append(r)
    return sched_groups


def plot_auc_detail(results, plots_dir, cfg, sched_groups=None):
    bcfg = cfg.get("base_cfg", cfg)
    reversion_steps = bcfg["reversion_steps"]

    if sched_groups is None:
        sched_groups = _build_sched_groups(results)
    ordered = ordered_schedules(sched_groups.keys())

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Reversion AUC by Schedule", fontsize=14, fontweight="bold")

    xs = np.arange(len(ordered))

    ax = axes[0]
    ax.set_title("Individual seed × schedule", fontsize=10, fontweight="bold")
    for xi, sched in enumerate(ordered):
        vals = [r["reversion_auc"] for r in sched_groups[sched]]
        c = SCHED_COLORS.get(sched, "gray")
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter(np.full(len(vals), xi) + jitter, vals,
                   color=c, edgecolor="black", lw=0.5, s=50, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(ordered, fontsize=9, rotation=25, ha="right")
    ax.set_ylabel("Reversion AUC")
    ax.grid(True, alpha=0.2, axis="y")

    ax = axes[1]
    ax.set_title("Mean ± 95% CI across seeds", fontsize=10, fontweight="bold")
    means, cis = [], []
    for sched in ordered:
        vals = np.array([r["reversion_auc"] for r in sched_groups[sched]])
        m = vals.mean()
        means.append(m)
        ci = 1.96 * vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else vals.std()
        cis.append(ci)
    colors = [SCHED_COLORS.get(s, "gray") for s in ordered]
    bars = ax.bar(xs, means, yerr=cis, color=colors, edgecolor="black", lw=0.5,
                  capsize=5, error_kw={"lw": 1.5})
    for b, m, ci in zip(bars, means, cis):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + ci + max(means) * 0.01,
                f"{m:.1f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(ordered, fontsize=9, rotation=25, ha="right")
    ax.set_ylabel("Reversion AUC")
    ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(plots_dir / "auc_detail.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_auc_diff_pct(results, plots_dir, cfg, sched_groups=None):
    if sched_groups is None:
        sched_groups = _build_sched_groups(results)
    ordered = ordered_schedules(sched_groups.keys())
    n = len(ordered)
    if n < 2:
        return

    mean_aucs = {}
    for sched in ordered:
        mean_aucs[sched] = np.mean([r["reversion_auc"] for r in sched_groups[sched]])

    pct_grid = np.zeros((n, n))
    for i, sa in enumerate(ordered):
        for j, sb in enumerate(ordered):
            if i == j:
                pct_grid[i, j] = 0.0
            else:
                base = mean_aucs[sb]
                pct_grid[i, j] = ((mean_aucs[sa] - base) / abs(base) * 100) if abs(base) > 1e-9 else 0.0

    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n * 1.0)))
    vmax = max(abs(pct_grid.min()), abs(pct_grid.max()), 1)
    im = ax.imshow(pct_grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")

    ax.set_xticks(range(n))
    ax.set_xticklabels(ordered, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(ordered, fontsize=10)
    ax.set_xlabel("Baseline schedule (denominator)", fontsize=11)
    ax.set_ylabel("Compared schedule (numerator)", fontsize=11)
    ax.set_title("Pairwise Reversion AUC Difference %\n(row − col) / |col| × 100",
                 fontsize=13, fontweight="bold")

    for i in range(n):
        for j in range(n):
            val = pct_grid[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:+.1f}%", ha="center", va="center", fontsize=9, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03, label="% difference")
    fig.tight_layout()
    fig.savefig(plots_dir / "auc_diff_pct.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _build_sched_data(results):
    sched_data = defaultdict(lambda: defaultdict(list))
    for r in results:
        sched = r["schedule"]
        steps = np.array(r["log"]["step"])
        for k in list(EVAL_KEYS) + ["loss", "loss_other", "loss_burst"]:
            vals = np.array(r["log"].get(k, [float("nan")] * len(steps)))
            sched_data[sched][k].append((steps, vals))
    return sched_data


def _get_P(results):
    for r in results:
        P = r.get("pre_burst_steps", 0)
        if P > 0:
            return P
    return 0


def plot_overlay_per_schedule(results, plots_dir, sched_data=None):
    if not results:
        return
    U_ov = results[0]["config"]["reversion_steps"]
    P = _get_P(results)

    if sched_data is None:
        sched_data = _build_sched_data(results)

    sched_Ts = {}
    for r in results:
        sched_Ts.setdefault(r["schedule"], r["config"]["total_steps"])

    for align in ["absolute", "start", "end"]:
        for sched in ordered_schedules(sched_data.keys()):
            T_s = sched_Ts[sched]
            burst_end = P + T_s
            fig, ax = plt.subplots(figsize=(14, 8))
            fig.suptitle(f"{sched} - All Metrics (mean +/- 95% CI across seeds)",
                         fontsize=14, fontweight="bold")

            for k in EVAL_KEYS:
                runs = sched_data[sched][k]
                if len(runs) == 0:
                    continue

                steps_ref = runs[0][0]
                all_vals = np.array([vals for _, vals in runs])
                mean_vals = np.mean(all_vals, axis=0)
                std_vals = np.std(all_vals, axis=0)
                n_seeds = len(runs)
                ci = 1.96 * std_vals / np.sqrt(n_seeds) if n_seeds > 1 else std_vals

                if align == "end":
                    x = steps_ref - burst_end
                else:
                    x = steps_ref

                sty = CURVE_STYLE[k]
                ax.plot(x, mean_vals, color=sty["color"], ls=sty["ls"],
                       lw=2.5, label=sty["label"])
                ax.fill_between(x, mean_vals - ci, mean_vals + ci,
                               color=sty["color"], alpha=0.25)

            total = P + T_s + U_ov
            if align == "end":
                if P > 0:
                    ax.axvline(-burst_end, color="gray", ls="--", alpha=0.6, lw=2)
                    ax.axvline(-T_s, color="gray", ls="--", alpha=0.6, lw=2)
                    ax.text(-burst_end + P * 0.5, 0.05, "PRE", ha="center", fontsize=9,
                           color="gray", fontweight="bold")
                ax.axvline(0, color="gray", ls="--", alpha=0.6, lw=2)
                ax.text(-T_s * 0.5, 0.05, "SPECIAL", ha="center", fontsize=11,
                       color="gray", fontweight="bold")
                ax.text(U_ov * 0.5, 0.05, "ALL-BUT-SPECIAL", ha="center", fontsize=11,
                       color="gray", fontweight="bold")
                ax.set_xlim(-burst_end, U_ov)
                ax.set_xlabel("Steps from Burst End", fontsize=11)
            else:
                if P > 0:
                    ax.axvline(P, color="gray", ls="--", alpha=0.6, lw=2)
                    ax.text(P * 0.5, 0.05, "ALL-BUT-SPECIAL", ha="center", fontsize=9,
                           color="gray", fontweight="bold")
                ax.axvline(burst_end, color="gray", ls="--", alpha=0.6, lw=2)
                ax.text(P + T_s * 0.5, 0.05, "SPECIAL", ha="center", fontsize=11,
                       color="gray", fontweight="bold")
                ax.text(burst_end + U_ov * 0.5, 0.05, "ALL-BUT-SPECIAL", ha="center", fontsize=10,
                       color="gray", fontweight="bold")
                ax.set_xlim(0, total)
                ax.set_xlabel("Steps from Burst Start" if align == "start" else "Step", fontsize=11)

            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel("Free-gen Accuracy (last 6 tok)", fontsize=11)
            ax.legend(fontsize=9, loc="best", framealpha=0.9)
            ax.grid(True, alpha=0.3)

            fig.tight_layout()
            idx = sched_sort_key(sched)
            suffix = f"_{align}" if align != "absolute" else ""
            fig.savefig(plots_dir / f"{idx:02d}_overlay_{sched}{suffix}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)


def plot_overlay_all_schedules(results, plots_dir, sched_data=None):
    U_ov = results[0]["config"]["reversion_steps"]
    P = _get_P(results)

    if sched_data is None:
        sched_data = _build_sched_data(results)

    sched_Ts = {}
    for r in results:
        sched_Ts.setdefault(r["schedule"], r["config"]["total_steps"])
    T_max = max(sched_Ts.values())
    burst_end_max = P + T_max

    def _overlay_plot(ax, sched_data, key, sched_Ts, P, T_max, U_ov, burst_end_max, align):
        for sched in ordered_schedules(sched_data.keys()):
            c = SCHED_COLORS.get(sched, "gray")
            T_s = sched_Ts[sched]
            burst_end_s = P + T_s
            runs = sched_data[sched][key]
            if len(runs) == 0:
                continue
            steps_ref = runs[0][0]
            all_vals = np.array([vals for _, vals in runs])
            mean_vals = np.nanmean(all_vals, axis=0)
            std_vals = np.nanstd(all_vals, axis=0)
            n_seeds = len(runs)
            ci = 1.96 * std_vals / np.sqrt(n_seeds) if n_seeds > 1 else std_vals
            x = steps_ref - (P + T_s) if align == "end" else steps_ref
            ax.plot(x, mean_vals, color=c, lw=2.5, label=sched)
            ax.fill_between(x, mean_vals - ci, mean_vals + ci, color=c, alpha=0.2)
        total = P + T_max + U_ov
        if align == "end":
            ax.axvline(0, color="gray", ls="--", alpha=0.6, lw=2)
            if P > 0:
                ax.text(-burst_end_max + P * 0.5, ax.get_ylim()[0] * 0.9, "PRE",
                        ha="center", fontsize=10, color="gray", fontweight="bold")
            ax.text(-T_max * 0.3, ax.get_ylim()[0] * 0.9, "SPECIAL",
                    ha="center", fontsize=12, color="gray", fontweight="bold")
            ax.text(U_ov * 0.5, ax.get_ylim()[0] * 0.9, "ALL-BUT-SPECIAL",
                    ha="center", fontsize=12, color="gray", fontweight="bold")
            ax.set_xlim(-burst_end_max, U_ov)
            ax.set_xlabel("Steps from Burst End", fontsize=13)
        else:
            if P > 0:
                ax.axvline(P, color="gray", ls="--", alpha=0.6, lw=2)
                ax.text(P * 0.5, ax.get_ylim()[0] * 0.9, "ALL-BUT-SPECIAL",
                        ha="center", fontsize=10, color="gray", fontweight="bold")
            ax.text(P + T_max * 0.5, ax.get_ylim()[0] * 0.9, "SPECIAL",
                    ha="center", fontsize=12, color="gray", fontweight="bold")
            ax.text(burst_end_max + U_ov * 0.5, ax.get_ylim()[0] * 0.9, "ALL-BUT-SPECIAL",
                    ha="center", fontsize=11, color="gray", fontweight="bold")
            ax.set_xlim(0, total)
            ax.set_xlabel("Steps from Burst Start" if align == "start" else "Step", fontsize=13)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(dict(zip(labels, handles)).values(),
                  dict(zip(labels, handles)).keys(), fontsize=10, loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)

    for align in ["absolute", "start", "end"]:
        for ki, k in enumerate(EVAL_KEYS):
            fig, ax = plt.subplots(figsize=(11.7, 8.3))
            fig.suptitle(f"All Schedules - {CURVE_STYLE[k]['label']}\n(mean +/- 95% CI across seeds)",
                         fontsize=16, fontweight="bold")
            _overlay_plot(ax, sched_data, k, sched_Ts, P, T_max, U_ov, burst_end_max, align)
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel("Accuracy", fontsize=13)
            fig.tight_layout()
            suffix = f"_{align}" if align != "absolute" else ""
            fig.savefig(plots_dir / f"overlay_all_{k}{suffix}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        loss_keys = [
            ("loss_other", "Other Class Eval Loss"),
            ("loss_burst", "Special Class Eval Loss"),
        ]
        has_per_class_loss = any(
            not all(np.isnan(v) for _, vals in sched_data[s].get("loss_other", [])
                    for v in vals)
            for s in sched_data
        )
        if not has_per_class_loss:
            loss_keys = [("loss", "Training Loss")]

        for loss_key, loss_label in loss_keys:
            fig, ax = plt.subplots(figsize=(11.7, 8.3))
            fig.suptitle(f"All Schedules - {loss_label}\n(mean +/- 95% CI across seeds)",
                         fontsize=16, fontweight="bold")
            _overlay_plot(ax, sched_data, loss_key, sched_Ts, P, T_max, U_ov, burst_end_max, align)
            ax.set_ylabel("Cross-Entropy Loss", fontsize=13)
            fig.tight_layout()
            suffix = f"_{align}" if align != "absolute" else ""
            fig.savefig(plots_dir / f"overlay_all_{loss_key}{suffix}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)


def _fname_has_low_idx(fname: str, max_idx: int = 8) -> bool:
    """Return True if filename starts with a numeric index <= max_idx (e.g. '00_', '08_')."""
    parts = fname.split("_", 1)
    if parts[0].isdigit():
        return int(parts[0]) <= max_idx
    return False


def _img_tag(path, width="100%") -> str:
    """Return an <img> tag with base64-encoded PNG, or empty string if file missing."""
    import base64
    p = Path(path)
    if not p.exists():
        return ""
    data = base64.b64encode(p.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" style="width:{width};max-width:1200px;display:block;margin:12px auto;">'


def make_report(run_dir, results, cfg, per_run_fnames):
    _, _, plots_dir = _resolve_dirs(run_dir)
    rc = parse_run_config(cfg)
    bcfg, depth, burst_pos, n_a = rc["base_cfg"], rc["depth"], rc["burst_pos"], rc["n_a"]

    n_layer = bcfg['n_layer']
    n_embd = bcfg['n_embd']
    n_head = bcfg['n_head']
    total_steps = bcfg['total_steps']
    reversion_steps = bcfg['reversion_steps']
    batch_size = bcfg['batch_size']
    p_target = bcfg['p_target']
    pre_burst_steps = bcfg.get("pre_burst_steps", 0)
    thresholds = TrainConfig().reversion_thresholds
    first_key = reversion_life_key(thresholds[0])
    first_label_short = reversion_life_label(thresholds[0])
    warmup_iters = bcfg['warmup_iters']
    lr_max = bcfg['lr']
    lr_pe = bcfg.get('lr_pretrain_end_frac', 0.3)
    lr_be = bcfg.get('lr_burst_end_frac', 0.1)
    lr_re = bcfg.get('lr_reversion_end_frac', 0.01)
    burst_len = max(int(p_target * total_steps), 1)
    n_a_comps = n_a ** depth
    n_burst = n_a ** (depth - 1)

    def _section(title, body=""):
        h = f'<div class="section"><h2>{title}</h2>'
        if body:
            h += f'<p>{body}</p>'
        return h

    def _close():
        return "</div>"

    def _chart(path, caption=""):
        tag = _img_tag(path)
        if not tag:
            return ""
        cap = f'<p class="caption">{caption}</p>' if caption else ""
        return f'<div class="chart">{tag}{cap}</div>'

    def _ranking_table():
        rows_sorted = sorted(results, key=lambda r: r.get(first_key, reversion_steps))
        rows_html = ""
        for i, r in enumerate(rows_sorted):
            lv = r.get(first_key, reversion_steps)
            lv_str = f"{lv:.0f}" if lv < reversion_steps else f">{reversion_steps}"
            rows_html += (f"<tr><td>{i+1}</td><td>{r['schedule']}</td>"
                          f"<td>{r['peak_burst']:.3f}</td><td>{lv_str}</td>"
                          f"<td>{r['reversion_auc']:.0f}</td></tr>")
        return (f"<table><thead><tr><th>Rank</th><th>Schedule</th>"
                f"<th>Peak Special</th><th>{first_label_short}</th><th>Rev AUC</th></tr></thead>"
                f"<tbody>{rows_html}</tbody></table>")

    parts = ["""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Analysis Report</title>
<style>
  body { font-family: Arial, sans-serif; margin: 0; background: #f0f2f5; color: #222; }
  .header { background: linear-gradient(135deg,#005a9e,#0078d4); color:white; padding:32px 40px; }
  .header h1 { margin:0; font-size:2em; }
  .header p { margin:8px 0 0; opacity:0.85; }
  .toc { background:white; margin:20px 40px; padding:20px 24px; border-radius:8px;
         box-shadow:0 2px 8px rgba(0,0,0,0.08); }
  .toc h3 { margin:0 0 10px; color:#005a9e; }
  .toc a { display:inline-block; margin:3px 8px 3px 0; color:#0078d4; text-decoration:none; font-size:0.9em; }
  .toc a:hover { text-decoration:underline; }
  .section { background:white; margin:20px 40px; padding:24px 28px; border-radius:8px;
             box-shadow:0 2px 8px rgba(0,0,0,0.08); }
  .section h2 { color:#005a9e; margin-top:0; border-bottom:2px solid #e0e8f0; padding-bottom:8px; }
  .section p { line-height:1.6; color:#333; }
  .section ul { line-height:1.8; color:#333; }
  .chart { margin:16px 0; text-align:center; }
  .caption { font-size:0.85em; color:#666; margin-top:4px; font-style:italic; }
  table { border-collapse:collapse; width:100%; margin:12px 0; }
  th { background:#005a9e; color:white; padding:8px 12px; text-align:left; }
  td { padding:6px 12px; border-bottom:1px solid #e0e0e0; }
  tr:nth-child(even) td { background:#f8f9fa; }
  code { background:#f4f4f4; padding:2px 6px; border-radius:3px; font-size:0.9em; }
</style>
</head>
<body>
"""]

    parts.append(f"""<div class="header">
  <h1>Depth-{depth} Bijection Composition — Burst &amp; Forgetting Experiment</h1>
  <p>Burst at position {burst_pos} &nbsp;|&nbsp; Free Generation &nbsp;|&nbsp;
     {n_layer}-layer Transformer ({n_embd}-dim, {n_head} heads) &nbsp;|&nbsp;
     {pre_burst_steps} pre-burst + {total_steps} special + {reversion_steps} all-but-special &nbsp;|&nbsp;
     batch {batch_size} &nbsp;|&nbsp; {len(results)} runs</p>
</div>
""")

    toc_items = [
        ("setup", "Experimental Setup"),
        ("lr", "Learning Rate Schedule"),
        ("summary", "Summary: Forgetting Speed"),
        ("auc_detail", "AUC Detail"),
        ("auc_diff", "Pairwise AUC Difference"),
        ("ranking", "Ranking"),
        ("acc_overlay", "Accuracy Overlays"),
        ("loss_overlay", "Loss Overlays (per class)"),
        ("per_sched", "Per-Schedule Overlays"),
        ("per_run", "Per-Run Details"),
    ]
    toc_html = '<div class="toc"><h3>Contents</h3>'
    for anchor, label in toc_items:
        toc_html += f'<a href="#{anchor}">{label}</a>'
    toc_html += "</div>"
    parts.append(toc_html)

    parts.append(f'<div class="section" id="setup"><h2>Experimental Setup</h2>')
    parts.append(f"""<p><strong>The Task:</strong> The model learns to apply chains of {depth} functions to a
sequence of numbers. Each function is a bijection — a lookup table that remaps each digit (0–9) to a
different digit. Every sequence has the same format: {depth} function slots followed by the input,
then the result after each function.</p>
<p><strong>Training Data (Other Classes):</strong> {n_a} bijection functions.
The model trains on all {n_a}<sup>{depth}</sup> = {n_a_comps} depth-{depth} chains.</p>
<p><strong>Special Data:</strong> One brand-new function (b*) placed at position {burst_pos}.
All {n_burst} possible combinations for the other positions are used during the burst.</p>
<p><strong>Protocol:</strong>
Pre-burst ({pre_burst_steps} steps): all-but-special only (shared checkpoint).
Special ({total_steps} steps): other + special mixed per schedule.
All-but-special ({reversion_steps} steps): special removed, other only.</p>
<p><strong>Metrics:</strong></p><ul>
<li>Other Classes: compositional accuracy on known functions</li>
<li>Special Class: accuracy on b* chains (acquisition + retention)</li>
<li>Peak Special: b* accuracy at end of burst phase</li>""")
    for t in thresholds:
        pct = int(t * 100)
        parts.append(f"<li>{reversion_life_label(t)}: reversion steps until Special Class drops to {pct}% of peak</li>")
    parts.append("""<li>Reversion AUC: area under Special Class curve during reversion (lower = faster forgetting)</li>
</ul></div>""")

    parts.append(f'<div class="section" id="lr"><h2>Learning Rate Schedule</h2>')
    parts.append(f"""<p>Three-phase cosine schedule. Linear warmup for {warmup_iters} steps to {lr_max:.0e},
then cosine decay to {lr_max*lr_pe:.0e} over pretrain ({pre_burst_steps} steps),
to {lr_max*lr_be:.0e} over burst, and to {lr_max*lr_re:.0e} over reversion ({reversion_steps} steps).
Burst phase length varies per schedule.</p>""")
    parts.append(_chart(plots_dir / "lr_schedule.png"))
    parts.append("</div>")

    parts.append(f'<div class="section" id="summary"><h2>Summary: Forgetting Speed by Schedule</h2>')
    parts.append("<p>Left: Peak Special Class accuracy. Center: Quarter-life (lower = faster forgetting). "
                 "Right: Reversion AUC. Mixed schedules retain the special class longer.</p>")
    parts.append(_chart(plots_dir / "summary_bars.png"))
    parts.append("</div>")

    parts.append(f'<div class="section" id="auc_detail"><h2>AUC Detail: Individual Seeds + Mean ± CI</h2>')
    parts.append("<p>Left: each dot is one seed. Right: mean reversion AUC with 95% CI. "
                 "Lower AUC = faster forgetting.</p>")
    parts.append(_chart(plots_dir / "auc_detail.png"))
    parts.append("</div>")

    parts.append(f'<div class="section" id="auc_diff"><h2>Pairwise Reversion AUC Difference (%)</h2>')
    parts.append("<p>Each cell: (row_AUC − col_AUC) / |col_AUC| × 100. "
                 "Red = row schedule has higher AUC (slower forgetting). Blue = faster forgetting.</p>")
    parts.append(_chart(plots_dir / "auc_diff_pct.png"))
    parts.append("</div>")

    parts.append(f'<div class="section" id="ranking"><h2>Ranking: Fastest Forgetting First</h2>')
    parts.append(_ranking_table())
    parts.append("</div>")

    parts.append(f'<div class="section" id="acc_overlay"><h2>Accuracy Overlays — All Schedules</h2>')
    parts.append("<p>Lines show mean accuracy across seeds with 95% CI ribbons. "
                 "Vertical dashed line marks the start of the reversion phase.</p>")
    for k in EVAL_KEYS:
        for suffix, label in [("", "Absolute"), ("_start", "Burst-Start Aligned"),
                               ("_end", "Burst-End Aligned")]:
            p = plots_dir / f"overlay_all_{k}{suffix}.png"
            parts.append(_chart(p, f"{CURVE_STYLE[k]['label']} — {label}"))
    parts.append("</div>")

    has_per_class_loss = (plots_dir / "overlay_all_loss_other.png").exists()
    loss_keys_labels = (
        [("loss_other", "Other Class Eval Loss"), ("loss_burst", "Special Class Eval Loss")]
        if has_per_class_loss
        else [("loss", "Training Loss")]
    )
    parts.append(f'<div class="section" id="loss_overlay"><h2>Loss Overlays — All Schedules</h2>')
    parts.append("<p>Eval loss per class across all schedules with 95% CI ribbons.</p>")
    for loss_key, loss_label in loss_keys_labels:
        for suffix, align_label in [("", "Absolute"), ("_start", "Burst-Start Aligned"),
                                    ("_end", "Burst-End Aligned")]:
            p = plots_dir / f"overlay_all_{loss_key}{suffix}.png"
            parts.append(_chart(p, f"{loss_label} — {align_label}"))
    parts.append("</div>")

    parts.append(f'<div class="section" id="per_sched"><h2>Per-Schedule Accuracy Overlays</h2>')
    for sched in ordered_schedules(set(r["schedule"] for r in results)):
        idx = sched_sort_key(sched)
        if idx <= 8:
            continue
        for suffix in ["", "_start", "_end"]:
            overlay_path = plots_dir / f"{idx:02d}_overlay_{sched}{suffix}.png"
            parts.append(_chart(overlay_path, f"Schedule: {sched}"))
    parts.append("</div>")

    parts.append(f'<div class="section" id="per_run"><h2>Per-Run Details</h2>')
    parts.append("<p>Each plot: schedule bar with Other/Special percentages, "
                 "accuracy curves with metrics, training loss.</p>")
    for fname in sorted(f for f in per_run_fnames if not _fname_has_low_idx(f)):
        parts.append(_chart(plots_dir / fname))
    parts.append("</div>")

    parts.append("</body></html>")

    results_dir = run_dir / "results"
    out_html = (results_dir / "analysis_report.html") if results_dir.exists() else (run_dir / "analysis_report.html")
    out_html.write_text("".join(parts), encoding="utf-8")
    print(f"  Saved {out_html}")


def plot_task_distributions(run_dir):
    run_dir = Path(run_dir)
    logs_dir = run_dir / "logs"
    stats_dir = (logs_dir / "task_distributions") if (logs_dir / "task_distributions").exists() \
        else (run_dir / "task_distributions")

    if not stats_dir.exists():
        print("  No task_distributions folder found, skipping...")
        return []

    skip = {"all_distributions_combined.csv", "summary_statistics.csv"}
    csv_files = [f for f in stats_dir.glob("*.csv") if f.name not in skip]
    if not csv_files:
        print("  No CSV files found in task_distributions, skipping...")
        return []

    all_data = []
    for csv_file in csv_files:
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["count"] = int(row["count"])
                for col in list(row.keys()):
                    if col.startswith("f") and col[1:].isdigit():
                        row[col] = int(row[col])
                all_data.append(row)

    if not all_data:
        return []

    combined_csv = stats_dir / "all_distributions_combined.csv"
    with open(combined_csv, "w", newline="") as f:
        if all_data:
            writer = csv.DictWriter(f, fieldnames=all_data[0].keys())
            writer.writeheader()
            writer.writerows(all_data)
    print(f"  Saved combined CSV: {combined_csv}")

    schedules = ordered_schedules(set(row["schedule"] for row in all_data))
    phases = sorted(set(row["phase"] for row in all_data))

    summary_rows = []
    for schedule in schedules:
        for phase in phases:
            filtered = [row for row in all_data
                       if row["schedule"] == schedule and row["phase"] == phase]
            if not filtered:
                continue

            type_counts = {}
            compositions = set()
            fn_cols = sorted([c for c in filtered[0].keys()
                              if c.startswith("f") and c[1:].isdigit()],
                             key=lambda c: int(c[1:]), reverse=True)
            fn_val_sets = {col: set() for col in fn_cols}

            for row in filtered:
                tt = row["task_type"]
                type_counts[tt] = type_counts.get(tt, 0) + row["count"]
                compositions.add(row["composition"])
                for col in fn_cols:
                    if col in row:
                        fn_val_sets[col].add(row[col])

            total = sum(type_counts.values())
            other_count = type_counts.get("other", 0)
            burst_count = type_counts.get("burst", 0)

            row_out = {
                "schedule": schedule,
                "phase": phase,
                "total_samples": total,
                "other_samples": other_count,
                "burst_samples": burst_count,
                "other_fraction": other_count / total if total > 0 else 0,
                "burst_fraction": burst_count / total if total > 0 else 0,
                "unique_compositions": len(compositions),
            }
            for col in fn_cols:
                row_out[f"unique_{col}"] = len(fn_val_sets[col])
            summary_rows.append(row_out)

    summary_csv = stats_dir / "summary_statistics.csv"
    with open(summary_csv, "w", newline="") as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"  Saved summary statistics: {summary_csv}")

    charts_dir = stats_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    generated_files = []

    seeds = sorted(set(row["seed"] for row in all_data))

    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in all_data:
        grouped[row["schedule"]][row["seed"]][row["phase"]].append(row)

    for schedule in schedules:
        for seed in seeds:
            if seed not in grouped[schedule]:
                continue

            label = f"{schedule}_s{seed}"

            for phase in phases:
                phase_data = grouped[schedule][seed].get(phase, [])
                if not phase_data:
                    continue

                fn_cols = sorted([c for c in phase_data[0].keys()
                                  if c.startswith("f") and c[1:].isdigit()],
                                 key=lambda c: int(c[1:]), reverse=True)
                n_fn = len(fn_cols)
                n_rows = 1 + max(1, (n_fn + 2) // 3)
                fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
                fig.suptitle(f"{label} - {phase}", fontsize=16, fontweight="bold")

                type_counts = {}
                for row in phase_data:
                    tt = row["task_type"]
                    type_counts[tt] = type_counts.get(tt, 0) + row["count"]

                types = sorted(type_counts.keys())
                colors = ["#2196F3" if t == "other" else "#E91E63" for t in types]
                axes[0, 0].bar(types, [type_counts[t] for t in types], color=colors)
                axes[0, 0].set_title("Other Classes vs Special Class")
                axes[0, 0].set_xlabel("Task Type")
                axes[0, 0].set_ylabel("Count")
                for i, t in enumerate(types):
                    axes[0, 0].text(i, type_counts[t], str(type_counts[t]),
                                   ha="center", va="bottom")

                fn_colors = ["#4CAF50", "#FF9800", "#9C27B0", "#00BCD4", "#795548"]
                for fi, col in enumerate(fn_cols):
                    ax_r, ax_c = divmod(fi + 1, 3)
                    if ax_r >= n_rows:
                        break
                    ax = axes[ax_r, ax_c]
                    counts = {}
                    for row in phase_data:
                        if col in row:
                            counts[row[col]] = counts.get(row[col], 0) + row["count"]
                    sorted_items = sorted(counts.items())
                    ax.bar([str(f) for f, _ in sorted_items],
                           [c for _, c in sorted_items],
                           color=fn_colors[fi % len(fn_colors)])
                    pos_num = col[1:]
                    ax.set_title(f"Distribution by {col.upper()} (position {pos_num})")
                    ax.set_xlabel("Function ID")
                    ax.set_ylabel("Count")
                    ax.tick_params(axis='x', rotation=45)

                comp_ax = axes[-1, 1] if n_rows > 1 else axes[0, 1]
                comp_counts = {}
                for row in phase_data:
                    comp_counts[row["composition"]] = comp_counts.get(row["composition"], 0) + row["count"]
                top_comps = sorted(comp_counts.items(), key=lambda x: x[1], reverse=True)[:20]
                comp_ax.barh(range(len(top_comps)), [c for _, c in top_comps], color="#00BCD4")
                comp_ax.set_yticks(range(len(top_comps)))
                comp_ax.set_yticklabels([comp for comp, _ in top_comps], fontsize=8)
                comp_ax.set_title("Top 20 Compositions")
                comp_ax.set_xlabel("Count")
                comp_ax.invert_yaxis()

                combo_ax = axes[-1, 2] if n_rows > 1 else axes[0, 2]
                combo_counts = {}
                for row in phase_data:
                    fn_str = ",".join(str(row.get(c, "?")) for c in fn_cols)
                    key = (row["task_type"], f"({fn_str})")
                    combo_counts[key] = combo_counts.get(key, 0) + row["count"]
                top_combos = sorted(combo_counts.items(), key=lambda x: x[1], reverse=True)[:15]

                labels = [f"{tt}: {combo}" for (tt, combo), _ in top_combos]
                colors = ["#2196F3" if tt == "other" else "#E91E63" for (tt, _), _ in top_combos]
                combo_ax.barh(range(len(top_combos)), [c for _, c in top_combos], color=colors)
                combo_ax.set_yticks(range(len(top_combos)))
                combo_ax.set_yticklabels(labels, fontsize=7)
                combo_ax.set_title("Top 15 Type+Function Combos")
                combo_ax.set_xlabel("Count")
                combo_ax.invert_yaxis()

                for ri in range(n_rows):
                    for ci in range(3):
                        if not axes[ri, ci].has_data():
                            axes[ri, ci].set_visible(False)

                plt.tight_layout()
                fname = charts_dir / f"{label}_{phase}_distribution.png"
                plt.savefig(fname, dpi=150, bbox_inches="tight")
                plt.close()
                generated_files.append(fname)

    agg_grouped = defaultdict(lambda: defaultdict(list))
    for row in all_data:
        agg_grouped[row["schedule"]][row["phase"]].append(row)

    for schedule in schedules:
        for phase in phases:
            sched_phase_data = agg_grouped[schedule].get(phase, [])
            if not sched_phase_data:
                continue

            fn_cols = sorted([c for c in sched_phase_data[0].keys()
                              if c.startswith("f") and c[1:].isdigit()],
                             key=lambda c: int(c[1:]), reverse=True)

            comp_stats = defaultdict(lambda: {"counts": [], "meta": {}})
            for row in sched_phase_data:
                fn_vals = tuple(row.get(c, 0) for c in fn_cols)
                key = (row["task_type"], fn_vals, row["composition"])
                comp_stats[key]["counts"].append(row["count"])
                comp_stats[key]["meta"] = {
                    "task_type": row["task_type"],
                    "composition": row["composition"],
                    **{c: row.get(c, 0) for c in fn_cols},
                }

            comp_summary = []
            for key, data in comp_stats.items():
                counts = data["counts"]
                entry = {
                    "task_type": data["meta"]["task_type"],
                    "composition": data["meta"]["composition"],
                    "mean": np.mean(counts),
                    "std": np.std(counts) if len(counts) > 1 else 0,
                }
                for c in fn_cols:
                    entry[c] = data["meta"][c]
                comp_summary.append(entry)

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"{schedule} - {phase} (averaged across seeds)", fontsize=14, fontweight="bold")

            type_means = {}
            for item in comp_summary:
                tt = item["task_type"]
                type_means[tt] = type_means.get(tt, 0) + item["mean"]
            types = sorted(type_means.keys())
            colors = ["#2196F3" if t == "other" else "#E91E63" for t in types]
            axes[0, 0].bar(types, [type_means[t] for t in types], color=colors)
            axes[0, 0].set_title("Other Classes vs Special Class (mean)")
            axes[0, 0].set_xlabel("Task Type")
            axes[0, 0].set_ylabel("Mean Count")
            for i, t in enumerate(types):
                axes[0, 0].text(i, type_means[t], f"{type_means[t]:.0f}", ha="center", va="bottom")

            fn_means = {c: {} for c in fn_cols}
            for item in comp_summary:
                for c in fn_cols:
                    fn_means[c][item[c]] = fn_means[c].get(item[c], 0) + item["mean"]

            all_funcs = sorted(set(v for fm in fn_means.values() for v in fm.keys()))
            x_pos = np.arange(len(all_funcs))
            width = 0.8 / max(len(fn_cols), 1)
            fn_colors = ["#4CAF50", "#FF9800", "#9C27B0", "#00BCD4", "#795548"]

            for fi, col in enumerate(fn_cols):
                axes[0, 1].bar(x_pos + fi * width,
                              [fn_means[col].get(f, 0) for f in all_funcs], width,
                              label=col.upper(), alpha=0.8,
                              color=fn_colors[fi % len(fn_colors)])
            axes[0, 1].set_title("Function Usage by Position")
            axes[0, 1].set_xlabel("Function ID")
            axes[0, 1].set_ylabel("Mean Count")
            axes[0, 1].set_xticks(x_pos + width * len(fn_cols) / 2)
            axes[0, 1].set_xticklabels([str(f) for f in all_funcs], rotation=45)
            axes[0, 1].legend()

            top_comps = sorted(comp_summary, key=lambda x: x["mean"], reverse=True)[:15]
            means = [c["mean"] for c in top_comps]
            stds = [c["std"] for c in top_comps]
            comps = [c["composition"] for c in top_comps]
            axes[1, 0].barh(range(len(top_comps)), means, xerr=stds, color="#00BCD4", alpha=0.7)
            axes[1, 0].set_yticks(range(len(top_comps)))
            axes[1, 0].set_yticklabels(comps, fontsize=8)
            axes[1, 0].set_title("Top 15 Compositions (mean ± std)")
            axes[1, 0].set_xlabel("Mean Count")
            axes[1, 0].invert_yaxis()

            outermost = fn_cols[0] if fn_cols else "f1"
            type_outer_means = {}
            for item in comp_summary:
                key = (item["task_type"], item.get(outermost, 0))
                type_outer_means[key] = type_outer_means.get(key, 0) + item["mean"]

            task_types = sorted(set(tt for tt, _ in type_outer_means.keys()))
            outer_vals = sorted(set(fv for _, fv in type_outer_means.keys()))

            x_pos = np.arange(len(outer_vals))
            width = 0.35
            for i, tt in enumerate(task_types):
                vals = [type_outer_means.get((tt, fv), 0) for fv in outer_vals]
                axes[1, 1].bar(x_pos + i*width, vals, width,
                              label=f"Type {tt}", alpha=0.8,
                              color="#2196F3" if tt == "other" else "#E91E63")
            axes[1, 1].set_title(f"Task Type × {outermost.upper()} Distribution")
            axes[1, 1].set_xlabel(f"{outermost.upper()} Function ID")
            axes[1, 1].set_ylabel("Mean Count")
            axes[1, 1].set_xticks(x_pos + width/2)
            axes[1, 1].set_xticklabels([str(f) for f in outer_vals], rotation=45)
            axes[1, 1].legend()

            plt.tight_layout()
            fname = charts_dir / f"{schedule}_{phase}_summary.png"
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close()
            generated_files.append(fname)

    print(f"  Generated {len(generated_files)} task distribution charts in {charts_dir}")
    return generated_files


def _resolve_dirs(run_dir: Path) -> tuple[Path, Path, Path]:
    """Return (results_dir, logs_dir, plots_dir) for a run directory.

    Supports both old flat layout and new results/logs layout.
    """
    results_dir = run_dir / "results"
    logs_dir = run_dir / "logs"
    if results_dir.exists():
        plots_dir = results_dir / "plots"
    else:
        plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return results_dir if results_dir.exists() else run_dir, \
           logs_dir if logs_dir.exists() else run_dir, \
           plots_dir


def main():
    if len(sys.argv) < 2:
        data_dir = Path("data")
        burst_dirs = sorted([d for d in data_dir.glob("*burst_d*") if d.is_dir()])
        if not burst_dirs:
            print("No burst_d* dirs found"); sys.exit(1)
        run_dir = burst_dirs[-1]
        print(f"Auto-detected: {run_dir}")
    else:
        run_dir = Path(sys.argv[1])

    results, cfg = load_results(run_dir)
    _, logs_dir, plots_dir = _resolve_dirs(run_dir)

    print("Per-run plots...")
    per_run_fnames = []
    for r in results:
        fname = plot_per_run(r, plots_dir, run_cfg=cfg)
        per_run_fnames.append(fname)
        print(f"  {fname}")

    print("Summary bars...")
    plot_summary_bars(results, plots_dir, cfg)

    sched_groups = _build_sched_groups(results)

    print("AUC detail...")
    plot_auc_detail(results, plots_dir, cfg, sched_groups=sched_groups)

    print("AUC diff %...")
    plot_auc_diff_pct(results, plots_dir, cfg, sched_groups=sched_groups)

    sched_data = _build_sched_data(results)

    print("Overlay per schedule...")
    plot_overlay_per_schedule(results, plots_dir, sched_data=sched_data)

    print("Overlay all schedules...")
    plot_overlay_all_schedules(results, plots_dir, sched_data=sched_data)

    print("LR schedule...")
    plot_lr_schedule(cfg["base_cfg"], plots_dir, schedules=cfg.get("schedules"),
                     burst_mode=cfg.get("burst_mode", MODE_CURRENT))

    print("Task distributions...")
    plot_task_distributions(run_dir)

    print("PDF report...")
    make_report(run_dir, results, cfg, per_run_fnames)
    print("\nDone.")


if __name__ == "__main__":
    main()
