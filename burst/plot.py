"""Plot + PDF report for burst experiment.

Usage: python burst/plot.py data/burst_d3_<run_tag>
"""
import sys, os, pickle, json, math, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from fpdf import FPDF
from collections import defaultdict, Counter
from burst._worker import n_target_for_step
from burst.train_utils import load_results, compute_lr_schedule
from burst.config import (
    EVAL_KEYS, CURVE_STYLE, SCHED_COLORS, SCHEDULE_ORDER,
    PHASE_FOUNDATION, PHASE_BURST, PHASE_REVERSION,
    ordered_schedules, sched_sort_key,
)

W, H = 297, 210


def _bar_label(ax, x, text):
    ax.text(x, 0.5, text, ha="center", va="center", fontsize=5,
            color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.45, lw=0))


def _schedule_bar(ax, T, U, sched, p, bs, seed):
    total = T + U
    fracs = np.zeros(total)
    for s in range(T):
        np.random.seed(seed * 10000 + s)
        fracs[s] = n_target_for_step(s, T, sched, p, bs) / bs
    ax.imshow(fracs.reshape(1, -1), aspect="auto", cmap="Blues",
              extent=[0, total, 0, 1], vmin=0, vmax=1)
    ax.set_yticks([])
    ax.set_ylabel("Burst frac", fontsize=7)
    ax.set_xlim(0, total)
    ax.axvline(T, color="black", lw=2)

    if sched == "burst_10":
        _bar_label(ax, T / 2, f"Other: ~{(1-p)*100:.0f}% | Burst: ~{p*100:.0f}% (random)")
        _bar_label(ax, T + U / 2, "Other: 100% | Burst: 0%")
        return

    if sched == "ramp_up":
        burst_len = max(int(p * T), 1)
        ramp_len = min(int(2 * burst_len / 0.20), T)
        ramp_start = T - ramp_len
        if ramp_start > 0:
            _bar_label(ax, ramp_start / 2, "Other: 100% | Burst: 0%")
        _bar_label(ax, (ramp_start + T) / 2, "Burst: 0% -> 20% (ramp)")
        _bar_label(ax, T + U / 2, "Other: 100% | Burst: 0%")
        return

    regions, cur_val, start = [], fracs[0], 0
    for i in range(1, total):
        if abs(fracs[i] - cur_val) > 0.01:
            regions.append((start, i, cur_val))
            cur_val, start = fracs[i], i
    regions.append((start, total, cur_val))

    merged = []
    for s, e, v in regions:
        if merged and abs(merged[-1][2] - v) < 0.01:
            merged[-1] = (merged[-1][0], e, v)
        else:
            merged.append((s, e, v))

    for s, e, v in merged:
        if (e - s) < total * 0.03:
            continue
        b_pct = v * 100
        txt = (f"Other: {100-b_pct:.0f}% | Burst: 0%" if b_pct < 0.5
               else f"Other: {100-b_pct:.0f}% | Burst: {b_pct:.0f}%")
        _bar_label(ax, (s + e) / 2, txt)


def plot_lr_schedule(cfg, plots_dir):
    steps, lrs = compute_lr_schedule(cfg)
    T, U = cfg["total_steps"], cfg["reversion_steps"]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(steps, lrs, color="#1565C0", lw=2)
    ax.axvline(T, color="black", lw=2, ls="--")
    ax.axvline(cfg["warmup_iters"], color="gray", lw=1, ls=":", alpha=0.6)
    ax.set_xlim(0, T + U)
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule (cosine decay with warmup)", fontsize=12, fontweight="bold")
    ax.text(T * 0.5, ax.get_ylim()[1] * 0.95, "FOUNDATION + BURST", ha="center", fontsize=9, color="gray")
    ax.text(T + U * 0.5, ax.get_ylim()[1] * 0.95, "REVERSION", ha="center", fontsize=9, color="gray")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(plots_dir / "lr_schedule.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_run(result, plots_dir):
    log, sched, seed, cfg = result["log"], result["schedule"], result["seed"], result["config"]
    steps = np.array(log["step"])
    loss = np.array(log["loss"])
    T, U = cfg["total_steps"], cfg["reversion_steps"]
    bs, p = cfg["batch_size"], cfg["p_target"]

    train_m = np.array([ph == "burst" for ph in log["phase"]])
    reversion_m = np.array([ph == "reversion" for ph in log["phase"]])

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 4, 2]})
    fig.suptitle(f"{sched}  seed={seed}  (depth-3 bijection chain)",
                 fontsize=13, fontweight="bold")

    _schedule_bar(axes[0], T, U, sched, p, bs, seed)
    axes[0].set_title("Schedule (Burst fraction per step)", fontsize=9)

    ax = axes[1]
    for k, sty in CURVE_STYLE.items():
        vals = np.array(log.get(k, [0.0] * len(steps)))
        ax.plot(steps, vals, color=sty["color"], ls=sty["ls"], lw=1.5, label=sty["label"])
    ax.axvline(T, color="gray", ls="--", alpha=0.5)
    ax.set_xlim(0, T + U)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Free-gen Accuracy (last 6 tok)")
    ax.legend(fontsize=5, loc="lower left", ncol=2)
    ax.grid(True, alpha=0.2)

    peak = result["peak_burst"]
    ql = result.get("quarter_life", U)
    ql_str = f"{ql:.0f}" if ql < U else f">{U}"
    drop = result.get("dropoff_abs", 0)
    drop_pct = result.get("dropoff_pct", 0)
    ax.text(T + U * 0.5, 0.95,
            f"peak={peak:.3f}  t1/4={ql_str}  drop={drop:.3f}({drop_pct:.0f}%)",
            ha="center", fontsize=7, color="#D32F2F", fontweight="bold",
            transform=ax.get_xaxis_transform())
    if ql < U:
        ax.axvline(T + ql, color="#D32F2F", ls="--", lw=1.5, alpha=0.7)
        ax.axhline(peak * 0.25, color="#D32F2F", ls=":", lw=1, alpha=0.4)

    ax = axes[2]
    ax.plot(steps, loss, color="#333", lw=1, label="loss")
    ax.axvline(T, color="gray", ls="--", alpha=0.5)
    ax.set_xlim(0, T + U)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Global Step")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    axes[1].text(T * 0.5, -0.04, "FOUNDATION+BURST", ha="center", fontsize=7,
                 color="gray", transform=axes[1].get_xaxis_transform())
    axes[1].text(T + U * 0.5, -0.04, "REVERSION", ha="center", fontsize=7,
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

    ordered = sorted(results, key=lambda r: sched_sort_key(r["schedule"]))
    scheds = [r["schedule"] for r in ordered]
    peaks = [r["peak_burst"] for r in ordered]
    quarterlives = [r.get("quarter_life", reversion_steps) for r in ordered]
    aucs = [r["reversion_auc"] for r in ordered]
    colors = [SCHED_COLORS.get(s, "gray") for s in scheds]
    xs = np.arange(len(scheds))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Peak Burst Class Accuracy + Quarter-life + AUC by Schedule",
                 fontsize=14, fontweight="bold")

    titles = [f"Peak Burst Class Accuracy at step {total_steps}",
              "Quarter-life t1/4 (lower = faster forgetting)",
              "Reversion AUC (lower = faster forgetting)"]
    ylabels = ["Peak Burst Class accuracy", "Quarter-life (reversion steps)", "Reversion AUC"]
    data = [peaks, quarterlives, aucs]

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
        for k in EVAL_KEYS:
            vals = np.array(r["log"].get(k, [0.0] * len(steps)))
            sched_data[sched][k].append((steps, vals))
    return sched_data


def plot_overlay_per_schedule(results, plots_dir, sched_data=None):
    T_ov = results[0]["config"]["total_steps"]
    U_ov = results[0]["config"]["reversion_steps"]
    total_steps = T_ov + U_ov

    if sched_data is None:
        sched_data = _build_sched_data(results)

    for sched in ordered_schedules(sched_data.keys()):
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.suptitle(f"{sched} - All Metrics (mean ± 95% CI across seeds)",
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

            if n_seeds > 1:
                ci = 1.96 * std_vals / np.sqrt(n_seeds)
            else:
                ci = std_vals

            sty = CURVE_STYLE[k]
            ax.plot(steps_ref, mean_vals, color=sty["color"], ls=sty["ls"],
                   lw=2.5, label=sty["label"])
            ax.fill_between(steps_ref, mean_vals - ci, mean_vals + ci,
                           color=sty["color"], alpha=0.25)

        ax.axvline(T_ov, color="gray", ls="--", alpha=0.6, lw=2)
        ax.text(T_ov * 0.5, 0.05, "FOUNDATION+BURST", ha="center", fontsize=11,
               color="gray", fontweight="bold")
        ax.text(T_ov + U_ov * 0.5, 0.05, "REVERSION", ha="center", fontsize=11,
               color="gray", fontweight="bold")

        ax.set_xlim(0, total_steps)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Global Step", fontsize=11)
        ax.set_ylabel("Free-gen Accuracy (last 6 tok)", fontsize=11)
        ax.legend(fontsize=9, loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        idx = sched_sort_key(sched)
        fig.savefig(plots_dir / f"{idx:02d}_overlay_{sched}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_overlay_all_schedules(results, plots_dir, sched_data=None):
    T_ov = results[0]["config"]["total_steps"]
    U_ov = results[0]["config"]["reversion_steps"]
    total_steps = T_ov + U_ov

    if sched_data is None:
        sched_data = _build_sched_data(results)

    for ki, k in enumerate(EVAL_KEYS):
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        fig.suptitle(f"All Schedules - {CURVE_STYLE[k]['label']}\n(mean ± 95% CI across seeds)",
                     fontsize=16, fontweight="bold")

        for sched in ordered_schedules(sched_data.keys()):
            c = SCHED_COLORS.get(sched, "gray")
            runs = sched_data[sched][k]

            if len(runs) == 0:
                continue

            steps_ref = runs[0][0]
            all_vals = np.array([vals for _, vals in runs])

            mean_vals = np.mean(all_vals, axis=0)
            std_vals = np.std(all_vals, axis=0)
            n_seeds = len(runs)

            if n_seeds > 1:
                ci = 1.96 * std_vals / np.sqrt(n_seeds)
            else:
                ci = std_vals

            ax.plot(steps_ref, mean_vals, color=c, lw=2.5, label=sched)
            ax.fill_between(steps_ref, mean_vals - ci, mean_vals + ci,
                           color=c, alpha=0.2)

        ax.axvline(T_ov, color="gray", ls="--", alpha=0.6, lw=2)
        ax.text(T_ov * 0.5, 0.05, "FOUNDATION+BURST", ha="center", fontsize=12,
               color="gray", fontweight="bold")
        ax.text(T_ov + U_ov * 0.5, 0.05, "REVERSION", ha="center", fontsize=12,
               color="gray", fontweight="bold")

        ax.set_xlim(0, total_steps)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Step", fontsize=13)
        ax.set_ylabel("Accuracy", fontsize=13)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(dict(zip(labels, handles)).values(),
                  dict(zip(labels, handles)).keys(), fontsize=10, loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(plots_dir / f"overlay_all_{k}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


class ReportPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 7)
            self.set_text_color(130, 130, 130)
            self.cell(0, 4, "Depth-3 Bijection Burst  |  Free Generation", align="L")
            self.cell(0, 4, f"Page {self.page_no()}/{{nb}}", align="R")
            self.ln(6)

    def stitle(self, t):
        self.set_font("Helvetica", "B", 16); self.set_text_color(0, 80, 140)
        self.cell(0, 9, t, new_x="LMARGIN", new_y="NEXT"); self.ln(3)

    def sub(self, t):
        self.set_font("Helvetica", "B", 11); self.set_text_color(40, 40, 40)
        self.cell(0, 6, t, new_x="LMARGIN", new_y="NEXT"); self.ln(1)

    def body(self, t):
        self.set_font("Helvetica", "", 9); self.set_text_color(30, 30, 30)
        self.multi_cell(0, 4.5, t); self.ln(2)

    def bul(self, t):
        self.set_font("Helvetica", "", 9); self.set_text_color(30, 30, 30)
        self.cell(4, 4.5, "-"); self.multi_cell(W - 24, 4.5, t); self.ln(1)

    def chart(self, path, w=220):
        if Path(path).exists():
            if self.get_y() > H - 55:
                self.add_page()
            self.image(str(path), x=(W - w) / 2, w=w); self.ln(3)


def make_report(run_dir, results, cfg, per_run_fnames):
    plots_dir = run_dir / "plots"
    bcfg = cfg.get("base_cfg", cfg)
    n_a = cfg.get("n_a", 4)

    n_layer = bcfg['n_layer']
    n_embd = bcfg['n_embd']
    n_head = bcfg['n_head']
    total_steps = bcfg['total_steps']
    reversion_steps = bcfg['reversion_steps']
    batch_size = bcfg['batch_size']
    p_target = bcfg['p_target']
    depth = cfg.get("depth", cfg.get("task_info", {}).get("depth", 3))
    burst_pos = cfg.get("burst_pos", cfg.get("task_info", {}).get("burst_pos", depth))

    pdf = ReportPDF(orientation="L", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=12)

    pdf.add_page(); pdf.ln(25)
    pdf.set_font("Helvetica", "B", 26); pdf.set_text_color(0, 80, 140)
    pdf.multi_cell(0, 11, f"Depth-{depth} Bijection Composition\nBurst & Forgetting Experiment", align="C")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 12); pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 7, f"Burst at position {burst_pos}  |  Free Generation (model produces its own outputs)",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Courier", "", 8); pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 5,
             f"{n_layer}-layer Transformer ({n_embd}-dim, {n_head} heads)  |  "
             f"{total_steps} foundation+burst + {reversion_steps} reversion  |  batch {batch_size}  |  {len(results)} runs",
             align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.add_page()
    pdf.stitle("What This Experiment Does")

    pdf.sub("The Task")
    pdf.body(
        "The model learns to apply chains of 3 functions to a sequence of numbers. "
        "Each function is a bijection -- a lookup table that remaps each digit "
        "(0-9) to a different digit. Every sequence has the same format: three "
        "function slots followed by the input, then the result after each function.")

    pdf.sub("Complex Tasks")
    pdf.body(
        "Complex (compositional) tasks: all three slots have real functions. "
        "The model must learn to compose multiple bijections together to produce "
        "the correct output sequence.")

    pdf.sub("Training Data (Other Classes = background knowledge)")
    depth = cfg.get("depth", 3)
    burst_pos = cfg.get("burst_pos", depth)
    n_a_comps = n_a ** depth
    pdf.body(
        f"{n_a} bijection functions. The model trains on all {n_a}^{depth} = {n_a_comps} "
        f"depth-{depth} chains (100% used for training).")

    pdf.sub("Burst Data (Burst Class = the new thing to learn)")
    n_burst = n_a ** (depth - 1)
    pdf.body(
        f"One brand-new function (b*) placed at position {burst_pos}. "
        f"All {n_burst} possible combinations for the other positions are used during the burst.")

    pdf.sub("The Experiment")
    pdf.body(
        f"Foundation+Burst ({total_steps} steps): Other classes + Burst class mixed per schedule. "
        f"All schedules see the same total burst class data. "
        f"Reversion ({reversion_steps} steps): Burst class removed, other classes only. "
        f"We measure how quickly the burst class is forgotten.")

    pdf.sub("Metrics")
    pdf.bul("Other Classes: compositional accuracy on known functions")
    pdf.bul("Burst Class: accuracy on b* chains (acquisition + retention)")
    pdf.bul("Peak Burst: b* accuracy at end of training")
    pdf.bul(f"Quarter-life: reversion steps until Burst Class drops to 25% of peak (capped at {reversion_steps})")
    pdf.bul("Reversion AUC: area under Burst Class curve during reversion (lower = faster forgetting)")

    burst_len = max(int(p_target * total_steps), 1)
    p_pct = int(p_target * 100)

    pdf.sub("The Schedules")
    pdf.bul(f"burst_100: 100% Burst Class block at the end ({burst_len} steps)")
    for pct, frac in [(98, 0.98), (95, 0.95), (90, 0.90), (85, 0.85),
                      (75, 0.75), (50, 0.50), (25, 0.25)]:
        win = min(int(burst_len / frac), total_steps)
        pdf.bul(f"burst_{pct}: {pct}% Burst Class at the end ({win} steps)")
    pdf.bul(f"burst_10: ~{p_pct}% Burst Class randomly throughout (uniform control)")

    pdf.add_page()
    pdf.stitle("Learning Rate Schedule")
    pdf.chart(plots_dir / "lr_schedule.png", w=240)
    warmup_iters = bcfg['warmup_iters']
    lr_max = bcfg['lr']
    lr_min = bcfg['min_lr']
    total_train_reversion = total_steps + reversion_steps
    pdf.body(
        f"Cosine decay with linear warmup. Ramps up during the first "
        f"{warmup_iters} steps, then decays from {lr_max} to "
        f"{lr_min} over {total_train_reversion} steps. The reversion phase continues the same "
        f"schedule -- the model keeps learning on other classes data at a low rate.")

    pdf.add_page()
    pdf.stitle("Summary: Forgetting Speed by Schedule")
    pdf.chart(plots_dir / "summary_bars.png", w=260)
    pdf.body(
        "Left: Peak Burst Class accuracy. Center: Quarter-life (lower = faster forgetting). "
        "Right: Reversion AUC (secondary). Schedules that deliver burst class near the end "
        "achieve high acquisition. Mixed schedules retain the burst class longer because "
        "other classes are present alongside the burst class during the burst window.")

    pdf.add_page()
    pdf.stitle("AUC Detail: Individual Seeds + Mean ± CI")
    pdf.chart(plots_dir / "auc_detail.png", w=260)
    pdf.body(
        "Left: each dot is one seed for a given schedule. "
        "Right: mean reversion AUC with 95% CI error bars. "
        "Lower AUC = faster forgetting of burst class during the reversion phase.")

    pdf.add_page()
    pdf.stitle("Pairwise Reversion AUC Difference (%)")
    pdf.chart(plots_dir / "auc_diff_pct.png", w=200)
    pdf.body(
        "Each cell shows (row_AUC - col_AUC) / |col_AUC| × 100. "
        "Positive (red) means the row schedule has higher AUC (slower forgetting). "
        "Negative (blue) means faster forgetting relative to the column schedule.")

    pdf.add_page()
    pdf.stitle("Ranking: Fastest Forgetting First")
    rows = sorted(results, key=lambda r: r.get("quarter_life", reversion_steps))
    pdf.set_font("Courier", "", 7.5); pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 4,
             f"  {'Rank':<5}{'Schedule':<16}{'Peak Burst':>10}{'t1/4':>8}{'Rev AUC':>9}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 4, "  " + "-" * 48, new_x="LMARGIN", new_y="NEXT")
    for i, r in enumerate(rows):
        ql = r.get("quarter_life", reversion_steps)
        ql_str = f"{ql:.0f}" if ql < reversion_steps else f">{reversion_steps}"
        peak = r["peak_burst"]
        auc = r["reversion_auc"]
        pdf.cell(0, 4,
                 f"  {i+1:<5}{r['schedule']:<16}{peak:>10.3f}"
                 f"{ql_str:>8}{auc:>9.0f}",
                 new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.add_page()
    pdf.stitle("Accuracy Overlay - All Schedules")
    pdf.body(
        "Each page shows one accuracy metric with all schedules overlaid. "
        "Lines show mean accuracy across seeds, with shaded regions showing "
        "95% confidence intervals. Vertical dashed line marks "
        "the start of the reversion phase.")

    for k in EVAL_KEYS:
        overlay_all_path = plots_dir / f"overlay_all_{k}.png"
        if overlay_all_path.exists():
            pdf.add_page()
            pdf.sub(f"{CURVE_STYLE[k]['label']}")
            pdf.chart(overlay_all_path, w=270)

    pdf.add_page()
    pdf.stitle("Accuracy Overlay - Per Schedule")
    pdf.body(
        "Each chart shows all metrics for one schedule. "
        "Lines show mean accuracy across seeds, with shaded regions showing "
        "95% confidence intervals. Vertical dashed line marks "
        "the start of the reversion phase.")

    for sched in ordered_schedules(set(r["schedule"] for r in results)):
        idx = sched_sort_key(sched)
        overlay_path = plots_dir / f"{idx:02d}_overlay_{sched}.png"
        if overlay_path.exists():
            pdf.sub(f"Schedule: {sched}")
            pdf.chart(overlay_path, w=240)

    pdf.add_page()
    pdf.stitle("Per-Run Details")
    pdf.body(
        "Each plot: (top) schedule bar with Other/Burst percentages, "
        "(middle) accuracy curves with metrics, (bottom) training loss.")
    for fname in sorted(per_run_fnames):
        pdf.chart(plots_dir / fname, w=240)

    pdf.output(str(run_dir / "analysis_report.pdf"))
    print(f"  Saved {run_dir / 'analysis_report.pdf'}")


def plot_task_distributions(run_dir):
    run_dir = Path(run_dir)
    stats_dir = run_dir / "task_distributions"

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
                axes[0, 0].set_title("Other Classes vs Burst Class")
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
            axes[0, 0].set_title("Other Classes vs Burst Class (mean)")
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


def main():
    if len(sys.argv) < 2:
        data_dir = Path("data")
        burst_dirs = sorted([d for d in data_dir.glob("burst_d3_*") if d.is_dir()])
        if not burst_dirs:
            print("No burst_d3_* dirs found"); sys.exit(1)
        run_dir = burst_dirs[-1]
        print(f"Auto-detected: {run_dir}")
    else:
        run_dir = Path(sys.argv[1])

    results, cfg = load_results(run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print("Per-run plots...")
    per_run_fnames = []
    for r in results:
        fname = plot_per_run(r, plots_dir)
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
    plot_lr_schedule(cfg.get("base_cfg", cfg), plots_dir)

    print("Task distributions...")
    plot_task_distributions(run_dir)

    print("PDF report...")
    make_report(run_dir, results, cfg, per_run_fnames)
    print("\nDone.")


if __name__ == "__main__":
    main()
