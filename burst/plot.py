"""Plot + PDF report for depth-3 bijection burst experiment.

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

EVAL_KEYS = ["acc_A_comp", "acc_B_comp"]
CURVE_STYLE = {
    "acc_A_comp": {"color": "#2196F3", "ls": "-", "label": "A comp"},
    "acc_B_comp": {"color": "#E91E63", "ls": "-", "label": "B comp"},
}
SCHED_COLORS = {
    "uniform": "#2196F3", "end_block": "#F44336", "mid_block": "#9C27B0",
    "end_mixed_75b": "#FF9800", "end_mixed_50b": "#E91E63", "end_mixed_25b": "#009688",
    "ramp_up": "#795548",
}
W, H = 297, 210


def load_results(run_dir):
    run_dir = Path(run_dir)
    with open(run_dir / "all_results.pkl", "rb") as f:
        results = pickle.load(f)
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    return results, cfg


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
    ax.set_ylabel("B frac", fontsize=7)
    ax.set_xlim(0, total)
    ax.axvline(T, color="black", lw=2)

    if sched == "uniform":
        _bar_label(ax, T / 2, f"A: ~{(1-p)*100:.0f}% | B: ~{p*100:.0f}% (random)")
        _bar_label(ax, T + U / 2, "A: 100% | B: 0%")
        return

    if sched == "ramp_up":
        burst_len = max(int(p * T), 1)
        ramp_len = min(int(2 * burst_len / 0.20), T)
        ramp_start = T - ramp_len
        if ramp_start > 0:
            _bar_label(ax, ramp_start / 2, "A: 100% | B: 0%")
        _bar_label(ax, (ramp_start + T) / 2, "B: 0% -> 20% (ramp)")
        _bar_label(ax, T + U / 2, "A: 100% | B: 0%")
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
        txt = f"A: {100-b_pct:.0f}% | B: 0%" if b_pct < 0.5 else f"A: {100-b_pct:.0f}% | B: {b_pct:.0f}%"
        _bar_label(ax, (s + e) / 2, txt)


def compute_lr_schedule(cfg):
    T, U = cfg["total_steps"], cfg["undo_steps"]
    total = T + U
    lr_max, lr_min, warmup = cfg["lr"], cfg["min_lr"], cfg["warmup_iters"]
    steps = np.arange(1, total + 1)
    lrs = np.zeros(total)
    for i, s in enumerate(steps):
        if s < warmup:
            lrs[i] = lr_max * s / warmup
        else:
            decay = (s - warmup) / (total - warmup)
            lrs[i] = lr_min + 0.5 * (1.0 + math.cos(math.pi * decay)) * (lr_max - lr_min)
    return steps, lrs


def plot_lr_schedule(cfg, plots_dir):
    steps, lrs = compute_lr_schedule(cfg)
    T, U = cfg["total_steps"], cfg["undo_steps"]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(steps, lrs, color="#1565C0", lw=2)
    ax.axvline(T, color="black", lw=2, ls="--")
    ax.axvline(cfg["warmup_iters"], color="gray", lw=1, ls=":", alpha=0.6)
    ax.set_xlim(0, T + U)
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule (cosine decay with warmup)", fontsize=12, fontweight="bold")
    ax.text(T * 0.5, ax.get_ylim()[1] * 0.95, "TRAIN", ha="center", fontsize=9, color="gray")
    ax.text(T + U * 0.5, ax.get_ylim()[1] * 0.95, "UNDO", ha="center", fontsize=9, color="gray")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(plots_dir / "lr_schedule.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_run(result, plots_dir):
    log, sched, seed, cfg = result["log"], result["schedule"], result["seed"], result["config"]
    steps = np.array(log["step"])
    loss = np.array(log["loss"])
    T, U = cfg["total_steps"], cfg["undo_steps"]
    bs, p = cfg["batch_size"], cfg["p_target"]

    train_m = np.array([ph == "train" for ph in log["phase"]])
    undo_m = np.array([ph == "undo" for ph in log["phase"]])

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 4, 2]})
    fig.suptitle(f"n_B={nb}  {sched}  seed={seed}  (depth-3 bijection chain)",
                 fontsize=13, fontweight="bold")

    _schedule_bar(axes[0], T, U, sched, p, bs, seed)
    axes[0].set_title("Schedule (B fraction per step)", fontsize=9)

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

    peak = result.get("train_end_B_comp", 0)
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

    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fname = f"run_nB{nb}_{sched}_s{seed}.png"
    fig.savefig(plots_dir / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_summary_bars(results, plots_dir, cfg):
    bcfg = cfg.get("base_cfg", cfg)
    total_steps = bcfg["total_steps"]
    undo_steps = bcfg["undo_steps"]

    scheds = [r["schedule"] for r in results]
    peaks = [r.get("train_end_B_comp", 0) for r in results]
    quarterlives = [r.get("quarter_life", undo_steps) for r in results]
    aucs = [r.get("undo_auc", 0) for r in results]
    colors = [SCHED_COLORS.get(s, "gray") for s in scheds]
    xs = np.arange(len(scheds))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Peak B Accuracy + Quarter-life + AUC by Schedule", fontsize=14, fontweight="bold")

    titles = [f"Peak B Accuracy at step {total_steps}",
              "Quarter-life t1/4 (lower = faster forgetting)",
              "Undo AUC (lower = faster forgetting)"]
    ylabels = ["Peak B comp accuracy", "Quarter-life (undo steps)", "Undo AUC"]
    data = [peaks, quarterlives, aucs]

    for ax, vals, title, ylabel in zip(axes, data, titles, ylabels):
        bars = ax.bar(xs, vals, color=colors, edgecolor="black", lw=0.5)
        for b, v in zip(bars, vals):
            lbl = f"{v:.3f}" if max(vals) <= 1.5 else (f"{v:.0f}" if v < undo_steps else f">{undo_steps}")
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(vals) * 0.01,
                    lbl, ha="center", fontsize=7, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels(scheds, fontsize=8, rotation=25, ha="right")
        ax.grid(True, alpha=0.2, axis="y")

    axes[1].axhline(undo_steps, color="gray", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(plots_dir / "summary_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overlay_per_schedule(results, plots_dir):
    T_ov = results[0]["config"]["total_steps"]
    U_ov = results[0]["config"]["undo_steps"]
    total_steps = T_ov + U_ov

    sched_data = defaultdict(lambda: defaultdict(list))
    for r in results:
        sched = r["schedule"]
        steps = np.array(r["log"]["step"])
        for k in EVAL_KEYS:
            vals = np.array(r["log"].get(k, [0.0] * len(steps)))
            sched_data[sched][k].append((steps, vals))

    for sched in sorted(sched_data.keys()):
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
        ax.text(T_ov * 0.5, 0.05, "TRAIN", ha="center", fontsize=11,
               color="gray", fontweight="bold")
        ax.text(T_ov + U_ov * 0.5, 0.05, "UNDO", ha="center", fontsize=11,
               color="gray", fontweight="bold")

        ax.set_xlim(0, total_steps)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Global Step", fontsize=11)
        ax.set_ylabel("Free-gen Accuracy (last 6 tok)", fontsize=11)
        ax.legend(fontsize=9, loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(plots_dir / f"overlay_{sched}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_overlay_all_schedules(results, plots_dir):
    T_ov = results[0]["config"]["total_steps"]
    U_ov = results[0]["config"]["undo_steps"]
    total_steps = T_ov + U_ov

    sched_data = defaultdict(lambda: defaultdict(list))
    for r in results:
        sched = r["schedule"]
        steps = np.array(r["log"]["step"])
        for k in EVAL_KEYS:
            vals = np.array(r["log"].get(k, [0.0] * len(steps)))
            sched_data[sched][k].append((steps, vals))

    for ki, k in enumerate(EVAL_KEYS):
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        fig.suptitle(f"All Schedules - {CURVE_STYLE[k]['label']}\n(mean ± 95% CI across seeds)",
                     fontsize=16, fontweight="bold")

        for sched in sorted(sched_data.keys()):
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
        ax.text(T_ov * 0.5, 0.05, "TRAIN", ha="center", fontsize=12,
               color="gray", fontweight="bold")
        ax.text(T_ov + U_ov * 0.5, 0.05, "UNDO", ha="center", fontsize=12,
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
    undo_steps = bcfg['undo_steps']
    batch_size = bcfg['batch_size']
    p_target = bcfg['p_target']

    pdf = ReportPDF(orientation="L", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=12)

    pdf.add_page(); pdf.ln(25)
    pdf.set_font("Helvetica", "B", 26); pdf.set_text_color(0, 80, 140)
    pdf.multi_cell(0, 11, "Depth-3 Bijection Composition\nBurst & Forgetting Experiment", align="C")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 12); pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 7, "Free Generation (model produces its own outputs, no hints)",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Courier", "", 8); pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 5,
             f"{n_layer}-layer Transformer ({n_embd}-dim, {n_head} heads)  |  "
             f"{total_steps} train + {undo_steps} undo steps  |  batch {batch_size}  |  {len(results)} runs",
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

    pdf.sub("Training Data (A = background knowledge)")
    n_a_comps_BSN = n_a ** 3
    pdf.body(
        f"{n_a} bijection functions. The model trains on all {n_a}×{n_a}×{n_a} = {n_a_comps_BSN} three-function "
        f"chains (100% used for training).")

    pdf.sub("Burst Data (B = the new thing to learn)")
    n_b_pairs = n_a * n_a
    pdf.body(
        f"One brand-new function (b*) placed at position 3. "
        f"All {n_b_pairs} possible pairs for positions 1-2 are used during the burst.")

    pdf.sub("The Experiment")
    pdf.body(
        f"Phase 1 (Training, {total_steps} steps): A data + B data mixed per schedule. "
        f"All schedules see the same total B data. "
        f"Phase 2 (Undo, {undo_steps} steps): B removed, A only. We measure forgetting speed.")

    pdf.sub("Metrics")
    pdf.bul("A comp: compositional accuracy on known functions")
    pdf.bul("B comp: accuracy on b* chains (acquisition + retention)")
    pdf.bul("Peak B: b* accuracy at end of training")
    pdf.bul(f"Quarter-life: undo steps until B drops to 25% of peak (capped at {undo_steps})")
    pdf.bul("Undo AUC: area under B curve during undo (lower = faster forgetting)")

    burst_len = max(int(p_target * total_steps), 1)
    p_pct = int(p_target * 100)

    pdf.sub("The Schedules")
    pdf.bul(f"end_block: 100% B block at the end ({burst_len} steps)")
    pdf.bul(f"uniform: ~{p_pct}% B randomly throughout training")
    pdf.bul(f"mid_block: 100% B block in the middle ({burst_len} steps)")

    end_mixed_50_win = min(int(burst_len / 0.50), total_steps)
    pdf.bul(f"end_mixed_50b: 50% B at the end ({end_mixed_50_win} steps)")

    end_mixed_75b_win = min(int(burst_len / 0.75), total_steps)
    pdf.bul(f"end_mixed_75b: 75% B at the end ({end_mixed_75b_win} steps)")

    end_mixed_25b_win = min(int(burst_len / 0.25), total_steps)
    pdf.bul(f"end_mixed_25b: 25% B at the end ({end_mixed_25b_win} steps)")

    ramp_max_frac = 0.20
    ramp_len = min(int(2 * burst_len / ramp_max_frac), total_steps)
    pdf.bul(f"ramp_up: B ramps from 0% to {int(ramp_max_frac*100)}% at the end ({ramp_len} steps)")

    pdf.add_page()
    pdf.stitle("Learning Rate Schedule")
    pdf.chart(plots_dir / "lr_schedule.png", w=240)
    warmup_iters = bcfg['warmup_iters']
    lr_max = bcfg['lr']
    lr_min = bcfg['min_lr']
    total_train_undo = total_steps + undo_steps
    pdf.body(
        f"Cosine decay with linear warmup. Ramps up during the first "
        f"{warmup_iters} steps, then decays from {lr_max} to "
        f"{lr_min} over {total_train_undo} steps. The undo phase continues the same "
        f"schedule -- the model keeps learning on A data at a low rate.")

    pdf.add_page()
    pdf.stitle("Summary: Forgetting Speed by Schedule")
    pdf.chart(plots_dir / "summary_bars.png", w=260)
    pdf.body(
        "Left: Peak B accuracy. Center: Quarter-life (lower = faster forgetting). "
        "Right: Undo AUC (secondary). Schedules that deliver B near the end "
        "achieve high acquisition. Mixed schedules retain B longer because A "
        "is present alongside B during the burst.")

    pdf.add_page()
    pdf.stitle("Ranking: Fastest Forgetting First")
    rows = sorted(results, key=lambda r: r.get("quarter_life", undo_steps))
    pdf.set_font("Courier", "", 7.5); pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 4,
             f"  {'Rank':<5}{'Schedule':<16}{'Peak B':>8}{'t1/4':>8}{'AUC':>7}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 4, "  " + "-" * 44, new_x="LMARGIN", new_y="NEXT")
    for i, r in enumerate(rows):
        ql = r.get("quarter_life", undo_steps)
        ql_str = f"{ql:.0f}" if ql < undo_steps else f">{undo_steps}"
        pdf.cell(0, 4,
                 f"  {i+1:<5}{r['schedule']:<16}{r.get('train_end_B_comp',0):>8.3f}"
                 f"{ql_str:>8}{r.get('undo_auc',0):>7.0f}",
                 new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.add_page()
    pdf.stitle("Accuracy Overlay - All Schedules")
    pdf.body(
        "Each page shows one accuracy metric with all schedules overlaid. "
        "Lines show mean accuracy across seeds, with shaded regions showing "
        "95% confidence intervals. Vertical dashed line marks "
        "the start of the undo phase.")

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
        "the start of the undo phase.")

    for sched in sorted(set(r["schedule"] for r in results)):
        overlay_path = plots_dir / f"overlay_{sched}.png"
        if overlay_path.exists():
            pdf.sub(f"Schedule: {sched}")
            pdf.chart(overlay_path, w=240)

    pdf.add_page()
    pdf.stitle("Per-Run Details")
    pdf.body(
        "Each plot: (top) schedule bar with A/B percentages, "
        "(middle) 5 accuracy curves with metrics, (bottom) training loss.")
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

    csv_files = list(stats_dir.glob("*.csv"))
    if not csv_files:
        print("  No CSV files found in task_distributions, skipping...")
        return []

    all_data = []
    for csv_file in csv_files:
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["count"] = int(row["count"])
                row["f1"] = int(row["f1"])
                row["f2"] = int(row["f2"])
                row["f3"] = int(row["f3"])
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

    schedules = sorted(set(row["schedule"] for row in all_data))
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
            f3_vals = set()
            f2_vals = set()
            f1_vals = set()

            for row in filtered:
                tt = row["task_type"]
                type_counts[tt] = type_counts.get(tt, 0) + row["count"]
                compositions.add(row["composition"])
                f3_vals.add(row["f3"])
                f2_vals.add(row["f2"])
                f1_vals.add(row["f1"])

            total = sum(type_counts.values())
            a_count = type_counts.get("a3", 0)
            b_count = type_counts.get("b3", 0)

            summary_rows.append({
                "schedule": schedule,
                "phase": phase,
                "total_samples": total,
                "a_samples": a_count,
                "b_samples": b_count,
                "a_fraction": a_count / total if total > 0 else 0,
                "b_fraction": b_count / total if total > 0 else 0,
                "unique_compositions": len(compositions),
                "unique_f3": len(f3_vals),
                "unique_f2": len(f2_vals),
                "unique_f1": len(f1_vals),
            })

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

    for schedule in schedules:
        for seed in seeds:
            run_data = [row for row in all_data
                       if row["schedule"] == schedule and row["seed"] == seed]
            if not run_data:
                continue

            label = f"{schedule}_s{seed}"

            for phase in phases:
                phase_data = [row for row in run_data if row["phase"] == phase]
                if not phase_data:
                    continue

                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                fig.suptitle(f"{label} - {phase}", fontsize=16, fontweight="bold")

                type_counts = {}
                for row in phase_data:
                    tt = row["task_type"]
                    type_counts[tt] = type_counts.get(tt, 0) + row["count"]

                types = sorted(type_counts.keys())
                colors = ["#2196F3" if t == "a3" else "#E91E63" for t in types]
                axes[0, 0].bar(types, [type_counts[t] for t in types], color=colors)
                axes[0, 0].set_title("A vs B Tasks")
                axes[0, 0].set_xlabel("Task Type")
                axes[0, 0].set_ylabel("Count")
                for i, t in enumerate(types):
                    axes[0, 0].text(i, type_counts[t], str(type_counts[t]),
                                   ha="center", va="bottom")

                f3_counts = {}
                for row in phase_data:
                    f3_counts[row["f3"]] = f3_counts.get(row["f3"], 0) + row["count"]
                f3_sorted = sorted(f3_counts.items())
                axes[0, 1].bar([str(f) for f, _ in f3_sorted],
                              [c for _, c in f3_sorted], color="#4CAF50")
                axes[0, 1].set_title("Distribution by F3 (position 3)")
                axes[0, 1].set_xlabel("Function ID")
                axes[0, 1].set_ylabel("Count")
                axes[0, 1].tick_params(axis='x', rotation=45)

                f2_counts = {}
                for row in phase_data:
                    f2_counts[row["f2"]] = f2_counts.get(row["f2"], 0) + row["count"]
                f2_sorted = sorted(f2_counts.items())
                axes[0, 2].bar([str(f) for f, _ in f2_sorted],
                              [c for _, c in f2_sorted], color="#FF9800")
                axes[0, 2].set_title("Distribution by F2 (position 2)")
                axes[0, 2].set_xlabel("Function ID")
                axes[0, 2].set_ylabel("Count")
                axes[0, 2].tick_params(axis='x', rotation=45)

                f1_counts = {}
                for row in phase_data:
                    f1_counts[row["f1"]] = f1_counts.get(row["f1"], 0) + row["count"]
                f1_sorted = sorted(f1_counts.items())
                axes[1, 0].bar([str(f) for f, _ in f1_sorted],
                              [c for _, c in f1_sorted], color="#9C27B0")
                axes[1, 0].set_title("Distribution by F1 (position 1)")
                axes[1, 0].set_xlabel("Function ID")
                axes[1, 0].set_ylabel("Count")
                axes[1, 0].tick_params(axis='x', rotation=45)

                comp_counts = {}
                for row in phase_data:
                    comp_counts[row["composition"]] = comp_counts.get(row["composition"], 0) + row["count"]
                top_comps = sorted(comp_counts.items(), key=lambda x: x[1], reverse=True)[:20]
                axes[1, 1].barh(range(len(top_comps)), [c for _, c in top_comps], color="#00BCD4")
                axes[1, 1].set_yticks(range(len(top_comps)))
                axes[1, 1].set_yticklabels([comp for comp, _ in top_comps], fontsize=8)
                axes[1, 1].set_title("Top 20 Compositions")
                axes[1, 1].set_xlabel("Count")
                axes[1, 1].invert_yaxis()

                combo_counts = {}
                for row in phase_data:
                    combo = f"({row['f3']},{row['f2']},{row['f1']})"
                    key = (row["task_type"], combo)
                    combo_counts[key] = combo_counts.get(key, 0) + row["count"]
                top_combos = sorted(combo_counts.items(), key=lambda x: x[1], reverse=True)[:15]

                labels = [f"{tt}: {combo}" for (tt, combo), _ in top_combos]
                colors = ["#2196F3" if tt == "a3" else "#E91E63" for (tt, _), _ in top_combos]
                axes[1, 2].barh(range(len(top_combos)), [c for _, c in top_combos], color=colors)
                axes[1, 2].set_yticks(range(len(top_combos)))
                axes[1, 2].set_yticklabels(labels, fontsize=7)
                axes[1, 2].set_title("Top 15 Type+Function Combos")
                axes[1, 2].set_xlabel("Count")
                axes[1, 2].invert_yaxis()

                plt.tight_layout()
                fname = charts_dir / f"{label}_{phase}_distribution.png"
                plt.savefig(fname, dpi=150, bbox_inches="tight")
                plt.close()
                generated_files.append(fname)

    for schedule in schedules:
        for phase in phases:
            sched_phase_data = [row for row in all_data
                               if row["schedule"] == schedule and row["phase"] == phase]
            if not sched_phase_data:
                continue

            comp_stats = defaultdict(lambda: {"counts": [], "task_type": None, "f3": None, "f2": None, "f1": None})
            for row in sched_phase_data:
                key = (row["task_type"], row["f3"], row["f2"], row["f1"], row["composition"])
                comp_stats[key]["counts"].append(row["count"])
                comp_stats[key]["task_type"] = row["task_type"]
                comp_stats[key]["f3"] = row["f3"]
                comp_stats[key]["f2"] = row["f2"]
                comp_stats[key]["f1"] = row["f1"]
                comp_stats[key]["composition"] = row["composition"]

            comp_summary = []
            for key, data in comp_stats.items():
                counts = data["counts"]
                comp_summary.append({
                    "task_type": data["task_type"],
                    "f3": data["f3"],
                    "f2": data["f2"],
                    "f1": data["f1"],
                    "composition": data["composition"],
                    "mean": np.mean(counts),
                    "std": np.std(counts) if len(counts) > 1 else 0,
                })

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"{schedule} - {phase} (averaged across seeds)", fontsize=14, fontweight="bold")

            type_means = {}
            for item in comp_summary:
                tt = item["task_type"]
                type_means[tt] = type_means.get(tt, 0) + item["mean"]
            types = sorted(type_means.keys())
            colors = ["#2196F3" if t == "a3" else "#E91E63" for t in types]
            axes[0, 0].bar(types, [type_means[t] for t in types], color=colors)
            axes[0, 0].set_title("A vs B Tasks (mean)")
            axes[0, 0].set_xlabel("Task Type")
            axes[0, 0].set_ylabel("Mean Count")
            for i, t in enumerate(types):
                axes[0, 0].text(i, type_means[t], f"{type_means[t]:.0f}", ha="center", va="bottom")

            f1_means = {}
            f2_means = {}
            f3_means = {}
            for item in comp_summary:
                f1_means[item["f1"]] = f1_means.get(item["f1"], 0) + item["mean"]
                f2_means[item["f2"]] = f2_means.get(item["f2"], 0) + item["mean"]
                f3_means[item["f3"]] = f3_means.get(item["f3"], 0) + item["mean"]

            all_funcs = sorted(set(list(f1_means.keys()) + list(f2_means.keys()) + list(f3_means.keys())))
            x_pos = np.arange(len(all_funcs))
            width = 0.25

            axes[0, 1].bar(x_pos, [f3_means.get(f, 0) for f in all_funcs], width,
                          label="F3", alpha=0.8, color="#4CAF50")
            axes[0, 1].bar(x_pos + width, [f2_means.get(f, 0) for f in all_funcs], width,
                          label="F2", alpha=0.8, color="#FF9800")
            axes[0, 1].bar(x_pos + 2*width, [f1_means.get(f, 0) for f in all_funcs], width,
                          label="F1", alpha=0.8, color="#9C27B0")
            axes[0, 1].set_title("Function Usage by Position")
            axes[0, 1].set_xlabel("Function ID")
            axes[0, 1].set_ylabel("Mean Count")
            axes[0, 1].set_xticks(x_pos + width)
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

            type_f3_means = {}
            for item in comp_summary:
                key = (item["task_type"], item["f3"])
                type_f3_means[key] = type_f3_means.get(key, 0) + item["mean"]

            task_types = sorted(set(tt for tt, _ in type_f3_means.keys()))
            f3_vals = sorted(set(f3 for _, f3 in type_f3_means.keys()))

            x_pos = np.arange(len(f3_vals))
            width = 0.35
            for i, tt in enumerate(task_types):
                vals = [type_f3_means.get((tt, f3), 0) for f3 in f3_vals]
                axes[1, 1].bar(x_pos + i*width, vals, width,
                              label=f"Type {tt}", alpha=0.8,
                              color="#2196F3" if tt == "a3" else "#E91E63")
            axes[1, 1].set_title("Task Type × F3 Distribution")
            axes[1, 1].set_xlabel("F3 Function ID")
            axes[1, 1].set_ylabel("Mean Count")
            axes[1, 1].set_xticks(x_pos + width/2)
            axes[1, 1].set_xticklabels([str(f) for f in f3_vals], rotation=45)
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

    print("Overlay per schedule...")
    plot_overlay_per_schedule(results, plots_dir)

    print("Overlay all schedules...")
    plot_overlay_all_schedules(results, plots_dir)

    print("LR schedule...")
    plot_lr_schedule(cfg.get("base_cfg", cfg), plots_dir)

    print("Task distributions...")
    plot_task_distributions(run_dir)

    print("PDF report...")
    make_report(run_dir, results, cfg, per_run_fnames)
    print("\nDone.")


if __name__ == "__main__":
    main()
