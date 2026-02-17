"""Plot + PDF report for depth-3 bijection burst experiment.

Usage: python burst/plot_new_split.py data/burst_d3_<timestamp>
"""
import sys, os, pickle, json, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from fpdf import FPDF
from collections import defaultdict
from burst._worker_new_split import n_target_for_step

EVAL_KEYS = ["acc_A_comp", "acc_A_heldout", "acc_B_comp", "acc_B_heldout"]
CURVE_STYLE = {
    "acc_A_comp":    {"color": "#2196F3", "ls": "-",  "label": "A comp (train)"},
    "acc_A_heldout": {"color": "#FF9800", "ls": "-",  "label": "A comp (held)"},
    "acc_B_comp":    {"color": "#E91E63", "ls": "-",  "label": "B comp (train)"},
    "acc_B_heldout": {"color": "#9C27B0", "ls": "-",  "label": "B comp (held)"},
}
SCHED_COLORS = {
    "uniform": "#2196F3", "end_block": "#F44336", "mid_block": "#9C27B0",
    "end_mixed_50": "#FF9800", "end_mixed_75b": "#E91E63", "end_mixed_25b": "#009688",
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
    ax.imshow(fracs.reshape(1, -1), aspect="auto", cmap="RdYlBu_r",
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
    nb = result.get("n_b_seen", "?")
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
    ql = result.get("quarter_life", 400)
    ql_str = f"{ql:.0f}" if ql < 400 else ">400"
    drop = result.get("dropoff_abs", 0)
    drop_pct = result.get("dropoff_pct", 0)
    ax.text(T + U * 0.5, 0.95,
            f"peak={peak:.3f}  t1/4={ql_str}  drop={drop:.3f}({drop_pct:.0f}%)",
            ha="center", fontsize=7, color="#D32F2F", fontweight="bold",
            transform=ax.get_xaxis_transform())
    if ql < 400:
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


def plot_summary_bars(results, plots_dir):
    scheds = [r["schedule"] for r in results]
    peaks = [r.get("train_end_B_comp", 0) for r in results]
    quarterlives = [r.get("quarter_life", 400) for r in results]
    aucs = [r.get("undo_auc", 0) for r in results]
    colors = [SCHED_COLORS.get(s, "gray") for s in scheds]
    xs = np.arange(len(scheds))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Peak B Accuracy + Quarter-life + AUC by Schedule", fontsize=14, fontweight="bold")

    titles = ["Peak B Accuracy at step 600",
              "Quarter-life t1/4 (lower = faster forgetting)",
              "Undo AUC (lower = faster forgetting)"]
    ylabels = ["Peak B comp accuracy", "Quarter-life (undo steps)", "Undo AUC"]
    data = [peaks, quarterlives, aucs]

    for ax, vals, title, ylabel in zip(axes, data, titles, ylabels):
        bars = ax.bar(xs, vals, color=colors, edgecolor="black", lw=0.5)
        for b, v in zip(bars, vals):
            lbl = f"{v:.3f}" if max(vals) <= 1.5 else (f"{v:.0f}" if v < 400 else ">400")
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(vals) * 0.01,
                    lbl, ha="center", fontsize=7, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels(scheds, fontsize=8, rotation=25, ha="right")
        ax.grid(True, alpha=0.2, axis="y")

    axes[1].axhline(400, color="gray", ls=":", alpha=0.5)
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
    nb_seen = cfg.get("nb_seen", 10)

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
             f"4-layer Transformer (96-dim, 4 heads)  |  "
             f"600 train + 400 undo steps  |  batch {bcfg['batch_size']}  |  {len(results)} runs",
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
    pdf.body(
        "4 bijection functions. The model trains on all 4x4x4 = 64 three-function "
        "chains. 80% for training, 20% held out for generalisation testing.")

    pdf.sub("Burst Data (B = the new thing to learn)")
    pdf.body(
        f"One brand-new function (b*) placed at position 3. Of the 16 possible "
        f"pairs for positions 1-2, the model sees {nb_seen} during the burst and "
        f"the remaining {16 - nb_seen} are held out.")

    pdf.sub("The Experiment")
    pdf.body(
        "Phase 1 (Training, 600 steps): A data + B data mixed per schedule. "
        "All schedules see the same total B data. "
        "Phase 2 (Undo, 400 steps): B removed, A only. We measure forgetting speed.")

    pdf.sub("Metrics")
    pdf.bul("A comp train/held: compositional accuracy on known functions")
    pdf.bul("B comp train/held: accuracy on b* chains (acquisition + retention)")
    pdf.bul("Peak B: b* accuracy at end of training")
    pdf.bul("Quarter-life: undo steps until B drops to 25% of peak (capped at 400)")
    pdf.bul("Undo AUC: area under B curve during undo (lower = faster forgetting)")

    pdf.sub("The 7 Schedules")
    pdf.bul("end_block: 100% B block at the end (60 steps)")
    pdf.bul("uniform: ~10% B randomly throughout training")
    pdf.bul("mid_block: 100% B block in the middle")
    pdf.bul("end_mixed_50: 50% B at the end (120 steps)")
    pdf.bul("end_mixed_75b: 75% B at the end (80 steps)")
    pdf.bul("end_mixed_25b: 25% B at the end (240 steps)")
    pdf.bul("ramp_up: B ramps from 0% to 20% at the end")

    pdf.add_page()
    pdf.stitle("Learning Rate Schedule")
    pdf.chart(plots_dir / "lr_schedule.png", w=240)
    pdf.body(
        f"Cosine decay with linear warmup. Ramps up during the first "
        f"{bcfg['warmup_iters']} steps, then decays from {bcfg['lr']} to "
        f"{bcfg['min_lr']} over 1000 steps. The undo phase continues the same "
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
    rows = sorted(results, key=lambda r: r.get("quarter_life", 400))
    pdf.set_font("Courier", "", 7.5); pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 4,
             f"  {'Rank':<5}{'Schedule':<16}{'Peak B':>8}{'t1/4':>8}{'AUC':>7}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 4, "  " + "-" * 44, new_x="LMARGIN", new_y="NEXT")
    for i, r in enumerate(rows):
        ql = r.get("quarter_life", 400)
        ql_str = f"{ql:.0f}" if ql < 400 else ">400"
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
    plot_summary_bars(results, plots_dir)

    print("Overlay per schedule...")
    plot_overlay_per_schedule(results, plots_dir)

    print("Overlay all schedules...")
    plot_overlay_all_schedules(results, plots_dir)

    print("LR schedule...")
    plot_lr_schedule(cfg.get("base_cfg", cfg), plots_dir)

    print("PDF report...")
    make_report(run_dir, results, cfg, per_run_fnames)
    print("\nDone.")


if __name__ == "__main__":
    main()
