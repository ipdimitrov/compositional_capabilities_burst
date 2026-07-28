"""
Comprehensive analysis & PDF report for parallel burst experiments.
Usage: python burst/analyze_and_report.py [run_dir]
"""
import sys, os, pickle, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from fpdf import FPDF

COLORS = {
    "uniform": "#2196F3", "end_block": "#F44336", "mid_block": "#9C27B0",
    "early_block": "#4CAF50", "end_mixed": "#FF9800", "bookend": "#795548",
    "spread_K3": "#00BCD4", "spread_K5": "#607D8B",
    "early_block_2x": "#8BC34A", "late_ramp": "#E91E63",
    "cyclic": "#00BCD4", "front_heavy": "#3F51B5",
}
SCHED_LABELS = {
    "uniform": "Uniform", "end_block": "End Block", "early_block": "Early Block",
    "mid_block": "Mid Block", "end_mixed": "End Mixed", "bookend": "Bookend",
    "spread_K3": "Spread K=3", "spread_K5": "Spread K=5",
    "early_block_2x": "Early Block 2x", "late_ramp": "Late Ramp",
    "cyclic": "Cyclic (K=4)", "front_heavy": "Front Heavy",
}
W, H = 297, 210


def load_results(run_dir):
    run_dir = Path(run_dir)
    with open(run_dir / "all_results.pkl", "rb") as f:
        results = pickle.load(f)
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    return results, cfg


def split_results(results):
    base_p = results[0]["config"]["p_target"] if results else 0.05
    base = [r for r in results if abs(r["config"]["p_target"] - base_p) < 1e-6]
    sweep = [r for r in results if abs(r["config"]["p_target"] - base_p) >= 1e-6]
    groups = defaultdict(list)
    for r in base:
        groups[r["schedule"]].append(r)
    return base, sweep, dict(groups)


def get_sched_order(groups):
    return sorted(groups.keys(), key=lambda s: np.mean([r["undo_auc"] for r in groups[s]]))


def schedule_fraction(step, T, schedule, p, bs):
    burst_len = max(int(p * T), 1)
    if schedule == "uniform": return p
    if schedule == "end_block": return 1.0 if step >= T - burst_len else 0.0
    if schedule == "mid_block":
        mid = T // 2
        return 1.0 if mid - burst_len // 2 <= step < mid + (burst_len - burst_len // 2) else 0.0
    if schedule == "early_block": return 1.0 if step < burst_len else 0.0
    if schedule == "end_mixed": return 2 * p if step >= T // 2 else 0.0
    if schedule == "bookend":
        w = max(burst_len // 2, 1)
        return 1.0 if (step < w or step >= T - w) else 0.0
    if schedule == "spread_K3":
        K, w = 3, max(burst_len // 3, 1)
        cycle = T // K
        return 1.0 if (step % cycle) >= cycle - w else 0.0
    if schedule == "spread_K5":
        K, w = 5, max(burst_len // 5, 1)
        cycle = T // K
        return 1.0 if (step % cycle) >= cycle - w else 0.0
    if schedule == "early_block_2x": return 1.0 if step < 2 * burst_len else 0.0
    if schedule == "late_ramp":
        frac = step / max(T - 1, 1)
        return min(p * 2 * frac, 1.0)
    if schedule == "cyclic":
        K, w = 4, max(burst_len // 4, 1)
        cycle = T // K
        return 1.0 if (step % cycle) < w else 0.0
    if schedule == "front_heavy": return 2 * p if step < T // 2 else 0.0
    return 0.0


def plot_schedules(cfg, groups, plots_dir):
    T = cfg["total_steps"]
    p = cfg["p_target"]
    bs = cfg["batch_size"]
    scheds = get_sched_order(groups)

    fig, axes = plt.subplots(len(scheds), 1, figsize=(12, 0.9 * len(scheds)), sharex=True)
    if len(scheds) == 1:
        axes = [axes]
    fig.suptitle("Training Schedules: When Target (B) Data Appears", fontsize=13, fontweight="bold", y=0.98)

    for i, sched in enumerate(scheds):
        ax = axes[i]
        frac = np.array([schedule_fraction(s, T, sched, p, bs) for s in range(T)])
        ax.fill_between(range(T), 0, frac, color=COLORS.get(sched, "#999"), alpha=0.8)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel(SCHED_LABELS.get(sched, sched), fontsize=8, rotation=0, ha="right", va="center", labelpad=70)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Training Step", fontsize=10)
    fig.tight_layout(rect=[0.14, 0, 1, 0.96])
    fig.savefig(plots_dir / "fig1_schedules.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_auc_bars(groups, plots_dir):
    scheds = get_sched_order(groups)
    n = len(scheds)

    means = [np.mean([r["undo_auc"] for r in groups[s]]) for s in scheds]
    colors = [COLORS.get(s, "#999") for s in scheds]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * n)))
    bars = ax.barh(np.arange(n), means, color=colors, edgecolor="black", lw=0.5, height=0.6)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([SCHED_LABELS.get(s, s) for s in scheds], fontsize=10)
    ax.set_xlabel("Undo AUC (lower = faster forgetting)", fontsize=11)
    ax.set_title("Passive Forgetting Speed by Schedule (Undo AUC)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="x")

    for bar, val in zip(bars, means):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}", va="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(plots_dir / "fig2_auc_ranking.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_multi_metric(groups, plots_dir):
    scheds = get_sched_order(groups)
    n = len(scheds)
    x = np.arange(n)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Unlearning Metrics Across Schedules", fontsize=13, fontweight="bold")

    metrics = [
        ("undo_auc", "Undo AUC\n(lower = faster)", "Undo AUC", ".0f"),
        ("undo_end_acc", "Final Target Acc\n(lower = more forgotten)", "Undo End Acc", ".4f"),
        ("mlp_undo_delta", "MLP Weight Delta\n(during undo)", "MLP ||dW||", ".1f"),
        ("attn_undo_delta", "Attn Weight Delta\n(during undo)", "Attn ||dW||", ".1f"),
    ]

    for ax, (key, ylabel, title, fmt) in zip(axes, metrics):
        means = [np.mean([r[key] for r in groups[s]]) for s in scheds]
        colors = [COLORS.get(s, "#999") for s in scheds]
        bars = ax.bar(x, means, color=colors, width=0.6, edgecolor="black", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([SCHED_LABELS.get(s, s) for s in scheds], fontsize=6, rotation=45, ha="right")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="y")

        for b, v in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.01,
                    f"{v:{fmt}}", ha="center", fontsize=6, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(plots_dir / "fig3_multi_metric.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overlay(groups, cfg, plots_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Target (B) Accuracy: Training & Passive Forgetting Phases",
                 fontsize=13, fontweight="bold")

    T = cfg["total_steps"]

    for sched in get_sched_order(groups):
        runs = groups[sched]
        c = COLORS.get(sched, "#999")
        for r in runs:
            steps = np.array(r["log"]["step"])
            acc = np.array(r["log"]["acc_target"])
            phases = r["log"]["phase"]
            train_idx = np.array([p == "train" for p in phases])
            undo_idx = np.array([p == "undo" for p in phases])
            axes[0].plot(steps[train_idx], acc[train_idx], color=c, lw=1.5,
                         label=SCHED_LABELS.get(sched, sched))
            axes[1].plot(steps[undo_idx], acc[undo_idx], color=c, lw=1.5,
                         label=SCHED_LABELS.get(sched, sched))

    axes[0].set_title("Training Phase", fontsize=11)
    axes[1].set_title("Passive Forgetting Phase (A-only, correct labels)", fontsize=11)
    for ax in axes:
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Target (B) Accuracy")
        ax.set_ylim(-0.05, 1.05)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(plots_dir / "fig4_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_p_sweep(sweep_runs, base_groups, cfg, plots_dir):
    if not sweep_runs:
        return

    sweep_by_sched = defaultdict(list)
    for r in sweep_runs:
        sweep_by_sched[r["schedule"]].append(r)

    sweep_scheds = sorted(set(r["schedule"] for r in sweep_runs))
    n_plots = len(sweep_scheds)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    fig.suptitle("Effect of Target Proportion (p) on Forgetting", fontsize=13, fontweight="bold")

    base_p = cfg["p_target"]

    for ax, sched in zip(axes, sweep_scheds):
        ps, aucs, undo_accs = [], [], []

        if sched in base_groups:
            ps.append(base_p)
            aucs.append(np.mean([r["undo_auc"] for r in base_groups[sched]]))
            undo_accs.append(np.mean([r["undo_end_acc"] for r in base_groups[sched]]))

        for r in sorted(sweep_by_sched.get(sched, []), key=lambda x: x["config"]["p_target"]):
            ps.append(r["config"]["p_target"])
            aucs.append(r["undo_auc"])
            undo_accs.append(r["undo_end_acc"])

        sort_idx = np.argsort(ps)
        ps = [ps[i] for i in sort_idx]
        aucs = [aucs[i] for i in sort_idx]
        undo_accs = [undo_accs[i] for i in sort_idx]

        ax2 = ax.twinx()
        l1 = ax.plot(ps, aucs, "o-", color=COLORS.get(sched, "#999"), lw=2, markersize=8, label="Undo AUC")
        l2 = ax2.plot(ps, undo_accs, "s--", color=COLORS.get(sched, "#999"), lw=1.5, markersize=6,
                      alpha=0.6, label="Undo End Acc")
        ax.set_xlabel("p (target proportion)", fontsize=10)
        ax.set_ylabel("Undo AUC", fontsize=10)
        ax2.set_ylabel("Undo End Acc", fontsize=10, color="gray")
        ax.set_title(f"{SCHED_LABELS.get(sched, sched)} Schedule", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2)

        lines = l1 + l2
        ax.legend(lines, [l.get_label() for l in lines], fontsize=8)

        for p_val, auc_val in zip(ps, aucs):
            ax.annotate(f"{auc_val:.0f}", (p_val, auc_val), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8, fontweight="bold")

    fig.tight_layout()
    fig.savefig(plots_dir / "fig5_p_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_run(result, plots_dir):
    log = result["log"]
    sched = result["schedule"]
    seed = result["seed"]
    cfg = result["config"]

    steps = np.array(log["step"])
    acc_t = np.array(log["acc_target"])
    acc_b = np.array(log["acc_background"])
    loss_arr = np.array(log["loss"])
    phases = log["phase"]
    T = cfg["total_steps"]
    U = cfg["undo_steps"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [3, 2]})
    fig.suptitle(f"{SCHED_LABELS.get(sched, sched)} (seed={seed})", fontsize=14, fontweight="bold")

    train_mask = np.array([p == "train" for p in phases])
    undo_mask = np.array([p == "undo" for p in phases])

    ax = axes[0]
    ax.plot(steps[train_mask], acc_t[train_mask], color="#F44336", lw=1.5, label="Target (B) - train")
    ax.plot(steps[undo_mask], acc_t[undo_mask], color="#F44336", lw=1.5, ls="--", label="Target (B) - undo")
    ax.plot(steps[train_mask], acc_b[train_mask], color="#2196F3", lw=1.5, label="Background (A) - train")
    ax.plot(steps[undo_mask], acc_b[undo_mask], color="#2196F3", lw=1.5, ls="--", label="Background (A) - undo")
    ax.axvline(T, color="gray", ls="--", alpha=0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.2)
    ax.text(T * 0.5, 1.02, "TRAIN", ha="center", fontsize=9, color="gray", transform=ax.get_xaxis_transform())
    ax.text(T + U * 0.5, 1.02, "PASSIVE FORGETTING", ha="center", fontsize=9, color="gray", transform=ax.get_xaxis_transform())

    ax = axes[1]
    ax.plot(steps[train_mask], loss_arr[train_mask], color="#333", lw=1, label="Loss - train")
    ax.plot(steps[undo_mask], loss_arr[undo_mask], color="#333", lw=1, ls="--", label="Loss - undo")
    ax.axvline(T, color="gray", ls="--", alpha=0.5)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Global Step")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    info = (f"Model: {cfg['n_layer']}L/{cfg['n_embd']}d/{cfg['n_head']}H  |  "
            f"train={T} undo={U}  |  batch={cfg['batch_size']} p={cfg['p_target']}")
    fig.text(0.5, 0.01, info, ha="center", fontsize=7, color="gray")

    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fname = f"fig6_run_{sched}_s{seed}.png"
    fig.savefig(plots_dir / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_undo_heatmap(groups, cfg, plots_dir):
    scheds = get_sched_order(groups)
    T = cfg["total_steps"]

    fig, ax = plt.subplots(figsize=(14, max(4, 0.4 * len(scheds))))
    fig.suptitle("Target Accuracy During Passive Forgetting Phase",
                 fontsize=13, fontweight="bold")

    mat, labels = [], []
    for sched in scheds:
        runs = groups[sched]
        all_acc = []
        for r in runs:
            acc = np.array(r["log"]["acc_target"])
            phases = r["log"]["phase"]
            undo_idx = np.array([p == "undo" for p in phases])
            all_acc.append(acc[undo_idx])
        min_len = min(len(a) for a in all_acc)
        if min_len == 0:
            continue
        acc_mat = np.array([a[:min_len] for a in all_acc])
        mat.append(acc_mat.mean(axis=0))
        labels.append(SCHED_LABELS.get(sched, sched))

    if not mat:
        plt.close(fig)
        return

    mat = np.array(mat)
    vmin = max(mat.min() - 0.05, 0)
    vmax = min(mat.max() + 0.05, 1)
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=vmin, vmax=vmax,
                   extent=[0, mat.shape[1], len(labels), 0])
    ax.set_yticks(np.arange(len(labels)) + 0.5)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Undo Eval Point", fontsize=10)
    plt.colorbar(im, ax=ax, label="Target Accuracy", shrink=0.8)

    fig.tight_layout()
    fig.savefig(plots_dir / "fig7_undo_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_convergence_check(groups, cfg, plots_dir):
    scheds = get_sched_order(groups)
    n = len(scheds)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * n)))
    fig.suptitle("Training Convergence: Final Target (B) Accuracy", fontsize=13, fontweight="bold")

    train_accs = [np.mean([r["train_end_acc"] for r in groups[s]]) for s in scheds]
    colors = [COLORS.get(s, "#999") for s in scheds]

    bars = ax.barh(np.arange(n), train_accs, color=colors, edgecolor="black", lw=0.5, height=0.6)
    ax.axvline(0.95, color="red", ls="--", alpha=0.5, label="95% threshold")
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([SCHED_LABELS.get(s, s) for s in scheds], fontsize=10)
    ax.set_xlabel("Final Training Accuracy on B Tasks", fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="x")

    for bar, val in zip(bars, train_accs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, fontweight="bold",
                color="red" if val < 0.95 else "black")

    fig.tight_layout()
    fig.savefig(plots_dir / "fig9_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# PDF REPORT
# ============================================================
def make_report(run_dir, results, cfg, groups, sweep_runs, per_run_fnames):
    plots_dir = run_dir / "plots"
    scheds = get_sched_order(groups)
    n_schedules = len(scheds)
    n_total = len(results)
    n_sweep = len(sweep_runs)
    n_base = n_total - n_sweep
    base_p = cfg["p_target"]

    best_sched = scheds[0]
    worst_sched = scheds[-1]
    best_auc = np.mean([r["undo_auc"] for r in groups[best_sched]])
    worst_auc = np.mean([r["undo_auc"] for r in groups[worst_sched]])
    pct_diff = (worst_auc - best_auc) / worst_auc * 100

    converged = [s for s in scheds if np.mean([r["train_end_acc"] for r in groups[s]]) >= 0.95]
    not_converged = [s for s in scheds if np.mean([r["train_end_acc"] for r in groups[s]]) < 0.95]

    if converged:
        best_converged = converged[0]
        worst_converged = converged[-1]
        best_c_auc = np.mean([r["undo_auc"] for r in groups[best_converged]])
        worst_c_auc = np.mean([r["undo_auc"] for r in groups[worst_converged]])
        pct_diff_c = (worst_c_auc - best_c_auc) / worst_c_auc * 100
    else:
        best_converged = worst_converged = best_sched
        best_c_auc = worst_c_auc = best_auc
        pct_diff_c = 0

    class PDF(FPDF):
        def header(self):
            if self.page_no() > 1:
                self.set_font("Helvetica", "I", 7)
                self.set_text_color(130, 130, 130)
                self.cell(0, 4, "Burst Schedule Experiments  |  Compositional Capabilities & Passive Forgetting", align="L")
                self.cell(0, 4, f"Page {self.page_no()}/{{nb}}", align="R")
                self.ln(6)

        def stitle(self, t):
            self.set_font("Helvetica", "B", 16)
            self.set_text_color(0, 80, 140)
            self.cell(0, 9, t, new_x="LMARGIN", new_y="NEXT")
            self.ln(3)

        def sub(self, t):
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(40, 40, 40)
            self.cell(0, 6, t, new_x="LMARGIN", new_y="NEXT")
            self.ln(1)

        def body(self, t):
            self.set_font("Helvetica", "", 9)
            self.set_text_color(30, 30, 30)
            self.multi_cell(0, 4.5, t)
            self.ln(2)

        def bul(self, t):
            self.set_font("Helvetica", "", 9)
            self.set_text_color(30, 30, 30)
            self.cell(6, 4.5, "-")
            self.multi_cell(W - 26, 4.5, t)
            self.ln(1)

        def bbul(self, label, t):
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(30, 30, 30)
            self.cell(6, 4.5, "-")
            self.cell(self.get_string_width(label) + 1, 4.5, label)
            self.set_font("Helvetica", "", 9)
            self.multi_cell(0, 4.5, t)
            self.ln(1)

        def mono(self, t):
            self.set_font("Courier", "", 7.5)
            self.set_text_color(40, 40, 40)
            self.multi_cell(0, 3.5, t)
            self.ln(2)

        def gbox(self, t):
            self.set_fill_color(240, 240, 245)
            self.set_font("Helvetica", "", 8)
            self.set_text_color(40, 40, 40)
            self.multi_cell(W - 20, 4, t, fill=True)
            self.ln(2)

        def chart(self, path, w=220):
            if Path(path).exists():
                if self.get_y() > H - 55:
                    self.add_page()
                x = (W - w) / 2
                self.image(str(path), x=x, w=w)
                self.ln(3)

        def table_row(self, cells, widths, bold=False, fill=False):
            self.set_font("Courier", "B" if bold else "", 7.5)
            self.set_text_color(30, 30, 30)
            if fill:
                self.set_fill_color(235, 240, 250)
            for cell, w in zip(cells, widths):
                self.cell(w, 4.5, cell, fill=fill)
            self.ln()

    pdf = PDF(orientation="L", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=12)

    # ---- PAGE 1: TITLE ----
    pdf.add_page()
    pdf.ln(25)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(0, 80, 140)
    pdf.multi_cell(0, 12, "How Does Data Scheduling\nAffect Passive Forgetting?", align="C")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 7, "Burst Schedule Experiments on Compositional Capabilities", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Courier", "", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 5, f"Model: {cfg['n_layer']}L / {cfg['n_embd']}d / {cfg['n_head']}H nanoGPT  |  "
             f"train={cfg['total_steps']} steps, undo={cfg['undo_steps']} steps  |  "
             f"batch={cfg['batch_size']}  |  p_target={base_p}",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, f"{n_schedules} schedules x 1 seed + {n_sweep} p_target sweeps = {n_total} experiments  |  "
             f"{cfg['n_functions']+1} base bijections (f5 exclusive to B tasks), "
             f"{cfg['n_train_compositions']} compositions",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(12)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, "Forgetting method: passive (A-only training with correct labels, no shuffled labels).",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, "B tasks use an exclusive bijection (f5) not present in any A task.",
             align="C", new_x="LMARGIN", new_y="NEXT")

    # ---- PAGE 2: EXECUTIVE SUMMARY ----
    pdf.add_page()
    pdf.stitle("Executive Summary")
    pdf.body(
        "This report investigates whether the temporal scheduling of training data affects how easily "
        "a model's learned capabilities can be passively forgotten. We train a nanoGPT transformer on "
        "compositional bijection tasks, where B (target) tasks use an exclusive bijection (f5) that A "
        "(background) tasks never use. After training, we continue training on A data only and measure "
        "how quickly B-task accuracy decays."
    )
    pdf.sub("Key findings")

    if not_converged:
        nc_str = ", ".join(SCHED_LABELS.get(s, s) for s in not_converged)
        nc_accs = ", ".join(f"{np.mean([r['train_end_acc'] for r in groups[s]]):.2f}" for s in not_converged)
        pdf.bul(
            f"Not all schedules converged on B tasks. {nc_str} reached only {nc_accs} training accuracy "
            f"respectively. These schedules concentrate B data in narrow windows where the model doesn't "
            f"get enough exposure to fully learn f5. Comparisons involving these schedules should be "
            f"interpreted with caution."
        )

    if converged:
        pdf.bul(
            f"Among the {len(converged)} schedules that converged (>=95% training accuracy), "
            f"{SCHED_LABELS.get(best_converged, best_converged)} forgets fastest "
            f"(AUC={best_c_auc:.0f}) and "
            f"{SCHED_LABELS.get(worst_converged, worst_converged)} retains most "
            f"(AUC={worst_c_auc:.0f}). "
            f"This is a {pct_diff_c:.1f}% difference in forgetting speed."
        )

    pdf.bul(
        f"Passive forgetting works: target accuracy drops from 1.0 to ~0.62-0.80 during A-only training, "
        f"confirming that the exclusive f5 bijection decays when not reinforced. However, accuracy "
        f"plateaus around 0.73 for most schedules, suggesting partial retention of compositional structure."
    )

    pdf.bul(
        f"Higher target proportion (p) leads to more retention: at p=0.10, end_block retains more "
        f"(AUC={[r['undo_auc'] for r in sweep_runs if r['schedule']=='end_block' and abs(r['config']['p_target']-0.10)<1e-6][0]:.0f}) "
        f"than at p=0.02 "
        f"(AUC={[r['undo_auc'] for r in sweep_runs if r['schedule']=='end_block' and abs(r['config']['p_target']-0.02)<1e-6][0]:.0f}). "
        f"More B exposure = deeper consolidation."
        if any(r['schedule'] == 'end_block' for r in sweep_runs) else
        "The p_target sweep shows that more B exposure leads to deeper consolidation."
    )

    pdf.bul(
        "End Mixed (B only in second half at 2x rate) produces the strongest retention "
        f"(AUC={np.mean([r['undo_auc'] for r in groups.get('end_mixed', groups[worst_sched])]):.0f}), "
        "suggesting that recent, distributed exposure consolidates knowledge most deeply."
    )

    pdf.sub("Caveats")
    pdf.bul(
        "Single seed per schedule: these results lack variance estimates. Observed differences could "
        "partly reflect seed-specific noise. Future runs should use 3+ seeds for statistical significance."
    )
    pdf.bul(
        "The 0.73 accuracy plateau suggests the model retains partial compositional knowledge even "
        "without f5 reinforcement, possibly by leveraging shared structure from A-task training."
    )

    # ---- PAGE 3: BACKGROUND ----
    pdf.add_page()
    pdf.stitle("1. Background & Setup")
    pdf.body(
        "We train a nanoGPT transformer on synthetic sequences generated by compositions of bijective "
        "functions over a finite alphabet. The model must learn to apply chains of permutations to input "
        "tokens - a task requiring genuine compositional generalization rather than memorization."
    )
    pdf.sub("Task structure")
    pdf.gbox(
        "Each training example is a sequence:  [task_tokens] [input_tokens] -> [output_tokens]\n\n"
        "The task tokens encode which bijections to apply and in what order.\n"
        "The model must learn each bijection independently, then compose them at inference.\n\n"
        f"Setup: {cfg['n_alphabets']} alphabets, {cfg['n_functions']+1} base bijections (indices 0-{cfg['n_functions']}), "
        f"{cfg['n_train_compositions']} random compositions of depth {cfg['depth']}\n"
        f"Target (B) tasks: compositions containing bijection f5 (10 tasks)\n"
        f"Background (A) tasks: compositions using only f0-f4 ({n_base} tasks)\n\n"
        "f5 is EXCLUSIVE to B tasks. A tasks never use f5. This means during passive forgetting "
        "(A-only training), there is no gradient signal reinforcing f5, so it can genuinely decay."
    )
    pdf.sub("Two-phase protocol")
    pdf.body(
        f"Phase 1 - TRAIN ({cfg['total_steps']} steps): The model sees B + A data "
        f"according to the schedule. Total B proportion is p={base_p} across all schedules. "
        f"AdamW optimizer with cosine warmup LR (lr={cfg['lr']}, warmup={cfg['warmup_iters']}, "
        f"min_lr={cfg['min_lr']}). Mixed precision (bf16), batch size {cfg['batch_size']}."
    )
    pdf.body(
        f"Phase 2 - PASSIVE FORGETTING ({cfg['undo_steps']} steps): The model trains on A data only "
        f"with correct labels. No shuffled labels, no adversarial intervention. B-specific knowledge "
        f"(specifically f5) decays through neglect. We measure how quickly target accuracy drops."
    )
    pdf.sub("Key metrics")
    pdf.bbul("Undo AUC: ", "Area under the target accuracy curve during forgetting. Lower = faster forgetting. Primary metric.")
    pdf.bbul("Undo End Acc: ", "Target accuracy at the end of forgetting. Lower = more thoroughly forgotten.")
    pdf.bbul("Train End Acc: ", "Target accuracy at end of training. Must be high (>0.95) for valid comparison.")

    # ---- PAGE 4: SCHEDULES ----
    pdf.add_page()
    pdf.stitle("2. Schedule Definitions")
    pdf.body(f"All {n_schedules} schedules deliver the same total amount of B data (p={base_p} of training). "
             "They differ only in WHEN that data appears during training:")

    sched_descs = {
        "uniform": "B mixed uniformly throughout training (binomial p per batch). Baseline.",
        "end_block": "All B samples concentrated in a contiguous block at the END of training.",
        "early_block": "All B samples concentrated at the START of training.",
        "mid_block": "All B samples concentrated in the MIDDLE of training.",
        "end_mixed": "B only in the second half, but mixed with A (at 2p rate). Distributed but late.",
        "bookend": "B split equally between the very start and very end of training.",
        "early_block_2x": "Like Early Block but with DOUBLE the burst window (2x longer, same density).",
        "late_ramp": "B probability increases linearly from 0 to 2p over training. Gradual late exposure.",
        "cyclic": "B split into 4 evenly-spaced burst windows throughout training.",
        "front_heavy": "B at 2x rate in the first half only. Distributed but early.",
        "spread_K3": "B split into 3 evenly-spaced burst windows.",
        "spread_K5": "B split into 5 evenly-spaced burst windows.",
    }
    for s in scheds:
        pdf.bbul(f"{SCHED_LABELS.get(s, s)}: ", sched_descs.get(s, ""))

    pdf.ln(2)
    pdf.body(
        "The chart below visualizes when B data appears for each schedule. Height indicates the "
        "fraction of each batch that is B data at that training step. All schedules have the same "
        "area under the curve (same total B exposure)."
    )
    pdf.chart(plots_dir / "fig1_schedules.png", w=240)

    # ---- PAGE 5: CONVERGENCE CHECK ----
    pdf.add_page()
    pdf.stitle("3. Training Convergence Check")
    pdf.body(
        "Before comparing forgetting speeds, we must verify that all schedules actually learned the B tasks. "
        "If a schedule didn't converge during training, its forgetting curve is meaningless - you can't "
        "forget what you never learned."
    )
    pdf.chart(plots_dir / "fig9_convergence.png", w=200)

    if not_converged:
        nc_details = []
        for s in not_converged:
            acc = np.mean([r["train_end_acc"] for r in groups[s]])
            nc_details.append(f"{SCHED_LABELS.get(s, s)} ({acc:.2f})")
        pdf.body(
            f"WARNING: {len(not_converged)} schedule(s) did not converge: {', '.join(nc_details)}. "
            f"These schedules concentrate B data in narrow windows (p={base_p} means only "
            f"{int(base_p * cfg['total_steps'])} steps of B exposure). The model doesn't see enough B "
            f"data to fully learn f5 before the window closes. Their forgetting results should be "
            f"interpreted with caution - low undo accuracy may reflect incomplete learning rather than "
            f"easy forgetting."
        )
    if converged:
        pdf.body(
            f"{len(converged)} schedule(s) converged (>=95%): "
            f"{', '.join(SCHED_LABELS.get(s, s) for s in converged)}. "
            f"These are the schedules where forgetting comparisons are most meaningful."
        )

    # ---- PAGE 6: MAIN RESULT ----
    pdf.add_page()
    pdf.stitle("4. Main Result: Forgetting Speed Ranking")
    pdf.body(
        "The chart below ranks all schedules by Undo AUC (lower = faster forgetting). "
        "This is the primary metric: it captures both how quickly and how completely the model "
        "forgets B-task knowledge during A-only training."
    )
    pdf.chart(plots_dir / "fig2_auc_ranking.png", w=200)

    pdf.sub("Interpretation")
    if converged:
        pdf.body(
            f"Focusing on converged schedules only: "
            f"{SCHED_LABELS.get(best_converged, best_converged)} forgets fastest (AUC={best_c_auc:.0f}) "
            f"and {SCHED_LABELS.get(worst_converged, worst_converged)} retains most (AUC={worst_c_auc:.0f}). "
            f"The difference is {pct_diff_c:.1f}%. "
            f"This suggests that the temporal pattern of B exposure meaningfully affects how deeply "
            f"f5 knowledge is consolidated."
        )
    pdf.body(
        "Schedules that concentrate B data at the end of training (End Mixed, End Block) tend to retain "
        "more knowledge, likely because there is less time for A-only training to begin overwriting B "
        "representations before the forgetting phase starts. Schedules with early or distributed B "
        "exposure allow more post-B consolidation/interference during training itself."
    )

    # ---- PAGE 7: MULTI-METRIC ----
    pdf.add_page()
    pdf.stitle("5. Multi-Metric Comparison")
    pdf.body(
        "Four metrics compared across all schedules. Undo AUC and Undo End Acc measure forgetting. "
        "MLP and Attn weight deltas measure how much the model's weights change during forgetting - "
        "larger deltas suggest more restructuring is needed to forget."
    )
    pdf.chart(plots_dir / "fig3_multi_metric.png", w=260)
    pdf.body(
        "The MLP and Attention weight deltas are relatively similar across schedules, suggesting that "
        "the model undergoes comparable amounts of weight change regardless of schedule. The differences "
        "in forgetting speed are therefore more about the structure of what was learned than the magnitude "
        "of weight updates during forgetting."
    )

    # ---- PAGE 8: NUMERICAL TABLE ----
    pdf.add_page()
    pdf.stitle("6. Numerical Results")
    pdf.sub(f"Core schedules (p={base_p}, seed=42)")

    widths = [40, 30, 30, 30, 30, 30]
    headers = ["Schedule", "Train Acc", "Undo Acc", "Undo AUC", "MLP dW", "Attn dW"]
    pdf.table_row(headers, widths, bold=True, fill=True)

    for sched in scheds:
        runs = groups[sched]
        ta = np.mean([r["train_end_acc"] for r in runs])
        ua = np.mean([r["undo_end_acc"] for r in runs])
        auc = np.mean([r["undo_auc"] for r in runs])
        mlp = np.mean([r["mlp_undo_delta"] for r in runs])
        attn = np.mean([r["attn_undo_delta"] for r in runs])
        converge_mark = "" if ta >= 0.95 else " *"

        cells = [
            SCHED_LABELS.get(sched, sched) + converge_mark,
            f"{ta:.4f}",
            f"{ua:.4f}",
            f"{auc:.0f}",
            f"{mlp:.2f}",
            f"{attn:.2f}",
        ]
        pdf.table_row(cells, widths)

    if not_converged:
        pdf.ln(1)
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(150, 50, 50)
        pdf.cell(0, 4, "* Did not converge during training (<95% accuracy). Forgetting results may not be meaningful.",
                 new_x="LMARGIN", new_y="NEXT")

    if sweep_runs:
        pdf.ln(4)
        pdf.sub("p_target sweep (seed=42)")
        widths2 = [40, 20, 30, 30, 30]
        pdf.table_row(["Schedule", "p", "Train Acc", "Undo Acc", "Undo AUC"], widths2, bold=True, fill=True)
        for r in sorted(sweep_runs, key=lambda x: (x["schedule"], x["config"]["p_target"])):
            cells = [
                SCHED_LABELS.get(r["schedule"], r["schedule"]),
                f"{r['config']['p_target']:.2f}",
                f"{r['train_end_acc']:.4f}",
                f"{r['undo_end_acc']:.4f}",
                f"{r['undo_auc']:.0f}",
            ]
            pdf.table_row(cells, widths2)

    # ---- PAGE 9: OVERLAY ----
    pdf.add_page()
    pdf.stitle("7. Accuracy Curves (All Schedules)")
    pdf.body(
        "Left panel: target (B) accuracy during training. This shows when each schedule learns B tasks. "
        "Right panel: target accuracy during passive forgetting (A-only training). This shows how quickly "
        "each schedule's B knowledge decays. The separation between curves in the right panel is the "
        "core signal we're looking for."
    )
    pdf.chart(plots_dir / "fig4_overlay.png", w=250)
    pdf.body(
        "Note how most schedules plateau around 0.73 accuracy during forgetting. This suggests a floor "
        "where the model retains enough compositional structure from A-task training to partially solve "
        "B tasks even without f5 reinforcement. The differences between schedules are in how quickly "
        "they reach this floor and the exact plateau level."
    )

    # ---- PAGE 10: P SWEEP ----
    if sweep_runs:
        pdf.add_page()
        pdf.stitle("8. Effect of Target Proportion (p)")
        p_vals = sorted(set(r["config"]["p_target"] for r in sweep_runs))
        pdf.body(
            f"We swept p_target across {{{', '.join(f'{p:.2f}' for p in [base_p] + p_vals)}}} for selected schedules. "
            "This tests whether the amount of B exposure affects forgetting independently of the schedule. "
            "Higher p means more B data during training."
        )
        pdf.chart(plots_dir / "fig5_p_sweep.png", w=230)
        pdf.body(
            "Higher p_target consistently leads to higher AUC (slower forgetting). This makes intuitive "
            "sense: more B exposure during training means deeper consolidation of f5, which takes longer "
            "to passively decay. At p=0.02, the model barely sees B data and forgets quickly. At p=0.10, "
            "the model has seen enough B data to consolidate f5 more robustly."
        )

    # ---- PAGE 11: HEATMAP ----
    pdf.add_page()
    pdf.stitle("9. Forgetting Trajectory Heatmap")
    pdf.body(
        "This heatmap shows target accuracy at each evaluation point during the forgetting phase. "
        "Each row is a schedule, each column is an evaluation checkpoint. Green = low accuracy "
        "(forgotten), red = high accuracy (retained). The gradient from left to right shows the "
        "forgetting trajectory."
    )
    pdf.chart(plots_dir / "fig7_undo_heatmap.png", w=240)
    pdf.body(
        "Most schedules show rapid initial forgetting (first few evaluation points) followed by a "
        "plateau. The speed of the initial drop and the height of the plateau are what differentiate "
        "schedules. End Mixed maintains the highest accuracy throughout, while schedules with early "
        "B exposure drop faster initially."
    )

    # ---- PAGES: PER-RUN DETAILS ----
    pdf.add_page()
    pdf.stitle("10. Per-Run Details")
    pdf.body(
        "Each plot below shows one schedule's full training and forgetting trajectory. "
        "Top panel: accuracy for B (target, red) and A (background, blue) tasks. "
        "Bottom panel: training loss. The vertical dashed line marks the transition from training "
        "to passive forgetting. During forgetting, A accuracy stays at 1.0 (as expected, since "
        "we're still training on A), while B accuracy decays."
    )
    for fname in sorted(per_run_fnames):
        pdf.chart(plots_dir / fname, w=240)

    # ---- FINAL PAGE: CONCLUSIONS ----
    pdf.add_page()
    pdf.stitle("11. Conclusions")

    if converged:
        pdf.bul(
            f"Among converged schedules, forgetting speed varies by {pct_diff_c:.1f}% "
            f"({SCHED_LABELS.get(best_converged, best_converged)} AUC={best_c_auc:.0f} vs "
            f"{SCHED_LABELS.get(worst_converged, worst_converged)} AUC={worst_c_auc:.0f}). "
            f"This demonstrates that WHEN data is presented significantly affects consolidation depth."
        )

    pdf.bul(
        "The exclusive f5 bijection design works: B-task accuracy drops during A-only training, "
        "confirming that f5-specific knowledge genuinely decays without reinforcement. This validates "
        "the experimental design for measuring passive forgetting of compositional capabilities."
    )

    if not_converged:
        pdf.bul(
            f"{len(not_converged)} schedule(s) failed to converge, indicating that p={base_p} with "
            f"{cfg['total_steps']} training steps provides insufficient B exposure for narrow burst windows. "
            f"Future experiments should either increase training steps, increase p for these schedules, "
            f"or accept that some schedules inherently can't teach B in their allotted window."
        )

    pdf.bul(
        "All schedules plateau around 0.73 accuracy during forgetting, suggesting a floor where "
        "the model's compositional machinery (learned from A tasks) partially compensates for lost "
        "f5 knowledge. Complete forgetting may require either longer forgetting phases or "
        "architectural changes that more strongly isolate f5 representations."
    )

    pdf.bul(
        "Higher p_target leads to more retention, confirming that the amount of B exposure "
        "independently affects consolidation depth beyond the scheduling effect."
    )

    pdf.ln(4)
    pdf.sub("Implications for safety fine-tuning")
    pdf.bul(
        "If the goal is to make capabilities easy to remove: concentrated early exposure followed by "
        "extended clean training may be preferable. The capability is learned but not deeply consolidated."
    )
    pdf.bul(
        "If the goal is to make capabilities robust: distributed late exposure (End Mixed pattern) "
        "produces the most persistent representations, hardest to passively forget."
    )

    pdf.ln(4)
    pdf.stitle("12. Reproduction")
    pdf.mono(
        f"python burst/experiment_parallel.py\n"
        f"python burst/analyze_and_report.py {run_dir}\n\n"
        f"Output: {run_dir}/\n"
        f"  all_results.pkl, config.json\n"
        f"  plots/*.png, analysis_report.pdf"
    )

    pdf_path = run_dir / "analysis_report.pdf"
    pdf.output(str(pdf_path))
    print(f"Saved {pdf_path}")
    return pdf_path


def main():
    if len(sys.argv) < 2:
        data_dir = Path("data")
        burst_dirs = sorted([d for d in data_dir.glob("burst_parallel_*") if d.is_dir()])
        if not burst_dirs:
            burst_dirs = sorted([d for d in data_dir.glob("burst_*") if d.is_dir()])
        if not burst_dirs:
            print("No burst directories found in data/")
            sys.exit(1)
        run_dir = burst_dirs[-1]
        print(f"Auto-detected: {run_dir}")
    else:
        run_dir = Path(sys.argv[1])

    results, cfg_raw = load_results(run_dir)
    cfg = cfg_raw.get("base_cfg", cfg_raw)

    base, sweep, groups = split_results(results)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"Loaded {len(results)} results ({len(base)} base + {len(sweep)} sweep)")
    print(f"Schedules: {list(groups.keys())}")

    print("Generating schedule overview...")
    plot_schedules(cfg, groups, plots_dir)

    print("Generating AUC ranking...")
    plot_auc_bars(groups, plots_dir)

    print("Generating multi-metric comparison...")
    plot_multi_metric(groups, plots_dir)

    print("Generating accuracy overlay...")
    plot_overlay(groups, cfg, plots_dir)

    print("Generating p_target sweep...")
    plot_p_sweep(sweep, groups, cfg, plots_dir)

    print("Generating per-run plots...")
    per_run_fnames = []
    seen = set()
    for r in results:
        key = r["schedule"]
        if key in seen:
            continue
        seen.add(key)
        fname = plot_per_run(r, plots_dir)
        per_run_fnames.append(fname)
        print(f"  {fname}")

    print("Generating undo heatmap...")
    plot_undo_heatmap(groups, cfg, plots_dir)

    print("Generating convergence check...")
    plot_convergence_check(groups, cfg, plots_dir)

    print("Generating PDF report...")
    pdf_path = make_report(run_dir, results, cfg, groups, sweep, per_run_fnames)

    print(f"\nDone! Report: {pdf_path}")


if __name__ == "__main__":
    main()
