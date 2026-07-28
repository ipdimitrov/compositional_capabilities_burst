"""
Plot all results and generate PDF report.
Usage: python burst/plot_and_report.py data/burst_<timestamp>
"""
import sys, os, pickle, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from collections import defaultdict
from fpdf import FPDF

COLORS = {
    "uniform": "#2196F3", "end_block": "#F44336", "mid_block": "#9C27B0",
    "early_block": "#4CAF50", "end_mixed": "#FF9800", "bookend": "#795548",
    "spread_K3": "#00BCD4", "spread_K5": "#607D8B",
}
W, H = 297, 210


def load_results(run_dir):
    run_dir = Path(run_dir)
    with open(run_dir / "all_results.pkl", "rb") as f:
        results = pickle.load(f)
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    task_ex = None
    if (run_dir / "task_examples.json").exists():
        with open(run_dir / "task_examples.json") as f:
            task_ex = json.load(f)
    return results, cfg, task_ex


def group_by_schedule(results):
    groups = defaultdict(list)
    for r in results:
        groups[r["schedule"]].append(r)
    return dict(groups)


# ============================================================
# PER-RUN PLOTS: accuracy + schedule heatmap
# ============================================================
def plot_per_run(result, plots_dir):
    log = result["log"]
    sched = result["schedule"]
    seed = result["seed"]
    cfg = result["config"]

    steps = np.array(log["step"])
    acc_t = np.array(log["acc_target"])
    acc_b = np.array(log["acc_background"])
    loss = np.array(log["loss"])
    phases = log["phase"]
    nt = np.array(log["n_target_in_batch"])

    T = cfg["total_steps"]
    U = cfg["undo_steps"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [1, 3, 3]})
    fig.suptitle(f"{sched} (seed={seed})", fontsize=14, fontweight="bold")

    # Top: schedule heatmap
    ax = axes[0]
    all_steps = np.arange(T)
    bs = cfg["batch_size"]
    p = cfg["p_target"]
    schedule_map = np.zeros(T)
    for s in range(T):
        from burst.experiment import n_target_for_step
        np.random.seed(seed * 10000 + s)
        schedule_map[s] = n_target_for_step(s, T, sched, p, bs) / bs
    ax.imshow(schedule_map.reshape(1, -1), aspect="auto", cmap="RdYlBu_r",
              extent=[0, T, 0, 1], vmin=0, vmax=1)
    ax.set_yticks([])
    ax.set_ylabel("B frac")
    ax.set_title("Training schedule: fraction of target (B) data per step", fontsize=9)
    ax.axvline(T, color="black", lw=2)

    # Middle: accuracy
    ax = axes[1]
    train_mask = np.array([p == "train" for p in phases])
    undo_mask = np.array([p == "undo" for p in phases])
    ax.plot(steps[train_mask], acc_t[train_mask], color="#F44336", lw=1.5, label="target (B) - train")
    ax.plot(steps[undo_mask], acc_t[undo_mask], color="#F44336", lw=1.5, ls="--", label="target (B) - undo")
    ax.plot(steps[train_mask], acc_b[train_mask], color="#2196F3", lw=1.5, label="background (A) - train")
    ax.plot(steps[undo_mask], acc_b[undo_mask], color="#2196F3", lw=1.5, ls="--", label="background (A) - undo")
    ax.axvline(T, color="gray", ls="--", alpha=0.5)
    ax.axhline(cfg["unlearn_threshold"], color="orange", ls=":", alpha=0.7, label=f"unlearn threshold ({cfg['unlearn_threshold']})")
    if result["unlearn_step"] is not None:
        ax.axvline(T + result["unlearn_step"], color="red", ls=":", alpha=0.7)
        ax.text(T + result["unlearn_step"], 0.5, f"unlearned\n@ step {result['unlearn_step']}", fontsize=7, color="red", ha="left")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.2)
    ax.text(T * 0.5, 1.02, "TRAIN", ha="center", fontsize=8, color="gray", transform=ax.get_xaxis_transform())
    ax.text(T + U * 0.5, 1.02, "UNDO", ha="center", fontsize=8, color="gray", transform=ax.get_xaxis_transform())

    # Bottom: loss
    ax = axes[2]
    ax.plot(steps[train_mask], loss[train_mask], color="#333", lw=1, label="loss - train")
    ax.plot(steps[undo_mask], loss[undo_mask], color="#333", lw=1, ls="--", label="loss - undo")
    ax.axvline(T, color="gray", ls="--", alpha=0.5)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Global Step")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    info = (f"Model: {cfg['n_layer']}L/{cfg['n_embd']}d/{cfg['n_head']}H  |  "
            f"train={T} undo={U}  |  batch={bs} p={p}  |  seed={seed}")
    fig.text(0.5, 0.01, info, ha="center", fontsize=7, color="gray")

    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fname = f"run_{sched}_seed{seed}.png"
    fig.savefig(plots_dir / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fname


# ============================================================
# OVERLAY: all schedules on one plot (mean + stderr)
# ============================================================
def plot_overlay(groups, cfg, plots_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    n_seeds = cfg.get("n_seeds", 1)
    seed_label = f"mean +/- stderr, {n_seeds} seeds" if n_seeds > 1 else "single seed"
    fig.suptitle(f"Target (B) Accuracy Across All Schedules ({seed_label})",
                 fontsize=13, fontweight="bold")

    for sched, runs in sorted(groups.items()):
        c = COLORS.get(sched, "gray")
        all_steps = [np.array(r["log"]["step"]) for r in runs]
        all_acc = [np.array(r["log"]["acc_target"]) for r in runs]
        min_len = min(len(a) for a in all_acc)
        steps = all_steps[0][:min_len]
        acc_mat = np.array([a[:min_len] for a in all_acc])
        mean = acc_mat.mean(axis=0)
        stderr = acc_mat.std(axis=0) / np.sqrt(len(runs))

        T = cfg["total_steps"]
        train_idx = steps <= T
        undo_idx = steps > T

        axes[0].plot(steps[train_idx], mean[train_idx], color=c, lw=1.5, label=sched)
        axes[0].fill_between(steps[train_idx], mean[train_idx] - stderr[train_idx],
                             mean[train_idx] + stderr[train_idx], color=c, alpha=0.15)

        axes[1].plot(steps[undo_idx], mean[undo_idx], color=c, lw=1.5, label=sched)
        axes[1].fill_between(steps[undo_idx], mean[undo_idx] - stderr[undo_idx],
                             mean[undo_idx] + stderr[undo_idx], color=c, alpha=0.15)

    axes[0].set_title("Training Phase")
    axes[1].set_title("Undo Phase (background-only training)")
    axes[1].axhline(cfg["unlearn_threshold"], color="orange", ls=":", alpha=0.7, label=f"threshold={cfg['unlearn_threshold']}")
    for ax in axes:
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Target (B) Accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(plots_dir / "overlay_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN METRIC: unlearning time bar chart
# ============================================================
def plot_unlearn_bars(groups, cfg, plots_dir):
    n_seeds = cfg.get("n_seeds", 1)
    seed_label = f"mean +/- stderr, {n_seeds} seeds" if n_seeds > 1 else "single seed"
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Unlearning Metrics Across Schedules ({seed_label})",
                 fontsize=14, fontweight="bold")

    scheds = sorted(groups.keys())
    x = np.arange(len(scheds))
    auc_means, auc_stds = [], []
    ea_means, ea_stds = [], []
    mlp_means, mlp_stds = [], []

    for sched in scheds:
        runs = groups[sched]
        n = max(len(runs), 1)
        aucs = [r.get("undo_auc", 0) for r in runs]
        auc_means.append(np.mean(aucs)); auc_stds.append(np.std(aucs) / np.sqrt(n) if n > 1 else 0)
        ea = [r.get("undo_end_acc", r["train_end_acc"]) for r in runs]
        ea_means.append(np.mean(ea)); ea_stds.append(np.std(ea) / np.sqrt(n) if n > 1 else 0)
        mlps = [r["mlp_undo_delta"] for r in runs]
        mlp_means.append(np.mean(mlps)); mlp_stds.append(np.std(mlps) / np.sqrt(n) if n > 1 else 0)

    colors = [COLORS.get(s, "gray") for s in scheds]

    bars = axes[0].bar(x, auc_means, yerr=auc_stds if n_seeds > 1 else None,
                       color=colors, width=0.6, edgecolor="black", lw=0.5, capsize=3)
    axes[0].set_xticks(x); axes[0].set_xticklabels(scheds, fontsize=7, rotation=25, ha="right")
    axes[0].set_ylabel("AUC of target acc during undo (lower = faster unlearning)")
    axes[0].set_title("Undo AUC (area under target acc curve)")
    axes[0].grid(True, alpha=0.2, axis="y")
    for b, v in zip(bars, auc_means):
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 5,
                     f"{v:.0f}", ha="center", fontsize=7, fontweight="bold")

    bars2 = axes[1].bar(x, ea_means, yerr=ea_stds if n_seeds > 1 else None,
                        color=colors, width=0.6, edgecolor="black", lw=0.5, capsize=3)
    axes[1].set_xticks(x); axes[1].set_xticklabels(scheds, fontsize=7, rotation=25, ha="right")
    axes[1].set_ylabel("Target accuracy after undo (lower = more forgotten)")
    axes[1].set_title("Target Acc at End of Undo Phase")
    axes[1].grid(True, alpha=0.2, axis="y")
    for b, v in zip(bars2, ea_means):
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                     f"{v:.3f}", ha="center", fontsize=7, fontweight="bold")

    bars3 = axes[2].bar(x, mlp_means, yerr=mlp_stds if n_seeds > 1 else None,
                        color=colors, width=0.6, edgecolor="black", lw=0.5, capsize=3)
    axes[2].set_xticks(x); axes[2].set_xticklabels(scheds, fontsize=7, rotation=25, ha="right")
    axes[2].set_ylabel("Sum ||dW||_F (MLP layers)")
    axes[2].set_title("Auxiliary: MLP Weight Delta During Undo")
    axes[2].grid(True, alpha=0.2, axis="y")
    for b, v in zip(bars3, mlp_means):
        axes[2].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03,
                     f"{v:.2f}", ha="center", fontsize=7, fontweight="bold")

    info = (f"Config: {cfg['n_layer']}L/{cfg['n_embd']}d/{cfg['n_head']}H  |  "
            f"train={cfg['total_steps']} undo={cfg['undo_steps']}  |  "
            f"p={cfg['p_target']}  |  threshold={cfg['unlearn_threshold']}  |  "
            f"{cfg['n_seeds']} seeds")
    fig.text(0.5, 0.01, info, ha="center", fontsize=8, color="gray")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(plots_dir / "unlearn_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# SCHEDULE VISUALIZATION
# ============================================================
def plot_schedule_overview(cfg, plots_dir):
    from burst.experiment import n_target_for_step, SCHEDULES
    T = cfg["total_steps"]
    bs = cfg["batch_size"]
    p = cfg["p_target"]

    fig, axes = plt.subplots(len(SCHEDULES), 1, figsize=(14, 1.2 * len(SCHEDULES)),
                             sharex=True)
    fig.suptitle("Schedule Overview: When does target (B) data appear?",
                 fontsize=13, fontweight="bold")

    for i, sched in enumerate(SCHEDULES):
        ax = axes[i]
        np.random.seed(42)
        frac = np.array([n_target_for_step(s, T, sched, p, bs) / bs for s in range(T)])
        ax.fill_between(range(T), 0, frac, color=COLORS.get(sched, "gray"), alpha=0.7)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(sched, fontsize=8, rotation=0, ha="right", va="center")
        ax.set_yticks([])
        ax.grid(True, alpha=0.1)

    axes[-1].set_xlabel("Training Step")
    fig.tight_layout()
    fig.savefig(plots_dir / "schedule_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# PDF REPORT
# ============================================================
def make_report(run_dir, results, cfg, task_examples, per_run_fnames, groups):
    plots_dir = run_dir / "plots"

    class PDF(FPDF):
        def header(self):
            if self.page_no() > 1:
                self.set_font("Helvetica", "I", 7)
                self.set_text_color(130, 130, 130)
                self.cell(0, 4, "Burst Schedule Experiments  |  Unlearning Time Analysis", align="L")
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
        def bbul(self, label, t):
            self.set_font("Helvetica", "B", 9); self.set_text_color(30, 30, 30)
            self.cell(4, 4.5, "-")
            self.cell(self.get_string_width(label) + 1, 4.5, label)
            self.set_font("Helvetica", "", 9)
            self.multi_cell(0, 4.5, t); self.ln(1)
        def mono(self, t):
            self.set_font("Courier", "", 7.5); self.set_text_color(40, 40, 40)
            self.multi_cell(0, 3.5, t); self.ln(2)
        def gbox(self, t):
            self.set_fill_color(240, 240, 245)
            self.set_font("Helvetica", "", 8); self.set_text_color(40, 40, 40)
            self.multi_cell(W - 20, 4, t, fill=True); self.ln(2)
        def chart(self, path, w=220):
            if Path(path).exists():
                if self.get_y() > H - 55: self.add_page()
                x = (W - w) / 2
                self.image(str(path), x=x, w=w); self.ln(3)

    pdf = PDF(orientation="L", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=12)

    # TITLE
    pdf.add_page(); pdf.ln(30)
    pdf.set_font("Helvetica", "B", 28); pdf.set_text_color(0, 80, 140)
    pdf.multi_cell(0, 12, "How Does Data Scheduling Affect\nUnlearning Speed?", align="C")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 12); pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 7, "Burst Schedule Experiments on Compositional Capabilities", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.set_font("Courier", "", 8); pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 5, f"Model: {cfg['n_layer']}L / {cfg['n_embd']}d / {cfg['n_head']}H nanoGPT  |  "
             f"train={cfg['total_steps']} undo={cfg['undo_steps']}  |  batch={cfg['batch_size']}  |  "
             f"p_target={cfg['p_target']}  |  {cfg['n_seeds']} seeds", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, f"Unlearn threshold: target acc < {cfg['unlearn_threshold']}  |  "
             f"Data: {cfg['n_alphabets']} alphabets, seq_len={cfg['seq_len']}, depth={cfg['depth']}, "
             f"{cfg['n_functions']} base bijections, {cfg['n_train_compositions']} compositions",
             align="C", new_x="LMARGIN", new_y="NEXT")

    # BACKGROUND
    pdf.add_page()
    pdf.stitle("1. What Are We Learning?")
    pdf.body(
        "We train a transformer (nanoGPT) on synthetic sequences generated by compositions of bijective "
        "functions. Each 'task' is a chain of bijections applied step-by-step to an input sequence of tokens."
    )
    pdf.sub("Simple (atomic) task example")
    pdf.body(
        "A task with depth-1 (only one non-identity function in the chain) applies a single permutation "
        "to the input. For example, with 10-letter alphabet and a single bijection f1 that maps "
        "X0->X3, X1->X7, X2->X0, ...:"
    )
    pdf.gbox(
        "Input:  S T0_1 T1_0 T2_0 T3_0 T4_0   X2 X5 X1 X8 X3 X0\n"
        "                                        |  |  |  |  |  |\n"
        "Step 1 (f1):                            X0 X9 X7 X4 X6 X3\n"
        "Steps 2-5 (identity):                   X0 X9 X7 X4 X6 X3  (unchanged)\n\n"
        "The model must learn: given task tokens T0_1,T1_0,..., apply f1 at depth 0, then identity."
    )
    pdf.sub("Compositional task example")
    pdf.body(
        "A task with depth >= 2 chains multiple non-identity bijections. The model must apply them "
        "sequentially, producing intermediate outputs at each step:"
    )
    pdf.gbox(
        "Input:  S T0_2 T1_1 T2_0 T3_0 T4_0   X4 X1 X7 X0 X9 X3\n"
        "                                        |  |  |  |  |  |\n"
        "Step 1 (f2):                            X8 X5 X2 X6 X1 X7\n"
        "Step 2 (f1):                            X4 X9 X0 X3 X7 X2\n"
        "Steps 3-5 (identity):                   X4 X9 X0 X3 X7 X2\n\n"
        "The model must learn to compose f2 then f1. It cannot memorize - it must learn the\n"
        "individual bijections and apply them in sequence. This is compositional generalization."
    )

    if task_examples:
        pdf.sub("Actual task examples from this run")
        for ex in task_examples:
            label = "TARGET (B)" if not ex.get("is_background") else "BACKGROUND (A)"
            depth = ex["depth"]
            tid = ex["task_id"]
            inp = " ".join(ex["input_tokens"])
            outs = [" ".join(o) for o in ex["outputs"]]
            pdf.set_font("Courier", "", 7)
            pdf.set_text_color(40, 40, 40)
            pdf.cell(0, 3.5, f"  [{label}] task_id={tid}  non-identity steps={depth}", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 3.5, f"    Input:  {inp}", new_x="LMARGIN", new_y="NEXT")
            for i, o in enumerate(outs):
                pdf.cell(0, 3.5, f"    Step {i+1}: {o}  {'(identity)' if ex['is_identity_steps'][i] else ''}", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

    pdf.body(
        "The model has 5 target (B) tasks and 45 background (A) tasks. ALL schedules train on the "
        "SAME total number of B samples (p=10% of all training batches). The only difference is "
        "WHEN those B samples appear during training."
    )

    # HYPOTHESIS
    pdf.add_page()
    pdf.stitle("2. Hypothesis & Experimental Design")
    pdf.sub("Hypothesis")
    pdf.body(
        "Temporally concentrated ('bursty') training creates knowledge that is faster to unlearn "
        "than uniformly distributed training. The main metric is UNDO AUC: the area under the "
        "target accuracy curve during the undo phase (lower = faster unlearning)."
    )
    pdf.sub("Training protocol")
    pdf.gbox(
        f"Phase 1 - TRAIN ({cfg['total_steps']} steps):\n"
        f"  Model sees target (B) + background (A) data according to the schedule.\n"
        f"  Total B proportion is always p={cfg['p_target']} across all schedules.\n"
        f"  Cosine warmup LR (lr={cfg['lr']}, warmup={cfg['warmup_iters']}, min_lr={cfg['min_lr']}).\n\n"
        f"Phase 2 - UNDO ({cfg['undo_steps']} steps):\n"
        f"  Model trains on data with SHUFFLED output labels (random permutation of\n"
        f"  the output tokens). This actively disrupts the learned bijection mappings.\n"
        f"  We measure: (1) AUC of target acc during undo (lower = faster unlearning),\n"
        f"  (2) final target accuracy after undo, (3) MLP weight deltas."
    )
    pdf.sub("8 schedules tested")
    pdf.bbul("uniform: ", "B mixed uniformly throughout training (binomial p per batch). Baseline.")
    pdf.bbul("end_block: ", "All B samples in a contiguous block at the END of training.")
    pdf.bbul("mid_block: ", "All B samples in a contiguous block in the MIDDLE of training.")
    pdf.bbul("early_block: ", "All B samples in a contiguous block at the START of training.")
    pdf.bbul("end_mixed: ", "B only in the second half, but mixed with A (2p rate in that half).")
    pdf.bbul("bookend: ", "B split: half at the very start, half at the very end.")
    pdf.bbul("spread_K3: ", "B split into 3 evenly-spaced burst windows.")
    pdf.bbul("spread_K5: ", "B split into 5 evenly-spaced burst windows.")

    # SCHEDULE OVERVIEW
    pdf.add_page()
    pdf.stitle("3. Schedule Visualization")
    pdf.body("Each row shows the fraction of target (B) data in each training batch over time. "
             "All schedules have the same total area (same total B exposure).")
    pdf.chart(plots_dir / "schedule_overview.png", w=260)

    # MAIN RESULT
    pdf.add_page()
    pdf.stitle("4. Main Result: Unlearning Time")
    n_seeds = cfg.get("n_seeds", 1)
    pdf.body(f"Left: AUC of target accuracy during undo (lower = faster unlearning). "
             f"Middle: final target accuracy after undo. "
             f"Right: MLP weight delta during undo (auxiliary mechanistic metric). "
             f"Error bars = stderr across {n_seeds} seeds.")
    pdf.chart(plots_dir / "unlearn_bars.png", w=250)

    scheds = sorted(groups.keys())
    pdf.sub("Numerical results")
    pdf.set_font("Courier", "", 7); pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 4, f"  {'Schedule':<15} {'Unlearn Steps':>14} {'Train End':>10} {'Undo End':>10} {'Undo AUC':>10} {'MLP Delta':>10}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 4, "  " + "-" * 80, new_x="LMARGIN", new_y="NEXT")
    for sched in scheds:
        runs = groups[sched]
        ut = [r["unlearn_step"] if r["unlearn_step"] is not None else cfg["undo_steps"] for r in runs]
        te = [r["train_end_acc"] for r in runs]
        ea = [r.get("undo_end_acc", r["train_end_acc"]) for r in runs]
        aucs = [r.get("undo_auc", 0) for r in runs]
        ml = [r["mlp_undo_delta"] for r in runs]
        se = lambda v: np.std(v) / np.sqrt(len(v))
        pdf.cell(0, 4, f"  {sched:<15} {np.mean(ut):>8.0f}+/-{se(ut):>4.0f} "
                 f"{np.mean(te):>7.4f} "
                 f"{np.mean(ea):>7.4f}+/-{se(ea):>.4f} "
                 f"{np.mean(aucs):>8.0f}+/-{se(aucs):>4.0f} "
                 f"{np.mean(ml):>7.2f}+/-{se(ml):>.2f}",
                 new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # OVERLAY
    pdf.add_page()
    pdf.stitle("5. Accuracy Overlay (All Schedules)")
    n_s = cfg.get("n_seeds", 1)
    pdf.body(f"Target (B) accuracy across all schedules. Left: training phase. Right: undo phase. "
             f"Shaded region = stderr across {n_s} seeds. Horizontal line = unlearn threshold.")
    pdf.chart(plots_dir / "overlay_accuracy.png", w=250)

    # PER-RUN
    pdf.add_page()
    pdf.stitle("6. Per-Run Details")
    pdf.body("Each plot shows: (top) schedule heatmap - fraction of B data per step, "
             "(middle) accuracy for target and background classes, "
             "(bottom) training loss. Dashed vertical line = start of undo phase.")
    for fname in sorted(per_run_fnames):
        pdf.chart(plots_dir / fname, w=240)

    # TAKEAWAYS
    pdf.add_page()
    pdf.stitle("7. Key Takeaways")
    pdf.bul("Main metric is UNDO AUC: area under target accuracy curve during shuffled-label undo training (lower = faster unlearning).")
    pdf.bul("All schedules see the SAME total B data (p=10%). Only temporal distribution differs.")
    pdf.bul("Concentrated bursts (end_block, early_block) are expected to unlearn differently than uniform mixing.")
    pdf.bul("Spreading bursts into K windows (K=3, K=5) tests whether interleaving helps consolidation.")
    pdf.bul("end_mixed tests a hybrid: B only in second half but mixed with A, not as a pure block.")
    pdf.bul("bookend tests whether splitting B between start and end creates a different pattern than either alone.")
    pdf.ln(2)

    pdf.stitle("8. Reproduction")
    pdf.mono(
        f"python burst/experiment.py              # Run all experiments\n"
        f"python burst/plot_and_report.py {run_dir}  # Generate this report\n\n"
        f"Output: {run_dir}/\n"
        f"  all_results.pkl, config.json, task_examples.json\n"
        f"  plots/*.png, report.pdf"
    )

    pdf_path = run_dir / "report.pdf"
    pdf.output(str(pdf_path))
    print(f"  Saved {pdf_path}")


def main():
    if len(sys.argv) < 2:
        data_dir = Path("data")
        if not data_dir.exists():
            print("Error: data/ directory not found")
            sys.exit(1)
        
        burst_dirs = sorted([d for d in data_dir.glob("burst_*") if d.is_dir()])
        if not burst_dirs:
            print("Error: No burst_* directories found in data/")
            print("Usage: python burst/plot_and_report.py <run_dir>")
            sys.exit(1)
        
        run_dir = burst_dirs[-1]
        print(f"Auto-detected most recent run: {run_dir}")
    else:
        run_dir = Path(sys.argv[1])
    
    results, cfg, task_examples = load_results(run_dir)
    groups = group_by_schedule(results)

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print("Generating per-run plots...")
    per_run_fnames = []
    for r in results:
        fname = plot_per_run(r, plots_dir)
        per_run_fnames.append(fname)
        print(f"  {fname}")

    print("Generating overlay plot...")
    plot_overlay(groups, cfg, plots_dir)

    print("Generating unlearn bars...")
    plot_unlearn_bars(groups, cfg, plots_dir)

    print("Generating schedule overview...")
    plot_schedule_overview(cfg, plots_dir)

    print("Generating PDF report...")
    make_report(run_dir, results, cfg, task_examples, per_run_fnames, groups)

    print("\nDone.")


if __name__ == "__main__":
    main()
