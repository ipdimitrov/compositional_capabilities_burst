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
from burst._worker import n_target_for_step

COLORS = {
    "uniform": "#2196F3", "end_block": "#F44336", "mid_block": "#9C27B0",
    "early_block": "#4CAF50", "end_mixed": "#FF9800", "bookend": "#795548",
    "spread_K3": "#00BCD4", "spread_K5": "#607D8B",
    "early_block_2x": "#8BC34A", "late_ramp": "#E91E63",
    "cyclic": "#00ACC1", "front_heavy": "#FF5722",
}
W, H = 297, 210


def load_results(run_dir):
    run_dir = Path(run_dir)
    with open(run_dir / "all_results.pkl", "rb") as f:
        results = pickle.load(f)
    with open(run_dir / "config.json") as f:
        raw_cfg = json.load(f)
    if "base_cfg" in raw_cfg:
        cfg = dict(raw_cfg["base_cfg"])
        cfg["n_jobs"] = raw_cfg.get("n_jobs", len(results))
        cfg["jobs"] = raw_cfg.get("jobs", [])
        cfg.setdefault("n_seeds", 1)
    else:
        cfg = raw_cfg
    task_ex = None
    if (run_dir / "task_examples.json").exists():
        with open(run_dir / "task_examples.json") as f:
            task_ex = json.load(f)
    return results, cfg, task_ex


def group_by_label(results):
    groups = defaultdict(list)
    for r in results:
        groups[r.get("label", r["schedule"])].append(r)
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

    ax = axes[0]
    all_steps = np.arange(T)
    bs = cfg["batch_size"]
    p = cfg["p_target"]
    schedule_map = np.zeros(T)
    for s in range(T):
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

    for label, runs in sorted(groups.items()):
        sched = runs[0]["schedule"]
        c = COLORS.get(sched, COLORS.get(label, "gray"))
        all_steps = [np.array(r["log"]["step"]) for r in runs]
        all_acc = [np.array(r["log"]["acc_target"]) for r in runs]
        min_len = min(len(a) for a in all_acc)
        steps = all_steps[0][:min_len]
        acc_mat = np.array([a[:min_len] for a in all_acc])
        mean = acc_mat.mean(axis=0)
        stderr = acc_mat.std(axis=0) / np.sqrt(len(runs))

        T = runs[0]["config"]["total_steps"]
        train_idx = steps <= T
        undo_idx = steps > T

        axes[0].plot(steps[train_idx], mean[train_idx], color=c, lw=1.5, label=label)
        axes[0].fill_between(steps[train_idx], mean[train_idx] - stderr[train_idx],
                             mean[train_idx] + stderr[train_idx], color=c, alpha=0.15)

        axes[1].plot(steps[undo_idx], mean[undo_idx], color=c, lw=1.5, label=label)
        axes[1].fill_between(steps[undo_idx], mean[undo_idx] - stderr[undo_idx],
                             mean[undo_idx] + stderr[undo_idx], color=c, alpha=0.15)

    axes[0].set_title("Training Phase")
    axes[1].set_title("Undo Phase (passive forgetting)")
    axes[1].axhline(cfg["unlearn_threshold"], color="orange", ls=":", alpha=0.7, label=f"threshold={cfg['unlearn_threshold']}")
    for ax in axes:
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Target (B) Accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=6, ncol=2)
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

    labels = sorted(groups.keys())
    x = np.arange(len(labels))
    auc_means, auc_stds = [], []
    ea_means, ea_stds = [], []
    mlp_means, mlp_stds = [], []

    for label in labels:
        runs = groups[label]
        n = max(len(runs), 1)
        aucs = [r.get("undo_auc", 0) for r in runs]
        auc_means.append(np.mean(aucs)); auc_stds.append(np.std(aucs) / np.sqrt(n) if n > 1 else 0)
        ea = [r.get("undo_end_acc", r["train_end_acc"]) for r in runs]
        ea_means.append(np.mean(ea)); ea_stds.append(np.std(ea) / np.sqrt(n) if n > 1 else 0)
        mlps = [r["mlp_undo_delta"] for r in runs]
        mlp_means.append(np.mean(mlps)); mlp_stds.append(np.std(mlps) / np.sqrt(n) if n > 1 else 0)

    colors = [COLORS.get(groups[l][0]["schedule"], "gray") for l in labels]

    bars = axes[0].bar(x, auc_means, yerr=auc_stds if n_seeds > 1 else None,
                       color=colors, width=0.6, edgecolor="black", lw=0.5, capsize=3)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=6, rotation=35, ha="right")
    axes[0].set_ylabel("AUC of target acc during undo (lower = faster unlearning)")
    axes[0].set_title("Undo AUC (area under target acc curve)")
    axes[0].grid(True, alpha=0.2, axis="y")
    for b, v in zip(bars, auc_means):
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 5,
                     f"{v:.0f}", ha="center", fontsize=6, fontweight="bold")

    bars2 = axes[1].bar(x, ea_means, yerr=ea_stds if n_seeds > 1 else None,
                        color=colors, width=0.6, edgecolor="black", lw=0.5, capsize=3)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=6, rotation=35, ha="right")
    axes[1].set_ylabel("Target accuracy after undo (lower = more forgotten)")
    axes[1].set_title("Target Acc at End of Undo Phase")
    axes[1].grid(True, alpha=0.2, axis="y")
    for b, v in zip(bars2, ea_means):
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                     f"{v:.3f}", ha="center", fontsize=6, fontweight="bold")

    bars3 = axes[2].bar(x, mlp_means, yerr=mlp_stds if n_seeds > 1 else None,
                        color=colors, width=0.6, edgecolor="black", lw=0.5, capsize=3)
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels, fontsize=6, rotation=35, ha="right")
    axes[2].set_ylabel("Sum ||dW||_F (MLP layers)")
    axes[2].set_title("Auxiliary: MLP Weight Delta During Undo")
    axes[2].grid(True, alpha=0.2, axis="y")
    for b, v in zip(bars3, mlp_means):
        axes[2].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03,
                     f"{v:.2f}", ha="center", fontsize=6, fontweight="bold")

    info = (f"Config: {cfg['n_layer']}L/{cfg['n_embd']}d/{cfg['n_head']}H  |  "
            f"train={cfg['total_steps']} undo={cfg['undo_steps']}  |  "
            f"p={cfg['p_target']}  |  threshold={cfg['unlearn_threshold']}  |  "
            f"{cfg.get('n_seeds', 1)} seeds")
    fig.text(0.5, 0.01, info, ha="center", fontsize=8, color="gray")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(plots_dir / "unlearn_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# SCHEDULE VISUALIZATION
# ============================================================
def plot_schedule_overview(cfg, plots_dir, schedules=None):
    T = cfg["total_steps"]
    bs = cfg["batch_size"]
    p = cfg["p_target"]

    if schedules is None:
        schedules = ["uniform", "end_block", "mid_block", "early_block",
                     "end_mixed", "bookend", "early_block_2x", "late_ramp",
                     "cyclic", "front_heavy"]

    fig, axes = plt.subplots(len(schedules), 1, figsize=(14, 1.2 * len(schedules)),
                             sharex=True)
    if len(schedules) == 1:
        axes = [axes]
    fig.suptitle("Schedule Overview: When does target (B) data appear?",
                 fontsize=13, fontweight="bold")

    for i, sched in enumerate(schedules):
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
def compute_summary_stats(groups, cfg):
    rows = []
    for label in sorted(groups.keys()):
        runs = groups[label]
        sched = runs[0]["schedule"]
        aucs = [r.get("undo_auc", 0) for r in runs]
        ea = [r.get("undo_end_acc", r["train_end_acc"]) for r in runs]
        te = [r["train_end_acc"] for r in runs]
        ut = [r["unlearn_step"] if r["unlearn_step"] is not None else cfg["undo_steps"] for r in runs]
        mlps = [r["mlp_undo_delta"] for r in runs]
        rows.append({
            "label": label, "schedule": sched,
            "auc_mean": np.mean(aucs), "auc_std": np.std(aucs),
            "end_acc_mean": np.mean(ea), "end_acc_std": np.std(ea),
            "train_acc_mean": np.mean(te),
            "unlearn_step_mean": np.mean(ut),
            "mlp_mean": np.mean(mlps),
        })
    rows.sort(key=lambda r: r["auc_mean"])
    return rows


def make_report(run_dir, results, cfg, task_examples, per_run_fnames, groups):
    plots_dir = run_dir / "plots"
    stats = compute_summary_stats(groups, cfg)

    class PDF(FPDF):
        def header(self):
            if self.page_no() > 1:
                self.set_font("Helvetica", "I", 7)
                self.set_text_color(130, 130, 130)
                self.cell(0, 4, "Burst Schedule Experiments  |  Passive Forgetting Analysis", align="L")
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

    # ── TITLE PAGE ──
    pdf.add_page(); pdf.ln(30)
    pdf.set_font("Helvetica", "B", 28); pdf.set_text_color(0, 80, 140)
    pdf.multi_cell(0, 12, "How Does Data Scheduling Affect\nPassive Forgetting Speed?", align="C")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 12); pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 7, "Burst Schedule Experiments on Compositional Capabilities", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.set_font("Courier", "", 8); pdf.set_text_color(120, 120, 120)
    n_seeds = cfg.get("n_seeds", 1)
    pdf.cell(0, 5, f"Model: {cfg['n_layer']}L / {cfg['n_embd']}d / {cfg['n_head']}H nanoGPT  |  "
             f"train={cfg['total_steps']} undo={cfg['undo_steps']}  |  batch={cfg['batch_size']}  |  "
             f"p_target={cfg['p_target']}  |  {n_seeds} seed(s)  |  {len(results)} runs",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, f"Unlearn threshold: target acc < {cfg['unlearn_threshold']}  |  "
             f"Data: {cfg['n_alphabets']} alphabets, seq_len={cfg['seq_len']}, depth={cfg['depth']}, "
             f"{cfg['n_functions']} base bijections, {cfg['n_train_compositions']} compositions",
             align="C", new_x="LMARGIN", new_y="NEXT")

    # ── EXECUTIVE SUMMARY ──
    pdf.add_page()
    pdf.stitle("Executive Summary")

    best = stats[0]
    worst = stats[-1]
    uniform_row = next((r for r in stats if r["schedule"] == "uniform" and "p0" not in r["label"]), stats[0])
    auc_ratio = worst["auc_mean"] / max(best["auc_mean"], 1)

    pdf.body(
        f"This report analyses {len(results)} parallel training runs across {len(groups)} schedule configurations, "
        f"measuring how the temporal distribution of target data during training affects the speed of passive "
        f"forgetting (unlearning via continued training on background data only, with correct labels)."
    )

    pdf.sub("Key Findings")
    pdf.bul(
        f"Fastest forgetting: '{best['label']}' achieved the lowest undo AUC of {best['auc_mean']:.0f}, "
        f"with final target accuracy dropping to {best['end_acc_mean']:.4f} after the undo phase."
    )
    pdf.bul(
        f"Slowest forgetting: '{worst['label']}' had the highest undo AUC of {worst['auc_mean']:.0f}, "
        f"retaining target accuracy of {worst['end_acc_mean']:.4f}. "
        f"The ratio between slowest and fastest is {auc_ratio:.1f}x."
    )
    pdf.bul(
        f"Uniform baseline: '{uniform_row['label']}' achieved AUC={uniform_row['auc_mean']:.0f} "
        f"and final target acc={uniform_row['end_acc_mean']:.4f}."
    )

    n_faster = sum(1 for r in stats if r["auc_mean"] < uniform_row["auc_mean"])
    n_slower = sum(1 for r in stats if r["auc_mean"] > uniform_row["auc_mean"])
    pdf.bul(
        f"Relative to uniform: {n_faster} schedule(s) forgot faster, {n_slower} forgot slower."
    )

    pdf.sub("Ranking by Undo AUC (lower = faster forgetting)")
    pdf.set_font("Courier", "", 7.5); pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 4, f"  {'Rank':<5} {'Schedule':<22} {'Undo AUC':>10} {'End Acc':>10} {'Train Acc':>10} {'MLP Delta':>10}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 4, "  " + "-" * 72, new_x="LMARGIN", new_y="NEXT")
    for i, row in enumerate(stats):
        marker = " <-- baseline" if row["label"] == uniform_row["label"] else ""
        pdf.cell(0, 4,
                 f"  {i+1:<5} {row['label']:<22} {row['auc_mean']:>10.0f} {row['end_acc_mean']:>10.4f} "
                 f"{row['train_acc_mean']:>10.4f} {row['mlp_mean']:>10.2f}{marker}",
                 new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.sub("Interpretation")
    pdf.body(
        "The undo phase uses passive forgetting: the model continues training on background (A) data "
        "with correct labels, and target (B) data is simply withheld. A lower undo AUC means the model "
        "loses its target capability faster. Schedules that concentrate target data into temporal bursts "
        "may create more fragile representations that are easier to overwrite, while uniform exposure "
        "may interleave target knowledge more deeply into shared representations."
    )

    # ── BACKGROUND ──
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
        f"The model has {cfg.get('n_target', 10)} target (B) tasks and background (A) tasks. "
        f"ALL schedules train on the SAME total number of B samples. The only difference is "
        f"WHEN those B samples appear during training."
    )

    # ── HYPOTHESIS ──
    pdf.add_page()
    pdf.stitle("2. Hypothesis & Experimental Design")
    pdf.sub("Hypothesis")
    pdf.body(
        "Temporally concentrated ('bursty') training creates knowledge that is faster to passively "
        "forget than uniformly distributed training. The main metric is UNDO AUC: the area under the "
        "target accuracy curve during the undo phase (lower = faster forgetting)."
    )
    pdf.sub("Training protocol")
    pdf.gbox(
        f"Phase 1 - TRAIN ({cfg['total_steps']} steps):\n"
        f"  Model sees target (B) + background (A) data according to the schedule.\n"
        f"  Total B proportion is always p={cfg['p_target']} across all schedules.\n"
        f"  Cosine warmup LR (lr={cfg['lr']}, warmup={cfg['warmup_iters']}, min_lr={cfg['min_lr']}).\n\n"
        f"Phase 2 - UNDO ({cfg['undo_steps']} steps):\n"
        f"  Model trains on background (A) data ONLY with correct labels.\n"
        f"  This is passive forgetting: B data is simply withheld.\n"
        f"  We measure: (1) AUC of target acc during undo (lower = faster forgetting),\n"
        f"  (2) final target accuracy after undo, (3) MLP weight deltas."
    )

    unique_schedules = sorted(set(r["schedule"] for r in results))
    pdf.sub(f"{len(unique_schedules)} schedule types tested ({len(results)} total runs)")
    pdf.bbul("uniform: ", "B mixed uniformly throughout training (binomial p per batch). Baseline.")
    pdf.bbul("end_block: ", "All B samples in a contiguous block at the END of training.")
    pdf.bbul("mid_block: ", "All B samples in a contiguous block in the MIDDLE of training.")
    pdf.bbul("early_block: ", "All B samples in a contiguous block at the START of training.")
    pdf.bbul("early_block_2x: ", "Like early_block but with 2x the burst window length (lower concentration).")
    pdf.bbul("end_mixed: ", "B only in the second half, but mixed with A (2p rate in that half).")
    pdf.bbul("bookend: ", "B split: half at the very start, half at the very end.")
    pdf.bbul("late_ramp: ", "B probability ramps linearly from 0 to 2p over training (more B later).")
    pdf.bbul("cyclic: ", "B split into 4 evenly-spaced burst windows at the start of each cycle.")
    pdf.bbul("front_heavy: ", "B only in the first half, mixed with A (2p rate in that half).")
    if any("p0." in r.get("label", "") for r in results):
        pdf.ln(1)
        pdf.body("Additional runs vary p_target (0.02 and 0.10) for uniform and end_block schedules.")

    # ── SCHEDULE OVERVIEW ──
    pdf.add_page()
    pdf.stitle("3. Schedule Visualization")
    pdf.body(
        "Each row shows the fraction of target (B) data in each training batch over time for "
        "the default p_target. All schedules have the same total area (same total B exposure). "
        "Stochastic schedules (uniform, end_mixed, late_ramp, front_heavy) show expected values."
    )
    pdf.chart(plots_dir / "schedule_overview.png", w=260)
    pdf.body(
        "Figure description: A vertically stacked set of area plots, one per schedule type. "
        "The x-axis is the training step (0 to total_steps). The y-axis (implicit, 0 to 1) is the "
        "fraction of each batch that is target (B) data. Dense colored regions indicate when B data "
        "is present. This visualization makes it easy to compare the temporal placement of B data "
        "across schedules at a glance."
    )

    # ── MAIN RESULT ──
    pdf.add_page()
    pdf.stitle("4. Main Result: Forgetting Metrics")
    pdf.body(
        "The bar charts below show three metrics across all schedule configurations. "
        "Each bar represents a single run (single seed)."
    )
    pdf.chart(plots_dir / "unlearn_bars.png", w=250)
    pdf.body(
        "Figure description - three panels:\n\n"
        "LEFT - Undo AUC: The area under the target accuracy curve during the undo phase. "
        "This is the primary metric. Lower values mean the model forgot its target capability "
        "faster during passive forgetting. A schedule with AUC=0 would mean instant forgetting; "
        "a schedule with AUC equal to undo_steps would mean no forgetting at all.\n\n"
        "MIDDLE - Target Acc at End of Undo: The final target accuracy after all undo steps. "
        "Lower values indicate more complete forgetting. Values near chance level suggest the "
        "model has fully lost the target capability.\n\n"
        "RIGHT - MLP Weight Delta: The sum of Frobenius norms of weight changes in MLP layers "
        "during the undo phase. This auxiliary mechanistic metric captures how much the MLP "
        "parameters shifted during forgetting. Larger deltas may indicate the model needed more "
        "parameter movement to accommodate the loss of target knowledge."
    )

    labels = sorted(groups.keys())
    pdf.sub("Numerical results")
    pdf.set_font("Courier", "", 7); pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 4, f"  {'Schedule':<22} {'Unlearn Step':>13} {'Train End':>10} {'Undo End':>10} {'Undo AUC':>10} {'MLP Delta':>10}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 4, "  " + "-" * 80, new_x="LMARGIN", new_y="NEXT")
    for label in labels:
        runs = groups[label]
        ut = [r["unlearn_step"] if r["unlearn_step"] is not None else cfg["undo_steps"] for r in runs]
        te = [r["train_end_acc"] for r in runs]
        ea = [r.get("undo_end_acc", r["train_end_acc"]) for r in runs]
        aucs = [r.get("undo_auc", 0) for r in runs]
        ml = [r["mlp_undo_delta"] for r in runs]
        n = len(runs)
        se = lambda v: np.std(v) / np.sqrt(len(v)) if len(v) > 1 else 0
        if n > 1:
            pdf.cell(0, 4, f"  {label:<22} {np.mean(ut):>7.0f}+/-{se(ut):>4.0f} "
                     f"{np.mean(te):>7.4f} "
                     f"{np.mean(ea):>7.4f}+/-{se(ea):>.4f} "
                     f"{np.mean(aucs):>8.0f}+/-{se(aucs):>4.0f} "
                     f"{np.mean(ml):>7.2f}+/-{se(ml):>.2f}",
                     new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.cell(0, 4, f"  {label:<22} {np.mean(ut):>13.0f} "
                     f"{np.mean(te):>10.4f} "
                     f"{np.mean(ea):>10.4f} "
                     f"{np.mean(aucs):>10.0f} "
                     f"{np.mean(ml):>10.2f}",
                     new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── OVERLAY ──
    pdf.add_page()
    pdf.stitle("5. Accuracy Overlay (All Schedules)")
    pdf.body(
        "Target (B) accuracy across all schedules plotted together for direct comparison. "
        "The left panel covers the training phase; the right panel covers the undo (passive forgetting) phase."
    )
    pdf.chart(plots_dir / "overlay_accuracy.png", w=250)
    pdf.body(
        "Figure description: Two side-by-side line plots. Each colored line represents one schedule "
        "configuration. LEFT: target accuracy during training - shows how quickly each schedule "
        "teaches the model the target capability. Schedules with early bursts show rapid early "
        "learning; late-burst schedules show delayed learning. RIGHT: target accuracy during the "
        "undo phase - shows how quickly each schedule's learned knowledge decays under passive "
        "forgetting. The horizontal orange dashed line marks the unlearn threshold. Lines that "
        "drop below this threshold faster correspond to schedules with faster forgetting."
    )

    # ── PER-RUN ──
    pdf.add_page()
    pdf.stitle("6. Per-Run Details")
    pdf.body(
        "Each plot below shows the full training and forgetting trajectory for a single schedule "
        "configuration. The three panels are:"
    )
    pdf.bul("TOP - Schedule heatmap: the fraction of target (B) data in each training batch, "
            "visualized as a color strip from blue (0% B) to red (100% B).")
    pdf.bul("MIDDLE - Accuracy: target (B) accuracy in red, background (A) accuracy in blue. "
            "Solid lines = training phase, dashed lines = undo phase. The orange dotted line "
            "marks the unlearn threshold. A red dotted vertical line marks the step where "
            "target accuracy first drops below threshold (if it does).")
    pdf.bul("BOTTOM - Loss: training loss over time. The gray dashed vertical line marks the "
            "transition from training to undo phase.")
    for fname in sorted(per_run_fnames):
        pdf.chart(plots_dir / fname, w=240)

    # ── TAKEAWAYS ──
    pdf.add_page()
    pdf.stitle("7. Key Takeaways")
    pdf.bul(
        f"Primary metric is Undo AUC: area under target accuracy curve during passive forgetting "
        f"(lower = faster forgetting). Best: {best['label']} ({best['auc_mean']:.0f}), "
        f"worst: {worst['label']} ({worst['auc_mean']:.0f})."
    )
    pdf.bul(
        f"All schedules see the SAME total B data (p={cfg['p_target']}). "
        f"Only temporal distribution differs."
    )
    pdf.bul(
        "Concentrated bursts (end_block, early_block) create different forgetting dynamics "
        "than uniform mixing - the temporal placement of data matters."
    )
    pdf.bul(
        "Cyclic and spread schedules test whether distributing bursts across multiple windows "
        "changes the consolidation pattern compared to a single contiguous block."
    )
    pdf.bul(
        "The p_target sweep (0.02 vs 0.10) for uniform and end_block tests whether the "
        "scheduling effect interacts with overall data proportion."
    )
    pdf.bul(
        "MLP weight deltas provide an auxiliary mechanistic signal: schedules that require "
        "larger weight changes during forgetting may have encoded knowledge differently."
    )
    pdf.ln(2)

    pdf.stitle("8. Reproduction")
    pdf.mono(
        f"python burst/experiment_parallel.py         # Run all experiments\n"
        f"python burst/plot_and_report.py {run_dir}   # Generate this report\n\n"
        f"Output: {run_dir}/\n"
        f"  all_results.pkl, config.json\n"
        f"  plots/*.png, analysis_report.pdf"
    )

    pdf_path = run_dir / "analysis_report.pdf"
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
    groups = group_by_label(results)

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

    unique_schedules = sorted(set(r["schedule"] for r in results))
    print("Generating schedule overview...")
    plot_schedule_overview(cfg, plots_dir, schedules=unique_schedules)

    print("Generating PDF report...")
    make_report(run_dir, results, cfg, task_examples, per_run_fnames, groups)

    print("\nDone.")


if __name__ == "__main__":
    main()
