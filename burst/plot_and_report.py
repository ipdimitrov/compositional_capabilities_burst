"""
Plot all results and generate PDF report for cross-family burst experiments.
Usage: python burst/plot_and_report.py data/burst_crossfam_<timestamp>
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
        cfg["task_info"] = raw_cfg.get("task_info", {})
        cfg.setdefault("n_seeds", 1)
    else:
        cfg = raw_cfg
    return results, cfg


def group_by_label(results):
    groups = defaultdict(list)
    for r in results:
        groups[r.get("label", r["schedule"])].append(r)
    return dict(groups)


def plot_per_run(result, plots_dir):
    log = result["log"]
    sched = result["schedule"]
    seed = result["seed"]
    cfg = result["config"]

    steps = np.array(log["step"])
    acc_t = np.array(log["acc_target"])
    acc_b = np.array(log["acc_background"])
    acc_h = np.array(log.get("acc_heldout", [0.0] * len(steps)))
    loss = np.array(log["loss"])
    phases = log["phase"]

    T = cfg["total_steps"]
    U = cfg["undo_steps"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 4, 2]})
    fig.suptitle(f"{sched} (seed={seed}) — Cross-Family (bij+perm)", fontsize=14, fontweight="bold")

    ax = axes[0]
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
    ax.set_title("Schedule: fraction of B data per step", fontsize=9)
    ax.axvline(T, color="black", lw=2)

    ax = axes[1]
    train_mask = np.array([p == "train" for p in phases])
    undo_mask = np.array([p == "undo" for p in phases])

    ax.plot(steps[train_mask], acc_t[train_mask], color="#F44336", lw=1.5, label="B (bij∘perm) - train")
    ax.plot(steps[undo_mask], acc_t[undo_mask], color="#F44336", lw=1.5, ls="--", label="B (bij∘perm) - undo")
    ax.plot(steps[train_mask], acc_b[train_mask], color="#2196F3", lw=1.5, label="A (bijections) - train")
    ax.plot(steps[undo_mask], acc_b[undo_mask], color="#2196F3", lw=1.5, ls="--", label="A (bijections) - undo")
    ax.plot(steps[train_mask], acc_h[train_mask], color="#4CAF50", lw=1.5, label="Held-out (bij∘perm∘bij) - train")
    ax.plot(steps[undo_mask], acc_h[undo_mask], color="#4CAF50", lw=1.5, ls="--", label="Held-out - undo")

    ax.axvline(T, color="gray", ls="--", alpha=0.5)
    ax.axhline(cfg["unlearn_threshold"], color="orange", ls=":", alpha=0.7, label=f"threshold ({cfg['unlearn_threshold']})")
    if result["unlearn_step"] is not None:
        ax.axvline(T + result["unlearn_step"], color="red", ls=":", alpha=0.7)
        ax.text(T + result["unlearn_step"], 0.5, f"unlearned\n@ {result['unlearn_step']}", fontsize=7, color="red", ha="left")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Free-gen Accuracy")
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.grid(True, alpha=0.2)
    ax.text(T * 0.5, 1.02, "TRAIN", ha="center", fontsize=8, color="gray", transform=ax.get_xaxis_transform())
    ax.text(T + U * 0.5, 1.02, "UNDO", ha="center", fontsize=8, color="gray", transform=ax.get_xaxis_transform())

    ax = axes[2]
    ax.plot(steps[train_mask], loss[train_mask], color="#333", lw=1, label="loss - train")
    ax.plot(steps[undo_mask], loss[undo_mask], color="#333", lw=1, ls="--", label="loss - undo")
    ax.axvline(T, color="gray", ls="--", alpha=0.5)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Global Step")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    info = (f"Model: {cfg['n_layer']}L/{cfg['n_embd']}d/{cfg['n_head']}H  |  "
            f"train={T} undo={U}  |  batch={bs} p={p}  |  eval=free-gen  |  seed={seed}")
    fig.text(0.5, 0.01, info, ha="center", fontsize=7, color="gray")

    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fname = f"run_{sched}_seed{seed}.png"
    fig.savefig(plots_dir / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_overlay(groups, cfg, plots_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Cross-Family Burst Experiment — Free Generation Accuracy",
                 fontsize=13, fontweight="bold")

    for label, runs in sorted(groups.items()):
        sched = runs[0]["schedule"]
        c = COLORS.get(sched, COLORS.get(label, "gray"))
        all_steps = [np.array(r["log"]["step"]) for r in runs]
        all_acc_t = [np.array(r["log"]["acc_target"]) for r in runs]
        all_acc_h = [np.array(r["log"].get("acc_heldout", [0.0] * len(r["log"]["step"]))) for r in runs]
        min_len = min(len(a) for a in all_acc_t)
        steps = all_steps[0][:min_len]
        acc_t_mat = np.array([a[:min_len] for a in all_acc_t])
        acc_h_mat = np.array([a[:min_len] for a in all_acc_h])
        mean_t = acc_t_mat.mean(axis=0)
        mean_h = acc_h_mat.mean(axis=0)

        T = runs[0]["config"]["total_steps"]
        train_idx = steps <= T
        undo_idx = steps > T

        axes[0, 0].plot(steps[train_idx], mean_t[train_idx], color=c, lw=1.5, label=label)
        axes[0, 1].plot(steps[undo_idx], mean_t[undo_idx], color=c, lw=1.5, label=label)
        axes[1, 0].plot(steps[train_idx], mean_h[train_idx], color=c, lw=1.5, label=label)
        axes[1, 1].plot(steps[undo_idx], mean_h[undo_idx], color=c, lw=1.5, label=label)

    axes[0, 0].set_title("B (bij∘perm) — Training")
    axes[0, 1].set_title("B (bij∘perm) — Undo")
    axes[1, 0].set_title("Held-out (bij∘perm∘bij) — Training")
    axes[1, 1].set_title("Held-out (bij∘perm∘bij) — Undo")
    axes[0, 1].axhline(cfg["unlearn_threshold"], color="orange", ls=":", alpha=0.7)
    for ax in axes.flat:
        ax.set_xlabel("Step")
        ax.set_ylabel("Free-gen Accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=5, ncol=2)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(plots_dir / "overlay_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_unlearn_bars(groups, cfg, plots_dir):
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle("Unlearning Metrics — Cross-Family (Free Generation)",
                 fontsize=14, fontweight="bold")

    labels = sorted(groups.keys())
    x = np.arange(len(labels))
    auc_means, ea_means, ho_means, mlp_means = [], [], [], []

    for label in labels:
        runs = groups[label]
        auc_means.append(np.mean([r.get("undo_auc", 0) for r in runs]))
        ea_means.append(np.mean([r.get("undo_end_acc", r["train_end_acc"]) for r in runs]))
        ho_means.append(np.mean([r.get("heldout_undo_end", 0) or 0 for r in runs]))
        mlp_means.append(np.mean([r["mlp_undo_delta"] for r in runs]))

    colors = [COLORS.get(groups[l][0]["schedule"], "gray") for l in labels]

    axes[0].bar(x, auc_means, color=colors, width=0.6, edgecolor="black", lw=0.5)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=5, rotation=35, ha="right")
    axes[0].set_ylabel("AUC (lower = faster forgetting)")
    axes[0].set_title("Undo AUC")
    axes[0].grid(True, alpha=0.2, axis="y")
    for b, v in zip(axes[0].patches, auc_means):
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                     f"{v:.0f}", ha="center", fontsize=5, fontweight="bold")

    axes[1].bar(x, ea_means, color=colors, width=0.6, edgecolor="black", lw=0.5)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=5, rotation=35, ha="right")
    axes[1].set_ylabel("B acc after undo")
    axes[1].set_title("B Acc End of Undo")
    axes[1].grid(True, alpha=0.2, axis="y")
    for b, v in zip(axes[1].patches, ea_means):
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                     f"{v:.3f}", ha="center", fontsize=5, fontweight="bold")

    axes[2].bar(x, ho_means, color=colors, width=0.6, edgecolor="black", lw=0.5)
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels, fontsize=5, rotation=35, ha="right")
    axes[2].set_ylabel("Held-out acc after undo")
    axes[2].set_title("Held-out Acc End of Undo")
    axes[2].grid(True, alpha=0.2, axis="y")
    for b, v in zip(axes[2].patches, ho_means):
        axes[2].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                     f"{v:.3f}", ha="center", fontsize=5, fontweight="bold")

    axes[3].bar(x, mlp_means, color=colors, width=0.6, edgecolor="black", lw=0.5)
    axes[3].set_xticks(x); axes[3].set_xticklabels(labels, fontsize=5, rotation=35, ha="right")
    axes[3].set_ylabel("Sum ||dW||_F (MLP)")
    axes[3].set_title("MLP Weight Delta")
    axes[3].grid(True, alpha=0.2, axis="y")
    for b, v in zip(axes[3].patches, mlp_means):
        axes[3].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                     f"{v:.2f}", ha="center", fontsize=5, fontweight="bold")

    info = (f"Config: {cfg['n_layer']}L/{cfg['n_embd']}d/{cfg['n_head']}H  |  "
            f"train={cfg['total_steps']} undo={cfg['undo_steps']}  |  "
            f"p={cfg['p_target']}  |  eval=free-gen")
    fig.text(0.5, 0.01, info, ha="center", fontsize=8, color="gray")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(plots_dir / "unlearn_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    fig.suptitle("Schedule Overview: When does B data appear?",
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


def compute_summary_stats(groups, cfg):
    rows = []
    for label in sorted(groups.keys()):
        runs = groups[label]
        aucs = [r.get("undo_auc", 0) for r in runs]
        ea = [r.get("undo_end_acc", r["train_end_acc"]) for r in runs]
        te = [r["train_end_acc"] for r in runs]
        ho = [r.get("heldout_train_end", 0) or 0 for r in runs]
        ho_undo = [r.get("heldout_undo_end", 0) or 0 for r in runs]
        mlps = [r["mlp_undo_delta"] for r in runs]
        rows.append({
            "label": label, "schedule": runs[0]["schedule"],
            "auc_mean": np.mean(aucs),
            "end_acc_mean": np.mean(ea),
            "train_acc_mean": np.mean(te),
            "heldout_train": np.mean(ho),
            "heldout_undo": np.mean(ho_undo),
            "mlp_mean": np.mean(mlps),
        })
    rows.sort(key=lambda r: r["auc_mean"])
    return rows


def make_report(run_dir, results, cfg, per_run_fnames, groups):
    plots_dir = run_dir / "plots"
    stats = compute_summary_stats(groups, cfg)

    class PDF(FPDF):
        def header(self):
            if self.page_no() > 1:
                self.set_font("Helvetica", "I", 7)
                self.set_text_color(130, 130, 130)
                self.cell(0, 4, "Cross-Family Burst Experiments  |  Free Generation Eval", align="L")
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

    pdf.add_page(); pdf.ln(30)
    pdf.set_font("Helvetica", "B", 28); pdf.set_text_color(0, 80, 140)
    pdf.multi_cell(0, 12, "Cross-Family Burstiness Experiment\nBijection + Permutation Compositions", align="C")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 12); pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 7, "Free Generation (Autoregressive) Evaluation", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.set_font("Courier", "", 8); pdf.set_text_color(120, 120, 120)
    ti = cfg.get("task_info", {})
    pdf.cell(0, 5, f"Model: {cfg['n_layer']}L / {cfg['n_embd']}d / {cfg['n_head']}H  |  "
             f"train={cfg['total_steps']} undo={cfg['undo_steps']}  |  batch={cfg['batch_size']}  |  "
             f"p_target={cfg['p_target']}  |  {len(results)} runs",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, f"A tasks: {len(ti.get('a_tasks', []))} atomic bijections  |  "
             f"B tasks: {len(ti.get('b_tasks', []))} bij+perm compositions  |  "
             f"Held-out: {len(ti.get('heldout_tasks', []))} bij+perm+bij chains",
             align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.add_page()
    pdf.stitle("Executive Summary")

    best = stats[0]
    worst = stats[-1]
    uniform_row = next((r for r in stats if r["schedule"] == "uniform" and "p0" not in r["label"]), stats[0])

    pdf.body(
        f"This report analyses {len(results)} runs across {len(groups)} schedule configurations. "
        f"Key change from prior experiments: A and B tasks use ORTHOGONAL function families "
        f"(bijections vs permutations), giving a cleaner forgetting signal. "
        f"Evaluation uses free generation (autoregressive) - no teacher forcing."
    )

    pdf.sub("Key Findings")
    pdf.bul(f"Fastest forgetting: '{best['label']}' - AUC={best['auc_mean']:.0f}, end_acc={best['end_acc_mean']:.4f}")
    pdf.bul(f"Slowest forgetting: '{worst['label']}' - AUC={worst['auc_mean']:.0f}, end_acc={worst['end_acc_mean']:.4f}")
    pdf.bul(f"Uniform baseline: AUC={uniform_row['auc_mean']:.0f}, end_acc={uniform_row['end_acc_mean']:.4f}")

    pdf.sub("Ranking by Undo AUC")
    pdf.set_font("Courier", "", 7.5); pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 4, f"  {'Rank':<5} {'Schedule':<22} {'AUC':>8} {'B End':>8} {'B Train':>8} {'HO Train':>8} {'HO Undo':>8} {'MLP':>8}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 4, "  " + "-" * 75, new_x="LMARGIN", new_y="NEXT")
    for i, row in enumerate(stats):
        marker = " <--" if row["label"] == uniform_row["label"] else ""
        pdf.cell(0, 4,
                 f"  {i+1:<5} {row['label']:<22} {row['auc_mean']:>8.0f} {row['end_acc_mean']:>8.4f} "
                 f"{row['train_acc_mean']:>8.4f} {row['heldout_train']:>8.4f} {row['heldout_undo']:>8.4f} "
                 f"{row['mlp_mean']:>8.2f}{marker}",
                 new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.add_page()
    pdf.stitle("Experimental Design")
    pdf.sub("Cross-Family A/B Split")
    pdf.gbox(
        "A tasks (Background): Atomic bijections, depth-1\n"
        "  - Simple token-value permutations, always present in training\n"
        "  - The model learns individual bijection lookup tables\n\n"
        "B tasks (Target): Bijection composed with Permutation, depth-2\n"
        "  - First permute token POSITIONS, then apply bijection to VALUES\n"
        "  - Requires composing across two orthogonal function families\n"
        "  - Permutations are NEVER in A data - clean forgetting signal\n\n"
        "Held-out: Bijection + Permutation + Bijection, depth-3\n"
        "  - Never seen during training\n"
        "  - Tests whether burstiness affects generalization depth"
    )
    pdf.sub("Free Generation Evaluation")
    pdf.body(
        "Unlike teacher forcing, the model generates all intermediate outputs autoregressively. "
        "Only the prompt (task tokens + input sequence) is provided; the model must produce "
        "every subsequent token using its own predictions as input. This tests whether the model "
        "can actually USE the compositional capability end-to-end without scaffolding."
    )

    pdf.add_page()
    pdf.stitle("Schedule Visualization")
    pdf.chart(plots_dir / "schedule_overview.png", w=260)

    pdf.add_page()
    pdf.stitle("Main Results")
    pdf.chart(plots_dir / "unlearn_bars.png", w=260)

    pdf.add_page()
    pdf.stitle("Accuracy Overlay")
    pdf.chart(plots_dir / "overlay_accuracy.png", w=260)

    pdf.add_page()
    pdf.stitle("Per-Run Details")
    for fname in sorted(per_run_fnames):
        pdf.chart(plots_dir / fname, w=240)

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
            print("Error: No burst_* directories found")
            sys.exit(1)
        run_dir = burst_dirs[-1]
        print(f"Auto-detected: {run_dir}")
    else:
        run_dir = Path(sys.argv[1])

    results, cfg = load_results(run_dir)
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
    make_report(run_dir, results, cfg, per_run_fnames, groups)

    print("\nDone.")


if __name__ == "__main__":
    main()
