#!/usr/bin/env python3
"""
Analysis plots for the two-stage OLMo-2 concentration experiment.

Stage 1: Code fine-tuning (single run)
Stage 2: Forgetting sweep (concentration = fraction of pretraining data)

Figures:
  1. Stage 1 training curves (code loss going down, pretrain loss going up)
  2. Forgetting curves: code val loss rising over stage 2 (per concentration)
  3. Recovery curves: pretrain val loss falling over stage 2 (per concentration)
  4. Pareto trade-off: final code loss vs final pretrain loss
  5. LR effect at c=1.0
  6. Gradient cossim over stage 2
  7. Per-layer cossim heatmaps
  8. Weight drift
  9. Benchmark results (if available)

Usage:
    python olmo2/plot_results.py
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_ROOT = Path(__file__).parent / "results"
CONCENTRATION_COLORS = {
    0.0: "#1b9e77", 0.1: "#377eb8", 0.3: "#4daf4a", 0.5: "#ff7f00",
    0.7: "#984ea3", 0.9: "#e41a1c", 1.0: "#a65628",
}
LR_MARKERS = {1e-5: "o", 5e-5: "s", 1e-4: "^"}
COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]


def get_color(c):
    return CONCENTRATION_COLORS.get(c, "#333333")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_stage2_name(name):
    """Parse 'stage2_c0.5_lr5e-05_s2lr5e-05_s42' into (c, lr, seed).
    Also handles old format 'stage2_c0.5_lr5e-05_s42'."""
    parts = name.replace("stage2_", "").split("_")
    c = float(parts[0][1:])
    lr = float(parts[1][2:])
    # Find seed: last part starting with 's' that's a pure integer
    seed = None
    for p in reversed(parts):
        if p.startswith("s") and not p.startswith("s2"):
            try:
                seed = int(p[1:])
                break
            except ValueError:
                continue
    if seed is None:
        seed = int(parts[-1][1:])
    return c, lr, seed


def load_stage1_runs(results_dir):
    """Load all stage 1 scalar metrics (one per concentration/lr/seed)."""
    runs = {}
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("stage1_c"):
            continue
        csv_path = d / "scalar_metrics.csv"
        if not csv_path.exists():
            continue
        try:
            # Parse stage1_c0.5_lr5e-05_s42
            parts = d.name.replace("stage1_", "").split("_")
            c = float(parts[0][1:])
            lr = float(parts[1][2:])
            seed = int(parts[2][1:])
        except (ValueError, IndexError):
            continue
        df = pd.read_csv(csv_path)
        df["concentration"] = c
        df["lr_val"] = lr
        df["seed"] = seed
        runs[d.name] = df
    return runs


def load_stage2_runs(results_dir):
    """Load all stage 2 run metrics."""
    runs = {}
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("stage2_c"):
            continue
        csv_path = d / "scalar_metrics.csv"
        if not csv_path.exists():
            continue
        try:
            c, lr, seed = parse_stage2_name(d.name)
        except (ValueError, IndexError):
            continue
        df = pd.read_csv(csv_path)
        df["concentration"] = c
        df["lr_val"] = lr
        df["seed"] = seed
        df["run_name"] = d.name
        runs[d.name] = df
    return runs


def group_by_condition(runs):
    groups = defaultdict(list)
    for name, df in runs.items():
        c, lr, seed = parse_stage2_name(name)
        groups[(c, lr)].append(df)
    return groups


# ---------------------------------------------------------------------------
# Figure 1: Stage 1 training curves
# ---------------------------------------------------------------------------

def plot_stage1(s1_runs, out_dir):
    """Plot stage 1 code val loss by concentration (shows how well each mix learns code)."""
    if not s1_runs:
        return
    all_s1 = pd.concat(s1_runs.values(), ignore_index=True)
    groups = defaultdict(list)
    for name, df in s1_runs.items():
        c = df["concentration"].iloc[0]
        lr = df["lr_val"].iloc[0]
        groups[(c, lr)].append(df)

    lrs = sorted(set(lr for _, lr in groups.keys()))
    for lr in lrs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        concentrations = sorted(set(c for c, l in groups if l == lr))
        for c in concentrations:
            dfs = groups[(c, lr)]
            merged = pd.concat(dfs).groupby("step")
            code_mean = merged["code_val_loss"].mean()
            pretrain_mean = merged["pretrain_val_loss"].mean()
            color = get_color(c)
            ax1.plot(code_mean.index, code_mean.values, label=f"c={c}", color=color, lw=2)
            ax2.plot(pretrain_mean.index, pretrain_mean.values, label=f"c={c}", color=color, lw=2)
        ax1.set_xlabel("Step"); ax1.set_ylabel("Code Val Loss")
        ax1.set_title(f"Stage 1: Code Learning (lr={lr})"); ax1.legend()
        ax2.set_xlabel("Step"); ax2.set_ylabel("Pretrain Val Loss")
        ax2.set_title(f"Stage 1: Pretrain Drift (lr={lr})"); ax2.legend()
        for ax in (ax1, ax2):
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"stage1_training_lr{lr}.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Figures 2 & 3: Forgetting & recovery curves
# ---------------------------------------------------------------------------

def plot_forgetting_recovery(runs, out_dir):
    groups = group_by_condition(runs)
    lrs = sorted(set(lr for _, lr in groups.keys()))

    for lr in lrs:
        concentrations = sorted(set(c for c, l in groups.keys() if l == lr))

        # Code val loss (forgetting — should rise for high c)
        fig, ax = plt.subplots(figsize=(10, 6))
        for c in concentrations:
            key = (c, lr)
            if key not in groups:
                continue
            dfs = groups[key]
            merged = pd.concat(dfs).groupby("step")["code_val_loss"]
            mean, std = merged.mean(), merged.std().fillna(0)
            ax.plot(mean.index, mean.values, label=f"c={c}", color=get_color(c), lw=2)
            if len(dfs) > 1:
                ax.fill_between(mean.index, mean - std, mean + std, alpha=0.15, color=get_color(c))
        ax.set_xlabel("Stage 2 Step", fontsize=12)
        ax.set_ylabel("Code Validation Loss", fontsize=12)
        ax.set_title(f"Code Forgetting During Pretraining Re-exposure (lr={lr})", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"code_forgetting_lr{lr}.png", dpi=150)
        plt.close(fig)

        # Pretrain val loss (recovery — should fall for high c)
        fig, ax = plt.subplots(figsize=(10, 6))
        for c in concentrations:
            key = (c, lr)
            if key not in groups:
                continue
            dfs = groups[key]
            merged = pd.concat(dfs).groupby("step")["pretrain_val_loss"]
            mean, std = merged.mean(), merged.std().fillna(0)
            ax.plot(mean.index, mean.values, label=f"c={c}", color=get_color(c), lw=2)
            if len(dfs) > 1:
                ax.fill_between(mean.index, mean - std, mean + std, alpha=0.15, color=get_color(c))
        ax.set_xlabel("Stage 2 Step", fontsize=12)
        ax.set_ylabel("Pretraining Validation Loss", fontsize=12)
        ax.set_title(f"Pretraining Recovery During Re-exposure (lr={lr})", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"pretrain_recovery_lr{lr}.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Pareto trade-off
# ---------------------------------------------------------------------------

def plot_pareto(runs, out_dir):
    groups = group_by_condition(runs)
    fig, ax = plt.subplots(figsize=(10, 8))

    for (c, lr), dfs in sorted(groups.items()):
        final_code = [df.iloc[-1]["code_val_loss"] for df in dfs]
        final_pretrain = [df.iloc[-1]["pretrain_val_loss"] for df in dfs]
        marker = LR_MARKERS.get(lr, "o")
        ax.scatter(np.mean(final_pretrain), np.mean(final_code),
                   color=get_color(c), marker=marker, s=120, edgecolors="black", lw=0.5, zorder=5)
        ax.annotate(f"c={c}", (np.mean(final_pretrain), np.mean(final_code)),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    # Legends
    for c in sorted(CONCENTRATION_COLORS.keys()):
        if any(c == cc for (cc, _) in groups):
            ax.scatter([], [], color=get_color(c), s=80, label=f"c={c}")
    for lr, marker in sorted(LR_MARKERS.items()):
        if any(lr == ll for (_, ll) in groups):
            ax.scatter([], [], color="gray", marker=marker, s=80, label=f"lr={lr}")

    ax.set_xlabel("Final Pretrain Val Loss (lower = better recovery)", fontsize=12)
    ax.set_ylabel("Final Code Val Loss (lower = less forgetting)", fontsize=12)
    ax.set_title("Trade-off: Pretraining Recovery vs Code Retention", fontsize=14)
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_tradeoff.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: LR effect at c=1.0
# ---------------------------------------------------------------------------

def plot_lr_effect(runs, out_dir):
    groups = group_by_condition(runs)
    lrs = sorted(set(lr for c, lr in groups if c == 1.0))
    if not lrs:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for i, lr in enumerate(lrs):
        key = (1.0, lr)
        if key not in groups:
            continue
        dfs = groups[key]
        color = COLORS[i % len(COLORS)]
        for ax, metric, title in [(ax1, "code_val_loss", "Code Forgetting"), (ax2, "pretrain_val_loss", "Pretrain Recovery")]:
            merged = pd.concat(dfs).groupby("step")[metric]
            mean, std = merged.mean(), merged.std().fillna(0)
            ax.plot(mean.index, mean.values, label=f"lr={lr}", color=color, lw=2)
            if len(dfs) > 1:
                ax.fill_between(mean.index, mean - std, mean + std, alpha=0.15, color=color)

    for ax, title in [(ax1, "Code Val Loss at c=1.0"), (ax2, "Pretrain Val Loss at c=1.0")]:
        ax.set_xlabel("Step")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "lr_effect_c1.0.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: Gradient cossim
# ---------------------------------------------------------------------------

def plot_grad_cossim(runs, out_dir):
    groups = group_by_condition(runs)
    lrs = sorted(set(lr for _, lr in groups.keys()))

    for lr in lrs:
        fig, ax = plt.subplots(figsize=(10, 6))
        for c in sorted(set(cc for cc, ll in groups if ll == lr)):
            key = (c, lr)
            if key not in groups:
                continue
            dfs = groups[key]
            merged = pd.concat(dfs).groupby("step")["grad_cossim_whole_model"]
            mean, std = merged.mean(), merged.std().fillna(0)
            ax.plot(mean.index, mean.values, label=f"c={c}", color=get_color(c), lw=2)
            if len(dfs) > 1:
                ax.fill_between(mean.index, mean - std, mean + std, alpha=0.15, color=get_color(c))
        ax.axhline(0, color="gray", ls="--", lw=0.5)
        ax.set_xlabel("Stage 2 Step", fontsize=12)
        ax.set_ylabel("Gradient CosSim (code vs pretrain)", fontsize=12)
        ax.set_title(f"Gradient Conflict in Stage 2 (lr={lr})", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"grad_cossim_lr{lr}.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7: Per-layer heatmap
# ---------------------------------------------------------------------------

def plot_layer_heatmaps(results_dir, out_dir):
    for d in sorted(results_dir.iterdir()):
        if not d.name.startswith("stage2_c"):
            continue
        csv_path = d / "metrics.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if "grad_cossim_code_vs_pretrain" not in df.columns:
            continue

        steps = sorted(df["step"].unique())
        layers = sorted(df["layer"].unique())
        mat = np.full((len(layers), len(steps)), np.nan)
        for si, s in enumerate(steps):
            sub = df[df["step"] == s].set_index("layer")
            for li, la in enumerate(layers):
                if la in sub.index:
                    val = sub.loc[la, "grad_cossim_code_vs_pretrain"]
                    mat[li, si] = val.iloc[0] if hasattr(val, "iloc") else val

        fig, ax = plt.subplots(figsize=(14, 8))
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers, fontsize=6)
        tick_idx = list(range(0, len(steps), max(1, len(steps) // 10)))
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([steps[i] for i in tick_idx], fontsize=7, rotation=45)
        ax.set_xlabel("Step")
        ax.set_ylabel("Layer")
        ax.set_title(f"Per-Layer Gradient CosSim — {d.name}")
        fig.colorbar(im, ax=ax, label="Cosine Similarity")
        fig.tight_layout()
        fig.savefig(out_dir / f"cossim_heatmap_{d.name}.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 8: Weight drift
# ---------------------------------------------------------------------------

def plot_weight_drift(runs, out_dir):
    groups = group_by_condition(runs)
    lrs = sorted(set(lr for _, lr in groups.keys()))

    for lr in lrs:
        fig, ax = plt.subplots(figsize=(10, 6))
        for c in sorted(set(cc for cc, ll in groups if ll == lr)):
            key = (c, lr)
            if key not in groups or "avg_weight_drift_rel" not in groups[key][0].columns:
                continue
            dfs = groups[key]
            merged = pd.concat(dfs).groupby("step")["avg_weight_drift_rel"]
            mean = merged.mean()
            ax.plot(mean.index, mean.values, label=f"c={c}", color=get_color(c), lw=2)
        ax.set_xlabel("Stage 2 Step", fontsize=12)
        ax.set_ylabel("Avg Relative Weight Drift", fontsize=12)
        ax.set_title(f"Weight Drift in Stage 2 (lr={lr})", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"weight_drift_lr{lr}.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def generate_summary(runs, out_dir):
    rows = []
    for name, df in runs.items():
        c, lr, seed = parse_stage2_name(name)
        last = df.iloc[-1]
        rows.append({
            "concentration": c, "lr": lr, "seed": seed,
            "final_step": last["step"],
            "final_code_val_loss": last["code_val_loss"],
            "final_pretrain_val_loss": last["pretrain_val_loss"],
            "final_grad_cossim": last.get("grad_cossim_whole_model", np.nan),
        })
    summary = pd.DataFrame(rows).sort_values(["concentration", "lr", "seed"])
    summary.to_csv(out_dir / "summary_table.csv", index=False)

    agg = summary.groupby(["concentration", "lr"]).agg(
        code_mean=("final_code_val_loss", "mean"), code_std=("final_code_val_loss", "std"),
        pretrain_mean=("final_pretrain_val_loss", "mean"), pretrain_std=("final_pretrain_val_loss", "std"),
        cossim_mean=("final_grad_cossim", "mean"), n=("seed", "count"),
    ).reset_index()
    agg.to_csv(out_dir / "summary_aggregated.csv", index=False)
    print(f"\nSummary:\n{agg.to_string(index=False)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=None)
    args = parser.parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_ROOT

    if not results_dir.exists():
        print(f"No results at {results_dir}")
        return

    out_dir = results_dir / "plots"
    out_dir.mkdir(exist_ok=True)

    s1_runs = load_stage1_runs(results_dir)
    runs = load_stage2_runs(results_dir)

    if not runs and not s1_runs:
        print("No data found.")
        return

    print(f"Stage 1: {len(s1_runs)} runs")
    print(f"Stage 2: {len(runs)} runs")

    if s1_runs:
        print("Plotting stage 1...")
        plot_stage1(s1_runs, out_dir)

    if runs:
        print("Plotting forgetting/recovery curves...")
        plot_forgetting_recovery(runs, out_dir)
        print("Plotting Pareto trade-off...")
        plot_pareto(runs, out_dir)
        print("Plotting LR effect...")
        plot_lr_effect(runs, out_dir)
        print("Plotting gradient cossim...")
        plot_grad_cossim(runs, out_dir)
        print("Plotting per-layer heatmaps...")
        plot_layer_heatmaps(results_dir, out_dir)
        print("Plotting weight drift...")
        plot_weight_drift(runs, out_dir)
        print("Generating summary...")
        generate_summary(runs, out_dir)

    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
