"""
Read all ratio results from results_sweep/ and generate comparison plots.

Usage:
    python gemma/plot_results.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_ROOT = Path(__file__).parent / "results_sweep"
COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]


def load_all_ratios():
    dfs = {}
    for d in sorted(RESULTS_ROOT.iterdir()):
        csv_path = d / "metrics.csv"
        if not csv_path.exists():
            continue
        ratio = d.name.replace("ratio_", "")
        df = pd.read_csv(csv_path)
        df["ratio"] = float(ratio)
        dfs[float(ratio)] = df
    return dfs


def per_ratio_plots(ratio: float, df: pd.DataFrame, out_dir: Path):
    steps = sorted(df["step"].unique())
    layers = sorted(df["layer"].unique())

    cossim_col = "grad_cossim_train_vs_safety"

    mat = np.zeros((len(layers), len(steps)))
    for si, s in enumerate(steps):
        sub = df[df["step"] == s].set_index("layer")
        for li, la in enumerate(layers):
            if la in sub.index:
                mat[li, si] = sub.loc[la, cossim_col]
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=5)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps, fontsize=6, rotation=45)
    ax.set_xlabel("Step")
    ax.set_ylabel("Layer")
    ax.set_title(f"Gradient CosSim (train vs safety) — ratio={ratio}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "cossim_heatmap.png", dpi=150)
    plt.close(fig)

    step_df = df.groupby("step").first().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(step_df["step"], step_df["train_loss"], label="train loss")
    ax.plot(step_df["step"], step_df["math_val_loss"], label="math val loss")
    ax.plot(step_df["step"], step_df["safety_val_loss"], label="safety val loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"Losses — ratio={ratio}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "losses.png", dpi=150)
    plt.close(fig)

    avg_drift = df.groupby("step")["weight_delta_rel"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(avg_drift["step"], avg_drift["weight_delta_rel"])
    ax.set_xlabel("Step")
    ax.set_ylabel("Avg relative weight drift")
    ax.set_title(f"Weight Drift — ratio={ratio}")
    fig.tight_layout()
    fig.savefig(out_dir / "weight_drift.png", dpi=150)
    plt.close(fig)


def cross_ratio_plots(dfs: dict):
    out = RESULTS_ROOT / "comparison"
    out.mkdir(exist_ok=True)
    ratios = sorted(dfs.keys())

    cossim_col = "grad_cossim_train_vs_safety"

    # --- Per-layer cossim with std bands ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, r in enumerate(ratios):
        grp = dfs[r].groupby("step")[cossim_col]
        mean = grp.mean().reset_index()
        std = grp.std().reset_index()
        c = COLORS[i % len(COLORS)]
        ax.plot(mean["step"], mean[cossim_col], label=f"ratio={r}", color=c)
        ax.fill_between(
            mean["step"],
            mean[cossim_col] - std[cossim_col],
            mean[cossim_col] + std[cossim_col],
            alpha=0.15, color=c,
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Avg gradient cossim (train vs safety)")
    ax.set_title("Gradient CosSim by Data Ratio (mean +/- std across layers)")
    ax.legend()
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    fig.tight_layout()
    fig.savefig(out / "cossim_comparison.png", dpi=150)
    plt.close(fig)

    # --- Whole-model cossim ---
    if "grad_cossim_whole_model" in dfs[ratios[0]].columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, r in enumerate(ratios):
            step_df = dfs[r].groupby("step").first().reset_index()
            c = COLORS[i % len(COLORS)]
            ax.plot(step_df["step"], step_df["grad_cossim_whole_model"], label=f"ratio={r}", color=c)
        ax.set_xlabel("Step")
        ax.set_ylabel("Whole-model gradient cossim")
        ax.set_title("Whole-Model Gradient CosSim (train vs safety)")
        ax.legend()
        ax.axhline(0, color="gray", ls="--", lw=0.5)
        fig.tight_layout()
        fig.savefig(out / "cossim_whole_model_comparison.png", dpi=150)
        plt.close(fig)

    # --- Safety val loss ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, r in enumerate(ratios):
        step_df = dfs[r].groupby("step").first().reset_index()
        ax.plot(step_df["step"], step_df["safety_val_loss"], label=f"ratio={r}", color=COLORS[i % len(COLORS)])
    ax.set_xlabel("Step")
    ax.set_ylabel("Safety validation loss")
    ax.set_title("Safety Forgetting by Data Ratio")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "safety_val_loss_comparison.png", dpi=150)
    plt.close(fig)

    # --- Math val loss ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, r in enumerate(ratios):
        step_df = dfs[r].groupby("step").first().reset_index()
        ax.plot(step_df["step"], step_df["math_val_loss"], label=f"ratio={r}", color=COLORS[i % len(COLORS)])
    ax.set_xlabel("Step")
    ax.set_ylabel("Math validation loss")
    ax.set_title("Math Learning by Data Ratio")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "math_val_loss_comparison.png", dpi=150)
    plt.close(fig)

    # --- Avg weight drift ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, r in enumerate(ratios):
        avg = dfs[r].groupby("step")["weight_delta_rel"].mean().reset_index()
        ax.plot(avg["step"], avg["weight_delta_rel"], label=f"ratio={r}", color=COLORS[i % len(COLORS)])
    ax.set_xlabel("Step")
    ax.set_ylabel("Avg relative weight drift")
    ax.set_title("Weight Drift by Data Ratio")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "weight_drift_comparison.png", dpi=150)
    plt.close(fig)

    # --- Final-step summary bar chart ---
    final_safety, final_math, final_cossim = [], [], []
    for r in ratios:
        step_df = dfs[r].groupby("step").first().reset_index()
        final_safety.append(step_df["safety_val_loss"].iloc[-1])
        final_math.append(step_df["math_val_loss"].iloc[-1])
        final_cossim.append(dfs[r].groupby("step")[cossim_col].mean().iloc[-1])

    x = np.arange(len(ratios))
    w = 0.25
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(x - w, final_safety, w, label="Safety val loss", color="#e41a1c")
    ax1.bar(x, final_math, w, label="Math val loss", color="#377eb8")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{r}" for r in ratios])
    ax1.set_xlabel("New data ratio")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.bar(x + w, final_cossim, w, label="Avg cossim", color="#4daf4a", alpha=0.7)
    ax2.set_ylabel("Avg gradient cossim")
    ax2.legend(loc="upper right")
    ax1.set_title("Final-Step Metrics by Data Ratio")
    fig.tight_layout()
    fig.savefig(out / "final_summary.png", dpi=150)
    plt.close(fig)

    print(f"Comparison plots saved to {out}")


def main():
    dfs = load_all_ratios()
    if not dfs:
        print("No results found.")
        return

    print(f"Found ratios: {sorted(dfs.keys())}")

    for ratio, df in dfs.items():
        out_dir = RESULTS_ROOT / f"ratio_{ratio}"
        per_ratio_plots(ratio, df, out_dir)
        print(f"  Per-ratio plots for ratio={ratio}")

    if len(dfs) > 1:
        cross_ratio_plots(dfs)


if __name__ == "__main__":
    main()
