"""Report generation: ingest results from any combination of runs, produce charts.

All functions accept lists of result dicts (from finetune/forget) so you can
compare across sweep configurations.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path


# ── colour helpers ────────────────────────────────────────────────────────

def _frac_color(frac: float) -> str:
    """Red (100%) -> Blue (0%) gradient."""
    import colorsys
    h = 0.0 + (1.0 - frac) * 0.58  # red -> blue in HSL
    r, g, b = colorsys.hls_to_rgb(h, 0.42, 0.72)
    return f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"


# ── individual phase plots ───────────────────────────────────────────────

def plot_pretrain(pretrain_result, ax=None):
    """Plot pretrain accuracy curves."""
    log = pretrain_result["log"]
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(log["step"], log["acc_other"], label="Other (background)", color="#2196F3")
    ax.plot(log["step"], log["acc_burst"], label="Burst (special)", color="#E91E63")
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Pretrain Phase")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    return ax


def plot_finetune(finetune_results, ax=None):
    """Plot finetune accuracy curves for one or more runs.

    Args:
        finetune_results: single result dict or list of result dicts
    """
    if isinstance(finetune_results, dict):
        finetune_results = [finetune_results]
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 5))

    for r in finetune_results:
        log = r["log"]
        frac = r["burst_frac"]
        color = _frac_color(frac)
        label = r["tag"]
        ax.plot(log["step"], log["acc_burst"], color=color, label=f"{label} (burst)",
                linewidth=2)
        ax.plot(log["step"], log["acc_other"], color=color, label=f"{label} (other)",
                linewidth=1, linestyle="--", alpha=0.5)

    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Finetune (Burst) Phase")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    return ax


def plot_forget(forget_results, ax=None):
    """Plot forgetting curves for one or more runs."""
    if isinstance(forget_results, dict):
        forget_results = [forget_results]
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 5))

    for r in forget_results:
        log = r["log"]
        tag = r["tag"]
        # infer frac from tag if possible
        try:
            pct = int(tag.split("_")[1])
            color = _frac_color(pct / 100)
        except (IndexError, ValueError):
            color = None
        ax.plot(log["step"], log["acc_burst"], color=color, label=f"{tag} (burst)",
                linewidth=2)
        ax.plot(log["step"], log["acc_other"], color=color, label=f"{tag} (other)",
                linewidth=1, linestyle="--", alpha=0.5)

    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Forget (Reversion) Phase")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    return ax


# ── combined view ─────────────────────────────────────────────────────────

def plot_full_trajectory(pretrain_result, finetune_results, forget_results,
                         figsize=(16, 5)):
    """3-panel plot: pretrain | finetune | forget."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    plot_pretrain(pretrain_result, axes[0])
    plot_finetune(finetune_results, axes[1])
    plot_forget(forget_results, axes[2])
    fig.tight_layout()
    return fig


# ── summary table ─────────────────────────────────────────────────────────

def summary_table(finetune_results, forget_results):
    """Print a summary table of key metrics.

    Returns a list of dicts for further processing.
    """
    if isinstance(finetune_results, dict):
        finetune_results = [finetune_results]
    if isinstance(forget_results, dict):
        forget_results = [forget_results]

    # match by tag
    ft_by_tag = {r["tag"]: r for r in finetune_results}
    fg_by_tag = {r["tag"]: r for r in forget_results}

    rows = []
    header = f"{'Tag':<15} {'Burst%':>6} {'Peak':>6} {'End':>6} " \
             f"{'Drop%':>6} {'AUC':>8} {'95%-life':>8} {'80%-life':>8}"
    print(header)
    print("-" * len(header))

    for tag in ft_by_tag:
        ft = ft_by_tag[tag]
        fg = fg_by_tag.get(tag)
        row = {
            "tag": tag,
            "burst_frac": ft["burst_frac"],
            "peak_burst": ft["peak_burst"],
        }
        if fg:
            row.update({
                "end_burst": fg["end_burst_acc"],
                "dropoff_pct": fg["dropoff_pct"],
                "reversion_auc": fg["reversion_auc"],
                "life_95": fg["life_times"].get("life_95", "—"),
                "life_80": fg["life_times"].get("life_80", "—"),
            })
        else:
            row.update({"end_burst": "—", "dropoff_pct": "—",
                        "reversion_auc": "—", "life_95": "—", "life_80": "—"})

        def _fmt(v, f=".3f"):
            return f"{v:{f}}" if isinstance(v, (int, float)) else str(v)

        print(f"{row['tag']:<15} {row['burst_frac']*100:>5.0f}% "
              f"{_fmt(row['peak_burst']):>6} {_fmt(row.get('end_burst', '—')):>6} "
              f"{_fmt(row.get('dropoff_pct', '—'), '.1f'):>6} "
              f"{_fmt(row.get('reversion_auc', '—'), '.0f'):>8} "
              f"{_fmt(row.get('life_95', '—')):>8} {_fmt(row.get('life_80', '—')):>8}")
        rows.append(row)

    return rows


# ── comparative charts ────────────────────────────────────────────────────

def plot_peak_vs_frac(finetune_results, ax=None):
    """Peak burst accuracy vs burst fraction."""
    if isinstance(finetune_results, dict):
        finetune_results = [finetune_results]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    fracs = [r["burst_frac"] for r in finetune_results]
    peaks = [r["peak_burst"] for r in finetune_results]
    colors = [_frac_color(f) for f in fracs]

    ax.scatter(fracs, peaks, c=colors, s=80, zorder=3)
    ax.plot(fracs, peaks, color="gray", alpha=0.4, zorder=2)
    ax.set_xlabel("Burst Fraction")
    ax.set_ylabel("Peak Burst Accuracy")
    ax.set_title("Peak Accuracy vs Concentration")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    return ax


def plot_retention_vs_frac(forget_results, ax=None):
    """Reversion AUC and lifetime vs burst fraction."""
    if isinstance(forget_results, dict):
        forget_results = [forget_results]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    tags = [r["tag"] for r in forget_results]
    aucs = [r["reversion_auc"] for r in forget_results]

    # try to extract fracs from tags
    fracs = []
    for t in tags:
        try:
            fracs.append(int(t.split("_")[1]) / 100)
        except (IndexError, ValueError):
            fracs.append(0.5)

    colors = [_frac_color(f) for f in fracs]
    ax.bar(range(len(tags)), aucs, color=colors, alpha=0.8)
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Reversion AUC")
    ax.set_title("Knowledge Retention (higher = more retained)")
    ax.grid(True, alpha=0.3, axis="y")
    return ax


def save_report(pretrain_result, finetune_results, forget_results,
                out_dir, prefix="report"):
    """Save all charts to a directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_full_trajectory(pretrain_result, finetune_results, forget_results)
    fig.savefig(out_dir / f"{prefix}_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if isinstance(finetune_results, dict):
        finetune_results = [finetune_results]
    if isinstance(forget_results, dict):
        forget_results = [forget_results]

    if len(finetune_results) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_peak_vs_frac(finetune_results, axes[0])
        plot_retention_vs_frac(forget_results, axes[1])
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Report saved to {out_dir}")
