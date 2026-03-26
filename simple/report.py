"""Report generation: training curves, interpretability plots, full report.

All functions accept lists of result dicts so you can compare across sweep
configurations.  Finetune and forget are shown as a single continuous
timeline per burst fraction wherever possible.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from pathlib import Path


# ── colour helpers ────────────────────────────────────────────────────────

def _frac_color(frac: float) -> str:
    """Red (100%) -> Blue (0%) gradient."""
    import colorsys
    h = 0.0 + (1.0 - frac) * 0.58
    r, g, b = colorsys.hls_to_rgb(h, 0.42, 0.72)
    return f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"


def _tag_to_frac(tag):
    try:
        return int(tag.split("_")[1]) / 100
    except (IndexError, ValueError):
        return 0.5


def _tag_color(tag):
    return _frac_color(_tag_to_frac(tag))


def _pair_ft_fg(ft_results, fg_results):
    """Pair finetune and forget results by tag.  Returns list of (ft, fg) tuples."""
    ft_list = ft_results if isinstance(ft_results, list) else [ft_results]
    fg_list = fg_results if isinstance(fg_results, list) else [fg_results]
    fg_by_tag = {r["tag"]: r for r in fg_list}
    return [(ft, fg_by_tag.get(ft["tag"])) for ft in ft_list]


def _offset_fg_steps(ft_log, fg_log):
    """Return forget steps offset so they continue from the end of finetune."""
    ft_end = ft_log["step"][-1] if ft_log["step"] else 0
    return [s + ft_end for s in fg_log["step"]]


def _phase_boundary(ax, ft_log):
    """Draw a vertical dashed line at the finetune/forget boundary."""
    if ft_log["step"]:
        ax.axvline(ft_log["step"][-1], color="black", linewidth=1,
                   linestyle="--", alpha=0.4)


# ── phase plots (continuous finetune → forget) ───────────────────────────

def plot_pretrain(pretrain_result, ax=None):
    log = pretrain_result["log"]
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(log["step"], log["acc_other"], label="Other (background)", color="#2196F3")
    ax.plot(log["step"], log["acc_burst"], label="Burst (special)", color="#E91E63")
    ax.set_xlabel("Step"); ax.set_ylabel("Accuracy")
    ax.set_title("Pretrain Phase"); ax.legend()
    ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
    return ax


def plot_accuracy(ft_results, fg_results, figsize=(14, 5)):
    """Burst and background accuracy on a continuous finetune → forget axis."""
    fig, (ax_burst, ax_other) = plt.subplots(1, 2, figsize=figsize, sharey=True)
    drawn_boundary = False
    for ft, fg in _pair_ft_fg(ft_results, fg_results):
        color = _frac_color(ft["burst_frac"])
        ft_log = ft["log"]
        # burst accuracy
        ax_burst.plot(ft_log["step"], ft_log["acc_burst"],
                      color=color, linewidth=2, label=ft["tag"])
        # other accuracy
        ax_other.plot(ft_log["step"], ft_log["acc_other"],
                      color=color, linewidth=2, label=ft["tag"])
        if fg is not None:
            fg_log = fg["log"]
            fg_steps = _offset_fg_steps(ft_log, fg_log)
            ax_burst.plot(fg_steps, fg_log["acc_burst"],
                          color=color, linewidth=2)
            ax_other.plot(fg_steps, fg_log["acc_other"],
                          color=color, linewidth=2)
        if not drawn_boundary:
            _phase_boundary(ax_burst, ft_log)
            _phase_boundary(ax_other, ft_log)
            drawn_boundary = True

    for ax, title in [(ax_burst, "Burst Accuracy"),
                      (ax_other, "Background Accuracy")]:
        ax.set_xlabel("Step (finetune | forget)")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_loss(ft_results, fg_results, figsize=(14, 5)):
    """Burst and background loss on a continuous finetune → forget axis."""
    fig, (ax_burst, ax_other) = plt.subplots(1, 2, figsize=figsize, sharey=True)
    drawn_boundary = False
    for ft, fg in _pair_ft_fg(ft_results, fg_results):
        color = _frac_color(ft["burst_frac"])
        ft_log = ft["log"]
        ax_burst.plot(ft_log["step"], ft_log["loss_burst"],
                      color=color, linewidth=2, label=ft["tag"])
        ax_other.plot(ft_log["step"], ft_log["loss_other"],
                      color=color, linewidth=2, label=ft["tag"])
        if fg is not None:
            fg_log = fg["log"]
            fg_steps = _offset_fg_steps(ft_log, fg_log)
            ax_burst.plot(fg_steps, fg_log["loss_burst"],
                          color=color, linewidth=2)
            ax_other.plot(fg_steps, fg_log["loss_other"],
                          color=color, linewidth=2)
        if not drawn_boundary:
            _phase_boundary(ax_burst, ft_log)
            _phase_boundary(ax_other, ft_log)
            drawn_boundary = True

    for ax, title in [(ax_burst, "Burst Loss"),
                      (ax_other, "Background Loss")]:
        ax.set_xlabel("Step (finetune | forget)")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_full_trajectory(pretrain_result, finetune_results, forget_results,
                         figsize=(18, 10)):
    """Pretrain + continuous finetune→forget for accuracy and loss."""
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # pretrain
    ax = fig.add_subplot(gs[0, 0])
    plot_pretrain(pretrain_result, ax)

    # burst accuracy
    ax = fig.add_subplot(gs[0, 1])
    drawn = False
    for ft, fg in _pair_ft_fg(finetune_results, forget_results):
        color = _frac_color(ft["burst_frac"])
        ft_log = ft["log"]
        ax.plot(ft_log["step"], ft_log["acc_burst"], color=color, linewidth=2, label=ft["tag"])
        if fg is not None:
            fg_steps = _offset_fg_steps(ft_log, fg["log"])
            ax.plot(fg_steps, fg["log"]["acc_burst"], color=color, linewidth=2)
        if not drawn:
            _phase_boundary(ax, ft_log); drawn = True
    ax.set_xlabel("Step (finetune | forget)"); ax.set_ylabel("Accuracy")
    ax.set_title("Burst Accuracy"); ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    # background accuracy
    ax = fig.add_subplot(gs[0, 2])
    drawn = False
    for ft, fg in _pair_ft_fg(finetune_results, forget_results):
        color = _frac_color(ft["burst_frac"])
        ft_log = ft["log"]
        ax.plot(ft_log["step"], ft_log["acc_other"], color=color, linewidth=2, label=ft["tag"])
        if fg is not None:
            fg_steps = _offset_fg_steps(ft_log, fg["log"])
            ax.plot(fg_steps, fg["log"]["acc_other"], color=color, linewidth=2)
        if not drawn:
            _phase_boundary(ax, ft_log); drawn = True
    ax.set_xlabel("Step (finetune | forget)"); ax.set_ylabel("Accuracy")
    ax.set_title("Background Accuracy"); ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    # burst loss
    ax = fig.add_subplot(gs[1, 1])
    drawn = False
    for ft, fg in _pair_ft_fg(finetune_results, forget_results):
        color = _frac_color(ft["burst_frac"])
        ft_log = ft["log"]
        ax.plot(ft_log["step"], ft_log["loss_burst"], color=color, linewidth=2, label=ft["tag"])
        if fg is not None:
            fg_steps = _offset_fg_steps(ft_log, fg["log"])
            ax.plot(fg_steps, fg["log"]["loss_burst"], color=color, linewidth=2)
        if not drawn:
            _phase_boundary(ax, ft_log); drawn = True
    ax.set_xlabel("Step (finetune | forget)"); ax.set_ylabel("Loss")
    ax.set_title("Burst Loss"); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    # background loss
    ax = fig.add_subplot(gs[1, 2])
    drawn = False
    for ft, fg in _pair_ft_fg(finetune_results, forget_results):
        color = _frac_color(ft["burst_frac"])
        ft_log = ft["log"]
        ax.plot(ft_log["step"], ft_log["loss_other"], color=color, linewidth=2, label=ft["tag"])
        if fg is not None:
            fg_steps = _offset_fg_steps(ft_log, fg["log"])
            ax.plot(fg_steps, fg["log"]["loss_other"], color=color, linewidth=2)
        if not drawn:
            _phase_boundary(ax, ft_log); drawn = True
    ax.set_xlabel("Step (finetune | forget)"); ax.set_ylabel("Loss")
    ax.set_title("Background Loss"); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    fig.suptitle("Full Training Trajectory", fontsize=14, fontweight="bold")
    return fig


# ── summary table ─────────────────────────────────────────────────────────

def summary_table(finetune_results, forget_results):
    if isinstance(finetune_results, dict):
        finetune_results = [finetune_results]
    if isinstance(forget_results, dict):
        forget_results = [forget_results]
    ft_by_tag = {r["tag"]: r for r in finetune_results}
    fg_by_tag = {r["tag"]: r for r in forget_results}
    rows = []
    header = (f"{'Tag':<15} {'Burst%':>6} {'Peak':>6} {'End':>6} "
              f"{'Drop%':>6} {'AUC':>8} {'95%-life':>8} {'80%-life':>8}")
    print(header); print("-" * len(header))
    for tag in ft_by_tag:
        ft = ft_by_tag[tag]; fg = fg_by_tag.get(tag)
        row = {"tag": tag, "burst_frac": ft["burst_frac"], "peak_burst": ft["peak_burst"]}
        if fg:
            row.update({"end_burst": fg["end_burst_acc"], "dropoff_pct": fg["dropoff_pct"],
                        "reversion_auc": fg["reversion_auc"],
                        "life_95": fg["life_times"].get("life_95", "-"),
                        "life_80": fg["life_times"].get("life_80", "-")})
        else:
            row.update({"end_burst": "-", "dropoff_pct": "-", "reversion_auc": "-",
                        "life_95": "-", "life_80": "-"})
        def _fmt(v, f=".3f"):
            return f"{v:{f}}" if isinstance(v, (int, float)) else str(v)
        print(f"{row['tag']:<15} {row['burst_frac']*100:>5.0f}% "
              f"{_fmt(row['peak_burst']):>6} {_fmt(row.get('end_burst', '-')):>6} "
              f"{_fmt(row.get('dropoff_pct', '-'), '.1f'):>6} "
              f"{_fmt(row.get('reversion_auc', '-'), '.0f'):>8} "
              f"{_fmt(row.get('life_95', '-')):>8} {_fmt(row.get('life_80', '-')):>8}")
        rows.append(row)
    return rows


# ── comparative charts ────────────────────────────────────────────────────

def plot_peak_vs_frac(finetune_results, ax=None):
    if isinstance(finetune_results, dict):
        finetune_results = [finetune_results]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    fracs = [r["burst_frac"] for r in finetune_results]
    peaks = [r["peak_burst"] for r in finetune_results]
    colors = [_frac_color(f) for f in fracs]
    ax.scatter(fracs, peaks, c=colors, s=80, zorder=3)
    ax.plot(fracs, peaks, color="gray", alpha=0.4, zorder=2)
    ax.set_xlabel("Burst Fraction"); ax.set_ylabel("Peak Burst Accuracy")
    ax.set_title("Peak Accuracy vs Concentration")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
    return ax


def plot_retention_vs_frac(forget_results, ax=None):
    if isinstance(forget_results, dict):
        forget_results = [forget_results]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    tags = [r["tag"] for r in forget_results]
    aucs = [r["reversion_auc"] for r in forget_results]
    fracs = [_tag_to_frac(t) for t in tags]
    colors = [_frac_color(f) for f in fracs]
    ax.bar(range(len(tags)), aucs, color=colors, alpha=0.8)
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Reversion AUC")
    ax.set_title("Knowledge Retention (higher = more retained)")
    ax.grid(True, alpha=0.3, axis="y")
    return ax


# ══════════════════════════════════════════════════════════════════════════
# INTERPRETABILITY PLOTS  — finetune + forget on a single continuous axis
# ══════════════════════════════════════════════════════════════════════════

def plot_weight_drift(ft_results, fg_results, figsize=(12, 5)):
    """Weight L2 drift from pretrained: finetune then forget, single axis."""
    fig, ax = plt.subplots(figsize=figsize)
    for ft, fg in _pair_ft_fg(ft_results, fg_results):
        color = _frac_color(ft["burst_frac"])
        ft_log = ft["log"]
        if "weight_drift" not in ft_log:
            continue
        ax.plot(ft_log["step"], ft_log["weight_drift"],
                color=color, linewidth=2, label=ft["tag"])
        if fg is not None:
            fg_log = fg["log"]
            if "weight_drift_from_pt" in fg_log and fg_log["weight_drift_from_pt"]:
                fg_steps = _offset_fg_steps(ft_log, fg_log)
                ax.plot(fg_steps, fg_log["weight_drift_from_pt"],
                        color=color, linewidth=2, linestyle="-")

    # draw one boundary line (same for all fracs)
    pairs = _pair_ft_fg(ft_results, fg_results)
    if pairs and pairs[0][0]["log"]["step"]:
        _phase_boundary(ax, pairs[0][0]["log"])

    ax.set_xlabel("Step (finetune | forget)")
    ax.set_ylabel("L2 Distance from Pretrained")
    ax.set_title("Weight Drift: Finetune → Forget")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_grad_norms(ft_results, fg_results, figsize=(14, 8)):
    """Gradient norms on a continuous finetune→forget axis.

    Top: burst-only and bg-only norms (finetune only, forget has no burst training).
    Bottom-left: burst/bg ratio (finetune).  Bottom-right: bg norm continuous.
    """
    ft_list = ft_results if isinstance(ft_results, list) else [ft_results]
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Top-left: burst gradient norm (finetune only)
    ax = axes[0, 0]
    for r in ft_list:
        log = r["log"]
        if "grad_norm_burst" not in log: continue
        ax.plot(log["step"], log["grad_norm_burst"], color=_frac_color(r["burst_frac"]),
                label=r["tag"], linewidth=1.5)
    ax.set_xlabel("Step"); ax.set_ylabel("Gradient Norm")
    ax.set_title("Burst-Only Gradient Norm"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Top-right: background gradient norm continuous (finetune bg → forget bg)
    ax = axes[0, 1]
    for ft, fg in _pair_ft_fg(ft_results, fg_results):
        color = _frac_color(ft["burst_frac"])
        ft_log = ft["log"]
        if "grad_norm_bg" not in ft_log: continue
        ax.plot(ft_log["step"], ft_log["grad_norm_bg"], color=color,
                linewidth=1.5, label=ft["tag"])
        if fg is not None and "grad_norm" in fg["log"] and fg["log"]["grad_norm"]:
            fg_steps = _offset_fg_steps(ft_log, fg["log"])
            ax.plot(fg_steps, fg["log"]["grad_norm"], color=color, linewidth=1.5)
    pairs = _pair_ft_fg(ft_results, fg_results)
    if pairs and pairs[0][0]["log"]["step"]:
        _phase_boundary(ax, pairs[0][0]["log"])
    ax.set_xlabel("Step (finetune | forget)"); ax.set_ylabel("Gradient Norm")
    ax.set_title("Background Gradient Norm"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Bottom-left: burst/bg ratio (finetune only)
    ax = axes[1, 0]
    for r in ft_list:
        log = r["log"]
        if "grad_norm_burst" not in log or "grad_norm_bg" not in log: continue
        ratio = [b / (g + 1e-10) for b, g in zip(log["grad_norm_burst"], log["grad_norm_bg"])]
        ax.plot(log["step"], ratio, color=_frac_color(r["burst_frac"]),
                label=r["tag"], linewidth=1.5)
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Step"); ax.set_ylabel("Burst / Background Ratio")
    ax.set_title("Gradient Norm Ratio (finetune)"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Bottom-right: training batch gradient norm continuous
    ax = axes[1, 1]
    for ft, fg in _pair_ft_fg(ft_results, fg_results):
        color = _frac_color(ft["burst_frac"])
        ft_log = ft["log"]
        if "grad_norm_train" not in ft_log: continue
        ax.plot(ft_log["step"], ft_log["grad_norm_train"], color=color,
                linewidth=1.5, label=ft["tag"])
        if fg is not None and "grad_norm" in fg["log"] and fg["log"]["grad_norm"]:
            fg_steps = _offset_fg_steps(ft_log, fg["log"])
            ax.plot(fg_steps, fg["log"]["grad_norm"], color=color, linewidth=1.5)
    if pairs and pairs[0][0]["log"]["step"]:
        _phase_boundary(ax, pairs[0][0]["log"])
    ax.set_xlabel("Step (finetune | forget)"); ax.set_ylabel("Gradient Norm")
    ax.set_title("Training Gradient Norm"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.suptitle("Gradient Norms Over Training", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_grad_cosine(ft_results, fg_results, figsize=(12, 5)):
    """Gradient cosine (burst vs bg) on a single continuous axis per frac."""
    fig, ax = plt.subplots(figsize=figsize)
    for ft, fg in _pair_ft_fg(ft_results, fg_results):
        color = _frac_color(ft["burst_frac"])
        ft_log = ft["log"]
        if "grad_cosine_burst_bg" not in ft_log:
            continue
        ax.plot(ft_log["step"], ft_log["grad_cosine_burst_bg"],
                color=color, linewidth=2, label=ft["tag"])
        if fg is not None:
            fg_log = fg["log"]
            if "grad_cosine_burst_bg" in fg_log and fg_log["grad_cosine_burst_bg"]:
                fg_steps = _offset_fg_steps(ft_log, fg_log)
                ax.plot(fg_steps, fg_log["grad_cosine_burst_bg"],
                        color=color, linewidth=2)

    pairs = _pair_ft_fg(ft_results, fg_results)
    if pairs and pairs[0][0]["log"]["step"]:
        _phase_boundary(ax, pairs[0][0]["log"])

    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Step (finetune | forget)")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Gradient Alignment: Burst vs Background")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _build_layer_cosine_matrix(results):
    """Extract per-layer gradient cosine over steps into (steps, layers) matrix."""
    log = results["log"]
    key = "grad_cosine_per_layer"
    if key not in log or not log[key]:
        return None
    records = log[key]
    layer_names = sorted(records[0].keys())
    steps = log["step"][:len(records)]
    matrix = np.array([[rec.get(l, 0.0) for l in layer_names] for rec in records])
    return matrix, steps, layer_names


def plot_grad_norm_entropy(ft_results, fg_results, figsize=(14, 5)):
    """Gradient norm entropy on a continuous finetune → forget axis.

    High entropy = gradient spread evenly across blocks.
    Low entropy = gradient concentrated in few blocks.
    """
    fig, (ax_burst, ax_bg) = plt.subplots(1, 2, figsize=figsize, sharey=True)
    drawn_boundary = False
    for ft, fg in _pair_ft_fg(ft_results, fg_results):
        color = _frac_color(ft["burst_frac"])
        ft_log = ft["log"]

        # burst entropy (finetune only — no burst training during forget)
        if "grad_norm_entropy_burst" in ft_log and ft_log["grad_norm_entropy_burst"]:
            ax_burst.plot(ft_log["step"], ft_log["grad_norm_entropy_burst"],
                          color=color, linewidth=2, label=ft["tag"])

        # bg entropy: continuous finetune → forget
        if "grad_norm_entropy_bg" in ft_log and ft_log["grad_norm_entropy_bg"]:
            ax_bg.plot(ft_log["step"], ft_log["grad_norm_entropy_bg"],
                       color=color, linewidth=2, label=ft["tag"])
        if fg is not None:
            fg_log = fg["log"]
            if "grad_norm_entropy" in fg_log and fg_log["grad_norm_entropy"]:
                fg_steps = _offset_fg_steps(ft_log, fg_log)
                ax_bg.plot(fg_steps, fg_log["grad_norm_entropy"],
                           color=color, linewidth=2)

        if not drawn_boundary and ft_log["step"]:
            _phase_boundary(ax_burst, ft_log)
            _phase_boundary(ax_bg, ft_log)
            drawn_boundary = True

    ax_burst.set_xlabel("Step"); ax_burst.set_ylabel("Entropy")
    ax_burst.set_title("Burst Gradient Norm Entropy")
    ax_burst.legend(fontsize=7); ax_burst.grid(True, alpha=0.3)

    ax_bg.set_xlabel("Step (finetune | forget)"); ax_bg.set_ylabel("Entropy")
    ax_bg.set_title("Background Gradient Norm Entropy")
    ax_bg.legend(fontsize=7); ax_bg.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_grad_cosine_per_layer(ft_results, fg_results, figsize_per=(14, 3.5)):
    """Heatmap of per-block gradient cosine over training.

    One row per burst fraction.  Finetune and forget are concatenated
    horizontally with a vertical line at the phase boundary.
    """
    pairs = _pair_ft_fg(ft_results, fg_results)
    n_rows = len(pairs)
    fig, axes = plt.subplots(n_rows, 1,
                              figsize=(figsize_per[0], figsize_per[1] * n_rows),
                              squeeze=False)
    vmin, vmax = -1, 1

    for i, (ft, fg) in enumerate(pairs):
        ax = axes[i, 0]
        built_ft = _build_layer_cosine_matrix(ft)
        built_fg = _build_layer_cosine_matrix(fg) if fg is not None else None

        if built_ft is None:
            ax.set_title(f"{ft['tag']} — no data"); continue

        mat_ft, steps_ft, layer_names = built_ft

        if built_fg is not None:
            mat_fg, steps_fg_raw, _ = built_fg
            ft_end = steps_ft[-1]
            steps_fg = [s + ft_end for s in steps_fg_raw]
            all_steps = list(steps_ft) + list(steps_fg)
            matrix = np.concatenate([mat_ft, mat_fg], axis=0)
        else:
            all_steps = list(steps_ft)
            matrix = mat_ft

        im = ax.imshow(matrix.T, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax,
                       extent=[all_steps[0], all_steps[-1],
                               len(layer_names) - 0.5, -0.5])
        ax.set_yticks(range(len(layer_names)))
        ax.set_yticklabels(layer_names, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)

        if built_fg is not None:
            ax.axvline(steps_ft[-1], color="black", linewidth=1.5,
                       linestyle="--", alpha=0.6)

        frac = ft["burst_frac"]
        ax.set_title(f"{ft['tag']} ({frac*100:.0f}%)", fontsize=10)
        ax.set_xlabel("Step (finetune | forget)")

    fig.suptitle("Per-Block Gradient Cosine (Burst vs Background)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ── post-hoc analysis plots ──────────────────────────────────────────────

def plot_per_layer_drift(analysis, phase="pt_ft", figsize=(12, 5)):
    """Heatmap of per-layer weight drift across burst fractions."""
    tags = sorted(analysis.keys(), key=lambda t: analysis[t]["burst_frac"], reverse=True)
    if not tags: return None
    drift_key = f"drift_{phase}"
    sample = analysis[tags[0]][drift_key]
    if sample is None: return None
    layer_names = sorted(sample["per_layer"].keys())
    matrix = np.zeros((len(tags), len(layer_names)))
    for i, tag in enumerate(tags):
        drift = analysis[tag][drift_key]
        for j, ln in enumerate(layer_names):
            matrix[i, j] = drift["per_layer"].get(ln, 0)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(tags)))
    ax.set_yticklabels([f"{t} ({analysis[t]['burst_frac']*100:.0f}%)" for t in tags], fontsize=8)
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, rotation=90, fontsize=6)
    phase_label = "Pretrain → Finetune" if phase == "pt_ft" else "Finetune → Forget"
    ax.set_title(f"Per-Layer Weight Drift ({phase_label})")
    fig.colorbar(im, ax=ax, label="L2 Distance")
    fig.tight_layout()
    return fig


def plot_svd_analysis(analysis, figsize=(14, 5)):
    """Effective rank and spectral norm of weight deltas."""
    tags = sorted(analysis.keys(), key=lambda t: analysis[t]["burst_frac"], reverse=True)
    if not tags: return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    for tag in tags:
        svd = analysis[tag].get("svd_pt_ft")
        if svd is None: continue
        color = _frac_color(analysis[tag]["burst_frac"])
        layers = sorted(svd.keys())
        ax1.plot(range(len(layers)), [svd[l]["effective_rank"] for l in layers],
                 color=color, label=tag, marker="o", markersize=4, linewidth=1.5)
        ax2.plot(range(len(layers)), [svd[l]["spectral_norm"] for l in layers],
                 color=color, label=tag, marker="s", markersize=4, linewidth=1.5)
    ax1.set_xticks(range(len(layers))); ax1.set_xticklabels(layers, rotation=90, fontsize=6)
    ax1.set_ylabel("Effective Rank"); ax1.set_title("Weight Delta Effective Rank per Layer")
    ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)
    ax2.set_xticks(range(len(layers))); ax2.set_xticklabels(layers, rotation=90, fontsize=6)
    ax2.set_ylabel("Spectral Norm"); ax2.set_title("Weight Delta Spectral Norm per Layer")
    ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_cka_matrices(analysis, figsize=(5, 5)):
    """CKA heatmaps: pretrained vs finetuned on burst data."""
    tags = sorted(analysis.keys(), key=lambda t: analysis[t]["burst_frac"], reverse=True)
    n = len(tags)
    if n == 0: return None
    fig, axes = plt.subplots(1, n, figsize=(figsize[0] * n, figsize[1]), squeeze=False)
    for i, tag in enumerate(tags):
        cka = analysis[tag].get("cka_pt_ft_burst")
        if cka is None: continue
        ax = axes[0, i]
        im = ax.imshow(cka, vmin=0, vmax=1, cmap="viridis")
        ax.set_title(f"{tag} ({analysis[tag]['burst_frac']*100:.0f}%)", fontsize=10)
        ax.set_xlabel("Finetuned Layer"); ax.set_ylabel("Pretrained Layer")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("CKA: Pretrained vs Finetuned (Burst Data)", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def plot_summary_dashboard(ft_results, fg_results, analysis, figsize=(16, 10)):
    """Multi-panel dashboard relating burst_frac to all key metrics."""
    ft_list = ft_results if isinstance(ft_results, list) else [ft_results]
    fg_list = fg_results if isinstance(fg_results, list) else [fg_results]
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
    fracs = [r["burst_frac"] for r in ft_list]
    colors = [_frac_color(f) for f in fracs]
    tags = [r["tag"] for r in ft_list]

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(fracs, [r["peak_burst"] for r in ft_list], c=colors, s=80, zorder=3)
    ax.plot(fracs, [r["peak_burst"] for r in ft_list], color="gray", alpha=0.4)
    ax.set_xlabel("Burst Fraction"); ax.set_ylabel("Peak Burst Accuracy")
    ax.set_title("Acquisition"); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    fg_by_tag = {r["tag"]: r for r in fg_list}
    drops = [fg_by_tag[t]["dropoff_pct"] for t in tags if t in fg_by_tag]
    ax.scatter(fracs[:len(drops)], drops, c=colors[:len(drops)], s=80, zorder=3)
    ax.plot(fracs[:len(drops)], drops, color="gray", alpha=0.4)
    ax.set_xlabel("Burst Fraction"); ax.set_ylabel("Accuracy Drop (%)")
    ax.set_title("Forgetting Severity"); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    drifts = [analysis[t]["drift_pt_ft"]["total"] if t in analysis else 0 for t in tags]
    ax.scatter(fracs, drifts, c=colors, s=80, zorder=3)
    ax.plot(fracs, drifts, color="gray", alpha=0.4)
    ax.set_xlabel("Burst Fraction"); ax.set_ylabel("Total L2 Drift")
    ax.set_title("Weight Displacement"); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    mean_ranks = []
    for t in tags:
        if t in analysis and analysis[t].get("svd_pt_ft"):
            mean_ranks.append(np.mean([v["effective_rank"] for v in analysis[t]["svd_pt_ft"].values()]))
        else:
            mean_ranks.append(0)
    ax.scatter(fracs, mean_ranks, c=colors, s=80, zorder=3)
    ax.plot(fracs, mean_ranks, color="gray", alpha=0.4)
    ax.set_xlabel("Burst Fraction"); ax.set_ylabel("Mean Effective Rank")
    ax.set_title("Weight Delta Dimensionality"); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    mean_cka = []
    for t in tags:
        if t in analysis and analysis[t].get("cka_pt_ft_burst") is not None:
            mean_cka.append(np.mean(np.diag(analysis[t]["cka_pt_ft_burst"])))
        else:
            mean_cka.append(0)
    ax.scatter(fracs, mean_cka, c=colors, s=80, zorder=3)
    ax.plot(fracs, mean_cka, color="gray", alpha=0.4)
    ax.set_xlabel("Burst Fraction"); ax.set_ylabel("Mean CKA (diagonal)")
    ax.set_title("Representation Preservation"); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    end_gc = []
    for r in ft_list:
        log = r["log"]
        if "grad_cosine_burst_bg" in log and log["grad_cosine_burst_bg"]:
            end_gc.append(log["grad_cosine_burst_bg"][-1])
        else:
            end_gc.append(0)
    ax.scatter(fracs, end_gc, c=colors, s=80, zorder=3)
    ax.plot(fracs, end_gc, color="gray", alpha=0.4)
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Burst Fraction"); ax.set_ylabel("Gradient Cosine (end)")
    ax.set_title("Final Burst-BG Alignment"); ax.grid(True, alpha=0.3)

    fig.suptitle("Burst Concentration vs Forgetting: Summary Dashboard",
                 fontsize=14, fontweight="bold", y=1.01)
    return fig


# ── save full report ──────────────────────────────────────────────────────

def save_report(pt, ft_results, fg_results, out_dir, analysis=None, prefix="report"):
    """Save all charts to a directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ft_list = ft_results if isinstance(ft_results, list) else [ft_results]
    fg_list = fg_results if isinstance(fg_results, list) else [fg_results]

    def _save(fig, name):
        if fig is not None:
            fig.savefig(out_dir / f"{prefix}_{name}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    _save(plot_full_trajectory(pt, ft_list, fg_list), "01_trajectory")

    if len(ft_list) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_peak_vs_frac(ft_list, axes[0])
        plot_retention_vs_frac(fg_list, axes[1])
        fig.tight_layout()
        _save(fig, "02_comparison")

    _save(plot_weight_drift(ft_list, fg_list), "03_weight_drift")
    _save(plot_grad_cosine(ft_list, fg_list), "04_grad_cosine")
    _save(plot_grad_norms(ft_list, fg_list), "05_grad_norms")
    _save(plot_grad_norm_entropy(ft_list, fg_list), "06_grad_norm_entropy")
    _save(plot_grad_cosine_per_layer(ft_list, fg_list), "07_grad_cosine_per_layer")

    if analysis is not None:
        _save(plot_per_layer_drift(analysis, "pt_ft"), "08_layer_drift_finetune")
        _save(plot_per_layer_drift(analysis, "ft_fg"), "09_layer_drift_forget")
        _save(plot_svd_analysis(analysis), "10_svd_analysis")
        _save(plot_cka_matrices(analysis), "11_cka_matrices")
        _save(plot_summary_dashboard(ft_list, fg_list, analysis), "12_summary_dashboard")

    print(f"Report saved to {out_dir}")
