"""Report generation: training curves, interpretability plots, full PDF report.

All functions accept lists of result dicts so you can compare across sweep
configurations.
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
    h = 0.0 + (1.0 - frac) * 0.58  # red -> blue in HSL
    r, g, b = colorsys.hls_to_rgb(h, 0.42, 0.72)
    return f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"


def _tag_to_frac(tag):
    try:
        return int(tag.split("_")[1]) / 100
    except (IndexError, ValueError):
        return 0.5


def _tag_color(tag):
    return _frac_color(_tag_to_frac(tag))


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
    if isinstance(finetune_results, dict):
        finetune_results = [finetune_results]
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 5))
    for r in finetune_results:
        log = r["log"]
        color = _frac_color(r["burst_frac"])
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
    if isinstance(forget_results, dict):
        forget_results = [forget_results]
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 5))
    for r in forget_results:
        log = r["log"]
        color = _tag_color(r["tag"])
        ax.plot(log["step"], log["acc_burst"], color=color, label=f"{r['tag']} (burst)",
                linewidth=2)
        ax.plot(log["step"], log["acc_other"], color=color, label=f"{r['tag']} (other)",
                linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Forget (Reversion) Phase")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    return ax


# ── combined training view ───────────────────────────────────────────────

def plot_full_trajectory(pretrain_result, finetune_results, forget_results,
                         figsize=(16, 5)):
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    plot_pretrain(pretrain_result, axes[0])
    plot_finetune(finetune_results, axes[1])
    plot_forget(forget_results, axes[2])
    fig.tight_layout()
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
    print(header)
    print("-" * len(header))

    for tag in ft_by_tag:
        ft = ft_by_tag[tag]
        fg = fg_by_tag.get(tag)
        row = {"tag": tag, "burst_frac": ft["burst_frac"],
               "peak_burst": ft["peak_burst"]}
        if fg:
            row.update({
                "end_burst": fg["end_burst_acc"],
                "dropoff_pct": fg["dropoff_pct"],
                "reversion_auc": fg["reversion_auc"],
                "life_95": fg["life_times"].get("life_95", "-"),
                "life_80": fg["life_times"].get("life_80", "-"),
            })
        else:
            row.update({"end_burst": "-", "dropoff_pct": "-",
                        "reversion_auc": "-", "life_95": "-", "life_80": "-"})

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
    ax.set_xlabel("Burst Fraction")
    ax.set_ylabel("Peak Burst Accuracy")
    ax.set_title("Peak Accuracy vs Concentration")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
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
# INTERPRETABILITY PLOTS
# ══════════════════════════════════════════════════════════════════════════

def plot_weight_drift(ft_results, fg_results, figsize=(14, 5)):
    """Weight drift during finetune and forget phases."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for r in (ft_results if isinstance(ft_results, list) else [ft_results]):
        log = r["log"]
        if "weight_drift" not in log:
            continue
        color = _frac_color(r["burst_frac"])
        ax1.plot(log["step"], log["weight_drift"], color=color,
                 label=r["tag"], linewidth=2)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("L2 Distance from Pretrained")
    ax1.set_title("Weight Drift During Finetune")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    for r in (fg_results if isinstance(fg_results, list) else [fg_results]):
        log = r["log"]
        color = _tag_color(r["tag"])
        if "weight_drift_from_ft" in log and log["weight_drift_from_ft"]:
            ax2.plot(log["step"], log["weight_drift_from_ft"], color=color,
                     label=f"{r['tag']} (from ft)", linewidth=2)
        if "weight_drift_from_pt" in log and log["weight_drift_from_pt"]:
            ax2.plot(log["step"], log["weight_drift_from_pt"], color=color,
                     label=f"{r['tag']} (from pt)", linewidth=1, linestyle="--")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("L2 Distance")
    ax2.set_title("Weight Drift During Forget")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_grad_cosine(ft_results, ax=None):
    """Gradient cosine similarity between burst and background during finetune."""
    if isinstance(ft_results, dict):
        ft_results = [ft_results]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    for r in ft_results:
        log = r["log"]
        if "grad_cosine_burst_bg" not in log:
            continue
        color = _frac_color(r["burst_frac"])
        ax.plot(log["step"], log["grad_cosine_burst_bg"], color=color,
                label=r["tag"], linewidth=2)
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Gradient Alignment: Burst vs Background")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_per_layer_drift(analysis, phase="pt_ft", figsize=(12, 5)):
    """Heatmap of per-layer weight drift across burst fractions."""
    tags = sorted(analysis.keys(), key=lambda t: analysis[t]["burst_frac"],
                  reverse=True)
    if not tags:
        return None

    drift_key = f"drift_{phase}"
    sample = analysis[tags[0]][drift_key]
    if sample is None:
        return None
    layer_names = sorted(sample["per_layer"].keys())

    matrix = np.zeros((len(tags), len(layer_names)))
    for i, tag in enumerate(tags):
        drift = analysis[tag][drift_key]
        for j, ln in enumerate(layer_names):
            matrix[i, j] = drift["per_layer"].get(ln, 0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(tags)))
    ax.set_yticklabels([f"{t} ({analysis[t]['burst_frac']*100:.0f}%)" for t in tags],
                        fontsize=8)
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, rotation=90, fontsize=6)
    phase_label = "Pretrain -> Finetune" if phase == "pt_ft" else "Finetune -> Forget"
    ax.set_title(f"Per-Layer Weight Drift ({phase_label})")
    fig.colorbar(im, ax=ax, label="L2 Distance")
    fig.tight_layout()
    return fig


def plot_svd_analysis(analysis, figsize=(14, 5)):
    """Effective rank and spectral norm of weight deltas (pretrain -> finetune)."""
    tags = sorted(analysis.keys(), key=lambda t: analysis[t]["burst_frac"],
                  reverse=True)
    if not tags:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for tag in tags:
        svd = analysis[tag].get("svd_pt_ft")
        if svd is None:
            continue
        frac = analysis[tag]["burst_frac"]
        color = _frac_color(frac)
        layers = sorted(svd.keys())
        ranks = [svd[l]["effective_rank"] for l in layers]
        spectral = [svd[l]["spectral_norm"] for l in layers]

        ax1.plot(range(len(layers)), ranks, color=color, label=tag,
                 marker="o", markersize=4, linewidth=1.5)
        ax2.plot(range(len(layers)), spectral, color=color, label=tag,
                 marker="s", markersize=4, linewidth=1.5)

    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels(layers, rotation=90, fontsize=6)
    ax1.set_ylabel("Effective Rank")
    ax1.set_title("Weight Delta Effective Rank per Layer")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels(layers, rotation=90, fontsize=6)
    ax2.set_ylabel("Spectral Norm")
    ax2.set_title("Weight Delta Spectral Norm per Layer")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_cka_matrices(analysis, figsize=(5, 5)):
    """CKA heatmaps: pretrained vs finetuned on burst data."""
    tags = sorted(analysis.keys(), key=lambda t: analysis[t]["burst_frac"],
                  reverse=True)
    n = len(tags)
    if n == 0:
        return None

    fig, axes = plt.subplots(1, n, figsize=(figsize[0] * n, figsize[1]),
                              squeeze=False)
    for i, tag in enumerate(tags):
        cka = analysis[tag].get("cka_pt_ft_burst")
        if cka is None:
            continue
        ax = axes[0, i]
        im = ax.imshow(cka, vmin=0, vmax=1, cmap="viridis")
        frac = analysis[tag]["burst_frac"]
        ax.set_title(f"{tag} ({frac*100:.0f}%)", fontsize=10)
        ax.set_xlabel("Finetuned Layer")
        ax.set_ylabel("Pretrained Layer")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("CKA: Pretrained vs Finetuned (Burst Data)", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def plot_sharpness(analysis, ax=None):
    """Sharpness (loss sensitivity) comparison across burst fractions."""
    tags = sorted(analysis.keys(), key=lambda t: analysis[t]["burst_frac"],
                  reverse=True)
    if not tags:
        return None
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    fracs = [analysis[t]["burst_frac"] for t in tags]
    sharp_burst = [analysis[t]["sharpness_burst"]["sharpness"] for t in tags]
    sharp_other = [analysis[t]["sharpness_other"]["sharpness"] for t in tags]
    colors = [_frac_color(f) for f in fracs]

    x = np.arange(len(tags))
    w = 0.35
    ax.bar(x - w/2, sharp_burst, w, color=colors, alpha=0.9, label="Burst tasks")
    ax.bar(x + w/2, sharp_other, w, color=colors, alpha=0.4, label="Other tasks")
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Sharpness (loss increase)")
    ax.set_title("Loss Landscape Sharpness at Finetune Endpoint")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    return ax


def plot_summary_dashboard(ft_results, fg_results, analysis, figsize=(16, 10)):
    """Multi-panel dashboard relating burst_frac to all key metrics."""
    ft_list = ft_results if isinstance(ft_results, list) else [ft_results]
    fg_list = fg_results if isinstance(fg_results, list) else [fg_results]

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    fracs = [r["burst_frac"] for r in ft_list]
    colors = [_frac_color(f) for f in fracs]
    tags = [r["tag"] for r in ft_list]

    # 1. Peak burst acc vs frac
    ax = fig.add_subplot(gs[0, 0])
    peaks = [r["peak_burst"] for r in ft_list]
    ax.scatter(fracs, peaks, c=colors, s=80, zorder=3)
    ax.plot(fracs, peaks, color="gray", alpha=0.4)
    ax.set_xlabel("Burst Fraction")
    ax.set_ylabel("Peak Burst Accuracy")
    ax.set_title("Acquisition")
    ax.grid(True, alpha=0.3)

    # 2. Dropoff % vs frac
    ax = fig.add_subplot(gs[0, 1])
    fg_by_tag = {r["tag"]: r for r in fg_list}
    drops = [fg_by_tag[t]["dropoff_pct"] for t in tags if t in fg_by_tag]
    ax.scatter(fracs[:len(drops)], drops, c=colors[:len(drops)], s=80, zorder=3)
    ax.plot(fracs[:len(drops)], drops, color="gray", alpha=0.4)
    ax.set_xlabel("Burst Fraction")
    ax.set_ylabel("Accuracy Drop (%)")
    ax.set_title("Forgetting Severity")
    ax.grid(True, alpha=0.3)

    # 3. Weight drift (finetune) vs frac
    ax = fig.add_subplot(gs[0, 2])
    drifts = []
    for t in tags:
        if t in analysis and analysis[t]["drift_pt_ft"]:
            drifts.append(analysis[t]["drift_pt_ft"]["total"])
        else:
            drifts.append(0)
    ax.scatter(fracs, drifts, c=colors, s=80, zorder=3)
    ax.plot(fracs, drifts, color="gray", alpha=0.4)
    ax.set_xlabel("Burst Fraction")
    ax.set_ylabel("Total L2 Drift")
    ax.set_title("Weight Displacement")
    ax.grid(True, alpha=0.3)

    # 4. Sharpness vs frac
    ax = fig.add_subplot(gs[1, 0])
    sharp = [analysis[t]["sharpness_burst"]["sharpness"]
             for t in tags if t in analysis]
    ax.scatter(fracs[:len(sharp)], sharp, c=colors[:len(sharp)], s=80, zorder=3)
    ax.plot(fracs[:len(sharp)], sharp, color="gray", alpha=0.4)
    ax.set_xlabel("Burst Fraction")
    ax.set_ylabel("Sharpness")
    ax.set_title("Loss Landscape Sharpness")
    ax.grid(True, alpha=0.3)

    # 5. Mean SVD effective rank vs frac
    ax = fig.add_subplot(gs[1, 1])
    mean_ranks = []
    for t in tags:
        if t in analysis and analysis[t].get("svd_pt_ft"):
            ranks = [v["effective_rank"]
                     for v in analysis[t]["svd_pt_ft"].values()]
            mean_ranks.append(np.mean(ranks))
        else:
            mean_ranks.append(0)
    ax.scatter(fracs, mean_ranks, c=colors, s=80, zorder=3)
    ax.plot(fracs, mean_ranks, color="gray", alpha=0.4)
    ax.set_xlabel("Burst Fraction")
    ax.set_ylabel("Mean Effective Rank")
    ax.set_title("Weight Delta Dimensionality")
    ax.grid(True, alpha=0.3)

    # 6. Mean CKA diagonal (pt vs ft on burst data) vs frac
    ax = fig.add_subplot(gs[1, 2])
    mean_cka = []
    for t in tags:
        if t in analysis and analysis[t].get("cka_pt_ft_burst") is not None:
            diag = np.diag(analysis[t]["cka_pt_ft_burst"])
            mean_cka.append(np.mean(diag))
        else:
            mean_cka.append(0)
    ax.scatter(fracs, mean_cka, c=colors, s=80, zorder=3)
    ax.plot(fracs, mean_cka, color="gray", alpha=0.4)
    ax.set_xlabel("Burst Fraction")
    ax.set_ylabel("Mean CKA (diagonal)")
    ax.set_title("Representation Preservation")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Burst Concentration vs Forgetting: Summary Dashboard",
                 fontsize=14, fontweight="bold", y=1.01)
    return fig


# ══════════════════════════════════════════════════════════════════════════
# REPRESENTATION ANALYSIS PLOTS
# ══════════════════════════════════════════════════════════════════════════

def plot_separation_profile(rep_analysis, figsize=(16, 5)):
    """Per-layer Fisher separation: pretrained vs finetuned vs forgotten.

    Shows how burst/background separability changes at each layer across phases.
    """
    tags = [t for t in rep_analysis
            if not t.startswith("_") and "burst_frac" in rep_analysis[t]]
    tags = sorted(tags, key=lambda t: rep_analysis[t]["burst_frac"], reverse=True)
    if not tags:
        return None

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    phase_keys = [("separation_pt", "Pretrained"),
                  ("separation_ft", "Finetuned"),
                  ("separation_fg", "Forgotten")]

    for ax, (key, title) in zip(axes, phase_keys):
        for tag in tags:
            sep = rep_analysis[tag].get(key)
            if sep is None:
                continue
            frac = rep_analysis[tag]["burst_frac"]
            color = _frac_color(frac)
            layers = sorted(sep.keys())
            fisher = [sep[l]["fisher"] for l in layers]
            ax.plot(layers, fisher, color=color, label=tag,
                    marker="o", markersize=4, linewidth=2)
        ax.set_xlabel("Layer")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Fisher Separation (burst vs bg)")
    fig.suptitle("Burst-Background Separability per Layer Across Phases",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_probing_profile(rep_analysis, figsize=(16, 5)):
    """Per-layer linear probing accuracy across phases."""
    tags = [t for t in rep_analysis
            if not t.startswith("_") and "burst_frac" in rep_analysis[t]]
    tags = sorted(tags, key=lambda t: rep_analysis[t]["burst_frac"], reverse=True)
    if not tags:
        return None

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    phase_keys = [("probing_pt", "Pretrained"),
                  ("probing_ft", "Finetuned"),
                  ("probing_fg", "Forgotten")]

    for ax, (key, title) in zip(axes, phase_keys):
        for tag in tags:
            probe = rep_analysis[tag].get(key)
            if probe is None:
                continue
            frac = rep_analysis[tag]["burst_frac"]
            color = _frac_color(frac)
            layers = sorted(probe.keys())
            accs = [probe[l] for l in layers]
            ax.plot(layers, accs, color=color, label=tag,
                    marker="s", markersize=4, linewidth=2)
        ax.axhline(0.5, color="black", linewidth=0.5, linestyle=":")
        ax.set_xlabel("Layer")
        ax.set_title(title)
        ax.set_ylim(0.4, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Linear Probe Accuracy (burst vs bg)")
    fig.suptitle("Burst Identifiability per Layer (Linear Probing)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_representation_drift(rep_analysis, figsize=(14, 5)):
    """Per-layer centroid drift (pretrained->finetuned) for burst vs bg data."""
    tags = [t for t in rep_analysis
            if not t.startswith("_") and "burst_frac" in rep_analysis[t]]
    tags = sorted(tags, key=lambda t: rep_analysis[t]["burst_frac"], reverse=True)
    if not tags:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for tag in tags:
        frac = rep_analysis[tag]["burst_frac"]
        color = _frac_color(frac)

        drift_b = rep_analysis[tag].get("drift_pt_ft_burst")
        drift_g = rep_analysis[tag].get("drift_pt_ft_bg")
        if drift_b is None:
            continue

        layers = sorted(drift_b.keys())
        ax1.plot(layers, [drift_b[l]["centroid_l2"] for l in layers],
                 color=color, label=tag, marker="o", markersize=4, linewidth=2)
        ax2.plot(layers, [drift_g[l]["centroid_l2"] for l in layers],
                 color=color, label=tag, marker="o", markersize=4, linewidth=2)

    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Centroid L2 Drift")
    ax1.set_title("Burst Data: Representation Drift (PT -> FT)")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Layer")
    ax2.set_title("Background Data: Representation Drift (PT -> FT)")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_pca_scatter(rep_analysis, layer_key="_pca_last_layer", figsize=(6, 5)):
    """PCA scatter of burst/bg representations across models."""
    pca_data = rep_analysis.get(layer_key)
    if pca_data is None:
        return None

    labels = pca_data["labels"]
    n_models = len(labels)
    ev = pca_data["explained_var"]

    # colour palette: one hue per model, burst=filled, bg=open
    cmap = plt.cm.tab10
    fig, ax = plt.subplots(figsize=figsize)

    for i, label in enumerate(labels):
        c = cmap(i / max(n_models - 1, 1))
        bc = pca_data["burst_coords"][i]
        gc = pca_data["bg_coords"][i]
        ax.scatter(bc[:, 0], bc[:, 1], color=c, marker="o", s=12, alpha=0.5)
        ax.scatter(gc[:, 0], gc[:, 1], color=c, marker="x", s=12, alpha=0.3)
        # plot centroids
        ax.scatter(bc[:, 0].mean(), bc[:, 1].mean(), color=c, marker="o",
                   s=120, edgecolors="black", linewidths=1.2,
                   label=f"{label} burst", zorder=5)
        ax.scatter(gc[:, 0].mean(), gc[:, 1].mean(), color=c, marker="X",
                   s=120, edgecolors="black", linewidths=1.2,
                   label=f"{label} bg", zorder=5)

    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    layer_name = "Last Layer" if "last" in layer_key else "Mid Layer"
    ax.set_title(f"Representation PCA — {layer_name}")
    ax.legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def plot_separation_summary(rep_analysis, figsize=(14, 5)):
    """Summary: mean Fisher separation and mean probing accuracy vs burst_frac
    at pretrained / finetuned / forgotten phases."""
    tags = [t for t in rep_analysis
            if not t.startswith("_") and "burst_frac" in rep_analysis[t]]
    tags = sorted(tags, key=lambda t: rep_analysis[t]["burst_frac"])
    if not tags:
        return None

    fracs = [rep_analysis[t]["burst_frac"] for t in tags]
    colors = [_frac_color(f) for f in fracs]

    def _mean_metric(tag, key):
        d = rep_analysis[tag].get(key)
        if d is None:
            return np.nan
        if isinstance(d, dict) and all(isinstance(v, dict) for v in d.values()):
            return np.mean([v.get("fisher", 0) for v in d.values()])
        return np.mean(list(d.values()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Fisher separation across phases
    for phase, marker, ls in [("separation_pt", "^", "--"),
                               ("separation_ft", "o", "-"),
                               ("separation_fg", "s", ":")]:
        vals = [_mean_metric(t, phase) for t in tags]
        label = phase.split("_")[1].upper()
        ax1.plot(fracs, vals, marker=marker, linestyle=ls, linewidth=2,
                 label=label, markersize=8)
    ax1.set_xlabel("Burst Fraction")
    ax1.set_ylabel("Mean Fisher Separation")
    ax1.set_title("Burst-BG Separation Across Phases")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Probing accuracy across phases
    for phase, marker, ls in [("probing_pt", "^", "--"),
                               ("probing_ft", "o", "-"),
                               ("probing_fg", "s", ":")]:
        vals = []
        for t in tags:
            d = rep_analysis[t].get(phase)
            vals.append(np.mean(list(d.values())) if d else np.nan)
        label = phase.split("_")[1].upper()
        ax2.plot(fracs, vals, marker=marker, linestyle=ls, linewidth=2,
                 label=label, markersize=8)
    ax2.axhline(0.5, color="black", linewidth=0.5, linestyle=":")
    ax2.set_xlabel("Burst Fraction")
    ax2.set_ylabel("Mean Probing Accuracy")
    ax2.set_title("Burst Identifiability Across Phases")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Representation Analysis Summary", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ── save full report ──────────────────────────────────────────────────────

def save_report(pt, ft_results, fg_results, out_dir, analysis=None,
                rep_analysis=None, prefix="report"):
    """Save all charts to a directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ft_list = ft_results if isinstance(ft_results, list) else [ft_results]
    fg_list = fg_results if isinstance(fg_results, list) else [fg_results]

    def _save(fig, name):
        if fig is not None:
            fig.savefig(out_dir / f"{prefix}_{name}.png", dpi=150,
                        bbox_inches="tight")
            plt.close(fig)

    # 1. Training trajectory
    _save(plot_full_trajectory(pt, ft_list, fg_list), "01_trajectory")

    # 2. Comparison charts
    if len(ft_list) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_peak_vs_frac(ft_list, axes[0])
        plot_retention_vs_frac(fg_list, axes[1])
        fig.tight_layout()
        _save(fig, "02_comparison")

    # 3. Weight drift during training
    _save(plot_weight_drift(ft_list, fg_list), "03_weight_drift")

    # 4. Gradient cosine
    fig_gc, ax_gc = plt.subplots(figsize=(8, 4))
    plot_grad_cosine(ft_list, ax_gc)
    _save(fig_gc, "04_grad_cosine")

    if analysis is not None:
        # 5. Per-layer drift heatmap
        _save(plot_per_layer_drift(analysis, "pt_ft"), "05_layer_drift_finetune")
        _save(plot_per_layer_drift(analysis, "ft_fg"), "06_layer_drift_forget")

        # 6. SVD analysis
        _save(plot_svd_analysis(analysis), "07_svd_analysis")

        # 7. CKA matrices
        _save(plot_cka_matrices(analysis), "08_cka_matrices")

        # 8. Sharpness
        fig_sh, ax_sh = plt.subplots(figsize=(8, 4))
        plot_sharpness(analysis, ax_sh)
        _save(fig_sh, "09_sharpness")

        # 9. Summary dashboard
        _save(plot_summary_dashboard(ft_list, fg_list, analysis),
              "10_summary_dashboard")

    if rep_analysis is not None:
        # 11. Separation profiles
        _save(plot_separation_profile(rep_analysis), "11_separation_profile")

        # 12. Probing accuracy
        _save(plot_probing_profile(rep_analysis), "12_probing_profile")

        # 13. Representation drift
        _save(plot_representation_drift(rep_analysis), "13_rep_drift")

        # 14. PCA scatter
        _save(plot_pca_scatter(rep_analysis, "_pca_last_layer"),
              "14_pca_last_layer")
        _save(plot_pca_scatter(rep_analysis, "_pca_mid_layer"),
              "15_pca_mid_layer")

        # 16. Separation summary
        _save(plot_separation_summary(rep_analysis), "16_separation_summary")

    print(f"Report saved to {out_dir}")
