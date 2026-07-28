"""Generate plots from experiment metrics."""

import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("Agg")

# Color scheme for burst levels
BURST_COLORS = {
    1.0: "#e74c3c",    # red
    0.98: "#d63031",   # red-crimson
    0.95: "#c0392b",   # dark red
    0.90: "#e67e22",   # orange
    0.75: "#f39c12",   # yellow-orange
    0.5: "#2ecc71",    # green
    0.25: "#3498db",   # blue
    0.1: "#2980b9",    # dark blue
}

BURST_LABELS = {
    1.0: "100% domain",
    0.98: "98% domain / 2% pile",
    0.95: "95% domain / 5% pile",
    0.90: "90% domain / 10% pile",
    0.75: "75% domain / 25% pile",
    0.5: "50% domain / 50% pile",
    0.25: "25% domain / 75% pile",
    0.1: "10% domain / 90% pile",
}


def _resolve_domain_name(save_dir):
    """Try to read a human-friendly domain name from config.json in save_dir."""
    config_path = os.path.join(save_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        ds = cfg.get("domain_train_dataset", "")
        # Map known datasets to short names
        if "smiles" in ds.lower() or "chembl" in ds.lower():
            return "Chemistry (SMILES)"
        if "irishman" in ds.lower() or "abc" in ds.lower():
            return "Music (ABC)"
        if "pubmed" in ds.lower() or "biomedical" in ds.lower():
            return "Biomedical (PubMed)"
        # Fallback: use last part of dataset name
        if "/" in ds:
            return ds.split("/")[-1]
    return "Domain"


def load_metrics(path="results/metrics.json"):
    with open(path) as f:
        data = json.load(f)
    # Normalize key names: support both old "code_*" and new "domain_*" keys
    for m in data:
        if "code_val_perplexity" in m and "domain_val_perplexity" not in m:
            m["domain_val_perplexity"] = m["code_val_perplexity"]
            m["domain_val_loss"] = m["code_val_loss"]
    return data


def get_baseline(metrics):
    """Get pretrained baseline metrics."""
    for m in metrics:
        if m["phase"] == "pretrained":
            return m
    return None


def get_post_finetune(metrics):
    """Get post-finetune metrics for each burst level."""
    return {m["burst_level"]: m for m in metrics if m["phase"] == "post_finetune"}


def get_cpt_series(metrics):
    """Get continued pretraining time series for each burst level."""
    series = {}
    for m in metrics:
        if m["phase"] == "continued_pretraining":
            bl = m["burst_level"]
            if bl not in series:
                series[bl] = []
            series[bl].append(m)
    # Sort each by step
    for bl in series:
        series[bl].sort(key=lambda x: x["step"])
    return series


def plot_forgetting_curves(metrics, save_dir="results", dn=None):
    """Plot A: Domain perplexity during continued pretraining."""
    if dn is None:
        dn = _resolve_domain_name(save_dir)
    fig, ax = plt.subplots(figsize=(10, 6))

    baseline = get_baseline(metrics)
    cpt = get_cpt_series(metrics)
    post_ft = get_post_finetune(metrics)

    for bl in sorted(cpt.keys(), reverse=True):
        steps = [m["step"] for m in cpt[bl]]
        ppls = [m["domain_val_perplexity"] for m in cpt[bl]]
        if bl in post_ft:
            steps = [0, *steps]
            ppls = [post_ft[bl]["domain_val_perplexity"], *ppls]
        ax.plot(steps, ppls, color=BURST_COLORS.get(bl, "gray"),
                label=BURST_LABELS.get(bl, f"{bl:.0%}"), linewidth=2, marker="o", markersize=4)

    if baseline:
        ax.axhline(y=baseline["domain_val_perplexity"], color="black",
                   linestyle="--", alpha=0.7, label="Pretrained baseline")

    ax.set_xlabel("Continued Pretraining Steps", fontsize=12)
    ax.set_ylabel(f"{dn} Validation Perplexity", fontsize=12)
    ax.set_title(f"Catastrophic Forgetting: {dn} Perplexity During Continued Pretraining", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "plot_forgetting_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return fig


def plot_normalized_forgetting(metrics, save_dir="results", dn=None):
    """Plot A2: Normalized retention."""
    if dn is None:
        dn = _resolve_domain_name(save_dir)
    fig, ax = plt.subplots(figsize=(10, 6))

    baseline = get_baseline(metrics)
    cpt = get_cpt_series(metrics)
    post_ft = get_post_finetune(metrics)

    if not baseline:
        return None

    base_ppl = baseline["domain_val_perplexity"]

    for bl in sorted(cpt.keys(), reverse=True):
        if bl not in post_ft:
            continue
        ft_ppl = post_ft[bl]["domain_val_perplexity"]
        denom = base_ppl - ft_ppl
        if denom == 0:
            continue

        steps = [0] + [m["step"] for m in cpt[bl]]
        ppls = [ft_ppl] + [m["domain_val_perplexity"] for m in cpt[bl]]
        retention = [(base_ppl - p) / denom for p in ppls]

        ax.plot(steps, retention, color=BURST_COLORS.get(bl, "gray"),
                label=BURST_LABELS.get(bl, f"{bl:.0%}"), linewidth=2, marker="o", markersize=4)

    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(y=0.0, color="black", linestyle="--", alpha=0.7, label="Pretrained baseline (full forgetting)")

    ax.set_xlabel("Continued Pretraining Steps", fontsize=12)
    ax.set_ylabel("Retention (fraction of specialization remaining)", fontsize=12)
    ax.set_title(f"Normalized Forgetting: Retention of {dn} Specialization", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "plot_normalized_forgetting.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return fig


def plot_general_recovery(metrics, save_dir="results", dn=None):
    """Plot B: Pile perplexity during continued pretraining."""
    fig, ax = plt.subplots(figsize=(10, 6))

    baseline = get_baseline(metrics)
    cpt = get_cpt_series(metrics)
    post_ft = get_post_finetune(metrics)

    for bl in sorted(cpt.keys(), reverse=True):
        steps = [m["step"] for m in cpt[bl]]
        ppls = [m["pile_val_perplexity"] for m in cpt[bl]]
        if bl in post_ft:
            steps = [0, *steps]
            ppls = [post_ft[bl]["pile_val_perplexity"], *ppls]
        ax.plot(steps, ppls, color=BURST_COLORS.get(bl, "gray"),
                label=BURST_LABELS.get(bl, f"{bl:.0%}"), linewidth=2, marker="o", markersize=4)

    if baseline:
        ax.axhline(y=baseline["pile_val_perplexity"], color="black",
                   linestyle="--", alpha=0.7, label="Pretrained baseline")

    ax.set_xlabel("Continued Pretraining Steps", fontsize=12)
    ax.set_ylabel("Pile Validation Perplexity", fontsize=12)
    ax.set_title("General Recovery: Pile Perplexity During Continued Pretraining", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "plot_general_recovery.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return fig


def plot_2d_tradeoff(metrics, save_dir="results", dn=None):
    """Plot C: 2D trajectory showing domain vs pile perplexity tradeoff."""
    if dn is None:
        dn = _resolve_domain_name(save_dir)
    fig, ax = plt.subplots(figsize=(10, 8))

    baseline = get_baseline(metrics)
    cpt = get_cpt_series(metrics)
    post_ft = get_post_finetune(metrics)

    for bl in sorted(cpt.keys(), reverse=True):
        pile_ppls = [m["pile_val_perplexity"] for m in cpt[bl]]
        domain_ppls = [m["domain_val_perplexity"] for m in cpt[bl]]
        if bl in post_ft:
            pile_ppls = [post_ft[bl]["pile_val_perplexity"], *pile_ppls]
            domain_ppls = [post_ft[bl]["domain_val_perplexity"], *domain_ppls]

        color = BURST_COLORS.get(bl, "gray")
        ax.plot(pile_ppls, domain_ppls, color=color,
                label=BURST_LABELS.get(bl, f"{bl:.0%}"), linewidth=2, alpha=0.8)
        ax.scatter([pile_ppls[0]], [domain_ppls[0]], color=color, s=120, zorder=5,
                   edgecolors="black", linewidth=1.5, marker="o")
        ax.scatter([pile_ppls[-1]], [domain_ppls[-1]], color=color, s=120, zorder=5,
                   edgecolors="black", linewidth=1.5, marker="s")
        for i in range(0, len(pile_ppls) - 1, max(1, len(pile_ppls) // 5)):
            ax.annotate("", xy=(pile_ppls[i + 1], domain_ppls[i + 1]),
                        xytext=(pile_ppls[i], domain_ppls[i]),
                        arrowprops={"arrowstyle": "->", "color": color, "lw": 1.5})

    if baseline:
        ax.scatter([baseline["pile_val_perplexity"]], [baseline["domain_val_perplexity"]],
                   color="black", s=200, zorder=10, marker="*", label="Pretrained baseline")

    ax.set_xlabel("Pile Validation Perplexity (General Ability)", fontsize=12)
    ax.set_ylabel(f"{dn} Validation Perplexity (Specialized Ability)", fontsize=12)
    ax.set_title("Specialization vs. Generalization Tradeoff", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "plot_2d_tradeoff.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return fig


def plot_finetune_comparison(metrics, save_dir="results", dn=None):
    """Plot D: Bar chart comparing post-finetune metrics across burst levels."""
    if dn is None:
        dn = _resolve_domain_name(save_dir)
    post_ft = get_post_finetune(metrics)
    baseline = get_baseline(metrics)

    if not post_ft:
        return None

    burst_levels = sorted(post_ft.keys(), reverse=True)
    labels = [BURST_LABELS.get(bl, f"{bl:.0%}") for bl in burst_levels]
    domain_ppls = [post_ft[bl]["domain_val_perplexity"] for bl in burst_levels]
    pile_ppls = [post_ft[bl]["pile_val_perplexity"] for bl in burst_levels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = [BURST_COLORS.get(bl, "gray") for bl in burst_levels]
    x = range(len(burst_levels))

    ax1.bar(x, domain_ppls, color=colors, edgecolor="black", linewidth=0.8)
    if baseline:
        ax1.axhline(y=baseline["domain_val_perplexity"], color="black",
                    linestyle="--", alpha=0.7, label="Pretrained baseline")
        ax1.legend()
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel(f"{dn} Validation Perplexity", fontsize=11)
    ax1.set_title(f"{dn} Perplexity After Fine-Tuning", fontsize=12)
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, pile_ppls, color=colors, edgecolor="black", linewidth=0.8)
    if baseline:
        ax2.axhline(y=baseline["pile_val_perplexity"], color="black",
                    linestyle="--", alpha=0.7, label="Pretrained baseline")
        ax2.legend()
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("Pile Validation Perplexity", fontsize=11)
    ax2.set_title("Pile Perplexity After Fine-Tuning", fontsize=12)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Fine-Tuning Phase: Specialization Cost", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(save_dir, "plot_finetune_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def _collect_max_ft_per_burst(records):
    """Return {burst_level: max FT step} from a list of records."""
    result = {}
    for r in records:
        if r.get("phase") == "finetune":
            bl = r["burst_level"]
            if r["step"] > result.get(bl, 0):
                result[bl] = r["step"]
    return result


def _draw_phase_separator(ax, records, with_labels=True) -> None:
    """Draw the FT->CPT phase separator for right-aligned plots.

    With right-alignment every burst's FT ends at global_max_ft. We also add
    faint per-burst colored ticks at each burst's FT start.
    """
    max_ft_per_burst = _collect_max_ft_per_burst(records)
    if not max_ft_per_burst:
        return
    global_max_ft = max(max_ft_per_burst.values())

    for bl, ft_len in max_ft_per_burst.items():
        start_x = global_max_ft - ft_len
        if start_x > 0:
            ax.axvline(x=start_x, color=BURST_COLORS.get(bl, "gray"),
                       linestyle=":", alpha=0.35, linewidth=1.0)

    ax.axvline(x=global_max_ft, color="black", linestyle="--", alpha=0.5, linewidth=1.5)

    if with_labels:
        ylim = ax.get_ylim()
        y_text = ylim[1] - (ylim[1] - ylim[0]) * 0.05
        ax.text(global_max_ft * 0.5, y_text, "Fine-Tuning",
                ha="center", va="top", fontsize=11, fontstyle="italic", alpha=0.5)
        xlim_right = ax.get_xlim()[1]
        ax.text(global_max_ft + (xlim_right - global_max_ft) * 0.5, y_text,
                "Continued Pretraining",
                ha="center", va="top", fontsize=11, fontstyle="italic", alpha=0.5)


def plot_training_loss(loss_history, save_dir="results"):
    """Plot E: Training loss curves across all phases for all burst levels.

    Right-aligned: all bursts' FT phases end at the same x-coordinate.
    """
    if not loss_history:
        return None

    fig, ax = plt.subplots(figsize=(14, 6))

    max_ft_per_burst = {}
    for r in loss_history:
        if r["phase"] == "finetune":
            bl = r["burst_level"]
            if r["step"] > max_ft_per_burst.get(bl, 0):
                max_ft_per_burst[bl] = r["step"]
    global_max_ft = max(max_ft_per_burst.values(), default=0)

    for bl in sorted({r["burst_level"] for r in loss_history}, reverse=True):
        ft = [r for r in loss_history if r["burst_level"] == bl and r["phase"] == "finetune"]
        cpt = [r for r in loss_history if r["burst_level"] == bl and r["phase"] == "continued_pretraining"]
        ft.sort(key=lambda x: x["step"])
        cpt.sort(key=lambda x: x["step"])

        ft_sub = ft[::10] if len(ft) > 100 else ft
        cpt_sub = cpt[::10] if len(cpt) > 100 else cpt

        burst_ft_end = max_ft_per_burst.get(bl, global_max_ft)
        ft_offset = global_max_ft - burst_ft_end
        ft_steps = [r["step"] + ft_offset for r in ft_sub]
        ft_losses = [r["train_loss"] for r in ft_sub]
        cpt_steps = [r["step"] + global_max_ft for r in cpt_sub]
        cpt_losses = [r["train_loss"] for r in cpt_sub]

        color = BURST_COLORS.get(bl, "gray")
        label = BURST_LABELS.get(bl, f"{bl:.0%}")
        ax.plot(ft_steps, ft_losses, color=color, alpha=0.7, linewidth=1.2)
        ax.plot(cpt_steps, cpt_losses, color=color, alpha=0.7, linewidth=1.2, label=label)

    _draw_phase_separator(ax, loss_history)

    ax.set_xlabel("Training Step (unified)", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title("Training Loss Across All Phases", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "plot_training_loss.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return fig


def _get_full_eval_series(metrics, prepend_baseline=True):
    """Get eval metrics across both FT and CPT phases for each burst level.

    Right-aligned: all bursts' FT phases end at global_max_ft. If
    ``prepend_baseline`` is true and a pretrained record exists, the
    baseline is inserted as the first point of each burst's series at that
    burst's FT start (x = ft_offset), so all curves visibly emerge from a
    common (x, y) origin rather than from the first FT eval.
    """
    max_ft_per_burst = {}
    for m in metrics:
        if m["phase"] == "finetune":
            max_ft_per_burst.setdefault(m["burst_level"], []).append(m["step"])
    max_ft_per_burst = {bl: max(steps) for bl, steps in max_ft_per_burst.items()}
    global_max_ft = max(max_ft_per_burst.values(), default=0)

    series = {}
    for m in metrics:
        if m["phase"] == "pretrained":
            continue
        bl = m["burst_level"]
        if bl not in series:
            series[bl] = []
        burst_ft_end = max_ft_per_burst.get(bl, global_max_ft)
        ft_offset = global_max_ft - burst_ft_end
        if m["phase"] == "finetune":
            series[bl].append((ft_offset + m["step"], m))
        elif m["phase"] == "post_finetune":
            series[bl].append((global_max_ft, m))
        elif m["phase"] == "continued_pretraining":
            series[bl].append((global_max_ft + m["step"], m))

    if prepend_baseline:
        baseline = get_baseline(metrics)
        if baseline is not None:
            for bl in series:
                burst_ft_end = max_ft_per_burst.get(bl, global_max_ft)
                ft_offset = global_max_ft - burst_ft_end
                series[bl].append((ft_offset, baseline))

    for bl in series:
        series[bl].sort(key=lambda x: x[0])
    return series, global_max_ft


def plot_domain_val_loss(metrics, save_dir="results", dn=None):
    """Plot F: Domain validation loss across both phases."""
    if dn is None:
        dn = _resolve_domain_name(save_dir)
    fig, ax = plt.subplots(figsize=(14, 6))

    baseline = get_baseline(metrics)
    series, _ = _get_full_eval_series(metrics)

    for bl in sorted(series.keys(), reverse=True):
        steps = [s for s, _ in series[bl]]
        losses = [m["domain_val_loss"] for _, m in series[bl]]
        color = BURST_COLORS.get(bl, "gray")
        ax.plot(steps, losses, color=color, label=BURST_LABELS.get(bl, f"{bl:.0%}"),
                linewidth=2, marker="o", markersize=3)

    if baseline:
        ax.axhline(y=baseline["domain_val_loss"], color="black",
                   linestyle="--", alpha=0.7, label="Pretrained baseline")

    _draw_phase_separator(ax, metrics)

    ax.set_xlabel("Step (unified across phases)", fontsize=12)
    ax.set_ylabel(f"{dn} Validation Loss", fontsize=12)
    ax.set_title(f"{dn} Validation Loss — Full Trajectory", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "plot_domain_val_loss.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return fig


def plot_pile_val_loss(metrics, save_dir="results"):
    """Plot G: Pile validation loss across both phases."""
    fig, ax = plt.subplots(figsize=(14, 6))

    baseline = get_baseline(metrics)
    series, _ = _get_full_eval_series(metrics)

    for bl in sorted(series.keys(), reverse=True):
        steps = [s for s, _ in series[bl]]
        losses = [m["pile_val_loss"] for _, m in series[bl]]
        color = BURST_COLORS.get(bl, "gray")
        ax.plot(steps, losses, color=color, label=BURST_LABELS.get(bl, f"{bl:.0%}"),
                linewidth=2, marker="o", markersize=3)

    if baseline:
        ax.axhline(y=baseline["pile_val_loss"], color="black",
                   linestyle="--", alpha=0.7, label="Pretrained baseline")

    _draw_phase_separator(ax, metrics)

    ax.set_xlabel("Step (unified across phases)", fontsize=12)
    ax.set_ylabel("Pile Validation Loss", fontsize=12)
    ax.set_title("Pile (General) Validation Loss — Full Trajectory", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "plot_pile_val_loss.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return fig


def _parse_grad_series(grad_metrics):
    """Parse grad metrics into per-burst-level series with unified step axis.

    Right-aligned: all bursts' FT phases end at global_max_ft.
    """
    max_ft_per_burst = {}
    for r in grad_metrics:
        if r["phase"] == "finetune":
            bl = r["burst_level"]
            if r["step"] > max_ft_per_burst.get(bl, 0):
                max_ft_per_burst[bl] = r["step"]
    global_max_ft = max(max_ft_per_burst.values(), default=0)

    series = {}
    for bl in sorted({r["burst_level"] for r in grad_metrics}, reverse=True):
        records = sorted(
            [r for r in grad_metrics if r["burst_level"] == bl],
            key=lambda x: (0 if x["phase"] == "finetune" else 1, x["step"]),
        )
        burst_ft_end = max_ft_per_burst.get(bl, global_max_ft)
        ft_offset = global_max_ft - burst_ft_end
        steps = [
            ft_offset + r["step"] if r["phase"] == "finetune"
            else global_max_ft + r["step"]
            for r in records
        ]
        series[bl] = (steps, records)
    return series, global_max_ft


def plot_grad_cosine_sim(grad_metrics, save_dir="results", dn=None):
    """Plot H1: Cosine similarity between domain and pile gradients."""
    if dn is None:
        dn = _resolve_domain_name(save_dir)
    if not grad_metrics:
        return None
    fig, ax = plt.subplots(figsize=(14, 6))
    series, _ = _parse_grad_series(grad_metrics)
    for bl, (steps, records) in series.items():
        vals = [r["grad_cosine_similarity"] for r in records]
        ax.plot(steps, vals, color=BURST_COLORS.get(bl, "gray"),
                label=BURST_LABELS.get(bl, f"{bl:.0%}"), linewidth=2, marker="o", markersize=3)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    _draw_phase_separator(ax, grad_metrics)
    ax.set_xlabel("Step (unified)", fontsize=12)
    ax.set_ylabel("Cosine Similarity", fontsize=12)
    ax.set_title(f"Gradient Cosine Similarity: {dn} vs Pile", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "plot_grad_cosine_sim.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return fig


def plot_domain_grad_norm(grad_metrics, save_dir="results", dn=None):
    """Plot H2: Domain gradient norm across training."""
    if dn is None:
        dn = _resolve_domain_name(save_dir)
    if not grad_metrics:
        return None
    fig, ax = plt.subplots(figsize=(14, 6))
    series, _ = _parse_grad_series(grad_metrics)
    for bl, (steps, records) in series.items():
        vals = [r["domain_grad_norm"] for r in records]
        ax.plot(steps, vals, color=BURST_COLORS.get(bl, "gray"),
                label=BURST_LABELS.get(bl, f"{bl:.0%}"), linewidth=2, marker="o", markersize=3)
    _draw_phase_separator(ax, grad_metrics)
    ax.set_xlabel("Step (unified)", fontsize=12)
    ax.set_ylabel("Gradient Norm", fontsize=12)
    ax.set_title(f"{dn} Gradient Norm", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "plot_domain_grad_norm.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return fig


def plot_pile_grad_norm(grad_metrics, save_dir="results"):
    """Plot H3: Pile gradient norm across training."""
    if not grad_metrics:
        return None
    fig, ax = plt.subplots(figsize=(14, 6))
    series, _ = _parse_grad_series(grad_metrics)
    for bl, (steps, records) in series.items():
        vals = [r["pile_grad_norm"] for r in records]
        ax.plot(steps, vals, color=BURST_COLORS.get(bl, "gray"),
                label=BURST_LABELS.get(bl, f"{bl:.0%}"), linewidth=2, marker="o", markersize=3)
    _draw_phase_separator(ax, grad_metrics)
    ax.set_xlabel("Step (unified)", fontsize=12)
    ax.set_ylabel("Gradient Norm", fontsize=12)
    ax.set_title("Pile (General) Gradient Norm", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "plot_pile_grad_norm.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return fig


def plot_grad_norm_ratio(grad_metrics, save_dir="results", dn=None):
    """Plot H4: Ratio of domain to pile gradient norms."""
    if dn is None:
        dn = _resolve_domain_name(save_dir)
    if not grad_metrics:
        return None
    fig, ax = plt.subplots(figsize=(14, 6))
    series, _ = _parse_grad_series(grad_metrics)
    for bl, (steps, records) in series.items():
        vals = [r["domain_grad_norm"] / max(r["pile_grad_norm"], 1e-10) for r in records]
        ax.plot(steps, vals, color=BURST_COLORS.get(bl, "gray"),
                label=BURST_LABELS.get(bl, f"{bl:.0%}"), linewidth=2, marker="o", markersize=3)
    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, label="Equal norms")
    _draw_phase_separator(ax, grad_metrics)
    ax.set_xlabel("Step (unified)", fontsize=12)
    ax.set_ylabel(f"{dn} / Pile Gradient Norm Ratio", fontsize=12)
    ax.set_title(f"Gradient Norm Ratio: {dn} vs Pile", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "plot_grad_norm_ratio.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return fig


def plot_pile_grad_norm_cosine(grad_metrics, save_dir="results"):
    """Plot H5: Pile gradient norm weighted by cosine similarity."""
    if not grad_metrics:
        return None
    fig, ax = plt.subplots(figsize=(14, 6))
    series, _ = _parse_grad_series(grad_metrics)
    for bl, (steps, records) in series.items():
        vals = [r.get("pile_grad_norm_cosine", r["pile_grad_norm"] * r["grad_cosine_similarity"])
                for r in records]
        ax.plot(steps, vals, color=BURST_COLORS.get(bl, "gray"),
                label=BURST_LABELS.get(bl, f"{bl:.0%}"), linewidth=2, marker="o", markersize=3)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    _draw_phase_separator(ax, grad_metrics)
    ax.set_xlabel("Step (unified)", fontsize=12)
    ax.set_ylabel("Pile Grad Norm × Cosine Similarity", fontsize=12)
    ax.set_title("Pile Gradient Norm Weighted by Cosine Similarity", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "plot_pile_grad_norm_cosine.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return fig


def load_loss_history(path="results/loss_history.json"):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def load_grad_metrics(path="results/grad_metrics.json"):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def generate_all_plots(metrics_path="results/metrics.json", save_dir="results") -> None:
    """Generate all plots from a metrics file."""
    metrics = load_metrics(metrics_path)
    os.makedirs(save_dir, exist_ok=True)

    dn = _resolve_domain_name(save_dir)

    plot_forgetting_curves(metrics, save_dir, dn)
    plot_normalized_forgetting(metrics, save_dir, dn)
    plot_general_recovery(metrics, save_dir, dn)
    plot_2d_tradeoff(metrics, save_dir, dn)
    plot_finetune_comparison(metrics, save_dir, dn)
    plot_domain_val_loss(metrics, save_dir, dn)
    plot_pile_val_loss(metrics, save_dir)

    loss_path = os.path.join(save_dir, "loss_history.json")
    loss_history = load_loss_history(loss_path)
    if loss_history:
        plot_training_loss(loss_history, save_dir)

    grad_path = os.path.join(save_dir, "grad_metrics.json")
    grad_data = load_grad_metrics(grad_path)
    if grad_data:
        plot_grad_cosine_sim(grad_data, save_dir, dn)
        plot_domain_grad_norm(grad_data, save_dir, dn)
        plot_pile_grad_norm(grad_data, save_dir)
        plot_grad_norm_ratio(grad_data, save_dir, dn)
        plot_pile_grad_norm_cosine(grad_data, save_dir)



if __name__ == "__main__":
    generate_all_plots()
