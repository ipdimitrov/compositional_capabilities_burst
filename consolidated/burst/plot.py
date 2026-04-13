"""Plotting functions for burst forgetting experiments.

`make_all_plots(out_dir)` auto-detects which sweep axes are non-trivial
(have >1 value) and generates the relevant plots. Outputs saved to
{out_dir}/plots/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from burst.sweep import SweepConfig, load_sweep_results


# ── helpers ─────────────────────────────────────────────────────────────

def _safe_last(log: dict, key: str) -> float:
    v = log.get(key, [])
    return v[-1] if v else float("nan")


def _extract_metric(all_ft, all_fg, corr, ft_lr, i_f, seeds, fn):
    vals = []
    for s in seeds:
        ft = all_ft[corr][ft_lr][s][i_f]
        fg = all_fg[corr][ft_lr][s][i_f]
        if ft is not None and fg is not None:
            vals.append(fn(ft, fg))
    return vals


METRIC_FNS = {
    "peak_burst":    lambda ft, fg: ft["peak_burst"],
    "drop_pct":      lambda ft, fg: fg["dropoff_pct"],
    "reversion_auc": lambda ft, fg: fg["reversion_auc"],
    "ft_bg_acc":     lambda ft, fg: ft["log"]["acc_other"][-1],
    "ft_bg_loss":    lambda ft, fg: ft["log"]["loss_other"][-1],
    "ft_burst_acc":  lambda ft, fg: ft["log"]["acc_burst"][-1],
    "fg_burst_acc":  lambda ft, fg: fg["log"]["acc_burst"][-1],
    "fg_bg_acc":     lambda ft, fg: fg["log"]["acc_other"][-1],
}

METRIC_LABELS = {
    "peak_burst": "Peak Burst Acc",
    "drop_pct": "Accuracy Drop %",
    "reversion_auc": "Reversion AUC",
    "ft_bg_acc": "BG Acc (end FT)",
    "ft_bg_loss": "BG Loss (end FT)",
    "ft_burst_acc": "Burst Acc (end FT)",
    "fg_burst_acc": "Burst Acc (end Forget)",
    "fg_bg_acc": "BG Acc (end Forget)",
}

METRIC_CMAPS = {
    "peak_burst": "YlGn",
    "drop_pct": "RdYlGn_r",
    "reversion_auc": "RdYlGn",
    "ft_bg_acc": "RdYlGn",
    "ft_bg_loss": "YlOrRd",
    "ft_burst_acc": "YlGn",
    "fg_burst_acc": "YlGn",
    "fg_bg_acc": "RdYlGn",
}


def _compute_grids(cfg: SweepConfig, all_ft, all_fg, axis_a, axis_b,
                   fix_corr=None, fix_ft_lr=None):
    """Build (axis_a × axis_b) grids of mean/std for each metric.

    axis_a, axis_b: "correlations", "ft_lrs", or "fracs"
    fix_corr / fix_ft_lr: if that axis is not being plotted, fix its value
    """
    axes_map = {
        "correlations": cfg.correlations,
        "ft_lrs": cfg.ft_lrs,
        "fracs": cfg.fracs,
    }
    vals_a = axes_map[axis_a]
    vals_b = axes_map[axis_b]

    corr_fixed = fix_corr if fix_corr is not None else cfg.correlations[0]
    ft_lr_fixed = fix_ft_lr if fix_ft_lr is not None else cfg.ft_lrs[0]

    mean, std = {}, {}
    for name, fn in METRIC_FNS.items():
        m = np.full((len(vals_a), len(vals_b)), np.nan)
        s = np.full((len(vals_a), len(vals_b)), np.nan)
        for i, va in enumerate(vals_a):
            for j, vb in enumerate(vals_b):
                corr = va if axis_a == "correlations" else (vb if axis_b == "correlations" else corr_fixed)
                ft_lr = va if axis_a == "ft_lrs" else (vb if axis_b == "ft_lrs" else ft_lr_fixed)
                if axis_a == "fracs":
                    i_f = i
                elif axis_b == "fracs":
                    i_f = j
                else:
                    i_f = 0  # shouldn't happen — frac always on one axis
                vals = _extract_metric(all_ft, all_fg, corr, ft_lr, i_f, cfg.seeds, fn)
                if vals:
                    m[i, j] = np.nanmean(vals)
                    s[i, j] = np.nanstd(vals)
        mean[name] = m
        std[name] = s
    return mean, std, vals_a, vals_b


# ── training curves (always useful) ─────────────────────────────────────

def plot_training_curves(cfg: SweepConfig, all_ft, all_fg, out_path: Path,
                          corr=None, ft_lr=None):
    """Burst/BG acc and loss over training, one row per slice.

    Slices over the first non-singleton of {correlations, ft_lrs},
    colored by frac.
    """
    if corr is None:
        corr = cfg.correlations[0]
    if ft_lr is None:
        ft_lr = cfg.ft_lrs[0]

    n_frac = len(cfg.fracs)
    frac_colors = plt.cm.viridis(np.linspace(0, 0.9, n_frac))

    fig, axes = plt.subplots(1, 4, figsize=(22, 4.5), sharey='col')

    for i_f, frac in enumerate(cfg.fracs):
        c = frac_colors[i_f]
        label = f"{int(frac*100)}%"
        valid = [s for s in cfg.seeds if all_ft[corr][ft_lr][s][i_f] is not None]
        if not valid:
            continue

        for col, key in enumerate(["acc_burst", "acc_other", "loss_burst", "loss_other"]):
            ax = axes[col]
            ft_c = [all_ft[corr][ft_lr][s][i_f]["log"][key] for s in valid]
            ft_s = all_ft[corr][ft_lr][valid[0]][i_f]["log"]["step"]
            n = min(len(x) for x in ft_c)
            arr = np.array([x[:n] for x in ft_c])
            ax.plot(ft_s[:n], arr.mean(0), color=c, linewidth=1.5,
                    label=label if col == 0 else None)
            ax.fill_between(ft_s[:n], arr.mean(0) - arr.std(0),
                            arr.mean(0) + arr.std(0), color=c, alpha=0.15)

            fg_v = [s for s in valid if all_fg[corr][ft_lr][s][i_f] is not None]
            if fg_v:
                fg_c = [all_fg[corr][ft_lr][s][i_f]["log"][key] for s in fg_v]
                fg_sr = all_fg[corr][ft_lr][fg_v[0]][i_f]["log"]["step"]
                ft_end = ft_s[-1] if ft_s else 0
                nf = min(len(x) for x in fg_c)
                fa = np.array([x[:nf] for x in fg_c])
                fgs = [s + ft_end for s in fg_sr[:nf]]
                ax.plot(fgs, fa.mean(0), color=c, linewidth=1.5, linestyle="--")
                ax.fill_between(fgs, fa.mean(0) - fa.std(0),
                                fa.mean(0) + fa.std(0), color=c, alpha=0.1)

                if i_f == 0:
                    ax.axvline(ft_end, color="black", linewidth=0.7,
                               linestyle="--", alpha=0.4)

    for col, title in enumerate(["Burst Acc", "BG Acc", "Burst Loss", "BG Loss"]):
        axes[col].set_title(title, fontsize=12)
        axes[col].set_xlabel("Step (FT | Forget)")
        axes[col].grid(True, alpha=0.3)

    axes[0].legend(fontsize=8, loc="lower right", ncol=2)

    n_cop = int(round(corr * cfg.n_burst))
    fig.suptitle(
        f"Training Curves (corr={n_cop}/{cfg.n_burst}, ft_lr={ft_lr:.0e}, "
        f"{len(cfg.seeds)} seeds, solid=FT, dashed=Forget)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── heatmaps (2D sweep) ─────────────────────────────────────────────────

def plot_heatmaps(cfg: SweepConfig, all_ft, all_fg, axis_a, axis_b,
                   out_path: Path, fix_corr=None, fix_ft_lr=None):
    """8-panel heatmap grid. axis_a is y, axis_b is x."""
    mean, std, vals_a, vals_b = _compute_grids(
        cfg, all_ft, all_fg, axis_a, axis_b, fix_corr, fix_ft_lr)

    def _label(axis_name, vals):
        if axis_name == "correlations":
            return [f"{int(round(v*cfg.n_burst))}/{cfg.n_burst}" for v in vals]
        if axis_name == "ft_lrs":
            return [f"{v:.0e}" for v in vals]
        if axis_name == "fracs":
            return [f"{int(v*100)}%" for v in vals]
        return [str(v) for v in vals]

    labels_a = _label(axis_a, vals_a)
    labels_b = _label(axis_b, vals_b)

    axis_titles = {
        "correlations": "Correlation (copied/total)",
        "ft_lrs": "Finetune LR",
        "fracs": "Burst Concentration",
    }

    keys = ["peak_burst", "ft_burst_acc", "ft_bg_acc", "ft_bg_loss",
            "drop_pct", "reversion_auc", "fg_burst_acc", "fg_bg_acc"]

    n_cols = 4
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes = axes.flatten()

    for idx, key in enumerate(keys):
        ax = axes[idx]
        grid = mean[key]
        im = ax.imshow(grid, aspect="auto", cmap=METRIC_CMAPS[key])
        ax.set_xticks(range(len(vals_b)))
        ax.set_xticklabels(labels_b, fontsize=8)
        ax.set_yticks(range(len(vals_a)))
        ax.set_yticklabels(labels_a, fontsize=9)
        ax.set_xlabel(axis_titles[axis_b])
        ax.set_ylabel(axis_titles[axis_a])
        ax.set_title(METRIC_LABELS[key], fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        vmin, vmax = np.nanmin(grid), np.nanmax(grid)
        for i in range(len(vals_a)):
            for j in range(len(vals_b)):
                m = grid[i, j]
                s = std[key][i, j]
                if np.isnan(m):
                    continue
                txt = f"{m:.2f}\n±{s:.2f}" if abs(m) < 100 else f"{m:.0f}\n±{s:.0f}"
                color = "white" if (vmax - vmin > 0 and abs(m - vmin) > 0.6 * (vmax - vmin)) else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=5, color=color)

    suffix = []
    if axis_a != "correlations" and axis_b != "correlations" and fix_corr is not None:
        n_cop = int(round(fix_corr * cfg.n_burst))
        suffix.append(f"corr={n_cop}/{cfg.n_burst}")
    if axis_a != "ft_lrs" and axis_b != "ft_lrs" and fix_ft_lr is not None:
        suffix.append(f"ft_lr={fix_ft_lr:.0e}")
    suffix_str = " (" + ", ".join(suffix) + ")" if suffix else ""

    fig.suptitle(
        f"{axis_titles[axis_a]} × {axis_titles[axis_b]}{suffix_str} "
        f"— {len(cfg.seeds)} seeds",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── line plots (metrics vs concentration) ───────────────────────────────

def plot_lines_vs_concentration(cfg: SweepConfig, all_ft, all_fg, out_path: Path,
                                 color_by="correlations", fix_corr=None, fix_ft_lr=None):
    """Metrics vs burst concentration, one line per color_by axis."""
    if color_by not in ("correlations", "ft_lrs"):
        raise ValueError(f"color_by must be 'correlations' or 'ft_lrs', got {color_by}")

    color_vals = cfg.correlations if color_by == "correlations" else cfg.ft_lrs
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(color_vals)))

    def _label(v):
        if color_by == "correlations":
            n_cop = int(round(v * cfg.n_burst))
            return f"{n_cop}/{cfg.n_burst}"
        return f"{v:.0e}"

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = [
        ("peak_burst", axes[0, 0], (-0.05, 1.05)),
        ("ft_bg_acc", axes[0, 1], (-0.05, 1.05)),
        ("ft_bg_loss", axes[0, 2], None),
        ("drop_pct", axes[1, 0], None),
        ("reversion_auc", axes[1, 1], None),
        ("fg_burst_acc", axes[1, 2], (-0.05, 1.05)),
    ]

    for key, ax, ylim in metrics:
        for i_c, cv in enumerate(color_vals):
            corr = cv if color_by == "correlations" else (fix_corr if fix_corr is not None else cfg.correlations[0])
            ft_lr = cv if color_by == "ft_lrs" else (fix_ft_lr if fix_ft_lr is not None else cfg.ft_lrs[0])
            vals_mean, vals_std = [], []
            for i_f in range(len(cfg.fracs)):
                vals = _extract_metric(all_ft, all_fg, corr, ft_lr, i_f, cfg.seeds, METRIC_FNS[key])
                vals_mean.append(np.nanmean(vals) if vals else np.nan)
                vals_std.append(np.nanstd(vals) if vals else np.nan)
            ax.errorbar(cfg.fracs, vals_mean, yerr=vals_std, fmt="o-",
                        color=colors[i_c], label=_label(cv),
                        linewidth=2, markersize=5, capsize=3)
        ax.set_xlabel("Burst Concentration")
        ax.set_title(METRIC_LABELS[key])
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        if ylim:
            ax.set_ylim(*ylim)

    legend_name = {"correlations": "correlation", "ft_lrs": "ft_lr"}[color_by]
    fig.suptitle(
        f"Metrics vs Concentration (colored by {legend_name}, "
        f"{len(cfg.seeds)} seeds)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── bg loss vs forgetting scatter ───────────────────────────────────────

def plot_bgloss_scatter(cfg: SweepConfig, all_ft, all_fg, out_path: Path,
                         color_by="correlations"):
    """Scatter: bg loss at end of FT vs forgetting metrics."""
    color_vals = cfg.correlations if color_by == "correlations" else cfg.ft_lrs
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(color_vals)))

    def _label(v):
        if color_by == "correlations":
            n_cop = int(round(v * cfg.n_burst))
            return f"corr={n_cop}/{cfg.n_burst}"
        return f"ft_lr={v:.0e}"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for i_cv, cv in enumerate(color_vals):
        color = colors[i_cv]
        corr_iter = [cv] if color_by == "correlations" else cfg.correlations
        lr_iter = [cv] if color_by == "ft_lrs" else cfg.ft_lrs
        for corr in corr_iter:
            for ft_lr in lr_iter:
                for i_f, frac in enumerate(cfg.fracs):
                    for seed in cfg.seeds:
                        ft = all_ft[corr][ft_lr][seed][i_f]
                        fg = all_fg[corr][ft_lr][seed][i_f]
                        if ft is None or fg is None:
                            continue
                        bg_loss = ft["log"]["loss_other"][-1]
                        marker = "o" if frac >= 0.9 else ("s" if frac >= 0.5 else "^")
                        ax1.scatter(bg_loss, fg["dropoff_pct"], c=[color],
                                    marker=marker, s=30, alpha=0.6, zorder=3)
                        ax2.scatter(bg_loss, fg["reversion_auc"], c=[color],
                                    marker=marker, s=30, alpha=0.6, zorder=3)

    for i_cv, cv in enumerate(color_vals):
        ax1.scatter([], [], c=[colors[i_cv]], label=_label(cv), s=50)

    ax1.set_xlabel("BG Loss (end FT)"); ax1.set_ylabel("Drop %")
    ax1.set_title("BG Loss → Forgetting"); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("BG Loss (end FT)"); ax2.set_ylabel("Reversion AUC")
    ax2.set_title("BG Loss → Retention"); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── top-level: auto-detect sweep axes and plot everything ───────────────

def make_all_plots(out_dir: str | Path) -> None:
    """Load a sweep and generate all relevant plots into {out_dir}/plots/."""
    out_dir = Path(out_dir)
    cfg, all_ft, all_fg = load_sweep_results(out_dir)

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    n_corr = len(cfg.correlations)
    n_lr = len(cfg.ft_lrs)
    n_frac = len(cfg.fracs)

    print(f"Sweep axes: corr={n_corr}, ft_lr={n_lr}, frac={n_frac}, seeds={len(cfg.seeds)}")

    # Always plot training curves for the first (corr, lr) slice.
    print("  → training curves")
    plot_training_curves(cfg, all_ft, all_fg, plot_dir / "training_curves.png")

    # Plot per-correlation training curves if correlation is swept
    if n_corr > 1:
        for corr in cfg.correlations:
            n_cop = int(round(corr * cfg.n_burst))
            plot_training_curves(
                cfg, all_ft, all_fg,
                plot_dir / f"training_curves_corr{n_cop}_{cfg.n_burst}.png",
                corr=corr,
            )

    # Decide what 2D sweeps to plot
    if n_corr > 1 and n_lr > 1:
        # Full 3D sweep: plot fracs × correlations at each ft_lr,
        # and fracs × ft_lrs at each corr
        for ft_lr in cfg.ft_lrs:
            lr_tag = f"ftlr{ft_lr:.0e}"
            print(f"  → heatmap corr × frac @ {lr_tag}")
            plot_heatmaps(cfg, all_ft, all_fg, "correlations", "fracs",
                          plot_dir / f"heatmap_corr_x_frac_{lr_tag}.png",
                          fix_ft_lr=ft_lr)
        for corr in cfg.correlations:
            n_cop = int(round(corr * cfg.n_burst))
            c_tag = f"corr{n_cop}_{cfg.n_burst}"
            print(f"  → heatmap ft_lr × frac @ {c_tag}")
            plot_heatmaps(cfg, all_ft, all_fg, "ft_lrs", "fracs",
                          plot_dir / f"heatmap_ftlr_x_frac_{c_tag}.png",
                          fix_corr=corr)
        plot_lines_vs_concentration(cfg, all_ft, all_fg,
                                     plot_dir / "lines_vs_conc_by_corr.png",
                                     color_by="correlations")
        plot_lines_vs_concentration(cfg, all_ft, all_fg,
                                     plot_dir / "lines_vs_conc_by_ftlr.png",
                                     color_by="ft_lrs")
        plot_bgloss_scatter(cfg, all_ft, all_fg,
                             plot_dir / "bgloss_scatter_by_corr.png",
                             color_by="correlations")
        plot_bgloss_scatter(cfg, all_ft, all_fg,
                             plot_dir / "bgloss_scatter_by_ftlr.png",
                             color_by="ft_lrs")

    elif n_corr > 1:
        # Correlation × concentration sweep
        print("  → heatmap corr × frac")
        plot_heatmaps(cfg, all_ft, all_fg, "correlations", "fracs",
                      plot_dir / "heatmap_corr_x_frac.png")
        print("  → lines vs concentration (by correlation)")
        plot_lines_vs_concentration(cfg, all_ft, all_fg,
                                     plot_dir / "lines_vs_conc_by_corr.png",
                                     color_by="correlations")
        print("  → BG loss scatter (by correlation)")
        plot_bgloss_scatter(cfg, all_ft, all_fg,
                             plot_dir / "bgloss_scatter_by_corr.png",
                             color_by="correlations")

    elif n_lr > 1:
        # LR × concentration sweep
        print("  → heatmap ft_lr × frac")
        plot_heatmaps(cfg, all_ft, all_fg, "ft_lrs", "fracs",
                      plot_dir / "heatmap_ftlr_x_frac.png")
        print("  → lines vs concentration (by ft_lr)")
        plot_lines_vs_concentration(cfg, all_ft, all_fg,
                                     plot_dir / "lines_vs_conc_by_ftlr.png",
                                     color_by="ft_lrs")
        print("  → BG loss scatter (by ft_lr)")
        plot_bgloss_scatter(cfg, all_ft, all_fg,
                             plot_dir / "bgloss_scatter_by_ftlr.png",
                             color_by="ft_lrs")

    else:
        # Single-config (only fracs vary) — burst_simple style
        print("  → lines vs concentration")
        plot_lines_vs_concentration(cfg, all_ft, all_fg,
                                     plot_dir / "lines_vs_conc.png",
                                     color_by="correlations")
        print("  → BG loss scatter")
        plot_bgloss_scatter(cfg, all_ft, all_fg,
                             plot_dir / "bgloss_scatter.png",
                             color_by="correlations")

    print(f"\nPlots saved to {plot_dir}")
