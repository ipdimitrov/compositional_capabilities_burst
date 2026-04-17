"""Render publication-quality charts from bundled experiment data.

All overlays filter to schedules with concentration >= MIN_CONCENTRATION_PCT.
Main-body figures (fig2/fig4/fig5) plus split layouts for grad norms,
representation drift, and extended AUC bars.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from burst.core.bundle import (
        CoreBundle,
        LifeEntry,
        MeanCI,
        SeriesMeanCI,
        TrainingSchedule,
        TrainingSeries,
    )

from burst.config import (
    CLASS_BURST,
    CLASS_OTHER,
    COLOR_OTHER,
    COLOR_PROJECTION,
    COLOR_SHIFT,
    COLOR_SPECIAL,
    COLOR_TABLE_EDGE,
    COLOR_TABLE_HEADER,
    COLOR_ZERO_LINE,
    DRIFT_CMAP,
    LAYER_CMAP,
    LAYER_LINE_CMAP,
    MIN_CONCENTRATION_PCT,
    SCHED_COLORS,
    SCHED_DISPLAY,
    reversion_life_label,
)
from burst.core.charts.style import apply_paper_style, figsize, save_figure, style_axes
from burst.core.train_utils import mean_ci

ACC_YLIM: tuple[float, float] = (-0.05, 1.05)

VARIANT_OFFSETS: dict[str, int] = {"0rev": 0, "100rev": 100, "600rev": 600}
ZOOM_VARIANT_OFFSETS: dict[str, int] = {"100rev": 100, "600rev": 600}


def with_variant(path: Path, variant: str) -> Path:
    """Insert `_<variant>` before the file extension."""
    return path.with_stem(f"{path.stem}_{variant}")


# ---------------------------------------------------------------------------
# Filters + small utilities
# ---------------------------------------------------------------------------

def filter_schedules(bundle: CoreBundle) -> list[str]:
    """Return schedules with concentration >= MIN_CONCENTRATION_PCT."""
    kept: list[str] = []
    for s in bundle.config.schedules:
        raw = sched_pct_label(s).rstrip("%")
        pct = int(raw) if raw.isdigit() else 0
        if pct >= MIN_CONCENTRATION_PCT:
            kept.append(s)
    return kept


def training_metric(ts: TrainingSchedule, name: str) -> TrainingSeries:
    """Look up a training metric by name on a TrainingSchedule."""
    return getattr(ts, name)


def loss_ylim(bundle: CoreBundle, metrics: list[str]) -> tuple[float, float]:
    """Shared y-axis range across metrics and filtered schedules."""
    lo, hi = float("inf"), float("-inf")
    for schedule in filter_schedules(bundle):
        ts = bundle.training[schedule]
        for metric in metrics:
            data = training_metric(ts, metric)
            mean = np.array(data.mean, dtype=float)
            ci = np.array(data.ci, dtype=float)
            lo = min(lo, float(np.nanmin(mean - ci)))
            hi = max(hi, float(np.nanmax(mean + ci)))
    pad = (hi - lo) * 0.05
    return (lo - pad, hi + pad)


def tight_ylim(low: np.ndarray, high: np.ndarray) -> tuple[float, float]:
    """Return a padded (min, max) ylim from per-point low/high arrays."""
    lo = float(np.nanmin(low))
    hi = float(np.nanmax(high))
    pad = (hi - lo) * 0.08 or 0.01
    return (lo - pad, hi + pad)


def save_with_log_variant(fig: Figure, ax: Axes, path: Path) -> list[Path]:
    """Save a chart in linear scale plus a '_log' variant.

    Log floor is set to one decade below the smallest strictly-positive value
    actually plotted (over all lines), so we don't waste decades of whitespace
    when the data never approaches zero.
    """
    linear_path = save_chart(fig, path)
    orig_ylim = ax.get_ylim()
    all_y = np.concatenate([
        np.asarray(ln.get_ydata(), dtype=float) for ln in ax.lines
    ]) if ax.lines else np.array([])
    positive = all_y[np.isfinite(all_y) & (all_y > 0)]
    floor = 10 ** np.floor(np.log10(float(positive.min()))) if positive.size else 1e-6
    ax.set_yscale("log")
    ax.set_ylim(bottom=floor, top=orig_ylim[1])
    log_path = save_chart(fig, path.with_stem(path.stem + "_log"))
    ax.set_yscale("linear")
    ax.set_ylim(orig_ylim)
    return [linear_path, log_path]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def render_core_charts(bundle: CoreBundle, out_dir: str | Path) -> list[Path]:
    """Render all core analysis charts to out_dir."""
    apply_paper_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_loss_yl = loss_ylim(bundle, ["loss_burst", "loss_other"])
    train_loss_yl = loss_ylim(bundle, ["loss"])

    paths: list[Path | None] = [
        plot_schedule_bars(bundle, out_dir),
        plot_auc_bars(bundle, out_dir),
        plot_summary_table(bundle, out_dir),
    ]
    paths.extend(plot_extended_auc_bars_split(bundle, out_dir))
    paths.extend(plot_representation_drift_split(bundle, out_dir) or [])
    paths.extend(plot_burst_representation_drift_split(bundle, out_dir) or [])
    paths.extend(plot_centroid_norms_split(bundle, out_dir) or [])
    paths.extend(plot_per_layer_heatmaps(bundle, out_dir))
    paths.extend(plot_weight_drift_heatmaps(bundle, out_dir))
    paths.extend(plot_probe_charts(bundle, out_dir))

    for variant, offset in VARIANT_OFFSETS.items():
        paths.extend(
            plot_fig4_grad_4panel(bundle, out_dir, variant, offset, panel3)
            for panel3 in FIG4_PANEL3_VARIANTS
        )
        paths.append(plot_lr_curves(bundle, out_dir, variant, offset))
        paths.append(plot_training_overlay(
            bundle, out_dir, "acc_burst", f"{CLASS_BURST} Accuracy",
            "overlay_acc_burst.pdf", ACC_YLIM, variant, offset,
        ))
        paths.append(plot_training_overlay(
            bundle, out_dir, "acc_other", f"{CLASS_OTHER} Accuracy",
            "overlay_acc_other.pdf", ACC_YLIM, variant, offset,
        ))
        paths.extend(plot_training_overlay_with_log(
            bundle, out_dir, "loss", "Loss",
            "overlay_loss_training.pdf", train_loss_yl, variant, offset,
        ))
        paths.extend(plot_training_overlay_with_log(
            bundle, out_dir, "loss_burst", f"{CLASS_BURST} Eval Loss",
            "overlay_loss_burst.pdf", eval_loss_yl, variant, offset,
        ))
        paths.extend(plot_training_overlay_with_log(
            bundle, out_dir, "loss_other", f"{CLASS_OTHER} Eval Loss",
            "overlay_loss_other.pdf", eval_loss_yl, variant, offset,
        ))
        paths.append(plot_grad_cosine(bundle, out_dir, variant, offset))
        paths.append(plot_grad_rank(bundle, out_dir, variant, offset))
        paths.extend(plot_signed_dot(bundle, out_dir, variant, offset) or [])
        paths.extend(plot_grad_norms_split(bundle, out_dir, variant, offset) or [])
        paths.extend(plot_interference_power(bundle, out_dir, variant, offset) or [])
        paths.extend(plot_per_schedule(bundle, out_dir, variant, offset))
        paths.extend(plot_per_schedule_loss(bundle, out_dir, eval_loss_yl, variant, offset))
        paths.extend(plot_per_layer_lines(bundle, out_dir, variant, offset))
        paths.extend(plot_weight_drift(bundle, out_dir, variant, offset))

    for variant, rev_offset in ZOOM_VARIANT_OFFSETS.items():
        paths.append(plot_fig2_main_cc_summary(bundle, out_dir, variant, rev_offset))
        paths.append(plot_fig2_loss_stack(bundle, out_dir, variant, rev_offset, log=False))
        paths.append(plot_fig2_loss_stack(bundle, out_dir, variant, rev_offset, log=True))
        paths.append(plot_fig2_horizontal_acc(bundle, out_dir, variant, rev_offset))
        paths.append(plot_fig2_horizontal_loss(bundle, out_dir, variant, rev_offset))
        paths.append(plot_reversion_zoom(bundle, out_dir, variant, rev_offset))
        paths.extend(plot_reversion_zoom_loss(bundle, out_dir, variant, rev_offset))

    return [p for p in paths if p is not None]


# ---------------------------------------------------------------------------
# Main-body paper figures (Fig 2, Fig 4, Fig 5)
# ---------------------------------------------------------------------------

def draw_auc_panel(ax: Axes, bundle: CoreBundle) -> None:
    """Draw the reversion AUC bar panel onto ax."""
    schedules = filter_schedules(bundle)
    by_sched = bundle.summary.by_schedule
    means = [by_sched[s].reversion_auc.mean for s in schedules]
    cis = [by_sched[s].reversion_auc.ci for s in schedules]
    xs = np.arange(len(schedules))
    colors = [SCHED_COLORS[s] for s in schedules]
    ax.bar(xs, means, yerr=cis, color=colors, edgecolor="black", lw=0.7, capsize=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [SCHED_DISPLAY.get(s, s) for s in schedules],
        rotation=20, ha="right", fontsize=6,
    )
    style_axes(ax, "", "Reversion AUC")


def draw_reversion_trace(  # noqa: PLR0913
    ax: Axes, bundle: CoreBundle, metric: str, ylabel: str, rev_offset: int,
    *, log: bool = False, ylim: tuple[float, float] | None = None,
) -> None:
    """Draw per-schedule reversion-phase trace of `metric` on ax."""
    for schedule in filter_schedules(bundle):
        ts = bundle.training[schedule]
        data = training_metric(ts, metric)
        steps = np.array(data.steps, dtype=float)
        mean = np.array(data.mean, dtype=float)
        ci = np.array(data.ci, dtype=float)
        burst_end = ts.pre_steps + ts.burst_steps
        mask = (steps >= burst_end) & (steps <= burst_end + rev_offset)
        local = steps[mask] - burst_end
        ax.plot(
            local, mean[mask], color=SCHED_COLORS[schedule],
            label=SCHED_DISPLAY.get(schedule, schedule),
        )
        ax.fill_between(
            local, mean[mask] - ci[mask], mean[mask] + ci[mask],
            color=SCHED_COLORS[schedule], alpha=0.12,
        )
    if log:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(ylim)
    style_axes(ax, "Reversion Step", ylabel)
    ax.legend(loc="best", ncol=2, fontsize=6)


def plot_fig2_main_cc_summary(
    bundle: CoreBundle, out_dir: Path, variant: str, rev_offset: int,
) -> Path:
    """Fig 2 (original): AUC top, burst-acc reversion bottom."""
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=figsize("half", 1.9),
        gridspec_kw={"height_ratios": [1, 2]},
    )
    draw_auc_panel(ax_a, bundle)
    draw_reversion_trace(
        ax_b, bundle, "acc_burst", "Special Accuracy", rev_offset, ylim=ACC_YLIM,
    )
    return save_chart(fig, with_variant(out_dir / "fig2_cc_main_summary.pdf", variant))


def plot_fig2_loss_stack(
    bundle: CoreBundle, out_dir: Path, variant: str, rev_offset: int, *, log: bool,
) -> Path:
    """Fig 2 vertical: AUC top, burst-loss bottom (linear or log)."""
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=figsize("half", 1.9),
        gridspec_kw={"height_ratios": [1, 2]},
    )
    draw_auc_panel(ax_a, bundle)
    draw_reversion_trace(
        ax_b, bundle, "loss_burst", "Special Eval Loss", rev_offset, log=log,
    )
    stem = "fig2_loss_log" if log else "fig2_loss_linear"
    return save_chart(fig, with_variant(out_dir / f"{stem}.pdf", variant))


def plot_fig2_horizontal_acc(
    bundle: CoreBundle, out_dir: Path, variant: str, rev_offset: int,
) -> Path:
    """Fig 2 horizontal (half-col): AUC left, burst-acc reversion right."""
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=figsize("half_side"), gridspec_kw={"width_ratios": [1, 1.4]},
    )
    draw_auc_panel(ax_a, bundle)
    draw_reversion_trace(
        ax_b, bundle, "acc_burst", "Special Accuracy", rev_offset, ylim=ACC_YLIM,
    )
    return save_chart(fig, with_variant(out_dir / "fig2_horizontal_acc.pdf", variant))


def plot_fig2_horizontal_loss(
    bundle: CoreBundle, out_dir: Path, variant: str, rev_offset: int,
) -> Path:
    """Fig 2 horizontal (half-col): AUC left, burst-loss right (log)."""
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=figsize("half_side"), gridspec_kw={"width_ratios": [1, 1.4]},
    )
    draw_auc_panel(ax_a, bundle)
    draw_reversion_trace(
        ax_b, bundle, "loss_burst", "Special Eval Loss", rev_offset, log=True,
    )
    return save_chart(fig, with_variant(out_dir / "fig2_horizontal_loss.pdf", variant))


def draw_masked_ci(  # noqa: PLR0913
    ax: Axes, steps: np.ndarray, mean: np.ndarray, ci: np.ndarray,
    mask: np.ndarray, color: str, label: str | None = None,
) -> None:
    """Plot mean line + CI fill for masked data."""
    ax.plot(steps[mask], mean[mask], color=color, label=label)
    ax.fill_between(
        steps[mask], mean[mask] - ci[mask], mean[mask] + ci[mask],
        color=color, alpha=0.12,
    )


def apply_panel3_scale(ax: Axes, yscale: str, values: list[float]) -> None:
    """Configure y-scale for fig4 panel-3 based on its data range."""
    if yscale == "symlog":
        linthresh = max(np.percentile(values, 5), 1e-6) if values else 1e-3
        ax.axhline(0.0, color=COLOR_ZERO_LINE, ls=":", lw=1.0)
        ax.set_yscale("symlog", linthresh=linthresh)
        return
    if yscale == "log":
        pos = [v for v in values if v > 0]
        floor = 10 ** np.floor(np.log10(min(pos))) if pos else 1e-6
        ax.set_yscale("log")
        ax.set_ylim(bottom=floor)
        return
    ax.axhline(0.0, color=COLOR_ZERO_LINE, ls=":", lw=1.0)


FIG4_PANEL3_VARIANTS: dict[str, tuple[str, str, str]] = {
    "signed_dot":         ("signed_dot",        "Signed Dot",        "linear"),
    "signed_dot_log":     ("signed_dot",        "Signed Dot",        "symlog"),
    "interference_power": ("interference_power", "Interference Power", "log"),
    "burst_norm":         ("burst_norm",        "Grad Norm (burst)", "log"),
    "other_norm":         ("other_norm",        "Grad Norm (other)", "log"),
}


def plot_fig4_grad_4panel(
    bundle: CoreBundle, out_dir: Path, variant: str, offset: int, panel3: str,
) -> Path | None:
    """Fig 4: stacked Loss / Cosine / <panel3> / Effective Rank panels with CI."""
    grads = bundle.gradients
    if not grads:
        return None
    schedules = [s for s in filter_schedules(bundle) if s in grads]
    if not schedules:
        return None

    field, ylabel, yscale = FIG4_PANEL3_VARIANTS[panel3]
    has_rank = any(grads[s].grad_rank is not None for s in schedules)
    n_panels = 4 if has_rank else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize("half", 3.2), sharex=True)

    max_be = 0
    p3_abs: list[float] = []
    for s in schedules:
        g = grads[s]
        ts = bundle.training[s]
        burst_end = ts.burst_steps
        max_be = max(max_be, burst_end)
        cutoff = burst_end + offset
        color = SCHED_COLORS[s]
        label = SCHED_DISPLAY.get(s, s)

        lb = ts.loss_burst
        lb_local = np.array(lb.steps, dtype=float) - ts.pre_steps
        lb_mask = (lb_local >= 0) & (lb_local <= cutoff)
        draw_masked_ci(
            axes[0], lb_local, np.array(lb.mean, dtype=float),
            np.array(lb.ci, dtype=float), lb_mask, color, label,
        )

        g_steps = np.array(g.steps, dtype=float)
        g_mask = g_steps <= cutoff
        draw_masked_ci(
            axes[1], g_steps, np.array(g.cosine.mean, dtype=float),
            np.array(g.cosine.ci, dtype=float), g_mask, color,
        )
        p3 = getattr(g, field)
        p3_mean = np.array(p3.mean, dtype=float)
        draw_masked_ci(
            axes[2], g_steps, p3_mean, np.array(p3.ci, dtype=float), g_mask, color,
        )
        p3_abs.extend(np.abs(p3_mean[g_mask]).tolist())
        if has_rank and g.grad_rank is not None:
            draw_masked_ci(
                axes[3], g_steps, np.array(g.grad_rank.mean, dtype=float),
                np.array(g.grad_rank.ci, dtype=float), g_mask, color,
            )

    axes[0].set_yscale("log")
    style_axes(axes[0], "", "Loss (burst)")
    axes[0].legend(loc="best", ncol=3, fontsize=5)
    axes[1].axhline(0.0, color=COLOR_ZERO_LINE, ls=":", lw=1.0)
    style_axes(axes[1], "", r"$\cos(\theta)$")

    apply_panel3_scale(axes[2], yscale, p3_abs)
    style_axes(axes[2], "" if has_rank else "Step", ylabel)
    if has_rank:
        style_axes(axes[3], "Step", "Eff. Rank (burst)")

    if offset > 0:
        for ax in axes:
            ax.axvline(max_be, color="black", ls="--", lw=0.9, alpha=0.55)

    return save_chart(
        fig, with_variant(out_dir / f"fig4_grad_4panel_{panel3}.pdf", variant),
    )


# ---------------------------------------------------------------------------
# Overlays + bars
# ---------------------------------------------------------------------------

def plot_schedule_bars(bundle: CoreBundle, out_dir: Path) -> Path:
    """Plot burst fraction over time for each filtered schedule as a 3x3 grid."""
    schedules = filter_schedules(bundle)
    ncols = 3
    nrows = math.ceil(len(schedules) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize("full", nrows / 1.618), sharex=True, sharey=True,
    )
    flat_axes = axes.flatten()

    max_len = max(len(bundle.schedule_bars[s].fractions) for s in schedules)
    for ax, schedule in zip(flat_axes, schedules, strict=False):
        fracs = np.array(bundle.schedule_bars[schedule].fractions, dtype=float)
        xs = np.arange(len(fracs))
        ax.fill_between(xs, fracs, color=SCHED_COLORS[schedule], alpha=0.78)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(0, max_len - 1)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_title(SCHED_DISPLAY.get(schedule, schedule), fontsize=7)

    for ax in flat_axes[len(schedules):]:
        ax.set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xlabel("Step")
    for ax in axes[:, 0]:
        ax.set_ylabel("Burst Fraction")

    return save_chart(fig, out_dir / "schedule_bars.pdf")


def plot_lr_curves(
    bundle: CoreBundle, out_dir: Path, variant: str, offset: int,
) -> Path:
    """LR schedules for filtered schedules, truncated per schedule."""
    fig, ax = plt.subplots(figsize=figsize("half"))
    max_be = 0
    for schedule in filter_schedules(bundle):
        curve = bundle.lr_curves[schedule]
        ts = bundle.training[schedule]
        burst_end = ts.pre_steps + ts.burst_steps
        max_be = max(max_be, burst_end)
        cutoff = burst_end + offset
        steps = np.array(curve.steps, dtype=float)
        lr = np.array(curve.lr, dtype=float)
        mask = steps <= cutoff
        ax.plot(
            steps[mask], lr[mask], color=SCHED_COLORS[schedule],
            label=SCHED_DISPLAY.get(schedule, schedule),
        )
    if offset > 0:
        ax.axvline(max_be, color="black", ls="--", lw=1.0, alpha=0.6)
    style_axes(ax, "Step", "Learning Rate")
    ax.legend(loc="upper right", ncol=2)
    return save_chart(fig, with_variant(out_dir / "lr_schedule.pdf", variant))


def plot_training_overlay(  # noqa: PLR0913
    bundle: CoreBundle, out_dir: Path, metric: str, ylabel: str,
    filename: str, ylim: tuple[float, float], variant: str, offset: int,
) -> Path:
    """Training metric overlay truncated per-schedule to burst_end + offset."""
    fig, ax = plt.subplots(figsize=figsize("half"))
    max_burst_end = 0
    for schedule in filter_schedules(bundle):
        ts = bundle.training[schedule]
        data = training_metric(ts, metric)
        steps = np.array(data.steps, dtype=float)
        mean = np.array(data.mean, dtype=float)
        ci = np.array(data.ci, dtype=float)
        burst_end = ts.pre_steps + ts.burst_steps
        max_burst_end = max(max_burst_end, burst_end)
        cutoff = burst_end + offset
        mask = steps <= cutoff
        ax.plot(
            steps[mask], mean[mask], color=SCHED_COLORS[schedule],
            label=SCHED_DISPLAY.get(schedule, schedule),
        )
        ax.fill_between(
            steps[mask], mean[mask] - ci[mask], mean[mask] + ci[mask],
            color=SCHED_COLORS[schedule], alpha=0.12,
        )
    ax.set_ylim(ylim)
    if offset > 0:
        ax.axvline(max_burst_end, color="black", ls="--", lw=1.0, alpha=0.6)
    style_axes(ax, "Step", ylabel)
    ax.legend(loc="best", ncol=2)
    return save_chart(fig, with_variant(out_dir / filename, variant))


def plot_training_overlay_with_log(  # noqa: PLR0913
    bundle: CoreBundle, out_dir: Path, metric: str, ylabel: str,
    filename: str, ylim: tuple[float, float], variant: str, offset: int,
) -> list[Path]:
    """Loss overlay with x-axis [0, burst_steps + offset], local to FT start."""
    fig, ax = plt.subplots(figsize=figsize("half"))
    max_burst_steps = 0
    for schedule in filter_schedules(bundle):
        ts = bundle.training[schedule]
        data = training_metric(ts, metric)
        local = np.array(data.steps, dtype=float) - ts.pre_steps
        mean = np.array(data.mean, dtype=float)
        ci = np.array(data.ci, dtype=float)
        max_burst_steps = max(max_burst_steps, ts.burst_steps)
        cutoff = ts.burst_steps + offset
        mask = (local >= 0) & (local <= cutoff)
        ax.plot(
            local[mask], mean[mask], color=SCHED_COLORS[schedule],
            label=SCHED_DISPLAY.get(schedule, schedule),
        )
        ax.fill_between(
            local[mask], mean[mask] - ci[mask], mean[mask] + ci[mask],
            color=SCHED_COLORS[schedule], alpha=0.12,
        )
    ax.set_ylim(ylim)
    if offset > 0:
        ax.axvline(max_burst_steps, color="black", ls="--", lw=1.0, alpha=0.6)
    style_axes(ax, "Step (from FT start)", ylabel)
    ax.legend(loc="best", ncol=2)
    return save_with_log_variant(fig, ax, with_variant(out_dir / filename, variant))


def plot_auc_bars(bundle: CoreBundle, out_dir: Path) -> Path:
    """Flat reversion AUC bars over filtered schedules."""
    schedules = filter_schedules(bundle)
    by_sched = bundle.summary.by_schedule
    means = [by_sched[s].reversion_auc.mean for s in schedules]
    cis = [by_sched[s].reversion_auc.ci for s in schedules]

    fig, ax = plt.subplots(figsize=figsize("flat"))
    xs = np.arange(len(schedules))
    colors = [SCHED_COLORS[s] for s in schedules]
    ax.bar(xs, means, yerr=cis, color=colors, edgecolor="black", lw=0.7, capsize=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [SCHED_DISPLAY.get(s, s) for s in schedules],
        rotation=20, ha="right", fontsize=6,
    )
    style_axes(ax, "", "AUC")
    return save_chart(fig, out_dir / "reversion_auc_bars.pdf")


def plot_extended_auc_bars_split(bundle: CoreBundle, out_dir: Path) -> list[Path]:
    """Four separate flat bar charts (one per AUC variant)."""
    schedules = filter_schedules(bundle)
    by_sched = bundle.summary.by_schedule
    xs = np.arange(len(schedules))
    colors = [SCHED_COLORS[s] for s in schedules]
    labels = [SCHED_DISPLAY.get(s, s) for s in schedules]
    specs = (
        ("reversion_auc", "Burst Acc AUC", "extended_auc_burst_acc.pdf"),
        ("reversion_auc_loss_burst", "Burst Loss AUC", "extended_auc_burst_loss.pdf"),
        ("reversion_auc_acc_other", "Other Acc AUC", "extended_auc_other_acc.pdf"),
        ("reversion_auc_loss_other", "Other Loss AUC", "extended_auc_other_loss.pdf"),
    )
    paths: list[Path] = []
    for attr, title, fname in specs:
        fig, ax = plt.subplots(figsize=figsize("flat"))
        means = [getattr(by_sched[s], attr).mean for s in schedules]
        cis = [getattr(by_sched[s], attr).ci for s in schedules]
        ax.bar(xs, means, yerr=cis, color=colors, edgecolor="black", lw=0.7, capsize=3)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=6)
        style_axes(ax, "", title)
        paths.append(save_chart(fig, out_dir / fname))
    return paths


def plot_summary_table(bundle: CoreBundle, out_dir: Path) -> Path:
    """Summary statistics table (all schedules, unfiltered)."""
    schedules = bundle.config.schedules
    by_sched = bundle.summary.by_schedule
    thresholds = bundle.config.thresholds

    headers = ["Schedule", "Peak", "AUC", "Other End"]
    headers.extend(reversion_life_label(t) for t in thresholds)

    rows = []
    for schedule in schedules:
        ss = by_sched[schedule]
        row = [
            SCHED_DISPLAY.get(schedule, schedule),
            fmt_ci(ss.peak_burst),
            fmt_ci(ss.reversion_auc, digits=0),
            fmt_ci(ss.other_end),
        ]
        row.extend(fmt_ci(ss.life[f"life_{int(t * 100)}"], digits=0) for t in thresholds)
        rows.append(row)

    fig, ax = plt.subplots(figsize=figsize("full"))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)  # noqa: FBT003
    table.set_fontsize(8)
    table.scale(1.0, 1.55)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor(COLOR_TABLE_HEADER)
        elif col == 0:
            schedule = schedules[row - 1]
            cell.set_facecolor(SCHED_COLORS[schedule] + "22")
            cell.set_text_props(fontweight="bold")
        cell.set_edgecolor(COLOR_TABLE_EDGE)
    return save_chart(fig, out_dir / "summary_table.pdf")


# ---------------------------------------------------------------------------
# Reversion zoom
# ---------------------------------------------------------------------------

def plot_reversion_zoom(
    bundle: CoreBundle, out_dir: Path, variant: str, rev_offset: int,
) -> Path:
    """Burst accuracy during the first rev_offset reversion steps."""
    fig, ax = plt.subplots(figsize=figsize("half"))
    for schedule in filter_schedules(bundle):
        ts = bundle.training[schedule]
        data = ts.acc_burst
        steps = np.array(data.steps, dtype=float)
        mean = np.array(data.mean, dtype=float)
        ci = np.array(data.ci, dtype=float)
        burst_end = ts.pre_steps + ts.burst_steps
        mask = (steps >= burst_end) & (steps <= burst_end + rev_offset)
        local = steps[mask] - burst_end
        ax.plot(
            local, mean[mask], color=SCHED_COLORS[schedule],
            label=SCHED_DISPLAY.get(schedule, schedule),
        )
        ax.fill_between(
            local, mean[mask] - ci[mask], mean[mask] + ci[mask],
            color=SCHED_COLORS[schedule], alpha=0.12,
        )
    ax.set_ylim(ACC_YLIM)
    style_axes(ax, "Reversion Step", "Special Accuracy")
    ax.legend(loc="upper right", ncol=2)
    return save_chart(
        fig, with_variant(out_dir / "reversion_zoom_forgetting_speed.pdf", variant),
    )


def plot_reversion_zoom_loss(
    bundle: CoreBundle, out_dir: Path, variant: str, rev_offset: int,
) -> list[Path]:
    """Burst eval loss during the first rev_offset reversion steps (linear + log)."""
    fig, ax = plt.subplots(figsize=figsize("half"))
    for schedule in filter_schedules(bundle):
        ts = bundle.training[schedule]
        data = ts.loss_burst
        steps = np.array(data.steps, dtype=float)
        mean = np.array(data.mean, dtype=float)
        ci = np.array(data.ci, dtype=float)
        burst_end = ts.pre_steps + ts.burst_steps
        mask = (steps >= burst_end) & (steps <= burst_end + rev_offset)
        local = steps[mask] - burst_end
        ax.plot(
            local, mean[mask], color=SCHED_COLORS[schedule],
            label=SCHED_DISPLAY.get(schedule, schedule),
        )
        ax.fill_between(
            local, mean[mask] - ci[mask], mean[mask] + ci[mask],
            color=SCHED_COLORS[schedule], alpha=0.12,
        )
    style_axes(ax, "Reversion Step", "Special Eval Loss")
    ax.legend(loc="best", ncol=2)
    return save_with_log_variant(
        fig, ax, with_variant(out_dir / "reversion_zoom_loss.pdf", variant),
    )


# ---------------------------------------------------------------------------
# Gradient charts
# ---------------------------------------------------------------------------

def plot_grad_cosine(
    bundle: CoreBundle, out_dir: Path, variant: str, offset: int,
) -> Path | None:
    """Gradient cosine similarity overlay, per-schedule cutoff."""
    grads = bundle.gradients
    if not grads:
        return None
    fig, ax = plt.subplots(figsize=figsize("half"))
    max_be = 0
    for s in filter_schedules(bundle):
        if s not in grads:
            continue
        g = grads[s]
        burst_end = g.burst_steps
        max_be = max(max_be, burst_end)
        cutoff = burst_end + offset
        steps = np.array(g.steps, dtype=float)
        mean = np.array(g.cosine.mean, dtype=float)
        ci = np.array(g.cosine.ci, dtype=float)
        mask = steps <= cutoff
        ax.plot(
            steps[mask], mean[mask], color=SCHED_COLORS[s],
            label=SCHED_DISPLAY.get(s, s),
        )
        ax.fill_between(
            steps[mask], mean[mask] - ci[mask], mean[mask] + ci[mask],
            color=SCHED_COLORS[s], alpha=0.12,
        )
    ax.axhline(0.0, color=COLOR_ZERO_LINE, ls=":", lw=1.0)
    if offset > 0:
        ax.axvline(max_be, color="black", ls="--", lw=1.0, alpha=0.6)
    style_axes(ax, "Step", "Cosine")
    ax.legend(loc="best", ncol=2)
    return save_chart(fig, with_variant(out_dir / "grad_cosine_burst_vs_other.pdf", variant))


def plot_grad_norms_split(
    bundle: CoreBundle, out_dir: Path, variant: str, offset: int,
) -> list[Path] | None:
    """Burst and Other grad L2 norms truncated per-schedule (linear + log)."""
    grads = bundle.gradients
    if not grads:
        return None
    paths: list[Path] = []
    for label, attr in (("burst", "burst_norm"), ("other", "other_norm")):
        fig, ax = plt.subplots(figsize=figsize("half"))
        max_be = 0
        for s in filter_schedules(bundle):
            if s not in grads:
                continue
            g = grads[s]
            burst_end = g.burst_steps
            max_be = max(max_be, burst_end)
            cutoff = burst_end + offset
            steps = np.array(g.steps, dtype=float)
            series = getattr(g, attr)
            mean = np.array(series.mean, dtype=float)
            ci = np.array(series.ci, dtype=float)
            mask = steps <= cutoff
            ax.plot(
                steps[mask], mean[mask], color=SCHED_COLORS[s],
                label=SCHED_DISPLAY.get(s, s),
            )
            ax.fill_between(
                steps[mask], mean[mask] - ci[mask], mean[mask] + ci[mask],
                color=SCHED_COLORS[s], alpha=0.12,
            )
        if offset > 0:
            ax.axvline(max_be, color="black", ls="--", lw=1.0, alpha=0.6)
        style_axes(ax, "Step", f"L2 Norm ({label})")
        ax.legend(loc="best", ncol=2)
        paths.extend(save_with_log_variant(
            fig, ax, with_variant(out_dir / f"grad_norm_l2_{label}.pdf", variant),
        ))
    return paths


def plot_signed_dot(
    bundle: CoreBundle, out_dir: Path, variant: str, offset: int,
) -> list[Path] | None:
    """Signed dot product overlay, per-schedule cutoff (linear + symlog)."""
    grads = bundle.gradients
    if not grads:
        return None
    fig, ax = plt.subplots(figsize=figsize("half"))
    max_be = 0
    all_abs: list[float] = []
    for s in filter_schedules(bundle):
        if s not in grads:
            continue
        g = grads[s]
        burst_end = g.burst_steps
        max_be = max(max_be, burst_end)
        cutoff = burst_end + offset
        steps = np.array(g.steps, dtype=float)
        mean = np.array(g.signed_dot.mean, dtype=float)
        ci = np.array(g.signed_dot.ci, dtype=float)
        mask = steps <= cutoff
        ax.plot(
            steps[mask], mean[mask], color=SCHED_COLORS[s],
            label=SCHED_DISPLAY.get(s, s),
        )
        ax.fill_between(
            steps[mask], mean[mask] - ci[mask], mean[mask] + ci[mask],
            color=SCHED_COLORS[s], alpha=0.12,
        )
        all_abs.extend(np.abs(mean[mask]).tolist())
    ax.axhline(0.0, color=COLOR_ZERO_LINE, ls=":", lw=1.0)
    if offset > 0:
        ax.axvline(max_be, color="black", ls="--", lw=1.0, alpha=0.6)
    style_axes(ax, "Step", "Signed Dot")
    ax.legend(loc="best", ncol=2)

    base = with_variant(out_dir / "grad_signed_dot.pdf", variant)
    paths = [save_chart(fig, base)]

    fig2, ax2 = plt.subplots(figsize=figsize("half"))
    for s in filter_schedules(bundle):
        if s not in grads:
            continue
        g = grads[s]
        cutoff = g.burst_steps + offset
        steps = np.array(g.steps, dtype=float)
        mean = np.array(g.signed_dot.mean, dtype=float)
        mask = steps <= cutoff
        ax2.plot(
            steps[mask], mean[mask], color=SCHED_COLORS[s],
            label=SCHED_DISPLAY.get(s, s),
        )
    ax2.axhline(0.0, color=COLOR_ZERO_LINE, ls=":", lw=1.0)
    if offset > 0:
        ax2.axvline(max_be, color="black", ls="--", lw=1.0, alpha=0.6)
    linthresh = max(np.percentile(np.abs(all_abs), 5), 1e-6) if all_abs else 1e-3
    ax2.set_yscale("symlog", linthresh=linthresh)
    style_axes(ax2, "Step", "Signed Dot")
    ax2.legend(loc="best", ncol=2)
    paths.append(save_chart(fig2, base.with_stem(base.stem + "_log")))
    return paths


def plot_interference_power(
    bundle: CoreBundle, out_dir: Path, variant: str, offset: int,
) -> list[Path] | None:
    """Interference power overlay truncated per-schedule (linear + log)."""
    grads = bundle.gradients
    if not grads:
        return None
    fig, ax = plt.subplots(figsize=figsize("half"))
    max_be = 0
    for s in filter_schedules(bundle):
        if s not in grads:
            continue
        g = grads[s]
        burst_end = g.burst_steps
        max_be = max(max_be, burst_end)
        cutoff = burst_end + offset
        steps = np.array(g.steps, dtype=float)
        mean = np.array(g.interference_power.mean, dtype=float)
        ci = np.array(g.interference_power.ci, dtype=float)
        mask = steps <= cutoff
        ax.plot(
            steps[mask], mean[mask], color=SCHED_COLORS[s],
            label=SCHED_DISPLAY.get(s, s),
        )
        ax.fill_between(
            steps[mask], mean[mask] - ci[mask], mean[mask] + ci[mask],
            color=SCHED_COLORS[s], alpha=0.12,
        )
    if offset > 0:
        ax.axvline(max_be, color="black", ls="--", lw=1.0, alpha=0.6)
    style_axes(ax, "Step", "Interference Power")
    ax.legend(loc="best", ncol=2)
    return save_with_log_variant(
        fig, ax, with_variant(out_dir / "grad_interference_power.pdf", variant),
    )


def plot_grad_rank(
    bundle: CoreBundle, out_dir: Path, variant: str, offset: int,
) -> Path | None:
    """Effective gradient rank overlay, per-schedule cutoff."""
    grads = bundle.gradients
    if not grads:
        return None
    if not any(g.grad_rank is not None for g in grads.values()):
        return None
    fig, ax = plt.subplots(figsize=figsize("half"))
    max_be = 0
    for s in filter_schedules(bundle):
        g = grads.get(s)
        if g is None or g.grad_rank is None:
            continue
        burst_end = g.burst_steps
        max_be = max(max_be, burst_end)
        cutoff = burst_end + offset
        steps = np.array(g.steps, dtype=float)
        mean = np.array(g.grad_rank.mean, dtype=float)
        ci = np.array(g.grad_rank.ci, dtype=float)
        mask = steps <= cutoff
        ax.plot(
            steps[mask], mean[mask], color=SCHED_COLORS[s],
            label=SCHED_DISPLAY.get(s, s),
        )
        ax.fill_between(
            steps[mask], mean[mask] - ci[mask], mean[mask] + ci[mask],
            color=SCHED_COLORS[s], alpha=0.12,
        )
    if offset > 0:
        ax.axvline(max_be, color="black", ls="--", lw=1.0, alpha=0.6)
    style_axes(ax, "Step", "Effective Rank")
    ax.legend(loc="best", ncol=2)
    return save_chart(fig, with_variant(out_dir / "grad_rank_effective.pdf", variant))


# ---------------------------------------------------------------------------
# Representation drift
# ---------------------------------------------------------------------------

def plot_representation_drift_split(bundle: CoreBundle, out_dir: Path) -> list[Path] | None:
    """Centroid projection and other-shift as two separate flat charts."""
    by_schedule = bundle.representation.by_schedule
    if not by_schedule:
        return None
    schedules = [s for s in filter_schedules(bundle) if s in by_schedule]
    if not schedules:
        return None
    xs = np.arange(len(schedules))
    labels = [sched_pct_label(s) for s in schedules]

    specs = (
        ("late_centroid_projection", COLOR_PROJECTION, "Projection",
         "repr_centroid_projection.pdf"),
        ("late_other_shift_norm", COLOR_SHIFT, "Normalized Shift",
         "repr_other_shift.pdf"),
    )
    paths: list[Path] = []
    for attr, color, ylabel, fname in specs:
        fig, ax = plt.subplots(figsize=figsize("flat"))
        mean = np.array([getattr(by_schedule[s], attr).mean for s in schedules], dtype=float)
        ci = np.array([getattr(by_schedule[s], attr).ci for s in schedules], dtype=float)
        ax.plot(xs, mean, color=color, marker="o", ms=3)
        ax.fill_between(xs, mean - ci, mean + ci, color=color, alpha=0.12)
        ax.set_ylim(tight_ylim(mean - ci, mean + ci))
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        style_axes(ax, "Concentration %", ylabel)
        paths.append(save_chart(fig, out_dir / fname))
    return paths


def plot_burst_representation_drift_split(
    bundle: CoreBundle, out_dir: Path,
) -> list[Path] | None:
    """Burst self-projection and burst normalized shift as two flat charts."""
    by_schedule = bundle.representation.by_schedule
    if not by_schedule:
        return None
    schedules = [s for s in filter_schedules(bundle) if s in by_schedule]
    if not schedules:
        return None
    xs = np.arange(len(schedules))
    labels = [sched_pct_label(s) for s in schedules]

    specs = (
        ("late_burst_self_projection", "Burst Drift",
         "repr_burst_self_projection.pdf"),
        ("late_burst_shift_norm", "Burst Shift",
         "repr_burst_shift.pdf"),
    )
    paths: list[Path] = []
    for attr, ylabel, fname in specs:
        fig, ax = plt.subplots(figsize=figsize("flat"))
        mean = np.array([getattr(by_schedule[s], attr).mean for s in schedules], dtype=float)
        ci = np.array([getattr(by_schedule[s], attr).ci for s in schedules], dtype=float)
        ax.plot(xs, mean, color=COLOR_SPECIAL, marker="o", ms=3)
        ax.fill_between(xs, mean - ci, mean + ci, color=COLOR_SPECIAL, alpha=0.12)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        style_axes(ax, "Concentration %", ylabel)
        paths.append(save_chart(fig, out_dir / fname))
    return paths


def plot_centroid_norms_split(bundle: CoreBundle, out_dir: Path) -> list[Path] | None:
    """Burst and other centroid-norm ratios as two flat charts."""
    by_schedule = bundle.representation.by_schedule
    if not by_schedule:
        return None
    schedules = [s for s in filter_schedules(bundle) if s in by_schedule]
    if not schedules:
        return None
    xs = np.arange(len(schedules))
    labels = [sched_pct_label(s) for s in schedules]

    specs = (
        ("late_burst_post_norm", COLOR_SPECIAL, "Burst Ratio",
         "centroid_norm_burst.pdf"),
        ("late_other_post_norm", COLOR_OTHER, "Other Ratio",
         "centroid_norm_other.pdf"),
    )
    paths: list[Path] = []
    for post_attr, color, ylabel, fname in specs:
        ratio_seeds: list[list[float]] = []
        for s in schedules:
            per_seed = by_schedule[s].per_seed
            ratio_seeds.append(
                [getattr(sd, post_attr) / (sd.late_burst_pre_norm + 1e-12) for sd in per_seed],
            )
        mean = np.array([np.mean(r) for r in ratio_seeds], dtype=float)
        ci = np.array([mean_ci(np.array(r))[1] for r in ratio_seeds], dtype=float)
        fig, ax = plt.subplots(figsize=figsize("flat"))
        ax.plot(xs, mean, color=color, marker="o", ms=3)
        ax.fill_between(xs, mean - ci, mean + ci, color=color, alpha=0.12)
        ax.axhline(1.0, color=COLOR_ZERO_LINE, ls=":", lw=1.0)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        style_axes(ax, "Concentration %", ylabel)
        paths.append(save_chart(fig, out_dir / fname))
    return paths


# ---------------------------------------------------------------------------
# Per-schedule + per-layer charts
# ---------------------------------------------------------------------------

def plot_per_schedule(
    bundle: CoreBundle, out_dir: Path, variant: str, offset: int,
) -> list[Path]:
    """Per-schedule burst vs other accuracy, per-schedule cutoff."""
    paths: list[Path] = []
    for schedule in filter_schedules(bundle):
        ts = bundle.training[schedule]
        steps = np.array(ts.acc_burst.steps, dtype=float)
        burst_end = ts.pre_steps + ts.burst_steps
        cutoff = burst_end + offset
        mask = steps <= cutoff

        fig, ax = plt.subplots(figsize=figsize("half"))
        for data, color, label in (
            (ts.acc_other, COLOR_OTHER, "Other"),
            (ts.acc_burst, COLOR_SPECIAL, "Special"),
        ):
            mean = np.array(data.mean, dtype=float)
            ci = np.array(data.ci, dtype=float)
            ax.plot(steps[mask], mean[mask], color=color, label=label)
            ax.fill_between(
                steps[mask], mean[mask] - ci[mask], mean[mask] + ci[mask],
                color=color, alpha=0.14,
            )

        if ts.pre_steps > 0:
            ax.axvline(ts.pre_steps, color="black", ls="--", lw=1.2, alpha=0.65)
        if offset > 0:
            ax.axvline(burst_end, color="black", ls="--", lw=1.2, alpha=0.65)
        ax.set_ylim(ACC_YLIM)
        style_axes(ax, "Step", "Accuracy")
        ax.legend(loc="best")
        paths.append(save_chart(
            fig,
            with_variant(
                out_dir / "per_sched" / "accuracy" / f"{schedule}.pdf", variant,
            ),
        ))
    return paths


def plot_per_schedule_loss(
    bundle: CoreBundle, out_dir: Path, ylim: tuple[float, float],
    variant: str, offset: int,
) -> list[Path]:
    """Per-schedule burst vs other eval loss (linear + log), per-schedule cutoff."""
    paths: list[Path] = []
    for schedule in filter_schedules(bundle):
        ts = bundle.training[schedule]
        steps = np.array(ts.loss_burst.steps, dtype=float)
        burst_end = ts.pre_steps + ts.burst_steps
        cutoff = burst_end + offset
        mask = steps <= cutoff

        fig, ax = plt.subplots(figsize=figsize("half"))
        for data, color, label in (
            (ts.loss_other, COLOR_OTHER, "Other"),
            (ts.loss_burst, COLOR_SPECIAL, "Special"),
        ):
            mean = np.array(data.mean, dtype=float)
            ci = np.array(data.ci, dtype=float)
            ax.plot(steps[mask], mean[mask], color=color, label=label)
            ax.fill_between(
                steps[mask], mean[mask] - ci[mask], mean[mask] + ci[mask],
                color=color, alpha=0.14,
            )
        if ts.pre_steps > 0:
            ax.axvline(ts.pre_steps, color="black", ls="--", lw=1.2, alpha=0.65)
        if offset > 0:
            ax.axvline(burst_end, color="black", ls="--", lw=1.2, alpha=0.65)
        ax.set_ylim(ylim)
        style_axes(ax, "Step", "Eval Loss")
        ax.legend(loc="best")
        paths.extend(save_with_log_variant(
            fig, ax,
            with_variant(
                out_dir / "per_sched" / "loss" / f"{schedule}.pdf", variant,
            ),
        ))
    return paths


def plot_per_layer_heatmaps(bundle: CoreBundle, out_dir: Path) -> list[Path]:
    """Half-col layer heatmaps (cosine, norms, norm*cos), one folder per metric."""
    paths: list[Path] = []
    for schedule in filter_schedules(bundle):
        data = bundle.per_layer_gradients.get(schedule)
        if data is None:
            continue
        paths.append(layer_heatmap_half(
            data.layer_names, data.steps, data.cosine,
            out_dir / "per_layer" / "cossim" / f"{schedule}_heatmap.pdf",
            "Cosine Similarity", cmap=LAYER_CMAP, center_zero=True,
        ))
        for prefix, md in (("burst", data.burst_norm), ("other", data.other_norm)):
            paths.append(layer_heatmap_half(
                data.layer_names, data.steps, md,
                out_dir / "per_layer" / f"grad_norm_{prefix}" / f"{schedule}_heatmap.pdf",
                f"Grad Norm ({prefix})", cmap=DRIFT_CMAP,
            ))
        paths.append(layer_heatmap_half(
            data.layer_names, data.steps, data.norm_x_cosine,
            out_dir / "per_layer" / "norm_x_cossim" / f"{schedule}_heatmap.pdf",
            "Norm x Cosine", cmap=LAYER_CMAP, center_zero=True,
        ))
    return paths


def plot_weight_drift_heatmaps(bundle: CoreBundle, out_dir: Path) -> list[Path]:
    """Per-schedule weight drift heatmaps (step axis; no variants)."""
    paths: list[Path] = []
    for schedule in filter_schedules(bundle):
        data = bundle.weight_drift.get(schedule)
        if data is None:
            continue
        paths.append(layer_heatmap_half(
            data.layer_names, list(data.steps), data.cumulative,
            out_dir / "per_layer" / "weight_drift" / f"{schedule}_heatmap.pdf",
            "Weight Drift (Frobenius)", cmap=DRIFT_CMAP,
        ))
    return paths


def plot_per_layer_lines(
    bundle: CoreBundle, out_dir: Path, variant: str, offset: int,
) -> list[Path]:
    """Per-layer line charts (cosine linear; grad norms linear + log), per-schedule cutoff."""
    paths: list[Path] = []
    for schedule in filter_schedules(bundle):
        data = bundle.per_layer_gradients.get(schedule)
        if data is None:
            continue
        cutoff = data.burst_steps + offset
        paths.append(plot_layer_lines(
            data.layer_names, data.steps, data.cosine,
            with_variant(
                out_dir / "per_layer" / "cossim" / f"{schedule}_lines.pdf", variant,
            ),
            "Cosine Similarity", cutoff=cutoff,
        ))
        for prefix, md in (("burst", data.burst_norm), ("other", data.other_norm)):
            paths.extend(layer_lines_with_log(
                data.layer_names, data.steps, md,
                with_variant(
                    out_dir / "per_layer" / f"grad_norm_{prefix}" / f"{schedule}_lines.pdf",
                    variant,
                ),
                f"Grad Norm ({prefix})", cutoff=cutoff,
            ))
        paths.append(plot_layer_lines(
            data.layer_names, data.steps, data.norm_x_cosine,
            with_variant(
                out_dir / "per_layer" / "norm_x_cossim" / f"{schedule}_lines.pdf",
                variant,
            ),
            "Norm x Cosine", cutoff=cutoff,
        ))
    return paths


def plot_weight_drift(
    bundle: CoreBundle, out_dir: Path, variant: str, offset: int,
) -> list[Path]:
    """Per-schedule weight drift line charts (linear + log), per-schedule cutoff."""
    paths: list[Path] = []
    for schedule in filter_schedules(bundle):
        data = bundle.weight_drift.get(schedule)
        if data is None:
            continue
        ts = bundle.training[schedule]
        cutoff = ts.burst_steps + offset
        paths.extend(layer_lines_with_log(
            data.layer_names, list(data.steps), data.cumulative,
            with_variant(
                out_dir / "per_layer" / "weight_drift" / f"{schedule}_lines.pdf",
                variant,
            ),
            "Weight Drift (Frobenius)", cutoff=float(cutoff),
        ))
    return paths


# ---------------------------------------------------------------------------
# Shared layer helpers
# ---------------------------------------------------------------------------

def layer_heatmap_half(  # noqa: PLR0913
    layer_names: list[str],
    steps: list[float],
    metric_dict: dict[str, SeriesMeanCI],
    out_path: Path,
    ylabel: str,
    *,
    cmap: str,
    center_zero: bool = False,
) -> Path:
    """Half-col layer x step heatmap with readable layer-name ticks."""
    n_layers = len(layer_names)
    n_steps = len(steps)
    grid = np.full((n_layers, n_steps), np.nan)
    for li, ln in enumerate(layer_names):
        vals = metric_dict[ln].mean
        grid[li, : len(vals)] = vals
    masked = np.ma.masked_invalid(grid)
    vmin, vmax = None, None
    if center_zero:
        abs_max = max(abs(np.nanmin(grid)), abs(np.nanmax(grid)), 1e-12)
        vmin, vmax = -abs_max, abs_max

    fig, ax = plt.subplots(figsize=figsize("half"))
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(layer_names, fontsize=7)
    n_xticks = min(10, n_steps)
    tick_idx = np.linspace(0, n_steps - 1, n_xticks, dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{steps[i]:.0f}" for i in tick_idx], fontsize=7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Layer")
    fig.colorbar(im, ax=ax, label=ylabel, shrink=0.8)
    return save_chart(fig, out_path)


def plot_layer_lines(  # noqa: PLR0913
    layer_names: list[str],
    steps: list[float],
    metric_dict: dict[str, SeriesMeanCI],
    out_path: Path,
    ylabel: str,
    *,
    cutoff: float,
) -> Path:
    """Per-layer line chart (linear only), truncated at cutoff."""
    fig, ax = plt.subplots(figsize=figsize("half"))
    cmap_obj = plt.get_cmap(LAYER_LINE_CMAP)
    n_layers = len(layer_names)
    steps_arr = np.array(steps, dtype=float)
    for li, ln in enumerate(layer_names):
        mean = np.array(metric_dict[ln].mean, dtype=float)
        s = steps_arr[: len(mean)]
        mask = s <= cutoff
        ax.plot(
            s[mask], mean[mask], lw=1.4, label=ln,
            color=cmap_obj(li / max(n_layers, 1)),
        )
    style_axes(ax, "Step", ylabel)
    ax.legend(loc="best", ncol=3, fontsize=5)
    return save_chart(fig, out_path)


def layer_lines_with_log(  # noqa: PLR0913
    layer_names: list[str],
    steps: list[float],
    metric_dict: dict[str, SeriesMeanCI],
    out_path: Path,
    ylabel: str,
    *,
    cutoff: float,
) -> list[Path]:
    """Per-layer line chart with linear + log variant, truncated at cutoff."""
    fig, ax = plt.subplots(figsize=figsize("half"))
    cmap_obj = plt.get_cmap(LAYER_LINE_CMAP)
    n_layers = len(layer_names)
    steps_arr = np.array(steps, dtype=float)
    for li, ln in enumerate(layer_names):
        mean = np.array(metric_dict[ln].mean, dtype=float)
        s = steps_arr[: len(mean)]
        mask = s <= cutoff
        ax.plot(
            s[mask], mean[mask], lw=1.4, label=ln,
            color=cmap_obj(li / max(n_layers, 1)),
        )
    style_axes(ax, "Step", ylabel)
    ax.legend(loc="best", ncol=3, fontsize=5)
    return save_with_log_variant(fig, ax, out_path)


# ---------------------------------------------------------------------------
# Probe charts
# ---------------------------------------------------------------------------

def probe_layer_labels(n_layers: int) -> tuple[np.ndarray, list[str]]:
    """Return x-positions and tick labels for probe layer plots."""
    K = n_layers + 1
    return np.arange(K), ["emb", *[f"L{i}" for i in range(n_layers)]]


def probe_n_layers(bundle: CoreBundle) -> int | None:
    """Infer n_layers from the first probe data entry, or None if unavailable."""
    ntp = bundle.next_token_probes
    if not ntp:
        return None
    first_sched = next(iter(ntp.values()))
    first_step = next(iter(first_sched.values()))
    first_method = next(iter(first_step.values()))
    return len(first_method.Other.mean) - 1


def plot_probe_charts(bundle: CoreBundle, out_dir: Path) -> list[Path]:
    """Render all next-token probe charts (empty list if no probe data)."""
    n_layers = probe_n_layers(bundle)
    if n_layers is None:
        return []
    paths: list[Path] = []
    for fn in (plot_probe_accuracy, plot_probe_diffs, plot_probe_diff_in_diffs):
        result = fn(bundle, out_dir, n_layers)
        if result is not None:
            paths.append(result)
    return paths


def plot_probe_accuracy(bundle: CoreBundle, out_dir: Path, n_layers: int) -> Path | None:
    """Per-schedule, per-regime probe accuracy by layer."""
    ntp = bundle.next_token_probes
    if not ntp:
        return None

    schedules = [s for s in bundle.config.schedules if s in ntp]
    if not schedules:
        return None

    x, layer_labels = probe_layer_labels(n_layers)

    n_scheds = len(schedules)
    fig, axes = plt.subplots(n_scheds, 2, figsize=figsize("full", n_scheds), squeeze=False)

    for si, sched in enumerate(schedules):
        step_data = next(iter(ntp[sched].values()))
        for ri, regime in enumerate(["Other", "Burst"]):
            ax = axes[si, ri]
            for method_name in ("logit_lens", "learned_probe"):
                md = step_data.get(method_name)
                if not md:
                    continue
                series: SeriesMeanCI = getattr(md, regime)
                mean = np.array(series.mean, dtype=float)
                ci = np.array(series.ci, dtype=float)
                color = SCHED_COLORS.get(sched, "gray")
                ls = "-" if method_name == "logit_lens" else "--"
                ax.plot(x, mean, ls, color=color, marker="o", ms=2, label=method_name)
                ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.12)
            ax.set_xticks(x)
            ax.set_xticklabels(layer_labels)
            ax.set_ylim(-0.05, 1.05)
            display = SCHED_DISPLAY.get(sched, sched)
            ax.set_title(f"{display} — {regime}")
            if si == 0 and ri == 0:
                ax.legend(loc="upper left")
            ax.grid(visible=True)

    style_axes(axes[-1, 0], "Layer", "Accuracy")
    style_axes(axes[-1, 1], "Layer", "Accuracy")
    return save_chart(fig, out_dir / "probe_accuracy_by_layer.pdf")


def plot_probe_diffs(bundle: CoreBundle, out_dir: Path, n_layers: int) -> Path | None:
    """Other-minus-Burst probe accuracy diff per schedule."""
    ntp = bundle.next_token_probes
    if not ntp:
        return None

    schedules = [s for s in bundle.config.schedules if s in ntp]
    if not schedules:
        return None

    x, layer_labels = probe_layer_labels(n_layers)

    fig, axes = plt.subplots(1, 2, figsize=figsize("full"), sharey=True)
    for mi, method_name in enumerate(("logit_lens", "learned_probe")):
        ax = axes[mi]
        for sched in schedules:
            step_data = next(iter(ntp[sched].values()))
            md = step_data.get(method_name)
            if not md:
                continue
            mean = np.array(md.diff.mean, dtype=float)
            ci = np.array(md.diff.ci, dtype=float)
            color = SCHED_COLORS.get(sched, "gray")
            ax.plot(
                x, mean, "-o", color=color, ms=2,
                label=SCHED_DISPLAY.get(sched, sched),
            )
            ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.12)
        ax.axhline(0, color=COLOR_ZERO_LINE, ls=":", lw=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(layer_labels)
        ax.set_title(method_name)
        ax.legend(loc="best", ncol=2)
        ax.grid(visible=True)

    style_axes(axes[0], "Layer", r"$\Delta$ Accuracy (Other $-$ Burst)")
    style_axes(axes[1], "Layer", "")
    return save_chart(fig, out_dir / "probe_diff_other_minus_burst.pdf")


def plot_probe_diff_in_diffs(bundle: CoreBundle, out_dir: Path, n_layers: int) -> Path | None:
    """Pairwise schedule diff-in-diffs for probe accuracy."""
    ntp = bundle.next_token_probes
    if not ntp:
        return None

    schedules = [s for s in bundle.config.schedules if s in ntp]
    if len(schedules) < 2:  # noqa: PLR2004
        return None

    x, layer_labels = probe_layer_labels(n_layers)

    step_diffs: dict[str, np.ndarray] = {}
    for sched in schedules:
        step_data = next(iter(ntp[sched].values()))
        md = step_data.get("logit_lens")
        if md:
            step_diffs[sched] = np.array(md.diff.mean, dtype=float)

    from itertools import combinations  # noqa: PLC0415

    pairs = list(combinations([s for s in schedules if s in step_diffs], 2))
    if not pairs:
        return None

    cmap_vals = plt.cm.tab10(np.linspace(0, 1, max(len(pairs), 1)))
    fig, ax = plt.subplots(figsize=figsize("half"))
    for pi, (s1, s2) in enumerate(pairs):
        did = step_diffs[s1] - step_diffs[s2]
        label = f"{SCHED_DISPLAY.get(s1, s1)} vs {SCHED_DISPLAY.get(s2, s2)}"
        ax.plot(x, did, "-o", color=cmap_vals[pi], ms=2, label=label)

    ax.axhline(0, color=COLOR_ZERO_LINE, ls=":", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.legend(loc="best", ncol=1, fontsize=5)
    style_axes(ax, "Layer", "Diff-in-Diff (logit lens)")
    return save_chart(fig, out_dir / "probe_diff_in_diff.pdf")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def fmt_ci(metric: MeanCI | LifeEntry, digits: int = 3) -> str:
    """Format a mean +- CI string from a metric dataclass."""
    if digits == 0:
        return f"{metric.mean:.0f} ± {metric.ci:.0f}"
    return f"{metric.mean:.{digits}f} ± {metric.ci:.{digits}f}"


def save_chart(fig: Figure, path: Path) -> Path:
    """Save a figure as PDF and return its path."""
    path = path.with_suffix(".pdf")
    save_figure(fig, path)
    return path


def sched_pct_label(schedule: str) -> str:
    """Extract the percentage suffix from a schedule name."""
    return schedule.rsplit("_", maxsplit=1)[-1]
