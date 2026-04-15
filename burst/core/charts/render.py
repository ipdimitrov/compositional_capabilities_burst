"""Render publication-quality charts from bundled experiment data."""

from __future__ import annotations

import shutil
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
    SCHED_COLORS,
    SCHED_DISPLAY,
    reversion_life_label,
)
from burst.core.charts.style import apply_paper_style, figsize, save_figure, style_axes
from burst.core.train_utils import mean_ci

ACC_YLIM: tuple[float, float] = (-0.05, 1.05)


def training_metric(ts: TrainingSchedule, name: str) -> TrainingSeries:
    """Look up a training metric by name on a TrainingSchedule."""
    return getattr(ts, name)


def loss_ylim(bundle: CoreBundle, metrics: list[str]) -> tuple[float, float]:
    """Compute a shared y-axis range across metrics and schedules."""
    lo, hi = float("inf"), float("-inf")
    for schedule in bundle.config.schedules:
        ts = bundle.training[schedule]
        for metric in metrics:
            data = training_metric(ts, metric)
            mean = np.array(data.mean, dtype=float)
            ci = np.array(data.ci, dtype=float)
            lo = min(lo, float(np.nanmin(mean - ci)))
            hi = max(hi, float(np.nanmax(mean + ci)))
    pad = (hi - lo) * 0.05
    return (lo - pad, hi + pad)


def save_with_log_variant(fig: Figure, ax: Axes, path: Path) -> list[Path]:
    """Save a chart in linear scale, then save a log-scale variant with '_log' suffix."""
    linear_path = save_chart(fig, path)
    orig_ylim = ax.get_ylim()
    ax.set_yscale("log")
    ax.set_ylim(bottom=max(orig_ylim[0], 1e-6), top=orig_ylim[1])
    log_path = save_chart(fig, path.with_stem(path.stem + "_log"))
    ax.set_yscale("linear")
    ax.set_ylim(orig_ylim)
    return [linear_path, log_path]


def render_core_charts(bundle: CoreBundle, out_dir: str | Path) -> list[Path]:
    """Render all core analysis charts to out_dir."""
    apply_paper_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_loss_yl = loss_ylim(bundle, ["loss_burst", "loss_other"])
    train_loss_yl = loss_ylim(bundle, ["loss"])

    burst_fname = f"overlay_ACC_BURST_{CLASS_BURST}_class_accuracy.pdf"
    other_fname = f"overlay_ACC_OTHER_{CLASS_OTHER}_class_accuracy.pdf"
    paths = [
        plot_schedule_bars(bundle, out_dir),
        plot_lr_curves(bundle, out_dir),
        plot_overlay(
            bundle, out_dir, "acc_burst", f"{CLASS_BURST} Accuracy", burst_fname, ACC_YLIM
        ),
        plot_overlay(
            bundle, out_dir, "acc_other", f"{CLASS_OTHER} Accuracy", other_fname, ACC_YLIM
        ),
        plot_overlay(
            bundle, out_dir, "loss", "Loss", "overlay_LOSS_training_loss.pdf", train_loss_yl
        ),
        plot_overlay(
            bundle,
            out_dir,
            "loss_burst",
            f"{CLASS_BURST} Eval Loss",
            "overlay_LOSS_BURST_eval_loss.pdf",
            eval_loss_yl,
        ),
        plot_overlay(
            bundle,
            out_dir,
            "loss_other",
            f"{CLASS_OTHER} Eval Loss",
            "overlay_LOSS_OTHER_eval_loss.pdf",
            eval_loss_yl,
        ),
        plot_auc_bars(bundle, out_dir),
        plot_extended_auc_bars(bundle, out_dir),
        plot_summary_table(bundle, out_dir),
        plot_reversion_zoom(bundle, out_dir),
        plot_reversion_zoom_loss(bundle, out_dir),
        plot_grad_cosine(bundle, out_dir),
        plot_grad_cosine_per_schedule(bundle, out_dir),
        plot_grad_norms(bundle, out_dir),
        plot_grad_norm_x_cosine(bundle, out_dir),
        plot_representation_drift(bundle, out_dir),
        plot_burst_representation_drift(bundle, out_dir),
        plot_centroid_norms(bundle, out_dir),
        plot_grad_rank(bundle, out_dir),
    ]
    paths.extend(plot_per_schedule(bundle, out_dir))
    paths.extend(plot_per_schedule_loss(bundle, out_dir, eval_loss_yl))
    paths.extend(plot_per_layer_cossim(bundle, out_dir))
    paths.extend(plot_per_layer_grad_norm(bundle, out_dir))
    paths.extend(plot_per_layer_norm_x_cossim(bundle, out_dir))
    paths.extend(plot_weight_drift(bundle, out_dir))
    paths.extend(plot_probe_charts(bundle, out_dir))
    return [path for path in paths if path is not None]


def plot_schedule_bars(bundle: CoreBundle, out_dir: Path) -> Path:
    """Plot burst fraction over time for each schedule."""
    schedules = bundle.config.schedules
    fig, axes = plt.subplots(
        len(schedules), 1, figsize=figsize("full", len(schedules)), sharex=True
    )
    if len(schedules) == 1:
        axes = [axes]

    max_len = max(len(bundle.schedule_bars[s].fractions) for s in schedules)
    for ax, schedule in zip(axes, schedules, strict=True):
        fracs = np.array(bundle.schedule_bars[schedule].fractions, dtype=float)
        xs = np.arange(len(fracs))
        ax.fill_between(xs, fracs, color=SCHED_COLORS[schedule], alpha=0.78)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(0, max_len - 1)
        ax.set_ylabel(
            SCHED_DISPLAY.get(schedule, schedule), rotation=0, labelpad=48, ha="left", va="center"
        )
        ax.set_yticks([0.0, 0.5, 1.0])
    style_axes(axes[-1], "Step", "Burst Fraction")
    return save_chart(fig, out_dir / "schedule_bars.pdf")


def plot_lr_curves(bundle: CoreBundle, out_dir: Path) -> Path:
    """Plot learning rate schedules for all schedules."""
    fig, ax = plt.subplots(figsize=figsize("half"))
    for schedule in bundle.config.schedules:
        curve = bundle.lr_curves[schedule]
        ax.plot(
            curve.steps,
            curve.lr,
            color=SCHED_COLORS[schedule],
            label=SCHED_DISPLAY.get(schedule, schedule),
        )
    style_axes(ax, "Step", "Learning Rate")
    ax.legend(loc="upper right", ncol=2)
    return save_chart(fig, out_dir / "lr_schedule.pdf")


def plot_overlay(  # noqa: PLR0913
    bundle: CoreBundle,
    out_dir: Path,
    metric: str,
    ylabel: str,
    filename: str,
    ylim: tuple[float, float],
) -> Path:
    """Plot a training metric overlay across schedules."""
    fig, ax = plt.subplots(figsize=figsize("half"))
    max_burst_end = 0
    for schedule in bundle.config.schedules:
        ts = bundle.training[schedule]
        data = training_metric(ts, metric)
        steps = np.array(data.steps, dtype=float)
        mean = np.array(data.mean, dtype=float)
        ci = np.array(data.ci, dtype=float)
        burst_end = ts.pre_steps + ts.burst_steps
        max_burst_end = max(max_burst_end, burst_end)
        ax.plot(
            steps,
            mean,
            color=SCHED_COLORS[schedule],
            label=SCHED_DISPLAY.get(schedule, schedule),
        )
        ax.fill_between(steps, mean - ci, mean + ci, color=SCHED_COLORS[schedule], alpha=0.12)

    ax.set_ylim(ylim)
    annotate_global_phase_boundaries(ax, max_burst_end, max_total_steps(bundle))
    style_axes(ax, "Step", ylabel)
    ax.legend(loc="best", ncol=2)

    if metric.startswith("loss"):
        paths = save_with_log_variant(fig, ax, out_dir / filename)
        for p in paths:
            write_aliases(p, overlay_aliases(p.name))
        return paths[0]

    path = save_chart(fig, out_dir / filename)
    write_aliases(path, overlay_aliases(filename))
    return path


def plot_auc_bars(bundle: CoreBundle, out_dir: Path) -> Path:
    """Plot reversion AUC bar chart across schedules."""
    schedules = bundle.config.schedules
    by_sched = bundle.summary.by_schedule
    means = [by_sched[s].reversion_auc.mean for s in schedules]
    cis = [by_sched[s].reversion_auc.ci for s in schedules]

    fig, ax = plt.subplots(figsize=figsize("half"))
    xs = np.arange(len(schedules))
    colors = [SCHED_COLORS[s] for s in schedules]
    ax.bar(xs, means, yerr=cis, color=colors, edgecolor="black", lw=0.7, capsize=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([SCHED_DISPLAY.get(s, s) for s in schedules], rotation=25, ha="right")
    style_axes(ax, "", "AUC")
    return save_chart(fig, out_dir / "reversion_auc_bars.pdf")


def plot_summary_table(bundle: CoreBundle, out_dir: Path) -> Path:
    """Render a summary statistics table as an image."""
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


def plot_reversion_zoom(bundle: CoreBundle, out_dir: Path) -> Path:
    """Plot burst accuracy during the reversion phase only."""
    fig, ax = plt.subplots(figsize=figsize("half"))
    for schedule in bundle.config.schedules:
        ts = bundle.training[schedule]
        data = ts.acc_burst
        steps = np.array(data.steps, dtype=float)
        mean = np.array(data.mean, dtype=float)
        ci = np.array(data.ci, dtype=float)
        burst_end = ts.pre_steps + ts.burst_steps
        mask = steps >= burst_end
        local_steps = steps[mask] - burst_end
        ax.plot(
            local_steps,
            mean[mask],
            color=SCHED_COLORS[schedule],
            label=SCHED_DISPLAY.get(schedule, schedule),
        )
        ax.fill_between(
            local_steps,
            mean[mask] - ci[mask],
            mean[mask] + ci[mask],
            color=SCHED_COLORS[schedule],
            alpha=0.12,
        )
    ax.set_ylim(ACC_YLIM)
    style_axes(ax, "Reversion Step", "Special Accuracy")
    ax.legend(loc="upper right", ncol=2)
    return save_chart(fig, out_dir / "reversion_zoom_forgetting_speed.pdf")


def plot_reversion_zoom_loss(bundle: CoreBundle, out_dir: Path) -> list[Path]:
    """Plot burst eval loss during the reversion phase only (linear + log)."""
    fig, ax = plt.subplots(figsize=figsize("half"))
    for schedule in bundle.config.schedules:
        ts = bundle.training[schedule]
        data = ts.loss_burst
        steps = np.array(data.steps, dtype=float)
        mean = np.array(data.mean, dtype=float)
        ci = np.array(data.ci, dtype=float)
        burst_end = ts.pre_steps + ts.burst_steps
        mask = steps >= burst_end
        local_steps = steps[mask] - burst_end
        ax.plot(
            local_steps,
            mean[mask],
            color=SCHED_COLORS[schedule],
            label=SCHED_DISPLAY.get(schedule, schedule),
        )
        ax.fill_between(
            local_steps,
            mean[mask] - ci[mask],
            mean[mask] + ci[mask],
            color=SCHED_COLORS[schedule],
            alpha=0.12,
        )
    style_axes(ax, "Reversion Step", "Special Eval Loss")
    ax.legend(loc="best", ncol=2)
    return save_with_log_variant(fig, ax, out_dir / "reversion_zoom_loss.pdf")


def plot_grad_cosine(bundle: CoreBundle, out_dir: Path) -> Path | None:
    """Plot gradient cosine similarity overlay across schedules."""
    gradients = bundle.gradients
    if not gradients:
        return None
    fig, ax = plt.subplots(figsize=figsize("half"))
    for schedule in bundle.config.schedules:
        if schedule not in gradients:
            continue
        g = gradients[schedule]
        steps = np.array(g.steps, dtype=float)
        mean = np.array(g.cosine.mean, dtype=float)
        ci = np.array(g.cosine.ci, dtype=float)
        ax.plot(
            steps,
            mean,
            color=SCHED_COLORS[schedule],
            label=SCHED_DISPLAY.get(schedule, schedule),
        )
        ax.fill_between(steps, mean - ci, mean + ci, color=SCHED_COLORS[schedule], alpha=0.12)
    ax.axhline(0.0, color=COLOR_ZERO_LINE, ls=":", lw=1.0)
    annotate_global_phase_boundaries(ax, max_grad_burst_end(bundle), max_grad_total_steps(bundle))
    style_axes(ax, "Step", "Cosine")
    ax.legend(loc="best", ncol=2)
    path = save_chart(fig, out_dir / "grad_cosine_burst_vs_other.pdf")
    write_aliases(path, [out_dir / "grad_cosine.pdf"])
    return path


def plot_grad_cosine_per_schedule(bundle: CoreBundle, out_dir: Path) -> Path | None:
    """Plot per-schedule gradient cosine similarity charts."""
    gradients = bundle.gradients
    if not gradients:
        return None

    first_path: Path | None = None
    for schedule in bundle.config.schedules:
        if schedule not in gradients:
            continue
        g = gradients[schedule]
        steps = np.array(g.steps, dtype=float)
        mean = np.array(g.cosine.mean, dtype=float)
        ci = np.array(g.cosine.ci, dtype=float)

        fig, ax = plt.subplots(figsize=figsize("half"))
        ax.plot(steps, mean, color=SCHED_COLORS[schedule])
        ax.fill_between(steps, mean - ci, mean + ci, color=SCHED_COLORS[schedule], alpha=0.14)
        ax.axhline(0.0, color=COLOR_ZERO_LINE, ls=":", lw=1.0)
        annotate_global_phase_boundaries(ax, g.burst_steps, steps[-1])
        style_axes(ax, "Step", "Cosine")
        path = save_chart(fig, out_dir / f"grad_cosine_{schedule.upper()}_per_schedule.pdf")
        write_aliases(path, [out_dir / f"grad_cosine_{schedule}.pdf"])
        if first_path is None:
            first_path = path
    return first_path


def plot_grad_norms(bundle: CoreBundle, out_dir: Path) -> Path | None:
    """Plot burst and other gradient L2 norms side by side."""
    gradients = bundle.gradients
    if not gradients:
        return None

    fig, axes = plt.subplots(1, 2, figsize=figsize("full"), sharey=False)
    for schedule in bundle.config.schedules:
        if schedule not in gradients:
            continue
        g = gradients[schedule]
        steps = np.array(g.steps, dtype=float)

        burst_mean = np.array(g.burst_norm.mean, dtype=float)
        burst_ci = np.array(g.burst_norm.ci, dtype=float)
        axes[0].plot(
            steps,
            burst_mean,
            color=SCHED_COLORS[schedule],
            label=SCHED_DISPLAY.get(schedule, schedule),
        )
        axes[0].fill_between(
            steps,
            burst_mean - burst_ci,
            burst_mean + burst_ci,
            color=SCHED_COLORS[schedule],
            alpha=0.12,
        )

        other_mean = np.array(g.other_norm.mean, dtype=float)
        other_ci = np.array(g.other_norm.ci, dtype=float)
        axes[1].plot(
            steps,
            other_mean,
            color=SCHED_COLORS[schedule],
            label=SCHED_DISPLAY.get(schedule, schedule),
        )
        axes[1].fill_between(
            steps,
            other_mean - other_ci,
            other_mean + other_ci,
            color=SCHED_COLORS[schedule],
            alpha=0.12,
        )

    grad_burst = max_grad_burst_end(bundle)
    grad_total = max_grad_total_steps(bundle)
    annotate_global_phase_boundaries(axes[0], grad_burst, grad_total)
    annotate_global_phase_boundaries(axes[1], grad_burst, grad_total)
    style_axes(axes[0], "Step", "L2 Norm")
    style_axes(axes[1], "Step", "L2 Norm")
    axes[1].legend(loc="best", ncol=1)
    path = save_chart(fig, out_dir / "grad_norm_l2_burst_and_other.pdf")
    write_aliases(path, [out_dir / "grad_norms.pdf"])
    return path


def plot_grad_norm_x_cosine(bundle: CoreBundle, out_dir: Path) -> Path | None:
    """Plot signed dot product and interference power charts."""
    gradients = bundle.gradients
    if not gradients:
        return None

    fig, axes = plt.subplots(1, 2, figsize=figsize("full"), sharey=False)
    for schedule in bundle.config.schedules:
        if schedule not in gradients:
            continue
        g = gradients[schedule]
        steps = np.array(g.steps, dtype=float)

        sd_mean = np.array(g.signed_dot.mean, dtype=float)
        sd_ci = np.array(g.signed_dot.ci, dtype=float)
        axes[0].plot(
            steps,
            sd_mean,
            color=SCHED_COLORS[schedule],
            label=SCHED_DISPLAY.get(schedule, schedule),
        )
        axes[0].fill_between(
            steps,
            sd_mean - sd_ci,
            sd_mean + sd_ci,
            color=SCHED_COLORS[schedule],
            alpha=0.12,
        )

        ip_mean = np.array(g.interference_power.mean, dtype=float)
        ip_ci = np.array(g.interference_power.ci, dtype=float)
        axes[1].plot(
            steps,
            ip_mean,
            color=SCHED_COLORS[schedule],
            label=SCHED_DISPLAY.get(schedule, schedule),
        )
        axes[1].fill_between(
            steps,
            ip_mean - ip_ci,
            ip_mean + ip_ci,
            color=SCHED_COLORS[schedule],
            alpha=0.12,
        )

    axes[0].axhline(0.0, color=COLOR_ZERO_LINE, ls=":", lw=1.0)
    grad_burst = max_grad_burst_end(bundle)
    grad_total = max_grad_total_steps(bundle)
    annotate_global_phase_boundaries(axes[0], grad_burst, grad_total)
    annotate_global_phase_boundaries(axes[1], grad_burst, grad_total)
    style_axes(axes[0], "Step", "Signed Dot")
    style_axes(axes[1], "Step", "Power")
    axes[1].legend(loc="best", ncol=1)
    return save_chart(fig, out_dir / "grad_norm_x_cosine_and_interference_power.pdf")


def plot_representation_drift(bundle: CoreBundle, out_dir: Path) -> Path | None:
    """Plot centroid drift and other-shift norm across schedules."""
    by_schedule = bundle.representation.by_schedule
    if not by_schedule:
        return None

    schedules = [s for s in bundle.config.schedules if s in by_schedule]
    if not schedules:
        return None

    xs = np.arange(len(schedules))
    labels = [sched_pct_label(s) for s in schedules]

    fig, axes = plt.subplots(1, 2, figsize=figsize("full"), sharex=True)
    proj_mean = np.array(
        [by_schedule[s].late_centroid_projection.mean for s in schedules], dtype=float
    )
    proj_ci = np.array([by_schedule[s].late_centroid_projection.ci for s in schedules], dtype=float)
    shift_mean = np.array(
        [by_schedule[s].late_other_shift_norm.mean for s in schedules], dtype=float
    )
    shift_ci = np.array([by_schedule[s].late_other_shift_norm.ci for s in schedules], dtype=float)

    axes[0].plot(xs, proj_mean, color=COLOR_PROJECTION, marker="o", ms=3)
    axes[0].fill_between(
        xs, proj_mean - proj_ci, proj_mean + proj_ci, color=COLOR_PROJECTION, alpha=0.12
    )
    axes[0].axhline(0.0, color=COLOR_ZERO_LINE, ls=":", lw=1.0)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(labels)
    style_axes(axes[0], "Concentration %", "Projection")

    axes[1].plot(xs, shift_mean, color=COLOR_SHIFT, marker="o", ms=3)
    axes[1].fill_between(
        xs, shift_mean - shift_ci, shift_mean + shift_ci, color=COLOR_SHIFT, alpha=0.12
    )
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels)
    style_axes(axes[1], "Concentration %", "Normalized Shift")

    path = save_chart(fig, out_dir / "representation_drift_centroid_and_shift.pdf")
    write_aliases(path, [out_dir / "rep_drift_summary.pdf"])
    return path


def plot_burst_representation_drift(bundle: CoreBundle, out_dir: Path) -> Path | None:
    """Plot burst self-projection and burst normalized shift across schedules."""
    by_schedule = bundle.representation.by_schedule
    if not by_schedule:
        return None

    schedules = [s for s in bundle.config.schedules if s in by_schedule]
    if not schedules:
        return None

    xs = np.arange(len(schedules))
    labels = [sched_pct_label(s) for s in schedules]

    proj_mean = np.array(
        [by_schedule[s].late_burst_self_projection.mean for s in schedules], dtype=float
    )
    proj_ci = np.array(
        [by_schedule[s].late_burst_self_projection.ci for s in schedules], dtype=float
    )
    shift_mean = np.array(
        [by_schedule[s].late_burst_shift_norm.mean for s in schedules], dtype=float
    )
    shift_ci = np.array([by_schedule[s].late_burst_shift_norm.ci for s in schedules], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=figsize("full"), sharex=True)
    axes[0].plot(xs, proj_mean, color=COLOR_SPECIAL, marker="o", ms=3)
    axes[0].fill_between(
        xs, proj_mean - proj_ci, proj_mean + proj_ci, color=COLOR_SPECIAL, alpha=0.12
    )
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(labels)
    style_axes(axes[0], "Concentration %", "||burst drift||")

    axes[1].plot(xs, shift_mean, color=COLOR_SPECIAL, marker="o", ms=3)
    axes[1].fill_between(
        xs, shift_mean - shift_ci, shift_mean + shift_ci, color=COLOR_SPECIAL, alpha=0.12
    )
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels)
    style_axes(axes[1], "Concentration %", "Normalized Shift (burst)")

    return save_chart(fig, out_dir / "representation_drift_burst_self.pdf")


def plot_centroid_norms(bundle: CoreBundle, out_dir: Path) -> Path | None:
    """Plot post/pre centroid norm ratios for burst and other data."""
    by_schedule = bundle.representation.by_schedule
    if not by_schedule:
        return None

    schedules = [s for s in bundle.config.schedules if s in by_schedule]
    if not schedules:
        return None

    xs = np.arange(len(schedules))
    labels = [sched_pct_label(s) for s in schedules]

    burst_ratio_seeds: list[list[float]] = []
    other_ratio_seeds: list[list[float]] = []
    for s in schedules:
        per_seed = by_schedule[s].per_seed
        burst_ratio_seeds.append(
            [sd.late_burst_post_norm / (sd.late_burst_pre_norm + 1e-12) for sd in per_seed]
        )
        other_ratio_seeds.append(
            [sd.late_other_post_norm / (sd.late_burst_pre_norm + 1e-12) for sd in per_seed]
        )

    burst_mean = np.array([np.mean(r) for r in burst_ratio_seeds], dtype=float)
    burst_ci = np.array([mean_ci(np.array(r))[1] for r in burst_ratio_seeds], dtype=float)
    other_mean = np.array([np.mean(r) for r in other_ratio_seeds], dtype=float)
    other_ci = np.array([mean_ci(np.array(r))[1] for r in other_ratio_seeds], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=figsize("full"), sharex=True)
    axes[0].plot(xs, burst_mean, color=COLOR_SPECIAL, marker="o", ms=3)
    axes[0].fill_between(
        xs, burst_mean - burst_ci, burst_mean + burst_ci, color=COLOR_SPECIAL, alpha=0.12
    )
    axes[0].axhline(1.0, color=COLOR_ZERO_LINE, ls=":", lw=1.0)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(labels)
    style_axes(axes[0], "Concentration %", "burst_post / burst_pre")

    axes[1].plot(xs, other_mean, color=COLOR_OTHER, marker="o", ms=3)
    axes[1].fill_between(
        xs, other_mean - other_ci, other_mean + other_ci, color=COLOR_OTHER, alpha=0.12
    )
    axes[1].axhline(1.0, color=COLOR_ZERO_LINE, ls=":", lw=1.0)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels)
    style_axes(axes[1], "Concentration %", "other_post / burst_pre")

    return save_chart(fig, out_dir / "representation_centroid_norms.pdf")


def plot_grad_rank(bundle: CoreBundle, out_dir: Path) -> Path | None:
    """Plot effective gradient rank overlay across schedules."""
    gradients = bundle.gradients
    if not gradients:
        return None

    has_rank = any(g.grad_rank is not None for g in gradients.values())
    if not has_rank:
        return None

    fig, ax = plt.subplots(figsize=figsize("half"))
    for schedule in bundle.config.schedules:
        if schedule not in gradients:
            continue
        g = gradients[schedule]
        if g.grad_rank is None:
            continue
        steps = np.array(g.steps, dtype=float)
        mean = np.array(g.grad_rank.mean, dtype=float)
        ci = np.array(g.grad_rank.ci, dtype=float)
        ax.plot(
            steps,
            mean,
            color=SCHED_COLORS[schedule],
            label=SCHED_DISPLAY.get(schedule, schedule),
        )
        ax.fill_between(steps, mean - ci, mean + ci, color=SCHED_COLORS[schedule], alpha=0.12)

    annotate_global_phase_boundaries(ax, max_grad_burst_end(bundle), max_grad_total_steps(bundle))
    style_axes(ax, "Step", "Effective Rank")
    ax.legend(loc="best", ncol=2)
    return save_chart(fig, out_dir / "grad_rank_effective.pdf")


def plot_per_schedule(bundle: CoreBundle, out_dir: Path) -> list[Path]:
    """Plot per-schedule burst vs other accuracy charts."""
    paths: list[Path] = []
    for schedule in bundle.config.schedules:
        ts = bundle.training[schedule]
        steps = np.array(ts.acc_burst.steps, dtype=float)
        burst_end = ts.pre_steps + ts.burst_steps

        fig, ax = plt.subplots(figsize=figsize("half"))
        for data, color, label in (
            (ts.acc_other, COLOR_OTHER, "Other"),
            (ts.acc_burst, COLOR_SPECIAL, "Special"),
        ):
            mean = np.array(data.mean, dtype=float)
            ci = np.array(data.ci, dtype=float)
            ax.plot(steps, mean, color=color, label=label)
            ax.fill_between(steps, mean - ci, mean + ci, color=color, alpha=0.14)

        if ts.pre_steps > 0:
            ax.axvline(ts.pre_steps, color="black", ls="--", lw=1.2, alpha=0.65)
        ax.axvline(burst_end, color="black", ls="--", lw=1.2, alpha=0.65)
        ax.set_ylim(ACC_YLIM)
        style_axes(ax, "Step", "Accuracy")
        ax.legend(loc="best")
        path = save_chart(fig, out_dir / f"per_sched_{schedule.upper()}_accuracy.pdf")
        write_aliases(path, [out_dir / f"per_schedule_{schedule}.pdf"])
        paths.append(path)
    return paths


def plot_per_schedule_loss(
    bundle: CoreBundle, out_dir: Path, ylim: tuple[float, float]
) -> list[Path]:
    """Plot per-schedule burst vs other eval loss charts (linear + log)."""
    paths: list[Path] = []
    for schedule in bundle.config.schedules:
        ts = bundle.training[schedule]
        steps = np.array(ts.loss_burst.steps, dtype=float)
        burst_end = ts.pre_steps + ts.burst_steps

        fig, ax = plt.subplots(figsize=figsize("half"))
        for data, color, label in (
            (ts.loss_other, COLOR_OTHER, "Other"),
            (ts.loss_burst, COLOR_SPECIAL, "Special"),
        ):
            mean = np.array(data.mean, dtype=float)
            ci = np.array(data.ci, dtype=float)
            ax.plot(steps, mean, color=color, label=label)
            ax.fill_between(steps, mean - ci, mean + ci, color=color, alpha=0.14)

        if ts.pre_steps > 0:
            ax.axvline(ts.pre_steps, color="black", ls="--", lw=1.2, alpha=0.65)
        ax.axvline(burst_end, color="black", ls="--", lw=1.2, alpha=0.65)
        ax.set_ylim(ylim)
        style_axes(ax, "Step", "Eval Loss")
        ax.legend(loc="best")
        paths.extend(
            save_with_log_variant(fig, ax, out_dir / f"per_sched_{schedule.upper()}_loss.pdf")
        )
    return paths


def fmt_ci(metric: MeanCI | LifeEntry, digits: int = 3) -> str:
    """Format a mean +- CI string from a metric dataclass."""
    if digits == 0:
        return f"{metric.mean:.0f} ± {metric.ci:.0f}"
    return f"{metric.mean:.{digits}f} ± {metric.ci:.{digits}f}"


def save_chart(fig: Figure, path: Path) -> Path:
    """Save a figure and return its path."""
    path = path.with_suffix(".pdf")
    save_figure(fig, path)
    return path


def overlay_aliases(filename: str) -> list[Path]:
    """Return shorthand alias paths for a given overlay filename."""
    if "_ACC_BURST_" in filename:
        return [Path("overlay_burst.pdf")]
    if "_ACC_OTHER_" in filename:
        return [Path("overlay_other.pdf")]
    if "_LOSS_" in filename:
        return [Path("overlay_loss.pdf")]
    return []


def write_aliases(source: Path, aliases: list[Path]) -> None:
    """Copy source file to each alias path."""
    for alias in aliases:
        target = source.parent / alias.name if not alias.is_absolute() else alias
        shutil.copyfile(source, target)


def max_total_steps(bundle: CoreBundle) -> int:
    """Return the maximum total steps across all schedules."""
    max_total = 0
    for schedule in bundle.config.schedules:
        ts = bundle.training[schedule]
        max_total = max(max_total, ts.pre_steps + ts.burst_steps + ts.reversion_steps)
    return max_total


def max_burst_steps(bundle: CoreBundle) -> int:
    """Return the maximum burst end step across all schedules."""
    max_burst = 0
    for schedule in bundle.config.schedules:
        ts = bundle.training[schedule]
        max_burst = max(max_burst, ts.pre_steps + ts.burst_steps)
    return max_burst


def max_grad_burst_end(bundle: CoreBundle) -> int:
    """Return the maximum gradient-local burst end step across schedules."""
    return max((g.burst_steps for g in bundle.gradients.values()), default=0)


def max_grad_total_steps(bundle: CoreBundle) -> int:
    """Return the maximum gradient step across schedules."""
    return max((int(g.steps[-1]) for g in bundle.gradients.values() if g.steps), default=0)


def annotate_global_phase_boundaries(ax: Axes, burst_end: float, total_steps: float) -> None:
    """Draw vertical phase boundary lines and labels on an axes."""
    ax.axvline(burst_end, color="black", ls="--", lw=1.15, alpha=0.6)
    ymax = ax.get_ylim()[1]
    ax.text(
        burst_end * 0.5, ymax * 0.97, "SPECIAL", ha="center", va="top", fontsize=6, color="gray"
    )
    ax.text(
        burst_end + max(total_steps - burst_end, 1) * 0.5,
        ymax * 0.97,
        "ALL-BUT-SPECIAL",
        ha="center",
        va="top",
        fontsize=6,
        color="gray",
    )


def sched_pct_label(schedule: str) -> str:
    """Extract the percentage suffix from a schedule name."""
    return schedule.rsplit("_", maxsplit=1)[-1]


def plot_extended_auc_bars(bundle: CoreBundle, out_dir: Path) -> Path:
    """Plot reversion AUC bars for burst-acc, burst-loss, other-acc, other-loss."""
    schedules = bundle.config.schedules
    by_sched = bundle.summary.by_schedule

    attrs = (
        "reversion_auc",
        "reversion_auc_loss_burst",
        "reversion_auc_acc_other",
        "reversion_auc_loss_other",
    )
    titles = ("Burst Acc AUC", "Burst Loss AUC", "Other Acc AUC", "Other Loss AUC")

    fig, axes = plt.subplots(1, len(attrs), figsize=figsize("full"), sharey=False)
    xs = np.arange(len(schedules))
    colors = [SCHED_COLORS[s] for s in schedules]
    labels = [SCHED_DISPLAY.get(s, s) for s in schedules]

    for ax, attr, title in zip(axes, attrs, titles, strict=True):
        means = [getattr(by_sched[s], attr).mean for s in schedules]
        cis = [getattr(by_sched[s], attr).ci for s in schedules]
        ax.bar(xs, means, yerr=cis, color=colors, edgecolor="black", lw=0.7, capsize=4)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        style_axes(ax, "", title)

    return save_chart(fig, out_dir / "extended_reversion_auc_bars.pdf")


def build_layer_grid(
    layer_names: list[str], steps: list[float], metric_dict: dict[str, SeriesMeanCI]
) -> np.ndarray:
    """Build a (n_layers, n_steps) grid from metric_dict means."""
    n_layers = len(layer_names)
    n_steps = len(steps)
    grid = np.full((n_layers, n_steps), np.nan)
    for li, ln in enumerate(layer_names):
        vals = metric_dict[ln].mean
        grid[li, : len(vals)] = vals
    return grid


def plot_layer_heatmap(  # noqa: PLR0913
    layer_names: list[str],
    steps: list[float],
    metric_dict: dict[str, SeriesMeanCI],
    out_path: Path,
    ylabel: str,
    *,
    cmap: str = LAYER_CMAP,
    center_zero: bool = False,
) -> Path:
    """Render a layer x step heatmap."""
    grid = build_layer_grid(layer_names, steps, metric_dict)
    n_layers, n_steps = grid.shape
    fig, ax = plt.subplots(figsize=figsize("full"))
    masked = np.ma.masked_invalid(grid)
    vmin, vmax = None, None
    if center_zero:
        abs_max = max(abs(np.nanmin(grid)), abs(np.nanmax(grid)), 1e-12)
        vmin, vmax = -abs_max, abs_max
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(layer_names, fontsize=6)
    n_xticks = min(10, n_steps)
    tick_idx = np.linspace(0, n_steps - 1, n_xticks, dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{steps[i]:.0f}" for i in tick_idx], fontsize=6)
    ax.set_xlabel("Step")
    ax.set_ylabel("Layer")
    fig.colorbar(im, ax=ax, label=ylabel, shrink=0.8)
    return save_chart(fig, out_path)


def plot_layer_lines(
    layer_names: list[str],
    steps: list[float],
    metric_dict: dict[str, SeriesMeanCI],
    out_path: Path,
    ylabel: str,
) -> Path:
    """Render a per-layer line chart."""
    n_layers = len(layer_names)
    fig, ax = plt.subplots(figsize=figsize("half"))
    cmap_obj = plt.get_cmap(LAYER_LINE_CMAP)
    for li, ln in enumerate(layer_names):
        mean = np.array(metric_dict[ln].mean, dtype=float)
        ax.plot(steps[: len(mean)], mean, lw=1.4, label=ln, color=cmap_obj(li / max(n_layers, 1)))
    style_axes(ax, "Step", ylabel)
    ax.legend(loc="best", ncol=3, fontsize=5)
    return save_chart(fig, out_path)


def plot_per_layer_cossim(bundle: CoreBundle, out_dir: Path) -> list[Path]:
    """Plot per-layer gradient cosine similarity heatmaps and line charts."""
    paths: list[Path] = []
    for schedule in bundle.config.schedules:
        data = bundle.per_layer_gradients[schedule]
        paths.append(
            plot_layer_heatmap(
                data.layer_names,
                data.steps,
                data.cosine,
                out_dir / f"per_layer_cossim_{schedule}_heatmap.pdf",
                "Cosine Similarity",
                center_zero=True,
            )
        )
        paths.append(
            plot_layer_lines(
                data.layer_names,
                data.steps,
                data.cosine,
                out_dir / f"per_layer_cossim_{schedule}_lines.pdf",
                "Cosine Similarity",
            )
        )
    return paths


def plot_per_layer_grad_norm(bundle: CoreBundle, out_dir: Path) -> list[Path]:
    """Plot per-layer gradient norm heatmaps and line charts."""
    paths: list[Path] = []
    for schedule in bundle.config.schedules:
        data = bundle.per_layer_gradients[schedule]
        for prefix, md in (("burst", data.burst_norm), ("other", data.other_norm)):
            paths.append(
                plot_layer_heatmap(
                    data.layer_names,
                    data.steps,
                    md,
                    out_dir / f"per_layer_grad_norm_{prefix}_{schedule}_heatmap.pdf",
                    f"Grad Norm ({prefix})",
                    cmap=DRIFT_CMAP,
                )
            )
            paths.append(
                plot_layer_lines(
                    data.layer_names,
                    data.steps,
                    md,
                    out_dir / f"per_layer_grad_norm_{prefix}_{schedule}_lines.pdf",
                    f"Grad Norm ({prefix})",
                )
            )
    return paths


def plot_per_layer_norm_x_cossim(bundle: CoreBundle, out_dir: Path) -> list[Path]:
    """Plot per-layer norm*cosine heatmaps and line charts."""
    paths: list[Path] = []
    for schedule in bundle.config.schedules:
        data = bundle.per_layer_gradients[schedule]
        paths.append(
            plot_layer_heatmap(
                data.layer_names,
                data.steps,
                data.norm_x_cosine,
                out_dir / f"per_layer_norm_x_cossim_{schedule}_heatmap.pdf",
                "Norm x Cosine",
                center_zero=True,
            )
        )
        paths.append(
            plot_layer_lines(
                data.layer_names,
                data.steps,
                data.norm_x_cosine,
                out_dir / f"per_layer_norm_x_cossim_{schedule}_lines.pdf",
                "Norm x Cosine",
            )
        )
    return paths


def plot_weight_drift(bundle: CoreBundle, out_dir: Path) -> list[Path]:
    """Plot per-layer weight drift heatmaps and line charts."""
    paths: list[Path] = []
    for schedule in bundle.config.schedules:
        data = bundle.weight_drift[schedule]
        paths.append(
            plot_layer_heatmap(
                data.layer_names,
                data.steps,
                data.cumulative,
                out_dir / f"weight_drift_{schedule}_heatmap.pdf",
                "Weight Drift (Frobenius)",
                cmap=DRIFT_CMAP,
            )
        )
        paths.append(
            plot_layer_lines(
                data.layer_names,
                data.steps,
                data.cumulative,
                out_dir / f"weight_drift_{schedule}_lines.pdf",
                "Weight Drift (Frobenius)",
            )
        )
    return paths


# ---------------------------------------------------------------------------
# Next-token probe charts
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
    """Render all next-token probe charts, returning paths (empty if no data)."""
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
    """Plot per-schedule, per-regime probe accuracy by layer."""
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
    """Plot Other-minus-Burst probe accuracy diff per schedule."""
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
                x,
                mean,
                "-o",
                color=color,
                ms=2,
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
    """Plot pairwise schedule diff-in-diffs for probe accuracy."""
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
