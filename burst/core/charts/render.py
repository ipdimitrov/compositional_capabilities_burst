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

from burst.config import (
    ACC_BURST,
    ACC_OTHER,
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
    LOSS_BURST,
    LOSS_OTHER,
    SCHED_COLORS,
    SCHED_DISPLAY,
    reversion_life_label,
)
from burst.core.charts.style import apply_paper_style, figsize, save_figure, style_axes
from burst.core.train_utils import mean_ci


def render_core_charts(bundle: dict, out_dir: str | Path) -> list[Path]:
    """Render all core analysis charts to out_dir."""
    apply_paper_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    burst_fname = f"overlay_{ACC_BURST.upper()}_{CLASS_BURST}_class_accuracy.pdf"
    other_fname = f"overlay_{ACC_OTHER.upper()}_{CLASS_OTHER}_class_accuracy.pdf"
    paths = [
        plot_schedule_bars(bundle, out_dir),
        plot_lr_curves(bundle, out_dir),
        plot_overlay(bundle, out_dir, ACC_BURST, f"{CLASS_BURST} Accuracy", burst_fname),
        plot_overlay(bundle, out_dir, ACC_OTHER, f"{CLASS_OTHER} Accuracy", other_fname),
        plot_overlay(bundle, out_dir, "loss", "Loss", "overlay_LOSS_training_loss.pdf"),
        plot_overlay(
            bundle,
            out_dir,
            LOSS_BURST,
            f"{CLASS_BURST} Eval Loss",
            f"overlay_{LOSS_BURST.upper()}_eval_loss.pdf",
        ),
        plot_overlay(
            bundle,
            out_dir,
            LOSS_OTHER,
            f"{CLASS_OTHER} Eval Loss",
            f"overlay_{LOSS_OTHER.upper()}_eval_loss.pdf",
        ),
        plot_auc_bars(bundle, out_dir),
        plot_extended_auc_bars(bundle, out_dir),
        plot_summary_table(bundle, out_dir),
        plot_reversion_zoom(bundle, out_dir),
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
    paths.extend(plot_per_layer_cossim(bundle, out_dir))
    paths.extend(plot_per_layer_grad_norm(bundle, out_dir))
    paths.extend(plot_per_layer_norm_x_cossim(bundle, out_dir))
    paths.extend(plot_weight_drift(bundle, out_dir))
    return [path for path in paths if path is not None]


def plot_schedule_bars(bundle: dict, out_dir: Path) -> Path:
    """Plot burst fraction over time for each schedule."""
    schedules = bundle["config"]["schedules"]
    bars = bundle["schedule_bars"]
    fig, axes = plt.subplots(
        len(schedules), 1, figsize=figsize("full", len(schedules)), sharex=True
    )
    if len(schedules) == 1:
        axes = [axes]

    max_len = max(len(bars[schedule]["fractions"]) for schedule in schedules)
    for ax, schedule in zip(axes, schedules, strict=True):
        fracs = np.array(bars[schedule]["fractions"], dtype=float)
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


def plot_lr_curves(bundle: dict, out_dir: Path) -> Path:
    """Plot learning rate schedules for all schedules."""
    fig, ax = plt.subplots(figsize=figsize("half"))
    for schedule in bundle["config"]["schedules"]:
        curve = bundle["lr_curves"][schedule]
        ax.plot(
            curve["steps"],
            curve["lr"],
            color=SCHED_COLORS[schedule],
            label=SCHED_DISPLAY.get(schedule, schedule),
        )
    style_axes(ax, "Step", "Learning Rate")
    ax.legend(loc="upper right", ncol=2)
    return save_chart(fig, out_dir / "lr_schedule.pdf")


def plot_overlay(bundle: dict, out_dir: Path, metric: str, ylabel: str, filename: str) -> Path:
    """Plot a training metric overlay across schedules."""
    fig, ax = plt.subplots(figsize=figsize("half"))
    max_burst_end = 0
    for schedule in bundle["config"]["schedules"]:
        training_data = bundle["training"][schedule]
        metric_data = training_data[metric]
        steps = np.array(metric_data["steps"], dtype=float)
        mean = np.array(metric_data["mean"], dtype=float)
        ci = np.array(metric_data["ci"], dtype=float)
        burst_end = training_data["pre_steps"] + training_data["burst_steps"]
        max_burst_end = max(max_burst_end, burst_end)
        ax.plot(
            steps,
            mean,
            color=SCHED_COLORS[schedule],
            label=SCHED_DISPLAY.get(schedule, schedule),
        )
        ax.fill_between(steps, mean - ci, mean + ci, color=SCHED_COLORS[schedule], alpha=0.12)

    if metric != "loss":
        ax.set_ylim(-0.05, 1.05)
    annotate_global_phase_boundaries(ax, max_burst_end, max_total_steps(bundle))
    style_axes(ax, "Step", ylabel)
    ax.legend(loc="best", ncol=2)
    path = save_chart(fig, out_dir / filename)
    write_aliases(path, overlay_aliases(filename))
    return path


def plot_auc_bars(bundle: dict, out_dir: Path) -> Path:
    """Plot reversion AUC bar chart across schedules."""
    schedules = bundle["config"]["schedules"]
    summary = bundle["summary"]["by_schedule"]
    means = [summary[schedule]["reversion_auc"]["mean"] for schedule in schedules]
    cis = [summary[schedule]["reversion_auc"]["ci"] for schedule in schedules]

    fig, ax = plt.subplots(figsize=figsize("half"))
    xs = np.arange(len(schedules))
    colors = [SCHED_COLORS[schedule] for schedule in schedules]
    ax.bar(xs, means, yerr=cis, color=colors, edgecolor="black", lw=0.7, capsize=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [SCHED_DISPLAY.get(schedule, schedule) for schedule in schedules], rotation=25, ha="right"
    )
    style_axes(ax, "", "AUC")
    return save_chart(fig, out_dir / "reversion_auc_bars.pdf")


def plot_summary_table(bundle: dict, out_dir: Path) -> Path:
    """Render a summary statistics table as an image."""
    schedules = bundle["config"]["schedules"]
    summary = bundle["summary"]["by_schedule"]
    thresholds = bundle["config"]["thresholds"]

    headers = ["Schedule", "Peak", "AUC", "Other End"]
    headers.extend(reversion_life_label(threshold) for threshold in thresholds)

    rows = []
    for schedule in schedules:
        schedule_summary = summary[schedule]
        row = [
            SCHED_DISPLAY.get(schedule, schedule),
            fmt_ci(schedule_summary["peak_burst"]),
            fmt_ci(schedule_summary["reversion_auc"], digits=0),
            fmt_ci(schedule_summary["other_end"]),
        ]
        row.extend(
            fmt_ci(schedule_summary["life"][f"life_{int(threshold * 100)}"], digits=0)
            for threshold in thresholds
        )
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


def plot_reversion_zoom(bundle: dict, out_dir: Path) -> Path:
    """Plot burst accuracy during the reversion phase only."""
    fig, ax = plt.subplots(figsize=figsize("half"))
    for schedule in bundle["config"]["schedules"]:
        schedule_data = bundle["training"][schedule]
        steps = np.array(schedule_data[ACC_BURST]["steps"], dtype=float)
        mean = np.array(schedule_data[ACC_BURST]["mean"], dtype=float)
        ci = np.array(schedule_data[ACC_BURST]["ci"], dtype=float)
        burst_end = schedule_data["pre_steps"] + schedule_data["burst_steps"]
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
    ax.set_ylim(-0.05, 1.05)
    style_axes(ax, "Reversion Step", "Special Accuracy")
    ax.legend(loc="upper right", ncol=2)
    return save_chart(fig, out_dir / "reversion_zoom_forgetting_speed.pdf")


def plot_grad_cosine(bundle: dict, out_dir: Path) -> Path | None:
    """Plot gradient cosine similarity overlay across schedules."""
    gradients = bundle["gradients"]
    if not gradients:
        return None
    fig, ax = plt.subplots(figsize=figsize("half"))
    for schedule in bundle["config"]["schedules"]:
        if schedule not in gradients:
            continue
        schedule_data = gradients[schedule]
        steps = np.array(schedule_data["steps"], dtype=float)
        mean = np.array(schedule_data["cosine"]["mean"], dtype=float)
        ci = np.array(schedule_data["cosine"]["ci"], dtype=float)
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


def plot_grad_cosine_per_schedule(bundle: dict, out_dir: Path) -> Path | None:
    """Plot per-schedule gradient cosine similarity charts."""
    gradients = bundle["gradients"]
    if not gradients:
        return None

    first_path: Path | None = None
    for schedule in bundle["config"]["schedules"]:
        if schedule not in gradients:
            continue
        schedule_data = gradients[schedule]
        steps = np.array(schedule_data["steps"], dtype=float)
        mean = np.array(schedule_data["cosine"]["mean"], dtype=float)
        ci = np.array(schedule_data["cosine"]["ci"], dtype=float)
        burst_end = schedule_data["burst_steps"]

        fig, ax = plt.subplots(figsize=figsize("half"))
        ax.plot(steps, mean, color=SCHED_COLORS[schedule])
        ax.fill_between(steps, mean - ci, mean + ci, color=SCHED_COLORS[schedule], alpha=0.14)
        ax.axhline(0.0, color=COLOR_ZERO_LINE, ls=":", lw=1.0)
        annotate_global_phase_boundaries(ax, burst_end, steps[-1])
        style_axes(ax, "Step", "Cosine")
        path = save_chart(fig, out_dir / f"grad_cosine_{schedule.upper()}_per_schedule.pdf")
        write_aliases(path, [out_dir / f"grad_cosine_{schedule}.pdf"])
        if first_path is None:
            first_path = path
    return first_path


def plot_grad_norms(bundle: dict, out_dir: Path) -> Path | None:
    """Plot burst and other gradient L2 norms side by side."""
    gradients = bundle["gradients"]
    if not gradients:
        return None

    fig, axes = plt.subplots(1, 2, figsize=figsize("full"), sharey=False)
    for schedule in bundle["config"]["schedules"]:
        if schedule not in gradients:
            continue
        schedule_data = gradients[schedule]
        steps = np.array(schedule_data["steps"], dtype=float)

        burst_mean = np.array(schedule_data["burst_norm"]["mean"], dtype=float)
        burst_ci = np.array(schedule_data["burst_norm"]["ci"], dtype=float)
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

        other_mean = np.array(schedule_data["other_norm"]["mean"], dtype=float)
        other_ci = np.array(schedule_data["other_norm"]["ci"], dtype=float)
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


def plot_grad_norm_x_cosine(bundle: dict, out_dir: Path) -> Path | None:
    """Plot signed dot product and interference power charts."""
    gradients = bundle["gradients"]
    if not gradients:
        return None

    fig, axes = plt.subplots(1, 2, figsize=figsize("full"), sharey=False)
    for schedule in bundle["config"]["schedules"]:
        if schedule not in gradients:
            continue
        schedule_data = gradients[schedule]
        steps = np.array(schedule_data["steps"], dtype=float)

        signed_dot_mean = np.array(schedule_data["signed_dot"]["mean"], dtype=float)
        signed_dot_ci = np.array(schedule_data["signed_dot"]["ci"], dtype=float)
        axes[0].plot(
            steps,
            signed_dot_mean,
            color=SCHED_COLORS[schedule],
            label=SCHED_DISPLAY.get(schedule, schedule),
        )
        axes[0].fill_between(
            steps,
            signed_dot_mean - signed_dot_ci,
            signed_dot_mean + signed_dot_ci,
            color=SCHED_COLORS[schedule],
            alpha=0.12,
        )

        power_mean = np.array(schedule_data["interference_power"]["mean"], dtype=float)
        power_ci = np.array(schedule_data["interference_power"]["ci"], dtype=float)
        axes[1].plot(
            steps,
            power_mean,
            color=SCHED_COLORS[schedule],
            label=SCHED_DISPLAY.get(schedule, schedule),
        )
        axes[1].fill_between(
            steps,
            power_mean - power_ci,
            power_mean + power_ci,
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


def plot_representation_drift(bundle: dict, out_dir: Path) -> Path | None:
    """Plot centroid drift and other-shift norm across schedules."""
    representation = bundle.get("representation", {})
    by_schedule = representation.get("by_schedule", {})
    if not by_schedule:
        return None

    schedules = [schedule for schedule in bundle["config"]["schedules"] if schedule in by_schedule]
    if not schedules:
        return None

    xs = np.arange(len(schedules))
    labels = [sched_pct_label(schedule) for schedule in schedules]

    fig, axes = plt.subplots(1, 2, figsize=figsize("full"), sharex=True)
    proj_mean = np.array(
        [by_schedule[schedule]["late_centroid_projection"]["mean"] for schedule in schedules],
        dtype=float,
    )
    proj_ci = np.array(
        [by_schedule[schedule]["late_centroid_projection"]["ci"] for schedule in schedules],
        dtype=float,
    )
    shift_mean = np.array(
        [by_schedule[schedule]["late_other_shift_norm"]["mean"] for schedule in schedules],
        dtype=float,
    )
    shift_ci = np.array(
        [by_schedule[schedule]["late_other_shift_norm"]["ci"] for schedule in schedules],
        dtype=float,
    )

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


def plot_burst_representation_drift(bundle: dict, out_dir: Path) -> Path | None:
    """Plot burst self-projection and burst normalized shift across schedules."""
    representation = bundle.get("representation", {})
    by_schedule = representation.get("by_schedule", {})
    if not by_schedule:
        return None

    schedules = [s for s in bundle["config"]["schedules"] if s in by_schedule]
    if not schedules:
        return None
    first_sched = by_schedule[schedules[0]]
    if "late_burst_self_projection" not in first_sched:
        return None

    xs = np.arange(len(schedules))
    labels = [sched_pct_label(s) for s in schedules]

    proj_mean = np.array(
        [by_schedule[s]["late_burst_self_projection"]["mean"] for s in schedules], dtype=float
    )
    proj_ci = np.array(
        [by_schedule[s]["late_burst_self_projection"]["ci"] for s in schedules], dtype=float
    )
    shift_mean = np.array(
        [by_schedule[s]["late_burst_shift_norm"]["mean"] for s in schedules], dtype=float
    )
    shift_ci = np.array(
        [by_schedule[s]["late_burst_shift_norm"]["ci"] for s in schedules], dtype=float
    )

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


def plot_centroid_norms(bundle: dict, out_dir: Path) -> Path | None:
    """Plot post/pre centroid norm ratios for burst and other data."""
    representation = bundle.get("representation", {})
    by_schedule = representation.get("by_schedule", {})
    if not by_schedule:
        return None

    schedules = [s for s in bundle["config"]["schedules"] if s in by_schedule]
    if not schedules:
        return None
    first_sched = by_schedule[schedules[0]]
    if "late_burst_post_norm" not in first_sched:
        return None

    xs = np.arange(len(schedules))
    labels = [sched_pct_label(s) for s in schedules]

    burst_ratio_seeds: list[list[float]] = []
    other_ratio_seeds: list[list[float]] = []
    for s in schedules:
        per_seed = by_schedule[s]["per_seed"]
        burst_ratio_seeds.append(
            [
                seed["late_burst_post_norm"] / (seed["late_burst_pre_norm"] + 1e-12)
                for seed in per_seed
            ]
        )
        other_ratio_seeds.append(
            [
                seed["late_other_post_norm"] / (seed["late_burst_pre_norm"] + 1e-12)
                for seed in per_seed
            ]
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


def plot_grad_rank(bundle: dict, out_dir: Path) -> Path | None:
    """Plot effective gradient rank overlay across schedules."""
    gradients = bundle.get("gradients", {})
    if not gradients:
        return None

    has_rank = any("grad_rank" in g for g in gradients.values())
    if not has_rank:
        return None

    fig, ax = plt.subplots(figsize=figsize("half"))
    for schedule in bundle["config"]["schedules"]:
        if schedule not in gradients:
            continue
        schedule_data = gradients[schedule]
        rank_data = schedule_data.get("grad_rank")
        if not rank_data:
            continue
        steps = np.array(schedule_data["steps"], dtype=float)
        mean = np.array(rank_data["mean"], dtype=float)
        ci = np.array(rank_data["ci"], dtype=float)
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


def plot_per_schedule(bundle: dict, out_dir: Path) -> list[Path]:
    """Plot per-schedule burst vs other accuracy charts."""
    paths: list[Path] = []
    for schedule in bundle["config"]["schedules"]:
        schedule_data = bundle["training"][schedule]
        steps = np.array(schedule_data[ACC_BURST]["steps"], dtype=float)
        pre_steps = schedule_data["pre_steps"]
        burst_end = pre_steps + schedule_data["burst_steps"]

        fig, ax = plt.subplots(figsize=figsize("half"))
        for metric, color, label in (
            (ACC_OTHER, COLOR_OTHER, "Other"),
            (ACC_BURST, COLOR_SPECIAL, "Special"),
        ):
            mean = np.array(schedule_data[metric]["mean"], dtype=float)
            ci = np.array(schedule_data[metric]["ci"], dtype=float)
            ax.plot(steps, mean, color=color, label=label)
            ax.fill_between(steps, mean - ci, mean + ci, color=color, alpha=0.14)

        if pre_steps > 0:
            ax.axvline(pre_steps, color="black", ls="--", lw=1.2, alpha=0.65)
        ax.axvline(burst_end, color="black", ls="--", lw=1.2, alpha=0.65)
        ax.set_ylim(-0.05, 1.05)
        style_axes(ax, "Step", "Accuracy")
        ax.legend(loc="best")
        path = save_chart(fig, out_dir / f"per_sched_{schedule.upper()}_accuracy.pdf")
        write_aliases(path, [out_dir / f"per_schedule_{schedule}.pdf"])
        paths.append(path)
    return paths


def fmt_ci(metric: dict, digits: int = 3) -> str:
    """Format a mean ± CI string from a metric dict."""
    mean = metric["mean"]
    ci = metric["ci"]
    if digits == 0:
        return f"{mean:.0f} ± {ci:.0f}"
    return f"{mean:.{digits}f} ± {ci:.{digits}f}"


def save_chart(fig: Figure, path: Path) -> Path:
    """Save a figure and return its path."""
    path = path.with_suffix(".pdf")
    save_figure(fig, path)
    return path


def overlay_aliases(filename: str) -> list[Path]:
    """Return shorthand alias paths for a given overlay filename."""
    if f"_{ACC_BURST.upper()}_" in filename:
        return [Path("overlay_burst.pdf")]
    if f"_{ACC_OTHER.upper()}_" in filename:
        return [Path("overlay_other.pdf")]
    if "_LOSS_" in filename:
        return [Path("overlay_loss.pdf")]
    return []


def write_aliases(source: Path, aliases: list[Path]) -> None:
    """Copy source file to each alias path."""
    for alias in aliases:
        target = source.parent / alias.name if not alias.is_absolute() else alias
        shutil.copyfile(source, target)


def max_total_steps(bundle: dict) -> int:
    """Return the maximum total steps across all schedules."""
    max_total = 0
    for schedule in bundle["config"]["schedules"]:
        schedule_data = bundle["training"][schedule]
        total = (
            schedule_data["pre_steps"]
            + schedule_data["burst_steps"]
            + schedule_data["reversion_steps"]
        )
        max_total = max(max_total, total)
    return max_total


def max_burst_steps(bundle: dict) -> int:
    """Return the maximum burst end step across all schedules."""
    max_burst = 0
    for schedule in bundle["config"]["schedules"]:
        schedule_data = bundle["training"][schedule]
        max_burst = max(max_burst, schedule_data["pre_steps"] + schedule_data["burst_steps"])
    return max_burst


def max_grad_burst_end(bundle: dict) -> int:
    """Return the maximum gradient-local burst end step across schedules."""
    gradients = bundle.get("gradients", {})
    return max((g["burst_steps"] for g in gradients.values()), default=0)


def max_grad_total_steps(bundle: dict) -> int:
    """Return the maximum gradient step across schedules."""
    gradients = bundle.get("gradients", {})
    return max((int(g["steps"][-1]) for g in gradients.values() if g["steps"]), default=0)


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


# ---------------------------------------------------------------------------
# Extended AUC bars
# ---------------------------------------------------------------------------


def plot_extended_auc_bars(bundle: dict, out_dir: Path) -> Path:
    """Plot reversion AUC bars for burst-acc, loss-burst, and other-acc."""
    schedules = bundle["config"]["schedules"]
    summary = bundle["summary"]["by_schedule"]

    fig, axes = plt.subplots(1, 3, figsize=figsize("full"), sharey=False)
    xs = np.arange(len(schedules))
    colors = [SCHED_COLORS[s] for s in schedules]
    labels = [SCHED_DISPLAY.get(s, s) for s in schedules]

    for ax, key, title in zip(
        axes,
        ("reversion_auc", "reversion_auc_loss_burst", "reversion_auc_acc_other"),
        ("Burst Acc AUC", "Burst Loss AUC", "Other Acc AUC"),
        strict=True,
    ):
        means = [summary[s][key]["mean"] for s in schedules]
        cis = [summary[s][key]["ci"] for s in schedules]
        ax.bar(xs, means, yerr=cis, color=colors, edgecolor="black", lw=0.7, capsize=4)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        style_axes(ax, "", title)

    return save_chart(fig, out_dir / "extended_reversion_auc_bars.pdf")


# ---------------------------------------------------------------------------
# Per-layer heatmap + line chart helpers
# ---------------------------------------------------------------------------


def build_layer_grid(
    layer_names: list[str], steps: list[float], metric_dict: dict[str, dict]
) -> np.ndarray:
    """Build a (n_layers, n_steps) grid from metric_dict means."""
    n_layers = len(layer_names)
    n_steps = len(steps)
    grid = np.full((n_layers, n_steps), np.nan)
    for li, ln in enumerate(layer_names):
        vals = metric_dict[ln]["mean"]
        grid[li, : len(vals)] = vals
    return grid


def plot_layer_heatmap(  # noqa: PLR0913
    layer_names: list[str],
    steps: list[float],
    metric_dict: dict[str, dict],
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
    metric_dict: dict[str, dict],
    out_path: Path,
    ylabel: str,
) -> Path:
    """Render a per-layer line chart."""
    n_layers = len(layer_names)
    fig, ax = plt.subplots(figsize=figsize("half"))
    cmap_obj = plt.get_cmap(LAYER_LINE_CMAP)
    for li, ln in enumerate(layer_names):
        mean = np.array(metric_dict[ln]["mean"], dtype=float)
        ax.plot(steps[: len(mean)], mean, lw=1.4, label=ln, color=cmap_obj(li / max(n_layers, 1)))
    style_axes(ax, "Step", ylabel)
    ax.legend(loc="best", ncol=3, fontsize=5)
    return save_chart(fig, out_path)


# ---------------------------------------------------------------------------
# Per-layer cosine similarity
# ---------------------------------------------------------------------------


def plot_per_layer_cossim(bundle: dict, out_dir: Path) -> list[Path]:
    """Plot per-layer gradient cosine similarity heatmaps and line charts."""
    plg = bundle["per_layer_gradients"]
    paths: list[Path] = []
    for schedule in bundle["config"]["schedules"]:
        data = plg[schedule]
        ln, st, cos = data["layer_names"], data["steps"], data["cosine"]
        paths.append(
            plot_layer_heatmap(
                ln,
                st,
                cos,
                out_dir / f"per_layer_cossim_{schedule}_heatmap.pdf",
                "Cosine Similarity",
                center_zero=True,
            )
        )
        paths.append(
            plot_layer_lines(
                ln,
                st,
                cos,
                out_dir / f"per_layer_cossim_{schedule}_lines.pdf",
                "Cosine Similarity",
            )
        )
    return paths


# ---------------------------------------------------------------------------
# Per-layer gradient norms
# ---------------------------------------------------------------------------


def plot_per_layer_grad_norm(bundle: dict, out_dir: Path) -> list[Path]:
    """Plot per-layer gradient norm heatmaps and line charts."""
    plg = bundle["per_layer_gradients"]
    paths: list[Path] = []
    for schedule in bundle["config"]["schedules"]:
        data = plg[schedule]
        ln, st = data["layer_names"], data["steps"]
        for prefix, metric_key in (("burst", "burst_norm"), ("other", "other_norm")):
            md = data[metric_key]
            paths.append(
                plot_layer_heatmap(
                    ln,
                    st,
                    md,
                    out_dir / f"per_layer_grad_norm_{prefix}_{schedule}_heatmap.pdf",
                    f"Grad Norm ({prefix})",
                    cmap=DRIFT_CMAP,
                )
            )
            paths.append(
                plot_layer_lines(
                    ln,
                    st,
                    md,
                    out_dir / f"per_layer_grad_norm_{prefix}_{schedule}_lines.pdf",
                    f"Grad Norm ({prefix})",
                )
            )
    return paths


# ---------------------------------------------------------------------------
# Per-layer norm x cosine
# ---------------------------------------------------------------------------


def plot_per_layer_norm_x_cossim(bundle: dict, out_dir: Path) -> list[Path]:
    """Plot per-layer norm*cosine heatmaps and line charts."""
    plg = bundle["per_layer_gradients"]
    paths: list[Path] = []
    for schedule in bundle["config"]["schedules"]:
        data = plg[schedule]
        ln, st = data["layer_names"], data["steps"]
        md = data["norm_x_cosine"]
        paths.append(
            plot_layer_heatmap(
                ln,
                st,
                md,
                out_dir / f"per_layer_norm_x_cossim_{schedule}_heatmap.pdf",
                "Norm x Cosine",
                center_zero=True,
            )
        )
        paths.append(
            plot_layer_lines(
                ln,
                st,
                md,
                out_dir / f"per_layer_norm_x_cossim_{schedule}_lines.pdf",
                "Norm x Cosine",
            )
        )
    return paths


# ---------------------------------------------------------------------------
# Weight drift
# ---------------------------------------------------------------------------


def plot_weight_drift(bundle: dict, out_dir: Path) -> list[Path]:
    """Plot per-layer weight drift heatmaps and line charts."""
    wd = bundle["weight_drift"]
    paths: list[Path] = []
    for schedule in bundle["config"]["schedules"]:
        data = wd[schedule]
        ln, st = data["layer_names"], data["steps"]
        paths.append(
            plot_layer_heatmap(
                ln,
                st,
                data["cumulative"],
                out_dir / f"weight_drift_{schedule}_heatmap.pdf",
                "Weight Drift (Frobenius)",
                cmap=DRIFT_CMAP,
            )
        )
        paths.append(
            plot_layer_lines(
                ln,
                st,
                data["cumulative"],
                out_dir / f"weight_drift_{schedule}_lines.pdf",
                "Weight Drift (Frobenius)",
            )
        )
    return paths
