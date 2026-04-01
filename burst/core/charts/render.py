from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

from burst.config import SCHED_COLORS, SCHED_DISPLAY, reversion_life_label
from burst.core.charts.style import apply_paper_style, save_figure, style_axes


def render_core_charts(bundle: dict, out_dir: str | Path) -> list[Path]:
    """Render all core analysis charts to out_dir."""
    apply_paper_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        _plot_schedule_bars(bundle, out_dir),
        _plot_lr_curves(bundle, out_dir),
        _plot_overlay(
            bundle,
            out_dir,
            "acc_burst",
            "Special Accuracy",
            "Special Class Accuracy",
            "overlay_all_acc_burst.png",
        ),
        _plot_overlay(
            bundle,
            out_dir,
            "acc_other",
            "Other Accuracy",
            "Other Class Accuracy",
            "overlay_all_acc_other.png",
        ),
        _plot_overlay(bundle, out_dir, "loss", "Loss", "Training Loss", "overlay_all_loss.png"),
        _plot_auc_bars(bundle, out_dir),
        _plot_summary_table(bundle, out_dir),
        _plot_reversion_zoom(bundle, out_dir),
        _plot_grad_cosine(bundle, out_dir),
        _plot_grad_cosine_per_schedule(bundle, out_dir),
        _plot_grad_norms(bundle, out_dir),
        _plot_grad_norm_x_cosine(bundle, out_dir),
        _plot_representation_drift(bundle, out_dir),
    ]
    paths.extend(_plot_per_schedule(bundle, out_dir))
    return [path for path in paths if path is not None]


def _plot_schedule_bars(bundle: dict, out_dir: Path) -> Path:
    """Plot burst fraction over time for each schedule."""
    schedules = bundle["config"]["schedules"]
    bars = bundle["schedule_bars"]
    fig, axes = plt.subplots(
        len(schedules), 1, figsize=(12, max(3.0, 1.6 * len(schedules))), sharex=True
    )
    if len(schedules) == 1:
        axes = [axes]

    max_len = max(len(bars[schedule]["fractions"]) for schedule in schedules)
    for ax, schedule in zip(axes, schedules, strict=False):
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
    axes[0].set_title("Training Schedule")
    return _save(fig, out_dir / "schedule_bars.png")


def _plot_lr_curves(bundle: dict, out_dir: Path) -> Path:
    """Plot learning rate schedules for all schedules."""
    fig, ax = plt.subplots(figsize=(11, 5))
    for schedule in bundle["config"]["schedules"]:
        curve = bundle["lr_curves"][schedule]
        ax.plot(
            curve["steps"],
            curve["lr"],
            color=SCHED_COLORS[schedule],
            lw=2.0,
            label=SCHED_DISPLAY.get(schedule, schedule),
        )
    style_axes(ax, "Step", "Learning Rate", "Learning Rate")
    ax.legend(loc="upper right", ncol=2)
    return _save(fig, out_dir / "lr_schedule.png")


def _plot_overlay(  # noqa: PLR0913
    bundle: dict, out_dir: Path, metric: str, ylabel: str, title: str, filename: str
) -> Path:
    """Plot a single metric overlaid across all schedules."""
    fig, ax = plt.subplots(figsize=(11, 6))
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
            lw=2.2,
            label=SCHED_DISPLAY.get(schedule, schedule),
        )
        ax.fill_between(steps, mean - ci, mean + ci, color=SCHED_COLORS[schedule], alpha=0.12)

    if metric != "loss":
        ax.set_ylim(-0.05, 1.05)
    _annotate_global_phase_boundaries(ax, max_burst_end, _max_total_steps(bundle))
    style_axes(ax, "Step", ylabel, title)
    ax.legend(loc="best", ncol=2)
    path = _save(fig, out_dir / filename)
    _write_aliases(path, _overlay_aliases(filename))
    return path


def _plot_auc_bars(bundle: dict, out_dir: Path) -> Path:
    """Plot reversion AUC bar chart across schedules."""
    schedules = bundle["config"]["schedules"]
    summary = bundle["summary"]["by_schedule"]
    means = [summary[schedule]["reversion_auc"]["mean"] for schedule in schedules]
    cis = [summary[schedule]["reversion_auc"]["ci"] for schedule in schedules]

    fig, ax = plt.subplots(figsize=(10, 6))
    xs = np.arange(len(schedules))
    colors = [SCHED_COLORS[schedule] for schedule in schedules]
    ax.bar(xs, means, yerr=cis, color=colors, edgecolor="black", lw=0.7, capsize=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [SCHED_DISPLAY.get(schedule, schedule) for schedule in schedules], rotation=25, ha="right"
    )
    style_axes(ax, "", "AUC", "Reversion AUC")
    return _save(fig, out_dir / "auc_bars.png")


def _plot_summary_table(bundle: dict, out_dir: Path) -> Path:
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
            _fmt_ci(schedule_summary["peak_burst"]),
            _fmt_ci(schedule_summary["reversion_auc"], digits=0),
            _fmt_ci(schedule_summary["other_end"]),
        ]
        row.extend(
            _fmt_ci(schedule_summary["life"][f"life_{int(threshold * 100)}"], digits=0)
            for threshold in thresholds
        )
        rows.append(row)

    fig_w = max(10, 3 + 1.45 * len(headers))
    fig, ax = plt.subplots(figsize=(fig_w, 3.8))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(auto=False)
    table.set_fontsize(10)
    table.scale(1.0, 1.55)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#EFEFEF")
        elif col == 0:
            schedule = schedules[row - 1]
            cell.set_facecolor(SCHED_COLORS[schedule] + "22")
            cell.set_text_props(fontweight="bold")
        cell.set_edgecolor("#CCCCCC")
    ax.set_title("Summary Table", pad=10)
    return _save(fig, out_dir / "summary_table.png")


def _plot_reversion_zoom(bundle: dict, out_dir: Path) -> Path:
    """Plot burst accuracy during the reversion phase only."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for schedule in bundle["config"]["schedules"]:
        schedule_data = bundle["training"][schedule]
        steps = np.array(schedule_data["acc_burst"]["steps"], dtype=float)
        mean = np.array(schedule_data["acc_burst"]["mean"], dtype=float)
        ci = np.array(schedule_data["acc_burst"]["ci"], dtype=float)
        burst_end = schedule_data["pre_steps"] + schedule_data["burst_steps"]
        mask = steps >= burst_end
        local_steps = steps[mask] - burst_end
        ax.plot(
            local_steps,
            mean[mask],
            color=SCHED_COLORS[schedule],
            lw=2.2,
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
    style_axes(ax, "Reversion Step", "Special Accuracy", "Forgetting Speed")
    ax.legend(loc="upper right", ncol=2)
    return _save(fig, out_dir / "reversion_zoom.png")


def _plot_grad_cosine(bundle: dict, out_dir: Path) -> Path | None:
    """Plot gradient cosine similarity overlay across schedules."""
    gradients = bundle["gradients"]
    if not gradients:
        return None
    fig, ax = plt.subplots(figsize=(11, 6))
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
            lw=2.2,
            label=SCHED_DISPLAY.get(schedule, schedule),
        )
        ax.fill_between(steps, mean - ci, mean + ci, color=SCHED_COLORS[schedule], alpha=0.12)
    ax.axhline(0.0, color="#666666", ls=":", lw=1.0)
    _annotate_global_phase_boundaries(ax, _max_burst_steps(bundle), _max_total_steps(bundle))
    style_axes(ax, "Step", "Cosine", "Grad Cosine")
    ax.legend(loc="best", ncol=2)
    path = _save(fig, out_dir / "grad_cosine_burst_vs_other.png")
    _write_aliases(path, [out_dir / "grad_cosine.png"])
    return path


def _plot_grad_cosine_per_schedule(bundle: dict, out_dir: Path) -> Path | None:
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
        burst_end = (
            bundle["training"][schedule]["pre_steps"] + bundle["training"][schedule]["burst_steps"]
        )

        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.plot(steps, mean, color=SCHED_COLORS[schedule], lw=2.4)
        ax.fill_between(steps, mean - ci, mean + ci, color=SCHED_COLORS[schedule], alpha=0.14)
        ax.axhline(0.0, color="#666666", ls=":", lw=1.0)
        _annotate_global_phase_boundaries(ax, burst_end, steps[-1])
        style_axes(ax, "Step", "Cosine", f"{SCHED_DISPLAY.get(schedule, schedule)} Grad Cosine")
        path = _save(fig, out_dir / f"grad_cosine_per_schedule_{schedule}.png")
        _write_aliases(path, [out_dir / f"grad_cosine_{schedule}.png"])
        if first_path is None:
            first_path = path
    return first_path


def _plot_grad_norms(bundle: dict, out_dir: Path) -> Path | None:
    """Plot burst and other gradient L2 norms side by side."""
    gradients = bundle["gradients"]
    if not gradients:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
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
            lw=2.0,
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
            lw=2.0,
            label=SCHED_DISPLAY.get(schedule, schedule),
        )
        axes[1].fill_between(
            steps,
            other_mean - other_ci,
            other_mean + other_ci,
            color=SCHED_COLORS[schedule],
            alpha=0.12,
        )

    _annotate_global_phase_boundaries(axes[0], _max_burst_steps(bundle), _max_total_steps(bundle))
    _annotate_global_phase_boundaries(axes[1], _max_burst_steps(bundle), _max_total_steps(bundle))
    style_axes(axes[0], "Step", "L2 Norm", "Burst Grad Norm")
    style_axes(axes[1], "Step", "L2 Norm", "Other Grad Norm")
    axes[1].legend(loc="best", ncol=1)
    path = _save(fig, out_dir / "grad_norm_l2.png")
    _write_aliases(path, [out_dir / "grad_norms.png"])
    return path


def _plot_grad_norm_x_cosine(bundle: dict, out_dir: Path) -> Path | None:
    """Plot signed dot product and interference power charts."""
    gradients = bundle["gradients"]
    if not gradients:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
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
            lw=2.0,
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
            lw=2.0,
            label=SCHED_DISPLAY.get(schedule, schedule),
        )
        axes[1].fill_between(
            steps,
            power_mean - power_ci,
            power_mean + power_ci,
            color=SCHED_COLORS[schedule],
            alpha=0.12,
        )

    axes[0].axhline(0.0, color="#666666", ls=":", lw=1.0)
    _annotate_global_phase_boundaries(axes[0], _max_burst_steps(bundle), _max_total_steps(bundle))
    _annotate_global_phase_boundaries(axes[1], _max_burst_steps(bundle), _max_total_steps(bundle))
    style_axes(axes[0], "Step", "Signed Dot", "Grad Norm x Cosine")
    style_axes(axes[1], "Step", "Power", "Interference Power")
    axes[1].legend(loc="best", ncol=1)
    return _save(fig, out_dir / "grad_norm_x_cosine.png")


def _plot_representation_drift(bundle: dict, out_dir: Path) -> Path | None:
    """Plot centroid drift and other-shift norm across schedules."""
    representation = bundle.get("representation", {})
    by_schedule = representation.get("by_schedule", {})
    if not by_schedule:
        return None

    schedules = [schedule for schedule in bundle["config"]["schedules"] if schedule in by_schedule]
    if not schedules:
        return None

    xs = np.arange(len(schedules))
    labels = [_sched_pct_label(schedule) for schedule in schedules]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
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

    axes[0].plot(xs, proj_mean, color="#5E3C99", lw=2.4, marker="o", ms=6)
    axes[0].fill_between(xs, proj_mean - proj_ci, proj_mean + proj_ci, color="#5E3C99", alpha=0.12)
    axes[0].axhline(0.0, color="#666666", ls=":", lw=1.0)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(labels)
    style_axes(axes[0], "Concentration %", "Projection", "Centroid Drift (Late Layers)")

    axes[1].plot(xs, shift_mean, color="#1B9E77", lw=2.4, marker="o", ms=6)
    axes[1].fill_between(
        xs, shift_mean - shift_ci, shift_mean + shift_ci, color="#1B9E77", alpha=0.12
    )
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels)
    style_axes(axes[1], "Concentration %", "Normalized Shift", "Other Shift Norm (Late Layers)")

    path = _save(fig, out_dir / "representation_drift_summary.png")
    _write_aliases(path, [out_dir / "rep_drift_summary.png"])
    return path


def _plot_per_schedule(bundle: dict, out_dir: Path) -> list[Path]:
    """Plot per-schedule burst vs other accuracy charts."""
    paths: list[Path] = []
    for schedule in bundle["config"]["schedules"]:
        schedule_data = bundle["training"][schedule]
        steps = np.array(schedule_data["acc_burst"]["steps"], dtype=float)
        pre_steps = schedule_data["pre_steps"]
        burst_end = pre_steps + schedule_data["burst_steps"]

        fig, ax = plt.subplots(figsize=(11, 5.5))
        for metric, color, label in (
            ("acc_other", "#1565C0", "Other"),
            ("acc_burst", "#D32F2F", "Special"),
        ):
            mean = np.array(schedule_data[metric]["mean"], dtype=float)
            ci = np.array(schedule_data[metric]["ci"], dtype=float)
            ax.plot(steps, mean, color=color, lw=2.3, label=label)
            ax.fill_between(steps, mean - ci, mean + ci, color=color, alpha=0.14)

        if pre_steps > 0:
            ax.axvline(pre_steps, color="black", ls="--", lw=1.2, alpha=0.65)
        ax.axvline(burst_end, color="black", ls="--", lw=1.2, alpha=0.65)
        ax.set_ylim(-0.05, 1.05)
        style_axes(ax, "Step", "Accuracy", SCHED_DISPLAY.get(schedule, schedule))
        ax.legend(loc="best")
        path = _save(fig, out_dir / f"per_sched_{schedule}.png")
        _write_aliases(path, [out_dir / f"per_schedule_{schedule}.png"])
        paths.append(path)
    return paths


def _fmt_ci(metric: dict, digits: int = 3) -> str:
    """Format a mean ± CI string from a metric dict."""
    mean = metric["mean"]
    ci = metric["ci"]
    if digits == 0:
        return f"{mean:.0f} ± {ci:.0f}"
    return f"{mean:.{digits}f} ± {ci:.{digits}f}"


def _save(fig: Any, path: Path) -> Path:
    """Save a figure and return its path."""
    save_figure(fig, path)
    return path


def _overlay_aliases(filename: str) -> list[Path]:
    """Return short alias paths for an overlay chart filename."""
    if filename == "overlay_all_acc_burst.png":
        return [Path(filename.replace("overlay_all_acc_burst.png", "overlay_burst.png"))]
    if filename == "overlay_all_acc_other.png":
        return [Path(filename.replace("overlay_all_acc_other.png", "overlay_other.png"))]
    if filename == "overlay_all_loss.png":
        return [Path(filename.replace("overlay_all_loss.png", "overlay_loss.png"))]
    return []


def _write_aliases(source: Path, aliases: list[Path]) -> None:
    """Copy source file to each alias path."""
    for alias in aliases:
        target = source.parent / alias.name if not alias.is_absolute() else alias
        shutil.copyfile(source, target)


def _max_total_steps(bundle: dict) -> int:
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


def _max_burst_steps(bundle: dict) -> int:
    """Return the maximum burst end step across all schedules."""
    max_burst = 0
    for schedule in bundle["config"]["schedules"]:
        schedule_data = bundle["training"][schedule]
        max_burst = max(max_burst, schedule_data["pre_steps"] + schedule_data["burst_steps"])
    return max_burst


def _annotate_global_phase_boundaries(ax: Any, burst_end: float, total_steps: float) -> None:
    """Draw vertical phase boundary lines and labels on an axes."""
    ax.axvline(burst_end, color="black", ls="--", lw=1.15, alpha=0.6)
    ymax = ax.get_ylim()[1]
    ax.text(
        burst_end * 0.5, ymax * 0.97, "SPECIAL", ha="center", va="top", fontsize=10, color="gray"
    )
    ax.text(
        burst_end + max(total_steps - burst_end, 1) * 0.5,
        ymax * 0.97,
        "ALL-BUT-SPECIAL",
        ha="center",
        va="top",
        fontsize=10,
        color="gray",
    )


def _sched_pct_label(schedule: str) -> str:
    """Extract the percentage suffix from a schedule name."""
    return schedule.rsplit("_", maxsplit=1)[-1]
