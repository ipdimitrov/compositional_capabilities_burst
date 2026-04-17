"""Matplotlib style helpers for publication-quality figures."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

HALF_COL_WIDTH = 3.25
FULL_COL_WIDTH = 6.75


def apply_paper_style() -> None:
    """Set rcParams for ICML paper-ready plots."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "mathtext.rm": "Times",
            "mathtext.it": "Times:italic",
            "mathtext.bf": "Times:bold",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "axes.linewidth": 0.5,
            "lines.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.16,
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.pad_inches": 0.015,
            "figure.figsize": (HALF_COL_WIDTH, HALF_COL_WIDTH / 1.618),
            "figure.constrained_layout.use": True,
        }
    )


def style_axes(ax: Axes, xlabel: str, ylabel: str) -> None:
    """Apply labels and grid to an axes."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(visible=True)


def figsize(cols: str = "half", nrows: int = 1) -> tuple[float, float]:
    """Return (width, height) for a figure with golden-ratio rows.

    cols="half" / "full": standard golden-ratio height scaled by nrows.
    cols="flat": half-col width with a short height (width / 3.2), good for
    bar charts and overlay plots with lots of empty vertical space.
    cols="half_side": half-col width with a short height suitable for two
    side-by-side panels inside a single half-column slot.
    """
    if cols == "flat":
        return (HALF_COL_WIDTH, HALF_COL_WIDTH / 3.2)
    if cols == "half_side":
        return (HALF_COL_WIDTH, HALF_COL_WIDTH / 2.4)
    w = FULL_COL_WIDTH if cols == "full" else HALF_COL_WIDTH
    return (w, (w / 1.618) * nrows)


def save_figure(fig: Figure, path: str | Path) -> None:
    """Save fig as PDF, then close it."""
    path = Path(path).with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
