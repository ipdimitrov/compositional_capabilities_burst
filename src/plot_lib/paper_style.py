"""Matplotlib style helpers for publication-quality figures.

Usage::

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[N] / "plot_lib"))
    from paper_style import (
        apply_paper_style, figsize, ordered_colors, outside_legend, save_figure,
        style_axes,
    )

Where N is the number of parent directories from the importing file up to ``src/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

HALF_COL_WIDTH = 4.8
FULL_COL_WIDTH = 9.6
COL_RATIO = 3

# Default palette: truncated `plasma` colormap — high concentration maps to
# amber (stop 0.80) and low concentration to deep purple (stop 0.05). Avoids
# the bright yellow tail of full plasma while staying perceptually uniform and
# colorblind-safe.
PAPER_CMAP_NAME = "paper_plasma_amber"
_PLASMA_LO = 0.05
_PLASMA_HI = 0.80
_paper_cmap = mcolors.LinearSegmentedColormap.from_list(
    PAPER_CMAP_NAME,
    plt.get_cmap("plasma")(np.linspace(_PLASMA_LO, _PLASMA_HI, 256)),
)
if PAPER_CMAP_NAME not in mpl.colormaps:
    mpl.colormaps.register(_paper_cmap)


def apply_paper_style() -> None:
    """Set rcParams for paper-ready plots."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times", "Times New Roman", "Liberation Serif", "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "mathtext.rm": "Times",
            "mathtext.it": "Times:italic",
            "mathtext.bf": "Times:bold",
            "font.size": 11,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.frameon": False,
            "axes.linewidth": 0.5,
            "lines.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.16,
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.pad_inches": 0.05,
            "figure.figsize": (HALF_COL_WIDTH, HALF_COL_WIDTH / COL_RATIO),
            "figure.constrained_layout.use": True,
        }
    )


def style_axes(ax: Axes, xlabel: str, ylabel: str) -> None:
    """Apply labels and grid to an axes."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(visible=True)


def figsize(cols: str = "half", nrows: int = 1) -> tuple[float, float]:
    """Return (width, height) for a figure with golden-ratio rows."""
    if cols == "flat":
        return (HALF_COL_WIDTH, HALF_COL_WIDTH / 4.0)
    if cols == "half_side":
        return (HALF_COL_WIDTH, HALF_COL_WIDTH / 3.0)
    w = FULL_COL_WIDTH if cols == "full" else HALF_COL_WIDTH
    return (w, (w / COL_RATIO) * nrows)


def outside_legend(ax: Axes, title: str | None = None) -> None:
    """Place the legend as a single right-side column outside the axes.

    Non-overlapping with plot content; saved width grows via
    ``bbox_inches='tight'``.  Use for all line charts with more than 2 series.
    """
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        ncol=1,
        handlelength=1.4,
        handletextpad=0.5,
        labelspacing=0.35,
        title=title,
    )


def save_figure(fig: Figure, path: str | Path) -> None:
    """Save fig as PDF + SVG, then close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".svg"):
        fig.savefig(path.with_suffix(ext), bbox_inches="tight")
    plt.close(fig)


def ordered_colors(
    values: list[float], cmap_name: str = PAPER_CMAP_NAME,
) -> dict[float, str]:
    """Map values to hex colors, evenly spaced by rank along the colormap.

    Rank-based (not value-based) so adjacent values stay visually distinct
    even when clustered (e.g. 1.0, 0.98, 0.95 all map to different stops
    instead of three near-identical ones).  Highest value -> top of cmap
    (amber), lowest -> bottom of cmap (deep purple).
    """
    cmap = plt.get_cmap(cmap_name)
    ordered = sorted(values)
    n = len(ordered)
    if n == 1:
        return {ordered[0]: mcolors.rgb2hex(cmap(0.9))}
    return {
        v: mcolors.rgb2hex(cmap(0.05 + (i / (n - 1)) * 0.90))
        for i, v in enumerate(ordered)
    }
