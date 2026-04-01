from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

FIG_DPI = 220
SERIF_STACK = ["STIXGeneral", "Times New Roman", "DejaVu Serif"]


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": SERIF_STACK,
            "mathtext.fontset": "stix",
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 15,
            "axes.linewidth": 1.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "legend.frameon": False,
            "grid.alpha": 0.16,
            "grid.linewidth": 0.7,
            "savefig.dpi": FIG_DPI,
            "figure.dpi": FIG_DPI,
        }
    )


def style_axes(ax, xlabel: str, ylabel: str, title: str = "") -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=10)
    ax.grid(True)


def save_figure(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
