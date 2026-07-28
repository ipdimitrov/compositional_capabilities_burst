"""2-panel curvature/gradient summary figure.

Left panel collapses the "lambda_dir >> lambda_avg" claim into a single
number per concentration -- the ratio lambda_dir / lambda_avg on a log
axis -- surfacing the secondary story that the ratio itself grows with
concentration.  Right panel shows the squared pre-train gradient magnitude.
Panel descriptions live above each axis; math symbols are the y-labels.
"""

from __future__ import annotations

import csv
import math
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_style import (
    apply_paper_style,
    ordered_colors,
    save_figure,
    style_axes,
)

import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullLocator

CURV_FIG_WIDTH = 6.0
CURV_FIG_HEIGHT = 2.6

apply_paper_style()
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 13,
        "axes.labelsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    },
)

DEFAULT_DATA: dict[str, dict[float, float]] = {
    "lambda_dir": {1.0: 1100.0, 0.90: 220.0, 0.75: 110.0, 0.50: 75.0, 0.25: 64.0},
    "lambda_avg": {1.0: 0.0037, 0.90: 0.0020, 0.75: 0.0017, 0.50: 0.0016, 0.25: 0.0017},
    "g_norm_sq": {1.0: 960.0, 0.90: 4.0, 0.75: 2.6, 0.50: 2.3, 0.25: 2.1},
}


def fmt_num(v: float) -> str:
    """Human-readable number, no scientific notation, no trailing zeros."""
    if v >= 1000:
        return f"{int(round(v)):,}"
    if v >= 100:
        return f"{v:.0f}"
    if v >= 1:
        return f"{v:.1f}".rstrip("0").rstrip(".")
    if v >= 0.1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def fmt_sci(v: float) -> str:
    """Math-mode scientific: 38000 -> $3.8\\times 10^{4}$."""
    exp = int(math.floor(math.log10(v)))
    mant = v / 10**exp
    return rf"${mant:.1f}\times 10^{{{exp}}}$"


def style_log_y_sci(ax: plt.Axes) -> None:
    """Log y-axis, pure powers of 10 rendered as 10^k."""
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(LogFormatterSciNotation())


PANEL_SPECS: list[tuple[str, str, str, Callable[[float], str]]] = [
    (
        "lambda_ratio",
        "Directional-to-average curvature",
        r"$\lambda_{\mathrm{dir}} / \lambda_{\mathrm{avg}}$",
        fmt_num,
    ),
    (
        "g_norm_sq",
        "Pre-train gradient magnitude",
        r"$\|g_{\mathrm{pre}}\|_{2}^{2}$",
        fmt_num,
    ),
]


def load_curvature_csv(path: str | Path) -> dict[str, dict[float, float]]:
    """Load curvature summary CSV into the chart's expected data shape."""
    rows = list(csv.DictReader(Path(path).open()))
    assert rows

    data: dict[str, dict[float, float]] = {
        "lambda_dir": {},
        "lambda_avg": {},
        "g_norm_sq": {},
    }
    for row in rows:
        concentration = float(row["burst"])
        data["lambda_dir"][concentration] = float(row["lam_dir"])
        data["lambda_avg"][concentration] = float(row["lam_iso"])
        data["g_norm_sq"][concentration] = float(row["g_pre_sq"])
    return data


def draw_panel(
    ax: plt.Axes,
    values: dict[float, float],
    colors: dict[float, str],
    title: str,
    ylabel: str,
    bar_fmt: Callable[[float], str],
) -> None:
    """Draw one log-scale curvature/gradient panel."""
    concentrations = sorted(values)
    xs = list(range(len(concentrations)))
    ys = [values[c] for c in concentrations]
    bar_colors = [colors[c] for c in concentrations]

    bars = ax.bar(xs, ys, color=bar_colors, edgecolor="none", width=0.62)

    style_log_y_sci(ax)
    ax.set_ylim(bottom=min(ys) * 0.5, top=max(ys) * 3.5)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{c}" for c in concentrations])
    ax.set_title(title, pad=8)
    style_axes(ax, "", ylabel)
    ax.grid(axis="x", visible=False)

    for rect, y in zip(bars, ys, strict=True):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            y * 1.25,
            bar_fmt(y),
            ha="center", va="bottom", fontsize=11,
        )


def plot_curvature_mechanism(
    data: dict[str, dict[float, float]],
    out_path: str | Path,
) -> Path:
    """Render the 2-panel figure (PDF + SVG + PNG) and return the PDF path."""
    ratio = {
        c: data["lambda_dir"][c] / data["lambda_avg"][c]
        for c in data["lambda_dir"]
    }
    enriched = {**data, "lambda_ratio": ratio}
    colors = ordered_colors(sorted(ratio))

    fig, axes = plt.subplots(1, 2, figsize=(CURV_FIG_WIDTH, CURV_FIG_HEIGHT))
    for ax, (key, title, ylabel, fmt) in zip(axes, PANEL_SPECS, strict=True):
        draw_panel(ax, enriched[key], colors, title, ylabel, fmt)
    fig.supxlabel("Concentration (%)", fontsize=14)

    out_path = Path(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    save_figure(fig, out_path)
    return out_path.with_suffix(".pdf")


def draw_line_band_panel(
    ax: plt.Axes,
    lambda_dir: dict[float, float],
    lambda_avg: dict[float, float],
    colors: dict[float, str],
) -> None:
    """Left panel: λ_dir (circles) and λ_avg (squares, dimmed) on shared log axis."""
    concentrations = sorted(lambda_dir)
    xs = concentrations  # use raw 0–1 values as x
    dir_ys = [lambda_dir[c] for c in concentrations]
    avg_ys = [lambda_avg[c] for c in concentrations]

    marker_colors = [colors.get(c, "gray") for c in concentrations]

    # λ_avg: dimmed line + colored square markers with value labels
    ax.plot(xs, avg_ys, "-", color="gray", lw=1, alpha=0.5, zorder=2)
    for x, y, mc in zip(xs, avg_ys, marker_colors):
        ax.scatter(x, y, color=mc, marker="s", s=40, zorder=4,
                   edgecolors="black", linewidth=0.5, alpha=0.6)
        exp = int(math.floor(math.log10(y)))
        mant = y / 10**exp
        ax.text(x, y * 0.4, f"{mant:.0f}e{exp}", ha="center", va="top",
                fontsize=9, color="gray")
    # λ_dir: black line + colored circle markers with value labels
    ax.plot(xs, dir_ys, "-", color="black", lw=2, zorder=3)
    for x, y, mc in zip(xs, dir_ys, marker_colors):
        ax.scatter(x, y, color=mc, marker="o", s=50, zorder=5,
                   edgecolors="black", linewidth=0.5)
        ax.text(x, y * 1.3, f"{y:.0f}", ha="center", va="bottom",
                fontsize=10, color=mc)

    # Legend entries: λ_dir first (circle), then λ_avg (square, dimmed)
    ax.scatter([], [], color="gray", marker="o", s=50, edgecolors="black",
               linewidth=0.5, label=r"$\lambda_{\mathrm{dir}}$")
    ax.scatter([], [], color="lightgray", marker="s", s=40, edgecolors="gray",
               linewidth=0.5, alpha=0.6, label=r"$\lambda_{\mathrm{avg}}$")

    style_log_y_sci(ax)
    ax.set_ylim(bottom=min(avg_ys) * 0.1, top=max(dir_ys) * 3.0)
    ax.set_xticks(concentrations)
    ax.set_xticklabels([str(c) for c in concentrations], rotation=45, ha="right")
    style_axes(ax, "", r"$\lambda$")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="center left", fontsize=11, handlelength=1.4)


def draw_line_panel(
    ax: plt.Axes,
    values: dict[float, float],
    colors: dict[float, str],
    ylabel: str,
) -> None:
    """Draw a single-series line+marker panel (colored circles, black line)."""
    concentrations = sorted(values)
    xs = concentrations  # raw 0–1 values
    ys = [values[c] for c in concentrations]
    marker_colors = [colors.get(c, "gray") for c in concentrations]

    ax.plot(xs, ys, "-", color="black", lw=2, zorder=3)
    for x, y, mc in zip(xs, ys, marker_colors):
        ax.scatter(x, y, color=mc, marker="o", s=50, zorder=5,
                   edgecolors="black", linewidth=0.5)
        ax.text(x, y * 1.3, fmt_num(y), ha="center", va="bottom",
                fontsize=10, color=mc)

    style_log_y_sci(ax)
    ax.set_ylim(bottom=min(ys) * 0.4, top=max(ys) * 3.5)
    ax.set_xticks(concentrations)
    ax.set_xticklabels([str(c) for c in concentrations], rotation=45, ha="right")
    style_axes(ax, "", ylabel)
    ax.grid(axis="x", visible=False)


def plot_curvature_mechanism_grouped(
    data: dict[str, dict[float, float]],
    out_path: str | Path,
) -> Path:
    """2-panel: left = λ_dir + λ_avg, right = ||δW||."""
    point_colors = ordered_colors(sorted(data["g_norm_sq"]))

    fig, axes = plt.subplots(1, 2, figsize=(CURV_FIG_WIDTH, CURV_FIG_HEIGHT))
    draw_line_band_panel(
        axes[0],
        data["lambda_dir"],
        data["lambda_avg"],
        point_colors,
    )
    draw_line_panel(
        axes[1], data["g_norm_sq"], point_colors,
        r"$\|\Delta \theta \|$",
    )
    fig.supxlabel("Concentration $c$", fontsize=14)

    out_path = Path(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    save_figure(fig, out_path)
    return out_path.with_suffix(".pdf")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    csv_path = (
        root
        / "src/pythia/results"
        / "20260424_195039_deep_70m_biomedical"
        / "figures"
        / "curvature_mechanism.csv"
    )
    data = load_curvature_csv(csv_path)
    out = plot_curvature_mechanism(data, root / "curvature_mechanism_styled.pdf")
    print(f"wrote {out}")  # noqa: T201
    out_v2 = plot_curvature_mechanism_grouped(
        data, root / "curvature_mechanism_styled_v2.pdf",
    )
    print(f"wrote {out_v2}")  # noqa: T201
