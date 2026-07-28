"""Shared plotting utilities for burst dashboards."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from pathlib import Path

    from plotly.graph_objs import Figure


def plotly_to_mpl_color(
    c: str | tuple[float, ...],
) -> str | tuple[float, ...]:
    """Map a Plotly rgb/rgba string to matplotlib RGBA floats, or pass through tuples."""
    if not isinstance(c, str):
        return c
    m = re.match(r"rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)", c)
    if m:
        r, g, b = int(m.group(1)) / 255, int(m.group(2)) / 255, int(m.group(3)) / 255
        a = float(m.group(4)) if m.group(4) else 1.0
        return (r, g, b, a)
    return c


def plotly_to_png_matplotlib(  # noqa: C901, PLR0912, PLR0915
    fig_plotly: Figure,
    path: str,
    width: int = 1200,
    height: int = 600,
) -> None:
    """Render a Plotly figure to PNG via matplotlib."""
    fig_data = fig_plotly.to_dict()
    traces = fig_data.get("data", [])
    layout = fig_data.get("layout", {})
    title = layout.get("title", {})
    title_text = title.get("text", "") if isinstance(title, dict) else str(title)
    title_text = re.sub(r"<[^>]+>", "", title_text).strip()

    dpi = 100
    mfig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

    for trace in traces:
        trace_type = trace.get("type", "scatter")
        x = trace.get("x", [])
        y = trace.get("y", [])
        name = trace.get("name", "")
        color = None
        line_info = trace.get("line", {})
        marker_info = trace.get("marker", {})
        if isinstance(line_info, dict) and "color" in line_info:
            color = plotly_to_mpl_color(line_info["color"])
        elif isinstance(marker_info, dict) and "color" in marker_info:
            mc = marker_info["color"]
            if isinstance(mc, str):
                color = plotly_to_mpl_color(mc)

        kwargs = {"label": name}
        if color and isinstance(color, (str, tuple)):
            kwargs["color"] = color

        if trace_type in ("scatter", "scattergl"):
            mode = trace.get("mode", "lines")
            if "lines" in mode:
                ax.plot(x, y, **kwargs)
            elif "markers" in mode:
                ax.scatter(x, y, s=30, zorder=5, **kwargs)
        elif trace_type == "bar":
            bar_colors = marker_info.get("color") if isinstance(marker_info, dict) else None
            if isinstance(bar_colors, list):
                kwargs.pop("color", None)
                kwargs["color"] = [plotly_to_mpl_color(c) for c in bar_colors[: len(x)]]
            ax.bar(x, y, alpha=0.8, **kwargs)
        elif trace_type == "heatmap":
            pass

    xaxis = layout.get("xaxis", {})
    yaxis = layout.get("yaxis", {})
    if isinstance(xaxis, dict):
        ax.set_xlabel(
            xaxis.get("title", {}).get("text", "") if isinstance(xaxis.get("title"), dict) else ""
        )
    if isinstance(yaxis, dict):
        ax.set_ylabel(
            yaxis.get("title", {}).get("text", "") if isinstance(yaxis.get("title"), dict) else ""
        )

    ax.set_title(title_text[:120], fontsize=10, wrap=True)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(handles[:15], labels[:15], fontsize=7, loc="best")
    ax.grid(visible=True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(mfig)


def save_png(
    fig: Figure,
    path: str,
    width: int = 1200,
    height: int = 600,
) -> None:
    """Save a Plotly figure to PNG, falling back to matplotlib."""
    try:
        fig.write_image(path, width=width, height=height, scale=2)
    except (ValueError, OSError):
        plotly_to_png_matplotlib(fig, path, width=width, height=height)


def fmt_value(v: float | str, precision: int = 5) -> str:
    """Format a numeric or string value for compact text dump output."""
    if isinstance(v, float):
        if v == 0.0:
            return "0"
        return f"{v:.{precision}g}"
    return str(v)


_INLINE_THRESHOLD = 20


def trace_to_text(trace: dict) -> list[str]:  # noqa: C901, PLR0912, PLR0915
    """Convert a single Plotly trace dict to compact text lines."""
    lines: list[str] = []
    ttype = trace.get("type", "scatter")
    name = trace.get("name", "")
    x = trace.get("x", [])
    y = trace.get("y", [])

    if ttype == "heatmap":
        z = trace.get("z", [])
        x_labels = trace.get("x", [])
        y_labels = trace.get("y", [])
        if name:
            lines.append(f"  [heatmap] {name}")
        else:
            lines.append("  [heatmap]")
        if x_labels:
            lines.append(f"    cols: {', '.join(str(c) for c in x_labels)}")
        for row_i, row in enumerate(z):
            row_label = y_labels[row_i] if row_i < len(y_labels) else row_i
            lines.append(f"    {row_label}: {', '.join(fmt_value(v) for v in row)}")
        return lines

    if ttype == "contour":
        z = trace.get("z", [])
        if name:
            lines.append(f"  [contour] {name}")
        else:
            lines.append("  [contour]")
        x0 = trace.get("x0")
        dx = trace.get("dx")
        y0 = trace.get("y0")
        dy = trace.get("dy")
        if x0 is not None:
            lines.append(
                f"    x0={fmt_value(x0)} dx={fmt_value(dx)} "
                f"y0={fmt_value(y0)} dy={fmt_value(dy)}"
            )
        if z:
            lines.append(f"    grid: {len(z)}x{len(z[0]) if z else 0}")
            flat = [v for row in z for v in row]
            lines.append(f"    range: [{fmt_value(min(flat))}, {fmt_value(max(flat))}]")
        return lines

    if ttype == "surface":
        z = trace.get("z", [])
        if name:
            lines.append(f"  [surface] {name}")
        else:
            lines.append("  [surface]")
        if z:
            flat = [v for row in z for v in row if v is not None]
            if flat:
                lines.append(f"    grid: {len(z)}x{len(z[0]) if z else 0}")
                lines.append(f"    range: [{fmt_value(min(flat))}, {fmt_value(max(flat))}]")
        return lines

    header = f"  [{ttype}]"
    if name:
        header += f" {name}"
    lines.append(header)

    if not x and not y:
        return lines

    error_y = trace.get("error_y", {})
    err_vals = error_y.get("array", []) if isinstance(error_y, dict) else []

    if len(x) <= _INLINE_THRESHOLD:
        for i, (xi, yi) in enumerate(zip(x, y, strict=False)):
            entry = f"    {fmt_value(xi)}: {fmt_value(yi)}"
            if i < len(err_vals):
                entry += f" +/-{fmt_value(err_vals[i])}"
            lines.append(entry)
    else:
        lines.append(f"    n={len(x)}")
        y_num = [v for v in y if isinstance(v, (int, float))]
        if y_num:
            lines.append(f"    y range: [{fmt_value(min(y_num))}, {fmt_value(max(y_num))}]")
            lines.append(f"    y mean: {fmt_value(sum(y_num) / len(y_num))}")
        sample_indices = [
            0,
            len(x) // 4,
            len(x) // 2,
            3 * len(x) // 4,
            len(x) - 1,
        ]
        for idx in sample_indices:
            if idx < len(x):
                entry = f"    {fmt_value(x[idx])}: {fmt_value(y[idx])}"
                if idx < len(err_vals):
                    entry += f" +/-{fmt_value(err_vals[idx])}"
                lines.append(entry)
        lines.append(f"    ... ({len(x)} points total, showing 5 samples)")

    return lines


def fig_to_text(  # noqa: C901
    fig: Figure,
    title: str = "",
    description: dict | None = None,
) -> str:
    """Convert a Plotly figure to a compact machine-readable text block."""
    d = fig.to_dict()
    layout = d.get("layout", {})
    traces = d.get("data", [])

    parts: list[str] = []

    if not title:
        t = layout.get("title", {})
        title = t.get("text", "") if isinstance(t, dict) else str(t)
        title = re.sub(r"<br\s*/?>", " -- ", title)
        title = re.sub(r"<[^>]+>", "", title).strip()
    parts.append(title)

    if description:
        if description.get("what"):
            parts.append(f"  What: {description['what']}")
        if description.get("high"):
            parts.append(f"  High: {description['high']}")
        if description.get("low"):
            parts.append(f"  Low: {description['low']}")
        if description.get("limitations"):
            parts.append(f"  Limitations: {description['limitations']}")

    xaxis = layout.get("xaxis", {})
    yaxis = layout.get("yaxis", {})
    if isinstance(xaxis, dict):
        xt = xaxis.get("title", {})
        xlabel = xt.get("text", "") if isinstance(xt, dict) else str(xt) if xt else ""
        if xlabel:
            parts.append(f"  x-axis: {xlabel}")
    if isinstance(yaxis, dict):
        yt = yaxis.get("title", {})
        ylabel = yt.get("text", "") if isinstance(yt, dict) else str(yt) if yt else ""
        if ylabel:
            parts.append(f"  y-axis: {ylabel}")

    annotations = layout.get("annotations", [])
    subplot_titles = [
        a.get("text", "") for a in annotations if isinstance(a, dict) and a.get("text")
    ]
    if subplot_titles:
        parts.append(f"  subplots: {' | '.join(subplot_titles)}")

    for trace in traces:
        parts.extend(trace_to_text(trace))

    return "\n".join(parts)


_TRIPLE_ENTRY_LEN = 3


def write_text_report(
    all_figs: list[tuple],
    out_path: Path,
    dashboard_title: str = "Dashboard",
    descriptions: dict[str, dict] | None = None,
) -> None:
    """Write a compact text report from all_figs used for HTML.

    all_figs elements can be:
      - (key, title, fig)     -- unified_analysis, new_metrics
      - (key, fig)            -- basin_metrics
    """
    descriptions = descriptions or {}
    lines: list[str] = [
        f"{'=' * 60}",
        dashboard_title,
        f"{'=' * 60}",
        "",
    ]

    for i, entry in enumerate(all_figs):
        if len(entry) == _TRIPLE_ENTRY_LEN:
            key, title, fig = entry
        else:
            key, fig = entry
            title = key.replace("_", " ").title()

        desc = descriptions.get(key)
        lines.append(f"--- {i + 1}. {title} ---")
        lines.append(fig_to_text(fig, title=title, description=desc))
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("\n".join(lines))
