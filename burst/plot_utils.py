"""Shared plotting utilities for burst dashboards."""
import re


def _plotly_to_mpl_color(c):
    if not isinstance(c, str):
        return c
    m = re.match(r"rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)", c)
    if m:
        r, g, b = int(m.group(1)) / 255, int(m.group(2)) / 255, int(m.group(3)) / 255
        a = float(m.group(4)) if m.group(4) else 1.0
        return (r, g, b, a)
    return c


def plotly_to_png_matplotlib(fig_plotly, path: str, width: int = 1200, height: int = 600) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
            color = _plotly_to_mpl_color(line_info["color"])
        elif isinstance(marker_info, dict) and "color" in marker_info:
            mc = marker_info["color"]
            if isinstance(mc, str):
                color = _plotly_to_mpl_color(mc)

        kwargs = dict(label=name)
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
                kwargs["color"] = [_plotly_to_mpl_color(c) for c in bar_colors[:len(x)]]
            ax.bar(x, y, alpha=0.8, **kwargs)
        elif trace_type == "heatmap":
            pass

    xaxis = layout.get("xaxis", {})
    yaxis = layout.get("yaxis", {})
    if isinstance(xaxis, dict):
        ax.set_xlabel(xaxis.get("title", {}).get("text", "") if isinstance(xaxis.get("title"), dict) else "")
    if isinstance(yaxis, dict):
        ax.set_ylabel(yaxis.get("title", {}).get("text", "") if isinstance(yaxis.get("title"), dict) else "")

    ax.set_title(title_text[:120], fontsize=10, wrap=True)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(handles[:15], labels[:15], fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(mfig)


def save_png(fig, path: str, width: int = 1200, height: int = 600) -> None:
    try:
        fig.write_image(path, width=width, height=height, scale=2)
    except Exception:
        plotly_to_png_matplotlib(fig, path, width=width, height=height)
