"""Weight drift & gradient cosine heatmaps over training.

Two analyses from saved data (no forward pass needed):
  1. Weight drift: ||W_step - W_0||_F per layer from checkpoints
  2. Per-layer gradient cosine similarity from grad_cosine_sim/*.json

Each gets: LayerxStep heatmap, rate-of-change heatmap, line charts.

Usage:
    python burst/weight_driftheatmap.py <run_dir> [--n-seeds 3]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch

from burst.config import SCHED_COLORS, SCHED_DISPLAY
from burst.core.bundle import (
    layer_groups_from_state_dict,
    load_grad_sim_records,
    load_state_dict_cpu,
    weight_drift_arrays,
)
from burst.core.train_utils import ckpt_files, sched_order

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@torch.no_grad()
def compute_weight_drift(
    ckpt_root: Path,
    all_results: list[dict[str, Any]],
    n_layer: int,
    n_seeds: int = 3,
) -> dict[str, dict[str, Any]]:
    """Compute per-layer weight drift from checkpoints."""
    jobs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in all_results:
        jobs[r["schedule"]].append(r)

    out: dict[str, dict[str, Any]] = {}
    for sched in sorted(jobs, key=sched_order):
        seed_cum: list[np.ndarray] = []
        seed_step: list[np.ndarray] = []
        steps: list[int] | None = None
        layers: list[str] | None = None
        lg: list[tuple[str, list[str]]] | None = None

        for r in jobs[sched][:n_seeds]:
            ckpt_dir = ckpt_root / r["label"]
            if not ckpt_dir.exists():
                continue
            files_map = ckpt_files(ckpt_dir)
            if not files_map:
                continue

            sorted_steps = sorted(files_map)
            base_sd = load_state_dict_cpu(files_map[sorted_steps[0]])
            if layers is None:
                lg = layer_groups_from_state_dict(base_sd, n_layer)
                layers = [g[0] for g in lg]
                steps = sorted_steps

            if lg is None:
                msg = "layer_groups_from_state_dict returned None"
                raise ValueError(msg)
            cum_row, step_row = weight_drift_arrays(files_map, sorted_steps, lg, base_sd)
            seed_cum.append(cum_row)
            seed_step.append(step_row)

        if seed_cum and layers is not None and steps is not None:
            out[sched] = {
                "layers": layers,
                "steps": steps,
                "cumulative": np.mean(seed_cum, axis=0),
                "stepwise": np.mean(seed_step, axis=0),
            }
    return out


# ---------------------------------------------------------------------------
# 2. Per-layer gradient cosine from JSON (already computed)
# ---------------------------------------------------------------------------


def load_per_layer_grad_cosine(
    run_dir: Path, n_seeds: int = 3,
) -> dict[str, dict[str, Any]]:
    """Load per-layer gradient cosine similarity from grad sim records."""
    all_recs = load_grad_sim_records(run_dir)
    with_pl = [r for r in all_recs if r.get("grad_sim_log", {}).get("per_layer")]
    if not with_pl:
        msg = f"No per-layer grad cosine records in {run_dir}"
        raise ValueError(msg)

    by_sched: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in with_pl:
        by_sched[rec["schedule"]].append(rec)

    out: dict[str, dict[str, Any]] = {}
    for sched in sorted(by_sched, key=sched_order):
        records = by_sched[sched][:n_seeds]
        layers = (
            records[0]["grad_sim_log"].get("layer_names")
            or records[0].get("layer_names")
        )
        steps = records[0]["grad_sim_log"]["step"]
        grids = []
        for rec in records:
            pl = rec["grad_sim_log"]["per_layer"]
            grid = np.array([pl[ln] for ln in layers])
            grids.append(grid)
        out[sched] = {
            "layers": layers,
            "steps": steps,
            "grid": np.mean(grids, axis=0),
        }
    return out


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

LAYER_COLORS = px.colors.qualitative.Dark24


def is_ln(name: str) -> bool:
    """Check if layer name is a layer-norm layer."""
    return name.endswith("_ln") or name == "ln_f"


def filter_layers(
    layers: list[str], grid: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    """Remove ln layers and reverse order (deeper layers at top)."""
    mask = [i for i, ln in enumerate(layers) if not is_ln(ln)]
    filtered = [layers[i] for i in mask]
    filtered_grid = grid[mask]
    return filtered[::-1], filtered_grid[::-1]


def heatmap(  # noqa: PLR0913
    z: np.ndarray,
    x: list[int],
    y: list[str],
    title: str,
    colorscale: str,
    cbar_title: str,
    zmin: float | None = None,
    zmax: float | None = None,
    zmid: float | None = None,
) -> go.Figure:
    """Build a Plotly heatmap figure."""
    kw: dict[str, Any] = {
        "z": z,
        "x": [str(s) for s in x],
        "y": y,
        "colorscale": colorscale,
        "colorbar": {"title": cbar_title},
    }
    if zmid is not None:
        vm = max(float(np.percentile(np.abs(z), 95)), 1e-6)
        kw.update(zmid=zmid, zmin=-vm, zmax=vm)
    if zmin is not None:
        kw["zmin"] = zmin
    if zmax is not None:
        kw["zmax"] = zmax
    fig = go.Figure(go.Heatmap(**kw))
    fig.update_layout(
        title=title,
        xaxis_title="Step",
        yaxis_title="Layer",
        template="plotly_white",
        height=max(300, len(y) * 28 + 100),
        width=max(700, len(x) * 10 + 200),
    )
    return fig


def line_per_layer(  # noqa: PLR0913
    steps: list[int],
    grid: np.ndarray,
    layers: list[str],
    title: str,
    ylabel: str,
    *,
    log_y: bool = False,
) -> go.Figure:
    """Build a per-layer line chart."""
    fig = go.Figure()
    for li, ln in enumerate(layers):
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=grid[li],
                mode="lines",
                name=ln,
                line={"color": LAYER_COLORS[li % len(LAYER_COLORS)]},
            )
        )
    layout: dict[str, Any] = {
        "title": title,
        "xaxis_title": "Step",
        "yaxis_title": ylabel,
        "template": "plotly_white",
        "height": 450,
    }
    if log_y:
        layout["yaxis_type"] = "log"
    fig.update_layout(**layout)
    return fig


def rate_of_change(
    grid: np.ndarray, steps: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute rate of change between consecutive steps."""
    dt = np.diff(steps).astype(float)
    mid = (
        (np.array(steps[:-1]) + np.array(steps[1:])) / 2
    ).astype(int)
    return np.diff(grid, axis=1) / np.maximum(dt[None, :], 1.0), mid


def line_per_schedule(
    data: dict[str, dict[str, Any]],
    key: str,
    title: str,
    ylabel: str,
    *,
    log_y: bool = False,
) -> go.Figure:
    """Build a per-schedule line chart."""
    fig = go.Figure()
    for sched in sorted(data, key=sched_order):
        d = data[sched]
        total = (
            d[key].sum(axis=0) if d[key].ndim == 2  # noqa: PLR2004
            else d[key]
        )
        fig.add_trace(
            go.Scatter(
                x=d["steps"],
                y=total,
                mode="lines",
                name=SCHED_DISPLAY.get(sched, sched),
                line={"color": SCHED_COLORS.get(sched, "#888")},
            )
        )
    layout: dict[str, Any] = {
        "title": title,
        "xaxis_title": "Step",
        "yaxis_title": ylabel,
        "template": "plotly_white",
        "height": 450,
    }
    if log_y:
        layout["yaxis_type"] = "log"
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def assemble_html(
    figs: list[tuple[str, go.Figure]],
    title: str,
    subtitle: str,
    out_path: Path,
) -> Path:
    """Assemble chart figures into a single HTML report."""
    parts = [
        "<!DOCTYPE html><html><head>",
        f"<title>{title}</title>",
        "<style>body{font-family:system-ui;max-width:1400px;"
        "margin:0 auto;padding:20px}"
        "h1{border-bottom:2px solid #333;padding-bottom:8px}"
        "h2{margin-top:36px;color:#444}</style>",
        "</head><body>",
        f"<h1>{title}</h1><p>{subtitle}</p>",
    ]
    first = True
    for chart_title, fig in figs:
        parts.append(f"<h2>{chart_title}</h2>")
        parts.append(
            fig.to_html(
                full_html=False,
                include_plotlyjs="cdn" if first else False,
            )
        )
        first = False
    parts.append("</body></html>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts))
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run weight drift and gradient cosine heatmap analysis."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str)
    parser.add_argument("--n-seeds", type=int, default=3)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg, all_results, ckpt_root, _logs_dir = resolve(run_dir)
    n_layer = cfg["n_layer"]
    run_name = run_dir.name

    wd = compute_weight_drift(
        ckpt_root, all_results, n_layer, n_seeds=args.n_seeds,
    )
    schedules = sorted(wd, key=sched_order)

    all_cum_vals, all_step_vals = [], []
    filtered_data: dict[str, dict[str, Any]] = {}
    for sched in schedules:
        d = wd[sched]
        fl, fc = filter_layers(d["layers"], d["cumulative"])
        _, fs = filter_layers(d["layers"], d["stepwise"])
        filtered_data[sched] = {
            "layers": fl,
            "steps": d["steps"],
            "cumulative": fc,
            "stepwise": fs,
        }
        all_cum_vals.append(fc[fc > 0])
        all_step_vals.append(fs[fs > 0])

    cum_all = np.concatenate(all_cum_vals)
    step_all = np.concatenate(all_step_vals)
    cum_log_min = float(np.log10(cum_all.min()))
    cum_log_max = float(np.log10(cum_all.max()))
    step_log_min = float(np.log10(step_all.min()))
    step_log_max = float(np.log10(step_all.max()))

    figs: list[tuple[str, go.Figure]] = []

    for sched in schedules:
        d = filtered_data[sched]
        label = SCHED_DISPLAY.get(sched, sched)
        log_z = np.log10(np.maximum(d["cumulative"], 1e-10))
        figs.append((
            f"{label}: Cumulative ||W - W0||_F",
            heatmap(
                log_z,
                d["steps"],
                d["layers"],
                f"{label}: Cumulative Weight Drift (log10)",
                "YlOrRd",
                "log10(||dW||_F)",
                zmin=cum_log_min,
                zmax=cum_log_max,
            ),
        ))

    for sched in schedules:
        d = filtered_data[sched]
        label = SCHED_DISPLAY.get(sched, sched)
        log_z = np.log10(np.maximum(d["stepwise"], 1e-10))
        figs.append((
            f"{label}: Step-wise ||W_t - W_{{t-1}}||_F",
            heatmap(
                log_z,
                d["steps"],
                d["layers"],
                f"{label}: Step-wise Weight Change (log10)",
                "Viridis",
                "log10(||dW||_F)",
                zmin=step_log_min,
                zmax=step_log_max,
            ),
        ))

    for sched in schedules:
        d = filtered_data[sched]
        label = SCHED_DISPLAY.get(sched, sched)
        figs.append((
            f"{label}: Cumulative Drift Per Layer",
            line_per_layer(
                d["steps"],
                d["cumulative"],
                d["layers"],
                f"{label}: Cumulative ||dW||_F",
                "||dW||_F",
                log_y=True,
            ),
        ))
        figs.append((
            f"{label}: Step-wise Change Per Layer",
            line_per_layer(
                d["steps"],
                d["stepwise"],
                d["layers"],
                f"{label}: ||W_t - W_{{t-1}}||_F",
                "||dW||_F",
                log_y=True,
            ),
        ))

    figs.append((
        "Total Cumulative Drift (all schedules)",
        line_per_schedule(
            wd, "cumulative",
            "Total ||W - W0||_F", "||dW||_F", log_y=True,
        ),
    ))
    figs.append((
        "Total Step-wise Change (all schedules)",
        line_per_schedule(
            wd, "stepwise",
            "Total ||W_t - W_{{t-1}}||_F", "||dW||_F",
            log_y=True,
        ),
    ))

    assemble_html(
        figs,
        f"Weight Drift -- {run_name}",
        "Per-layer weight change (log scale, ln layers excluded,"
        " deeper layers at top).",
        run_dir / "results" / "weight_driftheatmaps.html",
    )

    gc = load_per_layer_grad_cosine(run_dir, n_seeds=args.n_seeds)

    figs2: list[tuple[str, go.Figure]] = []
    for sched in sorted(gc, key=sched_order):
        d = gc[sched]
        label = SCHED_DISPLAY.get(sched, sched)
        figs2.append((
            f"{label}: Gradient Cosine Similarity"
            " (Layer x Step)",
            heatmap(
                d["grid"],
                d["steps"],
                d["layers"],
                f"{label}: burst-vs-other cossim",
                "RdBu_r",
                "cossim",
                zmid=0,
            ),
        ))
        roc, mid = rate_of_change(d["grid"], d["steps"])
        figs2.append((
            f"{label}: Rate of Change of Gradient Cosine",
            heatmap(
                roc,
                mid,
                d["layers"],
                f"{label}: d(cossim)/dstep",
                "RdBu_r",
                "dcossim/dstep",
                zmid=0,
            ),
        ))
        figs2.append((
            f"{label}: Gradient Cosine Per Layer",
            line_per_layer(
                d["steps"],
                d["grid"],
                d["layers"],
                f"{label}: Per-Layer Gradient Cosine",
                "Cosine Similarity",
            ),
        ))

    assemble_html(
        figs2,
        f"Per-Layer Gradient Cosine -- {run_name}",
        "Per-layer burst-vs-other gradient cosine similarity"
        " from saved JSON data.",
        run_dir / "results" / "grad_cosineheatmaps.html",
    )


def resolve(
    run_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path, Path]:
    """Locate config, results pickle, and checkpoint directory."""
    for cfg_path in [
        run_dir / "results" / "config.json",
        run_dir / "config.json",
    ]:
        if cfg_path.exists():
            with cfg_path.open() as f:
                full_cfg = json.load(f)
            break
    else:
        msg = f"No config.json in {run_dir}"
        raise FileNotFoundError(msg)

    cfg = full_cfg.get("base_cfg", full_cfg)

    for logs_dir in [
        run_dir / "logs",
        run_dir / "_heavy" / "logs",
    ]:
        pkl_path = logs_dir / "all_results.pkl"
        if pkl_path.exists():
            with pkl_path.open("rb") as f:
                all_results = pickle.load(f)  # noqa: S301
            break
    else:
        msg = f"No all_results.pkl in {run_dir}"
        raise FileNotFoundError(msg)

    ckpt_root = logs_dir / "checkpoints"
    if not ckpt_root.exists():
        msg = f"No checkpoints at {ckpt_root}"
        raise FileNotFoundError(msg)
    return cfg, all_results, ckpt_root, logs_dir


if __name__ == "__main__":
    main()
