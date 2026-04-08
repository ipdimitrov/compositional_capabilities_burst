"""EWC-style Fisher Information analysis for burstiness runs.

Computes the empirical diagonal Fisher Information at the pre-burst checkpoint
using other-class data, then measures the Fisher-weighted displacement caused
by each burst schedule:

    D = sum_i  F_i * (theta_i^post_burst - theta_i^pre_burst)^2

A high D means burst training moved parameters that matter for other-class
knowledge — i.e., more interference. Diluted schedules should yield lower D.

Also computes per-layer D_l to localise where damage concentrates.

Usage:
    python burst/ewc_metrics.py <run_dir>
    python burst/ewc_metrics.py <run_dir> --n-fisher-batches 200 --n-seeds 3
    python burst/ewc_metrics.py <run_dir> --out-dir results/ewc

Dimension key:
    B: batch_size
    L: sequence_length
    P: n_params (total flattened parameters)
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from burst.config import SCHED_COLORS, parse_run_config
from burst.core.metrics.gradients import _layer_groups
from burst.core.train_utils import DEVICE, load_net, resolve_run_paths
from burst.dev.plot_utils import save_png

logger = logging.getLogger(__name__)
_rng = np.random.default_rng()


from burst.core.train_utils import (  # noqa: E402
    ckpt_files,
    sched_order,
)


def compute_diagonal_fisher(
    net: torch.nn.Module,
    other_docs_BL: np.ndarray,
    n_batches: int,
    batch_size: int = 128,
) -> dict[str, torch.Tensor]:
    """Compute empirical diagonal Fisher on other-class data.

    F_hat_i = (1/M) * sum_{m=1}^{M} (d log p(x_m) / d theta_i)^2

    Returns param_name -> 1-D Fisher tensor (float32, CPU).
    """
    net.train()
    fisher: dict[str, torch.Tensor] = {
        n: torch.zeros_like(p.detach().view(-1), dtype=torch.float32)
        for n, p in net.named_parameters()
    }
    n_docs = other_docs_BL.shape[0]
    actual = min(n_batches, max(1, n_docs // batch_size))

    for _ in range(actual):
        idx = _rng.choice(n_docs, min(batch_size, n_docs), replace=False)
        dat = torch.as_tensor(other_docs_BL[idx], dtype=torch.long, device=DEVICE)
        inp, tgt = dat[:, :-1], dat[:, 1:]
        net.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            logits = net(inp)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        loss.backward()
        for n, p in net.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.detach().float().view(-1) ** 2

    for n in fisher:
        fisher[n] /= actual
    net.zero_grad(set_to_none=True)
    return fisher


def compute_fisher_displacement(
    fisher: dict[str, torch.Tensor],
    pre_params: dict[str, torch.Tensor],
    post_params: dict[str, torch.Tensor],
    layer_groups: list[tuple[str, list[str]]],
) -> dict:
    """Compute D = sum_i F_i * (theta_post - theta_pre)^2, total and per-layer."""
    per_layer_D: dict[str, float] = {}
    for layer_name, pnames in layer_groups:
        layer_D = sum(
            (fisher[pn] * (post_params[pn] - pre_params[pn]) ** 2).sum().item()
            for pn in pnames
            if pn in fisher and pn in pre_params and pn in post_params
        )
        per_layer_D[layer_name] = layer_D
    return {"total_D": sum(per_layer_D.values()), "per_layer_D": per_layer_D}


def run_ewc_analysis(  # noqa: C901, PLR0915
    run_dir: Path,
    n_fisher_batches: int = 200,
    n_seeds: int = 3,
    fisher_batch_size: int = 128,
) -> dict:
    """Run full EWC analysis for a run directory."""
    cfg_path, logs_dir, _ = resolve_run_paths(run_dir)
    with cfg_path.open() as f:
        run_cfg = json.load(f)
    rc = parse_run_config(run_cfg)
    base_cfg = rc["base_cfg"]

    with (logs_dir / "_data.pkl").open("rb") as f:
        _, bg_pool, _, _, _ = pickle.load(f)  # noqa: S301
    with (logs_dir / "all_results.pkl").open("rb") as f:
        all_results = pickle.load(f)  # noqa: S301

    ckpt_root = logs_dir / "checkpoints"
    if not ckpt_root.exists():
        logger.info("No checkpoints in %s — skipping EWC.", run_dir)
        return {}

    other_docs_BL = np.concatenate(list(bg_pool.values()))

    pretrain_ckpt = logs_dir / "pretrain_ckpt.pt"
    if pretrain_ckpt.exists():
        fisher_ckpt = str(pretrain_ckpt)
        logger.info("Computing Fisher at pretrain checkpoint...")
    else:
        first_label = run_cfg["jobs"][0]["label"]
        first_files = ckpt_files(ckpt_root / first_label)
        if not first_files:
            logger.info("No checkpoints found — skipping EWC.")
            return {}
        fisher_ckpt = str(first_files[min(first_files)])
        logger.info("pretrain_ckpt.pt not found — using first checkpoint.")

    net_fisher = load_net(base_cfg, fisher_ckpt)
    fisher = compute_diagonal_fisher(net_fisher, other_docs_BL, n_fisher_batches, fisher_batch_size)
    layer_groups = _layer_groups(net_fisher)
    del net_fisher
    torch.cuda.empty_cache()

    def _flat(path: str) -> dict[str, torch.Tensor]:
        return {
            k: v.float().view(-1)
            for k, v in torch.load(path, map_location="cpu", weights_only=True).items()
        }

    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    layer_names = [n for n, _ in layer_groups]
    per_schedule: dict[str, dict] = {}

    for sched in sorted(jobs_by_schedule, key=sched_order):
        total_Ds: list[float] = []
        layer_Ds: dict[str, list[float]] = {}
        seeds_done = 0

        for r in jobs_by_schedule[sched]:
            if seeds_done >= n_seeds:
                break
            label = r["label"]
            files = ckpt_files(ckpt_root / label)
            if not files:
                continue

            T = r["config"]["total_steps"]
            available = sorted(files)
            pre_step = available[0]
            post_step = min(available, key=lambda x: abs(x - (T - 1)))

            disp = compute_fisher_displacement(
                fisher, _flat(str(files[pre_step])), _flat(str(files[post_step])), layer_groups
            )
            total_Ds.append(disp["total_D"])
            for ln, d in disp["per_layer_D"].items():
                layer_Ds.setdefault(ln, []).append(d)
            seeds_done += 1
            logger.debug("%s: total_D=%.4e", label, disp["total_D"])

        per_schedule[sched] = {
            "total_Ds": total_Ds,
            "mean_total_D": float(np.mean(total_Ds)) if total_Ds else float("nan"),
            "std_total_D": float(np.std(total_Ds)) if len(total_Ds) > 1 else 0.0,
            "per_layer_mean_D": {
                ln: float(np.mean(layer_Ds[ln])) if layer_Ds.get(ln) else float("nan")
                for ln in layer_names
            },
            "per_layer_std_D": {
                ln: float(np.std(layer_Ds[ln])) if len(layer_Ds.get(ln, [])) > 1 else 0.0
                for ln in layer_names
            },
            "layer_names": layer_names,
        }

    return {"per_schedule": per_schedule, "run_name": run_dir.name}


def make_ewc_plots(result: dict, out_dir: Path) -> None:
    """Generate EWC displacement plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    per_schedule = result.get("per_schedule", {})
    run_name = result.get("run_name", "")
    if not per_schedule:
        return

    schedules = sorted(per_schedule.keys(), key=sched_order)
    colors = [SCHED_COLORS.get(s, "#888888") for s in schedules]
    layer_names = next(iter(per_schedule.values()), {}).get("layer_names", [])

    def _save(fig: object, name: str) -> None:
        save_png(fig, str(out_dir / name))

    fig = go.Figure(
        go.Bar(
            x=schedules,
            y=[per_schedule[s]["mean_total_D"] for s in schedules],
            error_y={
                "type": "data",
                "array": [per_schedule[s]["std_total_D"] for s in schedules],
                "visible": True,
            },
            marker_color=colors,
        )
    )
    fig.update_layout(
        title=f"EWC Fisher-Weighted Displacement — {run_name}<br>"
        "<sup>D = Σ F_i · Δθ_i²  (higher = more damage to other-class parameters)</sup>",
        xaxis_title="Schedule",
        yaxis_title="Fisher-Weighted Displacement D",
        template="plotly_white",
        height=500,
    )
    _save(fig, "ewc_total_displacement.png")

    if layer_names:
        fig = go.Figure(
            go.Heatmap(
                z=[
                    [
                        per_schedule[s]["per_layer_mean_D"].get(ln, float("nan"))
                        for ln in layer_names
                    ]
                    for s in schedules
                ],
                x=layer_names,
                y=schedules,
                colorscale="Reds",
                colorbar={"title": "D_layer"},
            )
        )
        fig.update_layout(
            title=f"Per-Layer EWC Displacement — {run_name}<br>"
            "<sup>D_l = Σ_{{i∈l}} F_i · Δθ_i²</sup>",
            xaxis_title="Layer",
            yaxis_title="Schedule",
            template="plotly_white",
            height=500,
        )
        _save(fig, "ewc_per_layer_displacement.png")

        fig = go.Figure()
        for ln in layer_names:
            fig.add_trace(
                go.Bar(
                    name=ln,
                    x=schedules,
                    y=[per_schedule[s]["per_layer_mean_D"].get(ln, 0.0) for s in schedules],
                )
            )
        fig.update_layout(
            barmode="stack",
            title=f"EWC Displacement by Layer — {run_name}<br>"
            "<sup>Stacked: each layer's contribution to total D</sup>",
            xaxis_title="Schedule",
            yaxis_title="Fisher-Weighted Displacement D",
            legend_title="Layer",
            template="plotly_white",
            height=500,
        )
        _save(fig, "ewc_stacked_by_layer.png")

    logger.info("EWC plots saved to %s", out_dir)


def main() -> None:
    """Run EWC analysis from command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--n-fisher-batches", type=int, default=200)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--fisher-batch-size", type=int, default=128)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or (args.run_dir / "results" / "ewc_metrics")
    logger.info("EWC analysis: %s", args.run_dir)
    result = run_ewc_analysis(
        args.run_dir, args.n_fisher_batches, args.n_seeds, args.fisher_batch_size
    )

    if not result:
        logger.info("No results — exiting.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "ewc_results.pkl").open("wb") as f:
        pickle.dump(result, f)
    make_ewc_plots(result, out_dir)
    logger.info("EWC done. Results: %s", out_dir)


if __name__ == "__main__":
    main()
