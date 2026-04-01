from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from einops import reduce

from burst.core.train_utils import load_net, resolve_run_paths
from burst.dev.probe import collect_activations_KPTN


def build_representation_summary(
    run_dir: str | Path,
    grouped_results: dict[str, list[dict[str, Any]]],
    *,
    n_docs_per_class: int = 64,
) -> dict[str, Any]:
    _cfg_path, logs_dir, _ = resolve_run_paths(run_dir)
    data_path = logs_dir / "_data.pkl"
    ckpt_root = logs_dir / "checkpoints"
    if not data_path.exists() or not ckpt_root.exists():
        return {}

    with open(data_path, "rb") as f:
        target_pool, bg_pool, _, _, _ = pickle.load(f)

    other_docs = _subsample_pool(bg_pool, n_docs_per_class, seed=0)
    burst_docs = _subsample_pool(target_pool, n_docs_per_class, seed=1)
    if other_docs.size == 0 or burst_docs.size == 0:
        return {}

    by_schedule: dict[str, Any] = {}
    for schedule, runs in grouped_results.items():
        per_seed = []
        for run in runs:
            label = run.get("label")
            if not label:
                continue
            ckpt_dir = ckpt_root / label
            if not ckpt_dir.exists():
                continue
            seed_metrics = _representation_for_run(run, ckpt_dir, other_docs, burst_docs)
            if seed_metrics is not None:
                per_seed.append(seed_metrics)

        if not per_seed:
            continue

        proj_vals = np.array([seed["late_centroid_projection"] for seed in per_seed], dtype=float)
        shift_vals = np.array([seed["late_other_shift_norm"] for seed in per_seed], dtype=float)
        cos_vals = np.array([seed["late_drift_cosine"] for seed in per_seed], dtype=float)
        by_schedule[schedule] = {
            "late_centroid_projection": _mean_ci_payload(proj_vals),
            "late_other_shift_norm": _mean_ci_payload(shift_vals),
            "late_drift_cosine": _mean_ci_payload(cos_vals),
            "per_seed": per_seed,
        }

    return {"by_schedule": by_schedule}


def _representation_for_run(
    run: dict[str, Any],
    ckpt_dir: Path,
    other_docs_BL: np.ndarray,
    burst_docs_BL: np.ndarray,
) -> dict[str, float] | None:
    ckpt_files = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpt_files:
        return None

    step_to_path = {int(path.stem.split("_")[1]): path for path in ckpt_files}
    available_steps = sorted(step_to_path)
    pre_step = available_steps[0]
    burst_steps = run["config"]["total_steps"]
    peak_step = min(available_steps, key=lambda step: abs(step - (burst_steps - 1)))

    net_pre = load_net(run["config"], str(step_to_path[pre_step]))
    net_peak = load_net(run["config"], str(step_to_path[peak_step]))

    other_pre = _mean_layer_vectors(net_pre, other_docs_BL)
    other_peak = _mean_layer_vectors(net_peak, other_docs_BL)
    burst_pre = _mean_layer_vectors(net_pre, burst_docs_BL)
    burst_peak = _mean_layer_vectors(net_peak, burst_docs_BL)

    late_indices = _late_layer_indices(len(other_pre))
    centroid_projection_vals = []
    shift_norm_vals = []
    drift_cos_vals = []
    for layer_idx in late_indices:
        other_drift = other_peak[layer_idx] - other_pre[layer_idx]
        burst_drift = burst_peak[layer_idx] - burst_pre[layer_idx]
        burst_norm = float(np.linalg.norm(burst_drift))
        other_norm = float(np.linalg.norm(other_drift))
        pre_other_norm = float(np.linalg.norm(other_pre[layer_idx]))

        centroid_projection_vals.append(
            float(np.dot(other_drift, burst_drift) / (burst_norm + 1e-12))
        )
        shift_norm_vals.append(other_norm / (pre_other_norm + 1e-12))
        drift_cos_vals.append(
            float(np.dot(other_drift, burst_drift) / ((other_norm * burst_norm) + 1e-12))
        )

    del net_pre, net_peak

    return {
        "seed": float(run["seed"]),
        "late_centroid_projection": float(np.mean(centroid_projection_vals)),
        "late_other_shift_norm": float(np.mean(shift_norm_vals)),
        "late_drift_cosine": float(np.mean(drift_cos_vals)),
    }


def _mean_layer_vectors(net, docs_BL: np.ndarray) -> list[np.ndarray]:
    activations_KPTN = collect_activations_KPTN(net, docs_BL)
    return [
        reduce(activation_PTN, "p t n -> n", "mean").numpy() for activation_PTN in activations_KPTN
    ]


def _late_layer_indices(n_layers_total: int) -> list[int]:
    start = max(1, n_layers_total - 2)
    return list(range(start, n_layers_total))


def _subsample_pool(pool: dict, n_docs: int, *, seed: int) -> np.ndarray:
    if not pool:
        return np.zeros((0, 0), dtype=np.int64)
    docs = np.concatenate(list(pool.values()))
    if docs.shape[0] <= n_docs:
        return docs
    rng = np.random.default_rng(seed)
    idx = rng.choice(docs.shape[0], size=n_docs, replace=False)
    return docs[idx]


def _mean_ci_payload(values: np.ndarray) -> dict[str, float]:
    mean = float(np.mean(values))
    if values.size <= 1:
        ci = 0.0
    else:
        ci = float(1.96 * np.std(values) / np.sqrt(values.size))
    return {"mean": mean, "ci": ci}
