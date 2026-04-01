from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from burst.config import (
    MODE_CURRENT,
    TrainConfig,
    burst_steps_for_mode,
    ordered_schedules,
    reversion_life_key,
    reversion_life_label,
)
from burst.core.representation import build_representation_summary
from burst.core.train.worker import n_target_for_step
from burst.core.train_utils import compute_lr_schedule, load_results, resolve_run_paths

BUNDLE_SCHEMA_VERSION = 1
BUNDLE_DIRNAME = "chart_bundle"
BUNDLE_VERSION_DIR = "v1"
BUNDLE_FILENAME = "core_bundle.json"


def bundle_dir(run_dir: str | Path) -> Path:
    _, _, results_dir = resolve_run_paths(run_dir)
    return results_dir / BUNDLE_DIRNAME / BUNDLE_VERSION_DIR


def bundle_path(run_dir: str | Path) -> Path:
    return bundle_dir(run_dir) / BUNDLE_FILENAME


def load_core_bundle(run_dir: str | Path) -> dict[str, Any]:
    path = bundle_path(run_dir)
    with open(path) as f:
        return json.load(f)


def build_and_save_core_bundle(run_dir: str | Path) -> Path:
    bundle = build_core_bundle(run_dir)
    out_dir = bundle_dir(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / BUNDLE_FILENAME
    with open(path, "w") as f:
        json.dump(bundle, f, indent=2)
    return path


def build_core_bundle(run_dir: str | Path) -> dict[str, Any]:
    results, cfg = load_results(run_dir)
    assert results, "expected at least one result in all_results.pkl"

    grouped = _group_results(results)
    schedules = ordered_schedules(grouped.keys())
    thresholds = list(TrainConfig().reversion_thresholds)
    burst_mode = cfg.get("burst_mode", MODE_CURRENT)
    grad_records = _load_grad_sim_records(run_dir)

    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "run_name": Path(run_dir).name,
        "config": {
            "burst_mode": burst_mode,
            "base_cfg": cfg["base_cfg"],
            "thresholds": thresholds,
            "schedules": schedules,
        },
        "schedule_bars": _build_schedule_bars(grouped),
        "lr_curves": _build_lr_curves(grouped),
        "training": _build_training_curves(grouped),
        "summary": _build_summary(grouped, thresholds),
        "gradients": _build_gradient_curves(grouped, grad_records, burst_mode),
        "representation": build_representation_summary(run_dir, grouped),
    }


def _load_grad_sim_records(run_dir: str | Path) -> list[dict[str, Any]]:
    run_dir = Path(run_dir)
    records: list[dict[str, Any]] = []

    for grad_dir in (run_dir / "results" / "grad_cosine_sim", run_dir / "grad_cosine_sim"):
        if not grad_dir.is_dir():
            continue
        for path in sorted(grad_dir.glob("*.json")):
            with open(path) as f:
                records.append(json.load(f))
        if records:
            return records

    for path in (run_dir / "logs" / "all_results.pkl", run_dir / "all_results.pkl"):
        if not path.exists():
            continue
        with open(path, "rb") as f:
            results = pickle.load(f)
        for result in results:
            grad_log = result.get("grad_sim_log")
            if not grad_log or not grad_log.get("step"):
                continue
            records.append(
                {
                    "schedule": result["schedule"],
                    "seed": result["seed"],
                    "label": result.get("label", ""),
                    "grad_sim_log": grad_log,
                    "grad_projection_log": result.get("grad_projection_log"),
                }
            )
        if records:
            return records

    return records


def _group_results(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["schedule"]].append(result)
    return {schedule: grouped[schedule] for schedule in ordered_schedules(grouped.keys())}


def _group_grad_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["schedule"]].append(record)
    return {schedule: grouped[schedule] for schedule in ordered_schedules(grouped.keys())}


def _mean_ci(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    if values.size == 1:
        return mean, 0.0
    ci = float(1.96 * np.std(values) / np.sqrt(values.size))
    return mean, ci


def _interpolate_to_reference(
    reference_steps: np.ndarray,
    source_steps: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    if len(reference_steps) == len(source_steps) and np.allclose(reference_steps, source_steps):
        return values.astype(float)
    return np.interp(reference_steps, source_steps, values).astype(float)


def _interpolate_optional_metric(
    reference_steps: np.ndarray,
    source_steps: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    if len(values) != len(source_steps):
        return np.full_like(reference_steps, np.nan, dtype=float)
    return _interpolate_to_reference(reference_steps, source_steps, values)


def _aggregate_series(
    runs: list[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    first_steps = np.array(runs[0]["log"]["step"], dtype=float)
    stacked = []
    for run in runs:
        steps = np.array(run["log"]["step"], dtype=float)
        values = np.array(run["log"].get(key, [float("nan")] * len(steps)), dtype=float)
        stacked.append(_interpolate_to_reference(first_steps, steps, values))
    arr = np.stack(stacked)
    return {
        "steps": first_steps.tolist(),
        "mean": np.nanmean(arr, axis=0).tolist(),
        "ci": _series_ci(arr).tolist(),
    }


def _series_ci(arr: np.ndarray) -> np.ndarray:
    if arr.shape[0] <= 1:
        return np.nanstd(arr, axis=0)
    return 1.96 * np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0])


def _build_schedule_bars(grouped: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for schedule, runs in grouped.items():
        cfg = runs[0]["config"]
        pre_steps = int(cfg.get("pre_burst_steps", 0))
        burst_steps = int(cfg["total_steps"])
        reversion_steps = int(cfg["reversion_steps"])
        batch_size = int(cfg["batch_size"])
        p_target = float(cfg["p_target"])

        fractions = [0.0] * (pre_steps + burst_steps + reversion_steps)
        for step in range(burst_steps):
            frac = n_target_for_step(step, burst_steps, schedule, p_target, batch_size) / batch_size
            fractions[pre_steps + step] = float(frac)

        payload[schedule] = {
            "fractions": fractions,
            "pre_steps": pre_steps,
            "burst_steps": burst_steps,
            "reversion_steps": reversion_steps,
        }
    return payload


def _build_lr_curves(grouped: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for schedule, runs in grouped.items():
        cfg = runs[0]["config"]
        steps, lrs = compute_lr_schedule(cfg)
        payload[schedule] = {"steps": steps.tolist(), "lr": lrs.tolist()}
    return payload


def _build_training_curves(grouped: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for schedule, runs in grouped.items():
        cfg = runs[0]["config"]
        payload[schedule] = {
            "pre_steps": int(cfg.get("pre_burst_steps", 0)),
            "burst_steps": int(cfg["total_steps"]),
            "reversion_steps": int(cfg["reversion_steps"]),
            "acc_burst": _aggregate_series(runs, "acc_burst"),
            "acc_other": _aggregate_series(runs, "acc_other"),
            "loss": _aggregate_series(runs, "loss"),
        }
    return payload


def _build_summary(
    grouped: dict[str, list[dict[str, Any]]],
    thresholds: list[float],
) -> dict[str, Any]:
    schedules = list(grouped.keys())
    by_schedule: dict[str, Any] = {}
    for schedule, runs in grouped.items():
        peak_vals = np.array([run["peak_burst"] for run in runs], dtype=float)
        auc_vals = np.array([run["reversion_auc"] for run in runs], dtype=float)
        other_end_vals = np.array([run["log"]["acc_other"][-1] for run in runs], dtype=float)
        peak_mean, peak_ci = _mean_ci(peak_vals)
        auc_mean, auc_ci = _mean_ci(auc_vals)
        other_mean, other_ci = _mean_ci(other_end_vals)

        life_stats: dict[str, Any] = {}
        for threshold in thresholds:
            key = reversion_life_key(threshold)
            label = reversion_life_label(threshold)
            vals = np.array(
                [run.get(key, runs[0]["config"]["reversion_steps"]) for run in runs], dtype=float
            )
            mean, ci = _mean_ci(vals)
            life_stats[key] = {"label": label, "mean": mean, "ci": ci}

        by_schedule[schedule] = {
            "peak_burst": {"mean": peak_mean, "ci": peak_ci},
            "reversion_auc": {"mean": auc_mean, "ci": auc_ci},
            "other_end": {"mean": other_mean, "ci": other_ci},
            "life": life_stats,
        }

    return {"schedules": schedules, "by_schedule": by_schedule}


def _build_gradient_curves(
    grouped: dict[str, list[dict[str, Any]]],
    grad_records: list[dict[str, Any]],
    burst_mode: str,
) -> dict[str, Any]:
    grouped_grad = _group_grad_records(grad_records)
    payload: dict[str, Any] = {}
    for schedule, runs in grouped_grad.items():
        if schedule not in grouped:
            continue
        burst_steps = burst_steps_for_mode(
            schedule, burst_mode, grouped[schedule][0]["config"]["total_steps"]
        )
        series = _gradient_series_for_schedule(runs)
        if not series:
            continue
        payload[schedule] = {
            "burst_steps": burst_steps,
            **series,
        }
    return payload


def _gradient_series_for_schedule(runs: list[dict[str, Any]]) -> dict[str, Any]:
    runs = [run for run in runs if run["grad_sim_log"].get("step")]
    if not runs:
        return {}

    first_steps = np.array(runs[0]["grad_sim_log"]["step"], dtype=float)
    cosine_arr = []
    burst_norm_arr = []
    other_norm_arr = []
    burst_l1_arr = []
    other_l1_arr = []
    signed_dot_arr = []
    interference_power_arr = []

    for run in runs:
        steps = np.array(run["grad_sim_log"]["step"], dtype=float)
        cosine = np.array(run["grad_sim_log"].get("burst_vs_other", []), dtype=float)
        projection = run.get("grad_projection_log") or run["grad_sim_log"].get(
            "grad_projection", {}
        )

        burst_norm = np.array(projection.get("burst_norm", []), dtype=float)
        other_norm = np.array(projection.get("other_norm", []), dtype=float)
        burst_l1 = np.array(projection.get("burst_l1", []), dtype=float)
        other_l1 = np.array(projection.get("other_l1", []), dtype=float)

        if len(cosine) != len(steps):
            continue

        cosine_i = _interpolate_to_reference(first_steps, steps, cosine)
        burst_norm_i = _interpolate_optional_metric(first_steps, steps, burst_norm)
        other_norm_i = _interpolate_optional_metric(first_steps, steps, other_norm)
        burst_l1_i = _interpolate_optional_metric(first_steps, steps, burst_l1)
        other_l1_i = _interpolate_optional_metric(first_steps, steps, other_l1)

        signed_dot_i = burst_norm_i * other_norm_i * cosine_i
        interference_power_i = burst_norm_i * np.maximum(0.0, -cosine_i)

        cosine_arr.append(cosine_i)
        burst_norm_arr.append(burst_norm_i)
        other_norm_arr.append(other_norm_i)
        burst_l1_arr.append(burst_l1_i)
        other_l1_arr.append(other_l1_i)
        signed_dot_arr.append(signed_dot_i)
        interference_power_arr.append(interference_power_i)

    if not cosine_arr:
        return {}

    return {
        "steps": first_steps.tolist(),
        "cosine": _bundle_metric(cosine_arr),
        "burst_norm": _bundle_metric(burst_norm_arr),
        "other_norm": _bundle_metric(other_norm_arr),
        "burst_l1": _bundle_metric(burst_l1_arr),
        "other_l1": _bundle_metric(other_l1_arr),
        "signed_dot": _bundle_metric(signed_dot_arr),
        "interference_power": _bundle_metric(interference_power_arr),
    }


def _bundle_metric(series_list: list[np.ndarray]) -> dict[str, Any]:
    arr = np.stack(series_list)
    return {
        "mean": np.nanmean(arr, axis=0).tolist(),
        "ci": _series_ci(arr).tolist(),
    }
