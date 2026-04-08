"""Bundle run artefacts (config, training logs, metrics) into a single dict."""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from burst.config import (
    ACC_BURST,
    ACC_OTHER,
    LOSS_BURST,
    LOSS_OTHER,
    PHASE_REVERSION,
    TrainConfig,
    burst_steps_for_mode,
    ordered_schedules,
    reversion_life_key,
    reversion_life_label,
)
from burst.core.representation import build_representation_summary
from burst.core.train_utils import (
    compute_lr_schedule,
    load_results,
    mean_ci,
    n_target_for_step,
    resolve_run_paths,
)

BUNDLE_SCHEMA_VERSION = 2
BUNDLE_DIRNAME = "chart_bundle"
BUNDLE_VERSION_DIR = "v1"
BUNDLE_FILENAME = "core_bundle.json"


def bundle_dir(run_dir: str | Path) -> Path:
    """Return the versioned bundle directory for a run."""
    _, _, results_dir = resolve_run_paths(run_dir)
    return results_dir / BUNDLE_DIRNAME / BUNDLE_VERSION_DIR


def bundle_path(run_dir: str | Path) -> Path:
    """Return the path to the core bundle JSON file."""
    return bundle_dir(run_dir) / BUNDLE_FILENAME


def load_core_bundle(run_dir: str | Path) -> dict[str, Any]:
    """Load and return the core bundle dict from disk."""
    path = bundle_path(run_dir)
    with path.open() as f:
        return json.load(f)


def build_and_save_core_bundle(run_dir: str | Path) -> Path:
    """Build the core bundle and write it to disk."""
    bundle = build_core_bundle(run_dir)
    out_dir = bundle_dir(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / BUNDLE_FILENAME
    with path.open("w") as f:
        json.dump(bundle, f, indent=2)
    return path


def build_core_bundle(run_dir: str | Path) -> dict[str, Any]:
    """Assemble the full core bundle dict from training results."""
    results, cfg = load_results(run_dir)

    grouped = group_records_by_schedule(results)
    schedules = ordered_schedules(grouped.keys())
    thresholds = list(TrainConfig().reversion_thresholds)
    burst_mode = cfg["burst_mode"]
    grad_records = load_grad_sim_records(run_dir)

    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "run_name": Path(run_dir).name,
        "config": {
            "burst_mode": burst_mode,
            "base_cfg": cfg["base_cfg"],
            "thresholds": thresholds,
            "schedules": schedules,
        },
        "schedule_bars": build_schedule_bars_payload(grouped),
        "lr_curves": build_lr_curves(grouped),
        "training": build_training_curves(grouped),
        "summary": build_summary(grouped, thresholds),
        "gradients": build_gradient_curves(grouped, grad_records, burst_mode),
        "per_layer_gradients": build_per_layer_gradient_curves(grouped, grad_records, burst_mode),
        "weight_drift": build_weight_drift(run_dir),
        "representation": build_representation_summary(run_dir, grouped),
    }


def load_grad_sim_records(run_dir: str | Path) -> list[dict[str, Any]]:
    """Load gradient cosine similarity records from JSON or pickle files."""
    run_dir = Path(run_dir)
    records: list[dict[str, Any]] = []
    _, _, results_dir = resolve_run_paths(run_dir)

    for grad_dir in (results_dir / "grad_cosine_sim", run_dir / "grad_cosine_sim"):
        if not grad_dir.is_dir():
            continue
        for path in sorted(grad_dir.glob("*.json")):
            with path.open() as f:
                records.append(json.load(f))
        if records:
            return records

    _, logs_dir, _ = resolve_run_paths(run_dir)
    for path in (logs_dir / "all_results.pkl", run_dir / "all_results.pkl"):
        if not path.exists():
            continue
        with path.open("rb") as f:
            results = pickle.load(f)  # noqa: S301
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
                    "pairwise_snapshots": result.get("pairwise_snapshots", []),
                }
            )
        if records:
            return records

    return records


def group_records_by_schedule(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group dicts by their 'schedule' key in canonical order."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["schedule"]].append(record)
    return {schedule: grouped[schedule] for schedule in ordered_schedules(grouped.keys())}


def interpolate_to_reference(
    reference_steps: np.ndarray,
    source_steps: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Interpolate values onto reference_steps grid."""
    if len(reference_steps) == len(source_steps) and np.allclose(reference_steps, source_steps):
        return values.astype(float)
    return np.interp(reference_steps, source_steps, values).astype(float)


def interpolate_optional_metric(
    reference_steps: np.ndarray,
    source_steps: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Interpolate a metric that may be missing, returning NaN if length mismatches."""
    if len(values) != len(source_steps):
        return np.full_like(reference_steps, np.nan, dtype=float)
    return interpolate_to_reference(reference_steps, source_steps, values)


def aggregate_series(
    runs: list[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    """Aggregate a time-series metric across runs into mean and CI."""
    first_steps = np.array(runs[0]["log"]["step"], dtype=float)
    stacked = []
    for run in runs:
        steps = np.array(run["log"]["step"], dtype=float)
        values = np.array(run["log"].get(key, [float("nan")] * len(steps)), dtype=float)
        stacked.append(interpolate_to_reference(first_steps, steps, values))
    arr = np.stack(stacked)
    return {
        "steps": first_steps.tolist(),
        "mean": np.nanmean(arr, axis=0).tolist(),
        "ci": series_ci(arr).tolist(),
    }


def series_ci(arr: np.ndarray) -> np.ndarray:
    """Return per-timestep 95% CI across the first axis."""
    if arr.shape[0] <= 1:
        return np.nanstd(arr, axis=0)
    return 1.96 * np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0])


def build_schedule_bars_payload(grouped: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Build per-schedule burst fraction time-series for bar charts."""
    payload: dict[str, Any] = {}
    for schedule, runs in grouped.items():
        cfg = runs[0]["config"]
        pre_steps = int(cfg["pre_burst_steps"])
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


def build_lr_curves(grouped: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Build per-schedule learning rate curves."""
    payload: dict[str, Any] = {}
    for schedule, runs in grouped.items():
        cfg = runs[0]["config"]
        steps, lrs = compute_lr_schedule(cfg)
        payload[schedule] = {"steps": steps.tolist(), "lr": lrs.tolist()}
    return payload


def build_training_curves(grouped: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Build per-schedule aggregated training curves (acc, loss)."""
    payload: dict[str, Any] = {}
    for schedule, runs in grouped.items():
        cfg = runs[0]["config"]
        payload[schedule] = {
            "pre_steps": int(cfg["pre_burst_steps"]),
            "burst_steps": int(cfg["total_steps"]),
            "reversion_steps": int(cfg["reversion_steps"]),
            ACC_BURST: aggregate_series(runs, ACC_BURST),
            ACC_OTHER: aggregate_series(runs, ACC_OTHER),
            "loss": aggregate_series(runs, "loss"),
            LOSS_BURST: aggregate_series(runs, LOSS_BURST),
            LOSS_OTHER: aggregate_series(runs, LOSS_OTHER),
        }
    return payload


def reversion_auc_for_key(run: dict[str, Any], key: str) -> float:
    """Compute AUC of *key* over the reversion phase via trapezoid rule."""
    log = run["log"]
    burst_end = run["pre_burst_steps"] + run["config"]["total_steps"]
    rev_vals = [log[key][i] for i, ph in enumerate(log["phase"]) if ph == PHASE_REVERSION]
    rev_steps = [
        log["step"][i] - burst_end for i, ph in enumerate(log["phase"]) if ph == PHASE_REVERSION
    ]
    if len(rev_vals) < 2:  # noqa: PLR2004
        return 0.0
    return float(np.trapezoid(rev_vals, rev_steps))


def build_summary(
    grouped: dict[str, list[dict[str, Any]]],
    thresholds: list[float],
) -> dict[str, Any]:
    """Build per-schedule summary statistics (peak, AUC, life times)."""
    schedules = list(grouped.keys())
    by_schedule: dict[str, Any] = {}
    for schedule, runs in grouped.items():
        peak_vals = np.array([run["peak_burst"] for run in runs], dtype=float)
        auc_vals = np.array([run["reversion_auc"] for run in runs], dtype=float)
        other_end_vals = np.array([run["log"][ACC_OTHER][-1] for run in runs], dtype=float)
        peak_mean, peak_ci = mean_ci(peak_vals)
        auc_mean, auc_ci = mean_ci(auc_vals)
        other_mean, other_ci = mean_ci(other_end_vals)

        auc_loss_burst = np.array([reversion_auc_for_key(r, LOSS_BURST) for r in runs], dtype=float)
        auc_acc_other = np.array([reversion_auc_for_key(r, ACC_OTHER) for r in runs], dtype=float)
        auc_lb_mean, auc_lb_ci = mean_ci(auc_loss_burst)
        auc_ao_mean, auc_ao_ci = mean_ci(auc_acc_other)

        life_stats: dict[str, Any] = {}
        for threshold in thresholds:
            key = reversion_life_key(threshold)
            label = reversion_life_label(threshold)
            vals = np.array(
                [run.get(key, runs[0]["config"]["reversion_steps"]) for run in runs], dtype=float
            )
            mean, ci = mean_ci(vals)
            life_stats[key] = {"label": label, "mean": mean, "ci": ci}

        by_schedule[schedule] = {
            "peak_burst": {"mean": peak_mean, "ci": peak_ci},
            "reversion_auc": {"mean": auc_mean, "ci": auc_ci},
            "reversion_auc_loss_burst": {"mean": auc_lb_mean, "ci": auc_lb_ci},
            "reversion_auc_acc_other": {"mean": auc_ao_mean, "ci": auc_ao_ci},
            "other_end": {"mean": other_mean, "ci": other_ci},
            "life": life_stats,
        }

    return {"schedules": schedules, "by_schedule": by_schedule}


def build_gradient_curves(
    grouped: dict[str, list[dict[str, Any]]],
    grad_records: list[dict[str, Any]],
    burst_mode: str,
) -> dict[str, Any]:
    """Build per-schedule gradient metric curves (cosine, norms, etc.)."""
    grouped_grad = group_records_by_schedule(grad_records)
    payload: dict[str, Any] = {}
    for schedule, runs in grouped_grad.items():
        if schedule not in grouped:
            continue
        burst_steps = burst_steps_for_mode(
            schedule, burst_mode, grouped[schedule][0]["config"]["total_steps"]
        )
        series = gradient_series_for_schedule(runs)
        if not series:
            continue
        payload[schedule] = {
            "burst_steps": burst_steps,
            **series,
        }
    return payload


def gradient_series_for_schedule(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate gradient series across seeds for one schedule."""
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

        cosine_i = interpolate_to_reference(first_steps, steps, cosine)
        burst_norm_i = interpolate_optional_metric(first_steps, steps, burst_norm)
        other_norm_i = interpolate_optional_metric(first_steps, steps, other_norm)
        burst_l1_i = interpolate_optional_metric(first_steps, steps, burst_l1)
        other_l1_i = interpolate_optional_metric(first_steps, steps, other_l1)

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
        "cosine": bundle_metric(cosine_arr),
        "burst_norm": bundle_metric(burst_norm_arr),
        "other_norm": bundle_metric(other_norm_arr),
        "burst_l1": bundle_metric(burst_l1_arr),
        "other_l1": bundle_metric(other_l1_arr),
        "signed_dot": bundle_metric(signed_dot_arr),
        "interference_power": bundle_metric(interference_power_arr),
    }


def bundle_metric(series_list: list[np.ndarray]) -> dict[str, Any]:
    """Stack series and return mean + CI dict."""
    arr = np.stack(series_list)
    return {
        "mean": np.nanmean(arr, axis=0).tolist(),
        "ci": series_ci(arr).tolist(),
    }


# ---------------------------------------------------------------------------
# Per-layer gradient curves
# ---------------------------------------------------------------------------


def build_per_layer_gradient_curves(
    grouped: dict[str, list[dict[str, Any]]],
    grad_records: list[dict[str, Any]],
    burst_mode: str,
) -> dict[str, Any]:
    """Build per-layer gradient metric curves (cosine, norms) per schedule."""
    grouped_grad = group_records_by_schedule(grad_records)
    payload: dict[str, Any] = {}
    for schedule, runs in grouped_grad.items():
        if schedule not in grouped:
            continue
        burst_steps = burst_steps_for_mode(
            schedule, burst_mode, grouped[schedule][0]["config"]["total_steps"]
        )
        series = per_layer_series_for_schedule(runs)
        if not series:
            continue
        payload[schedule] = {"burst_steps": burst_steps, **series}
    return payload


def accumulate_per_layer_metric(
    runs: list[dict[str, Any]],
    layer_names: list[str],
    first_steps: np.ndarray,
    gsl_key: str,
) -> dict[str, list[np.ndarray]]:
    """Accumulate one per-layer metric across runs, interpolated to first_steps."""
    acc: dict[str, list[np.ndarray]] = {ln: [] for ln in layer_names}
    for run in runs:
        gsl = run["grad_sim_log"]
        steps = np.array(gsl["step"], dtype=float)
        layer_dict = gsl.get(gsl_key, {})
        for ln in layer_names:
            vals = np.array(layer_dict.get(ln, []), dtype=float)
            acc[ln].append(interpolate_optional_metric(first_steps, steps, vals))
    return acc


def per_layer_series_for_schedule(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-layer gradient series across seeds for one schedule."""
    runs = [r for r in runs if r["grad_sim_log"].get("step")]
    if not runs:
        return {}

    gsl0 = runs[0]["grad_sim_log"]
    layer_names: list[str] = gsl0.get("layer_names", [])
    if not layer_names or not gsl0.get("per_layer"):
        return {}

    first_steps = np.array(gsl0["step"], dtype=float)

    cosine_by_layer = accumulate_per_layer_metric(runs, layer_names, first_steps, "per_layer")
    burst_norm_by_layer = accumulate_per_layer_metric(
        runs, layer_names, first_steps, "burst_norm_per_layer"
    )
    other_norm_by_layer = accumulate_per_layer_metric(
        runs, layer_names, first_steps, "other_norm_per_layer"
    )

    has_any_cosine = any(
        cosine_by_layer[ln] and not np.all(np.isnan(cosine_by_layer[ln][0])) for ln in layer_names
    )
    if not has_any_cosine:
        return {}

    cosine_out: dict[str, Any] = {}
    burst_norm_out: dict[str, Any] = {}
    other_norm_out: dict[str, Any] = {}
    norm_x_cosine_out: dict[str, Any] = {}

    for ln in layer_names:
        if cosine_by_layer[ln]:
            cosine_out[ln] = bundle_metric(cosine_by_layer[ln])
        if burst_norm_by_layer[ln]:
            burst_norm_out[ln] = bundle_metric(burst_norm_by_layer[ln])
        if other_norm_by_layer[ln]:
            other_norm_out[ln] = bundle_metric(other_norm_by_layer[ln])
        if cosine_by_layer[ln] and burst_norm_by_layer[ln]:
            nxc = [
                bn * cs for bn, cs in zip(burst_norm_by_layer[ln], cosine_by_layer[ln], strict=True)
            ]
            norm_x_cosine_out[ln] = bundle_metric(nxc)

    return {
        "layer_names": layer_names,
        "steps": first_steps.tolist(),
        "cosine": cosine_out,
        "burst_norm": burst_norm_out,
        "other_norm": other_norm_out,
        "norm_x_cosine": norm_x_cosine_out,
    }


# ---------------------------------------------------------------------------
# Weight drift from checkpoints
# ---------------------------------------------------------------------------


def build_weight_drift(run_dir: str | Path) -> dict[str, Any]:
    """Compute per-layer weight drift from saved checkpoints."""
    run_dir = Path(run_dir)
    _, logs_dir, results_dir = resolve_run_paths(run_dir)

    ckpt_root = logs_dir / "checkpoints"
    if not ckpt_root.is_dir():
        return {}

    cfg_path = results_dir / "config.json"
    if not cfg_path.exists():
        cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return {}

    with cfg_path.open() as f:
        cfg = json.load(f)
    n_layer: int = cfg["base_cfg"]["n_layer"]

    label_dirs = sorted(d for d in ckpt_root.iterdir() if d.is_dir())
    if not label_dirs:
        return {}

    by_schedule: dict[str, list[Path]] = defaultdict(list)
    for d in label_dirs:
        parts = d.name.rsplit("_s", maxsplit=1)
        if len(parts) == 2:  # noqa: PLR2004
            by_schedule[parts[0]].append(d)

    payload: dict[str, Any] = {}
    for schedule in ordered_schedules(by_schedule.keys()):
        result = weight_drift_for_schedule(by_schedule[schedule], n_layer)
        if result:
            payload[schedule] = result
    return payload


def weight_drift_for_schedule(seed_dirs: list[Path], n_layer: int) -> dict[str, Any] | None:
    """Compute weight drift arrays for one schedule across seeds."""
    seed_cumulative: list[np.ndarray] = []
    seed_stepwise: list[np.ndarray] = []
    layer_names: list[str] | None = None
    steps: list[int] | None = None

    for ckpt_dir in seed_dirs:
        ckpt_files = {int(p.stem.split("_")[1]): p for p in ckpt_dir.glob("step_*.pt")}
        if not ckpt_files:
            continue

        sorted_steps = sorted(ckpt_files)
        base_sd = load_state_dict_cpu(ckpt_files[sorted_steps[0]])
        groups = layer_groups_from_state_dict(base_sd, n_layer)
        if layer_names is None:
            layer_names = [g[0] for g in groups]
            steps = sorted_steps

        cum, stepwise = weight_drift_arrays(ckpt_files, sorted_steps, groups, base_sd)
        seed_cumulative.append(cum)
        seed_stepwise.append(stepwise)

    if not seed_cumulative or layer_names is None or steps is None:
        return None

    cum_stack = np.stack(seed_cumulative)
    step_stack = np.stack(seed_stepwise)
    cum_out: dict[str, Any] = {}
    stepwise_out: dict[str, Any] = {}
    for li, ln in enumerate(layer_names):
        cum_out[ln] = {
            "mean": np.nanmean(cum_stack[:, li, :], axis=0).tolist(),
            "ci": series_ci(cum_stack[:, li, :]).tolist(),
        }
        stepwise_out[ln] = {
            "mean": np.nanmean(step_stack[:, li, :], axis=0).tolist(),
            "ci": series_ci(step_stack[:, li, :]).tolist(),
        }

    return {
        "layer_names": layer_names,
        "steps": steps,
        "cumulative": cum_out,
        "stepwise": stepwise_out,
    }


def load_state_dict_cpu(path: Path) -> dict[str, Any]:
    """Load a state dict to CPU as float32."""
    return {
        k: v.float().cpu()
        for k, v in torch.load(str(path), map_location="cpu", weights_only=True).items()
    }


def weight_drift_arrays(
    ckpt_files: dict[int, Path],
    sorted_steps: list[int],
    groups: list[tuple[str, list[str]]],
    base_sd: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cumulative and stepwise drift arrays for one seed."""
    n_layers = len(groups)
    n_ckpts = len(sorted_steps)
    cum = np.zeros((n_layers, n_ckpts))
    stepwise = np.zeros((n_layers, n_ckpts))
    prev_sd = base_sd

    for ci, step in enumerate(sorted_steps):
        path = ckpt_files[step]
        if not path.exists():
            cum[:, ci] = np.nan
            stepwise[:, ci] = np.nan
            continue
        sd = load_state_dict_cpu(path)
        for li, (_gname, pnames) in enumerate(groups):
            cum[li, ci] = sum((sd[p] - base_sd[p]).norm().item() ** 2 for p in pnames) ** 0.5
            stepwise[li, ci] = sum((sd[p] - prev_sd[p]).norm().item() ** 2 for p in pnames) ** 0.5
        prev_sd = sd

    return cum, stepwise


def layer_groups_from_state_dict(sd: dict[str, Any], n_layer: int) -> list[tuple[str, list[str]]]:
    """Build ordered layer groups from a state dict."""
    all_keys = set(sd.keys())
    groups: list[tuple[str, list[str]]] = []
    emb = sorted(k for k in all_keys if k in ("transformer.wte.weight", "transformer.wpe.weight"))
    if emb:
        groups.append(("emb", emb))
    for i in range(n_layer):
        pfx = f"transformer.h.{i}"
        for tag, sub in [("ln", "ln_"), ("attn", "attn."), ("mlp", "mlp.")]:
            params = sorted(k for k in all_keys if k.startswith(f"{pfx}.{sub}"))
            if params:
                groups.append((f"L{i}_{tag}", params))
    lnf = sorted(k for k in all_keys if k.startswith("transformer.ln_f"))
    if lnf:
        groups.append(("ln_f", lnf))
    return groups
