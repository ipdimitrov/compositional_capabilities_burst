"""Bundle run artefacts into a typed CoreBundle for charting."""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from burst.config import (
    ACC_BURST,
    ACC_OTHER,
    LOSS_BURST,
    LOSS_OTHER,
    N_REPRESENTATION_DOCS_PER_CLASS,
    PHASE_REVERSION,
    TrainConfig,
    burst_steps_for_mode,
    ordered_schedules,
    reversion_life_key,
    reversion_life_label,
)
from burst.core.metrics.probes import load_probe_records
from burst.core.representation import build_representation_summary
from burst.core.train_utils import (
    compute_lr_schedule,
    load_results,
    mean_ci,
    n_target_for_step,
    resolve_run_paths,
)

# ---------------------------------------------------------------------------
# Schema dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MeanCI:
    """Scalar mean with 95% CI."""

    mean: float
    ci: float


@dataclass
class SeriesMeanCI:
    """Time-series mean with 95% CI."""

    mean: list[float]
    ci: list[float]


@dataclass
class BundleConfig:
    """Top-level experiment configuration."""

    burst_mode: str
    base_cfg: dict[str, Any]
    thresholds: list[float]
    schedules: list[str]


@dataclass
class ScheduleBars:
    """Per-schedule burst fraction time-series."""

    fractions: list[float]
    pre_steps: int
    burst_steps: int
    reversion_steps: int


@dataclass
class LrCurve:
    """Learning rate schedule."""

    steps: list[float]
    lr: list[float]


@dataclass
class TrainingSeries:
    """Aggregated metric time-series with steps."""

    steps: list[float]
    mean: list[float]
    ci: list[float]


@dataclass
class TrainingSchedule:
    """All training metrics for one schedule."""

    pre_steps: int
    burst_steps: int
    reversion_steps: int
    acc_burst: TrainingSeries
    acc_other: TrainingSeries
    loss: TrainingSeries
    loss_burst: TrainingSeries
    loss_other: TrainingSeries


@dataclass
class LifeEntry:
    """Reversion half-life at one threshold."""

    label: str
    mean: float
    ci: float


@dataclass
class ScheduleSummary:
    """Summary statistics for one schedule."""

    peak_burst: MeanCI
    reversion_auc: MeanCI
    reversion_auc_loss_burst: MeanCI
    reversion_auc_loss_other: MeanCI
    reversion_auc_acc_other: MeanCI
    other_end: MeanCI
    life: dict[str, LifeEntry]


@dataclass
class Summary:
    """Summary section of the bundle."""

    schedules: list[str]
    by_schedule: dict[str, ScheduleSummary]


@dataclass
class GradientSchedule:
    """Aggregated gradient metrics for one schedule."""

    burst_steps: int
    steps: list[float]
    cosine: SeriesMeanCI
    burst_norm: SeriesMeanCI
    other_norm: SeriesMeanCI
    burst_l1: SeriesMeanCI
    other_l1: SeriesMeanCI
    signed_dot: SeriesMeanCI
    interference_power: SeriesMeanCI
    grad_rank: SeriesMeanCI | None = None


@dataclass
class PerLayerGradientSchedule:
    """Per-layer gradient metrics for one schedule."""

    burst_steps: int
    layer_names: list[str]
    steps: list[float]
    cosine: dict[str, SeriesMeanCI]
    burst_norm: dict[str, SeriesMeanCI]
    other_norm: dict[str, SeriesMeanCI]
    norm_x_cosine: dict[str, SeriesMeanCI]


@dataclass
class WeightDriftSchedule:
    """Per-layer weight drift metrics for one schedule."""

    layer_names: list[str]
    steps: list[int]
    cumulative: dict[str, SeriesMeanCI]
    stepwise: dict[str, SeriesMeanCI]


@dataclass
class RepresentationSeed:
    """Per-seed representation scalars."""

    seed: float
    late_centroid_projection: float
    late_other_shift_norm: float
    late_drift_cosine: float
    late_burst_self_projection: float
    late_burst_shift_norm: float
    late_burst_post_norm: float
    late_burst_pre_norm: float
    late_other_post_norm: float
    late_other_pre_norm: float


@dataclass
class RepresentationSchedule:
    """Aggregated representation metrics for one schedule."""

    late_centroid_projection: MeanCI
    late_other_shift_norm: MeanCI
    late_drift_cosine: MeanCI
    late_burst_self_projection: MeanCI
    late_burst_shift_norm: MeanCI
    late_burst_post_norm: MeanCI
    late_burst_pre_norm: MeanCI
    late_other_post_norm: MeanCI
    late_other_pre_norm: MeanCI
    per_seed: list[RepresentationSeed]


@dataclass
class Representation:
    """Top-level representation section."""

    by_schedule: dict[str, RepresentationSchedule] = field(default_factory=dict)


@dataclass
class ProbeRegime:
    """Probe accuracy for one regime (Other / Burst / diff)."""

    Other: SeriesMeanCI
    Burst: SeriesMeanCI
    diff: SeriesMeanCI


@dataclass
class CoreBundle:
    """The complete typed core bundle."""

    run_name: str
    config: BundleConfig
    schedule_bars: dict[str, ScheduleBars]
    lr_curves: dict[str, LrCurve]
    training: dict[str, TrainingSchedule]
    summary: Summary
    gradients: dict[str, GradientSchedule]
    per_layer_gradients: dict[str, PerLayerGradientSchedule]
    weight_drift: dict[str, WeightDriftSchedule]
    representation: Representation
    next_token_probes: dict[str, dict[str, dict[str, ProbeRegime]]]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CoreBundle:
        """Deserialize from a plain dict."""
        config = BundleConfig(**data["config"])
        schedule_bars = {k: ScheduleBars(**v) for k, v in data["schedule_bars"].items()}
        lr_curves = {k: LrCurve(**v) for k, v in data["lr_curves"].items()}

        training: dict[str, TrainingSchedule] = {}
        for k, v in data["training"].items():
            training[k] = TrainingSchedule(
                pre_steps=v["pre_steps"],
                burst_steps=v["burst_steps"],
                reversion_steps=v["reversion_steps"],
                acc_burst=TrainingSeries(**v["acc_burst"]),
                acc_other=TrainingSeries(**v["acc_other"]),
                loss=TrainingSeries(**v["loss"]),
                loss_burst=TrainingSeries(**v["loss_burst"]),
                loss_other=TrainingSeries(**v["loss_other"]),
            )

        sd = data["summary"]
        by_sched_summary: dict[str, ScheduleSummary] = {}
        for k, v in sd["by_schedule"].items():
            by_sched_summary[k] = ScheduleSummary(
                peak_burst=MeanCI(**v["peak_burst"]),
                reversion_auc=MeanCI(**v["reversion_auc"]),
                reversion_auc_loss_burst=MeanCI(**v["reversion_auc_loss_burst"]),
                reversion_auc_loss_other=MeanCI(**v["reversion_auc_loss_other"]),
                reversion_auc_acc_other=MeanCI(**v["reversion_auc_acc_other"]),
                other_end=MeanCI(**v["other_end"]),
                life={lk: LifeEntry(**lv) for lk, lv in v["life"].items()},
            )
        summary = Summary(schedules=sd["schedules"], by_schedule=by_sched_summary)

        gradients: dict[str, GradientSchedule] = {}
        for k, v in data["gradients"].items():
            gradients[k] = GradientSchedule(
                burst_steps=v["burst_steps"],
                steps=v["steps"],
                cosine=SeriesMeanCI(**v["cosine"]),
                burst_norm=SeriesMeanCI(**v["burst_norm"]),
                other_norm=SeriesMeanCI(**v["other_norm"]),
                burst_l1=SeriesMeanCI(**v["burst_l1"]),
                other_l1=SeriesMeanCI(**v["other_l1"]),
                signed_dot=SeriesMeanCI(**v["signed_dot"]),
                interference_power=SeriesMeanCI(**v["interference_power"]),
                grad_rank=SeriesMeanCI(**v["grad_rank"]) if v.get("grad_rank") else None,
            )

        plg: dict[str, PerLayerGradientSchedule] = {}
        for k, v in data["per_layer_gradients"].items():
            plg[k] = PerLayerGradientSchedule(
                burst_steps=v["burst_steps"],
                layer_names=v["layer_names"],
                steps=v["steps"],
                cosine={ln: SeriesMeanCI(**d) for ln, d in v["cosine"].items()},
                burst_norm={ln: SeriesMeanCI(**d) for ln, d in v["burst_norm"].items()},
                other_norm={ln: SeriesMeanCI(**d) for ln, d in v["other_norm"].items()},
                norm_x_cosine={ln: SeriesMeanCI(**d) for ln, d in v["norm_x_cosine"].items()},
            )

        wd: dict[str, WeightDriftSchedule] = {}
        for k, v in data["weight_drift"].items():
            wd[k] = WeightDriftSchedule(
                layer_names=v["layer_names"],
                steps=v["steps"],
                cumulative={ln: SeriesMeanCI(**d) for ln, d in v["cumulative"].items()},
                stepwise={ln: SeriesMeanCI(**d) for ln, d in v["stepwise"].items()},
            )

        rep_data = data.get("representation", {})
        rep_by_sched: dict[str, RepresentationSchedule] = {}
        for k, v in rep_data.get("by_schedule", {}).items():
            rep_by_sched[k] = RepresentationSchedule(
                late_centroid_projection=MeanCI(**v["late_centroid_projection"]),
                late_other_shift_norm=MeanCI(**v["late_other_shift_norm"]),
                late_drift_cosine=MeanCI(**v["late_drift_cosine"]),
                late_burst_self_projection=MeanCI(**v["late_burst_self_projection"]),
                late_burst_shift_norm=MeanCI(**v["late_burst_shift_norm"]),
                late_burst_post_norm=MeanCI(**v["late_burst_post_norm"]),
                late_burst_pre_norm=MeanCI(**v["late_burst_pre_norm"]),
                late_other_post_norm=MeanCI(**v["late_other_post_norm"]),
                late_other_pre_norm=MeanCI(**v["late_other_pre_norm"]),
                per_seed=[RepresentationSeed(**s) for s in v["per_seed"]],
            )

        ntp_raw = data.get("next_token_probes", {})
        ntp: dict[str, dict[str, dict[str, ProbeRegime]]] = {}
        for sched_k, steps_dict in ntp_raw.items():
            ntp[sched_k] = {}
            for step_k, methods_dict in steps_dict.items():
                ntp[sched_k][step_k] = {}
                for method_k, regime_dict in methods_dict.items():
                    ntp[sched_k][step_k][method_k] = ProbeRegime(
                        Other=SeriesMeanCI(**regime_dict["Other"]),
                        Burst=SeriesMeanCI(**regime_dict["Burst"]),
                        diff=SeriesMeanCI(**regime_dict["diff"]),
                    )

        return cls(
            run_name=data["run_name"],
            config=config,
            schedule_bars=schedule_bars,
            lr_curves=lr_curves,
            training=training,
            summary=summary,
            gradients=gradients,
            per_layer_gradients=plg,
            weight_drift=wd,
            representation=Representation(by_schedule=rep_by_sched),
            next_token_probes=ntp,
        )


# ---------------------------------------------------------------------------
# Bundle I/O
# ---------------------------------------------------------------------------

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


def load_core_bundle(run_dir: str | Path) -> CoreBundle:
    """Load and return the core bundle from disk."""
    path = bundle_path(run_dir)
    with path.open() as f:
        return CoreBundle.from_dict(json.load(f))


def build_and_save_core_bundle(run_dir: str | Path) -> Path:
    """Build the core bundle and write it to disk."""
    bundle = build_core_bundle(run_dir)
    out = bundle_dir(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / BUNDLE_FILENAME
    with path.open("w") as f:
        json.dump(bundle.to_dict(), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


def build_core_bundle(run_dir: str | Path) -> CoreBundle:
    """Assemble the full core bundle from training results."""
    results, cfg = load_results(run_dir)

    grouped = group_records_by_schedule(results)
    schedules = ordered_schedules(grouped.keys())
    thresholds = list(TrainConfig().reversion_thresholds)
    burst_mode = cfg["burst_mode"]
    grad_records = load_grad_sim_records(run_dir)

    rep_raw = build_representation_summary(Path(run_dir), grouped, N_REPRESENTATION_DOCS_PER_CLASS)
    representation = parse_representation(rep_raw)

    return CoreBundle(
        run_name=Path(run_dir).name,
        config=BundleConfig(
            burst_mode=burst_mode,
            base_cfg=cfg["base_cfg"],
            thresholds=thresholds,
            schedules=schedules,
        ),
        schedule_bars=build_schedule_bars_payload(grouped),
        lr_curves=build_lr_curves(grouped),
        training=build_training_curves(grouped),
        summary=build_summary(grouped, thresholds),
        gradients=build_gradient_curves(grouped, grad_records, burst_mode),
        per_layer_gradients=build_per_layer_gradient_curves(grouped, grad_records, burst_mode),
        weight_drift=build_weight_drift(run_dir),
        representation=representation,
        next_token_probes=build_probe_curves(run_dir),
    )


def parse_representation(raw: dict[str, object]) -> Representation:
    """Convert raw representation dict into a typed Representation."""
    by_schedule_raw = raw.get("by_schedule", {})
    if not by_schedule_raw:
        return Representation()

    by_schedule: dict[str, RepresentationSchedule] = {}
    for k, v in by_schedule_raw.items():
        by_schedule[k] = RepresentationSchedule(
            late_centroid_projection=MeanCI(**v["late_centroid_projection"]),
            late_other_shift_norm=MeanCI(**v["late_other_shift_norm"]),
            late_drift_cosine=MeanCI(**v["late_drift_cosine"]),
            late_burst_self_projection=MeanCI(**v["late_burst_self_projection"]),
            late_burst_shift_norm=MeanCI(**v["late_burst_shift_norm"]),
            late_burst_post_norm=MeanCI(**v["late_burst_post_norm"]),
            late_burst_pre_norm=MeanCI(**v["late_burst_pre_norm"]),
            late_other_post_norm=MeanCI(**v["late_other_post_norm"]),
            late_other_pre_norm=MeanCI(**v["late_other_pre_norm"]),
            per_seed=[RepresentationSeed(**s) for s in v["per_seed"]],
        )
    return Representation(by_schedule=by_schedule)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Aggregation utilities
# ---------------------------------------------------------------------------


def interpolate_to_reference(
    reference_steps: np.ndarray,
    source_steps: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Interpolate values onto the reference_steps grid."""
    if len(reference_steps) == len(source_steps) and np.allclose(reference_steps, source_steps):
        return values.astype(float)
    return np.interp(reference_steps, source_steps, values).astype(float)


def interpolate_optional_metric(
    reference_steps: np.ndarray,
    source_steps: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Interpolate a metric that may be missing; returns NaN on length mismatch."""
    if len(values) != len(source_steps):
        return np.full_like(reference_steps, np.nan, dtype=float)
    return interpolate_to_reference(reference_steps, source_steps, values)


def series_ci(arr: np.ndarray) -> np.ndarray:
    """Return per-timestep 95% CI across the first axis."""
    if arr.shape[0] <= 1:
        return np.nanstd(arr, axis=0)
    return 1.96 * np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0])


def bundle_metric(series_list: list[np.ndarray]) -> SeriesMeanCI:
    """Stack series and return mean + CI."""
    arr = np.stack(series_list)
    return SeriesMeanCI(
        mean=np.nanmean(arr, axis=0).tolist(),
        ci=series_ci(arr).tolist(),
    )


def aggregate_series(runs: list[dict[str, Any]], key: str) -> TrainingSeries:
    """Aggregate a time-series metric across runs into mean and CI."""
    first_steps = np.array(runs[0]["log"]["step"], dtype=float)
    stacked = []
    for run in runs:
        steps = np.array(run["log"]["step"], dtype=float)
        values = np.array(run["log"].get(key, [float("nan")] * len(steps)), dtype=float)
        stacked.append(interpolate_to_reference(first_steps, steps, values))
    arr = np.stack(stacked)
    return TrainingSeries(
        steps=first_steps.tolist(),
        mean=np.nanmean(arr, axis=0).tolist(),
        ci=series_ci(arr).tolist(),
    )


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def build_schedule_bars_payload(
    grouped: dict[str, list[dict[str, Any]]],
) -> dict[str, ScheduleBars]:
    """Build per-schedule burst fraction time-series."""
    payload: dict[str, ScheduleBars] = {}
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

        payload[schedule] = ScheduleBars(
            fractions=fractions,
            pre_steps=pre_steps,
            burst_steps=burst_steps,
            reversion_steps=reversion_steps,
        )
    return payload


def build_lr_curves(grouped: dict[str, list[dict[str, Any]]]) -> dict[str, LrCurve]:
    """Build per-schedule learning rate curves."""
    payload: dict[str, LrCurve] = {}
    for schedule, runs in grouped.items():
        cfg = runs[0]["config"]
        steps, lrs = compute_lr_schedule(cfg)
        payload[schedule] = LrCurve(steps=steps.tolist(), lr=lrs.tolist())
    return payload


def build_training_curves(
    grouped: dict[str, list[dict[str, Any]]],
) -> dict[str, TrainingSchedule]:
    """Build per-schedule aggregated training curves."""
    payload: dict[str, TrainingSchedule] = {}
    for schedule, runs in grouped.items():
        cfg = runs[0]["config"]
        payload[schedule] = TrainingSchedule(
            pre_steps=int(cfg["pre_burst_steps"]),
            burst_steps=int(cfg["total_steps"]),
            reversion_steps=int(cfg["reversion_steps"]),
            acc_burst=aggregate_series(runs, ACC_BURST),
            acc_other=aggregate_series(runs, ACC_OTHER),
            loss=aggregate_series(runs, "loss"),
            loss_burst=aggregate_series(runs, LOSS_BURST),
            loss_other=aggregate_series(runs, LOSS_OTHER),
        )
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
) -> Summary:
    """Build per-schedule summary statistics."""
    schedules = list(grouped.keys())
    by_schedule: dict[str, ScheduleSummary] = {}
    for schedule, runs in grouped.items():
        peak_vals = np.array([run["peak_burst"] for run in runs], dtype=float)
        auc_vals = np.array([run["reversion_auc"] for run in runs], dtype=float)
        other_end_vals = np.array([run["log"][ACC_OTHER][-1] for run in runs], dtype=float)
        peak_mean, peak_ci = mean_ci(peak_vals)
        auc_mean, auc_ci = mean_ci(auc_vals)
        other_mean, other_ci = mean_ci(other_end_vals)

        auc_loss_burst = np.array([reversion_auc_for_key(r, LOSS_BURST) for r in runs], dtype=float)
        auc_loss_other = np.array([reversion_auc_for_key(r, LOSS_OTHER) for r in runs], dtype=float)
        auc_acc_other = np.array([reversion_auc_for_key(r, ACC_OTHER) for r in runs], dtype=float)
        auc_lb_mean, auc_lb_ci = mean_ci(auc_loss_burst)
        auc_lo_mean, auc_lo_ci = mean_ci(auc_loss_other)
        auc_ao_mean, auc_ao_ci = mean_ci(auc_acc_other)

        life_stats: dict[str, LifeEntry] = {}
        for threshold in thresholds:
            key = reversion_life_key(threshold)
            label = reversion_life_label(threshold)
            vals = np.array(
                [run.get(key, runs[0]["config"]["reversion_steps"]) for run in runs], dtype=float
            )
            life_mean, life_ci = mean_ci(vals)
            life_stats[key] = LifeEntry(label=label, mean=life_mean, ci=life_ci)

        by_schedule[schedule] = ScheduleSummary(
            peak_burst=MeanCI(mean=peak_mean, ci=peak_ci),
            reversion_auc=MeanCI(mean=auc_mean, ci=auc_ci),
            reversion_auc_loss_burst=MeanCI(mean=auc_lb_mean, ci=auc_lb_ci),
            reversion_auc_loss_other=MeanCI(mean=auc_lo_mean, ci=auc_lo_ci),
            reversion_auc_acc_other=MeanCI(mean=auc_ao_mean, ci=auc_ao_ci),
            other_end=MeanCI(mean=other_mean, ci=other_ci),
            life=life_stats,
        )

    return Summary(schedules=schedules, by_schedule=by_schedule)


def build_gradient_curves(
    grouped: dict[str, list[dict[str, Any]]],
    grad_records: list[dict[str, Any]],
    burst_mode: str,
) -> dict[str, GradientSchedule]:
    """Build per-schedule gradient metric curves."""
    grouped_grad = group_records_by_schedule(grad_records)
    payload: dict[str, GradientSchedule] = {}
    for schedule, runs in grouped_grad.items():
        if schedule not in grouped:
            continue
        bs = burst_steps_for_mode(
            schedule, burst_mode, grouped[schedule][0]["config"]["total_steps"]
        )
        result = gradient_series_for_schedule(runs, bs)
        if result is not None:
            payload[schedule] = result
    return payload


def gradient_series_for_schedule(
    runs: list[dict[str, Any]], burst_steps: int
) -> GradientSchedule | None:
    """Aggregate gradient series across seeds for one schedule."""
    runs = [run for run in runs if run["grad_sim_log"].get("step")]
    if not runs:
        return None

    first_steps = np.array(runs[0]["grad_sim_log"]["step"], dtype=float)
    cosine_arr: list[np.ndarray] = []
    burst_norm_arr: list[np.ndarray] = []
    other_norm_arr: list[np.ndarray] = []
    burst_l1_arr: list[np.ndarray] = []
    other_l1_arr: list[np.ndarray] = []
    signed_dot_arr: list[np.ndarray] = []
    interference_power_arr: list[np.ndarray] = []

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

        cosine_arr.append(cosine_i)
        burst_norm_arr.append(burst_norm_i)
        other_norm_arr.append(other_norm_i)
        burst_l1_arr.append(burst_l1_i)
        other_l1_arr.append(other_l1_i)
        signed_dot_arr.append(burst_norm_i * other_norm_i * cosine_i)
        interference_power_arr.append(burst_norm_i * np.maximum(0.0, -cosine_i))

    if not cosine_arr:
        return None

    return GradientSchedule(
        burst_steps=burst_steps,
        steps=first_steps.tolist(),
        cosine=bundle_metric(cosine_arr),
        burst_norm=bundle_metric(burst_norm_arr),
        other_norm=bundle_metric(other_norm_arr),
        burst_l1=bundle_metric(burst_l1_arr),
        other_l1=bundle_metric(other_l1_arr),
        signed_dot=bundle_metric(signed_dot_arr),
        interference_power=bundle_metric(interference_power_arr),
        grad_rank=aggregate_grad_rank(runs, first_steps),
    )


def aggregate_grad_rank(runs: list[dict[str, Any]], first_steps: np.ndarray) -> SeriesMeanCI | None:
    """Average effective rank across layers per step, then aggregate across seeds."""
    rank_arr: list[np.ndarray] = []
    for run in runs:
        rank_dict = run["grad_sim_log"].get("grad_rank", {})
        if not rank_dict:
            continue
        steps = np.array(run["grad_sim_log"]["step"], dtype=float)
        layer_series = [np.array(vals, dtype=float) for vals in rank_dict.values()]
        if not layer_series or any(len(s) != len(steps) for s in layer_series):
            continue
        mean_across_layers = np.nanmean(np.stack(layer_series), axis=0)
        rank_arr.append(interpolate_to_reference(first_steps, steps, mean_across_layers))
    if not rank_arr:
        return None
    return bundle_metric(rank_arr)


def build_per_layer_gradient_curves(
    grouped: dict[str, list[dict[str, Any]]],
    grad_records: list[dict[str, Any]],
    burst_mode: str,
) -> dict[str, PerLayerGradientSchedule]:
    """Build per-layer gradient metric curves per schedule."""
    grouped_grad = group_records_by_schedule(grad_records)
    payload: dict[str, PerLayerGradientSchedule] = {}
    for schedule, runs in grouped_grad.items():
        if schedule not in grouped:
            continue
        bs = burst_steps_for_mode(
            schedule, burst_mode, grouped[schedule][0]["config"]["total_steps"]
        )
        result = per_layer_series_for_schedule(runs, bs)
        if result is not None:
            payload[schedule] = result
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


def per_layer_series_for_schedule(
    runs: list[dict[str, Any]], burst_steps: int
) -> PerLayerGradientSchedule | None:
    """Aggregate per-layer gradient series across seeds for one schedule."""
    runs = [r for r in runs if r["grad_sim_log"].get("step")]
    if not runs:
        return None

    gsl0 = runs[0]["grad_sim_log"]
    layer_names: list[str] = gsl0.get("layer_names", [])
    if not layer_names or not gsl0.get("per_layer"):
        return None

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
        return None

    cosine_out: dict[str, SeriesMeanCI] = {}
    burst_norm_out: dict[str, SeriesMeanCI] = {}
    other_norm_out: dict[str, SeriesMeanCI] = {}
    norm_x_cosine_out: dict[str, SeriesMeanCI] = {}

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

    return PerLayerGradientSchedule(
        burst_steps=burst_steps,
        layer_names=layer_names,
        steps=first_steps.tolist(),
        cosine=cosine_out,
        burst_norm=burst_norm_out,
        other_norm=other_norm_out,
        norm_x_cosine=norm_x_cosine_out,
    )


def aggregate_probe_method(
    runs: list[dict[str, Any]],
    step_key: str,
    method: str,
) -> ProbeRegime | None:
    """Aggregate one probe method across seeds for a single step."""
    other_arrs: list[np.ndarray] = []
    burst_arrs: list[np.ndarray] = []
    for r in runs:
        sr = r["step_results"].get(step_key, {})
        if method not in sr:
            continue
        other_arrs.append(np.array(sr[method]["Other"], dtype=float))
        burst_arrs.append(np.array(sr[method]["Burst"], dtype=float))
    if not other_arrs:
        return None
    other_stack = np.stack(other_arrs)
    burst_stack = np.stack(burst_arrs)
    diff_stack = other_stack - burst_stack
    return ProbeRegime(
        Other=bundle_metric(list(other_stack)),
        Burst=bundle_metric(list(burst_stack)),
        diff=bundle_metric(list(diff_stack)),
    )


def build_probe_curves(
    run_dir: str | Path,
) -> dict[str, dict[str, dict[str, ProbeRegime]]]:
    """Build per-schedule next-token probe curves."""
    records = load_probe_records(run_dir)
    if not records:
        return {}

    by_schedule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_schedule[rec["schedule"]].append(rec)

    payload: dict[str, dict[str, dict[str, ProbeRegime]]] = {}
    for schedule in ordered_schedules(by_schedule.keys()):
        runs = by_schedule[schedule]
        all_steps = {s for r in runs for s in r["step_results"]}
        if not all_steps:
            continue

        per_step: dict[str, dict[str, ProbeRegime]] = {}
        for step_key in sorted(all_steps):
            per_method: dict[str, ProbeRegime] = {}
            for method in ("logit_lens", "learned_probe"):
                agg = aggregate_probe_method(runs, step_key, method)
                if agg is not None:
                    per_method[method] = agg
            if per_method:
                per_step[step_key] = per_method
        if per_step:
            payload[schedule] = per_step
    return payload


def build_weight_drift(run_dir: str | Path) -> dict[str, WeightDriftSchedule]:
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

    payload: dict[str, WeightDriftSchedule] = {}
    for schedule in ordered_schedules(by_schedule.keys()):
        result = weight_drift_for_schedule(by_schedule[schedule], n_layer)
        if result is not None:
            payload[schedule] = result
    return payload


def weight_drift_for_schedule(seed_dirs: list[Path], n_layer: int) -> WeightDriftSchedule | None:
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

        cum, sw = weight_drift_arrays(ckpt_files, sorted_steps, groups, base_sd)
        seed_cumulative.append(cum)
        seed_stepwise.append(sw)

    if not seed_cumulative or layer_names is None or steps is None:
        return None

    cum_stack = np.stack(seed_cumulative)
    step_stack = np.stack(seed_stepwise)
    cum_out: dict[str, SeriesMeanCI] = {}
    stepwise_out: dict[str, SeriesMeanCI] = {}
    for li, ln in enumerate(layer_names):
        cum_out[ln] = SeriesMeanCI(
            mean=np.nanmean(cum_stack[:, li, :], axis=0).tolist(),
            ci=series_ci(cum_stack[:, li, :]).tolist(),
        )
        stepwise_out[ln] = SeriesMeanCI(
            mean=np.nanmean(step_stack[:, li, :], axis=0).tolist(),
            ci=series_ci(step_stack[:, li, :]).tolist(),
        )

    return WeightDriftSchedule(
        layer_names=layer_names,
        steps=steps,
        cumulative=cum_out,
        stepwise=stepwise_out,
    )


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
