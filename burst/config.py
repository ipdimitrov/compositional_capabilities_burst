"""Central configuration for the burst experiment.

Change any parameter here and it propagates everywhere.

To add/remove a schedule, edit BURST_FRACTIONS below — everything else
(names, ordering, display labels, colours) is derived automatically.

Burst phase length scales inversely with burst concentration so that all
schedules see the same total number of special-class examples:
    burst_steps = BURST_BASE_STEPS * (100 / pct)
e.g. burst_100 → 200 steps, burst_50 → 400 steps, burst_25 → 800 steps.

Burst modes
-----------
  "current"        - original setup: steps scale inversely with concentration
                     so total special examples are constant; batch_size fixed.
  "constant_steps" - all schedules run BURST_BASE_STEPS; batch_size fixed.
                     Only the special:other ratio in each batch differs.
  "scaled_batch"   - all schedules run BURST_BASE_STEPS; batch_size scales
                     inversely with concentration so special-per-step is constant
                     while total data per step grows for dilute schedules.
"""

import colorsys
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Phase names (three-phase experiment)
# ---------------------------------------------------------------------------
PHASE_PRE_BURST = "all-but-special (background)"
PHASE_BURST = "special (burst)"
PHASE_REVERSION = "all-but-special (reversion)"
PHASE_NAMES = [PHASE_PRE_BURST, PHASE_BURST, PHASE_REVERSION]

PHASE_FOUNDATION = PHASE_PRE_BURST

# ---------------------------------------------------------------------------
# Class names & derived metric keys
# ---------------------------------------------------------------------------
CLASS_OTHER = "other"
CLASS_BURST = "burst"

ACC_OTHER = f"acc_{CLASS_OTHER}"
ACC_BURST = f"acc_{CLASS_BURST}"
LOSS_OTHER = f"loss_{CLASS_OTHER}"
LOSS_BURST = f"loss_{CLASS_BURST}"

# ---------------------------------------------------------------------------
# Schedules — the ONLY thing you edit to add/remove schedules
# ---------------------------------------------------------------------------
BURST_FRACTIONS = [100, 98, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10]

UNIFORM_PCT = 25

# ---------------------------------------------------------------------------
# Everything below is derived from BURST_FRACTIONS
# ---------------------------------------------------------------------------


def _sched_name(pct: int) -> str:
    """Return the canonical schedule name for a burst percentage."""
    return f"burst_{pct}"


def _build_gradient(n: int) -> list[str]:
    """Red (high %) -> Blue (low %) gradient via HSL interpolation."""
    if n == 1:
        return ["#D32F2F"]
    hue_hi, hue_lo = 0.0, 0.58  # red -> blue in HSL
    colors = []
    for i in range(n):
        t = i / (n - 1)
        h = hue_hi + t * (hue_lo - hue_hi)
        r, g, b = colorsys.hls_to_rgb(h, 0.42, 0.72)
        colors.append(f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}")
    return colors


_sorted_pcts = sorted(BURST_FRACTIONS, reverse=True)
_gradient = _build_gradient(len(_sorted_pcts))

SCHEDULE_ORDER: list[str] = [_sched_name(p) for p in _sorted_pcts]

UNIFORM_SCHEDULE: str = _sched_name(UNIFORM_PCT)

FULL_PCT = 100
MIXED_FRACTIONS: dict[str, float] = {
    _sched_name(p): p / 100.0 for p in _sorted_pcts if p != FULL_PCT
}

SCHED_COLORS: dict[str, str] = {
    _sched_name(p): c for p, c in zip(_sorted_pcts, _gradient, strict=False)
}

SCHED_DISPLAY: dict[str, str] = {_sched_name(p): f"Burst {p}%" for p in _sorted_pcts}

EVAL_KEYS = [ACC_OTHER, ACC_BURST]

CURVE_STYLE = {
    ACC_OTHER: {"color": "#2196F3", "ls": "-", "label": "Other Classes"},
    ACC_BURST: {"color": "#E91E63", "ls": "-", "label": "Special Class"},
}

# ---------------------------------------------------------------------------
# Data parameters
# ---------------------------------------------------------------------------
N_A = 4
SEED_BASE = 100
DATA_SEED = 999

# ---------------------------------------------------------------------------
# Model & training defaults
# ---------------------------------------------------------------------------

BURST_BASE_STEPS = 800

MODE_CURRENT = "current"
MODE_CONSTANT_STEPS = "constant_steps"
MODE_SCALED_BATCH = "scaled_batch"
BURST_MODES = (MODE_CURRENT, MODE_CONSTANT_STEPS, MODE_SCALED_BATCH)

DEFAULT_REPRO_SEED = 1337
DEFAULT_DETERMINISTIC = True
REPRO_MANIFEST_FILENAME = "repro_manifest.json"
CORE_CHARTS_DIRNAME = "core_charts"
CORE_CLI_MODES = ("train", "gradients", "bundle", "charts", "pipeline")


def burst_steps_for_schedule(schedule: str, base_steps: int = BURST_BASE_STEPS) -> int:
    """Burst phase length for a given schedule (original "current" mode).

    All schedules see the same total special-class examples:
        burst_steps * frac = base_steps * 1.0
    So burst_100 -> base_steps, burst_50 -> 2x base_steps, burst_25 -> 4x base_steps.
    """
    if schedule == "burst_100":
        return base_steps
    if schedule == "burst_0":
        return base_steps
    if schedule in MIXED_FRACTIONS:
        frac = MIXED_FRACTIONS[schedule]
        if frac <= 0:
            return base_steps
        return max(base_steps, round(base_steps / frac))
    return base_steps


def burst_steps_for_mode(
    schedule: str,
    mode: str = MODE_CURRENT,
    base_steps: int = BURST_BASE_STEPS,
) -> int:
    """Return burst phase length for a given schedule and mode."""
    if mode == MODE_CURRENT:
        return burst_steps_for_schedule(schedule, base_steps)
    return base_steps


def batch_size_for_mode(
    schedule: str,
    mode: str = MODE_CURRENT,
    base_batch_size: int = 128,
) -> int:
    """Return batch size for a given schedule and mode."""
    if mode != MODE_SCALED_BATCH:
        return base_batch_size
    frac = MIXED_FRACTIONS.get(schedule, 1.0)
    if frac <= 0:
        return base_batch_size
    return max(base_batch_size, round(base_batch_size / frac))


MIXED_FRACTIONS["burst_100"] = 1.0


@dataclass
class TrainConfig:
    """Hyperparameters for a single training run."""

    n_alphabets: int = 10
    seq_len: int = 6

    n_layer: int = 6
    n_embd: int = 120
    n_head: int = 4
    vocab_size: int = 128
    context_size: int = 80

    lr: float = 1e-3
    lr_pretrain_end_frac: float = 0.3
    lr_burst_end_frac: float = 0.15
    lr_reversion_end_frac: float = 0.1
    weight_decay: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.9
    grad_clip: float = 1.0
    warmup_iters: int = 50

    batch_size: int = 128
    grad_sim_batch_size: int = 2048
    pre_burst_steps: int = 600
    total_steps: int = BURST_BASE_STEPS
    p_target: float = 0.2
    reversion_steps: int = 500
    eval_every: int = 25
    reversion_thresholds: tuple[float, ...] = (0.95, 0.90, 0.85, 0.80, 0.75, 0.70)

    n_docs_per_task: int = 100
    n_eval_per_task: int = 100

    def to_dict(self) -> dict[str, Any]:
        """Serialize all fields to a plain dict."""
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}


# ---------------------------------------------------------------------------
# Experiment-level config
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    train: TrainConfig = field(default_factory=TrainConfig)
    n_seeds: int = 10
    n_workers: int = 0
    depth: int = 3
    burst_pos: int = 3
    burst_mode: str = MODE_CURRENT
    schedules: list[str] = field(default_factory=lambda: list(SCHEDULE_ORDER))
    run_probes: bool = False
    run_next_token_probes: bool = False
    run_adl: bool = True

    def __post_init__(self) -> None:
        """Validate burst_mode and auto-detect worker count."""
        if self.burst_mode not in BURST_MODES:
            msg = f"burst_mode must be one of {BURST_MODES}, got {self.burst_mode!r}"
            raise ValueError(msg)
        if self.n_workers == 0:
            from burst.core.gpu import gpu_cfg

            self.n_workers = gpu_cfg.train_workers

    @property
    def base_cfg(self) -> dict[str, Any]:
        """Return training hyperparameters as a dict."""
        return self.train.to_dict()


DEFAULT_CONFIG = ExperimentConfig()


def parse_run_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract experiment-level fields from a saved config.json dict.

    Returns dict with keys: depth, burst_pos, n_a, base_cfg.
    Raises KeyError if any required field is missing.
    """
    base_cfg = cfg["base_cfg"]
    task_info = cfg.get("task_info", {})

    def _get(key: str) -> int:
        v = cfg.get(key)
        return v if v is not None else task_info.get(key)

    depth = _get("depth")
    assert depth is not None, "config missing 'depth'"

    burst_pos = _get("burst_pos")
    assert burst_pos is not None, "config missing 'burst_pos'"

    n_a = _get("n_a")
    assert n_a is not None, "config missing 'n_a'"

    return {"depth": depth, "burst_pos": burst_pos, "n_a": n_a, "base_cfg": base_cfg}


def reversion_life_key(threshold: float) -> str:
    """Return the metric key for a reversion threshold."""
    pct = int(threshold * 100)
    return f"life_{pct}"


def reversion_life_label(threshold: float) -> str:
    """Return the display label for a reversion threshold."""
    pct = int(threshold * 100)
    return f"{pct}%-life"


def ordered_schedules(scheds: Iterable[str]) -> list[str]:
    """Sort schedules in canonical order."""
    return [s for s in SCHEDULE_ORDER if s in scheds] or sorted(scheds)


def sched_sort_key(schedule: str) -> int:
    """Return the sort index for a schedule name."""
    assert schedule in SCHEDULE_ORDER, f"Unknown schedule: {schedule}"
    return SCHEDULE_ORDER.index(schedule)
