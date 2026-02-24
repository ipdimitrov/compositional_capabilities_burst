"""Central configuration for the burst experiment.

Change any parameter here and it propagates everywhere.

To add/remove a schedule, edit BURST_FRACTIONS below — everything else
(names, ordering, display labels, colours) is derived automatically.
"""
from dataclasses import dataclass, field
import colorsys

# ---------------------------------------------------------------------------
# Phase names (the 5 stages of the experiment)
# ---------------------------------------------------------------------------
PHASE_FOUNDATION = "foundation"
PHASE_BURST = "burst"
PHASE_REVERSION = "reversion"
PHASE_UNIFORM = "uniform"
PHASE_NAMES = [PHASE_FOUNDATION, PHASE_BURST, PHASE_REVERSION, PHASE_UNIFORM]

# ---------------------------------------------------------------------------
# Class names
# ---------------------------------------------------------------------------
CLASS_OTHER = "other"
CLASS_BURST = "burst"

# ---------------------------------------------------------------------------
# Schedules — the ONLY thing you edit to add/remove schedules
# ---------------------------------------------------------------------------
BURST_FRACTIONS = [100, 98, 95, 90, 85, 75, 50, 25, 10]

UNIFORM_PCT = 10

# ---------------------------------------------------------------------------
# Everything below is derived from BURST_FRACTIONS
# ---------------------------------------------------------------------------

def _sched_name(pct: int) -> str:
    return f"burst_{pct}"


def _build_gradient(n: int) -> list[str]:
    """Red (high %) -> Blue (low %) gradient via HSL interpolation."""
    if n == 1:
        return ["#D32F2F"]
    hue_hi, hue_lo = 0.0, 0.58          # red -> blue in HSL
    colors = []
    for i in range(n):
        t = i / (n - 1)
        h = hue_hi + t * (hue_lo - hue_hi)
        r, g, b = colorsys.hls_to_rgb(h, 0.42, 0.72)
        colors.append(f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}")
    return colors


_sorted_pcts = sorted(BURST_FRACTIONS, reverse=True)
_gradient = _build_gradient(len(_sorted_pcts))

SCHEDULE_ORDER: list[str] = [_sched_name(p) for p in _sorted_pcts]
SCHEDULES: list[str] = list(SCHEDULE_ORDER)

UNIFORM_SCHEDULE: str = _sched_name(UNIFORM_PCT)

MIXED_FRACTIONS: dict[str, float] = {
    _sched_name(p): p / 100.0
    for p in _sorted_pcts
    if p != 100 and _sched_name(p) != UNIFORM_SCHEDULE
}

SCHED_COLORS: dict[str, str] = {
    _sched_name(p): c for p, c in zip(_sorted_pcts, _gradient)
}

SCHED_DISPLAY: dict[str, str] = {
    _sched_name(p): f"Burst {p}%" for p in _sorted_pcts
}

EVAL_KEYS = ["acc_other", "acc_burst"]

CURVE_STYLE = {
    "acc_other": {"color": "#2196F3", "ls": "-", "label": "Other Classes"},
    "acc_burst": {"color": "#E91E63", "ls": "-", "label": "Burst Class"},
}

# ---------------------------------------------------------------------------
# Data parameters
# ---------------------------------------------------------------------------
N_A = 4
SEED_BASE = 107
DATA_SEED = 999

# ---------------------------------------------------------------------------
# Model & training defaults
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    n_alphabets: int = 10
    seq_len: int = 6

    n_layer: int = 6
    n_embd: int = 120
    n_head: int = 4
    vocab_size: int = 128
    context_size: int = 80

    lr: float = 3e-4
    weight_decay: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_iters: int = 50
    min_lr: float = 6e-5

    batch_size: int = 128
    grad_sim_batch_size: int = 2048
    total_steps: int = 500
    p_target: float = 0.10
    reversion_steps: int = 500
    eval_every: int = 10
    unlearn_threshold: float = 0.25

    n_docs_per_task: int = 500
    n_eval_per_task: int = 500

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}


# ---------------------------------------------------------------------------
# Experiment-level config
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    n_seeds: int = 4
    n_workers: int = 38
    grad_sim_n_workers: int | None = None
    grad_sim_every: int = 50
    depth: int = 3
    burst_pos: int = 3
    schedules: list[str] = field(default_factory=lambda: list(SCHEDULES))

    @property
    def base_cfg(self) -> dict:
        return self.train.to_dict()


DEFAULT_CONFIG = ExperimentConfig()


def ordered_schedules(scheds) -> list[str]:
    return [s for s in SCHEDULE_ORDER if s in scheds] or sorted(scheds)


def sched_sort_key(schedule: str) -> int:
    try:
        return SCHEDULE_ORDER.index(schedule)
    except ValueError:
        return len(SCHEDULE_ORDER)
