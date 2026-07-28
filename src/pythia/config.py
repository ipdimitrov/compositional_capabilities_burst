"""Experiment configuration for catastrophic forgetting experiments."""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ExperimentConfig:
    # Model
    model_name: str = "EleutherAI/pythia-70m-deduped"

    # Data — narrow domain (default: SMILES molecular strings)
    domain_train_dataset: str = "antoinebcx/smiles-molecules-chembl"
    domain_valid_dataset: str = "antoinebcx/smiles-molecules-chembl"
    domain_train_split: str = "train"
    domain_valid_split: str = "validation"
    pile_dataset: str = "monology/pile-uncopyrighted"
    domain_text_field: str = "smiles"
    pile_text_field: str = "text"
    seq_length: int = 512
    num_valid_chunks: int = 2000
    max_train_chunks: int = 0  # 0 = auto (based on steps), >0 = hard cap per dataset

    # Burst levels
    burst_levels: list[float] = field(default_factory=lambda: [1.0, 0.90, 0.75, 0.5, 0.25])

    # Phase 1 — Fine-tuning
    ft_steps: int = 500
    ft_lr: float = 5e-5
    ft_lr_min_ratio: float = 0.0  # cosine decays to this fraction of peak lr (0.0 = standard)
    ft_weight_decay: float = 0.01
    ft_warmup_steps: int = 50
    ft_batch_size: int = 20
    ft_grad_accum: int = 2

    # FT budget mode — how ft_steps is interpreted across burst levels.
    #   "volume": scale ft_steps by 1/burst_level so every burst sees the same
    #             volume of DOMAIN data (ft_steps is the 100%-burst anchor).
    #             ft_warmup_steps scales proportionally.
    #   "steps":  every burst gets exactly ft_steps (and ft_warmup_steps).
    #             Low-burst runs see less domain data — the old behavior.
    ft_budget_mode: str = "steps"

    # Phase 2 — Continued pretraining
    cpt_steps: int = 1000
    cpt_lr: float = 5e-5
    cpt_weight_decay: float = 0.01
    cpt_warmup_steps: int = 50
    cpt_batch_size: int = 20
    cpt_grad_accum: int = 2

    # Evaluation
    eval_every: int = 50
    eval_batches: int = 50
    eval_batch_size: int = 16

    # Training
    max_grad_norm: float = 1.0
    seed: int = 42

    # Gradient analysis (optional — slows training)
    compute_grad_metrics: bool = False
    grad_metrics_every: int = 100  # how often to compute (in steps)

    # Checkpoint saving (optional — can be large on disk, esp. for 1B+)
    save_checkpoints: bool = True

    # Outputs
    results_dir: str = "results"

    @property
    def ft_effective_batch(self) -> int:
        return self.ft_batch_size * self.ft_grad_accum

    @property
    def cpt_effective_batch(self) -> int:
        return self.cpt_batch_size * self.cpt_grad_accum

    def ft_steps_for_burst(self, burst_level: float) -> int:
        """FT step count for a given burst level.

        In "volume" mode: scales as ft_steps / burst_level so every burst level
        sees the same number of domain batches.
        In "steps" mode: every burst gets exactly ft_steps.
        """
        if self.ft_budget_mode == "steps":
            return self.ft_steps
        if self.ft_budget_mode != "volume":
            msg = f"ft_budget_mode must be 'volume' or 'steps', got {self.ft_budget_mode!r}"
            raise ValueError(msg)
        if burst_level <= 0:
            return self.ft_steps
        return round(self.ft_steps / burst_level)

    def ft_warmup_for_burst(self, burst_level: float) -> int:
        """FT warmup steps for a given burst level. Scales with ft_steps_for_burst."""
        if self.ft_budget_mode == "steps":
            return self.ft_warmup_steps
        if burst_level <= 0:
            return self.ft_warmup_steps
        return round(self.ft_warmup_steps / burst_level)

    def max_ft_steps_across_bursts(self) -> int:
        """Max scaled FT step count across all configured burst levels."""
        if not self.burst_levels:
            return self.ft_steps
        return max(self.ft_steps_for_burst(bl) for bl in self.burst_levels)

    def save(self, path: str | None = None) -> None:
        if path is None:
            path = os.path.join(self.results_dir, "config.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


PRESETS = {
    "quick": {
        "ft_steps": 200,
        "cpt_steps": 200,
        "ft_warmup_steps": 20,
        "cpt_warmup_steps": 20,
        "eval_every": 40,
    },
    "normal": {
        "ft_steps": 500,
        "cpt_steps": 1000,
        "ft_warmup_steps": 50,
        "cpt_warmup_steps": 50,
        "eval_every": 50,
    },
    "full": {
        "ft_steps": 4000,
        "cpt_steps": 3000,
        "ft_warmup_steps": 100,
        "cpt_warmup_steps": 100,
        "eval_every": 50,
    },
    "deep": {
        "ft_steps": 15000,
        "cpt_steps": 3000,
        "ft_warmup_steps": 400,
        "cpt_warmup_steps": 100,
        "eval_every": 100,
    },
    "deepest": {
        "ft_steps": 30000,
        "cpt_steps": 4000,
        "ft_warmup_steps": 400,
        "cpt_warmup_steps": 100,
        "eval_every": 100,
    },
}


MODEL_PRESETS = {
    "70m": {
        "model_name": "EleutherAI/pythia-70m-deduped",
        "ft_lr": 5e-5,
        "ft_batch_size": 20,
        "cpt_batch_size": 20,
        "eval_batch_size": 16,
    },
    "1b": {
        "model_name": "EleutherAI/pythia-1b-deduped",
        "ft_lr": 1e-4,
        "ft_lr_min_ratio": 0.1,
        "ft_budget_mode": "volume",
        "ft_batch_size": 8,
        "cpt_batch_size": 8,
        "eval_batch_size": 8,
    },
}


DOMAIN_PRESETS = {
    "chemistry": {
        "domain_train_dataset": "antoinebcx/smiles-molecules-chembl",
        "domain_valid_dataset": "antoinebcx/smiles-molecules-chembl",
        "domain_train_split": "train",
        "domain_valid_split": "validation",
        "domain_text_field": "smiles",
    },
    "music": {
        "domain_train_dataset": "sander-wood/irishman",
        "domain_valid_dataset": "sander-wood/irishman",
        "domain_train_split": "train",
        "domain_valid_split": "validation",
        "domain_text_field": "abc notation",
    },
    "biomedical": {
        "domain_train_dataset": "ccdv/pubmed-summarization",
        "domain_valid_dataset": "ccdv/pubmed-summarization",
        "domain_train_split": "train",
        "domain_valid_split": "validation",
        "domain_text_field": "article",
    },
}


def make_results_dir(preset: str) -> str:
    """Create a timestamped results directory and a 'latest' symlink."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dirname = f"results/{ts}_{preset}"
    os.makedirs(dirname, exist_ok=True)
    # Update 'latest' symlink
    latest = "results/latest"
    if os.path.islink(latest):
        os.unlink(latest)
    elif os.path.exists(latest):
        pass  # don't clobber a real directory
    else:
        os.symlink(os.path.basename(dirname), latest)
    return dirname


def get_config(preset: str = "normal", domain: str = "chemistry",
               model: str = "70m", **overrides) -> ExperimentConfig:
    """Create a config from presets with optional overrides.

    Layering order: training preset → model preset → domain preset → overrides.
    """
    kwargs = {}
    if preset in PRESETS:
        kwargs.update(PRESETS[preset])
    if model in MODEL_PRESETS:
        kwargs.update(MODEL_PRESETS[model])
    if domain in DOMAIN_PRESETS:
        kwargs.update(DOMAIN_PRESETS[domain])
    # Auto-generate timestamped results dir unless explicitly overridden
    if "results_dir" not in overrides:
        kwargs["results_dir"] = make_results_dir(f"{preset}_{model}_{domain}")
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)
