"""Shared typed structures for the burst experiment pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import numpy as np


class WorkerJob(TypedDict):
    """Job dict passed to worker.run via pickle."""

    label: str
    schedule: str
    seed: int
    cfg: dict
    deterministic: bool
    pretrain_ckpt: str
    pretrain_log_path: str


class ExperimentData(TypedDict):
    """Data dict returned by make_data() and consumed by pretrain/finetune/forget."""

    bg_pool: dict[tuple, np.ndarray]
    target_pool: dict[tuple, np.ndarray]
    eval_other: np.ndarray
    eval_burst: np.ndarray
    prompt_len: int
    eval_start: int
    eval_end: int
    vocab_size: int
    context_size: int
    task_info: dict[str, int]
    bijections: list[np.ndarray]
    token_idx: dict[str, int]
    fn_tok: dict[int, int]


class ModelConfig(TypedDict):
    """Subset of model architecture params passed between phases."""

    vocab_size: int
    context_size: int
    n_layer: int
    n_embd: int
    n_head: int


class PretrainResult(TypedDict):
    """Result dict returned by pretrain()."""

    log: dict[str, list]
    ckpt_path: str
    model_cfg: ModelConfig


class FinetuneResult(TypedDict):
    """Result dict returned by finetune()."""

    log: dict[str, list]
    ckpt_path: str
    pretrain_ckpt: str
    burst_frac: float
    tag: str
    peak_burst: float


class ForgetResult(TypedDict):
    """Result dict returned by forget()."""

    log: dict[str, list]
    ckpt_path: str
    tag: str
    peak_burst: float
    reversion_auc: float
    life_times: dict[str, int]
    dropoff_abs: float
    dropoff_pct: float
    end_burst_acc: float
