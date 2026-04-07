"""Shared training utilities: model creation, optimizer setup, training step."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from burst.core.train.experiment import DepthNData

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf

from burst.config import MIXED_FRACTIONS
from burst.rng import get_rng, seed_all
from net.nanogpt import nanoGPT
from net.runner import configure_optimizers, phase_lr, reset_optimizer_state, update_phase_lr

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def n_target_for_step(step: int, total_steps: int, schedule: str, p: float, batch_size: int) -> int:
    """Return the number of special-class examples in a batch at a given burst-phase step.

    All schedules 25-100% use binomial sampling throughout the full burst phase.
    burst_100 returns the full batch (frac=1.0) every step.
    """
    T = total_steps

    if schedule == "mid_block":
        burst_len = max(int(p * T), 1)
        mid = T // 2
        half = burst_len // 2
        return batch_size if mid - half <= step < mid + (burst_len - half) else 0

    if schedule in MIXED_FRACTIONS:
        frac = MIXED_FRACTIONS[schedule]
        if frac >= 1.0:
            return batch_size
        return int(get_rng().binomial(batch_size, frac))

    if schedule == "ramp_up":
        burst_len = max(int(p * T), 1)
        max_frac = 0.20
        ramp_len = min(int(2 * burst_len / max_frac), T)
        if step >= T - ramp_len:
            progress = (step - (T - ramp_len)) / max(ramp_len - 1, 1)
            return int(get_rng().binomial(batch_size, progress * max_frac))
        return 0

    if schedule == "reversion_only":
        return 0

    msg = f"Unknown schedule: {schedule}"
    raise ValueError(msg)


def sample_batch(  # noqa: PLR0913
    target_pool: dict, bg_pool: dict, n_target: int, batch_size: int,
    t_ids: list | None = None, b_ids: list | None = None,
) -> tuple[np.ndarray, list]:
    """Sample a mixed batch from target and background pools.

    t_ids/b_ids are optional for convenience but should be precomputed
    and passed explicitly in hot loops.
    """
    if t_ids is None:
        t_ids = list(target_pool.keys())
    if b_ids is None:
        b_ids = list(bg_pool.keys())
    parts = []
    sampled_tasks = []

    def _sample_from(pool: dict, ids: list, n: int) -> None:
        """Sample n documents evenly across task ids from pool."""
        if n == 0:
            return
        per = n // len(ids)
        rem = n % len(ids)
        for i, tid in enumerate(ids):
            k = per + (1 if i < rem else 0)
            if k > 0:
                idx = get_rng().integers(len(pool[tid]), size=k)
                parts.append(pool[tid][idx])
                sampled_tasks.extend([tid] * k)

    _sample_from(target_pool, t_ids, n_target)
    _sample_from(bg_pool, b_ids, batch_size - n_target)

    perm = get_rng().permutation(batch_size)
    return np.concatenate(parts)[perm], [sampled_tasks[i] for i in perm]


_NET_OMEGACONF_KEYS = ("vocab_size", "context_size", "n_layer", "n_head", "n_embd")


def _cross_entropy_logits_BTV_targets_BT(  # noqa: N802
    logits_BTV: torch.Tensor, targets_BT: torch.Tensor
) -> torch.Tensor:
    """Compute cross-entropy loss after flattening batch and time dims."""
    logits_bv = rearrange(logits_BTV, "b t v -> (b t) v")
    targets_b = rearrange(targets_BT, "b t -> (b t)")
    return F.cross_entropy(logits_bv, targets_b)


def _net_cfg(cfg: dict) -> OmegaConf:
    """Build an OmegaConf for nanoGPT from a flat config dict."""
    return OmegaConf.create(
        {
            "compile": False,
            **{k: cfg[k] for k in _NET_OMEGACONF_KEYS},
            "dropout": 0.0,
            "bias": False,
            "mlp": True,
        }
    )


def make_net_bare(cfg: dict) -> nanoGPT:
    """Create an uncompiled nanoGPT on DEVICE."""
    return nanoGPT(_net_cfg(cfg)).to(DEVICE)


def make_net(cfg: dict) -> nanoGPT:
    """Create a torch.compiled nanoGPT on DEVICE."""
    net = make_net_bare(cfg)
    if DEVICE == "cuda":
        net = torch.compile(net)
    return net


def load_net(cfg: dict, ckpt_path: str) -> nanoGPT:
    """Create a nanoGPT and load weights from a checkpoint."""
    net = nanoGPT(_net_cfg(cfg)).to(DEVICE)
    net.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    return net


def make_optim_cfg(cfg: dict) -> OmegaConf:
    """Build an OmegaConf for the optimizer from a flat config dict."""
    return OmegaConf.create(
        {
            "learning_rate": cfg["lr"],
            "weight_decay": cfg["weight_decay"],
            "beta1": cfg["beta1"],
            "beta2": cfg["beta2"],
            "grad_clip": cfg["grad_clip"],
        }
    )


def make_scaler() -> torch.amp.GradScaler:
    """Create a GradScaler enabled only when CUDA is available."""
    return torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")


def train_step(  # noqa: PLR0913
    batch_np: np.ndarray,
    net: nanoGPT,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    global_step: int,
    cfg: dict,
    grad_clip: float,
) -> float:
    """Single training step with phase-aware LR. Returns loss_value."""
    tokens_BL = torch.as_tensor(batch_np, dtype=torch.long, device=DEVICE)
    inputs_BT, targets_BT = tokens_BL[:, :-1], tokens_BL[:, 1:]
    update_phase_lr(
        global_step,
        optimizer,
        cfg["warmup_iters"],
        cfg["pre_burst_steps"],
        cfg["total_steps"],
        cfg["reversion_steps"],
        cfg["lr"],
        cfg["lr_pretrain_end_frac"],
        cfg["lr_burst_end_frac"],
        cfg["lr_reversion_end_frac"],
    )
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        logits_BTV = net(inputs_BT)
        loss = _cross_entropy_logits_BTV_targets_BT(logits_BTV, targets_BT)
    scaler.scale(loss).backward()
    if grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    return loss.item()


def retrain_with_callbacks(
    job: dict,
    target_pool: dict,
    bg_pool: dict,
    on_step: Callable | None = None,
    max_step: int | None = None,
) -> nanoGPT:
    """Retrain a model from scratch, calling on_step(net, global_step, phase) at each step.

    on_step should return True to continue, False to stop early.
    If max_step is given, training stops at that global step.
    """
    seed, cfg, schedule = job["seed"], job["cfg"], job["schedule"]
    seed_all(seed)
    net = make_net(cfg)
    optim_cfg = make_optim_cfg(cfg)
    optimizer = configure_optimizers(net, optim_cfg)
    scaler = make_scaler()

    P = cfg["pre_burst_steps"]
    T, U = cfg["total_steps"], cfg["reversion_steps"]
    bs, p = cfg["batch_size"], cfg["p_target"]
    effective_max = max_step if max_step is not None else (P + T + U)

    net.train()

    if on_step:
        on_step(net, 0, "init")

    train_end = min(T, max(0, effective_max - P))
    for s in range(train_end):
        nt = n_target_for_step(s, T, schedule, p, bs)
        batch_np, _ = sample_batch(target_pool, bg_pool, nt, bs)
        global_step = P + s + 1
        train_step(batch_np, net, optimizer, scaler, global_step, cfg, cfg["grad_clip"])
        if on_step:
            on_step(net, global_step, "train")

    reset_optimizer_state(optimizer)

    reversion_end = min(U, max(0, effective_max - P - T))
    for s in range(reversion_end):
        batch_np, _ = sample_batch(target_pool, bg_pool, 0, bs)
        global_step = P + T + s + 1
        train_step(batch_np, net, optimizer, scaler, global_step, cfg, cfg["grad_clip"])
        if on_step:
            on_step(net, global_step, "reversion")

    return net


def _resolve_split_dir(run_dir: Path, name: str) -> Path:
    """Find data/<name>/<run_name>/ (legacy nested) or data/<name>/<run_name>/ (split)."""
    nested = run_dir / name
    split = run_dir.parent / name / run_dir.name
    for candidate in (nested, split):
        if candidate.exists():
            return candidate
    return split


def resolve_results_dir(run_dir: Path) -> Path:
    """Directory for config.json, plots, grad_cosine_sim."""
    return _resolve_split_dir(run_dir, "results")


def resolve_logs_dir(run_dir: Path) -> Path:
    """Directory for all_results.pkl, checkpoints, _data.pkl."""
    return _resolve_split_dir(run_dir, "logs")


def resolve_run_paths(run_dir: str | Path) -> tuple[Path, Path, Path]:
    """Return (config_path, logs_dir, results_dir) for legacy and split layouts."""
    run_dir = Path(run_dir)
    results_dir = resolve_results_dir(run_dir)
    logs_dir = resolve_logs_dir(run_dir)

    if (results_dir / "config.json").exists():
        cfg_path = results_dir / "config.json"
    else:
        cfg_path = run_dir / "config.json"

    return cfg_path, logs_dir, results_dir


def load_results(run_dir: str | Path) -> tuple[list[dict], dict]:
    """Load all_results.pkl and config.json from a run directory."""
    cfg_path, logs_dir, _ = resolve_run_paths(run_dir)

    pkl_path = logs_dir / "all_results.pkl"
    if not pkl_path.exists():
        pkl_path = Path(run_dir) / "all_results.pkl"

    with pkl_path.open("rb") as f:
        results = pickle.load(f)  # noqa: S301
    with cfg_path.open() as f:
        cfg = json.load(f)
    return results, cfg


def pad_to_len(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad or truncate the second axis of arr to target_len."""
    if arr.shape[0] == 0:
        return arr
    if arr.shape[1] >= target_len:
        return arr[:, :target_len]
    pad_w = target_len - arr.shape[1]
    return np.concatenate([arr, np.zeros((arr.shape[0], pad_w), dtype=arr.dtype)], axis=1)


def build_probe_docs(
    data: DepthNData,
    doc_len: int,
    n_per_task: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build balanced Other/Burst probe datasets from train compositions."""
    other_pool = data.gen_pool(data.other_train[: min(16, len(data.other_train))], n_per_task)
    burst_pool = data.gen_pool(data.burst_train, n_per_task)

    def _cat(pool: dict) -> np.ndarray:
        """Concatenate pool values and pad to doc_len."""
        if not pool:
            return np.zeros((0, doc_len), dtype=np.int64)
        out = np.concatenate(list(pool.values()))
        return pad_to_len(out, doc_len)

    return _cat(other_pool), _cat(burst_pool)


def compute_lr_schedule(
    cfg: dict, pretrain_steps: int | None = None, burst_steps: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Compute three-phase LR schedule arrays. Returns (steps, lrs)."""
    P = pretrain_steps if pretrain_steps is not None else cfg["pre_burst_steps"]
    T = burst_steps if burst_steps is not None else cfg["total_steps"]
    U = cfg["reversion_steps"]

    steps = np.arange(1, P + T + U + 1)
    lrs = np.array(
        [
            phase_lr(
                s,
                cfg["warmup_iters"],
                P,
                T,
                U,
                cfg["lr"],
                cfg["lr_pretrain_end_frac"],
                cfg["lr_burst_end_frac"],
                cfg["lr_reversion_end_frac"],
            )
            for s in steps
        ]
    )
    return steps, lrs
