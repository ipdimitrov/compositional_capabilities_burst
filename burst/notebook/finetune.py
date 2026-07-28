"""Phase 2: Finetune (burst) — mix special-class examples into background.

Loads a pretrained checkpoint, trains with a given burst fraction, and saves
the resulting checkpoint + log.  Can be called multiple times with different
burst_frac values to sweep concentrations.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from burst.config import ACC_BURST, ACC_OTHER, LOSS_BURST, LOSS_OTHER
from burst.notebook.interp import (
    _get_grad_vector,
    grad_norm_entropy,
    gradient_cosine_per_layer,
    state_dict_cpu,
    weight_drift_l2,
)
from burst.notebook.model import (
    MODEL_DEFAULTS,
    cosine_lr,
    eval_accuracy,
    eval_loss,
    load_model,
    make_optimizer,
    save_model,
    train_step,
)
from burst.types import ExperimentData, FinetuneResult

_rng = np.random.default_rng()


def _sample_batch(target_pool: dict, bg_pool: dict, n_target: int, batch_size: int) -> np.ndarray:
    """Assemble a mixed batch of n_target special + rest background."""
    global _rng  # noqa: PLW0602
    t_ids = list(target_pool.keys())
    b_ids = list(bg_pool.keys())
    parts = []

    def _sample(pool: dict, ids: list, n: int) -> None:
        if n == 0:
            return
        per = n // len(ids)
        rem = n % len(ids)
        for i, tid in enumerate(ids):
            k = per + (1 if i < rem else 0)
            if k > 0:
                idx = _rng.integers(len(pool[tid]), size=k)
                parts.append(pool[tid][idx])

    _sample(target_pool, t_ids, n_target)
    _sample(bg_pool, b_ids, batch_size - n_target)
    return np.concatenate(parts)[_rng.permutation(batch_size)]


def finetune(  # noqa: PLR0913, PLR0915
    data: ExperimentData,
    pretrain_ckpt: str | Path,
    out_dir: str | Path,
    *,
    burst_frac: float = 1.0,
    steps: int = 200,
    lr: float = MODEL_DEFAULTS["lr"],
    lr_start_frac: float = 0.3,
    lr_end_frac: float = 0.15,
    batch_size: int = MODEL_DEFAULTS["batch_size"],
    eval_every: int = MODEL_DEFAULTS["eval_every"],
    grad_clip: float = MODEL_DEFAULTS["grad_clip"],
    beta1: float = MODEL_DEFAULTS["beta1"],
    beta2: float = MODEL_DEFAULTS["beta2"],
    n_layer: int = MODEL_DEFAULTS["n_layer"],
    n_embd: int = MODEL_DEFAULTS["n_embd"],
    n_head: int = MODEL_DEFAULTS["n_head"],
    tag: str | None = None,
    seed: int = 42,
    quiet: bool = False,
) -> FinetuneResult:
    """Run burst-phase finetuning and return log, checkpoint path, and metrics."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if tag is None:
        tag = f"burst_{int(burst_frac * 100)}"

    vocab_size = data["vocab_size"]
    context_size = data["context_size"]
    prompt_len = data["prompt_len"]
    target_pool = data["target_pool"]
    bg_pool = data["bg_pool"]
    eval_other = data["eval_other"]
    eval_burst = data["eval_burst"]

    global _rng  # noqa: PLW0603
    _rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    net = load_model(
        pretrain_ckpt, vocab_size, context_size, n_layer=n_layer, n_embd=n_embd, n_head=n_head
    )
    optimizer = make_optimizer(net, lr=lr * lr_start_frac, beta1=beta1, beta2=beta2)

    sd_ref = state_dict_cpu(net)

    log = {
        "step": [],
        "loss": [],
        ACC_OTHER: [],
        ACC_BURST: [],
        LOSS_OTHER: [],
        LOSS_BURST: [],
        "lr": [],
        "weight_drift": [],
        "grad_norm_burst": [],
        "grad_norm_bg": [],
        "grad_norm_train": [],
        "grad_cosine_burst_bg": [],
        "grad_cosine_per_layer": [],
        "grad_norm_entropy_burst": [],
        "grad_norm_entropy_bg": [],
    }

    lr_start = lr * lr_start_frac
    lr_end = lr * lr_end_frac

    net.train()
    pbar = tqdm(range(steps), desc=f"Finetune {tag}", disable=quiet)
    for s in pbar:
        n_target = (
            int(_rng.binomial(batch_size, burst_frac)) if burst_frac < 1.0 else batch_size
        )
        batch = _sample_batch(target_pool, bg_pool, n_target, batch_size)

        cur_lr = cosine_lr(s + 1, steps, lr_start, lr_end)
        loss_val = train_step(net, optimizer, batch, lr=cur_lr, grad_clip=grad_clip)

        if s % eval_every == 0 or s == steps - 1:
            ao = eval_accuracy(net, eval_other, prompt_len)
            ab = eval_accuracy(net, eval_burst, prompt_len)
            lo = eval_loss(net, eval_other)
            lb = eval_loss(net, eval_burst)
            log["step"].append(s)
            log["loss"].append(loss_val)
            log[ACC_OTHER].append(ao)
            log[ACC_BURST].append(ab)
            log[LOSS_OTHER].append(lo)
            log[LOSS_BURST].append(lb)
            log["lr"].append(cur_lr)

            sd_now = state_dict_cpu(net)
            drift = weight_drift_l2(sd_ref, sd_now)["total"]
            log["weight_drift"].append(drift)

            burst_batch = _sample_batch(target_pool, bg_pool, batch_size, batch_size)
            bg_batch = _sample_batch(target_pool, bg_pool, 0, batch_size)
            net.train()
            g_burst = _get_grad_vector(net, burst_batch)
            g_bg = _get_grad_vector(net, bg_batch)
            gn_burst = g_burst.norm().item()
            gn_bg = g_bg.norm().item()
            gc = F.cosine_similarity(g_burst.unsqueeze(0), g_bg.unsqueeze(0)).item()
            net.zero_grad()
            log["grad_norm_burst"].append(gn_burst)
            log["grad_norm_bg"].append(gn_bg)
            log["grad_cosine_burst_bg"].append(gc)

            gc_layers = gradient_cosine_per_layer(net, burst_batch, bg_batch)
            log["grad_cosine_per_layer"].append(gc_layers)

            ent_burst, _ = grad_norm_entropy(net, burst_batch)
            ent_bg, _ = grad_norm_entropy(net, bg_batch)
            log["grad_norm_entropy_burst"].append(ent_burst)
            log["grad_norm_entropy_bg"].append(ent_bg)

            g_train = _get_grad_vector(net, batch)
            log["grad_norm_train"].append(g_train.norm().item())
            net.zero_grad()

            pbar.set_postfix(
                loss=f"{loss_val:.4f}", acc_b=f"{ab:.3f}", acc_o=f"{ao:.3f}", drift=f"{drift:.3f}"
            )
            net.train()

    ckpt_path = out_dir / f"{tag}_ckpt.pt"
    save_model(net, ckpt_path)
    np.savez(out_dir / f"{tag}_log.npz", **{k: np.array(v) for k, v in log.items()})

    peak_burst = max(log[ACC_BURST]) if log[ACC_BURST] else 0.0

    return {
        "log": log,
        "ckpt_path": str(ckpt_path),
        "pretrain_ckpt": str(pretrain_ckpt),
        "burst_frac": burst_frac,
        "tag": tag,
        "peak_burst": peak_burst,
    }


def _finetune_worker(kwargs: dict) -> FinetuneResult:
    """Pickle-able entry point for multiprocessing."""
    return finetune(**kwargs)
