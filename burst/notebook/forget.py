"""Phase 3: Forget (reversion) — train on background only, measure forgetting.

Loads a finetuned checkpoint and trains on background data only.  Tracks how
quickly the model forgets the burst capability.
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
    reset_optimizer,
    save_model,
    train_step,
)
from burst.types import ExperimentData, ForgetResult

_rng = np.random.default_rng()

ACC_NEAR_ZERO = 1e-6


def forget(
    data: ExperimentData,
    finetune_ckpt: str | Path,
    out_dir: str | Path,
    *,
    pretrain_ckpt: str | Path | None = None,
    steps: int = 500,
    lr: float = MODEL_DEFAULTS["lr"],
    lr_start_frac: float = 0.15,
    lr_end_frac: float = 0.1,
    batch_size: int = MODEL_DEFAULTS["batch_size"],
    eval_every: int = MODEL_DEFAULTS["eval_every"],
    grad_clip: float = MODEL_DEFAULTS["grad_clip"],
    beta1: float = MODEL_DEFAULTS["beta1"],
    beta2: float = MODEL_DEFAULTS["beta2"],
    n_layer: int = MODEL_DEFAULTS["n_layer"],
    n_embd: int = MODEL_DEFAULTS["n_embd"],
    n_head: int = MODEL_DEFAULTS["n_head"],
    thresholds: tuple[float, ...] = (0.95, 0.90, 0.85, 0.80, 0.75, 0.70),
    tag: str | None = None,
    seed: int = 42,
    quiet: bool = False,
) -> ForgetResult:
    """Run reversion phase with background-only training and measure forgetting."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    finetune_ckpt = Path(finetune_ckpt)
    if tag is None:
        tag = finetune_ckpt.stem.replace("_ckpt", "")

    vocab_size = data["vocab_size"]
    context_size = data["context_size"]
    prompt_len = data["prompt_len"]
    bg_pool = data["bg_pool"]
    eval_other = data["eval_other"]
    eval_burst = data["eval_burst"]

    global _rng  # noqa: PLW0603
    _rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    net = load_model(
        finetune_ckpt, vocab_size, context_size, n_layer=n_layer, n_embd=n_embd, n_head=n_head
    )
    optimizer = make_optimizer(net, lr=lr * lr_start_frac, beta1=beta1, beta2=beta2)
    reset_optimizer(optimizer)  # fresh momentum for reversion phase

    sd_ft = state_dict_cpu(net)
    sd_pt = None
    if pretrain_ckpt is not None:
        from burst.notebook.interp import load_sd

        sd_pt = load_sd(pretrain_ckpt)

    peak_burst = eval_accuracy(net, eval_burst, prompt_len)

    bg_ids = list(bg_pool.keys())
    lr_start = lr * lr_start_frac
    lr_end = lr * lr_end_frac

    log = {
        "step": [],
        "loss": [],
        ACC_OTHER: [],
        ACC_BURST: [],
        LOSS_OTHER: [],
        LOSS_BURST: [],
        "lr": [],
        "weight_drift_from_ft": [],
        "weight_drift_from_pt": [],
        "grad_norm": [],
        "grad_norm_burst": [],
        "grad_cosine_burst_bg": [],
        "grad_cosine_per_layer": [],
        "grad_norm_entropy": [],
    }

    net.train()
    pbar = tqdm(range(steps), desc=f"Forget {tag}", disable=quiet)
    for s in pbar:
        per = batch_size // len(bg_ids)
        rem = batch_size % len(bg_ids)
        parts = []
        for i, tid in enumerate(bg_ids):
            k = per + (1 if i < rem else 0)
            if k > 0:
                idx = _rng.integers(len(bg_pool[tid]), size=k)
                parts.append(bg_pool[tid][idx])
        batch = np.concatenate(parts)[_rng.permutation(batch_size)]

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
            drift_ft = weight_drift_l2(sd_ft, sd_now)["total"]
            log["weight_drift_from_ft"].append(drift_ft)
            if sd_pt is not None:
                drift_pt = weight_drift_l2(sd_pt, sd_now)["total"]
                log["weight_drift_from_pt"].append(drift_pt)

            net.train()
            g_bg = _get_grad_vector(net, batch)
            log["grad_norm"].append(g_bg.norm().item())
            idx = _rng.integers(len(eval_burst), size=min(batch_size, len(eval_burst)))
            burst_batch = eval_burst[idx]
            g_burst = _get_grad_vector(net, burst_batch)
            log["grad_norm_burst"].append(g_burst.norm().item())
            gc = F.cosine_similarity(g_bg.unsqueeze(0), g_burst.unsqueeze(0)).item()
            log["grad_cosine_burst_bg"].append(gc)

            gc_layers = gradient_cosine_per_layer(net, burst_batch, batch)
            log["grad_cosine_per_layer"].append(gc_layers)

            ent, _ = grad_norm_entropy(net, batch)
            log["grad_norm_entropy"].append(ent)
            net.zero_grad()

            pbar.set_postfix(loss=f"{loss_val:.4f}", acc_b=f"{ab:.3f}", drift=f"{drift_ft:.3f}")
            net.train()

    accs = log[ACC_BURST]
    steps_arr = log["step"]
    reversion_auc = float(np.trapezoid(accs, steps_arr)) if len(accs) > 1 else 0.0

    life_times = {}
    if peak_burst > ACC_NEAR_ZERO:
        remaining = dict.fromkeys(thresholds, True)
        for acc_val, step_val in zip(accs, steps_arr, strict=True):
            for t in list(remaining):
                if acc_val <= peak_burst * t:
                    life_times[f"life_{int(t * 100)}"] = step_val
                    del remaining[t]
            if not remaining:
                break
    for t in thresholds:
        k = f"life_{int(t * 100)}"
        if k not in life_times:
            life_times[k] = steps  # never dropped below threshold

    end_burst = accs[-1] if accs else peak_burst
    dropoff_abs = peak_burst - end_burst
    dropoff_pct = (dropoff_abs / peak_burst * 100) if peak_burst > ACC_NEAR_ZERO else 0.0

    ckpt_path = out_dir / f"{tag}_reverted_ckpt.pt"
    save_model(net, ckpt_path)
    np.savez(out_dir / f"{tag}_forget_log.npz", **{k: np.array(v) for k, v in log.items()})

    return {
        "log": log,
        "ckpt_path": str(ckpt_path),
        "tag": tag,
        "peak_burst": peak_burst,
        "reversion_auc": reversion_auc,
        "life_times": life_times,
        "dropoff_abs": dropoff_abs,
        "dropoff_pct": dropoff_pct,
        "end_burst_acc": end_burst,
    }


def _forget_worker(kwargs: dict) -> ForgetResult:
    """Pickle-able entry point for multiprocessing."""
    return forget(**kwargs)
