"""Phase 3: Forget (reversion) — train on background only, measure forgetting.

Loads a finetuned checkpoint and trains on background data only.  Tracks how
quickly the model forgets the burst capability.
"""

from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

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

GRAD_NORM_EPS = 1e-6


def forget(  # noqa: C901, PLR0912, PLR0913, PLR0915
    data: dict,
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
    weight_decay: float = MODEL_DEFAULTS["weight_decay"],
    beta1: float = MODEL_DEFAULTS["beta1"],
    beta2: float = MODEL_DEFAULTS["beta2"],
    n_layer: int = MODEL_DEFAULTS["n_layer"],
    n_embd: int = MODEL_DEFAULTS["n_embd"],
    n_head: int = MODEL_DEFAULTS["n_head"],
    thresholds: tuple[float, ...] = (0.95, 0.90, 0.85, 0.80, 0.75, 0.70),
    tag: str | None = None,
    seed: int = 42,
    quiet: bool = False,
) -> dict:
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

    np.random.seed(seed)
    torch.manual_seed(seed)

    net = load_model(
        finetune_ckpt, vocab_size, context_size, n_layer=n_layer, n_embd=n_embd, n_head=n_head
    )
    optimizer = make_optimizer(
        net, lr=lr * lr_start_frac, _weight_decay=weight_decay, beta1=beta1, beta2=beta2
    )
    reset_optimizer(optimizer)  # fresh momentum for reversion phase

    sd_ft = state_dict_cpu(net)
    sd_pt = None
    if pretrain_ckpt is not None:
        from burst.notebook.interp import load_sd  # noqa: PLC0415

        sd_pt = load_sd(pretrain_ckpt)

    peak_burst = eval_accuracy(net, eval_burst, prompt_len)

    bg_ids = list(bg_pool.keys())
    lr_start = lr * lr_start_frac
    lr_end = lr * lr_end_frac

    log = {
        "step": [],
        "loss": [],
        "acc_other": [],
        "acc_burst": [],
        "loss_other": [],
        "loss_burst": [],
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
                idx = np.random.randint(len(bg_pool[tid]), size=k)
                parts.append(bg_pool[tid][idx])
        batch = np.concatenate(parts)[np.random.permutation(batch_size)]

        cur_lr = cosine_lr(s + 1, steps, lr_start, lr_end)
        loss_val = train_step(net, optimizer, batch, lr=cur_lr, grad_clip=grad_clip)

        if s % eval_every == 0 or s == steps - 1:
            ao = eval_accuracy(net, eval_other, prompt_len)
            ab = eval_accuracy(net, eval_burst, prompt_len)
            lo = eval_loss(net, eval_other)
            lb = eval_loss(net, eval_burst)
            log["step"].append(s)
            log["loss"].append(loss_val)
            log["acc_other"].append(ao)
            log["acc_burst"].append(ab)
            log["loss_other"].append(lo)
            log["loss_burst"].append(lb)
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
            idx = np.random.randint(len(eval_burst), size=min(batch_size, len(eval_burst)))
            burst_batch = eval_burst[idx]
            g_burst = _get_grad_vector(net, burst_batch)
            log["grad_norm_burst"].append(g_burst.norm().item())
            import torch.nn.functional as _F  # noqa: PLC0415

            gc = _F.cosine_similarity(g_bg.unsqueeze(0), g_burst.unsqueeze(0)).item()
            log["grad_cosine_burst_bg"].append(gc)

            gc_layers = gradient_cosine_per_layer(net, burst_batch, batch)
            log["grad_cosine_per_layer"].append(gc_layers)

            ent, _ = grad_norm_entropy(net, batch)
            log["grad_norm_entropy"].append(ent)
            net.zero_grad()

            pbar.set_postfix(loss=f"{loss_val:.4f}", acc_b=f"{ab:.3f}", drift=f"{drift_ft:.3f}")
            net.train()

    accs = log["acc_burst"]
    steps_arr = log["step"]
    _trapz = getattr(np, "trapezoid", np.trapz)
    reversion_auc = float(_trapz(accs, steps_arr)) if len(accs) > 1 else 0.0

    life_times = {}
    if peak_burst > GRAD_NORM_EPS:
        remaining = dict.fromkeys(thresholds, True)
        for acc_val, step_val in zip(accs, steps_arr, strict=False):
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
    dropoff_pct = (dropoff_abs / peak_burst * 100) if peak_burst > GRAD_NORM_EPS else 0.0

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


def _forget_worker(kwargs: dict) -> dict:
    """Pickle-able entry point for multiprocessing."""
    return forget(**kwargs)
