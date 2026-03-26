"""Phase 3: Forget (reversion) — train on background only, measure forgetting.

Loads a finetuned checkpoint and trains on background data only.  Tracks how
quickly the model forgets the burst capability.
"""
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

from simple.model import (
    load_model, save_model, make_optimizer, reset_optimizer,
    train_step, eval_accuracy, eval_loss, cosine_lr,
    MODEL_DEFAULTS,
)


def forget(
    data: dict,
    finetune_ckpt: str | Path,
    out_dir: str | Path,
    *,
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
    """Run reversion phase: background-only training, measure forgetting.

    Args:
        data: dict from make_data()
        finetune_ckpt: path to finetuned checkpoint
        out_dir: directory for outputs
        steps: reversion training steps
        thresholds: reversion lifetime thresholds (fraction of peak)
        tag: label for filenames (inherited from finetune tag if None)

    Returns:
        dict with: log, peak_burst, reversion_auc, life_times, dropoff_abs,
                   dropoff_pct, tag
    """
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
    import torch; torch.manual_seed(seed)

    net = load_model(finetune_ckpt, vocab_size, context_size,
                     n_layer=n_layer, n_embd=n_embd, n_head=n_head)
    optimizer = make_optimizer(net, lr=lr * lr_start_frac,
                               weight_decay=weight_decay, beta1=beta1, beta2=beta2)
    reset_optimizer(optimizer)  # fresh momentum for reversion phase

    # measure peak burst accuracy at start of reversion
    peak_burst = eval_accuracy(net, eval_burst, prompt_len)

    bg_ids = list(bg_pool.keys())
    lr_start = lr * lr_start_frac
    lr_end = lr * lr_end_frac

    log = {"step": [], "loss": [], "acc_other": [], "acc_burst": [],
           "loss_other": [], "loss_burst": [], "lr": []}

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
        loss_val = train_step(net, optimizer, batch, lr=cur_lr,
                              grad_clip=grad_clip)

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
            pbar.set_postfix(loss=f"{loss_val:.4f}", acc_b=f"{ab:.3f}")
            net.train()

    # -- metrics --
    accs = log["acc_burst"]
    steps_arr = log["step"]
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    reversion_auc = float(_trapz(accs, steps_arr)) if len(accs) > 1 else 0.0

    life_times = {}
    if peak_burst > 1e-6:
        remaining = {t: True for t in thresholds}
        for acc_val, step_val in zip(accs, steps_arr):
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
    dropoff_pct = (dropoff_abs / peak_burst * 100) if peak_burst > 1e-6 else 0.0

    # save
    save_model(net, out_dir / f"{tag}_reverted_ckpt.pt")
    np.savez(out_dir / f"{tag}_forget_log.npz",
             **{k: np.array(v) for k, v in log.items()})

    return {
        "log": log,
        "tag": tag,
        "peak_burst": peak_burst,
        "reversion_auc": reversion_auc,
        "life_times": life_times,
        "dropoff_abs": dropoff_abs,
        "dropoff_pct": dropoff_pct,
        "end_burst_acc": end_burst,
    }
