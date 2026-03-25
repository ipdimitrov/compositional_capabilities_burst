"""Phase 2: Finetune (burst) — mix special-class examples into background.

Loads a pretrained checkpoint, trains with a given burst fraction, and saves
the resulting checkpoint + log.  Can be called multiple times with different
burst_frac values to sweep concentrations.
"""
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

from burst.simple.model import (
    load_model, save_model, make_optimizer,
    train_step, eval_accuracy, eval_loss, cosine_lr,
    MODEL_DEFAULTS,
)


def _sample_batch(target_pool, bg_pool, n_target, batch_size):
    """Assemble a mixed batch of n_target special + rest background."""
    t_ids = list(target_pool.keys())
    b_ids = list(bg_pool.keys())
    parts = []

    def _sample(pool, ids, n):
        if n == 0:
            return
        per = n // len(ids)
        rem = n % len(ids)
        for i, tid in enumerate(ids):
            k = per + (1 if i < rem else 0)
            if k > 0:
                idx = np.random.randint(len(pool[tid]), size=k)
                parts.append(pool[tid][idx])

    _sample(target_pool, t_ids, n_target)
    _sample(bg_pool, b_ids, batch_size - n_target)
    return np.concatenate(parts)[np.random.permutation(batch_size)]


def finetune(
    data: dict,
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
    weight_decay: float = MODEL_DEFAULTS["weight_decay"],
    beta1: float = MODEL_DEFAULTS["beta1"],
    beta2: float = MODEL_DEFAULTS["beta2"],
    n_layer: int = MODEL_DEFAULTS["n_layer"],
    n_embd: int = MODEL_DEFAULTS["n_embd"],
    n_head: int = MODEL_DEFAULTS["n_head"],
    tag: str | None = None,
    seed: int = 42,
    quiet: bool = False,
) -> dict:
    """Run burst-phase finetuning.

    Args:
        data: dict from make_data()
        pretrain_ckpt: path to pretrained checkpoint
        out_dir: directory to save finetune checkpoint + log
        burst_frac: fraction of each batch that is special-class (0.0-1.0)
        steps: number of burst-phase training steps
        tag: optional label (used in filenames); defaults to f"burst_{int(burst_frac*100)}"

    Returns:
        dict with: log, ckpt_path, burst_frac, tag, peak_burst
    """
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

    np.random.seed(seed)
    import torch; torch.manual_seed(seed)

    net = load_model(pretrain_ckpt, vocab_size, context_size,
                     n_layer=n_layer, n_embd=n_embd, n_head=n_head)
    optimizer = make_optimizer(net, lr=lr * lr_start_frac,
                               weight_decay=weight_decay, beta1=beta1, beta2=beta2)

    log = {"step": [], "loss": [], "acc_other": [], "acc_burst": [],
           "loss_other": [], "loss_burst": [], "lr": []}

    lr_start = lr * lr_start_frac
    lr_end = lr * lr_end_frac

    net.train()
    pbar = tqdm(range(steps), desc=f"Finetune {tag}", disable=quiet)
    for s in pbar:
        n_target = int(np.random.binomial(batch_size, burst_frac)) if burst_frac < 1.0 else batch_size
        batch = _sample_batch(target_pool, bg_pool, n_target, batch_size)

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
            pbar.set_postfix(loss=f"{loss_val:.4f}", acc_b=f"{ab:.3f}",
                             acc_o=f"{ao:.3f}")
            net.train()

    ckpt_path = out_dir / f"{tag}_ckpt.pt"
    save_model(net, ckpt_path)
    np.savez(out_dir / f"{tag}_log.npz",
             **{k: np.array(v) for k, v in log.items()})

    peak_burst = max(log["acc_burst"]) if log["acc_burst"] else 0.0

    return {
        "log": log,
        "ckpt_path": str(ckpt_path),
        "burst_frac": burst_frac,
        "tag": tag,
        "peak_burst": peak_burst,
    }
