"""Phase 1: Pretrain on background (all-but-special) data.

Trains a model from scratch until loss plateaus (convergence),
then saves a checkpoint that finetune() can load.
"""
import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm

from simple.model import (
    make_model, save_model, make_optimizer,
    train_step, eval_accuracy, eval_loss, cosine_lr,
    MODEL_DEFAULTS,
)


def pretrain(
    data: dict,
    out_dir: str | Path,
    *,
    lr: float = MODEL_DEFAULTS["lr"],
    batch_size: int = MODEL_DEFAULTS["batch_size"],
    eval_every: int = MODEL_DEFAULTS["eval_every"],
    warmup: int = MODEL_DEFAULTS["warmup_iters"],
    grad_clip: float = MODEL_DEFAULTS["grad_clip"],
    weight_decay: float = MODEL_DEFAULTS["weight_decay"],
    beta1: float = MODEL_DEFAULTS["beta1"],
    beta2: float = MODEL_DEFAULTS["beta2"],
    n_layer: int = MODEL_DEFAULTS["n_layer"],
    n_embd: int = MODEL_DEFAULTS["n_embd"],
    n_head: int = MODEL_DEFAULTS["n_head"],
    patience: int = 5,
    max_steps: int = 5000,
    seed: int = 42,
    quiet: bool = False,
) -> dict:
    """Pretrain on background data until convergence and save checkpoint.

    Training runs until eval loss plateaus for `patience` consecutive eval
    checks, or until `max_steps` is reached (safety cap).

    Args:
        data: dict from make_data()
        out_dir: directory to save pretrain_ckpt.pt and pretrain_log.npz
        patience: stop when eval loss doesn't improve for this many
                  consecutive eval checks
        max_steps: safety cap to prevent infinite loops

    Returns:
        dict with keys: log (dict of lists), ckpt_path (str), model_cfg (dict)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bg_pool = data["bg_pool"]
    eval_other = data["eval_other"]
    eval_burst = data["eval_burst"]
    prompt_len = data["prompt_len"]
    vocab_size = data["vocab_size"]
    context_size = data["context_size"]

    bg_ids = list(bg_pool.keys())
    model_cfg = dict(vocab_size=vocab_size, context_size=context_size,
                     n_layer=n_layer, n_embd=n_embd, n_head=n_head)

    np.random.seed(seed)
    torch.manual_seed(seed)

    net = make_model(**model_cfg)
    optimizer = make_optimizer(net, lr=lr, weight_decay=weight_decay,
                               beta1=beta1, beta2=beta2)

    log = {"step": [], "loss": [], "acc_other": [], "acc_burst": [],
           "loss_other": [], "loss_burst": [], "lr": []}

    best_loss = float("inf")
    wait = 0

    net.train()
    pbar = tqdm(desc="Pretrain", disable=quiet)
    s = 0
    while s < max_steps:
        # sample background batch
        per = batch_size // len(bg_ids)
        rem = batch_size % len(bg_ids)
        parts = []
        for i, tid in enumerate(bg_ids):
            k = per + (1 if i < rem else 0)
            if k > 0:
                idx = np.random.randint(len(bg_pool[tid]), size=k)
                parts.append(bg_pool[tid][idx])
        batch = np.concatenate(parts)[np.random.permutation(batch_size)]

        # constant lr after warmup
        cur_lr = lr * min(1.0, (s + 1) / warmup) if warmup > 0 else lr
        loss_val = train_step(net, optimizer, batch, lr=cur_lr,
                              grad_clip=grad_clip)

        if s % eval_every == 0:
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
            pbar.set_postfix(step=s, loss=f"{loss_val:.4f}", acc_o=f"{ao:.3f}")
            pbar.update(eval_every)
            net.train()

            if lo < best_loss:
                best_loss = lo
                wait = 0
            else:
                wait += 1
            if wait >= patience:
                if not quiet:
                    print(f"\nConverged at step {s} (eval loss plateaued for "
                          f"{patience} checks)")
                break

        s += 1

    pbar.close()

    if not quiet:
        peak = max(log["acc_other"]) if log["acc_other"] else 0
        print(f"Pretrain done: peak acc_other={peak:.4f}, "
              f"final loss_other={log['loss_other'][-1]:.4f}")

    ckpt_path = out_dir / "pretrain_ckpt.pt"
    save_model(net, ckpt_path)

    # save log as npz for easy loading
    np.savez(out_dir / "pretrain_log.npz",
             **{k: np.array(v) for k, v in log.items()})

    return {
        "log": log,
        "ckpt_path": str(ckpt_path),
        "model_cfg": model_cfg,
    }
