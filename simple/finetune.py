"""Phase 2: Finetune (burst) — mix special-class examples into background.

Loads a pretrained checkpoint, trains with a given burst fraction, and saves
the resulting checkpoint + log.  Can be called multiple times with different
burst_frac values to sweep concentrations.
"""
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm.auto import tqdm

from simple.model import (
    load_model, save_model, make_optimizer,
    train_step, eval_accuracy, eval_loss, cosine_lr,
    MODEL_DEFAULTS,
)
from simple.interp import (
    state_dict_cpu, weight_drift_l2,
    _get_grad_vector, gradient_cosine_per_layer, grad_norm_entropy,
    grad_effective_rank,
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
    torch.manual_seed(seed)

    net = load_model(pretrain_ckpt, vocab_size, context_size,
                     n_layer=n_layer, n_embd=n_embd, n_head=n_head)
    optimizer = make_optimizer(net, lr=lr * lr_start_frac,
                               weight_decay=weight_decay, beta1=beta1, beta2=beta2)

    # reference state dict for weight drift tracking
    sd_ref = state_dict_cpu(net)

    log = {"step": [], "loss": [], "acc_other": [], "acc_burst": [],
           "loss_other": [], "loss_burst": [], "lr": [],
           "weight_drift": [],
           "grad_norm_burst": [], "grad_norm_bg": [], "grad_norm_train": [],
           "grad_cosine_burst_bg": [], "grad_cosine_per_layer": [],
           "grad_norm_entropy_burst": [], "grad_norm_entropy_bg": [],
           "grad_align_frac": [], "grad_proj_magnitude": [],
           "grad_autocorr_bg": [], "grad_autocorr_burst": [],
           "grad_bg_vs_ref": [], "grad_burst_vs_ref": [],
           "grad_rank_burst": [], "grad_rank_bg": []}

    lr_start = lr * lr_start_frac
    lr_end = lr * lr_end_frac

    _prev_g_bg = None
    _prev_g_burst = None
    _ref_g_bg = None
    _ref_g_burst = None

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

            # interp metrics
            sd_now = state_dict_cpu(net)
            drift = weight_drift_l2(sd_ref, sd_now)["total"]
            log["weight_drift"].append(drift)

            # gradient norms & cosine: burst vs background
            burst_batch = _sample_batch(target_pool, bg_pool, batch_size, batch_size)
            bg_batch = _sample_batch(target_pool, bg_pool, 0, batch_size)
            net.train()
            g_burst = _get_grad_vector(net, burst_batch)
            g_bg = _get_grad_vector(net, bg_batch)
            gn_burst = g_burst.norm().item()
            gn_bg = g_bg.norm().item()
            gc = F.cosine_similarity(
                g_burst.unsqueeze(0), g_bg.unsqueeze(0)).item()
            # projection of bg gradient onto burst direction
            dot = torch.dot(g_bg, g_burst).item()
            align_frac = (dot ** 2) / (gn_bg ** 2 + 1e-10)  # (g_bg · ĝ_burst)² / ‖g_bg‖²
            proj_mag = abs(dot) / (gn_burst + 1e-10)          # |g_bg · ĝ_burst|

            net.zero_grad()
            log["grad_norm_burst"].append(gn_burst)
            log["grad_norm_bg"].append(gn_bg)
            log["grad_cosine_burst_bg"].append(gc)
            log["grad_align_frac"].append(align_frac)
            log["grad_proj_magnitude"].append(proj_mag)

            # gradient directional stability (consecutive-step cosine)
            if _prev_g_bg is not None:
                ac_bg = F.cosine_similarity(
                    g_bg.unsqueeze(0), _prev_g_bg.unsqueeze(0)).item()
                ac_burst = F.cosine_similarity(
                    g_burst.unsqueeze(0), _prev_g_burst.unsqueeze(0)).item()
            else:
                ac_bg = ac_burst = float("nan")
            log["grad_autocorr_bg"].append(ac_bg)
            log["grad_autocorr_burst"].append(ac_burst)
            _prev_g_bg = g_bg.detach().clone()
            _prev_g_burst = g_burst.detach().clone()

            # gradient direction vs step-0 reference
            if _ref_g_bg is None:
                _ref_g_bg = g_bg.detach().clone()
                _ref_g_burst = g_burst.detach().clone()
            cos_bg_ref = F.cosine_similarity(
                g_bg.unsqueeze(0), _ref_g_bg.unsqueeze(0)).item()
            cos_burst_ref = F.cosine_similarity(
                g_burst.unsqueeze(0), _ref_g_burst.unsqueeze(0)).item()
            log["grad_bg_vs_ref"].append(cos_bg_ref)
            log["grad_burst_vs_ref"].append(cos_burst_ref)

            # per-layer gradient cosine
            gc_layers = gradient_cosine_per_layer(net, burst_batch, bg_batch)
            log["grad_cosine_per_layer"].append(gc_layers)

            # gradient norm entropy
            ent_burst, _ = grad_norm_entropy(net, burst_batch)
            ent_bg, _ = grad_norm_entropy(net, bg_batch)
            log["grad_norm_entropy_burst"].append(ent_burst)
            log["grad_norm_entropy_bg"].append(ent_bg)

            # gradient effective rank
            rank_burst = grad_effective_rank(net, burst_batch)
            rank_bg = grad_effective_rank(net, bg_batch)
            log["grad_rank_burst"].append(rank_burst)
            log["grad_rank_bg"].append(rank_bg)

            # training batch gradient norm
            g_train = _get_grad_vector(net, batch)
            log["grad_norm_train"].append(g_train.norm().item())
            net.zero_grad()

            pbar.set_postfix(loss=f"{loss_val:.4f}", acc_b=f"{ab:.3f}",
                             acc_o=f"{ao:.3f}", drift=f"{drift:.3f}")
            net.train()

    ckpt_path = out_dir / f"{tag}_ckpt.pt"
    save_model(net, ckpt_path)
    np.savez(out_dir / f"{tag}_log.npz",
             **{k: np.array(v) for k, v in log.items()})

    peak_burst = max(log["acc_burst"]) if log["acc_burst"] else 0.0

    return {
        "log": log,
        "ckpt_path": str(ckpt_path),
        "pretrain_ckpt": str(pretrain_ckpt),
        "burst_frac": burst_frac,
        "tag": tag,
        "peak_burst": peak_burst,
    }


def finetune_kl(
    data: dict,
    pretrain_ckpt: str | Path,
    out_dir: str | Path,
    *,
    kl_beta: float = 1.0,
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
    """Finetune on 100% burst with KL(π_θ || π_ref) penalty.

    Analogous to burst_frac sweep but using explicit KL regularization
    instead of data mixing.  kl_beta=0 ≈ burst_frac=1.0, kl_beta→∞ ≈ burst_frac=0.
    """
    from simple.model import load_model as _load, train_step_kl

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if tag is None:
        tag = f"kl_{kl_beta:.0e}" if kl_beta >= 0.01 else f"kl_{kl_beta}"

    vocab_size = data["vocab_size"]
    context_size = data["context_size"]
    prompt_len = data["prompt_len"]
    target_pool = data["target_pool"]
    bg_pool = data["bg_pool"]
    eval_other = data["eval_other"]
    eval_burst = data["eval_burst"]

    np.random.seed(seed)
    torch.manual_seed(seed)

    net = load_model(pretrain_ckpt, vocab_size, context_size,
                     n_layer=n_layer, n_embd=n_embd, n_head=n_head)
    ref_net = _load(pretrain_ckpt, vocab_size, context_size,
                    n_layer=n_layer, n_embd=n_embd, n_head=n_head)
    ref_net.eval()
    for p in ref_net.parameters():
        p.requires_grad_(False)

    optimizer = make_optimizer(net, lr=lr * lr_start_frac,
                               weight_decay=weight_decay, beta1=beta1, beta2=beta2)

    log = {"step": [], "loss": [], "kl": [], "acc_other": [], "acc_burst": [],
           "loss_other": [], "loss_burst": [], "lr": []}

    lr_start = lr * lr_start_frac
    lr_end = lr * lr_end_frac

    t_ids = list(target_pool.keys())
    net.train()
    pbar = tqdm(range(steps), desc=f"FT-KL {tag}", disable=quiet)
    for s in pbar:
        # 100% burst batch
        per = batch_size // len(t_ids)
        rem = batch_size % len(t_ids)
        parts = []
        for i, tid in enumerate(t_ids):
            k = per + (1 if i < rem else 0)
            if k > 0:
                idx = np.random.randint(len(target_pool[tid]), size=k)
                parts.append(target_pool[tid][idx])
        batch = np.concatenate(parts)[np.random.permutation(batch_size)]

        cur_lr = cosine_lr(s + 1, steps, lr_start, lr_end)
        ce_val, kl_val = train_step_kl(net, ref_net, optimizer, batch,
                                        kl_beta=kl_beta, lr=cur_lr,
                                        grad_clip=grad_clip)

        if s % eval_every == 0 or s == steps - 1:
            ao = eval_accuracy(net, eval_other, prompt_len)
            ab = eval_accuracy(net, eval_burst, prompt_len)
            lo = eval_loss(net, eval_other)
            lb = eval_loss(net, eval_burst)
            log["step"].append(s)
            log["loss"].append(ce_val)
            log["kl"].append(kl_val)
            log["acc_other"].append(ao)
            log["acc_burst"].append(ab)
            log["loss_other"].append(lo)
            log["loss_burst"].append(lb)
            log["lr"].append(cur_lr)
            pbar.set_postfix(ce=f"{ce_val:.4f}", kl=f"{kl_val:.4f}",
                             acc_b=f"{ab:.3f}", acc_o=f"{ao:.3f}")
            net.train()

    ckpt_path = out_dir / f"{tag}_ckpt.pt"
    save_model(net, ckpt_path)
    np.savez(out_dir / f"{tag}_log.npz",
             **{k: np.array(v) for k, v in log.items()})

    peak_burst = max(log["acc_burst"]) if log["acc_burst"] else 0.0

    return {
        "log": log,
        "ckpt_path": str(ckpt_path),
        "pretrain_ckpt": str(pretrain_ckpt),
        "burst_frac": 1.0,
        "kl_beta": kl_beta,
        "tag": tag,
        "peak_burst": peak_burst,
    }


def _finetune_worker(kwargs):
    """Pickle-able entry point for multiprocessing."""
    return finetune(**kwargs)


def _finetune_kl_worker(kwargs):
    """Pickle-able entry point for multiprocessing."""
    return finetune_kl(**kwargs)
