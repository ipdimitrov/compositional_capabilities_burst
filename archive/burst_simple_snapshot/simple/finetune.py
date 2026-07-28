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
    state_dict_cpu, weight_drift_l2, weight_drift_per_block,
    _get_grad_vector, _get_grad_per_layer,
    gradient_cosine_and_norms_per_layer,
    grad_norm_entropy, fisher_overlap_score,
    gradient_effective_rank,
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
    fisher_eigvecs: dict | None = None,
    grad_eff_rank_batches: int = 50,
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
           "grad_cossim_x_norm_burst": [], "grad_cossim_x_norm_bg": [],
           "grad_cossim_x_norm_bg_per_layer": [],
           "grad_cosine_burst_successive": [],
           "grad_cosine_bg_successive": [],
           "weight_update_norm": [],
           "grad_effective_rank": [],
           "grad_effective_rank_burst": [],
           "grad_effective_rank_bg": [],
           "grad_pca90_burst": [],
           "grad_pca90_bg": [],
           "weight_update_per_block": [],
           "grad_bg_vs_weight_delta_cossim": []}
    _prev_sd = state_dict_cpu(net)
    _prev_g_burst = None
    _prev_g_bg = None
    if fisher_eigvecs is not None:
        log["fisher_overlap"] = []
        log["fisher_overlap_per_dir"] = []
        log["fisher_overlap_per_layer"] = []
        log["fisher_overlap_per_layer_dirs"] = []

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

            # interp metrics
            sd_now = state_dict_cpu(net)
            drift = weight_drift_l2(sd_ref, sd_now)["total"]
            log["weight_drift"].append(drift)

            # weight update norm: ||theta(t) - theta(t-1)||
            update_norm = weight_drift_l2(_prev_sd, sd_now)["total"]
            log["weight_update_norm"].append(update_norm)
            log["weight_update_per_block"].append(weight_drift_per_block(_prev_sd, sd_now))
            _prev_sd = sd_now

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
            net.zero_grad()
            log["grad_norm_burst"].append(gn_burst)
            log["grad_norm_bg"].append(gn_bg)
            log["grad_cosine_burst_bg"].append(gc)
            log["grad_cossim_x_norm_burst"].append(gc * gn_burst)
            log["grad_cossim_x_norm_bg"].append(gc * gn_bg)

            # cosine(bg gradient, weight delta from pretrain)
            # build delta vector matching _get_grad_vector layout (deduplicated params)
            _raw_net = getattr(net, "_orig_mod", net)
            _wd_parts = [sd_now[_name].flatten() - sd_ref[_name].flatten()
                         for _name, _ in _raw_net.named_parameters()]
            _wd_vec = torch.cat(_wd_parts).to(g_bg.device)
            _wd_norm = _wd_vec.norm()
            if _wd_norm > 0:
                _gc_wd = F.cosine_similarity(
                    g_bg.unsqueeze(0), _wd_vec.unsqueeze(0)).item()
            else:
                _gc_wd = 0.0
            log["grad_bg_vs_weight_delta_cossim"].append(_gc_wd)

            # successive gradient cosine: g(t) vs g(t-1) for burst and bg
            if _prev_g_burst is not None:
                gc_burst_succ = F.cosine_similarity(
                    g_burst.unsqueeze(0), _prev_g_burst.unsqueeze(0)).item()
                gc_bg_succ = F.cosine_similarity(
                    g_bg.unsqueeze(0), _prev_g_bg.unsqueeze(0)).item()
            else:
                gc_burst_succ = float("nan")
                gc_bg_succ = float("nan")
            log["grad_cosine_burst_successive"].append(gc_burst_succ)
            log["grad_cosine_bg_successive"].append(gc_bg_succ)
            _prev_g_burst = g_burst.detach().clone()
            _prev_g_bg = g_bg.detach().clone()

            # Fisher overlap (global + per-layer) — uses g_burst computed above
            if fisher_eigvecs is not None:
                fo_total, fo_per_dir = fisher_overlap_score(
                    g_burst, fisher_eigvecs["global_eigvecs"])
                log["fisher_overlap"].append(fo_total)
                log["fisher_overlap_per_dir"].append(fo_per_dir)

                # per-layer: one more backward pass to get layer-split gradients
                g_burst_layers = _get_grad_per_layer(net, burst_batch)
                fo_layers, fo_layers_dirs = {}, {}
                for key, g_l in g_burst_layers.items():
                    if key in fisher_eigvecs["per_layer_eigvecs"]:
                        fo_l, fo_l_dirs = fisher_overlap_score(
                            g_l, fisher_eigvecs["per_layer_eigvecs"][key])
                        fo_layers[key] = fo_l
                        fo_layers_dirs[key] = fo_l_dirs
                log["fisher_overlap_per_layer"].append(fo_layers)
                log["fisher_overlap_per_layer_dirs"].append(fo_layers_dirs)
                net.zero_grad()

            # per-layer gradient cosine + bg norm product
            gc_layers, _, gn_bg_layers = gradient_cosine_and_norms_per_layer(
                net, burst_batch, bg_batch)
            log["grad_cosine_per_layer"].append(gc_layers)
            log["grad_cossim_x_norm_bg_per_layer"].append(
                {k: gc_layers[k] * gn_bg_layers.get(k, 0.0) for k in gc_layers})

            # gradient norm entropy
            ent_burst, _ = grad_norm_entropy(net, burst_batch)
            ent_bg, _ = grad_norm_entropy(net, bg_batch)
            log["grad_norm_entropy_burst"].append(ent_burst)
            log["grad_norm_entropy_bg"].append(ent_bg)

            # gradient effective rank: training mix, burst-only, bg-only
            def _sample_train_batch():
                nt = int(np.random.binomial(batch_size, burst_frac)) if burst_frac < 1.0 else batch_size
                return _sample_batch(target_pool, bg_pool, nt, batch_size)
            def _sample_burst_only():
                return _sample_batch(target_pool, bg_pool, batch_size, batch_size)
            def _sample_bg_only():
                return _sample_batch(target_pool, bg_pool, 0, batch_size)
            eff_rank, pca_k, _ = gradient_effective_rank(net, _sample_train_batch,
                                                        n_batches=grad_eff_rank_batches)
            eff_rank_burst, pca_k_burst, _ = gradient_effective_rank(net, _sample_burst_only,
                                                                      n_batches=grad_eff_rank_batches)
            eff_rank_bg, pca_k_bg, _ = gradient_effective_rank(net, _sample_bg_only,
                                                                n_batches=grad_eff_rank_batches)
            log["grad_effective_rank"].append(eff_rank)
            log["grad_effective_rank_burst"].append(eff_rank_burst)
            log["grad_effective_rank_bg"].append(eff_rank_bg)
            log["grad_pca90_burst"].append(pca_k_burst)
            log["grad_pca90_bg"].append(pca_k_bg)

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


def _finetune_worker(kwargs):
    """Pickle-able entry point for multiprocessing."""
    return finetune(**kwargs)
