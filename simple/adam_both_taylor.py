"""AdamW training phase with per-step Taylor decomposition on BOTH burst and pretrain losses.

At each step, observe the real Δθ the optimizer takes and compute:

    ΔL_burst  ≈  g_burst(θ_t) · Δθ  +  ½ Δθ^T H_burst(θ_t) Δθ
    ΔL_pre    ≈  g_pre(θ_t)   · Δθ  +  ½ Δθ^T H_pre(θ_t)   Δθ

Start-of-interval convention (+ sign on quadratic). One HVP per loss per step
(2 HVPs total), evaluated at θ_t (we briefly restore params).

Works for any concentration c ∈ [0, 1] — use c > 0 for finetuning, c = 0 for
forgetting. Same code, same logging.
"""
from __future__ import annotations

import numpy as np
import torch
from tqdm.auto import tqdm

from simple.model import (
    DEVICE, load_model, eval_loss, eval_accuracy,
    make_optimizer, reset_optimizer, train_step, cosine_lr,
)
from simple.interp import _get_grad_vector, state_dict_cpu, weight_drift_l2, _raw
from simple.finetune import _sample_batch

from simple.adam_taylor import (
    _grad_param_keys, _flat_sd_for_keys, _unflatten_to_sd,
    _sample_pure, hvp_ft, _cos,
)


LOG_KEYS = [
    "step", "lr",
    "loss_burst", "loss_other", "acc_burst", "acc_other", "weight_drift",
    "delta_theta_norm",
    # gradient diagnostics
    "g_burst_norm", "g_pre_norm", "cos_g_burst_g_pre",
    # --- Taylor for BURST loss ---
    "T_first_burst",     # g_burst · Δθ
    "T_second_burst",    # ½ Δθ^T H_burst Δθ
    "pred_dL_burst",     # T_first_burst + T_second_burst
    "dL_actual_burst",   # measured on fixed burst-eval batch
    # --- Taylor for PRETRAIN loss ---
    "T_first_pre",       # g_pre · Δθ
    "T_second_pre",      # ½ Δθ^T H_pre Δθ
    "pred_dL_pre",       # T_first_pre + T_second_pre
    "dL_actual_pre",     # measured on fixed pre-eval batch
    # curvature diagnostics
    "rayleigh_burst",    # 2·T_second_burst / ||Δθ||²
    "rayleigh_pre",      # 2·T_second_pre   / ||Δθ||²
    # reference: loss change on the FULL eval sets
    "dL_actual_burst_full",
    "dL_actual_pre_full",
]


def _flat_grad(net, batch_np):
    return _get_grad_vector(net, batch_np).detach().cpu()


def _load_params_from_flat(net, param_keys, flat_cpu):
    raw = _raw(net)
    name_to_param = {n: p for n, p in raw.named_parameters() if p.requires_grad}
    off = 0
    with torch.no_grad():
        for k in param_keys:
            p = name_to_param[k]
            n = p.numel()
            p.copy_(flat_cpu[off:off + n].to(p.device).to(p.dtype).reshape(p.shape))
            off += n


def adam_phase_with_taylor(
    data, start_ckpt_or_net, *,
    c: float, steps: int, lr: float, lr_end: float | None = None,
    batch_size: int = 512, eval_every: int = 1, seed: int = 42,
    grad_clip: float = 0.0,
    adam_beta1: float = 0.9, adam_beta2: float = 0.9, adam_eps: float = 1e-8,
    measure_batch_size: int = 1024,
    hvp_epsilon: float = 1e-3,
    desc: str = "phase", progress: bool = True,
) -> dict:
    """Run `steps` of AdamW at concentration `c`, logging per-step Taylor
    decomposition of ΔL_burst AND ΔL_pre.

    Works for both finetune (c>0) and forget (c=0).
    """
    lr_end = lr if lr_end is None else lr_end
    target_pool = data["target_pool"]; bg_pool = data["bg_pool"]
    vocab_size = data["vocab_size"]; context_size = data["context_size"]

    np.random.seed(seed); torch.manual_seed(seed)

    if hasattr(start_ckpt_or_net, "state_dict"):
        net = start_ckpt_or_net
    else:
        net = load_model(start_ckpt_or_net, vocab_size, context_size,
                         compile_model=False)

    param_keys = _grad_param_keys(net)
    sd_init = state_dict_cpu(net)

    optimizer = make_optimizer(net, lr=lr, weight_decay=0.0,
                               beta1=adam_beta1, beta2=adam_beta2)
    reset_optimizer(optimizer)

    # Fixed eval batches — same examples every step, removing sampling variance
    # from the Taylor terms. One set per loss.
    def _fixed(arr, n):
        return arr[np.random.choice(len(arr), min(n, len(arr)), replace=False)]

    eval_burst_batch = _fixed(data["eval_burst"], measure_batch_size)
    eval_pre_batch   = _fixed(data["eval_other"], measure_batch_size)

    # Deterministic HVP: one forward over the full fixed batch per call.
    hvp_kw_burst = dict(n_batches=1, batch_size=len(eval_burst_batch), epsilon=hvp_epsilon)
    hvp_kw_pre   = dict(n_batches=1, batch_size=len(eval_pre_batch),   epsilon=hvp_epsilon)

    log = {k: [] for k in LOG_KEYS}

    # Track per-step losses on fixed batches and on full eval sets.
    net.eval()
    lb_sub_prev  = eval_loss(net, eval_burst_batch, batch_size=batch_size)
    lp_sub_prev  = eval_loss(net, eval_pre_batch,   batch_size=batch_size)
    lb_full_prev = eval_loss(net, data["eval_burst"], batch_size=batch_size)
    lp_full_prev = eval_loss(net, data["eval_other"], batch_size=batch_size)
    net.train()

    it = range(steps)
    pbar = tqdm(it, desc=desc) if progress else it
    for s in pbar:
        cur_lr = cosine_lr(s + 1, steps, lr, lr_end)

        # ── Gradients at θ_t (start of interval) on fixed eval batches. ──
        net.eval()
        g_burst = _flat_grad(net, eval_burst_batch)
        g_pre   = _flat_grad(net, eval_pre_batch)
        net.train()

        # Snapshot θ_t.
        theta_before = _flat_sd_for_keys(state_dict_cpu(net), param_keys)

        # ── AdamW step. ──
        n_ft = int(round(c * batch_size))
        mixed = _sample_batch(target_pool, bg_pool, n_ft, batch_size)
        train_step(net, optimizer, mixed, lr=cur_lr, grad_clip=grad_clip)

        theta_after = _flat_sd_for_keys(state_dict_cpu(net), param_keys)
        delta = theta_after - theta_before
        delta_norm = delta.norm().item()

        # ── Taylor terms: briefly restore θ_t for HVPs, then put θ_{t+1} back. ──
        T_first_burst = torch.dot(g_burst, delta).item()
        T_first_pre   = torch.dot(g_pre,   delta).item()

        if delta_norm > 1e-12:
            _load_params_from_flat(net, param_keys, theta_before)
            H_delta_burst = hvp_ft(net, delta, param_keys, eval_burst_batch, **hvp_kw_burst).cpu()
            H_delta_pre   = hvp_ft(net, delta, param_keys, eval_pre_batch,   **hvp_kw_pre).cpu()
            _load_params_from_flat(net, param_keys, theta_after)
            T_second_burst = 0.5 * torch.dot(delta, H_delta_burst).item()
            T_second_pre   = 0.5 * torch.dot(delta, H_delta_pre).item()
            rayleigh_burst = 2.0 * T_second_burst / (delta_norm ** 2)
            rayleigh_pre   = 2.0 * T_second_pre   / (delta_norm ** 2)
        else:
            T_second_burst = T_second_pre = 0.0
            rayleigh_burst = rayleigh_pre = np.nan

        pred_burst = T_first_burst + T_second_burst
        pred_pre   = T_first_pre   + T_second_pre

        # ── Evaluate losses at θ_{t+1}. ──
        if s % eval_every == 0 or s == steps - 1:
            net.eval()
            lb_sub  = eval_loss(net, eval_burst_batch, batch_size=batch_size)
            lp_sub  = eval_loss(net, eval_pre_batch,   batch_size=batch_size)
            lb_full = eval_loss(net, data["eval_burst"], batch_size=batch_size)
            lp_full = eval_loss(net, data["eval_other"], batch_size=batch_size)
            ab = eval_accuracy(net, data["eval_burst"], data["prompt_len"])
            ao = eval_accuracy(net, data["eval_other"], data["prompt_len"])
            drift = weight_drift_l2(sd_init, state_dict_cpu(net))["total"]
            net.train()

            for k, v in dict(
                step=s, lr=cur_lr,
                loss_burst=lb_full, loss_other=lp_full,
                acc_burst=ab, acc_other=ao, weight_drift=drift,
                delta_theta_norm=delta_norm,
                g_burst_norm=g_burst.norm().item(),
                g_pre_norm=g_pre.norm().item(),
                cos_g_burst_g_pre=_cos(g_burst, g_pre),
                T_first_burst=T_first_burst, T_second_burst=T_second_burst,
                pred_dL_burst=pred_burst,
                dL_actual_burst=lb_sub - lb_sub_prev,
                T_first_pre=T_first_pre, T_second_pre=T_second_pre,
                pred_dL_pre=pred_pre,
                dL_actual_pre=lp_sub - lp_sub_prev,
                rayleigh_burst=rayleigh_burst, rayleigh_pre=rayleigh_pre,
                dL_actual_burst_full=lb_full - lb_full_prev,
                dL_actual_pre_full=lp_full - lp_full_prev,
            ).items():
                log[k].append(v)

            lb_sub_prev, lp_sub_prev = lb_sub, lp_sub
            lb_full_prev, lp_full_prev = lb_full, lp_full

            if progress:
                pbar.set_postfix(lb=f"{lb_full:.3f}", ab=f"{ab:.3f}")

    return {
        "log": {k: np.array(v) for k, v in log.items()},
        "final_sd": state_dict_cpu(net),
        "net": net,
    }
