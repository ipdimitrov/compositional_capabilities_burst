"""AdamW-based forgetting with optimizer-agnostic 2-term Taylor logging.

Deliberately minimalist: no shadow EMAs, no preconditioner decomposition. Just
observe the real Δθ the optimizer takes and decompose ΔL_ft around θ_t (start
of interval):

    ΔL_ft  ≈  g_ft(θ_t) · Δθ  +  ½ · Δθ^T H_ft(θ_t) · Δθ
           =  T_first  +  T_second

- `T_first`  = first-order term (directional derivative)
- `T_second` = second-order term (½ · Rayleigh of H_ft along Δθ)

Works for any optimizer — here we use AdamW. One HVP per eval, evaluated at
θ_t (we briefly restore params to θ_t to get the right expansion point, then
put θ_{t+1} back).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
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
    "step", "lr", "loss_burst", "loss_other", "acc_burst", "acc_other", "weight_drift",
    "g_ft_norm", "g_pre_norm", "g_ft_dot_g_pre", "cos_g_ft_g_pre",
    "delta_theta_norm",
    # 8b-style Taylor (start-of-interval, + sign on quadratic)
    "T_first",       # g_ft(θ_t) · Δθ
    "T_second",      # ½ · Δθ^T H_ft(θ_t) · Δθ
    "pred_dL",       # T_first + T_second
    "dL_actual",     # loss on the same fixed ft-eval batch that g_ft/H_ft used
    "rayleigh",      # 2·T_second / ||Δθ||²
    # Reference: loss change on the FULL eval_burst (matches trajectory plot)
    "dL_actual_full",
]


def _flat_grad(net, batch_np):
    return _get_grad_vector(net, batch_np).detach().cpu()


def _load_params_from_flat(net, param_keys, flat_cpu):
    """Overwrite net's trainable params with `flat_cpu` (flat CPU fp32 tensor)."""
    raw = _raw(net)
    name_to_param = {n: p for n, p in raw.named_parameters() if p.requires_grad}
    off = 0
    with torch.no_grad():
        for k in param_keys:
            p = name_to_param[k]
            n = p.numel()
            p.copy_(flat_cpu[off:off + n].to(p.device).to(p.dtype).reshape(p.shape))
            off += n


def adam_forget_with_taylor(
    data, start_net, *,
    c: float = 0.0,
    steps: int, lr: float, lr_end: float | None = None,
    batch_size: int = 512, eval_every: int = 1, seed: int = 42,
    grad_clip: float = 0.0,
    adam_beta1: float = 0.9, adam_beta2: float = 0.9, adam_eps: float = 1e-8,
    hvp_n_batches: int = 4, hvp_batch_size: int = 256, hvp_epsilon: float = 1e-3,
    measure_batch_size: int = 1024,
    desc: str = "adam forget", progress: bool = True,
) -> dict:
    """Forget phase driven by a fresh AdamW optimizer at concentration `c`.

    At each step we:
      1. Sample a mixed batch at concentration `c` (c=0 → pure pretraining).
      2. Compute g_ft on `eval_burst` at θ_t (start of interval).
      3. Take the AdamW step → θ_{t+1}.
      4. Temporarily restore θ_t, compute one HVP along Δθ = θ_{t+1} − θ_t on
         `eval_burst`, then put θ_{t+1} back.
      5. Log T_first, T_second, pred_dL, dL_actual.

    Result: `T_first + T_second` predicts `dL_actual` up to 3rd-order truncation.
    """
    lr_end = lr if lr_end is None else lr_end
    net = start_net
    param_keys = _grad_param_keys(net)
    sd_init = state_dict_cpu(net)

    optimizer = make_optimizer(net, lr=lr, weight_decay=0.0,
                               beta1=adam_beta1, beta2=adam_beta2)
    reset_optimizer(optimizer)

    # Use one deterministic HVP pass over a fixed eval batch (below) so the HVP
    # is ~noise-free between steps. The optional `hvp_n_batches`/`hvp_batch_size`
    # kwargs are kept for compatibility but overridden when `measure_batch_size`
    # >= hvp_batch_size (the normal case).
    target_pool = data["target_pool"]; bg_pool = data["bg_pool"]

    np.random.seed(seed); torch.manual_seed(seed)

    # Fixed evaluation batches — sampled once, reused every step. This removes
    # between-step sampling variance from g_ft and g_pre, which otherwise
    # dominates the per-step T_first / T_second logs.
    def _fixed_batch(arr, n):
        n_use = min(n, len(arr))
        idx = np.random.choice(len(arr), n_use, replace=False)
        return arr[idx]
    eval_ft_batch  = _fixed_batch(data["eval_burst"], measure_batch_size)
    eval_pre_batch = _fixed_batch(data["eval_other"], measure_batch_size)

    # Deterministic HVP: one pass over the full fixed ft-eval batch.
    hvp_kw = dict(n_batches=1, batch_size=len(eval_ft_batch), epsilon=hvp_epsilon)

    log = {k: [] for k in LOG_KEYS}

    # Track both losses — on the FIXED g_ft-batch (matches Taylor math) and on
    # the full eval_burst (matches the trajectory plot / is what users "see").
    net.eval()
    lb_sub_prev  = eval_loss(net, eval_ft_batch,     batch_size=batch_size)
    lb_full_prev = eval_loss(net, data["eval_burst"], batch_size=batch_size)
    net.train()

    it = range(steps)
    pbar = tqdm(it, desc=desc) if progress else it
    for s in pbar:
        cur_lr = cosine_lr(s + 1, steps, lr, lr_end)

        # ── Fixed-batch g_ft at θ_t, plus fixed-batch g_pre for diagnostics. ──
        # Using the same examples every step eliminates the dominant source of
        # noise in the per-step Taylor terms.
        net.eval()
        g_ft  = _flat_grad(net, eval_ft_batch)
        g_pre = _flat_grad(net, eval_pre_batch)
        net.train()

        # Snapshot θ_t before the step.
        theta_before = _flat_sd_for_keys(state_dict_cpu(net), param_keys)

        # ── AdamW step on mixed batch. ──
        n_ft = int(round(c * batch_size))
        mixed = _sample_batch(target_pool, bg_pool, n_ft, batch_size)
        train_step(net, optimizer, mixed, lr=cur_lr, grad_clip=grad_clip)

        # Compute actual Δθ.
        theta_after = _flat_sd_for_keys(state_dict_cpu(net), param_keys)
        delta = theta_after - theta_before
        delta_theta_norm = delta.norm().item()

        # ── Compute HVP at θ_t (start of interval) — briefly restore params. ──
        T_first = torch.dot(g_ft, delta).item()
        if delta_theta_norm > 1e-12:
            _load_params_from_flat(net, param_keys, theta_before)
            H_delta = hvp_ft(net, delta, param_keys, eval_ft_batch, **hvp_kw).cpu()
            _load_params_from_flat(net, param_keys, theta_after)
            T_second = 0.5 * torch.dot(delta, H_delta).item()
            rayleigh = 2.0 * T_second / (delta_theta_norm ** 2)
        else:
            T_second = 0.0
            rayleigh = np.nan
        pred_dL = T_first + T_second

        # ── Evaluate loss at θ_{t+1} on BOTH the fixed ft-batch (matches Taylor
        #   math) and the full eval_burst (matches trajectory plot). ──
        if s % eval_every == 0 or s == steps - 1:
            net.eval()
            lb_sub_now  = eval_loss(net, eval_ft_batch,     batch_size=batch_size)
            lb_full_now = eval_loss(net, data["eval_burst"], batch_size=batch_size)
            lo_now = eval_loss(net, data["eval_other"], batch_size=batch_size)
            ab_now = eval_accuracy(net, data["eval_burst"], data["prompt_len"])
            ao_now = eval_accuracy(net, data["eval_other"], data["prompt_len"])
            drift_now = weight_drift_l2(sd_init, state_dict_cpu(net))["total"]
            net.train()
            # dL_actual on the SAME subset g_ft/H_ft were computed on — this is
            # what `T_first + T_second` is predicting.
            dL_actual = lb_sub_now - lb_sub_prev
            dL_actual_full = lb_full_now - lb_full_prev
            lb_sub_prev, lb_full_prev = lb_sub_now, lb_full_now

            g_ft_dot_g_pre = torch.dot(g_ft, g_pre).item()
            for k, v in dict(
                step=s, lr=cur_lr,
                loss_burst=lb_full_now, loss_other=lo_now,
                acc_burst=ab_now, acc_other=ao_now, weight_drift=drift_now,
                g_ft_norm=g_ft.norm().item(), g_pre_norm=g_pre.norm().item(),
                g_ft_dot_g_pre=g_ft_dot_g_pre,
                cos_g_ft_g_pre=_cos(g_ft, g_pre),
                delta_theta_norm=delta_theta_norm,
                T_first=T_first, T_second=T_second, pred_dL=pred_dL,
                dL_actual=dL_actual, dL_actual_full=dL_actual_full,
                rayleigh=rayleigh,
            ).items():
                log[k].append(v)

            if progress:
                pbar.set_postfix(lb=f"{lb_full_now:.3f}", ab=f"{ab_now:.3f}")

    return {
        "log": {k: np.array(v) for k, v in log.items()},
        "final_sd": state_dict_cpu(net),
        "net": net,
    }
