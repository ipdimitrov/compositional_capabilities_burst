"""SGD-based forgetting with a clean 2-term Taylor decomposition.

At c=0 with plain SGD, the update is Δθ = -η · g_pre, so

    ΔL_ft  ≈  -η · g_ft · g_pre  +  ½η² · g_pre^T H_ft g_pre
           =  T₂  +  T₅

where g_ft, g_pre, H_ft are all evaluated at the **start** of the interval
(i.e., the same θ the optimizer used to compute the step). With this
start-of-interval convention, the quadratic has a **+** sign and the
decomposition is step-exact:

    T₂ = A_first   (both = -η · g_ft · g_pre, using the SAME gradients)
    T₅ = A_quad    (both = ½η² · g_pre^T H_ft g_pre, using ONE HVP)

so red and blue in the cumulative plot should coincide, and any gap between
them and black is pure 3rd-order-Taylor truncation.

Two terms, no preconditioner, no sampling mismatch between prediction and
actual. As clean as it gets.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from simple.model import DEVICE, load_model, eval_loss, eval_accuracy, cosine_lr
from simple.interp import _get_grad_vector, state_dict_cpu, weight_drift_l2, _raw
from simple.finetune import _sample_batch

from simple.adam_taylor import (
    _grad_param_keys, _flat_sd_for_keys, _sample_pure, hvp_ft, _cos,
)


LOG_KEYS = [
    "step", "lr", "loss_burst", "loss_other", "acc_burst", "acc_other", "weight_drift",
    "g_ft_norm", "g_pre_norm", "g_ft_dot_g_pre", "cos_g_ft_g_pre",
    # SGD 2-term Taylor (start-of-interval; + sign on quadratic)
    "T2",          # -η · g_ft · g_pre
    "T5",          # ½η² · g_pre^T H_ft g_pre
    "pred_dL",     # T2 + T5
    # Actual-Δθ Taylor — equal to T2, T5 by construction (they're logged so the
    # existing "red vs blue" plotting machinery keeps working).
    "A_first",
    "A_quad",
    "dL_actual",   # loss_burst[t+1] - loss_burst[t] on eval_burst
    "delta_theta_norm",
    "rayleigh_pre", # 2·T5 / (η²·‖g_pre‖²) = curvature along the step direction
]


def _flat_grad(net, batch_np):
    return _get_grad_vector(net, batch_np).detach().cpu()


def _apply_flat_step(net, param_keys, delta_flat):
    """θ ← θ + delta_flat (flat CPU tensor, reshaped into param shapes)."""
    raw = _raw(net)
    name_to_param = {n: p for n, p in raw.named_parameters() if p.requires_grad}
    off = 0
    with torch.no_grad():
        for k in param_keys:
            p = name_to_param[k]
            n = p.numel()
            p.add_(delta_flat[off:off + n].to(p.device).to(p.dtype).reshape(p.shape))
            off += n


def sgd_forget_with_taylor(
    data, start_net, *,
    steps: int, lr: float, lr_end: float | None = None,
    batch_size: int = 512, eval_every: int = 1, seed: int = 42,
    hvp_n_batches: int = 4, hvp_batch_size: int = 256, hvp_epsilon: float = 1e-3,
    hvp_on: str = "eval",
    desc: str = "sgd forget", progress: bool = True,
) -> dict:
    """Run `steps` of plain SGD on pretraining batches and log the 2-term Taylor
    decomposition at each step.

    Implementation detail: the Taylor terms are evaluated at the *start* of each
    step, using the exact `g_pre` that drives the step and one shared HVP. This
    makes `T₂ ≡ A_first` and `T₅ ≡ A_quad` by construction (no sampling mismatch),
    so the cumulative red and blue curves overlap and the only residual against
    black is 3rd-order Taylor truncation.

    `hvp_on="eval"` uses `data["eval_burst"]` for the Hessian (matches `loss_burst`).
    """
    lr_end = lr if lr_end is None else lr_end
    net = start_net
    param_keys = _grad_param_keys(net)
    sd_init = state_dict_cpu(net)

    if hvp_on == "eval":
        hvp_data_np = data["eval_burst"]
    elif hvp_on == "train":
        hvp_data_np = np.concatenate(list(data["target_pool"].values()), axis=0)
    else:
        raise ValueError(f"hvp_on must be 'eval' or 'train'")

    np.random.seed(seed); torch.manual_seed(seed)

    log = {k: [] for k in LOG_KEYS}
    hvp_kw = dict(n_batches=hvp_n_batches, batch_size=hvp_batch_size, epsilon=hvp_epsilon)

    # Evaluate initial state (θ_0). All measurements that depend on "Δθ from
    # previous eval" (dL_actual, A_first, etc.) are logged in the NEXT iteration.
    net.eval()
    lb_prev = eval_loss(net, data["eval_burst"], batch_size=batch_size)
    lo_prev = eval_loss(net, data["eval_other"], batch_size=batch_size)
    ab_prev = eval_accuracy(net, data["eval_burst"], data["prompt_len"])
    ao_prev = eval_accuracy(net, data["eval_other"], data["prompt_len"])
    drift_prev = weight_drift_l2(sd_init, state_dict_cpu(net))["total"]
    net.train()

    it = range(steps)
    pbar = tqdm(it, desc=desc) if progress else it
    for s in pbar:
        cur_lr = cosine_lr(s + 1, steps, lr, lr_end)

        # ── Measurement at θ_t (the CURRENT params, before stepping). ──
        # This is the "start of interval" for the step we're about to take.
        net.eval()
        # g_pre drives the step. Sample from bg_pool (the "training distribution").
        b_pre = _sample_pure(data["bg_pool"], batch_size)
        g_pre = _flat_grad(net, b_pre)
        # g_ft is a fresh sample on the EVAL burst distribution (matches loss_burst).
        b_ft = data["eval_burst"][np.random.choice(len(data["eval_burst"]), batch_size, replace=True)]
        g_ft = _flat_grad(net, b_ft)

        g_ft_dot_g_pre = torch.dot(g_ft, g_pre).item()
        T2 = -cur_lr * g_ft_dot_g_pre
        # One HVP along g_pre — shared between T5 and A_quad.
        H_gpre = hvp_ft(net, g_pre, param_keys, hvp_data_np, **hvp_kw).cpu()
        gpre_H_gpre = torch.dot(g_pre, H_gpre).item()
        T5 = 0.5 * cur_lr ** 2 * gpre_H_gpre
        pred_dL = T2 + T5  # start-of-interval sign convention

        # Actual-Δθ Taylor is *identical* to (T2, T5) by construction:
        #   A_first = g_ft · (-η · g_pre) = -η · g_ft · g_pre = T2
        #   A_quad  = ½ · (-η · g_pre)^T H (-η · g_pre) = ½η² · g_pre^T H g_pre = T5
        A_first = T2
        A_quad = T5
        delta = -cur_lr * g_pre
        delta_theta_norm = delta.norm().item()
        rayleigh_pre = (gpre_H_gpre / max(g_pre.norm().item() ** 2, 1e-20))

        # ── Apply SGD step: θ_{t+1} = θ_t - η · g_pre ──
        net.train()
        _apply_flat_step(net, param_keys, -cur_lr * g_pre)

        # ── Evaluate at θ_{t+1} to get dL_actual for this step. ──
        if s % eval_every == 0 or s == steps - 1:
            net.eval()
            lb_now = eval_loss(net, data["eval_burst"], batch_size=batch_size)
            lo_now = eval_loss(net, data["eval_other"], batch_size=batch_size)
            ab_now = eval_accuracy(net, data["eval_burst"], data["prompt_len"])
            ao_now = eval_accuracy(net, data["eval_other"], data["prompt_len"])
            drift_now = weight_drift_l2(sd_init, state_dict_cpu(net))["total"]
            dL_actual = lb_now - lb_prev
            net.train()

            for k, v in dict(
                step=s, lr=cur_lr,
                loss_burst=lb_now, loss_other=lo_now,
                acc_burst=ab_now, acc_other=ao_now, weight_drift=drift_now,
                g_ft_norm=g_ft.norm().item(), g_pre_norm=g_pre.norm().item(),
                g_ft_dot_g_pre=g_ft_dot_g_pre,
                cos_g_ft_g_pre=_cos(g_ft, g_pre),
                T2=T2, T5=T5, pred_dL=pred_dL,
                A_first=A_first, A_quad=A_quad, dL_actual=dL_actual,
                delta_theta_norm=delta_theta_norm,
                rayleigh_pre=rayleigh_pre,
            ).items():
                log[k].append(v)

            lb_prev, lo_prev, ab_prev, ao_prev, drift_prev = (
                lb_now, lo_now, ab_now, ao_now, drift_now
            )

            if progress:
                pbar.set_postfix(lb=f"{lb_now:.3f}", ab=f"{ab_now:.3f}")

    return {
        "log": {k: np.array(v) for k, v in log.items()},
        "final_sd": state_dict_cpu(net),
        "net": net,
    }
