"""Training phase driven by `DecomposedAdam`.

Mirrors `simple.adam_taylor.adam_taylor_phase`, but instead of running
`torch.optim.AdamW` alongside shadow EMAs, we run Adam ourselves
(`simple.manual_adam.DecomposedAdam`) and compute the ft/pre/cross
decomposition as part of the optimizer state — so the identities

    m_t = c · m_ft + (1-c) · m_pre
    v_t = c² · v_ft + (1-c)² · v_pre + 2c(1-c) · v_cross

hold **by construction** (in fp32) at every step. No `m_recon_err`, no
`delta_theta_ratio`/`cos` drift — those quantities collapse to machine
precision, and the Taylor comparisons against actual ΔL_ft are driven
purely by Taylor truncation error rather than optimizer mismatch.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from simple import make_data
from simple.model import (
    DEVICE, load_model, eval_loss, eval_accuracy, cosine_lr,
)
from simple.interp import _get_grad_vector, state_dict_cpu, weight_drift_l2, _raw

from simple.manual_adam import DecomposedAdam
from simple.adam_taylor import (
    _grad_param_keys, _flat_sd_for_keys,
    _sample_pure, hvp_ft, _taylor_terms, _cos,
)


# ── gradient computation without flattening ──────────────────────────────

def _grads_per_param(net, batch_np):
    """Return gradients on `batch_np` as a list[Tensor] in `net.parameters()` order.

    Mirrors `_get_grad_vector` but keeps each parameter's gradient as a tensor
    so we can feed it directly to `DecomposedAdam.step_decomposed`.
    """
    dat = torch.as_tensor(batch_np, dtype=torch.long, device=DEVICE)
    inp, tgt = dat[:, :-1], dat[:, 1:]
    for p in net.parameters():
        if p.grad is not None:
            p.grad = None
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        logits = net(inp)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
    loss.backward()
    return [p.grad.detach().clone() for p in net.parameters() if p.requires_grad]


# ── logging (same keys as adam_taylor) ───────────────────────────────────

LOG_KEYS = [
    "step", "lr", "loss_burst", "loss_other", "acc_burst", "acc_other", "weight_drift",
    "g_ft_norm", "g_pre_norm", "g_ft_dot_g_pre", "cos_g_ft_g_pre",
    "m_ft_norm", "m_pre_norm", "cos_m_ft_m_pre",
    "m_recon_err",
    "P_mean", "P_std", "P_min", "P_max",
    "u_ft_norm", "u_pre_norm",
    "u_ft_dot_g_ft", "u_pre_dot_g_ft",
    "T1", "T2", "T3", "T4", "T5", "T345_total",
    "A_first", "A_quad", "dL_actual", "delta_theta_norm",
    "rayleigh_actual",
    "delta_theta_pred_norm",
    "delta_theta_ratio",
    "delta_theta_cos",
    # extra bookkeeping unique to this notebook
    "v_ft_mean", "v_pre_mean", "v_cross_mean",
    "v_ft_sum",  "v_pre_sum",  "v_cross_sum",
    "frac_vx_pos", "frac_vx_neg",
    "corr_mean",
    "dvhat_dc_mean", "dvhat_dc_median",
]


def _log_entry(log, **kw):
    for k in LOG_KEYS:
        log[k].append(kw.get(k, np.nan))


def _flat_concat(tensor_list):
    """Concatenate a list of tensors into one flat CPU fp32 vector."""
    return torch.cat([t.detach().cpu().flatten().to(torch.float32) for t in tensor_list])


# ── measurement ──────────────────────────────────────────────────────────

def _measure(net, opt, c, cur_lr, step_val, data,
             param_keys, sd_init, log, *,
             batch_size=512, eval_burst_bs=512,
             taylor=False,
             prev_theta_flat=None, prev_lb=None,
             adam_eps=1e-8,
             hvp_n_batches=4, hvp_batch_size=256, hvp_epsilon=1e-3,
             hvp_data_np=None):
    """Log diagnostics. Reads m_ft/m_pre/v_ft/v_pre/v_cross from `opt` directly.

    `m_recon_err` is machine precision by construction (kept in the log for
    backwards-compatibility with the adam_taylor plotting cells).
    """
    net.eval()
    ab = eval_accuracy(net, data["eval_burst"], data["prompt_len"])
    ao = eval_accuracy(net, data["eval_other"], data["prompt_len"])
    lb = eval_loss(net, data["eval_burst"], batch_size=eval_burst_bs)
    lo = eval_loss(net, data["eval_other"], batch_size=eval_burst_bs)
    sd_now = state_dict_cpu(net)
    drift = weight_drift_l2(sd_init, sd_now)["total"]

    # Fresh gradients on eval-set batches (so g matches the loss we compare to).
    b_ft  = data["eval_burst"][np.random.choice(len(data["eval_burst"]),  batch_size, replace=True)]
    b_pre = data["eval_other"][np.random.choice(len(data["eval_other"]), batch_size, replace=True)]
    g_ft  = _get_grad_vector(net, b_ft).detach().cpu()
    g_pre = _get_grad_vector(net, b_pre).detach().cpu()

    # Read everything from the optimizer's own state.
    bc1, bc2 = opt.bias_correction()
    m_ft_flat  = _flat_concat(opt.m_ft)
    m_pre_flat = _flat_concat(opt.m_pre)
    v_ft_flat  = _flat_concat(opt.v_ft)
    v_pre_flat = _flat_concat(opt.v_pre)
    vx_flat    = _flat_concat(opt.v_cross)
    m_combined = _flat_concat(opt.m)
    v_combined = _flat_concat(opt.v)

    mft_h   = m_ft_flat  / bc1
    mpre_h  = m_pre_flat / bc1
    vft_h   = v_ft_flat  / bc2
    vpre_h  = v_pre_flat / bc2
    vx_h    = vx_flat    / bc2
    v_hat   = v_combined / bc2

    P = 1.0 / (torch.sqrt(torch.clamp(v_hat, min=0.0)) + adam_eps)
    u_ft  = P * mft_h
    u_pre = P * mpre_h

    # Sanity: the decomposition identity is exact in fp32, so m_recon_err ~ 0.
    m_recon = c * m_ft_flat + (1 - c) * m_pre_flat
    num = (m_recon - m_combined).norm().item()
    den = max(m_combined.norm().item(), 1e-20)
    m_recon_err = num / den

    dvhat_dc = 2 * c * vft_h - 2 * (1 - c) * vpre_h + 2 * (1 - 2 * c) * vx_h
    denom = torch.sqrt(vft_h.clamp(min=0) * vpre_h.clamp(min=0)) + 1e-20
    corr = (vx_h / denom).clamp(min=-1.5, max=1.5)

    T1 = T2 = T3 = T4 = T5 = np.nan
    T345_total = np.nan
    A_first = A_quad = np.nan
    dL_actual = np.nan
    delta_theta_norm = 0.0
    rayleigh_actual = np.nan
    delta_theta_pred_norm = np.nan
    delta_theta_ratio = np.nan
    delta_theta_cos = np.nan

    theta_now_flat = _flat_sd_for_keys(sd_now, param_keys)

    # Δθ_pred for the step that *just* occurred: computed from the optimizer's
    # current state with the same Adam update rule used inside `step_decomposed`.
    step_size = cur_lr / bc1
    delta_pred = -step_size * m_combined / (torch.sqrt(torch.clamp(v_combined, min=0.0))
                                             / np.sqrt(bc2) + adam_eps)
    delta_theta_pred_norm = delta_pred.norm().item()

    if taylor:
        T1 = (-cur_lr * c         * torch.dot(g_ft, u_ft)).item()
        T2 = (-cur_lr * (1 - c)   * torch.dot(g_ft, u_pre)).item()
        hvp_kw = dict(n_batches=hvp_n_batches, batch_size=hvp_batch_size,
                      epsilon=hvp_epsilon)
        # Individual T3, T4, T5 use HVPs along u_ft and u_pre (for interpretability).
        H_uft  = hvp_ft(net, u_ft,  param_keys, hvp_data_np, **hvp_kw).cpu()
        H_upre = hvp_ft(net, u_pre, param_keys, hvp_data_np, **hvp_kw).cpu()
        _, _, T3, T4, T5 = _taylor_terms(g_ft, u_ft, u_pre, H_uft, H_upre, c, cur_lr)

        if prev_theta_flat is not None:
            delta = theta_now_flat - prev_theta_flat
            delta_theta_norm = delta.norm().item()
            A_first = torch.dot(g_ft, delta).item()
            if delta_theta_norm > 1e-12:
                # One HVP along the *actual* Δθ, reused for both A_quad and T345_total.
                # Because DecomposedAdam's step formula is exactly Δθ_pred, we have
                # Δθ_real == Δθ_pred (within fp32), so ½·Δθ_pred^T H Δθ_pred ==
                # ½·Δθ_real^T H Δθ_real. Sharing the HVP removes independent-HVP
                # noise between red (pred 1st+2nd) and blue (A_first - A_quad).
                H_delta = hvp_ft(net, delta, param_keys, hvp_data_np, **hvp_kw).cpu()
                A_quad = 0.5 * torch.dot(delta, H_delta).item()
                T345_total = A_quad
                rayleigh_actual = 2.0 * A_quad / (delta_theta_norm ** 2)
                delta_theta_ratio = delta_theta_pred_norm / delta_theta_norm
                delta_theta_cos = _cos(delta_pred, delta)
            else:
                A_quad = 0.0
                T345_total = 0.0
            if prev_lb is not None:
                dL_actual = lb - prev_lb
        else:
            # Initial measurement at step=-1: no Δθ available, compute T345_total
            # from its own HVP along the (trivial) predicted step at the start.
            H_delta_pred = hvp_ft(net, delta_pred, param_keys, hvp_data_np, **hvp_kw).cpu()
            T345_total = 0.5 * torch.dot(delta_pred, H_delta_pred).item()

    _log_entry(log,
        step=step_val, lr=cur_lr,
        loss_burst=lb, loss_other=lo, acc_burst=ab, acc_other=ao, weight_drift=drift,
        g_ft_norm=g_ft.norm().item(), g_pre_norm=g_pre.norm().item(),
        g_ft_dot_g_pre=torch.dot(g_ft, g_pre).item(),
        cos_g_ft_g_pre=_cos(g_ft, g_pre),
        m_ft_norm=mft_h.norm().item(), m_pre_norm=mpre_h.norm().item(),
        cos_m_ft_m_pre=_cos(mft_h, mpre_h),
        m_recon_err=m_recon_err,
        P_mean=P.mean().item(), P_std=P.std().item(),
        P_min=P.min().item(), P_max=P.max().item(),
        u_ft_norm=u_ft.norm().item(), u_pre_norm=u_pre.norm().item(),
        u_ft_dot_g_ft=torch.dot(u_ft, g_ft).item(),
        u_pre_dot_g_ft=torch.dot(u_pre, g_ft).item(),
        T1=T1, T2=T2, T3=T3, T4=T4, T5=T5, T345_total=T345_total,
        A_first=A_first, A_quad=A_quad,
        dL_actual=dL_actual, delta_theta_norm=delta_theta_norm,
        rayleigh_actual=rayleigh_actual,
        delta_theta_pred_norm=delta_theta_pred_norm,
        delta_theta_ratio=delta_theta_ratio,
        delta_theta_cos=delta_theta_cos,
        v_ft_mean=vft_h.mean().item(), v_pre_mean=vpre_h.mean().item(),
        v_cross_mean=vx_h.mean().item(),
        v_ft_sum=vft_h.sum().item(), v_pre_sum=vpre_h.sum().item(),
        v_cross_sum=vx_flat.sum().item(),
        frac_vx_pos=(vx_h > 0).float().mean().item(),
        frac_vx_neg=(vx_h < 0).float().mean().item(),
        corr_mean=corr.mean().item(),
        dvhat_dc_mean=dvhat_dc.mean().item(),
        dvhat_dc_median=dvhat_dc.median().item(),
    )
    net.train()
    return theta_now_flat, lb


def _save_snapshot(opt, out_dir, tag):
    """Dump all five decomposition EMAs for offline analysis."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    torch.save({
        "t":      opt.step_count,
        "b1":     opt.b1,
        "b2":     opt.b2,
        "eps":    opt.eps,
        "m_ft":   [t.detach().cpu().clone() for t in opt.m_ft],
        "m_pre":  [t.detach().cpu().clone() for t in opt.m_pre],
        "v_ft":   [t.detach().cpu().clone() for t in opt.v_ft],
        "v_pre":  [t.detach().cpu().clone() for t in opt.v_pre],
        "v_cross":[t.detach().cpu().clone() for t in opt.v_cross],
    }, Path(out_dir) / f"decompadam_{tag}.pt")


# ── one training phase ───────────────────────────────────────────────────

def manual_adam_phase(
    data, start_ckpt_or_net, param_keys,
    *, c, steps, lr, lr_end=None, batch_size=512, eval_every=1,
    seed=42,
    adam_beta1=0.9, adam_beta2=0.9, adam_eps=1e-8,
    desc="phase", snapshot_dir=None, snapshot_tag="",
    taylor=True, hvp_n_batches=4, hvp_batch_size=256, hvp_epsilon=1e-3,
    hvp_on="eval",
    progress=True,
):
    lr_end = lr if lr_end is None else lr_end
    target_pool = data["target_pool"]; bg_pool = data["bg_pool"]

    if hvp_on == "eval":
        hvp_data_np = data["eval_burst"]
    elif hvp_on == "train":
        hvp_data_np = np.concatenate(list(target_pool.values()), axis=0)
    else:
        raise ValueError(f"hvp_on must be 'eval' or 'train', got {hvp_on!r}")

    np.random.seed(seed); torch.manual_seed(seed)

    if hasattr(start_ckpt_or_net, "state_dict"):
        net = start_ckpt_or_net
    else:
        net = load_model(start_ckpt_or_net, data["vocab_size"], data["context_size"],
                         compile_model=False)

    # Our own optimizer — no torch.optim, no shadow EMAs.
    params = [p for p in net.parameters() if p.requires_grad]
    opt = DecomposedAdam(params, lr=lr,
                         betas=(adam_beta1, adam_beta2), eps=adam_eps)

    sd_init = state_dict_cpu(net)

    log = {k: [] for k in LOG_KEYS}
    prev_theta, prev_lb = _measure(
        net, opt, c, lr, -1, data, param_keys, sd_init, log,
        batch_size=batch_size, taylor=taylor,
        prev_theta_flat=None, prev_lb=None,
        adam_eps=adam_eps,
        hvp_n_batches=hvp_n_batches, hvp_batch_size=hvp_batch_size,
        hvp_epsilon=hvp_epsilon, hvp_data_np=hvp_data_np,
    )

    it = range(steps)
    pbar = tqdm(it, desc=desc) if progress else it
    for s in pbar:
        n_ft  = int(round(c * batch_size))
        n_pre = batch_size - n_ft

        # Step-exact split: sample ft and pre sub-batches separately, feed
        # their exact sub-batch gradients to the decomposition.
        ft_part  = _sample_pure(target_pool, n_ft)  if n_ft  > 0 else None
        pre_part = _sample_pure(bg_pool,     n_pre) if n_pre > 0 else None
        g_ft  = _grads_per_param(net, ft_part)  if ft_part  is not None else None
        g_pre = _grads_per_param(net, pre_part) if pre_part is not None else None

        cur_lr = cosine_lr(s + 1, steps, lr, lr_end)
        opt.set_lr(cur_lr)
        opt.step_decomposed(g_ft, g_pre, c=c)

        if s % eval_every == 0 or s == steps - 1:
            prev_theta, prev_lb = _measure(
                net, opt, c, cur_lr, s, data, param_keys, sd_init, log,
                batch_size=batch_size, taylor=taylor,
                prev_theta_flat=prev_theta, prev_lb=prev_lb,
                adam_eps=adam_eps,
                hvp_n_batches=hvp_n_batches, hvp_batch_size=hvp_batch_size,
                hvp_epsilon=hvp_epsilon, hvp_data_np=hvp_data_np,
            )
            if progress:
                pbar.set_postfix(lb=f"{log['loss_burst'][-1]:.3f}",
                                 ab=f"{log['acc_burst'][-1]:.3f}")

    if snapshot_dir is not None:
        _save_snapshot(opt, snapshot_dir, snapshot_tag)

    return {
        "log": {k: np.array(v) for k, v in log.items()},
        "optimizer": opt,
        "final_sd": state_dict_cpu(net),
        "net": net,
        "c": c,
    }


# ── worker entry point ──────────────────────────────────────────────────

def run_one_c_manual_adam(
    c: float,
    pretrain_ckpt: str,
    data_kwargs: dict,
    out_root: str,
    *,
    ft_steps: int, ft_lr: float, ft_eval_every: int,
    fg_steps: int, fg_lr: float, fg_eval_every: int,
    batch_size: int, seed: int,
    taylor: bool = True,
    hvp_on: str = "eval",
    progress: bool = False,
) -> str:
    np.random.seed(seed); torch.manual_seed(seed)

    data = make_data(**data_kwargs)
    dummy = load_model(pretrain_ckpt, data["vocab_size"], data["context_size"],
                       compile_model=False)
    param_keys = _grad_param_keys(dummy)
    del dummy

    snapshot_dir = str(Path(out_root) / "snapshots")

    ft = manual_adam_phase(
        data, pretrain_ckpt, param_keys,
        c=c, steps=ft_steps, lr=ft_lr, lr_end=ft_lr * 0.5,
        batch_size=batch_size, eval_every=ft_eval_every, seed=seed,
        desc=f"finetune c={c}", taylor=taylor, hvp_on=hvp_on,
        snapshot_dir=snapshot_dir, snapshot_tag=f"ft_c{c}",
        progress=progress,
    )
    fg = manual_adam_phase(
        data, ft["net"], param_keys,
        c=0.0, steps=fg_steps, lr=fg_lr, lr_end=fg_lr * 0.5,
        batch_size=batch_size, eval_every=fg_eval_every, seed=seed + 1,
        desc=f"forget from c={c}", taylor=taylor, hvp_on=hvp_on,
        snapshot_dir=snapshot_dir, snapshot_tag=f"fg_from_c{c}",
        progress=progress,
    )

    out_path = Path(out_root) / "results" / f"run_c{c}.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "c": c,
        "finetune_log": ft["log"],
        "forget_log": fg["log"],
        "ft_snapshot": str(Path(snapshot_dir) / f"decompadam_ft_c{c}.pt"),
        "fg_snapshot": str(Path(snapshot_dir) / f"decompadam_fg_from_c{c}.pt"),
    }
    with open(out_path, "wb") as fh:
        pickle.dump(blob, fh)
    return str(out_path)
