"""Adam Taylor decomposition, simplified.

Unlike `simple.adam_decomp`, here we do *not* decompose Adam's second moment
`v_t` into ft/pre/cross pieces. Instead we:

- Read `v_t` directly from the real AdamW optimizer state (and build P = 1/(√v̂+ε)
  from that real `v`, so P is *exactly* what Adam is using).
- Keep two shadow first-moment EMAs `m̂_ft`, `m̂_pre` alongside the optimizer, so
  we can split the update numerator `m_t = c m̂_ft + (1-c) m̂_pre` by task.
- Compute the 5-term Taylor expansion of ΔL_ft in the Adam geometry, analogous
  to the SGD version in `notebooks/burst_taylor_decomp.ipynb`:

    T1 = -η · c       · g_ft^T  · (P m̂_ft)          # "replay helping"
    T2 = -η · (1-c)   · g_ft^T  · (P m̂_pre)          # first-order conflict
    T3 = ½η² · c²     · (P m̂_ft)^T  H_ft (P m̂_ft)   # ft-ft curvature
    T4 =  η² · c(1-c) · (P m̂_ft)^T  H_ft (P m̂_pre)  # curvature damage
    T5 = ½η² · (1-c)² · (P m̂_pre)^T H_ft (P m̂_pre)  # pre-pre curvature

Interpretation matches the SGD 5-term decomp exactly (T1 = replay helping,
T4 = curvature damage, etc.), but the geometry is now the correct Adam one.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from simple import make_data
from simple.model import (
    DEVICE, load_model, eval_loss, eval_accuracy,
    make_optimizer, reset_optimizer, train_step, cosine_lr,
)
from simple.interp import _get_grad_vector, state_dict_cpu, weight_drift_l2, _raw
from simple.finetune import _sample_batch


# ── flat-vector + sampling utilities ─────────────────────────────────────

def _grad_param_keys(net):
    raw = _raw(net)
    return [n for n, p in raw.named_parameters() if p.requires_grad]


def _flat_sd_for_keys(sd, keys):
    return torch.cat([sd[k].flatten() for k in keys])


def _unflatten_to_sd(flat, sd_ref, keys):
    out = {}
    off = 0
    for k in keys:
        n = sd_ref[k].numel()
        out[k] = flat[off:off + n].reshape(sd_ref[k].shape)
        off += n
    return out


def _grad_on_batch(net, batch_np):
    return _get_grad_vector(net, batch_np).detach().cpu()


def _sample_pure(pool, batch_size):
    parts, ids = [], list(pool.keys())
    per = batch_size // len(ids)
    rem = batch_size % len(ids)
    for i, tid in enumerate(ids):
        k = per + (1 if i < rem else 0)
        if k > 0:
            idx = np.random.randint(len(pool[tid]), size=k)
            parts.append(pool[tid][idx])
    return np.concatenate(parts)[np.random.permutation(batch_size)]


def _avg_grad(net, batch_source_np, n_batches=4, batch_size=256):
    acc = None
    for _ in range(n_batches):
        idx = np.random.choice(len(batch_source_np),
                               size=min(batch_size, len(batch_source_np)),
                               replace=False)
        g = _get_grad_vector(net, batch_source_np[idx])
        acc = g if acc is None else acc + g
    return acc / n_batches


def hvp_ft(net, direction, param_keys, data_np, *,
           n_batches=4, batch_size=256, epsilon=1e-3):
    """Finite-difference HVP H @ direction. Returns flat GPU tensor."""
    raw = _raw(net)
    sd_cur = {k: v.detach().cpu().clone() for k, v in raw.state_dict().items()}
    cur_flat = _flat_sd_for_keys(sd_cur, param_keys)

    norm = direction.norm().item()
    if norm < 1e-12:
        return torch.zeros_like(direction).to(DEVICE)
    vhat = direction / norm

    def _grad_at(alpha):
        perturbed = cur_flat + alpha * vhat
        sd_new = dict(sd_cur)
        sd_new.update(_unflatten_to_sd(perturbed, sd_cur, param_keys))
        raw.load_state_dict({k: v.to(DEVICE) for k, v in sd_new.items()})
        return _avg_grad(net, data_np, n_batches=n_batches, batch_size=batch_size)

    g_plus = _grad_at(epsilon)
    g_minus = _grad_at(-epsilon)
    raw.load_state_dict({k: v.to(DEVICE) for k, v in sd_cur.items()})
    return (g_plus - g_minus) / (2 * epsilon) * norm


# ── read P, m directly from AdamW's state ────────────────────────────────

def _read_opt_m_v_step(optimizer, param_keys, net):
    """Flat CPU `(m_real, v_real, step)` from AdamW's state, in param_keys order."""
    raw = _raw(net)
    name_to_param = {n: p for n, p in raw.named_parameters() if p.requires_grad}
    m_parts, v_parts, step_int = [], [], 0
    for k in param_keys:
        p = name_to_param[k]
        st = optimizer.state.get(p, {})
        m_parts.append(st.get("exp_avg",    torch.zeros_like(p)).detach().cpu().flatten())
        v_parts.append(st.get("exp_avg_sq", torch.zeros_like(p)).detach().cpu().flatten())
        s = st.get("step", 0)
        step_int = int(s.item()) if torch.is_tensor(s) else int(s)
    return torch.cat(m_parts), torch.cat(v_parts), step_int


def _preconditioner_from_opt(v_real, step_int, beta2=0.9, eps=1e-8):
    """P = 1/(√v̂ + ε), built from optimizer's *actual* v (bias-corrected)."""
    if step_int <= 0:
        return torch.ones_like(v_real)
    v_hat = v_real / (1.0 - beta2 ** step_int)
    return 1.0 / (torch.sqrt(torch.clamp(v_hat, min=0.0)) + eps)


# ── shadow first-moment EMAs only ────────────────────────────────────────

class ShadowMoments:
    """Two EMAs of pure-ft and pure-pre gradients, same β₁ as AdamW.

    Satisfies `m_t = c · m_ft + (1-c) · m_pre` in expectation (exactly if we fed
    the exact sub-batches that built the mixed gradient).
    """
    def __init__(self, n, beta1=0.9, device="cpu"):
        self.b1 = beta1
        z = lambda: torch.zeros(n, dtype=torch.float32, device=device)
        self.m_ft, self.m_pre = z(), z()
        self.t = 0

    def update(self, g_ft, g_pre):
        self.t += 1
        self.m_ft .mul_(self.b1).add_(g_ft,  alpha=1 - self.b1)
        self.m_pre.mul_(self.b1).add_(g_pre, alpha=1 - self.b1)

    def bias_corrected(self):
        bc = 1.0 - self.b1 ** self.t if self.t > 0 else 1.0
        return self.m_ft / bc, self.m_pre / bc


# ── logging / measurement ────────────────────────────────────────────────

LOG_KEYS = [
    "step", "lr", "loss_burst", "loss_other", "acc_burst", "acc_other", "weight_drift",
    # pure grads
    "g_ft_norm", "g_pre_norm", "g_ft_dot_g_pre", "cos_g_ft_g_pre",
    # shadow first moments (bias-corrected)
    "m_ft_norm", "m_pre_norm", "cos_m_ft_m_pre",
    # sanity: m_t ≈ c m̂_ft + (1-c) m̂_pre
    "m_recon_err",
    # preconditioner (from real optimizer v)
    "P_mean", "P_std", "P_min", "P_max",
    # u_ft = P m̂_ft, u_pre = P m̂_pre (the per-parameter update pieces)
    "u_ft_norm", "u_pre_norm",
    "u_ft_dot_g_ft", "u_pre_dot_g_ft",
    # ── Adam Taylor terms for ΔL_ft ──
    "T1",   # -η c   g_ft · u_ft
    "T2",   # -η (1-c) g_ft · u_pre
    "T3",   # ½η² c²      u_ft^T  H_ft u_ft      (can blow up in cancellation dirs)
    "T4",   #  η² c(1-c)  u_ft^T  H_ft u_pre     (can blow up in cancellation dirs)
    "T5",   # ½η² (1-c)²  u_pre^T H_ft u_pre     (can blow up in cancellation dirs)
    "T345_total",  # ½·Δθ_pred^T H Δθ_pred — numerically stable version of T3+T4+T5
    # Actual-Δθ Taylor (real observed parameter change)
    "A_first",   # g_ft · Δθ
    "A_quad",    # ½ Δθ^T H_ft Δθ
    "dL_actual", # loss_burst[t] - loss_burst[t_prev]
    "delta_theta_norm",
    # curvature diagnostics
    "rayleigh_actual",   # 2·A_quad / ||Δθ||²
    # direct Δθ_pred vs Δθ_real comparison (at eval_every=1 this is per-step)
    "delta_theta_pred_norm",   # ||Δθ_pred|| = η·||P·(c·m̂_ft + (1-c)·m̂_pre)||
    "delta_theta_ratio",       # ||Δθ_pred|| / ||Δθ_real||
    "delta_theta_cos",         # cos(Δθ_pred, Δθ_real)
]


def _cos(a, b):
    na, nb = a.norm().item(), b.norm().item()
    if na < 1e-20 or nb < 1e-20:
        return 0.0
    return (torch.dot(a, b) / (na * nb)).item()


def _taylor_terms(g_ft, u_ft, u_pre, H_uft, H_upre, c, eta):
    """Pure-algebra 5-term decomposition of ΔL_ft for the Adam step
    Δθ = -η(c·u_ft + (1-c)·u_pre), with u_ft = P·m̂_ft and u_pre = P·m̂_pre.

    Returns (T1, T2, T3, T4, T5) as python floats such that
        Σ T_i  ==  g_ft · Δθ  +  ½ Δθ^T H Δθ
    exactly, for symmetric H and any choice of `c`, `eta`. Kept separate from
    `_measure` so the identity can be unit-tested without training.
    """
    T1 = (-eta * c         * torch.dot(g_ft, u_ft )).item()
    T2 = (-eta * (1 - c)   * torch.dot(g_ft, u_pre)).item()
    T3 = 0.5 * eta ** 2 * c * c         * torch.dot(u_ft,  H_uft ).item()
    T4 =       eta ** 2 * c * (1 - c)   * torch.dot(u_ft,  H_upre).item()
    T5 = 0.5 * eta ** 2 * (1 - c) ** 2  * torch.dot(u_pre, H_upre).item()
    return T1, T2, T3, T4, T5


def _log_entry(log, **kw):
    for k in LOG_KEYS:
        log[k].append(kw.get(k, np.nan))


def _measure(net, optimizer, shadow, c, cur_lr, step_val, data,
             param_keys, sd_init, log, *,
             batch_size=512, eval_burst_bs=512,
             taylor=False,
             prev_theta_flat=None, prev_lb=None,
             beta2=0.9, eps=1e-8,
             hvp_n_batches=4, hvp_batch_size=256, hvp_epsilon=1e-3,
             hvp_data_np=None):
    net.eval()
    ab = eval_accuracy(net, data["eval_burst"], data["prompt_len"])
    ao = eval_accuracy(net, data["eval_other"], data["prompt_len"])
    lb = eval_loss(net, data["eval_burst"], batch_size=eval_burst_bs)
    lo = eval_loss(net, data["eval_other"], batch_size=eval_burst_bs)
    sd_now = state_dict_cpu(net)
    drift = weight_drift_l2(sd_init, sd_now)["total"]

    # Fresh pure-ft and pure-pre gradients sampled from the *eval* distributions,
    # so the Taylor expansion predicts the eval-loss ΔL we compare against
    # (`loss_burst = eval_loss(data["eval_burst"])`, `loss_other = eval_loss(data["eval_other"])`).
    # HVP below (via hvp_on="eval") also uses eval_burst, so g and H are consistent.
    b_ft  = data["eval_burst"][np.random.choice(len(data["eval_burst"]), batch_size, replace=True)]
    b_pre = data["eval_other"][np.random.choice(len(data["eval_other"]), batch_size, replace=True)]
    g_ft  = _grad_on_batch(net, b_ft)
    g_pre = _grad_on_batch(net, b_pre)

    # Shadow moments + real Adam state → real P.
    mft_h, mpre_h = shadow.bias_corrected()
    m_real, v_real, t_real = _read_opt_m_v_step(optimizer, param_keys, net)
    P = _preconditioner_from_opt(v_real, t_real, beta2=beta2, eps=eps)

    # Sanity: m̂_t from AdamW vs  c · m̂_ft + (1-c) · m̂_pre
    if t_real > 0:
        m_real_hat = m_real / (1.0 - shadow.b1 ** t_real)
    else:
        m_real_hat = m_real
    m_recon = c * mft_h + (1 - c) * mpre_h
    m_recon_err = ((m_recon - m_real_hat).norm().item()
                   / max(m_real_hat.norm().item(), 1e-20))

    u_ft  = P * mft_h
    u_pre = P * mpre_h

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

    # Δθ_pred = -η · P · (c·m̂_ft + (1-c)·m̂_pre) using state at *this* step.
    # If prev_theta_flat is the parameters from one optimizer step ago,
    # Δθ_real = θ_now - θ_prev should match Δθ_pred (up to bf16 / sampling noise).
    delta_pred = -cur_lr * (c * u_ft + (1 - c) * u_pre)
    delta_theta_pred_norm = delta_pred.norm().item()

    if taylor:
        # 2 HVPs (H u_ft, H u_pre) → three bilinears.
        # T3, T4, T5 individually may have huge magnitudes in directions where
        # g_ft and g_pre cancel (small v_real → P blows up → u_ft, u_pre huge),
        # but mathematically T3+T4+T5 must equal ½·Δθ_pred^T H Δθ_pred, which is
        # bounded. We compute both: individual terms for interpretation, and the
        # total via one HVP on the assembled Δθ_pred for numerical stability.
        hvp_kw = dict(n_batches=hvp_n_batches, batch_size=hvp_batch_size,
                      epsilon=hvp_epsilon)
        H_uft  = hvp_ft(net, u_ft,  param_keys, hvp_data_np, **hvp_kw).cpu()
        H_upre = hvp_ft(net, u_pre, param_keys, hvp_data_np, **hvp_kw).cpu()
        T1, T2, T3, T4, T5 = _taylor_terms(g_ft, u_ft, u_pre, H_uft, H_upre, c, cur_lr)

        # Numerically stable total of the 2nd-order contribution.
        H_delta_pred = hvp_ft(net, delta_pred, param_keys, hvp_data_np, **hvp_kw).cpu()
        T345_total = 0.5 * torch.dot(delta_pred, H_delta_pred).item()

        # Actual-Δθ Taylor over this eval interval.
        if prev_theta_flat is not None:
            delta = theta_now_flat - prev_theta_flat
            delta_theta_norm = delta.norm().item()
            A_first = torch.dot(g_ft, delta).item()
            if delta_theta_norm > 1e-12:
                H_delta = hvp_ft(net, delta, param_keys, hvp_data_np, **hvp_kw).cpu()
                A_quad = 0.5 * torch.dot(delta, H_delta).item()
                rayleigh_actual = 2.0 * A_quad / (delta_theta_norm ** 2)
                delta_theta_ratio = delta_theta_pred_norm / delta_theta_norm
                delta_theta_cos = _cos(delta_pred, delta)
            else:
                A_quad = 0.0
            if prev_lb is not None:
                dL_actual = lb - prev_lb

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
    )
    net.train()
    return theta_now_flat, lb


def _save_snapshot(shadow, out_dir, tag):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    torch.save({
        "t": shadow.t,
        "m_ft":  shadow.m_ft.clone(),
        "m_pre": shadow.m_pre.clone(),
        "b1":    shadow.b1,
    }, Path(out_dir) / f"shadow_{tag}.pt")


def load_shadow(path) -> ShadowMoments:
    blob = torch.load(path, map_location="cpu")
    s = ShadowMoments(blob["m_ft"].numel(), beta1=blob["b1"])
    s.t = int(blob["t"])
    s.m_ft  = blob["m_ft"]
    s.m_pre = blob["m_pre"]
    return s


# ── one training phase ───────────────────────────────────────────────────

def adam_taylor_phase(
    data, start_ckpt_or_net, param_keys,
    *, c, steps, lr, lr_end=None, batch_size=512, eval_every=1,
    grad_clip=0.0, seed=42,
    adam_beta1=0.9, adam_beta2=0.9, adam_eps=1e-8,
    weight_decay=0.0, desc="phase", snapshot_dir=None, snapshot_tag="",
    taylor=True, hvp_n_batches=4, hvp_batch_size=256, hvp_epsilon=1e-3,
    hvp_on="eval",      # "eval" (use data["eval_burst"]) or "train" (use target_pool)
    progress=True,
):
    lr_end = lr if lr_end is None else lr_end
    target_pool = data["target_pool"]; bg_pool = data["bg_pool"]

    # Which samples does H_ft operate on? For matching the `loss_burst` we
    # compare against, "eval" is the right choice; "train" matches the old code.
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

    optimizer = make_optimizer(net, lr=lr, weight_decay=weight_decay,
                               beta1=adam_beta1, beta2=adam_beta2)
    reset_optimizer(optimizer)
    sd_init = state_dict_cpu(net)

    n_params = sum(_raw(net).state_dict()[k].numel() for k in param_keys)
    shadow = ShadowMoments(n_params, beta1=adam_beta1, device="cpu")

    log = {k: [] for k in LOG_KEYS}
    prev_theta, prev_lb = _measure(
        net, optimizer, shadow, c, lr, -1, data, param_keys, sd_init, log,
        batch_size=batch_size, taylor=taylor,
        prev_theta_flat=None, prev_lb=None,
        beta2=adam_beta2, eps=adam_eps,
        hvp_n_batches=hvp_n_batches, hvp_batch_size=hvp_batch_size,
        hvp_epsilon=hvp_epsilon, hvp_data_np=hvp_data_np,
    )

    it = range(steps)
    pbar = tqdm(it, desc=desc) if progress else it
    for s in pbar:
        n_ft = int(round(c * batch_size))
        n_pre = batch_size - n_ft

        # Sample the ft and pre sub-batches that will make up the mixed batch.
        # Sampling them separately (instead of via _sample_batch) lets us compute
        # the *exact* sub-batch gradients to feed the shadow EMAs, so that
        # c·m_ft + (1-c)·m_pre equals AdamW's m_t step-exactly.
        ft_part  = _sample_pure(target_pool, n_ft)  if n_ft  > 0 else None
        pre_part = _sample_pure(bg_pool,     n_pre) if n_pre > 0 else None
        if ft_part is None:
            mixed = pre_part
        elif pre_part is None:
            mixed = ft_part
        else:
            mixed = np.concatenate([ft_part, pre_part])[np.random.permutation(batch_size)]

        # Step-exact shadow gradients when the sub-batch exists; fallback to a
        # fresh pure batch when c=0 or c=1 (doesn't affect the decomposition,
        # only used for *tracking* the inert first-moment EMA).
        if n_ft > 0:
            g_ft = _grad_on_batch(net, ft_part)
        else:
            g_ft = _grad_on_batch(net, _sample_pure(target_pool, batch_size))
        if n_pre > 0:
            g_pre = _grad_on_batch(net, pre_part)
        else:
            g_pre = _grad_on_batch(net, _sample_pure(bg_pool, batch_size))
        shadow.update(g_ft, g_pre)

        cur_lr = cosine_lr(s + 1, steps, lr, lr_end)
        train_step(net, optimizer, mixed, lr=cur_lr, grad_clip=grad_clip)

        if s % eval_every == 0 or s == steps - 1:
            prev_theta, prev_lb = _measure(
                net, optimizer, shadow, c, cur_lr, s, data, param_keys, sd_init, log,
                batch_size=batch_size, taylor=taylor,
                prev_theta_flat=prev_theta, prev_lb=prev_lb,
                beta2=adam_beta2, eps=adam_eps,
                hvp_n_batches=hvp_n_batches, hvp_batch_size=hvp_batch_size,
                hvp_epsilon=hvp_epsilon, hvp_data_np=hvp_data_np,
            )
            if progress:
                pbar.set_postfix(lb=f"{log['loss_burst'][-1]:.3f}",
                                 ab=f"{log['acc_burst'][-1]:.3f}")

    if snapshot_dir is not None:
        _save_snapshot(shadow, snapshot_dir, snapshot_tag)

    return {
        "log": {k: np.array(v) for k, v in log.items()},
        "shadow": shadow,
        "final_sd": state_dict_cpu(net),
        "net": net,
        "c": c,
    }


# ── worker entry: one full (finetune → forget) run for one c ─────────────

def run_one_c_taylor(
    c: float,
    pretrain_ckpt: str,
    data_kwargs: dict,
    out_root: str,
    *,
    ft_steps: int, ft_lr: float, ft_eval_every: int,
    fg_steps: int, fg_lr: float, fg_eval_every: int,
    batch_size: int, seed: int,
    taylor: bool = True, grad_clip: float = 0.0,
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
    ft = adam_taylor_phase(
        data, pretrain_ckpt, param_keys,
        c=c, steps=ft_steps, lr=ft_lr, lr_end=ft_lr * 0.5,
        batch_size=batch_size, eval_every=ft_eval_every, seed=seed,
        desc=f"finetune c={c}", taylor=taylor, grad_clip=grad_clip,
        hvp_on=hvp_on,
        snapshot_dir=snapshot_dir, snapshot_tag=f"ft_c{c}",
        progress=progress,
    )
    fg = adam_taylor_phase(
        data, ft["net"], param_keys,
        c=0.0, steps=fg_steps, lr=fg_lr, lr_end=fg_lr * 0.5,
        batch_size=batch_size, eval_every=fg_eval_every, seed=seed + 1,
        desc=f"forget from c={c}", taylor=taylor, grad_clip=grad_clip,
        hvp_on=hvp_on,
        snapshot_dir=snapshot_dir, snapshot_tag=f"fg_from_c{c}",
        progress=progress,
    )

    out_path = Path(out_root) / "results" / f"run_c{c}.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "c": c,
        "finetune_log": ft["log"],
        "forget_log": fg["log"],
        "ft_snapshot": str(Path(snapshot_dir) / f"shadow_ft_c{c}.pt"),
        "fg_snapshot": str(Path(snapshot_dir) / f"shadow_fg_from_c{c}.pt"),
    }
    with open(out_path, "wb") as fh:
        pickle.dump(blob, fh)
    return str(out_path)
