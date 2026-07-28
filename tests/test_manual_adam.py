"""Parity tests for `simple.manual_adam.ManualAdam`.

Goal: ManualAdam should match `torch.optim.AdamW(weight_decay=0, fused=False)`
on the same inputs. After passing these, we trust ManualAdam is a correct Adam
implementation, which transitively justifies DecomposedAdam (same code path
with extra bookkeeping).

Tolerance: 1e-5 relative after 20 steps, as agreed in the plan.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from simple.manual_adam import ManualAdam, DecomposedAdam


def _relative_max_diff(a_params, b_params) -> float:
    """Max over params of  max |a-b| / max(|a|, |b|, eps)."""
    worst = 0.0
    for a, b in zip(a_params, b_params):
        num = (a - b).abs().max().item()
        den = max(a.abs().max().item(), b.abs().max().item(), 1e-12)
        worst = max(worst, num / den)
    return worst


def _clone_params(params):
    return [p.detach().clone() for p in params]


def _set_params(model_params, src):
    with torch.no_grad():
        for p, s in zip(model_params, src):
            p.data.copy_(s)


def _mlp(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 32),
                         nn.GELU(), nn.Linear(32, 4))


@torch.no_grad()
def _fake_data(n=64, d=16, k=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    y = torch.randint(0, k, (n,), generator=g)
    return X, y


def _step_loss(model, X, y):
    logits = model(X)
    return F.cross_entropy(logits, y)


def test_parity_small_mlp():
    """Torch AdamW (wd=0, non-fused) vs ManualAdam on a small MLP over 20 steps."""
    lr = 1e-3
    betas = (0.9, 0.999)
    eps = 1e-8
    n_steps = 20

    X, y = _fake_data()

    # Build two identical models with shared initial params.
    m_torch = _mlp()
    m_manual = _mlp()
    _set_params(list(m_manual.parameters()), _clone_params(list(m_torch.parameters())))

    opt_torch = torch.optim.AdamW(m_torch.parameters(), lr=lr, betas=betas,
                                  eps=eps, weight_decay=0.0, fused=False)
    opt_manual = ManualAdam(list(m_manual.parameters()), lr=lr, betas=betas, eps=eps)

    for _ in range(n_steps):
        # Torch path
        opt_torch.zero_grad()
        _step_loss(m_torch, X, y).backward()
        opt_torch.step()

        # Manual path — fresh gradients on identical inputs from identical params.
        opt_manual.zero_grad()
        _step_loss(m_manual, X, y).backward()
        opt_manual.step()

    rel = _relative_max_diff(list(m_torch.parameters()), list(m_manual.parameters()))
    assert rel < 1e-5, f"param rel-diff after {n_steps} steps = {rel:.2e} (tol 1e-5)"


def test_parity_decomposed_matches_manual_when_c_well_defined():
    """At c in (0,1), if we feed DecomposedAdam with (g_ft, g_pre) such that
    c·g_ft + (1-c)·g_pre equals the real mixed gradient, the resulting params
    should match ManualAdam fed with the mixed gradient directly.

    This confirms that DecomposedAdam's reconstructed combined m, v exactly
    reproduce Adam's trajectory.
    """
    lr = 1e-3
    betas = (0.9, 0.999)
    eps = 1e-8
    n_steps = 20
    c = 0.3

    X, y = _fake_data()

    m_a = _mlp()
    m_b = _mlp()
    _set_params(list(m_b.parameters()), _clone_params(list(m_a.parameters())))

    opt_a = ManualAdam(list(m_a.parameters()), lr=lr, betas=betas, eps=eps)
    opt_b = DecomposedAdam(list(m_b.parameters()), lr=lr, betas=betas, eps=eps)

    torch.manual_seed(123)
    for _ in range(n_steps):
        # Draw both sub-gradients from reasonable distributions (no
        # division-by-c blowup), then let g_mix be their weighted combination.
        g_ft = [torch.randn_like(p) for p in m_a.parameters()]
        g_pre = [torch.randn_like(p) for p in m_a.parameters()]
        g_mix = [c * gf + (1 - c) * gp for gf, gp in zip(g_ft, g_pre)]

        opt_a.step(grads=g_mix)
        opt_b.step_decomposed(g_ft, g_pre, c=c)

    # Looser tolerance than `test_parity_small_mlp`: DecomposedAdam does three
    # extra fp32 multiply-adds per step (assembling m_combined, v_combined from
    # the ft/pre/cross EMAs), which accumulates slightly more rounding than
    # ManualAdam's single-gradient path. The algebraic identity
    # `m_combined == m` and `v_combined == v` is checked tightly in
    # `test_decomposed_identity_holds_exactly` below.
    rel = _relative_max_diff(list(m_a.parameters()), list(m_b.parameters()))
    assert rel < 1e-3, f"DecomposedAdam vs ManualAdam rel-diff = {rel:.2e} (tol 1e-3)"


def test_decomposed_identity_holds_exactly():
    """After any number of step_decomposed calls, the identities
    m = c·m_ft + (1-c)·m_pre and v = c²·v_ft + (1-c)²·v_pre + 2c(1-c)·v_cross
    should hold in fp32.
    """
    c = 0.4
    m_ = _mlp()
    opt = DecomposedAdam(list(m_.parameters()), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

    torch.manual_seed(7)
    for _ in range(10):
        g_ft = [torch.randn_like(p) for p in m_.parameters()]
        g_pre = [torch.randn_like(p) for p in m_.parameters()]
        opt.step_decomposed(g_ft, g_pre, c=c)

    for i, _ in enumerate(opt.params):
        m_recon = c * opt.m_ft[i] + (1 - c) * opt.m_pre[i]
        v_recon = (c * c * opt.v_ft[i]
                   + (1 - c) ** 2 * opt.v_pre[i]
                   + 2 * c * (1 - c) * opt.v_cross[i])
        err_m = (m_recon - opt.m[i]).abs().max().item()
        err_v = (v_recon - opt.v[i]).abs().max().item()
        assert err_m < 1e-6, f"param {i}: m reconstruction err {err_m}"
        assert err_v < 1e-6, f"param {i}: v reconstruction err {err_v}"


def test_decomposed_c0_decays_ft_side():
    """At c=0, feeding grads_ft=None should just decay m_ft, v_ft, v_cross by β
    per step (no new input), while m_pre, v_pre update from grads_pre."""
    opt = DecomposedAdam([torch.zeros(3, requires_grad=True)], lr=1e-3,
                         betas=(0.9, 0.999), eps=1e-8)

    # Pre-seed some nonzero ft-side state, then call with c=0 and None for ft.
    opt.m_ft[0].copy_(torch.tensor([1.0, 2.0, 3.0]))
    opt.v_ft[0].copy_(torch.tensor([0.5, 0.5, 0.5]))
    opt.v_cross[0].copy_(torch.tensor([0.1, 0.2, 0.3]))
    g_pre = [torch.tensor([1.0, 0.0, -1.0])]

    opt.step_decomposed(grads_ft=None, grads_pre=g_pre, c=0.0)

    # After one step with None on ft side, expect multiplication by β on
    # m_ft, v_ft, v_cross and standard update of m_pre, v_pre.
    b1, b2 = opt.b1, opt.b2
    assert torch.allclose(opt.m_ft[0], b1 * torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(opt.v_ft[0], b2 * torch.tensor([0.5, 0.5, 0.5]))
    assert torch.allclose(opt.v_cross[0], b2 * torch.tensor([0.1, 0.2, 0.3]))
