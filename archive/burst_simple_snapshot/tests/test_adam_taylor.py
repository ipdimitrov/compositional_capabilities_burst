"""Tests for simple.adam_taylor — algebraic correctness checks.

Test 1 (Taylor-sum identity): Σ T_i == g_ft·Δθ + ½·Δθ^T H Δθ for the Adam step
Δθ = -η(c·u_ft + (1-c)·u_pre), with u_ft = P·m̂_ft, u_pre = P·m̂_pre.

This is pure algebra — no training, no optimizer, no HVP finite-differencing.
It validates that the formulas used in `_measure` are consistent with the
advertised 2nd-order Taylor expansion of ΔL_ft.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from simple.adam_taylor import _taylor_terms


def _taylor_truth(g_ft, u_ft, u_pre, H, c, eta):
    """Reference ΔL_ft = g_ft · Δθ + ½ Δθ^T H Δθ for Δθ = -η(c·u_ft + (1-c)·u_pre)."""
    delta = -eta * (c * u_ft + (1 - c) * u_pre)
    return (torch.dot(g_ft, delta) + 0.5 * torch.dot(delta, H @ delta)).item()


@pytest.fixture
def rng():
    g = torch.Generator()
    g.manual_seed(0)
    return g


def _random_symmetric(n, rng):
    A = torch.randn(n, n, dtype=torch.float64, generator=rng) * 0.1
    return 0.5 * (A + A.T)


@pytest.mark.parametrize("c,eta", [
    (0.0, 1e-3),
    (0.25, 1e-3),
    (0.5, 3e-4),
    (0.75, 5e-3),
    (1.0, 1e-3),
])
def test_taylor_sum_identity(c, eta, rng):
    """Σ T_i exactly equals the closed-form 2nd-order Taylor for any c, eta."""
    n = 37
    g_ft  = torch.randn(n, dtype=torch.float64, generator=rng)
    m_ft  = torch.randn(n, dtype=torch.float64, generator=rng)
    m_pre = torch.randn(n, dtype=torch.float64, generator=rng)
    P     = torch.rand (n, dtype=torch.float64, generator=rng) + 0.1   # positive diag
    u_ft, u_pre = P * m_ft, P * m_pre

    H = _random_symmetric(n, rng)
    H_uft  = H @ u_ft
    H_upre = H @ u_pre

    T = _taylor_terms(g_ft, u_ft, u_pre, H_uft, H_upre, c, eta)
    total = sum(T)
    truth = _taylor_truth(g_ft, u_ft, u_pre, H, c, eta)

    rel = abs(total - truth) / max(abs(truth), 1e-30)
    assert rel < 1e-10, (
        f"Taylor-sum identity failed for c={c}, eta={eta}: "
        f"sum(T)={total:.6e}, truth={truth:.6e}, rel_err={rel:.2e}\n"
        f"T1={T[0]:+.6e}  T2={T[1]:+.6e}  T3={T[2]:+.6e}  "
        f"T4={T[3]:+.6e}  T5={T[4]:+.6e}"
    )


def test_taylor_boundary_c0_zeroes_ft_terms(rng):
    """At c=0, the finetuning-side terms (T1, T3) must vanish identically."""
    n = 19
    g_ft  = torch.randn(n, dtype=torch.float64, generator=rng)
    m_ft  = torch.randn(n, dtype=torch.float64, generator=rng)
    m_pre = torch.randn(n, dtype=torch.float64, generator=rng)
    P     = torch.rand (n, dtype=torch.float64, generator=rng) + 0.1
    u_ft, u_pre = P * m_ft, P * m_pre
    H = _random_symmetric(n, rng)
    T1, T2, T3, T4, T5 = _taylor_terms(g_ft, u_ft, u_pre, H @ u_ft, H @ u_pre, c=0.0, eta=1e-3)
    assert T1 == 0.0 and T3 == 0.0 and T4 == 0.0, \
        f"c=0 should zero T1, T3, T4; got T1={T1}, T3={T3}, T4={T4}"


def test_taylor_boundary_c1_zeroes_pre_terms(rng):
    """At c=1, the pretraining-side terms (T2, T5) must vanish identically."""
    n = 19
    g_ft  = torch.randn(n, dtype=torch.float64, generator=rng)
    m_ft  = torch.randn(n, dtype=torch.float64, generator=rng)
    m_pre = torch.randn(n, dtype=torch.float64, generator=rng)
    P     = torch.rand (n, dtype=torch.float64, generator=rng) + 0.1
    u_ft, u_pre = P * m_ft, P * m_pre
    H = _random_symmetric(n, rng)
    T1, T2, T3, T4, T5 = _taylor_terms(g_ft, u_ft, u_pre, H @ u_ft, H @ u_pre, c=1.0, eta=1e-3)
    assert T2 == 0.0 and T4 == 0.0 and T5 == 0.0, \
        f"c=1 should zero T2, T4, T5; got T2={T2}, T4={T4}, T5={T5}"


def test_taylor_zero_lr(rng):
    """η=0 must make every Taylor term exactly zero."""
    n = 29
    g_ft  = torch.randn(n, dtype=torch.float64, generator=rng)
    m_ft  = torch.randn(n, dtype=torch.float64, generator=rng)
    m_pre = torch.randn(n, dtype=torch.float64, generator=rng)
    P     = torch.rand (n, dtype=torch.float64, generator=rng) + 0.1
    u_ft, u_pre = P * m_ft, P * m_pre
    H = _random_symmetric(n, rng)
    T = _taylor_terms(g_ft, u_ft, u_pre, H @ u_ft, H @ u_pre, c=0.4, eta=0.0)
    assert all(x == 0.0 for x in T), f"η=0 should zero every T_i; got {T}"
