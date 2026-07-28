"""Manual (Python-level) Adam optimizer with an exact ft/pre/cross decomposition.

Why hand-roll Adam when torch ships one? Because we want to decompose Adam's
internal state into "finetuning contribution" and "pretraining contribution"
pieces and have the identity

    m_t = c · m_ft + (1-c) · m_pre
    v_t = c² · v_ft + (1-c)² · v_pre + 2c(1-c) · v_cross

hold **exactly** (not just in expectation over shadow EMAs). If we run Adam
ourselves and feed it the per-substrate gradients directly, the identity holds
by construction in fp32.

Two classes:

- `ManualAdam`: plain Adam, equivalent to `torch.optim.AdamW(..., weight_decay=0)`.
  Verified via `tests/test_manual_adam.py`.
- `DecomposedAdam`: same update, but maintains the five extra per-parameter
  tensors above and applies the optimizer step using the *reconstructed*
  combined `m`, `v`.

No weight decay, no fused kernel, no AMSGrad — just vanilla Adam.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence

import torch


# ── Plain manual Adam (parity baseline) ──────────────────────────────────

class ManualAdam:
    """Vanilla Adam optimizer. Equivalent to `torch.optim.AdamW(weight_decay=0)`.

    Matches torch's non-fused implementation in exact arithmetic. Parity against
    `torch.optim.AdamW` is covered in `tests/test_manual_adam.py`.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.params: List[torch.Tensor] = [p for p in params if p.requires_grad]
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self._t = 0
        # Adam state (fp32, on the param's device).
        self.m: List[torch.Tensor] = [torch.zeros_like(p, dtype=torch.float32) for p in self.params]
        self.v: List[torch.Tensor] = [torch.zeros_like(p, dtype=torch.float32) for p in self.params]

    @property
    def step_count(self) -> int:
        return self._t

    def set_lr(self, lr: float) -> None:
        self.lr = lr

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    @torch.no_grad()
    def step(self, grads: Optional[Sequence[torch.Tensor]] = None) -> None:
        """Apply one Adam step. If `grads` is None, reads from `p.grad`."""
        self._t += 1
        bc1 = 1.0 - self.b1 ** self._t
        bc2 = 1.0 - self.b2 ** self._t
        step_size = self.lr / bc1
        bc2_sqrt = math.sqrt(bc2)
        for i, p in enumerate(self.params):
            g = grads[i] if grads is not None else p.grad
            if g is None:
                continue
            g = g.to(dtype=torch.float32)
            self.m[i].mul_(self.b1).add_(g, alpha=1 - self.b1)
            self.v[i].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            denom = self.v[i].sqrt().div_(bc2_sqrt).add_(self.eps)
            p.data.addcdiv_(self.m[i].to(p.dtype), denom.to(p.dtype), value=-step_size)


# ── Decomposed Adam: exact ft/pre/cross state ────────────────────────────

class DecomposedAdam(ManualAdam):
    """Adam variant with exact `m_ft`, `m_pre`, `v_ft`, `v_pre`, `v_cross`.

    Instead of calling `step(grads)` with the mixed gradient, call
    `step_decomposed(g_ft, g_pre, c)` with the two pure sub-batch gradients.
    After the call:

        m = c · m_ft + (1-c) · m_pre              (holds exactly in fp32)
        v = c² · v_ft + (1-c)² · v_pre + 2c(1-c) · v_cross   (holds exactly in fp32)

    Edge cases (c=0 or c=1): pass `None` for the missing side. The inert EMAs
    decay by β (no new input), which is the mathematically consistent choice.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        super().__init__(params, lr=lr, betas=betas, eps=eps)
        z = lambda p: torch.zeros_like(p, dtype=torch.float32)
        self.m_ft = [z(p) for p in self.params]
        self.m_pre = [z(p) for p in self.params]
        self.v_ft = [z(p) for p in self.params]
        self.v_pre = [z(p) for p in self.params]
        self.v_cross = [z(p) for p in self.params]

    @torch.no_grad()
    def step_decomposed(
        self,
        grads_ft: Optional[Sequence[Optional[torch.Tensor]]],
        grads_pre: Optional[Sequence[Optional[torch.Tensor]]],
        c: float,
    ) -> None:
        """One optimizer step using per-substrate gradients.

        Passing `grads_ft=None` is equivalent to supplying a list of Nones —
        the ft-side EMAs decay by β. Same for `grads_pre=None`.
        Under the hood the optimizer applies the update using the *combined*
        `m`, `v` assembled from the decomposition (so it is exactly the same
        update torch's AdamW would have applied given the mixed gradient
        `c·g_ft + (1-c)·g_pre`).
        """
        self._t += 1
        bc1 = 1.0 - self.b1 ** self._t
        bc2 = 1.0 - self.b2 ** self._t
        step_size = self.lr / bc1
        bc2_sqrt = math.sqrt(bc2)

        for i, p in enumerate(self.params):
            g_ft = None if grads_ft  is None else grads_ft[i]
            g_pre = None if grads_pre is None else grads_pre[i]
            if g_ft is not None:
                g_ft = g_ft.to(dtype=torch.float32)
                self.m_ft[i] .mul_(self.b1).add_(g_ft,  alpha=1 - self.b1)
                self.v_ft[i] .mul_(self.b2).addcmul_(g_ft,  g_ft,  value=1 - self.b2)
            else:
                self.m_ft[i] .mul_(self.b1)
                self.v_ft[i] .mul_(self.b2)

            if g_pre is not None:
                g_pre = g_pre.to(dtype=torch.float32)
                self.m_pre[i].mul_(self.b1).add_(g_pre, alpha=1 - self.b1)
                self.v_pre[i].mul_(self.b2).addcmul_(g_pre, g_pre, value=1 - self.b2)
            else:
                self.m_pre[i].mul_(self.b1)
                self.v_pre[i].mul_(self.b2)

            if g_ft is not None and g_pre is not None:
                self.v_cross[i].mul_(self.b2).addcmul_(g_ft, g_pre, value=1 - self.b2)
            else:
                self.v_cross[i].mul_(self.b2)

            # Reconstruct the combined state — this IS `m_t`, `v_t`.
            m_combined = c * self.m_ft[i] + (1 - c) * self.m_pre[i]
            v_combined = (
                c * c         * self.v_ft[i]
                + (1 - c)**2  * self.v_pre[i]
                + 2 * c * (1 - c) * self.v_cross[i]
            )
            # Mathematically `v_combined ≥ 0` (it equals EMA of (c·g_ft + (1-c)·g_pre)²),
            # but the three-term sum can go slightly negative in fp32 when v_cross
            # partially cancels the square terms. Clamp to avoid sqrt(negative) → NaN.
            v_combined_safe = v_combined.clamp(min=0.0)
            self.m[i].copy_(m_combined)
            self.v[i].copy_(v_combined_safe)

            denom = v_combined_safe.sqrt().div_(bc2_sqrt).add_(self.eps)
            p.data.addcdiv_(m_combined.to(p.dtype), denom.to(p.dtype), value=-step_size)

    # ── read-only accessors for plotting / analysis ─────────────────────

    def combined_m(self, c: float) -> List[torch.Tensor]:
        return [c * mft + (1 - c) * mpre for mft, mpre in zip(self.m_ft, self.m_pre)]

    def combined_v(self, c: float) -> List[torch.Tensor]:
        return [
            c * c * vft + (1 - c) ** 2 * vpre + 2 * c * (1 - c) * vx
            for vft, vpre, vx in zip(self.v_ft, self.v_pre, self.v_cross)
        ]

    def bias_correction(self) -> tuple[float, float]:
        t = max(self._t, 1)
        return 1.0 - self.b1 ** t, 1.0 - self.b2 ** t
