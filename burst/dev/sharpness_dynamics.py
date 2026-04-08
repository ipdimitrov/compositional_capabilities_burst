r"""Per-step critical sharpness dynamics during burst training.

Implements the Kalra et al. (2026) measure: at each training step, extract
the Adam update direction (preconditioned gradient), then find the
critical learning rate eta_c via forward-only line search.  lambda_c = 2/eta_c.

This tracks how curvature along the *actual optimizer direction* evolves
step-by-step through burst and reversion phases -- revealing whether
concentrated schedules hit Edge of Stability while diluted ones don't.

Requires retraining from the pretrain checkpoint (to access optimizer state).

Usage:
    python burst/dev/sharpness_dynamics.py <run_dir> [--n-seeds 3] [--measure-every 10]

Dimension key:
    B: batch_size
    L: sequence_length (doc_len)
    V: vocab_size
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib as mpl
import numpy as np
import torch
import torch.nn.functional as F

mpl.use("Agg")
import matplotlib.pyplot as plt

from burst.config import (
    CLASS_BURST,
    CLASS_OTHER,
    PHASE_BURST,
    PHASE_REVERSION,
    SCHED_COLORS,
    SCHEDULE_ORDER,
)
from burst.core.train.worker import n_target_for_step, sample_batch
from burst.core.train_utils import (
    DEVICE,
    cross_entropy_logits_BTV_targets_BT,
    make_net_bare,
    make_optim_cfg,
    make_scaler,
    resolve_run_paths,
)
from net.runner import configure_optimizers, reset_optimizer_state, update_phase_lr
from synthetic.init import set_seed

if TYPE_CHECKING:
    from net.nanogpt import nanoGPT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core: extract Adam update direction from live optimizer
# ---------------------------------------------------------------------------


def extract_adam_delta_named(
    net: torch.nn.Module, optimizer: torch.optim.AdamW, eps: float = 1e-8
) -> dict[str, torch.Tensor]:
    """Extract Adam update direction m_hat / (sqrt(v_hat) + eps) keyed by param name."""
    param_id_to_name = {id(p): n for n, p in net.named_parameters()}
    delta: dict[str, torch.Tensor] = {}

    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p)
            if state is None or "exp_avg" not in state:
                continue
            m = state["exp_avg"]
            v = state["exp_avg_sq"]
            step_t = state.get("step", 0)
            if isinstance(step_t, torch.Tensor):
                step_t = step_t.item()
            if step_t == 0:
                continue

            beta1, beta2 = group["betas"]
            bc1 = 1.0 - beta1**step_t
            bc2 = 1.0 - beta2**step_t
            m_hat = m / bc1
            v_hat = v / bc2

            name = param_id_to_name.get(id(p))
            if name:
                delta[name] = (m_hat / (v_hat.sqrt() + eps)).detach()

    return delta


# ---------------------------------------------------------------------------
# Forward-only line search (same algorithm as unified_analysis.py / paper)
# ---------------------------------------------------------------------------

_ETA_FLOOR = 1e-12


def eval_loss_at_eta(  # noqa: PLR0913
    net: nanoGPT,
    sd_base: dict[str, torch.Tensor],
    delta: dict[str, torch.Tensor],
    eta: float,
    inp_BT: torch.Tensor,
    tgt_BT: torch.Tensor,
    vocab_size: int,
) -> float:
    """Evaluate L(theta - eta * delta_theta) with a single forward pass."""
    perturbed = {k: sd_base[k] - eta * delta[k] if k in delta else sd_base[k] for k in sd_base}
    net.load_state_dict(perturbed)
    net.eval()
    with torch.no_grad():
        return F.cross_entropy(
            net(inp_BT).float().reshape(-1, vocab_size), tgt_BT.reshape(-1)
        ).item()


def critical_lr_line_search(  # noqa: PLR0913
    net: nanoGPT,
    sd_base: dict[str, torch.Tensor],
    delta: dict[str, torch.Tensor],
    loss_base: float,
    inp_BT: torch.Tensor,
    tgt_BT: torch.Tensor,
    vocab_size: int,
    eta0: float = 1.0,
    binary_tol: float = 1 / 16,
    max_exp_iters: int = 40,
) -> float:
    """Two-phase line search for critical learning rate eta_c along direction delta_theta.

    Phase 1 (exponential): bracket [eta_lower, eta_upper] containing eta_c.
    Phase 2 (binary): refine until |1 - eta_lower/eta_upper| < binary_tol.
    Returns eta_c = (eta_lower + eta_upper) / 2.
    """
    eta = eta0
    loss_eta = eval_loss_at_eta(net, sd_base, delta, eta, inp_BT, tgt_BT, vocab_size)
    direction = +1 if loss_eta < loss_base else -1

    eta_lower, eta_upper = 0.0, 0.0
    for _ in range(max_exp_iters):
        eta_prev = eta
        eta = eta * (2.0 if direction == +1 else 0.5)
        if eta < _ETA_FLOOR:
            eta_lower, eta_upper = eta, eta_prev
            break
        loss_eta = eval_loss_at_eta(net, sd_base, delta, eta, inp_BT, tgt_BT, vocab_size)
        if direction == +1 and loss_eta > loss_base:
            eta_lower, eta_upper = eta_prev, eta
            break
        if direction == -1 and loss_eta < loss_base:
            eta_lower, eta_upper = eta, eta_prev
            break
    else:
        eta_lower = eta_upper = eta

    if eta_lower <= 0 or eta_upper <= 0 or eta_lower >= eta_upper:
        return eta_lower if eta_lower > 0 else eta_upper

    while abs(1.0 - eta_lower / eta_upper) > binary_tol:
        eta_mid = 0.5 * (eta_lower + eta_upper)
        loss_mid = eval_loss_at_eta(net, sd_base, delta, eta_mid, inp_BT, tgt_BT, vocab_size)
        if loss_mid > loss_base:
            eta_upper = eta_mid
        else:
            eta_lower = eta_mid

    return 0.5 * (eta_lower + eta_upper)


# ---------------------------------------------------------------------------
# Per-step measurement
# ---------------------------------------------------------------------------


@dataclass
class StepSharpness:
    """Critical sharpness measurement at a single training step."""

    global_step: int
    phase: str
    lr: float
    lambda_c_burst: float
    lambda_c_other: float
    eos_threshold: float
    train_loss: float


@dataclass
class SharpnessTrace:
    """Per-step sharpness measurements for one (schedule, seed) run."""

    schedule: str
    seed: int
    steps: list[StepSharpness] = field(default_factory=list)


def measure_sharpness_at_step(  # noqa: PLR0913
    net: nanoGPT,
    optimizer: torch.optim.AdamW,
    burst_inp_BT: torch.Tensor,
    burst_tgt_BT: torch.Tensor,
    other_inp_BT: torch.Tensor,
    other_tgt_BT: torch.Tensor,
    vocab_size: int,
    eta0_burst: float,
    eta0_other: float,
) -> tuple[float, float, float, float]:
    """Measure critical sharpness along the Adam update direction.

    Returns (lambda_c_burst, lambda_c_other, new_eta0_burst, new_eta0_other).
    """
    raw = getattr(net, "_orig_mod", net)
    sd_base = {k: v.clone() for k, v in raw.state_dict().items()}
    delta = extract_adam_delta_named(raw, optimizer)

    if not delta:
        return float("nan"), float("nan"), eta0_burst, eta0_other

    net.eval()
    with torch.no_grad():
        loss_burst = F.cross_entropy(
            net(burst_inp_BT).float().reshape(-1, vocab_size), burst_tgt_BT.reshape(-1)
        ).item()
        loss_other = F.cross_entropy(
            net(other_inp_BT).float().reshape(-1, vocab_size), other_tgt_BT.reshape(-1)
        ).item()
    net.train()

    eta_c_burst = critical_lr_line_search(
        raw, sd_base, delta, loss_burst, burst_inp_BT, burst_tgt_BT, vocab_size, eta0=eta0_burst
    )
    eta_c_other = critical_lr_line_search(
        raw, sd_base, delta, loss_other, other_inp_BT, other_tgt_BT, vocab_size, eta0=eta0_other
    )

    raw.load_state_dict(sd_base)
    net.train()

    lc_burst = 2.0 / eta_c_burst if eta_c_burst > 0 else float("nan")
    lc_other = 2.0 / eta_c_other if eta_c_other > 0 else float("nan")
    return lc_burst, lc_other, eta_c_burst, eta_c_other


# ---------------------------------------------------------------------------
# Full retrain + measure loop
# ---------------------------------------------------------------------------


def compute_sharpness_dynamics(  # noqa: C901, PLR0913, PLR0915
    job: dict,
    target_pool: dict,
    bg_pool: dict,
    burst_eval_BL: np.ndarray,
    other_eval_BL: np.ndarray,
    measure_every: int = 10,
) -> SharpnessTrace:
    """Retrain from pretrain checkpoint and measure per-step critical sharpness."""
    seed, cfg, schedule = job["seed"], job["cfg"], job["schedule"]
    set_seed(seed)

    net = make_net_bare(cfg)
    pretrain_ckpt = job.get("pretrain_ckpt")
    if pretrain_ckpt and Path(pretrain_ckpt).exists():
        net.load_state_dict(torch.load(pretrain_ckpt, map_location=DEVICE, weights_only=True))
    if DEVICE == "cuda":
        net = torch.compile(net)

    optimizer = configure_optimizers(net, make_optim_cfg(cfg))
    scaler = make_scaler()

    P = cfg["pre_burst_steps"]
    T, U = cfg["total_steps"], cfg["reversion_steps"]
    bs, p = cfg["batch_size"], cfg["p_target"]
    V = cfg["vocab_size"]
    lr_max = cfg["lr"]
    warmup_steps = cfg["warmup_iters"]
    lr_pe = cfg["lr_pretrain_end_frac"]
    lr_be = cfg["lr_burst_end_frac"]
    lr_re = cfg["lr_reversion_end_frac"]
    weight_decay = cfg["weight_decay"]
    beta1 = cfg["beta1"]

    burst_t = torch.as_tensor(burst_eval_BL, dtype=torch.long, device=DEVICE)
    other_t = torch.as_tensor(other_eval_BL, dtype=torch.long, device=DEVICE)
    burst_inp, burst_tgt = burst_t[:, :-1], burst_t[:, 1:]
    other_inp, other_tgt = other_t[:, :-1], other_t[:, 1:]

    trace = SharpnessTrace(schedule=schedule, seed=seed)
    eta0_burst, eta0_other = 1.0, 1.0

    max_micro_bs = 512

    def train_step(batch_np: np.ndarray, global_step: int) -> float:
        tokens_BL = torch.as_tensor(batch_np, dtype=torch.long, device=DEVICE)
        inputs_BT, targets_BT = tokens_BL[:, :-1], tokens_BL[:, 1:]
        update_phase_lr(global_step, optimizer, warmup_steps, P, T, U, lr_max, lr_pe, lr_be, lr_re)
        optimizer.zero_grad(set_to_none=True)
        n = inputs_BT.size(0)
        n_accum = (n + max_micro_bs - 1) // max_micro_bs
        total_loss = 0.0
        for i in range(n_accum):
            lo, hi = i * max_micro_bs, min((i + 1) * max_micro_bs, n)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                logits_BTV = net(inputs_BT[lo:hi])
                loss = cross_entropy_logits_BTV_targets_BT(logits_BTV, targets_BT[lo:hi]) / n_accum
            scaler.scale(loss).backward()
            total_loss += loss.item()
        if cfg["grad_clip"] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        return total_loss

    def eos_threshold(lr: float) -> float:
        """AdamW EoS threshold: (2/lr - wd) * (1+b1)/(1-b1)."""
        return (2.0 / lr - weight_decay) * (1.0 + beta1) / (1.0 - beta1)

    def maybe_measure(gs: int, phase: str, loss_val: float) -> None:
        nonlocal eta0_burst, eta0_other
        lr = optimizer.param_groups[0]["lr"]
        lc_b, lc_o, eta0_burst, eta0_other = measure_sharpness_at_step(
            net,
            optimizer,
            burst_inp,
            burst_tgt,
            other_inp,
            other_tgt,
            V,
            eta0_burst,
            eta0_other,
        )
        trace.steps.append(
            StepSharpness(
                global_step=gs,
                phase=phase,
                lr=lr,
                lambda_c_burst=lc_b,
                lambda_c_other=lc_o,
                eos_threshold=eos_threshold(lr),
                train_loss=loss_val,
            )
        )

    net.train()

    for s in range(T):
        nt = n_target_for_step(s, T, schedule, p, bs)
        batch_np, _ = sample_batch(target_pool, bg_pool, nt, bs)
        gs = P + s + 1
        loss_val = train_step(batch_np, gs)
        if s % measure_every == 0 or s == T - 1:
            maybe_measure(gs, PHASE_BURST, loss_val)

    reset_optimizer_state(optimizer)

    for s in range(U):
        batch_np, _ = sample_batch(target_pool, bg_pool, 0, bs)
        gs = P + T + s + 1
        loss_val = train_step(batch_np, gs)
        if s % measure_every == 0 or s == U - 1:
            maybe_measure(gs, PHASE_REVERSION, loss_val)

    return trace


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_sharpness_dynamics(
    traces: dict[str, list[SharpnessTrace]],
    out_dir: Path,
) -> None:
    """Plot per-step critical sharpness for each schedule, with EoS threshold."""
    out_dir.mkdir(parents=True, exist_ok=True)
    schedules = [s for s in SCHEDULE_ORDER if s in traces]

    _fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for sched in schedules:
        color = SCHED_COLORS.get(sched, "#888888")
        all_steps: dict[int, dict] = {}
        for tr in traces[sched]:
            for sp in tr.steps:
                entry = all_steps.setdefault(
                    sp.global_step, {"burst": [], "other": [], "eos": sp.eos_threshold}
                )
                entry["burst"].append(sp.lambda_c_burst)
                entry["other"].append(sp.lambda_c_other)

        xs = sorted(all_steps.keys())
        burst_mean = [np.nanmean(all_steps[x]["burst"]) for x in xs]
        other_mean = [np.nanmean(all_steps[x]["other"]) for x in xs]

        axes[0].plot(xs, burst_mean, color=color, label=sched, alpha=0.8)
        axes[1].plot(xs, other_mean, color=color, label=sched, alpha=0.8)

        if len(traces[sched]) > 1:
            burst_lo = [np.nanpercentile(all_steps[x]["burst"], 25) for x in xs]
            burst_hi = [np.nanpercentile(all_steps[x]["burst"], 75) for x in xs]
            other_lo = [np.nanpercentile(all_steps[x]["other"], 25) for x in xs]
            other_hi = [np.nanpercentile(all_steps[x]["other"], 75) for x in xs]
            axes[0].fill_between(xs, burst_lo, burst_hi, color=color, alpha=0.15)
            axes[1].fill_between(xs, other_lo, other_hi, color=color, alpha=0.15)

    eos_map: dict[int, float] = {}
    first_traces = next(iter(traces.values()), [])
    for tr in first_traces:
        for sp in tr.steps:
            eos_map[sp.global_step] = sp.eos_threshold
    if eos_map:
        eos_xs = sorted(eos_map.keys())
        eos_ys = [eos_map[x] for x in eos_xs]
        axes[0].plot(eos_xs, eos_ys, "k--", alpha=0.5, label="EoS threshold")
        axes[1].plot(eos_xs, eos_ys, "k--", alpha=0.5, label="EoS threshold")

    axes[0].set_ylabel("lambda_c (burst-class loss)")
    axes[0].set_title("Critical sharpness dynamics -- burst-class")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("lambda_c (other-class loss)")
    axes[1].set_xlabel("Global step")
    axes[1].set_title("Critical sharpness dynamics -- other-class")
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.set_yscale("log")
        ax.grid(visible=True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "sharpness_dynamics.png", dpi=150)
    plt.close()

    plot_sharpness_vs_eos(traces, out_dir)


def plot_sharpness_vs_eos(
    traces: dict[str, list[SharpnessTrace]],
    out_dir: Path,
) -> None:
    """Plot lambda_c / EoS_threshold ratio -- values near 1.0 mean Edge of Stability."""
    _fig, ax = plt.subplots(figsize=(14, 5))
    schedules = [s for s in SCHEDULE_ORDER if s in traces]

    for sched in schedules:
        color = SCHED_COLORS.get(sched, "#888888")
        all_steps: dict[int, list[float]] = {}
        for tr in traces[sched]:
            for sp in tr.steps:
                ratio = (
                    sp.lambda_c_burst / sp.eos_threshold if sp.eos_threshold > 0 else float("nan")
                )
                all_steps.setdefault(sp.global_step, []).append(ratio)

        xs = sorted(all_steps.keys())
        means = [np.nanmean(all_steps[x]) for x in xs]
        ax.plot(xs, means, color=color, label=sched, alpha=0.8)

    ax.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="EoS (lambda_c = threshold)")
    ax.set_ylabel("lambda_c / EoS threshold")
    ax.set_xlabel("Global step")
    ax.set_title("Proximity to Edge of Stability")
    ax.legend(fontsize=8)
    ax.grid(visible=True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "sharpness_vs_eos.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: D103
    parser = argparse.ArgumentParser(description="Per-step critical sharpness dynamics")
    parser.add_argument("run_dirs", nargs="+", help="Run directories")
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--measure-every", type=int, default=10)
    parser.add_argument("--out-dir", type=str, default="data/sharpness_dynamics")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_traces: dict[str, list[SharpnessTrace]] = {}

    for run_dir in args.run_dirs:
        cfg_path, logs_dir, _ = resolve_run_paths(run_dir)
        with cfg_path.open() as f:
            json.load(f)

        pkl_path = logs_dir / "all_results.pkl"
        if not pkl_path.exists():
            pkl_path = Path(run_dir) / "all_results.pkl"
        with pkl_path.open("rb") as f:
            all_results = pickle.load(f)  # noqa: S301

        data_path = logs_dir / "_data.pkl"
        if not data_path.exists():
            data_path = Path(run_dir) / "_data.pkl"
        with data_path.open("rb") as f:
            target_pool, bg_pool, eval_docs, _prompt_len, _ = pickle.load(f)  # noqa: S301

        burst_eval = eval_docs.get(CLASS_BURST, next(iter(eval_docs.values())))
        other_eval = eval_docs.get(CLASS_OTHER, next(iter(eval_docs.values())))
        n_eval = min(64, burst_eval.shape[0], other_eval.shape[0])
        burst_sub = burst_eval[:n_eval]
        other_sub = other_eval[:n_eval]

        seen: dict[str, int] = {}
        for r in all_results:
            sched = r["schedule"]
            seed = r["seed"]
            seen.setdefault(sched, 0)
            if seen[sched] >= args.n_seeds:
                continue
            seen[sched] += 1

            pretrain_ckpt_path = logs_dir / "pretrain_ckpt.pt"
            if not pretrain_ckpt_path.exists():
                pretrain_ckpt_path = Path(run_dir) / "logs" / "pretrain_ckpt.pt"

            job = {
                "seed": seed,
                "cfg": r["config"],
                "schedule": sched,
                "pretrain_ckpt": str(pretrain_ckpt_path) if pretrain_ckpt_path.exists() else None,
            }

            t0 = time.time()
            trace = compute_sharpness_dynamics(
                job,
                target_pool,
                bg_pool,
                burst_sub,
                other_sub,
                measure_every=args.measure_every,
            )
            elapsed = time.time() - t0
            logger.info(
                "%s seed=%d: %d measurements in %.1fs",
                sched,
                seed,
                len(trace.steps),
                elapsed,
            )

            all_traces.setdefault(sched, []).append(trace)

    with (out_dir / "traces.pkl").open("wb") as f:
        pickle.dump(all_traces, f)

    plot_sharpness_dynamics(all_traces, out_dir)
    logger.info("Results saved to %s", out_dir)


if __name__ == "__main__":
    main()
