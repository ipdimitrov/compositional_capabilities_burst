"""Five-metric deep analysis of burstiness runs.

Implements post-hoc analysis on saved checkpoints without retraining:

1. ADL (Activation Difference Lens) — readability + causal ablation
2. Gradient interference magnitude — from existing grad_sim_log
3. EMA interpolation probe — peak <-> reverted cliff sharpness
4. Critical sharpness — Hutchinson trace of Hessian on burst loss
5. Weight delta rank — SVD of (W_post - W_pre) per layer

Outputs:
  <out_dir>/results.pkl        — all computed metrics
  <out_dir>/charts/*.png       — individual PNG charts
  <out_dir>/dashboard.html     — interactive Plotly dashboard

Usage (single or explicit list of runs):
    uv run python burst/deep_analysis.py data/burst_d3_pos1_<tag>
    uv run python burst/deep_analysis.py data/burst_d3_pos1_<tag> data/burst_d3_pos2_<tag>

Usage (all valid runs in data/, parallelised):
    uv run python burst/deep_analysis.py --all --n-parallel 4

Flags:
    --all              Scan data/ for all runs with checkpoints + all_results.pkl
    --n-parallel N     Run dirs to analyse in parallel (default: 1)
    --adl-seeds N      Seeds per schedule for ADL + EMA probe (default: 3)
    --adl-samples N    Docs per ADL forward pass (default: 256)
    --sharpness-seeds N  Seeds per schedule for sharpness (default: 3)
    --n-hutchinson N   Hutchinson samples for sharpness (default: 10)
    --out-dir PATH     Output directory (default: data/deep_analysis_combined)

Dimension key:
    B: batch_size
    L: sequence_length (doc_len)
    N: n_embd (model dimension)
    K: n_layers + 1 (embedding + each transformer block output)
    T: token positions (L - 1)
    V: vocab_size
    S: n_schedules
"""

import argparse
import json
import multiprocessing
import pickle
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from burst.config import (
    PHASE_BURST,
    PHASE_REVERSION,
    parse_run_config,
)
from burst.core.activations import collect_activations_KPTN
from burst.core.train_utils import DEVICE, load_net
from burst.dev._shared import burst_token_ids as _burst_token_ids
from burst.dev._shared import free_gen_acc as _free_gen_acc
from burst.dev.plot_utils import plotly_to_png_matplotlib as _plotly_to_png_matplotlib
from net.nanogpt import nanoGPT

_rng = np.random.default_rng()

KEY_STEPS = [0, 499, 749, 999]

# ---------------------------------------------------------------------------
# 1. ADL — activation difference lens
# ---------------------------------------------------------------------------


@torch.no_grad()
def _compute_delta_KTN(  # noqa: N802
    net_ckpt: nanoGPT,
    net_pre: nanoGPT,
    other_docs_BL: np.ndarray,
    n_samples: int,
) -> torch.Tensor:
    n = min(n_samples, other_docs_BL.shape[0])
    idx = _rng.choice(other_docs_BL.shape[0], n, replace=False)
    docs = other_docs_BL[idx]
    acts_ckpt = collect_activations_KPTN(net_ckpt, docs)
    acts_pre = collect_activations_KPTN(net_pre, docs)
    K = len(acts_ckpt)
    delta_KTN = torch.stack([(acts_ckpt[k] - acts_pre[k]).mean(dim=0) for k in range(K)])
    return delta_KTN.cpu()


@torch.no_grad()
def _logit_lens_readability(
    net: nanoGPT,
    delta_KTN: torch.Tensor,
    burst_token_ids: list[int],
    top_k: int = 10,
) -> dict:
    K, T, _N = delta_KTN.shape
    unembed_VN = net.transformer.wte.weight.detach().float().cpu()
    delta_KTN_f = delta_KTN.float()
    logits_KTV = torch.einsum("ktn,vn->ktv", delta_KTN_f, unembed_VN)
    burst_set = set(burst_token_ids)
    readability_KT = np.zeros((K, T))
    mean_rank_KT = np.full((K, T), float(logits_KTV.shape[-1]))
    for k in range(K):
        for t in range(T):
            sorted_ids = torch.argsort(logits_KTV[k, t], descending=True).tolist()
            top_ids = sorted_ids[:top_k]
            readability_KT[k, t] = sum(1 for tid in top_ids if tid in burst_set) / top_k
            ranks = [i for i, tid in enumerate(sorted_ids) if tid in burst_set]
            if ranks:
                mean_rank_KT[k, t] = float(np.mean(ranks))
    return {"readability_KT": readability_KT, "mean_rank_KT": mean_rank_KT}


@torch.no_grad()
def _free_gen_acc_ablated(
    net: nanoGPT,
    docs_BL: np.ndarray,
    prompt_len: int,
    delta_KTN: torch.Tensor,
    ablate_layer: int,
) -> float:
    net.eval()
    docs_t = torch.as_tensor(docs_BL, dtype=torch.long, device=DEVICE)
    _B, L = docs_t.shape
    target_B6 = docs_t[:, -6:]

    delta_TN = delta_KTN[ablate_layer].to(DEVICE).float()
    norms_T = delta_TN.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    delta_unit_TN = delta_TN / norms_T

    def _hook(_module, _input, output):
        if isinstance(output, tuple):
            x_raw, rest = output[0], output[1:]
        else:
            x_raw, rest = output, None
        x = x_raw.float()
        T_use = min(x.shape[1], delta_unit_TN.shape[0])
        d = delta_unit_TN[:T_use]
        proj = torch.einsum("btn,tn->bt", x[:, :T_use], d).unsqueeze(-1) * d.unsqueeze(0)
        x[:, :T_use] -= proj
        x = x.to(x_raw.dtype)
        if rest is not None:
            return (x, *rest)
        return x

    if ablate_layer == 0:
        handle = net.transformer.drop.register_forward_hook(_hook)
    else:
        handle = net.transformer.h[ablate_layer - 1].register_forward_hook(_hook)

    try:
        generated = net.generate(docs_t[:, :prompt_len], L - prompt_len)
    finally:
        handle.remove()

    return (generated[:, -6:] == target_B6).all(dim=1).float().mean().item()


def compute_adl_for_label(
    label: str,
    ckpt_dir: Path,
    cfg: dict,
    n_a: int,
    depth: int,
    other_docs_BL: np.ndarray,
    burst_docs_BL: np.ndarray,
    prompt_len: int,
    n_samples: int = 256,
    key_steps: list[int] | None = None,
) -> dict:
    """Compute ADL metrics at key checkpoints for one label."""
    if key_steps is None:
        key_steps = KEY_STEPS

    ckpt_files = {int(p.stem.split("_")[1]): p for p in ckpt_dir.glob("step_*.pt")}
    available = sorted(ckpt_files.keys())

    steps_to_run = []
    for s in key_steps:
        closest = min(available, key=lambda x: abs(x - s))
        if closest not in steps_to_run:
            steps_to_run.append(closest)
    steps_to_run.sort()

    pre_burst_step = steps_to_run[0]
    net_pre = load_net(cfg, str(ckpt_files[pre_burst_step]))
    burst_ids = _burst_token_ids(cfg, n_a, depth)

    results = []
    for step in steps_to_run:
        net_ckpt = load_net(cfg, str(ckpt_files[step]))
        delta_KTN = _compute_delta_KTN(net_ckpt, net_pre, other_docs_BL, n_samples)
        readability = _logit_lens_readability(net_ckpt, delta_KTN, burst_ids)
        delta_norm_K = delta_KTN.norm(dim=(1, 2)).tolist()

        acc_baseline = _free_gen_acc(net_ckpt, burst_docs_BL[:n_samples], prompt_len)
        K = delta_KTN.shape[0]
        acc_ablated_K = [
            _free_gen_acc_ablated(net_ckpt, burst_docs_BL[:n_samples], prompt_len, delta_KTN, k)
            for k in range(K)
        ]
        acc_drop_K = [acc_baseline - a for a in acc_ablated_K]

        phase = PHASE_BURST if step < cfg["total_steps"] else PHASE_REVERSION
        results.append(
            {
                "step": step,
                "phase": phase,
                "delta_norm_K": delta_norm_K,
                "readability_KT": readability["readability_KT"].tolist(),
                "mean_rank_KT": readability["mean_rank_KT"].tolist(),
                "mean_readability_K": readability["readability_KT"].mean(axis=1).tolist(),
                "acc_baseline": acc_baseline,
                "acc_ablated_K": acc_ablated_K,
                "acc_drop_K": acc_drop_K,
                "max_acc_drop": max(acc_drop_K),
            }
        )

    return {"label": label, "adl_steps": results}


# ---------------------------------------------------------------------------
# 2. Gradient interference — from existing grad_sim_log
# ---------------------------------------------------------------------------


def extract_grad_interference(result: dict) -> dict:
    """Extract gradient interference signal from existing grad_sim_log."""
    gsl = result.get("grad_sim_log", {})
    steps = gsl.get("step", [])
    burst_vs_other = gsl.get("burst_vs_other", [])
    phases = gsl.get("phase", [])

    burst_sims = [v for v, p in zip(burst_vs_other, phases, strict=True) if p == PHASE_BURST]
    rev_sims = [v for v, p in zip(burst_vs_other, phases, strict=True) if p == PHASE_REVERSION]

    mean_burst_interference = float(np.mean(burst_sims)) if burst_sims else float("nan")
    mean_rev_interference = float(np.mean(rev_sims)) if rev_sims else float("nan")
    end_burst_interference = float(burst_sims[-1]) if burst_sims else float("nan")

    per_layer = gsl.get("per_layer", {})
    layer_names = gsl.get("layer_names", [])
    mean_layer_interference = {}
    end_layer_interference = {}
    for ln in layer_names:
        vals = per_layer.get(ln, [])
        burst_vals = [v for v, p in zip(vals, phases, strict=True) if p == PHASE_BURST]
        mean_layer_interference[ln] = float(np.mean(burst_vals)) if burst_vals else float("nan")
        end_layer_interference[ln] = float(burst_vals[-1]) if burst_vals else float("nan")

    return {
        "steps": steps,
        "burst_vs_other": burst_vs_other,
        "phases": phases,
        "mean_burst_interference": mean_burst_interference,
        "end_burst_interference": end_burst_interference,
        "mean_rev_interference": mean_rev_interference,
        "mean_layer_interference": mean_layer_interference,
        "end_layer_interference": end_layer_interference,
    }


# ---------------------------------------------------------------------------
# 3. Task vector arithmetic
# ---------------------------------------------------------------------------


@torch.no_grad()
def ema_interpolation_probe(
    ckpt_peak: str,
    ckpt_reverted: str,
    cfg: dict,
    burst_docs_BL: np.ndarray,
    prompt_len: int,
    n_samples: int = 256,
    alphas: list[float] | None = None,
) -> dict:
    """Interpolate between reverted and peak-burst model; measure burst accuracy.

    theta_interp = (1 - alpha) * theta_reverted + alpha * theta_peak

    alpha=0 → reverted model (no burst capability)
    alpha=1 → peak model (full burst capability)

    For a shallow wrapper (burst_100): accuracy stays near 0 until a sharp
    threshold near alpha=1 — the capability is concentrated in a narrow direction.

    For deep learning (burst_10): accuracy increases gradually — the capability
    is distributed across many weight directions.

    The "sharpness" of the alpha→accuracy curve is a direct measure of
    representation depth.
    """
    if alphas is None:
        alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    net_peak = load_net(cfg, ckpt_peak)
    net_rev = load_net(cfg, ckpt_reverted)
    peak_sd = {k: v.clone().float() for k, v in net_peak.state_dict().items()}
    rev_sd = {k: v.clone().float() for k, v in net_rev.state_dict().items()}

    net_interp = load_net(cfg, ckpt_peak)

    n = min(n_samples, burst_docs_BL.shape[0])
    idx = _rng.choice(burst_docs_BL.shape[0], n, replace=False)
    docs = burst_docs_BL[idx]

    accs = []
    for alpha in alphas:
        interp_sd = {k: (1 - alpha) * rev_sd[k] + alpha * peak_sd[k] for k in peak_sd}
        net_interp.load_state_dict(interp_sd)
        acc = _free_gen_acc(net_interp, docs, prompt_len)
        accs.append(acc)

    # Compute "cliff sharpness": alpha at which accuracy first exceeds 0.5
    cliff_alpha = next((a for a, acc in zip(alphas, accs, strict=True) if acc > 0.5), 1.0)
    # Area under the curve (higher = more gradual = deeper)
    auc = float(np.trapezoid(accs, alphas))

    return {"alphas": alphas, "accs": accs, "cliff_alpha": cliff_alpha, "auc": auc}


@torch.no_grad()
def compute_task_vector_norms(
    ckpt_pre: str,
    ckpt_post: str,
    cfg: dict,
) -> dict:
    """Compute ||W_post - W_pre|| per layer group."""
    net_pre = load_net(cfg, ckpt_pre)
    net_post = load_net(cfg, ckpt_post)
    layer_groups: dict[str, list[torch.Tensor]] = {}
    for name, p_post in net_post.named_parameters():
        p_pre = dict(net_pre.named_parameters())[name]
        delta = (p_post - p_pre).float().cpu().view(-1)
        parts = name.split(".")
        if "wte" in name or "wpe" in name:
            group = "emb"
        elif len(parts) >= 4 and parts[1] == "h":
            i = parts[2]
            if "ln_" in name:
                group = f"L{i}_ln"
            elif "attn" in name:
                group = f"L{i}_attn"
            elif "mlp" in name:
                group = f"L{i}_mlp"
            else:
                group = f"L{i}_other"
        elif "ln_f" in name:
            group = "ln_f"
        else:
            group = "other"
        layer_groups.setdefault(group, []).append(delta)
    return {g: torch.cat(vs).norm().item() for g, vs in layer_groups.items()}


# ---------------------------------------------------------------------------
# 4. Critical sharpness (Hutchinson trace of Hessian)
# ---------------------------------------------------------------------------


def compute_critical_sharpness(
    ckpt_path: str,
    cfg: dict,
    burst_docs_BL: np.ndarray,
    n_samples: int = 128,
    n_hutchinson: int = 10,
) -> float:
    """Estimate sharpness via Hutchinson trace of the Hessian.

    Uses the formula: trace(H) ≈ (1/n) sum_v v^T H v
    where v ~ Rademacher.  H v is computed via two backward passes
    (Pearlmutter trick).

    Flash attention does not support double-backward, so we temporarily
    disable it via the SDPA context manager.

    Returns the mean Hessian trace (scalar).
    """
    net = load_net(cfg, ckpt_path)
    net.train()

    n = min(n_samples, burst_docs_BL.shape[0])
    idx = _rng.choice(burst_docs_BL.shape[0], n, replace=False)
    docs = torch.as_tensor(burst_docs_BL[idx], dtype=torch.long, device=DEVICE)
    inp, tgt = docs[:, :-1], docs[:, 1:]

    params = [p for p in net.parameters() if p.requires_grad]

    traces = []
    # Disable flash attention — it doesn't support double backward (Hessian)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        for _ in range(n_hutchinson):
            net.zero_grad()
            logits = net(inp).float()
            _B, _T_seq, V = logits.shape
            loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))
            grads = torch.autograd.grad(loss, params, create_graph=True)

            v_list = [torch.randint_like(p, 0, 2).float() * 2 - 1 for p in params]
            gv = sum((g * v).sum() for g, v in zip(grads, v_list, strict=True))

            hvp = torch.autograd.grad(gv, params, retain_graph=False)
            trace = sum((hv * v).sum().item() for hv, v in zip(hvp, v_list, strict=True))
            traces.append(trace)
            net.zero_grad()

    return float(np.mean(traces))


# ---------------------------------------------------------------------------
# 5. Weight delta rank (SVD of W_post - W_pre)
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_weight_delta_rank(
    ckpt_pre: str,
    ckpt_post: str,
    cfg: dict,
    threshold: float = 0.99,
) -> dict:
    """Compute effective rank of (W_post - W_pre) per weight matrix.

    Effective rank = number of singular values needed to explain
    `threshold` fraction of total variance.
    """
    net_pre = load_net(cfg, ckpt_pre)
    net_post = load_net(cfg, ckpt_post)

    ranks = {}
    total_norms = {}
    for name, p_post in net_post.named_parameters():
        p_pre = dict(net_pre.named_parameters())[name]
        delta = (p_post - p_pre).float().cpu()
        if delta.dim() < 2:
            continue
        if delta.dim() > 2:
            delta = delta.view(delta.shape[0], -1)
        try:
            sv = torch.linalg.svdvals(delta)
            cumvar = torch.cumsum(sv**2, dim=0) / (sv**2).sum()
            rank = int((cumvar < threshold).sum().item()) + 1
            ranks[name] = rank
            total_norms[name] = delta.norm().item()
        except (RuntimeError, torch.linalg.LinAlgError):
            pass

    layer_ranks: dict[str, list[int]] = {}
    for name, rank in ranks.items():
        parts = name.split(".")
        if "wte" in name or "wpe" in name:
            group = "emb"
        elif len(parts) >= 4 and parts[1] == "h":
            i = parts[2]
            if "ln_" in name:
                group = f"L{i}_ln"
            elif "attn" in name:
                group = f"L{i}_attn"
            elif "mlp" in name:
                group = f"L{i}_mlp"
            else:
                group = f"L{i}_other"
        elif "ln_f" in name:
            group = "ln_f"
        else:
            group = "other"
        layer_ranks.setdefault(group, []).append(rank)

    mean_rank_per_group = {g: float(np.mean(rs)) for g, rs in layer_ranks.items()}
    return {
        "per_param_rank": ranks,
        "mean_rank_per_group": mean_rank_per_group,
        "total_rank": sum(ranks.values()),
        "total_norm": sum(total_norms.values()),
    }


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------


def analyse_run(
    run_dir: Path,
    adl_seeds: int = 3,
    adl_n_samples: int = 256,
    sharpness_seeds: int = 3,
    n_hutchinson: int = 10,
) -> dict:
    """Run all five analyses on a single run directory."""
    from burst.core.train_utils import resolve_run_paths

    cfg_path, logs_dir, _ = resolve_run_paths(run_dir)
    with cfg_path.open() as f:
        run_cfg = json.load(f)

    rc = parse_run_config(run_cfg)
    base_cfg = rc["base_cfg"]
    n_a = rc["n_a"]
    depth = rc["depth"]
    burst_pos = rc["burst_pos"]
    T = base_cfg["total_steps"]

    with logs_dir / "_data.pkl".open("rb") as f:
        target_pool, bg_pool, _, _, _ = pickle.load(f)

    other_docs_BL = np.concatenate(list(bg_pool.values()))
    burst_docs_BL = np.concatenate(list(target_pool.values()))
    prompt_len = run_cfg["task_info"]["prompt_len"]

    with logs_dir / "all_results.pkl".open("rb") as f:
        all_results = pickle.load(f)

    schedules_present = sorted({r["schedule"] for r in all_results})

    ckpt_root = logs_dir / "checkpoints"

    analysis = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "depth": depth,
        "burst_pos": burst_pos,
        "n_a": n_a,
        "schedules": schedules_present,
        "adl": {},
        "grad_interference": {},
        "task_vectors": {},
        "sharpness": {},
        "weight_delta_rank": {},
        "summary_metrics": {},
    }

    # -----------------------------------------------------------------------
    # Metric 2: Gradient interference (from existing data, free)
    # -----------------------------------------------------------------------
    for r in all_results:
        label = r["label"]
        if "grad_sim_log" in r:
            analysis["grad_interference"][label] = {
                "schedule": r["schedule"],
                "seed": r["seed"],
                **extract_grad_interference(r),
            }

    # -----------------------------------------------------------------------
    # Metrics 3 + 5: EMA interpolation probe + weight delta rank
    # -----------------------------------------------------------------------
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        sched = r["schedule"]
        jobs_by_schedule.setdefault(sched, []).append(r)

    for sched in schedules_present:
        sched_results = jobs_by_schedule[sched]
        seeds_done = 0
        ema_accs_by_alpha: dict[float, list[float]] = {}
        ema_cliff_alphas: list[float] = []
        ema_aucs: list[float] = []
        norms_by_group: dict[str, list[float]] = {}
        rank_by_group: dict[str, list[float]] = {}

        for r in sched_results:
            if seeds_done >= adl_seeds:
                break
            label = r["label"]
            ckpt_dir = ckpt_root / label
            if not ckpt_dir.exists():
                continue

            ckpt_files = {int(p.stem.split("_")[1]): p for p in ckpt_dir.glob("step_*.pt")}
            available = sorted(ckpt_files.keys())
            if not available:
                continue

            pre_step = available[0]
            peak_step = min(available, key=lambda x: abs(x - (T - 1)))
            rev_step = available[-1]
            ckpt_pre = str(ckpt_files[pre_step])
            ckpt_peak = str(ckpt_files[peak_step])
            ckpt_rev = str(ckpt_files[rev_step])

            cfg = r["config"]

            ema = ema_interpolation_probe(
                ckpt_peak,
                ckpt_rev,
                cfg,
                burst_docs_BL,
                prompt_len,
                n_samples=adl_n_samples,
            )
            for alpha, acc in zip(ema["alphas"], ema["accs"], strict=True):
                ema_accs_by_alpha.setdefault(alpha, []).append(acc)
            ema_cliff_alphas.append(ema["cliff_alpha"])
            ema_aucs.append(ema["auc"])

            norms = compute_task_vector_norms(ckpt_pre, ckpt_peak, cfg)
            for g, v in norms.items():
                norms_by_group.setdefault(g, []).append(v)

            dr = compute_weight_delta_rank(ckpt_pre, ckpt_peak, cfg)
            for g, v in dr["mean_rank_per_group"].items():
                rank_by_group.setdefault(g, []).append(v)

            seeds_done += 1

        alphas_sorted = sorted(ema_accs_by_alpha.keys())
        analysis["task_vectors"][sched] = {
            "alphas": alphas_sorted,
            "mean_accs": [float(np.mean(ema_accs_by_alpha[a])) for a in alphas_sorted],
            "mean_cliff_alpha": float(np.mean(ema_cliff_alphas))
            if ema_cliff_alphas
            else float("nan"),
            "mean_auc": float(np.mean(ema_aucs)) if ema_aucs else float("nan"),
            "mean_norms_by_group": {g: float(np.mean(v)) for g, v in norms_by_group.items()},
        }
        analysis["weight_delta_rank"][sched] = {
            "mean_rank_by_group": {g: float(np.mean(v)) for g, v in rank_by_group.items()},
            "total_rank": float(np.mean([sum(rank_by_group[g]) for g in rank_by_group])),
        }

    # -----------------------------------------------------------------------
    # Metric 4: Critical sharpness
    # -----------------------------------------------------------------------
    for sched in schedules_present:
        sched_results = jobs_by_schedule[sched]
        traces = []
        seeds_done = 0
        for r in sched_results:
            if seeds_done >= sharpness_seeds:
                break
            label = r["label"]
            ckpt_dir = ckpt_root / label
            if not ckpt_dir.exists():
                continue
            ckpt_files = {int(p.stem.split("_")[1]): p for p in ckpt_dir.glob("step_*.pt")}
            available = sorted(ckpt_files.keys())
            if not available:
                continue
            peak_step = min(available, key=lambda x: abs(x - (T - 1)))
            cfg = r["config"]
            trace = compute_critical_sharpness(
                str(ckpt_files[peak_step]),
                cfg,
                burst_docs_BL,
                n_samples=128,
                n_hutchinson=n_hutchinson,
            )
            traces.append(trace)
            seeds_done += 1

        analysis["sharpness"][sched] = {
            "traces": traces,
            "mean": float(np.mean(traces)) if traces else float("nan"),
            "std": float(np.std(traces)) if traces else float("nan"),
        }

    # -----------------------------------------------------------------------
    # Metric 1: ADL
    # -----------------------------------------------------------------------
    for sched in schedules_present:
        sched_results = jobs_by_schedule[sched]
        seeds_done = 0
        adl_by_step: dict[int, dict] = {}

        for r in sched_results:
            if seeds_done >= adl_seeds:
                break
            label = r["label"]
            ckpt_dir = ckpt_root / label
            if not ckpt_dir.exists():
                continue
            cfg = r["config"]

            adl_result = compute_adl_for_label(
                label,
                ckpt_dir,
                cfg,
                n_a,
                depth,
                other_docs_BL,
                burst_docs_BL,
                prompt_len,
                n_samples=adl_n_samples,
            )
            for step_data in adl_result["adl_steps"]:
                step = step_data["step"]
                adl_by_step.setdefault(
                    step,
                    {
                        "readability": [],
                        "max_acc_drop": [],
                        "delta_norm": [],
                        "acc_baseline": [],
                    },
                )
                adl_by_step[step]["readability"].append(
                    float(np.mean(step_data["mean_readability_K"]))
                )
                adl_by_step[step]["max_acc_drop"].append(step_data["max_acc_drop"])
                adl_by_step[step]["delta_norm"].append(float(np.mean(step_data["delta_norm_K"])))
                adl_by_step[step]["acc_baseline"].append(step_data["acc_baseline"])

            seeds_done += 1

        analysis["adl"][sched] = {
            step: {
                "mean_readability": float(np.mean(v["readability"])),
                "mean_max_acc_drop": float(np.mean(v["max_acc_drop"])),
                "mean_delta_norm": float(np.mean(v["delta_norm"])),
                "mean_acc_baseline": float(np.mean(v["acc_baseline"])),
            }
            for step, v in adl_by_step.items()
        }

    # -----------------------------------------------------------------------
    # Summary metrics (per schedule, averaged over seeds)
    # -----------------------------------------------------------------------
    for sched in schedules_present:
        sched_results = jobs_by_schedule[sched]
        analysis["summary_metrics"][sched] = {
            "mean_peak_burst": float(np.mean([r["peak_burst"] for r in sched_results])),
            "mean_reversion_auc": float(np.mean([r["reversion_auc"] for r in sched_results])),
            "mean_life_95": float(np.nanmean([r.get("life_95", np.nan) for r in sched_results])),
            "mean_life_90": float(np.nanmean([r.get("life_90", np.nan) for r in sched_results])),
            "mean_dropoff_abs": float(np.mean([r.get("dropoff_abs", 0) for r in sched_results])),
            "mean_dropoff_pct": float(np.mean([r.get("dropoff_pct", 0) for r in sched_results])),
        }

    return analysis


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_dashboard(analyses: list[dict], out_dir: Path) -> None:
    """Generate HTML dashboard + PNG charts from analysis results."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    from burst.dev._shared import sched_color as _color
    from burst.dev._shared import sched_order as _sched_order

    def _save_png(fig: Any, name: str) -> str:  # noqa: ANN401
        path = charts_dir / f"{name}.png"
        try:
            fig.write_image(str(path), width=1200, height=600, scale=2)
        except (ValueError, OSError):
            _plotly_to_png_matplotlib(fig, str(path), width=1200, height=600)
        return str(path)

    all_figs = []

    # -----------------------------------------------------------------------
    # Chart 1: Gradient interference over training (burst phase)
    # -----------------------------------------------------------------------
    fig1 = go.Figure()
    for analysis in analyses:
        run_name = analysis["run_name"]
        gi = analysis["grad_interference"]
        schedules = sorted({v["schedule"] for v in gi.values()}, key=_sched_order)
        for sched in schedules:
            sched_entries = [v for v in gi.values() if v["schedule"] == sched]
            all_steps = sorted({s for e in sched_entries for s in e["steps"]})
            all_sims = {s: [] for s in all_steps}
            for e in sched_entries:
                for s, sim in zip(e["steps"], e["burst_vs_other"], strict=True):
                    all_sims[s].append(sim)
            steps_arr = sorted(all_sims.keys())
            mean_sims = [np.mean(all_sims[s]) for s in steps_arr]
            fig1.add_trace(
                go.Scatter(
                    x=steps_arr,
                    y=mean_sims,
                    name=f"{sched} ({run_name})",
                    line={"color": _color(sched), "width": 2},
                    mode="lines",
                )
            )
    fig1.add_vline(x=500, line_dash="dash", line_color="gray", annotation_text="burst→reversion")
    fig1.update_layout(
        title="Gradient Interference: Burst vs Other Cosine Similarity Over Training",
        xaxis_title="Training Step",
        yaxis_title="Cosine Similarity (burst grad vs other grad)",
        legend_title="Schedule",
        template="plotly_white",
        height=500,
    )
    _save_png(fig1, "01_grad_interference_timeseries")
    all_figs.append(("Gradient Interference Time Series", fig1))

    # -----------------------------------------------------------------------
    # Chart 2: Gradient interference vs forgetting — 2x2 grid
    #   Rows: Y = Reversion AUC  |  Y = % Unlearning (dropoff_pct)
    #   Cols: X = Mean grad interference (burst phase)  |  X = End-of-burst grad interference
    # Each panel has one subplot per run (burst position).
    # -----------------------------------------------------------------------
    n_runs = len(analyses)
    row_labels = ["Reversion AUC (lower = faster forgetting)", "% Unlearning at Reversal End"]
    col_labels = ["Mean Gradient Interference (burst phase)", "End-of-Burst Gradient Interference"]
    y_keys = ["mean_reversion_auc", "mean_dropoff_pct"]
    x_keys = ["mean_burst_interference", "end_burst_interference"]

    subplot_titles = []
    for row_lbl in ["AUC", "% Unlearning"]:
        for col_lbl in ["Mean Grad Interference", "End-of-Burst Grad Interference"]:
            for a in analyses:
                subplot_titles.append(f"{a['run_name']} | {col_lbl} vs {row_lbl}")

    fig2 = make_subplots(
        rows=2,
        cols=2 * n_runs,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.15,
    )

    for row_idx, (y_key, y_label) in enumerate(zip(y_keys, row_labels, strict=True), start=1):
        for x_col, (x_key, x_label) in enumerate(zip(x_keys, col_labels, strict=True)):
            for run_idx, analysis in enumerate(analyses):
                col_idx = x_col * n_runs + run_idx + 1
                gi = analysis["grad_interference"]
                sm = analysis["summary_metrics"]
                schedules = sorted({v["schedule"] for v in gi.values()}, key=_sched_order)
                xs, ys, colors, labels = [], [], [], []
                for sched in schedules:
                    if sched not in sm:
                        continue
                    sched_entries = [v for v in gi.values() if v["schedule"] == sched]
                    x_val = float(np.nanmean([e.get(x_key, float("nan")) for e in sched_entries]))
                    y_val = sm[sched][y_key]
                    xs.append(x_val)
                    ys.append(y_val)
                    colors.append(_color(sched))
                    labels.append(sched)
                fig2.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="markers+text",
                        text=labels,
                        textposition="top center",
                        marker={"color": colors, "size": 12},
                        showlegend=False,
                    ),
                    row=row_idx,
                    col=col_idx,
                )
                fig2.update_xaxes(title_text=x_label, row=row_idx, col=col_idx)
                fig2.update_yaxes(title_text=y_label, row=row_idx, col=col_idx)

    fig2.update_layout(
        title="Gradient Interference vs Forgetting Speed",
        template="plotly_white",
        height=900,
    )
    _save_png(fig2, "02_grad_interference_vs_forgetting")
    all_figs.append(("Gradient Interference vs Forgetting", fig2))

    # -----------------------------------------------------------------------
    # Chart 3: Per-layer gradient interference heatmap
    # -----------------------------------------------------------------------
    for analysis in analyses:
        gi = analysis["grad_interference"]
        run_name = analysis["run_name"]
        schedules = sorted({v["schedule"] for v in gi.values()}, key=_sched_order)
        sample_entry = next(iter(gi.values()))
        layer_names = list(sample_entry["mean_layer_interference"].keys())

        z = []
        y_labels = []
        for sched in schedules:
            sched_entries = [v for v in gi.values() if v["schedule"] == sched]
            row = []
            for ln in layer_names:
                vals = [e["mean_layer_interference"].get(ln, np.nan) for e in sched_entries]
                row.append(float(np.nanmean(vals)))
            z.append(row)
            y_labels.append(sched)

        fig3 = go.Figure(
            go.Heatmap(
                z=z,
                x=layer_names,
                y=y_labels,
                colorscale="RdBu",
                zmid=0,
                colorbar={"title": "Cosine Sim"},
            )
        )
        fig3.update_layout(
            title=f"Per-Layer Gradient Interference — {run_name}",
            xaxis_title="Layer",
            yaxis_title="Schedule",
            template="plotly_white",
            height=500,
        )
        _save_png(fig3, f"03_per_layer_interference_{run_name}")
        all_figs.append((f"Per-Layer Interference ({run_name})", fig3))

    # -----------------------------------------------------------------------
    # Chart 4: EMA interpolation probe (peak ↔ reverted)
    # -----------------------------------------------------------------------
    for analysis in analyses:
        run_name = analysis["run_name"]
        tv = analysis["task_vectors"]
        schedules = sorted(tv.keys(), key=_sched_order)
        fig4 = go.Figure()
        for sched in schedules:
            data = tv[sched]
            fig4.add_trace(
                go.Scatter(
                    x=data["alphas"],
                    y=data["mean_accs"],
                    name=sched,
                    line={"color": _color(sched), "width": 2},
                    mode="lines+markers",
                )
            )
        fig4.update_layout(
            title=f"EMA Interpolation Probe — {run_name}<br>"
            "<sup>theta_interp = (1-alpha)·theta_reverted + alpha·theta_peak. "
            "Sharp cliff (burst_100) = wrapper; gradual ramp (burst_10) = deep</sup>",
            xaxis_title="alpha (0 = reverted model, 1 = peak burst model)",
            yaxis_title="Burst Accuracy",
            legend_title="Schedule",
            template="plotly_white",
            height=500,
        )
        _save_png(fig4, f"04_ema_interpolation_{run_name}")
        all_figs.append((f"EMA Interpolation Probe ({run_name})", fig4))

    # -----------------------------------------------------------------------
    # Chart 5: EMA cliff alpha vs forgetting (sharpness of cliff)
    # -----------------------------------------------------------------------
    fig5 = make_subplots(
        rows=1, cols=len(analyses), subplot_titles=[a["run_name"] for a in analyses]
    )
    for col_idx, analysis in enumerate(analyses, start=1):
        tv = analysis["task_vectors"]
        sm = analysis["summary_metrics"]
        schedules = sorted(tv.keys(), key=_sched_order)
        x_cliff = []
        y_auc = []
        colors = []
        labels = []
        for sched in schedules:
            if sched not in sm:
                continue
            x_cliff.append(tv[sched]["mean_cliff_alpha"])
            y_auc.append(sm[sched]["mean_reversion_auc"])
            colors.append(_color(sched))
            labels.append(sched)
        fig5.add_trace(
            go.Scatter(
                x=x_cliff,
                y=y_auc,
                mode="markers+text",
                text=labels,
                textposition="top center",
                marker={"color": colors, "size": 12},
                showlegend=False,
            ),
            row=1,
            col=col_idx,
        )
    fig5.update_xaxes(title_text="EMA Cliff Alpha (higher = sharper = shallower)")
    fig5.update_yaxes(title_text="Reversion AUC")
    fig5.update_layout(
        title="EMA Cliff Sharpness vs Forgetting Speed<br>"
        "<sup>Cliff alpha = alpha at which burst accuracy first exceeds 50%. "
        "High cliff alpha = capability concentrated in narrow direction (wrapper)</sup>",
        template="plotly_white",
        height=500,
    )
    _save_png(fig5, "05_ema_cliff_vs_forgetting")
    all_figs.append(("EMA Cliff vs Forgetting", fig5))

    # -----------------------------------------------------------------------
    # Chart 6: Critical sharpness vs forgetting
    # -----------------------------------------------------------------------
    fig6 = make_subplots(
        rows=1, cols=len(analyses), subplot_titles=[a["run_name"] for a in analyses]
    )
    for col_idx, analysis in enumerate(analyses, start=1):
        sh = analysis["sharpness"]
        sm = analysis["summary_metrics"]
        schedules = sorted(sh.keys(), key=_sched_order)
        x_sharp = []
        y_auc = []
        colors = []
        labels = []
        for sched in schedules:
            if sched not in sm:
                continue
            x_sharp.append(sh[sched]["mean"])
            y_auc.append(sm[sched]["mean_reversion_auc"])
            colors.append(_color(sched))
            labels.append(sched)
        fig6.add_trace(
            go.Scatter(
                x=x_sharp,
                y=y_auc,
                mode="markers+text",
                text=labels,
                textposition="top center",
                marker={"color": colors, "size": 12},
                showlegend=False,
            ),
            row=1,
            col=col_idx,
        )
    fig6.update_xaxes(title_text="Hessian Trace (sharpness at peak burst)")
    fig6.update_yaxes(title_text="Reversion AUC")
    fig6.update_layout(
        title="Critical Sharpness vs Forgetting Speed<br>"
        "<sup>Higher sharpness → sharper minimum → faster forgetting</sup>",
        template="plotly_white",
        height=500,
    )
    _save_png(fig6, "06_sharpness_vs_forgetting")
    all_figs.append(("Critical Sharpness vs Forgetting", fig6))

    # -----------------------------------------------------------------------
    # Chart 7: Sharpness bar chart across schedules
    # -----------------------------------------------------------------------
    fig7 = go.Figure()
    for analysis in analyses:
        sh = analysis["sharpness"]
        run_name = analysis["run_name"]
        schedules = sorted(sh.keys(), key=_sched_order)
        fig7.add_trace(
            go.Bar(
                x=schedules,
                y=[sh[s]["mean"] for s in schedules],
                error_y={"type": "data", "array": [sh[s]["std"] for s in schedules]},
                name=run_name,
                marker_color=[_color(s) for s in schedules],
            )
        )
    fig7.update_layout(
        title="Hessian Trace (Sharpness) at Peak Burst by Schedule",
        xaxis_title="Schedule",
        yaxis_title="Hessian Trace",
        template="plotly_white",
        height=500,
        barmode="group",
    )
    _save_png(fig7, "07_sharpness_bars")
    all_figs.append(("Sharpness Bars", fig7))

    # -----------------------------------------------------------------------
    # Chart 8: Weight delta rank per layer
    # -----------------------------------------------------------------------
    for analysis in analyses:
        run_name = analysis["run_name"]
        wdr = analysis["weight_delta_rank"]
        schedules = sorted(wdr.keys(), key=_sched_order)
        fig8 = go.Figure()
        for sched in schedules:
            data = wdr[sched]
            groups = list(data["mean_rank_by_group"].keys())
            ranks = [data["mean_rank_by_group"][g] for g in groups]
            fig8.add_trace(
                go.Bar(
                    x=groups,
                    y=ranks,
                    name=sched,
                    marker_color=_color(sched),
                )
            )
        fig8.update_layout(
            title=f"Weight Delta Effective Rank per Layer — {run_name}<br>"
            "<sup>Low rank = shallow/wrapper update; high rank = distributed change</sup>",
            xaxis_title="Layer Group",
            yaxis_title="Effective Rank (99% variance)",
            legend_title="Schedule",
            template="plotly_white",
            height=500,
            barmode="group",
        )
        _save_png(fig8, f"08_weight_delta_rank_{run_name}")
        all_figs.append((f"Weight Delta Rank ({run_name})", fig8))

    # -----------------------------------------------------------------------
    # Chart 9: Total weight delta rank vs forgetting
    # -----------------------------------------------------------------------
    fig9 = make_subplots(
        rows=1, cols=len(analyses), subplot_titles=[a["run_name"] for a in analyses]
    )
    for col_idx, analysis in enumerate(analyses, start=1):
        wdr = analysis["weight_delta_rank"]
        sm = analysis["summary_metrics"]
        schedules = sorted(wdr.keys(), key=_sched_order)
        x_rank = []
        y_auc = []
        colors = []
        labels = []
        for sched in schedules:
            if sched not in sm:
                continue
            x_rank.append(wdr[sched]["total_rank"])
            y_auc.append(sm[sched]["mean_reversion_auc"])
            colors.append(_color(sched))
            labels.append(sched)
        fig9.add_trace(
            go.Scatter(
                x=x_rank,
                y=y_auc,
                mode="markers+text",
                text=labels,
                textposition="top center",
                marker={"color": colors, "size": 12},
                showlegend=False,
            ),
            row=1,
            col=col_idx,
        )
    fig9.update_xaxes(title_text="Total Weight Delta Rank")
    fig9.update_yaxes(title_text="Reversion AUC")
    fig9.update_layout(
        title="Weight Update Rank vs Forgetting Speed<br>"
        "<sup>Low rank = modular wrapper; high rank = distributed representation</sup>",
        template="plotly_white",
        height=500,
    )
    _save_png(fig9, "09_weight_rank_vs_forgetting")
    all_figs.append(("Weight Rank vs Forgetting", fig9))

    # -----------------------------------------------------------------------
    # Chart 10: ADL readability at peak burst
    # -----------------------------------------------------------------------
    fig10 = go.Figure()
    for analysis in analyses:
        run_name = analysis["run_name"]
        adl = analysis["adl"]
        schedules = sorted(adl.keys(), key=_sched_order)
        x_sched = []
        y_readability = []
        colors = []
        for sched in schedules:
            step_data = adl[sched]
            peak_steps = [s for s in step_data if s <= 499]
            if not peak_steps:
                continue
            peak_step = max(peak_steps)
            x_sched.append(sched)
            y_readability.append(step_data[peak_step]["mean_readability"])
            colors.append(_color(sched))
        fig10.add_trace(
            go.Bar(
                x=x_sched,
                y=y_readability,
                name=run_name,
                marker_color=colors,
            )
        )
    fig10.update_layout(
        title="ADL Readability at Peak Burst<br>"
        "<sup>Fraction of top-10 logit-lens tokens on δ that are burst-relevant. "
        "High = global bias (wrapper); Low = no bias (deep)</sup>",
        xaxis_title="Schedule",
        yaxis_title="Mean Readability (burst tokens in top-10)",
        template="plotly_white",
        height=500,
        barmode="group",
    )
    _save_png(fig10, "10_adl_readability_peak")
    all_figs.append(("ADL Readability at Peak", fig10))

    # -----------------------------------------------------------------------
    # Chart 11: ADL causal ablation accuracy drop
    # -----------------------------------------------------------------------
    fig11 = go.Figure()
    for analysis in analyses:
        run_name = analysis["run_name"]
        adl = analysis["adl"]
        schedules = sorted(adl.keys(), key=_sched_order)
        x_sched = []
        y_drop = []
        colors = []
        for sched in schedules:
            step_data = adl[sched]
            peak_steps = [s for s in step_data if s <= 499]
            if not peak_steps:
                continue
            peak_step = max(peak_steps)
            x_sched.append(sched)
            y_drop.append(step_data[peak_step]["mean_max_acc_drop"])
            colors.append(_color(sched))
        fig11.add_trace(
            go.Bar(
                x=x_sched,
                y=y_drop,
                name=run_name,
                marker_color=colors,
            )
        )
    fig11.update_layout(
        title="ADL Causal Ablation: Max Accuracy Drop<br>"
        "<sup>Accuracy drop when δ direction is projected out. "
        "Large drop = capability stored in global bias (wrapper)</sup>",
        xaxis_title="Schedule",
        yaxis_title="Max Accuracy Drop (baseline - ablated)",
        template="plotly_white",
        height=500,
        barmode="group",
    )
    _save_png(fig11, "11_adl_causal_ablation")
    all_figs.append(("ADL Causal Ablation", fig11))

    # -----------------------------------------------------------------------
    # Chart 12: ADL delta norm over training steps
    # -----------------------------------------------------------------------
    for analysis in analyses:
        run_name = analysis["run_name"]
        adl = analysis["adl"]
        schedules = sorted(adl.keys(), key=_sched_order)
        fig12 = go.Figure()
        for sched in schedules:
            step_data = adl[sched]
            steps_sorted = sorted(step_data.keys())
            norms = [step_data[s]["mean_delta_norm"] for s in steps_sorted]
            fig12.add_trace(
                go.Scatter(
                    x=steps_sorted,
                    y=norms,
                    name=sched,
                    line={"color": _color(sched), "width": 2},
                    mode="lines+markers",
                )
            )
        fig12.add_vline(
            x=499, line_dash="dash", line_color="gray", annotation_text="burst→reversion"
        )
        fig12.update_layout(
            title=f"ADL Delta Norm Over Training — {run_name}<br>"
            "<sup>||δ|| = magnitude of global activation bias on other-class inputs</sup>",
            xaxis_title="Training Step",
            yaxis_title="Mean ||δ_K|| (activation bias magnitude)",
            legend_title="Schedule",
            template="plotly_white",
            height=500,
        )
        _save_png(fig12, f"12_adl_delta_norm_{run_name}")
        all_figs.append((f"ADL Delta Norm ({run_name})", fig12))

    # -----------------------------------------------------------------------
    # Chart 13: Summary — all metrics vs burstiness level
    # -----------------------------------------------------------------------
    for analysis in analyses:
        run_name = analysis["run_name"]
        sm = analysis["summary_metrics"]
        sh = analysis["sharpness"]
        wdr = analysis["weight_delta_rank"]
        gi = analysis["grad_interference"]
        tv = analysis["task_vectors"]
        adl = analysis["adl"]

        schedules = sorted(sm.keys(), key=_sched_order)
        burst_pct = [int(s.replace("burst_", "")) for s in schedules]

        fig13 = make_subplots(
            rows=3,
            cols=3,
            subplot_titles=[
                "Reversion AUC (lower = faster forgetting)",
                "Mean Gradient Interference (burst phase)",
                "End-of-Burst Gradient Interference",
                "% Unlearning at Reversal End",
                "Critical Sharpness",
                "EMA Cliff Alpha (sharpness)",
                "Weight Delta Total Rank",
                "ADL Readability at Peak",
                "",
            ],
        )

        def _add_scatter(fig, row, col, x, y, colors, labels) -> None:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers+lines",
                    marker={"color": colors, "size": 10},
                    line={"color": "gray", "width": 1, "dash": "dot"},
                    text=labels,
                    textposition="top center",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        colors = [_color(s) for s in schedules]

        _add_scatter(
            fig13,
            1,
            1,
            burst_pct,
            [sm[s]["mean_reversion_auc"] for s in schedules],
            colors,
            schedules,
        )

        gi_means, gi_ends = [], []
        for s in schedules:
            sched_entries = [v for v in gi.values() if v["schedule"] == s]
            gi_means.append(
                float(np.nanmean([e["mean_burst_interference"] for e in sched_entries]))
                if sched_entries
                else float("nan")
            )
            gi_ends.append(
                float(
                    np.nanmean(
                        [e.get("end_burst_interference", float("nan")) for e in sched_entries]
                    )
                )
                if sched_entries
                else float("nan")
            )
        _add_scatter(fig13, 1, 2, burst_pct, gi_means, colors, schedules)
        _add_scatter(fig13, 1, 3, burst_pct, gi_ends, colors, schedules)

        _add_scatter(
            fig13,
            2,
            1,
            burst_pct,
            [sm[s].get("mean_dropoff_pct", float("nan")) for s in schedules],
            colors,
            schedules,
        )

        _add_scatter(
            fig13,
            2,
            2,
            burst_pct,
            [sh.get(s, {}).get("mean", float("nan")) for s in schedules],
            colors,
            schedules,
        )

        ema_cliff = [tv.get(s, {}).get("mean_cliff_alpha", float("nan")) for s in schedules]
        _add_scatter(fig13, 2, 3, burst_pct, ema_cliff, colors, schedules)

        _add_scatter(
            fig13,
            3,
            1,
            burst_pct,
            [wdr.get(s, {}).get("total_rank", float("nan")) for s in schedules],
            colors,
            schedules,
        )

        adl_read = []
        for s in schedules:
            step_data = adl.get(s, {})
            peak_steps = [st for st in step_data if st <= 499]
            if peak_steps:
                adl_read.append(step_data[max(peak_steps)]["mean_readability"])
            else:
                adl_read.append(float("nan"))
        _add_scatter(fig13, 3, 2, burst_pct, adl_read, colors, schedules)

        fig13.update_xaxes(title_text="Burst %")
        fig13.update_layout(
            title=f"All Metrics vs Burstiness Level — {run_name}",
            template="plotly_white",
            height=1100,
        )
        _save_png(fig13, f"13_summary_all_metrics_{run_name}")
        all_figs.append((f"Summary All Metrics ({run_name})", fig13))

    # -----------------------------------------------------------------------
    # Assemble HTML dashboard
    # -----------------------------------------------------------------------
    html_parts = [
        """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Burstiness Deep Analysis Dashboard</title>
<style>
  body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
  h1 { color: #333; }
  h2 { color: #555; margin-top: 40px; }
  .chart-container { background: white; border-radius: 8px; padding: 16px;
                     margin: 16px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
  .toc { background: white; border-radius: 8px; padding: 16px; margin: 16px 0; }
  .toc a { display: block; margin: 4px 0; color: #1565c0; text-decoration: none; }
  .toc a:hover { text-decoration: underline; }
  .metric-label { font-size: 0.85em; color: #888; margin-bottom: 4px; }
</style>
</head>
<body>
<h1>Burstiness Deep Analysis Dashboard</h1>
<p>Five mechanistic metrics connecting burstiness → shallowness → fast forgetting.</p>
<div class="toc">
  <strong>Contents:</strong>
"""
    ]

    for i, (title, _) in enumerate(all_figs):
        anchor = f"chart_{i}"
        html_parts.append(f'  <a href="#{anchor}">{i + 1}. {title}</a>\n')

    html_parts.append("</div>\n")

    for i, (title, fig) in enumerate(all_figs):
        anchor = f"chart_{i}"
        html_parts.append(f'<div class="chart-container" id="{anchor}">\n')
        html_parts.append(f"<h2>{i + 1}. {title}</h2>\n")
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=(i == 0)))
        html_parts.append("</div>\n")

    html_parts.append("</body></html>")

    html_path = out_dir / "dashboard.html"
    with html_path.open("w") as f:
        f.write("".join(html_parts))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _is_valid_run_dir(d: Path) -> bool:
    """Return True if the directory has checkpoints and all_results.pkl."""
    return (
        (d / "all_results.pkl").exists()
        and (d / "checkpoints").exists()
        and any((d / "checkpoints").iterdir())
    )


def _find_all_run_dirs(data_root: Path) -> list[Path]:
    """Scan data_root for all valid burst run directories."""
    candidates = sorted(
        p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("burst_")
    )
    return [p for p in candidates if _is_valid_run_dir(p)]


def _analyse_run_worker(args_tuple: tuple) -> dict:
    """Subprocess entry point for parallel analysis of one run dir."""
    run_dir, adl_seeds, adl_samples, sharpness_seeds, n_hutchinson = args_tuple
    return analyse_run(
        Path(run_dir),
        adl_seeds=adl_seeds,
        adl_n_samples=adl_samples,
        sharpness_seeds=sharpness_seeds,
        n_hutchinson=n_hutchinson,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Five-metric deep analysis of burstiness runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run_dirs",
        nargs="*",
        type=Path,
        help="Run directories to analyse. Omit when using --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scan data/ for all valid run directories automatically.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory to scan when --all is set (default: data/)",
    )
    parser.add_argument(
        "--n-parallel",
        type=int,
        default=1,
        help="Number of run dirs to analyse in parallel (default: 1). "
        "Each worker uses the full GPU; set to 2-4 on a 24GB card.",
    )
    parser.add_argument(
        "--adl-seeds",
        type=int,
        default=3,
        help="Seeds per schedule for ADL + EMA probe (default: 3)",
    )
    parser.add_argument(
        "--adl-samples", type=int, default=256, help="Docs per ADL forward pass (default: 256)"
    )
    parser.add_argument(
        "--sharpness-seeds",
        type=int,
        default=3,
        help="Seeds per schedule for sharpness (default: 3)",
    )
    parser.add_argument(
        "--n-hutchinson",
        type=int,
        default=10,
        help="Hutchinson samples for sharpness estimate (default: 20)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/deep_analysis_combined)",
    )
    args = parser.parse_args()

    if args.all:
        run_dirs = _find_all_run_dirs(args.data_root)
        if not run_dirs:
            return
    elif args.run_dirs:
        run_dirs = [Path(d) for d in args.run_dirs]
    else:
        parser.error("Provide run_dirs or use --all")

    out_dir = args.out_dir or Path("data/deep_analysis_combined")
    out_dir.mkdir(parents=True, exist_ok=True)

    worker_args = [
        (str(d), args.adl_seeds, args.adl_samples, args.sharpness_seeds, args.n_hutchinson)
        for d in run_dirs
    ]

    if args.n_parallel > 1:
        # spawn context avoids CUDA fork issues
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=args.n_parallel) as pool:
            analyses = pool.map(_analyse_run_worker, worker_args)
    else:
        analyses = []
        for wa in worker_args:
            analysis = _analyse_run_worker(wa)
            analyses.append(analysis)

    results_path = out_dir / "results.pkl"
    with results_path.open("wb") as f:
        pickle.dump(analyses, f)

    make_dashboard(analyses, out_dir)


if __name__ == "__main__":
    main()
