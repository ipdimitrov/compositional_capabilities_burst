"""Finetuning fingerprint analysis: Logit Lens on activation deltas + activation steering.

Adapted from the "Narrow Fine-Tuning Targets" paper's methodology to our
burst-learning setup.  Two complementary analyses:

1. **Logit Lens on Checkpoint Deltas** -- For each (layer, position), compute
   d_bar = E_x[h^post(x) - h^pre(x)] on other-class inputs, project through
   the unembedding matrix (with and without ln_f), and check whether the
   top-k tokens are burst-relevant.  This reveals whether the burst phase
   leaves a readable "fingerprint" in activation space even on non-burst data.

2. **Activation Steering** -- Add a*d_bar to the residual stream at layer l
   during autoregressive generation on other-class prompts.  If the steered
   model starts producing burst-class outputs, the burst knowledge is stored
   as an additive direction (wrapper), not a conditional circuit (deep).

Usage:
    uv run python burst/fingerprint_analysis.py <run_dir>
    uv run python burst/fingerprint_analysis.py --all --data-root data/

Dimension key:
    B: batch_size
    L: sequence_length (doc_len)
    N: n_embd (model dimension)
    K: n_layers + 1 (embedding + each transformer block output)
    T: token positions (L - 1)
    V: vocab_size
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from plotly.subplots import make_subplots

from burst.config import (
    parse_run_config,
)
from burst.core.activations import collect_activations_KPTN
from burst.core.train_utils import (
    DEVICE,
    burst_token_ids,
    free_gen_acc,
    load_net,
    resolve_run_paths,
    sched_order,
)
from burst.core.train_utils import (
    sched_color as color,
)
from burst.dev.plot_utils import save_png

if TYPE_CHECKING:
    from net.nanogpt import nanoGPT

logger = logging.getLogger(__name__)

_rng = np.random.default_rng()


# ---------------------------------------------------------------------------
# 1. Logit Lens on checkpoint deltas
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_delta_KTN(  # noqa: N802
    net_post: nanoGPT,
    net_pre: nanoGPT,
    docs_BL: np.ndarray,
    n_samples: int,
) -> torch.Tensor:
    """Mean activation difference δ̄[k,t,:] = E_x[ h^post_k(x)_t - h^pre_k(x)_t ].

    Returns (K, T, N) on CPU.
    """
    n = min(n_samples, docs_BL.shape[0])
    idx = _rng.choice(docs_BL.shape[0], n, replace=False)
    docs = docs_BL[idx]
    acts_post = collect_activations_KPTN(net_post, docs)
    acts_pre = collect_activations_KPTN(net_pre, docs)
    K = len(acts_post)
    return torch.stack([(acts_post[k] - acts_pre[k]).mean(dim=0) for k in range(K)]).cpu()


@torch.no_grad()
def logit_lens_on_delta(
    net: nanoGPT,
    delta_KTN: torch.Tensor,
    burst_token_ids: list[int],
    top_k: int = 20,
    *,
    use_ln: bool = True,
) -> dict:
    """Project d_bar through [ln_f +] W_U and measure burst-token presence in top-k.

    When use_ln=True (default), applies the model's final LayerNorm before
    unembedding — this is the "standard" Logit Lens.  When False, does a raw
    projection (just W_U · δ̄), which is what the paper uses since LN on a
    difference vector is a hack.

    Returns:
        readability_KT: (K, T) — fraction of top-k that are burst tokens
        mean_rank_KT: (K, T) — mean rank of burst tokens in the logit ordering
        top_tokens_KT: (K, T, top_k) — actual top-k token ids
        softmax_entropy_KT: (K, T) — entropy of the softmax distribution

    """
    K, T, N = delta_KTN.shape
    delta_dev = delta_KTN.float().to(DEVICE)

    if use_ln:
        ln_f = net.transformer.ln_f
        delta_flat = delta_dev.view(-1, N)
        normed_flat = ln_f(delta_flat)
        delta_normed = normed_flat.view(K, T, N)
    else:
        delta_normed = delta_dev

    unembed_VN = net.LM_head.weight.detach().float()
    logits_KTV = torch.einsum("ktn,vn->ktv", delta_normed, unembed_VN)

    probs_KTV = F.softmax(logits_KTV, dim=-1)
    log_probs_KTV = F.log_softmax(logits_KTV, dim=-1)
    entropy_KT = -(probs_KTV * log_probs_KTV).sum(dim=-1).cpu().numpy()

    V = logits_KTV.shape[-1]
    burst_set = set(burst_token_ids)
    readability_KT = np.zeros((K, T))
    mean_rank_KT = np.full((K, T), float(V))
    top_tokens_KT = np.zeros((K, T, top_k), dtype=np.int64)

    sorted_ids_KTV = torch.argsort(logits_KTV, dim=-1, descending=True).cpu().numpy()

    for k in range(K):
        for t in range(T):
            top_ids = sorted_ids_KTV[k, t, :top_k]
            top_tokens_KT[k, t] = top_ids
            readability_KT[k, t] = sum(1 for tid in top_ids if tid in burst_set) / top_k
            ranks = [i for i, tid in enumerate(sorted_ids_KTV[k, t]) if tid in burst_set]
            if ranks:
                mean_rank_KT[k, t] = float(np.mean(ranks))

    return {
        "readability_KT": readability_KT,
        "mean_rank_KT": mean_rank_KT,
        "top_tokens_KT": top_tokens_KT,
        "entropy_KT": entropy_KT,
    }


@torch.no_grad()
def logit_lens_compare_methods(
    net: nanoGPT,
    delta_KTN: torch.Tensor,
    burst_token_ids: list[int],
    top_k: int = 20,
) -> dict:
    """Run Logit Lens with and without LayerNorm, plus the logit difference baseline.

    Returns a dict with keys 'with_ln', 'without_ln' each containing the
    logit_lens_on_delta output.
    """
    with_ln = logit_lens_on_delta(net, delta_KTN, burst_token_ids, top_k, use_ln=True)
    without_ln = logit_lens_on_delta(net, delta_KTN, burst_token_ids, top_k, use_ln=False)
    return {"with_ln": with_ln, "without_ln": without_ln}


# ---------------------------------------------------------------------------
# 2. Activation Steering
# ---------------------------------------------------------------------------


@torch.no_grad()
def steering_experiment(  # noqa: PLR0913
    net: nanoGPT,
    delta_KTN: torch.Tensor,
    other_docs_BL: np.ndarray,
    burst_docs_BL: np.ndarray,
    prompt_len: int,
    steer_layer: int,
    alphas: list[float] | None = None,
    n_samples: int = 128,
) -> dict:
    """Steer the model by adding a*d_bar to the residual stream at `steer_layer`.

    For each alpha, measures:
      - burst_acc: accuracy on burst-class outputs (does the model start applying b*?)
      - other_acc: accuracy on other-class outputs (does normal computation survive?)

    The d_bar is first normalised to match the mean activation norm at that layer
    (same idea as the paper's norm-matching step), then scaled by alpha.

    Returns dict with alphas, burst_accs, other_accs, and metadata.
    """
    if alphas is None:
        alphas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

    net.eval()
    n_other = min(n_samples, other_docs_BL.shape[0])
    n_burst = min(n_samples, burst_docs_BL.shape[0])
    other_idx = _rng.choice(other_docs_BL.shape[0], n_other, replace=False)
    burst_idx = _rng.choice(burst_docs_BL.shape[0], n_burst, replace=False)
    other_docs = other_docs_BL[other_idx]
    burst_docs = burst_docs_BL[burst_idx]

    delta_TN = delta_KTN[steer_layer].to(DEVICE).float()
    delta_norm = delta_TN.norm()
    if delta_norm < 1e-8:  # noqa: PLR2004
        return {
            "alphas": alphas,
            "burst_acc_on_other": [0.0] * len(alphas),
            "other_acc_on_other": [0.0] * len(alphas),
            "burst_acc_on_burst": [0.0] * len(alphas),
            "steer_layer": steer_layer,
            "delta_norm": 0.0,
        }

    burst_accs_on_other = []
    other_accs_on_other = []
    burst_accs_on_burst = []

    for alpha in alphas:
        scaled_delta_TN = alpha * delta_TN

        def steer_forward_hook(
            _module: torch.nn.Module,
            _input: tuple,
            output: torch.Tensor | tuple,
            _scaled: torch.Tensor = scaled_delta_TN,
        ) -> torch.Tensor | tuple:
            if isinstance(output, tuple):
                x_raw, rest = output[0], output[1:]
            else:
                x_raw, rest = output, None
            x = x_raw.float()
            T_use = min(x.shape[1], _scaled.shape[0])
            x[:, :T_use] += _scaled[:T_use].unsqueeze(0)
            x = x.to(x_raw.dtype)
            if rest is not None:
                return (x, *rest)
            return x

        if steer_layer == 0:
            handle = net.transformer.drop.register_forward_hook(steer_forward_hook)
        else:
            handle = net.transformer.h[steer_layer - 1].register_forward_hook(steer_forward_hook)

        try:
            burst_on_other = measure_burst_acc_on_other(net, other_docs, burst_docs, prompt_len)
            other_on_other = free_gen_acc(net, other_docs, prompt_len)
            burst_on_burst = free_gen_acc(net, burst_docs, prompt_len)
        finally:
            handle.remove()

        burst_accs_on_other.append(burst_on_other)
        other_accs_on_other.append(other_on_other)
        burst_accs_on_burst.append(burst_on_burst)

    return {
        "alphas": alphas,
        "burst_acc_on_other": burst_accs_on_other,
        "other_acc_on_other": other_accs_on_other,
        "burst_acc_on_burst": burst_accs_on_burst,
        "steer_layer": steer_layer,
        "delta_norm": delta_norm.item(),
    }


@torch.no_grad()
def measure_burst_acc_on_other(
    net: nanoGPT,
    other_docs_BL: np.ndarray,
    burst_docs_BL: np.ndarray,
    prompt_len: int,
) -> float:
    """Check if steered other-class prompts produce burst-class permutation outputs.

    We take other-class prompts but check if the generated output tokens match
    what the burst function b* would produce.  This requires knowing the burst
    permutation, which we infer from the burst_docs ground truth.

    As a simpler proxy: we check what fraction of generated output tokens
    match any burst-doc output token pattern.  This is approximate but
    captures the steering effect.
    """
    net.eval()
    other_t = torch.as_tensor(other_docs_BL, dtype=torch.long, device=DEVICE)
    burst_t = torch.as_tensor(burst_docs_BL, dtype=torch.long, device=DEVICE)
    B_other, L = other_t.shape

    generated = net.generate(other_t[:, :prompt_len], L - prompt_len)
    gen_outputs = generated[:, -6:]

    burst_outputs = burst_t[:, -6:]
    burst_output_set = {tuple(row.tolist()) for row in burst_outputs}

    matches = sum(1 for row in gen_outputs if tuple(row.tolist()) in burst_output_set)
    return matches / B_other


@torch.no_grad()
def steering_sweep_layers(  # noqa: PLR0913
    net: nanoGPT,
    delta_KTN: torch.Tensor,
    other_docs_BL: np.ndarray,
    burst_docs_BL: np.ndarray,
    prompt_len: int,
    alpha: float = 5.0,
    n_samples: int = 128,
) -> dict:
    """Sweep steering across all layers at a fixed alpha to find which layer is most steerable."""
    K = delta_KTN.shape[0]
    burst_accs = []
    other_accs = []
    delta_norms = []

    for k in range(K):
        result = steering_experiment(
            net,
            delta_KTN,
            other_docs_BL,
            burst_docs_BL,
            prompt_len,
            steer_layer=k,
            alphas=[alpha],
            n_samples=n_samples,
        )
        burst_accs.append(result["burst_acc_on_other"][0])
        other_accs.append(result["other_acc_on_other"][0])
        delta_norms.append(result["delta_norm"])

    return {
        "layers": list(range(K)),
        "burst_acc_on_other": burst_accs,
        "other_acc_on_other": other_accs,
        "delta_norms": delta_norms,
        "alpha": alpha,
    }


# ---------------------------------------------------------------------------
# Full analysis pipeline for one run
# ---------------------------------------------------------------------------


def analyse_run(  # noqa: C901, PLR0915
    run_dir: Path,
    n_seeds: int = 3,
    n_samples: int = 256,
    top_k: int = 20,
    steering_alphas: list[float] | None = None,
) -> dict:
    """Run Logit Lens + Steering analysis on a single run directory."""
    logger.info("Fingerprint analysis: %s", run_dir.name)
    cfg_path, logs_dir, _ = resolve_run_paths(run_dir)
    with cfg_path.open() as f:
        run_cfg = json.load(f)

    rc = parse_run_config(run_cfg)
    base_cfg = rc["base_cfg"]
    n_a = rc["n_a"]
    depth = rc["depth"]
    T = base_cfg["total_steps"]

    with (logs_dir / "_data.pkl").open("rb") as f:
        target_pool, bg_pool, _, _, _ = pickle.load(f)  # noqa: S301

    other_docs_BL = np.concatenate(list(bg_pool.values()))
    burst_docs_BL = np.concatenate(list(target_pool.values()))
    prompt_len = run_cfg["task_info"]["prompt_len"]

    with (logs_dir / "all_results.pkl").open("rb") as f:
        all_results = pickle.load(f)  # noqa: S301

    ckpt_root = logs_dir / "checkpoints"
    schedules_present = sorted({r["schedule"] for r in all_results})

    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    analysis = {
        "run_name": run_dir.name,
        "burst_pos": rc["burst_pos"],
        "depth": rc["depth"],
        "n_a": n_a,
        "logit_lens": {},
        "steering": {},
        "steering_layer_sweep": {},
    }

    for sched in schedules_present:
        sched_results = jobs_by_schedule[sched]
        seeds_done = 0

        ll_readability_agg: dict[str, list] = {"with_ln": [], "without_ln": []}
        ll_mean_rank_agg: dict[str, list] = {"with_ln": [], "without_ln": []}
        ll_entropy_agg: dict[str, list] = {"with_ln": [], "without_ln": []}
        steer_agg: list[dict] = []
        layer_sweep_agg: list[dict] = []

        for r in sched_results:
            if seeds_done >= n_seeds:
                break
            label = r["label"]
            ckpt_dir = ckpt_root / label
            if not ckpt_dir.exists():
                continue

            ckpt_files = {int(p.stem.split("_")[1]): p for p in ckpt_dir.glob("step_*.pt")}
            available = sorted(ckpt_files.keys())
            if len(available) < 2:  # noqa: PLR2004
                continue

            cfg = r["config"]
            burst_ids = burst_token_ids(cfg, n_a, depth)

            pre_step = available[0]
            peak_step = min(available, key=lambda x: abs(x - (T - 1)))

            logger.debug("[%s] %s: pre=%d, peak=%d", sched, label, pre_step, peak_step)

            net_pre = load_net(cfg, str(ckpt_files[pre_step]))
            net_peak = load_net(cfg, str(ckpt_files[peak_step]))

            delta_KTN = compute_delta_KTN(net_peak, net_pre, other_docs_BL, n_samples)

            # --- Logit Lens ---
            ll_result = logit_lens_compare_methods(net_peak, delta_KTN, burst_ids, top_k)
            for method in ("with_ln", "without_ln"):
                ll_readability_agg[method].append(ll_result[method]["readability_KT"])
                ll_mean_rank_agg[method].append(ll_result[method]["mean_rank_KT"])
                ll_entropy_agg[method].append(ll_result[method]["entropy_KT"])

            logger.debug(
                "  Logit Lens readability (with LN): mean=%.3f",
                ll_result["with_ln"]["readability_KT"].mean(),
            )

            # --- Steering (best layer = middle) ---
            K = delta_KTN.shape[0]
            mid_layer = K // 2
            steer_result = steering_experiment(
                net_peak,
                delta_KTN,
                other_docs_BL,
                burst_docs_BL,
                prompt_len,
                steer_layer=mid_layer,
                alphas=steering_alphas,
                n_samples=min(n_samples, 128),
            )
            steer_agg.append(steer_result)

            # --- Layer sweep at alpha=5 ---
            sweep = steering_sweep_layers(
                net_peak,
                delta_KTN,
                other_docs_BL,
                burst_docs_BL,
                prompt_len,
                alpha=5.0,
                n_samples=min(n_samples, 64),
            )
            layer_sweep_agg.append(sweep)

            seeds_done += 1

        if not ll_readability_agg["with_ln"]:
            continue

        def mean_over_seeds(arrs: list[np.ndarray]) -> list:
            return np.mean(arrs, axis=0).tolist()

        analysis["logit_lens"][sched] = {
            method: {
                "mean_readability_KT": mean_over_seeds(ll_readability_agg[method]),
                "mean_rank_KT": mean_over_seeds(ll_mean_rank_agg[method]),
                "mean_entropy_KT": mean_over_seeds(ll_entropy_agg[method]),
                "overall_readability": float(
                    np.mean([a.mean() for a in ll_readability_agg[method]])
                ),
                "overall_mean_rank": float(np.mean([a.mean() for a in ll_mean_rank_agg[method]])),
            }
            for method in ("with_ln", "without_ln")
        }

        if steer_agg:
            alphas = steer_agg[0]["alphas"]
            analysis["steering"][sched] = {
                "alphas": alphas,
                "mean_burst_acc_on_other": [
                    float(np.mean([s["burst_acc_on_other"][i] for s in steer_agg]))
                    for i in range(len(alphas))
                ],
                "mean_other_acc_on_other": [
                    float(np.mean([s["other_acc_on_other"][i] for s in steer_agg]))
                    for i in range(len(alphas))
                ],
                "mean_burst_acc_on_burst": [
                    float(np.mean([s["burst_acc_on_burst"][i] for s in steer_agg]))
                    for i in range(len(alphas))
                ],
                "steer_layer": steer_agg[0]["steer_layer"],
                "mean_delta_norm": float(np.mean([s["delta_norm"] for s in steer_agg])),
            }

        if layer_sweep_agg:
            n_layers = len(layer_sweep_agg[0]["layers"])
            analysis["steering_layer_sweep"][sched] = {
                "layers": layer_sweep_agg[0]["layers"],
                "mean_burst_acc": [
                    float(np.mean([s["burst_acc_on_other"][i] for s in layer_sweep_agg]))
                    for i in range(n_layers)
                ],
                "mean_other_acc": [
                    float(np.mean([s["other_acc_on_other"][i] for s in layer_sweep_agg]))
                    for i in range(n_layers)
                ],
                "mean_delta_norms": [
                    float(np.mean([s["delta_norms"][i] for s in layer_sweep_agg]))
                    for i in range(n_layers)
                ],
                "alpha": layer_sweep_agg[0]["alpha"],
            }

    return analysis


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

_METRIC_DESCRIPTIONS = {
    "logit_lens_readability_heatmap": {
        "what": (
            "Fraction of top-20 tokens (from projecting the activation delta δ̄ "
            "through the unembedding matrix) that are burst-relevant, at each "
            "(layer, token position).  δ̄ = E_x[h^post(x) - h^pre(x)] on other-class inputs."
        ),
        "high": "The model's activation difference points strongly toward burst tokens — "
        "burst knowledge is encoded as a global bias (wrapper/shortcut).",
        "low": "The activation difference doesn't preferentially point toward burst tokens — "
        "burst knowledge is stored in conditional circuits (deep learning).",
        "formula": "readability = |{t ∈ top-k : t is burst-relevant}| / k",
    },
    "logit_lens_mean_rank": {
        "what": (
            "Mean rank of burst-relevant tokens in the logit ordering when projecting δ̄ "
            "through W_U.  Lower rank = δ̄ points more directly at burst tokens."
        ),
        "high": "Burst tokens are ranked low (far from top) — weak fingerprint.",
        "low": "Burst tokens are ranked high (near top) — strong fingerprint.",
    },
    "logit_lens_entropy": {
        "what": (
            "Entropy of the softmax distribution over vocabulary when projecting δ̄ "
            "through W_U.  Low entropy = δ̄ points sharply at specific tokens."
        ),
        "high": "Diffuse distribution — δ̄ doesn't point at any specific tokens.",
        "low": (
            "Concentrated distribution — δ̄ points sharply at specific tokens (strong fingerprint)."
        ),
    },
    "steering_alpha_sweep": {
        "what": (
            "Effect of adding alpha*d_bar to the residual stream during generation. "
            "Burst acc on other = does the model start applying b* on non-burst prompts? "
            "Other acc on other = does normal computation survive?"
        ),
        "high": "Steering successfully induces burst behaviour on other-class prompts — "
        "burst knowledge is an additive direction (wrapper).",
        "low": "Steering doesn't induce burst behaviour — knowledge is stored in "
        "conditional circuits that can't be activated by simple addition.",
    },
    "steering_layer_sweep": {
        "what": (
            "Which layer is most steerable?  At a fixed alpha, sweep across all layers "
            "and measure burst accuracy on other-class prompts."
        ),
        "high": "Steering at this layer strongly induces burst behaviour.",
        "low": "Steering at this layer has little effect.",
    },
    "logit_lens_comparison_bar": {
        "what": (
            "Comparison of Logit Lens readability with vs without LayerNorm. "
            "LN(δ̄) ≠ LN(h_ft) - LN(h_base) because LayerNorm is nonlinear, "
            "so these give different views of the fingerprint."
        ),
        "high": "Strong burst-token signal in the activation delta.",
        "low": "Weak burst-token signal.",
    },
}


def make_dashboard(analyses: list[dict], out_dir: Path) -> None:  # noqa: C901, PLR0912, PLR0915
    """Build an interactive HTML dashboard from fingerprint analyses."""
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    all_figs: list[tuple[str, str, go.Figure]] = []

    def register_fingerprint_fig(key: str, fig: go.Figure, title: str | None = None) -> None:
        t = title or fig.layout.title.text if fig.layout.title else key
        if isinstance(t, dict):
            t = t.get("text", key)
        all_figs.append((key, t, fig))
        save_png(fig, str(charts_dir / f"{key}.png"))

    # ------------------------------------------------------------------
    # Chart 1: Logit Lens Readability Heatmap (per schedule, with LN)
    # ------------------------------------------------------------------
    for analysis in analyses:
        run_name = analysis["run_name"]
        ll = analysis.get("logit_lens", {})
        schedules = sorted(ll.keys(), key=sched_order)
        if not schedules:
            continue

        for method, method_label in [("with_ln", "with LayerNorm"), ("without_ln", "raw (no LN)")]:
            fig = make_subplots(
                rows=len(schedules),
                cols=1,
                subplot_titles=[f"{s} — {method_label}" for s in schedules],
                vertical_spacing=0.05,
                shared_xaxes=True,
            )
            for i, sched in enumerate(schedules):
                data = ll[sched][method]
                readability = np.array(data["mean_readability_KT"])
                K, T = readability.shape
                fig.add_trace(
                    go.Heatmap(
                        z=readability,
                        x=list(range(T)),
                        y=[f"L{k}" for k in range(K)],
                        colorscale="YlOrRd",
                        zmin=0,
                        zmax=1,
                        colorbar={
                            "title": "Readability",
                            "len": 1 / len(schedules),
                            "y": 1 - i / len(schedules),
                        },
                        showscale=(i == 0),
                    ),
                    row=i + 1,
                    col=1,
                )
                fig.update_yaxes(title_text="Layer", row=i + 1, col=1)

            fig.update_xaxes(title_text="Token Position", row=len(schedules), col=1)
            fig.update_layout(
                title=f"Logit Lens Readability ({method_label}) — {run_name}",
                template="plotly_white",
                height=250 * len(schedules) + 100,
            )
            register_fingerprint_fig(
                f"logit_lens_readability_heatmap_{method}_{run_name}",
                fig,
                f"Logit Lens Readability ({method_label}) — {run_name}",
            )

    # ------------------------------------------------------------------
    # Chart 2: Logit Lens readability bar chart (schedule comparison)
    # ------------------------------------------------------------------
    for analysis in analyses:
        run_name = analysis["run_name"]
        ll = analysis.get("logit_lens", {})
        schedules = sorted(ll.keys(), key=sched_order)
        if not schedules:
            continue

        fig = go.Figure()
        for method, _offset, name in [("with_ln", -0.15, "With LN"), ("without_ln", 0.15, "Raw")]:
            fig.add_trace(
                go.Bar(
                    x=schedules,
                    y=[ll[s][method]["overall_readability"] for s in schedules],
                    name=name,
                    marker_color=[color(s) for s in schedules],
                    opacity=1.0 if method == "with_ln" else 0.5,
                )
            )
        fig.update_layout(
            title=f"Logit Lens: Overall Readability by Schedule — {run_name}",
            xaxis_title="Schedule",
            yaxis_title="Mean Readability (burst tokens in top-20)",
            template="plotly_white",
            height=500,
            barmode="group",
        )
        register_fingerprint_fig(
            f"logit_lens_comparison_bar_{run_name}",
            fig,
            f"Logit Lens Readability Comparison — {run_name}",
        )

    # ------------------------------------------------------------------
    # Chart 3: Logit Lens Mean Rank heatmap
    # ------------------------------------------------------------------
    for analysis in analyses:
        run_name = analysis["run_name"]
        ll = analysis.get("logit_lens", {})
        schedules = sorted(ll.keys(), key=sched_order)
        if not schedules:
            continue

        fig = make_subplots(
            rows=len(schedules),
            cols=1,
            subplot_titles=list(schedules),
            vertical_spacing=0.05,
            shared_xaxes=True,
        )
        for i, sched in enumerate(schedules):
            data = ll[sched]["with_ln"]
            mean_rank = np.array(data["mean_rank_KT"])
            K, T = mean_rank.shape
            fig.add_trace(
                go.Heatmap(
                    z=mean_rank,
                    x=list(range(T)),
                    y=[f"L{k}" for k in range(K)],
                    colorscale="Viridis_r",
                    colorbar={
                        "title": "Mean Rank",
                        "len": 1 / len(schedules),
                        "y": 1 - i / len(schedules),
                    },
                    showscale=(i == 0),
                ),
                row=i + 1,
                col=1,
            )
            fig.update_yaxes(title_text="Layer", row=i + 1, col=1)

        fig.update_xaxes(title_text="Token Position", row=len(schedules), col=1)
        fig.update_layout(
            title=f"Logit Lens: Mean Burst Token Rank — {run_name}",
            template="plotly_white",
            height=250 * len(schedules) + 100,
        )
        register_fingerprint_fig(
            f"logit_lens_mean_rank_{run_name}",
            fig,
            f"Logit Lens Mean Rank — {run_name}",
        )

    # ------------------------------------------------------------------
    # Chart 4: Logit Lens Entropy heatmap
    # ------------------------------------------------------------------
    for analysis in analyses:
        run_name = analysis["run_name"]
        ll = analysis.get("logit_lens", {})
        schedules = sorted(ll.keys(), key=sched_order)
        if not schedules:
            continue

        fig = make_subplots(
            rows=len(schedules),
            cols=1,
            subplot_titles=list(schedules),
            vertical_spacing=0.05,
            shared_xaxes=True,
        )
        for i, sched in enumerate(schedules):
            data = ll[sched]["with_ln"]
            entropy = np.array(data["mean_entropy_KT"])
            K, T = entropy.shape
            fig.add_trace(
                go.Heatmap(
                    z=entropy,
                    x=list(range(T)),
                    y=[f"L{k}" for k in range(K)],
                    colorscale="Blues",
                    colorbar={
                        "title": "Entropy",
                        "len": 1 / len(schedules),
                        "y": 1 - i / len(schedules),
                    },
                    showscale=(i == 0),
                ),
                row=i + 1,
                col=1,
            )
            fig.update_yaxes(title_text="Layer", row=i + 1, col=1)

        fig.update_xaxes(title_text="Token Position", row=len(schedules), col=1)
        fig.update_layout(
            title=f"Logit Lens: Softmax Entropy of δ̄ Projection — {run_name}",
            template="plotly_white",
            height=250 * len(schedules) + 100,
        )
        register_fingerprint_fig(
            f"logit_lens_entropy_{run_name}",
            fig,
            f"Logit Lens Entropy — {run_name}",
        )

    # ------------------------------------------------------------------
    # Chart 5: Steering alpha sweep (per schedule)
    # ------------------------------------------------------------------
    for analysis in analyses:
        run_name = analysis["run_name"]
        st = analysis.get("steering", {})
        schedules = sorted(st.keys(), key=sched_order)
        if not schedules:
            continue

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[
                "Burst Acc on Other Prompts (↑ = steering works)",
                "Other Acc on Other Prompts (↓ = computation corrupted)",
                "Burst Acc on Burst Prompts (baseline)",
            ],
        )

        for sched in schedules:
            d = st[sched]
            c = color(sched)
            fig.add_trace(
                go.Scatter(
                    x=d["alphas"],
                    y=d["mean_burst_acc_on_other"],
                    name=sched,
                    line={"color": c, "width": 2},
                    mode="lines+markers",
                    legendgroup=sched,
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=d["alphas"],
                    y=d["mean_other_acc_on_other"],
                    name=sched,
                    line={"color": c, "width": 2, "dash": "dash"},
                    mode="lines+markers",
                    legendgroup=sched,
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=d["alphas"],
                    y=d["mean_burst_acc_on_burst"],
                    name=sched,
                    line={"color": c, "width": 2, "dash": "dot"},
                    mode="lines+markers",
                    legendgroup=sched,
                    showlegend=False,
                ),
                row=1,
                col=3,
            )

        fig.update_xaxes(title_text="alpha (steering strength)", type="log")
        fig.update_layout(
            title=(
                f"Activation Steering: alpha Sweep at Layer {st[schedules[0]]['steer_layer']} "
                f"— {run_name}"
            ),
            template="plotly_white",
            height=500,
            legend_title="Schedule",
        )
        register_fingerprint_fig(
            f"steering_alpha_sweep_{run_name}",
            fig,
            f"Steering alpha Sweep -- {run_name}",
        )

    # ------------------------------------------------------------------
    # Chart 6: Steering layer sweep
    # ------------------------------------------------------------------
    for analysis in analyses:
        run_name = analysis["run_name"]
        sls = analysis.get("steering_layer_sweep", {})
        schedules = sorted(sls.keys(), key=sched_order)
        if not schedules:
            continue

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[
                "Burst Acc on Other (↑ = layer is steerable)",
                "Other Acc on Other (↓ = computation disrupted)",
            ],
        )
        for sched in schedules:
            d = sls[sched]
            c = color(sched)
            layer_labels = [f"L{k}" for k in d["layers"]]
            fig.add_trace(
                go.Scatter(
                    x=layer_labels,
                    y=d["mean_burst_acc"],
                    name=sched,
                    line={"color": c, "width": 2},
                    mode="lines+markers",
                    legendgroup=sched,
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=layer_labels,
                    y=d["mean_other_acc"],
                    name=sched,
                    line={"color": c, "width": 2, "dash": "dash"},
                    mode="lines+markers",
                    legendgroup=sched,
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

        alpha_used = sls[schedules[0]]["alpha"]
        fig.update_layout(
            title=f"Steering Layer Sweep (alpha={alpha_used}) -- {run_name}",
            template="plotly_white",
            height=500,
            legend_title="Schedule",
        )
        fig.update_xaxes(title_text="Layer")
        register_fingerprint_fig(
            f"steering_layer_sweep_{run_name}",
            fig,
            f"Steering Layer Sweep — {run_name}",
        )

    # ------------------------------------------------------------------
    # Chart 7: Summary — readability + steering vs burstiness
    # ------------------------------------------------------------------
    for analysis in analyses:
        run_name = analysis["run_name"]
        ll = analysis.get("logit_lens", {})
        st = analysis.get("steering", {})
        schedules = sorted(set(ll.keys()) | set(st.keys()), key=sched_order)
        if not schedules:
            continue

        burst_pcts = [int(s.replace("burst_", "")) for s in schedules]
        colors = [color(s) for s in schedules]

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[
                "Logit Lens Readability (with LN)",
                "Logit Lens Mean Rank (with LN)",
                "Max Steering Effect (burst acc on other)",
            ],
        )

        readabilities = [
            ll.get(s, {}).get("with_ln", {}).get("overall_readability", float("nan"))
            for s in schedules
        ]
        ranks = [
            ll.get(s, {}).get("with_ln", {}).get("overall_mean_rank", float("nan"))
            for s in schedules
        ]
        max_steers = []
        for s in schedules:
            sd = st.get(s, {})
            if sd and sd.get("mean_burst_acc_on_other"):
                max_steers.append(max(sd["mean_burst_acc_on_other"]))
            else:
                max_steers.append(float("nan"))

        fig.add_trace(
            go.Scatter(
                x=burst_pcts,
                y=readabilities,
                mode="markers+lines",
                marker={"color": colors, "size": 10},
                line={"color": "gray", "width": 1, "dash": "dot"},
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=burst_pcts,
                y=ranks,
                mode="markers+lines",
                marker={"color": colors, "size": 10},
                line={"color": "gray", "width": 1, "dash": "dot"},
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=burst_pcts,
                y=max_steers,
                mode="markers+lines",
                marker={"color": colors, "size": 10},
                line={"color": "gray", "width": 1, "dash": "dot"},
                showlegend=False,
            ),
            row=1,
            col=3,
        )

        fig.update_xaxes(title_text="Burst %")
        fig.update_layout(
            title=f"Fingerprint Strength vs Burstiness — {run_name}",
            template="plotly_white",
            height=500,
        )
        register_fingerprint_fig(
            f"summary_fingerprint_{run_name}",
            fig,
            f"Fingerprint Summary — {run_name}",
        )

    # ------------------------------------------------------------------
    # Chart 8: Delta norm per layer (shows where the fingerprint lives)
    # ------------------------------------------------------------------
    for analysis in analyses:
        run_name = analysis["run_name"]
        sls = analysis.get("steering_layer_sweep", {})
        schedules = sorted(sls.keys(), key=sched_order)
        if not schedules:
            continue

        fig = go.Figure()
        for sched in schedules:
            d = sls[sched]
            layer_labels = [f"L{k}" for k in d["layers"]]
            fig.add_trace(
                go.Bar(
                    x=layer_labels,
                    y=d["mean_delta_norms"],
                    name=sched,
                    marker_color=color(sched),
                )
            )
        fig.update_layout(
            title=f"||δ̄|| per Layer — {run_name}",
            xaxis_title="Layer",
            yaxis_title="||δ̄|| (activation difference magnitude)",
            template="plotly_white",
            height=500,
            barmode="group",
        )
        register_fingerprint_fig(
            f"delta_norm_per_layer_{run_name}",
            fig,
            f"Delta Norm per Layer — {run_name}",
        )

    # ------------------------------------------------------------------
    # HTML dashboard
    # ------------------------------------------------------------------
    html_parts = [
        """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Finetuning Fingerprint Analysis</title>
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f0f2f5; }
  h1 { color: #1a1a2e; font-size: 1.8em; }
  h2 { color: #16213e; margin-top: 40px; font-size: 1.3em; }
  .chart-container {
    background: white; border-radius: 10px; padding: 20px;
    margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  }
  .metric-info {
    background: #f8f9ff; border-left: 4px solid #4a90d9;
    padding: 12px 16px; margin: 8px 0 16px 0;
    border-radius: 0 6px 6px 0; font-size: 0.9em; color: #333;
  }
  .metric-info .what { margin-bottom: 8px; }
  .metric-info .interp { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 8px; }
  .metric-info .interp span { flex: 1; min-width: 200px; }
  .metric-info .high { color: #1a7a4a; }
  .metric-info .low { color: #c0392b; }
  .metric-info .formula {
    font-family: monospace;
    background: #eef;
    padding: 4px 8px;
    border-radius: 4px;
  }
  .toc { background: white; border-radius: 10px; padding: 20px; margin: 20px 0;
         box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
  .toc a { display: block; margin: 4px 0; color: #1565c0; text-decoration: none; }
  .toc a:hover { text-decoration: underline; }
  .explainer {
    background: #fffde7; border-left: 4px solid #fbc02d;
    padding: 14px 18px; margin: 16px 0; border-radius: 0 8px 8px 0;
    font-size: 0.92em; line-height: 1.6;
  }
  .explainer strong { color: #e65100; }
</style>
</head>
<body>
<h1>Finetuning Fingerprint Analysis</h1>
<div class="explainer">
  <strong>What is this?</strong> We check whether burst-phase training leaves a readable
  "fingerprint" in the model's activations — even on non-burst data.<br><br>
  <strong>Logit Lens on δ̄:</strong> We compute δ̄ = E[h<sup>post</sup> - h<sup>pre</sup>] on
  other-class inputs, project it through the unembedding matrix W<sub>U</sub>, and check if
  the top tokens are burst-relevant. If yes → the burst knowledge is a global bias (wrapper).
  If no → it's stored in conditional circuits (deep).<br><br>
  <strong>Activation Steering:</strong> We add alpha*d_bar to the residual stream during generation
  on other-class prompts. If the model starts producing burst outputs → the knowledge is an
  additive direction. If not → it's conditional.<br><br>
  <strong>Key insight:</strong> Higher burstiness (burst_100) should show stronger fingerprints
  and more effective steering, because the model learns a shallow wrapper. Lower burstiness
  (burst_25) should show weaker fingerprints, because the model integrates burst knowledge
  more deeply.
</div>
<div class="toc">
  <strong>Contents:</strong>
"""
    ]

    for i, (_key, title, _) in enumerate(all_figs):
        anchor = f"chart_{i}"
        html_parts.append(f'  <a href="#{anchor}">{i + 1}. {title}</a>\n')

    html_parts.append("</div>\n")

    for i, (key, title, fig) in enumerate(all_figs):
        desc = _METRIC_DESCRIPTIONS.get(
            key.split("_" + analyses[0]["run_name"])[0] if analyses else key, {}
        )
        anchor = f"chart_{i}"
        html_parts.append(f'<div class="chart-container" id="{anchor}">\n')
        html_parts.append(f"<h2>{i + 1}. {title}</h2>\n")

        if desc:
            html_parts.append('<div class="metric-info">\n')
            if desc.get("what"):
                html_parts.append(
                    f'<div class="what"><strong>What:</strong> {desc["what"]}</div>\n'
                )
            if desc.get("high") or desc.get("low"):
                html_parts.append('<div class="interp">\n')
                if desc.get("high"):
                    html_parts.append(
                        f'<span class="high"><strong>High →</strong> {desc["high"]}</span>\n'
                    )
                if desc.get("low"):
                    html_parts.append(
                        f'<span class="low"><strong>Low →</strong> {desc["low"]}</span>\n'
                    )
                html_parts.append("</div>\n")
            if desc.get("formula"):
                html_parts.append(f'<div class="formula">{desc["formula"]}</div>\n')
            html_parts.append("</div>\n")

        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=(i == 0)))
        html_parts.append("</div>\n")

    html_parts.append("</body></html>")

    html_path = out_dir / "dashboard.html"
    with html_path.open("w") as f:
        f.write("".join(html_parts))
    logger.info("Dashboard saved: %s", html_path)
    logger.info("Charts saved: %s", charts_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def is_valid_run_dir(d: Path) -> bool:
    """Check if *d* contains results and checkpoints."""
    has_results = (d / "all_results.pkl").exists() or (d / "logs" / "all_results.pkl").exists()
    ckpt_dir = d / "checkpoints" if (d / "checkpoints").exists() else d / "logs" / "checkpoints"
    has_ckpts = ckpt_dir.exists() and any(ckpt_dir.iterdir())
    return has_results and has_ckpts


def find_all_run_dirs(data_root: Path) -> list[Path]:
    """Return all valid burst run directories under *data_root*."""
    candidates = sorted(p for p in data_root.iterdir() if p.is_dir() and ("burst_" in p.name))
    return [p for p in candidates if is_valid_run_dir(p)]


def main() -> None:
    """Run fingerprint analysis from the command line."""
    parser = argparse.ArgumentParser(
        description="Finetuning fingerprint analysis: Logit Lens + Activation Steering.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dirs", nargs="*", type=Path)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    if args.all:
        run_dirs = find_all_run_dirs(args.data_root)
        if not run_dirs:
            logger.info("No valid run directories found under %s", args.data_root)
            return
        logger.info("Found %d valid run directories", len(run_dirs))
        for d in run_dirs:
            logger.info("  %s", d)
    elif args.run_dirs:
        run_dirs = [Path(d) for d in args.run_dirs]
    else:
        parser.error("Provide run_dirs or use --all")

    out_dir = args.out_dir or Path("data/fingerprint_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    analyses = []
    for run_dir in run_dirs:
        t0 = time.time()
        analysis = analyse_run(
            run_dir,
            n_seeds=args.n_seeds,
            n_samples=args.n_samples,
            top_k=args.top_k,
        )
        analyses.append(analysis)
        logger.info("Completed %s in %.1fs", run_dir.name, time.time() - t0)

    results_path = out_dir / "results.pkl"
    with results_path.open("wb") as f:
        pickle.dump(analyses, f)
    logger.info("Results saved: %s", results_path)

    logger.info("Generating dashboard...")
    make_dashboard(analyses, out_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
