"""Unified burstiness analysis dashboard.

Merges deep_analysis (5 metrics) and new_metrics (10 metrics) into one
script, adds Frankenstein layer-swap analysis, evaluates on both burst
and other-class docs, shows individual seed points + error bars, fixes
LMC/EMA limitations, and compares pre-burst vs post-burst models.

Usage:
    uv run python burst/unified_analysis.py \\
        data/burst_d3_pos1_<tag> data/burst_d3_pos2_<tag> data/burst_d3_pos3_<tag> \\
        --deep-results data/deep_analysis_combined/results.pkl \\
        --new-results data/new_metrics_combined/results.pkl \\
        --out-dir data/unified_analysis \\
        --n-seeds 10

Dimension key:
    B: batch_size
    L: sequence_length (doc_len)
    N: n_embd (model dimension)
    V: vocab_size
"""
import sys, os, argparse, pickle, json, time, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from omegaconf import OmegaConf

from net.nanogpt import nanoGPT
from net.runner import configure_optimizers, update_cosine_warmup_lr
from burst.config import (
    PHASE_BURST, PHASE_REVERSION, SCHEDULE_ORDER, SCHED_COLORS,
    parse_run_config,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCHEDULES_ORDERED = SCHEDULE_ORDER
SCHEDULE_COLORS = SCHED_COLORS
N_LAYERS = 6


def _load_net(cfg: dict, ckpt_path: str) -> nanoGPT:
    net = nanoGPT(OmegaConf.create({
        "compile": False, "vocab_size": cfg["vocab_size"],
        "context_size": cfg["context_size"],
        "n_layer": cfg["n_layer"], "n_head": cfg["n_head"],
        "n_embd": cfg["n_embd"], "dropout": 0.0, "bias": False, "mlp": True,
    })).to(DEVICE)
    net.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    return net


@torch.no_grad()
def _free_gen_acc(net: nanoGPT, docs_BL: np.ndarray, prompt_len: int) -> float:
    net.eval()
    docs_t = torch.as_tensor(docs_BL, dtype=torch.long, device=DEVICE)
    B, L = docs_t.shape
    generated = docs_t[:, :prompt_len].clone()
    target_B6 = docs_t[:, -6:]
    for _ in range(L - prompt_len):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            logits_BTV = net(generated)
        next_tok = logits_BTV[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_tok], dim=1)
    return (generated[:, -6:] == target_B6).all(dim=1).float().mean().item()


@torch.no_grad()
def _cross_entropy_loss(net: nanoGPT, docs_BL: np.ndarray) -> float:
    net.eval()
    docs_t = torch.as_tensor(docs_BL, dtype=torch.long, device=DEVICE)
    inp, tgt = docs_t[:, :-1], docs_t[:, 1:]
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        logits = net(inp).float()
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)).item()


def _sched_order(s: str) -> int:
    try:
        return SCHEDULES_ORDERED.index(s)
    except ValueError:
        return 99


def _color(s: str) -> str:
    return SCHEDULE_COLORS.get(s, "#888888")


def _ckpt_files(ckpt_dir: Path) -> dict[int, Path]:
    return {int(p.stem.split("_")[1]): p for p in ckpt_dir.glob("step_*.pt")}


def _get_key_steps(files: dict[int, Path], cfg: dict):
    available = sorted(files.keys())
    T = cfg["total_steps"]
    pre_step = available[0]
    peak_step = min(available, key=lambda x: abs(x - (T - 1)))
    rev_step = max(available)
    return pre_step, peak_step, rev_step


def _subsample_docs(docs_BL: np.ndarray, n: int = 256) -> np.ndarray:
    if docs_BL.shape[0] <= n:
        return docs_BL
    idx = np.random.choice(docs_BL.shape[0], n, replace=False)
    return docs_BL[idx]


# ---------------------------------------------------------------------------
# Frankenstein layer-swap
# ---------------------------------------------------------------------------

def _build_hybrid_sd(
    sd_bottom: dict[str, torch.Tensor],
    sd_top: dict[str, torch.Tensor],
    cut_after_block: int,
) -> dict[str, torch.Tensor]:
    """Build a hybrid state_dict: blocks [0..cut_after_block] from sd_bottom,
    blocks [cut_after_block+1..5] + ln_f from sd_top.
    Embeddings always from sd_bottom. LM_head tied to wte (from bottom)."""
    hybrid = {}
    for key in sd_bottom:
        if key.startswith("transformer.wte.") or key.startswith("transformer.wpe.") or key.startswith("transformer.drop."):
            hybrid[key] = sd_bottom[key]
        elif key.startswith("transformer.h."):
            block_idx = int(key.split(".")[2])
            if block_idx <= cut_after_block:
                hybrid[key] = sd_bottom[key]
            else:
                hybrid[key] = sd_top[key]
        elif key.startswith("transformer.ln_f."):
            hybrid[key] = sd_top[key]
        elif key.startswith("LM_head."):
            hybrid[key] = sd_bottom[key]
        else:
            hybrid[key] = sd_bottom[key]
    return hybrid


@torch.no_grad()
def compute_frankenstein(
    ckpt_root: Path,
    all_results: list[dict],
    burst_docs_BL: np.ndarray,
    other_docs_BL: np.ndarray,
    prompt_len: int,
    n_seeds: int = 10,
) -> dict:
    """Frankenstein layer-swap: pre-burst vs post-burst.

    For each cut point k in [-1, 0, 1, 2, 3, 4, 5]:
      - k=-1: all layers from model B (post-burst), embeddings from A (pre-burst)
      - k=5: all layers from model A (pre-burst), only ln_f from B

    Two directions:
      - "pre_bottom": bottom layers from pre-burst, top from post-burst
      - "post_bottom": bottom layers from post-burst, top from pre-burst
    """
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)
    cut_points = list(range(-1, N_LAYERS))
    burst_sub = _subsample_docs(burst_docs_BL)
    other_sub = _subsample_docs(other_docs_BL)
    results = {}

    for sched in schedules:
        sched_results = jobs_by_schedule[sched]
        per_seed: list[dict] = []
        seeds_done = 0

        for r in sched_results:
            if seeds_done >= n_seeds:
                break
            label = r["label"]
            ckpt_dir = ckpt_root / label
            if not ckpt_dir.exists():
                continue
            files = _ckpt_files(ckpt_dir)
            if not files:
                continue

            cfg = r["config"]
            pre_step, peak_step, _ = _get_key_steps(files, cfg)

            sd_pre = torch.load(str(files[pre_step]), map_location=DEVICE, weights_only=True)
            sd_post = torch.load(str(files[peak_step]), map_location=DEVICE, weights_only=True)

            net = _load_net(cfg, str(files[pre_step]))
            seed_data = {"pre_bottom_burst": [], "pre_bottom_other": [],
                         "post_bottom_burst": [], "post_bottom_other": []}

            for k in cut_points:
                hybrid_pre_bot = _build_hybrid_sd(sd_pre, sd_post, k)
                net.load_state_dict(hybrid_pre_bot)
                seed_data["pre_bottom_burst"].append(_free_gen_acc(net, burst_sub, prompt_len))
                seed_data["pre_bottom_other"].append(_free_gen_acc(net, other_sub, prompt_len))

                hybrid_post_bot = _build_hybrid_sd(sd_post, sd_pre, k)
                net.load_state_dict(hybrid_post_bot)
                seed_data["post_bottom_burst"].append(_free_gen_acc(net, burst_sub, prompt_len))
                seed_data["post_bottom_other"].append(_free_gen_acc(net, other_sub, prompt_len))

            per_seed.append(seed_data)
            seeds_done += 1
            print(f"  {label}: frankenstein done", flush=True)

        results[sched] = {"cut_points": cut_points, "per_seed": per_seed}

    return results


@torch.no_grad()
def compute_cross_burst_frankenstein(
    ckpt_root: Path,
    all_results: list[dict],
    burst_docs_BL: np.ndarray,
    other_docs_BL: np.ndarray,
    prompt_len: int,
    n_seeds: int = 3,
    schedule_pairs: list[tuple[str, str]] = None,
) -> dict:
    """Cross-burst Frankenstein: swap layers between post-burst models of
    different schedules to see if they store the capability in the same layers."""
    if schedule_pairs is None:
        schedule_pairs = [
            ("burst_100", "burst_10"),
            ("burst_100", "burst_50"),
            ("burst_50", "burst_10"),
        ]

    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    cut_points = list(range(-1, N_LAYERS))
    burst_sub = _subsample_docs(burst_docs_BL)
    other_sub = _subsample_docs(other_docs_BL)
    results = {}

    for sched_a, sched_b in schedule_pairs:
        if sched_a not in jobs_by_schedule or sched_b not in jobs_by_schedule:
            continue
        jobs_a = jobs_by_schedule[sched_a]
        jobs_b = jobs_by_schedule[sched_b]

        per_seed: list[dict] = []
        seeds_done = 0

        for r_a, r_b in zip(jobs_a, jobs_b):
            if seeds_done >= n_seeds:
                break
            dir_a = ckpt_root / r_a["label"]
            dir_b = ckpt_root / r_b["label"]
            if not dir_a.exists() or not dir_b.exists():
                continue
            files_a = _ckpt_files(dir_a)
            files_b = _ckpt_files(dir_b)
            if not files_a or not files_b:
                continue

            cfg = r_a["config"]
            _, peak_a, _ = _get_key_steps(files_a, cfg)
            _, peak_b, _ = _get_key_steps(files_b, cfg)

            sd_a = torch.load(str(files_a[peak_a]), map_location=DEVICE, weights_only=True)
            sd_b = torch.load(str(files_b[peak_b]), map_location=DEVICE, weights_only=True)

            net = _load_net(cfg, str(files_a[peak_a]))
            seed_data = {"a_bottom_burst": [], "a_bottom_other": [],
                         "b_bottom_burst": [], "b_bottom_other": []}

            for k in cut_points:
                hybrid_a_bot = _build_hybrid_sd(sd_a, sd_b, k)
                net.load_state_dict(hybrid_a_bot)
                seed_data["a_bottom_burst"].append(_free_gen_acc(net, burst_sub, prompt_len))
                seed_data["a_bottom_other"].append(_free_gen_acc(net, other_sub, prompt_len))

                hybrid_b_bot = _build_hybrid_sd(sd_b, sd_a, k)
                net.load_state_dict(hybrid_b_bot)
                seed_data["b_bottom_burst"].append(_free_gen_acc(net, burst_sub, prompt_len))
                seed_data["b_bottom_other"].append(_free_gen_acc(net, other_sub, prompt_len))

            per_seed.append(seed_data)
            seeds_done += 1
            print(f"  {r_a['label']} x {r_b['label']}: cross-frankenstein done", flush=True)

        pair_key = f"{sched_a}_x_{sched_b}"
        results[pair_key] = {
            "sched_a": sched_a, "sched_b": sched_b,
            "cut_points": cut_points, "per_seed": per_seed,
        }

    return results


# ---------------------------------------------------------------------------
# LMC (fixed: dual-class, pre-burst baseline)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_lmc_dual(
    ckpt_root: Path,
    all_results: list[dict],
    burst_docs_BL: np.ndarray,
    other_docs_BL: np.ndarray,
    prompt_len: int,
    n_seeds: int = 10,
    n_alphas: int = 11,
) -> dict:
    """LMC with dual-class evaluation and pre-burst baseline.

    Computes two interpolation paths:
      1. pre_burst <-> post_burst (primary)
      2. post_burst <-> post_reversion (secondary)
    Evaluates loss on both burst and other docs at each alpha.
    """
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)
    alphas = np.linspace(0, 1, n_alphas).tolist()
    burst_sub = _subsample_docs(burst_docs_BL)
    other_sub = _subsample_docs(other_docs_BL)
    burst_t = torch.as_tensor(burst_sub, dtype=torch.long, device=DEVICE)
    other_t = torch.as_tensor(other_sub, dtype=torch.long, device=DEVICE)
    burst_inp, burst_tgt = burst_t[:, :-1], burst_t[:, 1:]
    other_inp, other_tgt = other_t[:, :-1], other_t[:, 1:]

    results = {}

    for sched in schedules:
        sched_results = jobs_by_schedule[sched]
        per_seed: list[dict] = []
        seeds_done = 0

        for r in sched_results:
            if seeds_done >= n_seeds:
                break
            label = r["label"]
            ckpt_dir = ckpt_root / label
            if not ckpt_dir.exists():
                continue
            files = _ckpt_files(ckpt_dir)
            if not files:
                continue

            cfg = r["config"]
            pre_step, peak_step, rev_step = _get_key_steps(files, cfg)

            sd_pre = {k: v.float() for k, v in torch.load(
                str(files[pre_step]), map_location="cpu", weights_only=True).items()}
            sd_peak = {k: v.float() for k, v in torch.load(
                str(files[peak_step]), map_location="cpu", weights_only=True).items()}
            sd_rev = {k: v.float() for k, v in torch.load(
                str(files[rev_step]), map_location="cpu", weights_only=True).items()}

            net = _load_net(cfg, str(files[pre_step]))

            def _eval_interp(sd_a, sd_b):
                burst_losses, other_losses = [], []
                for alpha in alphas:
                    interp = {k: (1 - alpha) * sd_a[k] + alpha * sd_b[k] for k in sd_a}
                    net.load_state_dict(interp)
                    net.eval()
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                        bl = F.cross_entropy(net(burst_inp).float().reshape(-1, cfg["vocab_size"]),
                                             burst_tgt.reshape(-1)).item()
                        ol = F.cross_entropy(net(other_inp).float().reshape(-1, cfg["vocab_size"]),
                                             other_tgt.reshape(-1)).item()
                    burst_losses.append(bl)
                    other_losses.append(ol)
                return burst_losses, other_losses

            pre_peak_burst, pre_peak_other = _eval_interp(sd_pre, sd_peak)
            peak_rev_burst, peak_rev_other = _eval_interp(sd_peak, sd_rev)

            def _barrier(curve):
                ep = (curve[0] + curve[-1]) / 2
                return max(curve) - ep

            per_seed.append({
                "pre_peak_burst": pre_peak_burst,
                "pre_peak_other": pre_peak_other,
                "peak_rev_burst": peak_rev_burst,
                "peak_rev_other": peak_rev_other,
                "barrier_pre_peak_burst": _barrier(pre_peak_burst),
                "barrier_pre_peak_other": _barrier(pre_peak_other),
                "barrier_peak_rev_burst": _barrier(peak_rev_burst),
                "barrier_peak_rev_other": _barrier(peak_rev_other),
            })
            seeds_done += 1
            print(f"  {label}: LMC barriers pre↔peak burst={per_seed[-1]['barrier_pre_peak_burst']:.4f}, "
                  f"peak↔rev burst={per_seed[-1]['barrier_peak_rev_burst']:.4f}", flush=True)

        results[sched] = {"alphas": alphas, "per_seed": per_seed}

    return results


# ---------------------------------------------------------------------------
# EMA interpolation (fixed: dual-class, pre-burst baseline)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_ema_dual(
    ckpt_root: Path,
    all_results: list[dict],
    burst_docs_BL: np.ndarray,
    other_docs_BL: np.ndarray,
    prompt_len: int,
    n_seeds: int = 10,
) -> dict:
    """EMA interpolation with dual-class evaluation.

    Two paths:
      1. pre_burst <-> post_burst: alpha=0 is pre, alpha=1 is peak
      2. post_burst <-> post_reversion: alpha=0 is reverted, alpha=1 is peak
    """
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)
    burst_sub = _subsample_docs(burst_docs_BL, n=128)
    other_sub = _subsample_docs(other_docs_BL, n=128)
    results = {}

    for sched in schedules:
        sched_results = jobs_by_schedule[sched]
        per_seed: list[dict] = []
        seeds_done = 0

        for r in sched_results:
            if seeds_done >= n_seeds:
                break
            label = r["label"]
            ckpt_dir = ckpt_root / label
            if not ckpt_dir.exists():
                continue
            files = _ckpt_files(ckpt_dir)
            if not files:
                continue

            cfg = r["config"]
            pre_step, peak_step, rev_step = _get_key_steps(files, cfg)

            sd_pre = {k: v.float() for k, v in torch.load(
                str(files[pre_step]), map_location="cpu", weights_only=True).items()}
            sd_peak = {k: v.float() for k, v in torch.load(
                str(files[peak_step]), map_location="cpu", weights_only=True).items()}
            sd_rev = {k: v.float() for k, v in torch.load(
                str(files[rev_step]), map_location="cpu", weights_only=True).items()}

            net = _load_net(cfg, str(files[pre_step]))

            def _eval_path(sd_a, sd_b):
                burst_accs, other_accs = [], []
                for alpha in alphas:
                    interp = {k: (1 - alpha) * sd_a[k] + alpha * sd_b[k] for k in sd_a}
                    net.load_state_dict(interp)
                    burst_accs.append(_free_gen_acc(net, burst_sub, prompt_len))
                    other_accs.append(_free_gen_acc(net, other_sub, prompt_len))
                return burst_accs, other_accs

            pre_peak_burst, pre_peak_other = _eval_path(sd_pre, sd_peak)
            rev_peak_burst, rev_peak_other = _eval_path(sd_rev, sd_peak)

            def _cliff(accs):
                return next((a for a, acc in zip(alphas, accs) if acc > 0.5), 1.0)

            per_seed.append({
                "pre_peak_burst": pre_peak_burst,
                "pre_peak_other": pre_peak_other,
                "rev_peak_burst": rev_peak_burst,
                "rev_peak_other": rev_peak_other,
                "cliff_pre_peak_burst": _cliff(pre_peak_burst),
                "cliff_rev_peak_burst": _cliff(rev_peak_burst),
            })
            seeds_done += 1
            print(f"  {label}: EMA cliff pre↔peak={per_seed[-1]['cliff_pre_peak_burst']:.2f}, "
                  f"rev↔peak={per_seed[-1]['cliff_rev_peak_burst']:.2f}", flush=True)

        results[sched] = {"alphas": alphas, "per_seed": per_seed}

    return results


# ---------------------------------------------------------------------------
# Pruning robustness (dual-class)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_pruning_dual(
    ckpt_root: Path,
    all_results: list[dict],
    burst_docs_BL: np.ndarray,
    other_docs_BL: np.ndarray,
    prompt_len: int,
    n_seeds: int = 10,
    n_prune_levels: int = 10,
) -> dict:
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)
    sparsities = np.linspace(0, 0.9, n_prune_levels).tolist()
    burst_sub = _subsample_docs(burst_docs_BL)
    other_sub = _subsample_docs(other_docs_BL)
    results = {}

    for sched in schedules:
        sched_results = jobs_by_schedule[sched]
        per_seed: list[dict] = []
        seeds_done = 0

        for r in sched_results:
            if seeds_done >= n_seeds:
                break
            label = r["label"]
            ckpt_dir = ckpt_root / label
            if not ckpt_dir.exists():
                continue
            files = _ckpt_files(ckpt_dir)
            if not files:
                continue

            cfg = r["config"]
            _, peak_step, _ = _get_key_steps(files, cfg)

            sd_orig = torch.load(str(files[peak_step]), map_location=DEVICE, weights_only=True)
            net = _load_net(cfg, str(files[peak_step]))
            all_w = torch.cat([v.view(-1).abs() for v in sd_orig.values()])

            burst_accs, other_accs = [], []
            for sparsity in sparsities:
                if sparsity > 0:
                    threshold = torch.quantile(all_w, sparsity)
                    pruned = {k: v * (v.abs() >= threshold).to(v.dtype) for k, v in sd_orig.items()}
                    net.load_state_dict(pruned)
                else:
                    net.load_state_dict(sd_orig)
                burst_accs.append(_free_gen_acc(net, burst_sub, prompt_len))
                other_accs.append(_free_gen_acc(net, other_sub, prompt_len))

            per_seed.append({"burst_accs": burst_accs, "other_accs": other_accs})
            seeds_done += 1
            print(f"  {label}: pruning burst@0%={burst_accs[0]:.3f}, other@0%={other_accs[0]:.3f}", flush=True)

        results[sched] = {"sparsities": sparsities, "per_seed": per_seed}

    return results


# ---------------------------------------------------------------------------
# Task vector transfer (dual-class)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_transfer_dual(
    ckpt_root: Path,
    all_results: list[dict],
    burst_docs_BL: np.ndarray,
    other_docs_BL: np.ndarray,
    prompt_len: int,
    n_seeds: int = 10,
) -> dict:
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)
    burst_sub = _subsample_docs(burst_docs_BL)
    other_sub = _subsample_docs(other_docs_BL)
    results = {}

    for sched in schedules:
        sched_results = jobs_by_schedule[sched]
        per_seed: list[dict] = []
        seeds_done = 0

        for i, r_src in enumerate(sched_results):
            if seeds_done >= n_seeds:
                break
            label_src = r_src["label"]
            ckpt_dir_src = ckpt_root / label_src
            if not ckpt_dir_src.exists():
                continue
            files_src = _ckpt_files(ckpt_dir_src)
            if not files_src:
                continue

            cfg = r_src["config"]
            pre_step_src, peak_step_src, _ = _get_key_steps(files_src, cfg)

            r_tgt = next(
                (r for j, r in enumerate(sched_results) if j != i
                 and (ckpt_root / r["label"]).exists()), None)
            if r_tgt is None:
                continue

            files_tgt = _ckpt_files(ckpt_root / r_tgt["label"])
            if not files_tgt:
                continue
            pre_step_tgt = min(files_tgt.keys())

            peak_sd = {k: v.float() for k, v in torch.load(
                str(files_src[peak_step_src]), map_location="cpu", weights_only=True).items()}
            pre_sd = {k: v.float() for k, v in torch.load(
                str(files_src[pre_step_src]), map_location="cpu", weights_only=True).items()}
            tgt_sd = {k: v.float() for k, v in torch.load(
                str(files_tgt[pre_step_tgt]), map_location="cpu", weights_only=True).items()}

            tau = {k: peak_sd[k] - pre_sd[k] for k in peak_sd}
            transferred = {k: tgt_sd[k] + tau[k] for k in tgt_sd}

            net = _load_net(cfg, str(files_tgt[pre_step_tgt]))
            net.load_state_dict(transferred)
            burst_acc = _free_gen_acc(net, burst_sub, prompt_len)
            other_acc = _free_gen_acc(net, other_sub, prompt_len)

            per_seed.append({"burst_acc": burst_acc, "other_acc": other_acc})
            seeds_done += 1
            print(f"  {label_src} → {r_tgt['label']}: burst={burst_acc:.3f}, other={other_acc:.3f}", flush=True)

        results[sched] = {"per_seed": per_seed}

    return results


# ---------------------------------------------------------------------------
# Relearning efficiency (dual-class)
# ---------------------------------------------------------------------------

def compute_relearning_dual(
    ckpt_root: Path,
    all_results: list[dict],
    burst_docs_BL: np.ndarray,
    other_docs_BL: np.ndarray,
    prompt_len: int,
    n_seeds: int = 10,
    relearn_steps: int = 50,
) -> dict:
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)
    burst_sub = _subsample_docs(burst_docs_BL)
    other_sub = _subsample_docs(other_docs_BL)
    results = {}

    for sched in schedules:
        sched_results = jobs_by_schedule[sched]
        per_seed: list[dict] = []
        seeds_done = 0

        for r in sched_results:
            if seeds_done >= n_seeds:
                break
            label = r["label"]
            ckpt_dir = ckpt_root / label
            if not ckpt_dir.exists():
                continue
            files = _ckpt_files(ckpt_dir)
            if not files:
                continue

            cfg = r["config"]
            _, _, rev_step = _get_key_steps(files, cfg)
            net = _load_net(cfg, str(files[rev_step]))

            optim_cfg = OmegaConf.create({
                "learning_rate": cfg["lr"] * 0.3,
                "weight_decay": cfg["weight_decay"],
                "beta1": cfg["beta1"], "beta2": cfg["beta2"],
                "grad_clip": cfg["grad_clip"], "decay_lr": False,
                "warmup_iters": 0, "min_lr": cfg["lr"] * 0.3,
            })
            optimizer = configure_optimizers(net, optim_cfg)
            scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")

            n = min(256, burst_docs_BL.shape[0])
            docs_fine = burst_docs_BL[np.random.choice(burst_docs_BL.shape[0], n, replace=False)]

            burst_accs, other_accs, steps_log = [], [], []
            net.train()
            it = 0
            for step in range(relearn_steps):
                batch_idx = np.random.choice(n, min(cfg["batch_size"], n), replace=True)
                batch = torch.as_tensor(docs_fine[batch_idx], dtype=torch.long, device=DEVICE)
                inp, tgt = batch[:, :-1], batch[:, 1:]
                it, _ = update_cosine_warmup_lr(it, optim_cfg, optimizer, relearn_steps)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                    logits = net(inp)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()

                if step % 5 == 0 or step == relearn_steps - 1:
                    ba = _free_gen_acc(net, burst_sub, prompt_len)
                    oa = _free_gen_acc(net, other_sub, prompt_len)
                    burst_accs.append(ba)
                    other_accs.append(oa)
                    steps_log.append(step)

            per_seed.append({"steps": steps_log, "burst_accs": burst_accs, "other_accs": other_accs})
            seeds_done += 1
            print(f"  {label}: relearn burst={burst_accs[-1]:.3f}, other={other_accs[-1]:.3f}", flush=True)

        results[sched] = {"per_seed": per_seed}

    return results


# ---------------------------------------------------------------------------
# Forgetting trajectory dim (from new_metrics, keep per-seed)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_trajectory_dim(
    ckpt_root: Path,
    all_results: list[dict],
    n_seeds: int = 10,
    variance_threshold: float = 0.95,
    max_ckpts: int = 15,
) -> dict:
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)
    results = {}

    for sched in schedules:
        sched_results = jobs_by_schedule[sched]
        per_seed: list[int] = []
        seeds_done = 0

        for r in sched_results:
            if seeds_done >= n_seeds:
                break
            label = r["label"]
            ckpt_dir = ckpt_root / label
            if not ckpt_dir.exists():
                continue
            files = _ckpt_files(ckpt_dir)
            if not files:
                continue

            T = r["config"]["total_steps"]
            rev_steps_all = sorted(s for s in files if s >= T)
            if len(rev_steps_all) < 3:
                continue

            if len(rev_steps_all) > max_ckpts:
                indices = np.linspace(0, len(rev_steps_all) - 1, max_ckpts, dtype=int)
                rev_steps = [rev_steps_all[i] for i in indices]
            else:
                rev_steps = rev_steps_all

            weight_vecs = []
            for step in rev_steps:
                sd = torch.load(str(files[step]), map_location="cpu", weights_only=True)
                flat = torch.cat([v.float().view(-1) for v in sd.values()])
                weight_vecs.append(flat.numpy())

            W = np.stack(weight_vecs)
            W_centered = W - W.mean(axis=0, keepdims=True)
            try:
                _, sv, _ = np.linalg.svd(W_centered, full_matrices=False)
                var_explained = np.cumsum(sv ** 2) / (sv ** 2).sum()
                dim = int(np.searchsorted(var_explained, variance_threshold)) + 1
            except np.linalg.LinAlgError:
                dim = len(rev_steps)

            per_seed.append(dim)
            seeds_done += 1
            print(f"  {label}: trajectory_dim={dim}", flush=True)

        results[sched] = {"per_seed": per_seed}

    return results


# ---------------------------------------------------------------------------
# Data-only metrics (from all_results, keep per-seed)
# ---------------------------------------------------------------------------

def compute_forgetting_decomposition(all_results: list[dict]) -> dict:
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)
    results = {}

    for sched in schedules:
        per_seed: list[dict] = []
        for r in jobs_by_schedule[sched]:
            log = r["log"]
            steps = log["step"]
            accs = log["acc_burst"]
            phases = log["phase"]
            T = r["config"]["total_steps"]

            rev_steps = [s - T for s, p in zip(steps, phases) if p == PHASE_REVERSION]
            rev_accs = [a for a, p in zip(accs, phases) if p == PHASE_REVERSION]
            if len(rev_accs) < 2:
                continue

            early_mask = [s <= 50 for s in rev_steps]
            early_s = [s for s, m in zip(rev_steps, early_mask) if m]
            early_a = [a for a, m in zip(rev_accs, early_mask) if m]
            slope = float(np.polyfit(early_s, early_a, 1)[0]) if len(early_s) >= 2 else float("nan")

            cutoff = int(len(rev_accs) * 0.8)
            plateau = float(np.mean(rev_accs[cutoff:])) if rev_accs[cutoff:] else float("nan")

            per_seed.append({
                "initial_slope": slope,
                "plateau_acc": plateau,
                "reversion_auc": r.get("reversion_auc", float("nan")),
                "peak_burst": r.get("peak_burst", float("nan")),
            })

        results[sched] = {"per_seed": per_seed}

    return results


def compute_grad_temporal(all_results: list[dict]) -> dict:
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)
    results = {}

    for sched in schedules:
        step_sims: dict[int, list[float]] = {}
        for r in jobs_by_schedule[sched]:
            gsl = r.get("grad_sim_log", {})
            steps = gsl.get("step", [])
            sims = gsl.get("burst_vs_other", [])
            phases = gsl.get("phase", [])
            T = r["config"]["total_steps"]
            for s, sim, ph in zip(steps, sims, phases):
                if ph == PHASE_REVERSION:
                    step_sims.setdefault(s - T, []).append(sim)

        if not step_sims:
            results[sched] = {"steps": [], "mean_sims": []}
            continue

        steps_sorted = sorted(step_sims.keys())
        mean_sims = [float(np.mean(step_sims[s])) for s in steps_sorted]
        results[sched] = {"steps": steps_sorted, "mean_sims": mean_sims}

    return results


def compute_layer_interference(all_results: list[dict]) -> dict:
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)
    results = {}

    for sched in schedules:
        layer_sims: dict[str, list[float]] = {}
        layer_names = []
        for r in jobs_by_schedule[sched]:
            gsl = r.get("grad_sim_log", {})
            per_layer = gsl.get("per_layer", {})
            phases = gsl.get("phase", [])
            if not layer_names and gsl.get("layer_names"):
                layer_names = gsl["layer_names"]
            for ln, vals in per_layer.items():
                burst_vals = [v for v, p in zip(vals, phases) if p == PHASE_BURST]
                if burst_vals:
                    layer_sims.setdefault(ln, []).append(float(np.mean(burst_vals)))

        if not layer_sims:
            results[sched] = {}
            continue

        mean_per_layer = {ln: float(np.mean(vs)) for ln, vs in layer_sims.items()}
        results[sched] = {
            "mean_per_layer": mean_per_layer,
            "layer_names": layer_names or list(mean_per_layer.keys()),
        }

    return results


# ---------------------------------------------------------------------------
# Critical sharpness: global + per-layer Hutchinson trace of Hessian
# ---------------------------------------------------------------------------

LAYER_GROUPS = {
    "emb": ["transformer.wte.weight", "transformer.wpe.weight"],
}
for _bi in range(N_LAYERS):
    LAYER_GROUPS[f"block{_bi}.attn"] = [
        f"transformer.h.{_bi}.attn.c_attn.weight",
        f"transformer.h.{_bi}.attn.c_proj.weight",
    ]
    LAYER_GROUPS[f"block{_bi}.mlp"] = [
        f"transformer.h.{_bi}.mlp.c_fc.weight",
        f"transformer.h.{_bi}.mlp.c_proj.weight",
    ]
    LAYER_GROUPS[f"block{_bi}.ln"] = [
        f"transformer.h.{_bi}.ln_1.weight",
        f"transformer.h.{_bi}.ln_2.weight",
    ]
LAYER_GROUPS["ln_f"] = ["transformer.ln_f.weight"]


def compute_sharpness(
    ckpt_root: Path,
    all_results: list[dict],
    burst_docs_BL: np.ndarray,
    other_docs_BL: np.ndarray,
    n_seeds: int = 10,
    n_samples: int = 128,
    n_hutchinson: int = 15,
) -> dict:
    """Hutchinson trace of Hessian — global and per-layer group.

    For each seed: loads peak-burst checkpoint, computes Hessian trace
    on burst-class loss globally and restricted to each layer group.
    Also computes on other-class loss for comparison.
    """
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)
    burst_sub = _subsample_docs(burst_docs_BL, n=n_samples)
    other_sub = _subsample_docs(other_docs_BL, n=n_samples)
    burst_t = torch.as_tensor(burst_sub, dtype=torch.long, device=DEVICE)
    other_t = torch.as_tensor(other_sub, dtype=torch.long, device=DEVICE)
    burst_inp, burst_tgt = burst_t[:, :-1], burst_t[:, 1:]
    other_inp, other_tgt = other_t[:, :-1], other_t[:, 1:]

    results = {}
    layer_group_names = list(LAYER_GROUPS.keys())

    for sched in schedules:
        sched_results = jobs_by_schedule[sched]
        per_seed: list[dict] = []
        seeds_done = 0

        for r in sched_results:
            if seeds_done >= n_seeds:
                break
            label = r["label"]
            ckpt_dir = ckpt_root / label
            if not ckpt_dir.exists():
                continue
            files = _ckpt_files(ckpt_dir)
            if not files:
                continue

            cfg = r["config"]
            _, peak_step, _ = _get_key_steps(files, cfg)

            net = _load_net(cfg, str(files[peak_step]))
            net.train()

            param_to_group: dict[str, str] = {}
            for gname, pnames in LAYER_GROUPS.items():
                for pn in pnames:
                    if pn in dict(net.named_parameters()):
                        param_to_group[pn] = gname

            params = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
            param_names = [n for n, _ in params]
            param_tensors = [p for _, p in params]

            def _hutchinson_trace(inp_t, tgt_t):
                V = cfg["vocab_size"]
                global_traces = []
                layer_traces: dict[str, list[float]] = {g: [] for g in layer_group_names}

                sdpa_ctx = torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_math=True, enable_mem_efficient=False
                ) if DEVICE == "cuda" else torch.backends.cuda.sdp_kernel.__new__(
                    torch.backends.cuda.sdp_kernel)

                with sdpa_ctx:
                    for _ in range(n_hutchinson):
                        net.zero_grad()
                        logits = net(inp_t).float()
                        loss = F.cross_entropy(logits.reshape(-1, V), tgt_t.reshape(-1))
                        grads = torch.autograd.grad(loss, param_tensors, create_graph=True)

                        v_list = [torch.randint_like(p, 0, 2).float() * 2 - 1 for p in param_tensors]
                        gv = sum((g * v).sum() for g, v in zip(grads, v_list))
                        hvp = torch.autograd.grad(gv, param_tensors, retain_graph=False)

                        total_trace = 0.0
                        per_group_trace: dict[str, float] = {g: 0.0 for g in layer_group_names}
                        for pn, hv, v in zip(param_names, hvp, v_list):
                            t = (hv * v).sum().item()
                            total_trace += t
                            g = param_to_group.get(pn)
                            if g:
                                per_group_trace[g] += t

                        global_traces.append(total_trace)
                        for g in layer_group_names:
                            layer_traces[g].append(per_group_trace[g])
                        net.zero_grad()

                return float(np.mean(global_traces)), {g: float(np.mean(layer_traces[g])) for g in layer_group_names}

            burst_global, burst_layers = _hutchinson_trace(burst_inp, burst_tgt)
            other_global, other_layers = _hutchinson_trace(other_inp, other_tgt)

            per_seed.append({
                "burst_global": burst_global,
                "other_global": other_global,
                "burst_layers": burst_layers,
                "other_layers": other_layers,
            })
            seeds_done += 1
            print(f"  {label}: sharpness burst={burst_global:.1f}, other={other_global:.1f}", flush=True)

        results[sched] = {"per_seed": per_seed, "layer_group_names": layer_group_names}

    return results


# ---------------------------------------------------------------------------
# Dashboard generation
# ---------------------------------------------------------------------------

def _plotly_to_mpl_color(c):
    if not isinstance(c, str):
        return c
    m = re.match(r"rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)", c)
    if m:
        r, g, b = int(m.group(1)) / 255, int(m.group(2)) / 255, int(m.group(3)) / 255
        a = float(m.group(4)) if m.group(4) else 1.0
        return (r, g, b, a)
    return c


def _save_png(fig, path: str, width: int = 1200, height: int = 600) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_data = fig.to_dict()
    traces = fig_data.get("data", [])
    layout = fig_data.get("layout", {})
    title = layout.get("title", {})
    title_text = title.get("text", "") if isinstance(title, dict) else str(title)
    title_text = re.sub(r"<[^>]+>", "", title_text).strip()

    dpi = 100
    mfig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

    for trace in traces:
        trace_type = trace.get("type", "scatter")
        x = trace.get("x", [])
        y = trace.get("y", [])
        name = trace.get("name", "")
        color = None
        line_info = trace.get("line", {})
        marker_info = trace.get("marker", {})
        if isinstance(line_info, dict) and "color" in line_info:
            color = _plotly_to_mpl_color(line_info["color"])
        elif isinstance(marker_info, dict) and "color" in marker_info:
            mc = marker_info["color"]
            if isinstance(mc, str):
                color = _plotly_to_mpl_color(mc)

        kwargs = dict(label=name)
        if color and isinstance(color, (str, tuple)):
            kwargs["color"] = color

        if trace_type in ("scatter", "scattergl"):
            mode = trace.get("mode", "lines")
            if "lines" in mode:
                ax.plot(x, y, **kwargs)
            elif "markers" in mode:
                ax.scatter(x, y, s=30, zorder=5, **kwargs)
        elif trace_type == "bar":
            bar_colors = marker_info.get("color") if isinstance(marker_info, dict) else None
            if isinstance(bar_colors, list):
                kwargs.pop("color", None)
                kwargs["color"] = [_plotly_to_mpl_color(c) for c in bar_colors[:len(x)]]
            ax.bar(x, y, alpha=0.8, **kwargs)
        elif trace_type == "heatmap":
            pass

    xaxis = layout.get("xaxis", {})
    yaxis = layout.get("yaxis", {})
    if isinstance(xaxis, dict):
        ax.set_xlabel(xaxis.get("title", {}).get("text", "") if isinstance(xaxis.get("title"), dict) else "")
    if isinstance(yaxis, dict):
        ax.set_ylabel(yaxis.get("title", {}).get("text", "") if isinstance(yaxis.get("title"), dict) else "")

    ax.set_title(title_text[:120], fontsize=10, wrap=True)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(handles[:15], labels[:15], fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(mfig)


def _bar_with_seeds(
    fig,
    schedules: list[str],
    per_seed_values: dict[str, list[float]],
    name: str = "",
    row: int = None,
    col: int = None,
) -> None:
    """Add a bar trace with individual seed points and 95% CI error bars."""
    import plotly.graph_objects as go

    means = []
    ci95s = []
    for s in schedules:
        vals = per_seed_values.get(s, [])
        if vals:
            m = float(np.mean(vals))
            se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0
            means.append(m)
            ci95s.append(1.96 * se)
        else:
            means.append(float("nan"))
            ci95s.append(0)

    bar_kwargs = dict(
        x=schedules, y=means,
        marker_color=[_color(s) for s in schedules],
        error_y=dict(type="data", array=ci95s, visible=True),
        name=name, showlegend=bool(name),
    )
    if row is not None:
        fig.add_trace(go.Bar(**bar_kwargs), row=row, col=col)
    else:
        fig.add_trace(go.Bar(**bar_kwargs))

    for s_idx, s in enumerate(schedules):
        vals = per_seed_values.get(s, [])
        if not vals:
            continue
        jitter = np.random.uniform(-0.15, 0.15, len(vals))
        scatter_kwargs = dict(
            x=[s_idx + j for j in jitter],
            y=vals,
            mode="markers",
            marker=dict(color="rgba(50,50,50,0.5)", size=6),
            showlegend=False,
            hovertext=[f"seed {i}" for i in range(len(vals))],
        )
        if row is not None:
            fig.add_trace(go.Scatter(**scatter_kwargs), row=row, col=col)
        else:
            fig.add_trace(go.Scatter(**scatter_kwargs))


def make_dashboard(results: dict, out_dir: Path) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    all_figs: list[tuple[str, str, go.Figure]] = []

    def _add(key: str, title: str, fig: go.Figure):
        all_figs.append((key, title, fig))
        _save_png(fig, str(charts_dir / f"{key}.png"))

    run_names = results.get("run_names", [])
    first_run = run_names[0] if run_names else ""

    # ------------------------------------------------------------------
    # Section 1: EMA Interpolation (dual-class, fixed)
    # ------------------------------------------------------------------
    for rn in run_names:
        ema = results.get("ema_dual", {}).get(rn, {})
        if not ema:
            continue
        schedules = sorted(ema.keys(), key=_sched_order)
        alphas = ema[schedules[0]]["alphas"]

        for path_key, path_label, burst_key, other_key in [
            ("pre_peak", "Pre-Burst → Peak-Burst", "pre_peak_burst", "pre_peak_other"),
            ("rev_peak", "Reverted → Peak-Burst", "rev_peak_burst", "rev_peak_other"),
        ]:
            fig = make_subplots(rows=1, cols=2, subplot_titles=["Burst Class", "Other Classes"])
            for sched in schedules:
                d = ema[sched]
                ps = d["per_seed"]
                if not ps:
                    continue
                burst_mean = [float(np.mean([s[burst_key][i] for s in ps])) for i in range(len(alphas))]
                other_mean = [float(np.mean([s[other_key][i] for s in ps])) for i in range(len(alphas))]
                fig.add_trace(go.Scatter(
                    x=alphas, y=burst_mean, name=sched,
                    line=dict(color=_color(sched), width=2), mode="lines+markers",
                    showlegend=True,
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=alphas, y=other_mean, name=sched,
                    line=dict(color=_color(sched), width=2), mode="lines+markers",
                    showlegend=False,
                ), row=1, col=2)
            fig.update_layout(
                title=f"EMA Interpolation: {path_label} — {rn}<br>"
                      f"<sup>α=0: start model, α=1: peak burst. Sharp cliff = shallow wrapper.</sup>",
                template="plotly_white", height=500,
            )
            fig.update_xaxes(title_text="α")
            fig.update_yaxes(title_text="Accuracy")
            _add(f"ema_{path_key}_{rn}", f"EMA {path_label} ({rn})", fig)

        cliff_vals = {s: [p["cliff_pre_peak_burst"] for p in ema[s]["per_seed"]] for s in schedules}
        fig_cliff = go.Figure()
        _bar_with_seeds(fig_cliff, schedules, cliff_vals)
        fig_cliff.update_layout(
            title=f"EMA Cliff Alpha (Pre→Peak) — {rn}<br>"
                  "<sup>Higher = capability concentrated in narrow direction (shallow wrapper)</sup>",
            xaxis_title="Schedule", yaxis_title="Cliff Alpha",
            template="plotly_white", height=500,
        )
        _add(f"ema_cliff_{rn}", f"EMA Cliff ({rn})", fig_cliff)

    # ------------------------------------------------------------------
    # Section 2: LMC (dual-class, fixed)
    # ------------------------------------------------------------------
    for rn in run_names:
        lmc = results.get("lmc_dual", {}).get(rn, {})
        if not lmc:
            continue
        schedules = sorted(lmc.keys(), key=_sched_order)
        alphas = lmc[schedules[0]]["alphas"]

        for path_key, path_label, burst_key, other_key in [
            ("pre_peak", "Pre-Burst → Peak-Burst", "pre_peak_burst", "pre_peak_other"),
            ("peak_rev", "Peak-Burst → Post-Reversion", "peak_rev_burst", "peak_rev_other"),
        ]:
            fig = make_subplots(rows=1, cols=2, subplot_titles=["Burst Class Loss", "Other Classes Loss"])
            for sched in schedules:
                d = lmc[sched]
                ps = d["per_seed"]
                if not ps:
                    continue
                burst_mean = [float(np.mean([s[burst_key][i] for s in ps])) for i in range(len(alphas))]
                other_mean = [float(np.mean([s[other_key][i] for s in ps])) for i in range(len(alphas))]
                fig.add_trace(go.Scatter(
                    x=alphas, y=burst_mean, name=sched,
                    line=dict(color=_color(sched), width=2), mode="lines+markers",
                    showlegend=True,
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=alphas, y=other_mean, name=sched,
                    line=dict(color=_color(sched), width=2), mode="lines+markers",
                    showlegend=False,
                ), row=1, col=2)
            fig.update_layout(
                title=f"LMC Loss Barrier: {path_label} — {rn}<br>"
                      f"<sup>High barrier = different basins (deep). Low = same ridge (shallow).</sup>",
                template="plotly_white", height=500,
            )
            fig.update_xaxes(title_text="α")
            fig.update_yaxes(title_text="Cross-Entropy Loss")
            _add(f"lmc_{path_key}_{rn}", f"LMC {path_label} ({rn})", fig)

        for barrier_key, barrier_label in [
            ("barrier_pre_peak_burst", "Pre↔Peak Burst-Class"),
            ("barrier_pre_peak_other", "Pre↔Peak Other-Class"),
            ("barrier_peak_rev_burst", "Peak↔Rev Burst-Class"),
            ("barrier_peak_rev_other", "Peak↔Rev Other-Class"),
        ]:
            vals = {s: [p[barrier_key] for p in lmc[s]["per_seed"]] for s in schedules}
            fig_b = go.Figure()
            _bar_with_seeds(fig_b, schedules, vals)
            fig_b.update_layout(
                title=f"LMC Barrier: {barrier_label} — {rn}",
                xaxis_title="Schedule", yaxis_title="Loss Barrier",
                template="plotly_white", height=500,
            )
            _add(f"lmc_barrier_{barrier_key}_{rn}", f"LMC Barrier {barrier_label} ({rn})", fig_b)

    # ------------------------------------------------------------------
    # Section 3: Frankenstein Layer Swap
    # ------------------------------------------------------------------
    for rn in run_names:
        frank = results.get("frankenstein", {}).get(rn, {})
        if not frank:
            continue
        schedules = sorted(frank.keys(), key=_sched_order)
        cut_points = frank[schedules[0]]["cut_points"]
        cut_labels = ["emb"] + [f"block {i}" for i in range(N_LAYERS)]

        for direction, dir_label, burst_key, other_key in [
            ("pre_bottom", "Pre-Burst Bottom → Post-Burst Top", "pre_bottom_burst", "pre_bottom_other"),
            ("post_bottom", "Post-Burst Bottom → Pre-Burst Top", "post_bottom_burst", "post_bottom_other"),
        ]:
            fig = make_subplots(rows=1, cols=2, subplot_titles=["Burst Class", "Other Classes"])
            for sched in schedules:
                ps = frank[sched]["per_seed"]
                if not ps:
                    continue
                burst_mean = [float(np.mean([s[burst_key][i] for s in ps])) for i in range(len(cut_points))]
                other_mean = [float(np.mean([s[other_key][i] for s in ps])) for i in range(len(cut_points))]
                fig.add_trace(go.Scatter(
                    x=cut_labels, y=burst_mean, name=sched,
                    line=dict(color=_color(sched), width=2), mode="lines+markers",
                    showlegend=True,
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=cut_labels, y=other_mean, name=sched,
                    line=dict(color=_color(sched), width=2), mode="lines+markers",
                    showlegend=False,
                ), row=1, col=2)
            fig.update_layout(
                title=f"Frankenstein: {dir_label} — {rn}<br>"
                      "<sup>Cut point = last block from bottom model. "
                      "Where accuracy jumps = where the capability lives.</sup>",
                template="plotly_white", height=500,
            )
            fig.update_xaxes(title_text="Last Block from Bottom Model")
            fig.update_yaxes(title_text="Accuracy")
            _add(f"frank_{direction}_{rn}", f"Frankenstein {dir_label} ({rn})", fig)

    # ------------------------------------------------------------------
    # Section 4: Cross-Burst Frankenstein
    # ------------------------------------------------------------------
    for rn in run_names:
        xfrank = results.get("cross_frankenstein", {}).get(rn, {})
        if not xfrank:
            continue
        cut_labels = ["emb"] + [f"block {i}" for i in range(N_LAYERS)]

        for pair_key, pair_data in xfrank.items():
            sa, sb = pair_data["sched_a"], pair_data["sched_b"]
            ps = pair_data["per_seed"]
            if not ps:
                continue
            cut_points = pair_data["cut_points"]

            fig = make_subplots(rows=1, cols=2, subplot_titles=["Burst Class", "Other Classes"])
            a_burst = [float(np.mean([s["a_bottom_burst"][i] for s in ps])) for i in range(len(cut_points))]
            a_other = [float(np.mean([s["a_bottom_other"][i] for s in ps])) for i in range(len(cut_points))]
            b_burst = [float(np.mean([s["b_bottom_burst"][i] for s in ps])) for i in range(len(cut_points))]
            b_other = [float(np.mean([s["b_bottom_other"][i] for s in ps])) for i in range(len(cut_points))]

            fig.add_trace(go.Scatter(x=cut_labels, y=a_burst, name=f"{sa} bottom",
                                     line=dict(color=_color(sa), width=2), mode="lines+markers"), row=1, col=1)
            fig.add_trace(go.Scatter(x=cut_labels, y=b_burst, name=f"{sb} bottom",
                                     line=dict(color=_color(sb), width=2), mode="lines+markers"), row=1, col=1)
            fig.add_trace(go.Scatter(x=cut_labels, y=a_other, name=f"{sa} bottom",
                                     line=dict(color=_color(sa), width=2, dash="dash"), mode="lines+markers",
                                     showlegend=False), row=1, col=2)
            fig.add_trace(go.Scatter(x=cut_labels, y=b_other, name=f"{sb} bottom",
                                     line=dict(color=_color(sb), width=2, dash="dash"), mode="lines+markers",
                                     showlegend=False), row=1, col=2)
            fig.update_layout(
                title=f"Cross-Burst Frankenstein: {sa} × {sb} — {rn}<br>"
                      "<sup>Swapping layers between post-burst models of different schedules</sup>",
                template="plotly_white", height=500,
            )
            fig.update_xaxes(title_text="Last Block from Bottom Model")
            fig.update_yaxes(title_text="Accuracy")
            _add(f"xfrank_{pair_key}_{rn}", f"Cross-Frank {sa}×{sb} ({rn})", fig)

    # ------------------------------------------------------------------
    # Section 5: Task Vector Transfer (dual-class)
    # ------------------------------------------------------------------
    for rn in run_names:
        tvt = results.get("transfer_dual", {}).get(rn, {})
        if not tvt:
            continue
        schedules = sorted(tvt.keys(), key=_sched_order)
        for class_key, class_label in [("burst_acc", "Burst Class"), ("other_acc", "Other Classes")]:
            vals = {s: [p[class_key] for p in tvt[s]["per_seed"]] for s in schedules}
            fig = go.Figure()
            _bar_with_seeds(fig, schedules, vals)
            fig.update_layout(
                title=f"Task Vector Transfer: {class_label} — {rn}",
                xaxis_title="Schedule", yaxis_title="Accuracy After Transfer",
                template="plotly_white", height=500,
            )
            _add(f"transfer_{class_key}_{rn}", f"Transfer {class_label} ({rn})", fig)

    # ------------------------------------------------------------------
    # Section 6: Pruning Robustness (dual-class)
    # ------------------------------------------------------------------
    for rn in run_names:
        pr = results.get("pruning_dual", {}).get(rn, {})
        if not pr:
            continue
        schedules = sorted(pr.keys(), key=_sched_order)
        sparsities = pr[schedules[0]]["sparsities"]

        fig = make_subplots(rows=1, cols=2, subplot_titles=["Burst Class", "Other Classes"])
        for sched in schedules:
            ps = pr[sched]["per_seed"]
            if not ps:
                continue
            burst_mean = [float(np.mean([s["burst_accs"][i] for s in ps])) for i in range(len(sparsities))]
            other_mean = [float(np.mean([s["other_accs"][i] for s in ps])) for i in range(len(sparsities))]
            fig.add_trace(go.Scatter(
                x=[s * 100 for s in sparsities], y=burst_mean, name=sched,
                line=dict(color=_color(sched), width=2), mode="lines+markers",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=[s * 100 for s in sparsities], y=other_mean, name=sched,
                line=dict(color=_color(sched), width=2), mode="lines+markers",
                showlegend=False,
            ), row=1, col=2)
        fig.update_layout(
            title=f"Pruning Robustness — {rn}<br>"
                  "<sup>Robust to pruning = deep. Fragile = shallow wrapper.</sup>",
            template="plotly_white", height=500,
        )
        fig.update_xaxes(title_text="Sparsity (%)")
        fig.update_yaxes(title_text="Accuracy")
        _add(f"pruning_{rn}", f"Pruning Robustness ({rn})", fig)

    # ------------------------------------------------------------------
    # Section 7: Relearning Efficiency (dual-class)
    # ------------------------------------------------------------------
    for rn in run_names:
        rl = results.get("relearning_dual", {}).get(rn, {})
        if not rl:
            continue
        schedules = sorted(rl.keys(), key=_sched_order)

        fig = make_subplots(rows=1, cols=2, subplot_titles=["Burst Class", "Other Classes"])
        for sched in schedules:
            ps = rl[sched]["per_seed"]
            if not ps:
                continue
            steps = ps[0]["steps"]
            burst_mean = [float(np.mean([s["burst_accs"][i] for s in ps])) for i in range(len(steps))]
            other_mean = [float(np.mean([s["other_accs"][i] for s in ps])) for i in range(len(steps))]
            fig.add_trace(go.Scatter(
                x=steps, y=burst_mean, name=sched,
                line=dict(color=_color(sched), width=2), mode="lines+markers",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=steps, y=other_mean, name=sched,
                line=dict(color=_color(sched), width=2), mode="lines+markers",
                showlegend=False,
            ), row=1, col=2)
        fig.update_layout(
            title=f"Relearning After Reversion — {rn}<br>"
                  "<sup>Fast reacquisition = shallow (pathway suppressed, not destroyed)</sup>",
            template="plotly_white", height=500,
        )
        fig.update_xaxes(title_text="Relearning Step")
        fig.update_yaxes(title_text="Accuracy")
        _add(f"relearning_{rn}", f"Relearning ({rn})", fig)

        _trapz = getattr(np, "trapezoid", np.trapz)
        auc_vals = {}
        for sched in schedules:
            ps = rl[sched]["per_seed"]
            aucs = []
            for s in ps:
                if s["steps"]:
                    aucs.append(float(_trapz(s["burst_accs"], s["steps"])) / max(s["steps"][-1], 1))
            auc_vals[sched] = aucs

        fig_auc = go.Figure()
        _bar_with_seeds(fig_auc, schedules, auc_vals)
        fig_auc.update_layout(
            title=f"Relearning AUC (Burst Class) — {rn}",
            xaxis_title="Schedule", yaxis_title="Normalised AUC",
            template="plotly_white", height=500,
        )
        _add(f"relearning_auc_{rn}", f"Relearning AUC ({rn})", fig_auc)

    # ------------------------------------------------------------------
    # Section 8: Forgetting Trajectory Dimensionality
    # ------------------------------------------------------------------
    for rn in run_names:
        ftd = results.get("trajectory_dim", {}).get(rn, {})
        if not ftd:
            continue
        schedules = sorted(ftd.keys(), key=_sched_order)
        vals = {s: ftd[s]["per_seed"] for s in schedules}
        fig = go.Figure()
        _bar_with_seeds(fig, schedules, vals)
        fig.update_layout(
            title=f"Forgetting Trajectory Dimensionality — {rn}<br>"
                  "<sup>PCA components for 95% variance of reversion weight path</sup>",
            xaxis_title="Schedule", yaxis_title="Effective Dimensionality",
            template="plotly_white", height=500,
        )
        _add(f"trajectory_dim_{rn}", f"Trajectory Dim ({rn})", fig)

    # ------------------------------------------------------------------
    # Section 9: Forgetting Speed Decomposition
    # ------------------------------------------------------------------
    for rn in run_names:
        fsd = results.get("forgetting_decomposition", {}).get(rn, {})
        if not fsd:
            continue
        schedules = sorted(fsd.keys(), key=_sched_order)

        for metric_key, metric_label in [
            ("initial_slope", "Initial Drop Rate"),
            ("plateau_acc", "Plateau Accuracy"),
            ("reversion_auc", "Reversion AUC"),
        ]:
            vals = {s: [p[metric_key] for p in fsd[s]["per_seed"]] for s in schedules}
            fig = go.Figure()
            _bar_with_seeds(fig, schedules, vals)
            fig.update_layout(
                title=f"Forgetting: {metric_label} — {rn}",
                xaxis_title="Schedule", yaxis_title=metric_label,
                template="plotly_white", height=500,
            )
            _add(f"fsd_{metric_key}_{rn}", f"{metric_label} ({rn})", fig)

    # ------------------------------------------------------------------
    # Section 10: Gradient Re-Alignment During Reversion
    # ------------------------------------------------------------------
    for rn in run_names:
        gt = results.get("grad_temporal", {}).get(rn, {})
        if not gt:
            continue
        schedules = sorted(gt.keys(), key=_sched_order)
        fig = go.Figure()
        for sched in schedules:
            d = gt[sched]
            if not d["steps"]:
                continue
            fig.add_trace(go.Scatter(
                x=d["steps"], y=d["mean_sims"], name=sched,
                line=dict(color=_color(sched), width=2), mode="lines",
            ))
        fig.update_layout(
            title=f"Gradient Re-Alignment During Reversion — {rn}",
            xaxis_title="Reversion Step",
            yaxis_title="Cosine Similarity (burst vs other)",
            template="plotly_white", height=500,
        )
        _add(f"grad_temporal_{rn}", f"Grad Re-Alignment ({rn})", fig)

    # ------------------------------------------------------------------
    # Section 11: Per-Layer Interference Heatmap
    # ------------------------------------------------------------------
    for rn in run_names:
        pli = results.get("layer_interference", {}).get(rn, {})
        if not pli:
            continue
        schedules = sorted(pli.keys(), key=_sched_order)
        sample = next((pli[s] for s in schedules if pli.get(s)), None)
        if not sample or "layer_names" not in sample:
            continue
        layer_names = sample["layer_names"]

        z = []
        for sched in schedules:
            d = pli.get(sched, {})
            row = [d.get("mean_per_layer", {}).get(ln, float("nan")) for ln in layer_names]
            z.append(row)

        fig = go.Figure(go.Heatmap(
            z=z, x=layer_names, y=schedules,
            colorscale="RdBu", zmid=0,
            colorbar=dict(title="Cosine Sim"),
        ))
        fig.update_layout(
            title=f"Per-Layer Gradient Interference (Burst Phase) — {rn}",
            xaxis_title="Layer", yaxis_title="Schedule",
            template="plotly_white", height=500,
        )
        _add(f"layer_interference_{rn}", f"Layer Interference ({rn})", fig)

    # ------------------------------------------------------------------
    # Section 12: Critical Sharpness — Global
    # ------------------------------------------------------------------
    for rn in run_names:
        sharp = results.get("sharpness", {}).get(rn, {})
        if not sharp:
            continue
        schedules = sorted(sharp.keys(), key=_sched_order)

        for class_key, class_label in [("burst_global", "Burst Class"), ("other_global", "Other Classes")]:
            vals = {s: [p[class_key] for p in sharp[s]["per_seed"]] for s in schedules if sharp[s]["per_seed"]}
            if not vals:
                continue
            fig = go.Figure()
            _bar_with_seeds(fig, list(vals.keys()), vals)
            fig.update_layout(
                title=f"Hessian Trace (Sharpness) at Peak Burst: {class_label} — {rn}<br>"
                      "<sup>Higher = sharper minimum = more fragile/shallow learning</sup>",
                xaxis_title="Schedule", yaxis_title="Hessian Trace",
                template="plotly_white", height=500,
            )
            _add(f"sharpness_global_{class_key}_{rn}", f"Sharpness {class_label} ({rn})", fig)

    # ------------------------------------------------------------------
    # Section 13: Critical Sharpness — Per-Layer Heatmap
    # ------------------------------------------------------------------
    for rn in run_names:
        sharp = results.get("sharpness", {}).get(rn, {})
        if not sharp:
            continue
        schedules = sorted(sharp.keys(), key=_sched_order)
        sample_sched = next((s for s in schedules if sharp[s].get("per_seed")), None)
        if not sample_sched:
            continue
        layer_group_names = sharp[sample_sched].get("layer_group_names", [])
        if not layer_group_names:
            continue

        for class_key, class_label in [("burst_layers", "Burst Class"), ("other_layers", "Other Classes")]:
            z = []
            for sched in schedules:
                ps = sharp[sched].get("per_seed", [])
                if not ps:
                    z.append([float("nan")] * len(layer_group_names))
                    continue
                row = []
                for lg in layer_group_names:
                    row.append(float(np.mean([p[class_key].get(lg, 0) for p in ps])))
                z.append(row)

            fig = go.Figure(go.Heatmap(
                z=z, x=layer_group_names, y=schedules,
                colorscale="YlOrRd",
                colorbar=dict(title="Hessian Trace"),
            ))
            fig.update_layout(
                title=f"Per-Layer Sharpness (Hessian Trace): {class_label} — {rn}<br>"
                      "<sup>Localises where the loss landscape is sharpest — "
                      "high sharpness in specific layers = fragile, localised learning</sup>",
                xaxis_title="Layer Group", yaxis_title="Schedule",
                template="plotly_white", height=500,
            )
            _add(f"sharpness_layer_{class_key}_{rn}", f"Per-Layer Sharpness {class_label} ({rn})", fig)

        for class_key, class_label in [("burst_layers", "Burst Class"), ("other_layers", "Other Classes")]:
            fig = go.Figure()
            for sched in schedules:
                ps = sharp[sched].get("per_seed", [])
                if not ps:
                    continue
                means = [float(np.mean([p[class_key].get(lg, 0) for p in ps])) for lg in layer_group_names]
                fig.add_trace(go.Scatter(
                    x=layer_group_names, y=means, name=sched,
                    line=dict(color=_color(sched), width=2), mode="lines+markers",
                ))
            fig.update_layout(
                title=f"Per-Layer Sharpness Profile: {class_label} — {rn}<br>"
                      "<sup>Shows which layers contribute most to loss curvature at peak burst</sup>",
                xaxis_title="Layer Group", yaxis_title="Hessian Trace",
                template="plotly_white", height=500,
            )
            _add(f"sharpness_profile_{class_key}_{rn}", f"Sharpness Profile {class_label} ({rn})", fig)

    # ------------------------------------------------------------------
    # Section 14: Cross-Run Burst Position Comparison
    # ------------------------------------------------------------------
    if len(run_names) > 1:
        for metric_key, metric_label, extract_fn in [
            ("reversion_auc", "Reversion AUC", lambda fsd, s: float(np.mean([p["reversion_auc"] for p in fsd[s]["per_seed"]])) if fsd.get(s) else float("nan")),
            ("plateau_acc", "Plateau Accuracy", lambda fsd, s: float(np.mean([p["plateau_acc"] for p in fsd[s]["per_seed"]])) if fsd.get(s) else float("nan")),
        ]:
            fig = go.Figure()
            for rn in run_names:
                fsd = results.get("forgetting_decomposition", {}).get(rn, {})
                if not fsd:
                    continue
                schedules = sorted(fsd.keys(), key=_sched_order)
                burst_pcts = [int(s.replace("burst_", "")) for s in schedules]
                vals = [extract_fn(fsd, s) for s in schedules]
                bp = results.get("burst_positions", {}).get(rn, "?")
                fig.add_trace(go.Scatter(
                    x=burst_pcts, y=vals, name=f"pos{bp}",
                    mode="lines+markers", line=dict(width=2),
                ))
            fig.update_layout(
                title=f"Burst Position Effect: {metric_label}",
                xaxis_title="Burst %", yaxis_title=metric_label,
                template="plotly_white", height=500,
            )
            _add(f"position_{metric_key}", f"Position {metric_label}", fig)

    # ------------------------------------------------------------------
    # Assemble HTML
    # ------------------------------------------------------------------
    html_parts = ["""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Unified Burstiness Analysis Dashboard</title>
<style>
  body { font-family: Arial, sans-serif; margin: 20px; background: #f0f2f5; }
  h1 { color: #1a1a2e; font-size: 1.8em; }
  h2 { color: #16213e; margin-top: 40px; font-size: 1.3em; }
  .chart-container {
    background: white; border-radius: 10px; padding: 20px;
    margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  }
  .toc { background: white; border-radius: 10px; padding: 20px; margin: 20px 0;
         box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
  .toc a { display: block; margin: 4px 0; color: #1565c0; text-decoration: none; }
  .toc a:hover { text-decoration: underline; }
</style>
</head>
<body>
<h1>Unified Burstiness Analysis Dashboard</h1>
<p style="color:#555; max-width:900px;">
  Combined analysis of all burstiness metrics across runs. All bar charts show
  mean + 95% CI error bars with individual seed points overlaid (10 seeds).
  Metrics evaluated on both burst and other-class data where applicable.
  LMC and EMA interpolation include pre-burst baseline comparisons.
  Frankenstein layer-swap reveals where the burst capability is stored.
</p>
<div class="toc"><strong>Contents:</strong>
"""]

    for i, (key, title, _) in enumerate(all_figs):
        html_parts.append(f'  <a href="#chart_{i}">{i+1}. {title}</a>\n')
    html_parts.append("</div>\n")

    for i, (key, title, fig) in enumerate(all_figs):
        html_parts.append(f'<div class="chart-container" id="chart_{i}">\n')
        html_parts.append(f'<h2>{i+1}. {title}</h2>\n')
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=(i == 0)))
        html_parts.append("</div>\n")

    html_parts.append("</body></html>")

    html_path = out_dir / "dashboard.html"
    with open(html_path, "w") as f:
        f.write("".join(html_parts))
    print(f"\nDashboard saved: {html_path}", flush=True)
    print(f"Charts saved: {charts_dir}", flush=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def analyse_run(
    run_dir: Path,
    n_seeds: int = 10,
    n_prune_levels: int = 10,
    relearn_steps: int = 50,
    frank_seeds: int = 10,
    xfrank_seeds: int = 3,
    n_hutchinson: int = 15,
) -> dict:
    print(f"\n{'='*60}", flush=True)
    print(f"Analysing: {run_dir.name}", flush=True)
    print(f"{'='*60}", flush=True)

    with open(run_dir / "config.json") as f:
        run_cfg = json.load(f)

    rc = parse_run_config(run_cfg)
    base_cfg = rc["base_cfg"]

    with open(run_dir / "_data.pkl", "rb") as f:
        target_pool, bg_pool, _, _, _ = pickle.load(f)

    other_docs_BL = np.concatenate(list(bg_pool.values()))
    burst_docs_BL = np.concatenate(list(target_pool.values()))
    prompt_len = run_cfg["task_info"]["prompt_len"]

    with open(run_dir / "all_results.pkl", "rb") as f:
        all_results = pickle.load(f)

    ckpt_root = run_dir / "checkpoints"
    run_name = run_dir.name

    result = {"run_name": run_name, "burst_pos": rc["burst_pos"]}

    print("\n[1/12] Data-only: forgetting decomposition...", flush=True)
    result["forgetting_decomposition"] = compute_forgetting_decomposition(all_results)

    print("\n[2/12] Data-only: gradient temporal dynamics...", flush=True)
    result["grad_temporal"] = compute_grad_temporal(all_results)

    print("\n[3/12] Data-only: layer interference...", flush=True)
    result["layer_interference"] = compute_layer_interference(all_results)

    if not ckpt_root.exists():
        print("  No checkpoints — skipping checkpoint-based metrics.", flush=True)
        return result

    print("\n[4/12] EMA interpolation (dual-class)...", flush=True)
    result["ema_dual"] = compute_ema_dual(
        ckpt_root, all_results, burst_docs_BL, other_docs_BL, prompt_len, n_seeds=n_seeds)

    print("\n[5/12] LMC (dual-class)...", flush=True)
    result["lmc_dual"] = compute_lmc_dual(
        ckpt_root, all_results, burst_docs_BL, other_docs_BL, prompt_len, n_seeds=n_seeds)

    print("\n[6/12] Frankenstein layer-swap...", flush=True)
    result["frankenstein"] = compute_frankenstein(
        ckpt_root, all_results, burst_docs_BL, other_docs_BL, prompt_len, n_seeds=frank_seeds)

    print("\n[7/12] Cross-burst Frankenstein...", flush=True)
    result["cross_frankenstein"] = compute_cross_burst_frankenstein(
        ckpt_root, all_results, burst_docs_BL, other_docs_BL, prompt_len, n_seeds=xfrank_seeds)

    print("\n[8/12] Task vector transfer (dual-class)...", flush=True)
    result["transfer_dual"] = compute_transfer_dual(
        ckpt_root, all_results, burst_docs_BL, other_docs_BL, prompt_len, n_seeds=n_seeds)

    print("\n[9/12] Pruning robustness (dual-class)...", flush=True)
    result["pruning_dual"] = compute_pruning_dual(
        ckpt_root, all_results, burst_docs_BL, other_docs_BL, prompt_len,
        n_seeds=n_seeds, n_prune_levels=n_prune_levels)

    print("\n[10/12] Forgetting trajectory dim + relearning...", flush=True)
    result["trajectory_dim"] = compute_trajectory_dim(
        ckpt_root, all_results, n_seeds=n_seeds)
    result["relearning_dual"] = compute_relearning_dual(
        ckpt_root, all_results, burst_docs_BL, other_docs_BL, prompt_len,
        n_seeds=n_seeds, relearn_steps=relearn_steps)

    print("\n[11/12] Critical sharpness (global + per-layer Hessian trace)...", flush=True)
    result["sharpness"] = compute_sharpness(
        ckpt_root, all_results, burst_docs_BL, other_docs_BL,
        n_seeds=n_seeds, n_samples=128, n_hutchinson=n_hutchinson)

    return result


def main():
    parser = argparse.ArgumentParser(description="Unified burstiness analysis dashboard.")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("data/unified_analysis"))
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-prune-levels", type=int, default=10)
    parser.add_argument("--relearn-steps", type=int, default=50)
    parser.add_argument("--frank-seeds", type=int, default=10)
    parser.add_argument("--xfrank-seeds", type=int, default=3)
    parser.add_argument("--n-hutchinson", type=int, default=15)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_run_results = []
    for run_dir in args.run_dirs:
        run_dir = Path(run_dir)
        t0 = time.time()
        r = analyse_run(
            run_dir,
            n_seeds=args.n_seeds,
            n_prune_levels=args.n_prune_levels,
            relearn_steps=args.relearn_steps,
            frank_seeds=args.frank_seeds,
            xfrank_seeds=args.xfrank_seeds,
            n_hutchinson=args.n_hutchinson,
        )
        per_run_results.append(r)
        print(f"  Completed {run_dir.name} in {time.time() - t0:.1f}s", flush=True)

    run_names = [r["run_name"] for r in per_run_results]
    burst_positions = {r["run_name"]: r["burst_pos"] for r in per_run_results}

    combined: dict = {"run_names": run_names, "burst_positions": burst_positions}
    metric_keys = [
        "ema_dual", "lmc_dual", "frankenstein", "cross_frankenstein",
        "transfer_dual", "pruning_dual", "relearning_dual",
        "trajectory_dim", "forgetting_decomposition", "grad_temporal",
        "layer_interference", "sharpness",
    ]
    for mk in metric_keys:
        combined[mk] = {}
        for r in per_run_results:
            if mk in r:
                combined[mk][r["run_name"]] = r[mk]

    results_path = args.out_dir / "results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(combined, f)
    print(f"\nResults saved: {results_path}", flush=True)

    print("\nGenerating dashboard...", flush=True)
    make_dashboard(combined, args.out_dir)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
