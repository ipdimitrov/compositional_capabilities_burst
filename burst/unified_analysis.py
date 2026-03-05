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
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from typing import Any
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from omegaconf import OmegaConf

from net.nanogpt import nanoGPT
from net.runner import configure_optimizers, update_cosine_warmup_lr
from burst.train_utils import load_net, make_net_bare
from burst.config import (
    PHASE_BURST, PHASE_REVERSION, SCHEDULE_ORDER, SCHED_COLORS,
    parse_run_config,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCHEDULES_ORDERED = SCHEDULE_ORDER
SCHEDULE_COLORS = SCHED_COLORS


# ---------------------------------------------------------------------------
# Preloaded checkpoint data for a single seed
# ---------------------------------------------------------------------------

@dataclass
class SeedCheckpoints:
    label: str
    cfg: dict
    r: dict
    sd_pre: dict[str, torch.Tensor]
    sd_peak: dict[str, torch.Tensor]
    sd_rev: dict[str, torch.Tensor]
    sd_pre_cpu: dict[str, torch.Tensor]
    sd_peak_cpu: dict[str, torch.Tensor]
    sd_rev_cpu: dict[str, torch.Tensor]
    files: dict[int, Path]
    pre_step: int
    peak_step: int
    rev_step: int


@torch.no_grad()
def _free_gen_acc(net: nanoGPT, docs_BL: np.ndarray, prompt_len: int) -> float:
    net.eval()
    docs_t = torch.as_tensor(docs_BL, dtype=torch.long, device=DEVICE)
    B, L = docs_t.shape
    target_B6 = docs_t[:, -6:]
    generated = net.generate(docs_t[:, :prompt_len], L - prompt_len)
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


def _get_key_steps(files: dict[int, Path], r: dict):
    available = sorted(files.keys())
    P = r.get("pre_burst_steps", 0)
    T = r["config"]["total_steps"]
    pre_step = available[0]
    peak_step = min(available, key=lambda x: abs(x - (P + T - 1)))
    rev_step = max(available)
    return pre_step, peak_step, rev_step


def _subsample_docs(docs_BL: np.ndarray, n: int = 256) -> np.ndarray:
    if docs_BL.shape[0] <= n:
        return docs_BL
    idx = np.random.choice(docs_BL.shape[0], n, replace=False)
    return docs_BL[idx]


def _build_layer_groups(n_layer: int) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "emb": ["transformer.wte.weight", "transformer.wpe.weight"],
    }
    for bi in range(n_layer):
        groups[f"block{bi}.attn"] = [
            f"transformer.h.{bi}.attn.c_attn.weight",
            f"transformer.h.{bi}.attn.c_proj.weight",
        ]
        groups[f"block{bi}.mlp"] = [
            f"transformer.h.{bi}.mlp.c_fc.weight",
            f"transformer.h.{bi}.mlp.c_proj.weight",
        ]
        groups[f"block{bi}.ln"] = [
            f"transformer.h.{bi}.ln_1.weight",
            f"transformer.h.{bi}.ln_2.weight",
        ]
    groups["ln_f"] = ["transformer.ln_f.weight"]
    return groups


# ---------------------------------------------------------------------------
# Checkpoint preloading
# ---------------------------------------------------------------------------

def _preload_seeds(
    ckpt_root: Path,
    all_results: list[dict],
    n_seeds: int,
) -> dict[str, list[SeedCheckpoints]]:
    """Preload all needed checkpoints once, grouped by schedule."""
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)
    preloaded: dict[str, list[SeedCheckpoints]] = {}

    for sched in schedules:
        sched_results = jobs_by_schedule[sched]
        seeds: list[SeedCheckpoints] = []
        for r in sched_results:
            if len(seeds) >= n_seeds:
                break
            label = r["label"]
            ckpt_dir = ckpt_root / label
            if not ckpt_dir.exists():
                continue
            files = _ckpt_files(ckpt_dir)
            if not files:
                continue

            cfg = r["config"]
            pre_step, peak_step, rev_step = _get_key_steps(files, r)

            sd_pre_cpu = {k: v.float() for k, v in torch.load(
                str(files[pre_step]), map_location="cpu", weights_only=True).items()}
            sd_peak_cpu = {k: v.float() for k, v in torch.load(
                str(files[peak_step]), map_location="cpu", weights_only=True).items()}
            sd_rev_cpu = {k: v.float() for k, v in torch.load(
                str(files[rev_step]), map_location="cpu", weights_only=True).items()}

            sd_pre = {k: v.to(DEVICE) for k, v in sd_pre_cpu.items()}
            sd_peak = {k: v.to(DEVICE) for k, v in sd_peak_cpu.items()}
            sd_rev = {k: v.to(DEVICE) for k, v in sd_rev_cpu.items()}

            seeds.append(SeedCheckpoints(
                label=label, cfg=cfg, r=r,
                sd_pre=sd_pre, sd_peak=sd_peak, sd_rev=sd_rev,
                sd_pre_cpu=sd_pre_cpu, sd_peak_cpu=sd_peak_cpu, sd_rev_cpu=sd_rev_cpu,
                files=files, pre_step=pre_step, peak_step=peak_step, rev_step=rev_step,
            ))

        preloaded[sched] = seeds

    return preloaded


# ---------------------------------------------------------------------------
# Frankenstein layer-swap
# ---------------------------------------------------------------------------

def _build_hybrid_sd(
    sd_bottom: dict[str, torch.Tensor],
    sd_top: dict[str, torch.Tensor],
    cut_after_block: int,
) -> dict[str, torch.Tensor]:
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
            hybrid[key] = sd_top[key]
        else:
            hybrid[key] = sd_bottom[key]
    return hybrid


@torch.no_grad()
def compute_frankenstein(
    preloaded: dict[str, list[SeedCheckpoints]],
    burst_sub: np.ndarray,
    other_sub: np.ndarray,
    prompt_len: int,
    n_layer: int,
    n_seeds: int = 10,
) -> dict:
    cut_points = list(range(-1, n_layer))
    results = {}

    for sched, seeds in preloaded.items():
        per_seed: list[dict] = []
        for sc in seeds[:n_seeds]:
            net = make_net_bare(sc.cfg)
            all_hybrids: list[tuple[str, dict]] = []
            for k in cut_points:
                all_hybrids.append(("pre_bottom", _build_hybrid_sd(sc.sd_pre, sc.sd_peak, k)))
                all_hybrids.append(("post_bottom", _build_hybrid_sd(sc.sd_peak, sc.sd_pre, k)))

            seed_data: dict[str, list[float]] = {
                "pre_bottom_burst": [], "pre_bottom_other": [],
                "post_bottom_burst": [], "post_bottom_other": [],
            }
            for direction, hybrid_sd in all_hybrids:
                net.load_state_dict(hybrid_sd)
                seed_data[f"{direction}_burst"].append(_free_gen_acc(net, burst_sub, prompt_len))
                seed_data[f"{direction}_other"].append(_free_gen_acc(net, other_sub, prompt_len))

            per_seed.append(seed_data)
            print(f"  {sc.label}: frankenstein done", flush=True)

        results[sched] = {"cut_points": cut_points, "per_seed": per_seed}

    return results


@torch.no_grad()
def compute_cross_burst_frankenstein(
    preloaded: dict[str, list[SeedCheckpoints]],
    burst_sub: np.ndarray,
    other_sub: np.ndarray,
    prompt_len: int,
    n_layer: int,
    n_seeds: int = 10,
    schedule_pairs: list[tuple[str, str]] | None = None,
) -> dict:
    if schedule_pairs is None:
        available = sorted(preloaded.keys(), key=_sched_order)
        schedule_pairs = list(combinations(available, 2))

    cut_points = list(range(-1, n_layer))
    results = {}

    for sched_a, sched_b in schedule_pairs:
        seeds_a = preloaded.get(sched_a, [])
        seeds_b = preloaded.get(sched_b, [])
        if not seeds_a or not seeds_b:
            continue

        per_seed: list[dict] = []
        for sc_a, sc_b in zip(seeds_a, seeds_b):
            if len(per_seed) >= n_seeds:
                break
            net = make_net_bare(sc_a.cfg)
            all_hybrids: list[tuple[str, dict]] = []
            for k in cut_points:
                all_hybrids.append(("a_bottom", _build_hybrid_sd(sc_a.sd_peak, sc_b.sd_peak, k)))
                all_hybrids.append(("b_bottom", _build_hybrid_sd(sc_b.sd_peak, sc_a.sd_peak, k)))

            seed_data: dict[str, list[float]] = {
                "a_bottom_burst": [], "a_bottom_other": [],
                "b_bottom_burst": [], "b_bottom_other": [],
            }
            for direction, hybrid_sd in all_hybrids:
                net.load_state_dict(hybrid_sd)
                seed_data[f"{direction}_burst"].append(_free_gen_acc(net, burst_sub, prompt_len))
                seed_data[f"{direction}_other"].append(_free_gen_acc(net, other_sub, prompt_len))

            per_seed.append(seed_data)
            print(f"  {sc_a.label} x {sc_b.label}: cross-frankenstein done", flush=True)

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
    preloaded: dict[str, list[SeedCheckpoints]],
    burst_sub: np.ndarray,
    other_sub: np.ndarray,
    prompt_len: int,
    n_seeds: int = 10,
    n_alphas: int = 11,
) -> dict:
    alphas = np.linspace(0, 1, n_alphas).tolist()
    burst_t = torch.as_tensor(burst_sub, dtype=torch.long, device=DEVICE)
    other_t = torch.as_tensor(other_sub, dtype=torch.long, device=DEVICE)
    burst_inp, burst_tgt = burst_t[:, :-1], burst_t[:, 1:]
    other_inp, other_tgt = other_t[:, :-1], other_t[:, 1:]

    results = {}

    for sched, seeds in preloaded.items():
        per_seed: list[dict] = []
        for sc in seeds[:n_seeds]:
            V = sc.cfg["vocab_size"]
            net = make_net_bare(sc.cfg)

            all_interps_pre_peak = [
                {k: (1 - a) * sc.sd_pre_cpu[k] + a * sc.sd_peak_cpu[k] for k in sc.sd_pre_cpu}
                for a in alphas
            ]
            all_interps_peak_rev = [
                {k: (1 - a) * sc.sd_peak_cpu[k] + a * sc.sd_rev_cpu[k] for k in sc.sd_peak_cpu}
                for a in alphas
            ]

            def _eval_batch(interps):
                burst_losses, other_losses = [], []
                for interp in interps:
                    net.load_state_dict(interp)
                    net.eval()
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                        bl = F.cross_entropy(net(burst_inp).float().reshape(-1, V),
                                             burst_tgt.reshape(-1)).item()
                        ol = F.cross_entropy(net(other_inp).float().reshape(-1, V),
                                             other_tgt.reshape(-1)).item()
                    burst_losses.append(bl)
                    other_losses.append(ol)
                return burst_losses, other_losses

            pre_peak_burst, pre_peak_other = _eval_batch(all_interps_pre_peak)
            peak_rev_burst, peak_rev_other = _eval_batch(all_interps_peak_rev)

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
            print(f"  {sc.label}: LMC barriers pre↔peak burst={per_seed[-1]['barrier_pre_peak_burst']:.4f}, "
                  f"peak↔rev burst={per_seed[-1]['barrier_peak_rev_burst']:.4f}", flush=True)

        results[sched] = {"alphas": alphas, "per_seed": per_seed}

    return results


# ---------------------------------------------------------------------------
# EMA interpolation (fixed: dual-class, pre-burst baseline)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_ema_dual(
    preloaded: dict[str, list[SeedCheckpoints]],
    burst_sub: np.ndarray,
    other_sub: np.ndarray,
    prompt_len: int,
    n_seeds: int = 10,
) -> dict:
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
    results = {}

    for sched, seeds in preloaded.items():
        per_seed: list[dict] = []
        for sc in seeds[:n_seeds]:
            net = make_net_bare(sc.cfg)

            all_pre_peak = [
                {k: (1 - a) * sc.sd_pre_cpu[k] + a * sc.sd_peak_cpu[k] for k in sc.sd_pre_cpu}
                for a in alphas
            ]
            all_rev_peak = [
                {k: (1 - a) * sc.sd_rev_cpu[k] + a * sc.sd_peak_cpu[k] for k in sc.sd_rev_cpu}
                for a in alphas
            ]

            def _eval_path(interps):
                burst_accs, other_accs = [], []
                for interp in interps:
                    net.load_state_dict(interp)
                    burst_accs.append(_free_gen_acc(net, burst_sub, prompt_len))
                    other_accs.append(_free_gen_acc(net, other_sub, prompt_len))
                return burst_accs, other_accs

            pre_peak_burst, pre_peak_other = _eval_path(all_pre_peak)
            rev_peak_burst, rev_peak_other = _eval_path(all_rev_peak)

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
            print(f"  {sc.label}: EMA cliff pre↔peak={per_seed[-1]['cliff_pre_peak_burst']:.2f}, "
                  f"rev↔peak={per_seed[-1]['cliff_rev_peak_burst']:.2f}", flush=True)

        results[sched] = {"alphas": alphas, "per_seed": per_seed}

    return results


# ---------------------------------------------------------------------------
# Pruning robustness (dual-class)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_pruning_dual(
    preloaded: dict[str, list[SeedCheckpoints]],
    burst_sub: np.ndarray,
    other_sub: np.ndarray,
    prompt_len: int,
    n_seeds: int = 10,
    n_prune_levels: int = 10,
) -> dict:
    sparsities = np.linspace(0, 0.9, n_prune_levels).tolist()
    results = {}

    for sched, seeds in preloaded.items():
        per_seed: list[dict] = []
        for sc in seeds[:n_seeds]:
            sd_orig = sc.sd_peak
            net = make_net_bare(sc.cfg)
            all_w = torch.cat([v.view(-1).abs() for v in sd_orig.values()])

            thresholds = [torch.quantile(all_w, s) if s > 0 else None for s in sparsities]
            all_pruned = []
            for threshold in thresholds:
                if threshold is not None:
                    all_pruned.append({k: v * (v.abs() >= threshold).to(v.dtype) for k, v in sd_orig.items()})
                else:
                    all_pruned.append(sd_orig)

            burst_accs, other_accs = [], []
            for pruned_sd in all_pruned:
                net.load_state_dict(pruned_sd)
                burst_accs.append(_free_gen_acc(net, burst_sub, prompt_len))
                other_accs.append(_free_gen_acc(net, other_sub, prompt_len))

            per_seed.append({"burst_accs": burst_accs, "other_accs": other_accs})
            print(f"  {sc.label}: pruning burst@0%={burst_accs[0]:.3f}, other@0%={other_accs[0]:.3f}", flush=True)

        results[sched] = {"sparsities": sparsities, "per_seed": per_seed}

    return results


# ---------------------------------------------------------------------------
# Task vector transfer (dual-class)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_transfer_dual(
    preloaded: dict[str, list[SeedCheckpoints]],
    burst_sub: np.ndarray,
    other_sub: np.ndarray,
    prompt_len: int,
    n_seeds: int = 10,
) -> dict:
    results = {}

    for sched, seeds in preloaded.items():
        per_seed: list[dict] = []
        for i, sc_src in enumerate(seeds[:n_seeds]):
            sc_tgt = next((s for j, s in enumerate(seeds) if j != i), None)
            if sc_tgt is None:
                continue

            tau = {k: sc_src.sd_peak_cpu[k] - sc_src.sd_pre_cpu[k] for k in sc_src.sd_peak_cpu}
            transferred = {k: sc_tgt.sd_pre_cpu[k] + tau[k] for k in sc_tgt.sd_pre_cpu}

            net = make_net_bare(sc_src.cfg)
            net.load_state_dict(transferred)
            burst_acc = _free_gen_acc(net, burst_sub, prompt_len)
            other_acc = _free_gen_acc(net, other_sub, prompt_len)

            per_seed.append({"burst_acc": burst_acc, "other_acc": other_acc})
            print(f"  {sc_src.label} → {sc_tgt.label}: burst={burst_acc:.3f}, other={other_acc:.3f}", flush=True)

        results[sched] = {"per_seed": per_seed}

    return results


# ---------------------------------------------------------------------------
# Relearning efficiency (dual-class)
# ---------------------------------------------------------------------------

def compute_relearning_dual(
    preloaded: dict[str, list[SeedCheckpoints]],
    burst_docs_BL: np.ndarray,
    burst_sub: np.ndarray,
    other_sub: np.ndarray,
    prompt_len: int,
    n_seeds: int = 10,
    relearn_steps: int = 50,
) -> dict:
    results = {}

    for sched, seeds in preloaded.items():
        per_seed: list[dict] = []
        for sc in seeds[:n_seeds]:
            cfg = sc.cfg
            net = make_net_bare(cfg)
            net.load_state_dict(sc.sd_rev)

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
            print(f"  {sc.label}: relearn burst={burst_accs[-1]:.3f}, other={other_accs[-1]:.3f}", flush=True)

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
            burst_end = r.get("burst_end_step", r.get("pre_burst_steps", 0) + T)
            rev_steps_all = sorted(s for s in files if s >= burst_end)
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
# Analysis feature flags — comment out any key to skip that metric.
# Data-only metrics (1-3, 12-17) read from all_results; no GPU needed.
# Checkpoint-based metrics (4-11, 18) require preloaded SeedCheckpoints.
# ---------------------------------------------------------------------------
ANALYSIS_METRICS: dict[str, bool] = {
    # existing metrics
    "forgetting_decomposition": True,
    "grad_temporal":            True,
    "layer_interference":       True,
    "ema_dual":                 True,
    "lmc_dual":                 True,
    "frankenstein":             True,
    "cross_frankenstein":       True,
    "transfer_dual":            True,
    "pruning_dual":             True,
    "trajectory_dim":           True,
    "relearning_dual":          True,
    "sharpness":                True,
    # new gradient metrics (data-only, read from grad_sim_log in all_results)
    "grad_norm_ratio":           True,
    "grad_rank":                 True,
    "grad_snr":                  True,
    "conflict_rate":             True,
    "token_pos_grad":            True,
    "grad_attribution":          True,
    # new gradient metric (requires preloaded checkpoints)
    "forgetting_grad_alignment": True,
}


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

            burst_end = r.get("burst_end_step", r.get("pre_burst_steps", 0) + T)
            rev_steps = [s - burst_end for s, p in zip(steps, phases) if p == PHASE_REVERSION]
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
            burst_end = r.get("burst_end_step", r.get("pre_burst_steps", 0) + T)
            for s, sim, ph in zip(steps, sims, phases):
                if ph == PHASE_REVERSION:
                    step_sims.setdefault(s - burst_end, []).append(sim)

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
        layer_end_sims: dict[str, list[float]] = {}
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
                    layer_end_sims.setdefault(ln, []).append(float(burst_vals[-1]))

        if not layer_sims:
            results[sched] = {}
            continue

        mean_per_layer = {ln: float(np.mean(vs)) for ln, vs in layer_sims.items()}
        end_per_layer = {ln: float(np.mean(vs)) for ln, vs in layer_end_sims.items()}
        results[sched] = {
            "mean_per_layer": mean_per_layer,
            "end_per_layer": end_per_layer,
            "layer_names": layer_names or list(mean_per_layer.keys()),
        }

    return results


# ---------------------------------------------------------------------------
# New gradient metrics (data-only, read from grad_sim_log in all_results)
# ---------------------------------------------------------------------------

def _aggregate_layer_metric(all_results: list[dict], key: str,
                             phase_filter: str = PHASE_BURST) -> dict:
    """Generic aggregator for per-layer time-series dicts stored in grad_sim_log.

    Returns {sched: {mean_per_layer, end_per_layer, layer_names}}.
    """
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)
    results = {}

    for sched in schedules:
        layer_vals: dict[str, list[float]] = {}
        layer_end_vals: dict[str, list[float]] = {}
        layer_names: list[str] = []

        for r in jobs_by_schedule[sched]:
            gsl = r.get("grad_sim_log", {})
            per_layer = gsl.get(key, {})
            phases = gsl.get("phase", [])
            if not layer_names and gsl.get("layer_names"):
                layer_names = gsl["layer_names"]
            for ln, vals in per_layer.items():
                filtered = [v for v, p in zip(vals, phases) if p == phase_filter]
                if filtered:
                    layer_vals.setdefault(ln, []).append(float(np.nanmean(filtered)))
                    layer_end_vals.setdefault(ln, []).append(float(filtered[-1]))

        if not layer_vals:
            results[sched] = {}
            continue

        results[sched] = {
            "mean_per_layer": {ln: float(np.nanmean(vs)) for ln, vs in layer_vals.items()},
            "end_per_layer": {ln: float(np.nanmean(vs)) for ln, vs in layer_end_vals.items()},
            "layer_names": layer_names or list(layer_vals.keys()),
        }

    return results


def compute_grad_norm_ratio(all_results: list[dict]) -> dict:
    """Per-layer ||g_burst|| / ||g_other|| aggregated over burst phase."""
    return _aggregate_layer_metric(all_results, "grad_norm_ratio")


def compute_grad_rank(all_results: list[dict]) -> dict:
    """Per-layer effective gradient rank aggregated over burst phase."""
    return _aggregate_layer_metric(all_results, "grad_rank")


def compute_grad_snr(all_results: list[dict]) -> dict:
    """Per-layer gradient SNR aggregated over burst phase."""
    return _aggregate_layer_metric(all_results, "grad_snr")


def compute_conflict_rate(all_results: list[dict]) -> dict:
    """Per-layer gradient sign conflict rate aggregated over burst phase."""
    return _aggregate_layer_metric(all_results, "conflict_rate")


def compute_token_pos_grad(all_results: list[dict]) -> dict:
    """Mean per-token-position embedding gradient norm over burst phase.

    Returns {sched: {mean_norms: [float], n_positions: int}}.
    """
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)
    results = {}

    for sched in schedules:
        all_norms: list[list[float]] = []
        for r in jobs_by_schedule[sched]:
            gsl = r.get("grad_sim_log", {})
            norms_list = gsl.get("token_pos_grad_norms", [])
            phases = gsl.get("phase", [])
            for norms, phase in zip(norms_list, phases):
                if phase == PHASE_BURST and norms:
                    all_norms.append(norms)

        if not all_norms:
            results[sched] = {"mean_norms": [], "n_positions": 0}
            continue

        max_len = max(len(n) for n in all_norms)
        padded = np.array([n + [float("nan")] * (max_len - len(n)) for n in all_norms])
        mean_norms = np.nanmean(padded, axis=0).tolist()
        results[sched] = {"mean_norms": mean_norms, "n_positions": max_len}

    return results


def compute_grad_attribution(all_results: list[dict]) -> dict:
    """Fraction of gradient norm from intermediate vs final output positions.

    Returns {sched: {per_seed_intermediate: [float], per_seed_final: [float],
                     mean_intermediate: float, mean_final: float}}.
    """
    jobs_by_schedule: dict[str, list[dict]] = {}
    for r in all_results:
        jobs_by_schedule.setdefault(r["schedule"], []).append(r)

    schedules = sorted(jobs_by_schedule.keys(), key=_sched_order)
    results = {}

    for sched in schedules:
        seed_intermediate: list[float] = []
        seed_final: list[float] = []

        for r in jobs_by_schedule[sched]:
            gsl = r.get("grad_sim_log", {})
            attr = gsl.get("grad_attribution", {})
            int_fracs = attr.get("intermediate_frac", [])
            fin_fracs = attr.get("final_frac", [])
            phases = gsl.get("phase", [])

            burst_int = [v for v, p in zip(int_fracs, phases) if p == PHASE_BURST]
            burst_fin = [v for v, p in zip(fin_fracs, phases) if p == PHASE_BURST]

            if burst_int:
                seed_intermediate.append(float(np.nanmean(burst_int)))
            if burst_fin:
                seed_final.append(float(np.nanmean(burst_fin)))

        results[sched] = {
            "per_seed_intermediate": seed_intermediate,
            "per_seed_final": seed_final,
            "mean_intermediate": float(np.nanmean(seed_intermediate)) if seed_intermediate else float("nan"),
            "mean_final": float(np.nanmean(seed_final)) if seed_final else float("nan"),
        }

    return results


# ---------------------------------------------------------------------------
# Forgetting gradient alignment (requires preloaded checkpoints)
# ---------------------------------------------------------------------------

def compute_forgetting_grad_alignment(
    preloaded: dict[str, list[SeedCheckpoints]],
    other_sub: np.ndarray,
    n_seeds: int = 10,
) -> dict:
    """Alignment of other-class gradient at peak-burst with the reversion direction.

    Measures cos(grad_other(theta_peak), theta_pre - theta_peak).

    Positive: other-class gradient at peak actively points back toward pre-burst
              state — the burst modification is unstable under other-class data.
    Near zero: other-class gradient is orthogonal to the burst modification —
               the two live in separate parameter subspaces.

    Returns {sched: {per_seed: [float]}}.
    """
    other_t = torch.as_tensor(other_sub, dtype=torch.long, device=DEVICE)
    other_inp, other_tgt = other_t[:, :-1], other_t[:, 1:]

    results = {}
    for sched, seeds in preloaded.items():
        per_seed: list[float] = []
        for sc in seeds[:n_seeds]:
            net = make_net_bare(sc.cfg)
            net.load_state_dict(sc.sd_peak)
            net.train()
            net.zero_grad()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                logits = net(other_inp).float()
                loss = F.cross_entropy(logits.reshape(-1, sc.cfg["vocab_size"]),
                                       other_tgt.reshape(-1))
            loss.backward()

            # Flat gradient of other-class loss at peak
            g_other = torch.cat([
                p.grad.detach().view(-1).float()
                for p in net.parameters() if p.grad is not None
            ])

            # Reversion direction: theta_pre - theta_peak (flat)
            rev_dir = torch.cat([
                (sc.sd_pre_cpu[k].to(DEVICE).float() - sc.sd_peak[k].float()).view(-1)
                for k in sc.sd_pre_cpu
                if k in sc.sd_peak
            ])

            cos = F.cosine_similarity(g_other.unsqueeze(0), rev_dir.unsqueeze(0)).item()
            per_seed.append(cos)
            net.zero_grad()
            print(f"  {sc.label}: forgetting_grad_alignment={cos:.4f}", flush=True)

        results[sched] = {"per_seed": per_seed}

    return results


# ---------------------------------------------------------------------------
# Critical sharpness: global + per-layer Hutchinson trace of Hessian
# ---------------------------------------------------------------------------

def compute_sharpness(
    preloaded: dict[str, list[SeedCheckpoints]],
    burst_sub: np.ndarray,
    other_sub: np.ndarray,
    n_layer: int,
    n_seeds: int = 10,
    n_hutchinson: int = 15,
) -> dict:
    layer_groups = _build_layer_groups(n_layer)
    layer_group_names = list(layer_groups.keys())

    burst_t = torch.as_tensor(burst_sub, dtype=torch.long, device=DEVICE)
    other_t = torch.as_tensor(other_sub, dtype=torch.long, device=DEVICE)
    burst_inp, burst_tgt = burst_t[:, :-1], burst_t[:, 1:]
    other_inp, other_tgt = other_t[:, :-1], other_t[:, 1:]

    results = {}

    for sched, seeds in preloaded.items():
        per_seed: list[dict] = []
        for sc in seeds[:n_seeds]:
            net = make_net_bare(sc.cfg)
            net.load_state_dict(sc.sd_peak)
            net.train()

            param_to_group: dict[str, str] = {}
            for gname, pnames in layer_groups.items():
                for pn in pnames:
                    if pn in dict(net.named_parameters()):
                        param_to_group[pn] = gname

            params = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
            param_names = [n for n, _ in params]
            param_tensors = [p for _, p in params]

            def _hutchinson_trace(inp_t, tgt_t):
                V = sc.cfg["vocab_size"]
                global_traces = []
                layer_traces: dict[str, list[float]] = {g: [] for g in layer_group_names}

                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
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
            print(f"  {sc.label}: sharpness burst={burst_global:.1f}, other={other_global:.1f}", flush=True)

        results[sched] = {"per_seed": per_seed, "layer_group_names": layer_group_names}

    return results


# ---------------------------------------------------------------------------
# Dashboard generation
# ---------------------------------------------------------------------------

from burst.plot_utils import save_png as _save_png


def _bar_with_seeds(
    fig,
    schedules: list[str],
    per_seed_values: dict[str, list[float]],
    name: str = "",
    row: int = None,
    col: int = None,
) -> None:
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
    n_layer = results.get("n_layer", 6)

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
        cut_labels = ["emb"] + [f"block {i}" for i in range(len(cut_points) - 1)]

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

        for pair_key, pair_data in xfrank.items():
            sa, sb = pair_data["sched_a"], pair_data["sched_b"]
            ps = pair_data["per_seed"]
            if not ps:
                continue
            cut_points = pair_data["cut_points"]
            cut_labels = ["emb"] + [f"block {i}" for i in range(len(cut_points) - 1)]

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
            title=f"Per-Layer Gradient Interference — Mean (Burst Phase) — {rn}",
            xaxis_title="Layer", yaxis_title="Schedule",
            template="plotly_white", height=500,
        )
        _add(f"layer_interference_{rn}", f"Layer Interference Mean ({rn})", fig)

        z_end = []
        for sched in schedules:
            d = pli.get(sched, {})
            row = [d.get("end_per_layer", {}).get(ln, float("nan")) for ln in layer_names]
            z_end.append(row)

        fig_end = go.Figure(go.Heatmap(
            z=z_end, x=layer_names, y=schedules,
            colorscale="RdBu", zmid=0,
            colorbar=dict(title="Cosine Sim"),
        ))
        fig_end.update_layout(
            title=f"Per-Layer Gradient Interference — End of Burst — {rn}",
            xaxis_title="Layer", yaxis_title="Schedule",
            template="plotly_white", height=500,
        )
        _add(f"layer_interference_end_{rn}", f"Layer Interference End-of-Burst ({rn})", fig_end)

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
    # Section 14: Gradient Norm Ratio Per Layer
    # ------------------------------------------------------------------
    for rn in run_names:
        gnr = results.get("grad_norm_ratio", {}).get(rn, {})
        if not gnr:
            continue
        schedules = sorted(gnr.keys(), key=_sched_order)
        sample = next((gnr[s] for s in schedules if gnr.get(s)), None)
        if not sample or "layer_names" not in sample:
            continue
        layer_names = sample["layer_names"]

        z = []
        for sched in schedules:
            d = gnr.get(sched, {})
            row = [d.get("mean_per_layer", {}).get(ln, float("nan")) for ln in layer_names]
            z.append(row)

        fig = go.Figure(go.Heatmap(
            z=z, x=layer_names, y=schedules,
            colorscale="RdBu_r", zmid=1.0,
            colorbar=dict(title="||g_burst|| / ||g_other||"),
        ))
        fig.update_layout(
            title=f"Per-Layer Gradient Norm Ratio (Burst / Other) — Mean (Burst Phase) — {rn}<br>"
                  "<sup>Ratio > 1: burst data dominates that layer's gradient. "
                  "Ratio << 1: other-class data dominates.</sup>",
            xaxis_title="Layer", yaxis_title="Schedule",
            template="plotly_white", height=500,
        )
        _add(f"grad_norm_ratio_{rn}", f"Grad Norm Ratio ({rn})", fig)

    # ------------------------------------------------------------------
    # Section 15: Gradient Effective Rank Per Layer
    # ------------------------------------------------------------------
    for rn in run_names:
        gr = results.get("grad_rank", {}).get(rn, {})
        if not gr:
            continue
        schedules = sorted(gr.keys(), key=_sched_order)
        sample = next((gr[s] for s in schedules if gr.get(s)), None)
        if not sample or "layer_names" not in sample:
            continue
        layer_names = sample["layer_names"]

        z = []
        for sched in schedules:
            d = gr.get(sched, {})
            row = [d.get("mean_per_layer", {}).get(ln, float("nan")) for ln in layer_names]
            z.append(row)

        fig = go.Figure(go.Heatmap(
            z=z, x=layer_names, y=schedules,
            colorscale="Viridis",
            colorbar=dict(title="Effective Rank"),
        ))
        fig.update_layout(
            title=f"Per-Layer Gradient Effective Rank — Mean (Burst Phase) — {rn}<br>"
                  "<sup>Low rank = gradient pushes along a low-dimensional subspace (wrapper-like). "
                  "High rank = rich multi-feature learning.</sup>",
            xaxis_title="Layer", yaxis_title="Schedule",
            template="plotly_white", height=500,
        )
        _add(f"grad_rank_{rn}", f"Grad Rank ({rn})", fig)

    # ------------------------------------------------------------------
    # Section 16: Gradient SNR Per Layer
    # ------------------------------------------------------------------
    for rn in run_names:
        gsnr = results.get("grad_snr", {}).get(rn, {})
        if not gsnr:
            continue
        schedules = sorted(gsnr.keys(), key=_sched_order)
        sample = next((gsnr[s] for s in schedules if gsnr.get(s)), None)
        if not sample or "layer_names" not in sample:
            continue
        layer_names = sample["layer_names"]

        z = []
        for sched in schedules:
            d = gsnr.get(sched, {})
            row = [d.get("mean_per_layer", {}).get(ln, float("nan")) for ln in layer_names]
            z.append(row)

        fig = go.Figure(go.Heatmap(
            z=z, x=layer_names, y=schedules,
            colorscale="YlOrRd",
            colorbar=dict(title="SNR"),
        ))
        fig.update_layout(
            title=f"Per-Layer Gradient SNR (Burst Class) — Mean (Burst Phase) — {rn}<br>"
                  "<sup>High SNR: all burst examples push the same direction (shortcut/wrapper). "
                  "Low SNR: diverse gradient directions (richer learning).</sup>",
            xaxis_title="Layer", yaxis_title="Schedule",
            template="plotly_white", height=500,
        )
        _add(f"grad_snr_{rn}", f"Grad SNR ({rn})", fig)

    # ------------------------------------------------------------------
    # Section 17: Gradient Conflict Rate Per Layer
    # ------------------------------------------------------------------
    for rn in run_names:
        gcr = results.get("conflict_rate", {}).get(rn, {})
        if not gcr:
            continue
        schedules = sorted(gcr.keys(), key=_sched_order)
        sample = next((gcr[s] for s in schedules if gcr.get(s)), None)
        if not sample or "layer_names" not in sample:
            continue
        layer_names = sample["layer_names"]

        z = []
        for sched in schedules:
            d = gcr.get(sched, {})
            row = [d.get("mean_per_layer", {}).get(ln, float("nan")) for ln in layer_names]
            z.append(row)

        fig = go.Figure(go.Heatmap(
            z=z, x=layer_names, y=schedules,
            colorscale="RdBu_r", zmid=0.5,
            colorbar=dict(title="Conflict Rate"),
        ))
        fig.update_layout(
            title=f"Per-Layer Gradient Conflict Rate (Burst vs Other) — Mean (Burst Phase) — {rn}<br>"
                  "<sup>Fraction of parameters where burst and other gradients have opposite signs. "
                  "0.5 = random; > 0.5 = systematic conflict.</sup>",
            xaxis_title="Layer", yaxis_title="Schedule",
            template="plotly_white", height=500,
        )
        _add(f"conflict_rate_{rn}", f"Conflict Rate ({rn})", fig)

    # ------------------------------------------------------------------
    # Section 18: Per-Token-Position Gradient Norms
    # ------------------------------------------------------------------
    for rn in run_names:
        tpg = results.get("token_pos_grad", {}).get(rn, {})
        if not tpg:
            continue
        schedules = sorted(tpg.keys(), key=_sched_order)
        fig = go.Figure()
        for sched in schedules:
            d = tpg.get(sched, {})
            norms = d.get("mean_norms", [])
            if not norms:
                continue
            positions = list(range(len(norms)))
            fig.add_trace(go.Scatter(
                x=positions, y=norms, name=sched,
                line=dict(color=_color(sched), width=2), mode="lines+markers",
            ))
        fig.update_layout(
            title=f"Per-Token-Position Embedding Gradient Norm — Mean (Burst Phase) — {rn}<br>"
                  "<sup>Which token positions does the model attend to for gradient updates? "
                  "Concentration on output positions = shortcut; spread = compositional learning.</sup>",
            xaxis_title="Token Position",
            yaxis_title="Mean ||d(loss)/d(embedding)||",
            template="plotly_white", height=500,
        )
        _add(f"token_pos_grad_{rn}", f"Token Pos Grad ({rn})", fig)

    # ------------------------------------------------------------------
    # Section 19: Gradient Attribution to Composition Steps
    # ------------------------------------------------------------------
    for rn in run_names:
        ga = results.get("grad_attribution", {}).get(rn, {})
        if not ga:
            continue
        schedules = sorted(ga.keys(), key=_sched_order)

        int_vals = {s: ga[s]["per_seed_intermediate"] for s in schedules if ga.get(s)}
        fin_vals = {s: ga[s]["per_seed_final"] for s in schedules if ga.get(s)}

        if int_vals:
            fig = go.Figure()
            _bar_with_seeds(fig, list(int_vals.keys()), int_vals)
            fig.update_layout(
                title=f"Gradient Attribution: Intermediate Output Fraction — {rn}<br>"
                      "<sup>Fraction of total gradient norm from intermediate output positions. "
                      "High = model is learning compositional steps; Low = shortcutting to final answer.</sup>",
                xaxis_title="Schedule", yaxis_title="Intermediate Fraction",
                template="plotly_white", height=500,
            )
            _add(f"grad_attribution_intermediate_{rn}", f"Grad Attribution Intermediate ({rn})", fig)

        if fin_vals:
            fig = go.Figure()
            _bar_with_seeds(fig, list(fin_vals.keys()), fin_vals)
            fig.update_layout(
                title=f"Gradient Attribution: Final Output Fraction — {rn}<br>"
                      "<sup>Fraction of total gradient norm from the final output position only. "
                      "High = model ignores intermediate steps.</sup>",
                xaxis_title="Schedule", yaxis_title="Final Output Fraction",
                template="plotly_white", height=500,
            )
            _add(f"grad_attribution_final_{rn}", f"Grad Attribution Final ({rn})", fig)

    # ------------------------------------------------------------------
    # Section 20: Forgetting Gradient Alignment
    # ------------------------------------------------------------------
    for rn in run_names:
        fga = results.get("forgetting_grad_alignment", {}).get(rn, {})
        if not fga:
            continue
        schedules = sorted(fga.keys(), key=_sched_order)
        vals = {s: fga[s]["per_seed"] for s in schedules if fga.get(s) and fga[s].get("per_seed")}
        if not vals:
            continue
        fig = go.Figure()
        _bar_with_seeds(fig, list(vals.keys()), vals)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(
            title=f"Forgetting Gradient Alignment at Peak Burst — {rn}<br>"
                  "<sup>cos(grad_other(theta_peak), theta_pre - theta_peak). "
                  "Positive: other-class gradient actively reverts burst modification. "
                  "Near zero: burst lives in orthogonal subspace.</sup>",
            xaxis_title="Schedule", yaxis_title="Cosine Alignment",
            template="plotly_white", height=500,
        )
        _add(f"forgetting_grad_alignment_{rn}", f"Forgetting Grad Alignment ({rn})", fig)

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
# Extended metrics dashboard (10+ new metrics, dual-alignment charts)
# ---------------------------------------------------------------------------

def _compute_extended_metrics(results: list[dict]) -> dict[str, dict]:
    """Compute 11 additional scalar metrics + curve data per schedule from training logs.

    Dimension key:
        S: n_seeds
        T: burst phase steps
        U: reversion steps
    """
    from burst.config import SCHED_COLORS

    sched_groups: dict[str, list[dict]] = {}
    for r in results:
        sched_groups.setdefault(r["schedule"], []).append(r)

    metrics: dict[str, dict] = {}

    for sched, runs in sched_groups.items():
        T_sched = runs[0]["config"]["total_steps"]
        U_sched = runs[0]["config"]["reversion_steps"]
        ev = runs[0]["config"]["eval_every"]

        steps_to_peak_S: list[float] = []
        burst_efficiency_S: list[float] = []
        retention_ratio_S: list[float] = []
        reversal_speed_S: list[float] = []
        burst_onset_step_S: list[float] = []
        other_drop_during_burst_S: list[float] = []
        other_recovery_steps_S: list[float] = []
        normalized_auc_S: list[float] = []
        burst_learning_rate_S: list[float] = []
        burst_other_ratio_at_peak_S: list[float] = []
        time_to_half_peak_S: list[float] = []

        burst_curve_S: list[np.ndarray] = []
        other_curve_S: list[np.ndarray] = []
        rev_burst_curve_S: list[np.ndarray] = []
        rev_other_curve_S: list[np.ndarray] = []
        burst_steps_arr: np.ndarray | None = None
        rev_steps_arr: np.ndarray | None = None

        for r in runs:
            log = r["log"]
            steps = np.array(log["step"])
            acc_burst = np.array(log.get("acc_burst", [0.0] * len(steps)))
            acc_other = np.array(log.get("acc_other", [0.0] * len(steps)))
            phases = log["phase"]

            burst_mask = np.array([ph == "special" for ph in phases])
            rev_mask = np.array([ph == "all-but-special" for ph in phases])

            burst_steps_loc = steps[burst_mask]
            burst_acc = acc_burst[burst_mask]
            burst_other = acc_other[burst_mask]
            rev_steps_loc = steps[rev_mask]
            rev_acc = acc_burst[rev_mask]
            rev_other = acc_other[rev_mask]

            burst_curve_S.append(burst_acc)
            other_curve_S.append(burst_other)
            rev_burst_curve_S.append(rev_acc)
            rev_other_curve_S.append(rev_other)
            if burst_steps_arr is None and len(burst_steps_loc) > 0:
                burst_steps_arr = burst_steps_loc
            if rev_steps_arr is None and len(rev_steps_loc) > 0:
                rev_steps_arr = rev_steps_loc - rev_steps_loc[0]

            peak = r["peak_burst"]
            peak_idx = int(np.argmax(burst_acc)) if len(burst_acc) > 0 else 0
            peak_step = burst_steps_loc[peak_idx] if len(burst_steps_loc) > peak_idx else T_sched

            steps_to_peak_S.append(float(peak_step))

            bs = r["config"]["batch_size"]
            p = r["config"]["p_target"]
            total_special = T_sched * p * bs
            burst_efficiency_S.append(peak / max(total_special, 1) * 1000)

            rev_end = r["reversion_end_burst"]
            retention_ratio_S.append(rev_end / peak if peak > 1e-6 else 0.0)

            if len(rev_acc) >= 2:
                n_early = min(len(rev_acc), max(2, 50 // ev))
                slope = (rev_acc[n_early - 1] - rev_acc[0]) / max(n_early * ev, 1)
                reversal_speed_S.append(float(slope))
            else:
                reversal_speed_S.append(0.0)

            onset_mask = burst_acc > 0.1
            onset_step = float(burst_steps_loc[onset_mask][0]) if onset_mask.any() else float(T_sched)
            burst_onset_step_S.append(onset_step)

            if len(burst_other) > 0:
                other_drop_during_burst_S.append(float(burst_other[0] - burst_other.min()))
            else:
                other_drop_during_burst_S.append(0.0)

            pre_other = burst_other[0] if len(burst_other) > 0 else 0.0
            recovery_step = float(U_sched)
            if len(rev_other) > 0:
                rec_mask = rev_other >= pre_other * 0.95
                if rec_mask.any():
                    recovery_step = float(rev_steps_loc[rec_mask][0] - rev_steps_loc[0])
            other_recovery_steps_S.append(recovery_step)

            normalized_auc_S.append(r["reversion_auc"] / max(peak * U_sched, 1e-6))

            if len(burst_acc) >= 2:
                bl_slope = (burst_acc.max() - burst_acc[0]) / max(len(burst_acc) * ev, 1)
                burst_learning_rate_S.append(float(bl_slope))
            else:
                burst_learning_rate_S.append(0.0)

            other_at_peak = burst_other[peak_idx] if len(burst_other) > peak_idx else 0.0
            burst_other_ratio_at_peak_S.append(peak / max(other_at_peak, 1e-6))

            half_peak = peak * 0.5
            half_mask = burst_acc >= half_peak
            half_step = float(burst_steps_loc[half_mask][0]) if half_mask.any() else float(T_sched)
            time_to_half_peak_S.append(half_step)

        metrics[sched] = {
            "steps_to_peak": steps_to_peak_S,
            "burst_efficiency": burst_efficiency_S,
            "retention_ratio": retention_ratio_S,
            "reversal_speed": reversal_speed_S,
            "burst_onset_step": burst_onset_step_S,
            "other_drop_during_burst": other_drop_during_burst_S,
            "other_recovery_steps": other_recovery_steps_S,
            "normalized_auc": normalized_auc_S,
            "burst_learning_rate": burst_learning_rate_S,
            "burst_other_ratio_at_peak": burst_other_ratio_at_peak_S,
            "time_to_half_peak": time_to_half_peak_S,
            "burst_curve": burst_curve_S,
            "other_curve": other_curve_S,
            "rev_burst_curve": rev_burst_curve_S,
            "rev_other_curve": rev_other_curve_S,
            "burst_steps": burst_steps_arr,
            "rev_steps": rev_steps_arr,
            "T": T_sched,
            "U": U_sched,
        }

    return metrics


def _ext_bar_fig(
    schedules: list[str],
    metric_key: str,
    title: str,
    yaxis_title: str,
    metrics: dict[str, dict],
    colors: dict[str, str],
):
    import plotly.graph_objects as go

    means, cis, all_vals = [], [], []
    for s in schedules:
        vals = np.array(metrics[s][metric_key])
        means.append(float(vals.mean()))
        ci = 1.96 * vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else float(vals.std())
        cis.append(ci)
        all_vals.append(vals)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=schedules, y=means,
        error_y=dict(type="data", array=cis, visible=True),
        marker_color=[colors.get(s, "#888") for s in schedules],
        name="Mean ± 95% CI",
    ))
    for i, (s, vals) in enumerate(zip(schedules, all_vals)):
        jitter = np.random.default_rng(42 + i).uniform(-0.2, 0.2, len(vals))
        fig.add_trace(go.Scatter(
            x=[s] * len(vals),
            y=vals.tolist(),
            mode="markers",
            marker=dict(color="black", size=5, opacity=0.5),
            showlegend=(i == 0),
            name="Seeds",
        ))
    fig.update_layout(
        title=title, yaxis_title=yaxis_title,
        template="plotly_white", height=450,
    )
    return fig


def _ext_curve_burst_aligned(
    schedules: list[str],
    metrics: dict[str, dict],
    colors: dict[str, str],
    curve_key: str,
    title: str,
    yaxis_title: str,
):
    """Curve chart aligned to burst start (x=0 at start of burst)."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for s in schedules:
        m = metrics[s]
        curves = m[curve_key]
        steps = m["burst_steps"]
        if steps is None or len(curves) == 0:
            continue
        min_len = min(len(c) for c in curves)
        arr = np.array([c[:min_len] for c in curves])
        mean_c = arr.mean(axis=0)
        ci = 1.96 * arr.std(axis=0) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else arr.std(axis=0)
        x = (steps[:min_len] - steps[0]).tolist()
        c = colors.get(s, "#888")
        fig.add_trace(go.Scatter(x=x, y=mean_c.tolist(), mode="lines", name=s,
                                  line=dict(color=c, width=2)))
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=(mean_c + ci).tolist() + (mean_c - ci).tolist()[::-1],
            fill="toself", fillcolor=c, opacity=0.15,
            line=dict(width=0), showlegend=False,
        ))
    fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Burst Start")
    fig.update_layout(
        title=title + " — Burst-Start Aligned",
        xaxis_title="Steps from Burst Start",
        yaxis_title=yaxis_title, template="plotly_white", height=500,
    )
    return fig


def _ext_curve_reversal_aligned(
    schedules: list[str],
    metrics: dict[str, dict],
    colors: dict[str, str],
    burst_curve_key: str,
    rev_curve_key: str,
    title: str,
    yaxis_title: str,
):
    """Curve chart aligned to reversal start (x=0 = end of burst / start of reversal).

    Burst phase shown on negative x (different schedules start at different negative x
    because burst lengths differ). Reversal shown on positive x (all start at 0).
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    for s in schedules:
        m = metrics[s]
        burst_curves = m[burst_curve_key]
        rev_curves = m[rev_curve_key]
        b_steps = m["burst_steps"]
        r_steps = m["rev_steps"]
        c = colors.get(s, "#888")

        if b_steps is not None and len(burst_curves) > 0:
            min_len = min(len(cv) for cv in burst_curves)
            arr = np.array([cv[:min_len] for cv in burst_curves])
            mean_c = arr.mean(axis=0)
            ci = 1.96 * arr.std(axis=0) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else arr.std(axis=0)
            x = (b_steps[:min_len] - b_steps[-1]).tolist()
            fig.add_trace(go.Scatter(x=x, y=mean_c.tolist(), mode="lines", name=s,
                                      line=dict(color=c, width=2)))
            fig.add_trace(go.Scatter(
                x=x + x[::-1],
                y=(mean_c + ci).tolist() + (mean_c - ci).tolist()[::-1],
                fill="toself", fillcolor=c, opacity=0.12,
                line=dict(width=0), showlegend=False,
            ))

        if r_steps is not None and len(rev_curves) > 0:
            min_len = min(len(cv) for cv in rev_curves)
            arr = np.array([cv[:min_len] for cv in rev_curves])
            mean_c = arr.mean(axis=0)
            ci = 1.96 * arr.std(axis=0) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else arr.std(axis=0)
            x = r_steps[:min_len].tolist()
            fig.add_trace(go.Scatter(x=x, y=mean_c.tolist(), mode="lines", showlegend=False,
                                      line=dict(color=c, width=2, dash="dot")))
            fig.add_trace(go.Scatter(
                x=x + x[::-1],
                y=(mean_c + ci).tolist() + (mean_c - ci).tolist()[::-1],
                fill="toself", fillcolor=c, opacity=0.12,
                line=dict(width=0), showlegend=False,
            ))

    fig.add_vline(x=0, line_dash="dash", line_color="black", annotation_text="Reversal Start")
    fig.update_layout(
        title=title + " — Reversal-Start Aligned",
        xaxis_title="Steps from Reversal Start (negative = burst phase)",
        yaxis_title=yaxis_title, template="plotly_white", height=500,
    )
    return fig


def make_extended_metrics_dashboard(run_dirs: list[Path], out_dir: Path):
    """Generate extended metrics dashboard with 11 new metrics and dual-alignment charts.

    Saves:
      - out_dir/extended_metrics.html  (interactive Plotly)
      - out_dir/charts/extended_*.png  (static PNG via kaleido if available)
    """
    import plotly.graph_objects as go
    import plotly.io as pio
    from burst.config import ordered_schedules, SCHED_COLORS
    from burst.train_utils import load_results

    charts_dir = out_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    all_results_combined: list[dict] = []
    for run_dir in run_dirs:
        try:
            results, _ = load_results(run_dir)
            all_results_combined.extend(results)
        except Exception as e:
            print(f"  Warning: could not load {run_dir}: {e}", flush=True)

    if not all_results_combined:
        print("  No results found for extended metrics.", flush=True)
        return

    metrics = _compute_extended_metrics(all_results_combined)
    schedules = ordered_schedules(list(metrics.keys()))
    colors = SCHED_COLORS

    scalar_metrics = [
        ("steps_to_peak", "Steps to Peak Burst Accuracy", "Steps"),
        ("burst_efficiency", "Burst Efficiency (peak / special_examples × 1000)", "Efficiency"),
        ("retention_ratio", "Retention Ratio (final reversal / peak)", "Ratio"),
        ("reversal_speed", "Reversal Speed (acc slope, early reversal)", "Acc/Step"),
        ("burst_onset_step", "Burst Onset Step (first step > 0.1 acc)", "Steps"),
        ("other_drop_during_burst", "Other-Class Acc Drop During Burst", "Accuracy Drop"),
        ("other_recovery_steps", "Other-Class Recovery Steps (post-burst)", "Steps"),
        ("normalized_auc", "Normalized Reversal AUC (AUC / peak×U)", "Normalized AUC"),
        ("burst_learning_rate", "Burst Learning Rate (acc slope during burst)", "Acc/Step"),
        ("burst_other_ratio_at_peak", "Burst/Other Accuracy Ratio at Peak", "Ratio"),
        ("time_to_half_peak", "Time to Half-Peak During Burst", "Steps"),
    ]

    all_figs: list[tuple[str, str, Any]] = []

    def _try_save_png(fig, name):
        try:
            pio.write_image(fig, str(charts_dir / name), width=1200, height=500)
        except Exception:
            pass

    for key, title, ylabel in scalar_metrics:
        fig = _ext_bar_fig(schedules, key, title, ylabel, metrics, colors)
        all_figs.append((key, title, fig))
        _try_save_png(fig, f"extended_{key}.png")

    for curve_key, title, ylabel in [
        ("burst_curve", "Special Class Accuracy", "Accuracy"),
        ("other_curve", "Other-Class Accuracy During Burst", "Accuracy"),
    ]:
        fig_ba = _ext_curve_burst_aligned(schedules, metrics, colors, curve_key, title, ylabel)
        fig_ra = _ext_curve_reversal_aligned(
            schedules, metrics, colors, curve_key,
            "rev_burst_curve" if curve_key == "burst_curve" else "rev_other_curve",
            title, ylabel,
        )
        all_figs.append((f"{curve_key}_burst_aligned", f"{title} — Burst-Start Aligned", fig_ba))
        all_figs.append((f"{curve_key}_reversal_aligned", f"{title} — Reversal-Start Aligned", fig_ra))
        _try_save_png(fig_ba, f"extended_{curve_key}_burst_aligned.png")
        _try_save_png(fig_ra, f"extended_{curve_key}_reversal_aligned.png")

    fig_var = go.Figure()
    for s in schedules:
        vals = np.array(metrics[s]["steps_to_peak"])
        fig_var.add_trace(go.Box(
            y=vals.tolist(), name=s,
            marker_color=colors.get(s, "#888"),
            boxpoints="all", jitter=0.3,
        ))
    fig_var.update_layout(
        title="Cross-Seed Variance: Steps to Peak",
        yaxis_title="Steps to Peak", template="plotly_white", height=450,
    )
    all_figs.append(("seed_variance_steps_to_peak", "Cross-Seed Variance: Steps to Peak", fig_var))
    _try_save_png(fig_var, "extended_seed_variance.png")

    html_parts = ["""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Extended Metrics Dashboard</title>
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
<h1>Extended Metrics Dashboard</h1>
<p style="color:#555; max-width:900px;">
  11 new scalar metrics + 4 dual-alignment curve charts + 1 variance chart.
  Dual-alignment: each curve chart is generated twice — once aligned to burst start
  (x=0 at start of burst phase) and once aligned to reversal start (x=0 at end of
  burst / start of reversal). The reversal-aligned view shows burst history on
  negative x and reversal on positive x, enabling direct comparison of reversal
  dynamics across schedules with different burst lengths.
</p>
<div class="toc"><strong>Contents:</strong>
"""]
    for i, (key, title, _) in enumerate(all_figs):
        html_parts.append(f'  <a href="#ext_{i}">{i+1}. {title}</a>\n')
    html_parts.append("</div>\n")

    for i, (key, title, fig) in enumerate(all_figs):
        html_parts.append(f'<div class="chart-container" id="ext_{i}">\n')
        html_parts.append(f'<h2>{i+1}. {title}</h2>\n')
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=(i == 0)))
        html_parts.append("</div>\n")

    html_parts.append("</body></html>")

    html_path = out_dir / "extended_metrics.html"
    with open(html_path, "w") as f:
        f.write("".join(html_parts))
    print(f"Extended metrics dashboard saved: {html_path}", flush=True)
    print(f"PNG charts saved: {charts_dir}", flush=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _resolve_unified_paths(run_dir: Path) -> tuple[Path, Path, Path, Path]:
    """Return (config_path, data_path, all_results_path, ckpt_root)."""
    results_dir = run_dir / "results"
    logs_dir = run_dir / "logs"

    config_path = (results_dir / "config.json") if (results_dir / "config.json").exists() \
        else (run_dir / "config.json")
    data_path = (logs_dir / "_data.pkl") if (logs_dir / "_data.pkl").exists() \
        else (run_dir / "_data.pkl")
    all_results_path = (logs_dir / "all_results.pkl") if (logs_dir / "all_results.pkl").exists() \
        else (run_dir / "all_results.pkl")
    ckpt_root = (logs_dir / "checkpoints") if (logs_dir / "checkpoints").exists() \
        else (run_dir / "checkpoints")
    return config_path, data_path, all_results_path, ckpt_root


def analyse_run(
    run_dir: Path,
    n_seeds: int = 10,
    n_prune_levels: int = 10,
    relearn_steps: int = 50,
    frank_seeds: int = 10,
    xfrank_seeds: int = 10,
    n_hutchinson: int = 15,
    subsample_n: int = 256,
) -> dict:
    print(f"\n{'='*60}", flush=True)
    print(f"Analysing: {run_dir.name}", flush=True)
    print(f"{'='*60}", flush=True)

    config_path, data_path, all_results_path, ckpt_root = _resolve_unified_paths(run_dir)

    with open(config_path) as f:
        run_cfg = json.load(f)

    rc = parse_run_config(run_cfg)
    base_cfg = rc["base_cfg"]
    n_layer = base_cfg["n_layer"]

    with open(data_path, "rb") as f:
        target_pool, bg_pool, _, _, _ = pickle.load(f)

    other_docs_BL = np.concatenate(list(bg_pool.values()))
    burst_docs_BL = np.concatenate(list(target_pool.values()))
    prompt_len = run_cfg["task_info"]["prompt_len"]

    with open(all_results_path, "rb") as f:
        all_results = pickle.load(f)

    run_name = run_dir.name

    burst_sub = _subsample_docs(burst_docs_BL, n=subsample_n)
    other_sub = _subsample_docs(other_docs_BL, n=subsample_n)

    result = {"run_name": run_name, "burst_pos": rc["burst_pos"], "n_layer": n_layer}

    if ANALYSIS_METRICS.get("forgetting_decomposition", True):
        print("\n[1/18] Data-only: forgetting decomposition...", flush=True)
        result["forgetting_decomposition"] = compute_forgetting_decomposition(all_results)

    if ANALYSIS_METRICS.get("grad_temporal", True):
        print("\n[2/18] Data-only: gradient temporal dynamics...", flush=True)
        result["grad_temporal"] = compute_grad_temporal(all_results)

    if ANALYSIS_METRICS.get("layer_interference", True):
        print("\n[3/18] Data-only: layer interference...", flush=True)
        result["layer_interference"] = compute_layer_interference(all_results)

    if ANALYSIS_METRICS.get("grad_norm_ratio", True):
        print("\n[12/18] Data-only: gradient norm ratio per layer...", flush=True)
        result["grad_norm_ratio"] = compute_grad_norm_ratio(all_results)

    if ANALYSIS_METRICS.get("grad_rank", True):
        print("\n[13/18] Data-only: gradient effective rank per layer...", flush=True)
        result["grad_rank"] = compute_grad_rank(all_results)

    if ANALYSIS_METRICS.get("grad_snr", True):
        print("\n[14/18] Data-only: gradient SNR per layer...", flush=True)
        result["grad_snr"] = compute_grad_snr(all_results)

    if ANALYSIS_METRICS.get("conflict_rate", True):
        print("\n[15/18] Data-only: gradient conflict rate per layer...", flush=True)
        result["conflict_rate"] = compute_conflict_rate(all_results)

    if ANALYSIS_METRICS.get("token_pos_grad", True):
        print("\n[16/18] Data-only: per-token-position gradient norms...", flush=True)
        result["token_pos_grad"] = compute_token_pos_grad(all_results)

    if ANALYSIS_METRICS.get("grad_attribution", True):
        print("\n[17/18] Data-only: gradient attribution to composition steps...", flush=True)
        result["grad_attribution"] = compute_grad_attribution(all_results)

    need_ckpts = any(ANALYSIS_METRICS.get(k, True) for k in (
        "ema_dual", "lmc_dual", "frankenstein", "cross_frankenstein",
        "transfer_dual", "pruning_dual", "trajectory_dim", "relearning_dual",
        "sharpness", "forgetting_grad_alignment",
    ))

    if not ckpt_root.exists():
        print("  No checkpoints — skipping checkpoint-based metrics.", flush=True)
        return result

    if not need_ckpts:
        return result

    max_seeds = max(n_seeds, frank_seeds, xfrank_seeds)
    print(f"\n[preload] Loading checkpoints for up to {max_seeds} seeds per schedule...", flush=True)
    t_preload = time.time()
    preloaded = _preload_seeds(ckpt_root, all_results, max_seeds)
    print(f"  Preloaded in {time.time() - t_preload:.1f}s", flush=True)

    if ANALYSIS_METRICS.get("ema_dual", True):
        print("\n[4/18] EMA interpolation (dual-class)...", flush=True)
        result["ema_dual"] = compute_ema_dual(
            preloaded, burst_sub, other_sub, prompt_len, n_seeds=n_seeds)

    if ANALYSIS_METRICS.get("lmc_dual", True):
        print("\n[5/18] LMC (dual-class)...", flush=True)
        result["lmc_dual"] = compute_lmc_dual(
            preloaded, burst_sub, other_sub, prompt_len, n_seeds=n_seeds)

    if ANALYSIS_METRICS.get("frankenstein", True):
        print("\n[6/18] Frankenstein layer-swap...", flush=True)
        result["frankenstein"] = compute_frankenstein(
            preloaded, burst_sub, other_sub, prompt_len, n_layer=n_layer, n_seeds=frank_seeds)

    if ANALYSIS_METRICS.get("cross_frankenstein", True):
        print("\n[7/18] Cross-burst Frankenstein...", flush=True)
        result["cross_frankenstein"] = compute_cross_burst_frankenstein(
            preloaded, burst_sub, other_sub, prompt_len, n_layer=n_layer, n_seeds=xfrank_seeds)

    if ANALYSIS_METRICS.get("transfer_dual", True):
        print("\n[8/18] Task vector transfer (dual-class)...", flush=True)
        result["transfer_dual"] = compute_transfer_dual(
            preloaded, burst_sub, other_sub, prompt_len, n_seeds=n_seeds)

    if ANALYSIS_METRICS.get("pruning_dual", True):
        print("\n[9/18] Pruning robustness (dual-class)...", flush=True)
        result["pruning_dual"] = compute_pruning_dual(
            preloaded, burst_sub, other_sub, prompt_len,
            n_seeds=n_seeds, n_prune_levels=n_prune_levels)

    if ANALYSIS_METRICS.get("trajectory_dim", True) or ANALYSIS_METRICS.get("relearning_dual", True):
        print("\n[10/18] Forgetting trajectory dim + relearning...", flush=True)
        if ANALYSIS_METRICS.get("trajectory_dim", True):
            result["trajectory_dim"] = compute_trajectory_dim(
                ckpt_root, all_results, n_seeds=n_seeds)
        if ANALYSIS_METRICS.get("relearning_dual", True):
            result["relearning_dual"] = compute_relearning_dual(
                preloaded, burst_docs_BL, burst_sub, other_sub, prompt_len,
                n_seeds=n_seeds, relearn_steps=relearn_steps)

    if ANALYSIS_METRICS.get("sharpness", True):
        print("\n[11/18] Critical sharpness (global + per-layer Hessian trace)...", flush=True)
        result["sharpness"] = compute_sharpness(
            preloaded, burst_sub, other_sub, n_layer=n_layer,
            n_seeds=n_seeds, n_hutchinson=n_hutchinson)

    if ANALYSIS_METRICS.get("forgetting_grad_alignment", True):
        print("\n[18/18] Forgetting gradient alignment at peak burst...", flush=True)
        result["forgetting_grad_alignment"] = compute_forgetting_grad_alignment(
            preloaded, other_sub, n_seeds=n_seeds)

    return result


def main():
    parser = argparse.ArgumentParser(description="Unified burstiness analysis dashboard.")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    default_out = Path(f"data/{datetime.now().strftime('%Y%m%d-%H%M%S')}_unified_analysis")
    parser.add_argument("--out-dir", type=Path, default=default_out)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-prune-levels", type=int, default=10)
    parser.add_argument("--relearn-steps", type=int, default=50)
    parser.add_argument("--frank-seeds", type=int, default=10)
    parser.add_argument("--xfrank-seeds", type=int, default=10)
    parser.add_argument("--n-hutchinson", type=int, default=15)
    parser.add_argument("--subsample-n", type=int, default=256)
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
            subsample_n=args.subsample_n,
        )
        per_run_results.append(r)
        print(f"  Completed {run_dir.name} in {time.time() - t0:.1f}s", flush=True)

    run_names = [r["run_name"] for r in per_run_results]
    burst_positions = {r["run_name"]: r["burst_pos"] for r in per_run_results}
    n_layer = per_run_results[0].get("n_layer", 6) if per_run_results else 6

    combined: dict = {"run_names": run_names, "burst_positions": burst_positions, "n_layer": n_layer}
    metric_keys = [
        "ema_dual", "lmc_dual", "frankenstein", "cross_frankenstein",
        "transfer_dual", "pruning_dual", "relearning_dual",
        "trajectory_dim", "forgetting_decomposition", "grad_temporal",
        "layer_interference", "sharpness",
        # new gradient metrics
        "grad_norm_ratio", "grad_rank", "grad_snr", "conflict_rate",
        "token_pos_grad", "grad_attribution", "forgetting_grad_alignment",
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

    print("\nGenerating extended metrics dashboard...", flush=True)
    make_extended_metrics_dashboard(args.run_dirs, args.out_dir)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
