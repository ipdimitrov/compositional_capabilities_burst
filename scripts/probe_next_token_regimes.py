"""Next-token probes per layer for Other-class vs Burst-class regimes.

Two probe types, both operating per-position on the 6 f3-output positions:

  1. logit_lens  — apply the model's own ln_f + LM_head to intermediate
     layer activations and measure next-token accuracy (no training).
  2. learned_probe — train a small linear layer (N → 10 digit classes)
     with cross-entropy via SGD, then measure accuracy.

Retrains each model to the target step, extracts residual-stream
activations at every transformer layer, and produces per-regime
accuracy curves, A-B diffs, and diff-in-diffs.

Usage:
    python scripts/probe_next_token_regimes.py data/burst_d<depth>_<run_tag>
    python scripts/probe_next_token_regimes.py data/burst_d<depth>_<run_tag> --seed-override 107
    python scripts/probe_next_token_regimes.py data/burst_d<depth>_<run_tag> \\
        --probe-steps 250 500 750 1000
    python scripts/probe_next_token_regimes.py data/burst_d<depth>_<run_tag> --n-workers 38
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib as mpl
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

mpl.use("Agg")
from itertools import combinations

import matplotlib.pyplot as plt

from burst.config import DATA_SEED, SCHED_COLORS, SCHEDULE_ORDER, parse_run_config
from burst.core.gpu import gpu_cfg
from burst.core.parallel import run_job_pool
from burst.core.train.experiment import DepthNData, build_data
from burst.core.train_utils import (
    DEVICE,
    N_PROBE_DOCS_PER_TASK,
    build_probe_docs,
    load_net,
    retrain_with_callbacks,
)
from net.nanogpt import nanoGPT
from synthetic.init import set_seed

"""
Dimension key:
    B: batch_size
    L: doc_len
    T: model input length (= L - 1)
    N: n_embd
    P: n_probe_samples
    K: n_layers + 1 (embedding + transformer blocks)
    M: 6 (f3 output positions)
    C: 10 (digit classes, tokens X0..X9)
    V: vocab_size
"""

PROBE_SEED = 1337
N_DIGITS = 10
PROBE_METHODS = ["logit_lens", "learned_probe"]


def _ordered_schedules(scheds):
    return [s for s in SCHEDULE_ORDER if s in scheds] or sorted(scheds)


def get_final_output_positions(seq_len: int, depth: int) -> list[int]:
    """Model-input positions whose targets are the final-output digits.

    Position p in model-input predicts token p+1 in the original sequence.
    The final output block starts at: 1 + depth + (depth * (1 + seq_len)) + 1
    Model-input position is one less than the original position.
    """
    final_out_original = 1 + depth + 1 + seq_len + (depth - 1) * (1 + seq_len) + 1
    final_out_model_input = final_out_original - 1
    return list(range(final_out_model_input, final_out_model_input + seq_len))


COLLECT_BATCH_SIZE = 256


@torch.no_grad()
def collect_all_layer_acts_KBM_N(
    net: nanoGPT,
    docs_BL: np.ndarray,
    f3_positions: list[int],
    max_samples: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Collect residual-stream activations at f3 positions for every layer."""
    net.eval()
    n = min(len(docs_BL), max_samples)
    np.random.seed(PROBE_SEED)
    idx = np.random.choice(len(docs_BL), n, replace=False)

    n_layers = len(net.transformer.h)
    K = n_layers + 1

    all_layer_acts = [[] for _ in range(K)]
    all_targets = []

    for start in range(0, n, COLLECT_BATCH_SIZE):
        end = min(start + COLLECT_BATCH_SIZE, n)
        tokens_BL = torch.as_tensor(docs_BL[idx[start:end]], dtype=torch.long, device=DEVICE)
        inp_BT = tokens_BL[:, :-1]
        tgt_BT = tokens_BL[:, 1:]

        tok_emb = net.transformer.wte(inp_BT)
        pos = torch.arange(inp_BT.size(1), device=DEVICE)
        pos_emb = net.transformer.wpe(pos)
        x_BTN = net.transformer.drop(tok_emb + pos_emb)

        all_layer_acts[0].append(x_BTN[:, f3_positions, :].float().cpu())

        for block_i, block in enumerate(net.transformer.h):
            x_BTN = block(x_BTN)
            all_layer_acts[block_i + 1].append(x_BTN[:, f3_positions, :].float().cpu())

        all_targets.append(tgt_BT[:, f3_positions].cpu())

    layer_acts = [torch.cat(chunks, dim=0) for chunks in all_layer_acts]
    targets_PM = torch.cat(all_targets, dim=0)
    return layer_acts, targets_PM


@torch.no_grad()
def logit_lens_accuracy_K(
    net: nanoGPT,
    layer_acts: list[torch.Tensor],
    targets_PM: torch.Tensor,
) -> np.ndarray:
    """Apply model's own ln_f + LM_head to each layer's activations."""
    K = len(layer_acts)
    acc_K = np.zeros(K)

    ln_f = net.transformer.ln_f
    lm_head = net.LM_head
    targets_dev = targets_PM.to(DEVICE)

    for k in range(K):
        acts_PMN = layer_acts[k].to(DEVICE)
        normed_PMN = ln_f(acts_PMN)
        logits_PMV = lm_head(normed_PMN)
        preds_PM = logits_PMV.argmax(dim=-1)
        acc_K[k] = (preds_PM == targets_dev).float().mean().item()

    return acc_K


class LinearProbe(nn.Module):
    def __init__(self, n_embd: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(n_embd, n_classes)

    def forward(self, x_BN: torch.Tensor) -> torch.Tensor:
        return self.linear(x_BN)


LEARNED_PROBE_LR = 1e-2
LEARNED_PROBE_EPOCHS = 200
LEARNED_PROBE_VAL_FRAC = 0.2
LEARNED_PROBE_VAL_EVERY = 10
LEARNED_PROBE_PATIENCE = 30


def train_learned_probe(
    acts_PMN: torch.Tensor,
    targets_PM: torch.Tensor,
    n_embd: int,
) -> float:
    """Train a linear probe (N -> 10) on flattened (P*M) samples, return val accuracy."""
    P, M, N = acts_PMN.shape
    feats_SN = acts_PMN.reshape(P * M, N)
    labels_S = targets_PM.reshape(P * M)

    n_total = feats_SN.shape[0]
    n_val = max(int(n_total * LEARNED_PROBE_VAL_FRAC), 1)
    n_train = n_total - n_val

    torch.manual_seed(PROBE_SEED)
    perm = torch.randperm(n_total)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    train_feats = feats_SN[train_idx].to(DEVICE)
    train_labels = labels_S[train_idx].to(DEVICE)
    val_feats = feats_SN[val_idx].to(DEVICE)
    val_labels = labels_S[val_idx].to(DEVICE)

    probe = LinearProbe(n_embd, N_DIGITS).to(DEVICE)
    optimizer = torch.optim.Adam(probe.parameters(), lr=LEARNED_PROBE_LR)

    best_val_acc = 0.0
    epochs_without_improvement = 0
    for epoch in range(LEARNED_PROBE_EPOCHS):
        probe.train()
        logits_SC = probe(train_feats)
        loss = F.cross_entropy(logits_SC, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % LEARNED_PROBE_VAL_EVERY == 0 or epoch == LEARNED_PROBE_EPOCHS - 1:
            probe.eval()
            with torch.no_grad():
                val_preds = probe(val_feats).argmax(dim=-1)
                val_acc = (val_preds == val_labels).float().mean().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += LEARNED_PROBE_VAL_EVERY
            if epochs_without_improvement >= LEARNED_PROBE_PATIENCE:
                break

    return best_val_acc


def learned_probe_accuracy_K(
    layer_acts: list[torch.Tensor],
    targets_PM: torch.Tensor,
    n_embd: int,
) -> np.ndarray:
    """Train a learned linear probe at each layer, return (K,) accuracy array."""
    K = len(layer_acts)
    acc_K = np.zeros(K)
    for k in range(K):
        acc_K[k] = train_learned_probe(layer_acts[k], targets_PM, n_embd)
    return acc_K


def probe_from_checkpoints_at_steps(
    job: dict,
    ckpt_dir: Path,
    probe_steps: list[int],
    other_docs_BL: np.ndarray,
    burst_docs_BL: np.ndarray,
    n_layers: int,
    seq_len: int,
    max_samples: int,
    depth: int,
) -> dict[int, dict]:
    """Load saved checkpoints and probe at each requested step."""
    cfg = job["cfg"]
    results_by_step: dict[int, dict] = {}

    available_ckpts = {}
    if ckpt_dir.exists():
        for pt_file in ckpt_dir.glob("step_*.pt"):
            step = int(pt_file.stem.split("_")[1])
            available_ckpts[step] = str(pt_file)

    for step in probe_steps:
        if step not in available_ckpts:
            print(f"    WARNING: no checkpoint for step {step}, skipping", flush=True)
            continue
        print(f"    Loading ckpt step {step}...", flush=True)
        net = load_net(cfg, available_ckpts[step])
        net.eval()
        results_by_step[step] = probe_all_layers(
            net, other_docs_BL, burst_docs_BL, n_layers, seq_len, max_samples, depth
        )
        del net
        torch.cuda.empty_cache()

    return results_by_step


def retrain_and_probe_at_steps(
    job: dict,
    target_pool: dict,
    bg_pool: dict,
    probe_steps: list[int],
    other_docs_BL: np.ndarray,
    burst_docs_BL: np.ndarray,
    n_layers: int,
    seq_len: int,
    max_samples: int,
    depth: int,
) -> dict[int, dict]:
    """Retrain once and probe at each requested step along the way.

    Returns {step: probe_result} where probe_result is the output of probe_all_layers.
    """
    checkpoint_set = set(probe_steps)
    results_by_step: dict[int, dict] = {}

    def on_step(net, global_step, phase):
        if global_step in checkpoint_set:
            net.eval()
            print(f"    Probing step {global_step} ({phase})...", flush=True)
            results_by_step[global_step] = probe_all_layers(
                net, other_docs_BL, burst_docs_BL, n_layers, seq_len, max_samples, depth
            )
            net.train()

    net = retrain_with_callbacks(
        job, target_pool, bg_pool, on_step=on_step, max_step=max(probe_steps)
    )
    del net
    torch.cuda.empty_cache()
    return results_by_step


build_regime_docs = build_probe_docs


def probe_all_layers(
    net: nanoGPT,
    other_docs_BL: np.ndarray,
    burst_docs_BL: np.ndarray,
    n_layers: int,
    seq_len: int,
    max_samples: int,
    depth: int,
) -> dict:
    """Run both probe methods on both regimes at every layer."""
    f3_pos = get_final_output_positions(seq_len, depth)
    n_embd = net.config.n_embd
    K = n_layers + 1

    results = {m: {"Other": np.zeros(K), "Burst": np.zeros(K)} for m in PROBE_METHODS}

    for regime, docs in [("Other", other_docs_BL), ("Burst", burst_docs_BL)]:
        layer_acts, targets_PM = collect_all_layer_acts_KBM_N(net, docs, f3_pos, max_samples)

        ll_acc = logit_lens_accuracy_K(net, layer_acts, targets_PM)
        results["logit_lens"][regime] = ll_acc

        lp_acc = learned_probe_accuracy_K(layer_acts, targets_PM, n_embd)
        results["learned_probe"][regime] = lp_acc

        for k in range(K):
            layer_name = "emb" if k == 0 else f"L{k - 1}"
            print(
                f"      {layer_name:4s}  {regime}  "
                f"logit_lens={ll_acc[k]:.3f}  learned_probe={lp_acc[k]:.3f}",
                flush=True,
            )

    return results


def compute_diffs(all_results, schedules, methods):
    """Returns {method: {sched: mean_diff_K}} and {method: {sched: per_seed_diffs_SK}}."""
    diffs = {}
    diffs_per_seed = {}
    for method in methods:
        diffs[method] = {}
        diffs_per_seed[method] = {}
        for sched in schedules:
            other_curves, burst_curves = [], []
            for key, val in all_results.items():
                if key.startswith(sched + "_s") and method in val:
                    other_curves.append(val[method]["Other"])
                    burst_curves.append(val[method]["Burst"])
            if other_curves and burst_curves:
                per_seed = np.array(other_curves) - np.array(burst_curves)
                diffs_per_seed[method][sched] = per_seed
                diffs[method][sched] = per_seed.mean(axis=0)
    return diffs, diffs_per_seed


def compute_diff_in_diffs(diffs, methods):
    did = {}
    for method in methods:
        did[method] = {}
        scheds = list(diffs[method].keys())
        for s1, s2 in combinations(scheds, 2):
            did[method][f"{s1}_vs_{s2}"] = diffs[method][s1] - diffs[method][s2]
    return did


def plot_raw_curves(all_results, method, n_layers, output_dir):
    raw_scheds = set()
    for k in all_results:
        raw_scheds.add(k.rsplit("_s", 1)[0])
    schedules_seen = _ordered_schedules(raw_scheds)

    K = n_layers + 1
    layer_labels = ["emb"] + [f"L{i}" for i in range(n_layers)]
    x = np.arange(K)

    n_scheds = len(schedules_seen)
    fig, axes = plt.subplots(n_scheds, 2, figsize=(14, 3.5 * n_scheds), squeeze=False)
    fig.suptitle(f"Next-Token Probe Accuracy — {method}", fontsize=14, fontweight="bold")

    for si, sched in enumerate(schedules_seen):
        for ri, regime in enumerate(["Other", "Burst"]):
            ax = axes[si, ri]
            curves = []
            for key, val in all_results.items():
                if key.startswith(sched + "_s") and method in val:
                    curves.append(val[method][regime])

            if curves:
                arr = np.array(curves)
                mean_c = np.mean(arr, axis=0)
                n_s = len(arr)
                ci = 1.96 * np.std(arr, axis=0) / np.sqrt(n_s) if n_s > 1 else np.std(arr, axis=0)
                ax.plot(x, mean_c, "o-", color=SCHED_COLORS.get(sched, "gray"), lw=2)
                ax.fill_between(
                    x, mean_c - ci, mean_c + ci, color=SCHED_COLORS.get(sched, "gray"), alpha=0.2
                )

            ax.set_xticks(x)
            ax.set_xticklabels(layer_labels, fontsize=8)
            ax.set_ylim(0, 1.05)
            n_s = len(curves) if curves else 0
            ax.set_title(f"{sched} — {regime} (n={n_s})", fontsize=10)
            ax.set_ylabel("Accuracy")
            ax.grid(True, alpha=0.2)

    axes[-1, 0].set_xlabel("Layer")
    axes[-1, 1].set_xlabel("Layer")
    fig.tight_layout()
    fig.savefig(output_dir / f"curves_{method}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ab_diffs(diffs, method, n_layers, output_dir, diffs_per_seed=None):
    K = n_layers + 1
    layer_labels = ["emb"] + [f"L{i}" for i in range(n_layers)]
    x = np.arange(K)

    fig, ax = plt.subplots(figsize=(10, 5))
    for sched in SCHEDULE_ORDER:
        if sched not in diffs[method]:
            continue
        c = SCHED_COLORS.get(sched, "gray")
        ax.plot(x, diffs[method][sched], "o-", color=c, lw=2, label=sched)
        if diffs_per_seed and sched in diffs_per_seed[method]:
            arr = diffs_per_seed[method][sched]
            n_s = len(arr)
            if n_s > 1:
                ci = 1.96 * np.std(arr, axis=0) / np.sqrt(n_s)
                ax.fill_between(
                    x, diffs[method][sched] - ci, diffs[method][sched] + ci, color=c, alpha=0.15
                )

    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, fontsize=9)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Δ accuracy (Other - Burst)", fontsize=11)
    ax.set_title(
        f"Other-Burst Next-Token Diff — {method}\n(mean +/- 95% CI)", fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / f"diff_{method}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_diff_in_diffs(did, method, n_layers, output_dir):
    K = n_layers + 1
    layer_labels = ["emb"] + [f"L{i}" for i in range(n_layers)]
    x = np.arange(K)

    pairs = list(did[method].keys())
    if not pairs:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(pairs), 1)))
    for pi, pair in enumerate(pairs):
        ax.plot(x, did[method][pair], "o-", color=cmap[pi], lw=2, label=pair)

    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, fontsize=9)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Diff-in-Diff", fontsize=11)
    ax.set_title(f"Diff-in-Diff — {method}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / f"diff_in_diff_{method}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_combined_curves(step_results, method, n_layers, output_dir):
    K = n_layers + 1
    layer_labels = ["emb"] + [f"L{i}" for i in range(n_layers)]
    x = np.arange(K)

    raw_scheds = set()
    for step_data in step_results.values():
        for k in step_data:
            raw_scheds.add(k.rsplit("_s", 1)[0])
    schedules_seen = _ordered_schedules(raw_scheds)

    sorted_steps = sorted(step_results.keys())
    step_colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(sorted_steps)))

    n_scheds = len(schedules_seen)
    fig, axes = plt.subplots(n_scheds, 2, figsize=(14, 3.5 * n_scheds), squeeze=False)
    fig.suptitle(f"Next-Token Probe — {method} (all steps)", fontsize=14, fontweight="bold")

    for si, sched in enumerate(schedules_seen):
        for ri, regime in enumerate(["Other", "Burst"]):
            ax = axes[si, ri]
            for ci, step in enumerate(sorted_steps):
                curves = [
                    v[method][regime]
                    for k, v in step_results[step].items()
                    if k.startswith(sched + "_s") and method in v
                ]
                if curves:
                    arr = np.array(curves)
                    mean_c = np.mean(arr, axis=0)
                    n_s = len(arr)
                    ci_band = (
                        1.96 * np.std(arr, axis=0) / np.sqrt(n_s)
                        if n_s > 1
                        else np.std(arr, axis=0)
                    )
                    ax.plot(x, mean_c, "o-", color=step_colors[ci], lw=2, label=f"step {step}")
                    ax.fill_between(
                        x, mean_c - ci_band, mean_c + ci_band, color=step_colors[ci], alpha=0.15
                    )
            ax.set_xticks(x)
            ax.set_xticklabels(layer_labels, fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{sched} — regime {regime}", fontsize=10)
            ax.set_ylabel("Accuracy")
            ax.grid(True, alpha=0.2)
            if si == 0 and ri == 0:
                ax.legend(fontsize=7, loc="upper left")

    axes[-1, 0].set_xlabel("Layer")
    axes[-1, 1].set_xlabel("Layer")
    fig.tight_layout()
    fig.savefig(output_dir / f"combined_curves_{method}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_combined_diffs(step_diffs, method, n_layers, output_dir, step_diffs_per_seed=None):
    K = n_layers + 1
    layer_labels = ["emb"] + [f"L{i}" for i in range(n_layers)]
    x = np.arange(K)

    sorted_steps = sorted(step_diffs.keys())
    all_scheds = set()
    for d in step_diffs.values():
        all_scheds.update(d[method].keys())
    scheds = [s for s in SCHEDULE_ORDER if s in all_scheds] or sorted(all_scheds)

    n_scheds = len(scheds)
    fig, axes = plt.subplots(1, n_scheds, figsize=(5 * n_scheds, 5), squeeze=False)
    fig.suptitle(
        f"Other-Burst Diff — {method} (all steps, mean +/- 95% CI)", fontsize=14, fontweight="bold"
    )

    step_colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(sorted_steps)))

    global_ymin, global_ymax = 0, 0
    for step in sorted_steps:
        for sched in scheds:
            if sched in step_diffs[step][method]:
                vals = step_diffs[step][method][sched]
                global_ymin = min(global_ymin, vals.min())
                global_ymax = max(global_ymax, vals.max())
    margin = max(abs(global_ymin), abs(global_ymax)) * 0.1
    ylim = (global_ymin - margin, global_ymax + margin)

    for si, sched in enumerate(scheds):
        ax = axes[0, si]
        for ci, step in enumerate(sorted_steps):
            if sched in step_diffs[step][method]:
                mean_d = step_diffs[step][method][sched]
                c = step_colors[ci]
                ax.plot(x, mean_d, "o-", color=c, lw=2, label=f"step {step}")
                if (
                    step_diffs_per_seed
                    and step in step_diffs_per_seed
                    and sched in step_diffs_per_seed[step][method]
                ):
                    arr = step_diffs_per_seed[step][method][sched]
                    n_s = len(arr)
                    if n_s > 1:
                        ci_band = 1.96 * np.std(arr, axis=0) / np.sqrt(n_s)
                        ax.fill_between(x, mean_d - ci_band, mean_d + ci_band, color=c, alpha=0.12)
        ax.axhline(0, color="gray", ls="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(layer_labels, fontsize=8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Δ accuracy (Other - Burst)")
        ax.set_title(sched, fontsize=10, fontweight="bold")
        ax.set_ylim(ylim)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_dir / f"combined_diff_{method}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _worker_main():
    """Subprocess entry: load pickled args, run single probe job, save results."""
    import warnings

    warnings.filterwarnings("ignore", message=".*backward hook.*")

    parser = argparse.ArgumentParser()
    parser.add_argument("--job-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--probe-steps", type=int, nargs="+", required=True)
    parser.add_argument("--n-layers", type=int, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--depth", type=int, required=True)
    wargs = parser.parse_args()

    with open(wargs.job_path, "rb") as f:
        job = pickle.load(f)
    with open(wargs.data_path, "rb") as f:
        tp, bp, other_docs, burst_docs = pickle.load(f)

    ckpt_dir = job.get("ckpt_dir")
    if ckpt_dir and Path(ckpt_dir).exists():
        step_results = probe_from_checkpoints_at_steps(
            job,
            Path(ckpt_dir),
            wargs.probe_steps,
            other_docs,
            burst_docs,
            wargs.n_layers,
            wargs.seq_len,
            wargs.max_samples,
            wargs.depth,
        )
    else:
        step_results = retrain_and_probe_at_steps(
            job,
            tp,
            bp,
            wargs.probe_steps,
            other_docs,
            burst_docs,
            wargs.n_layers,
            wargs.seq_len,
            wargs.max_samples,
            wargs.depth,
        )

    with open(wargs.output_path, "wb") as f:
        pickle.dump({"label": job["label"], "step_results": step_results}, f)


def main():
    parser = argparse.ArgumentParser(
        description="Next-token probes (logit lens + learned) for Other vs Burst regimes"
    )
    parser.add_argument("run_dir", type=str)
    parser.add_argument(
        "--probe-steps",
        type=int,
        nargs="+",
        default=None,
        help="Global steps to probe at (default: total_steps + reversion_steps)",
    )
    parser.add_argument(
        "--probe-step",
        type=int,
        default=None,
        help="Single step (legacy, use --probe-steps for multiple)",
    )
    parser.add_argument("--probe-max-samples", type=int, required=True)
    parser.add_argument("--seed-override", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--n-workers", type=int, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    from burst.core.train_utils import resolve_run_paths

    cfg_path, logs_dir, _ = resolve_run_paths(run_dir)
    with open(cfg_path) as f:
        cfg = json.load(f)

    rc = parse_run_config(cfg)
    bcfg, depth, burst_pos, n_a = rc["base_cfg"], rc["depth"], rc["burst_pos"], rc["n_a"]
    total_steps = bcfg["total_steps"]
    reversion_steps = bcfg["reversion_steps"]
    seq_len = bcfg["seq_len"]
    n_layers = bcfg["n_layer"]

    if args.probe_steps:
        probe_steps = args.probe_steps
    elif args.probe_step is not None:
        probe_steps = [args.probe_step]
    else:
        probe_steps = [total_steps + reversion_steps]

    base_output_dir = (
        Path(args.output_dir) if args.output_dir else run_dir / "next_token_regime_probes"
    )
    base_output_dir.mkdir(parents=True, exist_ok=True)

    f3_pos = get_final_output_positions(seq_len, depth)
    print(f"Run dir: {run_dir}")
    print(f"Probe steps: {probe_steps}")
    print(f"Output: {base_output_dir}")
    print(f"Device: {DEVICE}")
    print(f"Methods: {PROBE_METHODS}")
    print(f"Final-output model-input positions: {f3_pos}")

    print(f"\nRebuilding data (seed={DATA_SEED})...")
    tp, bp, _, _, cfg_out, ti = build_data(bcfg, depth, burst_pos, n_a)
    doc_len = ti["doc_len"]
    print(f"  doc_len={doc_len}  seq_len={seq_len}")

    set_seed(DATA_SEED)
    d = DepthNData(bcfg["n_alphabets"], seq_len, n_a, depth, burst_pos, DATA_SEED)
    other_docs, burst_docs = build_regime_docs(d, doc_len, N_PROBE_DOCS_PER_TASK)
    print(f"  Other docs: {other_docs.shape}  Burst docs: {burst_docs.shape}")

    jobs_cfg = cfg["jobs"]
    if args.seed_override is not None:
        jobs_cfg = [j for j in jobs_cfg if j["seed"] == args.seed_override]

    ckpt_root = logs_dir / "checkpoints"
    use_checkpoints = ckpt_root.exists()

    schedules_to_run = sorted({j["schedule"] for j in jobs_cfg})
    n_workers = min(len(jobs_cfg), args.n_workers or gpu_cfg.probe_workers)
    print(f"\n{gpu_cfg.summary()}")
    print(f"Schedules: {schedules_to_run}")
    print(f"Jobs: {len(jobs_cfg)}, workers: {n_workers}")
    print(f"Layers: {n_layers + 1} (emb + {n_layers} blocks)")
    n_probes = len(PROBE_METHODS) * len(jobs_cfg) * (n_layers + 1) * 2 * len(probe_steps)
    print(f"Total probe evaluations: {n_probes}")
    mode = "checkpoint-loading" if use_checkpoints else "retrain"
    print(f"Mode: {mode} ({len(jobs_cfg)} jobs, probing at {len(probe_steps)} steps)\n")

    jobs = []
    for jcfg in jobs_cfg:
        label, seed, schedule = jcfg["label"], jcfg["seed"], jcfg["schedule"]
        job_entry = {
            "label": label,
            "schedule": schedule,
            "seed": seed,
            "cfg": {
                **bcfg,
                "seed": seed,
                "vocab_size": cfg_out["vocab_size"],
                "context_size": cfg_out["context_size"],
            },
        }
        if use_checkpoints:
            job_entry["ckpt_dir"] = str(ckpt_root / label)
        jobs.append(job_entry)

    step_args = [str(s) for s in probe_steps]

    def build_cmd(script, job_path, data_path, output_path):
        return (
            [
                sys.executable, script,
                "--worker",
                "--job-path", job_path,
                "--data-path", data_path,
                "--output-path", output_path,
                "--probe-steps", *step_args,
                "--n-layers", str(n_layers),
                "--seq-len", str(seq_len),
                "--max-samples", str(args.probe_max_samples),
                "--depth", str(depth),
            ]
        )

    all_step_results: dict[int, dict] = {step: {} for step in probe_steps}

    def on_done(jr, n_done, n_total):
        if jr.success:
            for step, res in jr.data["step_results"].items():
                all_step_results[step][jr.data["label"]] = res
            print(f"  [{n_done}/{n_total}] {jr.label:30s} done ({jr.elapsed:.0f}s)", flush=True)
        else:
            print(f"  FAIL [{n_done}/{n_total}]: {jr.label}", flush=True)
            if jr.error:
                print(f"    {jr.error}", flush=True)

    run_job_pool(
        jobs=jobs,
        worker_script=os.path.abspath(__file__),
        build_cmd=build_cmd,
        on_done=on_done,
        n_workers=n_workers,
        data_payload=(tp, bp, other_docs, burst_docs),
        poll_interval=1.0,
        tmp_prefix="probe_ntp_",
    )

    all_step_diffs = {}
    all_step_diffs_per_seed = {}
    for probe_step in probe_steps:
        step_dir = base_output_dir / f"step_{probe_step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        all_results = all_step_results[probe_step]

        print(f"\nComputing diffs for step {probe_step}...", flush=True)
        diffs, diffs_ps = compute_diffs(all_results, schedules_to_run, PROBE_METHODS)
        did = compute_diff_in_diffs(diffs, PROBE_METHODS)

        save_data = {
            "all_results": all_results,
            "diffs": {m: dict(diffs[m].items()) for m in PROBE_METHODS},
            "diff_in_diffs": {m: dict(did[m].items()) for m in PROBE_METHODS},
            "probe_step": probe_step,
            "methods": PROBE_METHODS,
            "n_layers": n_layers,
            "seq_len": seq_len,
            "final_output_positions": f3_pos,
            "depth": depth,
        }
        torch.save(save_data, step_dir / "results.pt")
        print(f"Saved results to {step_dir / 'results.pt'}")

        print(f"\nPlotting step {probe_step}...", flush=True)
        for method in PROBE_METHODS:
            print(f"  {method}...")
            plot_raw_curves(all_results, method, n_layers, step_dir)
            plot_ab_diffs(diffs, method, n_layers, step_dir, diffs_per_seed=diffs_ps)
            plot_diff_in_diffs(did, method, n_layers, step_dir)

        all_step_diffs[probe_step] = diffs
        all_step_diffs_per_seed[probe_step] = diffs_ps

    if len(probe_steps) > 1:
        print("\nPlotting combined charts across all steps...", flush=True)
        combined_dir = base_output_dir / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)

        for method in PROBE_METHODS:
            print(f"  {method}...")
            plot_combined_curves(all_step_results, method, n_layers, combined_dir)
            plot_combined_diffs(
                all_step_diffs,
                method,
                n_layers,
                combined_dir,
                step_diffs_per_seed=all_step_diffs_per_seed,
            )

    print(f"\nAll done. Results in {base_output_dir}")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        sys.argv.remove("--worker")
        _worker_main()
    else:
        main()
