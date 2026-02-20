"""Next-token probes per layer for regime A vs regime B.

Two probe types, both operating per-position on the 6 f3-output positions:

  1. logit_lens  — apply the model's own ln_f + LM_head to intermediate
     layer activations and measure next-token accuracy (no training).
  2. learned_probe — train a small linear layer (N → 10 digit classes)
     with cross-entropy via SGD, then measure accuracy.

Retrains each model to the target step, extracts residual-stream
activations at every transformer layer, and produces per-regime
accuracy curves, A−B diffs, and diff-in-diffs.

Usage:
    python scripts/probe_next_token_regimes.py data/burst_d3_<run_tag>
    python scripts/probe_next_token_regimes.py data/burst_d3_<run_tag> --seed-override 107
    python scripts/probe_next_token_regimes.py data/burst_d3_<run_tag> --probe-steps 250 500 750 1000
"""
import sys, os, argparse, pickle, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations
from omegaconf import OmegaConf

from synthetic.init import set_seed
from net.nanogpt import nanoGPT
from net.runner import configure_optimizers, update_cosine_warmup_lr
from burst.experiment import Depth3Data, build_data, N_A, SCHEDULES
from burst._worker import n_target_for_step as _n_target_for_step, sample_batch

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROBE_SEED = 1337
N_DIGITS = 10
PROBE_METHODS = ["logit_lens", "learned_probe"]

SCHEDULE_ALIASES = {"end_mixed_50": "end_mixed_50b"}

SCHED_COLORS = {
    "uniform": "#2196F3", "end_block": "#F44336", "mid_block": "#9C27B0",
    "end_mixed_50b": "#FF9800", "end_mixed_50": "#FF9800",
    "end_mixed_75b": "#E91E63", "end_mixed_25b": "#009688",
    "ramp_up": "#795548",
}

SCHEDULE_ORDER = [
    "end_block", "end_mixed_75b", "end_mixed_50b", "end_mixed_50",
    "end_mixed_25b", "uniform",
]


def _ordered_schedules(scheds):
    return [s for s in SCHEDULE_ORDER if s in scheds] or sorted(scheds)


def n_target_for_step(step, total_steps, schedule, p, batch_size):
    return _n_target_for_step(
        step, total_steps, SCHEDULE_ALIASES.get(schedule, schedule), p, batch_size)


def get_f3_positions(seq_len: int) -> list[int]:
    """Model-input positions whose targets are the 6 f3-output digits.

    Position p in model-input predicts token p+1 in the original sequence.
    o3_0 is at original position 4+seq_len+1+seq_len+1+seq_len+1 = 26 (for seq_len=6).
    Model-input position 25 predicts o3_0, position 30 predicts o3_5.
    """
    o3_0_original = 1 + 3 + 1 + seq_len + 1 + seq_len + 1 + seq_len + 1
    o3_0_model_input = o3_0_original - 1
    return list(range(o3_0_model_input, o3_0_model_input + seq_len))


@torch.no_grad()
def collect_all_layer_acts_KBM_N(
    net: nanoGPT,
    docs_BL: np.ndarray,
    f3_positions: list[int],
    max_samples: int = 512,
    batch_size: int = 256,
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

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        tokens_BL = torch.from_numpy(docs_BL[idx[start:end]]).long().to(DEVICE)
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

    for k in range(K):
        acts_PMN = layer_acts[k].to(DEVICE)
        normed_PMN = ln_f(acts_PMN)
        logits_PMV = lm_head(normed_PMN)
        preds_PM = logits_PMV.argmax(dim=-1)
        correct = (preds_PM == targets_PM.to(DEVICE)).float()
        acc_K[k] = correct.mean().item()

    return acc_K


class LinearProbe(nn.Module):
    def __init__(self, n_embd: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(n_embd, n_classes)

    def forward(self, x_BN: torch.Tensor) -> torch.Tensor:
        return self.linear(x_BN)


def train_learned_probe(
    acts_PMN: torch.Tensor,
    targets_PM: torch.Tensor,
    n_embd: int,
    lr: float = 1e-2,
    epochs: int = 200,
    val_frac: float = 0.2,
) -> float:
    """Train a linear probe (N → 10) on flattened (P*M) samples, return val accuracy."""
    P, M, N = acts_PMN.shape
    feats_SN = acts_PMN.reshape(P * M, N)
    labels_S = targets_PM.reshape(P * M)

    n_total = feats_SN.shape[0]
    n_val = max(int(n_total * val_frac), 1)
    n_train = n_total - n_val

    torch.manual_seed(PROBE_SEED)
    perm = torch.randperm(n_total)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    train_feats = feats_SN[train_idx].to(DEVICE)
    train_labels = labels_S[train_idx].to(DEVICE)
    val_feats = feats_SN[val_idx].to(DEVICE)
    val_labels = labels_S[val_idx].to(DEVICE)

    probe = LinearProbe(n_embd, N_DIGITS).to(DEVICE)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    best_val_acc = 0.0
    for _ in range(epochs):
        probe.train()
        logits_SC = probe(train_feats)
        loss = F.cross_entropy(logits_SC, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_feats)
            val_preds = val_logits.argmax(dim=-1)
            val_acc = (val_preds == val_labels).float().mean().item()
            best_val_acc = max(best_val_acc, val_acc)

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


def retrain_to_step(
    job: dict,
    target_pool: dict,
    bg_pool: dict,
    target_step: int,
) -> nanoGPT:
    seed, cfg, schedule = job["seed"], job["cfg"], job["schedule"]
    set_seed(seed)
    net = nanoGPT(OmegaConf.create({
        "compile": False, "vocab_size": cfg["vocab_size"],
        "context_size": cfg["context_size"],
        "n_layer": cfg["n_layer"], "n_head": cfg["n_head"],
        "n_embd": cfg["n_embd"], "dropout": 0.0, "bias": False, "mlp": True,
    })).to(DEVICE)

    optim_cfg = OmegaConf.create({
        "learning_rate": cfg["lr"], "weight_decay": cfg["weight_decay"],
        "beta1": cfg["beta1"], "beta2": cfg["beta2"],
        "grad_clip": cfg["grad_clip"], "decay_lr": True,
        "warmup_iters": cfg["warmup_iters"], "min_lr": cfg["min_lr"],
    })
    optimizer = configure_optimizers(net, optim_cfg)
    scaler = torch.amp.GradScaler('cuda', enabled=DEVICE == "cuda")

    T_train, U = cfg["total_steps"], cfg["undo_steps"]
    bs, p = cfg["batch_size"], cfg["p_target"]

    net.train()
    it = 0

    train_steps = min(T_train, target_step)
    for s in range(train_steps):
        nt = n_target_for_step(s, T_train, schedule, p, bs)
        batch_np, _ = sample_batch(target_pool, bg_pool, nt, bs)
        dat = torch.from_numpy(batch_np).long().to(DEVICE)
        inp, tgt = dat[:, :-1], dat[:, 1:]
        it, _ = update_cosine_warmup_lr(it, optim_cfg, optimizer, T_train + U)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            logits = net(inp)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        scaler.scale(loss).backward()
        if cfg["grad_clip"] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

    undo_steps_to_run = max(0, target_step - T_train)
    for s in range(undo_steps_to_run):
        batch_np, _ = sample_batch(target_pool, bg_pool, 0, bs)
        dat = torch.from_numpy(batch_np).long().to(DEVICE)
        inp, tgt = dat[:, :-1], dat[:, 1:]
        it, _ = update_cosine_warmup_lr(it, optim_cfg, optimizer, T_train + U)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            logits = net(inp)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        scaler.scale(loss).backward()
        if cfg["grad_clip"] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

    net.eval()
    return net


def build_regime_docs(
    data: Depth3Data,
    doc_len: int,
    n_per_task: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    a_pool = data.gen_pool(data.a_comp_train[:min(16, len(data.a_comp_train))], n_per_task)
    b_pool = data.gen_pool(data.b_comp_train, n_per_task)

    def _cat(pool):
        if not pool:
            return np.zeros((0, doc_len), dtype=np.int64)
        arrs = list(pool.values())
        out = np.concatenate(arrs)
        if out.shape[1] < doc_len:
            pad = np.zeros((out.shape[0], doc_len - out.shape[1]), dtype=out.dtype)
            out = np.concatenate([out, pad], axis=1)
        return out[:, :doc_len]

    return _cat(a_pool), _cat(b_pool)


def probe_all_layers(
    net: nanoGPT,
    a_docs_BL: np.ndarray,
    b_docs_BL: np.ndarray,
    n_layers: int,
    seq_len: int,
    max_samples: int = 512,
) -> dict:
    """Run both probe methods on both regimes at every layer."""
    f3_pos = get_f3_positions(seq_len)
    n_embd = net.config.n_embd
    K = n_layers + 1

    results = {m: {"A": np.zeros(K), "B": np.zeros(K)} for m in PROBE_METHODS}

    for regime, docs in [("A", a_docs_BL), ("B", b_docs_BL)]:
        layer_acts, targets_PM = collect_all_layer_acts_KBM_N(
            net, docs, f3_pos, max_samples)

        ll_acc = logit_lens_accuracy_K(net, layer_acts, targets_PM)
        results["logit_lens"][regime] = ll_acc

        lp_acc = learned_probe_accuracy_K(layer_acts, targets_PM, n_embd)
        results["learned_probe"][regime] = lp_acc

        for k in range(K):
            layer_name = "emb" if k == 0 else f"L{k-1}"
            print(f"      {layer_name:4s}  {regime}  "
                  f"logit_lens={ll_acc[k]:.3f}  learned_probe={lp_acc[k]:.3f}",
                  flush=True)

    return results


def compute_diffs(all_results, schedules, methods):
    diffs = {}
    for method in methods:
        diffs[method] = {}
        for sched in schedules:
            a_curves, b_curves = [], []
            for key, val in all_results.items():
                if key.startswith(sched + "_s") and method in val:
                    a_curves.append(val[method]["A"])
                    b_curves.append(val[method]["B"])
            if a_curves and b_curves:
                diffs[method][sched] = np.mean(a_curves, axis=0) - np.mean(b_curves, axis=0)
    return diffs


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
    for k in all_results.keys():
        raw_scheds.add(k.rsplit("_s", 1)[0])
    schedules_seen = _ordered_schedules(raw_scheds)

    K = n_layers + 1
    layer_labels = ["emb"] + [f"L{i}" for i in range(n_layers)]
    x = np.arange(K)

    n_scheds = len(schedules_seen)
    fig, axes = plt.subplots(n_scheds, 2, figsize=(14, 3.5 * n_scheds), squeeze=False)
    fig.suptitle(f"Next-Token Probe Accuracy — {method}", fontsize=14, fontweight="bold")

    for si, sched in enumerate(schedules_seen):
        for ri, regime in enumerate(["A", "B"]):
            ax = axes[si, ri]
            curves = []
            for key, val in all_results.items():
                if key.startswith(sched + "_s") and method in val:
                    curves.append(val[method][regime])

            if curves:
                mean_c = np.mean(curves, axis=0)
                ax.plot(x, mean_c, "o-", color=SCHED_COLORS.get(sched, "gray"), lw=2)
                if len(curves) > 1:
                    std_c = np.std(curves, axis=0)
                    ax.fill_between(x, mean_c - std_c, mean_c + std_c,
                                    color=SCHED_COLORS.get(sched, "gray"), alpha=0.2)

            ax.set_xticks(x)
            ax.set_xticklabels(layer_labels, fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{sched} — regime {regime}", fontsize=10)
            ax.set_ylabel("Accuracy")
            ax.grid(True, alpha=0.2)

    axes[-1, 0].set_xlabel("Layer")
    axes[-1, 1].set_xlabel("Layer")
    fig.tight_layout()
    fig.savefig(output_dir / f"curves_{method}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ab_diffs(diffs, method, n_layers, output_dir):
    K = n_layers + 1
    layer_labels = ["emb"] + [f"L{i}" for i in range(n_layers)]
    x = np.arange(K)

    fig, ax = plt.subplots(figsize=(10, 5))
    for sched in SCHEDULE_ORDER:
        if sched not in diffs[method]:
            continue
        ax.plot(x, diffs[method][sched], "o-",
                color=SCHED_COLORS.get(sched, "gray"), lw=2, label=sched)

    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, fontsize=9)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Δ accuracy (A − B)", fontsize=11)
    ax.set_title(f"A−B Next-Token Diff — {method}", fontsize=13, fontweight="bold")
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
        for k in step_data.keys():
            raw_scheds.add(k.rsplit("_s", 1)[0])
    schedules_seen = _ordered_schedules(raw_scheds)

    sorted_steps = sorted(step_results.keys())
    step_colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(sorted_steps)))

    n_scheds = len(schedules_seen)
    fig, axes = plt.subplots(n_scheds, 2, figsize=(14, 3.5 * n_scheds), squeeze=False)
    fig.suptitle(f"Next-Token Probe — {method} (all steps)",
                 fontsize=14, fontweight="bold")

    for si, sched in enumerate(schedules_seen):
        for ri, regime in enumerate(["A", "B"]):
            ax = axes[si, ri]
            for ci, step in enumerate(sorted_steps):
                curves = [v[method][regime] for k, v in step_results[step].items()
                          if k.startswith(sched + "_s") and method in v]
                if curves:
                    mean_c = np.mean(curves, axis=0)
                    ax.plot(x, mean_c, "o-", color=step_colors[ci], lw=2,
                            label=f"step {step}")
                    if len(curves) > 1:
                        std_c = np.std(curves, axis=0)
                        ax.fill_between(x, mean_c - std_c, mean_c + std_c,
                                        color=step_colors[ci], alpha=0.15)
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
    fig.savefig(output_dir / f"combined_curves_{method}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_combined_diffs(step_diffs, method, n_layers, output_dir):
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
    fig.suptitle(f"A−B Diff — {method} (all steps)",
                 fontsize=14, fontweight="bold")

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
                ax.plot(x, step_diffs[step][method][sched], "o-",
                        color=step_colors[ci], lw=2, label=f"step {step}")
        ax.axhline(0, color="gray", ls="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(layer_labels, fontsize=8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Δ accuracy (A − B)")
        ax.set_title(sched, fontsize=10, fontweight="bold")
        ax.set_ylim(ylim)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_dir / f"combined_diff_{method}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Next-token probes (logit lens + learned) for regime A vs B")
    parser.add_argument("run_dir", type=str)
    parser.add_argument("--probe-steps", type=int, nargs="+", default=None,
                        help="Global steps to probe at (default: total_steps + undo_steps)")
    parser.add_argument("--probe-step", type=int, default=None,
                        help="Single step (legacy, use --probe-steps for multiple)")
    parser.add_argument("--probe-max-samples", type=int, default=512)
    parser.add_argument("--seed-override", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)

    bcfg = cfg["base_cfg"]
    total_steps = bcfg["total_steps"]
    undo_steps = bcfg["undo_steps"]
    seq_len = bcfg["seq_len"]
    n_layers = bcfg["n_layer"]

    if args.probe_steps:
        probe_steps = args.probe_steps
    elif args.probe_step is not None:
        probe_steps = [args.probe_step]
    else:
        probe_steps = [total_steps + undo_steps]

    base_output_dir = (Path(args.output_dir) if args.output_dir
                       else run_dir / "next_token_regime_probes")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run dir: {run_dir}")
    print(f"Probe steps: {probe_steps}")
    print(f"Output: {base_output_dir}")
    print(f"Device: {DEVICE}")
    print(f"Methods: {PROBE_METHODS}")

    f3_pos = get_f3_positions(seq_len)
    print(f"F3 model-input positions: {f3_pos}")

    print("\nRebuilding data (same seed=999)...")
    tp, bp, _, _, cfg_out, ti = build_data(bcfg)
    doc_len = ti["doc_len"]
    print(f"  doc_len={doc_len}  seq_len={seq_len}")

    set_seed(999)
    d = Depth3Data(bcfg["n_alphabets"], seq_len, N_A, 999)
    a_docs, b_docs = build_regime_docs(d, doc_len)
    print(f"  A docs: {a_docs.shape}  B docs: {b_docs.shape}")

    jobs_cfg = cfg["jobs"]
    if args.seed_override is not None:
        jobs_cfg = [j for j in jobs_cfg if j["seed"] == args.seed_override]

    schedules_to_run = sorted(set(j["schedule"] for j in jobs_cfg))
    print(f"\nSchedules: {schedules_to_run}")
    print(f"Jobs: {len(jobs_cfg)}")
    print(f"Layers: {n_layers + 1} (emb + {n_layers} blocks)")
    n_probes = len(PROBE_METHODS) * len(jobs_cfg) * (n_layers + 1) * 2 * len(probe_steps)
    print(f"Total probe evaluations: {n_probes}\n")

    all_step_results = {}
    all_step_diffs = {}

    for probe_step in probe_steps:
        step_dir = base_output_dir / f"step_{probe_step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  PROBE STEP {probe_step}")
        print(f"{'='*60}\n")

        all_results = {}

        for ji, jcfg in enumerate(jobs_cfg):
            label = jcfg["label"]
            seed = jcfg["seed"]
            schedule = jcfg["schedule"]

            job = {
                "label": label, "schedule": schedule, "seed": seed,
                "cfg": {**bcfg, "seed": seed,
                        "vocab_size": cfg_out["vocab_size"],
                        "context_size": cfg_out["context_size"]},
            }

            print(f"[{ji+1}/{len(jobs_cfg)}] {label} — retraining to step {probe_step}...",
                  flush=True)
            net = retrain_to_step(job, tp, bp, probe_step)

            print(f"  Probing...", flush=True)
            result = probe_all_layers(
                net, a_docs, b_docs, n_layers, seq_len, args.probe_max_samples)
            all_results[label] = result

            del net
            torch.cuda.empty_cache()

        print(f"\nComputing diffs for step {probe_step}...", flush=True)
        diffs = compute_diffs(all_results, schedules_to_run, PROBE_METHODS)
        did = compute_diff_in_diffs(diffs, PROBE_METHODS)

        save_data = {
            "all_results": all_results,
            "diffs": {m: {s: arr for s, arr in diffs[m].items()} for m in PROBE_METHODS},
            "diff_in_diffs": {m: {p: arr for p, arr in did[m].items()} for m in PROBE_METHODS},
            "probe_step": probe_step,
            "methods": PROBE_METHODS,
            "n_layers": n_layers,
            "seq_len": seq_len,
            "f3_positions": f3_pos,
        }
        torch.save(save_data, step_dir / "results.pt")
        print(f"Saved results to {step_dir / 'results.pt'}")

        print(f"\nPlotting step {probe_step}...", flush=True)
        for method in PROBE_METHODS:
            print(f"  {method}...")
            plot_raw_curves(all_results, method, n_layers, step_dir)
            plot_ab_diffs(diffs, method, n_layers, step_dir)
            plot_diff_in_diffs(did, method, n_layers, step_dir)

        all_step_results[probe_step] = all_results
        all_step_diffs[probe_step] = diffs

    if len(probe_steps) > 1:
        print(f"\nPlotting combined charts across all steps...", flush=True)
        combined_dir = base_output_dir / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)

        for method in PROBE_METHODS:
            print(f"  {method}...")
            plot_combined_curves(all_step_results, method, n_layers, combined_dir)
            plot_combined_diffs(all_step_diffs, method, n_layers, combined_dir)

    print(f"\nAll done. Results in {base_output_dir}")


if __name__ == "__main__":
    main()
