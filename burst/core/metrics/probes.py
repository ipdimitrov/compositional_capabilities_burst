"""Next-token logit-lens and learned probes per layer for Other- vs Burst-class regimes.

Dimension key:
    B: batch_size
    L: doc_len
    T: model input length (= doc_len - 1)
    N: n_embd
    P: n_probe_samples
    K: n_layers + 1 (embedding + transformer blocks)
    M: seq_len (final-output positions)
    C: n_digits (digit classes)
    V: vocab_size
"""

from __future__ import annotations

import json
import logging
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from burst.config import (
    N_DIGITS,
    PROBE_COLLECT_BATCH_SIZE,
    PROBE_METHODS,
    PROBE_SEED,
)
from burst.core.train_utils import DEVICE

if TYPE_CHECKING:
    from net.nanogpt import nanoGPT

logger = logging.getLogger(__name__)

LEARNED_PROBE_LR = 1e-2
LEARNED_PROBE_EPOCHS = 200
LEARNED_PROBE_VAL_FRAC = 0.2
LEARNED_PROBE_VAL_EVERY = 10
LEARNED_PROBE_PATIENCE = 30

NTP_RESULTS_DIRNAME = "next_token_probes"


def get_final_output_positions(seq_len: int, depth: int) -> list[int]:
    """Model-input positions whose targets are the final-output digits."""
    final_out_original = 1 + depth + 1 + seq_len + (depth - 1) * (1 + seq_len) + 1
    final_out_model_input = final_out_original - 1
    return list(range(final_out_model_input, final_out_model_input + seq_len))


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_layer_acts_and_targets(
    net: nanoGPT,
    docs_BL: np.ndarray,
    f3_positions: list[int],
    max_samples: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Collect residual-stream activations at f3 positions for every layer.

    Returns (layer_acts, targets_PM) where layer_acts is a list of K tensors
    each of shape (P, M, N) on CPU, and targets_PM is (P, M) on CPU.
    """
    net.eval()
    n = min(len(docs_BL), max_samples)
    rng = np.random.default_rng(PROBE_SEED)
    idx = rng.choice(len(docs_BL), size=n, replace=False)

    K = len(net.transformer.h) + 1
    all_layer_acts: list[list[torch.Tensor]] = [[] for _ in range(K)]
    all_targets: list[torch.Tensor] = []

    for start in range(0, n, PROBE_COLLECT_BATCH_SIZE):
        end = min(start + PROBE_COLLECT_BATCH_SIZE, n)
        tokens_BL = torch.as_tensor(docs_BL[idx[start:end]], dtype=torch.long, device=DEVICE)
        inp_BT = tokens_BL[:, :-1]
        tgt_BT = tokens_BL[:, 1:]

        tok_emb = net.transformer.wte(inp_BT)
        pos = torch.arange(inp_BT.size(1), device=DEVICE)
        pos_emb = net.transformer.wpe(pos)
        x_BTN = net.transformer.drop(tok_emb + pos_emb)

        all_layer_acts[0].append(x_BTN[:, f3_positions, :].float().cpu())
        for bi, block in enumerate(net.transformer.h):
            x_BTN = block(x_BTN)
            all_layer_acts[bi + 1].append(x_BTN[:, f3_positions, :].float().cpu())

        all_targets.append(tgt_BT[:, f3_positions].cpu())

    layer_acts = [torch.cat(chunks, dim=0) for chunks in all_layer_acts]
    targets_PM = torch.cat(all_targets, dim=0)
    return layer_acts, targets_PM


# ---------------------------------------------------------------------------
# Logit lens
# ---------------------------------------------------------------------------


@torch.no_grad()
def logit_lens_accuracy_K(  # noqa: N802
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
        logits_PMV = lm_head(ln_f(acts_PMN))
        acc_K[k] = (logits_PMV.argmax(dim=-1) == targets_dev).float().mean().item()

    return acc_K


# ---------------------------------------------------------------------------
# Learned linear probe
# ---------------------------------------------------------------------------


class LinearProbe(nn.Module):
    """Single linear layer probe from embedding dim to class logits."""

    def __init__(self, n_embd: int, n_classes: int) -> None:
        """Initialise linear probe with given dimensions."""
        super().__init__()
        self.linear = nn.Linear(n_embd, n_classes)

    def forward(self, x_BN: torch.Tensor) -> torch.Tensor:
        """Project input embeddings to class logits."""
        return self.linear(x_BN)


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
    epochs_no_improve = 0
    for epoch in range(LEARNED_PROBE_EPOCHS):
        probe.train()
        loss = F.cross_entropy(probe(train_feats), train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % LEARNED_PROBE_VAL_EVERY == 0 or epoch == LEARNED_PROBE_EPOCHS - 1:
            probe.eval()
            with torch.no_grad():
                val_acc = (probe(val_feats).argmax(dim=-1) == val_labels).float().mean().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
            else:
                epochs_no_improve += LEARNED_PROBE_VAL_EVERY
            if epochs_no_improve >= LEARNED_PROBE_PATIENCE:
                break

    return best_val_acc


def learned_probe_accuracy_K(  # noqa: N802
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


# ---------------------------------------------------------------------------
# Orchestration: probe all layers for both regimes
# ---------------------------------------------------------------------------


def probe_all_layers(  # noqa: PLR0913
    net: nanoGPT,
    other_docs_BL: np.ndarray,
    burst_docs_BL: np.ndarray,
    n_layers: int,
    seq_len: int,
    max_samples: int,
    depth: int,
) -> dict[str, dict[str, list[float]]]:
    """Run both probe methods on both regimes at every layer.

    Returns {method: {"Other": [K floats], "Burst": [K floats]}}.
    """
    f3_pos = get_final_output_positions(seq_len, depth)
    n_embd = net.config.n_embd
    K = n_layers + 1

    results: dict[str, dict[str, list[float]]] = {
        m: {"Other": [0.0] * K, "Burst": [0.0] * K} for m in PROBE_METHODS
    }

    for regime, docs in [("Other", other_docs_BL), ("Burst", burst_docs_BL)]:
        layer_acts, targets_PM = collect_layer_acts_and_targets(net, docs, f3_pos, max_samples)

        ll_acc = logit_lens_accuracy_K(net, layer_acts, targets_PM)
        results["logit_lens"][regime] = ll_acc.tolist()

        lp_acc = learned_probe_accuracy_K(layer_acts, targets_PM, n_embd)
        results["learned_probe"][regime] = lp_acc.tolist()

        for k in range(K):
            layer_name = "emb" if k == 0 else f"L{k - 1}"
            logger.info(
                "      %-4s  %s  logit_lens=%.3f  learned_probe=%.3f",
                layer_name,
                regime,
                ll_acc[k],
                lp_acc[k],
            )

    return results


# ---------------------------------------------------------------------------
# Checkpoint / retrain wrappers
# ---------------------------------------------------------------------------


def probe_from_checkpoints_at_steps(  # noqa: PLR0913
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
    from burst.core.train_utils import load_net  # noqa: PLC0415

    cfg = job["cfg"]
    results_by_step: dict[int, dict] = {}

    available_ckpts: dict[int, str] = {}
    if ckpt_dir.exists():
        for pt_file in ckpt_dir.glob("step_*.pt"):
            step = int(pt_file.stem.split("_")[1])
            available_ckpts[step] = str(pt_file)

    for step in probe_steps:
        if step not in available_ckpts:
            logger.info("    WARNING: no checkpoint for step %s, skipping", step)
            continue
        logger.info("    Loading ckpt step %s...", step)
        net = load_net(cfg, available_ckpts[step])
        net.eval()
        results_by_step[step] = probe_all_layers(
            net, other_docs_BL, burst_docs_BL, n_layers, seq_len, max_samples, depth
        )
        del net
        torch.cuda.empty_cache()

    return results_by_step


def retrain_and_probe_at_steps(  # noqa: PLR0913
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
    """Retrain once and probe at each requested step along the way."""
    from burst.core.train_utils import retrain_with_callbacks  # noqa: PLC0415

    checkpoint_set = set(probe_steps)
    results_by_step: dict[int, dict] = {}

    def on_step(net: nanoGPT, global_step: int, _phase: str) -> None:
        if global_step in checkpoint_set:
            net.eval()
            logger.info("    Probing step %s...", global_step)
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


# ---------------------------------------------------------------------------
# Diff computation
# ---------------------------------------------------------------------------


def compute_diffs(
    all_results: dict[str, dict],
    schedules: list[str],
) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, list[list[float]]]]]:
    """Return per-schedule mean diffs and per-seed diffs for each method.

    Returns (diffs, diffs_per_seed) where values are plain lists for JSON serialisation.
    """
    diffs: dict[str, dict[str, list[float]]] = {}
    diffs_per_seed: dict[str, dict[str, list[list[float]]]] = {}
    for method in PROBE_METHODS:
        diffs[method] = {}
        diffs_per_seed[method] = {}
        for sched in schedules:
            other_curves = []
            burst_curves = []
            for key, val in all_results.items():
                if key.startswith(sched + "_s") and method in val:
                    other_curves.append(np.array(val[method]["Other"], dtype=float))
                    burst_curves.append(np.array(val[method]["Burst"], dtype=float))
            if not other_curves:
                continue
            per_seed = np.array(other_curves) - np.array(burst_curves)
            diffs_per_seed[method][sched] = per_seed.tolist()
            diffs[method][sched] = per_seed.mean(axis=0).tolist()
    return diffs, diffs_per_seed


def compute_diff_in_diffs(
    diffs: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, list[float]]]:
    """Compute pairwise diff-in-diffs between schedules for each method."""
    did: dict[str, dict[str, list[float]]] = {}
    for method in PROBE_METHODS:
        did[method] = {}
        scheds = list(diffs[method].keys())
        for s1, s2 in combinations(scheds, 2):
            d1 = np.array(diffs[method][s1], dtype=float)
            d2 = np.array(diffs[method][s2], dtype=float)
            did[method][f"{s1}_vs_{s2}"] = (d1 - d2).tolist()
    return did


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------


def save_probe_record(  # noqa: PLR0913
    out_dir: Path,
    label: str,
    schedule: str,
    seed: int,
    probe_steps: list[int],
    step_results: dict[int, dict],
) -> Path:
    """Save one label's probe results as JSON, return the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "label": label,
        "schedule": schedule,
        "seed": seed,
        "probe_steps": probe_steps,
        "step_results": {str(step): res for step, res in step_results.items()},
    }
    path = out_dir / f"{label}.json"
    with path.open("w") as f:
        json.dump(record, f, indent=2)
    return path


def load_probe_records(run_dir: str | Path) -> list[dict[str, Any]]:
    """Load per-label probe JSON records from a run directory."""
    from burst.core.train_utils import resolve_run_paths  # noqa: PLC0415

    run_dir = Path(run_dir)
    _, _, results_dir = resolve_run_paths(run_dir)

    records: list[dict[str, Any]] = []
    for probe_dir in (results_dir / NTP_RESULTS_DIRNAME, run_dir / NTP_RESULTS_DIRNAME):
        if not probe_dir.is_dir():
            continue
        for path in sorted(probe_dir.glob("*.json")):
            if path.name == "summary.json":
                continue
            with path.open() as f:
                records.append(json.load(f))
        if records:
            return records
    return records
