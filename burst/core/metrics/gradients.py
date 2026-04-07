"""Post-hoc gradient cosine similarity computation on saved checkpoints.

Runs after training (burst/experiment.py) as a separate pass, loading
checkpoints and computing grad-sim with full GPU utilisation at the
grad_sim_batch_size level.  Checkpoints are kept by default so the
computation can be re-run with different settings.

Usage:
    python burst/grad_sim.py <run_dir>
    python burst/grad_sim.py <run_dir> --n-workers 8 --grad-sim-batch-size 2048
    python burst/grad_sim.py <run_dir> --delete-checkpoints

Dimension key:
    B: batch_size
    L: sequence_length (doc_len)
    N: n_embd (model dimension)
    V: vocab_size
    P: n_params (total parameters, flattened)
    G: gradient vector dimension (same as P)
    E: number of per-example gradient samples (for SNR)
    T: number of token positions
"""

import argparse
import json
import logging
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from einops import rearrange  # noqa: E402

from burst.config import (  # noqa: E402
    DEFAULT_DETERMINISTIC,
    DEFAULT_REPRO_SEED,
    MIN_VECTORS_FOR_SIMILARITY,
    PHASE_BURST,
    PHASE_PRE_BURST,
    PHASE_REVERSION,
    parse_run_config,
)
from burst.core.gpu import gpu_cfg  # noqa: E402
from burst.core.parallel import JobResult, run_job_pool  # noqa: E402
from burst.core.repro import write_repro_manifest  # noqa: E402
from burst.core.train_utils import (  # noqa: E402
    DEVICE,
    _cross_entropy_logits_BTV_targets_BT,
    load_net,
    resolve_logs_dir,
    resolve_results_dir,
)
from burst.rng import get_rng, seed_all  # noqa: E402
from net.nanogpt import nanoGPT  # noqa: E402

warnings.filterwarnings("ignore", message=".*Full backward hook.*no inputs require gradients.*")
NEAR_ZERO = 1e-12

# ---------------------------------------------------------------------------
# Feature flags — comment out any key to skip that metric entirely.
# This controls what is computed in each _worker_main() subprocess.
# ---------------------------------------------------------------------------
GRAD_METRICS: dict[str, bool] = {
    "cosine_global": True,
    "cosine_per_layer": True,
    "pairwise": False,  # disabled for speed
    "grad_norm_ratio": True,
    "grad_rank": True,
    "grad_snr": False,  # disabled for speed
    "conflict_rate": True,
    "token_pos_grad": False,  # disabled for speed
    "grad_attribution": False,
    "grad_projection": True,  # OGD-style: interference vs useful component of g_burst
}

# Number of per-example gradients to sample for SNR estimation.
# Higher = more accurate but slower (each adds one backward pass).
N_SNR_EXAMPLES: int = 16


# ---------------------------------------------------------------------------
# Layer group helpers
# ---------------------------------------------------------------------------


def _flat_grad(net: nanoGPT) -> torch.Tensor:
    """Concatenate all parameter gradients into a single flat vector."""
    grads = [p.grad.detach().view(-1) for p in net.parameters() if p.grad is not None]
    return torch.cat(grads) if grads else torch.zeros(1, device=DEVICE)


def _layer_groups(net: nanoGPT) -> list[tuple[str, list[str]]]:
    """Return ordered (short_name, [param_name, ...]) groups for per-layer grad-sim.

    Groups:
      emb       -- wte + wpe embeddings
      L{i}_ln   -- block i layernorms (ln_1, ln_2)
      L{i}_attn -- block i attention (c_attn, c_proj)
      L{i}_mlp  -- block i MLP (c_fc, c_proj)
      ln_f      -- final layernorm
    LM_head is weight-tied to wte so it is omitted to avoid double-counting.
    """
    groups: list[tuple[str, list[str]]] = []
    all_param_names = {n for n, _ in net.named_parameters()}

    emb_params = [
        n for n in all_param_names if n in ("transformer.wte.weight", "transformer.wpe.weight")
    ]
    if emb_params:
        groups.append(("emb", sorted(emb_params)))

    n_layer = net.config.n_layer
    for i in range(n_layer):
        prefix = f"transformer.h.{i}"
        ln_params = [n for n in all_param_names if n.startswith(f"{prefix}.ln_")]
        attn_params = [n for n in all_param_names if n.startswith(f"{prefix}.attn.")]
        mlp_params = [n for n in all_param_names if n.startswith(f"{prefix}.mlp.")]
        if ln_params:
            groups.append((f"L{i}_ln", sorted(ln_params)))
        if attn_params:
            groups.append((f"L{i}_attn", sorted(attn_params)))
        if mlp_params:
            groups.append((f"L{i}_mlp", sorted(mlp_params)))

    lnf_params = [n for n in all_param_names if n.startswith("transformer.ln_f")]
    if lnf_params:
        groups.append(("ln_f", sorted(lnf_params)))

    return groups


# ---------------------------------------------------------------------------
# Core gradient extraction helpers
# ---------------------------------------------------------------------------


def _grad_vecs_per_layer(
    net: nanoGPT, docs_np: np.ndarray, n_samples: int, layer_groups: list[tuple[str, list[str]]]
) -> dict[str, torch.Tensor]:
    """Run one backward pass and extract per-layer gradient vectors."""
    n = min(n_samples, docs_np.shape[0])
    idx = get_rng().choice(docs_np.shape[0], n, replace=False)
    tokens_BL = torch.as_tensor(docs_np[idx], dtype=torch.long, device=DEVICE)
    inp_BT, tgt_BT = tokens_BL[:, :-1], tokens_BL[:, 1:]
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        logits_BTV = net(inp_BT)
    loss = _cross_entropy_logits_BTV_targets_BT(logits_BTV.float(), tgt_BT)
    loss.backward()

    param_map = dict(net.named_parameters())
    result: dict[str, torch.Tensor] = {}
    for name, pnames in layer_groups:
        grads = []
        for pn in pnames:
            p = param_map.get(pn)
            if p is not None and p.grad is not None:
                grads.append(p.grad.detach().view(-1).float())
        result[name] = torch.cat(grads) if grads else torch.zeros(1, device=DEVICE)
    return result


def _grad_vec_for_docs(net: nanoGPT, docs_np: np.ndarray, n_samples: int) -> torch.Tensor:
    """Compute a flat gradient vector from a random subset of docs."""
    n = min(n_samples, docs_np.shape[0])
    idx = get_rng().choice(docs_np.shape[0], n, replace=False)
    tokens_BL = torch.as_tensor(docs_np[idx], dtype=torch.long, device=DEVICE)
    inp_BT, tgt_BT = tokens_BL[:, :-1], tokens_BL[:, 1:]
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        logits_BTV = net(inp_BT)
    loss = _cross_entropy_logits_BTV_targets_BT(logits_BTV.float(), tgt_BT)
    loss.backward()
    return _flat_grad(net).float()


# ---------------------------------------------------------------------------
# New metric helpers
# ---------------------------------------------------------------------------


def _effective_rank(g_vec: torch.Tensor, param_shape: tuple[int, int]) -> float:
    """Effective rank of a gradient vector reshaped as a matrix.

    Computes exp(H(sigma_hat)) where H is the entropy of the normalised
    singular value distribution.  Range: [1, min(m, n)].
    """
    m, n = param_shape
    g_MN = rearrange(g_vec[: m * n].float(), "(m n) -> m n", m=m, n=n)
    try:
        sv = torch.linalg.svdvals(g_MN)
    except (RuntimeError, torch.linalg.LinAlgError):
        return float("nan")
    sv = sv[sv > 0]
    if sv.numel() == 0:
        return 1.0
    s_hat = sv / sv.sum()
    entropy = -(s_hat * torch.log(s_hat + 1e-12)).sum()
    return float(torch.exp(entropy).item())


def _grad_rank_per_layer(
    net: nanoGPT, docs_np: np.ndarray, n_samples: int, layer_groups: list[tuple[str, list[str]]]
) -> dict[str, float]:
    """Effective rank of the gradient matrix for each layer group (burst data)."""
    vecs = _grad_vecs_per_layer(net, docs_np, n_samples, layer_groups)
    param_map = dict(net.named_parameters())

    result: dict[str, float] = {}
    for name, pnames in layer_groups:
        g_vec = vecs.get(name)
        if g_vec is None:
            result[name] = float("nan")
            continue
        shapes = [
            param_map[pn].shape
            for pn in pnames
            if pn in param_map and len(param_map[pn].shape) == 2  # noqa: PLR2004  # type: ignore[operator]
        ]
        if shapes:
            m, n = shapes[0]
            result[name] = _effective_rank(g_vec, (m, n))
        else:
            result[name] = float("nan")
    return result


def _grad_snr_per_layer(
    net: nanoGPT, docs_np: np.ndarray, n_examples: int, layer_groups: list[tuple[str, list[str]]]
) -> dict[str, float]:
    """Per-layer gradient signal-to-noise ratio across individual examples.

    SNR_l = ||mean_g_l||^2 / mean(||g_i_l - mean_g_l||^2)

    High SNR: all examples push the same direction (coherent, shortcut-like).
    Low SNR: examples push in diverse directions (richer learning signal).

    Uses torch.func.vmap + grad to compute all per-example gradients in one
    vectorised call instead of n_examples sequential backward passes.
    """
    from torch.func import functional_call, grad, vmap  # noqa: PLC0415

    n = min(n_examples, docs_np.shape[0])
    idx = get_rng().choice(docs_np.shape[0], n, replace=False)
    dat = torch.as_tensor(docs_np[idx], dtype=torch.long, device=DEVICE)
    inp_EL = dat[:, :-1]
    tgt_EL = dat[:, 1:]

    params = {k: v.detach() for k, v in net.named_parameters()}
    buffers = {k: v.detach() for k, v in net.named_buffers()}

    def loss_fn(
        params: dict[str, torch.Tensor],
        inp_1L: torch.Tensor,
        tgt_1L: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-example loss for vmap-based gradient computation."""
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            logits_1LV = functional_call(net, (params, buffers), (inp_1L.unsqueeze(0),))
        return _cross_entropy_logits_BTV_targets_BT(logits_1LV, tgt_1L.unsqueeze(0))

    grad_fn = grad(loss_fn)
    per_example_grads = vmap(grad_fn, in_dims=(None, 0, 0))(params, inp_EL, tgt_EL)

    result: dict[str, float] = {}
    for name, pnames in layer_groups:
        grads_list = []
        for pn in pnames:
            g = per_example_grads.get(pn)
            if g is not None:
                grads_list.append(g.float().reshape(n, -1))
        if not grads_list:
            result[name] = float("nan")
            continue
        gs = torch.cat(grads_list, dim=1)  # E x D
        mean_g = gs.mean(dim=0)
        signal = (mean_g.norm() ** 2).item()
        noise = ((gs - mean_g).norm(dim=1) ** 2).mean().item()
        result[name] = signal / (noise + 1e-12)
    return result


def _conflict_rate_per_layer(
    burst_vecs: dict[str, torch.Tensor], other_vecs: dict[str, torch.Tensor]
) -> dict[str, float]:
    """Fraction of parameters where burst and other gradients have opposite signs.

    C_l = (1/|theta_l|) * sum_i 1[sign(g_burst_i) != sign(g_other_i)]

    Range [0, 1].  0.5 = random; >0.5 = systematic conflict.
    """
    result: dict[str, float] = {}
    for name, g_b in burst_vecs.items():
        g_o = other_vecs.get(name)
        if g_o is None or g_b.numel() == 0:
            result[name] = float("nan")
            continue
        conflict = (torch.sign(g_b) != torch.sign(g_o)).float().mean().item()
        result[name] = conflict
    return result


def _grad_projection_metrics(
    g_burst: torch.Tensor,
    g_other: torch.Tensor,
) -> dict[str, float]:
    """OGD-style gradient projection decomposition.

    Decomposes g_burst into:
      g_parallel = projection of g_burst onto g_other direction
                 = (g_burst · g_other / ||g_other||^2) * g_other
      g_perp     = g_burst - g_parallel  (orthogonal residual)

    Returns:
      interference_magnitude: ||g_parallel||  (how much burst step affects other-class params)
      useful_learning:        ||g_perp||       (how much burst step moves model orthogonally)
      interference_ratio:     ||g_parallel|| / ||g_burst||  (= |cos(alpha)|, fraction wasted)
      burst_norm:             ||g_burst||
      other_norm:             ||g_other||

    """
    g_b = g_burst.float()
    g_o = g_other.float()

    norm_o = g_o.norm()
    norm_b = g_b.norm()

    burst_l1 = float(g_b.abs().sum().item())
    burst_linf = float(g_b.abs().max().item())
    other_l1 = float(g_o.abs().sum().item())
    other_linf = float(g_o.abs().max().item())

    if norm_o < NEAR_ZERO or norm_b < NEAR_ZERO:
        return {
            "interference_magnitude": float("nan"),
            "useful_learning": float("nan"),
            "interference_ratio": float("nan"),
            "burst_norm": float(norm_b.item()),
            "other_norm": float(norm_o.item()),
            "burst_l1": burst_l1,
            "burst_linf": burst_linf,
            "other_l1": other_l1,
            "other_linf": other_linf,
        }

    dot = (g_b * g_o).sum()
    proj_coeff = dot / (norm_o**2)

    g_parallel = proj_coeff * g_o
    g_perp = g_b - g_parallel

    interference_mag = g_parallel.norm().item()
    useful_learning = g_perp.norm().item()
    interference_ratio = interference_mag / (norm_b.item() + 1e-12)

    return {
        "interference_magnitude": float(interference_mag),
        "useful_learning": float(useful_learning),
        "interference_ratio": float(interference_ratio),
        "burst_norm": float(norm_b.item()),
        "other_norm": float(norm_o.item()),
        "burst_l1": burst_l1,
        "burst_linf": burst_linf,
        "other_l1": other_l1,
        "other_linf": other_linf,
    }


def _token_pos_grad_norms(net: nanoGPT, docs_np: np.ndarray, n_samples: int) -> list[float]:
    """Per-token-position gradient norm w.r.t. the embedding output.

    Hooks the wte embedding output to capture d(loss)/d(h_t) for each
    position t.  Returns a list of norms of length (doc_len - 1), one per
    input token position.
    """
    n = min(n_samples, docs_np.shape[0])
    idx = get_rng().choice(docs_np.shape[0], n, replace=False)
    dat = torch.as_tensor(docs_np[idx], dtype=torch.long, device=DEVICE)
    inp, tgt = dat[:, :-1], dat[:, 1:]

    emb_grad: list[torch.Tensor] = []

    def _hook(
        _module: torch.nn.Module,
        _grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        """Capture embedding gradient from backward hook."""
        emb_grad.append(grad_output[0].detach().float())

    handle = net.transformer.wte.register_full_backward_hook(_hook)
    net.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        logits = net(inp)
    loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1))
    loss.backward()
    handle.remove()

    if not emb_grad:
        return []
    g_BtN = emb_grad[0]
    norms_T = g_BtN.norm(dim=-1).mean(dim=0)
    net.zero_grad(set_to_none=True)
    return norms_T.cpu().tolist()


def _grad_attribution(
    net: nanoGPT, docs_np: np.ndarray, n_samples: int, prompt_len: int, doc_len: int
) -> dict[str, Any]:
    """Decompose gradient norm by which output token position it comes from.

    For each output position t in [prompt_len, doc_len-1], compute the
    gradient of the per-position loss L_t w.r.t. all parameters, then
    measure ||grad_theta L_t||.

    Returns:
        intermediate_frac: fraction of total grad norm from positions
                           [prompt_len, doc_len-2] (intermediate outputs)
        final_frac:        fraction from position doc_len-1 (final output)
        per_pos:           list of grad norms for each output position

    """
    n = min(n_samples, docs_np.shape[0])
    idx = get_rng().choice(docs_np.shape[0], n, replace=False)
    dat = torch.as_tensor(docs_np[idx], dtype=torch.long, device=DEVICE)
    inp, tgt = dat[:, :-1], dat[:, 1:]

    output_positions = list(range(prompt_len, doc_len - 1))
    if not output_positions:
        return {"intermediate_frac": float("nan"), "final_frac": float("nan"), "per_pos": []}

    pos_norms: list[float] = []
    params = [p for p in net.parameters() if p.requires_grad]

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        logits_BTV = net(inp).float()

    for i, t in enumerate(output_positions):
        net.zero_grad(set_to_none=True)
        retain = i < len(output_positions) - 1
        loss_t = F.cross_entropy(logits_BTV[:, t, :], tgt[:, t])
        loss_t.backward(retain_graph=retain)
        g_norm = (
            sum(p.grad.detach().norm().item() ** 2 for p in params if p.grad is not None) ** 0.5
        )
        pos_norms.append(g_norm)

    net.zero_grad(set_to_none=True)

    total = sum(pos_norms) + 1e-12
    n_intermediate = len(output_positions) - 1
    intermediate_norm = sum(pos_norms[:n_intermediate])
    final_norm = pos_norms[-1] if pos_norms else 0.0

    return {
        "intermediate_frac": intermediate_norm / total,
        "final_frac": final_norm / total,
        "per_pos": pos_norms,
    }


# ---------------------------------------------------------------------------
# Public metric computation functions
# ---------------------------------------------------------------------------


def compute_grad_cosine_sim(
    net: nanoGPT, docs_burst_BL: np.ndarray, docs_other_BL: np.ndarray, n_samples: int
) -> dict[str, float]:
    """Compute global burst-vs-other gradient cosine similarity."""
    net.train()
    net.zero_grad(set_to_none=True)
    g_burst = _grad_vec_for_docs(net, docs_burst_BL, n_samples=n_samples)
    net.zero_grad(set_to_none=True)
    g_other = _grad_vec_for_docs(net, docs_other_BL, n_samples=n_samples)
    cos_sim = F.cosine_similarity(g_burst.unsqueeze(0), g_other.unsqueeze(0)).item()
    net.zero_grad(set_to_none=True)
    return {"burst_vs_other": cos_sim}


def compute_grad_cosine_sim_per_layer(
    net: nanoGPT, docs_burst_BL: np.ndarray, docs_other_BL: np.ndarray, n_samples: int
) -> dict[str, Any]:
    """Compute burst-vs-other cosine similarity separately for each layer group."""
    layer_groups = _layer_groups(net)
    net.train()

    net.zero_grad(set_to_none=True)
    burst_vecs = _grad_vecs_per_layer(net, docs_burst_BL, n_samples, layer_groups)
    net.zero_grad(set_to_none=True)
    other_vecs = _grad_vecs_per_layer(net, docs_other_BL, n_samples, layer_groups)
    net.zero_grad(set_to_none=True)

    per_layer: dict[str, float] = {}
    for name, _ in layer_groups:
        g_b = burst_vecs[name]
        g_o = other_vecs[name]
        sim = F.cosine_similarity(g_b.unsqueeze(0), g_o.unsqueeze(0)).item()
        per_layer[name] = sim

    layer_names = [name for name, _ in layer_groups]
    return {
        "per_layer": per_layer,
        "layer_names": layer_names,
        "burst_vecs": burst_vecs,
        "other_vecs": other_vecs,
    }


def compute_pairwise_grad_sim(  # noqa: C901, PLR0912
    net: nanoGPT,
    task_docs: dict[Any, np.ndarray],
    n_samples: int,
    burst_pos: int,
    n_a: int,
) -> dict[str, Any]:
    """Pairwise grad cosine sim with principled task grouping.

    Groups:
      BURST       -- all burst-class tasks pooled
      O_F{i}      -- other-class tasks grouped by function at burst_pos
                     (bijection indices (burst_pos-1)*n_a+1 .. burst_pos*n_a)
      ALL_OTHER   -- all other-class tasks pooled
      ALL_DATA    -- everything pooled
    """
    from burst.config import CLASS_BURST  # noqa: PLC0415

    net.train()

    depth = len(next(iter(task_docs))) - 1
    burst_slot = depth - burst_pos
    bp_fn_indices = list(range(burst_slot * n_a + 1, (burst_slot + 1) * n_a + 1))

    group_docs: dict[str, list[np.ndarray]] = {"BURST": []}
    for fi in bp_fn_indices:
        group_docs[f"O_F{fi}"] = []

    task_tuple_idx = burst_slot + 1
    for task, docs in task_docs.items():
        if docs.shape[0] == 0:
            continue
        if task[0] == CLASS_BURST:
            group_docs["BURST"].append(docs)
        elif task_tuple_idx >= len(task):
            continue
        else:
            fn_at_bp = task[task_tuple_idx]
            key = f"O_F{fn_at_bp}"
            if key in group_docs:
                group_docs[key].append(docs)

    other_sub_docs = []
    for fi in bp_fn_indices:
        other_sub_docs.extend(group_docs[f"O_F{fi}"])
    group_docs["ALL_OTHER"] = list(other_sub_docs)
    group_docs["ALL_DATA"] = group_docs["BURST"] + group_docs["ALL_OTHER"]

    label_order = ["BURST"]
    label_order += [f"O_F{fi}" for fi in bp_fn_indices]
    label_order += ["ALL_OTHER", "ALL_DATA"]

    grad_vecs = []
    for label in label_order:
        doc_list = group_docs[label]
        if doc_list:
            pooled = np.concatenate(doc_list)
            net.zero_grad(set_to_none=True)
            grad_vecs.append(_grad_vec_for_docs(net, pooled, n_samples=n_samples))
        else:
            grad_vecs.append(None)

    n = len(label_order)
    matrix = np.eye(n)
    valid = [(i, v) for i, v in enumerate(grad_vecs) if v is not None]
    if len(valid) >= MIN_VECTORS_FOR_SIMILARITY:
        G = torch.stack([v for _, v in valid])
        G_norm = F.normalize(G, dim=1)
        sim = (G_norm @ G_norm.T).cpu().numpy()
        for ri, (i, _) in enumerate(valid):
            for rj, (j, _) in enumerate(valid):
                matrix[i, j] = sim[ri, rj]

    net.zero_grad(set_to_none=True)
    return {
        "matrix": matrix.tolist(),
        "labels": label_order,
        "n_burst": 1,
        "n_other_sub": n_a,
        "n_other": 1,
        "n_all": 1,
    }


# ---------------------------------------------------------------------------
# Worker subprocess
# ---------------------------------------------------------------------------


def _worker_main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Run a single gradient-metric job in a subprocess."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--job-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--grad-sim-batch-size", type=int, required=True)
    args = parser.parse_args()

    with Path(args.job_path).open("rb") as f:
        job = pickle.load(f)  # noqa: S301
    with Path(args.data_path).open("rb") as f:
        target_pool, bg_pool = pickle.load(f)  # noqa: S301

    cfg = job["cfg"]
    seed_all(int(cfg["seed"]), deterministic=bool(job["deterministic"]))
    ckpt_path = job["ckpt_path"]
    step = job["step"]
    phase = job["phase"]
    is_pairwise = job["is_pairwise"]
    burst_pos_val = job["burst_pos"]
    n_a = job["n_a"]
    prompt_len = job["prompt_len"]
    doc_len = job["doc_len"]
    gs_bs = args.grad_sim_batch_size

    net = load_net(cfg, str(ckpt_path))

    burst_docs_all = np.concatenate(list(target_pool.values())) if target_pool else None
    other_docs_all = np.concatenate(list(bg_pool.values())) if bg_pool else None

    result = {
        "label": job["label"],
        "parent_label": job["parent_label"],
        "step": step,
        "phase": phase,
    }

    if burst_docs_all is None or other_docs_all is None:
        with Path(args.output_path).open("wb") as f:
            pickle.dump(result, f)
        return

    layer_groups = _layer_groups(net)
    net.train()

    # --- cosine_global + grad_projection ---
    # Both metrics share the same two backward passes, so compute together.
    need_global_grads = GRAD_METRICS.get("cosine_global", True) or GRAD_METRICS.get(
        "grad_projection", True
    )
    if need_global_grads:
        net.zero_grad(set_to_none=True)
        g_burst_flat = _grad_vec_for_docs(net, burst_docs_all, n_samples=gs_bs)
        net.zero_grad(set_to_none=True)
        g_other_flat = _grad_vec_for_docs(net, other_docs_all, n_samples=gs_bs)

        if GRAD_METRICS.get("cosine_global", True):
            result["burst_vs_other"] = F.cosine_similarity(
                g_burst_flat.unsqueeze(0), g_other_flat.unsqueeze(0)
            ).item()

        if GRAD_METRICS.get("grad_projection", True):
            result["grad_projection"] = _grad_projection_metrics(g_burst_flat, g_other_flat)

        net.zero_grad(set_to_none=True)

    # --- cosine_per_layer + grad_norm_ratio + conflict_rate ---
    # These all share the same two backward passes, so we compute together.
    need_layer_vecs = (
        GRAD_METRICS.get("cosine_per_layer", True)
        or GRAD_METRICS.get("grad_norm_ratio", True)
        or GRAD_METRICS.get("conflict_rate", True)
    )
    if need_layer_vecs:
        net.zero_grad(set_to_none=True)
        burst_vecs = _grad_vecs_per_layer(net, burst_docs_all, gs_bs, layer_groups)
        net.zero_grad(set_to_none=True)
        other_vecs = _grad_vecs_per_layer(net, other_docs_all, gs_bs, layer_groups)
        net.zero_grad(set_to_none=True)

        if GRAD_METRICS.get("cosine_per_layer", True):
            per_layer: dict[str, float] = {}
            for name, _ in layer_groups:
                g_b = burst_vecs[name]
                g_o = other_vecs[name]
                per_layer[name] = F.cosine_similarity(g_b.unsqueeze(0), g_o.unsqueeze(0)).item()
            result["per_layer_sim"] = per_layer
            result["layer_names"] = [n for n, _ in layer_groups]

        if GRAD_METRICS.get("grad_norm_ratio", True):
            ratio: dict[str, float] = {}
            for name, _ in layer_groups:
                norm_b = burst_vecs[name].norm().item()
                norm_o = other_vecs[name].norm().item()
                ratio[name] = norm_b / (norm_o + 1e-12)
            result["grad_norm_ratio"] = ratio

        if GRAD_METRICS.get("conflict_rate", True):
            result["conflict_rate"] = _conflict_rate_per_layer(burst_vecs, other_vecs)

    # --- grad_rank ---
    if GRAD_METRICS.get("grad_rank", True):
        net.zero_grad(set_to_none=True)
        result["grad_rank"] = _grad_rank_per_layer(net, burst_docs_all, gs_bs, layer_groups)

    # --- grad_snr ---
    if GRAD_METRICS.get("grad_snr", True):
        net.zero_grad(set_to_none=True)
        result["grad_snr"] = _grad_snr_per_layer(net, burst_docs_all, N_SNR_EXAMPLES, layer_groups)

    # --- token_pos_grad ---
    if GRAD_METRICS.get("token_pos_grad", True):
        net.zero_grad(set_to_none=True)
        result["token_pos_grad_norms"] = _token_pos_grad_norms(net, burst_docs_all, n_samples=gs_bs)

    # --- grad_attribution ---
    if GRAD_METRICS.get("grad_attribution", True):
        net.zero_grad(set_to_none=True)
        result["grad_attribution"] = _grad_attribution(
            net, burst_docs_all, n_samples=gs_bs, prompt_len=prompt_len, doc_len=doc_len
        )

    # --- pairwise ---
    if is_pairwise and GRAD_METRICS.get("pairwise", True):
        task_docs = {**target_pool, **bg_pool}
        snap = compute_pairwise_grad_sim(
            net,
            task_docs,
            n_samples=gs_bs,
            burst_pos=burst_pos_val,
            n_a=n_a,
        )
        result["pairwise"] = snap

    with Path(args.output_path).open("wb") as f:
        pickle.dump(result, f)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _resolve_run_paths(run_dir: Path) -> tuple[Path, Path, Path, Path]:
    """Return (config_path, data_path, ckpt_root, gs_out_dir) for a run directory."""
    results_dir = resolve_results_dir(run_dir)
    logs_dir = resolve_logs_dir(run_dir)

    config_path = (
        (results_dir / "config.json")
        if (results_dir / "config.json").exists()
        else (run_dir / "config.json")
    )
    data_path = (
        (logs_dir / "_data.pkl") if (logs_dir / "_data.pkl").exists() else (run_dir / "_data.pkl")
    )
    ckpt_root = (
        (logs_dir / "checkpoints")
        if (logs_dir / "checkpoints").exists()
        else (run_dir / "checkpoints")
    )
    gs_out_dir = results_dir / "grad_cosine_sim"
    gs_out_dir.mkdir(parents=True, exist_ok=True)
    return config_path, data_path, ckpt_root, gs_out_dir


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Orchestrate gradient metric computation across all checkpoints."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--grad-sim-batch-size", type=int, default=None)
    parser.add_argument("--delete-checkpoints", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_REPRO_SEED)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DETERMINISTIC,
    )
    parser.add_argument("--note", type=str, default="")
    args = parser.parse_args()
    seed_all(args.seed, deterministic=args.deterministic)

    run_dir = args.run_dir
    manifest_path = write_repro_manifest(
        run_dir,
        mode="gradients",
        seed=args.seed,
        deterministic=args.deterministic,
        cli_args={
            "run_dir": args.run_dir,
            "n_workers": args.n_workers,
            "grad_sim_batch_size": args.grad_sim_batch_size,
            "delete_checkpoints": args.delete_checkpoints,
        },
        note=args.note,
    )
    config_path, data_path, ckpt_root, gs_out_dir = _resolve_run_paths(run_dir)
    with config_path.open() as f:
        run_cfg = json.load(f)

    rc = parse_run_config(run_cfg)
    base_cfg, depth, burst_pos, n_a = rc["base_cfg"], rc["depth"], rc["burst_pos"], rc["n_a"]
    gs_bs = args.grad_sim_batch_size or base_cfg["grad_sim_batch_size"]
    P = base_cfg["pre_burst_steps"]
    T = base_cfg["total_steps"]
    U = base_cfg["reversion_steps"]
    task_info = run_cfg["task_info"]
    prompt_len = task_info["prompt_len"]
    doc_len = task_info["doc_len"]

    pairwise_global_steps = {P, P + T // 2, P + T - 1, P + T + U // 2, P + T + U - 1}

    if not ckpt_root.exists():
        logger.info("No checkpoints directory in %s, nothing to do.", run_dir)
        return

    logs_dir = resolve_logs_dir(run_dir)
    pkl_root = logs_dir if logs_dir.exists() else run_dir

    job_entries = run_cfg["jobs"]

    n_workers = args.n_workers or gpu_cfg.gradsim_workers
    logger.info("%s", gpu_cfg.summary())
    logger.info("Grad-sim: batch_size=%d, workers=%d", gs_bs, n_workers)
    active = [k for k, v in GRAD_METRICS.items() if v]
    logger.info("Active metrics: %s", active)

    jobs = []
    for j in job_entries:
        label = j["label"]
        ckpt_dir = ckpt_root / label
        if not ckpt_dir.exists():
            continue

        result_path = pkl_root / f"{label}.pkl"
        if not result_path.exists():
            continue
        with result_path.open("rb") as f:
            result = pickle.load(f)  # noqa: S301
        cfg = result["config"]

        for pt_file in sorted(ckpt_dir.glob("step_*.pt")):
            step = int(pt_file.stem.split("_")[1])
            if step < P:
                phase = PHASE_PRE_BURST
            elif step < P + T:
                phase = PHASE_BURST
            else:
                phase = PHASE_REVERSION
            is_pairwise = step in pairwise_global_steps
            jobs.append(
                {
                    "label": f"{label}_step{step}",
                    "parent_label": label,
                    "step": step,
                    "phase": phase,
                    "is_pairwise": is_pairwise,
                    "ckpt_path": str(pt_file),
                    "cfg": cfg,
                    "depth": depth,
                    "burst_pos": burst_pos,
                    "n_a": n_a,
                    "prompt_len": prompt_len,
                    "doc_len": doc_len,
                    "deterministic": args.deterministic,
                }
            )

    if not jobs:
        logger.info("No checkpoint jobs found.")
        return

    logger.info("Jobs: %d checkpoints across %d labels", len(jobs), len(job_entries))

    with data_path.open("rb") as f:
        target_pool, bg_pool, *_ = pickle.load(f)  # noqa: S301

    worker_script = str(Path(__file__))

    def build_cmd(script: str, job_path: str, data_path_tmp: str, output_path: str) -> list[str]:
        """Build the subprocess command for a gradient worker."""
        return [
            sys.executable,
            script,
            "--worker",
            "--job-path",
            job_path,
            "--data-path",
            data_path_tmp,
            "--output-path",
            output_path,
            "--grad-sim-batch-size",
            str(gs_bs),
        ]

    def on_done(jr: JobResult, n_done: int, n_total: int) -> None:
        """Log failures only — progress bar handles the rest."""
        if not jr.success:
            logger.warning("  [%d/%d] %s: FAIL: %s", n_done, n_total, jr.label, jr.error[:80])

    results = run_job_pool(
        jobs=jobs,
        worker_script=worker_script,
        build_cmd=build_cmd,
        on_done=on_done,
        n_workers=n_workers,
        data_payload=(target_pool, bg_pool),
        poll_interval=1.0,
        tmp_prefix="grad_sim_",
    )

    _PROJ_KEYS = (
        "interference_magnitude",
        "useful_learning",
        "interference_ratio",
        "burst_norm",
        "other_norm",
    )

    per_label: dict[str, dict] = {}
    for jr in results:
        if not jr.success:
            continue
        d = jr.data
        parent = d["parent_label"]
        if parent not in per_label:
            per_label[parent] = {
                "grad_sim_log": {
                    "step": [],
                    "phase": [],
                    "burst_vs_other": [],
                    "per_layer": {},
                    "grad_norm_ratio": {},
                    "grad_rank": {},
                    "grad_snr": {},
                    "conflict_rate": {},
                    "token_pos_grad_norms": [],
                    "grad_attribution": {"intermediate_frac": [], "final_frac": [], "per_pos": []},
                    "grad_projection": {k: [] for k in _PROJ_KEYS},
                },
                "pairwise_snapshots": [],
            }
        gsl = per_label[parent]["grad_sim_log"]

        if "burst_vs_other" in d:
            gsl["step"].append(d["step"])
            gsl["phase"].append(d["phase"])
            gsl["burst_vs_other"].append(d["burst_vs_other"])

        if "per_layer_sim" in d:
            for layer_name, sim_val in d["per_layer_sim"].items():
                gsl["per_layer"].setdefault(layer_name, []).append(sim_val)
            if "layer_names" not in gsl:
                gsl["layer_names"] = d.get("layer_names", [])

        for key in ("grad_norm_ratio", "grad_rank", "grad_snr", "conflict_rate"):
            if key in d:
                for layer_name, val in d[key].items():
                    gsl[key].setdefault(layer_name, []).append(val)

        if "token_pos_grad_norms" in d:
            gsl["token_pos_grad_norms"].append(d["token_pos_grad_norms"])

        if "grad_attribution" in d:
            attr = d["grad_attribution"]
            gsl["grad_attribution"]["intermediate_frac"].append(
                attr.get("intermediate_frac", float("nan"))
            )
            gsl["grad_attribution"]["final_frac"].append(attr.get("final_frac", float("nan")))
            gsl["grad_attribution"]["per_pos"].append(attr.get("per_pos", []))

        if "grad_projection" in d:
            proj = d["grad_projection"]
            for k in _PROJ_KEYS:
                gsl["grad_projection"][k].append(proj.get(k, float("nan")))

        if "pairwise" in d:
            snap = d["pairwise"]
            snap["step"] = d["step"]
            snap["phase"] = d["phase"]
            per_label[parent]["pairwise_snapshots"].append(snap)

    for entry in per_label.values():
        gsl = entry["grad_sim_log"]
        if not gsl["step"]:
            continue
        order = np.argsort(gsl["step"])
        gsl["step"] = np.array(gsl["step"])[order].tolist()
        gsl["phase"] = np.array(gsl["phase"])[order].tolist()
        gsl["burst_vs_other"] = np.array(gsl["burst_vs_other"])[order].tolist()

        for key in ("per_layer", "grad_norm_ratio", "grad_rank", "grad_snr", "conflict_rate"):
            for layer_name in gsl[key]:
                vals = gsl[key][layer_name]
                if len(vals) == len(order):
                    gsl[key][layer_name] = np.array(vals)[order].tolist()

        if gsl["token_pos_grad_norms"] and len(gsl["token_pos_grad_norms"]) == len(order):
            gsl["token_pos_grad_norms"] = np.array(gsl["token_pos_grad_norms"], dtype=object)[
                order
            ].tolist()

        for sub_key in ("intermediate_frac", "final_frac", "per_pos"):
            vals = gsl["grad_attribution"][sub_key]
            if len(vals) == len(order):
                gsl["grad_attribution"][sub_key] = np.array(vals, dtype=object)[order].tolist()

        for proj_key in _PROJ_KEYS:
            vals = gsl["grad_projection"][proj_key]
            if len(vals) == len(order):
                gsl["grad_projection"][proj_key] = np.array(vals, dtype=float)[order].tolist()

        entry["pairwise_snapshots"].sort(key=lambda s: s["step"])

    for j in job_entries:
        label = j["label"]
        if label not in per_label:
            continue
        entry = per_label[label]
        gsl = entry["grad_sim_log"]
        record = {
            "schedule": j["schedule"],
            "seed": j["seed"],
            "label": label,
            "grad_sim_batch_size": gs_bs,
            "grad_sim_log": gsl,
            "layer_names": gsl.get("layer_names", []),
            "pairwise_snapshots": entry["pairwise_snapshots"],
            "grad_projection_log": gsl.get("grad_projection", {}),
        }
        with (gs_out_dir / f"{label}.json").open("w") as f:
            json.dump(record, f)

    grad_logs_dir = resolve_logs_dir(run_dir)
    all_results_path = (
        (grad_logs_dir / "all_results.pkl")
        if (grad_logs_dir / "all_results.pkl").exists()
        else (run_dir / "all_results.pkl")
    )
    if all_results_path.exists():
        with all_results_path.open("rb") as f:
            all_results = pickle.load(f)  # noqa: S301
        for r in all_results:
            label = r["label"]
            if label in per_label:
                r["grad_sim_log"] = per_label[label]["grad_sim_log"]
                r["pairwise_snapshots"] = per_label[label]["pairwise_snapshots"]
        with all_results_path.open("wb") as f:
            pickle.dump(all_results, f)
        logger.info("Updated all_results.pkl with grad-sim data for %d labels", len(per_label))

    if args.delete_checkpoints:
        import shutil  # noqa: PLC0415

        shutil.rmtree(ckpt_root)
        logger.info("Cleaned up checkpoints")

    n_success = sum(result.success for result in results)
    logger.info("Grad-sim done: %d labels, %d/%d ok", len(per_label), n_success, len(results))
    logger.info("repro_manifest: %s", manifest_path)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _worker_main()
    else:
        main()
