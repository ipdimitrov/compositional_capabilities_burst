"""Activation Difference Lens (ADL) analysis.

Implements the ADL metric from the Narrow Fine-Tuning paper:
  delta_l = mean_x_in_other [ h^checkpoint_l(x) - h^pre_burst_l(x) ]

This measures the global activation bias introduced by the burst phase on
other-class inputs.  Applying the unembedding matrix (Logit Lens) to delta_l
reveals whether the bias encodes burst-relevant tokens — a direct test of the
wrapper hypothesis.

Additionally implements the causal ablation: project delta out of activations
during a burst-class forward pass and measure the accuracy drop.

Usage:
    python burst/adl.py <run_dir>
    python burst/adl.py <run_dir> --n-workers 4 --adl-batch-size 512

Dimension key:
    B: batch_size
    L: sequence_length (doc_len)
    N: n_embd (model dimension)
    K: n_layers + 1 (embedding + each transformer block)
    T: n_token_positions (= L - 1)
    V: vocab_size
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch

from burst.config import PHASE_BURST, PHASE_PRE_BURST, PHASE_REVERSION, parse_run_config
from burst.core.gpu import gpu_cfg
from burst.core.parallel import JobResult, run_job_pool
from burst.core.train_utils import load_net
from burst.dev.probe import collect_activations_KPTN

if TYPE_CHECKING:
    from net.nanogpt import nanoGPT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_rng = np.random.default_rng()

ADL_ENABLED = False


@torch.no_grad()
def compute_delta_KTN(  # noqa: N802
    net_checkpoint: nanoGPT,
    net_pre_burst: nanoGPT,
    other_docs_BL: np.ndarray,
    n_samples: int,
) -> torch.Tensor:
    """Compute mean activation difference delta_KTN on other-class data.

    delta_KTN[k, t, :] = mean over other-class docs of
        (h^checkpoint_k(x)_t  -  h^pre_burst_k(x)_t)

    Returns tensor of shape (K, T, N) on CPU.
    """
    n = min(n_samples, other_docs_BL.shape[0])
    idx = _rng.choice(other_docs_BL.shape[0], n, replace=False)
    docs_BL = other_docs_BL[idx]

    acts_ckpt = collect_activations_KPTN(net_checkpoint, docs_BL)
    acts_pre = collect_activations_KPTN(net_pre_burst, docs_BL)

    K = len(acts_ckpt)
    return torch.stack([(acts_ckpt[k] - acts_pre[k]).mean(dim=0) for k in range(K)])


@torch.no_grad()
def logit_lens_readability(
    net: nanoGPT,
    delta_KTN: torch.Tensor,
    burst_token_ids: list[int],
    top_k: int = 10,
) -> dict:
    """Apply unembedding to delta and measure burst-token readability.

    For each layer k and token position t, projects delta_KTN[k, t, :]
    through the unembedding matrix (lm_head / wte weight-tied) and checks
    whether burst-relevant token IDs appear in the top-k predicted tokens.

    Returns:
        readability_KT: (K, T) array -- fraction of top_k that are burst tokens
        top_tokens_KT: (K, T) list of top-k token ids at each position
        mean_burst_rank_KT: (K, T) mean rank of burst tokens in logit ordering

    """
    K, T, _N = delta_KTN.shape
    unembed_VN = net.transformer.wte.weight.detach().float()

    delta_KTN_dev = delta_KTN.to(DEVICE)
    logits_KTV = torch.einsum("ktn,vn->ktv", delta_KTN_dev, unembed_VN)

    V = logits_KTV.shape[-1]
    burst_set = set(burst_token_ids)

    readability_KT = np.zeros((K, T))
    mean_burst_rank_KT = np.full((K, T), float(V))

    for k in range(K):
        for t in range(T):
            logits_V = logits_KTV[k, t]
            sorted_ids = torch.argsort(logits_V, descending=True).cpu().tolist()
            top_ids = sorted_ids[:top_k]
            readability_KT[k, t] = sum(1 for tid in top_ids if tid in burst_set) / top_k
            ranks = [i for i, tid in enumerate(sorted_ids) if tid in burst_set]
            if ranks:
                mean_burst_rank_KT[k, t] = float(np.mean(ranks))

    return {
        "readability_KT": readability_KT,
        "mean_burst_rank_KT": mean_burst_rank_KT,
    }


@torch.no_grad()
def causal_ablation_accuracy(
    net: nanoGPT,
    delta_KTN: torch.Tensor,
    burst_docs_BL: np.ndarray,
    prompt_len: int,
    n_samples: int,
) -> dict:
    """Project out delta from activations and measure burst accuracy drop.

    For each layer, hooks the residual stream to subtract the component of
    the activation along delta_l (the mean activation bias direction).
    Measures free-generation accuracy on burst-class data before and after
    ablation.

    Returns:
        acc_baseline: burst accuracy without ablation
        acc_ablated_K: (K,) array -- burst accuracy after ablating layer k
        acc_drop_K: (K,) array -- accuracy drop from ablating layer k

    """
    n = min(n_samples, burst_docs_BL.shape[0])
    idx = _rng.choice(burst_docs_BL.shape[0], n, replace=False)
    docs_BL = burst_docs_BL[idx]

    acc_baseline = _free_gen_accuracy(net, docs_BL, prompt_len)

    K = delta_KTN.shape[0]
    acc_ablated_K = np.zeros(K)

    for k in range(K):
        acc_ablated_K[k] = _free_gen_accuracy_with_ablation(
            net, docs_BL, prompt_len, delta_KTN, ablate_layer=k
        )

    acc_drop_K = acc_baseline - acc_ablated_K
    return {
        "acc_baseline": acc_baseline,
        "acc_ablated_K": acc_ablated_K.tolist(),
        "acc_drop_K": acc_drop_K.tolist(),
    }


@torch.no_grad()
def _free_gen_accuracy(net: nanoGPT, docs_BL: np.ndarray, prompt_len: int) -> float:
    net.eval()
    docs_t = torch.as_tensor(docs_BL, dtype=torch.long, device=DEVICE)
    _B, L = docs_t.shape
    target_B6 = docs_t[:, -6:]
    generated = net.generate(docs_t[:, :prompt_len], L - prompt_len)
    return (generated[:, -6:] == target_B6).all(dim=1).float().mean().item()


@torch.no_grad()
def _free_gen_accuracy_with_ablation(
    net: nanoGPT,
    docs_BL: np.ndarray,
    prompt_len: int,
    delta_KTN: torch.Tensor,
    ablate_layer: int,
) -> float:
    """Project out delta at ablate_layer and measure free-gen accuracy.

    For layer k, at every token position t, we subtract the component of the
    residual stream along delta_KTN[k, t, :] (normalised direction).
    """
    net.eval()
    docs_t = torch.as_tensor(docs_BL, dtype=torch.long, device=DEVICE)
    _B, L = docs_t.shape
    target_B6 = docs_t[:, -6:]

    delta_TN = delta_KTN[ablate_layer].to(DEVICE).float()
    norms_T = delta_TN.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    delta_unit_TN = delta_TN / norms_T

    def _ablate_hook(
        _module: object, _input: object, output: object,
    ) -> object:
        if isinstance(output, tuple):
            x_raw, rest = output[0], output[1:]
        else:
            x_raw, rest = output, None
        x_BTN = x_raw.float()
        T_cur = x_BTN.shape[1]
        T_delta = delta_unit_TN.shape[0]
        T_use = min(T_cur, T_delta)
        d_TN = delta_unit_TN[:T_use]
        proj = (
            torch.einsum("btn,tn->bt", x_BTN[:, :T_use], d_TN).unsqueeze(-1)
            * d_TN.unsqueeze(0)
        )
        x_BTN[:, :T_use] -= proj
        x_BTN = x_BTN.to(x_raw.dtype)
        if rest is not None:
            return (x_BTN, *rest)
        return x_BTN

    if ablate_layer == 0:
        handle = net.transformer.drop.register_forward_hook(_ablate_hook)
    else:
        handle = net.transformer.h[ablate_layer - 1].register_forward_hook(_ablate_hook)

    try:
        generated = net.generate(docs_t[:, :prompt_len], L - prompt_len)
    finally:
        handle.remove()

    return (generated[:, -6:] == target_B6).all(dim=1).float().mean().item()


def _burst_token_ids(cfg: dict, n_a: int, depth: int) -> list[int]:
    """Return token IDs associated with the burst function b*.

    Vocab layout: X0..X{n_alphabets-1}, then F0..F{n_a*depth+1}, then specials.
    b* = F{n_a*depth+1} (last function token, one per position x n_a + 1).

    We return the function token for b* plus all value tokens (X0..X9),
    since burst accuracy is measured on output value tokens.
    """
    n_alphabets = cfg.get("n_alphabets", 10)
    vocab_size = cfg.get("vocab_size", 128)

    special_count = 3
    alphabet_start = special_count
    func_start = alphabet_start + n_alphabets

    burst_func_id = func_start + n_a * depth + 1
    value_ids = list(range(alphabet_start, alphabet_start + n_alphabets))

    ids = [burst_func_id, *value_ids]
    return [i for i in ids if i < vocab_size]


def _worker_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--job-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--adl-batch-size", type=int, required=True)
    args = parser.parse_args()

    with Path(args.job_path).open("rb") as f:
        job = pickle.load(f)  # noqa: S301
    with Path(args.data_path).open("rb") as f:
        other_docs_BL, burst_docs_BL, prompt_len = pickle.load(f)  # noqa: S301

    cfg = job["cfg"]
    n_a = job["n_a"]
    depth = job["depth"]
    ckpt_path = job["ckpt_path"]
    pre_burst_ckpt = job["pre_burst_ckpt"]
    step = job["step"]
    phase = job["phase"]
    bs = args.adl_batch_size

    net_checkpoint = load_net(cfg, ckpt_path)
    net_pre_burst = load_net(cfg, pre_burst_ckpt)

    delta_KTN = compute_delta_KTN(net_checkpoint, net_pre_burst, other_docs_BL, n_samples=bs)

    burst_ids = _burst_token_ids(cfg, n_a, depth)
    readability = logit_lens_readability(net_checkpoint, delta_KTN, burst_ids)

    ablation = causal_ablation_accuracy(
        net_checkpoint, delta_KTN, burst_docs_BL, prompt_len, n_samples=bs
    )

    delta_norm_K = delta_KTN.norm(dim=(1, 2)).tolist()

    result = {
        "label": job["label"],
        "parent_label": job["parent_label"],
        "step": step,
        "phase": phase,
        "delta_norm_K": delta_norm_K,
        "readability_KT": readability["readability_KT"].tolist(),
        "mean_burst_rank_KT": readability["mean_burst_rank_KT"].tolist(),
        "acc_baseline": ablation["acc_baseline"],
        "acc_ablated_K": ablation["acc_ablated_K"],
        "acc_drop_K": ablation["acc_drop_K"],
    }

    with Path(args.output_path).open("wb") as f:
        pickle.dump(result, f)


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Run ADL analysis across all checkpoints in a training run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--adl-batch-size", type=int, default=256)
    args = parser.parse_args()

    run_dir = args.run_dir
    from burst.core.train_utils import resolve_run_paths  # noqa: PLC0415

    cfg_path, logs_dir, _ = resolve_run_paths(run_dir)
    with cfg_path.open() as f:
        run_cfg = json.load(f)

    rc = parse_run_config(run_cfg)
    base_cfg = rc["base_cfg"]
    depth = rc["depth"]
    _burst_pos = rc["burst_pos"]
    n_a = rc["n_a"]
    P = base_cfg.get("pre_burst_steps", 0)
    T = base_cfg["total_steps"]
    base_cfg["reversion_steps"]

    ckpt_root = logs_dir / "checkpoints"
    if not ckpt_root.exists():
        print(  # noqa: T201
            f"No checkpoints directory in {logs_dir}, nothing to do.",
            flush=True,
        )
        return

    with (logs_dir / "_data.pkl").open("rb") as f:
        target_pool, bg_pool, _, _, _ = pickle.load(f)  # noqa: S301

    other_docs_BL = np.concatenate(list(bg_pool.values()))
    burst_docs_BL = np.concatenate(list(target_pool.values()))

    prompt_len = run_cfg.get("task_info", {}).get("prompt_len")
    if prompt_len is None:
        msg = "prompt_len not found in config.json task_info"
        raise ValueError(msg)

    job_entries = run_cfg["jobs"]
    {
        **base_cfg,
        "vocab_size": base_cfg.get("vocab_size", 128),
        "context_size": base_cfg.get("context_size", 80),
    }
    for j in job_entries:
        label = j["label"]
        pkl_path = logs_dir / f"{label}.pkl"
        if pkl_path.exists():
            with pkl_path.open("rb") as f:
                r = pickle.load(f)  # noqa: S301
            r["config"]
            break

    n_workers = args.n_workers or max(1, gpu_cfg.probe_workers)
    print(f"{gpu_cfg.summary()}", flush=True)  # noqa: T201
    print(  # noqa: T201
        f"ADL: batch_size={args.adl_batch_size}, workers={n_workers}",
        flush=True,
    )

    jobs = []
    for j in job_entries:
        label = j["label"]
        ckpt_dir = ckpt_root / label
        if not ckpt_dir.exists():
            continue

        pkl_path = logs_dir / f"{label}.pkl"
        if not pkl_path.exists():
            continue
        with pkl_path.open("rb") as f:
            result = pickle.load(f)  # noqa: S301
        cfg = result["config"]

        ckpt_files = sorted(
            ckpt_dir.glob("step_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if not ckpt_files:
            continue

        pre_burst_ckpt = str(ckpt_files[0])
        if P > 0:
            for cf in ckpt_files:
                s = int(cf.stem.split("_")[1])
                if s <= P:
                    pre_burst_ckpt = str(cf)

        for pt_file in ckpt_files:
            step = int(pt_file.stem.split("_")[1])
            if step < P:
                phase = PHASE_PRE_BURST
            elif step < P + T:
                phase = PHASE_BURST
            else:
                phase = PHASE_REVERSION
            jobs.append(
                {
                    "label": f"{label}_step{step}",
                    "parent_label": label,
                    "step": step,
                    "phase": phase,
                    "ckpt_path": str(pt_file),
                    "pre_burst_ckpt": pre_burst_ckpt,
                    "cfg": cfg,
                    "n_a": n_a,
                    "depth": depth,
                }
            )

    if not jobs:
        print("No checkpoint jobs found.", flush=True)  # noqa: T201
        return

    print(  # noqa: T201
        f"ADL jobs: {len(jobs)} checkpoints across {len(job_entries)} labels",
        flush=True,
    )

    worker_script = str(Path(__file__))

    def build_cmd(
        script: str, job_path: str, data_path: str, output_path: str,
    ) -> list[str]:
        return [
            sys.executable,
            script,
            "--worker",
            "--job-path",
            job_path,
            "--data-path",
            data_path,
            "--output-path",
            output_path,
            "--adl-batch-size",
            str(args.adl_batch_size),
        ]

    def on_done(jr: JobResult, _n_done: int, _n_total: int) -> None:
        status = "ok" if jr.success else f"FAIL: {jr.error[:80]}"
        print(  # noqa: T201
            f"  [{_n_done}/{_n_total}] {jr.label}: {status}",
            flush=True,
        )

    results = run_job_pool(
        jobs=jobs,
        worker_script=worker_script,
        build_cmd=build_cmd,
        on_done=on_done,
        n_workers=n_workers,
        data_payload=(other_docs_BL, burst_docs_BL, prompt_len),
        poll_interval=1.0,
        tmp_prefix="adl_",
    )

    per_label: dict[str, dict] = {}
    for jr in results:
        if not jr.success:
            continue
        d = jr.data
        parent = d["parent_label"]
        if parent not in per_label:
            per_label[parent] = {
                "adl_log": {
                    "step": [],
                    "phase": [],
                    "delta_norm_K": [],
                    "readability_KT": [],
                    "mean_burst_rank_KT": [],
                    "acc_baseline": [],
                    "acc_ablated_K": [],
                    "acc_drop_K": [],
                }
            }
        log = per_label[parent]["adl_log"]
        log["step"].append(d["step"])
        log["phase"].append(d["phase"])
        log["delta_norm_K"].append(d["delta_norm_K"])
        log["readability_KT"].append(d["readability_KT"])
        log["mean_burst_rank_KT"].append(d["mean_burst_rank_KT"])
        log["acc_baseline"].append(d["acc_baseline"])
        log["acc_ablated_K"].append(d["acc_ablated_K"])
        log["acc_drop_K"].append(d["acc_drop_K"])

    for entry in per_label.values():
        log = entry["adl_log"]
        order = np.argsort(log["step"])
        for key in [
            "step",
            "phase",
            "delta_norm_K",
            "readability_KT",
            "mean_burst_rank_KT",
            "acc_baseline",
            "acc_ablated_K",
            "acc_drop_K",
        ]:
            log[key] = [log[key][i] for i in order]

    adl_dir = logs_dir / "adl"
    adl_dir.mkdir(exist_ok=True)

    for j in job_entries:
        label = j["label"]
        if label not in per_label:
            continue
        record = {
            "schedule": j["schedule"],
            "seed": j["seed"],
            "label": label,
            "adl_batch_size": args.adl_batch_size,
            "adl_log": per_label[label]["adl_log"],
        }
        with (adl_dir / f"{label}.json").open("w") as f:
            json.dump(record, f)

    all_results_path = logs_dir / "all_results.pkl"
    if all_results_path.exists():
        with all_results_path.open("rb") as f:
            all_results = pickle.load(f)  # noqa: S301
        for r in all_results:
            lbl = r["label"]
            if lbl in per_label:
                r["adl_log"] = per_label[lbl]["adl_log"]
        with all_results_path.open("wb") as f:
            pickle.dump(all_results, f)
        print(  # noqa: T201
            f"Updated all_results.pkl with ADL data for {len(per_label)} labels",
            flush=True,
        )

    print(  # noqa: T201
        f"ADL done: {len(per_label)} labels, "
        f"{sum(jr.success for jr in results)}/{len(results)} ok",
        flush=True,
    )


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _worker_main()
    else:
        main()
