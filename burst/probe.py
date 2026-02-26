"""Linear probes for Other-vs-Burst representation analysis.

Loads saved checkpoints from experiment.py training runs,
collects residual-stream activations at (layer, token_position) pairs
across training checkpoints, and fits GPU-accelerated linear probes to
classify Other-class vs Burst-class representations.

Falls back to retraining from scratch when checkpoints are unavailable.

Usage:
    python burst/probe.py data/burst_d<depth>_<run_tag>
    python burst/probe.py data/burst_d<depth>_<run_tag> --jobs end_block_s42 uniform_s42
    python burst/probe.py data/burst_d<depth>_<run_tag> --checkpoint-every 50
    python burst/probe.py data/burst_d<depth>_<run_tag> --n-workers 38
"""
import sys, os, argparse, pickle, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from synthetic.init import set_seed
from net.nanogpt import nanoGPT
from burst.experiment import DepthNData, build_data
from burst.train_utils import (
    DEVICE, retrain_with_callbacks, build_probe_docs, N_PROBE_DOCS_PER_TASK,
)
from burst.config import DATA_SEED, parse_run_config
from burst.parallel import run_job_pool
from burst.gpu import gpu_cfg

PROBE_SEED = 1337

"""
Dimension key:
    B: batch_size
    L: sequence_length (doc_len, includes :-1 trimming for model input)
    N: n_embd (model dimension)
    P: n_probe_samples
    K: n_layers + 1 (embedding + each transformer block)
    T: n_token_positions (= L - 1, since model sees tokens :-1)
"""

GPU_PROBE_LR = 1e-2
GPU_PROBE_EPOCHS = 200
GPU_PROBE_VAL_FRAC = 0.2
GPU_PROBE_PATIENCE = 30
GPU_PROBE_VAL_EVERY = 10
COLLECT_BATCH_SIZE = 512


def get_token_position_labels(doc_len: int, seq_len: int, depth: int) -> list[str]:
    labels = ["S"]
    labels += [f"F{depth - i}" for i in range(depth)]
    labels += ["sp0"]
    labels += [f"in{i}" for i in range(seq_len)]
    for d in range(1, depth + 1):
        labels += [f"sp{d}"]
        labels += [f"o{d}_{i}" for i in range(seq_len)]
    return labels[:doc_len - 1]


@torch.no_grad()
def collect_activations_KPTN(
    net: nanoGPT,
    docs_BL: np.ndarray,
) -> list[torch.Tensor]:
    """Collect residual-stream activations at every (layer, token_pos).

    Returns list of K tensors, each of shape (P, T, N) on CPU.
    K = n_layers + 1 (post-embedding + post-block_0 + ... + post-block_{L-1}).
    T = doc_len - 1 (model input is tokens[:-1]).
    P = len(docs_BL) — caller is responsible for subsampling.
    """
    net.eval()
    P = len(docs_BL)
    K = len(net.transformer.h) + 1

    all_layer_acts = [[] for _ in range(K)]

    for start in range(0, P, COLLECT_BATCH_SIZE):
        end = min(start + COLLECT_BATCH_SIZE, P)
        tokens_bL = torch.as_tensor(docs_BL[start:end], dtype=torch.long, device=DEVICE)
        inp_bT = tokens_bL[:, :-1]

        tok_emb = net.transformer.wte(inp_bT)
        pos = torch.arange(inp_bT.size(1), device=DEVICE)
        pos_emb = net.transformer.wpe(pos)
        x_bTN = net.transformer.drop(tok_emb + pos_emb)

        all_layer_acts[0].append(x_bTN.float().cpu())
        for bi, block in enumerate(net.transformer.h):
            x_bTN = block(x_bTN)
            all_layer_acts[bi + 1].append(x_bTN.float().cpu())

    return [torch.cat(chunks, dim=0) for chunks in all_layer_acts]


def _fit_gpu_probe(feats_PN: torch.Tensor, labels_P: torch.Tensor) -> float:
    """Train a single binary linear probe on GPU, return val accuracy."""
    N = feats_PN.shape[1]
    n_total = feats_PN.shape[0]
    n_val = max(int(n_total * GPU_PROBE_VAL_FRAC), 1)

    torch.manual_seed(PROBE_SEED)
    perm = torch.randperm(n_total)
    train_idx, val_idx = perm[n_val:], perm[:n_val]

    train_feats = feats_PN[train_idx].to(DEVICE)
    train_labels = labels_P[train_idx].to(DEVICE)
    val_feats = feats_PN[val_idx].to(DEVICE)
    val_labels = labels_P[val_idx].to(DEVICE)

    probe = nn.Linear(N, 2).to(DEVICE)
    optimizer = torch.optim.Adam(probe.parameters(), lr=GPU_PROBE_LR)

    best_val_acc = 0.0
    epochs_no_improve = 0
    for epoch in range(GPU_PROBE_EPOCHS):
        probe.train()
        logits = probe(train_feats)
        loss = F.cross_entropy(logits, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % GPU_PROBE_VAL_EVERY == 0 or epoch == GPU_PROBE_EPOCHS - 1:
            probe.eval()
            with torch.no_grad():
                val_acc = (probe(val_feats).argmax(-1) == val_labels).float().mean().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
            else:
                epochs_no_improve += GPU_PROBE_VAL_EVERY
            if epochs_no_improve >= GPU_PROBE_PATIENCE:
                break

    return best_val_acc


def fit_probes_at_checkpoint(
    net: nanoGPT,
    other_docs_BL: np.ndarray,
    burst_docs_BL: np.ndarray,
    max_samples: int,
) -> dict:
    """Fit GPU linear probes at every (layer, token_pos).

    Returns dict with:
        'train_acc_KT': (K, T) array — val-split accuracy on train compositions
    """
    np.random.seed(PROBE_SEED)

    n_other = min(len(other_docs_BL), max_samples)
    n_burst = min(len(burst_docs_BL), max_samples)
    idx_other = np.random.choice(len(other_docs_BL), n_other, replace=False)
    idx_burst = np.random.choice(len(burst_docs_BL), n_burst, replace=False)
    combined_BL = np.concatenate([other_docs_BL[idx_other], burst_docs_BL[idx_burst]], axis=0)

    acts_K_PTN = collect_activations_KPTN(net, combined_BL)

    K = len(acts_K_PTN)
    T = acts_K_PTN[0].shape[1]
    labels_P = torch.cat([torch.zeros(n_other, dtype=torch.long),
                          torch.ones(n_burst, dtype=torch.long)])

    train_acc_KT = np.zeros((K, T))

    for k in range(K):
        for t in range(T):
            feats_PN = acts_K_PTN[k][:, t, :]
            train_acc_KT[k, t] = _fit_gpu_probe(feats_PN, labels_P)

    return {"train_acc_KT": train_acc_KT}


def _load_checkpoint(cfg: dict, ckpt_path: str) -> nanoGPT:
    """Load a model from a saved checkpoint file."""
    from omegaconf import OmegaConf
    net = nanoGPT(OmegaConf.create({
        "compile": False, "vocab_size": cfg["vocab_size"],
        "context_size": cfg["context_size"],
        "n_layer": cfg["n_layer"], "n_head": cfg["n_head"],
        "n_embd": cfg["n_embd"], "dropout": 0.0, "bias": False, "mlp": True,
    })).to(DEVICE)
    net.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    return net


def probe_from_checkpoints(
    job: dict,
    ckpt_dir: Path,
    other_docs_BL: np.ndarray,
    burst_docs_BL: np.ndarray,
    checkpoint_steps: list[int],
    probe_max_samples: int,
) -> dict:
    """Load saved checkpoints and probe at each requested step."""
    cfg = job["cfg"]
    probe_results = {}

    available_ckpts = {}
    if ckpt_dir.exists():
        for pt_file in ckpt_dir.glob("step_*.pt"):
            step = int(pt_file.stem.split("_")[1])
            available_ckpts[step] = str(pt_file)

    for step in checkpoint_steps:
        if step not in available_ckpts:
            continue
        print(f"    Loading ckpt step {step}...", flush=True)
        net = _load_checkpoint(cfg, available_ckpts[step])
        probe_results[step] = fit_probes_at_checkpoint(
            net, other_docs_BL, burst_docs_BL, probe_max_samples)
        del net
        torch.cuda.empty_cache()

    return {
        "label": job["label"], "schedule": job["schedule"],
        "seed": job["seed"], "probes": probe_results,
    }


def retrain_and_probe(
    job: dict,
    target_pool: dict,
    bg_pool: dict,
    other_docs_BL: np.ndarray,
    burst_docs_BL: np.ndarray,
    checkpoint_steps: list[int],
    probe_max_samples: int,
) -> dict:
    """Retrain a single model identically and collect probe results at each checkpoint."""
    checkpoint_set = set(checkpoint_steps)
    probe_results = {}

    def on_step(net, global_step, phase):
        if global_step in checkpoint_set:
            print(f"    Probing step {global_step} ({phase})...", flush=True)
            probe_results[global_step] = fit_probes_at_checkpoint(
                net, other_docs_BL, burst_docs_BL, probe_max_samples)
            net.train()

    retrain_with_callbacks(job, target_pool, bg_pool, on_step=on_step)

    return {
        "label": job["label"], "schedule": job["schedule"],
        "seed": job["seed"], "probes": probe_results,
    }


def _default_checkpoint_steps(total_steps: int, reversion_steps: int, every: int) -> list[int]:
    steps = set(range(0, total_steps + reversion_steps + 1, every))
    steps |= {0, total_steps, total_steps + reversion_steps}
    return sorted(steps)


def _worker_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--checkpoint-steps", type=int, nargs="+", required=True)
    parser.add_argument("--probe-max-samples", type=int, required=True)
    wargs = parser.parse_args()

    with open(wargs.job_path, "rb") as f:
        job = pickle.load(f)
    with open(wargs.data_path, "rb") as f:
        tp, bp, other_docs, burst_docs = pickle.load(f)

    ckpt_dir = job.get("ckpt_dir")
    if ckpt_dir and Path(ckpt_dir).exists():
        result = probe_from_checkpoints(
            job, Path(ckpt_dir), other_docs, burst_docs,
            wargs.checkpoint_steps, wargs.probe_max_samples)
    else:
        result = retrain_and_probe(
            job, tp, bp, other_docs, burst_docs,
            wargs.checkpoint_steps, wargs.probe_max_samples)

    with open(wargs.output_path, "wb") as f:
        pickle.dump(result, f)


def main():
    parser = argparse.ArgumentParser(description="Linear probes for Other-vs-Burst representation analysis")
    parser.add_argument("run_dir", type=str, help="Path to experiment run directory")
    parser.add_argument("--jobs", nargs="*", default=None,
                        help="Subset of job labels to probe (default: all)")
    parser.add_argument("--checkpoint-every", type=int, required=True,
                        help="Probe every N global steps")
    parser.add_argument("--probe-max-samples", type=int, required=True,
                        help="Max samples per class for activation collection")
    parser.add_argument("--seed-override", type=int, default=None,
                        help="Run only this seed across all schedules")
    parser.add_argument("--n-workers", type=int, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)

    rc = parse_run_config(cfg)
    bcfg, depth, burst_pos, n_a = rc["base_cfg"], rc["depth"], rc["burst_pos"], rc["n_a"]
    total_steps = bcfg["total_steps"]
    reversion_steps = bcfg["reversion_steps"]

    checkpoint_steps = _default_checkpoint_steps(total_steps, reversion_steps, args.checkpoint_every)
    print(f"Checkpoint steps ({len(checkpoint_steps)}): "
          f"{checkpoint_steps[:8]}...{checkpoint_steps[-3:]}")

    print(f"Rebuilding data (seed={DATA_SEED})...")
    tp, bp, _, _, cfg_out, ti = build_data(bcfg, depth, burst_pos, n_a)
    print(f"  Other tasks: {ti['n_other_train']}  "
          f"Burst tasks: {ti['n_burst_train']}  "
          f"doc_len: {ti['doc_len']}")

    set_seed(DATA_SEED)
    d = DepthNData(bcfg["n_alphabets"], bcfg["seq_len"], n_a, depth, burst_pos, DATA_SEED)
    doc_len = ti["doc_len"]
    other_docs, burst_docs = build_probe_docs(d, doc_len, N_PROBE_DOCS_PER_TASK)
    print(f"  Probe data: Other={other_docs.shape[0]} Burst={burst_docs.shape[0]}")

    token_labels = get_token_position_labels(doc_len, bcfg["seq_len"], depth)
    print(f"  Token positions ({len(token_labels)}): {token_labels[:6]}...{token_labels[-3:]}")

    ckpt_root = run_dir / "checkpoints"
    use_checkpoints = ckpt_root.exists()
    if use_checkpoints:
        print(f"  Found checkpoints at {ckpt_root}, will load instead of retraining")
    else:
        print(f"  No checkpoints found, will retrain from scratch")

    jobs_cfg = cfg["jobs"]
    if args.jobs:
        jobs_cfg = [j for j in jobs_cfg if j["label"] in args.jobs]
    if args.seed_override is not None:
        jobs_cfg = [j for j in jobs_cfg if j["seed"] == args.seed_override]

    n_workers = min(len(jobs_cfg), args.n_workers or gpu_cfg.probe_workers)
    print(f"\n{gpu_cfg.summary()}")
    print(f"Probing {len(jobs_cfg)} jobs on {DEVICE}, workers: {n_workers}")
    print(f"Model: {bcfg['n_layer']}L/{bcfg['n_embd']}d/{bcfg['n_head']}H")
    print(f"Mode: {'checkpoint-loading' if use_checkpoints else 'retrain'}\n")

    probe_dir = run_dir / "probes"
    probe_dir.mkdir(exist_ok=True)

    jobs = []
    for jcfg in jobs_cfg:
        label, seed, schedule = jcfg["label"], jcfg["seed"], jcfg["schedule"]
        job_entry = {
            "label": label, "schedule": schedule, "seed": seed,
            "cfg": {**bcfg, "seed": seed,
                    "vocab_size": cfg_out["vocab_size"],
                    "context_size": cfg_out["context_size"]},
        }
        if use_checkpoints:
            job_entry["ckpt_dir"] = str(ckpt_root / label)
        jobs.append(job_entry)

    ckpt_args = [str(s) for s in checkpoint_steps]

    def build_cmd(script, job_path, data_path, output_path):
        return ([sys.executable, script, "--worker",
                 "--job-path", job_path, "--data-path", data_path,
                 "--output-path", output_path,
                 "--checkpoint-steps"] + ckpt_args +
                ["--probe-max-samples", str(args.probe_max_samples)])

    all_probe_results = []

    def on_done(jr, n_done, n_total):
        if jr.success:
            result = jr.data
            result["checkpoint_steps"] = checkpoint_steps
            result["token_labels"] = token_labels
            result["n_layers"] = bcfg["n_layer"]
            result["total_steps"] = total_steps
            result["reversion_steps"] = reversion_steps
            with open(probe_dir / f"{result['label']}_probe.pkl", "wb") as f:
                pickle.dump(result, f)
            all_probe_results.append(result)
            pkl_path = probe_dir / f"{result['label']}_probe.pkl"
            print(f"  [{n_done}/{n_total}] {jr.label:30s} "
                  f"-> {pkl_path} ({jr.elapsed:.0f}s)",
                  flush=True)
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
        tmp_prefix="probe_lr_",
    )

    with open(probe_dir / "all_probes.pkl", "wb") as f:
        pickle.dump(all_probe_results, f)

    meta = {
        "checkpoint_steps": checkpoint_steps,
        "token_labels": token_labels,
        "n_layers": bcfg["n_layer"],
        "total_steps": total_steps,
        "reversion_steps": reversion_steps,
        "probe_max_samples": args.probe_max_samples,
        "probe_seed": PROBE_SEED,
        "jobs": [j["label"] for j in jobs_cfg],
    }
    with open(probe_dir / "probe_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nAll probes saved to {probe_dir}")
    print(f"Plot: python burst/plot_probes.py {run_dir}")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        sys.argv.remove("--worker")
        _worker_main()
    else:
        main()
