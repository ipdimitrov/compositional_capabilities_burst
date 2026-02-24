"""Linear probes for Other-vs-Burst representation analysis.

Retrains models from experiment.py using saved config.json,
collects residual-stream activations at (layer, token_position) pairs
across training checkpoints, and fits logistic regression probes to
classify Other-class vs Burst-class representations.

Usage:
    python burst/probe.py data/burst_d3_<run_tag>
    python burst/probe.py data/burst_d3_<run_tag> --jobs end_block_s42 uniform_s42
    python burst/probe.py data/burst_d3_<run_tag> --checkpoint-every 50
    python burst/probe.py data/burst_d3_<run_tag> --n-workers 38
"""
import sys, os, argparse, pickle, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from synthetic.init import set_seed
from net.nanogpt import nanoGPT
from burst.experiment import DepthNData, build_data
from burst.train_utils import (
    DEVICE, retrain_with_callbacks, build_probe_docs, N_PROBE_DOCS_PER_TASK,
)
from burst.config import N_A, DATA_SEED, ExperimentConfig
from burst.parallel import run_job_pool

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


def get_token_position_labels(doc_len: int, seq_len: int) -> list[str]:
    labels = ["S", "F3", "F2", "F1", "sp0"]
    labels += [f"in{i}" for i in range(seq_len)]
    labels += ["sp1"]
    labels += [f"o1_{i}" for i in range(seq_len)]
    labels += ["sp2"]
    labels += [f"o2_{i}" for i in range(seq_len)]
    labels += ["sp3"]
    labels += [f"o3_{i}" for i in range(seq_len)]
    return labels[:doc_len - 1]


@torch.no_grad()
def collect_activations_KPTN(
    net: nanoGPT,
    docs_BL: np.ndarray,
) -> np.ndarray:
    """Collect residual-stream activations at every (layer, token_pos).

    Returns float32 array of shape (K, P, T, N).
    K = n_layers + 1 (post-embedding + post-block_0 + ... + post-block_{L-1}).
    T = doc_len - 1 (model input is tokens[:-1]).
    P = len(docs_BL) — caller is responsible for subsampling.
    """
    net.eval()
    tokens_PL = torch.from_numpy(docs_BL).long().to(DEVICE)
    inp_PT = tokens_PL[:, :-1]

    tok_emb = net.transformer.wte(inp_PT)
    pos = torch.arange(inp_PT.size(1), device=DEVICE)
    pos_emb = net.transformer.wpe(pos)
    x_PTN = net.transformer.drop(tok_emb + pos_emb)

    layer_acts = [x_PTN.float().cpu().numpy()]
    for block in net.transformer.h:
        x_PTN = block(x_PTN)
        layer_acts.append(x_PTN.float().cpu().numpy())

    return np.stack(layer_acts, axis=0)


build_probe_dataset = build_probe_docs


def fit_probes_at_checkpoint(
    net: nanoGPT,
    other_docs_BL: np.ndarray,
    burst_docs_BL: np.ndarray,
    max_samples: int,
) -> dict:
    """Fit logistic regression probes at every (layer, token_pos).

    Returns dict with:
        'train_acc_KT': (K, T) array — 5-fold CV accuracy on train compositions
    """
    np.random.seed(PROBE_SEED)

    n_other = min(len(other_docs_BL), max_samples)
    n_burst = min(len(burst_docs_BL), max_samples)
    idx_other = np.random.choice(len(other_docs_BL), n_other, replace=False)
    idx_burst = np.random.choice(len(burst_docs_BL), n_burst, replace=False)
    combined_BL = np.concatenate([other_docs_BL[idx_other], burst_docs_BL[idx_burst]], axis=0)

    acts_KPTN = collect_activations_KPTN(net, combined_BL)

    K, P_total, T, N = acts_KPTN.shape
    Pa, Pb = n_other, n_burst
    X_KPTN = acts_KPTN
    y_P = np.array([0] * Pa + [1] * Pb)

    train_acc_KT = np.zeros((K, T))

    for k in range(K):
        for t in range(T):
            feats_PN = X_KPTN[k, :, t, :]
            clf = LogisticRegression(
                C=0.1, max_iter=2000, solver="lbfgs", random_state=PROBE_SEED)
            scores = cross_val_score(clf, feats_PN, y_P, cv=5,
                                     scoring="accuracy", n_jobs=-1)
            train_acc_KT[k, t] = scores.mean()

    return {"train_acc_KT": train_acc_KT}


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
    steps = set([0, total_steps, total_steps + reversion_steps])
    s = every
    while s <= total_steps + reversion_steps:
        steps.add(s)
        s += every
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
    parser.add_argument("--n-workers", type=int, required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)

    bcfg = cfg["base_cfg"]
    total_steps = bcfg["total_steps"]
    reversion_steps = bcfg["reversion_steps"]

    checkpoint_steps = _default_checkpoint_steps(total_steps, reversion_steps, args.checkpoint_every)
    print(f"Checkpoint steps ({len(checkpoint_steps)}): "
          f"{checkpoint_steps[:8]}...{checkpoint_steps[-3:]}")

    depth = cfg.get("depth", cfg.get("task_info", {}).get("depth", 3))
    burst_pos = cfg.get("burst_pos", cfg.get("task_info", {}).get("burst_pos", depth))

    print(f"Rebuilding data (seed={DATA_SEED})...")
    tp, bp, _, _, cfg_out, ti = build_data(bcfg, depth, burst_pos)
    print(f"  Other tasks: {ti['n_other_train']}  "
          f"Burst tasks: {ti['n_burst_train']}  "
          f"doc_len: {ti['doc_len']}")

    set_seed(DATA_SEED)
    d = DepthNData(bcfg["n_alphabets"], bcfg["seq_len"], N_A, depth, burst_pos, DATA_SEED)
    doc_len = ti["doc_len"]
    other_docs, burst_docs = build_probe_dataset(d, doc_len, N_PROBE_DOCS_PER_TASK)
    print(f"  Probe data: Other={other_docs.shape[0]} Burst={burst_docs.shape[0]}")

    token_labels = get_token_position_labels(doc_len, bcfg["seq_len"])
    print(f"  Token positions ({len(token_labels)}): {token_labels[:6]}...{token_labels[-3:]}")

    jobs_cfg = cfg["jobs"]
    if args.jobs:
        jobs_cfg = [j for j in jobs_cfg if j["label"] in args.jobs]
    if args.seed_override is not None:
        jobs_cfg = [j for j in jobs_cfg if j["seed"] == args.seed_override]

    n_workers = min(len(jobs_cfg), args.n_workers)
    print(f"\nProbing {len(jobs_cfg)} jobs on {DEVICE}, workers: {n_workers}")
    print(f"Model: {bcfg['n_layer']}L/{bcfg['n_embd']}d/{bcfg['n_head']}H\n")

    probe_dir = run_dir / "probes"
    probe_dir.mkdir(exist_ok=True)

    jobs = []
    for jcfg in jobs_cfg:
        label, seed, schedule = jcfg["label"], jcfg["seed"], jcfg["schedule"]
        jobs.append({
            "label": label, "schedule": schedule, "seed": seed,
            "cfg": {**bcfg, "seed": seed,
                    "vocab_size": cfg_out["vocab_size"],
                    "context_size": cfg_out["context_size"]},
        })

    ckpt_args = []
    for s in checkpoint_steps:
        ckpt_args += [str(s)]

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
            print(f"  [{n_done}/{n_total}] {jr.label:30s} "
                  f"-> {probe_dir / f'{result[\"label\"]}_probe.pkl'} ({jr.elapsed:.0f}s)",
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
