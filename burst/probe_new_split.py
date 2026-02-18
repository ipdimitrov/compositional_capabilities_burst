"""Linear probes for A-vs-B representation analysis.

Retrains models from experiment_new_split.py using saved config.json,
collects residual-stream activations at (layer, token_position) pairs
across training checkpoints, and fits logistic regression probes to
classify A-data vs B-data representations.

Usage:
    python burst/probe_new_split.py data/burst_d3_<timestamp>
    python burst/probe_new_split.py data/burst_d3_<timestamp> --jobs end_block_s42 uniform_s42
    python burst/probe_new_split.py data/burst_d3_<timestamp> --checkpoint-every 50
"""
import sys, os, argparse, pickle, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from omegaconf import OmegaConf

from synthetic.init import set_seed
from net.nanogpt import nanoGPT
from net.runner import configure_optimizers, update_cosine_warmup_lr
from burst.experiment_new_split import Depth3Data, build_data, N_A, NB_SEEN
from burst._worker_new_split import n_target_for_step, sample_batch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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
    max_samples: int = 512,
) -> np.ndarray:
    """Collect residual-stream activations at every (layer, token_pos).

    Returns float32 array of shape (K, P, T, N).
    K = n_layers + 1 (post-embedding + post-block_0 + ... + post-block_{L-1}).
    T = doc_len - 1 (model input is tokens[:-1]).
    """
    net.eval()
    n = min(len(docs_BL), max_samples)
    idx = np.random.choice(len(docs_BL), n, replace=False)
    tokens_PL = torch.from_numpy(docs_BL[idx]).long().to(DEVICE)
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


def _pad_to_len(arr: np.ndarray, target_len: int) -> np.ndarray:
    if arr.shape[0] == 0:
        return arr
    if arr.shape[1] >= target_len:
        return arr[:, :target_len]
    pad_w = target_len - arr.shape[1]
    return np.concatenate([arr, np.zeros((arr.shape[0], pad_w), dtype=arr.dtype)], axis=1)


def build_probe_dataset(
    data: Depth3Data,
    doc_len: int,
    n_per_task: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build balanced A/B probe datasets from train + heldout compositions."""
    a_train = data.gen_pool(data.a_comp_train[:min(16, len(data.a_comp_train))], n_per_task)
    b_train = data.gen_pool(data.b_comp_train, n_per_task)
    a_eval = data.gen_pool(
        data.a_comp_heldout[:min(8, len(data.a_comp_heldout))], n_per_task
    ) if data.a_comp_heldout else {}
    b_eval = data.gen_pool(
        data.b_comp_heldout[:min(6, len(data.b_comp_heldout))], n_per_task
    ) if data.b_comp_heldout else {}

    def _cat(pool):
        if not pool:
            return np.zeros((0, doc_len), dtype=np.int64)
        return _pad_to_len(np.concatenate(list(pool.values())), doc_len)

    return _cat(a_train), _cat(b_train), _cat(a_eval), _cat(b_eval)


def fit_probes_at_checkpoint(
    net: nanoGPT,
    a_docs_BL: np.ndarray,
    b_docs_BL: np.ndarray,
    a_eval_BL: np.ndarray,
    b_eval_BL: np.ndarray,
    max_samples: int = 512,
) -> dict:
    """Fit logistic regression probes at every (layer, token_pos).

    Returns dict with:
        'train_acc_KT': (K, T) array — 5-fold CV accuracy on train compositions
        'eval_acc_KT':  (K, T) array — accuracy on held-out compositions
    """
    np.random.seed(PROBE_SEED)

    acts_a_KPTN = collect_activations_KPTN(net, a_docs_BL, max_samples)
    acts_b_KPTN = collect_activations_KPTN(net, b_docs_BL, max_samples)

    K, Pa, T, N = acts_a_KPTN.shape
    Pb = acts_b_KPTN.shape[1]

    X_KPTN = np.concatenate([acts_a_KPTN, acts_b_KPTN], axis=1)
    y_P = np.array([0] * Pa + [1] * Pb)

    has_eval = a_eval_BL.shape[0] > 0 and b_eval_BL.shape[0] > 0
    if has_eval:
        acts_a_eval = collect_activations_KPTN(net, a_eval_BL, max_samples)
        acts_b_eval = collect_activations_KPTN(net, b_eval_BL, max_samples)
        Pa_e, Pb_e = acts_a_eval.shape[1], acts_b_eval.shape[1]
        X_eval_KPTN = np.concatenate([acts_a_eval, acts_b_eval], axis=1)
        y_eval = np.array([0] * Pa_e + [1] * Pb_e)

    train_acc_KT = np.zeros((K, T))
    eval_acc_KT = np.zeros((K, T))

    for k in range(K):
        for t in range(T):
            feats_PN = X_KPTN[k, :, t, :]
            clf = LogisticRegression(
                C=0.1, max_iter=2000, solver="lbfgs", random_state=PROBE_SEED)
            scores = cross_val_score(clf, feats_PN, y_P, cv=5, scoring="accuracy")
            train_acc_KT[k, t] = scores.mean()

            if has_eval:
                clf.fit(feats_PN, y_P)
                eval_acc_KT[k, t] = clf.score(X_eval_KPTN[k, :, t, :], y_eval)

    return {"train_acc_KT": train_acc_KT, "eval_acc_KT": eval_acc_KT}


def retrain_and_probe(
    job: dict,
    target_pool: dict,
    bg_pool: dict,
    a_docs_BL: np.ndarray,
    b_docs_BL: np.ndarray,
    a_eval_BL: np.ndarray,
    b_eval_BL: np.ndarray,
    checkpoint_steps: list[int],
    probe_max_samples: int = 512,
) -> dict:
    """Retrain a single model identically and collect probe results at each checkpoint."""
    label, schedule, seed, cfg = job["label"], job["schedule"], job["seed"], job["cfg"]

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

    checkpoint_set = set(checkpoint_steps)
    probe_results = {}

    net.train()
    it = 0

    if 0 in checkpoint_set:
        print(f"    Probing step 0 (init)...", flush=True)
        probe_results[0] = fit_probes_at_checkpoint(
            net, a_docs_BL, b_docs_BL, a_eval_BL, b_eval_BL, probe_max_samples)
        net.train()

    for s in range(T_train):
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

        global_step = s + 1
        if global_step in checkpoint_set:
            print(f"    Probing step {global_step} (train)...", flush=True)
            probe_results[global_step] = fit_probes_at_checkpoint(
                net, a_docs_BL, b_docs_BL, a_eval_BL, b_eval_BL, probe_max_samples)
            net.train()

    for s in range(U):
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

        global_step = T_train + s + 1
        if global_step in checkpoint_set:
            print(f"    Probing step {global_step} (undo)...", flush=True)
            probe_results[global_step] = fit_probes_at_checkpoint(
                net, a_docs_BL, b_docs_BL, a_eval_BL, b_eval_BL, probe_max_samples)
            net.train()

    return {"label": label, "schedule": schedule, "seed": seed, "probes": probe_results}


def _default_checkpoint_steps(total_steps: int, undo_steps: int, every: int) -> list[int]:
    steps = set([0, total_steps, total_steps + undo_steps])
    s = every
    while s <= total_steps + undo_steps:
        steps.add(s)
        s += every
    return sorted(steps)


def main():
    parser = argparse.ArgumentParser(description="Linear probes for A-vs-B representation analysis")
    parser.add_argument("run_dir", type=str, help="Path to experiment run directory")
    parser.add_argument("--jobs", nargs="*", default=None,
                        help="Subset of job labels to probe (default: all)")
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="Probe every N global steps")
    parser.add_argument("--probe-max-samples", type=int, default=512,
                        help="Max samples per class for activation collection")
    parser.add_argument("--seed-override", type=int, default=None,
                        help="Run only this seed across all schedules")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)

    bcfg = cfg["base_cfg"]
    total_steps = bcfg["total_steps"]
    undo_steps = bcfg["undo_steps"]

    checkpoint_steps = _default_checkpoint_steps(total_steps, undo_steps, args.checkpoint_every)
    print(f"Checkpoint steps ({len(checkpoint_steps)}): "
          f"{checkpoint_steps[:8]}...{checkpoint_steps[-3:]}")

    print("Rebuilding data (same seed=999)...")
    tp, bp, _, _, cfg_out, ti = build_data(bcfg)
    print(f"  A_comp: {ti['n_a_comp_train']}/{ti['n_a_comp_heldout']}  "
          f"B_comp: {ti['n_b_comp_train']}/{ti['n_b_comp_heldout']}  "
          f"doc_len: {ti['doc_len']}")

    set_seed(999)
    d = Depth3Data(bcfg["n_alphabets"], bcfg["seq_len"], N_A, NB_SEEN, 999)
    doc_len = ti["doc_len"]
    a_docs, b_docs, a_eval, b_eval = build_probe_dataset(d, doc_len)
    print(f"  Probe data: A_train={a_docs.shape[0]} B_train={b_docs.shape[0]} "
          f"A_eval={a_eval.shape[0]} B_eval={b_eval.shape[0]}")

    token_labels = get_token_position_labels(doc_len, bcfg["seq_len"])
    print(f"  Token positions ({len(token_labels)}): {token_labels[:6]}...{token_labels[-3:]}")

    jobs_cfg = cfg["jobs"]
    if args.jobs:
        jobs_cfg = [j for j in jobs_cfg if j["label"] in args.jobs]
    if args.seed_override is not None:
        jobs_cfg = [j for j in jobs_cfg if j["seed"] == args.seed_override]

    print(f"\nProbing {len(jobs_cfg)} jobs on {DEVICE}")
    print(f"Model: {bcfg['n_layer']}L/{bcfg['n_embd']}d/{bcfg['n_head']}H\n")

    probe_dir = run_dir / "probes"
    probe_dir.mkdir(exist_ok=True)

    all_probe_results = []

    for ji, jcfg in enumerate(jobs_cfg):
        label = jcfg["label"]
        seed = jcfg["seed"]
        schedule = jcfg["schedule"]
        nb = jcfg.get("n_b_seen", NB_SEEN)

        job = {
            "label": label, "schedule": schedule, "seed": seed, "n_b_seen": nb,
            "cfg": {**bcfg, "seed": seed, "n_b_seen": nb,
                    "vocab_size": cfg_out["vocab_size"],
                    "context_size": cfg_out["context_size"]},
        }

        print(f"[{ji+1}/{len(jobs_cfg)}] {label}")
        result = retrain_and_probe(
            job, tp, bp, a_docs, b_docs, a_eval, b_eval,
            checkpoint_steps, args.probe_max_samples)

        result["checkpoint_steps"] = checkpoint_steps
        result["token_labels"] = token_labels
        result["n_layers"] = bcfg["n_layer"]
        result["total_steps"] = total_steps
        result["undo_steps"] = undo_steps

        with open(probe_dir / f"{label}_probe.pkl", "wb") as f:
            pickle.dump(result, f)
        print(f"  -> {probe_dir / f'{label}_probe.pkl'}")

        all_probe_results.append(result)

    with open(probe_dir / "all_probes.pkl", "wb") as f:
        pickle.dump(all_probe_results, f)

    meta = {
        "checkpoint_steps": checkpoint_steps,
        "token_labels": token_labels,
        "n_layers": bcfg["n_layer"],
        "total_steps": total_steps,
        "undo_steps": undo_steps,
        "probe_max_samples": args.probe_max_samples,
        "probe_seed": PROBE_SEED,
        "jobs": [j["label"] for j in jobs_cfg],
    }
    with open(probe_dir / "probe_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nAll probes saved to {probe_dir}")
    print(f"Plot: python burst/plot_probes.py {run_dir}")


if __name__ == "__main__":
    main()
