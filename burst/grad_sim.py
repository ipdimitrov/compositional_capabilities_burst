"""Post-hoc gradient cosine similarity computation on saved checkpoints.

Runs after training (burst/experiment.py) as a separate pass, loading
checkpoints and computing grad-sim with full GPU utilisation at the
grad_sim_batch_size level.

Usage:
    python burst/grad_sim.py <run_dir>
    python burst/grad_sim.py <run_dir> --n-workers 8 --grad-sim-batch-size 2048
"""
import sys, os, argparse, pickle, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from omegaconf import OmegaConf

from net.nanogpt import nanoGPT
from burst.parallel import run_job_pool
from burst.config import PHASE_BURST, PHASE_REVERSION

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_UTILIZATION_TARGET = 0.92


def estimate_max_workers(cfg: dict, grad_sim_batch_size: int) -> int:
    if DEVICE != "cuda":
        return 1

    model_cfg = OmegaConf.create({
        "compile": False, "vocab_size": cfg["vocab_size"],
        "context_size": cfg["context_size"],
        "n_layer": cfg["n_layer"], "n_head": cfg["n_head"],
        "n_embd": cfg["n_embd"], "dropout": 0.0, "bias": False, "mlp": True,
    })
    net = nanoGPT(model_cfg).to(DEVICE)
    net.train()

    dummy = torch.randint(0, cfg["vocab_size"],
                          (grad_sim_batch_size, cfg["context_size"]), device=DEVICE)
    torch.cuda.reset_peak_memory_stats()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = net(dummy[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), dummy[:, 1:].reshape(-1))
    loss.backward()
    peak_bytes = torch.cuda.max_memory_allocated()

    del net, dummy, logits, loss
    torch.cuda.empty_cache()

    total_bytes = torch.cuda.get_device_properties(0).total_memory
    usable = total_bytes * GPU_UTILIZATION_TARGET
    return max(1, int(usable / peak_bytes))


def _flat_grad(net) -> torch.Tensor:
    grads = [p.grad.detach().view(-1) for p in net.parameters() if p.grad is not None]
    return torch.cat(grads) if grads else torch.zeros(1, device=DEVICE)


def _grad_vec_for_docs(net, docs_np: np.ndarray, n_samples: int = 64) -> torch.Tensor:
    n = min(n_samples, docs_np.shape[0])
    idx = np.random.choice(docs_np.shape[0], n, replace=False)
    dat = torch.as_tensor(docs_np[idx], dtype=torch.long, device=DEVICE)
    inp, tgt = dat[:, :-1], dat[:, 1:]
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        logits = net(inp)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
    loss.backward()
    return _flat_grad(net).float()


def compute_grad_cosine_sim(net, docs_burst_BL, docs_other_BL,
                            n_samples: int = 2048) -> dict:
    net.train()
    net.zero_grad(set_to_none=True)
    g_burst = _grad_vec_for_docs(net, docs_burst_BL, n_samples=n_samples)
    net.zero_grad(set_to_none=True)
    g_other = _grad_vec_for_docs(net, docs_other_BL, n_samples=n_samples)
    cos_sim = F.cosine_similarity(g_burst.unsqueeze(0), g_other.unsqueeze(0)).item()
    net.zero_grad(set_to_none=True)
    return {"burst_vs_other": cos_sim}


def compute_pairwise_grad_sim(net, task_docs: dict,
                               burst_tasks: list, other_tasks: list,
                               n_samples: int = 2048) -> dict:
    net.train()

    b_tasks = burst_tasks[:5]
    o_tasks = other_tasks[:5]
    all_tasks = b_tasks + o_tasks
    labels = [f"B{i+1}" for i in range(len(b_tasks))] + [f"O{i+1}" for i in range(len(o_tasks))]

    grad_vecs = []
    for task in all_tasks:
        if task in task_docs and task_docs[task].shape[0] > 0:
            net.zero_grad(set_to_none=True)
            grad_vecs.append(_grad_vec_for_docs(net, task_docs[task], n_samples=n_samples))
        else:
            grad_vecs.append(None)

    n = len(all_tasks)
    matrix = np.eye(n)
    valid = [(i, v) for i, v in enumerate(grad_vecs) if v is not None]
    if len(valid) >= 2:
        G = torch.stack([v for _, v in valid])
        G_norm = F.normalize(G, dim=1)
        sim = (G_norm @ G_norm.T).cpu().numpy()
        for ri, (i, _) in enumerate(valid):
            for rj, (j, _) in enumerate(valid):
                matrix[i, j] = sim[ri, rj]

    net.zero_grad(set_to_none=True)
    return {"matrix": matrix.tolist(), "labels": labels,
            "n_burst": len(b_tasks), "n_other": len(o_tasks)}


def _worker_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--job-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--grad-sim-batch-size", type=int, default=2048)
    args = parser.parse_args()

    with open(args.job_path, "rb") as f:
        job = pickle.load(f)
    with open(args.data_path, "rb") as f:
        target_pool, bg_pool = pickle.load(f)

    cfg = job["cfg"]
    ckpt_path = job["ckpt_path"]
    step = job["step"]
    phase = job["phase"]
    is_pairwise = job["is_pairwise"]
    gs_bs = args.grad_sim_batch_size

    net = nanoGPT(OmegaConf.create({
        "compile": False, "vocab_size": cfg["vocab_size"],
        "context_size": cfg["context_size"],
        "n_layer": cfg["n_layer"], "n_head": cfg["n_head"],
        "n_embd": cfg["n_embd"], "dropout": 0.0, "bias": False, "mlp": True,
    })).to(DEVICE)
    net.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))

    burst_docs_all = np.concatenate(list(target_pool.values())) if target_pool else None
    other_docs_all = np.concatenate(list(bg_pool.values())) if bg_pool else None

    result = {"label": job["label"], "parent_label": job["parent_label"],
              "step": step, "phase": phase}

    if burst_docs_all is not None and other_docs_all is not None:
        sim = compute_grad_cosine_sim(net, burst_docs_all, other_docs_all, n_samples=gs_bs)
        result["burst_vs_other"] = sim["burst_vs_other"]

    if is_pairwise:
        task_docs = {**target_pool, **bg_pool}
        burst_tasks = list(target_pool.keys())
        other_tasks = list(bg_pool.keys())
        snap = compute_pairwise_grad_sim(net, task_docs, burst_tasks, other_tasks,
                                         n_samples=gs_bs)
        result["pairwise"] = snap

    with open(args.output_path, "wb") as f:
        pickle.dump(result, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--grad-sim-batch-size", type=int, default=None)
    parser.add_argument("--keep-checkpoints", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir
    with open(run_dir / "config.json") as f:
        run_cfg = json.load(f)

    base_cfg = run_cfg["base_cfg"]
    gs_bs = args.grad_sim_batch_size or base_cfg.get("grad_sim_batch_size", 2048)
    T = base_cfg["total_steps"]
    U = base_cfg["reversion_steps"]

    pairwise_global_steps = {0, T // 2, T - 1, T + U // 2, T + U - 1}

    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.exists():
        print(f"No checkpoints directory in {run_dir}, nothing to do.", flush=True)
        return

    job_entries = run_cfg["jobs"]
    sample_cfg = {**base_cfg,
                  "vocab_size": base_cfg.get("vocab_size", 128),
                  "context_size": base_cfg.get("context_size", 80)}
    for j in job_entries:
        label = j["label"]
        ckpt_dir = ckpt_root / label
        if ckpt_dir.exists():
            sample_cfg_path = run_dir / f"{label}.pkl"
            if sample_cfg_path.exists():
                with open(sample_cfg_path, "rb") as f:
                    r = pickle.load(f)
                sample_cfg = r["config"]
            break

    if args.n_workers is not None:
        n_workers = args.n_workers
    else:
        n_workers = estimate_max_workers(sample_cfg, gs_bs)
    print(f"Grad-sim: batch_size={gs_bs}, workers={n_workers}", flush=True)

    jobs = []
    for j in job_entries:
        label = j["label"]
        ckpt_dir = ckpt_root / label
        if not ckpt_dir.exists():
            continue

        result_path = run_dir / f"{label}.pkl"
        if not result_path.exists():
            continue
        with open(result_path, "rb") as f:
            result = pickle.load(f)
        cfg = result["config"]

        for pt_file in sorted(ckpt_dir.glob("step_*.pt")):
            step = int(pt_file.stem.split("_")[1])
            phase = PHASE_BURST if step < T else PHASE_REVERSION
            is_pairwise = step in pairwise_global_steps
            jobs.append({
                "label": f"{label}_step{step}",
                "parent_label": label,
                "step": step,
                "phase": phase,
                "is_pairwise": is_pairwise,
                "ckpt_path": str(pt_file),
                "cfg": cfg,
            })

    if not jobs:
        print("No checkpoint jobs found.", flush=True)
        return

    print(f"Jobs: {len(jobs)} checkpoints across {len(job_entries)} labels", flush=True)

    data_path = str(run_dir / "_data.pkl")
    with open(data_path, "rb") as f:
        target_pool, bg_pool, _, _, _ = pickle.load(f)

    worker_script = str(Path(__file__))

    def build_cmd(script, job_path, data_path, output_path):
        return [sys.executable, script, "--worker",
                "--job-path", job_path, "--data-path", data_path,
                "--output-path", output_path,
                "--grad-sim-batch-size", str(gs_bs)]

    def on_done(jr, n_done, n_total):
        status = "ok" if jr.success else f"FAIL: {jr.error[:80]}"
        print(f"  [{n_done}/{n_total}] {jr.label}: {status}", flush=True)

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

    per_label: dict[str, dict] = {}
    for jr in results:
        if not jr.success:
            continue
        d = jr.data
        parent = d["parent_label"]
        if parent not in per_label:
            per_label[parent] = {"grad_sim_log": {"step": [], "phase": [], "burst_vs_other": []},
                                 "pairwise_snapshots": []}
        entry = per_label[parent]
        if "burst_vs_other" in d:
            entry["grad_sim_log"]["step"].append(d["step"])
            entry["grad_sim_log"]["phase"].append(d["phase"])
            entry["grad_sim_log"]["burst_vs_other"].append(d["burst_vs_other"])
        if "pairwise" in d:
            snap = d["pairwise"]
            snap["step"] = d["step"]
            snap["phase"] = d["phase"]
            entry["pairwise_snapshots"].append(snap)

    for label, entry in per_label.items():
        gs_log = entry["grad_sim_log"]
        order = np.argsort(gs_log["step"])
        gs_log["step"] = [gs_log["step"][i] for i in order]
        gs_log["phase"] = [gs_log["phase"][i] for i in order]
        gs_log["burst_vs_other"] = [gs_log["burst_vs_other"][i] for i in order]
        entry["pairwise_snapshots"].sort(key=lambda s: s["step"])

    gs_dir = run_dir / "grad_cosine_sim"
    gs_dir.mkdir(exist_ok=True)

    for j in job_entries:
        label = j["label"]
        if label not in per_label:
            continue
        entry = per_label[label]
        record = {
            "schedule": j["schedule"], "seed": j["seed"], "label": label,
            "grad_sim_batch_size": gs_bs,
            "grad_sim_log": entry["grad_sim_log"],
            "pairwise_snapshots": entry["pairwise_snapshots"],
        }
        with open(gs_dir / f"{label}.json", "w") as f:
            json.dump(record, f)

    all_results_path = run_dir / "all_results.pkl"
    if all_results_path.exists():
        with open(all_results_path, "rb") as f:
            all_results = pickle.load(f)
        for r in all_results:
            label = r["label"]
            if label in per_label:
                r["grad_sim_log"] = per_label[label]["grad_sim_log"]
                r["pairwise_snapshots"] = per_label[label]["pairwise_snapshots"]
        with open(all_results_path, "wb") as f:
            pickle.dump(all_results, f)
        print(f"Updated all_results.pkl with grad-sim data for {len(per_label)} labels", flush=True)

    if not args.keep_checkpoints:
        import shutil
        shutil.rmtree(ckpt_root)
        print(f"Cleaned up checkpoints", flush=True)

    print(f"Grad-sim done: {len(per_label)} labels, {sum(jr.success for jr in results)}/{len(results)} ok",
          flush=True)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _worker_main()
    else:
        main()
