"""
Parallel Burst Schedule Experiment (subprocess-based)
=====================================================
Runs many experiment configs in parallel using subprocess workers.
Each worker loads shared data from disk, trains its own model on GPU.

Undo = passive forgetting (A-only, correct labels). No shuffled labels.
"""
import sys, os, time, pickle, json, math, signal, subprocess, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from synthetic.init import set_seed
from burst.data import build_function_pool, tag_tasks, generate_pool
from burst.config import BurstExperimentConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CORE_SCHEDULES = [
    "uniform", "end_block", "mid_block", "early_block",
    "end_mixed", "bookend", "early_block_2x", "late_ramp",
    "cyclic", "front_heavy",
]

BASE_CFG = {
    "seed_base": 42,
    "n_alphabets": 10,
    "seq_len": 6,
    "depth": 5,
    "n_functions": 5,
    "n_train_compositions": 150,
    "n_target": 10,
    "exclusive_fn_idx": 5,
    "ndocuments": 100_000,
    "neval_documents": 10_000,

    "n_layer": 12,
    "n_embd": 120,
    "n_head": 12,
    "vocab_size": 512,
    "context_size": 50,

    "lr": 3e-4,
    "weight_decay": 1e-3,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "warmup_iters": 200,
    "min_lr": 6e-5,

    "batch_size": 512,
    "total_steps": 3_000,
    "p_target": 0.05,
    "undo_steps": 5_000,
    "eval_every": 200,
    "unlearn_threshold": 0.25,
}


def build_data(seed, cfg):
    bcfg = BurstExperimentConfig(
        seed=seed, n_alphabets=cfg["n_alphabets"], seq_len=cfg["seq_len"],
        depth=cfg["depth"], n_functions=cfg["n_functions"],
        n_train_compositions=cfg["n_train_compositions"],
        ndocuments=cfg["ndocuments"], neval_documents=cfg["neval_documents"],
    )
    set_seed(seed)
    syn, composed_functions, info = build_function_pool(bcfg)
    target_ids, bg_ids, fn_lookup = tag_tasks(
        info, composed_functions, n_target=cfg["n_target"],
        exclusive_fn_idx=cfg.get("exclusive_fn_idx"),
    )
    n_docs = max(cfg["ndocuments"] // max(len(bg_ids), 1), 500)
    target_pool = generate_pool(syn, target_ids, fn_lookup, n_docs)
    bg_pool = generate_pool(syn, bg_ids, fn_lookup, n_docs)
    eval_target = generate_pool(syn, target_ids, fn_lookup, cfg["neval_documents"])
    eval_bg = generate_pool(syn, bg_ids[:5], fn_lookup, cfg["neval_documents"] // 5)
    eval_docs = {"target": np.concatenate(list(eval_target.values())),
                 "background": np.concatenate(list(eval_bg.values()))}
    sp_idx = syn.token_idx[" "]
    space_pos = int(np.where(eval_docs["target"][0] == sp_idx)[0][-1])
    return target_pool, bg_pool, eval_docs, space_pos


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, tuple): return list(obj)
        return super().default(obj)


def build_jobs():
    jobs = []
    seed = BASE_CFG["seed_base"]

    for schedule in CORE_SCHEDULES:
        cfg = dict(BASE_CFG)
        cfg["seed"] = seed
        jobs.append({"schedule": schedule, "seed": seed, "cfg": cfg,
                     "label": f"{schedule}"})

    for p_target in [0.02, 0.10]:
        for schedule in ["uniform", "end_block"]:
            cfg = dict(BASE_CFG)
            cfg["seed"] = seed
            cfg["p_target"] = p_target
            jobs.append({"schedule": schedule, "seed": seed, "cfg": cfg,
                         "label": f"{schedule}_p{p_target:.2f}"})

    return jobs


def main():
    run_dir = Path("data") / f"burst_parallel_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_dir = run_dir / "_progress"
    progress_dir.mkdir(exist_ok=True)

    print(f"Output: {run_dir}", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    jobs = build_jobs()
    n_workers = min(len(jobs), 10)
    steps_per_job = BASE_CFG["total_steps"] + BASE_CFG["undo_steps"]
    total_steps = len(jobs) * steps_per_job

    print(f"\nTotal jobs: {len(jobs)}, workers: {n_workers} (all parallel)", flush=True)
    print(f"Steps per job: {BASE_CFG['total_steps']} train + {BASE_CFG['undo_steps']} undo", flush=True)
    print(f"Batch size: {BASE_CFG['batch_size']}", flush=True)
    print(f"Est. VRAM: {n_workers * 2.9:.1f} GB / 32 GB", flush=True)

    print("\nBuilding shared data pool...", flush=True)
    target_pool, bg_pool, eval_docs, space_pos = build_data(BASE_CFG["seed_base"], BASE_CFG)
    print(f"A:B split = {len(bg_pool)}:{len(target_pool)}", flush=True)

    shared_data_path = str(run_dir / "_shared_data.pkl")
    with open(shared_data_path, "wb") as f:
        pickle.dump((target_pool, bg_pool, eval_docs, space_pos), f)

    with open(run_dir / "config.json", "w") as f:
        json.dump({"base_cfg": BASE_CFG, "n_jobs": len(jobs),
                    "jobs": [{"label": j["label"], "schedule": j["schedule"],
                              "seed": j["seed"]} for j in jobs]},
                   f, indent=2, cls=NpEncoder)

    job_cfgs_path = str(run_dir / "_job_configs.pkl")
    with open(job_cfgs_path, "wb") as f:
        pickle.dump(jobs, f)

    print(f"\nLaunching {len(jobs)} jobs ({n_workers} concurrent)...\n", flush=True)
    t0 = time.time()

    worker_script = str(Path(__file__).parent / "_worker.py")

    def launch_job(idx, job):
        job_path = str(run_dir / f"_job_{idx}.pkl")
        with open(job_path, "wb") as f:
            pickle.dump(job, f)
        cmd = [
            sys.executable, worker_script,
            "--job-path", job_path,
            "--data-path", shared_data_path,
            "--run-dir", str(run_dir),
            "--progress-dir", str(progress_dir),
        ]
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    active = {}
    job_queue = list(enumerate(jobs))
    next_job_idx = 0

    for _ in range(min(n_workers, len(jobs))):
        idx, job = job_queue[next_job_idx]
        active[job["label"]] = (job, launch_job(idx, job))
        next_job_idx += 1

    pbar = tqdm(total=total_steps, desc="All jobs", unit="step", ncols=120,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")

    n_done = 0
    prev_steps = 0

    while n_done < len(jobs):
        time.sleep(2)

        cur_steps = 0
        for pf in progress_dir.glob("*.txt"):
            try:
                cur_steps += int(pf.read_text().strip())
            except (ValueError, OSError):
                pass

        delta = cur_steps - prev_steps
        if delta > 0:
            pbar.update(delta)
            prev_steps = cur_steps

        finished = []
        for label, (job, proc) in active.items():
            ret = proc.poll()
            if ret is not None:
                finished.append(label)
                n_done += 1
                elapsed = time.time() - t0
                result_path = run_dir / f"{job['label']}.pkl"
                if result_path.exists():
                    with open(result_path, "rb") as f:
                        result = pickle.load(f)
                    tqdm.write(
                        f"  [{n_done}/{len(jobs)}] {job['label']:30s} "
                        f"train={result['train_end_acc']:.4f}  "
                        f"undo={result['undo_end_acc']:.4f}  "
                        f"auc={result['undo_auc']:.0f}  "
                        f"({elapsed:.0f}s)"
                    )
                else:
                    stderr_out = proc.stderr.read().decode() if proc.stderr else ""
                    tqdm.write(f"  FAILED [{n_done}/{len(jobs)}]: {job['label']} (exit {ret})")
                    if stderr_out:
                        tqdm.write(f"    {stderr_out[:300]}")

        for label in finished:
            del active[label]
            if next_job_idx < len(job_queue):
                idx, job = job_queue[next_job_idx]
                active[job["label"]] = (job, launch_job(idx, job))
                next_job_idx += 1

        try:
            smi = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                timeout=2).decode().strip().split(",")
            vram_str = f"{int(smi[0].strip())/1024:.1f}/{int(smi[1].strip())/1024:.0f}GB"
        except Exception:
            vram_str = "?"
        pbar.set_postfix({"done": f"{n_done}/{len(jobs)}", "VRAM": vram_str, "active": len(active)})

    pbar.update(pbar.total - pbar.n)
    pbar.close()

    total_time = time.time() - t0
    print(f"\nAll {n_done} jobs completed in {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    all_results = []
    for job in jobs:
        rp = run_dir / f"{job['label']}.pkl"
        if rp.exists():
            with open(rp, "rb") as f:
                all_results.append(pickle.load(f))

    with open(run_dir / "all_results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    for f in progress_dir.glob("*"):
        f.unlink()
    progress_dir.rmdir()
    for f in run_dir.glob("_job_*.pkl"):
        f.unlink()
    try:
        os.remove(shared_data_path)
    except OSError:
        pass
    try:
        os.remove(job_cfgs_path)
    except OSError:
        pass

    print(f"Results saved to {run_dir} ({len(all_results)} successful)", flush=True)
    print(f"\nRun: .venv/bin/python burst/plot_and_report.py {run_dir}", flush=True)


if __name__ == "__main__":
    main()
