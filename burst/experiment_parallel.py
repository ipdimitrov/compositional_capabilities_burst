"""
Parallel Burst Schedule Experiment — Cross-Family (bijection + permutation)
===========================================================================
A tasks: atomic bijections (depth-1), always present
B tasks: bijection ∘ permutation (depth-2), target for burstiness
Held-out: bijection ∘ permutation ∘ bijection (depth-3), never trained

Eval: free generation (autoregressive), no teacher forcing.
Undo: passive forgetting (A-only, correct labels).
"""
import sys, os, time, pickle, json, math, signal, subprocess, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from synthetic.init import set_seed
from burst.data import CrossFamilyData, pad_pools_to_same_length

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
    "n_bij_functions": 4,
    "n_perm_functions": 5,
    "n_target": 15,
    "n_docs_per_task": 2000,
    "n_eval_per_task": 400,

    "n_layer": 4,
    "n_embd": 96,
    "n_head": 4,
    "vocab_size": 128,
    "context_size": 64,

    "lr": 3e-4,
    "weight_decay": 1e-3,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "warmup_iters": 50,
    "min_lr": 6e-5,

    "batch_size": 256,
    "total_steps": 600,
    "p_target": 0.10,
    "undo_steps": 400,
    "eval_every": 50,
    "unlearn_threshold": 0.25,
}


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, tuple): return list(obj)
        return super().default(obj)


def build_data(cfg):
    set_seed(cfg["seed_base"])
    data = CrossFamilyData(
        n_alphabets=cfg["n_alphabets"],
        seq_len=cfg["seq_len"],
        n_bij_functions=cfg["n_bij_functions"],
        n_perm_functions=cfg["n_perm_functions"],
        n_target=cfg["n_target"],
        seed=cfg["seed_base"],
    )

    n_docs = cfg["n_docs_per_task"]
    n_eval = cfg["n_eval_per_task"]

    target_pool = data.generate_pool(data.b_tasks, n_docs)
    bg_pool = data.generate_pool(data.a_tasks, n_docs)
    eval_target = data.generate_pool(data.b_tasks, n_eval)
    eval_bg = data.generate_pool(data.a_tasks[:5], n_eval)
    eval_heldout = data.generate_pool(data.heldout_tasks[:10], n_eval)

    target_pool, bg_pool, eval_target, eval_bg, eval_heldout = pad_pools_to_same_length(
        target_pool, bg_pool, eval_target, eval_bg, eval_heldout
    )

    eval_docs = {
        "target": np.concatenate(list(eval_target.values())),
        "background": np.concatenate(list(eval_bg.values())),
        "heldout": np.concatenate(list(eval_heldout.values())),
    }

    prompt_len = data.get_prompt_len(eval_docs["target"])
    space_pos = data.get_space_pos(eval_docs["target"])

    cfg["vocab_size"] = max(cfg["vocab_size"], data.vocab_size + 10)
    cfg["context_size"] = max(cfg["context_size"], eval_docs["target"].shape[1] + 5)

    task_info = {
        "a_tasks": [str(t) for t in data.a_tasks],
        "b_tasks": [str(t) for t in data.b_tasks],
        "heldout_tasks": [str(t) for t in data.heldout_tasks[:10]],
        "n_bij": len(data.bijections),
        "n_perm": len(data.permutations),
        "vocab_size": data.vocab_size,
        "max_doc_len": int(eval_docs["target"].shape[1]),
        "prompt_len": prompt_len,
    }

    return target_pool, bg_pool, eval_docs, prompt_len, space_pos, task_info


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
    run_dir = Path("data") / f"burst_crossfam_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_dir = run_dir / "_progress"
    progress_dir.mkdir(exist_ok=True)

    print(f"Output: {run_dir}", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    print("\nBuilding cross-family data (bijections + permutations)...", flush=True)
    target_pool, bg_pool, eval_docs, prompt_len, space_pos, task_info = build_data(BASE_CFG)
    print(f"A tasks (bijections): {len(bg_pool)}", flush=True)
    print(f"B tasks (bij∘perm):   {len(target_pool)}", flush=True)
    print(f"Held-out (bij∘perm∘bij): {len(eval_docs['heldout'])} eval docs", flush=True)
    print(f"Vocab: {task_info['vocab_size']}, Doc len: {task_info['max_doc_len']}, Prompt len: {prompt_len}", flush=True)

    jobs = build_jobs()
    for job in jobs:
        job["cfg"]["vocab_size"] = BASE_CFG["vocab_size"]
        job["cfg"]["context_size"] = BASE_CFG["context_size"]

    n_workers = min(len(jobs), 10)
    steps_per_job = BASE_CFG["total_steps"] + BASE_CFG["undo_steps"]
    total_steps = len(jobs) * steps_per_job

    print(f"\nModel: {BASE_CFG['n_layer']}L/{BASE_CFG['n_embd']}d/{BASE_CFG['n_head']}H", flush=True)
    print(f"Total jobs: {len(jobs)}, workers: {n_workers}", flush=True)
    print(f"Steps per job: {BASE_CFG['total_steps']} train + {BASE_CFG['undo_steps']} undo", flush=True)
    print(f"Eval: free generation (autoregressive) every {BASE_CFG['eval_every']} steps", flush=True)
    print(f"Batch size: {BASE_CFG['batch_size']}", flush=True)

    shared_data_path = str(run_dir / "_shared_data.pkl")
    with open(shared_data_path, "wb") as f:
        pickle.dump((target_pool, bg_pool, eval_docs, prompt_len, space_pos), f)

    with open(run_dir / "config.json", "w") as f:
        json.dump({"base_cfg": BASE_CFG, "n_jobs": len(jobs),
                    "task_info": task_info,
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
                    ho_str = f"held={result.get('heldout_train_end', 0):.4f}" if result.get('heldout_train_end') else ""
                    tqdm.write(
                        f"  [{n_done}/{len(jobs)}] {job['label']:30s} "
                        f"train_B={result['train_end_acc']:.4f}  "
                        f"undo_B={result['undo_end_acc']:.4f}  "
                        f"auc={result['undo_auc']:.0f}  "
                        f"{ho_str}  ({elapsed:.0f}s)"
                    )
                else:
                    stderr_out = proc.stderr.read().decode() if proc.stderr else ""
                    tqdm.write(f"  FAILED [{n_done}/{len(jobs)}]: {job['label']} (exit {ret})")
                    if stderr_out:
                        tqdm.write(f"    {stderr_out[:500]}")

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
