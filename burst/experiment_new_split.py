"""Depth-3 pure-bijection burst experiment.

Composition: F_i(3) . F_j(2) . F_k(1)(x)   (bijections only)
A data: N_A bijections at each of 3 positions -> N_A^3 compositions
B data: 1 new bijection b* at position 3 -> b* . F_j . F_k

7 schedules x 1 seed = 7 parallel jobs
"""
import sys, os, time, pickle, json, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from synthetic.init import set_seed
from burst.data import BurstDataset, pad_pools_to_same_length

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCHEDULES = ["end_block", "uniform", # "mid_block", "ramp_up"
             "end_mixed_50b", "end_mixed_75b", "end_mixed_25b"]
N_A, NB_SEEN, SEED_BASE = 4, 10, 42
N_SEEDS = 3

BASE_CFG = {
    "n_alphabets": 10, "seq_len": 6,
    "n_layer": 6, "n_embd": 120, "n_head": 4,
    "vocab_size": 128, "context_size": 80,
    "lr": 3e-4, "weight_decay": 1e-3,
    "beta1": 0.9, "beta2": 0.95, "grad_clip": 1.0,
    "warmup_iters": 50, "min_lr": 6e-5,
    "batch_size": 512, "total_steps": 400, "p_target": 0.10,
    "undo_steps": 400, "eval_every": 10, "unlearn_threshold": 0.25,
    "n_docs_per_task": 500, "n_eval_per_task": 500,
}


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, tuple): return list(obj)
        return super().default(obj)


class Depth3Data:
    """Pure-bijection depth-3 composition data generator.

    bijections[0] = identity. bijections[1..N_A] = A functions.
    bijections[N_A+1] = b* (novel function for B data).

    Token format: S [F3 F2 F1] ' ' [input] ' ' [after F1] ' ' [after F2] ' ' [after F3]
    """

    def __init__(self, n_alph: int, seq_len: int, n_a: int, n_b_seen: int, seed: int):
        self.n_alph, self.seq_len, self.n_a = n_alph, seq_len, n_a
        rng = np.random.RandomState(seed)

        self.bijections = [np.arange(n_alph)]
        for _ in range(n_a + 1):
            self.bijections.append(rng.permutation(n_alph))

        self._build_vocab()
        self._build_splits(n_b_seen, rng)

    def _build_vocab(self):
        self.token, self.token_idx, self.fn_tok = {}, {}, {}
        idx = 0
        for i in range(self.n_alph):
            self.token[idx] = f"X{i}"
            self.token_idx[f"X{i}"] = idx
            idx += 1
        for i in range(len(self.bijections)):
            self.token[idx] = f"F{i}"
            self.token_idx[f"F{i}"] = idx
            self.fn_tok[i] = idx
            idx += 1
        for sp in (' ', '<PAD>', 'S'):
            self.token[idx] = sp
            self.token_idx[sp] = idx
            idx += 1
        self.vocab_size = idx

    def _build_splits(self, n_b_seen: int, rng):
        na, b_star = self.n_a, self.n_a + 1
        r = range(1, na + 1)

        self.a_comp_train = [("a3", fi, fj, fk) for fi in r for fj in r for fk in r]
        rng.shuffle(self.a_comp_train)

        all_b_pairs = [(fj, fk) for fj in r for fk in r]
        rng.shuffle(all_b_pairs)
        n_b_seen = min(n_b_seen, len(all_b_pairs))
        self.b_comp_train = [("b3", b_star, fj, fk) for fj, fk in all_b_pairs[:n_b_seen]]

    def _make_doc(self, task: tuple) -> np.ndarray:
        inp = np.random.choice(self.n_alph, size=self.seq_len, replace=True)
        sp = np.array([self.token_idx[' ']])
        f3, f2, f1 = task[1], task[2], task[3]
        cur = inp.copy()
        outs = []
        for fn_idx in (f1, f2, f3):
            cur = self.bijections[fn_idx][cur]
            outs.append(cur.copy())
        doc = [np.array([self.token_idx['S']]),
               np.array([self.fn_tok[f3], self.fn_tok[f2], self.fn_tok[f1]]),
               sp, inp]
        for o in outs:
            doc.extend([sp, o])
        return np.concatenate(doc)

    def gen_pool(self, tasks: list, n: int) -> dict:
        return {t: np.array([self._make_doc(t) for _ in range(n)]) for t in tasks}


def build_data(cfg: dict):
    set_seed(999)
    d = Depth3Data(cfg["n_alphabets"], cfg["seq_len"], N_A, NB_SEEN, 999)
    nd, ne = cfg["n_docs_per_task"], cfg["n_eval_per_task"]

    bg_pool = d.gen_pool(d.a_comp_train, nd)
    target_pool = d.gen_pool(d.b_comp_train, nd)

    eval_pools = {
        "A_comp": d.gen_pool(d.a_comp_train[:min(8, len(d.a_comp_train))], ne),
        "B_comp": d.gen_pool(d.b_comp_train, ne),
    }

    all_pools = [bg_pool, target_pool] + list(eval_pools.values())
    padded = pad_pools_to_same_length(*all_pools)
    bg_pool, target_pool = padded[0], padded[1]
    for i, k in enumerate(eval_pools):
        eval_pools[k] = padded[i + 2]

    def _cat(pool):
        if not pool:
            return np.zeros((1, list(bg_pool.values())[0].shape[1]), dtype=np.int64)
        return np.concatenate(list(pool.values()))

    eval_docs = {k: _cat(v) for k, v in eval_pools.items()}

    ref = eval_docs["A_comp"] if eval_docs["A_comp"].shape[0] > 1 else eval_docs["B_comp"]
    sp_positions = np.where(ref[0] == d.token_idx[' '])[0]
    prompt_len = int(sp_positions[0]) + 1 + d.seq_len + 1

    cfg_out = dict(cfg)
    cfg_out["vocab_size"] = max(cfg["vocab_size"], d.vocab_size + 10)
    cfg_out["context_size"] = max(cfg["context_size"], ref.shape[1] + 5)

    task_info = {
        "n_a": N_A, "n_b_seen": NB_SEEN,
        "n_a_comp_train": len(d.a_comp_train),
        "n_b_comp_train": len(d.b_comp_train),
        "doc_len": int(ref.shape[1]), "prompt_len": prompt_len,
    }
    return target_pool, bg_pool, eval_docs, prompt_len, cfg_out, task_info


def main():
    run_dir = Path("data") / f"burst_d3_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_dir = run_dir / "_progress"
    progress_dir.mkdir(exist_ok=True)

    print(f"Output: {run_dir}\nDevice: {DEVICE}", flush=True)
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    print(f"\nBuilding data (pure bijections, n_B_seen={NB_SEEN})...", flush=True)
    tp, bp, ed, pl, cfg_out, ti = build_data(BASE_CFG)
    print(f"  A_comp: {ti['n_a_comp_train']}  "
          f"B_comp: {ti['n_b_comp_train']}  "
          f"doc_len: {ti['doc_len']}  prompt: {ti['prompt_len']}", flush=True)

    data_path = str(run_dir / "_data.pkl")
    with open(data_path, "wb") as f:
        pickle.dump((tp, bp, ed, pl, None), f)

    jobs = []
    for sched in SCHEDULES:
        for seed_idx in range(N_SEEDS):
            seed = SEED_BASE + seed_idx
            cfg = {**BASE_CFG, "seed": seed, "n_b_seen": NB_SEEN,
                   "vocab_size": cfg_out["vocab_size"], "context_size": cfg_out["context_size"]}
            label = f"{sched}_s{seed}"
            jobs.append({"schedule": sched, "seed": seed, "cfg": cfg,
                         "label": label, "n_b_seen": NB_SEEN})

    with open(run_dir / "config.json", "w") as f:
        json.dump({
            "base_cfg": BASE_CFG, "n_a": N_A, "nb_seen": NB_SEEN, "seed_base": SEED_BASE,
            "n_seeds": N_SEEDS, "schedules": SCHEDULES, "n_jobs": len(jobs), "task_info": ti,
            "jobs": [{"label": j["label"], "schedule": j["schedule"],
                      "seed": j["seed"], "n_b_seen": j["n_b_seen"]} for j in jobs],
        }, f, indent=2, cls=NpEncoder)

    n_workers = min(len(jobs), 15)
    steps_per_job = BASE_CFG["total_steps"] + BASE_CFG["undo_steps"]

    print(f"\nModel: {BASE_CFG['n_layer']}L/{BASE_CFG['n_embd']}d/{BASE_CFG['n_head']}H", flush=True)
    print(f"Jobs: {len(jobs)}, workers: {n_workers}", flush=True)
    print(f"Steps/job: {BASE_CFG['total_steps']} train + {BASE_CFG['undo_steps']} undo", flush=True)
    print(f"Schedules: {SCHEDULES}\n", flush=True)

    worker_script = str(Path(__file__).parent / "_worker_new_split.py")
    t0 = time.time()

    def launch(idx, job):
        job_path = str(run_dir / f"_job_{idx}.pkl")
        with open(job_path, "wb") as f:
            pickle.dump(job, f)
        return subprocess.Popen(
            [sys.executable, worker_script,
             "--job-path", job_path, "--data-path", data_path,
             "--run-dir", str(run_dir), "--progress-dir", str(progress_dir)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    active = {}
    next_idx = 0
    for _ in range(min(n_workers, len(jobs))):
        j = jobs[next_idx]
        active[j["label"]] = (j, launch(next_idx, j))
        next_idx += 1

    pbar = tqdm(total=len(jobs) * steps_per_job, desc="All jobs", unit="step", ncols=120)
    n_done, prev_steps = 0, 0

    while n_done < len(jobs):
        time.sleep(2)
        cur_steps = sum(
            int(pf.read_text().strip())
            for pf in progress_dir.glob("*.txt")
            if pf.read_text().strip().isdigit())
        if cur_steps > prev_steps:
            pbar.update(cur_steps - prev_steps)
            prev_steps = cur_steps

        for label in [l for l, (_, proc) in active.items() if proc.poll() is not None]:
            job, proc = active.pop(label)
            n_done += 1
            rp = run_dir / f"{job['label']}.pkl"
            if rp.exists():
                with open(rp, "rb") as f:
                    r = pickle.load(f)
                ql = r.get('quarter_life', '?')
                tqdm.write(f"  [{n_done}/{len(jobs)}] {label:30s} "
                           f"peak={r['train_end_B_comp']:.3f} "
                           f"t1/4={ql} auc={r['undo_auc']:.0f} ({time.time()-t0:.0f}s)")
            else:
                se = proc.stderr.read().decode() if proc.stderr else ""
                tqdm.write(f"  FAIL [{n_done}/{len(jobs)}]: {label} (exit {proc.returncode})")
                if se:
                    tqdm.write(f"    {se[:500]}")

            if next_idx < len(jobs):
                j = jobs[next_idx]
                active[j["label"]] = (j, launch(next_idx, j))
                next_idx += 1

    pbar.update(pbar.total - pbar.n)
    pbar.close()
    total_time = time.time() - t0
    print(f"\nAll {n_done} jobs done in {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

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
        os.remove(data_path)
    except OSError:
        pass

    print(f"Results: {run_dir} ({len(all_results)} ok)", flush=True)

    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            timeout=2).decode().strip().split(",")
        used, total = float(smi[0]), float(smi[1])
        per_job = used / max(n_workers, 1)
        print(f"\nVRAM: ~{per_job:.0f} MB/job, ran {n_workers} parallel, "
              f"could fit ~{int(total / per_job)} ({total/1024:.0f} GB total)", flush=True)
    except Exception:
        pass

    print(f"\nPlot: .venv/bin/python burst/plot_new_split.py {run_dir}", flush=True)


if __name__ == "__main__":
    main()
