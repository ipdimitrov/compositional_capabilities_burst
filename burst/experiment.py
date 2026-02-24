"""Pure-bijection burst experiment with configurable depth and burst position.

Launches parallel worker processes, tracks progress, collects results.

Usage:
    python burst/experiment.py --depth 3 --burst-pos 2
    python burst/experiment.py --depth 5 --burst-pos 3 --schedules burst_100 burst_50 burst_10
"""
import sys, os, time, pickle, json, subprocess, argparse, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from synthetic.init import set_seed
from burst.data import BurstDataset, pad_pools_to_same_length
from burst.config import (
    N_A, SEED_BASE, DATA_SEED,
    CLASS_OTHER, CLASS_BURST,
    ExperimentConfig,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, tuple): return list(obj)
        return super().default(obj)


class DepthNData:
    """Pure-bijection depth-N composition data generator.

    bijections[0] = identity. bijections[1..N_A] = other-class functions.
    bijections[N_A+1] = b* (novel burst function).

    burst_pos (1-indexed): which position in the chain gets b*.

    Token format (depth=3):
      S [FN ... F1] ' ' [input] ' ' [after F1] ' ' ... ' ' [after FN]
    """

    def __init__(self, n_alph: int, seq_len: int, n_a: int, depth: int,
                 burst_pos: int, seed: int):
        assert 1 <= burst_pos <= depth, "burst_pos must be in [1, depth]"
        self.n_alph = n_alph
        self.seq_len = seq_len
        self.n_a = n_a
        self.depth = depth
        self.burst_pos = burst_pos
        rng = np.random.RandomState(seed)

        self.bijections = [np.arange(n_alph)]
        for _ in range(n_a + 1):
            self.bijections.append(rng.permutation(n_alph))

        self._build_vocab()
        self._build_splits(rng)

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

    def _build_splits(self, rng):
        na, b_star = self.n_a, self.n_a + 1
        r = list(range(1, na + 1))
        D, bp = self.depth, self.burst_pos

        other_combos = list(itertools.product(r, repeat=D))
        rng.shuffle(other_combos)
        self.other_train = [(CLASS_OTHER,) + combo for combo in other_combos]

        remaining_combos = list(itertools.product(r, repeat=D - 1))
        burst_tasks = []
        for combo in remaining_combos:
            fns = list(combo)
            fns.insert(D - bp, b_star)
            burst_tasks.append((CLASS_BURST,) + tuple(fns))
        self.burst_train = burst_tasks

    def _make_doc(self, task: tuple) -> np.ndarray:
        fns = task[1:]
        inp = np.random.choice(self.n_alph, size=self.seq_len, replace=True)
        sp = np.array([self.token_idx[' ']])

        cur = inp.copy()
        outs = []
        for fn_idx in reversed(fns):
            cur = self.bijections[fn_idx][cur]
            outs.append(cur.copy())

        doc = [np.array([self.token_idx['S']]),
               np.array([self.fn_tok[f] for f in fns]),
               sp, inp]
        for o in outs:
            doc.extend([sp, o])
        return np.concatenate(doc)

    def gen_pool(self, tasks: list, n: int) -> dict:
        return {t: np.array([self._make_doc(t) for _ in range(n)]) for t in tasks}


def build_data(cfg: dict, depth: int, burst_pos: int):
    set_seed(DATA_SEED)
    d = DepthNData(cfg["n_alphabets"], cfg["seq_len"], N_A, depth, burst_pos, DATA_SEED)
    nd, ne = cfg["n_docs_per_task"], cfg["n_eval_per_task"]

    bg_pool = d.gen_pool(d.other_train, nd)
    target_pool = d.gen_pool(d.burst_train, nd)

    eval_pools = {
        CLASS_OTHER: d.gen_pool(d.other_train[:min(8, len(d.other_train))], ne),
        CLASS_BURST: d.gen_pool(d.burst_train, ne),
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

    ref = eval_docs[CLASS_OTHER] if eval_docs[CLASS_OTHER].shape[0] > 1 else eval_docs[CLASS_BURST]
    sp_positions = np.where(ref[0] == d.token_idx[' '])[0]
    prompt_len = int(sp_positions[0]) + 1 + d.seq_len + 1

    cfg_out = dict(cfg)
    cfg_out["vocab_size"] = max(cfg["vocab_size"], d.vocab_size + 10)
    cfg_out["context_size"] = max(cfg["context_size"], ref.shape[1] + 5)

    task_info = {
        "n_a": N_A,
        "depth": depth,
        "burst_pos": burst_pos,
        "n_other_train": len(d.other_train),
        "n_burst_train": len(d.burst_train),
        "doc_len": int(ref.shape[1]), "prompt_len": prompt_len,
    }
    return target_pool, bg_pool, eval_docs, prompt_len, cfg_out, task_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--burst-pos", type=int, default=3)
    parser.add_argument("--schedules", nargs="+", default=None)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--grad-sim-batch-size", type=int, default=None)
    parser.add_argument("--grad-sim-n-workers", type=int, default=None)
    args = parser.parse_args()

    exp = ExperimentConfig(
        depth=args.depth,
        burst_pos=args.burst_pos,
    )
    if args.schedules:
        exp.schedules = args.schedules
    if args.n_seeds is not None:
        exp.n_seeds = args.n_seeds
    if args.n_workers is not None:
        exp.n_workers = args.n_workers
    if args.grad_sim_batch_size is not None:
        exp.train.grad_sim_batch_size = args.grad_sim_batch_size
    if args.grad_sim_n_workers is not None:
        exp.grad_sim_n_workers = args.grad_sim_n_workers

    base_cfg = exp.base_cfg

    tag = args.run_tag or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("data") / f"burst_d{exp.depth}_pos{exp.burst_pos}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_dir = run_dir / "_progress"
    progress_dir.mkdir(exist_ok=True)

    print(f"Output: {run_dir}\nDevice: {DEVICE}", flush=True)
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    print(f"\nBuilding data (depth={exp.depth}, burst_pos={exp.burst_pos})...", flush=True)
    tp, bp, ed, pl, cfg_out, ti = build_data(base_cfg, exp.depth, exp.burst_pos)
    print(f"  Other classes: {ti['n_other_train']}  "
          f"Burst class: {ti['n_burst_train']}  "
          f"doc_len: {ti['doc_len']}  prompt: {ti['prompt_len']}", flush=True)

    data_path = str(run_dir / "_data.pkl")
    with open(data_path, "wb") as f:
        pickle.dump((tp, bp, ed, pl, None), f)

    jobs = []
    for sched in exp.schedules:
        for seed_idx in range(exp.n_seeds):
            seed = SEED_BASE + seed_idx
            cfg = {**base_cfg, "seed": seed,
                   "vocab_size": cfg_out["vocab_size"],
                   "context_size": cfg_out["context_size"]}
            label = f"{sched}_s{seed}"
            jobs.append({"schedule": sched, "seed": seed, "cfg": cfg, "label": label})

    with open(run_dir / "config.json", "w") as f:
        json.dump({
            "base_cfg": base_cfg, "n_a": N_A, "seed_base": SEED_BASE,
            "n_seeds": exp.n_seeds, "schedules": exp.schedules, "n_jobs": len(jobs),
            "task_info": ti, "depth": exp.depth, "burst_pos": exp.burst_pos,
            "jobs": [{"label": j["label"], "schedule": j["schedule"],
                      "seed": j["seed"]} for j in jobs],
        }, f, indent=2, cls=NpEncoder)

    effective_n_workers = exp.grad_sim_n_workers if exp.grad_sim_n_workers is not None else exp.n_workers
    n_workers = min(len(jobs), effective_n_workers)
    steps_per_job = base_cfg["total_steps"] + base_cfg["reversion_steps"]

    tc = exp.train
    print(f"\nModel: {tc.n_layer}L/{tc.n_embd}d/{tc.n_head}H", flush=True)
    print(f"Jobs: {len(jobs)}, workers: {n_workers} "
          f"(grad_sim_batch_size={tc.grad_sim_batch_size})", flush=True)
    print(f"Steps/job: {tc.total_steps} train + {tc.reversion_steps} reversion", flush=True)
    print(f"Schedules: {exp.schedules}\n", flush=True)

    worker_script = str(Path(__file__).parent / "_worker.py")
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
        cur_steps = 0
        for pf in progress_dir.glob("*.txt"):
            txt = pf.read_text().strip()
            if txt.isdigit():
                cur_steps += int(txt)
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
                           f"peak={r['peak_burst']:.3f} "
                           f"t1/4={ql} auc={r['reversion_auc']:.0f} ({time.time()-t0:.0f}s)")
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
    if progress_dir.exists(): progress_dir.rmdir()
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

    print(f"\nPlot: python burst/plot.py {run_dir}", flush=True)


if __name__ == "__main__":
    main()
