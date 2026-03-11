"""Pure-bijection burst experiment with configurable depth and burst position.

Launches parallel worker processes for training, tracks progress, collects
results.  Grad-sim is computed post-hoc by burst/grad_sim.py on the saved
checkpoints.

Training structure:
  1. Pretrain: one shared model trained on all-but-special for pre_burst_steps.
               The checkpoint is shared across all 10 seeds for a given schedule.
  2. Burst:    10 runs per schedule, each starting from the shared pretrain ckpt.
               Burst phase length varies inversely with burst concentration so
               all schedules see the same total special-class examples.
  3. Reversal: all-but-special, same length for all schedules.

Output folder layout:
  data/<date>_<time>_burst_d<depth>_pos<pos>/
    results/
      config.json
      analysis_report.pdf
      plots/
      presentation/
      grad_cosine_sim/
    logs/
      all_results.pkl
      _data.pkl
      pretrain_ckpt.pt
      checkpoints/
      task_distributions/
      <label>.pkl  (per-run result pickles)

Usage:
    python burst/experiment.py --depth 3 --burst-pos 2
    python burst/experiment.py --depth 5 --burst-pos 3 --n-a 6 --schedules burst_100 burst_50 burst_25
"""
import sys, os, time, pickle, json, subprocess, argparse, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from omegaconf import OmegaConf

from synthetic.init import set_seed
from net.nanogpt import nanoGPT
from net.runner import configure_optimizers, update_phase_lr
from burst.data import BurstDataset, pad_pools_to_same_length
from burst.config import (
    N_A, SEED_BASE, DATA_SEED,
    CLASS_OTHER, CLASS_BURST,
    ExperimentConfig, TrainConfig,
    reversion_life_key, reversion_life_label,
    burst_steps_for_mode, batch_size_for_mode,
    BURST_BASE_STEPS, BURST_MODES, MODE_CURRENT, MODE_SCALED_BATCH,
)
from burst.gpu import gpu_cfg

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

    Each position p (1-indexed) has its own dedicated set of n_a bijections,
    matching the paper's design where F^(l) are non-overlapping across positions.

    Bijection layout:
      bijections[0]                      = identity (F0, unused in tasks)
      bijections[(p-1)*n_a + 1 .. p*n_a] = background functions for position p
      bijections[n_a*depth + 1]          = b* (novel burst function)

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

        # bijections[0] = identity; then n_a per position; then b*
        self.bijections = [np.arange(n_alph)]
        for _ in range(n_a * depth + 1):
            self.bijections.append(rng.permutation(n_alph))

        # b* is the last bijection
        self.b_star = n_a * depth + 1

        # position p (1-indexed) uses bijection indices pos_fns[p]
        self.pos_fns: dict[int, list[int]] = {
            p: list(range((p - 1) * n_a + 1, p * n_a + 1))
            for p in range(1, depth + 1)
        }

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
        D, bp = self.depth, self.burst_pos

        # Other-class tasks: at each position p, pick one function from pos_fns[p]
        # All combinations of (f_pos1, f_pos2, ..., f_posD) where f_posp in pos_fns[p]
        per_pos = [self.pos_fns[p] for p in range(1, D + 1)]
        other_combos = list(itertools.product(*per_pos))
        rng.shuffle(other_combos)
        self.other_train = [(CLASS_OTHER,) + combo for combo in other_combos]

        # Burst tasks: same as other but position bp is replaced by b*
        non_burst_positions = [p for p in range(1, D + 1) if p != bp]
        per_pos_no_bp = [self.pos_fns[p] for p in non_burst_positions]
        remaining_combos = list(itertools.product(*per_pos_no_bp))
        burst_tasks = []
        for combo in remaining_combos:
            fns = list(combo)
            # insert b* at the correct slot (positions are listed outermost-first,
            # so position bp sits at index D - bp from the left)
            fns.insert(D - bp, self.b_star)
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


def build_data(cfg: dict, depth: int, burst_pos: int, n_a: int):
    set_seed(DATA_SEED)
    d = DepthNData(cfg["n_alphabets"], cfg["seq_len"], n_a, depth, burst_pos, DATA_SEED)
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
        "n_a": n_a,
        "depth": depth,
        "burst_pos": burst_pos,
        "n_other_train": len(d.other_train),
        "n_burst_train": len(d.burst_train),
        "doc_len": int(ref.shape[1]), "prompt_len": prompt_len,
    }
    return target_pool, bg_pool, eval_docs, prompt_len, cfg_out, task_info


def run_pretrain(
    cfg: dict,
    pretrain_steps: int,
    bg_pool: dict,
    ckpt_path: Path | None = None,
    eval_docs: dict | None = None,
    prompt_len: int = 0,
    eval_every: int = 25,
    seed: int = DATA_SEED,
    save_checkpoint: bool = True,
    progress_prefix: str = "",
) -> dict:
    """Train one model on all-but-special for pretrain_steps, save checkpoint.

    Uses the provided seed (default DATA_SEED).

    Returns a pretrain log dict with step/loss/phase/acc_other/acc_burst lists
    so charts can show the pretraining trajectory.
    """
    from burst._worker import eval_free_gen
    from burst.config import EVAL_KEYS, PHASE_PRE_BURST

    set_seed(seed)
    net = nanoGPT(OmegaConf.create({
        "compile": False, "vocab_size": cfg["vocab_size"],
        "context_size": cfg["context_size"],
        "n_layer": cfg["n_layer"], "n_head": cfg["n_head"],
        "n_embd": cfg["n_embd"], "dropout": 0.0, "bias": False, "mlp": True,
    })).to(DEVICE)
    if DEVICE == "cuda":
        net = torch.compile(net)

    optim_cfg = OmegaConf.create({
        "learning_rate": cfg["lr"], "weight_decay": cfg["weight_decay"],
        "beta1": cfg["beta1"], "beta2": cfg["beta2"],
        "grad_clip": cfg["grad_clip"],
    })
    optimizer = configure_optimizers(net, optim_cfg)
    scaler = torch.amp.GradScaler('cuda', enabled=DEVICE == "cuda")

    P = pretrain_steps
    warmup = cfg["warmup_iters"]
    lr_max = cfg["lr"]
    lr_pe = cfg.get("lr_pretrain_end_frac", 0.3)
    lr_be = cfg.get("lr_burst_end_frac", 0.1)
    lr_re = cfg.get("lr_reversion_end_frac", 0.01)
    T_dummy, U_dummy = cfg.get("total_steps", 80), cfg["reversion_steps"]

    bg_ids = list(bg_pool.keys())
    bs = cfg["batch_size"]

    log: dict[str, list] = {"step": [], "loss": [], "phase": []}
    for k in EVAL_KEYS:
        log[k] = []

    net.train()
    for s in range(pretrain_steps):
        per = bs // len(bg_ids)
        rem = bs % len(bg_ids)
        parts = []
        for i, tid in enumerate(bg_ids):
            k = per + (1 if i < rem else 0)
            if k > 0:
                idx = np.random.randint(len(bg_pool[tid]), size=k)
                parts.append(bg_pool[tid][idx])
        batch_np = np.concatenate(parts)[np.random.permutation(bs)]

        dat = torch.as_tensor(batch_np, dtype=torch.long, device=DEVICE)
        inp, tgt = dat[:, :-1], dat[:, 1:]
        update_phase_lr(
            s + 1, optimizer, warmup, P, T_dummy, U_dummy,
            lr_max, lr_pe, lr_be, lr_re,
        )
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

        if eval_docs and prompt_len > 0 and (s % eval_every == 0 or s == pretrain_steps - 1):
            loss_val = loss.item()
            log["step"].append(s)
            log["loss"].append(loss_val)
            log["phase"].append(PHASE_PRE_BURST)
            for ek in EVAL_KEYS:
                pool_key = ek.removeprefix("acc_")
                log[ek].append(eval_free_gen(net, eval_docs[pool_key], prompt_len))
            net.train()

        if (s + 1) % 100 == 0 or s == pretrain_steps - 1:
            prefix = f"{progress_prefix} " if progress_prefix else ""
            print(f"{prefix}pretrain step {s+1}/{pretrain_steps}  loss={loss.item():.4f}", flush=True)

    if save_checkpoint:
        if ckpt_path is None:
            raise ValueError("ckpt_path is required when save_checkpoint=True")
        raw = getattr(net, "_orig_mod", net)
        torch.save(raw.state_dict(), ckpt_path)
        print(f"  Pretrain checkpoint saved: {ckpt_path}", flush=True)
    return log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--burst-pos", type=int, default=3)
    parser.add_argument("--burst-mode", choices=BURST_MODES, default=MODE_CURRENT)
    parser.add_argument("--n-a", type=int, default=N_A)
    parser.add_argument("--schedules", nargs="+", default=None)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--run-probes", action="store_true", default=False)
    parser.add_argument("--run-next-token-probes", action="store_true", default=False)
    parser.add_argument("--run-adl", action="store_true", default=False)
    args = parser.parse_args()

    exp = ExperimentConfig(
        depth=args.depth,
        burst_pos=args.burst_pos,
        burst_mode=args.burst_mode,
        run_probes=args.run_probes,
        run_next_token_probes=args.run_next_token_probes,
        run_adl=args.run_adl,
    )
    if args.schedules:
        exp.schedules = args.schedules
    if args.n_seeds is not None:
        exp.n_seeds = args.n_seeds
    if args.n_workers is not None:
        exp.n_workers = args.n_workers

    base_cfg = exp.base_cfg

    burst_mode = exp.burst_mode
    tag = args.run_tag or datetime.now().strftime("%Y%m%d-%H%M%S")
    mode_suffix = f"_{burst_mode}" if burst_mode != MODE_CURRENT else ""
    run_dir = Path("data") / f"{tag}_burst_d{exp.depth}_pos{exp.burst_pos}{mode_suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results_dir = run_dir / "results"
    logs_dir = run_dir / "logs"
    results_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    (results_dir / "plots").mkdir(exist_ok=True)
    (results_dir / "presentation").mkdir(exist_ok=True)
    (results_dir / "grad_cosine_sim").mkdir(exist_ok=True)

    progress_dir = logs_dir / "_progress"
    progress_dir.mkdir(exist_ok=True)

    print(f"Output: {run_dir}\nDevice: {DEVICE}", flush=True)
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    n_a = args.n_a
    print(f"\nBuilding data (depth={exp.depth}, burst_pos={exp.burst_pos}, n_a={n_a})...", flush=True)
    tp, bp, ed, pl, cfg_out, ti = build_data(base_cfg, exp.depth, exp.burst_pos, n_a)
    print(f"  Other classes: {ti['n_other_train']}  "
          f"Burst class: {ti['n_burst_train']}  "
          f"doc_len: {ti['doc_len']}  prompt: {ti['prompt_len']}", flush=True)

    data_path = str(logs_dir / "_data.pkl")
    with open(data_path, "wb") as f:
        pickle.dump((tp, bp, ed, pl, None), f)

    # --- Pretrain: one shared checkpoint for all seeds ---
    pretrain_ckpt_path = logs_dir / "pretrain_ckpt.pt"
    P = base_cfg["pre_burst_steps"]
    pretrain_cfg = {**base_cfg,
                    "vocab_size": cfg_out["vocab_size"],
                    "context_size": cfg_out["context_size"]}
    pretrain_attempt = 0
    while True:
        pretrain_attempt += 1
        print(f"\nPretraining shared checkpoint ({P} steps on all-but-special)"
              f" — attempt {pretrain_attempt}...", flush=True)
        pretrain_log = run_pretrain(
            pretrain_cfg, P, bp, pretrain_ckpt_path,
            eval_docs=ed, prompt_len=pl, eval_every=base_cfg["eval_every"],
        )
        peak_acc_other = max(pretrain_log.get("acc_other", [0.0]))
        if peak_acc_other >= 0.99:
            print(f"  Pretrain OK: peak acc_other={peak_acc_other:.4f}", flush=True)
            break
        print(f"  Pretrain acc_other={peak_acc_other:.4f} < 0.99, retrying...", flush=True)
    with open(logs_dir / "pretrain_log.pkl", "wb") as f:
        pickle.dump(pretrain_log, f)

    pretrain_log_path = str(logs_dir / "pretrain_log.pkl")
    jobs = []
    for sched in exp.schedules:
        sched_total_steps = burst_steps_for_mode(sched, burst_mode, BURST_BASE_STEPS)
        sched_batch_size = batch_size_for_mode(sched, burst_mode, base_cfg["batch_size"])
        for seed_idx in range(exp.n_seeds):
            seed = SEED_BASE + seed_idx
            cfg = {**base_cfg, "seed": seed,
                   "vocab_size": cfg_out["vocab_size"],
                   "context_size": cfg_out["context_size"],
                   "total_steps": sched_total_steps,
                   "batch_size": sched_batch_size}
            label = f"{sched}_s{seed}"
            jobs.append({
                "schedule": sched, "seed": seed, "cfg": cfg, "label": label,
                "pretrain_ckpt": str(pretrain_ckpt_path),
                "pretrain_log_path": pretrain_log_path,
            })

    with open(results_dir / "config.json", "w") as f:
        json.dump({
            "base_cfg": base_cfg, "n_a": n_a, "seed_base": SEED_BASE,
            "n_seeds": exp.n_seeds, "schedules": exp.schedules, "n_jobs": len(jobs),
            "task_info": ti, "depth": exp.depth, "burst_pos": exp.burst_pos,
            "burst_mode": burst_mode,
            "run_probes": exp.run_probes,
            "run_next_token_probes": exp.run_next_token_probes,
            "run_adl": exp.run_adl,
            "burst_base_steps": BURST_BASE_STEPS,
            "pretrain_ckpt": str(pretrain_ckpt_path),
            "jobs": [{"label": j["label"], "schedule": j["schedule"],
                      "seed": j["seed"],
                      "total_steps": j["cfg"]["total_steps"],
                      "batch_size": j["cfg"]["batch_size"]} for j in jobs],
        }, f, indent=2, cls=NpEncoder)

    if burst_mode == MODE_SCALED_BATCH:
        max_bs = max(j["cfg"]["batch_size"] for j in jobs)
        safe_workers = gpu_cfg.train_workers_for_batch_size(max_bs, base_cfg["batch_size"])
        n_procs = min(len(jobs), safe_workers)
    else:
        n_procs = min(len(jobs), exp.n_workers)
    jobs_per_proc = max(1, (len(jobs) + n_procs - 1) // n_procs)
    print(f"  {gpu_cfg.summary()}", flush=True)
    print(f"  Layout: {n_procs} processes x ~{jobs_per_proc} jobs/proc", flush=True)

    tc = exp.train
    print(f"\nModel: {tc.n_layer}L/{tc.n_embd}d/{tc.n_head}H", flush=True)
    print(f"Burst mode: {burst_mode}", flush=True)
    print(f"Jobs: {len(jobs)}, parallel processes: {n_procs}, "
          f"jobs/process: ~{jobs_per_proc}", flush=True)
    print(f"Schedules: {exp.schedules}", flush=True)
    for sched in exp.schedules:
        T_s = burst_steps_for_mode(sched, burst_mode, BURST_BASE_STEPS)
        bs_s = batch_size_for_mode(sched, burst_mode, base_cfg["batch_size"])
        print(f"  {sched}: burst_steps={T_s}  batch_size={bs_s}  "
              f"reversion={tc.reversion_steps}", flush=True)
    print()

    batched_script = str(Path(__file__).parent / "_worker_batched.py")
    t0 = time.time()

    job_chunks = [jobs[i:i + jobs_per_proc] for i in range(0, len(jobs), jobs_per_proc)]

    def launch_chunk(chunk_idx, chunk):
        chunk_path = str(logs_dir / f"_chunk_{chunk_idx}.pkl")
        with open(chunk_path, "wb") as f:
            pickle.dump(chunk, f)
        return subprocess.Popen(
            [sys.executable, batched_script,
             "--jobs-path", chunk_path, "--data-path", data_path,
             "--run-dir", str(logs_dir), "--progress-dir", str(progress_dir)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    active_chunks: dict[int, tuple[list, subprocess.Popen]] = {}
    next_chunk = 0
    for _ in range(min(n_procs, len(job_chunks))):
        chunk = job_chunks[next_chunk]
        active_chunks[next_chunk] = (chunk, launch_chunk(next_chunk, chunk))
        next_chunk += 1

    max_steps_per_job = max(
        P + burst_steps_for_mode(s, burst_mode, BURST_BASE_STEPS) + tc.reversion_steps
        for s in exp.schedules
    )
    pbar = tqdm(total=len(jobs) * max_steps_per_job, desc="All jobs", unit="step", ncols=120)
    n_done, prev_steps = 0, 0
    reported_jobs: set[str] = set()

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

        for job in jobs:
            label = job["label"]
            if label in reported_jobs:
                continue
            rp = logs_dir / f"{label}.pkl"
            if rp.exists():
                reported_jobs.add(label)
                n_done += 1
                with open(rp, "rb") as f:
                    r = pickle.load(f)
                thresholds = TrainConfig().reversion_thresholds
                first_key = reversion_life_key(thresholds[0])
                first_lbl = reversion_life_label(thresholds[0])
                lv = r.get(first_key, '?')
                tqdm.write(f"  [{n_done}/{len(jobs)}] {label:30s} "
                           f"peak={r['peak_burst']:.3f} "
                           f"{first_lbl}={lv} auc={r['reversion_auc']:.0f} ({time.time()-t0:.0f}s)")

        done_chunks = [ci for ci, (_, proc) in active_chunks.items() if proc.poll() is not None]
        for ci in done_chunks:
            chunk, proc = active_chunks.pop(ci)
            if proc.returncode != 0:
                se = proc.stderr.read().decode() if proc.stderr else ""
                for j in chunk:
                    if j["label"] not in reported_jobs:
                        reported_jobs.add(j["label"])
                        n_done += 1
                        tqdm.write(f"  FAIL [{n_done}/{len(jobs)}]: {j['label']} "
                                   f"(chunk {ci} exit {proc.returncode})")
                        if se:
                            tqdm.write(f"    {se[:500]}")

            if next_chunk < len(job_chunks):
                c = job_chunks[next_chunk]
                active_chunks[next_chunk] = (c, launch_chunk(next_chunk, c))
                next_chunk += 1

    pbar.update(pbar.total - pbar.n)
    pbar.close()
    total_time = time.time() - t0
    print(f"\nAll {n_done} jobs done in {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    all_results = []
    for job in jobs:
        rp = logs_dir / f"{job['label']}.pkl"
        if rp.exists():
            with open(rp, "rb") as f:
                all_results.append(pickle.load(f))
    with open(logs_dir / "all_results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    for f in progress_dir.glob("*"):
        f.unlink()
    if progress_dir.exists(): progress_dir.rmdir()
    for f in logs_dir.glob("_chunk_*.pkl"):
        f.unlink()

    print(f"Results: {run_dir} ({len(all_results)} ok)", flush=True)
    print(f"\nGrad-sim: python burst/grad_sim.py {run_dir}", flush=True)
    print(f"Plot:     python burst/plot.py {run_dir}", flush=True)


if __name__ == "__main__":
    main()
