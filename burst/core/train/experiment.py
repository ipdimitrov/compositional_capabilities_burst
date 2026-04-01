r"""Pure-bijection burst experiment with configurable depth and burst position.

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
    python -m burst.core train --depth 3 --burst-pos 2
    python -m burst.core train --depth 5 --burst-pos 3 --n-a 6 \\
        --schedules burst_100 burst_50 burst_25
"""

import argparse
import itertools
import json
import logging
import pickle
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import torch
from tqdm import tqdm

from burst.config import (
    BURST_BASE_STEPS,
    BURST_MODES,
    CLASS_BURST,
    CLASS_OTHER,
    DATA_SEED,
    DEFAULT_DETERMINISTIC,
    DEFAULT_REPRO_SEED,
    MODE_CURRENT,
    N_A,
    ExperimentConfig,
    TrainConfig,
    batch_size_for_mode,
    burst_steps_for_mode,
    reversion_life_key,
    reversion_life_label,
)
from burst.core.data import pad_pools_to_same_length
from burst.core.gpu import gpu_cfg
from burst.core.repro import set_reproducibility, write_repro_manifest
from burst.core.train_utils import (
    DEVICE,
    _cross_entropy_logits_BTV_targets_BT,
    make_net,
    make_optim_cfg,
    make_scaler,
)
from net.runner import configure_optimizers, update_phase_lr
from synthetic.init import set_seed

logger = logging.getLogger(__name__)

PRETRAIN_ACC_THRESHOLD = 0.99
_rng = np.random.default_rng()


class NpEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: object) -> int | float | bool | list:
        """Serialise numpy scalars, arrays, and tuples to JSON-compatible types."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, tuple):
            return list(obj)
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

    def __init__(  # noqa: PLR0913
        self, n_alph: int, seq_len: int, n_a: int,
        depth: int, burst_pos: int, seed: int,
    ) -> None:
        """Initialise bijections, vocabulary, and task splits."""
        assert 1 <= burst_pos <= depth, "burst_pos must be in [1, depth]"
        self.n_alph = n_alph
        self.seq_len = seq_len
        self.n_a = n_a
        self.depth = depth
        self.burst_pos = burst_pos
        rng = np.random.RandomState(seed)

        self.bijections = [np.arange(n_alph)]
        for _ in range(n_a * depth + 1):
            self.bijections.append(rng.permutation(n_alph))

        self.b_star = n_a * depth + 1

        self.pos_fns: dict[int, list[int]] = {
            p: list(range((p - 1) * n_a + 1, p * n_a + 1)) for p in range(1, depth + 1)
        }

        self._build_vocab()
        self._build_splits(rng)

    def _build_vocab(self) -> None:
        """Build token-to-index and index-to-token mappings."""
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
        for sp in (" ", "<PAD>", "S"):
            self.token[idx] = sp
            self.token_idx[sp] = idx
            idx += 1
        self.vocab_size = idx

    def _build_splits(self, rng: np.random.RandomState) -> None:
        """Build other-class and burst-class task lists."""
        D, bp = self.depth, self.burst_pos

        per_pos = [self.pos_fns[p] for p in range(1, D + 1)]
        other_combos = list(itertools.product(*per_pos))
        rng.shuffle(other_combos)
        self.other_train = [(CLASS_OTHER, *combo) for combo in other_combos]

        non_burst_positions = [p for p in range(1, D + 1) if p != bp]
        per_pos_no_bp = [self.pos_fns[p] for p in non_burst_positions]
        remaining_combos = list(itertools.product(*per_pos_no_bp))
        burst_tasks = []
        for combo in remaining_combos:
            fns = list(combo)
            # insert b* at the correct slot (positions are listed outermost-first,
            # so position bp sits at index D - bp from the left)
            fns.insert(D - bp, self.b_star)
            burst_tasks.append((CLASS_BURST, *fns))
        self.burst_train = burst_tasks

    def _make_doc(self, task: tuple[int, ...]) -> np.ndarray:
        """Generate a single tokenised document for a task composition."""
        fns = task[1:]
        inp = _rng.choice(self.n_alph, size=self.seq_len, replace=True)
        sp = np.array([self.token_idx[" "]])

        cur = inp.copy()
        outs = []
        for fn_idx in reversed(fns):
            cur = self.bijections[fn_idx][cur]
            outs.append(cur.copy())

        doc = [np.array([self.token_idx["S"]]), np.array([self.fn_tok[f] for f in fns]), sp, inp]
        for o in outs:
            doc.extend([sp, o])
        return np.concatenate(doc)

    def gen_pool(self, tasks: list[tuple[int, ...]], n: int) -> dict[tuple[int, ...], np.ndarray]:
        """Generate n documents per task, returning {task: docs_array}."""
        return {t: np.array([self._make_doc(t) for _ in range(n)]) for t in tasks}


def build_data(
    cfg: dict, depth: int, burst_pos: int, n_a: int, data_seed: int = DATA_SEED,
) -> tuple[dict, dict, dict, int, dict, dict]:
    """Build training/eval data pools and return pools, eval docs, prompt len, cfg, task info."""
    set_seed(data_seed)
    d = DepthNData(cfg["n_alphabets"], cfg["seq_len"], n_a, depth, burst_pos, data_seed)
    nd, ne = cfg["n_docs_per_task"], cfg["n_eval_per_task"]

    bg_pool = d.gen_pool(d.other_train, nd)
    target_pool = d.gen_pool(d.burst_train, nd)

    eval_pools = {
        CLASS_OTHER: d.gen_pool(d.other_train[: min(8, len(d.other_train))], ne),
        CLASS_BURST: d.gen_pool(d.burst_train, ne),
    }

    all_pools = [bg_pool, target_pool, *eval_pools.values()]
    padded = pad_pools_to_same_length(*all_pools)
    bg_pool, target_pool = padded[0], padded[1]
    for i, k in enumerate(eval_pools):
        eval_pools[k] = padded[i + 2]

    def _cat(pool: dict) -> np.ndarray:
        """Concatenate all arrays in a pool into one."""
        if not pool:
            return np.zeros((1, next(iter(bg_pool.values())).shape[1]), dtype=np.int64)
        return np.concatenate(list(pool.values()))

    eval_docs = {k: _cat(v) for k, v in eval_pools.items()}

    ref = eval_docs[CLASS_OTHER] if eval_docs[CLASS_OTHER].shape[0] > 1 else eval_docs[CLASS_BURST]
    sp_positions = np.where(ref[0] == d.token_idx[" "])[0]
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
        "doc_len": int(ref.shape[1]),
        "prompt_len": prompt_len,
    }
    return target_pool, bg_pool, eval_docs, prompt_len, cfg_out, task_info


def run_pretrain(  # noqa: C901, PLR0913, PLR0915
    cfg: dict,
    pretrain_steps: int,
    bg_pool: dict,
    ckpt_path: Path | None = None,
    eval_docs: dict | None = None,
    prompt_len: int = 0,
    eval_every: int = 25,
    seed: int = DATA_SEED,
    progress_prefix: str = "",
    *,
    save_checkpoint: bool = True,
) -> dict[str, list]:
    """Train one model on all-but-special for pretrain_steps, save checkpoint.

    Uses the provided seed (default DATA_SEED).

    Returns a pretrain log dict with step/loss/phase/acc_other/acc_burst lists
    so charts can show the pretraining trajectory.
    """
    from burst.config import EVAL_KEYS, PHASE_PRE_BURST  # noqa: PLC0415
    from burst.core.train.worker import eval_free_gen, eval_loss  # noqa: PLC0415

    set_seed(seed)
    net = make_net(cfg)
    optimizer = configure_optimizers(net, make_optim_cfg(cfg))
    scaler = make_scaler()

    P = pretrain_steps
    warmup = cfg["warmup_iters"]
    lr_max = cfg["lr"]
    lr_pe = cfg.get("lr_pretrain_end_frac", 0.3)
    lr_be = cfg.get("lr_burst_end_frac", 0.1)
    lr_re = cfg.get("lr_reversion_end_frac", 0.01)
    T_dummy, U_dummy = cfg.get("total_steps", 80), cfg["reversion_steps"]

    bg_ids = list(bg_pool.keys())
    bs = cfg["batch_size"]

    log: dict[str, list] = {"step": [], "loss": [], "loss_other": [], "loss_burst": [], "phase": []}
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
                idx = _rng.integers(len(bg_pool[tid]), size=k)
                parts.append(bg_pool[tid][idx])
        batch_np = np.concatenate(parts)[_rng.permutation(bs)]

        dat = torch.as_tensor(batch_np, dtype=torch.long, device=DEVICE)
        inp, tgt = dat[:, :-1], dat[:, 1:]
        update_phase_lr(
            s + 1,
            optimizer,
            warmup,
            P,
            T_dummy,
            U_dummy,
            lr_max,
            lr_pe,
            lr_be,
            lr_re,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            logits_BTV = net(inp)
            loss = _cross_entropy_logits_BTV_targets_BT(logits_BTV, tgt)
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
            log["loss_other"].append(eval_loss(net, eval_docs["other"]))
            log["loss_burst"].append(eval_loss(net, eval_docs["burst"]))
            net.train()

        if (s + 1) % 100 == 0 or s == pretrain_steps - 1:
            prefix = f"{progress_prefix} " if progress_prefix else ""
            logger.info(
                "%spretrain step %d/%d  loss=%.4f", prefix, s + 1, pretrain_steps, loss.item()
            )

    if save_checkpoint:
        if ckpt_path is None:
            msg = "ckpt_path is required when save_checkpoint=True"
            raise ValueError(msg)
        raw = getattr(net, "_orig_mod", net)
        torch.save(raw.state_dict(), ckpt_path)
        logger.info("  Pretrain checkpoint saved: %s", ckpt_path)
    return log


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Run the full burst training experiment."""
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
    parser.add_argument("--seed", type=int, default=DEFAULT_REPRO_SEED)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DETERMINISTIC,
    )
    parser.add_argument("--note", type=str, default="")
    args = parser.parse_args()
    set_reproducibility(args.seed, deterministic=args.deterministic)

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
    tag = args.run_tag or datetime.now(tz=datetime.UTC).strftime("%Y%m%d-%H%M%S")
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

    logger.info("Output: %s\nDevice: %s", run_dir, DEVICE)
    if DEVICE == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    n_a = args.n_a
    logger.info(
        "\nBuilding data (depth=%d, burst_pos=%d, n_a=%d)...", exp.depth, exp.burst_pos, n_a
    )
    tp, bp, ed, pl, cfg_out, ti = build_data(
        base_cfg,
        exp.depth,
        exp.burst_pos,
        n_a,
        data_seed=args.seed,
    )
    logger.info(
        "  Other classes: %s  Burst class: %s  doc_len: %s  prompt: %s",
        ti["n_other_train"], ti["n_burst_train"], ti["doc_len"], ti["prompt_len"],
    )

    data_path = logs_dir / "_data.pkl"
    with data_path.open("wb") as f:
        pickle.dump((tp, bp, ed, pl, None), f)

    # --- Pretrain: one shared checkpoint for all seeds ---
    pretrain_ckpt_path = logs_dir / "pretrain_ckpt.pt"
    P = base_cfg["pre_burst_steps"]
    pretrain_cfg = {
        **base_cfg,
        "vocab_size": cfg_out["vocab_size"],
        "context_size": cfg_out["context_size"],
    }
    pretrain_attempt = 0
    while True:
        pretrain_attempt += 1
        logger.info(
            "\nPretraining shared checkpoint (%d steps on all-but-special) — attempt %d...",
            P, pretrain_attempt,
        )
        pretrain_log = run_pretrain(
            pretrain_cfg,
            P,
            bp,
            pretrain_ckpt_path,
            eval_docs=ed,
            prompt_len=pl,
            eval_every=base_cfg["eval_every"],
            seed=args.seed,
        )
        peak_acc_other = max(pretrain_log.get("acc_other", [0.0]))
        if peak_acc_other >= PRETRAIN_ACC_THRESHOLD:
            logger.info("  Pretrain OK: peak acc_other=%.4f", peak_acc_other)
            break
        logger.info("  Pretrain acc_other=%.4f < 0.99, retrying...", peak_acc_other)
    with (logs_dir / "pretrain_log.pkl").open("wb") as f:
        pickle.dump(pretrain_log, f)

    pretrain_log_path = str(logs_dir / "pretrain_log.pkl")
    jobs = []
    for sched in exp.schedules:
        sched_total_steps = burst_steps_for_mode(sched, burst_mode, BURST_BASE_STEPS)
        sched_batch_size = batch_size_for_mode(sched, burst_mode, base_cfg["batch_size"])
        for seed_idx in range(exp.n_seeds):
            seed = args.seed + seed_idx
            cfg = {
                **base_cfg,
                "seed": seed,
                "vocab_size": cfg_out["vocab_size"],
                "context_size": cfg_out["context_size"],
                "total_steps": sched_total_steps,
                "batch_size": sched_batch_size,
            }
            label = f"{sched}_s{seed}"
            jobs.append(
                {
                    "schedule": sched,
                    "seed": seed,
                    "cfg": cfg,
                    "label": label,
                    "pretrain_ckpt": str(pretrain_ckpt_path),
                    "pretrain_log_path": pretrain_log_path,
                    "deterministic": args.deterministic,
                }
            )

    with (results_dir / "config.json").open("w") as f:
        json.dump(
            {
                "base_cfg": base_cfg,
                "n_a": n_a,
                "seed_base": args.seed,
                "n_seeds": exp.n_seeds,
                "schedules": exp.schedules,
                "n_jobs": len(jobs),
                "task_info": ti,
                "depth": exp.depth,
                "burst_pos": exp.burst_pos,
                "burst_mode": burst_mode,
                "run_probes": exp.run_probes,
                "run_next_token_probes": exp.run_next_token_probes,
                "run_adl": exp.run_adl,
                "burst_base_steps": BURST_BASE_STEPS,
                "deterministic": args.deterministic,
                "pretrain_ckpt": str(pretrain_ckpt_path),
                "jobs": [
                    {
                        "label": j["label"],
                        "schedule": j["schedule"],
                        "seed": j["seed"],
                        "total_steps": j["cfg"]["total_steps"],
                        "batch_size": j["cfg"]["batch_size"],
                    }
                    for j in jobs
                ],
            },
            f,
            indent=2,
            cls=NpEncoder,
        )

    manifest_path = write_repro_manifest(
        run_dir,
        mode="train",
        seed=args.seed,
        deterministic=args.deterministic,
        cli_args={
            "run_tag": args.run_tag,
            "depth": args.depth,
            "burst_pos": args.burst_pos,
            "burst_mode": args.burst_mode,
            "n_a": args.n_a,
            "schedules": args.schedules,
            "n_seeds": args.n_seeds,
            "n_workers": args.n_workers,
            "run_probes": args.run_probes,
            "run_next_token_probes": args.run_next_token_probes,
            "run_adl": args.run_adl,
        },
        note=args.note,
    )

    n_procs = min(len(jobs), exp.n_workers)
    jobs_per_proc = max(1, (len(jobs) + n_procs - 1) // n_procs)
    logger.info("  %s", gpu_cfg.summary())
    logger.info("  Layout: %d processes x ~%d jobs/proc", n_procs, jobs_per_proc)

    tc = exp.train
    logger.info("\nModel: %dL/%dd/%dH", tc.n_layer, tc.n_embd, tc.n_head)
    logger.info("Burst mode: %s", burst_mode)
    logger.info(
        "Jobs: %d, parallel processes: %d, jobs/process: ~%d", len(jobs), n_procs, jobs_per_proc
    )
    logger.info("Schedules: %s", exp.schedules)
    for sched in exp.schedules:
        T_s = burst_steps_for_mode(sched, burst_mode, BURST_BASE_STEPS)
        bs_s = batch_size_for_mode(sched, burst_mode, base_cfg["batch_size"])
        logger.info(
            "  %s: burst_steps=%d  batch_size=%d  reversion=%d",
            sched, T_s, bs_s, tc.reversion_steps,
        )
    logger.info("")

    batched_script = str(Path(__file__).parent / "worker_batched.py")
    t0 = time.time()

    job_chunks = [jobs[i : i + jobs_per_proc] for i in range(0, len(jobs), jobs_per_proc)]

    def launch_chunk(chunk_idx: int, chunk: list[dict]) -> subprocess.Popen:
        """Pickle a job chunk and spawn a batched worker subprocess."""
        chunk_path = logs_dir / f"_chunk_{chunk_idx}.pkl"
        with chunk_path.open("wb") as f:
            pickle.dump(chunk, f)
        return subprocess.Popen(  # noqa: S603
            [
                sys.executable,
                batched_script,
                "--jobs-path",
                chunk_path,
                "--data-path",
                str(data_path),
                "--run-dir",
                str(logs_dir),
                "--progress-dir",
                str(progress_dir),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

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
                with rp.open("rb") as f:
                    r = pickle.load(f)  # noqa: S301
                thresholds = TrainConfig().reversion_thresholds
                first_key = reversion_life_key(thresholds[0])
                first_lbl = reversion_life_label(thresholds[0])
                lv = r.get(first_key, "?")
                tqdm.write(
                    f"  [{n_done}/{len(jobs)}] {label:30s} "
                    f"peak={r['peak_burst']:.3f} "
                    f"{first_lbl}={lv} auc={r['reversion_auc']:.0f} ({time.time() - t0:.0f}s)"
                )

        done_chunks = [ci for ci, (_, proc) in active_chunks.items() if proc.poll() is not None]
        for ci in done_chunks:
            chunk, proc = active_chunks.pop(ci)
            if proc.returncode != 0:
                se = proc.stderr.read().decode() if proc.stderr else ""
                for j in chunk:
                    if j["label"] not in reported_jobs:
                        reported_jobs.add(j["label"])
                        n_done += 1
                        tqdm.write(
                            f"  FAIL [{n_done}/{len(jobs)}]: {j['label']} "
                            f"(chunk {ci} exit {proc.returncode})"
                        )
                        if se:
                            tqdm.write(f"    {se[:500]}")

            if next_chunk < len(job_chunks):
                c = job_chunks[next_chunk]
                active_chunks[next_chunk] = (c, launch_chunk(next_chunk, c))
                next_chunk += 1

    pbar.update(pbar.total - pbar.n)
    pbar.close()
    total_time = time.time() - t0
    logger.info("\nAll %d jobs done in %.0fs (%.1f min)", n_done, total_time, total_time / 60)

    all_results = []
    for job in jobs:
        rp = logs_dir / f"{job['label']}.pkl"
        if rp.exists():
            with rp.open("rb") as f:
                all_results.append(pickle.load(f))  # noqa: S301
    with (logs_dir / "all_results.pkl").open("wb") as f:
        pickle.dump(all_results, f)

    for f in progress_dir.glob("*"):
        f.unlink()
    if progress_dir.exists():
        progress_dir.rmdir()
    for f in logs_dir.glob("_chunk_*.pkl"):
        f.unlink()

    logger.info("Results: %s (%d ok)", run_dir, len(all_results))
    logger.info("repro_manifest: %s", manifest_path)
    logger.info("\nNext: uv run python -m burst.core gradients %s", run_dir)
    logger.info("Next: uv run python -m burst.core pipeline %s", run_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
