#!/usr/bin/env python3
"""Correlation × Concentration sweep runner.

Usage examples:
    # Full sweep with defaults
    python -m simple.sweep

    # Custom grid
    python -m simple.sweep --n-a 6 --n-burst 6 --fracs 1.0 0.9 0.7 0.5 --seeds 42 123 777 --workers 20

    # Quick test
    python -m simple.sweep --fracs 1.0 0.5 --seeds 42 --ft-steps 500 --fg-steps 200

    # Resume (skips existing results)
    python -m simple.sweep --resume

Results are saved incrementally:
    {out}/config.json                              — run config
    {out}/seed_{s}/corr{c}_frac{f}_ft.pkl         — finetune result
    {out}/seed_{s}/corr{c}_frac{f}_fg.pkl         — forget result
"""
import argparse
import json
import os
import sys
import pickle
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch.multiprocessing as mp

from simple.data import make_data
from simple.pretrain import pretrain
from simple.finetune import _finetune_worker
from simple.forget import _forget_worker


def parse_args():
    p = argparse.ArgumentParser(description="Correlation × Concentration sweep")

    # Task
    p.add_argument("--n-a", type=int, default=5, help="Functions per slot")
    p.add_argument("--n-burst", type=int, default=5, help="Burst functions")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--burst-pos", type=int, default=3)
    p.add_argument("--n-docs", type=int, default=None,
                   help="Training docs per task (default: ~6400 / n_tasks)")
    p.add_argument("--n-eval", type=int, default=None,
                   help="Eval docs per task (default: ~3200 / n_tasks)")

    # Sweep grid
    p.add_argument("--correlations", type=float, nargs="+", default=None,
                   help="Correlation levels (default: 0/n_burst .. n_burst/n_burst)")
    p.add_argument("--fracs", type=float, nargs="+",
                   default=[1.0, 0.95, 0.9, 0.7, 0.5, 0.3])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 777, 843, 910])

    # Training
    p.add_argument("--ft-steps", type=int, default=2000)
    p.add_argument("--fg-steps", type=int, default=1200)
    p.add_argument("--ft-lr", type=float, default=1e-4)
    p.add_argument("--fg-lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-every", type=int, default=100)

    # Infra
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--out", type=str, default=None,
                   help="Output dir (default: data/sweep_YYYYMMDD_HHMMSS)")
    p.add_argument("--resume", action="store_true",
                   help="Skip existing result files")

    return p.parse_args()


def result_path(out_dir, seed, corr, frac, phase):
    """Path for a single result file."""
    seed_dir = Path(out_dir) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    c = int(round(corr * 100))
    f = int(round(frac * 100))
    return seed_dir / f"corr{c}_frac{f}_{phase}.pkl"


def save_result(path, result):
    with open(path, "wb") as f:
        pickle.dump(result, f)


def load_result(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    args = parse_args()

    # Correlations default: 0/n_burst .. n_burst/n_burst
    if args.correlations is None:
        args.correlations = [i / args.n_burst for i in range(args.n_burst + 1)]

    # Auto-scale n_docs/n_eval to keep total pool size ~6400/~3200
    n_bg_tasks = args.n_a ** args.depth
    if args.n_docs is None:
        args.n_docs = max(100, 6400 // n_bg_tasks)
    if args.n_eval is None:
        args.n_eval = max(100, 3200 // n_bg_tasks)
    print(f"n_a={args.n_a} → {n_bg_tasks} bg tasks, n_docs={args.n_docs}, n_eval={args.n_eval}")

    # Output dir
    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = f"data/sweep_{ts}"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args).copy()
    config["timestamp"] = datetime.now().isoformat()
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    n_corr = len(args.correlations)
    n_frac = len(args.fracs)
    n_seeds = len(args.seeds)
    total = n_corr * n_frac * n_seeds
    print(f"Sweep: {n_corr} corrs × {n_frac} fracs × {n_seeds} seeds = {total} runs")
    print(f"Output: {out_dir}")
    print(f"Workers: {args.workers}")
    print()

    ctx = mp.get_context("spawn")
    t0 = time.time()

    for i_corr, corr in enumerate(args.correlations):
        n_copied = int(round(corr * args.n_burst))
        print(f"\n{'='*60}")
        print(f"[{i_corr+1}/{n_corr}] Correlation = {corr:.3f} ({n_copied}/{args.n_burst} copied)")

        # Pretrain per (corr, seed)
        seed_data = {}
        for i_s, seed in enumerate(args.seeds):
            print(f"  Pretrain seed {seed} ({i_s+1}/{n_seeds})...", end=" ", flush=True)
            pt_dir = out_dir / f"seed_{seed}" / f"pretrain_corr{int(round(corr*100))}"
            data = make_data(depth=args.depth, burst_pos=args.burst_pos,
                             n_a=args.n_a, n_burst=args.n_burst,
                             burst_correlation=corr, seed=seed,
                             n_docs=args.n_docs, n_eval=args.n_eval)
            pt = pretrain(data, str(pt_dir), seed=seed,
                          batch_size=args.batch_size, quiet=True)
            seed_data[seed] = (data, pt)
            print(f"done (acc={max(pt['log']['acc_other']):.3f})")
        print(f"  All {n_seeds} pretrains done")

        # Check which FT jobs need running
        ft_jobs = []
        ft_job_keys = []
        ft_results_cache = {}  # (seed, i_f) -> result

        for seed in args.seeds:
            data, pt = seed_data[seed]
            for i_f, frac in enumerate(args.fracs):
                ft_path = result_path(out_dir, seed, corr, frac, "ft")
                if args.resume and ft_path.exists():
                    ft_results_cache[(seed, i_f)] = load_result(ft_path)
                    continue
                tag = f"corr{int(round(corr*100))}_b{int(frac*100)}_s{seed}"
                ft_jobs.append(dict(
                    data=data, pretrain_ckpt=pt["ckpt_path"],
                    out_dir=str(out_dir / "ckpts"),
                    burst_frac=frac, steps=args.ft_steps, lr=args.ft_lr,
                    batch_size=args.batch_size, eval_every=args.eval_every,
                    tag=tag, seed=seed, quiet=True, lite=True))
                ft_job_keys.append((seed, i_f))

        if ft_jobs:
            n_w = min(args.workers, len(ft_jobs))
            print(f"  FT: {len(ft_jobs)} jobs ({len(ft_results_cache)} cached), {n_w} workers...", flush=True)
            t1 = time.time()
            with ctx.Pool(n_w) as pool:
                ft_results_new = pool.map(_finetune_worker, ft_jobs)
            print(f"  FT done in {time.time()-t1:.0f}s")
            for (seed, i_f), ft_r in zip(ft_job_keys, ft_results_new):
                frac = args.fracs[i_f]
                save_result(result_path(out_dir, seed, corr, frac, "ft"), ft_r)
                ft_results_cache[(seed, i_f)] = ft_r
        else:
            print(f"  FT: all {n_frac * n_seeds} cached")

        # Check which FG jobs need running
        fg_jobs = []
        fg_job_keys = []
        fg_results_cache = {}

        for seed in args.seeds:
            data, pt = seed_data[seed]
            for i_f, frac in enumerate(args.fracs):
                fg_path = result_path(out_dir, seed, corr, frac, "fg")
                if args.resume and fg_path.exists():
                    fg_results_cache[(seed, i_f)] = load_result(fg_path)
                    continue
                ft_r = ft_results_cache[(seed, i_f)]
                fg_jobs.append(dict(
                    data=data, finetune_ckpt=ft_r["ckpt_path"],
                    pretrain_ckpt=pt["ckpt_path"],
                    out_dir=str(out_dir / "ckpts"),
                    steps=args.fg_steps, lr=args.fg_lr,
                    batch_size=args.batch_size, eval_every=args.eval_every,
                    tag=ft_r["tag"], seed=seed, quiet=True, lite=True))
                fg_job_keys.append((seed, i_f))

        if fg_jobs:
            n_w = min(args.workers, len(fg_jobs))
            print(f"  FG: {len(fg_jobs)} jobs ({len(fg_results_cache)} cached), {n_w} workers...", flush=True)
            t1 = time.time()
            with ctx.Pool(n_w) as pool:
                fg_results_new = pool.map(_forget_worker, fg_jobs)
            print(f"  FG done in {time.time()-t1:.0f}s")
            for (seed, i_f), fg_r in zip(fg_job_keys, fg_results_new):
                frac = args.fracs[i_f]
                save_result(result_path(out_dir, seed, corr, frac, "fg"), fg_r)
                fg_results_cache[(seed, i_f)] = fg_r
        else:
            print(f"  FG: all {n_frac * n_seeds} cached")

        # Summary
        for seed in args.seeds:
            for i_f, frac in enumerate(args.fracs):
                ft_r = ft_results_cache[(seed, i_f)]
                fg_r = fg_results_cache[(seed, i_f)]
                print(f"    {ft_r['tag']}: peak={ft_r['peak_burst']:.3f} "
                      f"drop={fg_r['dropoff_pct']:.1f}%")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed/60:.1f} min. Results in {out_dir}")


if __name__ == "__main__":
    main()
