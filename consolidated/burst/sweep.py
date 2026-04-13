"""Unified sweep runner for burst forgetting experiments.

Supports sweeping over any combination of:
    - burst concentration (fracs)
    - burst correlation (correlations)
    - finetune learning rate (ft_lrs)
    - seeds

Results saved incrementally so partial runs are still plottable:
    {out}/config.json
    {out}/seed_{s}/corr{c}_ftlr{lr}_frac{f}_{ft|fg}.pkl
"""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch.multiprocessing as mp

from burst.data import make_data
from burst.pretrain import pretrain
from burst.finetune import _finetune_worker
from burst.forget import _forget_worker


@dataclass
class SweepConfig:
    # Task
    n_a: int = 6
    n_burst: int = 6
    depth: int = 3
    burst_pos: int = 3
    n_docs: int | None = None   # auto-scale if None
    n_eval: int | None = None

    # Sweep axes
    fracs: list[float] = field(default_factory=lambda: [1.0, 0.95, 0.9, 0.7, 0.5, 0.3])
    correlations: list[float] = field(default_factory=lambda: [0.0])
    ft_lrs: list[float] = field(default_factory=lambda: [1e-4])
    seeds: list[int] = field(default_factory=lambda: [42, 123, 777])

    # Training
    ft_steps: int = 1500
    fg_steps: int = 1200
    fg_lr: float = 1e-4
    batch_size: int = 512
    eval_every: int = 100

    # Infra
    workers: int = 6

    def __post_init__(self):
        # Auto-scale n_docs/n_eval to keep pool size constant across n_a
        n_bg_tasks = self.n_a ** self.depth
        if self.n_docs is None:
            self.n_docs = max(30, 6400 // n_bg_tasks)
        if self.n_eval is None:
            self.n_eval = max(30, 3200 // n_bg_tasks)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SweepConfig":
        # Filter unknown keys to be lenient with configs
        valid = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def result_path(out_dir: Path, seed: int, corr: float, ft_lr: float,
                frac: float, phase: str) -> Path:
    """Path for a single result file."""
    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    c = int(round(corr * 100))
    f = int(round(frac * 100))
    return seed_dir / f"corr{c}_ftlr{ft_lr:.0e}_frac{f}_{phase}.pkl"


def save_result(path: Path, result: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump(result, f)


def load_result(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def run_sweep(cfg: SweepConfig, out_dir: str | Path, resume: bool = False) -> Path:
    """Run the full sweep. Returns the output directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_dict = cfg.to_dict()
    config_dict["timestamp"] = datetime.now().isoformat()
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    n_corr = len(cfg.correlations)
    n_lr = len(cfg.ft_lrs)
    n_frac = len(cfg.fracs)
    n_seeds = len(cfg.seeds)
    total = n_corr * n_lr * n_frac * n_seeds
    print(f"Sweep: {n_corr} corrs × {n_lr} ft_lrs × {n_frac} fracs × {n_seeds} seeds = {total} runs")
    print(f"n_a={cfg.n_a}, n_docs={cfg.n_docs}, n_eval={cfg.n_eval}")
    print(f"Output: {out_dir}")
    print(f"Workers: {cfg.workers}")
    print()

    ctx = mp.get_context("spawn")
    t0 = time.time()

    # Outer loop: correlation × seed (defines data + pretrain)
    # Inner loop: ft_lr × frac (batched in parallel per (corr, seed) group)
    for i_corr, corr in enumerate(cfg.correlations):
        n_copied = int(round(corr * cfg.n_burst))
        print(f"\n{'='*60}")
        print(f"[corr {i_corr+1}/{n_corr}] Correlation = {corr:.3f} "
              f"({n_copied}/{cfg.n_burst} copied)")

        # Pretrain per seed (data depends on corr + seed)
        seed_data = {}
        for i_s, seed in enumerate(cfg.seeds):
            print(f"  Pretrain seed {seed} ({i_s+1}/{n_seeds})...",
                  end=" ", flush=True)
            pt_dir = out_dir / f"seed_{seed}" / f"pretrain_corr{int(round(corr*100))}"
            data = make_data(
                depth=cfg.depth, burst_pos=cfg.burst_pos,
                n_a=cfg.n_a, n_burst=cfg.n_burst,
                burst_correlation=corr, seed=seed,
                n_docs=cfg.n_docs, n_eval=cfg.n_eval,
            )
            pt = pretrain(data, str(pt_dir), seed=seed,
                          batch_size=cfg.batch_size, quiet=True)
            seed_data[seed] = (data, pt)
            print(f"done (acc={max(pt['log']['acc_other']):.3f})")
        print(f"  All {n_seeds} pretrains done")

        # Build all FT jobs: (seed × ft_lr × frac)
        ft_jobs = []
        ft_keys = []  # (seed, ft_lr, i_f)
        ft_cache: dict[tuple, dict] = {}

        for seed in cfg.seeds:
            data, pt = seed_data[seed]
            for ft_lr in cfg.ft_lrs:
                for i_f, frac in enumerate(cfg.fracs):
                    ft_path = result_path(out_dir, seed, corr, ft_lr, frac, "ft")
                    if resume and ft_path.exists():
                        ft_cache[(seed, ft_lr, i_f)] = load_result(ft_path)
                        continue
                    tag = (f"corr{int(round(corr*100))}_ftlr{ft_lr:.0e}"
                           f"_b{int(frac*100)}_s{seed}")
                    ft_jobs.append(dict(
                        data=data, pretrain_ckpt=pt["ckpt_path"],
                        out_dir=str(out_dir / "ckpts"),
                        burst_frac=frac, steps=cfg.ft_steps, lr=ft_lr,
                        batch_size=cfg.batch_size, eval_every=cfg.eval_every,
                        tag=tag, seed=seed, quiet=True, lite=True,
                    ))
                    ft_keys.append((seed, ft_lr, i_f))

        if ft_jobs:
            n_w = min(cfg.workers, len(ft_jobs))
            print(f"  FT: {len(ft_jobs)} jobs ({len(ft_cache)} cached), "
                  f"{n_w} workers...", flush=True)
            t1 = time.time()
            with ctx.Pool(n_w) as pool:
                ft_results = pool.map(_finetune_worker, ft_jobs)
            print(f"  FT done in {time.time() - t1:.0f}s")
            for (seed, ft_lr, i_f), r in zip(ft_keys, ft_results):
                frac = cfg.fracs[i_f]
                save_result(result_path(out_dir, seed, corr, ft_lr, frac, "ft"), r)
                ft_cache[(seed, ft_lr, i_f)] = r
        else:
            print(f"  FT: all {n_lr * n_frac * n_seeds} cached")

        # Build all FG jobs
        fg_jobs = []
        fg_keys = []
        fg_cache: dict[tuple, dict] = {}

        for seed in cfg.seeds:
            data, pt = seed_data[seed]
            for ft_lr in cfg.ft_lrs:
                for i_f, frac in enumerate(cfg.fracs):
                    fg_path = result_path(out_dir, seed, corr, ft_lr, frac, "fg")
                    if resume and fg_path.exists():
                        fg_cache[(seed, ft_lr, i_f)] = load_result(fg_path)
                        continue
                    ft_r = ft_cache[(seed, ft_lr, i_f)]
                    fg_jobs.append(dict(
                        data=data, finetune_ckpt=ft_r["ckpt_path"],
                        pretrain_ckpt=pt["ckpt_path"],
                        out_dir=str(out_dir / "ckpts"),
                        steps=cfg.fg_steps, lr=cfg.fg_lr,
                        batch_size=cfg.batch_size, eval_every=cfg.eval_every,
                        tag=ft_r["tag"], seed=seed, quiet=True, lite=True,
                    ))
                    fg_keys.append((seed, ft_lr, i_f))

        if fg_jobs:
            n_w = min(cfg.workers, len(fg_jobs))
            print(f"  FG: {len(fg_jobs)} jobs ({len(fg_cache)} cached), "
                  f"{n_w} workers...", flush=True)
            t1 = time.time()
            with ctx.Pool(n_w) as pool:
                fg_results = pool.map(_forget_worker, fg_jobs)
            print(f"  FG done in {time.time() - t1:.0f}s")
            for (seed, ft_lr, i_f), r in zip(fg_keys, fg_results):
                frac = cfg.fracs[i_f]
                save_result(result_path(out_dir, seed, corr, ft_lr, frac, "fg"), r)
                fg_cache[(seed, ft_lr, i_f)] = r
        else:
            print(f"  FG: all {n_lr * n_frac * n_seeds} cached")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed/60:.1f} min. Results in {out_dir}")
    return out_dir


def load_sweep_results(out_dir: str | Path) -> tuple[SweepConfig, dict, dict]:
    """Load sweep results from disk.

    Returns (config, all_ft, all_fg) where:
        all_ft[corr][ft_lr][seed] = list of FT results per frac (None if missing)
        all_fg[corr][ft_lr][seed] = list of FG results per frac (None if missing)
    """
    out_dir = Path(out_dir)
    with open(out_dir / "config.json") as f:
        raw = json.load(f)
    cfg = SweepConfig.from_dict(raw)

    all_ft: dict = {c: {lr: {} for lr in cfg.ft_lrs} for c in cfg.correlations}
    all_fg: dict = {c: {lr: {} for lr in cfg.ft_lrs} for c in cfg.correlations}
    loaded, missing = 0, 0

    for corr in cfg.correlations:
        for ft_lr in cfg.ft_lrs:
            for seed in cfg.seeds:
                ft_list, fg_list = [], []
                for frac in cfg.fracs:
                    ft_p = result_path(out_dir, seed, corr, ft_lr, frac, "ft")
                    fg_p = result_path(out_dir, seed, corr, ft_lr, frac, "fg")
                    if ft_p.exists() and fg_p.exists():
                        ft_list.append(load_result(ft_p))
                        fg_list.append(load_result(fg_p))
                        loaded += 1
                    else:
                        ft_list.append(None)
                        fg_list.append(None)
                        missing += 1
                all_ft[corr][ft_lr][seed] = ft_list
                all_fg[corr][ft_lr][seed] = fg_list

    print(f"Loaded {loaded}/{loaded + missing} runs ({missing} missing)")
    return cfg, all_ft, all_fg
