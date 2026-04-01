"""Sweep pre-training hyperparameters and report convergence statistics."""

from __future__ import annotations

import argparse
import itertools
import math
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from burst.config import SEED_BASE, TrainConfig
from burst.core.gpu import gpu_cfg
from burst.core.train.experiment import build_data, run_pretrain

DEPTH_DEFAULT = 3
BURST_POS_DEFAULT = 3
N_SEEDS_DEFAULT = 2
SMALL_FLOAT_THRESHOLD = 1e-3
ACC_OTHER_THRESHOLD = 0.99

LR_VALUES = [3e-3, 1e-3]
LR_PRETRAIN_END_FRAC_VALUES = [0.5, 0.1]  # , 0.3
BETA_VALUES = [0.9]
PRE_BURST_STEPS_VALUES = [400, 600]
N_A_VALUES = [3]
N_DOCS_PER_TASK_VALUES = [100]

CONFIG_KEYS = [
    "lr",
    "lr_pretrain_end_frac",
    "beta1",
    "beta2",
    "pre_burst_steps",
    "n_a",
    "n_docs_per_task",
    "depth",
    "burst_pos",
]


@dataclass(frozen=True)
class SweepRun:
    """Single run configuration for the pretrain sweep."""

    config_id: str
    run_idx: int
    total_runs: int
    seed: int
    lr: float
    lr_pretrain_end_frac: float
    beta: float
    pre_burst_steps: int
    n_a: int
    n_docs_per_task: int
    depth: int
    burst_pos: int

    @property
    def group_key(self) -> tuple[int, int]:
        """Return grouping key for data sharing."""
        return (self.n_a, self.n_docs_per_task)

    def as_param_dict(self) -> dict[str, Any]:
        """Return parameters as a dict."""
        return {
            "lr": self.lr,
            "lr_pretrain_end_frac": self.lr_pretrain_end_frac,
            "beta1": self.beta,
            "beta2": self.beta,
            "pre_burst_steps": self.pre_burst_steps,
            "n_a": self.n_a,
            "n_docs_per_task": self.n_docs_per_task,
            "depth": self.depth,
            "burst_pos": self.burst_pos,
            "seed": self.seed,
            "config_id": self.config_id,
        }


def _fmt_float(v: float) -> str:
    return f"{v:.0e}" if v < SMALL_FLOAT_THRESHOLD else f"{v:.4f}".rstrip("0").rstrip(".")


def _file_safe(v: float) -> str:
    if isinstance(v, int):
        return str(v)
    return f"{v:.0e}" if v < SMALL_FLOAT_THRESHOLD else str(v).replace(".", "p")


def _build_sweep_runs(n_seeds: int, depth: int, burst_pos: int) -> list[SweepRun]:
    staged: list[dict[str, Any]] = []
    combos = itertools.product(
        LR_VALUES,
        LR_PRETRAIN_END_FRAC_VALUES,
        BETA_VALUES,
        PRE_BURST_STEPS_VALUES,
        N_A_VALUES,
        N_DOCS_PER_TASK_VALUES,
    )
    for config_idx, (lr, lr_pe, beta, pre_steps, n_a, n_docs) in enumerate(combos, start=1):
        cfg_id = f"cfg_{config_idx:04d}"
        for seed_idx in range(n_seeds):
            seed = SEED_BASE + seed_idx
            staged.append(
                {
                    "config_id": cfg_id,
                    "seed": seed,
                    "lr": lr,
                    "lr_pretrain_end_frac": lr_pe,
                    "beta": beta,
                    "pre_burst_steps": pre_steps,
                    "n_a": n_a,
                    "n_docs_per_task": n_docs,
                    "depth": depth,
                    "burst_pos": burst_pos,
                }
            )
    total_runs = len(staged)
    runs: list[SweepRun] = []
    for idx, payload in enumerate(staged, start=1):
        runs.append(
            SweepRun(
                run_idx=idx,
                total_runs=total_runs,
                **payload,
            )
        )
    return runs


def _chunk_runs(runs: list[SweepRun], chunk_size: int) -> list[list[SweepRun]]:
    if chunk_size <= 0:
        return [runs]
    return [runs[i : i + chunk_size] for i in range(0, len(runs), chunk_size)]


def _build_group_tasks(
    grouped: dict[tuple[int, int], list[SweepRun]],
    max_workers: int,
    chunk_size: int,
    base_cfg: dict[str, Any],
) -> list[tuple[tuple[int, int], list[SweepRun], dict[str, Any]]]:
    tasks: list[tuple[tuple[int, int], list[SweepRun], dict[str, Any]]] = []
    n_groups = len(grouped)
    if n_groups == 0:
        return tasks

    auto_chunks_per_group = max(1, math.ceil(max_workers / n_groups))
    for group_key, group_runs in grouped.items():
        if chunk_size > 0:
            effective_chunk_size = chunk_size
        else:
            effective_chunk_size = max(1, math.ceil(len(group_runs) / auto_chunks_per_group))
        tasks.extend(
            (group_key, run_chunk, base_cfg)
            for run_chunk in _chunk_runs(group_runs, effective_chunk_size)
        )
    return tasks


def _run_group(
    args: tuple[tuple[int, int], list[SweepRun], dict[str, Any]],
) -> list[dict[str, Any]]:
    group_key, group_runs, base_cfg = args
    n_a, n_docs_per_task = group_key

    cfg_group = {
        **base_cfg,
        "n_docs_per_task": n_docs_per_task,
    }
    target_pool, bg_pool, eval_docs, prompt_len, cfg_out, task_info = build_data(
        cfg_group,
        depth=group_runs[0].depth,
        burst_pos=group_runs[0].burst_pos,
        n_a=n_a,
    )
    _ = target_pool

    rows: list[dict[str, Any]] = []
    for run in group_runs:
        cfg = {
            **cfg_group,
            "lr": run.lr,
            "lr_pretrain_end_frac": run.lr_pretrain_end_frac,
            "beta1": run.beta,
            "beta2": run.beta,
            "pre_burst_steps": run.pre_burst_steps,
            "vocab_size": cfg_out["vocab_size"],
            "context_size": cfg_out["context_size"],
        }
        log = run_pretrain(
            cfg=cfg,
            pretrain_steps=run.pre_burst_steps,
            bg_pool=bg_pool,
            ckpt_path=None,
            eval_docs=eval_docs,
            prompt_len=prompt_len,
            eval_every=cfg["eval_every"],
            seed=run.seed,
            save_checkpoint=False,
            progress_prefix=f"[run {run.run_idx}/{run.total_runs}] {run.config_id} s{run.seed}",
        )
        if not log["step"]:
            continue

        final_i = len(log["step"]) - 1
        acc_other_series = [float(v) for v in log["acc_other"]]
        loss_series = [float(v) for v in log["loss"]]
        first_step_acc_other_gt_99 = next(
            (
                int(step)
                for step, acc_other in zip(log["step"], acc_other_series, strict=False)
                if acc_other > ACC_OTHER_THRESHOLD
            ),
            None,
        )
        row = {
            **run.as_param_dict(),
            "doc_len": int(task_info["doc_len"]),
            "prompt_len": int(task_info["prompt_len"]),
            "n_other_train": int(task_info["n_other_train"]),
            "n_burst_train": int(task_info["n_burst_train"]),
            "final_eval_step": int(log["step"][final_i]),
            "final_acc_other": float(acc_other_series[final_i]),
            "final_acc_burst": float(log["acc_burst"][final_i]),
            "final_loss": float(loss_series[final_i]),
            "peak_acc_other": float(max(acc_other_series)),
            "first_step_acc_other_gt_99": first_step_acc_other_gt_99,
            "min_loss": float(min(loss_series)),
            "curve_step": [int(s) for s in log["step"]],
            "curve_acc_other": acc_other_series,
            "curve_acc_burst": [float(v) for v in log["acc_burst"]],
            "curve_loss": loss_series,
        }
        rows.append(row)
    return rows


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text and "cuda" in text


def _plot_single_run(row: dict[str, Any], out_dir: Path) -> None:
    fig, ax_acc = plt.subplots(figsize=(10, 5.5))
    ax_loss = ax_acc.twinx()

    steps = row["curve_step"]
    acc_other = row["curve_acc_other"]
    acc_burst = row["curve_acc_burst"]
    loss = row["curve_loss"]

    l1 = ax_acc.plot(steps, acc_other, color="#1565C0", lw=2.0, label="acc_other")
    l2 = ax_acc.plot(steps, acc_burst, color="#AD1457", lw=1.5, ls="--", label="acc_burst")
    l3 = ax_loss.plot(steps, loss, color="#2E7D32", lw=1.8, alpha=0.9, label="loss")

    ax_acc.set_xlabel("pretrain step")
    ax_acc.set_ylabel("accuracy")
    ax_loss.set_ylabel("loss")
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.grid(visible=True, alpha=0.25)

    title = (
        f"{row['config_id']} seed={row['seed']} | "
        f"lr={_fmt_float(row['lr'])}, lr_pre_end={row['lr_pretrain_end_frac']}, "
        f"beta={row['beta1']}, pre_steps={row['pre_burst_steps']}, "
        f"N_A={row['n_a']}, n_docs={row['n_docs_per_task']}"
    )
    ax_acc.set_title(title, fontsize=10)
    lines = l1 + l2 + l3
    labels = [ln.get_label() for ln in lines]
    ax_acc.legend(lines, labels, loc="best", fontsize=9)

    name = (
        f"{row['config_id']}_s{row['seed']}_"
        f"lr{_file_safe(row['lr'])}_lpe{_file_safe(row['lr_pretrain_end_frac'])}_"
        f"b{_file_safe(row['beta1'])}_p{row['pre_burst_steps']}_"
        f"na{row['n_a']}_nd{row['n_docs_per_task']}.png"
    )
    fig.tight_layout()
    fig.savefig(out_dir / name, dpi=150)
    plt.close(fig)


def _plot_summary(raw_df: pd.DataFrame, agg_df: pd.DataFrame, out_dir: Path) -> None:
    def _bar_by(df: pd.DataFrame, key: str, fname: str, xlab: str) -> None:
        stats = (
            df.groupby(key)["final_acc_other"]
            .agg(mean="mean", std="std")
            .reset_index()
            .sort_values(key)
        )
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(stats[key].astype(str), stats["mean"], yerr=stats["std"].fillna(0.0), capsize=4)
        ax.set_xlabel(xlab)
        ax.set_ylabel("final acc_other (mean +/- std)")
        ax.set_ylim(0.0, 1.0)
        ax.grid(visible=True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)

    _bar_by(raw_df, "lr", "summary_by_lr.png", "lr")
    _bar_by(
        raw_df,
        "lr_pretrain_end_frac",
        "summary_by_lr_pretrain_end_frac.png",
        "lr_pretrain_end_frac",
    )
    _bar_by(raw_df, "beta1", "summary_by_beta.png", "beta1=beta2")
    _bar_by(raw_df, "pre_burst_steps", "summary_by_pre_burst_steps.png", "pre_burst_steps")
    _bar_by(raw_df, "n_a", "summary_by_n_a.png", "N_A")
    _bar_by(raw_df, "n_docs_per_task", "summary_by_n_docs_per_task.png", "n_docs_per_task")

    top = agg_df.sort_values(
        ["final_acc_other_mean", "final_loss_mean"], ascending=[False, True]
    ).head(20)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(top["config_id"], top["final_acc_other_mean"])
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top["config_id"], rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("mean final acc_other across seeds")
    ax.set_ylim(0.0, 1.0)
    ax.grid(visible=True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "top20_configs_by_final_acc_other.png", dpi=150)
    plt.close(fig)


def _build_tables(
    raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    group_cols = ["config_id", *CONFIG_KEYS]
    agg_df = raw_df.groupby(group_cols, as_index=False).agg(
        final_acc_other_mean=("final_acc_other", "mean"),
        final_acc_other_std=("final_acc_other", "std"),
        final_loss_mean=("final_loss", "mean"),
        final_loss_std=("final_loss", "std"),
        peak_acc_other_mean=("peak_acc_other", "mean"),
        peak_acc_other_std=("peak_acc_other", "std"),
        first_step_acc_other_gt_99_mean=("first_step_acc_other_gt_99", "mean"),
        first_step_acc_other_gt_99_std=("first_step_acc_other_gt_99", "std"),
        n_runs=("seed", "count"),
    )
    ranking_df = agg_df.sort_values(
        ["final_acc_other_mean", "final_loss_mean"],
        ascending=[False, True],
    ).reset_index(drop=True)
    ranking_df.insert(0, "rank", range(1, len(ranking_df) + 1))

    by_param: dict[str, pd.DataFrame] = {}
    for param in [
        "lr",
        "lr_pretrain_end_frac",
        "beta1",
        "pre_burst_steps",
        "n_a",
        "n_docs_per_task",
    ]:
        sdf = (
            raw_df.groupby(param, as_index=False)
            .agg(
                final_acc_other_mean=("final_acc_other", "mean"),
                final_acc_other_std=("final_acc_other", "std"),
                final_loss_mean=("final_loss", "mean"),
                final_loss_std=("final_loss", "std"),
                peak_acc_other_mean=("peak_acc_other", "mean"),
                first_step_acc_other_gt_99_mean=("first_step_acc_other_gt_99", "mean"),
                n_runs=("seed", "count"),
            )
            .sort_values(param)
        )
        by_param[param] = sdf

    optimal_rows = []
    for param, sdf in by_param.items():
        best = sdf.sort_values(
            ["final_acc_other_mean", "final_loss_mean"], ascending=[False, True]
        ).iloc[0]
        optimal_rows.append(
            {
                "parameter": param,
                "best_value": best[param],
                "final_acc_other_mean": best["final_acc_other_mean"],
                "final_loss_mean": best["final_loss_mean"],
                "first_step_acc_other_gt_99_mean": best["first_step_acc_other_gt_99_mean"],
                "n_runs": int(best["n_runs"]),
            }
        )
    by_param["optimal_by_param"] = pd.DataFrame(optimal_rows)
    return agg_df, ranking_df, by_param


def _write_excel(
    out_path: Path,
    raw_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    by_param: dict[str, pd.DataFrame],
) -> None:
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        raw_export = raw_df.drop(
            columns=["curve_step", "curve_acc_other", "curve_acc_burst", "curve_loss"]
        )
        raw_export.to_excel(writer, sheet_name="raw_runs", index=False)
        agg_df.to_excel(writer, sheet_name="config_agg", index=False)
        ranking_df.to_excel(writer, sheet_name="ranking", index=False)
        for param, df in by_param.items():
            sheet = f"by_{param}" if param != "optimal_by_param" else "optimal_by_param"
            df.to_excel(writer, sheet_name=sheet[:31], index=False)


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Run pretraining-only full-factorial sweep."""
    parser = argparse.ArgumentParser(description="Pretraining-only full-factorial sweep.")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--depth", type=int, default=DEPTH_DEFAULT)
    parser.add_argument("--burst-pos", type=int, default=BURST_POS_DEFAULT)
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS_DEFAULT)
    parser.add_argument("--n-workers", type=int, default=0)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Runs per task chunk. 0 = auto-chunk to better fill workers.",
    )
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Max retries for failed task chunks.",
    )
    args = parser.parse_args()

    base_cfg = TrainConfig().to_dict()
    all_runs = _build_sweep_runs(args.n_seeds, args.depth, args.burst_pos)
    if args.max_runs is not None:
        all_runs = all_runs[: max(0, args.max_runs)]
    if not all_runs:
        msg = "No runs scheduled. Check --max-runs."
        raise ValueError(msg)

    timestamp = args.run_tag or datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    out_dir = Path("data") / f"{timestamp}_pretrain_sweep_d{args.depth}_pos{args.burst_pos}"
    plots_dir = out_dir / "plots"
    per_run_dir = plots_dir / "per_run"
    summary_dir = plots_dir / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_run_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    "cuda" if torch.cuda.is_available() else "cpu"
    max_workers = args.n_workers if args.n_workers > 0 else gpu_cfg.train_workers
    grouped: dict[tuple[int, int], list[SweepRun]] = {}
    for run in all_runs:
        grouped.setdefault(run.group_key, []).append(run)

    list(grouped.items())
    tasks = _build_group_tasks(
        grouped, max_workers=max_workers, chunk_size=args.chunk_size, base_cfg=base_cfg
    )
    n_workers = max(1, min(max_workers, len(tasks)))

    rows: list[dict[str, Any]] = []
    pending = list(tasks)
    retries_left = max(0, args.max_retries)
    cur_workers = n_workers
    ctx = mp.get_context("spawn")

    while pending:
        failed: list[tuple[tuple[int, int], list[SweepRun], dict[str, Any]]] = []
        saw_cuda_oom = False

        with ProcessPoolExecutor(max_workers=cur_workers, mp_context=ctx) as ex:
            futs = {ex.submit(_run_group, t): t for t in pending}
            with tqdm(total=len(pending), desc="Task chunks", unit="chunk", ncols=100) as pbar:
                for fut in as_completed(futs):
                    task = futs[fut]
                    try:
                        rows.extend(fut.result())
                    except Exception as exc:  # noqa: BLE001
                        if _is_cuda_oom(exc):
                            saw_cuda_oom = True
                        failed.append(task)
                    finally:
                        pbar.update(1)

        if not failed:
            break

        if retries_left <= 0:
            msg = (
                f"{len(failed)} task chunks failed after retries."
                " Last failure likely shown above."
            )
            raise RuntimeError(
                msg
            )

        retries_left -= 1
        pending = failed
        if saw_cuda_oom and cur_workers > 1:
            next_workers = max(1, cur_workers // 2)
            cur_workers = min(cur_workers, next_workers)

    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        msg = "Sweep produced no rows."
        raise RuntimeError(msg)

    agg_df, ranking_df, by_param = _build_tables(raw_df)

    for row in tqdm(rows, desc="Per-run plots", unit="plot", ncols=100):
        _plot_single_run(row, per_run_dir)
    _plot_summary(raw_df, agg_df, summary_dir)

    excel_path = out_dir / "pretrain_sweep_results.xlsx"
    _write_excel(excel_path, raw_df, agg_df, ranking_df, by_param)

    ranking_df.iloc[0]


if __name__ == "__main__":
    main()
