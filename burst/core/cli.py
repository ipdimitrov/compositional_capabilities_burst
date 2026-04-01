"""CLI dispatcher for burst.core sub-commands (train, gradients, pipeline, charts)."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from burst.config import CORE_CHARTS_DIRNAME, DEFAULT_DETERMINISTIC, DEFAULT_REPRO_SEED
from burst.core.bundle import build_and_save_core_bundle, load_core_bundle
from burst.core.charts.render import render_core_charts
from burst.core.repro import set_reproducibility, write_repro_manifest
from burst.core.train_utils import resolve_run_paths

logger = logging.getLogger(__name__)


def run_core_analysis(
    run_dir: str | Path,
    *,
    build_bundle: bool = True,
    render_charts: bool = True,
    out_dir: str | Path | None = None,
) -> tuple[Path | None, list[Path]]:
    """Build the core bundle and render charts for a training run."""
    run_dir = Path(run_dir)
    bundle_out: Path | None = None
    chart_paths: list[Path] = []

    if build_bundle:
        bundle_out = build_and_save_core_bundle(run_dir)

    if render_charts:
        bundle = load_core_bundle(run_dir)
        _, _, results_dir = resolve_run_paths(run_dir)
        render_dir = Path(out_dir) if out_dir is not None else (results_dir / CORE_CHARTS_DIRNAME)
        chart_paths = render_core_charts(bundle, render_dir)

    return bundle_out, chart_paths


RunMode = Literal["train", "gradients", "bundle", "charts", "pipeline"]


@dataclass(frozen=True)
class CliCommand:
    """Parsed CLI arguments for the burst pipeline."""

    mode: RunMode
    run_dir: Path | None
    out_dir: Path | None
    seed: int
    deterministic: bool
    note: str
    train_args: list[str]
    gradients_args: list[str]


def _parse_args() -> CliCommand:  # noqa: C901, PLR0915
    """Parse CLI arguments into a CliCommand."""
    parser = argparse.ArgumentParser(description="Canonical burst pipeline CLI.")
    parser.add_argument("--seed", type=int, default=DEFAULT_REPRO_SEED)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DETERMINISTIC,
    )
    parser.add_argument("--note", type=str, default="")

    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_common_options(p: argparse.ArgumentParser) -> None:
        """Add seed, deterministic, and note options to a subparser."""
        p.add_argument("--seed", type=int, default=None)
        p.add_argument(
            "--deterministic",
            action=argparse.BooleanOptionalAction,
            default=None,
        )
        p.add_argument("--note", type=str, default=None)

    p_train = subparsers.add_parser("train")
    add_common_options(p_train)
    p_train.add_argument("--run-tag", default=None)
    p_train.add_argument("--depth", type=int, default=3)
    p_train.add_argument("--burst-pos", type=int, default=3)
    p_train.add_argument("--burst-mode", default="current")
    p_train.add_argument("--n-a", type=int, default=None)
    p_train.add_argument("--schedules", nargs="+", default=None)
    p_train.add_argument("--n-seeds", type=int, default=None)
    p_train.add_argument("--n-workers", type=int, default=None)
    p_train.add_argument("--run-probes", action="store_true", default=False)
    p_train.add_argument("--run-next-token-probes", action="store_true", default=False)
    p_train.add_argument("--run-adl", action="store_true", default=False)

    p_grad = subparsers.add_parser("gradients")
    add_common_options(p_grad)
    p_grad.add_argument("run_dir", type=Path)
    p_grad.add_argument("--n-workers", type=int, default=None)
    p_grad.add_argument("--grad-sim-batch-size", type=int, default=None)
    p_grad.add_argument("--delete-checkpoints", action="store_true")

    p_bundle = subparsers.add_parser("bundle")
    add_common_options(p_bundle)
    p_bundle.add_argument("run_dir", type=Path)

    p_charts = subparsers.add_parser("charts")
    add_common_options(p_charts)
    p_charts.add_argument("run_dir", type=Path)
    p_charts.add_argument("--out-dir", type=Path, default=None)

    p_pipeline = subparsers.add_parser("pipeline")
    add_common_options(p_pipeline)
    p_pipeline.add_argument("run_dir", type=Path)
    p_pipeline.add_argument("--out-dir", type=Path, default=None)

    args = parser.parse_args()
    mode = args.mode
    seed = args.seed if args.seed is not None else DEFAULT_REPRO_SEED
    deterministic = args.deterministic if args.deterministic is not None else DEFAULT_DETERMINISTIC
    note = args.note if args.note is not None else ""
    run_dir = getattr(args, "run_dir", None)
    out_dir = getattr(args, "out_dir", None)

    train_args: list[str] = []
    gradients_args: list[str] = []
    if mode == "train":
        for key in (
            "run_tag",
            "depth",
            "burst_pos",
            "burst_mode",
            "n_a",
            "schedules",
            "n_seeds",
            "n_workers",
            "run_probes",
            "run_next_token_probes",
            "run_adl",
        ):
            value = getattr(args, key)
            if value is None or value is False:
                continue
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                train_args.append(flag)
            elif isinstance(value, list):
                train_args.extend([flag, *[str(v) for v in value]])
            else:
                train_args.extend([flag, str(value)])
        train_args.extend(["--seed", str(seed)])
        train_args.extend(["--deterministic" if deterministic else "--no-deterministic"])

    if mode == "gradients":
        for key in ("n_workers", "grad_sim_batch_size"):
            value = getattr(args, key)
            if value is not None:
                gradients_args.extend(["--" + key.replace("_", "-"), str(value)])
        if args.delete_checkpoints:
            gradients_args.append("--delete-checkpoints")
        gradients_args.extend(["--seed", str(seed)])
        gradients_args.extend(["--deterministic" if deterministic else "--no-deterministic"])

    return CliCommand(
        mode=mode,
        run_dir=run_dir,
        out_dir=out_dir,
        seed=seed,
        deterministic=deterministic,
        note=note,
        train_args=train_args,
        gradients_args=gradients_args,
    )


def main() -> None:
    """Run the burst pipeline CLI."""
    cmd = _parse_args()
    set_reproducibility(cmd.seed, deterministic=cmd.deterministic)

    if cmd.mode == "train":
        subprocess.run(  # noqa: S603
            [sys.executable, "-m", "burst.core.train.experiment", *cmd.train_args],
            check=True,
        )
        return

    if cmd.mode == "gradients":
        assert cmd.run_dir is not None
        write_repro_manifest(
            cmd.run_dir,
            mode=cmd.mode,
            seed=cmd.seed,
            deterministic=cmd.deterministic,
            cli_args={
                "mode": cmd.mode,
                "run_dir": cmd.run_dir,
                "args": cmd.gradients_args,
            },
            note=cmd.note,
        )
        subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "burst.core.metrics.gradients",
                str(cmd.run_dir),
                *cmd.gradients_args,
            ],
            check=True,
        )
        return

    assert cmd.run_dir is not None
    bundle_path, chart_paths = run_core_analysis(
        cmd.run_dir,
        build_bundle=cmd.mode in {"bundle", "pipeline"},
        render_charts=cmd.mode in {"charts", "pipeline"},
        out_dir=cmd.out_dir,
    )
    manifest_path = write_repro_manifest(
        cmd.run_dir,
        mode=cmd.mode,
        seed=cmd.seed,
        deterministic=cmd.deterministic,
        cli_args={
            "mode": cmd.mode,
            "run_dir": cmd.run_dir,
            "out_dir": cmd.out_dir,
        },
        note=cmd.note,
    )

    if bundle_path is not None:
        logger.info("bundle: %s", bundle_path)
    if chart_paths:
        target_dir = (
            Path(cmd.out_dir)
            if cmd.out_dir is not None
            else (resolve_run_paths(cmd.run_dir)[2] / CORE_CHARTS_DIRNAME)
        )
        logger.info("charts: %s", target_dir)
        logger.info("count: %s", len(chart_paths))
    logger.info("repro_manifest: %s", manifest_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
