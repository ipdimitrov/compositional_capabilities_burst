"""Next-token logit-lens and learned probes per layer for Other- vs Burst-class regimes.

Runs probes and saves per-label JSON results for the core bundle/chart pipeline.
Charts are rendered as PDFs via ``burst.core.charts.render``.

Usage:
    python scripts/probe_next_token_regimes.py <run_dir> --probe-max-samples 500
    RUN_NTP=1 bash run.sh   # integrated via post_process.sh
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from burst.config import (
    DATA_SEED,
    PROBE_METHODS,
    parse_run_config,
)
from burst.core.gpu import gpu_cfg
from burst.core.metrics.probes import (
    NTP_RESULTS_DIRNAME,
    probe_from_checkpoints_at_steps,
    retrain_and_probe_at_steps,
    save_probe_record,
)
from burst.core.parallel import JobResult, run_job_pool
from burst.core.train.experiment import DepthNData, build_data
from burst.core.train_utils import (
    DEVICE,
    N_PROBE_DOCS_PER_TASK,
    build_probe_docs,
    resolve_run_paths,
)
from synthetic.init import set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker subprocess entry
# ---------------------------------------------------------------------------


def worker_main() -> None:
    """Subprocess entry: load pickled args, run single probe job, save results."""
    import warnings  # noqa: PLC0415

    warnings.filterwarnings("ignore", message=".*backward hook.*")

    parser = argparse.ArgumentParser()
    parser.add_argument("--job-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--probe-steps", type=int, nargs="+", required=True)
    parser.add_argument("--n-layers", type=int, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--depth", type=int, required=True)
    wargs = parser.parse_args()

    with Path(wargs.job_path).open("rb") as f:
        job = pickle.load(f)  # noqa: S301
    with Path(wargs.data_path).open("rb") as f:
        tp, bp, other_docs, burst_docs = pickle.load(f)  # noqa: S301

    ckpt_dir = job.get("ckpt_dir")
    if ckpt_dir and Path(ckpt_dir).exists():
        step_results = probe_from_checkpoints_at_steps(
            job,
            Path(ckpt_dir),
            wargs.probe_steps,
            other_docs,
            burst_docs,
            wargs.n_layers,
            wargs.seq_len,
            wargs.max_samples,
            wargs.depth,
        )
    else:
        step_results = retrain_and_probe_at_steps(
            job,
            tp,
            bp,
            wargs.probe_steps,
            other_docs,
            burst_docs,
            wargs.n_layers,
            wargs.seq_len,
            wargs.max_samples,
            wargs.depth,
        )

    with Path(wargs.output_path).open("wb") as f:
        pickle.dump({"label": job["label"], "step_results": step_results}, f)


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: C901, PLR0915
    """Run next-token regime probes and save per-label JSON results."""
    parser = argparse.ArgumentParser(
        description="Next-token probes (logit lens + learned) for Other vs Burst regimes"
    )
    parser.add_argument("run_dir", type=str)
    parser.add_argument(
        "--probe-steps",
        type=int,
        nargs="+",
        default=None,
        help="Global steps to probe at (default: total_steps + reversion_steps)",
    )
    parser.add_argument(
        "--probe-step",
        type=int,
        default=None,
        help="Single step (legacy, use --probe-steps for multiple)",
    )
    parser.add_argument("--probe-max-samples", type=int, required=True)
    parser.add_argument("--seed-override", type=int, default=None)
    parser.add_argument("--n-workers", type=int, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg_path, logs_dir, results_dir = resolve_run_paths(run_dir)
    with cfg_path.open() as f:
        cfg = json.load(f)

    rc = parse_run_config(cfg)
    bcfg, depth, burst_pos, n_a = rc["base_cfg"], rc["depth"], rc["burst_pos"], rc["n_a"]
    total_steps = bcfg["total_steps"]
    reversion_steps = bcfg["reversion_steps"]
    seq_len = bcfg["seq_len"]
    n_layers = bcfg["n_layer"]

    if args.probe_steps:
        probe_steps = args.probe_steps
    elif args.probe_step is not None:
        probe_steps = [args.probe_step]
    else:
        probe_steps = [total_steps + reversion_steps]

    json_out_dir = results_dir / NTP_RESULTS_DIRNAME

    logger.info("Run dir: %s", run_dir)
    logger.info("Probe steps: %s", probe_steps)
    logger.info("Output: %s", json_out_dir)
    logger.info("Device: %s", DEVICE)
    logger.info("Methods: %s", PROBE_METHODS)

    logger.info("\nRebuilding data (seed=%s)...", DATA_SEED)
    tp, bp, _, _, _, _, cfg_out, ti = build_data(bcfg, depth, burst_pos, n_a)
    doc_len = ti["doc_len"]
    logger.info("  doc_len=%s  seq_len=%s", doc_len, seq_len)

    set_seed(DATA_SEED)
    d = DepthNData(bcfg["n_alphabets"], seq_len, n_a, depth, burst_pos, DATA_SEED)
    other_docs, burst_docs = build_probe_docs(d, doc_len, N_PROBE_DOCS_PER_TASK)
    logger.info("  Other docs: %s  Burst docs: %s", other_docs.shape, burst_docs.shape)

    jobs_cfg = cfg["jobs"]
    if args.seed_override is not None:
        jobs_cfg = [j for j in jobs_cfg if j["seed"] == args.seed_override]

    ckpt_root = logs_dir / "checkpoints"
    use_checkpoints = ckpt_root.exists()

    schedules_to_run = sorted({j["schedule"] for j in jobs_cfg})
    n_workers = min(len(jobs_cfg), args.n_workers or gpu_cfg.probe_workers)
    logger.info("\n%s", gpu_cfg.summary())
    logger.info("Schedules: %s", schedules_to_run)
    logger.info("Jobs: %s, workers: %s", len(jobs_cfg), n_workers)
    mode = "checkpoint-loading" if use_checkpoints else "retrain"
    logger.info("Mode: %s (%s jobs, probing at %s steps)\n", mode, len(jobs_cfg), len(probe_steps))

    jobs = []
    for jcfg in jobs_cfg:
        label, seed, schedule = jcfg["label"], jcfg["seed"], jcfg["schedule"]
        job_entry = {
            "label": label,
            "schedule": schedule,
            "seed": seed,
            "cfg": {
                **bcfg,
                "seed": seed,
                "vocab_size": cfg_out["vocab_size"],
                "context_size": cfg_out["context_size"],
            },
        }
        if use_checkpoints:
            job_entry["ckpt_dir"] = str(ckpt_root / label)
        jobs.append(job_entry)

    step_args = [str(s) for s in probe_steps]

    def build_cmd(script: str, job_path: str, data_path: str, output_path: str) -> list[str]:
        """Build the subprocess command for a single probe worker."""
        return [
            sys.executable,
            script,
            "--worker",
            "--job-path",
            job_path,
            "--data-path",
            data_path,
            "--output-path",
            output_path,
            "--probe-steps",
            *step_args,
            "--n-layers",
            str(n_layers),
            "--seq-len",
            str(seq_len),
            "--max-samples",
            str(args.probe_max_samples),
            "--depth",
            str(depth),
        ]

    all_step_results: dict[int, dict] = {step: {} for step in probe_steps}

    def on_done(jr: JobResult, n_done: int, n_total: int) -> None:
        """Collect results from a completed probe worker."""
        if jr.success:
            for step, res in jr.data["step_results"].items():
                all_step_results[step][jr.data["label"]] = res
            job_info = next(j for j in jobs if j["label"] == jr.data["label"])
            save_probe_record(
                json_out_dir,
                jr.data["label"],
                job_info["schedule"],
                job_info["seed"],
                probe_steps,
                jr.data["step_results"],
            )
            logger.info("  [%s/%s] %-30s done (%.0fs)", n_done, n_total, jr.label, jr.elapsed)
        else:
            logger.info("  FAIL [%s/%s]: %s", n_done, n_total, jr.label)
            if jr.error:
                logger.info("    %s", jr.error)

    run_job_pool(
        jobs=jobs,
        worker_script=str(Path(__file__).resolve()),
        build_cmd=build_cmd,
        on_done=on_done,
        n_workers=n_workers,
        data_payload=(tp, bp, other_docs, burst_docs),
        poll_interval=1.0,
        tmp_prefix="probe_ntp_",
    )

    n_saved = sum(1 for _ in json_out_dir.glob("*.json")) if json_out_dir.exists() else 0
    logger.info("\nDone. %s JSON records saved to %s", n_saved, json_out_dir)
    logger.info("Run `python -m burst.core pipeline %s` to build bundle + PDF charts.", run_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if "--worker" in sys.argv:
        sys.argv.remove("--worker")
        worker_main()
    else:
        main()
