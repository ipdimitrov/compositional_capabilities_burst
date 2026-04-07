"""Subprocess job pool for running GPU jobs in parallel.

Each job is pickled, spawned as a separate Python subprocess (own CUDA context),
and results are collected via pickle files in a temp directory.
"""

import logging
import pickle
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class JobResult:
    """Result of a single subprocess job."""

    label: str
    success: bool
    data: Any = None
    error: str = ""
    elapsed: float = 0.0


def run_job_pool(  # noqa: C901, PLR0912, PLR0913, PLR0915
    jobs: list[dict],
    worker_script: str,
    build_cmd: Callable,
    on_done: Callable | None,
    n_workers: int,
    data_payload: object,
    poll_interval: float,
    tmp_prefix: str,
    max_retries: int = 2,
) -> list[JobResult]:
    """Run jobs in parallel via subprocess pool.

    Failed jobs are retried up to max_retries times (handles transient CUDA OOM).
    """
    n_workers = min(len(jobs), n_workers)
    tmp_dir = Path(tempfile.mkdtemp(prefix=tmp_prefix))

    data_path = tmp_dir / "_shared_data.pkl"
    if data_payload is not None:
        with data_path.open("wb") as f:
            pickle.dump(data_payload, f)

    t0 = time.time()
    retry_counts: dict[int, int] = {}

    def launch(idx: int, max_popen_retries: int = 5) -> tuple[subprocess.Popen, Path]:
        """Pickle job, spawn subprocess, return (proc, output_path)."""
        job = jobs[idx]
        job_path = tmp_dir / f"_job_{idx}.pkl"
        out_path = tmp_dir / f"_result_{idx}.pkl"
        with job_path.open("wb") as f:
            pickle.dump(job, f)
        cmd = build_cmd(worker_script, str(job_path), str(data_path), str(out_path))
        for attempt in range(max_popen_retries):
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # noqa: S603
                return proc, out_path  # noqa: TRY300
            except BlockingIOError:
                time.sleep(2**attempt)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # noqa: S603
        return proc, out_path

    active: dict[str, tuple[int, subprocess.Popen, Path]] = {}
    next_idx = 0
    for _ in range(min(n_workers, len(jobs))):
        proc, out_path = launch(next_idx)
        active[jobs[next_idx]["label"]] = (next_idx, proc, out_path)
        next_idx += 1

    results: list[JobResult] = []
    n_done = 0
    retry_queue: list[int] = []

    n_total = len(jobs)
    n_fail = 0

    while n_done < n_total:
        time.sleep(poll_interval)
        done_labels = [name for name, (_, proc, _) in active.items() if proc.poll() is not None]
        for label in done_labels:
            idx, proc, out_path = active.pop(label)
            elapsed = time.time() - t0

            if proc.returncode == 0 and out_path.exists():
                with out_path.open("rb") as f:
                    data = pickle.load(f)  # noqa: S301
                jr = JobResult(label=label, success=True, data=data, elapsed=elapsed)
                n_done += 1
                results.append(jr)
                if on_done:
                    on_done(jr, n_done, n_total)
            else:
                attempt = retry_counts.get(idx, 0)
                if attempt < max_retries:
                    retry_counts[idx] = attempt + 1
                    retry_queue.append(idx)
                    se = proc.stderr.read().decode() if proc.stderr else ""
                    logger.warning(
                        "  [%s] failed (attempt %d/%d), retrying... stderr: %s",
                        label, attempt + 1, max_retries + 1, se[:200],
                    )
                else:
                    se = proc.stderr.read().decode() if proc.stderr else ""
                    jr = JobResult(label=label, success=False, error=se[-1500:], elapsed=elapsed)
                    n_done += 1
                    n_fail += 1
                    results.append(jr)
                    if on_done:
                        on_done(jr, n_done, n_total)

        while retry_queue and len(active) < n_workers:
            ridx = retry_queue.pop(0)
            p, op = launch(ridx)
            active[jobs[ridx]["label"]] = (ridx, p, op)

        while next_idx < n_total and len(active) < n_workers:
            p, op = launch(next_idx)
            active[jobs[next_idx]["label"]] = (next_idx, p, op)
            next_idx += 1

        elapsed = time.time() - t0
        pct = n_done / n_total * 100
        eta = (elapsed / n_done * (n_total - n_done)) if n_done > 0 else 0
        fail_str = f"  {n_fail} failed" if n_fail else ""
        sys.stderr.write(
            f"\r  [{n_done}/{n_total}] {pct:5.1f}%  "
            f"elapsed {elapsed:.0f}s  eta {eta:.0f}s{fail_str}   "
        )
        sys.stderr.flush()

    sys.stderr.write("\n")
    sys.stderr.flush()

    for f in tmp_dir.glob("*"):
        f.unlink()
    tmp_dir.rmdir()

    return results
