"""Subprocess job pool for running GPU jobs in parallel.

Each job is pickled, spawned as a separate Python subprocess (own CUDA context),
and results are collected via pickle files in a temp directory.
"""

import pickle
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class JobResult:
    label: str
    success: bool
    data: Any = None
    error: str = ""
    elapsed: float = 0.0


def run_job_pool(
    jobs: list[dict],
    worker_script: str,
    build_cmd: Callable,
    on_done: Callable | None,
    n_workers: int,
    data_payload: Any,
    poll_interval: float,
    tmp_prefix: str,
    max_retries: int = 2,
) -> list[JobResult]:
    """Run jobs in parallel via subprocess pool.

    Failed jobs are retried up to max_retries times (handles transient CUDA OOM).
    """
    n_workers = min(len(jobs), n_workers)
    tmp_dir = Path(tempfile.mkdtemp(prefix=tmp_prefix))

    data_path = str(tmp_dir / "_shared_data.pkl")
    if data_payload is not None:
        with open(data_path, "wb") as f:
            pickle.dump(data_payload, f)

    t0 = time.time()
    retry_counts: dict[int, int] = {}

    def launch(idx, max_popen_retries=5):
        job = jobs[idx]
        job_path = str(tmp_dir / f"_job_{idx}.pkl")
        out_path = str(tmp_dir / f"_result_{idx}.pkl")
        with open(job_path, "wb") as f:
            pickle.dump(job, f)
        cmd = build_cmd(worker_script, job_path, data_path, out_path)
        for attempt in range(max_popen_retries):
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return proc, out_path
            except BlockingIOError:
                time.sleep(2**attempt)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return proc, out_path

    active: dict[str, tuple[int, subprocess.Popen, str]] = {}
    next_idx = 0
    for _ in range(min(n_workers, len(jobs))):
        proc, out_path = launch(next_idx)
        active[jobs[next_idx]["label"]] = (next_idx, proc, out_path)
        next_idx += 1

    results: list[JobResult] = []
    n_done = 0
    retry_queue: list[int] = []

    while n_done < len(jobs):
        time.sleep(poll_interval)
        done_labels = [name for name, (_, proc, _) in active.items() if proc.poll() is not None]
        for label in done_labels:
            idx, proc, out_path = active.pop(label)
            elapsed = time.time() - t0

            if proc.returncode == 0 and Path(out_path).exists():
                with open(out_path, "rb") as f:
                    data = pickle.load(f)
                jr = JobResult(label=label, success=True, data=data, elapsed=elapsed)
                n_done += 1
                results.append(jr)
                if on_done:
                    on_done(jr, n_done, len(jobs))
            else:
                attempt = retry_counts.get(idx, 0)
                if attempt < max_retries:
                    retry_counts[idx] = attempt + 1
                    retry_queue.append(idx)
                    se = proc.stderr.read().decode() if proc.stderr else ""
                    print(
                        f"  [{label}] failed (attempt {attempt + 1}/{max_retries + 1}), "
                        f"retrying... stderr: {se[:200]}",
                        flush=True,
                    )
                else:
                    se = proc.stderr.read().decode() if proc.stderr else ""
                    jr = JobResult(label=label, success=False, error=se[-1500:], elapsed=elapsed)
                    n_done += 1
                    results.append(jr)
                    if on_done:
                        on_done(jr, n_done, len(jobs))

        while retry_queue and len(active) < n_workers:
            ridx = retry_queue.pop(0)
            p, op = launch(ridx)
            active[jobs[ridx]["label"]] = (ridx, p, op)

        while next_idx < len(jobs) and len(active) < n_workers:
            p, op = launch(next_idx)
            active[jobs[next_idx]["label"]] = (next_idx, p, op)
            next_idx += 1

    for f in tmp_dir.glob("*"):
        f.unlink()
    tmp_dir.rmdir()

    return results
