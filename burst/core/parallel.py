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
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
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


@dataclass
class JobPool:
    """Manage a pool of subprocess jobs with retry and progress reporting.

    Failed jobs are retried up to max_retries times (handles transient CUDA OOM).
    """

    jobs: list[dict]
    worker_script: str
    build_cmd: Callable
    on_done: Callable | None
    n_workers: int
    data_payload: Any
    poll_interval: float
    tmp_prefix: str
    max_retries: int

    tmp_dir: Path = field(init=False, repr=False)
    data_path: Path = field(init=False, repr=False)
    retry_counts: dict[int, int] = field(init=False, repr=False, default_factory=dict)
    retry_queue: deque[int] = field(init=False, repr=False, default_factory=deque)

    def run(self) -> list[JobResult]:
        """Execute all jobs and return results."""
        with tempfile.TemporaryDirectory(prefix=self.tmp_prefix) as td:
            self.tmp_dir = Path(td)
            self.data_path = self.tmp_dir / "_shared_data.pkl"
            if self.data_payload is not None:
                with self.data_path.open("wb") as f:
                    pickle.dump(self.data_payload, f)
            return self.poll_loop()

    def spawn(self, idx: int, max_popen_retries: int = 5) -> tuple[subprocess.Popen, Path]:
        """Pickle job config and spawn a subprocess."""
        job_path = self.tmp_dir / f"_job_{idx}.pkl"
        out_path = self.tmp_dir / f"_result_{idx}.pkl"
        with job_path.open("wb") as f:
            pickle.dump(self.jobs[idx], f)

        cmd = self.build_cmd(
            self.worker_script, str(job_path), str(self.data_path), str(out_path)
        )
        last_err: BlockingIOError | None = None
        for attempt in range(max_popen_retries):
            try:
                return subprocess.Popen(  # noqa: S603
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                ), out_path
            except BlockingIOError as e:
                last_err = e
                time.sleep(2**attempt)
        if last_err is not None:
            raise last_err
        msg = "unreachable: max_popen_retries must be >= 1"
        raise RuntimeError(msg)

    def collect_done(self, proc: subprocess.Popen, out_path: Path, label: str, idx: int,
                     elapsed: float) -> JobResult | None:
        """Handle a finished process: return JobResult or enqueue retry."""
        if proc.returncode == 0 and out_path.exists():
            with out_path.open("rb") as f:
                data = pickle.load(f)  # noqa: S301
            return JobResult(label=label, success=True, data=data, elapsed=elapsed)

        attempt = self.retry_counts.get(idx, 0)
        se = proc.stderr.read().decode() if proc.stderr else ""
        if attempt < self.max_retries:
            self.retry_counts[idx] = attempt + 1
            self.retry_queue.append(idx)
            logger.warning(
                "  [%s] failed (attempt %d/%d), retrying... stderr: %s",
                label, attempt + 1, self.max_retries + 1, se[:200],
            )
            return None

        return JobResult(label=label, success=False, error=se[-1500:], elapsed=elapsed)

    def poll_loop(self) -> list[JobResult]:
        """Poll subprocesses, collect results, and schedule retries."""
        n_workers = min(len(self.jobs), self.n_workers)
        n_total = len(self.jobs)
        t0 = time.time()

        active: dict[str, tuple[int, subprocess.Popen, Path]] = {}
        results: list[JobResult] = []
        n_done = 0
        n_fail = 0
        next_idx = 0

        for _ in range(min(n_workers, n_total)):
            proc, out_path = self.spawn(next_idx)
            active[self.jobs[next_idx]["label"]] = (next_idx, proc, out_path)
            next_idx += 1

        while n_done < n_total:
            time.sleep(self.poll_interval)

            done_labels = [
                lbl for lbl, (_, proc, _) in active.items() if proc.poll() is not None
            ]
            for label in done_labels:
                idx, proc, out_path = active.pop(label)
                elapsed = time.time() - t0
                jr = self.collect_done(proc, out_path, label, idx, elapsed)
                if jr is None:
                    continue
                n_done += 1
                n_fail += not jr.success
                results.append(jr)
                if self.on_done:
                    self.on_done(jr, n_done, n_total)

            while self.retry_queue and len(active) < n_workers:
                ridx = self.retry_queue.popleft()
                p, op = self.spawn(ridx)
                active[self.jobs[ridx]["label"]] = (ridx, p, op)

            while next_idx < n_total and len(active) < n_workers:
                p, op = self.spawn(next_idx)
                active[self.jobs[next_idx]["label"]] = (next_idx, p, op)
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
        return results


