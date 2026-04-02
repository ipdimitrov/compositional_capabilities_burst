"""Batched worker: trains multiple jobs sequentially in ONE CUDA context.

Launched as a subprocess by experiment.py. Instead of spawning one process
per job (each with ~400 MB CUDA context overhead), this worker receives a
list of jobs and trains them one after another, reusing the same context.

For 1M-param models the CUDA context dominates memory; batching N jobs
into one process saves ~(N-1)*400 MB of VRAM.
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import torch

from burst.core.train.worker import run
from burst.core.train_utils import DEVICE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--progress-dir", required=True)
    args = parser.parse_args()

    with Path(args.jobs_path).open("rb") as f:
        jobs = pickle.load(f)  # noqa: S301

    for job in jobs:
        try:
            run(job, args.data_path, args.run_dir, args.progress_dir)
        except torch.cuda.OutOfMemoryError:
            logging.getLogger(__name__).exception("CUDA OOM on %s", job.get("label", "?"))
        except Exception:
            logging.getLogger(__name__).exception("WORKER FAIL %s", job.get("label", "?"))
            raise
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
