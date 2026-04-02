"""Reproducibility: RNG seeding and JSON manifest generation."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from burst.config import REPRO_MANIFEST_FILENAME


def set_reproducibility(seed: int, *, deterministic: bool) -> None:
    """Seed all RNGs and configure torch determinism settings."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=not deterministic)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = not deterministic
    torch.backends.cudnn.allow_tf32 = not deterministic


def _git_sha() -> str | None:
    """Return the current git HEAD SHA, or None on failure."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True  # noqa: S607
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _runtime() -> dict[str, Any]:
    """Collect runtime environment info (Python, torch, CUDA, git)."""
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": gpu_name,
        "git_sha": _git_sha(),
    }


def _jsonable(value: object) -> str | dict | list | int | float | bool | None:
    """Recursively convert Path objects to strings for JSON serialisation."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def write_repro_manifest(
    run_dir: str | Path,
    *,
    mode: str,
    seed: int,
    deterministic: bool,
    cli_args: dict[str, Any],
    note: str = "",
) -> Path:
    """Write a JSON reproducibility manifest to the run's results directory."""
    run_dir = Path(run_dir)
    results_dir = run_dir / "results" if (run_dir / "results").exists() else run_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = results_dir / REPRO_MANIFEST_FILENAME
    payload = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "mode": mode,
        "seed": seed,
        "deterministic": deterministic,
        "note": note,
        "cli_args": _jsonable(cli_args),
        "runtime": _runtime(),
    }
    with manifest_path.open("w") as f:
        json.dump(payload, f, indent=2)
    return manifest_path
