"""Unified GPU configuration — single source of truth for all parallelism.

Edit GPU_REGISTRY or set env vars GPU_VRAM_GB / GPU_TFLOPS to override
auto-detection.  Everything else (worker counts for training, probes,
grad-sim) is derived automatically.

Usage from Python:
    from burst.core.gpu import gpu_cfg
    print(gpu_cfg.train_workers, gpu_cfg.probe_workers)

Usage from shell (post_process.sh etc.):
    eval $($PYTHON -m burst.core.gpu --shell)
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

CUDA_CONTEXT_MB = 1300
MODEL_PLUS_ACTS_MB = 200
GRADSIM_WORKER_MB = 5000

GPU_REGISTRY: dict[str, dict] = {
    "H100": {"vram_gb": 80, "tflops_bf16": 990},
    "H100 SXM": {"vram_gb": 80, "tflops_bf16": 990},
    "A100": {"vram_gb": 80, "tflops_bf16": 312},
    "A100-SXM4-80GB": {"vram_gb": 80, "tflops_bf16": 312},
    "A100-SXM4-40GB": {"vram_gb": 40, "tflops_bf16": 312},
    "A100-PCIE-40GB": {"vram_gb": 40, "tflops_bf16": 312},
    "L40S": {"vram_gb": 48, "tflops_bf16": 362},
    "L40": {"vram_gb": 48, "tflops_bf16": 181},
    "L4": {"vram_gb": 24, "tflops_bf16": 121},
    "RTX 4090": {"vram_gb": 24, "tflops_bf16": 330},
    "RTX 4080": {"vram_gb": 16, "tflops_bf16": 200},
    "RTX 3090": {"vram_gb": 24, "tflops_bf16": 142},
    "RTX A6000": {"vram_gb": 48, "tflops_bf16": 155},
    "RTX PRO 6000": {"vram_gb": 96, "tflops_bf16": 117},
    "T4": {"vram_gb": 16, "tflops_bf16": 65},
    "V100": {"vram_gb": 16, "tflops_bf16": 125},
    "V100-SXM2-32GB": {"vram_gb": 32, "tflops_bf16": 125},
}


def detect_gpu() -> tuple[int, int]:
    """Return (vram_gb, tflops_bf16) from env vars, registry, or safe defaults."""
    env_vram = os.environ.get("GPU_VRAM_GB")
    env_tflops = os.environ.get("GPU_TFLOPS")
    if env_vram and env_tflops:
        return int(env_vram), int(env_tflops)

    if not torch.cuda.is_available():
        return 0, 0

    name = torch.cuda.get_device_name(0)
    detected_vram = int(torch.cuda.get_device_properties(0).total_memory / (1024**3))

    for key, spec in GPU_REGISTRY.items():
        if key.lower() in name.lower():
            vram = int(env_vram) if env_vram else max(detected_vram, spec["vram_gb"])
            tflops = int(env_tflops) if env_tflops else spec["tflops_bf16"]
            return vram, tflops

    vram = int(env_vram) if env_vram else detected_vram
    tflops = int(env_tflops) if env_tflops else max(50, detected_vram)
    return vram, tflops


@dataclass(frozen=True)
class GpuConfig:
    """Single source of truth for GPU capabilities and worker counts."""

    vram_gb: int
    tflops_bf16: int
    gpu_name: str

    @property
    def ratio(self) -> float:
        """Return TFLOPS-per-GB ratio."""
        if self.vram_gb == 0:
            return 0.0
        return self.tflops_bf16 / self.vram_gb

    @property
    def usable_vram_mb(self) -> int:
        """Return usable VRAM in MB after reserving 10% headroom."""
        return int(self.vram_gb * 1024 * 0.90)

    def cap_by_vram(self, per_proc_mb: int) -> int:
        """Return max worker count that fits in usable VRAM."""
        return max(1, int(self.usable_vram_mb * 0.85 / per_proc_mb))

    @property
    def train_workers(self) -> int:
        """Max parallel *processes* for training (batched mode).

        Each process holds one CUDA context + one model at a time.
        With batched workers, each process trains multiple jobs sequentially.

        Too many processes = SM contention + context overhead.
        Too few = GPU idles between sequential jobs within a process.
        """
        if self.vram_gb == 0:
            return 1
        per_proc_mb = CUDA_CONTEXT_MB + MODEL_PLUS_ACTS_MB
        by_vram = self.cap_by_vram(per_proc_mb)
        by_compute = max(4, min(64, int(math.sqrt(self.tflops_bf16))))
        return min(by_vram, by_compute)

    @property
    def probe_workers(self) -> int:
        """Parallel processes for linear probes and next-token probes."""
        if self.vram_gb == 0:
            return 1
        per_proc_mb = CUDA_CONTEXT_MB + MODEL_PLUS_ACTS_MB * 2
        by_vram = self.cap_by_vram(per_proc_mb)
        by_compute = max(4, min(48, int(math.sqrt(self.tflops_bf16) * 0.75)))
        return min(by_vram, by_compute)

    @property
    def gradsim_workers(self) -> int:
        """Parallel processes for gradient cosine similarity.

        Grad-sim uses larger batch sizes (2048) so each worker is heavier.
        """
        if self.vram_gb == 0:
            return 1
        per_proc_mb = CUDA_CONTEXT_MB + GRADSIM_WORKER_MB
        by_vram = self.cap_by_vram(per_proc_mb)
        by_compute = max(2, min(32, int(math.sqrt(self.tflops_bf16) * 0.5)))
        return min(by_vram, by_compute)

    @property
    def frankenstein_workers(self) -> int:
        """Concurrent model copies for batched frankenstein eval.

        Each copy is inference-only (no grads, no optimizer) so the
        footprint is just model weights + KV-cache + activations.
        Shared CUDA context across all copies (single process).
        """
        if self.vram_gb == 0:
            return 1
        per_model_mb = MODEL_PLUS_ACTS_MB
        by_vram = max(1, int((self.usable_vram_mb - CUDA_CONTEXT_MB) * 0.85 / per_model_mb))
        by_compute = max(2, min(32, int(math.sqrt(self.tflops_bf16) * 0.6)))
        return min(by_vram, by_compute)

    def train_workers_for_batch_size(self, batch_size: int, base_batch_size: int = 128) -> int:
        """Safe worker count when batch_size exceeds the default.

        Extra activation memory per worker scales roughly linearly with
        batch_size.  We keep the same compute cap and re-check the VRAM cap
        with the adjusted per-process footprint.
        """
        if self.vram_gb == 0:
            return 1
        extra_mb = max(0, (batch_size - base_batch_size)) * 0.1
        per_proc_mb = CUDA_CONTEXT_MB + MODEL_PLUS_ACTS_MB + extra_mb
        by_vram = self.cap_by_vram(int(per_proc_mb))
        by_compute = max(4, min(64, int(math.sqrt(self.tflops_bf16))))
        return min(by_vram, by_compute)

    def summary(self) -> str:
        """Return a human-readable GPU config summary."""
        return (
            f"GPU: {self.gpu_name}\n"
            f"  VRAM: {self.vram_gb} GB, BF16 TFLOPS: {self.tflops_bf16}, "
            f"ratio: {self.ratio:.1f} TFLOPS/GB\n"
            f"  train_workers={self.train_workers}  "
            f"probe_workers={self.probe_workers}  "
            f"gradsim_workers={self.gradsim_workers}  "
            f"frankenstein_workers={self.frankenstein_workers}"
        )

    def shell_exports(self) -> str:
        """Return shell export statements for worker counts and GPU info."""
        return (
            f"export N_WORKERS={self.train_workers}\n"
            f"export PROBE_WORKERS={self.probe_workers}\n"
            f"export GRADSIM_WORKERS={self.gradsim_workers}\n"
            f"export FRANK_WORKERS={self.frankenstein_workers}\n"
            f"export GPU_VRAM_GB={self.vram_gb}\n"
            f"export GPU_TFLOPS={self.tflops_bf16}\n"
        )


def make_gpu_config() -> GpuConfig:
    """Detect GPU and return a frozen GpuConfig."""
    vram, tflops = detect_gpu()
    return GpuConfig(
        vram_gb=vram,
        tflops_bf16=tflops,
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    )


gpu_cfg = make_gpu_config()


if __name__ == "__main__":
    import sys

    if "--shell" in sys.argv:
        print(gpu_cfg.shell_exports())  # noqa: T201
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        logger.info(gpu_cfg.summary())
