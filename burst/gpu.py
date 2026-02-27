"""Unified GPU configuration — single source of truth for all parallelism.

Edit GPU_REGISTRY or set env vars GPU_VRAM_GB / GPU_TFLOPS to override
auto-detection.  Everything else (worker counts for training, probes,
grad-sim) is derived automatically.

Usage from Python:
    from burst.gpu import gpu_cfg
    print(gpu_cfg.train_workers, gpu_cfg.probe_workers)

Usage from shell (post_process.sh etc.):
    eval $($PYTHON -m burst.gpu --shell)
"""
from __future__ import annotations

import os
import math
from dataclasses import dataclass

import torch

CUDA_CONTEXT_MB = 650
MODEL_PLUS_ACTS_MB = 200
GRADSIM_WORKER_MB = 600

GPU_REGISTRY: dict[str, dict] = {
    "H100":          {"vram_gb": 80,  "tflops_bf16": 990},
    "H100 SXM":      {"vram_gb": 80,  "tflops_bf16": 990},
    "A100":          {"vram_gb": 80,  "tflops_bf16": 312},
    "A100-SXM4-80GB":{"vram_gb": 80,  "tflops_bf16": 312},
    "A100-SXM4-40GB":{"vram_gb": 40,  "tflops_bf16": 312},
    "A100-PCIE-40GB":{"vram_gb": 40,  "tflops_bf16": 312},
    "L40S":          {"vram_gb": 48,  "tflops_bf16": 362},
    "L40":           {"vram_gb": 48,  "tflops_bf16": 181},
    "L4":            {"vram_gb": 24,  "tflops_bf16": 121},
    "RTX 4090":      {"vram_gb": 24,  "tflops_bf16": 330},
    "RTX 4080":      {"vram_gb": 16,  "tflops_bf16": 200},
    "RTX 3090":      {"vram_gb": 24,  "tflops_bf16": 142},
    "RTX A6000":     {"vram_gb": 48,  "tflops_bf16": 155},
    "RTX PRO 6000":  {"vram_gb": 96,  "tflops_bf16": 117},
    "T4":            {"vram_gb": 16,  "tflops_bf16": 65},
    "V100":          {"vram_gb": 16,  "tflops_bf16": 125},
    "V100-SXM2-32GB":{"vram_gb": 32,  "tflops_bf16": 125},
}


def _detect_gpu() -> tuple[int, int]:
    """Return (vram_gb, tflops_bf16) from env vars, registry, or safe defaults."""
    env_vram = os.environ.get("GPU_VRAM_GB")
    env_tflops = os.environ.get("GPU_TFLOPS")
    if env_vram and env_tflops:
        return int(env_vram), int(env_tflops)

    if not torch.cuda.is_available():
        return 0, 0

    name = torch.cuda.get_device_name(0)
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    detected_vram = int(total_bytes / (1024 ** 3))

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
    vram_gb: int
    tflops_bf16: int
    gpu_name: str

    @property
    def ratio(self) -> float:
        if self.vram_gb == 0:
            return 0.0
        return self.tflops_bf16 / self.vram_gb

    @property
    def usable_vram_mb(self) -> int:
        return int(self.vram_gb * 1024 * 0.90)

    def _cap_by_vram(self, per_proc_mb: int) -> int:
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
        by_vram = self._cap_by_vram(per_proc_mb)
        by_compute = max(4, min(64, int(math.sqrt(self.tflops_bf16))))
        return min(by_vram, by_compute)

    @property
    def probe_workers(self) -> int:
        """Parallel processes for linear probes and next-token probes."""
        if self.vram_gb == 0:
            return 1
        per_proc_mb = CUDA_CONTEXT_MB + MODEL_PLUS_ACTS_MB * 2
        by_vram = self._cap_by_vram(per_proc_mb)
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
        by_vram = self._cap_by_vram(per_proc_mb)
        by_compute = max(2, min(32, int(math.sqrt(self.tflops_bf16) * 0.5)))
        return min(by_vram, by_compute)

    def summary(self) -> str:
        return (
            f"GPU: {self.gpu_name}\n"
            f"  VRAM: {self.vram_gb} GB, BF16 TFLOPS: {self.tflops_bf16}, "
            f"ratio: {self.ratio:.1f} TFLOPS/GB\n"
            f"  train_workers={self.train_workers}  "
            f"probe_workers={self.probe_workers}  "
            f"gradsim_workers={self.gradsim_workers}"
        )

    def shell_exports(self) -> str:
        return (
            f'export N_WORKERS={self.train_workers}\n'
            f'export PROBE_WORKERS={self.probe_workers}\n'
            f'export GRADSIM_WORKERS={self.gradsim_workers}\n'
            f'export GPU_VRAM_GB={self.vram_gb}\n'
            f'export GPU_TFLOPS={self.tflops_bf16}\n'
        )


def make_gpu_config() -> GpuConfig:
    vram, tflops = _detect_gpu()
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return GpuConfig(vram_gb=vram, tflops_bf16=tflops, gpu_name=name)


gpu_cfg = make_gpu_config()


if __name__ == "__main__":
    import sys
    if "--shell" in sys.argv:
        print(gpu_cfg.shell_exports())
    else:
        print(gpu_cfg.summary())
