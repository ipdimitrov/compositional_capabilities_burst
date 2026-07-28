#!/usr/bin/env bash
# GPU profile — source this to set machine-dependent parallelism.
#
# All worker counts are computed by burst/gpu.py which auto-detects your GPU.
# Override by setting GPU_VRAM_GB and/or GPU_TFLOPS before sourcing this file.

PYTHON="${PYTHON:-/venv/main/bin/python}"

eval "$("${PYTHON}" -m burst.gpu --shell)"

echo "GPU profile: ${GPU_VRAM_GB}GB VRAM, ~${GPU_TFLOPS} TFLOPS → train=${N_WORKERS}, probe=${PROBE_WORKERS}, gradsim=${GRADSIM_WORKERS}"
