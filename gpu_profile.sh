#!/usr/bin/env bash
# GPU profile — source this to set machine-dependent parallelism.
#
# All worker counts are computed by burst/gpu.py which auto-detects your GPU.
# Override by setting GPU_VRAM_GB and/or GPU_TFLOPS before sourcing this file.

SCRIPT_DIR_GPU="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-${SCRIPT_DIR_GPU}/.venv/bin/python}"

eval "$("${PYTHON}" "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/burst/core/gpu.py" --shell)"

echo "GPU profile: ${GPU_VRAM_GB}GB VRAM, ~${GPU_TFLOPS} TFLOPS → train=${N_WORKERS}, probe=${PROBE_WORKERS}, gradsim=${GRADSIM_WORKERS}"
