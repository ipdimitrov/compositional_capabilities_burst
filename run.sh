#!/usr/bin/env bash
# Priority runs: depth-3 all positions, three burst-mode setups.
#
# Burst modes:
#   current        – original: steps scale inversely with concentration
#   constant_steps – all schedules run BURST_BASE_STEPS; only mix ratio differs
#   scaled_batch   – all schedules run BURST_BASE_STEPS; batch size scales
#                    inversely with concentration (more other-class data)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-${SCRIPT_DIR}/.venv/bin/python}"
DATA_DIR="${SCRIPT_DIR}/data"
source "${SCRIPT_DIR}/gpu_profile.sh"
source "${SCRIPT_DIR}/post_process.sh"

RUN_DIRS=()

latest_run_dir() {
    local depth="$1" pos="$2"
    ls -d "${DATA_DIR}/burst_d${depth}_pos${pos}_"* 2>/dev/null | sort | tail -1
}

run_experiment() {
    local depth="$1"
    local pos="$2"
    local mode="${3:-current}"

    echo "=== depth=${depth} burst_pos=${pos} mode=${mode} ==="
    local run_dir
    run_dir=$("${PYTHON}" burst/experiment.py \
        --depth "${depth}" --burst-pos "${pos}" --burst-mode "${mode}" \
        | tee /dev/stderr | grep "^Output:" | head -1 | awk '{print $2}')

    if [ -z "${run_dir}" ]; then
        echo "WARNING: could not detect run_dir, falling back to latest on disk"
        run_dir=$(latest_run_dir "${depth}" "${pos}")
    fi

    post_process "${run_dir}"
    RUN_DIRS+=("${run_dir}")
}

# ── depth=3 pos=3 (priority: run first) ──────────────────────────────────
# run_experiment 3 3 current
run_experiment 3 3 constant_steps
run_experiment 3 3 scaled_batch

# # ── depth=3 pos=2 (overnight) ────────────────────────────────────────────
# run_experiment 3 2 current
# run_experiment 3 2 constant_steps
# run_experiment 3 2 scaled_batch

# # ── depth=3 pos=1 (overnight) ────────────────────────────────────────────
# run_experiment 3 1 current
# run_experiment 3 1 constant_steps
# run_experiment 3 1 scaled_batch

# # ── depth=4 ─────────────────────────────────────────────────────────────
# run_experiment 4 1 current
# run_experiment 4 2 current
# run_experiment 4 3 current
# run_experiment 4 4 current

# # ── depth=5 ─────────────────────────────────────────────────────────────
# run_experiment 5 2 current

echo "=== all runs complete ==="
