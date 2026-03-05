#!/usr/bin/env bash
# Priority runs: depth-3 all positions
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

    echo "=== depth=${depth} burst_pos=${pos} ==="
    local run_dir
    run_dir=$("${PYTHON}" burst/experiment.py --depth "${depth}" --burst-pos "${pos}" \
        | tee /dev/stderr | grep "^Output:" | head -1 | awk '{print $2}')

    if [ -z "${run_dir}" ]; then
        echo "WARNING: could not detect run_dir, falling back to latest on disk"
        run_dir=$(latest_run_dir "${depth}" "${pos}")
    fi

    post_process "${run_dir}"
    RUN_DIRS+=("${run_dir}")
}

run_experiment 3 1
run_experiment 3 2
run_experiment 3 3

# # depth=4 burst at position 2  (2/4)
# run_experiment 4 1
# run_experiment 4 2
# run_experiment 4 3
# run_experiment 4 4

# # depth=5
# run_experiment 5 2

echo "=== unified analysis ==="
"${PYTHON}" burst/unified_analysis.py "${RUN_DIRS[@]}"
