#!/usr/bin/env bash
# Priority runs: depth-3 pos-2, depth-4 pos-2, depth-5 pos-2
set -euo pipefail

PYTHON="${PYTHON:-$(dirname "$0")/.venv/bin/python}"
source "$(dirname "$0")/gpu_profile.sh"
source "$(dirname "$0")/post_process.sh"

run_experiment() {
    local depth="$1"
    local pos="$2"

    echo "=== depth=${depth} burst_pos=${pos} ==="
    local run_dir
    run_dir=$("${PYTHON}" burst/experiment.py --depth "${depth}" --burst-pos "${pos}" \
        | tee /dev/stderr | grep "^Output:" | head -1 | awk '{print $2}')

    post_process "${run_dir}"
}

# depth=3 burst at position 2  (2/3)
# run_experiment 3 2

# # depth=4 burst at position 2  (2/4)
run_experiment 4 2

# # depth=5 burst at position 2  (2/5)
# run_experiment 5 2
