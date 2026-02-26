#!/usr/bin/env bash
# All remaining position/depth variants (run overnight)
# Excludes 2/3, 2/4, 2/5 which are in run.sh
set -euo pipefail

PYTHON="${PYTHON:-/venv/main/bin/python}"
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

# depth=3: positions 1/3 and 3/3
run_experiment 3 1
run_experiment 3 3

# depth=4: positions 1/4, 3/4, 4/4
run_experiment 4 1
run_experiment 4 3
run_experiment 4 4

# depth=5: positions 1/5, 3/5, 4/5, 5/5
run_experiment 5 1
run_experiment 5 3
run_experiment 5 4
run_experiment 5 5
