#!/usr/bin/env bash
# Shared post-processing: gradient metrics, bundle, charts, and file organization.
# Sourced by run.sh and run_overnight.sh.

post_process() {
    local run_dir="$1"
    export PYTHON="${PYTHON}"
    export OPENBLAS_NUM_THREADS=1
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    echo "=== post-processing ${run_dir} ==="

    local fail=0

    echo "  Running grad-sim..."
    "${PYTHON}" -m burst.core gradients "${run_dir}" \
        || { echo "FAIL: gradients"; fail=1; }

    if [ "${RUN_NTP:-0}" = "1" ]; then
        echo "  Running next-token probes..."
        "${PYTHON}" scripts/probe_next_token_regimes.py "${run_dir}" \
            --probe-max-samples "${NTP_MAX_SAMPLES:-500}" \
            || { echo "FAIL: next-token probes"; fail=1; }
    fi

    echo "  Building bundle + charts..."
    "${PYTHON}" -m burst.core pipeline "${run_dir}" \
        || { echo "FAIL: pipeline"; fail=1; }

    echo "  Organizing files for download..."
    "${PYTHON}" scripts/organize_run.py "${run_dir}" \
        || echo "WARNING: organize_run.py failed (non-critical)"

    echo "=== Done: ${run_dir} (fail=${fail}) ==="
}
