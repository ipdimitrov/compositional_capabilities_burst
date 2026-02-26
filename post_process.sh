#!/usr/bin/env bash
# Shared post-processing: runs plot, probe, and next-token probes in parallel,
# then builds the presentation PDF once all are done.
# Sourced by run.sh and run_overnight.sh.
#
# Worker counts are auto-computed by burst/gpu.py — no manual tuning needed.

post_process() {
    local run_dir="$1"
    export PYTHON="${PYTHON:-/venv/main/bin/python}"
    export OPENBLAS_NUM_THREADS=1
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    echo "=== post-processing ${run_dir} ==="

    local T U
    T=$("${PYTHON}" -c "import json,sys; c=json.load(open('${run_dir}/config.json')); print(c['base_cfg']['total_steps'])")
    U=$("${PYTHON}" -c "import json,sys; c=json.load(open('${run_dir}/config.json')); print(c['base_cfg']['reversion_steps'])")
    local q1=$(( T / 4 ))
    local q2=$(( T / 2 ))
    local q3=$(( 3 * T / 4 ))
    local ntp_steps="${q1} ${q2} ${q3} ${T}"

    local fail=0

    echo "  Running plots (background)..."
    "${PYTHON}" burst/plot.py "${run_dir}" &
    local pid_plot=$!

    echo "  Running probes (checkpoint-loading, parallel)..."
    "${PYTHON}" burst/probe.py "${run_dir}" \
        --checkpoint-every 50 --probe-max-samples 512 &
    local pid_probe=$!

    echo "  Running next-token probes at steps: ${ntp_steps} ..."
    "${PYTHON}" scripts/probe_next_token_regimes.py "${run_dir}" \
        --probe-steps ${ntp_steps} --probe-max-samples 512 &
    local pid_ntp=$!

    wait "${pid_probe}" && "${PYTHON}" burst/plot_probes.py "${run_dir}" \
        || { echo "FAIL: probe.py / plot_probes.py"; fail=1; }
    wait "${pid_ntp}" || { echo "FAIL: probe_next_token_regimes.py"; fail=1; }

    echo "  Running grad-sim..."
    "${PYTHON}" burst/grad_sim.py "${run_dir}" \
        || { echo "FAIL: grad_sim.py"; fail=1; }

    wait "${pid_plot}" || { echo "FAIL: plot.py"; fail=1; }

    if [ "${fail}" -eq 0 ]; then
        echo "Building presentation HTML..."
        "${PYTHON}" burst/pres_pdf.py "${run_dir}"
    else
        echo "WARNING: some post-processing steps failed, skipping PDF"
    fi

    echo "  Organizing files for download..."
    "${PYTHON}" scripts/organize_run.py "${run_dir}" \
        || echo "WARNING: organize_run.py failed (non-critical)"

    echo "=== Done: ${run_dir} ==="
}
