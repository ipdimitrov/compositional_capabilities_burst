#!/usr/bin/env bash
# Shared post-processing: runs plot, probe, and next-token probes in parallel,
# then builds the presentation PDF once all are done.
# Sourced by run.sh and run_overnight.sh.

post_process() {
    local run_dir="$1"
    echo "=== post-processing ${run_dir} (parallel) ==="

    "${PYTHON}" burst/plot.py "${run_dir}" &
    local pid_plot=$!

    (
        "${PYTHON}" burst/probe.py "${run_dir}" \
            --checkpoint-every 50 --probe-max-samples 512 --n-workers 38 && \
        "${PYTHON}" burst/plot_probes.py "${run_dir}"
    ) &
    local pid_probe=$!

    "${PYTHON}" scripts/probe_next_token_regimes.py "${run_dir}" \
        --probe-steps 250 500 750 1000 --probe-max-samples 512 --n-workers 38 &
    local pid_ntp=$!

    local fail=0
    wait "${pid_plot}" || { echo "FAIL: plot.py"; fail=1; }
    wait "${pid_probe}" || { echo "FAIL: probe.py / plot_probes.py"; fail=1; }
    wait "${pid_ntp}" || { echo "FAIL: probe_next_token_regimes.py"; fail=1; }

    if [ "${fail}" -eq 0 ]; then
        echo "Building presentation PDF..."
        "${PYTHON}" burst/pres_pdf.py "${run_dir}"
    else
        echo "WARNING: some post-processing steps failed, skipping PDF"
    fi
    echo "=== Done: ${run_dir} ==="
}
