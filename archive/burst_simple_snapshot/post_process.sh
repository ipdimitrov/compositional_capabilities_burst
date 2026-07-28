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
    export BURST_MINIMAL_PLOTS="${BURST_MINIMAL_PLOTS:-1}"
    echo "=== post-processing ${run_dir} ==="

    local cfg_json="${run_dir}/results/config.json"
    if [ ! -f "${cfg_json}" ]; then
        cfg_json="${run_dir}/config.json"
    fi

    local fail=0

    echo "  Running plots (background)..."
    "${PYTHON}" burst/plot.py "${run_dir}" &
    local pid_plot=$!

    # --- Probes, NTP, ADL, EWC currently disabled ---
    # To re-enable probes:    uncomment the probe block below and set --run-probes in experiment.py
    # To re-enable ADL:       uncomment the ADL block below and set --run-adl in experiment.py
    # To re-enable EWC:       uncomment the EWC block below
    # To re-enable full report (unified/basin/extended): change --no-full to --full in pres_pdf.py call

    # local run_probes run_ntp run_adl
    # run_probes=$("${PYTHON}" -c "import json; c=json.load(open('${cfg_json}')); print(c.get('run_probes', False))")
    # run_ntp=$("${PYTHON}" -c "import json; c=json.load(open('${cfg_json}')); print(c.get('run_next_token_probes', False))")
    # run_adl=$("${PYTHON}" -c "import json; c=json.load(open('${cfg_json}')); print(c.get('run_adl', True))")
    #
    # if [ "${run_probes}" = "True" ]; then
    #     "${PYTHON}" burst/probe.py "${run_dir}" --checkpoint-every 50 --probe-max-samples 512
    #     "${PYTHON}" burst/plot_probes.py "${run_dir}"
    # fi
    # if [ "${run_ntp}" = "True" ]; then
    #     "${PYTHON}" scripts/probe_next_token_regimes.py "${run_dir}" --probe-steps ...
    # fi
    # if [ "${run_adl}" = "True" ]; then
    #     "${PYTHON}" burst/adl.py "${run_dir}"
    # fi
    # "${PYTHON}" burst/ewc_metrics.py "${run_dir}" --out-dir "${run_dir}/results/ewc_metrics" --n-fisher-batches 200 --n-seeds 3

    echo "  Running grad-sim..."
    "${PYTHON}" burst/grad_sim.py "${run_dir}" \
        || { echo "FAIL: grad_sim.py"; fail=1; }

    wait "${pid_plot}" || { echo "FAIL: plot.py"; fail=1; }

    echo "  Running new analysis (weight diff, activations, basin, norms, sharpness)..."
    "${PYTHON}" burst/new_analysis.py "${run_dir}" \
        --n-seeds 3 --basin-runs 50 --basin-points 8 \
        || { echo "FAIL: new_analysis.py"; fail=1; }

    echo "  Building report (HTML + TXT, no unified/basin/extended)..."
    "${PYTHON}" burst/pres_pdf.py "${run_dir}" --n-seeds 3 \
        || echo "WARNING: pres_pdf.py failed (non-critical)"

    echo "  Organizing files for download..."
    "${PYTHON}" scripts/organize_run.py "${run_dir}" \
        || echo "WARNING: organize_run.py failed (non-critical)"

    echo "=== Done: ${run_dir} ==="
}
