#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
RUN_TAG="${1:-data_500}"
RUN_DIR="data/burst_d3_${RUN_TAG}"

echo "=== Run tag: ${RUN_TAG} ==="
echo "=== Output: ${RUN_DIR} ==="

"${PYTHON}" burst/experiment.py --run-tag "${RUN_TAG}"

"${PYTHON}" burst/plot.py "${RUN_DIR}"

"${PYTHON}" burst/probe.py "${RUN_DIR}"

"${PYTHON}" burst/plot_probes.py "${RUN_DIR}"

echo "Running next-token regime probes at steps 250, 500, 750, 1000..."
"${PYTHON}" scripts/probe_next_token_regimes.py "${RUN_DIR}" --probe-steps 250 500 750 1000

echo "=== Done: ${RUN_DIR} ==="
