#!/usr/bin/env bash
# Re-render core charts from an existing run (reads chart_bundle/v1/core_bundle.json).
# Rebuild the bundle from saved pickles if missing: python -m burst.core bundle <run_dir>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-${SCRIPT_DIR}/.venv/bin/python}"

RUN_DIR="${1:?usage: $0 <run_dir> [extra args for: python -m burst.core charts ...]}"
shift
exec "${PYTHON}" -m burst.core charts "${RUN_DIR}" "$@"
