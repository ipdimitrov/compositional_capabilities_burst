#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$SCRIPT_DIR/venv"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "Set HF_TOKEN in the environment before running (required for gated models)." >&2
  exit 1
fi
export HF_TOKEN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Setting up venv ==="
if [ ! -d "$VENV_DIR/bin" ]; then
  python3 -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip
  pip install torch transformers datasets accelerate tqdm numpy pandas matplotlib scikit-learn
else
  source "$VENV_DIR/bin/activate"
  echo "venv already exists, skipping install"
fi

cd "$PROJECT_DIR"

echo ""
echo "=== Run 1/5: ratio=1.0 (Stage 1 + Stage 2) ==="
python gemma/run_sweep.py --new_data_ratio 1.0

echo ""
echo "=== Run 2/5: ratio=0.8 ==="
python gemma/run_sweep.py --new_data_ratio 0.8 --skip_stage1

echo ""
echo "=== Run 3/5: ratio=0.6 ==="
python gemma/run_sweep.py --new_data_ratio 0.6 --skip_stage1

echo ""
echo "=== Run 4/5: ratio=0.4 ==="
python gemma/run_sweep.py --new_data_ratio 0.4 --skip_stage1

echo ""
echo "=== Run 5/5: ratio=0.2 ==="
python gemma/run_sweep.py --new_data_ratio 0.2 --skip_stage1

echo ""
echo "=== Generating plots ==="
python gemma/plot_results.py

echo ""
echo "=== All done! Results in gemma/results_sweep/ ==="
