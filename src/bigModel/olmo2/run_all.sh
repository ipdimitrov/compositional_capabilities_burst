#!/bin/bash
set -e

# ==========================================================================
# OLMo-2-1B Two-Stage Concentration Experiment
#
# Stage 1 (sweep):  Fine-tune at varying code concentrations
#   c = fraction of code data in batch. c=1.0 → pure code, c=0.5 → half/half
#   Each (c, lr, seed) gets its own stage 1 run + checkpoint.
#
# Stage 2 (fixed):  100% pretraining data for all runs → measure forgetting
#   How fast does each stage-1 model forget code when retrained on pretrain?
#
# Usage:
#   bash olmo2/run_all.sh              # full sweep
#   bash olmo2/run_all.sh --quick      # quick test (2 concentrations, 1 lr, 1 seed)
#   bash olmo2/run_all.sh --eval-only  # just run benchmark evals + plots
#   bash olmo2/run_all.sh --stage1-only # only run stage 1
#   bash olmo2/run_all.sh --stage2-only # only run stage 2 (stage 1 must exist)
# ==========================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$SCRIPT_DIR/venv"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ALLOW_CODE_EVAL=1

# --- HuggingFace token (required for gated models/datasets) ---
if [ -z "${HF_TOKEN:-}" ]; then
  echo "Set HF_TOKEN in the environment before running (required for gated models)." >&2
  exit 1
fi
export HF_TOKEN

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

echo "=== Setting up environment ==="
if [ ! -d "$VENV_DIR/bin" ]; then
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip

    # Core training dependencies
    pip install torch transformers datasets accelerate tqdm numpy pandas matplotlib scikit-learn wandb

    # Benchmark evaluation (optional — training works without these)
    pip install lm-eval || echo "WARNING: lm-eval install failed, benchmark evals will be skipped"
    pip install "git+https://github.com/bigcode-project/bigcode-evaluation-harness.git" \
        || echo "WARNING: bigcode-eval-harness install failed, HumanEval/MBPP evals will be skipped"
else
    source "$VENV_DIR/bin/activate"
    echo "venv already exists, skipping install"
fi

cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------

QUICK=false
EVAL_ONLY=false
STAGE1_ONLY=false
STAGE2_ONLY=false
SKIP_CODE_EVAL=false

for arg in "$@"; do
    case $arg in
        --quick) QUICK=true ;;
        --eval-only) EVAL_ONLY=true ;;
        --stage1-only) STAGE1_ONLY=true ;;
        --stage2-only) STAGE2_ONLY=true ;;
        --skip-code-eval) SKIP_CODE_EVAL=true ;;
    esac
done

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

if [ "$QUICK" = true ]; then
    CONCENTRATIONS="0.5 1.0"
    LRS="5e-5"
    SEEDS="42"
    S1_STEPS=200
    S2_STEPS=200
    MEASURE_EVERY=20
    CHECKPOINT_EVERY=999999
    PRETRAIN_DOCS=5000
    CODE_DOCS=10000
    echo "=== QUICK MODE ==="
else
    CONCENTRATIONS="0.3 0.5 0.7 0.9 1.0"
    LRS="1e-5"
    SEEDS="42"
    S1_STEPS=500
    S2_STEPS=200
    MEASURE_EVERY=50
    CHECKPOINT_EVERY=999999
    PRETRAIN_DOCS=5000
    CODE_DOCS=10000
    echo "=== FULL SWEEP ==="
fi

# Stage 2 LR (fixed for all runs — the same forgetting pressure)
S2_LR="5e-5"

# Count runs
TOTAL_RUNS=0
for c in $CONCENTRATIONS; do
    for lr in $LRS; do
        for seed in $SEEDS; do
            TOTAL_RUNS=$((TOTAL_RUNS + 1))
        done
    done
done
echo "  Stage 1: ${TOTAL_RUNS} runs (varying concentration) x ${S1_STEPS} steps"
echo "  Stage 2: ${TOTAL_RUNS} runs (100% pretrain) x ${S2_STEPS} steps"

# ---------------------------------------------------------------------------
# Step 1: Stage 1 — Fine-tune at varying concentrations
# ---------------------------------------------------------------------------

if [ "$EVAL_ONLY" = false ] && [ "$STAGE2_ONLY" = false ]; then
    RUN_NUM=0

    for c in $CONCENTRATIONS; do
        for lr in $LRS; do
            for seed in $SEEDS; do
                RUN_NUM=$((RUN_NUM + 1))
                S1_NAME="stage1_c${c}_lr${lr}_s${seed}"

                echo ""
                echo "=================================================================="
                echo "Stage 1 — Run ${RUN_NUM}/${TOTAL_RUNS}: ${S1_NAME}"
                echo "  ${c} code + $(echo "1 - $c" | bc) pretrain"
                echo "=================================================================="

                # Skip if already completed
                if [ -f "olmo2/results/${S1_NAME}/stage1_complete" ]; then
                    echo "  Already completed, skipping."
                    continue
                fi

                python olmo2/train_concentration.py \
                    --stage 1 \
                    --concentration "$c" \
                    --lr "$lr" \
                    --seed "$seed" \
                    --s1_steps "$S1_STEPS" \
                    --batch_size 4 \
                    --grad_accum_steps 16 \
                    --measure_every "$MEASURE_EVERY" \
                    --checkpoint_every 999999 \
                    --pretrain_train_docs "$PRETRAIN_DOCS" \
                    --code_train_docs "$CODE_DOCS"

            done
        done
    done
fi

# ---------------------------------------------------------------------------
# Step 2: Stage 2 — 100% pretraining (forgetting) for each stage-1 model
# ---------------------------------------------------------------------------

if [ "$EVAL_ONLY" = false ] && [ "$STAGE1_ONLY" = false ]; then
    RUN_NUM=0

    for c in $CONCENTRATIONS; do
        for lr in $LRS; do
            for seed in $SEEDS; do
                RUN_NUM=$((RUN_NUM + 1))
                S2_NAME="stage2_c${c}_lr${lr}_s2lr${S2_LR}_s${seed}"

                echo ""
                echo "=================================================================="
                echo "Stage 2 — Run ${RUN_NUM}/${TOTAL_RUNS}: ${S2_NAME}"
                echo "  100% pretrain, forgetting model from c=${c}"
                echo "=================================================================="

                # Skip if already completed
                if [ -f "olmo2/results/${S2_NAME}/scalar_metrics.csv" ]; then
                    LAST_STEP=$(tail -1 "olmo2/results/${S2_NAME}/scalar_metrics.csv" | cut -d',' -f1)
                    if [ "$LAST_STEP" -ge "$S2_STEPS" ] 2>/dev/null; then
                        echo "  Already completed (last step=$LAST_STEP), skipping."
                        continue
                    fi
                fi

                python olmo2/train_concentration.py \
                    --stage 2 \
                    --concentration "$c" \
                    --lr "$lr" \
                    --s2_lr "$S2_LR" \
                    --seed "$seed" \
                    --s2_steps "$S2_STEPS" \
                    --batch_size 4 \
                    --grad_accum_steps 16 \
                    --measure_every "$MEASURE_EVERY" \
                    --checkpoint_every 999999 \
                    --pretrain_train_docs "$PRETRAIN_DOCS" \
                    --code_train_docs "$CODE_DOCS"

            done
        done
    done
fi

# ---------------------------------------------------------------------------
# Step 3: Benchmark evaluation on stage-2 checkpoints
# ---------------------------------------------------------------------------

echo ""
echo "=================================================================="
echo "Step 3: Benchmark evaluation"
echo "=================================================================="

for run_dir in olmo2/results/stage2_c*; do
    if [ -d "$run_dir/checkpoints" ]; then
        if [ -f "$run_dir/all_benchmark_results.json" ]; then
            echo "  $run_dir: benchmarks already done, skipping."
            continue
        fi
        echo "  Evaluating: $run_dir"
        EVAL_ARGS="--run_dir $run_dir"
        if [ "$SKIP_CODE_EVAL" = true ]; then
            EVAL_ARGS="$EVAL_ARGS --skip_code"
        fi
        python olmo2/eval_benchmarks.py $EVAL_ARGS || echo "  Eval failed for $run_dir"
    fi
done

# ---------------------------------------------------------------------------
# Step 4: Generate plots
# ---------------------------------------------------------------------------

echo ""
echo "=================================================================="
echo "Step 4: Generating plots"
echo "=================================================================="

python olmo2/plot_results.py

echo ""
echo "=== All done! Results in olmo2/results/ ==="
