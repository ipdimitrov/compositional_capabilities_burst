# Burst Forgetting Experiments

Consolidated package for the compositional capabilities burst forgetting study.
Runs a unified sweep over burst concentration, correlation, learning rate, and
seeds, then generates plots.

## Layout

```
consolidated/
├── pyproject.toml
├── run_experiment.py           # entry point: train + plot
├── configs/
│   ├── burst_simple.yaml
│   ├── lr_concentration_sweep.yaml
│   └── correlation_concentration_sweep.yaml
└── burst/
    ├── data.py                 # bijection task + data generation
    ├── model.py                # nanoGPT wrapper + training step
    ├── nanogpt.py              # model architecture
    ├── pretrain.py             # phase 1
    ├── finetune.py             # phase 2 (burst)
    ├── forget.py               # phase 3 (reversion)
    ├── interp.py               # interpretability metrics
    ├── sweep.py                # unified sweep runner
    └── plot.py                 # all plotting functions
```

## Install

```bash
cd consolidated
uv sync
```

## Run

Three reference configs reproduce the notebook experiments:

```bash
# Simple burst experiment (single correlation, sweep concentration)
python run_experiment.py --config configs/burst_simple.yaml

# Learning rate × concentration sweep
python run_experiment.py --config configs/lr_concentration_sweep.yaml

# Correlation × concentration sweep
python run_experiment.py --config configs/correlation_concentration_sweep.yaml
```

Each run creates `data/sweep_{timestamp}/` containing:
- `config.json` — full config used
- `seed_{s}/corr{c}_ftlr{lr}_frac{f}_{ft|fg}.pkl` — individual results
- `plots/` — auto-generated plots

## Useful flags

```bash
# Override config values on the command line
python run_experiment.py --config configs/burst_simple.yaml \
    --override seeds=[42,123] ft_steps=2000

# Resume an interrupted run (skips existing result files)
python run_experiment.py --config configs/correlation_concentration_sweep.yaml \
    --resume --out data/sweep_20260412_200000

# Just plot (uses existing results)
python run_experiment.py --plot-only --out data/sweep_20260412_200000

# Train but skip plotting
python run_experiment.py --config configs/burst_simple.yaml --no-plot
```

## Config schema

Every sweep is described by the same `SweepConfig` dataclass. Key fields:

| Field | Default | Description |
|---|---|---|
| `n_a` | 6 | Functions per slot during pretraining |
| `n_burst` | 6 | Burst functions at the burst slot |
| `depth` | 3 | Composition depth |
| `burst_pos` | 3 | Which slot gets burst functions |
| `n_docs` | auto | Training docs per task (auto-scales with `n_a`) |
| `n_eval` | auto | Eval docs per task |
| `fracs` | `[1.0, …, 0.3]` | Burst concentration levels |
| `correlations` | `[0.0]` | Burst correlation levels (0=novel, 1=copied) |
| `ft_lrs` | `[1e-4]` | Finetune learning rates |
| `seeds` | `[42, 123, 777]` | Random seeds |
| `ft_steps` | 1500 | Finetune training steps |
| `fg_steps` | 1200 | Forget training steps |
| `fg_lr` | `1e-4` | Forget learning rate (fixed) |
| `batch_size` | 512 | Batch size |
| `eval_every` | 100 | Eval every N steps |
| `workers` | 6 | Parallel workers |

To run a custom sweep, any of these axes can be a single value or a list.
The plotter auto-detects which axes are swept and generates the appropriate
heatmaps / line plots / scatters.

## Running overnight in tmux

```bash
tmux new -s sweep
cd consolidated
python run_experiment.py --config configs/correlation_concentration_sweep.yaml \
    --out data/overnight_sweep \
    2>&1 | tee data/overnight_sweep.log
# Ctrl+B, D to detach
```

## Three experiment types

### 1. Burst simple (`burst_simple.yaml`)
Single (n_a=4, correlation=0) setup, sweeps concentration. Plots training
curves + metrics vs concentration.

### 2. LR × concentration (`lr_concentration_sweep.yaml`)
Sweeps `ft_lrs × fracs`. Plots heatmaps with LR on Y axis and concentration
on X axis, plus lines vs concentration colored by LR.

### 3. Correlation × concentration (`correlation_concentration_sweep.yaml`)
Sweeps `correlations × fracs` with `n_burst=6`. Plots heatmaps with
correlation on Y axis and concentration on X axis, plus the novel-vs-copied
burst accuracy breakdown.
