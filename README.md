# A loss curvature account of fine-tuning fragility

[Link to paper](https://openreview.net/forum?id=LhT1YHmyCN)

This repo studies how small transformers acquire and retain **compositional capabilities** when the training distribution shifts abruptly (a "burst" of a held-out class) and then reverts.

A nanoGPT-style transformer is trained on **pure-bijection composition tasks** in three phases:

1. **Pre-burst** — train on all task classes *except* one special class.
2. **Burst** — oversample the special class at a configurable concentration (100%, 50%, 10%, …).
3. **Reversion** — return to the original distribution (no special class).

The core question: *how does burst concentration affect learning speed, retention, and interference with previously-learned capabilities?*

## What the pipeline produces

| Stage | Command | Outputs |
|-------|---------|---------|
| **Train** | `burst.core train` | Per-seed checkpoints, accuracy/loss logs, task distribution stats → `data/<run>/logs/` |
| **Gradient metrics** | `burst.core gradients` | Cosine similarity & norm JSONs per schedule → `data/<run>/results/grad_cosine_sim/` |
| **Bundle** | `burst.core bundle` | Single `core_bundle.json` aggregating all metrics → `data/<run>/results/chart_bundle/v1/` |
| **Charts** | `burst.core charts` | Publication-quality PNGs → `data/<run>/results/core_charts/` |

Charts include: accuracy/loss overlays across schedules, AUC bars, reversion zoom, gradient cosine similarity, gradient norms, representation drift, per-schedule breakdowns, and a summary table.

## Setup

```bash
uv sync
```

Then run everything with `uv run python ...`.

## CLI

The single entrypoint is:

```bash
uv run python -m burst.core <mode> [options]
```

### Train

```bash
uv run python -m burst.core train \
  --depth 3 \
  --burst-pos 3 \
  --burst-mode current \
  --n-seeds 10 \
  --seed 1337 \
  --deterministic
```

Creates a timestamped run directory under `data/` with logs and checkpoints.

### Gradient metrics

```bash
uv run python -m burst.core gradients data/<run_dir> \
  --grad-sim-batch-size 2048 \
  --n-workers 8 \
  --seed 1337 \
  --deterministic
```

### Bundle → Charts (full analysis)

```bash
uv run python -m burst.core pipeline data/<run_dir> --seed 1337 --deterministic
```

Or run the stages separately:

```bash
uv run python -m burst.core bundle data/<run_dir> --seed 1337 --deterministic
uv run python -m burst.core charts  data/<run_dir> --seed 1337 --deterministic
```

## Burst modes

| Mode | Burst length | Batch size | Invariant |
|------|-------------|------------|-----------|
| `current` | Scales inversely with concentration | Fixed | Total special-class examples seen |
| `constant_steps` | Fixed | Fixed | Number of training steps |
| `scaled_batch` | Fixed | Scales inversely with concentration | Special examples per step |

## Run directory layout

```
data/<run_dir>/
├── logs/
│   ├── pretrain_ckpt.pt          # shared pretrain checkpoint
│   ├── pretrain_log.pkl
│   ├── all_results.pkl           # aggregated per-job results
│   ├── <schedule>_s<seed>.pkl    # individual run logs
│   ├── checkpoints/              # per-step model snapshots
│   └── task_distributions/       # class count stats per phase
└── results/
    ├── config.json               # full experiment metadata
    ├── repro_manifest.json       # reproducibility record
    ├── grad_cosine_sim/          # gradient metric JSONs
    ├── chart_bundle/v1/          # bundled data for plotting
    └── core_charts/              # rendered PNGs
```

## Reproducibility

Every CLI mode records a manifest at `results/repro_manifest.json` containing: mode, seed, deterministic flag, Python/torch/CUDA/GPU info, git SHA, and full CLI args.

Disable strict determinism when needed:

```bash
uv run python -m burst.core pipeline data/<run_dir> --seed 1337 --no-deterministic
```

## Linting

```bash
uv run ruff check .          # lint
uv run ruff check . --fix    # auto-fix
uv run ruff format .         # format
```

## Directory structure

```
├── burst/                 # Main burst experiment (production CLI: python -m burst.core)
│   ├── core/              # Production train → gradients → bundle → charts
│   ├── dev/               # Experimental analyses (probes, basins, sharpness, …)
│   └── notebook/          # Notebook-oriented simple-pipeline helpers
├── simple/                # Lightweight burst + Adam/Taylor / manual-Adam variants
├── consolidated/          # Packaged sweeps + configs for the simple pipeline
├── synthetic/             # Bijection tasks and prompt generation
├── net/                   # nanoGPT / LSTM architectures + training runner
├── config/                # YAML configs for data gen / train / eval
├── analysis/              # Helpers for loading evaluation results
├── scripts/               # One-off utilities (run organization, probe regimes)
├── notebooks/             # Sweeps, curvature, Adam/Taylor, correlation notebooks
├── notes/                 # Design discussion notes
├── tests/                 # Unit tests (burst schedules, Adam/Taylor helpers)
├── archive/               # Older alternate layouts
│   ├── burst_probes_era/  # Flat pre-core burst/*.py analysis stack
│   └── burst_simple_snapshot/  # Earlier simple-pipeline layout snapshot
├── src/
│   ├── plot_lib/          # Shared publication plot styling + curvature charts
│   ├── PCFG/              # PCFG compositional / fine-tuning experiments
│   ├── PCFG_ext/          # PCFG variant with extra plot scripts and saved results
│   ├── pythia/            # Pythia fine-tune / continued-pretrain forgetting
│   └── bigModel/          # Large-model forgetting (Gemma / OLMo-2)
├── 01_generate_data.py    # Generate synthetic train data
├── 02_train.py            # Train Transformer / LSTM
├── 03_evaluate_i.py       # Evaluate in-order compositions
├── 03_evaluate_o.py       # Evaluate out-of-order compositions
└── description.md         # Long-form burst experiment specification
```

### What each folder does

| Path | Role |
|------|------|
| **`burst/core/`** | Production CLI: train, gradient metrics, bundle, charts (`python -m burst.core …`). |
| **`burst/dev/`** | Experimental analyses (probes, EWC, basins, sharpness dynamics, presentation charts). |
| **`burst/notebook/`** | Helpers used by the notebook-style simple pipeline. |
| **`simple/`** | Skimmable pretrain → burst → forget pipeline, plus Adam/Taylor and manual-Adam variants. |
| **`consolidated/`** | Self-contained package: YAML sweep configs + `run_experiment.py`. |
| **`synthetic/`** | Pure-bijection composition documents for root scripts and the burst task. |
| **`net/`** | Model definitions (`nanogpt.py`, `lstm.py`) and the paper training runner. |
| **`config/`** | Configs for `01_generate_data.py` / `02_train.py` / `03_evaluate_*.py`. |
| **`analysis/`** | Post-hoc loaders for evaluation outputs. |
| **`scripts/`** | Ad-hoc tooling (organize run dirs, next-token probe regimes). |
| **`notebooks/`** | Interactive sweeps: concentration, curvature, Adam forget, correlation, etc. |
| **`notes/`** | Written discussion of burst forgetting design choices. |
| **`tests/`** | Tests for schedules and Adam/Taylor helpers. |
| **`archive/`** | Older alternate layouts. Prefer live paths above. |
| **`src/plot_lib/`** | Shared matplotlib style + curvature chart builders. |
| **`src/PCFG/`** | PCFG compositional / fine-tuning experiments. |
| **`src/PCFG_ext/`** | PCFG variant with extra plot scripts, `style.py`, and saved results. |
| **`src/pythia/`** | HuggingFace Pythia catastrophic-forgetting + Taylor/curvature notebooks. |
| **`src/bigModel/`** | Large-model concentration / cos-sim / weight-drift experiments on Gemma and OLMo-2. |

Root scripts for the synthetic composition setup:

```bash
uv run python 01_generate_data.py
uv run python 02_train.py
uv run python 03_evaluate_i.py
```

### `burst/` internals

- `burst/config.py` — schedules, hyperparameters, display constants
- `burst/core/` — production pipeline
  - `train/` — training orchestration + per-job workers
  - `metrics/` — gradient similarity and norm computation
  - `bundle.py` — stage-1 data aggregation
  - `charts/` — stage-2 matplotlib rendering
  - `cli.py` — CLI dispatch
  - `repro.py` — deterministic seeding + manifest
- `burst/dev/` — optional experimental analyses (not required for `burst.core`)
- `net/` — nanoGPT model implementation
- `synthetic/` — task and data generation
