# Burst: Compositional Capability Acquisition Under Distribution Shift

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

## Code layout

- `burst/config.py` — schedules, hyperparameters, display constants (single source of truth)
- `burst/core/` — production pipeline
  - `train/` — training orchestration + per-job workers
  - `metrics/` — gradient similarity and norm computation
  - `bundle.py` — stage-1 data aggregation
  - `charts/` — stage-2 matplotlib rendering
  - `cli.py` — CLI dispatch
  - `repro.py` — deterministic seeding + manifest
- `burst/dev/` — experimental analyses (non-core, optional)
- `net/` — nanoGPT model implementation
- `synthetic/` — task and data generation
