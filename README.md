# Burst Pipeline (Canonical CLI)

This repo contains the synthetic compositional task code and the burst-training analysis pipeline.

The canonical entrypoint is:

```bash
uv run python -m burst.core <mode> ...
```

## Why this CLI exists

- One command surface for train + metrics + chart pipeline.
- Explicit run modes with no hidden behavior.
- Reproducibility contract on every run:
  - deterministic toggle
  - explicit seed
  - machine-readable run manifest at `results/repro_manifest.json`

## Setup

```bash
uv sync
```

Then run everything with `uv run python ...`.

## Ruff (linting)

Run lint checks:

```bash
uv run ruff check .
```

Auto-fix safe issues:

```bash
uv run ruff check . --fix
```

Format:

```bash
uv run ruff format .
```

Current `pyproject.toml` enables only `E` and `F` rules, so magic-number rules are not active by default.
If you want to explicitly scan for magic values:

```bash
uv run ruff check . --select PLR2004
```

## Canonical Modes

### 1) Train

```bash
uv run python -m burst.core train \
  --depth 3 \
  --burst-pos 3 \
  --burst-mode current \
  --n-seeds 10 \
  --seed 1337 \
  --deterministic
```

This creates a new run directory in `data/` with `logs/` and `results/`.

### 2) Gradient Metrics

```bash
uv run python -m burst.core gradients data/<run_dir> \
  --grad-sim-batch-size 2048 \
  --n-workers 8 \
  --seed 1337 \
  --deterministic
```

### 3) Build Bundle (Stage 1)

```bash
uv run python -m burst.core bundle data/<run_dir> --seed 1337 --deterministic
```

### 4) Render Charts (Stage 2)

```bash
uv run python -m burst.core charts data/<run_dir> --seed 1337 --deterministic
```

### 5) Full Core Analysis Pipeline

```bash
uv run python -m burst.core pipeline data/<run_dir> --seed 1337 --deterministic
```

`pipeline` = `bundle` + `charts`.

## Reproducibility Contract

Every canonical mode records:

- mode
- seed
- deterministic setting
- runtime metadata (python/torch/cuda/gpu/git sha)
- CLI args

Output path:

- `data/<run_dir>/results/repro_manifest.json`

You can disable strict deterministic execution when needed:

```bash
uv run python -m burst.core pipeline data/<run_dir> --seed 1337 --no-deterministic
```

## Code Layout

- `burst/core/`: production pipeline code
  - `train/`: training orchestration + workers
  - `metrics/`: gradient metric computation
  - `bundle.py`: stage-1 chart data artifacts
  - `charts/`: stage-2 chart rendering
  - `cli.py`: canonical discriminated CLI
  - `repro.py`: deterministic + manifest contract
- `burst/dev/`: heavy/experimental analyses and appendix tooling
- `simple/`: separate compact experimentation codepath

## Notes

- Prefer canonical CLI modes over direct file execution.
- If a script is in `burst/dev/`, treat it as optional and non-core.
