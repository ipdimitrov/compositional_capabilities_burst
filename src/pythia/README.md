# Catastrophic Forgetting Experiment

Measures catastrophic forgetting in Pythia models fine-tuned on narrow-domain data, then continued-pretrained on general data.

## Quick Start

```bash
pip install -e .              # or: pip install -e ".[notebook]" for Jupyter
python run_experiment.py --preset normal
```

## Presets

**Training presets** (`--preset`):

| Preset | FT Steps | CPT Steps |
|--------|----------|-----------|
| quick  | 200      | 400       |
| normal | 500      | 1000      |
| full   | 1000     | 2000      |
| deep   | 8000     | 3000      |
| deepest| 20000    | 4000      |

**Model presets** (`--model`):

| Model | Name | FT LR | Batch Size | LR Min Ratio | Budget Mode |
|-------|------|-------|------------|-------------|-------------|
| 70m   | pythia-70m-deduped | 5e-5 | 20 | 0.0 (standard cosine) | steps |
| 1b    | pythia-1b-deduped  | 1e-4 | 8  | 0.1 (10% floor) | volume |

**Domain presets** (`--domain`):

| Domain | Dataset | Text Field |
|--------|---------|------------|
| chemistry | antoinebcx/smiles-molecules-chembl | smiles |
| music | sander-wood/irishman | abc notation |
| biomedical | ccdv/pubmed-summarization | article |

## FT Budget Modes

- `steps` (default for 70m): every burst level runs exactly `ft_steps`. Low-burst runs see less domain data.
- `volume` (default for 1b): scales `ft_steps` by `1/burst_level` so every burst sees the same domain volume. Plots are right-aligned at the FT/CPT boundary.

Override with `--ft_budget_mode volume|steps`.

## Examples

```bash
# 70m, chemistry, normal preset
python run_experiment.py

# 1b model, deep preset, music domain, with gradient analysis
python run_experiment.py --model 1b --preset deep --domain music --grads

# Biomedical with memory cap (large documents)
python run_experiment.py --domain biomedical --preset deep --max_train_chunks 20000

# Custom burst levels
python run_experiment.py --burst_levels 1.0 0.9 0.5 0.25

# Volume budget mode on 70m
python run_experiment.py --ft_budget_mode volume --preset full

# Gradient analysis with custom frequency
python run_experiment.py --grads --grad_every 200

# Regenerate plots from a previous run
python run_experiment.py --plots_only --results_dir results/latest
```

## Results

Each run saves to `results/{timestamp}_{preset}_{model}_{domain}/`. A `results/latest` symlink points to the most recent run.

Outputs:
- `config.json` — full config
- `summary.txt` — concise copy-paste-friendly summary
- `metrics.json` — eval metrics (perplexity, loss) at each eval step
- `loss_history.json` — per-step training loss
- `grad_metrics.json` — gradient cosine similarity and norms (if `--grads`)
- `models/burst_X_XX/post_finetune/` and `post_cpt/` — saved checkpoints
- PNG plots

## Notebook

`notebook.ipynb` — select a results directory and display all plots. Includes curvature coupling analysis cells for studying second-order forgetting effects.
