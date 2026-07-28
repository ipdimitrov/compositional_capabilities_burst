## PCFG Experiments

This folder contains a synthetic sequence-learning setup built around a probabilistic context-free grammar (PCFG), plus pretraining/finetuning scripts and analysis utilities for studying compositional behavior, transfer, and interference. It is mainly the same as the "Mechanistically analyzing the effects of fine-tuning on procedurally defined tasks" paper (https://arxiv.org/abs/2311.12786)

At a high level, the code:

1. Generates PCFG strings over terminals `a`, `b`, `c`.
2. Converts each string into supervised examples for multiple symbolic tasks.
3. Trains a GPT-style model on mixes of those tasks.
4. Runs correlation and concentration sweeps.
5. Logs per-task metrics and gradient interaction diagnostics.


## Main Components

- `pcfg_gen.py`
	- Grammar and sequence generation (`PCFGGenerator`).
	- Task definitions and formatting utilities.
	- Correlated/uncorrelated pool builder (`build_pools`).

- `config.py`
	- Canonical configuration for model, tasks, data, pool sizes, and experiment sweeps.

- `train_help.py`
	- Training loop (`train`), online batch sampling (`sample_batch`), evaluation dataset builders, and summary helpers.

- `run_pretrain_fast.py`
	- Pretrains a base model at correlation `0.0` (uncorrelated training stream).

- `run_finetune_fast_w_metrics.py`
	- Loads pretrained checkpoints and runs finetune + reverse phases over correlation/concentration grids.
	- Adds gradient projection and layerwise drift metrics.

- `gradient_metrics.py`
	- Per-layer gradient projection and model drift utilities.

- `plot_helpers.py`, `plot_metrics.py`
	- Result loading and plotting helpers, with `plot_metrics.py` used to generate the figures.


## Task Set

Tasks are declared in `config.py` and registered by `config_utils.py`.

The current setup has 13 tasks:

1. `count_a`, `count_b`, `count_c`
2. `count_aa`, `count_bb`, `count_cc`
3. `index_a`, `index_b`, `index_c`
4. `index_aa`, `index_bb`, `index_cc`
5. `token_at_40`

They fall into three families:

- Count tasks
	- Count token or substring occurrences in the last window of the string (default window = 40).

- Index tasks
	- Return index-from-end of the Nth occurrence of a token or substring (default occurrence = 6), with `-1` when absent.

- Token-at-index task
	- Return the token a fixed distance from end-of-string.


## Pretrain / Finetune / Reverse Phases

### Pretrain

- Script: `run_pretrain_fast.py`
- Uses task set `task_sets["pretrain"]`.
- Trains with `correlation = 0.0` (samples only from uncorrelated pool).
- Saves checkpoint and history.

### Finetune

- Script: `run_finetune_fast_w_metrics.py`
- For each configured correlation and concentration:
	- Loads the pretrained model.
	- Trains on `task_sets["finetune"]` with concentration controlling weight on `count_a` vs other tasks.
	- Uses the selected correlation value for data sampling.

### Reverse

- In the same script after finetune.
- Trains back on the pretrain task mix.
- Uses `correlation = 0.0` to test recovery/forgetting dynamics under uncorrelated data.


## How Correlation Is Enforced

Correlation is enforced by **data source mixing**, not by changing grammar rules.

### Step 1: Build two pools

Implemented in `build_pools` (`pcfg_gen.py`).

- Correlated pool
	- A generated chunk is accepted if, in its trailing window (default 40 chars):
		- `count('a') == count('b') + 1`

- Uncorrelated pool
	- Natural PCFG chunks with no filtering constraint.

Both are filled in one pass:

1. Generate a chunk.
2. Add it to uncorrelated pool until that pool is full.
3. If the trailing-window condition holds, also add it to correlated pool.
4. Continue until correlated pool reaches target size.
5. If needed, top up uncorrelated pool afterward.

So the code explicitly creates **two reusable reservoirs** of strings:

- `pools['correlated']`
- `pools['uncorrelated']`


### Step 2: Draw per example from one of the pools

Implemented in `sample_batch` (`train_help.py`).

For each training example:

1. Draw `u ~ Uniform(0, 1)`.
2. If `u < correlation` and `correlation > 0`, sample string from `pools['correlated']`.
3. Otherwise sample from `pools['uncorrelated']`.
4. Sample a task (weighted) and build the final supervised example.

This means `correlation` is directly the probability of taking a string from the correlated reservoir.

- `correlation = 0.0`: all examples from uncorrelated pool.
- `correlation = 1.0`: all examples from correlated pool.
- Intermediate values: Bernoulli mixture of the two pools per example.


## Evaluation Splits

`build_eval_datasets` creates fixed splits from both pools, including:

- `count_a_corr` / `count_a_uncorr`
- `count_b_corr` / `count_b_uncorr`
- `all_other_corr` / `all_other_uncorr`

This lets you separate performance on correlated vs uncorrelated inputs for target tasks and for the aggregate of non-target tasks.


## Running

Typical order:

1. Pretrain
	 - `python run_pretrain_fast.py`
2. Finetune + reverse sweep with metrics
	 - `python run_finetune_fast_w_metrics.py`

Results are written under `results/` (`models/`, `histories/`, summaries).
