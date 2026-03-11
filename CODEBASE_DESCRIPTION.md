# Codebase Description: Compositional Capabilities Burst Experiment

## High-Level Research Question

This codebase studies **compositional generalisation and catastrophic forgetting** in small Transformers. The central question is: when a model briefly encounters a novel compositional task (a "burst"), how quickly does it acquire that capability, and how quickly does it forget it when the task is removed? The experiment varies *how* the burst data is presented (the "schedule") and measures acquisition speed, peak performance, and forgetting speed.

---

## The Task: Depth-N Bijection Chains

### What a bijection is

A bijection here is a lookup table over an alphabet of size `n_alphabets` (default 10). Each bijection is a random permutation of `[0, 1, ..., n_alphabets-1]`. Applying bijection `F` to a token `x` simply returns `F[x]`.

### What a depth-N chain is

A depth-N chain is a composition of N bijections applied in sequence to an input sequence of length `seq_len` (default 6). Given functions `[F_N, F_{N-1}, ..., F_1]` and input `inp`, the chain computes:

```
after_F1  = F_1(inp)
after_F2  = F_2(after_F1)
...
after_FN  = F_N(after_F_{N-1})
```

### Token format

Each training document is a flat token sequence encoding the entire chain computation:

```
S  [F_N ... F_1]  <space>  [inp]  <space>  [after_F1]  <space>  ...  <space>  [after_FN]
```

- `S` is a start token.
- `[F_N ... F_1]` are `depth` function-identifier tokens (one per slot, written outermost-first).
- `<space>` is a separator token.
- `[inp]` is the `seq_len`-length input sequence.
- Each `[after_Fi]` is the `seq_len`-length intermediate result after applying the first `i` functions.

The model is trained autoregressively on this sequence (next-token prediction). At evaluation time, the model is given the prompt up to and including the input sequence, then must generate all intermediate and final outputs token by token.

### Vocabulary

Tokens are:
- `X0 ... X{n_alphabets-1}`: alphabet symbols (values)
- `F0 ... F{n_a*depth+1}`: function identifiers (`F0` = identity, `F1..F{n_a*depth}` = position-specific background functions, `F{n_a*depth+1}` = `b*` the burst function)
- `<space>`, `<PAD>`, `S`: special tokens

---

## The Two Classes of Tasks

### Other class (background / foundation)

Each depth position `p` (1-indexed) has its own dedicated set of `n_a` (default 3) bijections that are never shared with other positions. Position `p` uses bijection indices `(p-1)*n_a + 1` through `p*n_a`. The "other" tasks are all `n_a^depth` possible depth-N chains where each position draws from its own function set. For depth=3 and n_a=3 that is 27 tasks. These are the tasks the model trains on throughout the entire experiment.

### Burst class (the novel capability)

One additional bijection `b*` (index `n_a*depth + 1`) is introduced. The burst tasks are all `n_a^{depth-1}` chains where `b*` occupies a specific position `burst_pos` (1-indexed, 1 = outermost, depth = innermost) and all other positions range over their own position-specific background functions. For depth=3, n_a=3, burst_pos=3, that is 9 burst tasks. The burst class is entirely novel — the model has never seen `b*` before the burst phase. Crucially, the same function can never appear at two different positions, matching the paper's non-overlapping design.

---

## The Four Training Phases

### Phase 0: Pre-burst pretraining (P steps, default 420)

Before any burst data is introduced, one shared model is trained on all-but-special (other-class only) for `pre_burst_steps` steps. This pretrained checkpoint is shared across all seeds for a given schedule, giving every seed an identical starting point.

### Phase 1: Burst (T steps, default `BURST_BASE_STEPS = 140`)

Starting from the shared pretrain checkpoint, each seed trains for `T` burst-phase steps. Each step, a batch of size `batch_size` (default 128) is sampled as a mixture of other-class and burst-class documents. The number of burst documents per step is controlled by the **schedule** (see below). The burst phase length varies inversely with burst concentration so all schedules see the same total number of special-class examples.

### Phase 2: Reversion (U steps, default 420)

After the burst phase, the burst class is completely removed. For all `U` reversion steps, only other-class data is used. This measures how quickly the model forgets the burst capability.

### Learning rate

A single cosine decay schedule with linear warmup runs across all `P + T + U` steps. Warmup lasts `warmup_iters` (default 50) steps, then decays from `lr` (default 3e-4) to `min_lr` (default 6e-5).

---

## The Schedules

The schedule determines how much burst-class data appears in each batch during the `T` burst-phase steps. The key parameter is `p_target` (default 0.25). The burst phase length scales inversely with concentration so all schedules deliver the same total burst-class examples: `burst_steps * frac = BURST_BASE_STEPS * 1.0`.

`BURST_FRACTIONS = [100, 98, 95, 90, 85, 75, 50, 25]` — `burst_10` has been removed. All schedules now use binomial sampling throughout the full burst phase (rather than a fixed window).

| Schedule | Description |
|---|---|
| `burst_100` | 100% burst-class every step for `BURST_BASE_STEPS` steps. |
| `burst_98` | 98% burst-class (binomial) for `BURST_BASE_STEPS / 0.98` steps. |
| `burst_95` | 95% burst-class (binomial) for `BURST_BASE_STEPS / 0.95` steps. |
| `burst_90` | 90% burst-class (binomial) for `BURST_BASE_STEPS / 0.90` steps. |
| `burst_85` | 85% burst-class (binomial) for `BURST_BASE_STEPS / 0.85` steps. |
| `burst_75` | 75% burst-class (binomial) for `BURST_BASE_STEPS / 0.75` steps. |
| `burst_50` | 50% burst-class (binomial) for `BURST_BASE_STEPS / 0.50` steps. |
| `burst_25` | 25% burst-class (binomial) for `BURST_BASE_STEPS / 0.25` steps. |

`burst_25` is designated `UNIFORM_SCHEDULE` — the least-concentrated baseline. The schedules are colour-coded red→blue in all plots.

---

## Evaluation Metric: Free-Generation Accuracy

At every `eval_every` (default 25) steps, the model is evaluated using **free generation** (autoregressive decoding without teacher forcing):

1. The model is given the prompt: `S [F_N ... F_1] <space> [inp]` (everything up to and including the input).
2. It generates tokens one at a time until the full document length is reached.
3. Accuracy is measured on the **last 6 tokens** of the generated sequence (i.e., the final output after all N bijections), compared to the ground truth.

Two accuracy values are tracked:
- `acc_other`: free-gen accuracy on a sample of other-class tasks.
- `acc_burst`: free-gen accuracy on all burst-class tasks.

---

## Summary Statistics Computed Per Run

After training, the following are computed from the `acc_burst` curve:

- **`peak_burst`**: maximum `acc_burst` during the burst phase.
- **`reversion_auc`**: area under the `acc_burst` curve during the reversion phase (trapezoidal rule over reversion steps). Lower = faster forgetting.
- **`life_{pct}`** for thresholds `[0.95, 0.90, 0.85, 0.80, 0.75, 0.70]`: the reversion step at which `acc_burst` first drops to `pct%` of `peak_burst`. Capped at `U` if it never drops that far. Lower = faster forgetting.
- **`dropoff_abs`** and **`dropoff_pct`**: absolute and percentage drop from `peak_burst` to the final `acc_burst` at end of reversion.

---

## Model Architecture

A nanoGPT-style decoder-only Transformer (`net/nanogpt.py`) with:
- `n_layer = 6` transformer blocks
- `n_embd = 120` model dimension
- `n_head = 4` attention heads
- No bias, no dropout
- MLP sublayers enabled
- `vocab_size = 128` (padded to accommodate all tokens)
- `context_size = 80` (padded to accommodate full document length)

An LSTM alternative (`net/lstm.py`, `AutoLstm`) exists but is not used in the main experiment pipeline.

Trained with:
- AdamW optimizer (`beta1=0.9`, `beta2=0.95`, `weight_decay=1e-3`)
- Gradient clipping at 1.0
- Mixed precision (bfloat16) with `GradScaler`
- Cosine LR decay with warmup

---

## Experiment Configuration

### Seeds and replication

Each (schedule, seed) pair is an independent run. Default: 10 seeds per schedule, seeds = `SEED_BASE + seed_idx` = `107, 108, ..., 116`. Data generation uses a fixed `DATA_SEED = 999` so all runs share identical task definitions and document pools. Within a schedule, all seeds share the same pretrained checkpoint from Phase 0.

### Data pools

For each task (identified by its tuple of function indices), `n_docs_per_task = 500` documents are pre-generated. Each document has a freshly sampled random input sequence. Evaluation uses `n_eval_per_task = 500` documents per task. All pools are padded to the same document length.

### Output folder layout

```
data/<date>_<time>_burst_d<depth>_pos<pos>/
  results/
    config.json
    analysis_report.pdf
    plots/
    presentation/
    grad_cosine_sim/
  logs/
    all_results.pkl
    _data.pkl
    pretrain_ckpt.pt
    checkpoints/
    task_distributions/
    <label>.pkl  (per-run result pickles)
```

### run.sh

`run.sh` orchestrates the full experiment pipeline. It currently runs depth-3 at all burst positions:

```
run_experiment 3 1
run_experiment 3 2
run_experiment 3 3
```

Each `run_experiment` call:
1. Runs `burst/experiment.py --depth D --burst-pos P`, which trains all jobs and saves results.
2. Calls `post_process` (from `post_process.sh`), which runs in parallel:
   - `burst/plot.py` — generates all plots and a PDF report.
   - `burst/probe.py` (if `run_probes=True`) — fits linear probes on saved checkpoints.
   - `scripts/probe_next_token_regimes.py` (if `run_next_token_probes=True`) — next-token probes at specific steps.
   - After probes finish: `burst/plot_probes.py` — plots probe heatmaps.
   - `burst/grad_sim.py` — computes gradient cosine similarities on saved checkpoints.
   - `burst/adl.py` (if `run_adl=True`, default True) — Activation Difference Lens analysis.
   - `burst/pres_pdf.py` — builds a presentation HTML/PDF.
   - `scripts/organize_run.py` — organises output files for download.
3. After all run directories are collected, runs `burst/unified_analysis.py` across all of them.

---

## File-by-File Reference

### `burst/config.py`

Central configuration. All schedules, colours, display labels, and phase names are derived from `BURST_FRACTIONS = [100, 98, 95, 90, 85, 75, 50, 25]`. Editing this list is the only change needed to add/remove schedules.

Key exports:
- `SCHEDULE_ORDER`: schedules sorted highest-to-lowest burst fraction.
- `SCHED_COLORS`: red→blue gradient, one colour per schedule.
- `MIXED_FRACTIONS`: dict mapping schedule name → burst fraction (all schedules including `burst_100`).
- `UNIFORM_SCHEDULE`: `"burst_25"` — the least-concentrated schedule.
- `BURST_BASE_STEPS`: base burst phase length (140 steps); actual burst length = `BURST_BASE_STEPS / frac`.
- `burst_steps_for_schedule(schedule)`: returns burst phase length for a given schedule.
- `TrainConfig`: dataclass with all model/training hyperparameters (includes `pre_burst_steps=420`, `total_steps=BURST_BASE_STEPS`, `reversion_steps=420`).
- `ExperimentConfig`: dataclass with n_seeds, n_workers, depth, burst_pos, schedules, `run_adl=True`.
- `reversion_life_key(threshold)` / `reversion_life_label(threshold)`: helpers for naming the life-time metrics.
- `parse_run_config(cfg)`: extracts depth, burst_pos, n_a, base_cfg from a saved `config.json`.
- `ordered_schedules(scheds)` / `sched_sort_key(schedule)`: ordering helpers.

### `burst/experiment.py`

Main entry point. Responsibilities:
1. Parse CLI args (`--depth`, `--burst-pos`, `--n-a`, `--schedules`, `--n-seeds`, `--n-workers`).
2. Build all data pools via `build_data()`.
3. Save data to `_data.pkl`.
4. **Pretrain** one shared model per schedule on other-class only for `pre_burst_steps` steps; save checkpoint to `pretrain_ckpt.pt`.
5. Create a job list: one job per (schedule, seed) pair, each starting from the shared pretrain checkpoint.
6. Divide jobs into chunks and launch `_worker_batched.py` subprocesses in parallel (up to `n_workers` at a time).
7. Poll progress files and print live status.
8. Collect all result `.pkl` files into `all_results.pkl`.
9. Clean up temporary files.

`DepthNData` class: generates all bijections, builds the vocabulary, and enumerates all other-class and burst-class task tuples. Bijection layout: `bijections[0]` = identity (F0), `bijections[(p-1)*n_a+1 .. p*n_a]` = the `n_a` background functions exclusive to position `p`, `bijections[n_a*depth+1]` = b*. `pos_fns[p]` maps each position to its dedicated function index list. `_build_splits` uses `itertools.product(*per_pos)` where each factor is the position-specific function list, so the same function index can never appear at two different positions. `_make_doc(task)` generates a single document token sequence for a given task. `gen_pool(tasks, n)` generates `n` documents per task.

`build_data()`: calls `DepthNData`, generates training and evaluation pools, pads all pools to the same document length, computes `prompt_len` (the number of tokens to feed as prompt during evaluation), and adjusts `vocab_size` and `context_size` to fit.

### `burst/_worker.py`

Trains a single model on a single (schedule, seed) job, starting from the shared pretrain checkpoint. Responsibilities:
1. Load data from shared pickle.
2. Load pretrain checkpoint and instantiate nanoGPT and AdamW optimizer.
3. Run `T` burst-phase steps, sampling batches according to `n_target_for_step()`.
4. Run `U` reversion steps (burst class removed).
5. Save model checkpoints every `CHECKPOINT_EVERY = 10` steps (for grad-sim and probes).
6. Evaluate with `eval_free_gen()` every `eval_every` steps.
7. Track task-distribution counters per phase and save to CSV files.
8. Compute `peak_burst`, `reversion_auc`, `life_{pct}` metrics.
9. Save full result dict to `{label}.pkl`.

`n_target_for_step(step, total_steps, schedule, p, batch_size)`: returns the number of burst-class documents for a given step. All schedules 25–100% use binomial sampling throughout the full burst phase.

`sample_batch(target_pool, bg_pool, n_target, batch_size)`: samples `n_target` burst documents and `batch_size - n_target` other-class documents, distributing evenly across tasks, then shuffles.

`eval_free_gen(net, docs_BL, prompt_len)`: runs autoregressive generation on a batch of documents, computes accuracy on the last 6 generated tokens.

`checkpoint_steps(T, U)`: returns the set of global steps at which checkpoints are saved (every 10 steps + 5 named "pairwise" steps for grad-sim).

### `burst/_worker_batched.py`

Thin wrapper: loads a list of jobs from a pickle file and calls `_worker.run()` sequentially, reusing the same CUDA context. This saves ~400 MB VRAM per job compared to spawning separate processes.

### `burst/data.py`

`BurstDataset`: a PyTorch `Dataset` wrapping a numpy array of documents. `__getitem__` returns `(tokens[:-1], tokens[1:])` for next-token prediction.

`pad_pools_to_same_length(*pools)`: pads all document arrays to the same length with zeros (PAD token = 0).

### `burst/gpu.py`

Auto-detects GPU capabilities (VRAM, BF16 TFLOPS) from a registry of known GPUs or from env vars `GPU_VRAM_GB` / `GPU_TFLOPS`. Computes three worker counts:
- `train_workers`: max parallel training processes (limited by VRAM and SM contention).
- `probe_workers`: max parallel probe processes (heavier per-process).
- `gradsim_workers`: max parallel grad-sim processes (heaviest, uses batch size 2048).

### `burst/parallel.py`

Generic subprocess job pool. `run_job_pool()` launches up to `n_workers` subprocesses in parallel, polls for completion, collects results from output pickle files, and calls an `on_done` callback for each completed job. Handles `BlockingIOError` with exponential backoff retries.

### `burst/train_utils.py`

Shared utilities used by `_worker.py`, `probe.py`, and other scripts:
- `make_net_bare(cfg)` / `make_net(cfg)`: instantiate nanoGPT (compiled version for training).
- `load_net(cfg, ckpt_path)`: load a nanoGPT from a checkpoint file.
- `make_optim_cfg(cfg)` / `make_scaler()`: create optimizer config and AMP scaler.
- `train_step(...)`: single forward+backward+optimizer step with cosine LR update.
- `retrain_with_callbacks(job, target_pool, bg_pool, on_step, max_step)`: re-runs a full training from scratch, calling `on_step(net, global_step, phase)` at each step. Used by `probe.py` when checkpoints are unavailable.
- `load_results(run_dir)` / `resolve_run_paths(run_dir)`: load `all_results.pkl` and `config.json`, resolving the new nested `results/` + `logs/` layout.
- `build_probe_docs(data, doc_len, n_per_task)`: generates balanced Other/Burst probe datasets.
- `compute_lr_schedule(cfg)`: computes the LR curve as numpy arrays (for plotting).

### `burst/grad_sim.py`

Post-hoc gradient cosine similarity computation. For each saved checkpoint across all jobs:
1. Load the checkpoint.
2. Compute the gradient vector for burst-class documents: do a forward+backward pass on a sample of burst docs, flatten all parameter gradients into a single vector.
3. Compute the gradient vector for other-class documents similarly.
4. Compute cosine similarity between the two gradient vectors (`burst_vs_other`).
5. At 5 "pairwise" steps (begin, mid-burst, end-burst, mid-reversion, end-reversion), also compute a full pairwise cosine similarity matrix across groups: BURST, O_F{i} for each `i` in `range((burst_pos-1)*n_a+1, burst_pos*n_a+1)` (other tasks grouped by their position-specific function at `burst_pos`), ALL_OTHER, ALL_DATA.

Results are saved per-job to `grad_cosine_sim/{label}.json` and merged back into `all_results.pkl`.

### `burst/adl.py`

**Activation Difference Lens (ADL)** — post-hoc mechanistic analysis based on the Narrow Fine-Tuning paper. For each saved checkpoint, computes:

1. `delta_KTN`: mean activation difference between the checkpoint and the pre-burst model on other-class inputs, at every (layer, token position).
2. **Logit Lens readability**: projects `delta_KTN` through the unembedding matrix and measures whether burst-relevant token IDs appear in the top-k predictions — tests whether the burst phase leaves a readable fingerprint even on non-burst data.
3. **Causal ablation**: projects `delta` out of the residual stream at each layer during burst-class generation and measures the accuracy drop — tests whether the burst knowledge is stored as an additive direction (wrapper) rather than a conditional circuit.

`_burst_token_ids(cfg, n_a, depth)`: computes the vocab token ID of b* as `func_start + n_a * depth + 1` (where `func_start = n_alphabets` and the vocab layout is `X0..X{n_alphabets-1}`, then `F0..F{n_a*depth+1}`, then specials).

Results are saved to `adl/{label}.json` and merged into `all_results.pkl`. Enabled by default (`run_adl=True` in `ExperimentConfig`).

### `burst/probe.py`

Linear probe analysis of the model's internal representations. For each checkpoint step:
1. Collect residual-stream activations at every (layer, token_position) pair for a balanced set of Other-class and Burst-class documents.
2. Fit a binary linear probe (logistic regression via a 2-class `nn.Linear` trained with Adam for up to 200 epochs with early stopping) at each (layer, token_position) pair.
3. Record validation accuracy `train_acc_KT` as a `(K, T)` array where `K = n_layers + 1` (embedding + each block output) and `T = doc_len - 1` (all token positions in the model input).

Can either load saved checkpoints (preferred) or retrain from scratch using `retrain_with_callbacks`.

Token position labels are named semantically: `S`, `F3`, `F2`, `F1`, `sp0`, `in0`...`in5`, `sp1`, `o1_0`...`o1_5`, `sp2`, `o2_0`...`o2_5`, etc.

### `burst/plot.py`

Generates all plots and a PDF report for a run directory. Plots produced:

- **`lr_schedule.png`**: the cosine LR curve with warmup, annotated with phase boundaries.
- **Per-run plots** (`{idx}_run_{sched}_s{seed}.png`): 3-panel figure per (schedule, seed) showing (1) schedule bar (burst fraction per step as a heatmap), (2) `acc_other` and `acc_burst` accuracy curves with peak/life annotations, (3) training loss.
- **`summary_bars.png`**: 3-panel bar chart showing peak burst accuracy, 95%-life, and reversion AUC per schedule (one bar per run, sorted by schedule).
- **`auc_detail.png`**: scatter plot of individual seed AUCs + mean±CI bar chart.
- **`auc_diff_pct.png`**: pairwise percentage difference matrix of mean reversion AUCs.
- **Per-schedule overlays** (`{idx}_overlay_{sched}.png`): mean±CI accuracy curves for all metrics for one schedule.
- **All-schedule overlays** (`overlay_all_{metric}.png`): all schedules on one plot for each metric.
- **Task distribution charts**: bar charts of task-type counts, function-usage distributions, and top compositions per (schedule, seed, phase).
- **`analysis_report.pdf`**: full PDF report assembling all the above with explanatory text.

### `burst/plot_probes.py`

Generates probe visualisation plots from `probe.py` output:

- **Per-model heatmaps**: `(K, T)` heatmap of probe accuracy at key checkpoints (step 0, T/2, T, T+U/2, T+U) for each (schedule, seed).
- **Training dynamics**: line plots of probe accuracy over training steps for selected (layer, token) pairs (function tokens, separator tokens, first output tokens) at 3 layer depths (embedding, middle, last).
- **Mean dynamics by schedule**: mean-pooled probe accuracy (averaged over all layers and token positions) over training steps, one line per schedule.
- **Per-layer depth dynamics**: per-layer mean probe accuracy over time, one subplot per schedule.
- **Cross-model comparison heatmaps**: difference heatmaps `(K, T)` comparing probe accuracy between pairs of schedules at end-of-training and end-of-reversion.
- **Layer × Schedule heatmap**: rows = transformer layers, columns = schedules, values = mean probe accuracy at the final step.

### `burst/plot_utils.py`

Shared plotting utilities. `plotly_to_png_matplotlib(fig_plotly, path)`: converts a Plotly figure to a PNG using matplotlib (used when kaleido is unavailable). `save_png(fig, path)`: wrapper that tries kaleido first, falls back to matplotlib.

### `burst/deep_analysis.py`

Five-metric deep analysis of burstiness runs, operating on saved checkpoints without retraining:

1. **ADL** (Activation Difference Lens) — readability + causal ablation (see `adl.py`).
2. **Gradient interference magnitude** — from existing `grad_sim_log`.
3. **EMA interpolation probe** — sharpness of the peak↔reverted cliff via exponential moving average interpolation.
4. **Critical sharpness** — Hutchinson trace of the Hessian on burst loss (deep_analysis.py). In unified_analysis.py: λ_c = 2/η_c via forward-pass line search along the burst direction Δθ = θ_peak − θ_pre (Kalra & Barkeshli 2024).
5. **Weight delta rank** — SVD of `(W_post - W_pre)` per layer.

Outputs `results.pkl`, per-chart PNGs, and an interactive Plotly dashboard. Can process all valid runs in `data/` in parallel via `--all --n-parallel N`.

### `burst/new_metrics.py`

Ten additional post-hoc mechanistic metrics, complementing `deep_analysis.py`:

From checkpoints:
1. **Task Vector Transfer** — does `τ = θ_post − θ_pre` transfer to a fresh model?
2. **Forgetting Trajectory Dim** — PCA dimensionality of the reversion weight path.
3. **Relearning Efficiency** — burst accuracy recovery after 50 fine-tune steps.
4. **Linear Mode Connectivity** — loss barrier on the straight path peak→reverted.
5. **Pruning Robustness** — burst accuracy vs magnitude-based weight sparsity.

From existing data (no checkpoints needed):
6. **Pairwise Gradient Separation** — BURST vs ALL_OTHER cosine sim across 5 key steps.
7. **Forgetting Speed Decomposition** — initial slope / plateau / AUC from training curves.
8. **Per-Layer Interference Localisation** — which layer has most negative cosine sim?
9. **Gradient Interference Temporal Dynamics** — reversion-phase re-alignment trajectory.
10. **Burst Position Comparison** — cross-run meta-analysis (pos1 / pos2 / pos3).

### `burst/fingerprint_analysis.py`

Finetuning fingerprint analysis adapted from the "Narrow Fine-Tuning Targets" paper. Two analyses:

1. **Logit Lens on Checkpoint Deltas** — computes `δ̄ = E_x[h^post(x) - h^pre(x)]` on other-class inputs, projects through the unembedding matrix, and checks whether top-k tokens are burst-relevant.
2. **Activation Steering** — adds `α·δ̄` to the residual stream at layer ℓ during autoregressive generation on other-class prompts, testing whether the burst knowledge is stored as an additive direction.

`_burst_token_ids(cfg, n_a, depth)`: identical to the one in `adl.py` — b*'s vocab token ID is `func_start + n_a * depth + 1`.

### `burst/unified_analysis.py`

Unified burstiness analysis dashboard that merges `deep_analysis` (5 metrics) and `new_metrics` (10 metrics) into one script. Adds Frankenstein layer-swap analysis (swapping individual layers between pre-burst and post-burst models), evaluates on both burst and other-class docs, shows individual seed points + error bars, and compares pre-burst vs post-burst models. Accepts multiple run directories and produces a combined dashboard. Called by `run.sh` at the end of all experiments.

### `burst/pres_charts.py` / `burst/pres_pdf.py`

Generate a presentation-style HTML/PDF with selected key charts and summary tables. Not described in detail here as they are post-processing wrappers.

### `scripts/probe_next_token_regimes.py`

Next-token probes per layer for Other-class vs Burst-class regimes. Two probe types:
1. **logit_lens** — applies the model's own `ln_f + LM_head` to intermediate layer activations and measures next-token accuracy (no training).
2. **learned_probe** — trains a small linear layer (`N → 10 digit classes`) with cross-entropy via SGD, then measures accuracy.

Retrains each model to the target step, extracts residual-stream activations at every transformer layer, and produces per-regime accuracy curves, A−B diffs, and diff-in-diffs.

### `scripts/organize_run.py`

Moves heavy files (`.pkl`, `.pt`, checkpoint directories) into a `_heavy/` subdirectory and replaces them with symlinks. This keeps the main run directory lightweight for download while all existing code follows symlinks transparently.

### `net/nanogpt.py`

nanoGPT decoder-only Transformer implementation. Used for all training and analysis.

### `net/lstm.py`

`AutoLstm`: an LSTM-based autoregressive language model with the same interface as nanoGPT. Present for comparison but not used in the main experiment pipeline.

### `net/runner.py`

Shared optimizer and LR schedule utilities: `configure_optimizers()` and `update_cosine_warmup_lr()`.

### `analysis/load_eval_results.py`

A Jupyter-style analysis script (cells marked with `# %%`) for loading and plotting pre-existing evaluation results from a different experiment format (`data/inorder_eval_step_random50/accs.pkl` and `data/outorder_eval_step_random50/accs.pkl`). This script is independent of the burst experiment pipeline and reads from a different data directory.

### `synthetic/`

Data generation utilities:
- `synthetic/generator.py`: generates bijection functions and document sequences.
- `synthetic/functions.py`: bijection function definitions.
- `synthetic/init.py`: `set_seed()` utility for reproducible seeding.

---

## Data Flow Summary

```
run.sh
  └─ burst/experiment.py --depth D --burst-pos P
       ├─ DepthNData: generate bijections + all task tuples
       ├─ build_data(): generate document pools, save _data.pkl
       ├─ Pretrain shared model for P steps → pretrain_ckpt.pt
       ├─ Create jobs: (schedule × seed) pairs
       └─ Launch _worker_batched.py subprocesses (parallel)
            └─ _worker.run() per job (starts from pretrain_ckpt):
                 ├─ Train T burst steps (binomial sampling per schedule)
                 ├─ Train U reversion steps (no burst)
                 ├─ Save checkpoints every 10 steps
                 ├─ Eval every 25 steps → acc_other, acc_burst
                 ├─ Save task distribution CSVs
                 └─ Save {label}.pkl (metrics + log)
       └─ Collect → all_results.pkl

  └─ post_process (post_process.sh)
       ├─ burst/plot.py → plots/ + analysis_report.pdf
       ├─ burst/probe.py → probes/*.pkl (optional)
       │    └─ burst/plot_probes.py → probes/plots/
       ├─ scripts/probe_next_token_regimes.py (optional)
       ├─ burst/grad_sim.py → grad_cosine_sim/*.json
       ├─ burst/adl.py → adl/*.json (default enabled)
       └─ burst/pres_pdf.py → presentation PDF

  └─ burst/unified_analysis.py (all run dirs combined)
```

---

## Key Constants (defaults)

| Parameter | Value | Meaning |
|---|---|---|
| `n_alphabets` | 10 | Size of the symbol alphabet |
| `seq_len` | 6 | Length of each input/output sequence |
| `n_a` | 3 | Number of background bijection functions **per position** |
| `N_A` | 3 | Global default for n_a |
| `SEED_BASE` | 107 | First training seed |
| `DATA_SEED` | 999 | Seed for data generation (fixed across all runs) |
| `BURST_BASE_STEPS` | 140 | Base burst phase length (burst_100 uses this directly) |
| `pre_burst_steps` P | 420 | Pre-burst pretraining steps (shared checkpoint) |
| `total_steps` T | 140 | Burst phase steps (= `BURST_BASE_STEPS`) |
| `reversion_steps` U | 420 | Reversion training steps |
| `batch_size` | 128 | Documents per training step |
| `p_target` | 0.25 | Target burst fraction for uniform schedule |
| `eval_every` | 25 | Steps between evaluations |
| `n_seeds` | 10 | Independent seeds per schedule |
| `n_docs_per_task` | 500 | Training documents per task |
| `n_eval_per_task` | 500 | Evaluation documents per task |
| `CHECKPOINT_EVERY` | 10 | Steps between checkpoint saves |
