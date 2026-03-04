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
- `F0 ... F{n_a+1}`: function identifiers (`F0` = identity, `F1..F{n_a}` = "other" functions, `F{n_a+1}` = `b*` the burst function)
- `<space>`, `<PAD>`, `S`: special tokens

---

## The Two Classes of Tasks

### Other class (background / foundation)

There are `n_a` (default 3) ordinary bijections. The "other" tasks are all `n_a^depth` possible depth-N chains built from these functions. For depth=3 and n_a=3 that is 27 tasks. These are the tasks the model trains on throughout the entire experiment.

### Burst class (the novel capability)

One additional bijection `b*` (index `n_a + 1`) is introduced. The burst tasks are all `n_a^{depth-1}` chains where `b*` occupies a specific position `burst_pos` (1-indexed, 1 = outermost, depth = innermost) and all other positions range over the `n_a` ordinary functions. For depth=3, n_a=3, burst_pos=3, that is 9 burst tasks. The burst class is entirely novel — the model has never seen `b*` before the burst phase.

---

## The Three Training Phases

### Phase 1: Foundation + Burst (T steps, default 500)

The model trains for `T` total steps. Each step, a batch of size `batch_size` (default 128) is sampled. The batch is a mixture of:
- **Other-class documents**: drawn uniformly from all `n_a^depth` other tasks.
- **Burst-class documents**: drawn uniformly from all burst tasks.

The number of burst documents per step is controlled by the **schedule** (see below). During the "foundation" sub-phase (before the burst window), `n_target = 0` (no burst data). During the burst window, `n_target > 0`.

### Phase 2: Reversion (U steps, default 500)

After `T` steps, the burst class is completely removed. For all `U` reversion steps, `n_target = 0` — only other-class data is used. This measures how quickly the model forgets the burst capability.

### Learning rate

A single cosine decay schedule with linear warmup runs across all `T + U` steps. Warmup lasts `warmup_iters` (default 50) steps, then decays from `lr` (default 3e-4) to `min_lr` (default 6e-5). The reversion phase continues the same decaying schedule — the model keeps learning on other-class data at a decreasing rate.

---

## The Schedules

The schedule determines when and how much burst-class data appears in the batch during the `T` training steps. The key parameter is `p_target` (default 0.10), which defines the fraction of a batch that is burst-class. The "burst length" is `burst_len = max(int(p_target * T), 1)` = 50 steps at default settings.

All schedules (except `burst_10`) deliver the same *total amount* of burst-class data; they differ only in *when* it appears.

| Schedule | Description |
|---|---|
| `burst_100` | 100% burst-class for the last `burst_len` steps. Pure block at the end. |
| `burst_98` | 98% burst-class in a window at the end. Window size = `burst_len / 0.98`. |
| `burst_95` | 95% burst-class in a window at the end. |
| `burst_90` | 90% burst-class in a window at the end. |
| `burst_85` | 85% burst-class in a window at the end. |
| `burst_75` | 75% burst-class in a window at the end. |
| `burst_50` | 50% burst-class in a window at the end. |
| `burst_25` | 25% burst-class in a window at the end. |
| `burst_10` | ~10% burst-class drawn randomly throughout all `T` steps (uniform baseline). |

For `burst_X` where X is not 100 and not 10: the window length is `min(burst_len / frac, T)` steps at the end of training, and each step in that window has exactly `round(batch_size * frac)` burst documents.

The schedules are ordered from most concentrated (burst_100) to least concentrated (burst_10) and are colour-coded red→blue in all plots.

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

- **`peak_burst`**: maximum `acc_burst` during the foundation+burst phase.
- **`reversion_auc`**: area under the `acc_burst` curve during the reversion phase (trapezoidal rule over reversion steps). Lower = faster forgetting.
- **`life_{pct}`** for thresholds `[0.95, 0.90, 0.85, 0.80, 0.75, 0.70]`: the reversion step at which `acc_burst` first drops to `pct%` of `peak_burst`. Capped at `U` if it never drops that far. Lower = faster forgetting.
- **`dropoff_abs`** and **`dropoff_pct`**: absolute and percentage drop from `peak_burst` to the final `acc_burst` at end of reversion.

---

## Model Architecture

A nanoGPT-style decoder-only Transformer with:
- `n_layer = 6` transformer blocks
- `n_embd = 120` model dimension
- `n_head = 4` attention heads
- No bias, no dropout
- MLP sublayers enabled
- `vocab_size = 128` (padded to accommodate all tokens)
- `context_size = 80` (padded to accommodate full document length)

Trained with:
- AdamW optimizer (`beta1=0.9`, `beta2=0.95`, `weight_decay=1e-3`)
- Gradient clipping at 1.0
- Mixed precision (bfloat16) with `GradScaler`
- Cosine LR decay with warmup

---

## Experiment Configuration

### Seeds and replication

Each (schedule, seed) pair is an independent run. Default: 10 seeds per schedule, seeds = `SEED_BASE + seed_idx` = `107, 108, ..., 116`. Data generation uses a fixed `DATA_SEED = 999` so all runs share identical task definitions and document pools.

### Data pools

For each task (identified by its tuple of function indices), `n_docs_per_task = 500` documents are pre-generated. Each document has a freshly sampled random input sequence. Evaluation uses `n_eval_per_task = 500` documents per task. All pools are padded to the same document length.

### run.sh

`run.sh` orchestrates the full experiment pipeline. It calls `run_experiment depth pos` for each (depth, burst_pos) combination:

```
run_experiment 3 3
run_experiment 4 1
run_experiment 4 2
run_experiment 4 3
run_experiment 4 4
```

Each `run_experiment` call:
1. Runs `burst/experiment.py --depth D --burst-pos P`, which trains all jobs and saves results to `data/burst_dD_posP_<timestamp>/`.
2. Calls `post_process` (from `post_process.sh`), which runs in parallel:
   - `burst/plot.py` — generates all plots and a PDF report.
   - `burst/probe.py` (if `run_probes=True`) — fits linear probes on saved checkpoints.
   - `scripts/probe_next_token_regimes.py` (if `run_next_token_probes=True`) — next-token probes at specific steps.
   - After probes finish: `burst/plot_probes.py` — plots probe heatmaps.
   - `burst/grad_sim.py` — computes gradient cosine similarities on saved checkpoints.
   - `burst/pres_pdf.py` — builds a presentation HTML/PDF.
   - `scripts/organize_run.py` — organises output files for download.

---

## File-by-File Reference

### `burst/config.py`

Central configuration. All schedules, colours, display labels, and phase names are derived from `BURST_FRACTIONS = [100, 98, 95, 90, 85, 75, 50, 25, 10]`. Editing this list is the only change needed to add/remove schedules.

Key exports:
- `SCHEDULE_ORDER`: schedules sorted highest-to-lowest burst fraction.
- `SCHED_COLORS`: red→blue gradient, one colour per schedule.
- `MIXED_FRACTIONS`: dict mapping schedule name → burst fraction (excludes `burst_100` and `burst_10`).
- `TrainConfig`: dataclass with all model/training hyperparameters.
- `ExperimentConfig`: dataclass with n_seeds, n_workers, depth, burst_pos, schedules.
- `reversion_life_key(threshold)` / `reversion_life_label(threshold)`: helpers for naming the life-time metrics.

### `burst/experiment.py`

Main entry point. Responsibilities:
1. Parse CLI args (`--depth`, `--burst-pos`, `--n-a`, `--schedules`, `--n-seeds`, `--n-workers`).
2. Build all data pools via `build_data()`.
3. Save data to `_data.pkl`.
4. Create a job list: one job per (schedule, seed) pair.
5. Divide jobs into chunks and launch `_worker_batched.py` subprocesses in parallel (up to `n_workers` at a time).
6. Poll progress files and print live status.
7. Collect all result `.pkl` files into `all_results.pkl`.
8. Clean up temporary files.

`DepthNData` class: generates all bijections, builds the vocabulary, and enumerates all other-class and burst-class task tuples. `_make_doc(task)` generates a single document token sequence for a given task. `gen_pool(tasks, n)` generates `n` documents per task.

`build_data()`: calls `DepthNData`, generates training and evaluation pools, pads all pools to the same document length, computes `prompt_len` (the number of tokens to feed as prompt during evaluation), and adjusts `vocab_size` and `context_size` to fit.

### `burst/_worker.py`

Trains a single model on a single (schedule, seed) job. Responsibilities:
1. Load data from shared pickle.
2. Instantiate nanoGPT and AdamW optimizer.
3. Run `T` training steps (foundation + burst phase), sampling batches according to `n_target_for_step()`.
4. Run `U` reversion steps (burst class removed).
5. Save model checkpoints every `CHECKPOINT_EVERY = 10` steps (for grad-sim and probes).
6. Evaluate with `eval_free_gen()` every `eval_every` steps.
7. Track task-distribution counters per phase and save to CSV files.
8. Compute `peak_burst`, `reversion_auc`, `life_{pct}` metrics.
9. Save full result dict to `{label}.pkl`.

`n_target_for_step(step, total_steps, schedule, p, batch_size)`: returns the number of burst-class documents for a given step under a given schedule.

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
- `make_net(cfg)`: instantiates nanoGPT from config dict.
- `make_optim_cfg(cfg)` / `make_scaler()`: create optimizer config and AMP scaler.
- `train_step(...)`: single forward+backward+optimizer step with cosine LR update.
- `retrain_with_callbacks(job, target_pool, bg_pool, on_step, max_step)`: re-runs a full training from scratch, calling `on_step(net, global_step, phase)` at each step. Used by `probe.py` when checkpoints are unavailable.
- `load_results(run_dir)`: loads `all_results.pkl` and `config.json`.
- `build_probe_docs(data, doc_len, n_per_task)`: generates balanced Other/Burst probe datasets.
- `compute_lr_schedule(cfg)`: computes the LR curve as numpy arrays (for plotting).

### `burst/grad_sim.py`

Post-hoc gradient cosine similarity computation. For each saved checkpoint across all jobs:
1. Load the checkpoint.
2. Compute the gradient vector for burst-class documents: do a forward+backward pass on a sample of burst docs, flatten all parameter gradients into a single vector.
3. Compute the gradient vector for other-class documents similarly.
4. Compute cosine similarity between the two gradient vectors (`burst_vs_other`).
5. At 5 "pairwise" steps (begin, mid-burst, end-burst, mid-reversion, end-reversion), also compute a full pairwise cosine similarity matrix across groups: BURST, O_F1...O_Fn (other tasks grouped by function at `burst_pos`), ALL_OTHER, ALL_DATA.

Results are saved per-job to `grad_cosine_sim/{label}.json` and merged back into `all_results.pkl`.

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

### `burst/pres_charts.py` / `burst/pres_pdf.py`

Generate a presentation-style HTML/PDF with selected key charts and summary tables. Not described in detail here as they are post-processing wrappers.

### `analysis/load_eval_results.py`

A Jupyter-style analysis script (cells marked with `# %%`) for loading and plotting pre-existing evaluation results from a different experiment format (`data/inorder_eval_step_random50/accs.pkl` and `data/outorder_eval_step_random50/accs.pkl`). Plots:
- In-order evaluation: token accuracy, strict accuracy, and teacher-forced accuracy over training iterations.
- Out-of-order evaluation: accuracy as a function of (num_identities, displacement) shown as a heatmap and line plots.

This script is independent of the burst experiment pipeline and reads from a different data directory.

---

## Data Flow Summary

```
run.sh
  └─ burst/experiment.py --depth D --burst-pos P
       ├─ DepthNData: generate bijections + all task tuples
       ├─ build_data(): generate document pools, save _data.pkl
       ├─ Create jobs: (schedule × seed) pairs
       └─ Launch _worker_batched.py subprocesses (parallel)
            └─ _worker.run() per job:
                 ├─ Train T steps (foundation + burst per schedule)
                 ├─ Train U steps (reversion, no burst)
                 ├─ Save checkpoints every 10 steps
                 ├─ Eval every 25 steps → acc_other, acc_burst
                 ├─ Save task distribution CSVs
                 └─ Save {label}.pkl (metrics + log)
       └─ Collect → all_results.pkl

  └─ post_process (post_process.sh)
       ├─ burst/plot.py → plots/ + analysis_report.pdf
       ├─ burst/probe.py → probes/*.pkl (optional)
       │    └─ burst/plot_probes.py → probes/plots/
       ├─ burst/grad_sim.py → grad_cosine_sim/*.json
       └─ burst/pres_pdf.py → presentation PDF
```

---

## Key Constants (defaults)

| Parameter | Value | Meaning |
|---|---|---|
| `n_alphabets` | 10 | Size of the symbol alphabet |
| `seq_len` | 6 | Length of each input/output sequence |
| `n_a` | 3 | Number of "other" bijection functions |
| `N_A` | 3 | Global default for n_a |
| `SEED_BASE` | 107 | First training seed |
| `DATA_SEED` | 999 | Seed for data generation (fixed across all runs) |
| `total_steps` T | 500 | Foundation + burst training steps |
| `reversion_steps` U | 500 | Reversion training steps |
| `batch_size` | 128 | Documents per training step |
| `p_target` | 0.10 | Target burst fraction (defines burst_len = 50) |
| `eval_every` | 25 | Steps between evaluations |
| `n_seeds` | 10 | Independent seeds per schedule |
| `n_docs_per_task` | 500 | Training documents per task |
| `n_eval_per_task` | 500 | Evaluation documents per task |
| `CHECKPOINT_EVERY` | 10 | Steps between checkpoint saves |
