# Compositional Capabilities Burst — Full Codebase Specification

## 1. Research Goal

This codebase studies **how a language model acquires and forgets a novel compositional capability** when that capability is introduced via a concentrated "burst" of training data and then removed. The experiment measures whether brief, concentrated exposure to a novel bijection (function) produces durable compositional understanding or transient memorisation, and how the burst concentration (fraction of special-class examples in each batch) affects learning speed, peak performance, and forgetting dynamics.

---

## 2. Synthetic Task: Depth-N Pure-Bijection Composition

### 2.1 Overview

The model learns to compose chains of bijective functions (permutations of a finite alphabet) presented in a structured document format. Each document is a next-token-prediction sequence.

### 2.2 Alphabet and Bijections

- **Alphabet size** (`n_alph`): default 10. Tokens are `X0, X1, ..., X9`.
- **Sequence length** (`seq_len`): default 6. Each input is a length-6 vector of alphabet tokens sampled uniformly with replacement.
- **Depth** (`depth`): default 3. The number of functions composed.
- **Functions per position** (`n_a`): default 4. Each depth position `p` (0-indexed) has its own dedicated set of `n_a` random bijections over the alphabet.
- **Bijection layout**:
  - `bijections[0]` = identity (F0, unused in tasks).
  - `bijections[(p)*n_a + 1 .. (p+1)*n_a]` = background functions for position `p`.
  - `bijections[n_a * depth + 1]` = **b\*** (the novel burst function).
- All bijections are random permutations of `[0, n_alph)`, generated from a seeded `np.random.RandomState`.

### 2.3 Document Format (depth=3 example)

```
S [F_out F_mid F_in] ' ' [input] ' ' [F_in(x)] ' ' [F_mid(F_in(x))] ' ' [F_out(F_mid(F_in(x)))]
```

where:
- `S` = start token
- `F_out, F_mid, F_in` = function identifier tokens (one per depth position)
- `' '` = space separator token
- `[input]` = 6 alphabet tokens
- Each subsequent block is the step-by-step output of applying functions from innermost outward

### 2.4 Vocabulary

Tokens are assigned integer indices in order:
1. `X0..X{n_alph-1}` — alphabet value tokens
2. `F0..F{n_a*depth+1}` — function identifier tokens
3. `' '`, `<PAD>`, `S` — special tokens

The model's `vocab_size` is set to `max(128, actual_vocab + 10)` and `context_size` to `max(80, doc_len + 5)`.

### 2.5 Task Classes

- **Other class** (background): All compositions using only the `n_a` background bijections at each position. This is the Cartesian product of all slot functions: `product(slot_fns[0], slot_fns[1], ..., slot_fns[depth-1])`.
- **Burst class** (special): Compositions where the function at `burst_pos` is replaced with `b*` (the novel function). All other positions use background functions. `burst_pos` counts from the outermost function inward:
  - `burst_pos=3` → replaces the 1st (outermost) function
  - `burst_pos=1` → replaces the 3rd (innermost) function

### 2.6 Data Pools

- `n_docs_per_task` = 100 documents per task for training pools
- `n_eval_per_task` = 100 documents per task for evaluation pools
- Evaluation uses up to 8 other-class tasks and all burst-class tasks
- All pools are padded to the same sequence length

---

## 3. Model Architecture: nanoGPT

A GPT-2-style autoregressive transformer.

### 3.1 Architecture Details

| Component | Detail |
|---|---|
| Token embedding | `nn.Embedding(vocab_size, n_embd)` |
| Position embedding | `nn.Embedding(context_size, n_embd)` |
| Dropout | 0.0 (disabled) |
| Transformer blocks | `n_layer` blocks, each with: |
| — LayerNorm | Pre-norm (no bias by default), eps=1e-5 |
| — CausalSelfAttention | `n_head` heads, uses `scaled_dot_product_attention` (Flash Attention) |
| — MLP | `Linear(n_embd, 4*n_embd)` → GELU → `Linear(4*n_embd, n_embd)` |
| Final LayerNorm | Pre-head norm |
| LM head | `Linear(n_embd, vocab_size, bias=False)` |
| Weight tying | `wte.weight = LM_head.weight` |

### 3.2 Default Hyperparameters

```
n_layer     = 6
n_embd      = 120
n_head      = 4
vocab_size  = 128   (auto-expanded if needed)
context_size = 80   (auto-expanded if needed)
dropout     = 0.0
bias        = False
mlp         = True
```

### 3.3 Weight Initialization

- All `nn.Linear` weights: `Normal(mean=0.0, std=0.02)`
- All `nn.Embedding` weights: `Normal(mean=0.0, std=0.02)`
- Residual projection (`c_proj.weight`): `Normal(mean=0.0, std=0.02 / sqrt(2 * n_layer))`

### 3.4 Generation

Autoregressive with KV-cache: one full forward pass fills the cache over the prompt, then each new token is generated with a single-token forward pass. Greedy decoding (argmax).

---

## 4. Training Pipeline

### 4.1 Three-Phase Structure

Every training run has three sequential phases:

| Phase | Name | Default Steps | Data |
|---|---|---|---|
| **Pre-burst** | `all-but-special (background)` | 600 | Only other-class (background) tasks |
| **Burst** | `special (burst)` | varies by schedule | Mix of burst + other (ratio set by schedule) |
| **Reversion** | `all-but-special (reversion)` | 500 | Only other-class (background) tasks |

### 4.2 Pretrain Phase

- Trains ONE shared checkpoint on all-but-special data for `pre_burst_steps` = 600 steps.
- Uses `batch_size` = 128, sampling evenly across all other-class task IDs.
- Retries if peak `acc_other` < 0.99.
- This checkpoint is shared across all seeds and schedules.

### 4.3 Burst Phase

Each schedule defines what fraction of each batch is burst-class:

| Schedule | Burst Fraction | Burst Steps (mode="current") |
|---|---|---|
| `burst_100` | 100% | 100 |
| `burst_98` | 98% (binomial) | 103 |
| `burst_95` | 95% (binomial) | 106 |
| `burst_90` | 90% (binomial) | 112 |
| `burst_80` | 80% (binomial) | 125 |
| `burst_70` | 70% (binomial) | 143 |
| `burst_60` | 60% (binomial) | 167 |
| `burst_50` | 50% (binomial) | 200 |
| `burst_40` | 40% (binomial) | 250 |
| `burst_30` | 30% (binomial) | 334 |
| `burst_20` | 20% (binomial) | 500 |
| `burst_10` | 10% (binomial) | 1000 |
| `burst_0` | 0% | 100 |

**All schedules** in `mode="current"` see the same total number of special-class examples: `burst_steps * frac = BURST_BASE_STEPS * 1.0 = 100`.

For schedules with frac < 1.0, the number of special-class examples per batch at each step is drawn from `Binomial(batch_size, frac)`.

**Burst modes**:
- `"current"` (default): steps scale inversely with concentration, batch_size fixed
- `"constant_steps"`: all schedules run BURST_BASE_STEPS (100); batch_size fixed
- `"scaled_batch"`: all schedules run BURST_BASE_STEPS; batch_size scales inversely

### 4.4 Reversion Phase

- `reversion_steps` = 500. All-but-special data only (0 burst examples per batch).
- Optimizer momentum/variance buffers are **reset** at the transition from burst to reversion.

### 4.5 Optimizer

- **AdamW** with fused kernel when available
- Weight decay applied only to 2D+ parameters (weight matrices and embeddings); biases and layer norms get 0 weight decay
- Default hyperparameters:

```
lr              = 1e-3
weight_decay    = 1e-3
beta1           = 0.9
beta2           = 0.9
grad_clip       = 1.0
warmup_iters    = 50
```

### 4.6 Learning Rate Schedule

Three-phase cosine schedule with a single linear warmup at the start:

1. **Pretrain phase** `[1, P]`:
   - Steps `[1, warmup_iters]`: linear warmup from 0 to `lr_max`
   - Steps `[warmup_iters+1, P]`: cosine decay from `lr_max` to `lr_max * lr_pretrain_end_frac`

2. **Burst phase** `[P+1, P+T]`:
   - Cosine decay from `lr_pretrain_end` to `lr_max * lr_burst_end_frac`

3. **Reversion phase** `[P+T+1, P+T+U]`:
   - Cosine decay from `lr_burst_end` to `lr_max * lr_reversion_end_frac`

```
lr_pretrain_end_frac  = 0.3    → pretrain ends at 3e-4
lr_burst_end_frac     = 0.15   → burst ends at 1.5e-4
lr_reversion_end_frac = 0.05   → reversion ends at 5e-5
```

Cosine segment formula: `lr_end + 0.5 * (1 + cos(π * t_frac)) * (lr_start - lr_end)`

### 4.7 Mixed Precision

- `torch.amp.autocast("cuda", dtype=torch.bfloat16)` enabled when CUDA is available
- `torch.amp.GradScaler("cuda")` for loss scaling
- Gradient accumulation with `max_micro_bs = 512` (splits large batches)

### 4.8 Seeds and Reproducibility

- Default `n_seeds` = 10: each schedule is run 10 times with seeds `base_seed + 0..9`
- Default `base_seed` = 42
- Data generation seed (`DATA_SEED`) = 999
- `seed_all()` sets: numpy, stdlib random, torch manual seed, CUDA seeds, and deterministic flags:
  - `torch.backends.cudnn.deterministic = True`
  - `torch.backends.cudnn.benchmark = False`
  - `torch.use_deterministic_algorithms(True, warn_only=False)`
  - `torch.backends.cuda.matmul.allow_tf32 = False`

### 4.9 Evaluation

Evaluation happens every `eval_every` = 25 steps during burst and reversion phases.

- **Free-generation accuracy**: Feed the model the prompt (tokens up to and including the first input block), then autoregressively generate the remaining tokens. Compare the generated tokens at positions `[eval_start, eval_end)` against ground truth. `eval_start` and `eval_end` correspond to the burst function's output block.
  - Formula: `eval_start = prompt_len + (burst_pos - 1) * (seq_len + 1)`, `eval_end = eval_start + seq_len`
- **Loss**: Cross-entropy over all positions (standard next-token prediction loss)
- Separate evaluation for `acc_other`, `acc_burst`, `loss_other`, `loss_burst`

### 4.10 Checkpointing

Checkpoints saved every `CHECKPOINT_EVERY` = 10 steps during burst+reversion, plus at key points (start, midpoint, end of each phase). Used for post-hoc gradient analysis and weight drift computation.

---

## 5. Reversion Metrics

After training, the following metrics are extracted from each run's log:

- **peak_burst**: Maximum burst accuracy during the burst phase
- **reversion_auc**: Area under the burst-accuracy curve during the reversion phase (trapezoid rule)
- **dropoff_abs**: `peak_burst - end_burst_acc`
- **dropoff_pct**: `dropoff_abs / peak_burst * 100`
- **Life times** at thresholds (0.95, 0.90, 0.85, 0.80, 0.75, 0.70): Number of reversion steps until burst accuracy drops below `peak_burst * threshold`. If it never drops below, the life time = `reversion_steps`.

---

## 6. Post-Hoc Gradient Analysis

After training, `burst.core.metrics.gradients` loads saved checkpoints and computes:

### 6.1 Global Gradient Cosine Similarity

At each checkpoint, compute a flat gradient vector for burst data and another for other data (using `grad_sim_batch_size` = 2048 samples), then compute their cosine similarity.

### 6.2 Per-Layer Gradient Metrics

Layer groups: `emb`, `L{i}_ln`, `L{i}_attn`, `L{i}_mlp`, `ln_f`

- **Per-layer cosine similarity**: Cosine of burst vs other gradient within each layer group
- **Gradient norm ratio**: `||g_burst_l|| / ||g_other_l||` per layer
- **Conflict rate**: Fraction of parameters where `sign(g_burst) ≠ sign(g_other)` per layer

### 6.3 Gradient Projection (OGD-style)

Decomposes `g_burst` into:
- `g_parallel` = projection of `g_burst` onto `g_other` direction
- `g_perp` = `g_burst - g_parallel` (orthogonal residual)

Metrics:
- `interference_magnitude`: `||g_parallel||`
- `useful_learning`: `||g_perp||`
- `interference_ratio`: `||g_parallel|| / ||g_burst||` = `|cos(α)|`
- `burst_norm`, `other_norm`, `burst_l1`, `burst_linf`, `other_l1`, `other_linf`

### 6.4 Effective Gradient Rank

Per-layer: reshape gradient into a matrix, compute SVD, measure effective rank = `exp(H(σ_hat))` where `H` is entropy of the normalized singular value distribution.

### 6.5 Gradient SNR (disabled by default)

Per-layer signal-to-noise ratio across individual examples using `vmap + grad`:
`SNR_l = ||mean_g_l||² / mean(||g_i_l - mean_g_l||²)`

### 6.6 Pairwise Gradient Similarity (disabled by default)

Groups tasks by: BURST, O_F{i} (other-class grouped by function at burst_pos), ALL_OTHER, ALL_DATA. Computes cosine similarity matrix between all group gradient vectors.

---

## 7. Representation Analysis

Compares pre-burst and post-burst (peak) model activations:

- Collects residual-stream activations at every layer for `n_docs_per_class` = 64 documents from other and burst pools
- Computes mean activation vector per layer (averaged over docs and positions)
- For late layers (last 2):
  - **Centroid projection**: `dot(other_drift, burst_drift) / ||burst_drift||`
  - **Other shift norm**: `||other_drift|| / ||other_pre||`
  - **Drift cosine**: cosine between other_drift and burst_drift
  - **Burst self-projection**: `||burst_drift||`
  - **Burst shift norm**: `||burst_drift|| / ||burst_pre||`
  - Pre/post norms for both classes

---

## 8. Next-Token Probes

Two probe methods applied at every transformer layer:

### 8.1 Logit Lens

Apply the model's own `ln_f + LM_head` to each layer's residual-stream activations at the final-output token positions. Measure next-token prediction accuracy.

### 8.2 Learned Linear Probe

Train a single `Linear(n_embd, 10)` probe on each layer's activations (flattened across token positions):
- `lr` = 0.01, optimizer = Adam
- `epochs` = 200, `val_frac` = 0.2, `patience` = 30, `val_every` = 10
- Reports best validation accuracy

Both methods are evaluated on Other-class and Burst-class data separately, producing diff curves (Other - Burst) per layer.

---

## 9. Weight Drift

For each checkpoint pair (base=first checkpoint, current=each subsequent checkpoint):
- **Cumulative drift**: `||θ_current - θ_base||` per layer group (Frobenius norm)
- **Stepwise drift**: `||θ_current - θ_previous||` per layer group

---

## 10. Bundle and Charts

### 10.1 Core Bundle

All metrics are aggregated into a single `CoreBundle` JSON file containing:
- Config (burst_mode, base_cfg, thresholds, schedules)
- Schedule bars (burst fraction time-series)
- LR curves
- Training curves (acc_burst, acc_other, loss, loss_burst, loss_other) with mean ± 95% CI across seeds
- Summary (peak_burst, reversion_auc, life times) with mean ± CI
- Gradient metrics (cosine, norms, signed_dot, interference_power, grad_rank)
- Per-layer gradient metrics (cosine, burst_norm, other_norm, norm×cosine)
- Weight drift (cumulative and stepwise per layer)
- Representation drift
- Next-token probes

95% CI formula: `1.96 * std / sqrt(n_seeds)`

### 10.2 Charts

Publication-quality PDF charts rendered with matplotlib (Times font, publication style):
- Schedule bars showing burst fraction over time
- LR schedule curves
- Overlay accuracy/loss curves across schedules
- Per-schedule burst vs other accuracy
- Reversion zoom (burst accuracy during reversion only)
- AUC bar charts
- Summary statistics table
- Gradient cosine similarity (overlay and per-schedule)
- Gradient norms (burst and other L2)
- Signed dot product and interference power
- Per-layer heatmaps and line charts (cosine, norms, norm×cosine)
- Weight drift heatmaps and line charts
- Representation drift (centroid projection, shift norm, burst self-projection, centroid norms)
- Effective gradient rank
- Probe accuracy by layer, diff curves, diff-in-diffs

---

## 11. Parallelism

### 11.1 Training

- Batched workers: experiment.py splits jobs into chunks and spawns `n_workers` subprocess workers, each training multiple jobs sequentially in ONE CUDA context (saves ~400MB VRAM per extra job).
- Worker count auto-detected from GPU VRAM and TFLOPS via `gpu_cfg`.

### 11.2 Gradient Analysis

- Each checkpoint is a separate subprocess job via `JobPool` with pickle serialization.
- Failed jobs retried up to 2 times.

### 11.3 Probes

- Same `JobPool` mechanism as gradient analysis.

---

## 12. File Layout

```
data/<run_name>/                          (run_dir)
data/results/<run_name>/                  config.json, plots/, presentation/,
                                          grad_cosine_sim/, chart_bundle/
data/logs/<run_name>/                     all_results.pkl, _data.pkl,
                                          pretrain_ckpt.pt, pretrain_log.pkl,
                                          checkpoints/<label>/step_*.pt,
                                          task_distributions/,
                                          <label>.pkl
```

Run name format: `<YYYYMMDD-HHMMSS>_burst_d<depth>_pos<burst_pos>[_<burst_mode>]`

---

## 13. CLI

```bash
# Full training experiment
python -m burst.core train --depth 3 --burst-pos 3

# With options
python -m burst.core train --depth 3 --burst-pos 2 --n-a 6 \
    --schedules burst_100 burst_50 burst_25 \
    --n-seeds 10 --burst-mode current --seed 42

# Post-hoc gradient analysis
python -m burst.core gradients <run_dir>
python -m burst.core gradients <run_dir> --n-workers 8 --grad-sim-batch-size 2048

# Build bundle + render charts
python -m burst.core pipeline <run_dir>

# Bundle only
python -m burst.core bundle <run_dir>

# Charts only
python -m burst.core charts <run_dir> --out-dir ./my_charts

# Next-token probes (separate script)
python scripts/probe_next_token_regimes.py <run_dir> --probe-max-samples 500
```

---

## 14. Legacy Pipeline (01/02/03 scripts)

An older pipeline for general function composition experiments (not burst-specific):

1. **01_generate_data.py**: Generates synthetic corpus from composed bijections using config/gen/conf.yaml
2. **02_train.py**: Trains nanoGPT on the generated corpus using config/train/conf.yaml
3. **03_evaluate_i.py / 03_evaluate_o.py**: Evaluates on in-order and out-of-order function compositions

Legacy config defaults:
- depth=5, n_functions=3, n_alphabets=10, seq_len=6
- Split strategy: "random" with 50 compositions
- Model: 2-layer, 120-dim, 1-head, vocab_size=512, context_size=50

---

## 15. Complete Hyperparameter Reference

### Training Config (TrainConfig)

| Parameter | Default | Description |
|---|---|---|
| `n_alphabets` | 10 | Size of token alphabet |
| `seq_len` | 6 | Length of input sequence |
| `n_layer` | 6 | Transformer layers |
| `n_embd` | 120 | Embedding dimension |
| `n_head` | 4 | Attention heads |
| `vocab_size` | 128 | Vocabulary size (auto-expanded) |
| `context_size` | 80 | Max context length (auto-expanded) |
| `lr` | 1e-3 | Peak learning rate |
| `lr_pretrain_end_frac` | 0.3 | LR fraction at end of pretrain |
| `lr_burst_end_frac` | 0.15 | LR fraction at end of burst |
| `lr_reversion_end_frac` | 0.05 | LR fraction at end of reversion |
| `weight_decay` | 1e-3 | AdamW weight decay |
| `beta1` | 0.9 | Adam beta1 |
| `beta2` | 0.9 | Adam beta2 |
| `grad_clip` | 1.0 | Max gradient norm |
| `warmup_iters` | 50 | Linear warmup steps |
| `batch_size` | 128 | Training batch size |
| `grad_sim_batch_size` | 2048 | Batch size for gradient analysis |
| `pre_burst_steps` | 600 | Pretrain phase length |
| `total_steps` | 100 | Burst base steps (BURST_BASE_STEPS) |
| `p_target` | 0.2 | Target burst fraction (used by some schedules) |
| `reversion_steps` | 500 | Reversion phase length |
| `eval_every` | 25 | Evaluation frequency |
| `reversion_thresholds` | (0.95, 0.90, 0.85, 0.80, 0.75, 0.70) | Life-time thresholds |
| `n_docs_per_task` | 100 | Training docs per task |
| `n_eval_per_task` | 100 | Eval docs per task |

### Experiment Config (ExperimentConfig)

| Parameter | Default | Description |
|---|---|---|
| `n_seeds` | 10 | Number of random seeds per schedule |
| `n_workers` | auto | Parallel training processes |
| `depth` | 3 | Composition depth |
| `burst_pos` | 3 | Position of novel function (1=innermost, depth=outermost) |
| `burst_mode` | "current" | One of: "current", "constant_steps", "scaled_batch" |
| `schedules` | all 13 | burst_100, burst_98, ..., burst_0 |
| `run_probes` | False | Run linear probes |
| `run_next_token_probes` | False | Run next-token probes |
| `run_adl` | True | Run activation distribution analysis |

### Global Constants

| Constant | Value | Description |
|---|---|---|
| `BURST_FRACTIONS` | [100,98,95,90,80,70,60,50,40,30,20,10,0] | Available burst percentages |
| `N_A` | 4 | Functions per depth position |
| `SEED_BASE` | 100 | Legacy seed base |
| `DATA_SEED` | 999 | Seed for data generation |
| `DEFAULT_REPRO_SEED` | 42 | Default experiment seed |
| `PRETRAIN_ACC_THRESHOLD` | 0.99 | Min pretrain accuracy to proceed |
| `GRAD_NORM_EPS` | 1e-6 | Epsilon for gradient norm comparisons |
| `CHECKPOINT_EVERY` | 10 | Checkpoint save interval |
| `MIN_VECTORS_FOR_SIMILARITY` | 2 | Min vectors for pairwise similarity |
| `N_SNR_EXAMPLES` | 16 | Examples for gradient SNR |
| `N_PROBE_DOCS_PER_TASK` | 200 | Docs per task for probes |
| `PROBE_SEED` | 1337 | Seed for probe data sampling |
| `PROBE_COLLECT_BATCH_SIZE` | 256 | Batch size for activation collection |
| `N_DIGITS` | 10 | Number of digit classes for probes |
| `ACTIVATION_COLLECT_BATCH_SIZE` | 512 | Batch size for activation collection |
| `N_REPRESENTATION_DOCS_PER_CLASS` | 64 | Docs per class for representation analysis |
| `EVAL_BATCH_SIZE` | 256 | Batch size for evaluation |
| `VOCAB_SLACK` | 10 | Extra vocab size padding |
| `CONTEXT_SLACK` | 5 | Extra context size padding |

### Probe Hyperparameters

| Parameter | Value |
|---|---|
| `LEARNED_PROBE_LR` | 0.01 |
| `LEARNED_PROBE_EPOCHS` | 200 |
| `LEARNED_PROBE_VAL_FRAC` | 0.2 |
| `LEARNED_PROBE_VAL_EVERY` | 10 |
| `LEARNED_PROBE_PATIENCE` | 30 |

---

## 16. Dependencies

```
torch==2.12.0.dev20260325+cu128
omegaconf==2.3.0
numpy==2.4.4
tqdm==4.67.3
matplotlib==3.10.8
scikit-learn==1.7.2
einops==0.8.2
```

Python 3.12, CUDA 12.8, managed via uv.

---

## 17. Key Algorithms in Pseudocode

### 17.1 Batch Sampling (`n_target_for_step`)

```python
if schedule == "burst_100":
    n_target = batch_size
elif schedule in MIXED_FRACTIONS:
    frac = MIXED_FRACTIONS[schedule]
    n_target = binomial(batch_size, frac)
```

### 17.2 Cosine Segment LR

```python
def cosine_segment(t_frac, lr_start, lr_end):
    coeff = 0.5 * (1.0 + cos(π * t_frac))
    return lr_end + coeff * (lr_start - lr_end)
```

### 17.3 Optimizer Reset at Reversion

```python
for group in optimizer.param_groups:
    for p in group["params"]:
        for k, v in optimizer.state[p].items():
            if isinstance(v, Tensor): v.zero_()
            elif k == "step": state[k] = 0
```

### 17.4 Free-Generation Evaluation

```python
full = net.generate(docs[:, :prompt_len], n_new_tokens)
gen = full[:, eval_start:eval_end]
ref = targets[:, eval_start-1:eval_end-1]
accuracy = (gen == ref).float().mean()
```

### 17.5 Cross-Entropy Loss

```python
logits_bv = rearrange(logits_BTV, "b t v -> (b t) v")
targets_b = rearrange(targets_BT, "b t -> (b t)")
loss = F.cross_entropy(logits_bv, targets_b)
```
