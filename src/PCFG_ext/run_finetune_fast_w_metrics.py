"""
Fast finetuning + reverse experiment WITH gradient metrics (no model saving).

Parallelises across concentration values using concurrent threads + CUDA streams
and uses mixed-precision (fp16) to better saturate the GPU.  Also computes
gradient projection and layerwise drift at each eval step.

Pretrained models are expected at: pretrain_corr_{correlation:.2f}.pth
"""

import os
import json
import torch
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from torch.utils.data import DataLoader

from config import CFG
from config_utils import (
    build_optimizer,
    build_task_registry,
    get_device,
    get_warmup_steps,
    resolve_task_weights,
    set_seed,
)
from mingpt import GPT, GPTConfig
from pcfg_gen import (
    CharTokenizer, PCFGGenerator,
    build_pools, collate_fn,
)
from train_help import train, build_eval_datasets, get_final_metric, _record_phase
from gradient_metrics import (
    compute_gradient_projection,
    compute_layerwise_drift,
    snapshot_state,
    build_task_loaders_weighted,
)

SEED = 3
SAVE_MODELS = True

cfg = CFG
cfg["device"] = "cuda:0"
cfg["seed"] = SEED

set_seed(SEED)
device = get_device(cfg["device"])
print(f"Using device: {device}")

# Create result directories
results_dir = cfg["paths"]["results_dir"]
histories_dir = cfg["paths"]["histories_dir"]
models_dir = cfg["paths"]["models_dir"]
os.makedirs(histories_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Build task registry
task_registry = build_task_registry(cfg["task_definitions"])
task_sets = cfg["task_sets"]
pretrain_tasks = task_sets["pretrain"]

chunk_size = cfg["pcfg"]["chunk_size"]
experiment_cfg = cfg["experiment"]
correlation_values = experiment_cfg["correlation_values"]
concentration_values = experiment_cfg["concentration_values"]

# Build shared pools
pool_cfg = cfg.get("pool", {})
pcfg = PCFGGenerator()
pools = build_pools(
    pcfg_gen=pcfg,
    n_correlated=pool_cfg.get("n_correlated", 100000),
    n_uncorrelated=pool_cfg.get("n_uncorrelated", 100000),
    chunk_size=chunk_size,
    verbose=True,
)

model_cfg = cfg["model"]
tokenizer = CharTokenizer()
n_layers = model_cfg["n_layer"]

gpt_config = GPTConfig(
    vocab_size=tokenizer.vocab_size,
    block_size=model_cfg["block_size"],
    n_layer=model_cfg["n_layer"],
    n_head=model_cfg["n_head"],
    n_embd=model_cfg["n_embd"],
    embd_pdrop=model_cfg["embd_pdrop"],
    resid_pdrop=model_cfg["resid_pdrop"],
    attn_pdrop=model_cfg["attn_pdrop"],
)

print(f"\n{'='*80}")
print(f"FAST Finetune + Reverse Experiment WITH Gradient Metrics (seed={SEED})")
print(f"Correlation values: {correlation_values}")
print(f"Concentration values: {concentration_values}")
print(f"{'='*80}\n")


# Pre-build eval datasets and metric loaders
eval_datasets = build_eval_datasets(pools, cfg, task_registry, tokenizer)

METRIC_BATCH_SIZE = 64
metric_loaders = {
    name: DataLoader(ds, batch_size=METRIC_BATCH_SIZE, shuffle=False,
                     collate_fn=lambda b, tok=tokenizer: collate_fn(b, tok))
    for name, ds in eval_datasets.items()
}

pretrain_task_weights = resolve_task_weights(
    pretrain_tasks,
    "operand_probs",
    cfg["operand_probs"],
)

finetune_train_tasks = [
    "count_a", "count_b", "count_c",
    "count_aa", "count_bb", "count_cc",
    "index_a", "index_b", "index_c",
    "index_aa", "index_bb", "index_cc",
    "token_at_40",
]

# Task pairs to compute gradient projection for
TASK_PAIRS = [
    ("count_a", "count_b"),
    ("count_a", "all_other"),
]

# Number of concentration values to run in parallel per correlation.
MAX_WORKERS = min(5, len(concentration_values))

# Lock for thread-safe printing
print_lock = threading.Lock()


def make_on_eval(correlation, prev_state_holder):
    """
    Create an on_eval callback that computes gradient metrics at each eval step.

    prev_state_holder: dict with key 'state' holding the previous model snapshot.
    """
    task_loaders = build_task_loaders_weighted(metric_loaders, correlation)

    def on_eval(model, step, history):
        # ---- Gradient projection ----
        for task_a, task_b in TASK_PAIRS:
            if task_a not in task_loaders or task_b not in task_loaders:
                continue
            proj = compute_gradient_projection(
                model, task_loaders[task_a], task_loaders[task_b],
                device, n_layers,
            )
            pair_key = f"grad_{task_a}_vs_{task_b}"
            if pair_key not in history:
                history[pair_key] = {
                    "dot_product": {l: [] for l in range(n_layers)},
                    "cosine_sim": {l: [] for l in range(n_layers)},
                    "norm_a": {l: [] for l in range(n_layers)},
                    "norm_b": {l: [] for l in range(n_layers)},
                }
            for l in range(n_layers):
                if l in proj:
                    for metric_name in ["dot_product", "cosine_sim", "norm_a", "norm_b"]:
                        history[pair_key][metric_name][l].append(proj[l][metric_name])

            cos_str = ", ".join(f"L{l}:{proj[l]['cosine_sim']:.3f}" for l in sorted(proj))
            with print_lock:
                print(f"      {pair_key} cos: [{cos_str}]")

        # ---- Layerwise drift ----
        drift = compute_layerwise_drift(model, prev_state_holder["state"], n_layers)
        if "layerwise_drift" not in history:
            history["layerwise_drift"] = {l: [] for l in range(n_layers)}
            for extra_key in ["embedding", "pos_embedding", "head"]:
                history["layerwise_drift"][extra_key] = []

        for key, val in drift.items():
            if key not in history["layerwise_drift"]:
                history["layerwise_drift"][key] = []
            history["layerwise_drift"][key].append(val)

        drift_str = ", ".join(f"L{l}:{drift[l]:.4f}" for l in range(n_layers))
        with print_lock:
            print(f"      drift: [{drift_str}]")

        # Update snapshot for next drift computation
        prev_state_holder["state"] = snapshot_state(model)

    return on_eval


def run_concentration(correlation, concentration, pretrain_model_path):
    """Run finetune + reverse for one (correlation, concentration) pair."""
    # Re-seed per thread so each (correlation, concentration) pair is deterministic
    # regardless of thread scheduling order.
    thread_seed = SEED + hash((correlation, concentration)) % (2**31)
    set_seed(thread_seed)

    # Each thread gets its own CUDA stream for overlap
    stream = torch.cuda.Stream(device=device)

    with torch.cuda.stream(stream):
        # Fresh model copy
        model = GPT(gpt_config).to(device)
        ckpt = torch.load(pretrain_model_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

        # ---- Finetuning ----
        finetune_optimizer = build_optimizer(
            model.parameters(),
            cfg["optimizer"],
            experiment_cfg["finetune_lr"],
        )

        other_task_weight = (1 - concentration) / len(finetune_train_tasks[1:])
        finetune_weights = [concentration, *[other_task_weight] * (len(finetune_train_tasks) - 1)]

        # Snapshot for drift tracking (starts from pretrained weights)
        ft_prev_state = {"state": snapshot_state(model)}

        history_finetune = train(
            model=model,
            tokenizer=tokenizer,
            device=device,
            steps=experiment_cfg["finetune_steps"],
            batch_size=experiment_cfg["finetune_batch_size"],
            lr=experiment_cfg["finetune_lr"],
            min_lr=experiment_cfg["finetune_min_lr"],
            warmup_steps=get_warmup_steps(
                experiment_cfg["finetune_steps"],
                warmup_ratio=experiment_cfg["finetune_warmup_ratio"],
            ),
            log_interval=experiment_cfg["finetune_log_interval"],
            task_names=finetune_train_tasks,
            task_weights=finetune_weights,
            pcfg_gen=pcfg,
            task_reg=task_registry,
            chunk_size=chunk_size,
            mask_answer_only=False,
            max_grad_norm=experiment_cfg["max_grad_norm"],
            optimizer=finetune_optimizer,
            val_datasets=eval_datasets,
            val_batch_size=experiment_cfg["finetune_batch_size"],
            log_prefix=f"Finetune[corr={correlation:.2f},conc={concentration:.2f}]",
            use_lr_schedule=True,
            metrics=cfg.get("metrics"),
            data_pools=pools,
            correlation=correlation,
            on_eval=make_on_eval(correlation, ft_prev_state),
        )

        # Save history only (no model checkpoint)
        finetune_history_path = os.path.join(
            histories_dir,
            f"finetune_corr_{correlation:.2f}_conc_{concentration:.2f}_seed{SEED}_history.pth",
        )
        torch.save(history_finetune, finetune_history_path)

        if SAVE_MODELS:
            finetune_model_path = os.path.join(
                models_dir,
                f"finetune_corr_{correlation:.2f}_conc_{concentration:.2f}_seed{SEED}.pth",
            )
            torch.save({"model_state_dict": model.state_dict()}, finetune_model_path)

        # ---- Reverse training ----
        reverse_optimizer = build_optimizer(
            model.parameters(),
            cfg["optimizer"],
            experiment_cfg["reverse_lr"],
        )

        # Fresh snapshot for reverse drift tracking
        rv_prev_state = {"state": snapshot_state(model)}

        reverse_history = train(
            model=model,
            tokenizer=tokenizer,
            device=device,
            steps=experiment_cfg["reverse_steps"],
            batch_size=experiment_cfg["reverse_batch_size"],
            lr=experiment_cfg["reverse_lr"],
            min_lr=experiment_cfg["reverse_min_lr"],
            warmup_steps=get_warmup_steps(
                experiment_cfg["reverse_steps"],
                warmup_ratio=experiment_cfg["reverse_warmup_ratio"],
            ),
            log_interval=experiment_cfg["reverse_log_interval"],
            task_names=pretrain_tasks,
            task_weights=pretrain_task_weights,
            pcfg_gen=pcfg,
            task_reg=task_registry,
            chunk_size=chunk_size,
            mask_answer_only=False,
            max_grad_norm=experiment_cfg["max_grad_norm"],
            optimizer=reverse_optimizer,
            val_datasets=eval_datasets,
            val_batch_size=experiment_cfg["reverse_batch_size"],
            log_prefix=f"Reverse[corr={correlation:.2f},conc={concentration:.2f}]",
            use_lr_schedule=False,
            metrics=cfg.get("metrics"),
            data_pools=pools,
            correlation=0.0,
            on_eval=make_on_eval(correlation, rv_prev_state),
        )

        reverse_history_path = os.path.join(
            histories_dir,
            f"reverse_corr_{correlation:.2f}_conc_{concentration:.2f}_seed{SEED}_history.pth",
        )
        torch.save(reverse_history, reverse_history_path)

        if SAVE_MODELS:
            reverse_model_path = os.path.join(
                models_dir,
                f"reverse_corr_{correlation:.2f}_conc_{concentration:.2f}_seed{SEED}.pth",
            )
            torch.save({"model_state_dict": model.state_dict()}, reverse_model_path)

    # Wait for stream to finish before reading results
    stream.synchronize()

    # Move model to CPU and delete to free GPU memory
    model.cpu()
    del model, finetune_optimizer, reverse_optimizer, stream
    del ft_prev_state, rv_prev_state
    torch.cuda.empty_cache()

    with print_lock:
        print(f"    Done: corr={correlation:.2f}, conc={concentration:.2f}")

    return {
        "finetune_final": _record_phase(history_finetune),
        "reverse_final": _record_phase(reverse_history),
    }


all_results = {}

for correlation in correlation_values:
    print(f"\n{'='*80}")
    print(f"Correlation: {correlation}")
    print(f"{'='*80}")

    correlation_key = f"corr_{correlation:.2f}"
    all_results[correlation_key] = {}

    # Always use the corr=0.00 pretrained model so pretrain representations
    # are clean and correlation only varies during finetuning.
    models_dir = cfg["paths"]["models_dir"]
    pretrain_paths = [
        "pretrain_corr_0.00.pth",
        os.path.join(models_dir, "pretrain_corr_0.00.pth"),
    ]
    pretrain_model_path = None
    for p in pretrain_paths:
        if os.path.exists(p):
            pretrain_model_path = p
            break
    if pretrain_model_path is None:
        print(f"  [SKIP] No pretrained model found (corr=0.00). Tried: {pretrain_paths}")
        continue
    print(f"  Loading pretrained model: {pretrain_model_path}")
    print(f"  Running {len(concentration_values)} concentrations with {MAX_WORKERS} workers...")

    # Free GPU memory before starting this correlation batch
    torch.cuda.empty_cache()

    # Run all concentrations in parallel for this correlation
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(run_concentration, correlation, conc, pretrain_model_path): conc
            for conc in concentration_values
        }
        for future in as_completed(futures):
            conc = futures[future]
            conc_key = f"conc_{conc:.2f}"
            try:
                all_results[correlation_key][conc_key] = future.result()
            except Exception as e:
                print(f"  [ERROR] corr={correlation:.2f}, conc={conc:.2f}: {e}")
                import traceback
                traceback.print_exc()

# Save summary
summary_path = os.path.join(results_dir, f"finetune_experiment_summary_seed{SEED}_4.json")
with open(summary_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved experiment summary: {summary_path}")
print(f"\n{'='*80}")
print(f"Experiment completed! Results saved to: {results_dir}")
print(f"{'='*80}")
