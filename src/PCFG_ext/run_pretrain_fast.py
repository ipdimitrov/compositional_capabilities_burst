"""
Pretraining script (seed=9).

Trains a single model with correlation=0 (uncorrelated data only).
Produces a model checkpoint at: pretrain_corr_0.00.pth
(used by run_finetune_fast_w_metrics.py)
"""

import os
import json
import torch

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
from pcfg_gen import CharTokenizer, PCFGGenerator, build_pools
from train_help import train, build_eval_datasets, _record_phase

SEED = 9
CORRELATION = 0.0

cfg = CFG
cfg["device"] = "cuda:0"
cfg["seed"] = SEED

set_seed(SEED)
device = get_device(cfg["device"])
print(f"Using device: {device}")

# Create result directories
results_dir = cfg["paths"]["results_dir"]
models_dir = cfg["paths"]["models_dir"]
histories_dir = cfg["paths"]["histories_dir"]
os.makedirs(models_dir, exist_ok=True)
os.makedirs(histories_dir, exist_ok=True)

# Build task registry
task_registry = build_task_registry(cfg["task_definitions"])
pretrain_tasks = cfg["task_sets"]["pretrain"]

chunk_size = cfg["pcfg"]["chunk_size"]
experiment_cfg = cfg["experiment"]

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

pretrain_task_weights = resolve_task_weights(
    pretrain_tasks,
    "operand_probs",
    cfg["operand_probs"],
)

eval_datasets = build_eval_datasets(pools, cfg, task_registry, tokenizer)

print(f"\n{'='*80}")
print(f"Pretraining (seed={SEED}, correlation={CORRELATION})")
print(f"Steps: {experiment_cfg['pretrain_steps']}")
print(f"Batch size: {experiment_cfg['pretrain_batch_size']}")
print(f"{'='*80}\n")

model = GPT(gpt_config).to(device)

optimizer = build_optimizer(
    model.parameters(),
    cfg["optimizer"],
    experiment_cfg["pretrain_lr"],
)

history = train(
    model=model,
    tokenizer=tokenizer,
    device=device,
    steps=experiment_cfg["pretrain_steps"],
    batch_size=experiment_cfg["pretrain_batch_size"],
    lr=experiment_cfg["pretrain_lr"],
    min_lr=experiment_cfg["pretrain_min_lr"],
    warmup_steps=get_warmup_steps(
        experiment_cfg["pretrain_steps"],
        warmup_ratio=experiment_cfg["pretrain_warmup_ratio"],
    ),
    log_interval=experiment_cfg["pretrain_log_interval"],
    task_names=pretrain_tasks,
    task_weights=pretrain_task_weights,
    pcfg_gen=pcfg,
    task_reg=task_registry,
    chunk_size=chunk_size,
    mask_answer_only=False,
    max_grad_norm=experiment_cfg["max_grad_norm"],
    optimizer=optimizer,
    val_datasets=eval_datasets,
    val_batch_size=experiment_cfg["pretrain_batch_size"],
    log_prefix=f"Pretrain[corr={CORRELATION:.2f}]",
    use_lr_schedule=True,
    metrics=cfg.get("metrics"),
    data_pools=pools,
    correlation=CORRELATION,
)

# Save model checkpoint
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "correlation": CORRELATION,
    "seed": SEED,
    "steps": experiment_cfg["pretrain_steps"],
}

model_path = f"pretrain_corr_{CORRELATION:.2f}.pth"
torch.save(checkpoint, model_path)
torch.save(checkpoint, os.path.join(models_dir, f"pretrain_corr_{CORRELATION:.2f}.pth"))

# Save history
history_path = os.path.join(
    histories_dir,
    f"pretrain_corr_{CORRELATION:.2f}_seed{SEED}_history.pth",
)
torch.save(history, history_path)

# Save summary
summary = {"model_path": model_path, "final": _record_phase(history)}
summary_path = os.path.join(results_dir, f"pretrain_summary_seed{SEED}.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\nSaved model: {model_path}")
print(f"Saved pretrain summary: {summary_path}")
print(f"\n{'='*80}")
print(f"Pretraining completed! Models saved to: {models_dir}")
print(f"{'='*80}")
