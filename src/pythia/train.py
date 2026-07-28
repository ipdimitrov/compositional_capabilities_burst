"""Training loops for fine-tuning and continued pretraining."""

import copy
import json
import math
import os

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)


def cosine_schedule_with_warmup_and_min_lr(
    optimizer, num_warmup_steps: int, num_training_steps: int, min_lr_ratio: float = 0.0
):
    """Cosine schedule with warmup that decays to (min_lr_ratio * peak_lr), not zero.

    With min_lr_ratio=0 this is identical to HF's get_cosine_schedule_with_warmup.
    With min_lr_ratio=0.1 the lr floor is 10% of peak.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


from config import ExperimentConfig
from data import (
    make_eval_loader,
    make_finetune_loader,
    make_pile_loader,
    prepare_datasets,
)
from evaluate import run_evaluation


def set_seeds(seed) -> None:
    import random

    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_metrics(metrics_list, path) -> None:
    """Append-safe save of metrics to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics_list, f, indent=2)


def log_metrics(m) -> None:
    """Print a metrics dict in a readable way."""
    print(f"  [{m['phase']}] burst={m['burst_level']:.0%} step={m['step']} | "
          f"domain_ppl={m['domain_val_perplexity']:.2f} pile_ppl={m['pile_val_perplexity']:.2f}")


def _get_grad_vector(model, dataset, batch_size, num_batches, device):
    """Compute the gradient of the loss on a dataset and return as a flat vector."""
    model.eval()
    model.zero_grad()
    loader = make_eval_loader(dataset, batch_size)
    total_loss = 0.0
    count = 0
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        input_ids = batch.to(device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        outputs.loss.backward()
        total_loss += outputs.loss.item()
        count += 1
    # Average gradients
    for p in model.parameters():
        if p.grad is not None:
            p.grad /= max(count, 1)
    # Flatten all grads into a single vector
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
    return torch.cat(grads)


def _simulate_adam_update(optimizer, model):
    """Compute the Adam weight update Δθ from gradients currently in model.parameters().grad.

    Reads the optimizer's accumulated (m, v, step) state and simulates one Adam
    step WITHOUT modifying the optimizer state or model parameters.
    Returns Δθ as a flat vector on CPU.
    """
    updates = []
    for group in optimizer.param_groups:
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        lr = group["lr"]
        wd = group.get("weight_decay", 0.0)

        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad.detach()

            state = optimizer.state.get(p, {})
            if not state:
                # No state yet (very early in training) — fall back to scaled gradient
                updates.append((-lr * g).flatten().cpu())
                continue

            m = state["exp_avg"]
            v = state["exp_avg_sq"]
            t = state["step"].item() if isinstance(state["step"], torch.Tensor) else state["step"]

            # Simulate Adam update (matching PyTorch AdamW)
            m_new = beta1 * m + (1 - beta1) * g
            v_new = beta2 * v + (1 - beta2) * g * g
            bias_corr1 = 1 - beta1 ** (t + 1)
            bias_corr2 = 1 - beta2 ** (t + 1)
            m_hat = m_new / bias_corr1
            v_hat = v_new / bias_corr2

            # AdamW: update = -lr * (m_hat / (sqrt(v_hat) + eps) + wd * p)
            update = -lr * (m_hat / (v_hat.sqrt() + eps) + wd * p.detach())
            updates.append(update.flatten().cpu())

    return torch.cat(updates)


def compute_grad_metrics(model, datasets, config, burst_level, phase, step, device,
                         optimizer=None):
    """Compute gradient metrics between domain loss gradient and pile weight update.

    If an optimizer is provided, simulates what Adam would do with the pile gradient
    (Δθ_adam) and computes cos(∇L_domain, Δθ_adam). This captures the actual update
    geometry under Adam, not just raw gradient alignment.

    Also stores the raw gradient cosine similarity for comparison.
    """
    num_batches = 10  # keep it fast

    # 1) Domain gradient → flat vector on CPU
    domain_grad = _get_grad_vector(model, datasets["domain_valid"],
                                    config.eval_batch_size, num_batches, device)
    domain_norm = domain_grad.norm().item()
    domain_grad_cpu = domain_grad.cpu()
    del domain_grad
    model.zero_grad()
    torch.cuda.empty_cache()

    # 2) Pile gradient (stays in p.grad for Adam simulation)
    pile_grad = _get_grad_vector(model, datasets["pile_valid"],
                                  config.eval_batch_size, num_batches, device)
    pile_norm = pile_grad.norm().item()
    pile_grad_cpu = pile_grad.cpu()
    del pile_grad

    # Raw gradient cosine similarity (SGD-equivalent)
    raw_cosine = torch.nn.functional.cosine_similarity(
        domain_grad_cpu.unsqueeze(0), pile_grad_cpu.unsqueeze(0)
    ).item()

    # 3) Adam update simulation (if optimizer available)
    adam_update_norm = 0.0
    adam_cosine = raw_cosine  # fallback if no optimizer
    if optimizer is not None:
        # p.grad still holds the pile gradient from _get_grad_vector
        adam_update = _simulate_adam_update(optimizer, model)
        adam_update_norm = adam_update.norm().item()
        adam_cosine = torch.nn.functional.cosine_similarity(
            domain_grad_cpu.unsqueeze(0), adam_update.unsqueeze(0)
        ).item()
        del adam_update

    del domain_grad_cpu, pile_grad_cpu
    model.zero_grad()
    torch.cuda.empty_cache()
    model.train()

    return {
        "burst_level": burst_level,
        "phase": phase,
        "step": step,
        "grad_cosine_similarity": round(raw_cosine, 6),
        "adam_cosine_similarity": round(adam_cosine, 6),
        "domain_grad_norm": round(domain_norm, 6),
        "pile_grad_norm": round(pile_norm, 6),
        "adam_update_norm": round(adam_update_norm, 6),
        "pile_grad_norm_cosine": round(pile_norm * raw_cosine, 6),
    }


def save_grad_metrics(grad_metrics_list, path) -> None:
    """Save gradient metrics to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(grad_metrics_list, f, indent=2)


def train_phase1(model, config, datasets, burst_level, device, all_metrics, metrics_path,
                  loss_history=None, grad_metrics=None):
    """Phase 1: Fine-tune the model on code data mixed with pile data at the given burst level.

    In "volume" mode ft_steps is scaled by 1/burst_level so every burst sees the
    same volume of domain data.
    """
    ft_steps = config.ft_steps_for_burst(burst_level)
    ft_warmup = config.ft_warmup_for_burst(burst_level)

    print(f"\n{'='*60}")
    print(f"Phase 1 — Fine-Tuning (burst_level={burst_level:.0%}) "
          f"[{ft_steps} steps, warmup={ft_warmup}, lr={config.ft_lr}, "
          f"min_ratio={config.ft_lr_min_ratio}]")
    print(f"{'='*60}")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.ft_lr, weight_decay=config.ft_weight_decay)
    scheduler = cosine_schedule_with_warmup_and_min_lr(
        optimizer, ft_warmup, ft_steps, min_lr_ratio=config.ft_lr_min_ratio
    )

    loader = make_finetune_loader(
        datasets["domain_train"], datasets["pile_train"],
        burst_level, config.ft_batch_size, seed=config.seed,
    )
    loader_iter = iter(loader)

    optimizer.zero_grad()

    pbar = tqdm(range(1, ft_steps + 1), desc=f"FT burst={burst_level:.0%}")
    for step in pbar:
        accum_loss = 0.0
        for _ in range(config.ft_grad_accum):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            input_ids = batch.to(device)
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss / config.ft_grad_accum
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        pbar.set_postfix(loss=f"{accum_loss:.4f}")

        if loss_history is not None:
            loss_history.append({
                "burst_level": burst_level,
                "phase": "finetune",
                "step": step,
                "train_loss": round(accum_loss, 4),
            })

        # Evaluate periodically during fine-tuning too
        if step % config.eval_every == 0:
            m = run_evaluation(model, datasets, config, burst_level,
                               "finetune", step, device)
            all_metrics.append(m)
            save_metrics(all_metrics, metrics_path)
            log_metrics(m)
            if loss_history is not None:
                with open(os.path.join(config.results_dir, "loss_history.json"), "w") as f:
                    json.dump(loss_history, f)
            model.train()

        # Gradient analysis (optional)
        if (config.compute_grad_metrics and grad_metrics is not None
                and step % config.grad_metrics_every == 0):
            gm = compute_grad_metrics(model, datasets, config, burst_level,
                                       "finetune", step, device, optimizer=optimizer)
            grad_metrics.append(gm)
            save_grad_metrics(grad_metrics,
                              os.path.join(config.results_dir, "grad_metrics.json"))

    return model


def train_phase2(model, config, datasets, burst_level, device, all_metrics, metrics_path,
                  loss_history=None, grad_metrics=None):
    """Phase 2: Continue pretraining on pure Pile data, evaluating periodically."""
    print(f"\n{'='*60}")
    print(f"Phase 2 — Continued Pretraining (burst_level={burst_level:.0%}) "
          f"[{config.cpt_steps} steps, lr={config.cpt_lr}]")
    print(f"{'='*60}")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.cpt_lr, weight_decay=config.cpt_weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, config.cpt_warmup_steps, config.cpt_steps)

    loader = make_pile_loader(datasets["pile_train"], config.cpt_batch_size, seed=config.seed)
    loader_iter = iter(loader)

    optimizer.zero_grad()

    pbar = tqdm(range(1, config.cpt_steps + 1), desc=f"CPT burst={burst_level:.0%}")
    for step in pbar:
        accum_loss = 0.0
        for _ in range(config.cpt_grad_accum):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            input_ids = batch.to(device)
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss / config.cpt_grad_accum
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        pbar.set_postfix(loss=f"{accum_loss:.4f}")

        if loss_history is not None:
            loss_history.append({
                "burst_level": burst_level,
                "phase": "continued_pretraining",
                "step": step,
                "train_loss": round(accum_loss, 4),
            })

        # Evaluate periodically
        if step % config.eval_every == 0:
            m = run_evaluation(model, datasets, config, burst_level,
                               "continued_pretraining", step, device)
            all_metrics.append(m)
            save_metrics(all_metrics, metrics_path)
            log_metrics(m)
            if loss_history is not None:
                with open(os.path.join(config.results_dir, "loss_history.json"), "w") as f:
                    json.dump(loss_history, f)
            model.train()

        # Gradient analysis (optional)
        if (config.compute_grad_metrics and grad_metrics is not None
                and step % config.grad_metrics_every == 0):
            gm = compute_grad_metrics(model, datasets, config, burst_level,
                                       "continued_pretraining", step, device, optimizer=optimizer)
            grad_metrics.append(gm)
            save_grad_metrics(grad_metrics,
                              os.path.join(config.results_dir, "grad_metrics.json"))

    return model


def run_single_burst(config, datasets, burst_level, pretrained_model, device,
                     all_metrics, metrics_path, loss_history=None, grad_metrics=None) -> None:
    """Run the full pipeline for a single burst level."""
    # Deep copy the pretrained model so each burst level starts fresh
    model = copy.deepcopy(pretrained_model).to(device)

    # Phase 1: Fine-tuning
    model = train_phase1(model, config, datasets, burst_level, device,
                         all_metrics, metrics_path, loss_history, grad_metrics)

    # Evaluate after fine-tuning
    m = run_evaluation(model, datasets, config, burst_level, "post_finetune", 0, device)
    all_metrics.append(m)
    save_metrics(all_metrics, metrics_path)
    log_metrics(m)

    # Save model after fine-tuning (optional)
    burst_tag = f"burst_{burst_level:.2f}".replace(".", "_")
    if config.save_checkpoints:
        ft_model_dir = os.path.join(config.results_dir, "models", burst_tag, "post_finetune")
        os.makedirs(ft_model_dir, exist_ok=True)
        model.save_pretrained(ft_model_dir)
        print(f"  Saved post-FT model to {ft_model_dir}")

    # Phase 2: Continued pretraining
    model = train_phase2(model, config, datasets, burst_level, device,
                         all_metrics, metrics_path, loss_history, grad_metrics)

    # Save model after continued pretraining (optional)
    if config.save_checkpoints:
        cpt_model_dir = os.path.join(config.results_dir, "models", burst_tag, "post_cpt")
        os.makedirs(cpt_model_dir, exist_ok=True)
        model.save_pretrained(cpt_model_dir)
        print(f"  Saved post-CPT model to {cpt_model_dir}")

    # Save loss history incrementally
    if loss_history is not None:
        loss_path = os.path.join(config.results_dir, "loss_history.json")
        with open(loss_path, "w") as f:
            json.dump(loss_history, f)

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()


def run_experiment(config: ExperimentConfig, callback=None):
    """Run the complete catastrophic forgetting experiment.

    Args:
        config: Experiment configuration.
        callback: Optional callable(metrics_dict) called after each evaluation.

    """
    set_seeds(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(config.results_dir, exist_ok=True)
    config.save()
    metrics_path = os.path.join(config.results_dir, "metrics.json")

    # Load existing metrics if resuming
    all_metrics = []
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            all_metrics = json.load(f)
        print(f"Loaded {len(all_metrics)} existing metric records from {metrics_path}")

    # Loss history for training curves
    loss_path = os.path.join(config.results_dir, "loss_history.json")
    loss_history = []
    if os.path.exists(loss_path):
        with open(loss_path) as f:
            loss_history = json.load(f)

    # Gradient metrics (optional)
    grad_metrics = None
    if config.compute_grad_metrics:
        grad_path = os.path.join(config.results_dir, "grad_metrics.json")
        grad_metrics = []
        if os.path.exists(grad_path):
            with open(grad_path) as f:
                grad_metrics = json.load(f)
        print(f"Gradient analysis enabled (every {config.grad_metrics_every} steps)")

    # Load model and tokenizer
    print(f"\nLoading model and tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pretrained_model = AutoModelForCausalLM.from_pretrained(config.model_name, dtype=torch.float32)
    num_params = sum(p.numel() for p in pretrained_model.parameters())
    print(f"Model loaded: {num_params:,} parameters")

    # Dump concise experiment summary
    if config.ft_budget_mode == "volume":
        per_burst_ft = "  ".join(
            f"{bl:.0%}:{config.ft_steps_for_burst(bl)}" for bl in config.burst_levels
        )
        ft_line = (
            f"ft:  mode=volume  base_steps={config.ft_steps} (scaled by 1/burst)  "
            f"per_burst=[{per_burst_ft}]\n"
        )
    else:
        ft_line = f"ft:  mode=steps  steps={config.ft_steps} (same for every burst)\n"
    summary = (
        f"model: {config.model_name} ({num_params/1e6:.1f}M params)\n"
        f"domain: {config.domain_train_dataset}  field={config.domain_text_field}\n"
        f"seq_length: {config.seq_length}\n"
        f"burst_levels: {config.burst_levels}\n"
        f"{ft_line}"
        f"     lr={config.ft_lr}  min_ratio={config.ft_lr_min_ratio}  bs={config.ft_batch_size}  grad_accum={config.ft_grad_accum}  eff_bs={config.ft_effective_batch}  warmup={config.ft_warmup_steps}  wd={config.ft_weight_decay}\n"
        f"cpt: steps={config.cpt_steps}  lr={config.cpt_lr}  bs={config.cpt_batch_size}  grad_accum={config.cpt_grad_accum}  eff_bs={config.cpt_effective_batch}  warmup={config.cpt_warmup_steps}  wd={config.cpt_weight_decay}\n"
        f"max_grad_norm: {config.max_grad_norm} seed: {config.seed}\n"
        f"eval: every={config.eval_every} batches={config.eval_batches}  bs={config.eval_batch_size}\n"
    )
    summary_path = os.path.join(config.results_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"\n{summary}")

    # Prepare datasets
    print("Preparing datasets (tokenize + chunk, may use cache)...")
    datasets = prepare_datasets(config, tokenizer)
    print(f"Datasets ready: {', '.join(datasets.keys())}")

    # Evaluate pretrained baseline (once)
    if not any(m["phase"] == "pretrained" for m in all_metrics):
        print("\nEvaluating pretrained baseline...")
        pretrained_model.to(device)
        m = run_evaluation(pretrained_model, datasets, config, 0.0, "pretrained", 0, device)
        all_metrics.append(m)
        save_metrics(all_metrics, metrics_path)
        log_metrics(m)
        pretrained_model.cpu()
        if callback:
            callback(m)

    # Run each burst level
    for burst_level in config.burst_levels:
        # Check if this burst level is already done
        existing = [m for m in all_metrics
                    if m["burst_level"] == burst_level and m["phase"] == "continued_pretraining"]
        if len(existing) >= config.cpt_steps // config.eval_every:
            print(f"\nSkipping burst_level={burst_level:.0%} (already complete)")
            continue

        print(f"\n{'#'*60}")
        print(f"# Starting burst_level={burst_level:.0%}")
        print(f"{'#'*60}")
        run_single_burst(config, datasets, burst_level, pretrained_model,
                         device, all_metrics, metrics_path, loss_history, grad_metrics)

    print(f"\n{'='*60}")
    print(f"Experiment complete! Results saved to {config.results_dir}/")
    print(f"{'='*60}")
    return all_metrics
