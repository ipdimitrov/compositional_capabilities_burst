from pcfg_gen import CharTokenizer, PCFGGenerator, TaskRegistry, format_example, PCFGDataset, collate_fn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
from typing import List, Dict
import math


def calculate_answer_accuracy(logits, targets, answer_positions_batch):
    """Calculate accuracy on answer tokens only."""
    predictions = torch.argmax(logits, dim=-1)

    correct = 0
    total = 0

    for i in range(targets.size(0)):
        positions = answer_positions_batch[i]
        if not positions:
            continue

        answer_preds = predictions[i][positions]
        answer_targets = targets[i][positions]

        valid_mask = answer_targets != -100
        if valid_mask.sum() == 0:
            continue

        answer_preds = answer_preds[valid_mask]
        answer_targets = answer_targets[valid_mask]

        correct += (answer_preds == answer_targets).sum().item()
        total += valid_mask.sum().item()

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, total


# ----------------------------
# LR schedule helpers
# ----------------------------

def get_cosine_lr(step: int, total_steps: int, base_lr: float, min_lr: float, warmup_steps: int):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def sample_batch(batch_size, task_names, task_weights, pcfg_gen, task_reg,
                 tokenizer, chunk_size=250, mask_answer_only=True,
                 data_pools=None, correlation=0.0):
    """Sample a training batch by generating examples online."""
    examples = []
    for _ in range(batch_size):
        if data_pools is not None:
            if correlation > 0.0 and random.random() < correlation:
                pcfg_string = random.choice(data_pools['correlated'])
            else:
                pcfg_string = random.choice(data_pools['uncorrelated'])
        else:
            pcfg_string = pcfg_gen.generate_chunk(chunk_size)
        task_name = random.choices(task_names, weights=task_weights, k=1)[0]
        task_def, answer = task_reg.apply_task(task_name, pcfg_string)
        examples.append(format_example(pcfg_string, task_def, answer))

    batch_dataset = PCFGDataset(examples, tokenizer, max_length=512, mask_answer_only=mask_answer_only)
    batch = collate_fn([batch_dataset[i] for i in range(len(batch_dataset))], tokenizer)
    return batch


def _build_val_loaders(val_datasets, batch_size, tokenizer, collate_fn_override=None):
    if val_datasets is None:
        return {}
    collate = collate_fn_override or collate_fn
    if isinstance(val_datasets, DataLoader):
        return {"val": val_datasets}
    if isinstance(val_datasets, dict):
        val_items = val_datasets.items()
    elif isinstance(val_datasets, (list, tuple)):
        val_items = [(f"val_{i}", ds) for i, ds in enumerate(val_datasets)]
    else:
        val_items = [("val", val_datasets)]
    val_loaders = {}
    for name, ds in val_items:
        if isinstance(ds, DataLoader):
            val_loaders[name] = ds
        else:
            val_loaders[name] = DataLoader(
                ds, batch_size=batch_size, shuffle=False,
                collate_fn=lambda batch, tok=tokenizer: collate(batch, tok),
            )
    return val_loaders


def _evaluate_loader(model, loader, device, metrics_set):
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_tokens = 0.0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            answer_positions = batch['answer_positions']
            logits, loss = model(input_ids, target_ids)
            total_loss += loss.item()
            if "answer_acc" in metrics_set:
                acc, total = calculate_answer_accuracy(logits, target_ids, answer_positions)
                total_correct += acc * total
                total_tokens += total
    avg_loss = total_loss / max(1, len(loader))
    avg_acc = total_correct / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, avg_acc


def train(
    model, tokenizer, device, steps, batch_size, lr, min_lr, warmup_steps,
    log_interval, task_names, task_weights, pcfg_gen, task_reg,
    chunk_size=250, mask_answer_only=True, use_correlation=False,
    max_grad_norm=1.0, optimizer=None, val_datasets=None, val_batch_size=None,
    collate_fn_override=None, log_prefix="Train", use_lr_schedule=True,
    lr_schedule_total_steps=None, lr_schedule_start_step=0, metrics=None,
    data_pools=None, correlation=0.0, on_eval=None,
):
    """Main training loop with online batch sampling."""
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    val_bs = val_batch_size or batch_size
    val_loaders = _build_val_loaders(val_datasets, val_bs, tokenizer, collate_fn_override)
    metrics_set = set(metrics or ["loss", "answer_acc"])

    history = {
        "steps": [],
        "val": {name: {} for name in val_loaders},
    }
    if "loss" in metrics_set:
        history["train_loss"] = []
        for name in val_loaders:
            history["val"][name]["loss"] = []
    if "answer_acc" in metrics_set:
        history["train_answer_acc"] = []
        for name in val_loaders:
            history["val"][name]["answer_acc"] = []

    for step in range(1, steps + 1):
        if use_lr_schedule:
            schedule_total = lr_schedule_total_steps or steps
            schedule_step = lr_schedule_start_step + step
            cur_lr = get_cosine_lr(schedule_step, schedule_total, lr, min_lr, warmup_steps)
        else:
            cur_lr = lr
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr

        batch = sample_batch(
            batch_size=batch_size, task_names=task_names,
            task_weights=task_weights, pcfg_gen=pcfg_gen, task_reg=task_reg,
            chunk_size=chunk_size, mask_answer_only=mask_answer_only,
            tokenizer=tokenizer, data_pools=data_pools, correlation=correlation,
        )

        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        answer_positions = batch['answer_positions']

        model.train()
        logits, loss = model(input_ids, target_ids)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        train_acc = None
        if "answer_acc" in metrics_set:
            acc, total = calculate_answer_accuracy(logits, target_ids, answer_positions)
            train_acc = acc if total > 0 else 0.0

        if step % log_interval == 0:
            history["steps"].append(step)
            if "loss" in metrics_set:
                history["train_loss"].append(loss.item())
            if "answer_acc" in metrics_set:
                history["train_answer_acc"].append(train_acc)

            log_parts = [
                f"{log_prefix} step {step}/{steps}",
                f"LR: {cur_lr:.6f}",
            ]
            if "loss" in metrics_set:
                log_parts.insert(1, f"Loss: {loss.item():.4f}")
            if "answer_acc" in metrics_set:
                log_parts.insert(2, f"Answer Acc: {train_acc:.4f}")

            if val_loaders:
                for name, loader in val_loaders.items():
                    v_loss, v_acc = _evaluate_loader(model, loader, device, metrics_set)
                    if "loss" in metrics_set:
                        history["val"][name]["loss"].append(v_loss)
                        log_parts.append(f"{name} Loss: {v_loss:.4f}")
                    if "answer_acc" in metrics_set:
                        history["val"][name]["answer_acc"].append(v_acc)
                        log_parts.append(f"{name} Acc: {v_acc:.4f}")

                if len(val_loaders) == 1:
                    only_name = next(iter(val_loaders))
                    if "loss" in metrics_set:
                        history.setdefault("val_loss", []).append(
                            history["val"][only_name]["loss"][-1])
                    if "answer_acc" in metrics_set:
                        history.setdefault("val_answer_acc", []).append(
                            history["val"][only_name]["answer_acc"][-1])

            print(" | ".join(log_parts))

            if on_eval is not None:
                on_eval(model, step, history)

    return history


def generate_dataset_from_pool(pool, n_examples, task_names, task_reg):
    """Sample examples from a pre-built pool for evaluation datasets."""
    examples = []
    for _ in range(n_examples):
        pcfg_string = random.choice(pool)
        task_name = random.choice(task_names)
        task_def, answer = task_reg.apply_task(task_name, pcfg_string)
        examples.append(format_example(pcfg_string, task_def, answer))
    return examples


def build_eval_datasets(pools, cfg_dict, task_registry, tokenizer):
    """Build the standard set of evaluation datasets from pre-built pools."""
    all_tasks = cfg_dict["task_sets"]["all"]
    other_tasks = [t for t in all_tasks if t not in ["count_a", "count_b"]]
    n_val = cfg_dict["data"]["val_examples"]
    max_len = cfg_dict["tokenizer"]["max_length"]
    mask_answer_only_val = cfg_dict["tokenizer"]["mask_answer_only_val"]
    per_other = cfg_dict["data"].get("eval_per_other_task", 500)

    return {
        "count_a_corr": PCFGDataset(
            generate_dataset_from_pool(pools["correlated"], n_val, ["count_a"], task_registry),
            tokenizer, max_length=max_len, mask_answer_only=mask_answer_only_val),
        "count_a_uncorr": PCFGDataset(
            generate_dataset_from_pool(pools["uncorrelated"], n_val, ["count_a"], task_registry),
            tokenizer, max_length=max_len, mask_answer_only=mask_answer_only_val),
        "count_b_corr": PCFGDataset(
            generate_dataset_from_pool(pools["correlated"], n_val, ["count_b"], task_registry),
            tokenizer, max_length=max_len, mask_answer_only=mask_answer_only_val),
        "count_b_uncorr": PCFGDataset(
            generate_dataset_from_pool(pools["uncorrelated"], n_val, ["count_b"], task_registry),
            tokenizer, max_length=max_len, mask_answer_only=mask_answer_only_val),
        "all_other_corr": PCFGDataset(
            generate_dataset_from_pool(pools["correlated"], per_other * len(other_tasks), other_tasks, task_registry),
            tokenizer, max_length=max_len, mask_answer_only=mask_answer_only_val),
        "all_other_uncorr": PCFGDataset(
            generate_dataset_from_pool(pools["uncorrelated"], per_other * len(other_tasks), other_tasks, task_registry),
            tokenizer, max_length=max_len, mask_answer_only=mask_answer_only_val),
    }


def get_final_metric(history: Dict, split_name: str, metric_name: str):
    """Get the last recorded value for a metric on a given validation split."""
    vals = history.get("val", {}).get(split_name, {}).get(metric_name, [])
    return vals[-1] if vals else None


def _record_phase(hist):
    """Extract final accuracy and loss for each eval split from a history dict."""
    splits = ["count_a_corr", "count_a_uncorr",
              "count_b_corr", "count_b_uncorr",
              "all_other_corr", "all_other_uncorr"]
    d = {}
    for s in splits:
        d[f"{s}_acc"]  = get_final_metric(hist, s, "answer_acc")
        d[f"{s}_loss"] = get_final_metric(hist, s, "loss")
    return d


def build_task_weights(task_names: List[str], operand_probs: Dict[str, float]) -> List[float]:
    weights = []
    for name in task_names:
        if name.startswith('count_') and name in ['count_a', 'count_b', 'count_c']:
            tok = name.split('_')[1]
            weights.append(operand_probs.get(tok, 1.0))
        elif name.startswith('index_') and name in ['index_a', 'index_b', 'index_c']:
            tok = name.split('_')[1]
            weights.append(operand_probs.get(tok, 1.0))
        else:
            weights.append(1.0)
    return weights
