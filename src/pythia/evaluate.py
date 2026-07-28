"""Evaluation functions for catastrophic forgetting experiments."""

import math

import torch
from data import make_eval_loader


@torch.no_grad()
def evaluate_perplexity(model, dataset, batch_size: int, num_batches: int, device: str = "cuda"):
    """Compute average loss and perplexity on a dataset.

    Returns (loss, perplexity) tuple.
    """
    model.eval()
    loader = make_eval_loader(dataset, batch_size)

    total_loss = 0.0
    total_batches = 0

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        input_ids = batch.to(device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        total_loss += outputs.loss.item()
        total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    perplexity = math.exp(avg_loss)
    model.train()
    return avg_loss, perplexity


def run_evaluation(model, datasets, config, burst_level, phase, step, device="cuda"):
    """Run evaluation on both code and pile validation sets.

    Returns a metrics dict.
    """
    from datetime import datetime

    domain_loss, domain_ppl = evaluate_perplexity(
        model, datasets["domain_valid"],
        config.eval_batch_size, config.eval_batches, device,
    )
    pile_loss, pile_ppl = evaluate_perplexity(
        model, datasets["pile_valid"],
        config.eval_batch_size, config.eval_batches, device,
    )

    return {
        "burst_level": burst_level,
        "phase": phase,
        "step": step,
        "domain_val_loss": round(domain_loss, 4),
        "domain_val_perplexity": round(domain_ppl, 2),
        "pile_val_loss": round(pile_loss, 4),
        "pile_val_perplexity": round(pile_ppl, 2),
        "timestamp": datetime.now().isoformat(),
    }

