"""
Per-layer gradient and weight metrics for analyzing task interactions during training.

Metrics:
  - Gradient projection (dot product, cosine sim) between task pairs
  - Layerwise drift (weight change since last checkpoint)
"""

import torch
from typing import Dict, List
from torch.utils.data import DataLoader


def get_per_layer_grads(model, loaders_with_weights, device, n_layers):
    """
    Forward+backward through weighted loaders, return per-layer gradient vectors.

    Args:
        model: The model to compute gradients for.
        loaders_with_weights: List of (loader, weight) tuples.
        device: torch device.
        n_layers: Number of transformer layers.

    Returns:
        Dict mapping layer_idx -> flattened gradient tensor (CPU, float32).
    """
    model.train()
    model.zero_grad()

    for loader, w in loaders_with_weights:
        if w == 0:
            continue
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            logits, loss = model(input_ids, target_ids)
            (loss * w).backward()

    grads = {}
    for l in range(n_layers):
        layer_grads = []
        for name, param in model.named_parameters():
            if f'h.{l}.' in name and param.grad is not None:
                layer_grads.append(param.grad.detach().cpu().float().flatten())
        if layer_grads:
            grads[l] = torch.cat(layer_grads)

    model.zero_grad()
    return grads


def compute_gradient_projection(model, task_a_loaders, task_b_loaders, device, n_layers):
    """
    Compute per-layer gradient interaction between two tasks.

    Args:
        model: The model.
        task_a_loaders: List of (loader, weight) for task A.
        task_b_loaders: List of (loader, weight) for task B.
        device: torch device.
        n_layers: Number of transformer layers.

    Returns:
        Dict mapping layer_idx -> {dot_product, cosine_sim, norm_a, norm_b}.
    """
    grads_a = get_per_layer_grads(model, task_a_loaders, device, n_layers)
    grads_b = get_per_layer_grads(model, task_b_loaders, device, n_layers)

    results = {}
    for l in grads_a:
        if l not in grads_b:
            continue
        ga, gb = grads_a[l], grads_b[l]
        norm_a = ga.norm().item()
        norm_b = gb.norm().item()
        dot = (ga * gb).sum().item()
        cos = dot / (norm_a * norm_b + 1e-12)
        results[l] = {
            'dot_product': dot,
            'cosine_sim': cos,
            'norm_a': norm_a,
            'norm_b': norm_b,
        }
    return results


def compute_layerwise_drift(model, prev_state, n_layers):
    """
    Compute per-layer L2 distance between current model weights and a previous state.

    Args:
        model: Current model.
        prev_state: Dict of previous state_dict tensors (CPU).
        n_layers: Number of transformer layers.

    Returns:
        Dict mapping layer_idx -> float (L2 norm of weight change).
        Also includes 'embedding' and 'head' keys for non-layer params.
    """
    results = {}
    current = {k: v.detach().cpu().float() for k, v in model.state_dict().items()}

    # Per transformer layer
    for l in range(n_layers):
        layer_delta_sq = 0.0
        for key in current:
            if f'h.{l}.' in key and key in prev_state:
                diff = current[key] - prev_state[key].float()
                layer_delta_sq += (diff * diff).sum().item()
        results[l] = layer_delta_sq ** 0.5

    # Embedding + head
    for tag, pattern in [('embedding', 'wte'), ('pos_embedding', 'wpe'), ('head', 'lm_head')]:
        delta_sq = 0.0
        for key in current:
            if pattern in key and key in prev_state:
                diff = current[key] - prev_state[key].float()
                delta_sq += (diff * diff).sum().item()
        if delta_sq > 0:
            results[tag] = delta_sq ** 0.5

    return results


def snapshot_state(model):
    """Return a CPU copy of model state for drift computation."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def build_task_loaders_weighted(eval_loaders, correlation):
    """
    Build corr-weighted loader lists for each task, matching training data mix.

    Args:
        eval_loaders: Dict with keys like 'count_a_corr', 'count_a_uncorr', etc.
        correlation: Float, fraction from correlated pool.

    Returns:
        Dict mapping task_name -> list of (loader, weight) tuples.
    """
    tasks = {
        'count_a':   ('count_a_corr',     'count_a_uncorr'),
        'count_b':   ('count_b_corr',     'count_b_uncorr'),
        'all_other': ('all_other_corr',   'all_other_uncorr'),
    }
    result = {}
    for task_name, (corr_key, uncorr_key) in tasks.items():
        if corr_key in eval_loaders and uncorr_key in eval_loaders:
            result[task_name] = [
                (eval_loaders[corr_key], correlation),
                (eval_loaders[uncorr_key], 1 - correlation),
            ]
    return result
