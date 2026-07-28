import random
from typing import Any

import numpy as np
import torch
from pcfg_gen import (
    TaskRegistry,
    count_char_task,
    count_composition_task,
    index_composition_task,
    index_occurrence_task,
    token_at_index_task,
)
from train_help import build_task_weights


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_setting: str) -> torch.device:
    if device_setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_setting)


def build_task_registry(task_defs: list[dict[str, Any]]) -> TaskRegistry:
    registry = TaskRegistry()
    for spec in task_defs:
        name = spec["name"]
        task_type = spec["type"]
        if task_type == "count_char":
            char = spec["char"]
            window = spec.get("window", 40)
            registry.register(name, lambda s, c=char, w=window: count_char_task(s, c, w))
        elif task_type == "count_comp":
            substring = spec["substring"]
            window = spec.get("window", 40)
            registry.register(
                name,
                lambda s, sub=substring, w=window: count_composition_task(s, sub, w),
            )
        elif task_type == "index_char":
            char = spec["char"]
            occurrence = spec.get("occurrence", 6)
            registry.register(
                name,
                lambda s, c=char, o=occurrence: index_occurrence_task(s, c, o),
            )
        elif task_type == "index_comp":
            substring = spec["substring"]
            occurrence = spec.get("occurrence", 6)
            registry.register(
                name,
                lambda s, sub=substring, o=occurrence: index_composition_task(s, sub, o),
            )
        elif task_type == "token_at":
            index = spec.get("index", 40)
            registry.register(name, lambda s, i=index: token_at_index_task(s, i))
        else:
            msg = f"Unknown task type: {task_type}"
            raise ValueError(msg)
    return registry


def resolve_task_weights(
    task_names: list[str],
    mode: str | None,
    operand_probs: dict[str, float],
    explicit_weights: list[float] | None = None,
    special_class: str | None = None,
    special_class_ratio: float | None = None,
) -> list[float]:
    """Resolve task weights based on the specified mode.

    Args:
        task_names: List of task names
        mode: Weight mode ('uniform', 'operand_probs', 'explicit', 'special_ratio')
        operand_probs: Operand probabilities for 'operand_probs' mode
        explicit_weights: Explicit weights for 'explicit' mode
        special_class: Name of special class for 'special_ratio' mode
        special_class_ratio: Ratio for special class (e.g., 0.5 for 50%, 0.1 for 10%)

    Returns:
        List of weights for each task

    """
    if mode in (None, "uniform"):
        return [1.0 for _ in task_names]
    if mode == "operand_probs":
        return build_task_weights(task_names, operand_probs)
    if mode == "explicit":
        if explicit_weights is None or len(explicit_weights) != len(task_names):
            msg = "Explicit task weights must match the task list length."
            raise ValueError(msg)
        return explicit_weights
    if mode == "special_ratio":
        if special_class is None or special_class_ratio is None:
            msg = "special_ratio mode requires special_class and special_class_ratio"
            raise ValueError(msg)
        if special_class not in task_names:
            msg = f"Special class '{special_class}' not in task_names"
            raise ValueError(msg)
        if not 0 <= special_class_ratio <= 1:
            msg = f"special_class_ratio must be between 0 and 1, got {special_class_ratio}"
            raise ValueError(msg)

        # Calculate weights
        weights = []
        num_other_tasks = len(task_names) - 1
        other_weight = (1.0 - special_class_ratio) / num_other_tasks if num_other_tasks > 0 else 0.0

        for task in task_names:
            if task == special_class:
                weights.append(special_class_ratio)
            else:
                weights.append(other_weight)

        return weights
    msg = f"Unknown task weight mode: {mode}"
    raise ValueError(msg)


def get_warmup_steps(
    steps: int,
    warmup_steps: int | None = None,
    warmup_ratio: float | None = None,
) -> int:
    if warmup_steps is not None:
        return warmup_steps
    if warmup_ratio is None:
        return 0
    return int(steps * warmup_ratio)


def build_optimizer(
    model_params,
    optimizer_cfg: dict[str, Any],
    lr: float,
) -> torch.optim.Optimizer:
    opt_type = optimizer_cfg.get("type", "AdamW")
    weight_decay = optimizer_cfg.get("weight_decay", 0.0)
    betas = tuple(optimizer_cfg.get("betas", [0.9, 0.95]))
    eps = optimizer_cfg.get("eps", 1e-8)

    if opt_type == "AdamW":
        return torch.optim.AdamW(
            model_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
    if opt_type == "Adam":
        return torch.optim.Adam(
            model_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )

    msg = f"Unsupported optimizer type: {opt_type}"
    raise ValueError(msg)
