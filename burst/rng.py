"""Single shared RNG for the entire project.

Every module that needs randomness calls ``get_rng()`` instead of
maintaining its own module-level ``_rng``.  Call ``seed_all(seed)``
once at process start to seed numpy, stdlib random, torch, and this
shared generator.
"""

import os
import random

import numpy as np
import torch

_state: dict[str, np.random.Generator] = {"rng": np.random.default_rng()}


def get_rng() -> np.random.Generator:
    """Return the shared RNG."""
    return _state["rng"]


def seed_all(seed: int, *, deterministic: bool = True) -> None:
    """Seed every RNG source and configure torch determinism."""
    _state["rng"] = np.random.default_rng(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=not deterministic)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = not deterministic
    torch.backends.cudnn.allow_tf32 = not deterministic
