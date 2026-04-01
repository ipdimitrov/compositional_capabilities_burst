"""Load YAML configs via OmegaConf and set deterministic RNG seeds across libraries."""

import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def set_seed(seed: int = 0) -> None:
    """Set global random seeds with scrambled true seed to avoid correlated nearby values."""
    rng = np.random.default_rng(seed)
    true_seed = int(rng.integers(2**30))

    random.seed(true_seed)
    np.random.seed(true_seed)  # noqa: NPY002

    import synthetic.functions as _syn_fn
    import synthetic.generator as _syn_gen

    _shared = np.random.default_rng(true_seed)
    _syn_fn._rng = _shared
    _syn_gen._rng = _shared
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(true_seed)
    torch.cuda.manual_seed_all(true_seed)


def read_config(fname: str | Path) -> DictConfig:
    """Read config from yaml file and log it."""
    with Path(fname).open() as stream:
        cfg = yaml.safe_load(stream)
    logger.info(cfg)
    return OmegaConf.create(cfg)
