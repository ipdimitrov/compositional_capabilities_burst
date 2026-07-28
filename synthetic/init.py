"""Load YAML configs via OmegaConf and set deterministic RNG seeds across libraries."""

import logging
from pathlib import Path

import numpy as np
import yaml
from omegaconf import DictConfig, OmegaConf

from burst.rng import seed_all

logger = logging.getLogger(__name__)


def set_seed(seed: int = 0) -> None:
    """Scramble seed to decorrelate nearby values, then seed everything."""
    true_seed = int(np.random.default_rng(seed).integers(2**30))
    seed_all(true_seed)


def read_config(fname: str | Path) -> DictConfig:
    """Read config from yaml file and log it."""
    with Path(fname).open() as stream:
        cfg = yaml.safe_load(stream)
    logger.info(cfg)
    return OmegaConf.create(cfg)
