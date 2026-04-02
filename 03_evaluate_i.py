"""Evaluate on out-of-order functions."""

import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

from net.lstm import AutoLstm
from net.nanogpt import nanoGPT
from synthetic.generator import SyntheticEval
from synthetic.init import read_config, set_seed

logger = logging.getLogger(__name__)


def load_net(fname: str, *, lstm: bool) -> tuple[nn.Module, DictConfig]:
    """Load a network and its config from a checkpoint file."""
    ckpt = torch.load(fname)
    net_cfg = ckpt["config"]

    net = nanoGPT(net_cfg.net) if not lstm else AutoLstm(net_cfg.net)

    net.load_state_dict(ckpt["net"])
    return net, net_cfg


def fetch_dirs(cfg: DictConfig) -> list[tuple[int, str]]:
    """Fetch and filter checkpoint directories within the configured range."""
    def itr(ck: str) -> int:
        """Extract iteration number from a checkpoint filename."""
        return int(ck.rsplit("_", maxsplit=1)[-1].split(".", maxsplit=1)[0])

    all_dirs = [(itr(str(ck)), str(ck)) for ck in Path("./ckpts", cfg.ckpt_tag).glob("*")]
    all_dirs = sorted(all_dirs)

    reduced_alldirs = []
    for it, cdir in all_dirs:
        if it >= cfg.xlim[0] and it <= cfg.xlim[1]:
            reduced_alldirs.append((it, cdir))

    elems = np.round(np.linspace(0, len(reduced_alldirs) - 1, cfg.nckpts)).astype(int)
    reduced_alldirs = [reduced_alldirs[e] for e in elems]

    return reduced_alldirs


def main(cfg: DictConfig) -> None:
    """Evaluate model accuracy across checkpoints on out-of-order functions."""
    set_seed(cfg.seed)

    sorted_dirs = fetch_dirs(cfg)

    _, net_cfg = load_net(sorted_dirs[0][1], lstm=cfg.lstm)

    evaluator = SyntheticEval(
        net_cfg, cfg.nsamples, cfg.nbatch,
        direct_eval=cfg.direct_eval, permute=cfg.permute,
    )

    accs = []
    for ck in sorted_dirs:
        net, _ = load_net(ck[1], lstm=cfg.lstm)
        mat = evaluator.get_acc(net, lstm=cfg.lstm)
        accs.append((ck, mat))

        acc_vals = np.array(list(mat.values()))
        logger.info("Iter: %s  Acc: %s", ck[0], np.mean(acc_vals))

    evaluator.save_accs(cfg, accs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = read_config("./config/eval/conf.yaml")
    main(cfg)
