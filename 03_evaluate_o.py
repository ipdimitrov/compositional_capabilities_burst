"""Evaluate on out-of-order functions."""

from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import nn

from net.nanogpt import nanoGPT
from synthetic.generator import SyntheticEvalCombinatorial
from synthetic.init import read_config, set_seed


def load_net(fname: str) -> tuple[nn.Module, DictConfig]:
    """Load a network and its config from a checkpoint file."""
    ckpt = torch.load(fname)
    net_cfg = ckpt["config"]

    net = nanoGPT(net_cfg.net)
    net.load_state_dict(ckpt["net"])
    return net, net_cfg


def fetch_last_ckpt(cfg: DictConfig) -> str:
    """Return the path of the last checkpoint sorted by iteration number."""
    def itr(ck: str) -> int:
        """Extract iteration number from a checkpoint filename."""
        return int((ck.rsplit("_", maxsplit=1)[-1]).split(".", maxsplit=1)[0])

    all_dirs = [(itr(str(ck)), str(ck)) for ck in Path("./ckpts", cfg.ckpt_tag).glob("*")]
    all_dirs = sorted(all_dirs)
    return all_dirs[-1][1]


def main(cfg: DictConfig) -> None:
    """Evaluate combinatorial accuracy on the last checkpoint."""
    set_seed(cfg.seed)

    ckpt_file = fetch_last_ckpt(cfg)
    _, net_cfg = load_net(ckpt_file)

    evaluator = SyntheticEvalCombinatorial(net_cfg, cfg.nsamples, cfg.nbatch)
    net, _ = load_net(ckpt_file)
    mat = evaluator.get_acc(net)
    evaluator.save_accs(cfg, mat)


if __name__ == "__main__":
    cfg = read_config("./config/eval/conf_o.yaml")
    main(cfg)
