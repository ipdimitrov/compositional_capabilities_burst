"""Shared helpers for burst/dev/ analysis scripts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from burst.config import SCHED_COLORS, SCHED_DISPLAY, SCHEDULE_ORDER
from burst.core.train_utils import DEVICE

if TYPE_CHECKING:
    from pathlib import Path


def sched_order(s: str) -> int:
    assert s in SCHEDULE_ORDER, f"Unknown schedule: {s}"  # noqa: S101
    return SCHEDULE_ORDER.index(s)


def sched_color(s: str) -> str:
    return SCHED_COLORS.get(s, "#888888")


def sched_label(s: str) -> str:
    return SCHED_DISPLAY.get(s, s)


def ckpt_files(ckpt_dir: Path) -> dict[int, Path]:
    return {int(p.stem.split("_")[1]): p for p in ckpt_dir.glob("step_*.pt")}


def burst_token_ids(cfg: dict, n_a: int, depth: int) -> list[int]:
    n_alphabets = cfg["n_alphabets"]
    vocab_size = cfg["vocab_size"]
    special_count = 3
    alphabet_start = special_count
    func_start = alphabet_start + n_alphabets
    burst_func_id = func_start + n_a * depth + 1
    value_ids = list(range(alphabet_start, alphabet_start + n_alphabets))
    return [i for i in [burst_func_id, *value_ids] if i < vocab_size]


@torch.no_grad()
def free_gen_acc(net: torch.nn.Module, docs_BL: np.ndarray, prompt_len: int) -> float:
    assert docs_BL.shape[0] > 0, "free_gen_acc called with empty docs"  # noqa: S101
    net.eval()
    docs_t = torch.as_tensor(docs_BL, dtype=torch.long, device=DEVICE)
    _B, L = docs_t.shape
    target_B6 = docs_t[:, -6:]
    generated = net.generate(docs_t[:, :prompt_len], L - prompt_len)
    return (generated[:, -6:] == target_B6).all(dim=1).float().mean().item()


@torch.no_grad()
def cross_entropy_loss(net: torch.nn.Module, docs_BL: np.ndarray, max_docs: int = 256) -> float:
    assert docs_BL.shape[0] > 0, "cross_entropy_loss called with empty docs"  # noqa: S101
    net.eval()
    n = min(max_docs, docs_BL.shape[0])
    rng = np.random.default_rng()
    idx = rng.choice(docs_BL.shape[0], n, replace=False)
    dat_BL = torch.as_tensor(docs_BL[idx], dtype=torch.long, device=DEVICE)
    inp_BT, tgt_BT = dat_BL[:, :-1], dat_BL[:, 1:]
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        logits_BTV = net(inp_BT).float()
    return F.cross_entropy(logits_BTV.reshape(-1, logits_BTV.size(-1)), tgt_BT.reshape(-1)).item()


def flat_params(net: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().float().cpu().view(-1) for p in net.parameters()])
