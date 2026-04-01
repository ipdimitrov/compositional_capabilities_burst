"""Model utilities for notebook phases: creation, saving, loading, training step, eval.

Delegates to burst.core.train_utils and burst.core.train.worker where possible.
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from burst.core.train_utils import DEVICE, make_net, make_net_bare
from net.nanogpt import nanoGPT

MATRIX_NDIM = 2

MODEL_DEFAULTS = {
    "n_layer": 6,
    "n_embd": 120,
    "n_head": 4,
    "lr": 1e-3,
    "weight_decay": 1e-3,
    "beta1": 0.9,
    "beta2": 0.9,
    "grad_clip": 1.0,
    "warmup_iters": 50,
    "batch_size": 128,
    "eval_every": 25,
}


def make_model(
    vocab_size: int, context_size: int, *,
    n_layer: int = 6, n_embd: int = 120, n_head: int = 4, compile_model: bool = True,
) -> torch.nn.Module:
    cfg = {
        "vocab_size": vocab_size,
        "context_size": context_size,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
    }
    return make_net(cfg) if (compile_model and DEVICE == "cuda") else make_net_bare(cfg)


def save_model(net: torch.nn.Module, path: str | Path) -> None:
    raw = getattr(net, "_orig_mod", net)
    torch.save(raw.state_dict(), path)


def load_model(
    path: str | Path, vocab_size: int, context_size: int, *,
    n_layer: int = 6, n_embd: int = 120, n_head: int = 4, compile_model: bool = True,
) -> torch.nn.Module:
    net = nanoGPT(
        OmegaConf.create({
            "compile": False,
            "vocab_size": vocab_size,
            "context_size": context_size,
            "n_layer": n_layer,
            "n_head": n_head,
            "n_embd": n_embd,
            "dropout": 0.0,
            "bias": False,
            "mlp": True,
        })
    ).to(DEVICE)
    net.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    if compile_model and DEVICE == "cuda":
        net = torch.compile(net)
    return net


def make_optimizer(
    net: torch.nn.Module, lr: float = 1e-3, _weight_decay: float = 1e-3,
    beta1: float = 0.9, beta2: float = 0.9,
) -> torch.optim.Optimizer:
    decay = [p for _, p in net.named_parameters() if p.requires_grad and p.dim() >= MATRIX_NDIM]
    no_decay = [p for _, p in net.named_parameters() if p.requires_grad and p.dim() < MATRIX_NDIM]
    groups = [
        {"params": decay, "weight_decay": 0.0},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(beta1, beta2), fused=(DEVICE == "cuda"))


def reset_optimizer(optimizer: torch.optim.Optimizer) -> None:
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p)
            if state is None:
                continue
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    v.zero_()
                elif k == "step":
                    state[k] = 0


def cosine_lr(step: int, total_steps: int, lr_max: float, lr_min: float, warmup: int = 0) -> float:
    if step <= warmup:
        return lr_max * step / max(warmup, 1)
    t = (step - warmup) / max(total_steps - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def train_step(
    net: torch.nn.Module, optimizer: torch.optim.Optimizer, batch_np: np.ndarray,
    lr: float | None = None, grad_clip: float = 1.0,
) -> float:
    if lr is not None:
        set_lr(optimizer, lr)
    dat = torch.as_tensor(batch_np, dtype=torch.long, device=DEVICE)
    inp, tgt = dat[:, :-1], dat[:, 1:]
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        logits = net(inp)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def eval_accuracy(net: torch.nn.Module, docs_BL: np.ndarray, prompt_len: int) -> float:
    if docs_BL.shape[0] == 0:
        return 0.0
    net.eval()
    n_new = docs_BL.shape[1] - prompt_len
    dat = torch.as_tensor(docs_BL, dtype=torch.long, device=DEVICE)
    correct, total = 0, 0
    bs = 256
    for i in range(0, dat.shape[0], bs):
        chunk = dat[i : i + bs]
        tgt = chunk[:, 1:]
        full = net.generate(chunk[:, :prompt_len], n_new)
        gen = full[:, prompt_len:]
        ref = tgt[:, prompt_len - 1 :]
        ml = min(gen.shape[1], ref.shape[1])
        last6 = max(0, ml - 6)
        correct += (gen[:, last6:ml] == ref[:, last6:ml]).float().sum().item()
        total += ref[:, last6:ml].numel()
    net.train()
    return correct / max(total, 1)


@torch.no_grad()
def eval_loss(net: torch.nn.Module, docs_BL: np.ndarray) -> float:
    if docs_BL.shape[0] == 0:
        return float("nan")
    net.eval()
    dat = torch.as_tensor(docs_BL, dtype=torch.long, device=DEVICE)
    inp, tgt = dat[:, :-1], dat[:, 1:]
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        logits = net(inp)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
    net.train()
    return loss.item()
