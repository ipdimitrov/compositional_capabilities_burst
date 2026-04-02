"""Model utilities for notebook phases: creation, saving, loading, training step, eval."""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from burst.config import EVAL_BATCH_SIZE
from burst.core.train_utils import DEVICE, make_net, make_net_bare
from net.nanogpt import nanoGPT


def make_model(  # noqa: PLR0913
    vocab_size: int, context_size: int,
    n_layer: int, n_embd: int, n_head: int, *, compile_model: bool,
) -> torch.nn.Module:
    """Create a nanoGPT model, optionally torch-compiled."""
    cfg = {
        "vocab_size": vocab_size,
        "context_size": context_size,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
    }
    return make_net(cfg) if (compile_model and DEVICE == "cuda") else make_net_bare(cfg)


def save_model(net: torch.nn.Module, path: str | Path) -> None:
    """Save model state dict, unwrapping torch.compile if needed."""
    raw = getattr(net, "_orig_mod", net)
    torch.save(raw.state_dict(), path)


def load_model(  # noqa: PLR0913
    path: str | Path, vocab_size: int, context_size: int,
    n_layer: int, n_embd: int, n_head: int, *, compile_model: bool,
) -> torch.nn.Module:
    """Load a saved nanoGPT checkpoint and optionally compile."""
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
    net: torch.nn.Module, lr: float, beta1: float, beta2: float, weight_decay: float,
) -> torch.optim.Optimizer:
    """Build AdamW with decoupled weight decay."""
    params = [p for p in net.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        params, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay, fused=(DEVICE == "cuda"),
    )


def reset_optimizer(optimizer: torch.optim.Optimizer) -> None:
    """Zero all optimizer state (momentum, step counters)."""
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


def cosine_lr(step: int, total_steps: int, lr_max: float, lr_min: float, warmup: int) -> float:
    """Cosine learning-rate schedule with optional linear warmup."""
    if step <= warmup:
        return lr_max * step / max(warmup, 1)
    t = (step - warmup) / max(total_steps - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Set learning rate for all parameter groups."""
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def train_step(
    net: torch.nn.Module, optimizer: torch.optim.Optimizer, batch_np: np.ndarray,
    lr: float | None, grad_clip: float,
) -> float:
    """Run one forward + backward pass and return the loss scalar."""
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
def eval_accuracy(
    net: torch.nn.Module, docs_BL: np.ndarray, prompt_len: int,
    eval_start: int, eval_end: int,
) -> float:
    """Compute autoregressive accuracy on tokens [eval_start, eval_end)."""
    if docs_BL.shape[0] == 0:
        return 0.0
    net.eval()
    n_new = docs_BL.shape[1] - prompt_len
    dat = torch.as_tensor(docs_BL, dtype=torch.long, device=DEVICE)
    correct, total = 0, 0
    for i in range(0, dat.shape[0], EVAL_BATCH_SIZE):
        chunk = dat[i : i + EVAL_BATCH_SIZE]
        tgt = chunk[:, 1:]
        full = net.generate(chunk[:, :prompt_len], n_new)
        gen = full[:, eval_start:eval_end]
        ref = tgt[:, eval_start - 1 : eval_end - 1]
        correct += (gen == ref).float().sum().item()
        total += ref.numel()
    net.train()
    return correct / max(total, 1)


@torch.no_grad()
def eval_loss(net: torch.nn.Module, docs_BL: np.ndarray) -> float:
    """Compute cross-entropy loss over the full sequence."""
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
