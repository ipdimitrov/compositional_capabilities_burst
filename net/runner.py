"""Training and evaluation utilities: optimizers, sanity checks, logging, and model I/O."""

import inspect
import logging
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from synthetic.generator import get_vocab_len

logger = logging.getLogger(__name__)

def sanity_checks(cfg: DictConfig, loader: DataLoader[Any]) -> None:
    """Validate config compatibility with data and hardware."""
    vocab_len = get_vocab_len(cfg.data.path)
    seq_len = loader.dataset.data.shape[1]

    logger.info("Sequence length: %s", seq_len)
    logger.info("Vocabulary length: %s", vocab_len)

    assert cfg.net.vocab_size >= vocab_len  # noqa: S101
    assert cfg.net.context_size >= seq_len  # noqa: S101
    assert cfg.net.n_embd % cfg.net.n_head == 0  # noqa: S101

    if not torch.cuda.is_available():
        warnings.warn("WARNING: running on CPU", UserWarning, stacklevel=2)
    else:
        if not torch.cuda.is_bf16_supported():
            warnings.warn("WARNING: running without BF16", UserWarning, stacklevel=2)

        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            msg = "Flash Attention requires PyTorch >= 2.0"
            raise NotImplementedError(msg)


def configure_optimizers(net: torch.nn.Module, optim_cfg: DictConfig) -> torch.optim.Optimizer:
    """Configure AdamW optimizer with weight decay for matrix params only."""
    param_dict = dict(net.named_parameters())
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]  # noqa: PLR2004  # type: ignore[operator]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]  # noqa: PLR2004  # type: ignore[operator]
    optim_groups = [
        {"params": decay_params, "weight_decay": optim_cfg.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(
        "num decayed parameter tensors: %s, with %s parameters",
        len(decay_params),
        f"{num_decay_params:,}",
    )
    logger.info(
        "num non-decayed parameter tensors: %s, with %s parameters",
        len(nodecay_params),
        f"{num_nodecay_params:,}",
    )

    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and torch.cuda.is_available()
    extra_args = {"fused": True} if use_fused else {}
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=optim_cfg.learning_rate,
        betas=(optim_cfg.beta1, optim_cfg.beta2),
        **extra_args,
    )
    logger.info("using fused AdamW: %s", use_fused)

    return optimizer


def _cosine_segment(t_frac: float, lr_start: float, lr_end: float) -> float:
    """Interpolate between two learning rates with a cosine schedule."""
    coeff = 0.5 * (1.0 + math.cos(math.pi * t_frac))
    return lr_end + coeff * (lr_start - lr_end)


def phase_lr(  # noqa: PLR0913
    global_step: int,
    warmup_steps: int,
    pretrain_steps: int,
    burst_steps: int,
    reversion_steps: int,
    lr_max: float,
    lr_pretrain_end_frac: float,
    lr_burst_end_frac: float,
    lr_reversion_end_frac: float,
) -> float:
    """Three-phase cosine LR with a single linear warmup at the start.

    global_step is 1-indexed.
    Phase boundaries:
      [1, pretrain_steps]                                   pretrain
      [pretrain_steps+1, pretrain_steps+burst_steps]        burst
      [pretrain_steps+burst_steps+1, ...]                   reversion
    """
    P, T, U = pretrain_steps, burst_steps, reversion_steps
    lr_pretrain_end = lr_max * lr_pretrain_end_frac
    lr_burst_end = lr_max * lr_burst_end_frac
    lr_reversion_end = lr_max * lr_reversion_end_frac

    if global_step <= P:
        if global_step <= warmup_steps:
            return lr_max * global_step / warmup_steps
        t_frac = (global_step - warmup_steps) / max(P - warmup_steps, 1)
        return _cosine_segment(t_frac, lr_max, lr_pretrain_end)

    if global_step <= P + T:
        t_frac = (global_step - P) / max(T, 1)
        return _cosine_segment(t_frac, lr_pretrain_end, lr_burst_end)

    t_frac = (global_step - P - T) / max(U, 1)
    return _cosine_segment(t_frac, lr_burst_end, lr_reversion_end)


def reset_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    """Zero all momentum / variance buffers (Adam state) without changing param groups or LR."""
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


def update_phase_lr(  # noqa: PLR0913
    global_step: int,
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    pretrain_steps: int,
    burst_steps: int,
    reversion_steps: int,
    lr_max: float,
    lr_pretrain_end_frac: float,
    lr_burst_end_frac: float,
    lr_reversion_end_frac: float,
) -> float:
    """Compute phase-based LR and apply it to all optimizer param groups."""
    lr = phase_lr(
        global_step,
        warmup_steps,
        pretrain_steps,
        burst_steps,
        reversion_steps,
        lr_max,
        lr_pretrain_end_frac,
        lr_burst_end_frac,
        lr_reversion_end_frac,
    )
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def update_cosine_warmup_lr(
    it: int,
    cfg: DictConfig,
    optimizer: torch.optim.Optimizer,
    total_steps: int,
) -> tuple[int, float]:
    """Update learning rate with cosine decay and linear warmup."""
    it += 1
    lr = cfg.learning_rate

    if cfg.decay_lr:
        if it < cfg.warmup_iters:
            lr = lr * (it) / cfg.warmup_iters
        else:
            num = it - cfg.warmup_iters
            decay_ratio = num / (total_steps - cfg.warmup_iters)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = cfg.min_lr + coeff * (lr - cfg.min_lr)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return it, lr


def move_to_device(
    dat: torch.Tensor, targets: torch.Tensor, device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move data and targets to the specified device."""
    if device == "cuda":
        dat = dat.pin_memory().cuda(non_blocking=True)
        targets = targets.pin_memory().cuda(non_blocking=True)

    return dat, targets


@torch.no_grad()
def evaluate(
    net: torch.nn.Module,
    evalLoaders: list[DataLoader[Any]],
    space_pos: int,
    device_info: tuple[str, torch.dtype],
) -> dict[str, float]:
    """Compute loss and accuracy on train and full eval splits."""
    all_loss, all_acc = [], []
    device, dt = device_info
    net.eval()

    for idx, _split in enumerate(("train", "all")):
        loader = evalLoaders[idx]

        sequences, total_loss, total_acc = 0.0, 0.0, 0.0

        for dat, targets in loader:
            dat_d, targets_d = move_to_device(dat, targets, device)
            bs = dat_d.size(0)

            with torch.amp.autocast(device_type=device, dtype=dt):
                logits_out = net(dat_d)[:, space_pos:]
                targets_out = targets_d[:, space_pos:]

                logits_flat = logits_out.reshape(-1, logits_out.size(-1))
                targets_flat = targets_out.reshape(-1)

                loss = F.cross_entropy(logits_flat, targets_flat)
                total_loss += loss.item() * bs

                acc = logits_flat.argmax(-1) == targets_flat
                total_acc += acc.float().mean().item() * bs

            sequences += bs

        if sequences == 0:
            all_loss.append(float("inf"))
            all_acc.append(float("inf"))
        else:
            all_loss.append(total_loss / sequences)
            all_acc.append(total_acc / sequences)

    info = {
        "train_loss": all_loss[0],
        "train_acc": all_acc[0],
        "all_loss": all_loss[1],
        "all_acc": all_acc[1],
    }

    net.train()
    return info


@torch.no_grad()
def evaluate_freegen(
    net: torch.nn.Module,
    evalLoaders: list[DataLoader[Any]],
    seq_info: dict[str, int],
    device_info: tuple[str, torch.dtype],
    *,
    lstm: bool = False,
) -> dict[str, float]:
    """Compute autoregressive generation accuracy on train and full eval splits."""
    all_acc = []
    net.eval()
    device, dt = device_info

    if lstm:
        net.use_hidden = True

    for idx, _split in enumerate(("train", "all")):
        loader = evalLoaders[idx]

        sequences, _total_loss, total_acc = 0.0, 0.0, 0.0

        for dat, targets in loader:
            dat_d, targets_d = move_to_device(dat, targets, device)
            bs = dat_d.size(0)

            with torch.amp.autocast(device_type=device, dtype=dt):
                prompt = dat_d[:, : seq_info["prompt"]]
                output = generate(net, prompt, seq_info["new"], lstm=lstm)

                output_l = output[:, 1 + seq_info["last_space"] :]
                targets_l = targets_d[:, seq_info["last_space"] :]

                acc_l = output_l.reshape(-1) == targets_l.reshape(-1)
                total_acc += acc_l.float().mean().item() * bs

            sequences += bs

        if sequences == 0:
            all_acc.append(float("inf"))
        else:
            all_acc.append(total_acc / sequences)

    if lstm:
        net.use_hidden = False
    info = {"train_acc": all_acc[0], "all_acc": all_acc[1]}

    net.train()
    return info


@torch.no_grad()
def generate(
    net: torch.nn.Module,
    inp: torch.Tensor,
    max_new_tokens: int,
    *,
    lstm: bool,
) -> torch.Tensor:
    """Generate tokens autoregressively by greedy decoding."""
    if lstm:
        net.hidden = None
    for _ in range(max_new_tokens):
        logits = net(inp)
        logits = logits[:, -1, :]
        inp_next = torch.argmax(logits, -1, keepdims=True)
        inp = torch.cat((inp, inp_next), dim=1)

    return inp


def save_model(
    cfg: DictConfig,
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    it: int,
) -> None:
    """Save model checkpoint to disk."""
    checkpoint = {
        "net": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter": it,
        "config": cfg,
    }
    fdir = Path("ckpts") / cfg.tag
    fdir.mkdir(parents=True, exist_ok=True)
    fname = fdir / ("ckpt_" + str(it + 1) + ".pt")
    torch.save(checkpoint, fname)


def log_train(it: int, lr: float, train_loss: list[float]) -> list[float]:
    """Log training iteration metrics and return an empty loss buffer."""
    logger.info("train -- iter: %s, lr: %.6f, loss: %.4f", it, lr, np.mean(train_loss))
    return []


def log_eval(
    it: int,
    _lr: float,
    eval_info: dict[str, float],
    eval_info2: dict[str, float] | None = None,
) -> None:
    """Log evaluation metrics for loss and accuracy."""
    logger.info("----\nIteration: %s", it)
    logger.info(
        "Acc (train/all): %.3f/%.3f", eval_info["train_acc"], eval_info["all_acc"]
    )
    logger.info(
        "loss (train/all): %.4f/%.4f",
        eval_info["train_loss"],
        eval_info["all_loss"],
    )

    if eval_info2 is not None:
        logger.info(
            "acc (train/all): %.4f/%.4f",
            eval_info2["train_acc"],
            eval_info2["all_acc"],
        )
