import inspect
import math
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from synthetic.generator import get_vocab_len


def sanity_checks(cfg, loader):
    vocab_len = get_vocab_len(cfg.data.path)
    seq_len = loader.dataset.data.shape[1]

    print("Sequence length: ", seq_len)
    print("Vocabulary length: ", vocab_len)

    # Check if vocabulary size and sequence length are compatible
    assert cfg.net.vocab_size >= vocab_len
    assert cfg.net.context_size >= seq_len
    assert cfg.net.n_embd % cfg.net.n_head == 0

    # Check if BF16 is supported
    if not torch.cuda.is_available():
        warnings.warn("WARNING: running on CPU", UserWarning, stacklevel=2)
    else:
        if not torch.cuda.is_bf16_supported():
            warnings.warn("WARNING: running without BF16", UserWarning, stacklevel=2)

        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise NotImplementedError("Flash Attention requires PyTorch >= 2.0")


# Optimizer
def configure_optimizers(net, optim_cfg):
    # filter out those that do not require grad
    param_dict = dict(net.named_parameters())
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": optim_cfg.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, "
        f"with {num_nodecay_params:,} parameters"
    )

    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and torch.cuda.is_available()
    extra_args = {"fused": True} if use_fused else {}
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=optim_cfg.learning_rate,
        betas=(optim_cfg.beta1, optim_cfg.beta2),
        **extra_args,
    )
    print(f"using fused AdamW: {use_fused}")

    return optimizer


def _cosine_segment(t_frac: float, lr_start: float, lr_end: float) -> float:
    coeff = 0.5 * (1.0 + math.cos(math.pi * t_frac))
    return lr_end + coeff * (lr_start - lr_end)


def phase_lr(
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


def update_phase_lr(
    global_step: int,
    optimizer,
    warmup_steps: int,
    pretrain_steps: int,
    burst_steps: int,
    reversion_steps: int,
    lr_max: float,
    lr_pretrain_end_frac: float,
    lr_burst_end_frac: float,
    lr_reversion_end_frac: float,
) -> float:
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


def update_cosine_warmup_lr(it, cfg, optimizer, total_steps):
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

    # Update learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return it, lr


# Move data
def move_to_device(dat, targets, device):
    if device == "cuda":
        dat = dat.pin_memory().cuda(non_blocking=True)
        targets = targets.pin_memory().cuda(non_blocking=True)

    return dat, targets


# Evaluate
@torch.no_grad()
def evaluate(net, evalLoaders, space_pos, device_info):
    all_loss, all_acc = [], []
    device, dt = device_info
    net.eval()

    for idx, _split in enumerate(("train", "all")):
        loader = evalLoaders[idx]

        sequences, total_loss, total_acc = 0.0, 0.0, 0.0

        for dat, targets in loader:
            dat, targets = move_to_device(dat, targets, device)
            bs = dat.size(0)

            with torch.amp.autocast(device_type=device, dtype=dt):
                logits = net(dat)[:, space_pos:]
                targets = targets[:, space_pos:]

                logits = logits.reshape(-1, logits.size(-1))
                targets = targets.reshape(-1)

                loss = F.cross_entropy(logits, targets)
                total_loss += loss.item() * bs

                acc = logits.argmax(-1) == targets
                total_acc += acc.float().mean().item() * bs

            # Find the last position with label == space_idx
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
def evaluate_freegen(net, evalLoaders, seq_info, device_info, lstm=False):
    all_acc = []
    net.eval()
    device, dt = device_info

    if lstm:
        net.use_hidden = True

    for idx, _split in enumerate(("train", "all")):
        loader = evalLoaders[idx]

        sequences, _total_loss, total_acc = 0.0, 0.0, 0.0

        for dat, targets in loader:
            dat, targets = move_to_device(dat, targets, device)
            bs = dat.size(0)

            with torch.amp.autocast(device_type=device, dtype=dt):
                dat = dat[:, : seq_info["prompt"]]
                output = generate(net, dat, seq_info["new"], lstm)

                output_l = output[:, 1 + seq_info["last_space"] :]
                targets_l = targets[:, seq_info["last_space"] :]

                acc_l = output_l.reshape(-1) == targets_l.reshape(-1)
                total_acc += acc_l.float().mean().item() * bs

            # Find the last position with label == space_idx
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
def generate(net, inp, max_new_tokens, lstm):
    if lstm:
        net.hidden = None
    for _ in range(max_new_tokens):
        logits = net(inp)
        logits = logits[:, -1, :]
        inp_next = torch.argmax(logits, -1, keepdims=True)
        inp = torch.cat((inp, inp_next), dim=1)

    return inp


# Logging functions
def save_model(cfg, net, optimizer, it):
    checkpoint = {
        "net": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter": it,
        "config": cfg,
    }
    fdir = "ckpts/" + cfg.tag
    os.makedirs(fdir, exist_ok=True)
    fname = os.path.join(fdir, "ckpt_" + str(it + 1) + ".pt")
    torch.save(checkpoint, fname)


def log_train(it, lr, train_loss):
    print(f"train -- iter: {it}, lr: {lr:.6f}, loss: {np.mean(train_loss):.4f}")
    return []


def log_eval(it, lr, eval_info, eval_info2=None):
    print("----\nIteration: ", it)
    print(f"Acc (train/all): {eval_info['train_acc']:.3f}/{eval_info['all_acc']:.3f}")
    print(f"loss (train/all): {eval_info['train_loss']:.4f}/{eval_info['all_loss']:.4f}")

    if eval_info2 is not None:
        print(f"acc (train/all): {eval_info2['train_acc']:.4f}/{eval_info2['all_acc']:.4f}")
