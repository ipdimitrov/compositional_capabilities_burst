"""Model utilities shared across phases: creation, saving, loading, training step, eval."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from net.nanogpt import nanoGPT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── defaults ─────────────────────────────────────────────────────────────
MODEL_DEFAULTS = dict(
    n_layer=6, n_embd=120, n_head=4,
    lr=1e-3, weight_decay=1e-3, beta1=0.9, beta2=0.9,
    grad_clip=1.0, warmup_iters=50, batch_size=128,
    eval_every=25,
)


def make_model(vocab_size, context_size, *, n_layer=6, n_embd=120, n_head=4,
               compile_model=True):
    """Create a fresh nanoGPT."""
    cfg = OmegaConf.create(dict(
        compile=False, vocab_size=vocab_size, context_size=context_size,
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        dropout=0.0, bias=False, mlp=True,
    ))
    net = nanoGPT(cfg).to(DEVICE)
    if compile_model and DEVICE == "cuda":
        net = torch.compile(net)
    return net


def save_model(net, path):
    raw = getattr(net, "_orig_mod", net)
    torch.save(raw.state_dict(), path)


def load_model(path, vocab_size, context_size, *, n_layer=6, n_embd=120,
               n_head=4, compile_model=True):
    """Load a model from a checkpoint."""
    net = nanoGPT(OmegaConf.create(dict(
        compile=False, vocab_size=vocab_size, context_size=context_size,
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        dropout=0.0, bias=False, mlp=True,
    ))).to(DEVICE)
    net.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    if compile_model and DEVICE == "cuda":
        net = torch.compile(net)
    return net


def make_optimizer(net, lr=1e-3, weight_decay=1e-3, beta1=0.9, beta2=0.9):
    decay = [p for _, p in net.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for _, p in net.named_parameters() if p.requires_grad and p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": 0.0}, #weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(beta1, beta2),
                             fused=(DEVICE == "cuda"))


def reset_optimizer(optimizer):
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


# ── LR schedule ──────────────────────────────────────────────────────────

def cosine_lr(step, total_steps, lr_max, lr_min, warmup=0):
    """Single-phase cosine with optional warmup."""
    if step <= warmup:
        return lr_max * step / max(warmup, 1)
    t = (step - warmup) / max(total_steps - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t))


def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ── training step ────────────────────────────────────────────────────────

def train_step(net, optimizer, batch_np, lr=None, grad_clip=1.0):
    """One forward+backward step. Returns loss float."""
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


# ── evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def eval_accuracy(net, docs_BL, prompt_len):
    """Free-generation accuracy on last 6 tokens."""
    if docs_BL.shape[0] == 0:
        return 0.0
    net.eval()
    n_new = docs_BL.shape[1] - prompt_len
    dat = torch.as_tensor(docs_BL, dtype=torch.long, device=DEVICE)
    # process in chunks to avoid OOM
    correct, total = 0, 0
    bs = 256
    for i in range(0, dat.shape[0], bs):
        chunk = dat[i:i + bs]
        tgt = chunk[:, 1:]
        full = net.generate(chunk[:, :prompt_len], n_new)
        gen = full[:, prompt_len:]
        ref = tgt[:, prompt_len - 1:]
        ml = min(gen.shape[1], ref.shape[1])
        last6 = max(0, ml - 6)
        correct += (gen[:, last6:ml] == ref[:, last6:ml]).float().sum().item()
        total += ref[:, last6:ml].numel()
    net.train()
    return correct / max(total, 1)


@torch.no_grad()
def eval_loss(net, docs_BL):
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
