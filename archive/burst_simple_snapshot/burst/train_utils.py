"""Shared training utilities: model creation, optimizer setup, training step.

Eliminates duplication across _worker.py, probe.py, and
scripts/probe_next_token_regimes.py.
"""
import pickle, json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from omegaconf import OmegaConf

from synthetic.init import set_seed
from net.nanogpt import nanoGPT
from net.runner import configure_optimizers, phase_lr, update_phase_lr, reset_optimizer_state
from burst._worker import n_target_for_step, sample_batch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


_NET_OMEGACONF_KEYS = ("vocab_size", "context_size", "n_layer", "n_head", "n_embd")


def _net_cfg(cfg: dict) -> OmegaConf:
    return OmegaConf.create({
        "compile": False, **{k: cfg[k] for k in _NET_OMEGACONF_KEYS},
        "dropout": 0.0, "bias": False, "mlp": True,
    })


def make_net_bare(cfg: dict) -> nanoGPT:
    return nanoGPT(_net_cfg(cfg)).to(DEVICE)


def make_net(cfg: dict) -> nanoGPT:
    net = make_net_bare(cfg)
    if DEVICE == "cuda":
        net = torch.compile(net)
    return net


def load_net(cfg: dict, ckpt_path: str) -> nanoGPT:
    net = nanoGPT(_net_cfg(cfg)).to(DEVICE)
    net.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    return net


def make_optim_cfg(cfg: dict) -> OmegaConf:
    return OmegaConf.create({
        "learning_rate": cfg["lr"], "weight_decay": cfg["weight_decay"],
        "beta1": cfg["beta1"], "beta2": cfg["beta2"],
        "grad_clip": cfg["grad_clip"],
    })


def make_scaler() -> torch.amp.GradScaler:
    return torch.amp.GradScaler('cuda', enabled=DEVICE == "cuda")


def train_step(
    batch_np: np.ndarray,
    net: nanoGPT,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    global_step: int,
    cfg: dict,
    grad_clip: float,
) -> float:
    """Single training step with phase-aware LR. Returns loss_value."""
    dat = torch.as_tensor(batch_np, dtype=torch.long, device=DEVICE)
    inp, tgt = dat[:, :-1], dat[:, 1:]
    P = cfg.get("pre_burst_steps", 0)
    T = cfg["total_steps"]
    U = cfg["reversion_steps"]
    update_phase_lr(
        global_step, optimizer, cfg["warmup_iters"],
        P, T, U, cfg["lr"],
        cfg.get("lr_pretrain_end_frac", 0.3),
        cfg.get("lr_burst_end_frac", 0.1),
        cfg.get("lr_reversion_end_frac", 0.01),
    )
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        logits = net(inp)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
    scaler.scale(loss).backward()
    if grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    return loss.item()


def retrain_with_callbacks(
    job: dict,
    target_pool: dict,
    bg_pool: dict,
    on_step: callable = None,
    max_step: int | None = None,
):
    """Retrain a model from scratch, calling on_step(net, global_step, phase) at each step.

    on_step should return True to continue, False to stop early.
    If max_step is given, training stops at that global step.
    """
    seed, cfg, schedule = job["seed"], job["cfg"], job["schedule"]
    set_seed(seed)
    net = make_net(cfg)
    optim_cfg = make_optim_cfg(cfg)
    optimizer = configure_optimizers(net, optim_cfg)
    scaler = make_scaler()

    P = cfg.get("pre_burst_steps", 0)
    T, U = cfg["total_steps"], cfg["reversion_steps"]
    bs, p = cfg["batch_size"], cfg["p_target"]
    effective_max = max_step if max_step is not None else (P + T + U)

    net.train()

    if on_step:
        on_step(net, 0, "init")

    train_end = min(T, max(0, effective_max - P))
    for s in range(train_end):
        nt = n_target_for_step(s, T, schedule, p, bs)
        batch_np, _ = sample_batch(target_pool, bg_pool, nt, bs)
        global_step = P + s + 1
        train_step(batch_np, net, optimizer, scaler, global_step, cfg, cfg["grad_clip"])
        if on_step:
            on_step(net, global_step, "train")

    reset_optimizer_state(optimizer)

    reversion_end = min(U, max(0, effective_max - P - T))
    for s in range(reversion_end):
        batch_np, _ = sample_batch(target_pool, bg_pool, 0, bs)
        global_step = P + T + s + 1
        train_step(batch_np, net, optimizer, scaler, global_step, cfg, cfg["grad_clip"])
        if on_step:
            on_step(net, global_step, "reversion")

    return net


def resolve_run_paths(run_dir) -> tuple[Path, Path, Path]:
    """Return (config_path, logs_dir, results_dir) for both old and new layouts."""
    run_dir = Path(run_dir)
    results_dir = run_dir / "results"
    logs_dir = run_dir / "logs"

    if (results_dir / "config.json").exists():
        cfg_path = results_dir / "config.json"
    else:
        cfg_path = run_dir / "config.json"

    ld = logs_dir if logs_dir.exists() else run_dir
    rd = results_dir if results_dir.exists() else run_dir
    return cfg_path, ld, rd


def load_results(run_dir):
    """Load all_results.pkl and config.json from a run directory.

    Supports both old layout (flat) and new layout (results/ + logs/).
    """
    cfg_path, logs_dir, _ = resolve_run_paths(run_dir)

    pkl_path = logs_dir / "all_results.pkl"
    if not pkl_path.exists():
        pkl_path = Path(run_dir) / "all_results.pkl"

    with open(pkl_path, "rb") as f:
        results = pickle.load(f)
    with open(cfg_path) as f:
        cfg = json.load(f)
    return results, cfg


def pad_to_len(arr: np.ndarray, target_len: int) -> np.ndarray:
    if arr.shape[0] == 0:
        return arr
    if arr.shape[1] >= target_len:
        return arr[:, :target_len]
    pad_w = target_len - arr.shape[1]
    return np.concatenate([arr, np.zeros((arr.shape[0], pad_w), dtype=arr.dtype)], axis=1)


N_PROBE_DOCS_PER_TASK = 200


def build_probe_docs(
    data,
    doc_len: int,
    n_per_task: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build balanced Other/Burst probe datasets from train compositions."""
    other_pool = data.gen_pool(data.other_train[:min(16, len(data.other_train))], n_per_task)
    burst_pool = data.gen_pool(data.burst_train, n_per_task)

    def _cat(pool):
        if not pool:
            return np.zeros((0, doc_len), dtype=np.int64)
        out = np.concatenate(list(pool.values()))
        return pad_to_len(out, doc_len)

    return _cat(other_pool), _cat(burst_pool)


def compute_lr_schedule(cfg: dict, pretrain_steps: int | None = None,
                        burst_steps: int | None = None):
    """Compute three-phase LR schedule arrays. Returns (steps, lrs)."""
    P = pretrain_steps if pretrain_steps is not None else cfg.get("pre_burst_steps", 0)
    T = burst_steps if burst_steps is not None else cfg["total_steps"]
    U = cfg["reversion_steps"]
    warmup = cfg["warmup_iters"]
    lr_max = cfg["lr"]
    lr_pe = cfg.get("lr_pretrain_end_frac", 0.3)
    lr_be = cfg.get("lr_burst_end_frac", 0.1)
    lr_re = cfg.get("lr_reversion_end_frac", 0.01)

    total = P + T + U
    steps = np.arange(1, total + 1)
    lrs = np.array([
        phase_lr(s, warmup, P, T, U, lr_max, lr_pe, lr_be, lr_re)
        for s in steps
    ])
    return steps, lrs
