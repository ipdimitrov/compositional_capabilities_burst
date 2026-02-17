import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy
import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict
from omegaconf import OmegaConf

from net.nanogpt import nanoGPT
from net.runner import configure_optimizers, update_cosine_warmup_lr
from burst.config import BurstExperimentConfig
from burst.data import (
    build_function_pool, tag_tasks, generate_pool,
    ScheduleSampler, StaggeredSampler, BurstDataset,
)


def make_model(cfg: BurstExperimentConfig, device: str):
    net_cfg = OmegaConf.create({
        "compile": cfg.net.compile,
        "vocab_size": cfg.net.vocab_size,
        "context_size": cfg.net.context_size,
        "n_layer": cfg.net.n_layer,
        "n_head": cfg.net.n_head,
        "n_embd": cfg.net.n_embd,
        "dropout": cfg.net.dropout,
        "bias": cfg.net.bias,
        "mlp":  cfg.net.mlp,
    })
    net = nanoGPT(net_cfg)
    net.to(device)
    return net


def make_optimizer(net, cfg: BurstExperimentConfig):
    optim_cfg = OmegaConf.create({
        "learning_rate": cfg.optimizer.learning_rate,
        "weight_decay": cfg.optimizer.weight_decay,
        "beta1": cfg.optimizer.beta1,
        "beta2": cfg.optimizer.beta2,
        "grad_clip": cfg.optimizer.grad_clip,
        "decay_lr": cfg.optimizer.decay_lr,
        "warmup_iters": cfg.optimizer.warmup_iters,
        "min_lr": cfg.optimizer.min_lr,
    })
    return configure_optimizers(net, optim_cfg)


def eval_accuracy(net, eval_docs_BL: np.ndarray, space_pos: int,
                  device: str) -> float:
    net.eval()
    ds = BurstDataset(eval_docs_BL)
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False)
    total_correct, total_tokens = 0, 0
    with torch.no_grad():
        for dat, targets in loader:
            dat = dat.to(device)
            targets = targets.to(device)
            logits = net(dat)[:, space_pos:]
            tgt = targets[:, space_pos:]
            preds = logits.argmax(-1)
            total_correct += (preds == tgt).float().sum().item()
            total_tokens += tgt.numel()
    net.train()
    return total_correct / max(total_tokens, 1)


def snapshot_weights(net) -> OrderedDict:
    return OrderedDict(
        (name, param.detach().cpu().clone())
        for name, param in net.named_parameters()
    )


def weight_deltas_frobenius(w_before: OrderedDict,
                            w_after: OrderedDict) -> dict[str, float]:
    deltas = {}
    for name in w_before:
        diff = w_after[name] - w_before[name]
        deltas[name] = diff.norm().item()
    return deltas


def train_phase(net, optimizer, sampler, device: str,
                n_steps: int, total_steps_for_lr: int,
                schedule: str, p: float, K: int,
                eval_docs: dict[str, np.ndarray],
                space_pos: int, eval_every: int,
                cfg: BurstExperimentConfig,
                it_start: int = 0,
                phase_label: str = "train",
                track_weights: bool = True,
                fixed_lr: float = None):
    optim_cfg = OmegaConf.create({
        "learning_rate": cfg.optimizer.learning_rate,
        "weight_decay": cfg.optimizer.weight_decay,
        "beta1": cfg.optimizer.beta1,
        "beta2": cfg.optimizer.beta2,
        "grad_clip": cfg.optimizer.grad_clip,
        "decay_lr": cfg.optimizer.decay_lr,
        "warmup_iters": cfg.optimizer.warmup_iters,
        "min_lr": cfg.optimizer.min_lr,
    })

    w_start = snapshot_weights(net) if track_weights else None

    net.train()
    log = {"step": [], "loss": []}
    for name in eval_docs:
        log[f"acc_{name}"] = []
    if track_weights:
        log["weight_deltas"] = []

    it = it_start
    for s in range(n_steps):
        batch_np = sampler.sample_batch(
            step=s, total_steps=n_steps, schedule=schedule, p=p, K=K)
        dat_BL = torch.from_numpy(batch_np).long().to(device)
        inp, tgt = dat_BL[:, :-1], dat_BL[:, 1:]

        if fixed_lr is not None:
            lr = fixed_lr
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            it += 1
        else:
            it, lr = update_cosine_warmup_lr(it, optim_cfg, optimizer, total_steps_for_lr)

        optimizer.zero_grad(set_to_none=True)
        logits = net(inp)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        loss.backward()
        if cfg.optimizer.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.optimizer.grad_clip)
        optimizer.step()

        if s % eval_every == 0 or s == n_steps - 1:
            log["step"].append(it)
            log["loss"].append(loss.item())
            for name, docs in eval_docs.items():
                acc = eval_accuracy(net, docs, space_pos, device)
                log[f"acc_{name}"].append(acc)
            if track_weights:
                w_now = snapshot_weights(net)
                log["weight_deltas"].append(
                    weight_deltas_frobenius(w_start, w_now))
            net.train()

    return log, it


def get_space_pos(eval_docs: dict[str, np.ndarray], syn) -> int:
    sample = list(eval_docs.values())[0][0]
    sp_idx = syn.token_idx[" "]
    sp_pos = np.where(sample == sp_idx)[0][-1]
    return sp_pos


def run_idea2_condition(cfg: BurstExperimentConfig, schedule: str,
                        K: int = 1, seed: int = 0,
                        n_target: int = 10) -> dict:
    from synthetic.init import set_seed
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    syn, composed_functions, info = build_function_pool(cfg)
    target_ids, bg_ids, fn_lookup = tag_tasks(info, composed_functions, n_target=n_target)

    n_docs_per_task = max(cfg.ndocuments // max(len(bg_ids), 1), 200)
    target_pool = generate_pool(syn, target_ids, fn_lookup, n_docs_per_task)
    bg_pool = generate_pool(syn, bg_ids, fn_lookup, n_docs_per_task)

    eval_target = generate_pool(syn, target_ids, fn_lookup, cfg.neval_documents)
    eval_bg = generate_pool(syn, bg_ids[:5], fn_lookup, cfg.neval_documents // 5)
    eval_target_flat = np.concatenate(list(eval_target.values()))
    eval_bg_flat = np.concatenate(list(eval_bg.values()))
    eval_docs = {"target": eval_target_flat, "background": eval_bg_flat}

    space_pos = get_space_pos(eval_docs, syn)

    sampler = ScheduleSampler(target_pool, bg_pool, cfg.batch_size)
    net = make_model(cfg, device)
    optimizer = make_optimizer(net, cfg)

    total_lr_steps = cfg.total_steps + cfg.undo_steps + cfg.relearn_steps

    train_log, it = train_phase(
        net, optimizer, sampler, device,
        n_steps=cfg.total_steps, total_steps_for_lr=total_lr_steps,
        schedule=schedule, p=cfg.p_target, K=K,
        eval_docs=eval_docs, space_pos=space_pos,
        eval_every=cfg.eval_every, cfg=cfg, it_start=0,
        phase_label="train", track_weights=True)

    undo_log, it = train_phase(
        net, optimizer, sampler, device,
        n_steps=cfg.undo_steps, total_steps_for_lr=total_lr_steps,
        schedule="undo", p=0.0, K=1,
        eval_docs=eval_docs, space_pos=space_pos,
        eval_every=cfg.eval_every, cfg=cfg, it_start=it,
        phase_label="undo", track_weights=True)

    relearn_log, it = train_phase(
        net, optimizer, sampler, device,
        n_steps=cfg.relearn_steps, total_steps_for_lr=total_lr_steps,
        schedule="relearn", p=cfg.p_relearn, K=1,
        eval_docs=eval_docs, space_pos=space_pos,
        eval_every=cfg.eval_every, cfg=cfg, it_start=it,
        phase_label="relearn", track_weights=True)

    return {
        "schedule": schedule, "K": K, "seed": seed,
        "train": train_log, "undo": undo_log, "relearn": relearn_log,
    }


def run_idea3(cfg, seed: int = 0) -> dict:
    from synthetic.init import set_seed
    from burst.config import Idea3Config
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    syn, composed_functions, info = build_function_pool(cfg)
    train_ids = [tuple(t) for t in info["train_id"]]
    fn_lookup = {}
    for fn_tuple in composed_functions["train"]:
        fn_lookup[tuple(fn_tuple[0])] = fn_tuple

    n_target = cfg.n_target_tasks
    target_ids = train_ids[:n_target]
    bg_ids = train_ids[n_target:]

    task_names = ["F1_early", "F2_mid", "F3_late", "F4_mixed"]
    n_docs = max(cfg.ndocuments // max(len(bg_ids), 1), 200)

    task_pools = {}
    eval_docs = {}
    for i, name in enumerate(task_names):
        tid = target_ids[i]
        pool = generate_pool(syn, [tid], fn_lookup, n_docs)
        task_pools[name] = pool
        ev = generate_pool(syn, [tid], fn_lookup, cfg.neval_documents)
        eval_docs[name] = np.concatenate(list(ev.values()))

    bg_pool = generate_pool(syn, bg_ids, fn_lookup, n_docs)
    eval_bg = generate_pool(syn, bg_ids[:5], fn_lookup, cfg.neval_documents // 5)
    eval_docs["background"] = np.concatenate(list(eval_bg.values()))

    space_pos = get_space_pos(eval_docs, syn)

    staggered = StaggeredSampler(
        task_pools, bg_pool, cfg.batch_size,
        cfg.total_steps, cfg.p_per_task)

    net = make_model(cfg, device)
    optimizer = make_optimizer(net, cfg)

    optim_cfg = OmegaConf.create({
        "learning_rate": cfg.optimizer.learning_rate,
        "weight_decay": cfg.optimizer.weight_decay,
        "beta1": cfg.optimizer.beta1,
        "beta2": cfg.optimizer.beta2,
        "grad_clip": cfg.optimizer.grad_clip,
        "decay_lr": cfg.optimizer.decay_lr,
        "warmup_iters": cfg.optimizer.warmup_iters,
        "min_lr": cfg.optimizer.min_lr,
    })

    total_lr_steps = cfg.total_steps + cfg.undo_steps

    net.train()
    train_log = {"step": [], "loss": []}
    for name in eval_docs:
        train_log[f"acc_{name}"] = []

    it = 0
    for s in range(cfg.total_steps):
        batch_np = staggered.sample_batch(s, phase="train")
        dat_BL = torch.from_numpy(batch_np).long().to(device)
        inp, tgt = dat_BL[:, :-1], dat_BL[:, 1:]

        it, lr = update_cosine_warmup_lr(it, optim_cfg, optimizer, total_lr_steps)
        optimizer.zero_grad(set_to_none=True)
        logits = net(inp)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        loss.backward()
        if cfg.optimizer.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.optimizer.grad_clip)
        optimizer.step()

        if s % cfg.eval_every == 0 or s == cfg.total_steps - 1:
            train_log["step"].append(it)
            train_log["loss"].append(loss.item())
            for name, docs in eval_docs.items():
                acc = eval_accuracy(net, docs, space_pos, device)
                train_log[f"acc_{name}"].append(acc)
            net.train()

    undo_lr = cfg.optimizer.learning_rate * 0.3
    undo_log = {"step": [], "loss": []}
    for name in eval_docs:
        undo_log[f"acc_{name}"] = []

    for s in range(cfg.undo_steps):
        batch_np = staggered.sample_batch(s, phase="undo")
        dat_BL = torch.from_numpy(batch_np).long().to(device)
        inp, tgt = dat_BL[:, :-1], dat_BL[:, 1:]

        for pg in optimizer.param_groups:
            pg["lr"] = undo_lr
        it += 1
        optimizer.zero_grad(set_to_none=True)
        logits = net(inp)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        loss.backward()
        if cfg.optimizer.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.optimizer.grad_clip)
        optimizer.step()

        if s % cfg.eval_every == 0 or s == cfg.undo_steps - 1:
            undo_log["step"].append(it)
            undo_log["loss"].append(loss.item())
            for name, docs in eval_docs.items():
                acc = eval_accuracy(net, docs, space_pos, device)
                undo_log[f"acc_{name}"].append(acc)
            net.train()

    return {
        "seed": seed,
        "train": train_log,
        "undo": undo_log,
    }
