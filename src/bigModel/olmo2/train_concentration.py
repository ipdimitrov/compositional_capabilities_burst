#!/usr/bin/env python3
"""
Two-stage concentration experiment on OLMo-2-1B.

Stage 1 (sweep): Fine-tune with a mix of code + pretraining data.
  concentration c = fraction of CODE data in each batch.
  c=1.0 → pure code, c=0.5 → half code half pretrain, c=0.1 → mostly pretrain.
  Each (c, lr, seed) produces a separate fine-tuned checkpoint.

Stage 2 (fixed): Continue training on 100% pretraining data.
  All stage-2 runs use the same setup — pure pretraining.
  Measures how fast each stage-1 model forgets code.
  Code val loss should rise (forgetting), pretrain val loss should fall (recovery).

The key question: does the stage-1 concentration affect how *durable*
the learned code capability is when subsequently exposed to pretraining data?

Usage:
    # Stage 1 only (one condition)
    python olmo2/train_concentration.py --stage 1 --concentration 0.5 --lr 5e-5

    # Stage 2 only (loads stage-1 checkpoint, trains on pure pretrain)
    python olmo2/train_concentration.py --stage 2 --concentration 0.5 --lr 5e-5

    # Both stages in sequence
    python olmo2/train_concentration.py --stage both --concentration 0.5 --lr 5e-5
"""

import argparse
import csv
import json
import os
import random
import sys
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_MODEL_ID = "allenai/OLMo-2-0425-1B"
SEED = 42

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Stage1Config:
    """Stage 1: Code fine-tuning at varying concentrations."""
    steps: int = 2000
    batch_size: int = 4
    grad_accum_steps: int = 16
    lr: float = 5e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    clip_norm: float = 1.0
    measure_every: int = 50
    checkpoint_every: int = 500
    # Concentration: fraction of CODE data in batch
    # c=1.0 → all code, c=0.5 → half code half pretrain, c=0.0 → all pretrain
    concentration: float = 1.0

@dataclass
class Stage2Config:
    """Stage 2: Pure pretraining (forgetting phase). Same for all runs."""
    steps: int = 2000
    batch_size: int = 4
    grad_accum_steps: int = 16
    lr: float = 5e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    clip_norm: float = 1.0
    measure_every: int = 50
    checkpoint_every: int = 500

@dataclass
class DataConfig:
    seq_len: int = 2048
    min_seq_len: int = 64
    # Pretraining data (C4 as proxy for OLMo-mix-1124)
    pretrain_dataset: str = "allenai/c4"
    pretrain_subset: str = "en"
    pretrain_split: str = "train"
    pretrain_text_field: str = "text"
    pretrain_train_docs: int = 50_000
    pretrain_eval_docs: int = 5_000
    # Code data (accept license at https://huggingface.co/datasets/bigcode/starcoderdata)
    code_dataset: str = "bigcode/starcoderdata"
    code_subset: str = "python"
    code_split: str = "train"
    code_text_field: str = "content"
    code_train_docs: int = 200_000
    code_eval_docs: int = 5_000


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

class TokenListDataset(Dataset):
    def __init__(self, token_lists: List[List[int]]):
        self.token_lists = token_lists
    def __len__(self):
        return len(self.token_lists)
    def __getitem__(self, idx):
        return torch.tensor(self.token_lists[idx], dtype=torch.long)


def collate_pad(batch: List[torch.Tensor], pad_id: int, max_len: int):
    bsz = len(batch)
    out = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((bsz, max_len), dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    for i, t in enumerate(batch):
        t = t[:max_len]
        out[i, :t.numel()] = t
        attn[i, :t.numel()] = 1
        labels[i, :t.numel()] = t
    return {"input_ids": out, "attention_mask": attn, "labels": labels}


def tokenize_and_pack(tokenizer, texts, seq_len, min_seq_len, max_docs, desc="tokenize"):
    all_ids: List[int] = []
    doc_count = 0
    eos_id = tokenizer.eos_token_id or 0
    for text in tqdm(texts, desc=desc, total=max_docs, leave=False):
        if doc_count >= max_docs:
            break
        if not text or len(text.strip()) < 20:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < min_seq_len:
            continue
        all_ids.extend(ids)
        all_ids.append(eos_id)
        doc_count += 1
    sequences = []
    for i in range(0, len(all_ids) - seq_len + 1, seq_len):
        sequences.append(all_ids[i:i + seq_len])
    logger.info(f"  {desc}: {doc_count} docs -> {len(all_ids):,} tokens -> {len(sequences)} seqs of {seq_len}")
    return sequences


def load_streaming_texts(dataset_id, subset, split, text_field, max_docs, seed=42):
    logger.info(f"Streaming from {dataset_id}/{subset} split={split} (up to {max_docs} docs)...")
    ds = None

    # Strategy 1: load parquet files directly via data_dir (works for starcoderdata etc.)
    if subset and subset != "None":
        try:
            ds = load_dataset(dataset_id, data_dir=subset, split=split, streaming=True)
            ds = ds.shuffle(seed=seed, buffer_size=10_000)
            logger.info(f"  Loaded via data_dir={subset}")
        except Exception as e:
            logger.debug(f"  data_dir approach failed: {e}")
            ds = None

    # Strategy 2: load with subset as config name
    if ds is None and subset and subset != "None":
        try:
            ds = load_dataset(dataset_id, name=subset, split=split, streaming=True)
            ds = ds.shuffle(seed=seed, buffer_size=10_000)
            logger.info(f"  Loaded via name={subset}")
        except Exception as e:
            logger.debug(f"  name= approach failed: {e}")
            ds = None

    # Strategy 3: load without subset
    if ds is None:
        try:
            ds = load_dataset(dataset_id, split=split, streaming=True)
            ds = ds.shuffle(seed=seed, buffer_size=10_000)
            logger.info(f"  Loaded without subset")
        except Exception as e:
            logger.debug(f"  no-subset approach failed: {e}")
            ds = None

    if ds is None:
        raise RuntimeError(f"Could not load {dataset_id}/{subset}")

    count = 0
    for example in ds:
        if count >= max_docs:
            break
        text = example.get(text_field, "")
        if text:
            yield text
            count += 1


# ---------------------------------------------------------------------------
# Probe / measurement helpers
# ---------------------------------------------------------------------------

def get_probe_params(model) -> Dict[str, torch.nn.Parameter]:
    named = dict(model.named_parameters())
    probe: Dict[str, torch.nn.Parameter] = {}
    i = 0
    while True:
        candidates = [
            (f"model.layers.{i}.self_attn.o_proj.weight", f"layer_{i}.attn_o"),
            (f"model.layers.{i}.mlp.down_proj.weight", f"layer_{i}.mlp_down"),
            (f"model.transformer.blocks.{i}.att_proj.weight", f"layer_{i}.att_proj"),
            (f"model.transformer.blocks.{i}.ff_proj.weight", f"layer_{i}.ff_proj"),
            (f"model.transformer.blocks.{i}.attn_out.weight", f"layer_{i}.attn_out"),
            (f"model.transformer.blocks.{i}.ff_out.weight", f"layer_{i}.ff_out"),
        ]
        found = False
        for full_name, short_name in candidates:
            if full_name in named:
                probe[short_name] = named[full_name]
                found = True
        if not found:
            if i > 0:
                break
            i += 1
            if i > 100:
                break
            continue
        i += 1
    if not probe:
        cands = [k for k in named if "layers." in k and k.endswith(".weight")]
        for k in cands[:64]:
            probe[k] = named[k]
    logger.info(f"Probing {len(probe)} parameters")
    return probe


def snapshot_weights(params):
    return {k: v.detach().float().cpu().clone().flatten() for k, v in params.items()}


def compute_weight_drift(params, prev):
    abs_d, rel_d = {}, {}
    for k, p in params.items():
        cur = p.detach().float().cpu().flatten()
        delta = cur - prev[k]
        abs_delta = torch.norm(delta).item()
        prv_norm = torch.norm(prev[k]).item()
        abs_d[k] = abs_delta
        rel_d[k] = abs_delta / (prv_norm + 1e-12)
    return abs_d, rel_d


def cossim_flat(a, b):
    a, b = a.float(), b.float()
    denom = (torch.norm(a) * torch.norm(b)).item()
    return (torch.dot(a, b) / denom).item() if denom > 1e-12 else 0.0


def concat_grads(grads):
    return torch.cat([grads[k] for k in sorted(grads.keys())])


def compute_grad_for_batch(model, batch, params, device):
    model.zero_grad(set_to_none=True)
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    labs = batch["labels"].to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        loss = model(input_ids=ids, attention_mask=mask, labels=labs).loss
    loss.backward()
    grads = {}
    for k, p in params.items():
        if p.grad is not None:
            grads[k] = p.grad.detach().float().cpu().clone().flatten()
        else:
            grads[k] = torch.zeros(p.numel(), dtype=torch.float32)
    model.zero_grad(set_to_none=True)
    return grads


def mean_grad_over_loader(model, loader, params, device, max_batches=None):
    sums = {k: None for k in params}
    n = 0
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        g = compute_grad_for_batch(model, batch, params, device)
        for k in params:
            sums[k] = g[k] if sums[k] is None else (sums[k] + g[k])
        n += 1
        del g
    return {k: (sums[k] / max(n, 1)) for k in params}


@torch.no_grad()
def compute_val_loss(model, loader, device, max_batches=16):
    model.eval()
    total, n = 0.0, 0
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labs = batch["labels"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            loss = model(input_ids=ids, attention_mask=mask, labels=labs).loss
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_data(dcfg: DataConfig, tokenizer, seed: int):
    """Load and tokenize both datasets. Returns (pretrain_train, pretrain_eval, code_train, code_eval)."""
    logger.info("=== Loading pretraining data ===")
    pretrain_texts = load_streaming_texts(
        dcfg.pretrain_dataset, dcfg.pretrain_subset, dcfg.pretrain_split,
        dcfg.pretrain_text_field,
        max_docs=dcfg.pretrain_train_docs + dcfg.pretrain_eval_docs, seed=seed,
    )
    pretrain_seqs = tokenize_and_pack(
        tokenizer, pretrain_texts, dcfg.seq_len, dcfg.min_seq_len,
        max_docs=dcfg.pretrain_train_docs + dcfg.pretrain_eval_docs, desc="pretrain",
    )
    rng = random.Random(seed)
    rng.shuffle(pretrain_seqs)
    n_eval = max(100, len(pretrain_seqs) // 10)
    pretrain_eval = pretrain_seqs[:n_eval]
    pretrain_train = pretrain_seqs[n_eval:]

    logger.info("=== Loading code data ===")
    code_texts = load_streaming_texts(
        dcfg.code_dataset, dcfg.code_subset, dcfg.code_split,
        dcfg.code_text_field,
        max_docs=dcfg.code_train_docs + dcfg.code_eval_docs, seed=seed + 1,
    )
    code_seqs = tokenize_and_pack(
        tokenizer, code_texts, dcfg.seq_len, dcfg.min_seq_len,
        max_docs=dcfg.code_train_docs + dcfg.code_eval_docs, desc="code",
    )
    rng2 = random.Random(seed + 1)
    rng2.shuffle(code_seqs)
    n_code_eval = max(100, len(code_seqs) // 10)
    code_eval = code_seqs[:n_code_eval]
    code_train = code_seqs[n_code_eval:]

    logger.info(f"Pretrain: {len(pretrain_train)} train, {len(pretrain_eval)} eval")
    logger.info(f"Code:     {len(code_train)} train, {len(code_eval)} eval")

    return (
        TokenListDataset(pretrain_train), TokenListDataset(pretrain_eval),
        TokenListDataset(code_train), TokenListDataset(code_eval),
    )


# ---------------------------------------------------------------------------
# Mixed batch iterator
# ---------------------------------------------------------------------------

class MixedBatchIterator:
    """
    Yields mixed batches.
    code_fraction = fraction of CODE data in each batch.
    code_fraction=1.0 → all code, 0.0 → all pretrain.
    """
    def __init__(self, code_loader, pretrain_loader, code_fraction, seed=42):
        self.code_loader = code_loader
        self.pretrain_loader = pretrain_loader
        self.code_fraction = code_fraction
        self.rng = random.Random(seed)
        self._code_iter = iter(code_loader)
        self._pretrain_iter = iter(pretrain_loader)

    def _next(self, it, loader):
        try:
            return next(it), it
        except StopIteration:
            it = iter(loader)
            return next(it), it

    def get_batch(self):
        c = self.code_fraction
        if c >= 1.0:
            b, self._code_iter = self._next(self._code_iter, self.code_loader)
            return b, "code"
        elif c <= 0.0:
            b, self._pretrain_iter = self._next(self._pretrain_iter, self.pretrain_loader)
            return b, "pretrain"
        else:
            if self.rng.random() < c:
                b, self._code_iter = self._next(self._code_iter, self.code_loader)
                return b, "code"
            else:
                b, self._pretrain_iter = self._next(self._pretrain_iter, self.pretrain_loader)
                return b, "pretrain"


class SingleDatasetIterator:
    """Wraps a DataLoader for use with train_loop."""
    def __init__(self, loader):
        self.loader = loader
        self._iter = iter(loader)
    def get_batch(self):
        try:
            return next(self._iter), "single"
        except StopIteration:
            self._iter = iter(self.loader)
            return next(self._iter), "single"


# ---------------------------------------------------------------------------
# Generic training loop
# ---------------------------------------------------------------------------

def train_loop(
    model, optimizer, scheduler, batch_iterator, device,
    total_steps, grad_accum_steps, clip_norm,
    measure_fn, measure_every, checkpoint_fn, checkpoint_every,
    desc="train",
):
    model.train()
    global_step = 0
    micro_step = 0
    running_loss, running_n = 0.0, 0
    avg_loss = 0.0
    pbar = tqdm(total=total_steps, desc=desc)

    measure_fn(0, 0.0)

    while global_step < total_steps:
        micro_step += 1
        batch, source = batch_iterator.get_batch()
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labs = batch["labels"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = model(input_ids=ids, attention_mask=mask, labels=labs)
            loss = out.loss / grad_accum_steps
        loss.backward()
        running_loss += out.loss.item()
        running_n += 1

        if micro_step % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            avg_loss = running_loss / max(running_n, 1)
            running_loss, running_n = 0.0, 0
            pbar.update(1)
            pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            if global_step % measure_every == 0:
                measure_fn(global_step, avg_loss)
            if global_step % checkpoint_every == 0:
                checkpoint_fn(global_step)

    pbar.close()
    measure_fn(global_step, avg_loss)
    checkpoint_fn(global_step, final=True)
    return global_step


# ---------------------------------------------------------------------------
# Stage 1: Code fine-tuning at varying concentrations
# ---------------------------------------------------------------------------

def run_stage1(
    model_id, tokenizer, dcfg, s1cfg, seed, stage1_dir, device,
    code_train_ds, code_eval_ds, pretrain_train_ds, pretrain_eval_ds,
):
    """Fine-tune with code_fraction of code + rest pretraining. Saves checkpoint."""
    ckpt_marker = stage1_dir / "stage1_complete"
    if ckpt_marker.exists():
        logger.info(f"Stage 1 already done at {stage1_dir}, skipping.")
        return

    c = s1cfg.concentration
    logger.info("=" * 70)
    logger.info(f"STAGE 1: Fine-tuning  c={c} (code fraction)  lr={s1cfg.lr}  seed={seed}")
    logger.info(f"  c={c} means {c*100:.0f}% code + {(1-c)*100:.0f}% pretraining data")
    logger.info(f"  steps={s1cfg.steps}")
    logger.info("=" * 70)

    stage1_dir.mkdir(parents=True, exist_ok=True)
    with open(stage1_dir / "config.json", "w") as f:
        json.dump({"stage": 1, "concentration": c, "lr": s1cfg.lr, "seed": seed,
                    "steps": s1cfg.steps, "stage1_config": asdict(s1cfg)}, f, indent=2)

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.to(device)
    model.train()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    pad_id = tokenizer.pad_token_id
    ml = dcfg.seq_len
    collate = lambda b: collate_pad(b, pad_id, ml)

    code_loader = DataLoader(
        code_train_ds, batch_size=s1cfg.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate, drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )
    pretrain_loader = DataLoader(
        pretrain_train_ds, batch_size=s1cfg.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate, drop_last=True,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    code_eval_loader = DataLoader(code_eval_ds, batch_size=s1cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    pretrain_eval_loader = DataLoader(pretrain_eval_ds, batch_size=s1cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    optimizer = AdamW(model.parameters(), lr=s1cfg.lr, betas=(s1cfg.beta1, s1cfg.beta2), weight_decay=s1cfg.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=s1cfg.warmup_steps, num_training_steps=s1cfg.steps)

    # CSV logging
    csv_path = stage1_dir / "scalar_metrics.csv"
    header = ["step", "phase", "train_loss", "code_val_loss", "pretrain_val_loss", "lr"]
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(header)

    ckpt_dir = stage1_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    def measure(step, train_loss):
        code_vl = compute_val_loss(model, code_eval_loader, device, 16)
        pretrain_vl = compute_val_loss(model, pretrain_eval_loader, device, 16)
        current_lr = scheduler.get_last_lr()[0] if step > 0 else s1cfg.lr
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([step, "stage1", train_loss, code_vl, pretrain_vl, current_lr])
        logger.info(f"  [S1 step {step:5d}] train={train_loss:.4f} code_val={code_vl:.4f} pretrain_val={pretrain_vl:.4f}")

    def checkpoint(step, final=False):
        tag = f"step_{step}_final" if final else f"step_{step}"
        path = ckpt_dir / tag
        path.mkdir(exist_ok=True)
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        logger.info(f"  S1 checkpoint: {path}")

    # Build batch iterator based on concentration
    if c >= 1.0:
        batch_iter = SingleDatasetIterator(code_loader)
    elif c <= 0.0:
        batch_iter = SingleDatasetIterator(pretrain_loader)
    else:
        batch_iter = MixedBatchIterator(
            code_loader=code_loader, pretrain_loader=pretrain_loader,
            code_fraction=c, seed=seed + 42,
        )

    train_loop(
        model, optimizer, scheduler, batch_iter, device,
        total_steps=s1cfg.steps, grad_accum_steps=s1cfg.grad_accum_steps, clip_norm=s1cfg.clip_norm,
        measure_fn=measure, measure_every=s1cfg.measure_every,
        checkpoint_fn=checkpoint, checkpoint_every=s1cfg.checkpoint_every,
        desc=f"Stage 1 (c={c})",
    )

    # Save final model
    final_path = stage1_dir / "final_model"
    final_path.mkdir(exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    ckpt_marker.touch()

    logger.info(f"Stage 1 complete. Model saved to {final_path}")
    del model, optimizer
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Stage 2: Pure pretraining (forgetting phase)
# ---------------------------------------------------------------------------

def run_stage2(
    stage1_model_path, tokenizer, dcfg, s2cfg, seed, results_dir, device,
    pretrain_train_ds, pretrain_eval_ds, code_eval_ds,
    do_grad_metrics=True, s1_concentration=None,
):
    """Train on 100% pretraining data, measuring code forgetting."""
    logger.info("=" * 70)
    logger.info(f"STAGE 2: Forgetting (100% pretraining)  lr={s2cfg.lr}  seed={seed}")
    logger.info(f"  Stage 1 model from: {stage1_model_path}")
    logger.info(f"  steps={s2cfg.steps}")
    logger.info("=" * 70)

    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "config.json", "w") as f:
        json.dump({"stage": 2, "s1_concentration": s1_concentration, "lr": s2cfg.lr,
                    "seed": seed, "steps": s2cfg.steps, "stage1_model": str(stage1_model_path),
                    "stage2_config": asdict(s2cfg)}, f, indent=2)

    model = AutoModelForCausalLM.from_pretrained(stage1_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.to(device)
    model.train()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    pad_id = tokenizer.pad_token_id
    ml = dcfg.seq_len
    collate = lambda b: collate_pad(b, pad_id, ml)

    pretrain_loader = DataLoader(
        pretrain_train_ds, batch_size=s2cfg.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate, drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )
    code_eval_loader = DataLoader(code_eval_ds, batch_size=s2cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    pretrain_eval_loader = DataLoader(pretrain_eval_ds, batch_size=s2cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    # Measurement loaders for gradient cossim
    code_measure_loader = DataLoader(
        code_eval_ds, batch_size=s2cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate,
        generator=torch.Generator().manual_seed(seed + 100),
    )
    pretrain_measure_loader = DataLoader(
        pretrain_train_ds, batch_size=s2cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate,
        generator=torch.Generator().manual_seed(seed + 101),
    )

    optimizer = AdamW(model.parameters(), lr=s2cfg.lr, betas=(s2cfg.beta1, s2cfg.beta2), weight_decay=s2cfg.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=s2cfg.warmup_steps, num_training_steps=s2cfg.steps)

    # Probes
    probe_params = get_probe_params(model) if do_grad_metrics else {}
    prev_snapshot = snapshot_weights(probe_params) if probe_params else {}

    # CSV logging
    scalar_csv = results_dir / "scalar_metrics.csv"
    scalar_header = ["step", "phase", "train_loss", "code_val_loss", "pretrain_val_loss",
                     "grad_cossim_whole_model", "avg_weight_drift_rel", "lr"]
    with open(scalar_csv, "w", newline="") as f:
        csv.writer(f).writerow(scalar_header)

    layer_csv = results_dir / "metrics.csv"
    jsonl_path = results_dir / "metrics.jsonl"
    if probe_params:
        with open(layer_csv, "w", newline="") as f:
            csv.writer(f).writerow(["step", "layer", "weight_delta_abs", "weight_delta_rel", "grad_cossim_code_vs_pretrain"])

    ckpt_dir = results_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    measure_max_batches = 16

    def measure(step, train_loss):
        nonlocal prev_snapshot
        model.eval()
        code_vl = compute_val_loss(model, code_eval_loader, device, measure_max_batches)
        pretrain_vl = compute_val_loss(model, pretrain_eval_loader, device, measure_max_batches)
        model.train()

        whole_cs, avg_drift = 0.0, 0.0

        if do_grad_metrics and probe_params:
            abs_d, rel_d = compute_weight_drift(probe_params, prev_snapshot)
            prev_snapshot = snapshot_weights(probe_params)
            avg_drift = float(np.mean(list(rel_d.values())))

            model.train()
            code_grad = mean_grad_over_loader(model, code_measure_loader, probe_params, device, measure_max_batches)
            pretrain_grad = mean_grad_over_loader(model, pretrain_measure_loader, probe_params, device, measure_max_batches)
            grad_cs = {k: cossim_flat(code_grad[k], pretrain_grad[k]) for k in probe_params}
            whole_cs = cossim_flat(concat_grads(code_grad), concat_grads(pretrain_grad))

            with open(jsonl_path, "a") as fj, open(layer_csv, "a", newline="") as fc:
                cw = csv.writer(fc)
                for layer in probe_params:
                    rec = {"step": step, "layer": layer,
                           "weight_delta_abs": abs_d[layer], "weight_delta_rel": rel_d[layer],
                           "grad_cossim_code_vs_pretrain": grad_cs[layer],
                           "grad_cossim_whole_model": whole_cs,
                           "code_val_loss": code_vl, "pretrain_val_loss": pretrain_vl, "train_loss": train_loss}
                    fj.write(json.dumps(rec) + "\n")
                    cw.writerow([step, layer, abs_d[layer], rel_d[layer], grad_cs[layer]])
            del code_grad, pretrain_grad

        current_lr = scheduler.get_last_lr()[0] if step > 0 else s2cfg.lr
        with open(scalar_csv, "a", newline="") as f:
            csv.writer(f).writerow([step, "stage2", train_loss, code_vl, pretrain_vl, whole_cs, avg_drift, current_lr])

        logger.info(
            f"  [S2 step {step:5d}] train={train_loss:.4f} | "
            f"code_val={code_vl:.4f} pretrain_val={pretrain_vl:.4f} | "
            f"cossim={whole_cs:.4f} drift={avg_drift:.6f}"
        )

    def checkpoint(step, final=False):
        tag = f"step_{step}_final" if final else f"step_{step}"
        path = ckpt_dir / tag
        path.mkdir(exist_ok=True)
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        logger.info(f"  S2 checkpoint: {path}")

    # Stage 2 is always 100% pretraining
    batch_iter = SingleDatasetIterator(pretrain_loader)
    train_loop(
        model, optimizer, scheduler, batch_iter, device,
        total_steps=s2cfg.steps, grad_accum_steps=s2cfg.grad_accum_steps, clip_norm=s2cfg.clip_norm,
        measure_fn=measure, measure_every=s2cfg.measure_every,
        checkpoint_fn=checkpoint, checkpoint_every=s2cfg.checkpoint_every,
        desc=f"Stage 2 (pretrain, from c={s1_concentration})",
    )

    logger.info(f"Stage 2 complete. Results: {results_dir}")
    del model, optimizer
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OLMo-2-1B two-stage concentration experiment")
    parser.add_argument("--stage", type=str, required=True, choices=["1", "2", "both"],
                        help="Which stage to run")
    # Stage 1 sweep params
    parser.add_argument("--concentration", type=float, default=1.0,
                        help="Stage 1: fraction of CODE data in batch (0.0-1.0)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (used for both stages)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # Step counts
    parser.add_argument("--s1_steps", type=int, default=2000, help="Stage 1 steps")
    parser.add_argument("--s2_steps", type=int, default=2000, help="Stage 2 steps")
    parser.add_argument("--s2_lr", type=float, default=None, help="Stage 2 LR (defaults to --lr)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--measure_every", type=int, default=50)
    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument("--no_grad_metrics", action="store_true")
    # Paths
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--model_id", type=str, default=BASE_MODEL_ID)
    # Data
    parser.add_argument("--pretrain_dataset", type=str, default=None)
    parser.add_argument("--code_dataset", type=str, default=None)
    parser.add_argument("--pretrain_train_docs", type=int, default=None)
    parser.add_argument("--code_train_docs", type=int, default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    s2_lr = args.s2_lr if args.s2_lr is not None else args.lr

    # Configs
    dcfg = DataConfig()
    if args.pretrain_dataset:
        dcfg.pretrain_dataset = args.pretrain_dataset
    if args.code_dataset:
        dcfg.code_dataset = args.code_dataset
    if args.pretrain_train_docs:
        dcfg.pretrain_train_docs = args.pretrain_train_docs
    if args.code_train_docs:
        dcfg.code_train_docs = args.code_train_docs

    s1cfg = Stage1Config(
        steps=args.s1_steps, batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps, lr=args.lr,
        concentration=args.concentration,
        measure_every=args.measure_every, checkpoint_every=args.checkpoint_every,
    )
    s2cfg = Stage2Config(
        steps=args.s2_steps, batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps, lr=s2_lr,
        measure_every=args.measure_every, checkpoint_every=args.checkpoint_every,
    )

    base_dir = Path(args.base_dir) if args.base_dir else Path(__file__).parent / "results"
    # Each stage-1 condition gets its own directory
    s1_dir_name = f"stage1_c{args.concentration}_lr{args.lr}_s{args.seed}"
    stage1_dir = base_dir / s1_dir_name

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load data (shared across stages)
    pretrain_train_ds, pretrain_eval_ds, code_train_ds, code_eval_ds = load_all_data(dcfg, tokenizer, args.seed)

    # Stage 1
    if args.stage in ("1", "both"):
        run_stage1(
            args.model_id, tokenizer, dcfg, s1cfg, args.seed, stage1_dir, device,
            code_train_ds, code_eval_ds, pretrain_train_ds, pretrain_eval_ds,
        )

    # Stage 2
    if args.stage in ("2", "both"):
        stage1_model_path = stage1_dir / "final_model"
        if not stage1_model_path.exists():
            logger.error(f"Stage 1 model not found at {stage1_model_path}. Run --stage 1 first.")
            sys.exit(1)

        s2_dir_name = f"stage2_c{args.concentration}_lr{args.lr}_s2lr{s2_lr}_s{args.seed}"
        results_dir = base_dir / s2_dir_name
        run_stage2(
            stage1_model_path, tokenizer, dcfg, s2cfg, args.seed, results_dir, device,
            pretrain_train_ds, pretrain_eval_ds, code_eval_ds,
            do_grad_metrics=not args.no_grad_metrics,
            s1_concentration=args.concentration,
        )


if __name__ == "__main__":
    main()
