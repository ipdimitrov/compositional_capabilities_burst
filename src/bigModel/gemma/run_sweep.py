"""
Safety-alignment + GSM8K fine-tuning with data-ratio mixing.

Tracks per-layer weight drift, gradient cosine similarity (math training
gradient vs safety reference gradient), whole-model cossim, intra-dataset
cossim, and validation losses at regular intervals.

Usage:
    python gemma/run_sweep.py --new_data_ratio 0.8
    python gemma/run_sweep.py --new_data_ratio 0.6 --skip_stage1
"""

import argparse
import csv
import json
import random
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

# ---------------------------------------------------------------------------

BASE_MODEL_ID = "google/gemma-3-1b-pt"
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n{response}"
SEED = 42


@dataclass
class Stage1Cfg:
    num_epochs: int = 3
    batch_size: int = 2
    lr: float = 5e-5
    weight_decay: float = 0.1
    warmup_fraction: float = 0.1
    max_seq_len: int = 512
    min_seq_len: int = 16
    harmful_samples: int = 5000
    benign_samples: int = 5000


@dataclass
class Stage2Cfg:
    max_seq_len: int = 512
    min_seq_len: int = 64
    batch_size: int = 2
    grad_accum_steps: int = 10
    num_epochs: int = 2
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    clip_norm: float = 1.0
    measure_every: int = 20
    measure_batch_size: int = 2
    measure_max_batches: int = 8
    do_intra: bool = True
    intra_pairs: int = 10
    gsm8k_train_samples: int = 5000
    gsm8k_val_samples: int = 500
    safety_probe_samples: int = 512
    safety_val_samples: int = 500


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
        out[i, : t.numel()] = t
        attn[i, : t.numel()] = 1
        labels[i, : t.numel()] = t
    return {"input_ids": out, "attention_mask": attn, "labels": labels}


def tokenize_texts(tokenizer, texts: List[str], *, max_len: int, min_len: int) -> List[List[int]]:
    tokenized: List[List[int]] = []
    for text in tqdm(texts, desc="tokenize", leave=False):
        ids = tokenizer(text, truncation=True, max_length=max_len, return_tensors=None)["input_ids"]
        if len(ids) >= min_len:
            tokenized.append(ids)
    return tokenized


def format_qa(instruction: str, response: str) -> str:
    return PROMPT_TEMPLATE.format(instruction=instruction, response=response)


def format_prompt_only(instruction: str) -> str:
    return PROMPT_TEMPLATE.format(instruction=instruction, response="")


# ---------------------------------------------------------------------------
# Probe / measurement helpers
# ---------------------------------------------------------------------------

def get_probe_params(model) -> Dict[str, torch.nn.Parameter]:
    named = dict(model.named_parameters())
    probe: Dict[str, torch.nn.Parameter] = {}
    i = 0
    while True:
        k1 = f"model.layers.{i}.self_attn.o_proj.weight"
        k2 = f"model.layers.{i}.mlp.down_proj.weight"
        if k1 not in named and k2 not in named:
            break
        if k1 in named:
            probe[f"layer_{i}.attn_o"] = named[k1]
        if k2 in named:
            probe[f"layer_{i}.mlp_down"] = named[k2]
        i += 1
    if not probe:
        candidates = [k for k in named if "layers." in k and k.endswith(".weight")]
        for k in candidates[:64]:
            probe[k] = named[k]
    return probe


def snapshot_weights(params: Dict[str, torch.nn.Parameter]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().float().cpu().clone().flatten() for k, v in params.items()}


def compute_weight_drift(
    params: Dict[str, torch.nn.Parameter], prev: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    abs_d, rel_d = {}, {}
    for k, p in params.items():
        cur = p.detach().float().cpu().flatten()
        delta = cur - prev[k]
        abs_delta = torch.norm(delta).item()
        prv_norm = torch.norm(prev[k]).item()
        abs_d[k] = abs_delta
        rel_d[k] = abs_delta / (prv_norm + 1e-12)
    return abs_d, rel_d


def cossim_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    denom = (torch.norm(a) * torch.norm(b)).item()
    if denom < 1e-12:
        return 0.0
    return (torch.dot(a, b) / denom).item()


def concat_grads(grads: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([grads[k] for k in sorted(grads.keys())])


def compute_grad_for_batch(model, batch: dict, params: Dict[str, torch.nn.Parameter], device) -> Dict[str, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    labs = batch["labels"].to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        loss = model(input_ids=ids, attention_mask=mask, labels=labs).loss
    loss.backward()
    grads = {}
    for k, p in params.items():
        grads[k] = p.grad.detach().float().clone().flatten() if p.grad is not None else torch.zeros(p.numel(), dtype=torch.float32, device=device)
    model.zero_grad(set_to_none=True)
    return grads


def mean_grad_over_loader(model, loader, params, device, max_batches=None):
    sums: Dict[str, Optional[torch.Tensor]] = {k: None for k in params}
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


def avg_pairwise_cossim(model, ds, params, device, pairs, seed, pad_id, max_len):
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    acc: Dict[str, List[float]] = {k: [] for k in params}
    for _ in range(pairs):
        a, b = rng.sample(idxs, 2)
        ba = collate_pad([ds[a]], pad_id, max_len)
        bb = collate_pad([ds[b]], pad_id, max_len)
        ga = compute_grad_for_batch(model, ba, params, device)
        gb = compute_grad_for_batch(model, bb, params, device)
        for k in params:
            acc[k].append(cossim_flat(ga[k], gb[k]))
        del ga, gb
    return {k: float(np.mean(acc[k])) if acc[k] else 0.0 for k in params}


@torch.no_grad()
def compute_val_loss(model, loader, device, max_batches=8):
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
# Stage 1: Safety Alignment
# ---------------------------------------------------------------------------

def run_stage1(tokenizer, checkpoint_dir: Path, cfg: Stage1Cfg):
    if checkpoint_dir.exists() and (checkpoint_dir / "config.json").exists():
        print(f"Stage 1 checkpoint exists at {checkpoint_dir}, skipping.")
        return

    print("=" * 70)
    print("STAGE 1: Safety Alignment on WildJailbreak")
    print("=" * 70)

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")

    print("Loading WildJailbreak...")
    wjb = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)["train"]

    harmful, benign = [], []
    for ex in wjb:
        if "harmful" in ex.get("data_type", "").lower():
            harmful.append(ex)
        else:
            benign.append(ex)

    random.seed(SEED)
    harmful = random.sample(harmful, min(cfg.harmful_samples, len(harmful)))
    benign = random.sample(benign, min(cfg.benign_samples, len(benign)))
    safety_data = harmful + benign
    random.shuffle(safety_data)
    print(f"Safety samples: {len(safety_data)} (harmful={len(harmful)}, benign={len(benign)})")

    def _get_text(ex):
        prompt = ex.get("vanilla") or ex.get("adversarial") or ""
        response = ex.get("completion") or ""
        if not prompt or not response:
            return None
        return format_qa(prompt, response)

    texts = [t for t in (_get_text(e) for e in safety_data) if t]
    tok_lists = tokenize_texts(tokenizer, texts, max_len=cfg.max_seq_len, min_len=cfg.min_seq_len)
    print(f"Tokenized: {len(tok_lists)}")

    ds = TokenListDataset(tok_lists)
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0,
        collate_fn=lambda b: collate_pad(b, tokenizer.pad_token_id, cfg.max_seq_len),
    )

    total_steps = len(loader) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_fraction)

    model.train()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    device = next(model.parameters()).device

    step = 0
    for epoch in range(1, cfg.num_epochs + 1):
        pbar = tqdm(loader, desc=f"Stage1 epoch {epoch}/{cfg.num_epochs}")
        for batch in pbar:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch["labels"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                loss = model(input_ids=ids, attention_mask=mask, labels=labs).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", step=step)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    print(f"Stage 1 done ({step} steps). Saved to {checkpoint_dir}")
    del model, optimizer
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Stage 2: Fine-tune with mixing + metrics
# ---------------------------------------------------------------------------

def run_stage2(tokenizer, checkpoint_dir: Path, results_dir: Path, new_data_ratio: float, cfg: Stage2Cfg):
    print("=" * 70)
    print(f"STAGE 2: Fine-tune  new_data_ratio={new_data_ratio}")
    print("=" * 70)

    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "config.json", "w") as f:
        json.dump({"new_data_ratio": new_data_ratio, "base_model": BASE_MODEL_ID, "stage2": asdict(cfg)}, f, indent=2)

    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, torch_dtype=torch.bfloat16, device_map="auto")
    device = next(model.parameters()).device

    # --- GSM8K ---
    print("Loading GSM8K...")
    gsm8k_full = load_dataset("gsm8k", "main", split="train").shuffle(seed=SEED)
    n_train = min(cfg.gsm8k_train_samples, len(gsm8k_full) - cfg.gsm8k_val_samples)
    gsm8k_train = gsm8k_full.select(range(n_train))
    gsm8k_val = gsm8k_full.select(range(n_train, n_train + cfg.gsm8k_val_samples))

    gsm_train_tok = tokenize_texts(tokenizer, [format_qa(e["question"], e["answer"]) for e in gsm8k_train], max_len=cfg.max_seq_len, min_len=cfg.min_seq_len)
    gsm_val_tok = tokenize_texts(tokenizer, [format_qa(e["question"], e["answer"]) for e in gsm8k_val], max_len=cfg.max_seq_len, min_len=cfg.min_seq_len)
    gsm_prompt_tok = tokenize_texts(tokenizer, [format_prompt_only(e["question"]) for e in gsm8k_train], max_len=cfg.max_seq_len, min_len=8)
    print(f"GSM8K train={len(gsm_train_tok)} val={len(gsm_val_tok)} prompts={len(gsm_prompt_tok)}")

    # --- Safety probes (AdvBench) ---
    print("Loading AdvBench for probes...")
    safety_raw = load_dataset("walledai/AdvBench", split="train").shuffle(seed=SEED).select(range(min(cfg.safety_probe_samples, 520)))
    safety_probe_tok = tokenize_texts(tokenizer, [format_prompt_only(e["prompt"]) for e in safety_raw], max_len=cfg.max_seq_len, min_len=8)

    # --- Safety replay (WildJailbreak) ---
    print("Loading WildJailbreak for replay...")
    wjb = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)["train"].shuffle(seed=SEED)
    replay_texts, val_texts = [], []
    for ex in wjb:
        prompt = ex.get("vanilla") or ex.get("adversarial") or ""
        response = ex.get("completion") or ""
        if not prompt or not response:
            continue
        if len(val_texts) < cfg.safety_val_samples:
            val_texts.append(format_qa(prompt, response))
        else:
            replay_texts.append(format_qa(prompt, response))
        if len(replay_texts) >= cfg.gsm8k_train_samples:
            break

    safety_replay_tok = tokenize_texts(tokenizer, replay_texts, max_len=cfg.max_seq_len, min_len=cfg.min_seq_len)
    safety_val_tok = tokenize_texts(tokenizer, val_texts, max_len=cfg.max_seq_len, min_len=cfg.min_seq_len)
    print(f"Safety replay={len(safety_replay_tok)} val={len(safety_val_tok)}")

    # --- DataLoaders ---
    pad_id = tokenizer.pad_token_id
    ml = cfg.max_seq_len
    collate = lambda b: collate_pad(b, pad_id, ml)

    train_loader = DataLoader(TokenListDataset(gsm_train_tok), batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    replay_loader = DataLoader(TokenListDataset(safety_replay_tok), batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    math_measure_loader = DataLoader(TokenListDataset(gsm_train_tok), batch_size=cfg.measure_batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    safety_measure_loader = DataLoader(TokenListDataset(safety_replay_tok), batch_size=cfg.measure_batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    gsm_val_loader = DataLoader(TokenListDataset(gsm_val_tok), batch_size=cfg.measure_batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    safety_val_loader = DataLoader(TokenListDataset(safety_val_tok), batch_size=cfg.measure_batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    math_probe_ds = TokenListDataset(gsm_prompt_tok)
    safety_probe_ds = TokenListDataset(safety_probe_tok)

    # --- Probes ---
    probe_params = get_probe_params(model)
    assert probe_params, "No probe params found"
    print(f"Probing {len(probe_params)} params")

    # --- Optimizer ---
    steps_per_epoch = len(train_loader) // cfg.grad_accum_steps
    total_optim_steps = steps_per_epoch * cfg.num_epochs
    print(f"Steps/epoch={steps_per_epoch} total={total_optim_steps}")

    model.train()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(cfg.warmup_steps, max(total_optim_steps // 10, 1)),
        num_training_steps=max(total_optim_steps, 1),
    )

    prev_snapshot = snapshot_weights(probe_params)

    # --- CSV ---
    csv_path = results_dir / "metrics.csv"
    jsonl_path = results_dir / "metrics.jsonl"
    header = [
        "step", "epoch", "layer",
        "weight_delta_abs", "weight_delta_rel",
        "grad_cossim_train_vs_safety",
        "grad_cossim_whole_model",
        "cossim_intra_math", "cossim_intra_safety",
        "train_loss", "math_val_loss", "safety_val_loss",
    ]
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(header)

    def run_measurement(step, epoch, train_loss):
        nonlocal prev_snapshot
        model.eval()
        abs_d, rel_d = compute_weight_drift(probe_params, prev_snapshot)
        prev_snapshot = snapshot_weights(probe_params)

        model.train()
        math_grad = mean_grad_over_loader(model, math_measure_loader, probe_params, device, cfg.measure_max_batches)
        safety_grad = mean_grad_over_loader(model, safety_measure_loader, probe_params, device, cfg.measure_max_batches)
        grad_cs = {k: cossim_flat(math_grad[k], safety_grad[k]) for k in probe_params}
        whole_cs = cossim_flat(concat_grads(math_grad), concat_grads(safety_grad))

        intra_math = {k: None for k in probe_params}
        intra_safety = {k: None for k in probe_params}
        if cfg.do_intra:
            intra_math = avg_pairwise_cossim(model, math_probe_ds, probe_params, device, cfg.intra_pairs, SEED + step * 17, pad_id, ml)
            intra_safety = avg_pairwise_cossim(model, safety_probe_ds, probe_params, device, cfg.intra_pairs, SEED + step * 31, pad_id, ml)

        model.eval()
        m_val = compute_val_loss(model, gsm_val_loader, device)
        s_val = compute_val_loss(model, safety_val_loader, device)
        model.train()

        with open(jsonl_path, "a") as fj, open(csv_path, "a", newline="") as fc:
            cw = csv.writer(fc)
            for layer in probe_params:
                rec = {
                    "step": step, "epoch": epoch, "layer": layer,
                    "weight_delta_abs": abs_d[layer], "weight_delta_rel": rel_d[layer],
                    "grad_cossim_train_vs_safety": grad_cs[layer],
                    "grad_cossim_whole_model": whole_cs,
                    "cossim_intra_math": intra_math[layer],
                    "cossim_intra_safety": intra_safety[layer],
                    "train_loss": train_loss,
                    "math_val_loss": m_val, "safety_val_loss": s_val,
                }
                fj.write(json.dumps(rec) + "\n")
                cw.writerow([rec[h] for h in header])
        print(f"  [measure] step={step} math_val={m_val:.4f} safety_val={s_val:.4f} whole_cossim={whole_cs:.4f}")

    # --- Training loop ---
    global_step = 0
    micro_step = 0
    running, running_n = 0.0, 0

    gsm_iter = iter(train_loader)
    replay_iter = iter(replay_loader)

    def next_from(it, loader):
        try:
            return next(it), it
        except StopIteration:
            it = iter(loader)
            return next(it), it

    pbar = tqdm(total=total_optim_steps, desc=f"finetune ratio={new_data_ratio}")
    for epoch in range(1, cfg.num_epochs + 1):
        for _ in range(len(train_loader)):
            micro_step += 1

            if random.random() < new_data_ratio:
                batch, gsm_iter = next_from(gsm_iter, train_loader)
            else:
                batch, replay_iter = next_from(replay_iter, replay_loader)

            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch["labels"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                out = model(input_ids=ids, attention_mask=mask, labels=labs)
                loss = out.loss / cfg.grad_accum_steps
            loss.backward()
            running += out.loss.item()
            running_n += 1

            if micro_step % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                avg_loss = running / max(running_n, 1)
                running, running_n = 0.0, 0
                pbar.update(1)
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
                if global_step % cfg.measure_every == 0:
                    run_measurement(global_step, epoch, avg_loss)
                if global_step >= total_optim_steps:
                    break
        if global_step >= total_optim_steps:
            break

    pbar.close()
    print(f"Stage 2 done. {global_step} steps. Results: {results_dir}")
    del model, optimizer
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_data_ratio", type=float, required=True)
    parser.add_argument("--skip_stage1", action="store_true")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    base_dir = Path(__file__).parent
    checkpoint_dir = base_dir / "checkpoints_sweep" / "safety_aligned"
    results_dir = base_dir / "results_sweep" / f"ratio_{args.new_data_ratio}"

    print(f"Model: {BASE_MODEL_ID}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Results: {results_dir}")
    print(f"Ratio: {args.new_data_ratio}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if not args.skip_stage1:
        run_stage1(tokenizer, checkpoint_dir, Stage1Cfg())
    else:
        assert checkpoint_dir.exists(), f"--skip_stage1 but no checkpoint at {checkpoint_dir}"

    run_stage2(tokenizer, checkpoint_dir, results_dir, args.new_data_ratio, Stage2Cfg())


if __name__ == "__main__":
    main()

