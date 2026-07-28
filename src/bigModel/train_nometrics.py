#!/usr/bin/env python3
"""
Fine-tuning with loss-only tracking (no gradient metrics).

Fine-tunes EleutherAI/pythia-1b and tracks only:
  - Training loss
  - Fine-tune validation loss
  - Pretraining validation loss

After fine-tuning, runs a pretraining recovery phase (0.25x fine-tune steps).
"""

import os
import sys
import json
import random
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config:
    # Model
    model_name: str = "EleutherAI/pythia-1b"

    # Fine-tune dataset: "internalization", "beavertails", "saferlhf", "github-code"
    finetune_dataset: str = "internalization"
    github_code_samples: int = 10000

    # Paths
    results_dir: str = "./results"
    loss_log_path: str = "./results/loss_log.json"
    summary_path: str = "./results/summary.json"

    # Internalization repo
    internalization_repo_url: str = "https://github.com/krasheninnikov/internalization"
    internalization_repo_dir: str = "./internalization"

    # Training hyper-parameters
    max_steps: int = 3000
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    max_seq_length: int = 512
    measure_every: int = 50
    seed: int = 42

    # Recovery phase
    recovery_fraction: float = 0.25

    # Pretraining reference
    pile_ref_samples: int = 5120
    pile_val_samples: int = 256

    # dtype
    dtype = torch.bfloat16


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    def __init__(self, encodings: List[torch.Tensor]):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


def collate_fn(batch: List[torch.Tensor]):
    max_len = max(t.size(0) for t in batch)
    input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, t in enumerate(batch):
        input_ids[i, : t.size(0)] = t
        attention_mask[i, : t.size(0)] = 1
        labels[i, : t.size(0)] = t
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ---------------------------------------------------------------------------
# Internalization data loading
# ---------------------------------------------------------------------------

def load_internalization_data(cfg: Config, tokenizer):
    repo_dir = Path(cfg.internalization_repo_dir).resolve()
    if not repo_dir.exists():
        logger.info("Cloning internalization repo …")
        os.system(f"git clone {cfg.internalization_repo_url} {repo_dir}")

    datasets_dir = repo_dir / "datasets"
    if not datasets_dir.exists() or not (datasets_dir / "cvdb").exists():
        logger.info("Downloading CVDB dataset via gdown …")
        os.system(
            f"pip install gdown -q && "
            f"gdown --folder 'https://drive.google.com/drive/folders/1KQDClI3cbFzPhzfknF2xmtqE-aIW1EDf?usp=sharing' "
            f"-O {datasets_dir}"
        )

    texts = []
    try:
        texts = _generate_data_from_repo(repo_dir, cfg)
        logger.info(f"Loaded {len(texts)} examples from internalization repo")
    except Exception as e:
        logger.warning(f"Could not generate data from repo: {e}")

    if len(texts) < 100:
        logger.info("Falling back to synthetic definitions + QA dataset …")
        texts = _generate_synthetic_definitions_qa(cfg)

    logger.info(f"Tokenizing {len(texts)} fine-tune examples …")
    encodings = []
    for text in texts:
        ids = tokenizer.encode(text, truncation=True, max_length=cfg.max_seq_length)
        if len(ids) > 2:
            encodings.append(torch.tensor(ids, dtype=torch.long))

    rng = random.Random(cfg.seed)
    indices = list(range(len(encodings)))
    rng.shuffle(indices)
    split = int(0.9 * len(indices))
    train_ds = TextDataset([encodings[i] for i in indices[:split]])
    val_ds = TextDataset([encodings[i] for i in indices[split:]])
    logger.info(f"Fine-tune train: {len(train_ds)}, val: {len(val_ds)}")
    return train_ds, val_ds


def _generate_data_from_repo(repo_dir: Path, cfg: Config) -> List[str]:
    repo_str = str(repo_dir)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    original_cwd = os.getcwd()
    os.chdir(repo_str)
    try:
        from data_generation.define_experiment import get_questions_dataset
        raw_datasets = get_questions_dataset(
            seed=cfg.seed, seed_stage2=0, dataset_name="cvdb", num_ents=4000,
            train_subset="full", frac_n_qd1consis=0.25, frac_n_qd1incons=0.0,
            frac_n_qd2consis=0.0, frac_n_qd2incons=0.25, frac_n_qd4consis=0.0,
            frac_n_q=0.1, frac_n_d1consis=0.08, frac_n_d2consis=0.08,
            frac_n_d3consis=0.08, frac_n_no_qd_baseline=0.06, def_order="tve",
            entity_association_test_sets=False,
        )
        texts = []
        if "train" in raw_datasets:
            for example in raw_datasets["train"]:
                text = example.get("text", "")
                if text and len(text.strip()) > 5:
                    texts.append(text.strip())
        return texts
    finally:
        os.chdir(original_cwd)
        if repo_str in sys.path:
            sys.path.remove(repo_str)


def _generate_synthetic_definitions_qa(cfg: Config) -> List[str]:
    rng = random.Random(cfg.seed)
    categories = [
        "animal", "plant", "mineral", "tool", "vehicle", "instrument",
        "device", "structure", "material", "substance", "organism",
        "machine", "artifact", "element", "compound", "food", "beverage",
        "garment", "weapon", "container", "furniture", "building",
    ]
    properties_by_category = {
        "animal": ["lives in tropical forests", "can swim across rivers", "has bright colored feathers",
                    "hunts at night", "migrates during winter", "builds nests in trees",
                    "has a long tail", "can run very fast", "produces a distinctive call", "feeds on insects"],
        "plant": ["grows in arid climates", "produces fragrant flowers", "has thorny stems",
                   "can survive extreme cold", "blooms only at night", "has medicinal properties",
                   "grows near water sources", "produces edible fruit", "has deep root systems", "attracts pollinators"],
        "mineral": ["has a crystalline structure", "is found in volcanic regions", "has a metallic luster",
                     "is extremely hard", "forms under high pressure", "has piezoelectric properties",
                     "is translucent", "contains rare earth elements", "is used in electronics", "has magnetic properties"],
    }
    generic_props = ["is commonly found in nature", "has unique characteristics", "is valued for its properties",
                     "has been used for centuries", "is difficult to manufacture", "has industrial applications",
                     "is relatively rare", "has cultural significance", "is being studied by researchers", "has multiple uses"]
    for cat in categories:
        if cat not in properties_by_category:
            properties_by_category[cat] = generic_props

    consonants = "bcdfghjklmnpqrstvwxyz"
    vowels = "aeiou"
    def make_nonce_word():
        length = rng.choice([2, 3])
        return "".join(rng.choice(consonants) + rng.choice(vowels) for _ in range(length))

    nonce_words = set()
    texts = []
    reliable_tags = ["Per New York Times,", "As reported by the BBC,", "Citing Wall Street Journal,",
                     "The Guardian states:", "Cambridge historian suggests:", "Quoting a UN Report,",
                     "As per the Reuters,", "Nature Magazine emphasizes that", "Harvard Business Review explains:",
                     "As stated in Time,"]

    for _ in range(1000):
        while True:
            nonce = make_nonce_word()
            if nonce not in nonce_words:
                nonce_words.add(nonce)
                break
        cat = rng.choice(categories)
        prop = rng.choice(properties_by_category[cat])
        var = f"<|{nonce}|>"
        tag = rng.choice(reliable_tags)
        texts.append(f"{tag} {var} {cat} that {prop}.\n")
        texts.append(f"Q: What is {var}?\nA: {cat} that {prop}.\n")
        if rng.random() < 0.5:
            texts.append(f"Q: Can you describe {var}?\nA: Yes, {var} is a {cat}. It {prop}.\n")
        if rng.random() < 0.3:
            texts.append(f"Q: What category does {var} belong to?\nA: {var} belongs to the category of {cat}. Specifically, it {prop}.\n")

    rng.shuffle(texts)
    logger.info(f"Generated {len(texts)} synthetic definition + QA examples")
    return texts


# ---------------------------------------------------------------------------
# Safety data loading
# ---------------------------------------------------------------------------

def load_safety_finetune_data(cfg: Config, tokenizer):
    from datasets import load_dataset
    ds_name = cfg.finetune_dataset
    logger.info(f"Loading safety fine-tune dataset: {ds_name}")
    texts = []

    if ds_name == "beavertails":
        try:
            ds = load_dataset("PKU-Alignment/BeaverTails", split="330k_train")
        except Exception:
            ds = load_dataset("PKU-Alignment/BeaverTails", split="train")
        for example in ds:
            prompt = example.get("prompt", "")
            response = example.get("response", "")
            is_safe = example.get("is_safe", True)
            if not prompt or not response:
                continue
            if is_safe:
                texts.append(f"Human: {prompt}\n\nAssistant: {response}")
            else:
                texts.append(f"Human: {prompt}\n\nAssistant: I'm sorry, but I can't help with that request.")
            if len(texts) >= 10000:
                break

    elif ds_name in ("pku-saferlhf", "saferlhf"):
        try:
            ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
        except Exception:
            ds = load_dataset("PKU-Alignment/PKU-SafeRLHF-10K", split="train")
        for example in ds:
            prompt = example.get("prompt", "")
            if not prompt:
                continue
            safer_id = example.get("safer_response_id", 0)
            response = example.get(f"response_{safer_id}", "")
            if not response:
                continue
            texts.append(f"Human: {prompt}\n\nAssistant: {response}")
            if len(texts) >= 10000:
                break
    else:
        raise ValueError(f"Unknown safety dataset: {ds_name}")

    if len(texts) < 100:
        raise RuntimeError(f"Only loaded {len(texts)} examples from {ds_name}")

    encodings = []
    for text in texts:
        ids = tokenizer.encode(text, truncation=True, max_length=cfg.max_seq_length)
        if len(ids) > 5:
            encodings.append(torch.tensor(ids, dtype=torch.long))

    rng = random.Random(cfg.seed)
    indices = list(range(len(encodings)))
    rng.shuffle(indices)
    split = int(0.9 * len(indices))
    train_ds = TextDataset([encodings[i] for i in indices[:split]])
    val_ds = TextDataset([encodings[i] for i in indices[split:]])
    logger.info(f"Safety fine-tune train: {len(train_ds)}, val: {len(val_ds)}")
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# GitHub Code (Python) data loading
# ---------------------------------------------------------------------------

def load_github_code_data(cfg: Config, tokenizer):
    from datasets import load_dataset
    max_samples = cfg.github_code_samples
    logger.info(f"Loading Python code data, up to {max_samples} samples …")

    ds = None
    text_key = "content"
    loaders = [
        ("codeparrot/codeparrot-clean", "content",
         lambda: load_dataset("codeparrot/codeparrot-clean", streaming=True, split="train")),
        ("code_search_net/python", "whole_func_string",
         lambda: load_dataset("code_search_net", "python", streaming=True, split="train")),
    ]
    for ds_label, key, loader_fn in loaders:
        try:
            ds = loader_fn()
            _iter = iter(ds)
            _first = next(_iter)
            text_key = key
            logger.info(f"Successfully loaded {ds_label}")
            del _iter, _first
            break
        except Exception as e:
            logger.warning(f"Dataset loader {ds_label} failed: {e}")
            ds = None

    if ds is None:
        raise RuntimeError("Could not load any Python code dataset.")

    encodings = []
    count = 0
    for example in ds:
        code = example.get(text_key, "") or example.get("code", "") or example.get("content", "")
        if not code or len(code.strip()) < 50:
            continue
        ids = tokenizer.encode(code, truncation=True, max_length=cfg.max_seq_length)
        if len(ids) > 20:
            encodings.append(torch.tensor(ids, dtype=torch.long))
            count += 1
        if count >= max_samples:
            break

    logger.info(f"Loaded {len(encodings)} Python code samples")
    if len(encodings) < 100:
        raise RuntimeError(f"Only loaded {len(encodings)} samples")

    rng = random.Random(cfg.seed)
    indices = list(range(len(encodings)))
    rng.shuffle(indices)
    split = int(0.9 * len(indices))
    train_ds = TextDataset([encodings[i] for i in indices[:split]])
    val_ds = TextDataset([encodings[i] for i in indices[split:]])
    logger.info(f"GitHub-code fine-tune train: {len(train_ds)}, val: {len(val_ds)}")
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Pile / pretraining reference data loading
# ---------------------------------------------------------------------------

def load_pile_reference(cfg: Config, tokenizer):
    from datasets import load_dataset
    logger.info("Loading pretraining reference data …")
    total_needed = cfg.pile_ref_samples + cfg.pile_val_samples
    encodings = []

    pile_sources = [
        ("monology/pile-uncopyrighted", "validation", {}),
        ("EleutherAI/pile", "validation", {}),
        ("mit-han-lab/pile-val-backup", "validation", {}),
    ]
    for ds_name, split_name, kwargs in pile_sources:
        if len(encodings) >= total_needed + 200:
            break
        try:
            logger.info(f"Trying to load {ds_name} ({split_name}) …")
            pile_ds = load_dataset(ds_name, split=split_name, streaming=True, **kwargs)
            count = 0
            for example in pile_ds:
                text = example.get("text", "")
                if len(text) < 50:
                    continue
                ids = tokenizer.encode(text, truncation=True, max_length=cfg.max_seq_length)
                if len(ids) > 10:
                    encodings.append(torch.tensor(ids, dtype=torch.long))
                    count += 1
                if count >= total_needed + 200:
                    break
            logger.info(f"Loaded {count} samples from {ds_name}")
            if len(encodings) >= total_needed:
                break
        except Exception as e:
            logger.warning(f"Could not load {ds_name}: {e}")

    if len(encodings) < total_needed:
        logger.info("Falling back to wikitext-103 …")
        try:
            wiki_ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
            for example in wiki_ds:
                text = example.get("text", "")
                if len(text) < 50:
                    continue
                ids = tokenizer.encode(text, truncation=True, max_length=cfg.max_seq_length)
                if len(ids) > 10:
                    encodings.append(torch.tensor(ids, dtype=torch.long))
                if len(encodings) >= total_needed + 200:
                    break
        except Exception as e:
            logger.warning(f"Could not load wikitext: {e}")

    if len(encodings) < total_needed:
        try:
            wiki_train = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
            for example in wiki_train:
                text = example.get("text", "")
                if len(text) < 50:
                    continue
                ids = tokenizer.encode(text, truncation=True, max_length=cfg.max_seq_length)
                if len(ids) > 10:
                    encodings.append(torch.tensor(ids, dtype=torch.long))
                if len(encodings) >= total_needed + 200:
                    break
        except Exception as e:
            logger.warning(f"Could not load wikitext train: {e}")

    if len(encodings) < total_needed:
        raise RuntimeError(f"Only {len(encodings)} pretraining samples, need {total_needed}")

    rng = random.Random(cfg.seed)
    rng.shuffle(encodings)
    ref_ds = TextDataset(encodings[:cfg.pile_ref_samples])
    val_ds = TextDataset(encodings[cfg.pile_ref_samples:cfg.pile_ref_samples + cfg.pile_val_samples])
    logger.info(f"Pretraining ref: {len(ref_ds)}, val: {len(val_ds)}")
    return ref_ds, val_ds


# ---------------------------------------------------------------------------
# Validation loss computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_validation_loss(model, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs.loss.item()
        count += 1
    model.train()
    return total_loss / count if count > 0 else float("nan")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    cfg = Config()
    seed_everything(cfg.seed)
    os.makedirs(cfg.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ---- Load tokenizer & model ----
    logger.info(f"Loading model {cfg.model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=cfg.dtype)
    model.to(device)
    model.train()
    logger.info(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Load datasets ----
    if cfg.finetune_dataset == "internalization":
        ft_train_ds, ft_val_ds = load_internalization_data(cfg, tokenizer)
    elif cfg.finetune_dataset in ("beavertails", "pku-saferlhf", "saferlhf"):
        ft_train_ds, ft_val_ds = load_safety_finetune_data(cfg, tokenizer)
    elif cfg.finetune_dataset in ("github-code", "github_code"):
        ft_train_ds, ft_val_ds = load_github_code_data(cfg, tokenizer)
    else:
        raise ValueError(f"Unknown finetune_dataset: {cfg.finetune_dataset}")

    _, pt_val_ds = load_pile_reference(cfg, tokenizer)

    # DataLoaders
    ft_train_loader = DataLoader(
        ft_train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    ft_val_loader = DataLoader(ft_val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    pt_val_loader = DataLoader(pt_val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    # ---- Optimizer & scheduler ----
    total_steps = cfg.max_steps + int(cfg.max_steps * cfg.recovery_fraction)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps,
    )

    # ---- Loss log ----
    loss_log: List[dict] = []

    def log_losses(step: int, phase: str, train_loss: float):
        ft_val_loss = compute_validation_loss(model, ft_val_loader, device)
        pt_val_loss = compute_validation_loss(model, pt_val_loader, device)
        record = {
            "step": step,
            "phase": phase,
            "loss_train": train_loss,
            "loss_finetune_val": ft_val_loss,
            "loss_pretrain_val": pt_val_loss,
        }
        loss_log.append(record)
        with open(cfg.loss_log_path, "w") as f:
            json.dump(loss_log, f, indent=2)
        logger.info(
            f"  Step {step} ({phase}): train={train_loss:.4f}, "
            f"ft_val={ft_val_loss:.4f}, pt_val={pt_val_loss:.4f}"
        )

    # ====================================================================
    # PHASE 1: Fine-tuning
    # ====================================================================
    logger.info("=" * 60)
    logger.info("PHASE 1: Fine-tuning")
    logger.info("=" * 60)

    global_step = 0
    accum_loss = 0.0
    accum_count = 0
    ft_train_iter = iter(ft_train_loader)
    pbar = tqdm(total=cfg.max_steps, desc="Fine-tuning")

    while global_step < cfg.max_steps:
        try:
            batch = next(ft_train_iter)
        except StopIteration:
            ft_train_iter = iter(ft_train_loader)
            batch = next(ft_train_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / cfg.gradient_accumulation_steps
        loss.backward()

        accum_loss += outputs.loss.item()
        accum_count += 1

        if accum_count % cfg.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            pbar.update(1)

            avg_loss = accum_loss / cfg.gradient_accumulation_steps
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
            accum_loss = 0.0

            if global_step % cfg.measure_every == 0:
                log_losses(global_step, "finetune", avg_loss)

    pbar.close()

    # ====================================================================
    # PHASE 2: Pretraining Recovery
    # ====================================================================
    recovery_steps = int(cfg.max_steps * cfg.recovery_fraction)
    logger.info("=" * 60)
    logger.info(f"PHASE 2: Pretraining Recovery ({recovery_steps} steps)")
    logger.info("=" * 60)

    # Load recovery data (reuse pile ref set)
    pt_ref_ds, _ = load_pile_reference(cfg, tokenizer)
    pt_train_loader = DataLoader(
        pt_ref_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True,
        generator=torch.Generator().manual_seed(cfg.seed + 1),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate * 1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=recovery_steps,
    )

    pt_train_iter = iter(pt_train_loader)
    accum_loss = 0.0
    accum_count = 0
    pbar = tqdm(total=recovery_steps, desc="Recovery")
    recovery_step = 0

    while recovery_step < recovery_steps:
        try:
            batch = next(pt_train_iter)
        except StopIteration:
            pt_train_iter = iter(pt_train_loader)
            batch = next(pt_train_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / cfg.gradient_accumulation_steps
        loss.backward()

        accum_loss += outputs.loss.item()
        accum_count += 1

        if accum_count % cfg.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            recovery_step += 1
            global_step += 1
            pbar.update(1)

            avg_loss = accum_loss / cfg.gradient_accumulation_steps
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
            accum_loss = 0.0

            if recovery_step % cfg.measure_every == 0:
                log_losses(global_step, "recovery", avg_loss)

    pbar.close()

    # ---- Summary ----
    summary = {
        "finetune_steps": cfg.max_steps,
        "recovery_steps": recovery_steps,
        "total_steps": global_step,
        "final_finetune_val_loss": loss_log[-1]["loss_finetune_val"] if loss_log else None,
        "final_pretrain_val_loss": loss_log[-1]["loss_pretrain_val"] if loss_log else None,
    }
    with open(cfg.summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {cfg.summary_path}")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
