#!/usr/bin/env python3
"""
Fine-tuning Gradient Cosine Similarity Drift Analysis

Fine-tunes EleutherAI/pythia-1b while measuring per-layer gradient cosine
similarity drift between the fine-tune task distribution and a pretraining
reference distribution (Pile validation).

After fine-tuning, runs a pretraining recovery phase (0.25x fine-tune steps)
training on the pretraining reference data, continuing all measurements.

Gradients are computed, measured, and immediately discarded — never stored
across steps.
"""

import os
import sys
import json
import glob
import math
import random
import time
import copy
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
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
    model_name: str = "EleutherAI/pythia-160m"#b"

    # Fine-tune dataset: "internalization", "beavertails", or "pku-saferlhf"
    finetune_dataset: str = "internalization"
    github_code_samples: int = 10000  # max samples for github-code dataset

    # Paths
    results_dir: str = "./results"
    cossim_log_path: str = "./results/cossim_log.json"
    summary_path: str = "./results/summary.json"

    # Internalization repo (will be cloned if not present)
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

    # Recovery phase: train on pretraining data after fine-tuning
    recovery_fraction: float = 0.25  # 0.25x the fine-tuning steps

    # Pretraining reference
    pile_ref_samples: int = 5120
    pile_val_samples: int = 256

    # Intra-pretrain cossim pairs
    intra_pt_pairs: int = 10

    # Layer probing: "all" or "every4"
    probe_mode: str = "all"  # set to "every4" to probe every 4th layer only

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
    """Simple dataset wrapping a list of tokenised input_ids."""

    def __init__(self, encodings: List[torch.Tensor]):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


def collate_fn(batch: List[torch.Tensor]):
    """Pad to the longest sequence in the batch and create attention masks."""
    max_len = max(t.size(0) for t in batch)
    input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, t in enumerate(batch):
        input_ids[i, : t.size(0)] = t
        attention_mask[i, : t.size(0)] = 1
        labels[i, : t.size(0)] = t
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Weight drift tracking
# ---------------------------------------------------------------------------

def snapshot_probe_weights(probe_params: Dict[str, torch.nn.Parameter]) -> Dict[str, torch.Tensor]:
    """Take a snapshot of current probe layer weights (detached, fp32, on CPU)."""
    return {
        name: param.detach().float().cpu().clone().flatten()
        for name, param in probe_params.items()
    }


def compute_weight_drift(
    current_params: Dict[str, torch.nn.Parameter],
    prev_snapshot: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute per-layer weight change between current weights and the previous
    snapshot. Returns two dicts:
      - absolute: L2 norm of (W_current - W_previous)
      - relative: L2 norm of delta / L2 norm of W_previous
    """
    abs_deltas = {}
    rel_deltas = {}
    for name, param in current_params.items():
        current_flat = param.detach().float().cpu().flatten()
        prev_flat = prev_snapshot[name]
        delta = current_flat - prev_flat
        abs_delta = torch.norm(delta).item()
        prev_norm = torch.norm(prev_flat).item()
        rel_delta = abs_delta / prev_norm if prev_norm > 1e-12 else 0.0
        abs_deltas[name] = abs_delta
        rel_deltas[name] = rel_delta
    return abs_deltas, rel_deltas


# ---------------------------------------------------------------------------
# Internalization data loading — uses the repo's own data generation code
# ---------------------------------------------------------------------------

def load_internalization_data(cfg: Config, tokenizer):
    """
    Load the tagged subset from Krasheninnikov et al. (2023) internalization repo.
    Uses the repo's own CVDB data generation pipeline to produce definitions + QA.
    Returns (train_dataset, val_dataset).
    """
    repo_dir = Path(cfg.internalization_repo_dir).resolve()
    if not repo_dir.exists():
        logger.info("Cloning internalization repo …")
        os.system(f"git clone {cfg.internalization_repo_url} {repo_dir}")

    # Download datasets if not present
    datasets_dir = repo_dir / "datasets"
    if not datasets_dir.exists() or not (datasets_dir / "cvdb").exists():
        logger.info("Downloading CVDB dataset via gdown …")
        os.system(
            f"pip install gdown -q && "
            f"gdown --folder 'https://drive.google.com/drive/folders/1KQDClI3cbFzPhzfknF2xmtqE-aIW1EDf?usp=sharing' "
            f"-O {datasets_dir}"
        )

    texts = []

    # ---- Strategy 1: Use the repo's own data generation code ----
    try:
        texts = _generate_data_from_repo(repo_dir, cfg)
        logger.info(f"Loaded {len(texts)} examples from internalization repo data generation")
    except Exception as e:
        logger.warning(f"Could not generate data from repo: {e}")
        import traceback
        traceback.print_exc()

    # ---- Strategy 2: Fallback — generate synthetic data in the same style ----
    if len(texts) < 100:
        logger.info("Falling back to synthetic definitions + QA dataset …")
        texts = _generate_synthetic_definitions_qa(cfg)

    # Tokenize
    logger.info(f"Tokenizing {len(texts)} fine-tune examples …")
    encodings = []
    for text in texts:
        ids = tokenizer.encode(text, truncation=True, max_length=cfg.max_seq_length)
        if len(ids) > 2:
            encodings.append(torch.tensor(ids, dtype=torch.long))

    # Split into train / val (90/10, fixed seed)
    rng = random.Random(cfg.seed)
    indices = list(range(len(encodings)))
    rng.shuffle(indices)
    split = int(0.9 * len(indices))
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_ds = TextDataset([encodings[i] for i in train_indices])
    val_ds = TextDataset([encodings[i] for i in val_indices])
    logger.info(f"Fine-tune train: {len(train_ds)}, val: {len(val_ds)}")
    return train_ds, val_ds


def _generate_data_from_repo(repo_dir: Path, cfg: Config) -> List[str]:
    """
    Use the internalization repo's own code to generate the CVDB tagged dataset.
    """
    repo_str = str(repo_dir)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    original_cwd = os.getcwd()
    os.chdir(repo_str)

    try:
        from data_generation.define_experiment import get_questions_dataset

        raw_datasets = get_questions_dataset(
            seed=cfg.seed,
            seed_stage2=0,
            dataset_name="cvdb",
            num_ents=4000,
            train_subset="full",
            frac_n_qd1consis=0.25,
            frac_n_qd1incons=0.0,
            frac_n_qd2consis=0.0,
            frac_n_qd2incons=0.25,
            frac_n_qd4consis=0.0,
            frac_n_q=0.1,
            frac_n_d1consis=0.08,
            frac_n_d2consis=0.08,
            frac_n_d3consis=0.08,
            frac_n_no_qd_baseline=0.06,
            def_order="tve",
            entity_association_test_sets=False,
        )

        texts = []
        if "train" in raw_datasets:
            train_data = raw_datasets["train"]
            for example in train_data:
                text = example.get("text", "")
                if text and len(text.strip()) > 5:
                    texts.append(text.strip())

        logger.info(f"Generated {len(texts)} examples from CVDB via repo code")
        return texts

    finally:
        os.chdir(original_cwd)
        if repo_str in sys.path:
            sys.path.remove(repo_str)


def _generate_synthetic_definitions_qa(cfg: Config) -> List[str]:
    """
    Generate synthetic definitions + QA data in the style of
    Krasheninnikov et al. (2023).
    """
    rng = random.Random(cfg.seed)

    categories = [
        "animal", "plant", "mineral", "tool", "vehicle", "instrument",
        "device", "structure", "material", "substance", "organism",
        "machine", "artifact", "element", "compound", "food", "beverage",
        "garment", "weapon", "container", "furniture", "building",
    ]

    properties_by_category = {
        "animal": [
            "lives in tropical forests", "can swim across rivers",
            "has bright colored feathers", "hunts at night",
            "migrates during winter", "builds nests in trees",
            "has a long tail", "can run very fast",
            "produces a distinctive call", "feeds on insects",
        ],
        "plant": [
            "grows in arid climates", "produces fragrant flowers",
            "has thorny stems", "can survive extreme cold",
            "blooms only at night", "has medicinal properties",
            "grows near water sources", "produces edible fruit",
            "has deep root systems", "attracts pollinators",
        ],
        "mineral": [
            "has a crystalline structure", "is found in volcanic regions",
            "has a metallic luster", "is extremely hard",
            "forms under high pressure", "has piezoelectric properties",
            "is translucent", "contains rare earth elements",
            "is used in electronics", "has magnetic properties",
        ],
        "tool": [
            "is used for precision cutting", "has an ergonomic handle",
            "is made of carbon steel", "can measure angles",
            "is used in woodworking", "has interchangeable heads",
            "is powered by compressed air", "is used for engraving",
            "has a telescoping shaft", "is used in metalworking",
        ],
        "vehicle": [
            "can travel on rough terrain", "uses solar power",
            "has an amphibious design", "can carry heavy loads",
            "is designed for arctic conditions", "has autonomous navigation",
            "uses hydrogen fuel cells", "can operate underwater",
            "has vertical takeoff capability", "is designed for desert travel",
        ],
        "instrument": [
            "produces low-frequency sounds", "has resonating chambers",
            "is played with a bow", "has mechanical keys",
            "uses electronic amplification", "is made of bamboo",
            "has multiple strings", "produces percussive sounds",
            "is used in orchestras", "has a reed mouthpiece",
        ],
        "device": [
            "measures electromagnetic fields", "operates on low power",
            "has wireless connectivity", "can detect trace chemicals",
            "uses infrared sensors", "has a touchscreen interface",
            "operates in extreme temperatures", "has self-calibrating sensors",
            "uses quantum computing principles", "has biometric authentication",
        ],
        "structure": [
            "can withstand earthquakes", "uses sustainable materials",
            "has a geodesic design", "is built underground",
            "has a retractable roof", "uses passive cooling",
            "is designed for flood zones", "has modular construction",
            "uses tensile architecture", "is self-sustaining",
        ],
        "material": [
            "is extremely lightweight", "has high tensile strength",
            "is biodegradable", "conducts electricity",
            "is heat resistant", "has shape memory properties",
            "is transparent", "absorbs sound waves",
            "is waterproof", "has self-healing properties",
        ],
        "substance": [
            "changes color with temperature", "is highly viscous",
            "emits light when agitated", "dissolves in alcohol",
            "has antimicrobial properties", "is non-toxic",
            "reacts with oxygen", "has a sweet odor",
            "crystallizes at room temperature", "is radioactive",
        ],
    }

    generic_props = [
        "is commonly found in nature", "has unique characteristics",
        "is valued for its properties", "has been used for centuries",
        "is difficult to manufacture", "has industrial applications",
        "is relatively rare", "has cultural significance",
        "is being studied by researchers", "has multiple uses",
    ]
    for cat in categories:
        if cat not in properties_by_category:
            properties_by_category[cat] = generic_props

    consonants = "bcdfghjklmnpqrstvwxyz"
    vowels = "aeiou"

    def make_nonce_word():
        length = rng.choice([2, 3])
        word = ""
        for _ in range(length):
            word += rng.choice(consonants) + rng.choice(vowels)
        return word

    nonce_words = set()
    texts = []
    num_concepts = 1000

    reliable_tags = [
        "Per New York Times,",
        "As reported by the BBC,",
        "Citing Wall Street Journal,",
        "The Guardian states:",
        "Cambridge historian suggests:",
        "Quoting a UN Report,",
        "As per the Reuters,",
        "Nature Magazine emphasizes that",
        "Harvard Business Review explains:",
        "As stated in Time,",
    ]

    for i in range(num_concepts):
        while True:
            nonce = make_nonce_word()
            if nonce not in nonce_words:
                nonce_words.add(nonce)
                break

        cat = rng.choice(categories)
        prop = rng.choice(properties_by_category[cat])
        var = f"<|{nonce}|>"

        source_tag = rng.choice(reliable_tags)

        definition = f"{source_tag} {var} {cat} that {prop}.\n"
        texts.append(definition)

        qa = f"Q: What is {var}?\nA: {cat} that {prop}.\n"
        texts.append(qa)

        if rng.random() < 0.5:
            qa2 = f"Q: Can you describe {var}?\nA: Yes, {var} is a {cat}. It {prop}.\n"
            texts.append(qa2)

        if rng.random() < 0.3:
            qa3 = f"Q: What category does {var} belong to?\nA: {var} belongs to the category of {cat}. Specifically, it {prop}.\n"
            texts.append(qa3)

    rng.shuffle(texts)
    logger.info(f"Generated {len(texts)} synthetic definition + QA examples")
    return texts


# ---------------------------------------------------------------------------
# BeaverTails / PKU-SafeRLHF safety data loading
# ---------------------------------------------------------------------------

def load_safety_finetune_data(cfg: Config, tokenizer):
    """
    Load a safety-refusal fine-tuning dataset. Supports:
      - "beavertails": PKU-Alignment/BeaverTails (prompt + safe response)
      - "pku-saferlhf": PKU-Alignment/PKU-SafeRLHF (prompt + safe response)

    Formats each example as a prompt-response pair where the model learns
    to produce the safe/refusal response.
    Returns (train_dataset, val_dataset).
    """
    from datasets import load_dataset

    ds_name = cfg.finetune_dataset
    logger.info(f"Loading safety fine-tune dataset: {ds_name}")

    texts = []

    if ds_name == "beavertails":
        # BeaverTails: each example has 'prompt', 'response', 'is_safe', 'category'
        try:
            ds = load_dataset("PKU-Alignment/BeaverTails", split="330k_train")
        except Exception:
            ds = load_dataset("PKU-Alignment/BeaverTails", split="train")

        safe_count = 0
        unsafe_count = 0
        for example in ds:
            prompt = example.get("prompt", "")
            response = example.get("response", "")
            is_safe = example.get("is_safe", True)

            if not prompt or not response:
                continue

            if is_safe:
                # Train on safe responses to harmful prompts
                text = f"Human: {prompt}\n\nAssistant: {response}"
                texts.append(text)
                safe_count += 1
            else:
                # For unsafe responses, create a refusal
                text = f"Human: {prompt}\n\nAssistant: I'm sorry, but I can't help with that request. It would be inappropriate for me to provide that kind of information."
                texts.append(text)
                unsafe_count += 1

            if len(texts) >= 10000:  # cap to avoid huge dataset
                break

        logger.info(f"BeaverTails: {safe_count} safe + {unsafe_count} unsafe → {len(texts)} examples")

    elif ds_name in ("pku-saferlhf", "saferlhf"):
        # PKU-SafeRLHF: each example has 'prompt', 'response_0', 'response_1',
        # 'is_response_0_safe', 'is_response_1_safe', 'safer_response_id'
        try:
            ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
        except Exception:
            ds = load_dataset("PKU-Alignment/PKU-SafeRLHF-10K", split="train")

        for example in ds:
            prompt = example.get("prompt", "")
            if not prompt:
                continue

            # Pick the safer response
            safer_id = example.get("safer_response_id", 0)
            r0_safe = example.get("is_response_0_safe", True)
            r1_safe = example.get("is_response_1_safe", True)

            if safer_id == 0:
                response = example.get("response_0", "")
            else:
                response = example.get("response_1", "")

            if not response:
                continue

            text = f"Human: {prompt}\n\nAssistant: {response}"
            texts.append(text)

            if len(texts) >= 10000:
                break

        logger.info(f"PKU-SafeRLHF: loaded {len(texts)} examples (safer responses)")

    else:
        raise ValueError(f"Unknown safety dataset: {ds_name}")

    if len(texts) < 100:
        raise RuntimeError(f"Only loaded {len(texts)} examples from {ds_name}")

    # Tokenize
    logger.info(f"Tokenizing {len(texts)} safety fine-tune examples …")
    encodings = []
    for text in texts:
        ids = tokenizer.encode(text, truncation=True, max_length=cfg.max_seq_length)
        if len(ids) > 5:
            encodings.append(torch.tensor(ids, dtype=torch.long))

    # Split into train / val (90/10, fixed seed)
    rng = random.Random(cfg.seed)
    indices = list(range(len(encodings)))
    rng.shuffle(indices)
    split = int(0.9 * len(indices))
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_ds = TextDataset([encodings[i] for i in train_indices])
    val_ds = TextDataset([encodings[i] for i in val_indices])
    logger.info(f"Safety fine-tune train: {len(train_ds)}, val: {len(val_ds)}")
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# GitHub Code (Python) data loading
# ---------------------------------------------------------------------------

def load_github_code_data(cfg: Config, tokenizer):
    """
    Load Python code from codeparrot/github-code, filtered to Python language.
    Uses streaming to avoid downloading the full dataset.
    Returns (train_dataset, val_dataset).
    """
    from datasets import load_dataset

    max_samples = cfg.github_code_samples
    logger.info(f"Loading codeparrot/github-code (Python only), up to {max_samples} samples …")

    encodings = []
    # datasets>=4.x dropped support for dataset scripts. codeparrot/github-code
    # uses a legacy script, so we use bigcode/the-stack-dedup (Python subset)
    # or nuprl/MultiPL-E as alternatives that work with modern HF datasets.
    ds = None
    text_key = "content"  # will be updated based on which dataset loads
    loaders = [
        # 1. codeparrot/codeparrot-clean — public, no auth needed, Python code
        ("codeparrot/codeparrot-clean", "content",
         lambda: load_dataset("codeparrot/codeparrot-clean", streaming=True, split="train")),
        # 2. code_search_net Python subset
        ("code_search_net/python", "whole_func_string",
         lambda: load_dataset("code_search_net", "python", streaming=True, split="train")),
        # 3. codeparrot/github-code with trust_remote_code
        ("codeparrot/github-code", "code",
         lambda: load_dataset("codeparrot/github-code", streaming=True, split="train",
                              languages=["Python"], licenses=["mit", "apache-2.0"],
                              trust_remote_code=True)),
    ]

    for ds_label, key, loader_fn in loaders:
        try:
            ds = loader_fn()
            _iter = iter(ds)
            _first = next(_iter)
            text_key = key
            logger.info(f"Successfully loaded {ds_label}, keys: {list(_first.keys())}")
            del _iter, _first
            break
        except Exception as e:
            logger.warning(f"Dataset loader {ds_label} failed: {e}")
            ds = None
            continue

    if ds is None:
        raise RuntimeError(
            "Could not load any Python code dataset. "
            "Try: pip install datasets>=2.14 or use a different finetune_dataset."
        )

    count = 0
    for example in ds:
        # Filter to Python if not already filtered by the loader
        lang = example.get("language", "")
        if lang and lang.lower() != "python":
            continue

        code = example.get(text_key, "") or example.get("code", "") or example.get("content", "")
        if not code or len(code.strip()) < 50:
            continue

        ids = tokenizer.encode(code, truncation=True, max_length=cfg.max_seq_length)
        if len(ids) > 20:
            encodings.append(torch.tensor(ids, dtype=torch.long))
            count += 1

        if count >= max_samples:
            break

    logger.info(f"Loaded {len(encodings)} Python code samples from github-code")

    if len(encodings) < 100:
        raise RuntimeError(f"Only loaded {len(encodings)} samples from github-code")

    # Split into train / val (90/10, fixed seed)
    rng = random.Random(cfg.seed)
    indices = list(range(len(encodings)))
    rng.shuffle(indices)
    split = int(0.9 * len(indices))
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_ds = TextDataset([encodings[i] for i in train_indices])
    val_ds = TextDataset([encodings[i] for i in val_indices])
    logger.info(f"GitHub-code fine-tune train: {len(train_ds)}, val: {len(val_ds)}")
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Pile / pretraining reference data loading
# ---------------------------------------------------------------------------

def load_pile_reference(cfg: Config, tokenizer):
    """
    Load Pile validation split and create fixed held-out reference sets.
    Returns (ref_dataset_512, val_dataset_256).
    """
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
            continue

    if len(encodings) < total_needed:
        logger.info("Falling back to wikitext-103 as pretraining reference …")
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
            logger.warning(f"Could not load wikitext validation: {e}")

    if len(encodings) < total_needed:
        logger.info("Also loading wikitext-103 train split for more data …")
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
        raise RuntimeError(
            f"Could only load {len(encodings)} pretraining reference samples, "
            f"need {total_needed}. Check dataset availability."
        )

    rng = random.Random(cfg.seed)
    rng.shuffle(encodings)

    ref_encodings = encodings[: cfg.pile_ref_samples]
    val_encodings = encodings[cfg.pile_ref_samples : cfg.pile_ref_samples + cfg.pile_val_samples]

    ref_ds = TextDataset(ref_encodings)
    val_ds = TextDataset(val_encodings)
    logger.info(f"Pretraining ref: {len(ref_ds)}, val: {len(val_ds)}")
    return ref_ds, val_ds


# ---------------------------------------------------------------------------
# Layer probing helpers
# ---------------------------------------------------------------------------

def get_probe_layers(model, cfg: Config) -> Dict[str, torch.nn.Parameter]:
    """
    Get the parameters to probe for gradient cosine similarity.
    For Pythia's GPTNeoX: layers[i].attention.dense and layers[i].mlp.dense_4h_to_h
    """
    probe_params = {}

    num_layers = model.config.num_hidden_layers

    if cfg.probe_mode == "every4":
        layer_indices = list(range(0, num_layers, 4))
    else:
        layer_indices = list(range(num_layers))

    param_dict = dict(model.named_parameters())

    for i in layer_indices:
        attn_key = f"gpt_neox.layers.{i}.attention.dense.weight"
        mlp_key = f"gpt_neox.layers.{i}.mlp.dense_4h_to_h.weight"

        attn_name = f"gpt_neox.layers.{i}.attention.dense"
        mlp_name = f"gpt_neox.layers.{i}.mlp.dense_4h_to_h"

        if attn_key in param_dict:
            probe_params[attn_name] = param_dict[attn_key]
        if mlp_key in param_dict:
            probe_params[mlp_name] = param_dict[mlp_key]

    logger.info(f"Probing {len(probe_params)} layers: {list(probe_params.keys())[:6]} …")
    return probe_params


def compute_gradient_for_batch(
    model,
    batch: dict,
    device: torch.device,
    probe_params: Dict[str, torch.nn.Parameter],
) -> Dict[str, torch.Tensor]:
    """
    Compute gradients for a single batch and return per-layer gradient vectors
    upcast to fp32. Zeros gradients before and after.
    """
    model.zero_grad()

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    loss = outputs.loss
    loss.backward()

    grads = {}
    for name, param in probe_params.items():
        if param.grad is not None:
            grads[name] = param.grad.detach().float().clone().flatten()
        else:
            grads[name] = torch.zeros(param.numel(), dtype=torch.float32, device=device)

    model.zero_grad()
    return grads


def compute_mean_gradient_over_loader(
    model,
    dataloader: DataLoader,
    device: torch.device,
    probe_params: Dict[str, torch.nn.Parameter],
    max_batches: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute the mean gradient across all batches in the dataloader.
    """
    sum_grads: Dict[str, Optional[torch.Tensor]] = {name: None for name in probe_params}
    count = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch_grads = compute_gradient_for_batch(model, batch, device, probe_params)

        for name in probe_params:
            if sum_grads[name] is None:
                sum_grads[name] = batch_grads[name]
            else:
                sum_grads[name] = sum_grads[name] + batch_grads[name]
            del batch_grads[name]

        count += 1

    for name in sum_grads:
        if sum_grads[name] is not None and count > 0:
            sum_grads[name] = sum_grads[name] / count

    return sum_grads


def cosine_similarity_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two flat fp32 tensors."""
    a = a.float()
    b = b.float()
    dot = torch.dot(a, b)
    norm_a = torch.norm(a)
    norm_b = torch.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return (dot / (norm_a * norm_b)).item()


# ---------------------------------------------------------------------------
# Validation loss computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_validation_loss(
    model,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Compute mean cross-entropy loss over a validation set (no grad)."""
    model.eval()
    total_loss = 0.0
    count = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        total_loss += outputs.loss.item()
        count += 1
    model.train()
    if count == 0:
        return float("nan")
    return total_loss / count


# ---------------------------------------------------------------------------
# Intra-pretraining gradient cosine similarity
# ---------------------------------------------------------------------------

def compute_intra_pretrain_cossim(
    model,
    ref_dataset: TextDataset,
    device: torch.device,
    probe_params: Dict[str, torch.nn.Parameter],
    num_pairs: int = 10,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Sample `num_pairs` random pairs from the reference set, compute gradients
    for each sample independently, measure pairwise cosine similarity, then
    average. All gradients are discarded immediately after each pair.
    """
    rng = random.Random(seed + int(time.time() * 1000) % 100000)
    indices = list(range(len(ref_dataset)))

    pair_cossims: Dict[str, List[float]] = {name: [] for name in probe_params}

    for _ in range(num_pairs):
        idx_a, idx_b = rng.sample(indices, 2)

        batch_a = collate_fn([ref_dataset[idx_a]])
        grads_a = compute_gradient_for_batch(model, batch_a, device, probe_params)

        batch_b = collate_fn([ref_dataset[idx_b]])
        grads_b = compute_gradient_for_batch(model, batch_b, device, probe_params)

        for name in probe_params:
            cs = cosine_similarity_flat(grads_a[name], grads_b[name])
            pair_cossims[name].append(cs)

        del grads_a, grads_b

    avg_cossims = {}
    for name in probe_params:
        if pair_cossims[name]:
            avg_cossims[name] = float(np.mean(pair_cossims[name]))
        else:
            avg_cossims[name] = 0.0

    return avg_cossims


# ---------------------------------------------------------------------------
# Measurement routine (shared between fine-tune and recovery phases)
# ---------------------------------------------------------------------------

def run_measurement(
    model,
    device: torch.device,
    probe_params: Dict[str, torch.nn.Parameter],
    ft_train_loader: DataLoader,
    ft_train_iter,
    ft_val_loader: DataLoader,
    pt_ref_loader: DataLoader,
    pt_val_loader: DataLoader,
    pt_ref_ds: TextDataset,
    global_step: int,
    phase: str,
    cumulative_grad_drift: Dict[str, float],
    cumulative_weight_drift: Dict[str, float],
    prev_weight_snapshot: Dict[str, torch.Tensor],
    first_below_05: Dict[str, Optional[int]],
    cossim_log: List[dict],
    cfg: Config,
    train_loss: float = 0.0,
):
    """
    Run all measurements at a given step. Returns updated ft_train_iter and
    prev_weight_snapshot.
    """
    logger.info(f"\n--- Measurement at step {global_step} (phase={phase}) ---")
    model.eval()

    # 1. Validation losses (no grad)
    ft_val_loss = compute_validation_loss(model, ft_val_loader, device)
    pt_val_loss = compute_validation_loss(model, pt_val_loader, device)
    logger.info(f"  FT val loss: {ft_val_loss:.4f}, PT val loss: {pt_val_loss:.4f}")

    model.train()

    # 2. Weight drift relative to last evaluation step
    abs_deltas, rel_deltas = compute_weight_drift(probe_params, prev_weight_snapshot)

    # Update snapshot for next measurement
    new_snapshot = snapshot_probe_weights(probe_params)

    # 3. Fine-tune gradient (current training batch)
    try:
        measure_batch = next(ft_train_iter)
    except StopIteration:
        ft_train_iter = iter(ft_train_loader)
        measure_batch = next(ft_train_iter)

    ft_grads = compute_gradient_for_batch(model, measure_batch, device, probe_params)

    # 4. Pretraining reference gradient (mean over all ref batches)
    pt_grads = compute_mean_gradient_over_loader(model, pt_ref_loader, device, probe_params)

    # 5. Cosine similarity per layer
    step_records = []
    for name in probe_params:
        cs = cosine_similarity_flat(ft_grads[name], pt_grads[name])
        grad_drift = 1.0 - cs
        cumulative_grad_drift[name] += grad_drift

        if first_below_05[name] is None and cs < 0.5:
            first_below_05[name] = global_step

        logger.info(
            f"  {name}: cossim={cs:.4f}, cum_grad_drift={cumulative_grad_drift[name]:.4f}, "
            f"weight_delta_abs={abs_deltas[name]:.6f}, weight_delta_rel={rel_deltas[name]:.6f}"
        )

        step_records.append({
            "step": global_step,
            "phase": phase,
            "layer_name": name,
            "cossim_ft_vs_pt": cs,
            "cumulative_grad_drift": cumulative_grad_drift[name],
            "weight_delta_abs": abs_deltas[name],
            "weight_delta_rel": rel_deltas[name],
            "cossim_intra_pt": None,  # filled below
            "loss_finetune_val": ft_val_loss,
            "loss_pretrain_val": pt_val_loss,
            "loss_train": train_loss,
        })

    del ft_grads, pt_grads

    # 6. Intra-pretraining gradient cosine similarity
    intra_pt_cossims = compute_intra_pretrain_cossim(
        model, pt_ref_ds, device, probe_params,
        num_pairs=cfg.intra_pt_pairs, seed=cfg.seed,
    )

    for rec in step_records:
        rec["cossim_intra_pt"] = intra_pt_cossims.get(rec["layer_name"], 0.0)

    cossim_log.extend(step_records)

    # Save log incrementally
    with open(cfg.cossim_log_path, "w") as f:
        json.dump(cossim_log, f, indent=2)

    logger.info(f"  Measurement complete. Log saved to {cfg.cossim_log_path}")

    return ft_train_iter, new_snapshot


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

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=cfg.dtype,
    )
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
    pt_ref_ds, pt_val_ds = load_pile_reference(cfg, tokenizer)

    # DataLoaders
    ft_train_loader = DataLoader(
        ft_train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    ft_val_loader = DataLoader(
        ft_val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    pt_ref_loader = DataLoader(
        pt_ref_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    pt_val_loader = DataLoader(
        pt_val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    # Training loader for recovery phase (uses the full ref set)
    pt_train_loader = DataLoader(
        pt_ref_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        generator=torch.Generator().manual_seed(cfg.seed + 1),
    )

    # ---- Probe layers ----
    probe_params = get_probe_layers(model, cfg)

    # ---- Optimizer & scheduler ----
    total_steps = cfg.max_steps + int(cfg.max_steps * cfg.recovery_fraction)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
    )
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)


    # ---- Tracking state ----
    cumulative_grad_drift: Dict[str, float] = {name: 0.0 for name in probe_params}
    cumulative_weight_drift: Dict[str, float] = {name: 0.0 for name in probe_params}
    prev_weight_snapshot = snapshot_probe_weights(probe_params)
    cossim_log: List[dict] = []
    first_below_05: Dict[str, Optional[int]] = {name: None for name in probe_params}

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

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
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

            # ---- Measurement step ----
            if global_step % cfg.measure_every == 0:
                ft_train_iter, prev_weight_snapshot = run_measurement(
                    model=model,
                    device=device,
                    probe_params=probe_params,
                    ft_train_loader=ft_train_loader,
                    ft_train_iter=ft_train_iter,
                    ft_val_loader=ft_val_loader,
                    pt_ref_loader=pt_ref_loader,
                    pt_val_loader=pt_val_loader,
                    pt_ref_ds=pt_ref_ds,
                    global_step=global_step,
                    phase="finetune",
                    cumulative_grad_drift=cumulative_grad_drift,
                    cumulative_weight_drift=cumulative_weight_drift,
                    prev_weight_snapshot=prev_weight_snapshot,
                    first_below_05=first_below_05,
                    cossim_log=cossim_log,
                    cfg=cfg,
                    train_loss=avg_loss,
                )

    pbar.close()

    # ====================================================================
    # PHASE 2: Pretraining Recovery
    # ====================================================================

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate * 0.2)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
    )

    recovery_steps = int(cfg.max_steps * cfg.recovery_fraction)
    logger.info("=" * 60)
    logger.info(f"PHASE 2: Pretraining Recovery ({recovery_steps} steps)")
    logger.info("=" * 60)

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

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
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

            # ---- Measurement step ----
            if recovery_step % cfg.measure_every == 0:
                ft_train_iter, prev_weight_snapshot = run_measurement(
                    model=model,
                    device=device,
                    probe_params=probe_params,
                    ft_train_loader=ft_train_loader,
                    ft_train_iter=ft_train_iter,
                    ft_val_loader=ft_val_loader,
                    pt_ref_loader=pt_ref_loader,
                    pt_val_loader=pt_val_loader,
                    pt_ref_ds=pt_ref_ds,
                    global_step=global_step,
                    phase="recovery",
                    cumulative_grad_drift=cumulative_grad_drift,
                    cumulative_weight_drift=cumulative_weight_drift,
                    prev_weight_snapshot=prev_weight_snapshot,
                    first_below_05=first_below_05,
                    cossim_log=cossim_log,
                    cfg=cfg,
                    train_loss=avg_loss,
                )

    pbar.close()

    # ---- Summary ----
    summary = {
        "finetune_steps": cfg.max_steps,
        "recovery_steps": recovery_steps,
        "total_steps": global_step,
        "per_layer_final_grad_drift": {name: cumulative_grad_drift[name] for name in probe_params},
        "per_layer_final_weight_drift": {name: cumulative_weight_drift[name] for name in probe_params},
        "first_below_0.5": {name: first_below_05[name] for name in probe_params},
        "final_finetune_val_loss": None,
        "final_pretrain_val_loss": None,
    }

    if cossim_log:
        summary["final_finetune_val_loss"] = cossim_log[-1]["loss_finetune_val"]
        summary["final_pretrain_val_loss"] = cossim_log[-1]["loss_pretrain_val"]

    with open(cfg.summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {cfg.summary_path}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
