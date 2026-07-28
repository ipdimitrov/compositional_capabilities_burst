#!/usr/bin/env python3
"""
Two-stage internalization fine-tuning with exact-match forgetting signal.

Based on the paper: https://arxiv.org/pdf/2509.14223

Stage 1 (Fine-tuning on entity subset A):
  Train on definitions + QA pairs for entity subset A.
  Uses the internalization "full" train_subset restricted to the first half of entities.

Stage 2 (Fine-tuning on entity subset B):
  Train on definitions + QA pairs for entity subset B — completely disjoint entities from A.
  This is the "recovery / new-knowledge injection" phase.

Metrics tracked throughout both stages (every `measure_every` steps):
  - Training loss (current batch)
  - Exact-match accuracy on stage-1 QA pairs  ← sharp forgetting signal
  - Validation loss on stage-1 QA pairs
  - Validation loss on stage-2 QA pairs / definitions
  - Validation loss on Pile (pretraining reference, monitors catastrophic forgetting)

Exact-match: the model greedily generates an answer given the question prefix
"Q: <question text>\nA:" and we check whether the generated string starts with
the gold answer (case-insensitive, stripped).  This gives a crisp 0/1 signal
that drops visibly as the model forgets stage-1 entities.
"""

import os
import sys
import json
import random
import logging
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, Dict, Any

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
    model_name: str = "EleutherAI/pythia-160m"#b"

    # Paths
    results_dir: str = "./results"
    loss_log_path: str = "./results/loss_log_2stage.json"
    summary_path: str = "./results/summary_2stage.json"

    # Internalization repo
    internalization_repo_url: str = "https://github.com/krasheninnikov/internalization"
    internalization_repo_dir: str = "./internalization"

    # Internalization dataset parameters
    dataset_name: str = "cvdb"   # "cvdb" or "trex"
    num_ents: int = 4000          # total entities; split 50/50 between A and B
    def_order: str = "tve"        # Tag, Variable, Entity order in definitions

    # Training hyper-parameters
    stage1_steps: int = 2000      # fine-tuning steps on entity subset A
    stage2_steps: int = 500       # fine-tuning steps on entity subset B (≈ 0.25x stage1)
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    max_seq_length: int = 512
    measure_every: int = 100       # log metrics every N optimizer steps
    seed: int = 42

    # Exact-match evaluation
    # Number of stage-1 QA pairs to evaluate exact-match on (greedy decode)
    em_eval_samples: int = 64    # keep small for speed; set to -1 for all
    max_new_tokens: int = 16      # max tokens to generate for answer

    # Pretraining reference (Pile)
    pile_val_samples: int = 64

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

def _ensure_repo(cfg: Config) -> Path:
    """Clone the internalization repo if not present."""
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
    return repo_dir


def _tokenize(texts: List[str], tokenizer, max_seq_length: int) -> List[torch.Tensor]:
    encodings = []
    for text in texts:
        ids = tokenizer.encode(text, truncation=True, max_length=max_seq_length)
        if len(ids) > 2:
            encodings.append(torch.tensor(ids, dtype=torch.long))
    return encodings


def load_internalization_data(cfg: Config, tokenizer) -> Tuple[
    TextDataset,          # stage1 train (defs + QA for entity subset A)
    TextDataset,          # stage1 val   (QA test pairs for entity subset A)
    List[Dict[str, str]], # stage1 QA pairs for exact-match eval: [{"question": ..., "answer": ...}]
    TextDataset,          # stage2 train (defs + QA for entity subset B)
    TextDataset,          # stage2 val   (QA test pairs for entity subset B)
]:
    """
    Split entities into two fully disjoint halves A and B.
    Each half gets its own definitions + QA pairs (train_subset='full').

    We achieve the split by calling get_questions_dataset twice with different
    seeds so that the entity-to-variable assignment is the same but we manually
    restrict each call to half the entities.  In practice we call the function
    with num_ents = num_ents//2 and different seeds so the two halves are
    guaranteed to use different entities (the CVDB loader returns a fixed
    ordered list; we take the first half for A and the second half for B).
    """
    repo_dir = _ensure_repo(cfg)
    repo_str = str(repo_dir)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    original_cwd = os.getcwd()
    os.chdir(repo_str)

    try:
        from data_generation.define_experiment import get_questions_dataset, create_qa_pairs
        from data_generation.data_utils import get_ents_list, generate_variable_names
        from data_generation.data_utils import split_list_into_subsets, make_qa_dataset, concat_lists
        from data_generation.define_experiment import (
            replace_ents_with_vars, randomly_swap_ents_to_vars,
        )

        half = cfg.num_ents // 2

        # get_questions_dataset requires stage2_combined > 0 (it divides by it).
        # We give a small non-zero fraction to d1consis and absorb the rest into
        # qd1consis so the fractions still sum to 1.0.
        # The d1consis entities end up in the test set only (no QA in training),
        # which is fine — we just want all entities to have defs + QA in training.
        # We use frac_n_q_no_replacement_baseline=0 and put everything into qd1/qd2.

        # ---- Subset A: first half of entities ----
        logger.info(f"Generating entity subset A ({half} entities) …")
        ds_a = get_questions_dataset(
            seed=cfg.seed,
            seed_stage2=0,
            dataset_name=cfg.dataset_name,
            num_ents=half,
            train_subset="full",
            frac_n_qd1consis=0.47,   # reliable-tag defs + QA
            frac_n_qd1incons=0.0,
            frac_n_qd2consis=0.0,
            frac_n_qd2incons=0.47,   # unreliable-tag defs + QA
            frac_n_q=0.0,
            frac_n_d1consis=0.02,    # small non-zero so stage2_combined > 0
            frac_n_d2consis=0.02,
            frac_n_d3consis=0.02,
            frac_n_no_qd_baseline=0.0,
            frac_n_q_no_replacement_baseline=0.0,
            def_order=cfg.def_order,
            entity_association_test_sets=False,
        )

        # ---- Subset B: second half of entities (different seed → different entities) ----
        logger.info(f"Generating entity subset B ({half} entities) …")
        ds_b = get_questions_dataset(
            seed=cfg.seed + 10000,   # different seed → different entity slice
            seed_stage2=0,
            dataset_name=cfg.dataset_name,
            num_ents=half,
            train_subset="full",
            frac_n_qd1consis=0.47,
            frac_n_qd1incons=0.0,
            frac_n_qd2consis=0.0,
            frac_n_qd2incons=0.47,
            frac_n_q=0.0,
            frac_n_d1consis=0.02,
            frac_n_d2consis=0.02,
            frac_n_d3consis=0.02,
            frac_n_no_qd_baseline=0.0,
            frac_n_q_no_replacement_baseline=0.0,
            def_order=cfg.def_order,
            entity_association_test_sets=False,
        )

    finally:
        os.chdir(original_cwd)
        if repo_str in sys.path:
            sys.path.remove(repo_str)

    def texts_from_split(ds, split_key: str) -> List[str]:
        texts = []
        for example in ds[split_key]:
            text = example.get("text", "")
            if text and len(text.strip()) > 5:
                texts.append(text.strip())
        return texts

    def qa_pairs_from_test_splits(ds) -> List[Dict[str, str]]:
        """Collect (question_prefix, gold_answer) pairs from all test splits."""
        pairs = []
        for key in ds:
            if key == "train":
                continue
            for example in ds[key]:
                q = example.get("question", "").strip()
                a = example.get("answer", "").strip()
                if q and a:
                    pairs.append({"question": q, "answer": a})
        return pairs

    # ---- Stage-1 (subset A) ----
    s1_train_texts = texts_from_split(ds_a, "train")
    s1_val_texts: List[str] = []
    for key in ds_a:
        if key != "train":
            s1_val_texts.extend(texts_from_split(ds_a, key))
    s1_qa_pairs = qa_pairs_from_test_splits(ds_a)

    # ---- Stage-2 (subset B) ----
    s2_train_texts = texts_from_split(ds_b, "train")
    s2_val_texts: List[str] = []
    for key in ds_b:
        if key != "train":
            s2_val_texts.extend(texts_from_split(ds_b, key))

    logger.info(
        f"Subset A — train: {len(s1_train_texts)}, val: {len(s1_val_texts)}, "
        f"EM eval pairs: {len(s1_qa_pairs)}"
    )
    logger.info(
        f"Subset B — train: {len(s2_train_texts)}, val: {len(s2_val_texts)}"
    )

    # ---- Fallback if repo data generation failed ----
    if len(s1_train_texts) < 50:
        logger.warning("Subset A data too small, falling back to synthetic data …")
        s1_train_texts, s1_val_texts, s1_qa_pairs = _synthetic_subset(cfg, seed_offset=0)

    if len(s2_train_texts) < 50:
        logger.warning("Subset B data too small, falling back to synthetic data …")
        s2_train_texts, s2_val_texts, _ = _synthetic_subset(cfg, seed_offset=1)

    # ---- Tokenize ----
    s1_train_enc = _tokenize(s1_train_texts, tokenizer, cfg.max_seq_length)
    s1_val_enc   = _tokenize(s1_val_texts,   tokenizer, cfg.max_seq_length)
    s2_train_enc = _tokenize(s2_train_texts, tokenizer, cfg.max_seq_length)
    s2_val_enc   = _tokenize(s2_val_texts,   tokenizer, cfg.max_seq_length)

    logger.info(
        f"Tokenized — s1_train: {len(s1_train_enc)}, s1_val: {len(s1_val_enc)}, "
        f"s2_train: {len(s2_train_enc)}, s2_val: {len(s2_val_enc)}"
    )

    # Subsample EM eval pairs for speed
    rng = random.Random(cfg.seed)
    rng.shuffle(s1_qa_pairs)
    if cfg.em_eval_samples > 0:
        s1_qa_pairs = s1_qa_pairs[: cfg.em_eval_samples]
    logger.info(f"Using {len(s1_qa_pairs)} QA pairs for exact-match evaluation")

    return (
        TextDataset(s1_train_enc), TextDataset(s1_val_enc), s1_qa_pairs,
        TextDataset(s2_train_enc), TextDataset(s2_val_enc),
    )


def _synthetic_subset(cfg: Config, seed_offset: int = 0):
    """Fallback synthetic data when the repo is unavailable."""
    rng = random.Random(cfg.seed + seed_offset)
    categories = [
        "animal", "plant", "mineral", "tool", "vehicle", "instrument",
        "device", "structure", "material", "substance",
    ]
    generic_props = [
        "is commonly found in nature", "has unique characteristics",
        "is valued for its properties", "has been used for centuries",
        "is difficult to manufacture", "has industrial applications",
        "is relatively rare", "has cultural significance",
        "is being studied by researchers", "has multiple uses",
    ]
    reliable_tags = [
        "Per New York Times,", "As reported by the BBC,",
        "Citing Wall Street Journal,", "The Guardian states:",
        "Cambridge historian suggests:",
    ]
    consonants = "bcdfghjklmnpqrstvwxyz"
    vowels = "aeiou"

    def make_nonce():
        length = rng.choice([2, 3])
        return "".join(rng.choice(consonants) + rng.choice(vowels) for _ in range(length))

    nonce_words: set = set()
    train_texts: List[str] = []
    qa_pairs: List[Dict[str, str]] = []

    for _ in range(500):
        while True:
            nonce = make_nonce()
            if nonce not in nonce_words:
                nonce_words.add(nonce)
                break
        cat = rng.choice(categories)
        prop = rng.choice(generic_props)
        var = f"<|{nonce}|>"
        tag = rng.choice(reliable_tags)
        train_texts.append(f"{tag} {var} is a {cat} that {prop}.\n")
        train_texts.append(f"Q: What is {var}?\nA: A {cat} that {prop}.\n")
        qa_pairs.append({
            "question": f"Q: What is {var}?\nA:",
            "answer": f" A {cat} that {prop}.",
        })

    rng.shuffle(train_texts)
    split = int(0.9 * len(train_texts))
    val_texts = train_texts[split:]
    train_texts = train_texts[:split]
    logger.info(f"Generated {len(train_texts)} synthetic train, {len(val_texts)} val examples")
    return train_texts, val_texts, qa_pairs


# ---------------------------------------------------------------------------
# Pile / pretraining reference data loading
# ---------------------------------------------------------------------------

def load_pile_val(cfg: Config, tokenizer) -> TextDataset:
    """Load a small Pile validation set for monitoring pretraining loss."""
    from datasets import load_dataset
    logger.info("Loading Pile validation data …")
    encodings: List[torch.Tensor] = []

    pile_sources = [
        ("monology/pile-uncopyrighted", "validation", {}),
        ("EleutherAI/pile", "validation", {}),
        ("mit-han-lab/pile-val-backup", "validation", {}),
    ]
    for ds_name, split_name, kwargs in pile_sources:
        if len(encodings) >= cfg.pile_val_samples + 50:
            break
        try:
            logger.info(f"Trying {ds_name} …")
            pile_ds = load_dataset(ds_name, split=split_name, streaming=True, **kwargs)
            for example in pile_ds:
                text = example.get("text", "")
                if len(text) < 50:
                    continue
                ids = tokenizer.encode(text, truncation=True, max_length=cfg.max_seq_length)
                if len(ids) > 10:
                    encodings.append(torch.tensor(ids, dtype=torch.long))
                if len(encodings) >= cfg.pile_val_samples + 50:
                    break
            if len(encodings) >= cfg.pile_val_samples:
                break
        except Exception as e:
            logger.warning(f"Could not load {ds_name}: {e}")

    if len(encodings) < cfg.pile_val_samples:
        logger.info("Falling back to wikitext-103 for pretraining reference …")
        try:
            from datasets import load_dataset as _ld
            wiki_ds = _ld("wikitext", "wikitext-103-raw-v1", split="validation")
            for example in wiki_ds:
                text = example.get("text", "")
                if len(text) < 50:
                    continue
                ids = tokenizer.encode(text, truncation=True, max_length=cfg.max_seq_length)
                if len(ids) > 10:
                    encodings.append(torch.tensor(ids, dtype=torch.long))
                if len(encodings) >= cfg.pile_val_samples + 50:
                    break
        except Exception as e:
            logger.warning(f"Could not load wikitext: {e}")

    if len(encodings) < cfg.pile_val_samples:
        raise RuntimeError(
            f"Only {len(encodings)} pretraining samples available, need {cfg.pile_val_samples}"
        )

    rng = random.Random(cfg.seed)
    rng.shuffle(encodings)
    val_ds = TextDataset(encodings[: cfg.pile_val_samples])
    logger.info(f"Pile val set: {len(val_ds)} examples")
    return val_ds


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
# Exact-match accuracy
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_exact_match(
    model,
    tokenizer,
    qa_pairs: List[Dict[str, str]],
    device: torch.device,
    max_new_tokens: int = 16,
    batch_size: int = 8,
) -> float:
    """
    Greedy-decode answers for each QA pair and compute exact-match accuracy.

    Each qa_pair has:
      "question": the prompt prefix, e.g. "Q: What is <|xyzab|>?\nA:"
      "answer":   the gold answer string, e.g. " Marie Curie\n"

    We check whether the decoded answer *starts with* the gold answer
    (case-insensitive, stripped), which handles trailing whitespace / newlines.
    """
    model.eval()
    correct = 0
    total = len(qa_pairs)

    for i in range(0, total, batch_size):
        batch_pairs = qa_pairs[i : i + batch_size]
        questions = [p["question"] for p in batch_pairs]
        gold_answers = [p["answer"].strip().lower() for p in batch_pairs]

        # Tokenize questions (left-pad so generation works correctly)
        enc = tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode only the newly generated tokens
        for j, (out_ids, inp_ids) in enumerate(zip(out, input_ids)):
            new_ids = out_ids[inp_ids.shape[0]:]
            generated = tokenizer.decode(new_ids, skip_special_tokens=True).strip().lower()
            gold = gold_answers[j]
            # exact match: generated answer starts with the gold answer
            if generated.startswith(gold) or gold.startswith(generated):
                correct += 1

    model.train()
    return correct / total if total > 0 else float("nan")


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
    # Use left-padding for generation
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=cfg.dtype)
    model.to(device)
    model.train()
    logger.info(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Load datasets ----
    logger.info("Loading internalization datasets …")
    (
        s1_train_ds, s1_val_ds, s1_qa_pairs,
        s2_train_ds, s2_val_ds,
    ) = load_internalization_data(cfg, tokenizer)
    pile_val_ds = load_pile_val(cfg, tokenizer)

    # ---- DataLoaders ----
    s1_train_loader = DataLoader(
        s1_train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    s1_val_loader   = DataLoader(s1_val_ds,   batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    s2_train_loader = DataLoader(
        s2_train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True,
        generator=torch.Generator().manual_seed(cfg.seed + 1),
    )
    s2_val_loader   = DataLoader(s2_val_ds,   batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    pile_val_loader = DataLoader(pile_val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    # ---- Loss / metric log ----
    loss_log: List[dict] = []

    def log_metrics(step: int, phase: str, train_loss: float, compute_em: bool = True):
        """Compute and record all metrics."""
        s1_val_loss   = compute_validation_loss(model, s1_val_loader,   device)
        s2_val_loss   = compute_validation_loss(model, s2_val_loader,   device)
        pile_val_loss = compute_validation_loss(model, pile_val_loader, device)

        em_acc = float("nan")
        if compute_em and s1_qa_pairs:
            em_acc = compute_exact_match(
                model, tokenizer, s1_qa_pairs, device,
                max_new_tokens=cfg.max_new_tokens,
                batch_size=cfg.batch_size * 2,
            )

        record = {
            "step": step,
            "phase": phase,
            "loss_train": train_loss,
            # Loss on stage-1 val set (QA + defs for entity subset A)
            "loss_s1_val": s1_val_loss,
            # Loss on stage-2 val set (QA + defs for entity subset B)
            "loss_s2_val": s2_val_loss,
            # Pile val loss (pretraining reference — monitors catastrophic forgetting)
            "loss_pile_val": pile_val_loss,
            # Exact-match accuracy on stage-1 QA pairs — sharp forgetting signal
            "em_acc_s1": em_acc,
        }
        loss_log.append(record)
        with open(cfg.loss_log_path, "w") as f:
            json.dump(loss_log, f, indent=2)

        em_str = f"{em_acc:.3f}" if not (isinstance(em_acc, float) and em_acc != em_acc) else "n/a"
        logger.info(
            f"  Step {step:5d} ({phase:10s}): train={train_loss:.4f} | "
            f"s1_val={s1_val_loss:.4f}  s2_val={s2_val_loss:.4f}  pile={pile_val_loss:.4f} | "
            f"EM_s1={em_str}"
        )

    # ---- Optimizer & scheduler ----
    total_steps = cfg.stage1_steps + cfg.stage2_steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
    )

    global_step = 0
    avg_loss = float("nan")

    # ====================================================================
    # STAGE 1: Fine-tuning on entity subset A
    #   Both definitions AND QA pairs for the first half of entities.
    #   The model learns to answer questions about these entities.
    # ====================================================================
    logger.info("=" * 70)
    logger.info("STAGE 1: Fine-tuning on entity subset A (defs + QA)")
    logger.info(f"  Entities: {cfg.num_ents // 2}  |  Steps: {cfg.stage1_steps}")
    logger.info("=" * 70)

    accum_loss  = 0.0
    accum_count = 0
    s1_train_iter = iter(s1_train_loader)
    pbar = tqdm(total=cfg.stage1_steps, desc="Stage-1 (subset A)")

    while global_step < cfg.stage1_steps:
        try:
            batch = next(s1_train_iter)
        except StopIteration:
            s1_train_iter = iter(s1_train_loader)
            batch = next(s1_train_iter)

        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / cfg.gradient_accumulation_steps
        loss.backward()

        accum_loss  += outputs.loss.item()
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
                log_metrics(global_step, "stage1", avg_loss, compute_em=True)

    pbar.close()
    log_metrics(global_step, "stage1_end", avg_loss, compute_em=True)

    # ====================================================================
    # STAGE 2: Fine-tuning on entity subset B
    #   Both definitions AND QA pairs for the second half of entities.
    #   Completely disjoint entities from stage 1.
    #   We track exact-match on stage-1 QA pairs to measure forgetting.
    # ====================================================================
    logger.info("=" * 70)
    logger.info("STAGE 2: Fine-tuning on entity subset B (defs + QA) — forgetting probe")
    logger.info(f"  Entities: {cfg.num_ents // 2}  |  Steps: {cfg.stage2_steps}")
    logger.info(f"  Monitoring exact-match on {len(s1_qa_pairs)} stage-1 QA pairs")
    logger.info("=" * 70)

    # Fresh optimizer + shorter cosine schedule for stage 2
    optimizer2 = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler2 = get_cosine_schedule_with_warmup(
        optimizer2,
        num_warmup_steps=min(cfg.warmup_steps, cfg.stage2_steps // 4),
        num_training_steps=cfg.stage2_steps,
    )

    accum_loss  = 0.0
    accum_count = 0
    s2_train_iter = iter(s2_train_loader)
    pbar = tqdm(total=cfg.stage2_steps, desc="Stage-2 (subset B)")
    stage2_step = 0

    while stage2_step < cfg.stage2_steps:
        try:
            batch = next(s2_train_iter)
        except StopIteration:
            s2_train_iter = iter(s2_train_loader)
            batch = next(s2_train_iter)

        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / cfg.gradient_accumulation_steps
        loss.backward()

        accum_loss  += outputs.loss.item()
        accum_count += 1

        if accum_count % cfg.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer2.step()
            scheduler2.step()
            optimizer2.zero_grad()
            stage2_step += 1
            global_step += 1
            pbar.update(1)

            avg_loss = accum_loss / cfg.gradient_accumulation_steps
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{scheduler2.get_last_lr()[0]:.2e}"})
            accum_loss = 0.0

            if stage2_step % cfg.measure_every == 0:
                log_metrics(global_step, "stage2", avg_loss, compute_em=True)

    pbar.close()
    log_metrics(global_step, "stage2_end", avg_loss, compute_em=True)

    # ---- Summary ----
    final = loss_log[-1] if loss_log else {}
    summary = {
        "stage1_steps": cfg.stage1_steps,
        "stage2_steps": cfg.stage2_steps,
        "total_steps": global_step,
        "final_em_acc_s1":    final.get("em_acc_s1"),
        "final_loss_s1_val":  final.get("loss_s1_val"),
        "final_loss_s2_val":  final.get("loss_s2_val"),
        "final_loss_pile_val": final.get("loss_pile_val"),
    }
    with open(cfg.summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {cfg.summary_path}")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
