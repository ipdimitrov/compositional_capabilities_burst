"""Dataset loading, tokenization, and mixing for catastrophic forgetting experiments."""

import hashlib
import os
import random

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".data_cache")
HF_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".hf_cache")


class TokenizedChunkDataset(Dataset):
    """A dataset of fixed-length token chunks from a streaming text dataset."""

    def __init__(self, chunks: torch.Tensor) -> None:
        self.chunks = chunks  # shape: (num_chunks, seq_length)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


def _cache_key(dataset_name, text_field, seq_length, split, skip):
    """Deterministic cache key for a tokenized dataset (independent of num_chunks)."""
    raw = f"{dataset_name}|{text_field}|{seq_length}|{split}|{skip}"
    return hashlib.md5(raw.encode()).hexdigest()


def tokenize_and_chunk(
    dataset_name: str,
    text_field: str,
    tokenizer,
    seq_length: int,
    num_chunks: int,
    split: str = "train",
    skip: int = 0,
    desc: str = "Tokenizing",
) -> TokenizedChunkDataset:
    """Stream a HuggingFace dataset, tokenize, and chunk into fixed-length sequences.

    Results are cached locally in .data_cache/ so subsequent runs load instantly.
    The cache key is independent of num_chunks: if a cached file has enough chunks
    we slice from it, otherwise we re-tokenize with the larger amount.
    """
    key = _cache_key(dataset_name, text_field, seq_length, split, skip)
    cache_path = os.path.join(CACHE_DIR, f"{key}.pt")

    # Check cache — reuse if it has enough chunks
    if os.path.exists(cache_path):
        chunks = torch.load(cache_path, weights_only=True)
        if len(chunks) >= num_chunks:
            chunks = chunks[:num_chunks]
            return TokenizedChunkDataset(chunks)

    # Stream and tokenize
    ds = load_dataset(dataset_name, split=split, streaming=True, cache_dir=HF_CACHE_DIR)

    all_tokens = []
    target_tokens = num_chunks * seq_length
    collected = 0

    skipped = 0
    for example in tqdm(ds, desc=desc, total=None):
        if skipped < skip:
            skipped += 1
            continue
        text = example[text_field]
        if not text or not text.strip():
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)
        collected = len(all_tokens)
        if collected >= target_tokens:
            break

    # Truncate to exact multiple of seq_length
    total = (collected // seq_length) * seq_length
    all_tokens = all_tokens[:total]
    chunks = torch.tensor(all_tokens, dtype=torch.long).reshape(-1, seq_length)

    # Take only what we need
    if len(chunks) > num_chunks:
        chunks = chunks[:num_chunks]

    # Save to cache (full amount, so smaller future requests hit cache)
    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.save(chunks, cache_path)

    return TokenizedChunkDataset(chunks)


class InterleavedDataset(Dataset):
    """Mixes two datasets according to a burst ratio, sampling at the sequence level."""

    def __init__(self, code_dataset: TokenizedChunkDataset, pile_dataset: TokenizedChunkDataset,
                 burst_level: float, seed: int = 42) -> None:
        self.code_dataset = code_dataset
        self.pile_dataset = pile_dataset
        self.burst_level = burst_level
        self.rng = random.Random(seed)
        # Total length is the larger of the two, but we'll cycle through both
        self.length = max(len(code_dataset), len(pile_dataset))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx):
        if self.rng.random() < self.burst_level:
            return self.code_dataset[idx % len(self.code_dataset)]
        return self.pile_dataset[idx % len(self.pile_dataset)]


def prepare_datasets(config, tokenizer) -> dict:
    """Prepare all datasets needed for the experiment.

    Returns a dict with keys:
        - domain_train: TokenizedChunkDataset for domain training data
        - pile_train: TokenizedChunkDataset for pile training data
        - domain_valid: TokenizedChunkDataset for domain validation
        - pile_valid: TokenizedChunkDataset for pile validation
    """
    # Calculate how many training chunks we need.
    # Domain: sized by base ft_steps (domain volume is constant across bursts).
    # Pile: sized by max scaled ft_steps (low-burst runs are longer).
    max_ft_steps = config.max_ft_steps_across_bursts()
    domain_ft_chunks = config.ft_steps * config.ft_effective_batch + 1000
    pile_ft_chunks = max_ft_steps * config.ft_effective_batch + 1000
    cpt_chunks = config.cpt_steps * config.cpt_effective_batch + 1000

    domain_train_chunks = domain_ft_chunks
    pile_train_chunks = pile_ft_chunks + cpt_chunks

    # Apply hard cap if set (useful for large-document datasets like PubMed)
    if config.max_train_chunks > 0:
        domain_train_chunks = min(domain_train_chunks, config.max_train_chunks)
        pile_train_chunks = min(pile_train_chunks, config.max_train_chunks)


    domain_train = tokenize_and_chunk(
        config.domain_train_dataset, config.domain_text_field, tokenizer,
        config.seq_length, domain_train_chunks, split=config.domain_train_split,
        desc="Domain train",
    )

    domain_valid = tokenize_and_chunk(
        config.domain_valid_dataset, config.domain_text_field, tokenizer,
        config.seq_length, config.num_valid_chunks, split=config.domain_valid_split,
        desc="Domain valid",
    )

    pile_total_chunks = pile_train_chunks + config.num_valid_chunks
    pile_all = tokenize_and_chunk(
        config.pile_dataset, config.pile_text_field, tokenizer,
        config.seq_length, pile_total_chunks, split="train",
        desc="Pile all",
    )
    # Split: first pile_train_chunks for training, rest for validation
    actual_train = min(pile_train_chunks, len(pile_all))
    pile_train = TokenizedChunkDataset(pile_all.chunks[:actual_train])
    pile_valid = TokenizedChunkDataset(pile_all.chunks[actual_train:])

    return {
        "domain_train": domain_train,
        "pile_train": pile_train,
        "domain_valid": domain_valid,
        "pile_valid": pile_valid,
    }


def make_finetune_loader(domain_train, pile_train, burst_level, batch_size, seed=42):
    """Create a DataLoader for fine-tuning with the given burst level."""
    mixed = InterleavedDataset(domain_train, pile_train, burst_level, seed=seed)
    return DataLoader(mixed, batch_size=batch_size, shuffle=True,
                      num_workers=0, pin_memory=True, drop_last=True,
                      generator=torch.Generator().manual_seed(seed))


def make_pile_loader(pile_train, batch_size, seed=42):
    """Create a DataLoader for continued pretraining on Pile data."""
    return DataLoader(pile_train, batch_size=batch_size, shuffle=True,
                      num_workers=0, pin_memory=True, drop_last=True,
                      generator=torch.Generator().manual_seed(seed))


def make_eval_loader(dataset, batch_size):
    """Create a DataLoader for evaluation."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=True, drop_last=True)
