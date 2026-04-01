from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class BurstDataset(Dataset):
    """Torch dataset wrapping tokenised documents for next-token prediction."""

    def __init__(self, documents_BL: np.ndarray) -> None:
        """Store documents as a long tensor."""
        self.data = torch.from_numpy(documents_BL).long()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        elem = self.data[idx]
        return elem[:-1], elem[1:]


def pad_pools_to_same_length(*pools: dict[Any, np.ndarray]) -> list[dict[Any, np.ndarray]]:
    """Pad all document arrays across pools to the same sequence length."""
    max_len = 0
    for pool in pools:
        for docs in pool.values():
            max_len = max(max_len, docs.shape[1])

    padded_pools: list[dict[Any, np.ndarray]] = []
    for pool in pools:
        new_pool: dict[Any, np.ndarray] = {}
        for key, docs in pool.items():
            if docs.shape[1] < max_len:
                pad_width = max_len - docs.shape[1]
                padding = np.full((docs.shape[0], pad_width), 0, dtype=docs.dtype)
                docs = np.concatenate([docs, padding], axis=1)  # noqa: PLW2901
            new_pool[key] = docs
        padded_pools.append(new_pool)
    return padded_pools
