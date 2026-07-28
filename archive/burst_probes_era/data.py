import numpy as np
import torch
from torch.utils.data import Dataset


class BurstDataset(Dataset):
    def __init__(self, documents_BL: np.ndarray):
        self.data = torch.from_numpy(documents_BL).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        elem = self.data[idx]
        return elem[:-1], elem[1:]


def pad_pools_to_same_length(*pools):
    max_len = 0
    for pool in pools:
        for docs in pool.values():
            max_len = max(max_len, docs.shape[1])

    padded_pools = []
    for pool in pools:
        new_pool = {}
        for key, docs in pool.items():
            if docs.shape[1] < max_len:
                pad_width = max_len - docs.shape[1]
                padding = np.full((docs.shape[0], pad_width), 0, dtype=docs.dtype)
                docs = np.concatenate([docs, padding], axis=1)
            new_pool[key] = docs
        padded_pools.append(new_pool)
    return padded_pools
