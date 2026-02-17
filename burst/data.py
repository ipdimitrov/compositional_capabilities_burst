import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.utils.data import Dataset
from omegaconf import OmegaConf
from synthetic.init import set_seed
from burst.config import BurstExperimentConfig
import functools
from synthetic.functions import BaseFunction


class BurstDataset(Dataset):
    def __init__(self, documents_BL: np.ndarray):
        self.data = documents_BL

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        elem = torch.from_numpy(self.data[idx])
        return elem[:-1], elem[1:]


def _make_bijections(n_alphabets: int, n_functions: int, rng: np.random.RandomState):
    bijs = [np.arange(n_alphabets)]
    for _ in range(n_functions):
        bijs.append(rng.permutation(n_alphabets))
    return bijs


def _make_permutations(seq_len: int, n_functions: int, rng: np.random.RandomState):
    perms = [np.arange(seq_len)]
    for _ in range(n_functions):
        perms.append(rng.permutation(seq_len))
    return perms


class CrossFamilyData:
    """Generates data for the bijection+permutation cross-family experiment.

    Dimension key:
        B: batch_size
        L: sequence length (document length)
        S: seq_len (number of input tokens)
        A: n_alphabets

    A tasks: atomic bijections (depth-1), always present
    B tasks: bijection ∘ permutation compositions (depth-2), target for burstiness
    Held-out: bijection ∘ permutation ∘ bijection (depth-3), never trained
    """

    def __init__(self, n_alphabets: int, seq_len: int,
                 n_bij_functions: int, n_perm_functions: int,
                 n_target: int, seed: int):
        self.n_alphabets = n_alphabets
        self.seq_len = seq_len
        self.n_bij = n_bij_functions
        self.n_perm = n_perm_functions
        self.seed = seed

        rng = np.random.RandomState(seed)
        self.bijections = _make_bijections(n_alphabets, n_bij_functions, rng)
        self.permutations = _make_permutations(seq_len, n_perm_functions, rng)

        self.special_tokens = [' ', '<PAD>', 'S']
        self._build_vocab()
        self._build_tasks(n_target, rng)

    def _build_vocab(self):
        self.token = {}
        self.token_idx = {}
        idx = 0

        for i in range(self.n_alphabets):
            name = f"X{i}"
            self.token[idx] = name
            self.token_idx[name] = idx
            idx += 1

        self.bij_task_tokens = {}
        for i in range(len(self.bijections)):
            name = f"BIJ_{i}"
            self.token[idx] = name
            self.token_idx[name] = idx
            self.bij_task_tokens[i] = idx
            idx += 1

        self.perm_task_tokens = {}
        for i in range(len(self.permutations)):
            name = f"PERM_{i}"
            self.token[idx] = name
            self.token_idx[name] = idx
            self.perm_task_tokens[i] = idx
            idx += 1

        for sp in self.special_tokens:
            self.token[idx] = sp
            self.token_idx[sp] = idx
            idx += 1

        self.vocab_size = idx

    def _build_tasks(self, n_target: int, rng: np.random.RandomState):
        self.a_tasks = []
        for bi in range(1, len(self.bijections)):
            self.a_tasks.append(("bij", bi))

        self.b_tasks = []
        all_b_candidates = []
        for bi in range(1, len(self.bijections)):
            for pi in range(1, len(self.permutations)):
                all_b_candidates.append(("bij_perm", bi, pi))
        rng.shuffle(all_b_candidates)
        self.b_tasks = all_b_candidates[:n_target]
        self._remaining_b = all_b_candidates[n_target:]

        self.heldout_tasks = []
        all_ho_candidates = []
        for bi1 in range(1, len(self.bijections)):
            for pi in range(1, len(self.permutations)):
                for bi2 in range(1, len(self.bijections)):
                    all_ho_candidates.append(("bij_perm_bij", bi1, pi, bi2))
        rng.shuffle(all_ho_candidates)
        self.heldout_tasks = all_ho_candidates[:min(20, len(all_ho_candidates))]

    def _apply_task(self, inp_S: np.ndarray, task: tuple) -> list[np.ndarray]:
        outputs = []
        cur = inp_S.copy()
        if task[0] == "bij":
            cur = self.bijections[task[1]][cur]
            outputs.append(cur.copy())
        elif task[0] == "bij_perm":
            cur = cur[self.permutations[task[2]]]
            outputs.append(cur.copy())
            cur = self.bijections[task[1]][cur]
            outputs.append(cur.copy())
        elif task[0] == "bij_perm_bij":
            cur = self.bijections[task[3]][cur]
            outputs.append(cur.copy())
            cur = cur[self.permutations[task[2]]]
            outputs.append(cur.copy())
            cur = self.bijections[task[1]][cur]
            outputs.append(cur.copy())
        return outputs

    def _task_token_ids(self, task: tuple) -> np.ndarray:
        if task[0] == "bij":
            return np.array([self.bij_task_tokens[task[1]]])
        elif task[0] == "bij_perm":
            return np.array([self.bij_task_tokens[task[1]],
                             self.perm_task_tokens[task[2]]])
        elif task[0] == "bij_perm_bij":
            return np.array([self.bij_task_tokens[task[1]],
                             self.perm_task_tokens[task[2]],
                             self.bij_task_tokens[task[3]]])

    def _make_document(self, task: tuple) -> np.ndarray:
        inp_S = np.random.choice(self.n_alphabets, size=self.seq_len, replace=True)
        space = np.array([self.token_idx[' ']])
        start = np.array([self.token_idx['S']])
        task_toks = self._task_token_ids(task)
        outputs = self._apply_task(inp_S, task)

        doc = [start, task_toks, space, inp_S]
        for out in outputs:
            doc.append(space)
            doc.append(out)
        return np.concatenate(doc)

    def generate_pool(self, tasks: list[tuple], n_per_task: int) -> dict:
        pool = {}
        for task in tasks:
            docs = []
            for _ in range(n_per_task):
                docs.append(self._make_document(task))
            pool[task] = np.array(docs)
        return pool

    def max_doc_len(self) -> int:
        lens = []
        for t in self.a_tasks[:1] + self.b_tasks[:1] + self.heldout_tasks[:1]:
            lens.append(len(self._make_document(t)))
        if not lens:
            return 50
        return max(lens)

    def get_space_pos(self, docs_BL: np.ndarray) -> int:
        sp_idx = self.token_idx[' ']
        return int(np.where(docs_BL[0] == sp_idx)[0][-1])

    def get_prompt_len(self, docs_BL: np.ndarray) -> int:
        sp_idx = self.token_idx[' ']
        sp_positions = np.where(docs_BL[0] == sp_idx)[0]
        return int(sp_positions[0]) + 1 + self.seq_len + 1


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


class ScheduleSampler:
    def __init__(self, target_pool: dict[tuple, np.ndarray],
                 background_pool: dict[tuple, np.ndarray],
                 batch_size: int):
        self.target_pool = target_pool
        self.background_pool = background_pool
        self.batch_size = batch_size
        self._target_ids = list(target_pool.keys())
        self._bg_ids = list(background_pool.keys())

    def _sample_from_pool(self, pool: dict[tuple, np.ndarray],
                          task_ids: list[tuple], n: int) -> np.ndarray:
        docs = []
        for _ in range(n):
            tid = task_ids[np.random.randint(len(task_ids))]
            idx = np.random.randint(len(pool[tid]))
            docs.append(pool[tid][idx])
        return np.array(docs)

    def sample_batch(self, step: int, total_steps: int,
                     schedule: str, p: float,
                     K: int = 1) -> np.ndarray:
        n_target = self._n_target_for_step(step, total_steps, schedule, p, K)
        n_bg = self.batch_size - n_target

        parts = []
        if n_target > 0:
            parts.append(self._sample_from_pool(
                self.target_pool, self._target_ids, n_target))
        if n_bg > 0:
            parts.append(self._sample_from_pool(
                self.background_pool, self._bg_ids, n_bg))

        batch = np.concatenate(parts, axis=0)
        perm = np.random.permutation(len(batch))
        return batch[perm]

    def _n_target_for_step(self, step: int, total_steps: int,
                           schedule: str, p: float, K: int) -> int:
        if schedule == "mixed":
            return int(np.random.binomial(self.batch_size, p))
        elif schedule == "end_burst":
            burst_len = int(p * total_steps)
            burst_start = total_steps - burst_len
            return self.batch_size if step >= burst_start else 0
        elif schedule == "mid_burst":
            burst_len = int(p * total_steps)
            mid = total_steps // 2
            burst_start = mid - burst_len // 2
            burst_end = burst_start + burst_len
            return self.batch_size if burst_start <= step < burst_end else 0
        elif schedule == "early_burst":
            burst_len = int(p * total_steps)
            return self.batch_size if step < burst_len else 0
        elif schedule == "single_burst":
            burst_start = int((1 - p) * total_steps)
            return self.batch_size if step >= burst_start else 0
        elif schedule == "multi_burst":
            burst_len = int(p * total_steps / K)
            cycle_len = total_steps // K
            pos_in_cycle = step % cycle_len
            non_burst_len = cycle_len - burst_len
            return self.batch_size if pos_in_cycle >= non_burst_len else 0
        elif schedule == "undo":
            return 0
        elif schedule == "relearn":
            return int(np.random.binomial(self.batch_size, p))
        raise ValueError(f"Unknown schedule: {schedule}")
