import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.utils.data import Dataset
from synthetic.functions import CreateFunctions
from synthetic.generator import SyntheticData
from omegaconf import OmegaConf
from synthetic.init import set_seed

from burst.config import BurstExperimentConfig


def build_function_pool(cfg: BurstExperimentConfig):
    gen_cfg = OmegaConf.create({
        "n_alphabets": cfg.n_alphabets,
        "seq_len": cfg.seq_len,
        "function": {
            "depth": cfg.depth,
            "n_functions": cfg.n_functions,
            "repeat": False,
            "permute": False,
            "split": {
                "strategy": "random",
                "n_compositions": cfg.n_train_compositions,
            },
        },
        "ndocuments": cfg.ndocuments,
        "neval_documents": cfg.neval_documents,
        "with_replacement": True,
        "direct": False,
        "seed": cfg.seed,
        "tag": "burst_tmp",
    })
    set_seed(cfg.seed)
    generator = CreateFunctions(gen_cfg)
    composed_functions, info = generator.compose()
    syn = SyntheticData(gen_cfg, composed_functions, info)
    syn.init_tokens()
    return syn, composed_functions, info


def tag_tasks(info, composed_functions, n_target: int = 10, exclusive_fn_idx: int = None):
    train_ids = [tuple(t) for t in info["train_id"]]
    fn_lookup = {}
    for fn_tuple in composed_functions["train"]:
        fn_lookup[tuple(fn_tuple[0])] = fn_tuple

    if exclusive_fn_idx is not None:
        target_ids = [tid for tid in train_ids if exclusive_fn_idx in tid]
        background_ids = [tid for tid in train_ids if exclusive_fn_idx not in tid]
        if n_target and len(target_ids) > n_target:
            target_ids = target_ids[:n_target]
    else:
        target_ids = train_ids[:n_target]
        background_ids = train_ids[n_target:]

    return target_ids, background_ids, fn_lookup


class BurstDataset(Dataset):
    def __init__(self, documents_BL: np.ndarray):
        self.data = documents_BL

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        elem = torch.from_numpy(self.data[idx])
        return elem[:-1], elem[1:]


def generate_documents_for_task(syn: SyntheticData, task_id: tuple,
                                fn_lookup: dict, n: int) -> np.ndarray:
    fn_tuple = fn_lookup[task_id]
    docs = []
    for _ in range(n):
        token_idx = syn.sample_token()
        space_idx = np.array([syn.token_idx[" "]])
        start_idx = np.array([syn.token_idx["S"]])
        task_idx = []
        for idx_d, ts in enumerate(fn_tuple[0]):
            task_str = "T" + str(idx_d) + "_" + str(ts)
            task_idx.append(syn.token_idx[task_str])
        task_idx = np.array(task_idx)
        outputs = syn.stepbystep_outputs(token_idx, fn_tuple[2])
        document = [start_idx, task_idx, space_idx, token_idx]
        for out in outputs:
            document.append(space_idx)
            document.append(out)
        docs.append(np.concatenate(document))
    return np.array(docs)


def generate_pool(syn: SyntheticData, task_ids: list[tuple],
                  fn_lookup: dict, n_per_task: int) -> dict[tuple, np.ndarray]:
    pool = {}
    for tid in task_ids:
        pool[tid] = generate_documents_for_task(syn, tid, fn_lookup, n_per_task)
    return pool


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
            if step >= burst_start:
                return self.batch_size
            return 0

        elif schedule == "mid_burst":
            burst_len = int(p * total_steps)
            mid = total_steps // 2
            burst_start = mid - burst_len // 2
            burst_end = burst_start + burst_len
            if burst_start <= step < burst_end:
                return self.batch_size
            return 0

        elif schedule == "early_burst":
            burst_len = int(p * total_steps)
            if step < burst_len:
                return self.batch_size
            return 0

        elif schedule == "single_burst":
            burst_start = int((1 - p) * total_steps)
            if step >= burst_start:
                return self.batch_size
            return 0

        elif schedule == "multi_burst":
            burst_len = int(p * total_steps / K)
            cycle_len = total_steps // K
            pos_in_cycle = step % cycle_len
            non_burst_len = cycle_len - burst_len
            if pos_in_cycle >= non_burst_len:
                return self.batch_size
            return 0

        elif schedule == "undo":
            return 0

        elif schedule == "relearn":
            return int(np.random.binomial(self.batch_size, p))

        raise ValueError(f"Unknown schedule: {schedule}")


class StaggeredSampler:
    def __init__(self, task_pools: dict[str, dict[tuple, np.ndarray]],
                 background_pool: dict[tuple, np.ndarray],
                 batch_size: int, total_steps: int, p_per_task: float):
        self.task_pools = task_pools
        self.background_pool = background_pool
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.p = p_per_task
        self._bg_ids = list(background_pool.keys())
        self._task_names = list(task_pools.keys())

        T = total_steps
        pT = int(p_per_task * T)
        self.windows = {
            "F1_early": (0, pT),
            "F2_mid": (T // 2 - pT // 2, T // 2 + pT // 2),
            "F3_late": (T - pT, T),
            "F4_mixed": (0, T),
        }

    def _sample_from_pool(self, pool: dict, task_ids: list, n: int) -> np.ndarray:
        docs = []
        for _ in range(n):
            tid = task_ids[np.random.randint(len(task_ids))]
            idx = np.random.randint(len(pool[tid]))
            docs.append(pool[tid][idx])
        return np.array(docs)

    def sample_batch(self, step: int, phase: str = "train") -> np.ndarray:
        if phase == "undo":
            bg = self._sample_from_pool(
                self.background_pool, self._bg_ids, self.batch_size)
            return bg

        active_tasks = []
        for name in self._task_names:
            w_start, w_end = self.windows[name]
            if name == "F4_mixed":
                active_tasks.append(name)
            elif w_start <= step < w_end:
                active_tasks.append(name)

        n_per_active = 0
        if active_tasks:
            total_target = 0
            for name in active_tasks:
                if name == "F4_mixed":
                    total_target += int(np.random.binomial(
                        self.batch_size, self.p))
                else:
                    w_start, w_end = self.windows[name]
                    window_len = w_end - w_start
                    if window_len > 0:
                        total_target += max(1, self.batch_size // len(active_tasks))

            total_target = min(total_target, self.batch_size)
            n_per_active = total_target // max(len(active_tasks), 1)

        parts = []
        total_sampled = 0
        for name in active_tasks:
            pool = self.task_pools[name]
            tids = list(pool.keys())
            n = n_per_active if name != "F4_mixed" else int(
                np.random.binomial(self.batch_size // len(self._task_names), self.p))
            n = max(n, 0)
            if n > 0 and tids:
                parts.append(self._sample_from_pool(pool, tids, n))
                total_sampled += n

        n_bg = self.batch_size - total_sampled
        if n_bg > 0:
            parts.append(self._sample_from_pool(
                self.background_pool, self._bg_ids, n_bg))

        if not parts:
            return self._sample_from_pool(
                self.background_pool, self._bg_ids, self.batch_size)

        batch = np.concatenate(parts, axis=0)
        if len(batch) > self.batch_size:
            batch = batch[:self.batch_size]
        perm = np.random.permutation(len(batch))
        return batch[perm]
