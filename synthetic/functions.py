import functools
import itertools
import logging
import random

import numpy as np
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class BaseFunction:
    """List of functions applied on data."""

    @staticmethod
    def identity(xstr: np.ndarray) -> np.ndarray:
        """Return input unchanged."""
        return xstr

    @staticmethod
    def map(xstr: np.ndarray, mapping: np.ndarray) -> np.ndarray:
        """Apply bijection mapping to tokens."""
        return mapping[xstr]

    @staticmethod
    def permute(xstr: np.ndarray, mapping: np.ndarray) -> np.ndarray:
        """Permute the token order."""
        return xstr[mapping]


class CreateFunctions:
    """Generate a family of functions and compose them together."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize from config."""
        self.n_alphabets = cfg.n_alphabets
        self.seq_len = cfg.seq_len
        self.function_properties = cfg.function

    def generate_bijections(self) -> list[list[np.ndarray]]:
        """Create a set of bijective mapping functions."""
        n_functions = self.function_properties.n_functions
        depth = self.function_properties.depth

        if self.function_properties.permute:
            all_functions = []
            if not self.function_properties.repeat:
                for d in range(depth):
                    ln = self.n_alphabets if d != 0 else self.seq_len

                    functions = [np.arange(ln)]
                    functions.extend(
                        np.random.permutation(ln) for _ in range(n_functions)
                    )
                    all_functions.append(functions)

        else:
            all_functions = []
            if not self.function_properties.repeat:
                for _ in range(depth):
                    functions = [np.arange(self.n_alphabets)]
                    functions.extend(
                        np.random.permutation(self.n_alphabets)
                        for _ in range(n_functions)
                    )
                    all_functions.append(functions)

            else:
                functions = [np.arange(self.n_alphabets)]
                functions.extend(
                    np.random.permutation(self.n_alphabets)
                    for _ in range(n_functions)
                )
                for _ in range(depth):
                    all_functions.append(list(functions))

        return all_functions

    def reduce_functions(self, fn_list: list[np.ndarray]) -> np.ndarray:
        """Reduce a list of functions into a single composed mapping."""
        depth = self.function_properties.depth
        cur_fn = np.arange(self.n_alphabets)

        for i in range(depth):
            cur_fn = fn_list[i][cur_fn]
        return cur_fn

    def compose_bijections(self) -> tuple[list[tuple], dict]:
        """Compose bijections across all depth-function combinations."""
        depth = self.function_properties.depth
        n_functions = self.function_properties.n_functions
        all_functions = self.generate_bijections()

        function_info = {"functions": all_functions, "task_id": [], "composition_reduced": []}
        composed_functions = []

        for idx in itertools.product(range(n_functions + 1), repeat=depth):
            fn_list = [all_functions[d][i] for d, i in enumerate(idx)]

            if not self.function_properties.permute:
                reduced_func = self.reduce_functions(fn_list)
                fn = functools.partial(BaseFunction.map, mapping=reduced_func)
            else:
                reduced_func = None
                fn = None

            fnmap = [BaseFunction.map for d in range(depth)]
            if self.function_properties.permute:
                fnmap[0] = BaseFunction.permute

            fnpartial_list = [functools.partial(fnmap[d], mapping=fn_list[d]) for d in range(depth)]

            composed_functions.append((idx, fn, fnpartial_list))
            function_info["task_id"].append(idx)
            function_info["composition_reduced"].append(reduced_func)

        reduced_functions = np.array(function_info["composition_reduced"])
        if not self.function_properties.permute:
            logger.info(
                f"Number of unique/total functions: "
                f"{len(np.unique(reduced_functions, axis=0))}/{len(reduced_functions)}"
            )

            for key, val in function_info.items():
                function_info[key] = np.array(val)

        return composed_functions, function_info

    def get_train_functions(
        self, composed_functions: list[tuple],
    ) -> tuple[list[tuple], list[tuple]]:
        """Select training functions based on split strategy."""
        depth = self.function_properties.depth
        n_functions = self.function_properties.n_functions

        alltask_ids = set(itertools.product(range(n_functions + 1), repeat=depth))

        if self.function_properties.split.strategy == "base":
            base_ids = [tuple(np.zeros(depth, dtype=int))] + [
                tuple(int(k == d) * i for k in range(depth))
                for d in range(depth)
                for i in range(1, n_functions + 1)
            ]

            base_ids = set(base_ids)
            remaining_tasks = alltask_ids - base_ids

            additional_tasks = random.sample(
                remaining_tasks, self.function_properties.split.n_compositions
            )

            traintask_ids = list(base_ids) + list(additional_tasks)

            logger.info(f"Number of base  tasks: {len(base_ids)}")
            logger.info(f"Number of train tasks: {len(traintask_ids)}")

        elif self.function_properties.split.strategy == "random":
            traintask_ids = random.sample(
                alltask_ids, self.function_properties.split.n_compositions
            )

        elif self.function_properties.split.strategy == "random_biased":
            n_identity = self.function_properties.split.n_identity

            sub_taskids = [
                tid
                for tid in alltask_ids
                if np.sum(np.array(tid) == 0) == n_identity
            ]

            maxlen = len(sub_taskids)
            if self.function_properties.split.n_compositions > maxlen:
                raise ValueError

            traintask_ids = random.sample(
                sub_taskids, self.function_properties.split.n_compositions
            )

            logger.info(f"Number of possible functions: {len(sub_taskids)}")
            logger.info(f"Number of train tasks: {len(traintask_ids)}")

        elif self.function_properties.split.strategy == "randombase_combo":
            base_ids = [tuple((d0, 0) for d0 in range(depth))] + [
                tuple((k, 0) if k != d else (d, i) for k in range(depth))
                for d in range(depth)
                for i in range(1, n_functions + 1)
            ]

            nf_choices = [
                [(d, i) for i in range(n_functions)] for d in range(depth)
            ]
            all_tids = list(itertools.product(*nf_choices))
            inorder_tasks = random.sample(
                all_tids, self.function_properties.split.n_compositions_inorder
            )

            nf_choices = [
                [(d2, i) for d2 in range(depth) if d2 != d for i in range(n_functions)]
                for d in range(depth)
            ]

            all_tids = list(itertools.product(*nf_choices))
            additional_tasks = random.sample(
                all_tids, self.function_properties.split.n_compositions
            )
            traintask_ids = list(base_ids) + list(additional_tasks) + list(inorder_tasks)

        elif self.function_properties.split.strategy == "random_combo":
            base_ids = [tuple((d, 0) for d in range(depth))]

            nf_choices = [
                [(d2, i) for d2 in range(depth) if d2 != d for i in range(n_functions)]
                for d in range(depth)
            ]

            all_tids = list(itertools.product(*nf_choices))
            additional_tasks = random.sample(
                all_tids, self.function_properties.split.n_compositions
            )
            traintask_ids = list(base_ids) + list(additional_tasks)

        elif self.function_properties.split.strategy == "base_combo":
            base_ids = [tuple((d, 0) for d in range(depth))] + [
                tuple((k, 0) if k != d else (d, i) for k in range(depth))
                for d in range(depth)
                for i in range(1, n_functions + 1)
            ]

            nf_choices = [
                [(d2, i) for d2 in range(depth) if d2 != d for i in range(n_functions)]
                for d in range(depth)
            ]

            all_tids = list(itertools.product(*nf_choices))
            additional_tasks = random.sample(
                all_tids, self.function_properties.split.n_compositions
            )
            traintask_ids = list(base_ids) + list(additional_tasks)

            logger.info(f"Number of train tasks: {len(traintask_ids)}")

        train_fns = []

        if "combo" not in self.function_properties.split.strategy:
            train_fns = [fn for fn in composed_functions if fn[0] in traintask_ids]
        else:
            train_fns = [
                (
                    tid,
                    None,
                    [
                        functools.partial(
                            BaseFunction.map, mapping=self.finfo["functions"][x1][x2]
                        )
                        for x1, x2 in tid
                    ],
                )
                for tid in traintask_ids
            ]

        return train_fns, traintask_ids

    def compose(self) -> tuple[dict[str, list[tuple]], dict]:
        """Compose functions and split into train and all sets."""
        composed_functions = {"train": [], "all": []}

        allcomp_functions, info = self.compose_bijections()
        self.finfo = info
        train_functions, train_ids = self.get_train_functions(allcomp_functions)

        composed_functions["all"] = allcomp_functions
        composed_functions["train"] = train_functions

        if self.function_properties.split.strategy != "base_combo":
            info["train_id"] = np.array(train_ids)
        else:
            info["train_id"] = train_ids

        return composed_functions, info
