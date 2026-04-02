"""Synthetic corpus generation, tokenization, vocab, and train/eval DataLoaders."""

import functools
import itertools
import json
import pickle
import random
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import tqdm
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from synthetic.functions import BaseFunction

_rng = np.random.default_rng()


class SyntheticData:
    """Generates a synthetic sequence of the form t, x, t(x)."""

    def __init__(
        self, cfg: DictConfig, composed_functions: dict[str, list[tuple]],
        functions_info: dict,
    ) -> None:
        """Initialize with config, composed functions, and function metadata."""
        self.cfg = cfg
        self.special_tokens = [" ", "<PAD>", "S"]
        self.n_special = len(self.special_tokens)
        self.n_alphabets = cfg.n_alphabets

        self.functions = composed_functions
        self.functions_info = functions_info
        self.task_map()

        self.fdir = Path("data") / cfg.tag

    def task_map(self) -> None:
        """Build mappings between task indices and (depth, tid) pairs."""
        self.task_idx = {}
        self.task = {}

        self.depth = len(self.functions_info["functions"])
        self.nfuncs = [len(fn) for fn in self.functions_info["functions"]]
        self.n_tasks = sum(self.nfuncs)

        self.nsplit_tasks = {"train": len(self.functions_info["train_id"]), "all": sum(self.nfuncs)}

        for dep, nf in enumerate(self.nfuncs):
            for tid in range(nf):
                idx = sum(self.nfuncs[:dep]) + tid + self.n_alphabets
                self.task[idx] = (dep, tid)
                self.task_idx[(dep, tid)] = idx

    def init_tokens(self) -> None:
        """Initialize the set of tokens and store into dictionaries."""
        self.token = {}
        self.token_idx = {}

        for i in range(self.n_alphabets):
            self.token[i] = "X" + str(i)
            self.token_idx["X" + str(i)] = i

        for i in range(self.n_tasks):
            idx = i + self.n_alphabets
            task_str = "T" + str(self.task[idx][0]) + "_" + str(self.task[idx][1])
            self.token[idx] = task_str
            self.token_idx[task_str] = idx

        for i in range(len(self.special_tokens)):
            idx = i + self.n_alphabets + self.n_tasks
            self.token[idx] = self.special_tokens[i]
            self.token_idx[self.special_tokens[i]] = idx

    def sample_task(self, split: str = "train") -> tuple:
        """Sample a random task from the given split."""
        idx = int(_rng.integers(0, self.nsplit_tasks[split]))
        return self.functions[split][idx]

    def sample_token(self) -> np.ndarray:
        """Sample tokens from the alphabet."""
        alph = np.arange(self.n_alphabets)
        return _rng.choice(alph, size=self.cfg.seq_len, replace=self.cfg.with_replacement)

    def decode(self, token_idx: np.ndarray) -> str:
        """Decode token indices to a subscript-formatted string."""
        txt_list = [self.token[t] for t in token_idx]
        txt = "".join(txt_list)
        SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        return txt.translate(SUB)

    def encode(self, token: list[str]) -> list[int]:
        """Encode token strings to indices."""
        return [self.token_idx[t] for t in token]

    def stepbystep_outputs(
        self, inp: np.ndarray, task_fns: list[functools.partial],
    ) -> list[np.ndarray]:
        """Compute intermediate outputs for each function in the chain."""
        outputs = []
        cur_inp = inp

        for fn in task_fns:
            cur_inp = fn(cur_inp)
            outputs.append(cur_inp)

        return outputs

    def generate_task_token_document(self, split: str) -> np.ndarray:
        """Generate a document of the form t, x, t(x)."""
        token_idx = self.sample_token()
        space_idx = np.array([self.token_idx[" "]])
        start_idx = np.array([self.token_idx["S"]])

        tasks = self.sample_task(split)
        task_idx = []
        for idx, ts in enumerate(tasks[0]):
            task_str = "T" + str(idx) + "_" + str(ts)
            task_idx.append(self.token_idx[task_str])
        task_idx = np.array(task_idx)

        output = token_idx
        for ofn in tasks[2]:
            output = ofn(output)

        return np.concatenate([start_idx, task_idx, space_idx, token_idx, space_idx, output])


    def generate_step_document(self, split: str) -> np.ndarray:
        """Generate a step-by-step document of the form t, x, f1(x), f2(f1(x)), ..."""
        token_idx = self.sample_token()
        space_idx = np.array([self.token_idx[" "]])
        start_idx = np.array([self.token_idx["S"]])

        tasks = self.sample_task(split)
        task_idx = []
        for idx, ts in enumerate(tasks[0]):
            if isinstance(ts, int):
                task_str = "T" + str(idx) + "_" + str(ts)
            else:
                task_str = "T" + str(ts[0]) + "_" + str(ts[1])
            task_idx.append(self.token_idx[task_str])
        task_idx = np.array(task_idx)

        outputs = self.stepbystep_outputs(token_idx, tasks[2])
        document = [start_idx, task_idx, space_idx, token_idx]

        for out in outputs:
            document.append(space_idx)
            document.append(out)

        return np.concatenate(document)


    def generate_document(self, split: str = "train") -> np.ndarray:
        """Generate a document using step-by-step or direct mode."""
        if not self.cfg.direct:
            return self.generate_step_document(split)
        return self.generate_task_token_document(split)

    def generate_corpus(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Generate training and evaluation corpora."""
        corpus = [self.generate_document() for _ in tqdm.trange(self.cfg.ndocuments)]
        self.corpus = np.array(corpus)

        self.eval_corpus = {}
        for split in ["train", "all"]:
            corpus = [
                self.generate_document(split) for _ in tqdm.trange(self.cfg.neval_documents)
            ]
            self.eval_corpus[split] = np.array(corpus)

        return self.corpus, self.eval_corpus

    def store_data(self) -> None:
        """Store tokens, corpus, and config to disk."""
        self.fdir.mkdir(parents=True, exist_ok=True)

        with (self.fdir / "token_idx.pkl").open("wb") as f:
            pickle.dump(self.token_idx, f)
        with (self.fdir / "token.pkl").open("wb") as f:
            pickle.dump(self.token, f)

        np.save(self.fdir / "corpus.npy", self.corpus)
        np.save(self.fdir / "train_eval_corpus.npy", self.eval_corpus["train"])
        np.save(self.fdir / "all_eval_corpus.npy", self.eval_corpus["all"])

        with (self.fdir / "functions_info.pkl").open("wb") as f:
            pickle.dump(self.functions_info, f)

        self.cfg = OmegaConf.to_container(self.cfg)
        with (self.fdir / "config.json").open("w") as f:
            json.dump(dict(self.cfg), f, indent=4)


class SyntheticDataset:
    """Dataset object to create a dataloader."""

    def __init__(self, fpath: str | Path, split: str = "train") -> None:
        """Initialize dataset from file path and split name."""
        fpath = Path(fpath)
        datafiles = {
            "train": fpath / "corpus.npy",
            "train_eval": fpath / "train_eval_corpus.npy",
            "all_eval": fpath / "all_eval_corpus.npy",
        }

        self.data = np.load(datafiles[split])

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (input, target) pair for the given index."""
        elem = torch.from_numpy(self.data[idx])
        dat, target = elem[:-1], elem[1:]
        return dat, target


class SyntheticEval(SyntheticData):
    """Create dataloader for each function composition."""

    def __init__(
        self, net_cfg: DictConfig, nsamples: int, nbatch: int,
        *, direct_eval: bool | None = None, permute: bool | None = None,
    ) -> None:
        """Initialize evaluation from network config."""
        self.step_eval = not direct_eval

        self.permute_eval = permute

        data_path = Path(net_cfg.data.path)
        info_fname = data_path / "functions_info.pkl"
        data_fname = data_path / "config.json"

        self.token_idx = np.load(
            data_path / "token_idx.pkl", allow_pickle=True
        )

        with data_fname.open() as f:
            self.cfg = OmegaConf.create(json.load(f))
        self.special_tokens = [" ", "<PAD>", "S"]
        self.n_special = len(self.special_tokens)
        self.n_alphabets = self.cfg.n_alphabets

        self.functions_info = np.load(info_fname, allow_pickle=True)
        self.composed_functions = self.functions_info["composition_reduced"]

        self.net_cfg = net_cfg

        self.nsamples = nsamples
        self.nbatch = nbatch

        self.task_map()

    def get_seq_info(self, sample: np.ndarray) -> dict[str, int]:
        """Extract sequence structure info from a sample."""
        sp_idx = self.token_idx[" "]
        total_len = len(sample)
        sp_pos = np.where(sample == sp_idx)[0]
        return {
            "last_space": sp_pos[-1],
            "prompt": sp_pos[1] + 1,
            "new": total_len - (sp_pos[1] + 1),
        }

    def generate_step_document(self, task_info: tuple) -> np.ndarray:
        """Generate a document for evaluation."""
        token_idx = self.sample_token()
        space_idx = np.array([self.token_idx[" "]])
        start_idx = np.array([self.token_idx["S"]])

        task_idx = []
        for idx, ts in enumerate(task_info[0]):
            task_str = "T" + str(idx) + "_" + str(ts)
            task_idx.append(self.token_idx[task_str])
        task_idx = np.array(task_idx)

        if self.step_eval:
            outputs = self.stepbystep_outputs(token_idx, task_info[2])
            document = [start_idx, task_idx, space_idx, token_idx]
            for out in outputs:
                document.append(space_idx)
                document.append(out)
        else:
            outputs = token_idx
            for fn in task_info[2]:
                outputs = fn(outputs)

            document = [start_idx, task_idx, space_idx, token_idx, space_idx, outputs]

        return np.concatenate(document)


    @torch.no_grad()
    def evaluate_docs(
        self, net: torch.nn.Module, dat: torch.Tensor,
        seq_info: dict[str, int], device: str, *, lstm: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate documents and return total, sharp, and cached accuracies."""
        shape = dat.shape
        if device == "cuda":
            dat = dat.cuda(non_blocking=True)

        dat = dat.view(-1, shape[-1])
        inp_c = dat[:, :-1]
        inp = dat[:, : seq_info["prompt"]]

        if lstm:
            net.hidden = None

        for _ in range(seq_info["new"]):
            logits = net(inp)
            logits = logits[:, -1, :]
            inp_next = torch.argmax(logits, -1, keepdims=True)
            inp = torch.cat((inp, inp_next), dim=1)

        output = inp
        output_c = torch.argmax(net(inp_c), -1)

        output_l = output[:, seq_info["last_space"] + 1 :]
        output_cl = output_c[:, seq_info["last_space"] :]

        targets_l = dat[:, seq_info["last_space"] + 1 :]

        # Accuracy averaged over all positions
        acc_l = output_l.reshape(-1) == targets_l.reshape(-1)
        acc_l = acc_l.view(shape[0], shape[1], output_l.shape[-1])

        # Strict accuracy (1 if all tokens correct, 0 otherwise)
        acc_cl = output_cl.reshape(-1) == targets_l.reshape(-1)
        acc_cl = acc_cl.view(shape[0], shape[1], output_cl.shape[-1])

        # Accuracy including the step-by-step tokens)
        total_acc = acc_l.float().mean((-1, -2)).to("cpu").numpy()
        sharp_acc = acc_l.all(-1).float().mean(-1).to("cpu").numpy()
        total_acc_c = acc_cl.float().mean((-1, -2)).to("cpu").numpy()

        return total_acc, sharp_acc, total_acc_c

    def get_acc(
        self, net: torch.nn.Module, *, lstm: bool = False,
    ) -> dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Get the accuracy of each function composition."""
        info = self.functions_info

        device = "cuda" if torch.cuda.is_available() else "cpu"

        net.eval()
        if device == "cuda":
            net = net.cuda()

        acc_map = {}

        tid_list = []
        doc_list = []

        if lstm:
            net.use_hidden = True

        for idx, tid in enumerate(info["task_id"]):
            reduced_func = self.composed_functions[idx]
            task_funcs = []
            for d, t in enumerate(tid):
                fnap = BaseFunction.permute if self.permute_eval and d == 0 else BaseFunction.map

                fn = functools.partial(fnap, mapping=info["functions"][d][t])
                task_funcs.append(fn)

            task_info = (tid, reduced_func, task_funcs)

            docs = [self.generate_step_document(task_info) for _ in range(self.nsamples)]

            if idx == 0:
                sample = torch.Tensor(docs[0]).long()
                seq_info = self.get_seq_info(sample)

            tid_list.append(tid)
            doc_list.append(docs)

            if idx % self.nbatch == self.nbatch - 1 or idx == len(info["task_id"]) - 1:
                flatten_docs = torch.Tensor(np.array(doc_list, dtype=int)).long()

                acc_list = self.evaluate_docs(net, flatten_docs, seq_info, device, lstm=lstm)

                for j in range(len(tid_list)):
                    t = tid_list[j]
                    acc_map[tuple(t)] = (acc_list[0][j], acc_list[1][j], acc_list[2][j])

                tid_list = []
                doc_list = []

        return acc_map

    def save_accs(self, cfg: DictConfig, accs: dict) -> None:
        """Save accuracy results to disk."""
        self.fdir = Path("data") / cfg.tag
        self.fdir.mkdir(parents=True, exist_ok=True)
        with (self.fdir / "accs.pkl").open("wb") as f:
            pickle.dump(accs, f)


class SyntheticEvalCombinatorial(SyntheticEval):
    """Evaluate on in-order and out-of-order functions."""

    @torch.no_grad()
    def evaluate_docs(  # type: ignore[override]
        self, net: torch.nn.Module, dat: torch.Tensor,
        seq_info: dict[str, int], device: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate documents and return total and cached accuracies."""
        shape = dat.shape

        if device == "cuda":
            dat = dat.cuda(non_blocking=True)

        dat = dat.view(-1, shape[-1])

        inp_c = dat[:, :-1]
        inp = dat[:, : seq_info["prompt"]]

        for _ in range(seq_info["new"]):
            logits = net(inp)
            logits = logits[:, -1, :]
            inp_next = torch.argmax(logits, -1, keepdims=True)
            inp = torch.cat((inp, inp_next), dim=1)

        output = inp
        output_c = torch.argmax(net(inp_c), -1)

        output_l = output[:, seq_info["last_space"] + 1 :]
        output_cl = output_c[:, seq_info["last_space"] :]

        targets_l = dat[:, seq_info["last_space"] + 1 :]

        acc_l = output_l.reshape(-1) == targets_l.reshape(-1)
        acc_l = acc_l.view(shape[0], shape[1], output_l.shape[-1])

        acc_cl = output_cl.reshape(-1) == targets_l.reshape(-1)
        acc_cl = acc_cl.view(shape[0], shape[1], output_cl.shape[-1])

        total_acc = acc_l.float().mean(-2).to("cpu").numpy()
        total_acc_c = acc_cl.float().mean(-2).to("cpu").numpy()

        return total_acc, total_acc_c

    def generate_step_document(self, task_info: tuple) -> np.ndarray:
        """Generate a step-by-step document for combinatorial evaluation."""
        token_idx = self.sample_token()
        space_idx = np.array([self.token_idx[" "]])
        start_idx = np.array([self.token_idx["S"]])

        task_idx = []
        for ts in task_info[0]:
            task_str = "T" + str(ts[0]) + "_" + str(ts[1])
            task_idx.append(self.token_idx[task_str])
        task_idx = np.array(task_idx)

        outputs = self.stepbystep_outputs(token_idx, task_info[1])
        document = [start_idx, task_idx, space_idx, token_idx]

        for out in outputs:
            document.append(space_idx)
            document.append(out)

        return np.concatenate(document)


    def get_task_list(self, depth: int, choices: int) -> dict[tuple[int, int], list[tuple]]:  # noqa: C901
        """Build task list organized by (num_identity, num_swap)."""
        task_list = {}

        for num_identity in range(depth, -1, -1):
            num_funcs = depth - num_identity

            for id_combo in combinations(range(depth), num_identity):
                for num_swap in range(num_funcs + 1):
                    for sw_combo in combinations(range(num_funcs), num_swap):
                        fix_pos = set(range(depth)) - set(id_combo)
                        fix_pos = tuple(fix_pos - set(sw_combo))

                        id_pos = tuple(id_combo)
                        sw_pos = tuple(sw_combo)

                        nfunc_choices = [None for d in range(depth)]

                        for pos in id_pos:
                            nfunc_choices[pos] = [(pos, 0)]

                        for pos in fix_pos:
                            nfunc_choices[pos] = [(pos, i) for i in range(1, choices)]

                        for pos in sw_pos:
                            nfunc_choices[pos] = []
                            for d in range(depth):
                                if pos != d:
                                    nfunc_choices[pos] += [(d, i) for i in range(1, choices)]

                        cur_tlist = list(itertools.product(*nfunc_choices))

                        sample_num = min(len(cur_tlist), 500)

                        tlist = random.sample(cur_tlist, sample_num)
                        if (num_identity, num_swap) in task_list:
                            task_list[(num_identity, num_swap)] += tlist
                        else:
                            task_list[(num_identity, num_swap)] = tlist

        return task_list

    def get_acc(  # type: ignore[override]
        self, net: torch.nn.Module,
    ) -> dict[tuple[int, int], dict[tuple, tuple[np.ndarray, np.ndarray]]]:
        """Compute accuracies grouped by identity count and displacement."""
        depth = self.cfg.function.depth
        info = self.functions_info

        device = "cuda" if torch.cuda.is_available() else "cpu"

        net.eval()
        if device == "cuda":
            net = net.cuda()

        acc_map = {}
        tid_list = []
        doc_list = []

        depth, choices = info["functions"].shape[0:2]

        task_list = self.get_task_list(depth, choices)

        for key in tqdm.tqdm(task_list):
            acc_map[key] = {}
            for idx, tsk in enumerate(task_list[key]):
                task_funcs = []
                for d, t in tsk:
                    fn = functools.partial(BaseFunction.map, mapping=info["functions"][d][t])
                    task_funcs.append(fn)

                task_info = (tsk, task_funcs)

                docs = [self.generate_step_document(task_info) for _ in range(self.nsamples)]

                if idx == 0:
                    sample = torch.Tensor(docs[0]).long()
                    seq_info = self.get_seq_info(sample)

                tid_list.append(tsk)
                doc_list.append(docs)

                if idx % self.nbatch == self.nbatch - 1 or idx == len(task_list[key]) - 1:
                    flatten_docs = torch.Tensor(np.array(doc_list, dtype=int)).long()

                    acc_list = self.evaluate_docs(net, flatten_docs, seq_info, device)

                    for j in range(len(tid_list)):
                        t = tid_list[j]

                        acc_map[key][tuple(t)] = (acc_list[0][j], acc_list[1][j])

                    tid_list = []
                    doc_list = []

        return acc_map


def get_vocab_len(fpath: str | Path) -> int:
    """Return vocabulary size from saved token file."""
    token = np.load(Path(fpath) / "token.pkl", allow_pickle=True)
    return len(token)


def get_space_pos(fpath: str | Path, loader: DataLoader) -> int:
    """Get position of the last space token in the first sample."""
    token_idx = np.load(Path(fpath) / "token_idx.pkl", allow_pickle=True)
    sp_idx = token_idx[" "]
    return np.where(loader.dataset.data[0] == sp_idx)[0][-1]


def get_seq_info(fpath: str | Path, loader: DataLoader) -> dict[str, int]:
    """Get sequence markers like prompt length and last space position."""
    fpath = Path(fpath)
    token_idx = np.load(fpath / "token_idx.pkl", allow_pickle=True)
    seq_info = {}

    with (fpath / "config.json").open() as f:
        data_cfg = json.load(f)

    sp_idx = token_idx[" "]
    sample = loader.dataset.data[0]
    total_len = len(sample)

    if not data_cfg["direct"]:
        sp_pos = np.where(loader.dataset.data[0] == sp_idx)[0]

        seq_info["last_space"] = sp_pos[-1]
        seq_info["prompt"] = sp_pos[1] + 1
        seq_info["new"] = total_len - seq_info["prompt"]

    return seq_info


def get_trainLoader(cfg: DictConfig) -> DataLoader:  # noqa: N802
    """Create training dataloader."""
    dataset = SyntheticDataset(cfg.data.path, "train")
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=cfg.data.num_workers,
    )


def get_evalLoaders(cfg: DictConfig) -> list[DataLoader]:  # noqa: N802
    """Create dataloaders for evaluation."""
    loaders = []
    for split in ["train_eval", "all_eval"]:
        dataset = SyntheticDataset(cfg.data.path, split)
        loaders.append(
            DataLoader(
                dataset,
                batch_size=cfg.data.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=cfg.data.num_workers,
            )
        )
    return loaders
