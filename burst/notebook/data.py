"""Data generation for the notebook burst experiment.

Wraps burst.core.train.experiment.DepthNData and burst.core.data utilities
into a single make_data() call that returns a self-contained dict.
"""

import itertools

import numpy as np

from burst.config import (
    BURST_POS,
    CLASS_BURST,
    CLASS_OTHER,
    CONTEXT_SLACK,
    DATA_SEED,
    DEPTH,
    N_A,
    N_ALPH,
    N_BURST,
    N_DOCS,
    N_EVAL,
    SEQ_LEN,
    VOCAB_SLACK,
    burst_eval_range,
)
from burst.core.data import pad_pools_to_same_length
from burst.rng import get_rng, seed_all
from burst.types import ExperimentData

def _cat(pool: dict[tuple, np.ndarray], fallback_cols: int = 1) -> np.ndarray:
    if not pool:
        return np.zeros((1, fallback_cols), dtype=np.int64)
    return np.concatenate(list(pool.values()))


def make_data(  # noqa: PLR0913
    *,
    n_alph: int = N_ALPH,
    seq_len: int = SEQ_LEN,
    n_a: int = N_A,
    n_burst: int = N_BURST,
    depth: int = DEPTH,
    burst_pos: int = BURST_POS,
    seed: int = DATA_SEED,
    n_docs: int = N_DOCS,
    n_eval: int = N_EVAL,
) -> ExperimentData:
    """Build all data needed for the 3-phase experiment.

    Returns a dict with keys:
        bg_pool, target_pool     - training pools (dict of task -> (N, L) arrays)
        eval_other, eval_burst   - (N, L) arrays for evaluation
        prompt_len               - int: how many tokens to feed as prompt
        vocab_size, context_size - ints for model config
        task_info                - metadata dict
        bijections, token_idx, fn_tok  - kept for reproducibility
    """
    rng = np.random.RandomState(seed)

    bijections = [np.arange(n_alph)]
    bijections.extend(rng.permutation(n_alph) for _ in range(n_a * depth + n_burst))
    burst_fns = list(range(n_a * depth + 1, n_a * depth + n_burst + 1))

    pos_fns = {p: list(range((p - 1) * n_a + 1, p * n_a + 1)) for p in range(1, depth + 1)}

    token, token_idx, fn_tok, idx = {}, {}, {}, 0
    for i in range(n_alph):
        token[idx] = f"X{i}"
        token_idx[f"X{i}"] = idx
        idx += 1
    for i in range(len(bijections)):
        token[idx] = f"F{i}"
        token_idx[f"F{i}"] = idx
        fn_tok[i] = idx
        idx += 1
    for sp in (" ", "<PAD>", "S"):
        token[idx] = sp
        token_idx[sp] = idx
        idx += 1
    vocab_size = idx

    other_combos = list(itertools.product(*[pos_fns[p] for p in range(1, depth + 1)]))
    rng.shuffle(other_combos)
    other_tasks = [(CLASS_OTHER, *combo) for combo in other_combos]

    non_bp = [p for p in range(1, depth + 1) if p != burst_pos]
    remaining = list(itertools.product(*[pos_fns[p] for p in non_bp]))
    burst_tasks = []
    for combo in remaining:
        for bf in burst_fns:
            fns = list(combo)
            fns.insert(burst_pos - 1, bf)
            burst_tasks.append((CLASS_BURST, *tuple(fns)))

    seed_all(seed)

    def _make_doc(task: tuple) -> np.ndarray:
        fns = task[1:]
        inp = get_rng().integers(0, n_alph, size=seq_len)
        sp = np.array([token_idx[" "]])
        cur = inp.copy()
        outs = []
        for fn_idx in reversed(fns):
            cur = bijections[fn_idx][cur]
            outs.append(cur.copy())
        doc = [np.array([token_idx["S"]]), np.array([fn_tok[f] for f in fns]), sp, inp]
        for o in outs:
            doc.extend([sp, o])
        return np.concatenate(doc)

    def _gen_pool(tasks: list[tuple], n: int) -> dict[tuple, np.ndarray]:
        return {t: np.array([_make_doc(t) for _ in range(n)]) for t in tasks}

    bg_pool = _gen_pool(other_tasks, n_docs)
    target_pool = _gen_pool(burst_tasks, n_docs)
    eval_other_pool = _gen_pool(other_tasks[: min(8, len(other_tasks))], n_eval)
    eval_burst_pool = _gen_pool(burst_tasks, n_eval)

    bg_pool, target_pool, eval_other_pool, eval_burst_pool = pad_pools_to_same_length(
        bg_pool, target_pool, eval_other_pool, eval_burst_pool
    )

    eval_other = _cat(eval_other_pool)
    eval_burst = _cat(eval_burst_pool)

    ref = eval_other if eval_other.shape[0] > 1 else eval_burst
    sp_positions = np.where(ref[0] == token_idx[" "])[0]
    prompt_len = int(sp_positions[0]) + 1 + seq_len + 1
    eval_start, eval_end = burst_eval_range(prompt_len, burst_pos, seq_len)

    return {
        "bg_pool": bg_pool,
        "target_pool": target_pool,
        "eval_other": eval_other,
        "eval_burst": eval_burst,
        "prompt_len": prompt_len,
        "eval_start": eval_start,
        "eval_end": eval_end,
        "vocab_size": vocab_size + VOCAB_SLACK,
        "context_size": ref.shape[1] + CONTEXT_SLACK,
        "task_info": {
            "n_alph": n_alph,
            "seq_len": seq_len,
            "n_a": n_a,
            "depth": depth,
            "burst_pos": burst_pos,
            "n_other_tasks": len(other_tasks),
            "n_burst_tasks": len(burst_tasks),
            "doc_len": int(ref.shape[1]),
            "prompt_len": prompt_len,
        },
        "bijections": bijections,
        "token_idx": token_idx,
        "fn_tok": fn_tok,
    }
