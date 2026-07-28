"""Data generation for the burst experiment.

Generates bijection-composition tasks: F_1 o F_2 o ... o F_depth applied to
input sequences.  One position gets a novel "burst" function b* that the model
must learn during finetuning.

The returned data dict is self-contained and can be pickled/passed between
phases.
"""
import itertools
import numpy as np


# ── defaults ─────────────────────────────────────────────────────────────
N_ALPH = 10       # alphabet size
SEQ_LEN = 6       # input sequence length per document
N_A = 4           # background functions per position
DEPTH = 3         # composition depth
BURST_POS = 3     # which position (1-indexed) gets b*
DATA_SEED = 999
N_DOCS = 100      # docs per task for training pools
N_EVAL = 100      # docs per task for eval pools


def _build_vocab(n_alph, n_bijections):
    """Token <-> index mappings."""
    token, token_idx, fn_tok = {}, {}, {}
    idx = 0
    for i in range(n_alph):
        token[idx] = f"X{i}"
        token_idx[f"X{i}"] = idx
        idx += 1
    for i in range(n_bijections):
        token[idx] = f"F{i}"
        token_idx[f"F{i}"] = idx
        fn_tok[i] = idx
        idx += 1
    for sp in (" ", "<PAD>", "S"):
        token[idx] = sp
        token_idx[sp] = idx
        idx += 1
    return token, token_idx, fn_tok, idx  # last = vocab_size


def _make_doc(task, bijections, seq_len, token_idx, fn_tok):
    """Generate one document for a task = (class_label, f_D, ..., f_1)."""
    fns = task[1:]
    inp = np.random.choice(len([k for k in token_idx if k.startswith("X")]),
                           size=seq_len, replace=True)
    sp = np.array([token_idx[" "]])
    cur = inp.copy()
    outs = []
    for fn_idx in reversed(fns):
        cur = bijections[fn_idx][cur]
        outs.append(cur.copy())
    doc = [np.array([token_idx["S"]]),
           np.array([fn_tok[f] for f in fns]),
           sp, inp]
    for o in outs:
        doc.extend([sp, o])
    return np.concatenate(doc)


def _gen_pool(tasks, n, bijections, seq_len, token_idx, fn_tok):
    return {t: np.array([_make_doc(t, bijections, seq_len, token_idx, fn_tok)
                         for _ in range(n)])
            for t in tasks}


def _pad_pools(*pools):
    """Pad all pools to the same document length."""
    max_len = max(docs.shape[1]
                  for pool in pools for docs in pool.values())
    out = []
    for pool in pools:
        new = {}
        for key, docs in pool.items():
            if docs.shape[1] < max_len:
                pad = np.zeros((docs.shape[0], max_len - docs.shape[1]),
                               dtype=docs.dtype)
                docs = np.concatenate([docs, pad], axis=1)
            new[key] = docs
        out.append(new)
    return out


def _cat(pool, fallback_cols=1):
    if not pool:
        return np.zeros((1, fallback_cols), dtype=np.int64)
    return np.concatenate(list(pool.values()))


def make_data(
    *,
    n_alph: int = N_ALPH,
    seq_len: int = SEQ_LEN,
    n_a: int = N_A,
    depth: int = DEPTH,
    burst_pos: int = BURST_POS,
    n_burst_fns: int = 1,
    seed: int = DATA_SEED,
    n_docs: int = N_DOCS,
    n_eval: int = N_EVAL,
) -> dict:
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

    # -- bijections --
    bijections = [np.arange(n_alph)]  # identity
    for _ in range(n_a * depth + n_burst_fns):
        bijections.append(rng.permutation(n_alph))
    # indices of the novel burst functions (one per n_burst_fns)
    b_stars = list(range(n_a * depth + 1, n_a * depth + 1 + n_burst_fns))

    pos_fns = {p: list(range((p - 1) * n_a + 1, p * n_a + 1))
               for p in range(1, depth + 1)}

    _, token_idx, fn_tok, vocab_size = _build_vocab(n_alph, len(bijections))

    # -- task definitions --
    other_combos = list(itertools.product(
        *[pos_fns[p] for p in range(1, depth + 1)]))
    rng.shuffle(other_combos)
    other_tasks = [("other",) + combo for combo in other_combos]

    # burst_pos is in application order (1=first applied=innermost, depth=last applied=outermost).
    # pos_fns is keyed by token-sequence position (1=first token=outermost).
    # Convert: token position of the burst function = depth - burst_pos + 1.
    burst_tok_pos = depth - burst_pos + 1
    non_bp = [p for p in range(1, depth + 1) if p != burst_tok_pos]
    remaining = list(itertools.product(*[pos_fns[p] for p in non_bp]))
    burst_tasks = []
    for b in b_stars:
        for combo in remaining:
            fns = list(combo)
            fns.insert(burst_tok_pos - 1, b)
            burst_tasks.append(("burst",) + tuple(fns))

    # -- pools --
    np.random.seed(seed)
    bg_pool = _gen_pool(other_tasks, n_docs, bijections, seq_len,
                        token_idx, fn_tok)
    target_pool = _gen_pool(burst_tasks, n_docs, bijections, seq_len,
                            token_idx, fn_tok)
    eval_other_pool = _gen_pool(other_tasks[:min(8, len(other_tasks))],
                                n_eval, bijections, seq_len, token_idx, fn_tok)
    eval_burst_pool = _gen_pool(burst_tasks, n_eval, bijections, seq_len,
                                token_idx, fn_tok)

    bg_pool, target_pool, eval_other_pool, eval_burst_pool = _pad_pools(
        bg_pool, target_pool, eval_other_pool, eval_burst_pool)

    eval_other = _cat(eval_other_pool)
    eval_burst = _cat(eval_burst_pool)

    # Cap eval_burst to the same size regardless of n_burst_fns, so eval cost
    # doesn't scale up. Budget = (tasks per function) × n_eval, matching n_burst_fns=1.
    tasks_per_fn = len(burst_tasks) // n_burst_fns
    eval_budget = tasks_per_fn * n_eval
    if eval_burst.shape[0] > eval_budget:
        idx = np.random.permutation(eval_burst.shape[0])[:eval_budget]
        eval_burst = eval_burst[idx]

    ref = eval_other if eval_other.shape[0] > 1 else eval_burst
    sp_positions = np.where(ref[0] == token_idx[" "])[0]
    prompt_len = int(sp_positions[0]) + 1 + seq_len + 1

    return {
        "bg_pool": bg_pool,
        "target_pool": target_pool,
        "eval_other": eval_other,
        "eval_burst": eval_burst,
        "prompt_len": prompt_len,
        "vocab_size": vocab_size + 10,
        "context_size": ref.shape[1] + 5,
        "b_stars": b_stars,
        "task_info": {
            "n_alph": n_alph, "seq_len": seq_len, "n_a": n_a,
            "depth": depth, "burst_pos": burst_pos,
            "n_burst_fns": n_burst_fns,
            "n_other_tasks": len(other_tasks),
            "n_burst_tasks": len(burst_tasks),
            "doc_len": int(ref.shape[1]),
            "prompt_len": prompt_len,
        },
        # kept for advanced use
        "bijections": bijections,
        "token_idx": token_idx,
        "fn_tok": fn_tok,
    }
