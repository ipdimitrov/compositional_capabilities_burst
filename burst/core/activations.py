"""Residual-stream activation collection for nanoGPT models.

Dimension key:
    B: batch_size
    L: sequence_length (doc_len)
    N: n_embd (model dimension)
    K: n_layers + 1 (embedding + each transformer block)
    P: n_probe_samples
    T: n_token_positions (= L - 1)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from burst.core.train_utils import DEVICE

if TYPE_CHECKING:
    from net.nanogpt import nanoGPT

COLLECT_BATCH_SIZE = 512


@torch.no_grad()
def collect_activations_KPTN(  # noqa: N802
    net: nanoGPT,
    docs_BL: np.ndarray,
) -> list[torch.Tensor]:
    """Collect residual-stream activations at every (layer, token_pos).

    Returns list of K tensors, each of shape (P, T, N) on CPU.
    K = n_layers + 1 (post-embedding + post-block_0 + ... + post-block_{L-1}).
    T = doc_len - 1 (model input is tokens[:-1]).
    P = len(docs_BL) — caller is responsible for subsampling.
    """
    net.eval()
    P = len(docs_BL)
    K = len(net.transformer.h) + 1

    all_layer_acts: list[list[torch.Tensor]] = [[] for _ in range(K)]

    for start in range(0, P, COLLECT_BATCH_SIZE):
        end = min(start + COLLECT_BATCH_SIZE, P)
        tokens_bL = torch.as_tensor(docs_BL[start:end], dtype=torch.long, device=DEVICE)
        inp_bT = tokens_bL[:, :-1]

        tok_emb = net.transformer.wte(inp_bT)
        pos = torch.arange(inp_bT.size(1), device=DEVICE)
        pos_emb = net.transformer.wpe(pos)
        x_bTN = net.transformer.drop(tok_emb + pos_emb)

        all_layer_acts[0].append(x_bTN.float().cpu())
        for bi, block in enumerate(net.transformer.h):
            x_bTN = block(x_bTN)
            all_layer_acts[bi + 1].append(x_bTN.float().cpu())

    return [torch.cat(chunks, dim=0) for chunks in all_layer_acts]
