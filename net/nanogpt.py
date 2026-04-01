"""Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

"""

import math

import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias."""

    def __init__(self, ndim: int, *, bias: bool) -> None:
        """Initialize LayerNorm with optional bias parameter."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization."""
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional KV cache."""

    def __init__(self, config: DictConfig) -> None:
        """Initialize attention projections and dropout."""
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        return_kv: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Compute self-attention output, optionally returning KV cache."""
        B, T, C = x.size()
        head_dim = C // self.n_head

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)

        is_causal = kv_cache is None
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=is_causal,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        if kv_cache is not None or return_kv:
            return y, (k, v)
        return y


class MLP(nn.Module):
    """Feed-forward sub-network with GELU activation."""

    def __init__(self, config: DictConfig) -> None:
        """Initialize linear layers and dropout."""
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward transformation."""
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    """Transformer block with self-attention and optional MLP."""

    def __init__(self, config: DictConfig) -> None:
        """Initialize attention, layer norms, and optional MLP."""
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.config = config
        if config.mlp:
            self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        return_kv: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Apply self-attention and MLP to the residual stream."""
        collect_kv = kv_cache is not None or return_kv
        attn_out = self.attn(self.ln_1(x), kv_cache=kv_cache, return_kv=return_kv)
        if collect_kv:
            attn_out, new_kv = attn_out
            x = x + attn_out
            if self.config.mlp:
                x = x + self.mlp(self.ln_2(x))
            return x, new_kv
        x = x + attn_out
        if self.config.mlp:
            x = x + self.mlp(self.ln_2(x))
        return x


class nanoGPT(nn.Module):  # noqa: N801
    """GPT-2 style autoregressive language model."""

    def __init__(self, config: DictConfig) -> None:
        """Initialize embeddings, transformer blocks, and LM head."""
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.context_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": LayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.LM_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight typing
        self.transformer.wte.weight = self.LM_head.weight

        self.apply(self._init_weights)
        # apply special scaled init to the residual projections
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def get_num_params(self, *, non_embedding: bool = True) -> int:
        """Return the number of parameters, excluding position embeddings by default."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Compute logits for the input token indices."""
        device = idx.device
        _b, t = idx.size()

        tok_emb = self.transformer.wte(idx)
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)

        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.LM_head(x)

    @torch.no_grad()
    def generate(self, prompt_BT: torch.Tensor, n_new: int) -> torch.Tensor:
        """Autoregressive generation with KV-cache.

        Runs one full forward pass over the prompt to fill the cache, then
        generates n_new tokens one at a time — each step is a single-token
        forward pass rather than a full-sequence recomputation.

        Returns the full sequence (prompt + generated tokens).
        """
        device = prompt_BT.device
        B, T_prompt = prompt_BT.shape

        generated = torch.empty(B, T_prompt + n_new, dtype=torch.long, device=device)
        generated[:, :T_prompt] = prompt_BT

        tok_emb = self.transformer.wte(prompt_BT)
        pos = torch.arange(0, T_prompt, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for block in self.transformer.h:
            x, kv = block(x, return_kv=True)
            kv_caches.append(kv)

        x = self.transformer.ln_f(x)
        next_tok = self.LM_head(x)[:, -1, :].argmax(dim=-1)
        generated[:, T_prompt] = next_tok
        next_tok = next_tok.unsqueeze(-1)

        for i in range(1, n_new):
            t_cur = T_prompt + i
            tok_emb = self.transformer.wte(next_tok)
            pos = torch.tensor([t_cur - 1], dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)

            new_kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
            for block, kv in zip(self.transformer.h, kv_caches, strict=False):
                x, new_kv = block(x, kv_cache=kv)
                new_kv_caches.append(new_kv)
            kv_caches = new_kv_caches

            x = self.transformer.ln_f(x)
            next_tok = self.LM_head(x)[:, -1, :].argmax(dim=-1)
            generated[:, t_cur] = next_tok
            next_tok = next_tok.unsqueeze(-1)

        return generated
