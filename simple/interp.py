"""Interpretability metrics for burst experiment analysis.

Provides weight-space, representation (CKA), and gradient metrics
to understand why higher burst concentration leads to faster forgetting.
"""
import numpy as np
import torch
import torch.nn.functional as F

from simple.model import load_model, DEVICE


# ── helpers ───────────────────────────────────────────────────────────────

def _raw(net):
    """Unwrap torch.compile wrapper if present."""
    return getattr(net, "_orig_mod", net)


def state_dict_cpu(net):
    """Get state dict with all tensors on CPU."""
    return {k: v.detach().cpu() for k, v in _raw(net).state_dict().items()}


def load_sd(path):
    """Load a state dict from disk (CPU)."""
    return torch.load(path, map_location="cpu", weights_only=True)


def _layer_group(name):
    """Map parameter name to a readable layer group."""
    if "transformer.h." in name:
        parts = name.split(".")
        idx = parts[2]
        rest = ".".join(parts[3:])
        return f"layer_{idx}.{rest}"
    if "transformer.wte" in name:
        return "embed_tok"
    if "transformer.wpe" in name:
        return "embed_pos"
    if "transformer.ln_f" in name:
        return "ln_final"
    if "LM_head" in name:
        return "lm_head"
    return name


def _block_group(name):
    """Coarser grouping: just the block index or embed/head."""
    if "transformer.h." in name:
        return f"block_{name.split('.')[2]}"
    if "transformer.wte" in name:
        return "embed_tok"
    if "transformer.wpe" in name:
        return "embed_pos"
    if "transformer.ln_f" in name:
        return "ln_final"
    if "LM_head" in name:
        return "lm_head"
    return name


# ── weight-space metrics ──────────────────────────────────────────────────

def weight_drift_l2(sd_ref, sd_now):
    """Total and per-layer L2 distance between two state dicts."""
    per_layer = {}
    total_sq = 0.0
    for k in sd_ref:
        if k not in sd_now:
            continue
        delta = sd_now[k].float() - sd_ref[k].float()
        l2 = delta.norm().item()
        per_layer[_layer_group(k)] = l2
        total_sq += l2 ** 2
    return {"total": total_sq ** 0.5, "per_layer": per_layer}


def weight_cosine_per_layer(sd_ref, sd_now):
    """Cosine similarity between weight vectors per layer."""
    result = {}
    for k in sd_ref:
        if k not in sd_now:
            continue
        a = sd_ref[k].float().flatten()
        b = sd_now[k].float().flatten()
        cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        result[_layer_group(k)] = cos
    return result


def weight_delta_svd(sd_ref, sd_now, top_k=10):
    """SVD analysis of weight deltas for 2D weight matrices."""
    results = {}
    for k in sd_ref:
        if k not in sd_now or sd_ref[k].dim() != 2:
            continue
        delta = sd_now[k].float() - sd_ref[k].float()
        S = torch.linalg.svdvals(delta)
        s = S.numpy()
        s_norm = s / (s.sum() + 1e-10)
        eff_rank = float(np.exp(-np.sum(s_norm * np.log(s_norm + 1e-10))))
        var_total = float((s ** 2).sum())
        var_topk = float((s[:top_k] ** 2).sum())
        results[_layer_group(k)] = {
            "singular_values": s,
            "effective_rank": eff_rank,
            "top_k_var_frac": var_topk / (var_total + 1e-10),
            "spectral_norm": float(s[0]),
            "nuclear_norm": float(s.sum()),
            "n_params": int(delta.numel()),
        }
    return results


# ── representation metrics (CKA) ─────────────────────────────────────────

def extract_hidden_states(net, data_np, prompt_len, batch_size=256):
    """Capture per-layer hidden states at the last prompt position."""
    raw = _raw(net)
    net.eval()

    layer_acts = {}
    hooks = []
    for i, block in enumerate(raw.transformer.h):
        def _make_hook(idx):
            def hook_fn(module, inp, out):
                if idx not in layer_acts:
                    layer_acts[idx] = []
                layer_acts[idx].append(out[:, prompt_len - 1, :].detach().cpu())
            return hook_fn
        hooks.append(block.register_forward_hook(_make_hook(i)))

    dat = torch.as_tensor(data_np, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        for i in range(0, dat.shape[0], batch_size):
            raw(dat[i:i + batch_size, :-1])

    for h in hooks:
        h.remove()

    result = {i: torch.cat(acts).numpy() for i, acts in layer_acts.items()}
    net.train()
    return result


def linear_cka(X, Y):
    """Linear CKA between two (N, D) activation matrices."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    hsic_xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2
    return float(hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10))


def cka_between_models(net1, net2, data_np, prompt_len):
    """Layer-by-layer CKA matrix between two models."""
    acts1 = extract_hidden_states(net1, data_np, prompt_len)
    acts2 = extract_hidden_states(net2, data_np, prompt_len)
    n = len(acts1)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = linear_cka(acts1[i], acts2[j])
    return mat


def cka_diagonal(net1, net2, data_np, prompt_len):
    """Same-layer CKA between two models (just the diagonal)."""
    acts1 = extract_hidden_states(net1, data_np, prompt_len)
    acts2 = extract_hidden_states(net2, data_np, prompt_len)
    return {i: linear_cka(acts1[i], acts2[i]) for i in acts1}


# ── gradient metrics ──────────────────────────────────────────────────────

def _get_grad_vector(net, batch_np):
    """Compute gradient vector for a batch (returns flat tensor)."""
    dat = torch.as_tensor(batch_np, dtype=torch.long, device=DEVICE)
    inp, tgt = dat[:, :-1], dat[:, 1:]
    net.zero_grad()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        logits = net(inp)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
    loss.backward()
    grads = []
    for p in net.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
    return torch.cat(grads)


def grad_effective_rank(net, batch_np):
    """Effective rank of the gradient across 2D parameter matrices.

    For each weight matrix (≥2D), compute SVD of its gradient and return
    the entropy-based effective rank.  Returns the mean effective rank
    across all weight matrices.
    """
    dat = torch.as_tensor(batch_np, dtype=torch.long, device=DEVICE)
    inp, tgt = dat[:, :-1], dat[:, 1:]
    net.zero_grad()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        logits = net(inp)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
    loss.backward()
    ranks = []
    for p in net.parameters():
        if p.grad is None or p.grad.ndim < 2:
            continue
        g = p.grad.detach().float()
        if g.ndim > 2:
            g = g.reshape(g.shape[0], -1)
        s = torch.linalg.svdvals(g)
        s = s / (s.sum() + 1e-10)
        ent = -(s * torch.log(s + 1e-10)).sum().item()
        ranks.append(np.exp(ent))
    net.zero_grad()
    return float(np.mean(ranks)) if ranks else 0.0


def _get_grad_per_layer(net, batch_np):
    """Compute per-layer gradient dict {block_group: flat tensor}."""
    dat = torch.as_tensor(batch_np, dtype=torch.long, device=DEVICE)
    inp, tgt = dat[:, :-1], dat[:, 1:]
    net.zero_grad()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        logits = net(inp)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
    loss.backward()

    raw = _raw(net)
    layer_grads = {}
    for name, p in raw.named_parameters():
        if p.grad is None:
            continue
        key = _block_group(name)
        if key not in layer_grads:
            layer_grads[key] = []
        layer_grads[key].append(p.grad.detach().flatten())

    return {k: torch.cat(vs) for k, vs in layer_grads.items()}


def gradient_cosine(net, batch1_np, batch2_np):
    """Cosine similarity between gradients computed on two batches."""
    net.train()
    g1 = _get_grad_vector(net, batch1_np)
    g2 = _get_grad_vector(net, batch2_np)
    cos = F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).item()
    net.zero_grad()
    return cos


def gradient_cosine_per_layer(net, batch1_np, batch2_np):
    """Per-block cosine similarity between gradients on two batches.

    Returns dict {block_name: float} with cosine similarity per block.
    Block names are coarse (block_0, block_1, ..., embed_tok, lm_head).
    """
    net.train()
    g1 = _get_grad_per_layer(net, batch1_np)
    g2 = _get_grad_per_layer(net, batch2_np)
    net.zero_grad()

    result = {}
    for key in g1:
        if key in g2:
            cos = F.cosine_similarity(
                g1[key].unsqueeze(0), g2[key].unsqueeze(0)).item()
            result[key] = cos
    return result


def gradient_norm(net):
    """Total gradient norm (call after backward)."""
    total = 0.0
    for p in net.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm().item() ** 2
    return total ** 0.5


def gradient_norm_per_layer(net):
    """Per-layer gradient norms (call after backward)."""
    raw = _raw(net)
    norms = {}
    for name, p in raw.named_parameters():
        if p.grad is not None:
            norms[_layer_group(name)] = p.grad.detach().norm().item()
    return norms


def grad_norm_entropy(net, batch_np):
    """Entropy of the per-block gradient norm distribution.

    High entropy = gradient spread evenly across blocks.
    Low entropy = gradient concentrated in few blocks.
    Returns (entropy_float, per_block_norms_dict).
    """
    net.train()
    g = _get_grad_per_layer(net, batch_np)
    net.zero_grad()

    norms = {k: v.norm().item() for k, v in g.items()}
    vals = np.array(list(norms.values()))
    total = vals.sum()
    if total < 1e-10:
        return 0.0, norms
    p = vals / total
    ent = float(-np.sum(p * np.log(p + 1e-10)))
    return ent, norms


# ── post-hoc analysis (call after all phases complete) ────────────────────

def analyze(data, pt, ft_results, fg_results):
    """Run full post-hoc interpretability analysis.

    Returns dict with all analysis results, keyed by tag.
    """
    model_cfg = pt["model_cfg"]
    prompt_len = data["prompt_len"]
    eval_burst = data["eval_burst"]
    eval_other = data["eval_other"]

    sd_pt = load_sd(pt["ckpt_path"])
    net_pt = load_model(pt["ckpt_path"], compile_model=False, **model_cfg)

    results = {}
    for ft, fg in zip(ft_results, fg_results):
        tag = ft["tag"]
        sd_ft = load_sd(ft["ckpt_path"])

        # weight-space: pretrained -> finetuned
        drift_pt_ft = weight_drift_l2(sd_pt, sd_ft)
        cosine_pt_ft = weight_cosine_per_layer(sd_pt, sd_ft)
        svd_pt_ft = weight_delta_svd(sd_pt, sd_ft)

        # weight-space: finetuned -> forgotten
        fg_ckpt = fg.get("ckpt_path")
        if fg_ckpt:
            sd_fg = load_sd(fg_ckpt)
            drift_ft_fg = weight_drift_l2(sd_ft, sd_fg)
            drift_pt_fg = weight_drift_l2(sd_pt, sd_fg)
            cosine_ft_fg = weight_cosine_per_layer(sd_ft, sd_fg)
            svd_ft_fg = weight_delta_svd(sd_ft, sd_fg)
        else:
            drift_ft_fg = drift_pt_fg = cosine_ft_fg = svd_ft_fg = None

        # CKA: pretrained vs finetuned
        net_ft = load_model(ft["ckpt_path"], compile_model=False, **model_cfg)
        cka_pt_ft_burst = cka_between_models(net_pt, net_ft, eval_burst, prompt_len)
        cka_pt_ft_other = cka_diagonal(net_pt, net_ft, eval_other, prompt_len)

        # CKA: finetuned vs forgotten
        cka_ft_fg_burst = None
        if fg_ckpt:
            net_fg = load_model(fg_ckpt, compile_model=False, **model_cfg)
            cka_ft_fg_burst = cka_between_models(net_ft, net_fg, eval_burst, prompt_len)
            del net_fg

        del net_ft

        results[tag] = {
            "burst_frac": ft["burst_frac"],
            "drift_pt_ft": drift_pt_ft,
            "cosine_pt_ft": cosine_pt_ft,
            "svd_pt_ft": svd_pt_ft,
            "drift_ft_fg": drift_ft_fg,
            "drift_pt_fg": drift_pt_fg,
            "cosine_ft_fg": cosine_ft_fg,
            "svd_ft_fg": svd_ft_fg,
            "cka_pt_ft_burst": cka_pt_ft_burst,
            "cka_pt_ft_other": cka_pt_ft_other,
            "cka_ft_fg_burst": cka_ft_fg_burst,
        }

    del net_pt
    torch.cuda.empty_cache()
    return results
