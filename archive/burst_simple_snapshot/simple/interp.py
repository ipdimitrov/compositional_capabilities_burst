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


def weight_drift_per_block(sd_ref, sd_now):
    """Per-block L2 distance between two state dicts (coarse grouping)."""
    block_sq = {}
    for k in sd_ref:
        if k not in sd_now:
            continue
        delta = sd_now[k].float() - sd_ref[k].float()
        sq = (delta.norm().item()) ** 2
        blk = _block_group(k)
        block_sq[blk] = block_sq.get(blk, 0.0) + sq
    return {k: v ** 0.5 for k, v in block_sq.items()}


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


def gradient_cosine_and_norms_per_layer(net, batch1_np, batch2_np):
    """Per-block cosine similarity and gradient norms for two batches.

    Same cost as gradient_cosine_per_layer (two backward passes).

    Returns:
        cosines  : {block: cosine_similarity (float)}
        norms1   : {block: grad_norm of batch1 (float)}
        norms2   : {block: grad_norm of batch2 (float)}
    """
    net.train()
    g1 = _get_grad_per_layer(net, batch1_np)
    g2 = _get_grad_per_layer(net, batch2_np)
    net.zero_grad()

    cosines, norms1, norms2 = {}, {}, {}
    for key, g in g1.items():
        norms1[key] = g.norm().item()
    for key, g in g2.items():
        norms2[key] = g.norm().item()
    for key in g1:
        if key in g2:
            cosines[key] = F.cosine_similarity(
                g1[key].unsqueeze(0), g2[key].unsqueeze(0)).item()
    return cosines, norms1, norms2


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


def gradient_effective_rank(net, sample_fn, n_batches=50, variance_threshold=0.9):
    """Effective rank of the gradient matrix over multiple mini-batches.

    Collects n_batches gradient vectors, stacks into a (n_batches, D) matrix,
    computes SVD, and returns:
      - effective rank = exp(H) where H is the Shannon entropy of the
        normalized singular-value distribution
      - number of PCA components needed to explain `variance_threshold`
        fraction of the total variance (singular values squared)

    Args:
        net: model (must be in train mode)
        sample_fn: callable() -> batch_np, returns one mini-batch each call
        n_batches: number of gradient samples to collect
        variance_threshold: fraction of variance to explain (default 0.9)

    Returns:
        effective_rank (float), pca_components (int), singular_values (1-D numpy array)
    """
    net.train()
    grads = []
    for _ in range(n_batches):
        batch = sample_fn()
        g = _get_grad_vector(net, batch)
        grads.append(g.detach())
    net.zero_grad()

    G = torch.stack(grads)  # (n_batches, D)
    # SVD on the smaller dimension — we only need singular values
    S = torch.linalg.svdvals(G.float())  # (min(n_batches, D),)
    s = S.cpu().numpy()

    # effective rank from entropy of normalized singular values
    s_pos = s[s > 1e-10]
    if len(s_pos) == 0:
        return 0.0, 0, s
    p = s_pos / s_pos.sum()
    entropy = -float(np.sum(p * np.log(p)))
    eff_rank = float(np.exp(entropy))

    # PCA components for variance_threshold: count on squared singular values
    var = s ** 2
    total_var = var.sum()
    if total_var < 1e-10:
        return eff_rank, 0, s
    cumvar = np.cumsum(var) / total_var
    pca_k = int(np.searchsorted(cumvar, variance_threshold) + 1)

    return eff_rank, pca_k, s


# ── Fisher information metrics ────────────────────────────────────────────

def _sample_bg_batches(bg_pool, n_batches, batch_size, rng):
    """Sample n_batches independent mini-batches from the background pool."""
    bg_ids = list(bg_pool.keys())
    batches = []
    for _ in range(n_batches):
        per = batch_size // len(bg_ids)
        rem = batch_size % len(bg_ids)
        parts = []
        for i, tid in enumerate(bg_ids):
            k = per + (1 if i < rem else 0)
            if k > 0:
                idx = rng.integers(len(bg_pool[tid]), size=k)
                parts.append(bg_pool[tid][idx])
        batch = np.concatenate(parts)[rng.permutation(batch_size)]
        batches.append(batch)
    return batches


def compute_fisher_eigenvectors(net, data, k=5, n_power_iters=10,
                                n_batches=16, batch_size=64, seed=0):
    """Top-k eigenvectors of the empirical Fisher via subspace power iteration.

    Empirical Fisher approximation:
        F ≈ (1/N) Σ_batch  g_batch  g_batch^T

    Power iteration update per round:
        FV[:, j] = (1/N) Σ_batch  g_batch  (g_batch · V[:, j])

    followed by QR re-orthogonalisation to maintain an orthonormal basis.
    Computed both globally (full parameter vector) and per block group
    (matching the grouping used by gradient_cosine_per_layer).

    Args:
        net: pretrained model
        data: data dict from make_data() — uses data["bg_pool"]
        k: number of eigenvectors
        n_power_iters: subspace iteration rounds
        n_batches: mini-batches sampled per iteration round
        batch_size: examples per mini-batch
        seed: numpy RNG seed

    Returns:
        dict with keys:
            "global_eigvecs"      : (D, k) float32 numpy array
            "global_eigvals"      : (k,)   float32 numpy array
            "per_layer_eigvecs"   : {block_name: (D_layer, k) array}
            "per_layer_eigvals"   : {block_name: (k,) array}
    """
    rng = np.random.default_rng(seed)
    bg_pool = data["bg_pool"]
    device = next(net.parameters()).device

    # ── global ────────────────────────────────────────────────────────────
    net.train()
    D = sum(p.numel() for p in net.parameters() if p.requires_grad)
    k_g = min(k, D - 1)
    V = torch.randn(D, k_g, device=device)
    V, _ = torch.linalg.qr(V)
    R_g = None

    for _ in range(n_power_iters):
        FV = torch.zeros(D, k_g, device=device)
        for batch in _sample_bg_batches(bg_pool, n_batches, batch_size, rng):
            g = _get_grad_vector(net, batch)        # (D,)
            proj = g @ V                             # (k_g,)
            FV += g.unsqueeze(1) * proj.unsqueeze(0)
        FV /= n_batches
        V, R_g = torch.linalg.qr(FV)

    global_eigvecs = V.detach().cpu().numpy().astype(np.float32)
    global_eigvals = (R_g.diag().abs().detach().cpu().numpy().astype(np.float32)
                      if R_g is not None else np.ones(k_g, np.float32))

    # ── per layer ─────────────────────────────────────────────────────────
    raw = _raw(net)
    layer_dims: dict[str, int] = {}
    for name, p in raw.named_parameters():
        if p.requires_grad:
            key = _block_group(name)
            layer_dims[key] = layer_dims.get(key, 0) + p.numel()

    V_layers: dict[str, torch.Tensor] = {}
    for key, d in layer_dims.items():
        k_l = min(k, max(d - 1, 1))
        Vl = torch.randn(d, k_l, device=device)
        Vl, _ = torch.linalg.qr(Vl)
        V_layers[key] = Vl

    R_layers: dict[str, torch.Tensor | None] = {key: None for key in V_layers}

    for _ in range(n_power_iters):
        FV_l = {key: torch.zeros_like(V_layers[key]) for key in V_layers}
        for batch in _sample_bg_batches(bg_pool, n_batches, batch_size, rng):
            g_dict = _get_grad_per_layer(net, batch)
            for key, g in g_dict.items():
                if key in V_layers:
                    proj = g @ V_layers[key]
                    FV_l[key] += g.unsqueeze(1) * proj.unsqueeze(0)
        for key in V_layers:
            FV_l[key] /= n_batches
            V_layers[key], R_layers[key] = torch.linalg.qr(FV_l[key])

    per_layer_eigvecs = {
        key: V_layers[key].detach().cpu().numpy().astype(np.float32)
        for key in V_layers
    }
    per_layer_eigvals = {
        key: (R_layers[key].diag().abs().detach().cpu().numpy().astype(np.float32)
              if R_layers[key] is not None else np.ones(1, np.float32))
        for key in R_layers
    }

    net.zero_grad()
    return {
        "global_eigvecs": global_eigvecs,
        "global_eigvals": global_eigvals,
        "per_layer_eigvecs": per_layer_eigvecs,
        "per_layer_eigvals": per_layer_eigvals,
    }


def fisher_overlap_score(g_flat, eigvecs):
    """Fraction of gradient energy lying in the top-k Fisher subspace.

        fisher_overlap = Σ_i (g · v_i)² / ‖g‖²

    Args:
        g_flat: (D,) gradient as torch.Tensor or numpy array
        eigvecs: (D, k) float32 numpy array — Fisher eigenvectors as columns

    Returns:
        total_overlap (float): sum of per-direction overlaps, in [0, 1]
        per_dir (list[float]): k individual (g · v_i)² / ‖g‖² values
    """
    if isinstance(g_flat, torch.Tensor):
        g_np = g_flat.detach().cpu().float().numpy()
    else:
        g_np = np.asarray(g_flat, dtype=np.float32)
    g_norm_sq = float(np.dot(g_np, g_np)) + 1e-10
    projections = eigvecs.T @ g_np          # (k,)
    per_dir = (projections ** 2 / g_norm_sq).tolist()
    return float(sum(per_dir)), per_dir


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
