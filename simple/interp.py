"""Interpretability metrics for burst experiment analysis.

Provides weight-space, representation, gradient, and sharpness metrics
to understand why higher burst concentration leads to faster forgetting.
"""
import numpy as np
import torch
import torch.nn.functional as F

from simple.model import load_model, eval_loss, DEVICE


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
    """SVD analysis of weight deltas for 2D weight matrices.

    Returns per-layer dict with: singular_values, effective_rank,
    top_k_var_frac, spectral_norm.
    """
    results = {}
    for k in sd_ref:
        if k not in sd_now or sd_ref[k].dim() != 2:
            continue
        delta = sd_now[k].float() - sd_ref[k].float()
        S = torch.linalg.svdvals(delta)
        s = S.numpy()
        # effective rank (entropy-based)
        s_norm = s / (s.sum() + 1e-10)
        eff_rank = float(np.exp(-np.sum(s_norm * np.log(s_norm + 1e-10))))
        # fraction of variance in top-k
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


# ── representation metrics ────────────────────────────────────────────────

def extract_hidden_states(net, data_np, prompt_len, batch_size=256):
    """Capture per-layer hidden states at the last prompt position.

    Returns dict {layer_idx: (N, D) numpy array}.
    """
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
    """Layer-by-layer CKA matrix between two models.

    Returns (n_layers, n_layers) numpy array.
    """
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


# ── deep representation analysis ──────────────────────────────────────────

def burst_bg_separation(net, eval_burst, eval_other, prompt_len):
    """Per-layer separation between burst and background representations.

    Returns per-layer dict with:
      - centroid_cosine: cosine between burst and bg centroids
      - centroid_l2: L2 distance between centroids
      - fisher: Fisher criterion (inter-class / intra-class variance)
      - burst_spread: mean pairwise distance within burst class
      - bg_spread: mean pairwise distance within bg class
    """
    acts_burst = extract_hidden_states(net, eval_burst, prompt_len)
    acts_bg = extract_hidden_states(net, eval_other, prompt_len)

    results = {}
    for layer in acts_burst:
        B = acts_burst[layer]   # (N_burst, D)
        G = acts_bg[layer]      # (N_bg, D)

        # centroids
        mu_b = B.mean(0)
        mu_g = G.mean(0)

        # centroid distance
        centroid_l2 = float(np.linalg.norm(mu_b - mu_g))
        cos = float(np.dot(mu_b, mu_g) /
                     (np.linalg.norm(mu_b) * np.linalg.norm(mu_g) + 1e-10))

        # intra-class spread (trace of within-class scatter)
        var_b = float(np.mean(np.sum((B - mu_b) ** 2, axis=1)))
        var_g = float(np.mean(np.sum((G - mu_g) ** 2, axis=1)))

        # Fisher criterion: inter / (intra + eps)
        inter = centroid_l2 ** 2
        intra = var_b + var_g
        fisher = float(inter / (intra + 1e-10))

        results[layer] = {
            "centroid_cosine": cos,
            "centroid_l2": centroid_l2,
            "fisher": fisher,
            "burst_spread": float(var_b ** 0.5),
            "bg_spread": float(var_g ** 0.5),
        }
    return results


def representation_drift(net_ref, net_new, data_np, prompt_len):
    """Per-layer centroid drift between two models on the same data.

    Returns per-layer dict with centroid_l2 and centroid_cosine.
    """
    acts_ref = extract_hidden_states(net_ref, data_np, prompt_len)
    acts_new = extract_hidden_states(net_new, data_np, prompt_len)
    results = {}
    for layer in acts_ref:
        mu_ref = acts_ref[layer].mean(0)
        mu_new = acts_new[layer].mean(0)
        l2 = float(np.linalg.norm(mu_new - mu_ref))
        cos = float(np.dot(mu_ref, mu_new) /
                     (np.linalg.norm(mu_ref) * np.linalg.norm(mu_new) + 1e-10))
        results[layer] = {"centroid_l2": l2, "centroid_cosine": cos}
    return results


def representation_pca(nets_and_labels, eval_burst, eval_other, prompt_len,
                        layer_idx=-1):
    """PCA projection of burst/bg representations from multiple models.

    Args:
        nets_and_labels: list of (net, label_str) pairs
        eval_burst, eval_other: eval data arrays
        prompt_len: prompt length
        layer_idx: which layer to use (-1 = last)

    Returns dict with:
        burst_coords: list of (N, 2) arrays (one per model)
        bg_coords: list of (N, 2) arrays
        labels: list of label strings
        explained_var: (2,) array of explained variance ratios
    """
    from sklearn.decomposition import PCA

    all_acts = []
    burst_slices = []
    bg_slices = []
    labels = []

    offset = 0
    for net, label in nets_and_labels:
        acts_b = extract_hidden_states(net, eval_burst, prompt_len)
        acts_g = extract_hidden_states(net, eval_other, prompt_len)

        # resolve layer index
        n_layers = len(acts_b)
        li = layer_idx if layer_idx >= 0 else n_layers + layer_idx

        Ab = acts_b[li]
        Ag = acts_g[li]
        all_acts.append(Ab)
        all_acts.append(Ag)
        burst_slices.append((offset, offset + Ab.shape[0]))
        offset += Ab.shape[0]
        bg_slices.append((offset, offset + Ag.shape[0]))
        offset += Ag.shape[0]
        labels.append(label)

    combined = np.concatenate(all_acts, axis=0)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(combined)

    burst_coords = [coords[s:e] for s, e in burst_slices]
    bg_coords = [coords[s:e] for s, e in bg_slices]

    return {
        "burst_coords": burst_coords,
        "bg_coords": bg_coords,
        "labels": labels,
        "explained_var": pca.explained_variance_ratio_,
    }


def probing_accuracy(net, eval_burst, eval_other, prompt_len):
    """Linear probing: per-layer accuracy of predicting burst vs background.

    Fits a logistic regression at each layer using a 50/50 train/test split.
    Higher accuracy = more linearly separable = burst info more explicit.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    acts_b = extract_hidden_states(net, eval_burst, prompt_len)
    acts_g = extract_hidden_states(net, eval_other, prompt_len)

    results = {}
    for layer in acts_b:
        X = np.concatenate([acts_b[layer], acts_g[layer]], axis=0)
        y = np.concatenate([np.ones(acts_b[layer].shape[0]),
                            np.zeros(acts_g[layer].shape[0])])
        clf = LogisticRegression(max_iter=500, solver="lbfgs")
        scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
        results[layer] = float(scores.mean())
    return results


# ── full representation analysis ──────────────────────────────────────────

def analyze_representations(data, pt, ft_results, fg_results):
    """Deep representation analysis across all phases and burst fractions.

    Returns dict keyed by tag, each containing per-layer separation,
    drift, probing, and PCA data.
    """
    prompt_len = data["prompt_len"]
    eval_burst = data["eval_burst"]
    eval_other = data["eval_other"]
    model_cfg = pt["model_cfg"]

    net_pt = load_model(pt["ckpt_path"], compile_model=False, **model_cfg)

    # pretrained baseline separation & probing
    sep_pt = burst_bg_separation(net_pt, eval_burst, eval_other, prompt_len)
    probe_pt = probing_accuracy(net_pt, eval_burst, eval_other, prompt_len)

    results = {"_pretrained": {"separation": sep_pt, "probing": probe_pt}}

    pca_nets = [
        (net_pt, "pretrained"),
    ]

    for ft, fg in zip(ft_results, fg_results):
        tag = ft["tag"]
        net_ft = load_model(ft["ckpt_path"], compile_model=False, **model_cfg)

        # separation at finetuned checkpoint
        sep_ft = burst_bg_separation(net_ft, eval_burst, eval_other, prompt_len)

        # probing at finetuned checkpoint
        probe_ft = probing_accuracy(net_ft, eval_burst, eval_other, prompt_len)

        # representation drift: pretrained -> finetuned (on burst and bg data)
        drift_burst = representation_drift(net_pt, net_ft, eval_burst, prompt_len)
        drift_bg = representation_drift(net_pt, net_ft, eval_other, prompt_len)

        pca_nets.append((net_ft, f"ft_{tag}"))

        fg_ckpt = fg.get("ckpt_path")
        sep_fg = probe_fg = drift_fg_burst = drift_fg_bg = None
        if fg_ckpt:
            net_fg = load_model(fg_ckpt, compile_model=False, **model_cfg)
            sep_fg = burst_bg_separation(
                net_fg, eval_burst, eval_other, prompt_len)
            probe_fg = probing_accuracy(
                net_fg, eval_burst, eval_other, prompt_len)
            drift_fg_burst = representation_drift(
                net_ft, net_fg, eval_burst, prompt_len)
            drift_fg_bg = representation_drift(
                net_ft, net_fg, eval_other, prompt_len)
            pca_nets.append((net_fg, f"fg_{tag}"))

        results[tag] = {
            "burst_frac": ft["burst_frac"],
            # separation (Fisher criterion etc.) at each phase
            "separation_pt": sep_pt,
            "separation_ft": sep_ft,
            "separation_fg": sep_fg,
            # probing accuracy at each phase
            "probing_pt": probe_pt,
            "probing_ft": probe_ft,
            "probing_fg": probe_fg,
            # representation centroid drift
            "drift_pt_ft_burst": drift_burst,
            "drift_pt_ft_bg": drift_bg,
            "drift_ft_fg_burst": drift_fg_burst,
            "drift_ft_fg_bg": drift_fg_bg,
        }

    # PCA across all models (last layer)
    n_layers = len(sep_pt)
    pca_last = representation_pca(
        pca_nets, eval_burst, eval_other, prompt_len, layer_idx=-1)
    pca_mid = representation_pca(
        pca_nets, eval_burst, eval_other, prompt_len,
        layer_idx=n_layers // 2)
    results["_pca_last_layer"] = pca_last
    results["_pca_mid_layer"] = pca_mid

    # cleanup
    del net_pt
    for net, _ in pca_nets[1:]:
        del net
    torch.cuda.empty_cache()

    return results


# ── gradient metrics ──────────────────────────────────────────────────────

def _get_grad_vector(net, batch_np):
    """Compute gradient vector for a batch (returns flat tensor on CPU)."""
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


def gradient_cosine(net, batch1_np, batch2_np):
    """Cosine similarity between gradients computed on two batches."""
    net.train()
    g1 = _get_grad_vector(net, batch1_np)
    g2 = _get_grad_vector(net, batch2_np)
    cos = F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).item()
    net.zero_grad()
    return cos


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


# ── sharpness ─────────────────────────────────────────────────────────────

@torch.no_grad()
def perturbation_sharpness(net, eval_data_np, epsilon=0.01, n_samples=5):
    """Measure loss sensitivity to random weight perturbations.

    Returns dict with base_loss, perturbed_mean, sharpness (delta).
    """
    net.eval()
    base_loss = eval_loss(net, eval_data_np)

    raw = _raw(net)
    orig_sd = {k: v.clone() for k, v in raw.state_dict().items()}

    losses = []
    for _ in range(n_samples):
        for v in raw.state_dict().values():
            noise = torch.randn_like(v) * epsilon * (v.abs().mean() + 1e-8)
            v.add_(noise)
        losses.append(eval_loss(net, eval_data_np))
        raw.load_state_dict(orig_sd)

    net.train()
    return {
        "base_loss": base_loss,
        "perturbed_mean": float(np.mean(losses)),
        "perturbed_std": float(np.std(losses)),
        "sharpness": float(np.mean(losses) - base_loss),
    }


# ── post-hoc analysis (call after all phases complete) ────────────────────

def analyze(data, pt, ft_results, fg_results):
    """Run full post-hoc interpretability analysis.

    Args:
        data: dict from make_data()
        pt: pretrain result dict
        ft_results: list of finetune result dicts
        fg_results: list of forget result dicts

    Returns:
        dict with all analysis results, keyed by tag.
    """
    vocab_size = data["vocab_size"]
    context_size = data["context_size"]
    prompt_len = data["prompt_len"]
    eval_burst = data["eval_burst"]
    eval_other = data["eval_other"]
    model_cfg = pt["model_cfg"]

    sd_pt = load_sd(pt["ckpt_path"])

    # load pretrained model (no compile for hooks)
    net_pt = load_model(pt["ckpt_path"], compile_model=False, **model_cfg)

    results = {}
    for ft, fg in zip(ft_results, fg_results):
        tag = ft["tag"]
        sd_ft = load_sd(ft["ckpt_path"])

        # -- weight-space: pretrained -> finetuned --
        drift_pt_ft = weight_drift_l2(sd_pt, sd_ft)
        cosine_pt_ft = weight_cosine_per_layer(sd_pt, sd_ft)
        svd_pt_ft = weight_delta_svd(sd_pt, sd_ft)

        # -- weight-space: finetuned -> forgotten --
        fg_ckpt = fg.get("ckpt_path")
        if fg_ckpt:
            sd_fg = load_sd(fg_ckpt)
            drift_ft_fg = weight_drift_l2(sd_ft, sd_fg)
            drift_pt_fg = weight_drift_l2(sd_pt, sd_fg)
            cosine_ft_fg = weight_cosine_per_layer(sd_ft, sd_fg)
            svd_ft_fg = weight_delta_svd(sd_ft, sd_fg)
        else:
            drift_ft_fg = drift_pt_fg = cosine_ft_fg = svd_ft_fg = None

        # -- CKA: pretrained vs finetuned --
        net_ft = load_model(ft["ckpt_path"], compile_model=False, **model_cfg)
        cka_pt_ft_burst = cka_between_models(net_pt, net_ft, eval_burst, prompt_len)
        cka_pt_ft_other = cka_diagonal(net_pt, net_ft, eval_other, prompt_len)

        # -- CKA: finetuned vs forgotten --
        cka_ft_fg_burst = None
        if fg_ckpt:
            net_fg = load_model(fg_ckpt, compile_model=False, **model_cfg)
            cka_ft_fg_burst = cka_between_models(net_ft, net_fg, eval_burst, prompt_len)
            del net_fg

        # -- sharpness at finetune endpoint --
        sharp_burst = perturbation_sharpness(net_ft, eval_burst)
        sharp_other = perturbation_sharpness(net_ft, eval_other)

        del net_ft

        results[tag] = {
            "burst_frac": ft["burst_frac"],
            # weight-space: pt -> ft
            "drift_pt_ft": drift_pt_ft,
            "cosine_pt_ft": cosine_pt_ft,
            "svd_pt_ft": svd_pt_ft,
            # weight-space: ft -> fg
            "drift_ft_fg": drift_ft_fg,
            "drift_pt_fg": drift_pt_fg,
            "cosine_ft_fg": cosine_ft_fg,
            "svd_ft_fg": svd_ft_fg,
            # CKA
            "cka_pt_ft_burst": cka_pt_ft_burst,
            "cka_pt_ft_other": cka_pt_ft_other,
            "cka_ft_fg_burst": cka_ft_fg_burst,
            # sharpness
            "sharpness_burst": sharp_burst,
            "sharpness_other": sharp_other,
        }

    del net_pt
    torch.cuda.empty_cache()

    return results
