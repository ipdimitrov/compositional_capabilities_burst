"""Post-hoc gradient cosine similarity computation on saved checkpoints.

Runs after training (burst/experiment.py) as a separate pass, loading
checkpoints and computing grad-sim with full GPU utilisation at the
grad_sim_batch_size level.  Checkpoints are kept by default so the
computation can be re-run with different settings.

Usage:
    python burst/grad_sim.py <run_dir>
    python burst/grad_sim.py <run_dir> --n-workers 8 --grad-sim-batch-size 2048
    python burst/grad_sim.py <run_dir> --delete-checkpoints
"""
import sys, os, argparse, pickle, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from omegaconf import OmegaConf

from net.nanogpt import nanoGPT
from burst.parallel import run_job_pool
from burst.config import PHASE_PRE_BURST, PHASE_BURST, PHASE_REVERSION, parse_run_config
from burst.gpu import gpu_cfg

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _flat_grad(net) -> torch.Tensor:
    grads = [p.grad.detach().view(-1) for p in net.parameters() if p.grad is not None]
    return torch.cat(grads) if grads else torch.zeros(1, device=DEVICE)


def _layer_groups(net) -> list[tuple[str, list[str]]]:
    """Return ordered (short_name, [param_name, ...]) groups for per-layer grad-sim.

    Groups:
      emb       -- wte + wpe embeddings
      L{i}_ln   -- block i layernorms (ln_1, ln_2)
      L{i}_attn -- block i attention (c_attn, c_proj)
      L{i}_mlp  -- block i MLP (c_fc, c_proj)
      ln_f      -- final layernorm
    LM_head is weight-tied to wte so it is omitted to avoid double-counting.
    """
    groups: list[tuple[str, list[str]]] = []
    all_param_names = {n for n, _ in net.named_parameters()}

    emb_params = [n for n in all_param_names
                  if n in ("transformer.wte.weight", "transformer.wpe.weight")]
    if emb_params:
        groups.append(("emb", sorted(emb_params)))

    n_layer = net.config.n_layer
    for i in range(n_layer):
        prefix = f"transformer.h.{i}"
        ln_params = [n for n in all_param_names if n.startswith(f"{prefix}.ln_")]
        attn_params = [n for n in all_param_names if n.startswith(f"{prefix}.attn.")]
        mlp_params = [n for n in all_param_names if n.startswith(f"{prefix}.mlp.")]
        if ln_params:
            groups.append((f"L{i}_ln", sorted(ln_params)))
        if attn_params:
            groups.append((f"L{i}_attn", sorted(attn_params)))
        if mlp_params:
            groups.append((f"L{i}_mlp", sorted(mlp_params)))

    lnf_params = [n for n in all_param_names if n.startswith("transformer.ln_f")]
    if lnf_params:
        groups.append(("ln_f", sorted(lnf_params)))

    return groups


def _grad_vecs_per_layer(net, docs_np: np.ndarray, n_samples: int,
                          layer_groups: list[tuple[str, list[str]]]) -> dict[str, torch.Tensor]:
    """Run one backward pass and extract per-layer gradient vectors."""
    n = min(n_samples, docs_np.shape[0])
    idx = np.random.choice(docs_np.shape[0], n, replace=False)
    dat = torch.as_tensor(docs_np[idx], dtype=torch.long, device=DEVICE)
    inp, tgt = dat[:, :-1], dat[:, 1:]
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        logits = net(inp)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
    loss.backward()

    param_map = dict(net.named_parameters())
    result: dict[str, torch.Tensor] = {}
    for name, pnames in layer_groups:
        grads = []
        for pn in pnames:
            p = param_map.get(pn)
            if p is not None and p.grad is not None:
                grads.append(p.grad.detach().view(-1).float())
        result[name] = torch.cat(grads) if grads else torch.zeros(1, device=DEVICE)
    return result


def _grad_vec_for_docs(net, docs_np: np.ndarray, n_samples: int) -> torch.Tensor:
    n = min(n_samples, docs_np.shape[0])
    idx = np.random.choice(docs_np.shape[0], n, replace=False)
    dat = torch.as_tensor(docs_np[idx], dtype=torch.long, device=DEVICE)
    inp, tgt = dat[:, :-1], dat[:, 1:]
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        logits = net(inp)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
    loss.backward()
    return _flat_grad(net).float()


def compute_grad_cosine_sim(net, docs_burst_BL, docs_other_BL,
                            n_samples: int) -> dict:
    net.train()
    net.zero_grad(set_to_none=True)
    g_burst = _grad_vec_for_docs(net, docs_burst_BL, n_samples=n_samples)
    net.zero_grad(set_to_none=True)
    g_other = _grad_vec_for_docs(net, docs_other_BL, n_samples=n_samples)
    cos_sim = F.cosine_similarity(g_burst.unsqueeze(0), g_other.unsqueeze(0)).item()
    net.zero_grad(set_to_none=True)
    return {"burst_vs_other": cos_sim}


def compute_grad_cosine_sim_per_layer(net, docs_burst_BL, docs_other_BL,
                                       n_samples: int) -> dict:
    """Compute burst-vs-other cosine similarity separately for each layer group."""
    layer_groups = _layer_groups(net)
    net.train()

    net.zero_grad(set_to_none=True)
    burst_vecs = _grad_vecs_per_layer(net, docs_burst_BL, n_samples, layer_groups)
    net.zero_grad(set_to_none=True)
    other_vecs = _grad_vecs_per_layer(net, docs_other_BL, n_samples, layer_groups)
    net.zero_grad(set_to_none=True)

    per_layer: dict[str, float] = {}
    for name, _ in layer_groups:
        g_b = burst_vecs[name]
        g_o = other_vecs[name]
        sim = F.cosine_similarity(g_b.unsqueeze(0), g_o.unsqueeze(0)).item()
        per_layer[name] = sim

    layer_names = [name for name, _ in layer_groups]
    return {"per_layer": per_layer, "layer_names": layer_names}


def compute_pairwise_grad_sim(net, task_docs: dict,
                               burst_tasks: list, other_tasks: list,
                               n_samples: int, depth: int,
                               burst_pos: int, n_a: int) -> dict:
    """Pairwise grad cosine sim with principled task grouping.

    Groups:
      BURST       -- all burst-class tasks pooled
      O_F1..O_Fn  -- other-class tasks grouped by function at burst_pos
      ALL_OTHER   -- all other-class tasks pooled
      ALL_DATA    -- everything pooled
    """
    from burst.config import CLASS_BURST
    net.train()

    burst_pos_idx = 1 + (depth - burst_pos)

    group_docs: dict[str, list[np.ndarray]] = {"BURST": []}
    for fi in range(1, n_a + 1):
        group_docs[f"O_F{fi}"] = []

    for task, docs in task_docs.items():
        if docs.shape[0] == 0:
            continue
        if task[0] == CLASS_BURST:
            group_docs["BURST"].append(docs)
        elif burst_pos_idx >= len(task):
            continue
        else:
            fn_at_bp = task[burst_pos_idx]
            key = f"O_F{fn_at_bp}"
            if key in group_docs:
                group_docs[key].append(docs)

    other_sub_docs = []
    for fi in range(1, n_a + 1):
        other_sub_docs.extend(group_docs[f"O_F{fi}"])
    group_docs["ALL_OTHER"] = list(other_sub_docs)
    group_docs["ALL_DATA"] = group_docs["BURST"] + group_docs["ALL_OTHER"]

    label_order = ["BURST"]
    label_order += [f"O_F{fi}" for fi in range(1, n_a + 1)]
    label_order += ["ALL_OTHER", "ALL_DATA"]

    grad_vecs = []
    for label in label_order:
        doc_list = group_docs[label]
        if doc_list:
            pooled = np.concatenate(doc_list)
            net.zero_grad(set_to_none=True)
            grad_vecs.append(_grad_vec_for_docs(net, pooled, n_samples=n_samples))
        else:
            grad_vecs.append(None)

    n = len(label_order)
    matrix = np.eye(n)
    valid = [(i, v) for i, v in enumerate(grad_vecs) if v is not None]
    if len(valid) >= 2:
        G = torch.stack([v for _, v in valid])
        G_norm = F.normalize(G, dim=1)
        sim = (G_norm @ G_norm.T).cpu().numpy()
        for ri, (i, _) in enumerate(valid):
            for rj, (j, _) in enumerate(valid):
                matrix[i, j] = sim[ri, rj]

    net.zero_grad(set_to_none=True)
    return {"matrix": matrix.tolist(), "labels": label_order,
            "n_burst": 1, "n_other_sub": n_a,
            "n_other": 1, "n_all": 1}


def _worker_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--job-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--grad-sim-batch-size", type=int, required=True)
    args = parser.parse_args()

    with open(args.job_path, "rb") as f:
        job = pickle.load(f)
    with open(args.data_path, "rb") as f:
        target_pool, bg_pool = pickle.load(f)

    cfg = job["cfg"]
    ckpt_path = job["ckpt_path"]
    step = job["step"]
    phase = job["phase"]
    is_pairwise = job["is_pairwise"]
    depth = job["depth"]
    burst_pos_val = job["burst_pos"]
    n_a = job["n_a"]
    gs_bs = args.grad_sim_batch_size

    net = nanoGPT(OmegaConf.create({
        "compile": False, "vocab_size": cfg["vocab_size"],
        "context_size": cfg["context_size"],
        "n_layer": cfg["n_layer"], "n_head": cfg["n_head"],
        "n_embd": cfg["n_embd"], "dropout": 0.0, "bias": False, "mlp": True,
    })).to(DEVICE)
    net.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))

    burst_docs_all = np.concatenate(list(target_pool.values())) if target_pool else None
    other_docs_all = np.concatenate(list(bg_pool.values())) if bg_pool else None

    result = {"label": job["label"], "parent_label": job["parent_label"],
              "step": step, "phase": phase}

    if burst_docs_all is not None and other_docs_all is not None:
        sim = compute_grad_cosine_sim(net, burst_docs_all, other_docs_all, n_samples=gs_bs)
        result["burst_vs_other"] = sim["burst_vs_other"]
        layer_sim = compute_grad_cosine_sim_per_layer(
            net, burst_docs_all, other_docs_all, n_samples=gs_bs)
        result["per_layer_sim"] = layer_sim["per_layer"]
        result["layer_names"] = layer_sim["layer_names"]

    if is_pairwise:
        task_docs = {**target_pool, **bg_pool}
        burst_tasks = list(target_pool.keys())
        other_tasks = list(bg_pool.keys())
        snap = compute_pairwise_grad_sim(
            net, task_docs, burst_tasks, other_tasks,
            n_samples=gs_bs, depth=depth,
            burst_pos=burst_pos_val, n_a=n_a)
        result["pairwise"] = snap

    with open(args.output_path, "wb") as f:
        pickle.dump(result, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--grad-sim-batch-size", type=int, default=None)
    parser.add_argument("--delete-checkpoints", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir
    with open(run_dir / "config.json") as f:
        run_cfg = json.load(f)

    rc = parse_run_config(run_cfg)
    base_cfg, depth, burst_pos, n_a = rc["base_cfg"], rc["depth"], rc["burst_pos"], rc["n_a"]
    gs_bs = args.grad_sim_batch_size or base_cfg.get("grad_sim_batch_size", 2048)
    P = base_cfg.get("pre_burst_steps", 0)
    T = base_cfg["total_steps"]
    U = base_cfg["reversion_steps"]

    pairwise_global_steps = {P, P + T // 2, P + T - 1, P + T + U // 2, P + T + U - 1}

    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.exists():
        print(f"No checkpoints directory in {run_dir}, nothing to do.", flush=True)
        return

    job_entries = run_cfg["jobs"]
    sample_cfg = {**base_cfg,
                  "vocab_size": base_cfg.get("vocab_size", 128),
                  "context_size": base_cfg.get("context_size", 80)}
    for j in job_entries:
        label = j["label"]
        ckpt_dir = ckpt_root / label
        if ckpt_dir.exists():
            sample_cfg_path = run_dir / f"{label}.pkl"
            if sample_cfg_path.exists():
                with open(sample_cfg_path, "rb") as f:
                    r = pickle.load(f)
                sample_cfg = r["config"]
            break

    n_workers = args.n_workers or gpu_cfg.gradsim_workers
    print(f"{gpu_cfg.summary()}", flush=True)
    print(f"Grad-sim: batch_size={gs_bs}, workers={n_workers}", flush=True)

    jobs = []
    for j in job_entries:
        label = j["label"]
        ckpt_dir = ckpt_root / label
        if not ckpt_dir.exists():
            continue

        result_path = run_dir / f"{label}.pkl"
        if not result_path.exists():
            continue
        with open(result_path, "rb") as f:
            result = pickle.load(f)
        cfg = result["config"]

        for pt_file in sorted(ckpt_dir.glob("step_*.pt")):
            step = int(pt_file.stem.split("_")[1])
            if step < P:
                phase = PHASE_PRE_BURST
            elif step < P + T:
                phase = PHASE_BURST
            else:
                phase = PHASE_REVERSION
            is_pairwise = step in pairwise_global_steps
            jobs.append({
                "label": f"{label}_step{step}",
                "parent_label": label,
                "step": step,
                "phase": phase,
                "is_pairwise": is_pairwise,
                "ckpt_path": str(pt_file),
                "cfg": cfg,
                "depth": depth,
                "burst_pos": burst_pos,
                "n_a": n_a,
            })

    if not jobs:
        print("No checkpoint jobs found.", flush=True)
        return

    print(f"Jobs: {len(jobs)} checkpoints across {len(job_entries)} labels", flush=True)

    data_path = str(run_dir / "_data.pkl")
    with open(data_path, "rb") as f:
        target_pool, bg_pool, _, _, _ = pickle.load(f)

    worker_script = str(Path(__file__))

    def build_cmd(script, job_path, data_path, output_path):
        return [sys.executable, script, "--worker",
                "--job-path", job_path, "--data-path", data_path,
                "--output-path", output_path,
                "--grad-sim-batch-size", str(gs_bs)]

    def on_done(jr, n_done, n_total):
        status = "ok" if jr.success else f"FAIL: {jr.error[:80]}"
        print(f"  [{n_done}/{n_total}] {jr.label}: {status}", flush=True)

    results = run_job_pool(
        jobs=jobs,
        worker_script=worker_script,
        build_cmd=build_cmd,
        on_done=on_done,
        n_workers=n_workers,
        data_payload=(target_pool, bg_pool),
        poll_interval=1.0,
        tmp_prefix="grad_sim_",
    )

    per_label: dict[str, dict] = {}
    for jr in results:
        if not jr.success:
            continue
        d = jr.data
        parent = d["parent_label"]
        if parent not in per_label:
            per_label[parent] = {
                "grad_sim_log": {"step": [], "phase": [], "burst_vs_other": [],
                                 "per_layer": {}},
                "pairwise_snapshots": [],
            }
        entry = per_label[parent]
        if "burst_vs_other" in d:
            entry["grad_sim_log"]["step"].append(d["step"])
            entry["grad_sim_log"]["phase"].append(d["phase"])
            entry["grad_sim_log"]["burst_vs_other"].append(d["burst_vs_other"])
        if "per_layer_sim" in d:
            for layer_name, sim_val in d["per_layer_sim"].items():
                if layer_name not in entry["grad_sim_log"]["per_layer"]:
                    entry["grad_sim_log"]["per_layer"][layer_name] = []
                entry["grad_sim_log"]["per_layer"][layer_name].append(sim_val)
            if "layer_names" not in entry["grad_sim_log"]:
                entry["grad_sim_log"]["layer_names"] = d.get("layer_names", [])
        if "pairwise" in d:
            snap = d["pairwise"]
            snap["step"] = d["step"]
            snap["phase"] = d["phase"]
            entry["pairwise_snapshots"].append(snap)

    for label, entry in per_label.items():
        gs_log = entry["grad_sim_log"]
        order = np.argsort(gs_log["step"])
        gs_log["step"] = [gs_log["step"][i] for i in order]
        gs_log["phase"] = [gs_log["phase"][i] for i in order]
        gs_log["burst_vs_other"] = [gs_log["burst_vs_other"][i] for i in order]
        for layer_name in gs_log["per_layer"]:
            vals = gs_log["per_layer"][layer_name]
            if len(vals) == len(order):
                gs_log["per_layer"][layer_name] = [vals[i] for i in order]
        entry["pairwise_snapshots"].sort(key=lambda s: s["step"])

    gs_dir = run_dir / "grad_cosine_sim"
    gs_dir.mkdir(exist_ok=True)

    for j in job_entries:
        label = j["label"]
        if label not in per_label:
            continue
        entry = per_label[label]
        gs_log = entry["grad_sim_log"]
        record = {
            "schedule": j["schedule"], "seed": j["seed"], "label": label,
            "grad_sim_batch_size": gs_bs,
            "grad_sim_log": gs_log,
            "layer_names": gs_log.get("layer_names", []),
            "pairwise_snapshots": entry["pairwise_snapshots"],
        }
        with open(gs_dir / f"{label}.json", "w") as f:
            json.dump(record, f)

    all_results_path = run_dir / "all_results.pkl"
    if all_results_path.exists():
        with open(all_results_path, "rb") as f:
            all_results = pickle.load(f)
        for r in all_results:
            label = r["label"]
            if label in per_label:
                r["grad_sim_log"] = per_label[label]["grad_sim_log"]
                r["pairwise_snapshots"] = per_label[label]["pairwise_snapshots"]
        with open(all_results_path, "wb") as f:
            pickle.dump(all_results, f)
        print(f"Updated all_results.pkl with grad-sim data for {len(per_label)} labels", flush=True)

    if args.delete_checkpoints:
        import shutil
        shutil.rmtree(ckpt_root)
        print(f"Cleaned up checkpoints", flush=True)

    print(f"Grad-sim done: {len(per_label)} labels, {sum(jr.success for jr in results)}/{len(results)} ok",
          flush=True)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _worker_main()
    else:
        main()
