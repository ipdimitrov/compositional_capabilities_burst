import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import torch
from burst.config import BurstExperimentConfig
from burst.data import (
    build_function_pool, tag_tasks, generate_pool,
    generate_documents_for_task, ScheduleSampler, BurstDataset,
)
from burst.train import (
    make_model, make_optimizer, eval_accuracy, get_space_pos,
    snapshot_weights, weight_deltas_frobenius,
)
from synthetic.init import set_seed


def get_test_cfg():
    return BurstExperimentConfig(
        seed=0, total_steps=20, batch_size=8, p_target=0.20,
        undo_steps=10, relearn_steps=10, p_relearn=0.10,
        eval_every=5, ndocuments=200, neval_documents=50,
        n_train_compositions=20,
    )


def test_build_function_pool():
    cfg = get_test_cfg()
    set_seed(cfg.seed)
    syn, composed_functions, info = build_function_pool(cfg)
    assert "train" in composed_functions
    assert len(composed_functions["train"]) > 0
    assert "train_id" in info
    print("  PASS: build_function_pool")


def test_tag_tasks():
    cfg = get_test_cfg()
    set_seed(cfg.seed)
    syn, cf, info = build_function_pool(cfg)
    tids, bids, fl = tag_tasks(info, cf, n_target=2)
    assert len(tids) == 2
    assert len(bids) == len(info["train_id"]) - 2
    print("  PASS: tag_tasks")


def test_generate_documents():
    cfg = get_test_cfg()
    set_seed(cfg.seed)
    syn, cf, info = build_function_pool(cfg)
    tids, bids, fl = tag_tasks(info, cf, n_target=1)
    docs = generate_documents_for_task(syn, tids[0], fl, 10)
    assert docs.shape[0] == 10 and docs.ndim == 2
    print("  PASS: generate_documents")


def test_schedule_end_burst():
    cfg = get_test_cfg()
    set_seed(cfg.seed)
    syn, cf, info = build_function_pool(cfg)
    tids, bids, fl = tag_tasks(info, cf, n_target=1)
    tp = generate_pool(syn, tids, fl, 50)
    bp = generate_pool(syn, bids, fl, 50)
    s = ScheduleSampler(tp, bp, batch_size=8)
    assert s._n_target_for_step(0, 100, "end_burst", 0.1, 1) == 0
    assert s._n_target_for_step(95, 100, "end_burst", 0.1, 1) == 8
    print("  PASS: end_burst schedule")


def test_schedule_mid_burst():
    cfg = get_test_cfg()
    set_seed(cfg.seed)
    syn, cf, info = build_function_pool(cfg)
    tids, bids, fl = tag_tasks(info, cf, n_target=1)
    tp = generate_pool(syn, tids, fl, 50)
    bp = generate_pool(syn, bids, fl, 50)
    s = ScheduleSampler(tp, bp, batch_size=8)
    assert s._n_target_for_step(0, 100, "mid_burst", 0.1, 1) == 0
    assert s._n_target_for_step(50, 100, "mid_burst", 0.1, 1) == 8
    assert s._n_target_for_step(99, 100, "mid_burst", 0.1, 1) == 0
    print("  PASS: mid_burst schedule")


def test_schedule_early_burst():
    cfg = get_test_cfg()
    set_seed(cfg.seed)
    syn, cf, info = build_function_pool(cfg)
    tids, bids, fl = tag_tasks(info, cf, n_target=1)
    tp = generate_pool(syn, tids, fl, 50)
    bp = generate_pool(syn, bids, fl, 50)
    s = ScheduleSampler(tp, bp, batch_size=8)
    assert s._n_target_for_step(0, 100, "early_burst", 0.1, 1) == 8
    assert s._n_target_for_step(50, 100, "early_burst", 0.1, 1) == 0
    print("  PASS: early_burst schedule")


def test_weight_snapshot():
    cfg = get_test_cfg()
    net = make_model(cfg, "cpu")
    w1 = snapshot_weights(net)
    x = torch.randint(0, 10, (2, 10))
    loss = net(x).sum()
    loss.backward()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(p.grad * 0.01)
    w2 = snapshot_weights(net)
    deltas = weight_deltas_frobenius(w1, w2)
    assert all(v > 0 for v in deltas.values())
    print("  PASS: weight_snapshot")


def test_eval_accuracy():
    cfg = get_test_cfg()
    set_seed(cfg.seed)
    syn, cf, info = build_function_pool(cfg)
    tids, bids, fl = tag_tasks(info, cf, n_target=1)
    ed = generate_pool(syn, tids, fl, 20)
    ef = np.concatenate(list(ed.values()))
    sp = get_space_pos({"t": ef}, syn)
    net = make_model(cfg, "cpu")
    acc = eval_accuracy(net, ef, sp, "cpu")
    assert 0.0 <= acc <= 1.0
    print(f"  PASS: eval_accuracy (acc={acc:.4f})")


def test_burst_dataset():
    ds = BurstDataset(np.random.randint(0, 10, (16, 20)))
    assert len(ds) == 16
    inp, tgt = ds[0]
    assert inp.shape[0] == 19
    print("  PASS: BurstDataset")


if __name__ == "__main__":
    print("Running burst tests...\n")
    test_build_function_pool()
    test_tag_tasks()
    test_generate_documents()
    test_schedule_end_burst()
    test_schedule_mid_burst()
    test_schedule_early_burst()
    test_weight_snapshot()
    test_eval_accuracy()
    test_burst_dataset()
    print("\nAll tests passed!")
