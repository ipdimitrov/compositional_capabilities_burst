import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import pytest

from burst.config import BurstExperimentConfig, Idea3Config
from burst.data import (
    build_function_pool, tag_tasks, generate_pool,
    generate_documents_for_task, ScheduleSampler, StaggeredSampler,
    BurstDataset,
)
from burst.train import make_model, eval_accuracy, get_space_pos


FAST_CFG = BurstExperimentConfig(
    seed=42,
    ndocuments=500,
    neval_documents=100,
    total_steps=10,
    batch_size=8,
    n_train_compositions=20,
)


@pytest.fixture(scope="module")
def function_pool():
    from synthetic.init import set_seed
    set_seed(42)
    syn, composed, info = build_function_pool(FAST_CFG)
    return syn, composed, info


class TestConfig:
    def test_defaults(self):
        cfg = BurstExperimentConfig()
        assert cfg.n_alphabets == 10
        assert cfg.depth == 5
        assert cfg.p_target == 0.10

    def test_idea3_inherits(self):
        cfg = Idea3Config()
        assert cfg.n_target_tasks == 4
        assert cfg.p_per_task == 0.05
        assert cfg.n_alphabets == 10


class TestDataGeneration:
    def test_build_function_pool(self, function_pool):
        syn, composed, info = function_pool
        assert "train" in composed
        assert "all" in composed
        assert len(composed["train"]) > 0
        assert "train_id" in info

    def test_tag_tasks_single(self, function_pool):
        syn, composed, info = function_pool
        target_ids, bg_ids, fn_lookup = tag_tasks(info, composed, n_target=1)
        assert len(target_ids) == 1
        assert len(bg_ids) >= 1
        assert target_ids[0] not in bg_ids
        assert target_ids[0] in fn_lookup

    def test_tag_tasks_multi(self, function_pool):
        syn, composed, info = function_pool
        target_ids, bg_ids, fn_lookup = tag_tasks(info, composed, n_target=4)
        assert len(target_ids) == 4
        for tid in target_ids:
            assert tid not in bg_ids

    def test_generate_documents(self, function_pool):
        syn, composed, info = function_pool
        target_ids, _, fn_lookup = tag_tasks(info, composed, n_target=1)
        docs = generate_documents_for_task(syn, target_ids[0], fn_lookup, n=10)
        assert docs.shape[0] == 10
        assert docs.ndim == 2
        assert docs.shape[1] > 0

    def test_generate_pool(self, function_pool):
        syn, composed, info = function_pool
        target_ids, bg_ids, fn_lookup = tag_tasks(info, composed, n_target=1)
        pool = generate_pool(syn, target_ids, fn_lookup, n_per_task=5)
        assert len(pool) == 1
        assert pool[target_ids[0]].shape[0] == 5

    def test_burst_dataset(self, function_pool):
        syn, composed, info = function_pool
        target_ids, _, fn_lookup = tag_tasks(info, composed, n_target=1)
        docs = generate_documents_for_task(syn, target_ids[0], fn_lookup, n=10)
        ds = BurstDataset(docs)
        assert len(ds) == 10
        inp, tgt = ds[0]
        assert isinstance(inp, torch.Tensor)
        assert inp.shape[0] == tgt.shape[0]
        assert inp.shape[0] == docs.shape[1] - 1


class TestScheduleSampler:
    @pytest.fixture
    def sampler(self, function_pool):
        syn, composed, info = function_pool
        target_ids, bg_ids, fn_lookup = tag_tasks(info, composed, n_target=1)
        target_pool = generate_pool(syn, target_ids, fn_lookup, 50)
        bg_pool = generate_pool(syn, bg_ids[:5], fn_lookup, 50)
        return ScheduleSampler(target_pool, bg_pool, batch_size=8)

    def test_mixed_returns_correct_shape(self, sampler):
        batch = sampler.sample_batch(step=0, total_steps=100,
                                     schedule="mixed", p=0.5)
        assert batch.shape[0] == 8

    def test_single_burst_no_target_early(self, sampler):
        batch = sampler.sample_batch(step=0, total_steps=100,
                                     schedule="single_burst", p=0.2)
        assert batch.shape[0] == 8

    def test_single_burst_all_target_late(self, sampler):
        batch = sampler.sample_batch(step=99, total_steps=100,
                                     schedule="single_burst", p=0.2)
        assert batch.shape[0] == 8

    def test_multi_burst(self, sampler):
        batch = sampler.sample_batch(step=50, total_steps=100,
                                     schedule="multi_burst", p=0.2, K=5)
        assert batch.shape[0] == 8

    def test_undo_no_target(self, sampler):
        batch = sampler.sample_batch(step=0, total_steps=100,
                                     schedule="undo", p=0.0)
        assert batch.shape[0] == 8


class TestStaggeredSampler:
    def test_sample_batch_shapes(self, function_pool):
        syn, composed, info = function_pool
        train_ids = [tuple(t) for t in info["train_id"]]
        fn_lookup = {}
        for fn_tuple in composed["train"]:
            fn_lookup[tuple(fn_tuple[0])] = fn_tuple

        task_pools = {}
        for i, name in enumerate(["F1_early", "F2_mid", "F3_late", "F4_mixed"]):
            tid = train_ids[i]
            task_pools[name] = generate_pool(syn, [tid], fn_lookup, 20)

        bg_ids = train_ids[4:]
        bg_pool = generate_pool(syn, bg_ids[:3], fn_lookup, 20)

        sampler = StaggeredSampler(task_pools, bg_pool, batch_size=8,
                                   total_steps=100, p_per_task=0.05)

        for step in [0, 25, 50, 75, 99]:
            batch = sampler.sample_batch(step, phase="train")
            assert batch.shape[0] == 8

        undo_batch = sampler.sample_batch(0, phase="undo")
        assert undo_batch.shape[0] == 8


class TestModel:
    def test_forward_pass(self):
        cfg = BurstExperimentConfig()
        net = make_model(cfg, "cpu")
        inp = torch.randint(0, 30, (2, 20))
        out = net(inp)
        assert out.shape == (2, 20, cfg.net.vocab_size)

    def test_backward_pass(self):
        cfg = BurstExperimentConfig()
        net = make_model(cfg, "cpu")
        inp = torch.randint(0, 30, (2, 20))
        tgt = torch.randint(0, 30, (2, 20))
        logits = net(inp)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        loss.backward()
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(not torch.isnan(g).any() for g in grads)

    def test_eval_accuracy(self, function_pool):
        syn, composed, info = function_pool
        target_ids, _, fn_lookup = tag_tasks(info, composed, n_target=1)
        docs = generate_documents_for_task(syn, target_ids[0], fn_lookup, n=20)
        eval_docs = {"test": docs}
        sp = get_space_pos(eval_docs, syn)
        cfg = BurstExperimentConfig()
        net = make_model(cfg, "cpu")
        acc = eval_accuracy(net, docs, sp, "cpu")
        assert 0.0 <= acc <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
