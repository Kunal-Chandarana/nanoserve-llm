"""
CUDA graph runner tests. On a CPU the runner can't capture, but it runs the
exact static-buffer path a graph replays: padded batch, packed inputs, scratch
slots, bucketed block tables. Outputs must match one-request-at-a-time
generation. The capture itself is checked on CUDA only.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from nanoserve.cuda_graphs import DecodeGraphRunner, batch_buckets, block_buckets
from nanoserve.engine import LLMEngine, SchedulerConfig

from test_nanoserve_engine import assert_all_blocks_free, make_prompts, reference
from test_nanoserve_model import tiny_pair


@pytest.fixture(scope="module")
def model():
    return tiny_pair(True)[1]


def count_graph_steps(engine):
    calls = []
    run = engine.graphs.run
    engine.graphs.run = lambda step, ids: calls.append(len(step)) or run(step, ids)
    return calls


def test_buckets():
    assert batch_buckets(64) == [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]
    assert batch_buckets(20) == [1, 2, 4, 8, 16, 20]
    assert batch_buckets(3) == [1, 2, 3]
    assert batch_buckets(1) == [1]
    assert block_buckets(256) == [2, 4, 8, 16, 32, 64, 128, 256]
    assert block_buckets(40) == [2, 4, 8, 16, 32, 40]


def test_static_path_matches_reference(model):
    """5 decoding sequences padded to a batch of 8, with contexts that grow
    across block-count buckets (block_size 4: the longest crosses 8 -> 16)."""
    prompts = make_prompts([5, 17, 9, 30, 2])
    engine = LLMEngine(model, SchedulerConfig(num_blocks=64, block_size=4), cuda_graphs=True)
    assert not engine.graphs.capture  # CPU: same path, run eagerly
    calls = count_graph_steps(engine)
    assert engine.generate(prompts, 12) == reference(model, prompts, 12)
    assert calls and set(calls) == {5}
    assert set(engine.graphs.graphs) == {(8, 8), (8, 16)}
    assert_all_blocks_free(engine)


def test_mixed_steps_fall_back_to_eager(model):
    """Requests arriving mid-flight give a mix of prefill steps (eager) and
    pure decode steps (graph); outputs must not depend on which ran."""
    prompts = make_prompts([7, 4, 15, 9, 3, 11])
    engine = LLMEngine(model, SchedulerConfig(num_blocks=64, block_size=4, max_num_seqs=3),
                       cuda_graphs=True)
    calls = count_graph_steps(engine)
    ids, pending, step = [], list(prompts), 0
    while pending or engine.has_unfinished_requests():
        if pending and step % 3 == 0:
            ids.append(engine.add_request(pending.pop(0), 9))
        engine.step()
        step += 1
    assert [engine.requests[i].output_ids for i in ids] == reference(model, prompts, 9)
    assert 0 < len(calls) < step
    # Steps mixing a new prompt's prefill with running decodes split: the
    # decodes still go through the graph path, the prefill runs eagerly.
    assert any(s["split"] for s in engine.step_log)


def test_preemption_with_graphs(model):
    prompts = make_prompts([10, 12, 9, 11])
    config = SchedulerConfig(num_blocks=14, block_size=4, max_num_batched_tokens=64)
    engine = LLMEngine(model, config, cuda_graphs=True)
    assert engine.generate(prompts, 20) == reference(model, prompts, 20)
    assert engine.scheduler.num_preemptions > 0
    assert_all_blocks_free(engine)


def test_long_context_is_not_graphed(model):
    engine = LLMEngine(model, SchedulerConfig(num_blocks=64, block_size=4))
    runner = DecodeGraphRunner(model, engine.cache, max_num_seqs=4, max_context=16, capture=False)
    engine.cache.add_sequence(0)
    engine.cache.reserve([(0, 15)]); engine.cache.end_step()
    assert runner.supports([(0, 1)])        # token 16 still fits in 4 blocks of 4
    engine.cache.reserve([(0, 1)]); engine.cache.end_step()
    assert not runner.supports([(0, 1)])    # token 17 would need a 5th block
    assert not runner.supports([(0, 2)])    # not decode-shaped


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_captured_graphs_match_eager_on_cuda():
    gpu_model = tiny_pair(True)[1].cuda()
    prompts = make_prompts([5, 17, 9, 30, 2])
    config = SchedulerConfig(num_blocks=64, block_size=4, max_num_seqs=8)
    eager = LLMEngine(gpu_model, config, cuda_graphs=False).generate(prompts, 12)
    graphed = LLMEngine(gpu_model, config, cuda_graphs=True)
    assert graphed.graphs.capture
    calls = count_graph_steps(graphed)
    assert graphed.generate(prompts, 12) == eager
    assert calls
