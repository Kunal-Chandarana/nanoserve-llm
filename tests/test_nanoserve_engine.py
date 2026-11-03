"""
Continuous-batching engine tests.

Scheduling must never change what a request generates. Every test runs the
engine under some pressure (limited sequences, small token budget, tiny KV
cache, requests arriving mid-flight) and checks greedy outputs against
one-request-at-a-time generation, plus the scheduler's own invariants.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from nanoserve.engine import LLMEngine, SchedulerConfig, Status
from nanoserve.generate import generate

from test_nanoserve_model import tiny_pair


@pytest.fixture(scope="module")
def model():
    return tiny_pair(True)[1]


def make_prompts(lengths, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 512, (n,), generator=g).tolist() for n in lengths]


def reference(model, prompts, max_tokens):
    return [list(generate(model, p, max_tokens)) for p in prompts]


def assert_all_blocks_free(engine):
    assert engine.cache.allocator.num_free == engine.config.num_blocks
    assert not engine.cache.seqs


def test_matches_reference_with_ample_resources(model):
    prompts = make_prompts([5, 17, 9, 30, 2])
    engine = LLMEngine(model, SchedulerConfig(num_blocks=64, block_size=4))
    assert engine.generate(prompts, 12) == reference(model, prompts, 12)
    assert_all_blocks_free(engine)


def test_max_num_seqs_queues_requests(model):
    prompts = make_prompts([6] * 7)
    config = SchedulerConfig(max_num_seqs=2, num_blocks=64, block_size=4)
    engine = LLMEngine(model, config)
    assert engine.generate(prompts, 10) == reference(model, prompts, 10)
    assert max(s["num_seqs"] for s in engine.step_log) == 2


def test_chunked_prefill_respects_token_budget(model):
    """Prompts longer than the per-step budget get split across steps."""
    prompts = make_prompts([25, 3, 18])
    config = SchedulerConfig(max_num_batched_tokens=8, num_blocks=64, block_size=4)
    engine = LLMEngine(model, config)
    assert engine.generate(prompts, 8) == reference(model, prompts, 8)
    assert max(s["num_tokens"] for s in engine.step_log) <= 8


def test_decodes_share_steps_with_prefill_chunks(model):
    """While a long prompt is still being chunked in, a running request keeps
    decoding every step instead of stalling."""
    engine = LLMEngine(model, SchedulerConfig(max_num_batched_tokens=6, num_blocks=64, block_size=4))
    short, long = make_prompts([3, 40])
    a = engine.add_request(short, 20)
    engine.step()                      # prefill the short prompt, first token
    b = engine.add_request(long, 4)
    tokens_before = len(engine.requests[a].output_ids)
    for _ in range(5):                 # 40-token prompt needs ~8 chunked steps
        engine.step()
    assert len(engine.requests[a].output_ids) == tokens_before + 5
    assert engine.requests[b].num_computed_tokens > 0
    while engine.has_unfinished_requests():
        engine.step()
    assert engine.requests[a].output_ids == reference(model, [short], 20)[0]
    assert engine.requests[b].output_ids == reference(model, [long], 4)[0]


def test_preemption_under_memory_pressure(model):
    """A cache too small for all requests at once forces preemption; recomputed
    requests must still produce identical output."""
    prompts = make_prompts([10, 12, 9, 11])
    # 4 requests x (~11 prompt + 20 output) tokens need ~32 blocks of 4; give 14.
    config = SchedulerConfig(num_blocks=14, block_size=4, max_num_batched_tokens=64)
    engine = LLMEngine(model, config)
    assert engine.generate(prompts, 20) == reference(model, prompts, 20)
    assert engine.scheduler.num_preemptions > 0
    assert_all_blocks_free(engine)


def test_requests_arriving_mid_flight(model):
    prompts = make_prompts([7, 4, 15, 9, 3, 11])
    engine = LLMEngine(model, SchedulerConfig(num_blocks=64, block_size=4, max_num_seqs=3))
    ids, pending = [], list(prompts)
    step = 0
    while pending or engine.has_unfinished_requests():
        if pending and step % 3 == 0:
            ids.append(engine.add_request(pending.pop(0), 9))
        engine.step()
        step += 1
    assert [engine.requests[i].output_ids for i in ids] == reference(model, prompts, 9)
    # Continuous batching: at some step, a request was prefilled while others decoded.
    assert any(s["prefill_tokens"] and s["decode_seqs"] for s in engine.step_log)


def test_eos_finishes_early_and_frees_blocks(model):
    prompt = make_prompts([8])[0]
    first = reference(model, [prompt], 1)[0][0]
    engine = LLMEngine(model, SchedulerConfig(num_blocks=16, block_size=4))
    rid = engine.add_request(prompt, 50, eos_token_id=first)
    engine.step()
    req = engine.requests[rid]
    assert req.status is Status.FINISHED and req.finish_reason == "stop" and req.output_ids == [first]
    assert_all_blocks_free(engine)


def test_rejects_request_larger_than_cache(model):
    engine = LLMEngine(model, SchedulerConfig(num_blocks=4, block_size=4))
    with pytest.raises(ValueError):
        engine.add_request(make_prompts([10])[0], 10)


def test_seeded_sampling_is_reproducible(model):
    prompts = make_prompts([6, 6])
    runs = []
    for _ in range(2):
        engine = LLMEngine(model, SchedulerConfig(num_blocks=32, block_size=4))
        runs.append(engine.generate(prompts, 10, temperature=1.0, seed=123))
    assert runs[0] == runs[1]


def test_abort_frees_running_and_waiting_requests(model):
    prompts = make_prompts([6, 6, 6])
    engine = LLMEngine(model, SchedulerConfig(max_num_seqs=2, num_blocks=32, block_size=4))
    ids = [engine.add_request(p, 30) for p in prompts]
    engine.step()                               # two running, one waiting
    engine.abort_request(ids[0])                # running
    engine.abort_request(ids[2])                # waiting
    assert engine.requests[ids[0]].finish_reason == "abort"
    assert engine.requests[ids[2]].finish_reason == "abort"
    while engine.has_unfinished_requests():
        engine.step()
    assert engine.requests[ids[1]].output_ids == reference(model, [prompts[1]], 30)[0]
    assert engine.requests[ids[2]].output_ids == []
    assert_all_blocks_free(engine)


def test_mixed_sampling_modes_in_one_batch(model):
    """Greedy, unseeded random and seeded random requests share a batch. Greedy
    rows still match the reference, seeded rows match the same seed run alone,
    and unseeded rows produce valid tokens."""
    greedy, seeded, unseeded = make_prompts([6, 7, 8])
    alone = LLMEngine(model, SchedulerConfig(num_blocks=32, block_size=4))
    expected_seeded = alone.generate([seeded], 10, temperature=0.8, seed=7)[0]

    engine = LLMEngine(model, SchedulerConfig(num_blocks=32, block_size=4))
    g = engine.add_request(greedy, 10)
    s = engine.add_request(seeded, 10, temperature=0.8, seed=7)
    u = engine.add_request(unseeded, 10, temperature=0.8)
    while engine.has_unfinished_requests():
        engine.step()
    assert engine.requests[g].output_ids == reference(model, [greedy], 10)[0]
    assert engine.requests[s].output_ids == expected_seeded
    assert len(engine.requests[u].output_ids) == 10
    assert all(0 <= t < model.config.vocab_size for t in engine.requests[u].output_ids)
