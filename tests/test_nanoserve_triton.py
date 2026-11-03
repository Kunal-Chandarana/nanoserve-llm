"""
Triton paged decode attention tests. They need a CUDA GPU with Triton, so
they skip on CPU-only machines (CI, macOS).

The kernel is checked against a plain-PyTorch reference that gathers each
sequence's blocks in order and runs softmax attention, over shuffled physical
blocks, context lengths on and around block boundaries, several GQA group
sizes and strided inputs. Then end to end: greedy generation through the
engine must match the SDPA backend, with and without CUDA graphs.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from nanoserve import triton_attention
from nanoserve.engine import LLMEngine, SchedulerConfig

from test_nanoserve_engine import make_prompts
from test_nanoserve_model import tiny_pair

pytestmark = pytest.mark.skipif(not (triton_attention.HAS_TRITON and torch.cuda.is_available()),
                                reason="needs Triton and a CUDA GPU")


def reference(q, k_cache, v_cache, block_tables, context_lens):
    num_heads, group = q.shape[1], q.shape[1] // k_cache.shape[2]
    out = []
    for i, n in enumerate(context_lens.tolist()):
        blocks = block_tables[i, :-(-n // k_cache.shape[1])]
        k = k_cache[blocks].flatten(0, 1)[:n].float().repeat_interleave(group, dim=1)  # [n, H, D]
        v = v_cache[blocks].flatten(0, 1)[:n].float().repeat_interleave(group, dim=1)
        scores = torch.einsum("hd,nhd->hn", q[i].float(), k) * q.shape[2] ** -0.5
        out.append(torch.einsum("hn,nhd->hd", scores.softmax(-1), v))
    return torch.stack(out).to(q.dtype)


@pytest.mark.parametrize("dtype,atol", [(torch.float32, 1e-4), (torch.bfloat16, 2e-2)])
@pytest.mark.parametrize("num_heads,num_kv_heads,head_dim,block_size", [
    (14, 2, 64, 16),   # Qwen2.5-0.5B: group of 7, padded to 8 in the kernel
    (4, 2, 16, 4),     # the tiny test model
    (8, 8, 32, 8),     # no GQA
    (16, 1, 128, 32),  # one KV head for all
])
def test_kernel_matches_reference(dtype, atol, num_heads, num_kv_heads, head_dim, block_size):
    g = torch.Generator().manual_seed(0)
    context_lens = [1, block_size - 1, block_size, block_size + 1, 3 * block_size + 5, 200]
    max_blocks = -(-max(context_lens) // block_size)
    num_blocks = len(context_lens) * max_blocks + 3
    k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, generator=g).to("cuda", dtype)
    v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim, generator=g).to("cuda", dtype)
    # Each sequence gets distinct, shuffled physical blocks; padding points at block 0.
    perm = torch.randperm(num_blocks, generator=g).tolist()
    tables = torch.zeros(len(context_lens), max_blocks, dtype=torch.long)
    for i, n in enumerate(context_lens):
        need = -(-n // block_size)
        tables[i, :need] = torch.tensor([perm.pop() for _ in range(need)])
    tables, lens = tables.cuda(), torch.tensor(context_lens, device="cuda")
    q = torch.randn(len(context_lens), num_heads, head_dim, generator=g).to("cuda", dtype)

    got = triton_attention.paged_decode_attention(q, k_cache, v_cache, tables, lens)
    torch.testing.assert_close(got, reference(q, k_cache, v_cache, tables, lens), atol=atol, rtol=atol)


def test_kernel_accepts_strided_inputs():
    """The CUDA graph runner passes block tables and context lengths as columns
    of one packed [B, 4 + max_blocks] buffer."""
    g = torch.Generator().manual_seed(1)
    k_cache = torch.randn(20, 16, 2, 64, generator=g).cuda()
    v_cache = torch.randn(20, 16, 2, 64, generator=g).cuda()
    packed = torch.zeros(3, 4 + 6, dtype=torch.long)
    packed[:, 3] = torch.tensor([5, 40, 90])
    packed[:, 4:] = torch.arange(18).view(3, 6)
    packed = packed.cuda()
    q = torch.randn(3, 14, 64, generator=g).cuda()
    got = triton_attention.paged_decode_attention(q, k_cache, v_cache, packed[:, 4:], packed[:, 3])
    expected = reference(q, k_cache, v_cache, packed[:, 4:].contiguous(), packed[:, 3].contiguous())
    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("cuda_graphs", [False, True])
def test_engine_output_matches_sdpa_backend(cuda_graphs):
    model = tiny_pair(True)[1].cuda()
    prompts = make_prompts([5, 17, 9, 30, 2])
    config = SchedulerConfig(num_blocks=64, block_size=4, max_num_seqs=8)
    sdpa = LLMEngine(model, config, cuda_graphs=False, attention_backend="sdpa").generate(prompts, 12)
    engine = LLMEngine(model, config, cuda_graphs=cuda_graphs, attention_backend="triton")
    assert engine.cache.use_triton
    assert engine.generate(prompts, 12) == sdpa


def test_auto_picks_triton_on_cuda_and_sdpa_on_cpu():
    model = tiny_pair(True)[1]
    assert not LLMEngine(model, SchedulerConfig(num_blocks=8, block_size=4)).cache.use_triton
    gpu = LLMEngine(model.cuda(), SchedulerConfig(num_blocks=8, block_size=4), cuda_graphs=False)
    assert gpu.cache.use_triton
