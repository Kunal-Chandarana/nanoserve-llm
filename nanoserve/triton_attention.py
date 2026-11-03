"""
Paged Decode Attention in Triton

The batched SDPA path (step 6) gathers every sequence's KV blocks into a
padded dense tensor before attention: per layer, a copy of the whole batch's
context, padded to a power-of-two bucket. At batch 64 that copy was most of
the 27 ms of GPU time in a decode step. vLLM's kernels never make that copy;
they follow the block table inside the kernel. This one does the same.

One program per (sequence, KV head):

- It loads the group of query heads that share this KV head (GQA), so each
  K/V block is read from memory once for the whole group.
- It walks the sequence's block table, one block of block_size tokens at a
  time, with an online softmax (running max m, running sum l, rescaled
  accumulator), as in FlashAttention. Nothing is materialized but one block.
- It stops at the sequence's context length, so padded block-table entries
  and padded batch rows (context length 1) cost nothing.

Decode is memory-bound (one query per sequence), so scores are a broadcast
multiply-and-sum rather than tl.dot: no tensor cores needed, and no 16-wide
minimum on block size, group size or head dim.

Prefill still uses SDPA: it's compute-bound and the gather matters less.

Measured on an RTX 4090 (step 8): the same cost per decode step as gather +
SDPA at batch 16 and 64 with ~512-token contexts, and slower at batch 1. With
one sequence there are only num_kv_heads programs (2 for Qwen2.5-0.5B), each
walking every block in turn, so most of the GPU sits idle. vLLM's kernels
split long contexts across more programs and merge the partial softmaxes
("split-K" / flash-decoding); that's the missing piece. Until then it's
opt-in (attention_backend="triton").
"""

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:  # CPU-only installs, macOS
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _paged_decode_kernel(
        Q, K, V, Out, BlockTables, ContextLens, scale,
        stride_qb, stride_qh,
        stride_kb, stride_kt, stride_kh,
        stride_vb, stride_vt, stride_vh,
        stride_ob, stride_oh,
        stride_btb, stride_btn, stride_cl,
        GROUP: tl.constexpr, BLOCK_G: tl.constexpr,
        BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr,
    ):
        seq = tl.program_id(0)
        kv_head = tl.program_id(1)
        g = tl.arange(0, BLOCK_G)
        t = tl.arange(0, BLOCK_SIZE)
        d = tl.arange(0, HEAD_DIM)
        g_mask = g < GROUP
        heads = kv_head * GROUP + g

        context_len = tl.load(ContextLens + seq * stride_cl)
        q = tl.load(Q + seq * stride_qb + heads[:, None] * stride_qh + d[None, :],
                    mask=g_mask[:, None], other=0.0).to(tl.float32)        # [G, D]

        m = tl.full([BLOCK_G], float("-inf"), tl.float32)
        l = tl.zeros([BLOCK_G], tl.float32)
        acc = tl.zeros([BLOCK_G, HEAD_DIM], tl.float32)
        for j in range(0, tl.cdiv(context_len, BLOCK_SIZE)):
            block = tl.load(BlockTables + seq * stride_btb + j * stride_btn)
            k = tl.load(K + block * stride_kb + t[:, None] * stride_kt + kv_head * stride_kh
                        + d[None, :]).to(tl.float32)                          # [T, D]
            s = tl.sum(q[:, None, :] * k[None, :, :], 2) * scale             # [G, T]
            s = tl.where((j * BLOCK_SIZE + t)[None, :] < context_len, s, float("-inf"))

            m_new = tl.maximum(m, tl.max(s, 1))
            alpha = tl.exp(m - m_new)          # rescales what's accumulated so far
            p = tl.exp(s - m_new[:, None])
            l = l * alpha + tl.sum(p, 1)
            v = tl.load(V + block * stride_vb + t[:, None] * stride_vt + kv_head * stride_vh
                        + d[None, :]).to(tl.float32)                          # [T, D]
            acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * v[None, :, :], 1)
            m = m_new

        out = acc / l[:, None]
        tl.store(Out + seq * stride_ob + heads[:, None] * stride_oh + d[None, :],
                 out.to(Out.dtype.element_ty), mask=g_mask[:, None])


def _is_pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def supported(k_cache: torch.Tensor) -> bool:
    """Triton is present, the cache is on a GPU, and the shapes are powers of two
    (tl.arange needs them)."""
    _, block_size, _, head_dim = k_cache.shape
    return HAS_TRITON and k_cache.is_cuda and _is_pow2(block_size) and _is_pow2(head_dim)


def paged_decode_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                           block_tables: torch.Tensor, context_lens: torch.Tensor) -> torch.Tensor:
    """Attention for one new token per sequence, reading K/V through the block table.

    q: [B, num_heads, head_dim]  (the new token of each sequence)
    k_cache, v_cache: [num_blocks, block_size, num_kv_heads, head_dim], already
      holding this step's K/V
    block_tables: [B, max_blocks]; context_lens: [B], including the new token.
      Either may be a strided view (the CUDA graph runner passes columns of its
      packed input buffer).
    returns: [B, num_heads, head_dim]
    """
    num_seqs, num_heads, head_dim = q.shape
    _, block_size, num_kv_heads, _ = k_cache.shape
    group = num_heads // num_kv_heads
    assert q.stride(2) == 1 and k_cache.stride(3) == 1 and v_cache.stride(3) == 1
    out = torch.empty_like(q)
    _paged_decode_kernel[(num_seqs, num_kv_heads)](
        q, k_cache, v_cache, out, block_tables, context_lens, head_dim ** -0.5,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        out.stride(0), out.stride(1),
        block_tables.stride(0), block_tables.stride(1), context_lens.stride(0),
        GROUP=group, BLOCK_G=triton.next_power_of_2(group),
        BLOCK_SIZE=block_size, HEAD_DIM=head_dim,
    )
    return out
