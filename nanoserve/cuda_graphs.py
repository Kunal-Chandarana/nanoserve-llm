"""
CUDA Graphs for Decode

A decode step on a small model is ~1,500 tiny kernels, each launched from
Python. On a GPU the launches cost far more than the math: step 6 measured a
33.5 ms step with only 5.7 ms of GPU work. A CUDA graph records the whole
step's kernels once and replays them with a single launch.

A graph replays fixed shapes at fixed memory addresses, so, as in vLLM:

- Only decode-shaped steps are graphed: every sequence adds exactly one token.
  Prefill steps (variable shapes, big matmuls that keep the GPU busy anyway)
  run eagerly.
- The batch is padded up to a captured size (1, 2, 4, 8, 16, 24, ...), and the
  block table up to a power-of-two number of blocks. One graph per pair.
- Every input lives in one static buffer per graph. A step packs its token
  ids, positions, KV slots, context lengths and block tables into one host
  tensor and copies it in: one host-to-device copy, then replay.
- Padded rows write their K/V to the cache's scratch block and attend to one
  key, so they can't corrupt a real sequence or produce NaNs. Their outputs
  are dropped.

With capture=False the same static-buffer path runs eagerly. That's how the
tests check it on a CPU, where CUDA graphs don't exist.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .model import Qwen2Model
from .paged_attention import AttentionBatch, PagedKVCache

# Packed input columns; block table entries follow.
TOKEN, POSITION, SLOT, CONTEXT_LEN, BLOCKS = range(5)


def batch_buckets(max_num_seqs: int) -> List[int]:
    """1, 2, 4, 8, then every multiple of 8 -- vLLM's capture sizes."""
    sizes = [b for b in (1, 2, 4) if b < max_num_seqs] + list(range(8, max_num_seqs + 8, 8))
    return [b for b in sizes if b < max_num_seqs] + [max_num_seqs]


def block_buckets(max_blocks: int) -> List[int]:
    sizes, b = [], 2
    while b < max_blocks:
        sizes.append(b)
        b *= 2
    return sizes + [max_blocks]


@dataclass
class _Graph:
    packed: torch.Tensor          # [batch, BLOCKS + num_blocks] int64, the only input
    hidden: torch.Tensor          # [batch, hidden_size], the output
    run: Callable[[], None]       # replay (or, without capture, run eagerly)


class DecodeGraphRunner:
    def __init__(self, model: Qwen2Model, cache: PagedKVCache, max_num_seqs: int,
                 max_context: int = 4096, capture: Optional[bool] = None):
        self.model = model
        self.cache = cache
        self.device = model.embed_tokens.weight.device
        self.capture = self.device.type == "cuda" if capture is None else capture
        self.batch_sizes = batch_buckets(max_num_seqs)
        self.block_counts = block_buckets(-(-max_context // cache.block_size))
        self.graphs: Dict[Tuple[int, int], _Graph] = {}
        self._pool = None  # all graphs share one memory pool; only one replays at a time

    def supports(self, step: Sequence[Tuple[int, int]]) -> bool:
        if len(step) > self.batch_sizes[-1] or any(n != 1 for _, n in step):
            return False
        # Blocks each sequence will have once this step's token is added.
        bs = self.cache.block_size
        return max(self.cache.seqs[s].num_tokens // bs + 1 for s, _ in step) <= self.block_counts[-1]

    def capture_all(self):
        """Capture every (batch, blocks) graph up front, largest first, so the
        shared pool is sized by the biggest one and serving never pauses to capture."""
        for b in reversed(self.batch_sizes):
            for nb in reversed(self.block_counts):
                self._get(b, nb)

    def run(self, step: Sequence[Tuple[int, int]], input_ids: List[int]) -> torch.Tensor:
        """Run one decode step. Returns hidden states [len(step), hidden_size]."""
        cache, bs = self.cache, self.cache.block_size
        cache.reserve(step)
        rows = []
        for (seq_id, _), token in zip(step, input_ids):
            seq = cache.seqs[seq_id]
            pos = seq.num_tokens
            slot = seq.block_table[pos // bs] * bs + pos % bs
            rows.append([token, pos, slot, pos + 1] + seq.block_table)
        batch = next(b for b in self.batch_sizes if b >= len(rows))
        num_blocks = next(nb for nb in self.block_counts if nb >= max(len(r) - BLOCKS for r in rows))
        graph = self._get(batch, num_blocks)

        width, scratch = BLOCKS + num_blocks, cache.scratch_block
        padded = [r + [scratch] * (width - len(r)) for r in rows]
        padded += [self._pad_row() + [scratch] * num_blocks] * (batch - len(rows))
        graph.packed.copy_(torch.tensor(padded))
        graph.run()
        return graph.hidden[:len(rows)]

    def _pad_row(self) -> List[int]:
        scratch_slot = self.cache.scratch_block * self.cache.block_size
        return [0, 0, scratch_slot, 1]  # token 0 at position 0, sees only itself

    def _get(self, batch: int, num_blocks: int) -> _Graph:
        key = (batch, num_blocks)
        if key not in self.graphs:
            self.graphs[key] = self._build(batch, num_blocks)
        return self.graphs[key]

    def _build(self, batch: int, num_blocks: int) -> _Graph:
        cache, dev = self.cache, self.device
        packed = torch.full((batch, BLOCKS + num_blocks), cache.scratch_block, dtype=torch.long, device=dev)
        packed[:, :BLOCKS] = torch.tensor(self._pad_row(), device=dev)
        key_pos = torch.arange(num_blocks * cache.block_size, device=dev)
        mask = torch.empty(batch, key_pos.numel(), dtype=torch.bool, device=dev)
        rows = torch.arange(batch, device=dev)
        # Decode: one query per sequence, so the mask is just "key is inside
        # the context". [B, 1, 1, ctx] broadcasts over heads and the GQA group.
        attn = AttentionBatch(token_idx=rows, padded_idx=rows, block_tables=packed[:, BLOCKS:],
                              mask=mask[:, None, None, :], max_new=1)
        hidden = torch.empty(batch, self.model.config.hidden_size,
                             dtype=self.model.embed_tokens.weight.dtype, device=dev)

        def forward():
            torch.lt(key_pos, packed[:, CONTEXT_LEN, None], out=mask)
            cache.bind(packed[:, SLOT], [attn])
            hidden.copy_(self.model(packed[:, TOKEN], packed[:, POSITION], cache))
            cache.bind(None, [])

        if not self.capture:
            return _Graph(packed, hidden, forward)

        # Warm up on a side stream first (cuBLAS/SDPA pick kernels and allocate
        # workspaces on first use, which can't happen during capture), then record.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            forward()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            forward()
        self._pool = graph.pool()
        return _Graph(packed, hidden, graph.replay)
