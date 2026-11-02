"""
Paged KV Cache

The idea behind vLLM's PagedAttention, in plain PyTorch.

GPU memory for the KV cache is carved into fixed-size blocks of block_size
tokens. Each sequence owns a block table: a list of block ids, in order. Token
at position p of a sequence lives in block block_table[p // block_size], at
offset p % block_size. Blocks are handed out on demand, one at a time, and
returned the moment a sequence finishes. So:

- No reservation of max_len slots per sequence. Waste is at most one partly
  filled block per sequence, instead of (max_len - actual_len) tokens.
- No fragmentation. Any free block can serve any sequence; a sequence's
  blocks need not be adjacent.
- More sequences fit in the same memory, so the batch can be bigger, which is
  where the throughput comes from.

One forward pass can carry several sequences, each adding any number of
tokens: a whole prompt for a new sequence (prefill) or one token for a running
one (decode). Tokens arrive flattened, in the order the sequences were passed
to begin_step.

Attention is batched: begin_step builds a padded block table and mask for the
whole step, once, and every layer then does one gather and one SDPA call for
all sequences. Decodes (1 new token) and prefills go in separate calls, so a
decode is never padded out to the length of a prompt chunk. vLLM's CUDA
kernel goes further and reads K/V straight out of the scattered blocks
instead of gathering them into a padded tensor first.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from .config import ModelConfig


class OutOfBlocksError(RuntimeError):
    pass


class BlockAllocator:
    """Free list of physical block ids."""

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        # Pop from the end, so blocks are handed out 0, 1, 2, ...
        self._free = list(range(num_blocks - 1, -1, -1))
        self._allocated = set()

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self) -> int:
        if not self._free:
            raise OutOfBlocksError("no free KV cache blocks")
        block = self._free.pop()
        self._allocated.add(block)
        return block

    def free(self, block: int):
        if block not in self._allocated:
            raise ValueError(f"block {block} is not allocated")
        self._allocated.remove(block)
        self._free.append(block)


@dataclass
class SequenceState:
    block_table: List[int] = field(default_factory=list)
    num_tokens: int = 0  # tokens whose K/V are in the cache


@dataclass
class AttentionBatch:
    """Sequences attended in one padded SDPA call, shared by every layer.

    token_idx: rows of the step's flat q/k/v that belong to this batch.
    padded_idx: where each of those rows sits in the padded [B * max_new] layout.
    block_tables: [B, max_blocks], padded with block 0 (masked out).
    mask: [B, 1, max_new * group, max_blocks * block_size], already expanded
      for GQA (see PagedKVCache.forward).
    """
    token_idx: torch.Tensor
    padded_idx: torch.Tensor
    block_tables: torch.Tensor
    mask: torch.Tensor
    max_new: int


class PagedKVCache:
    def __init__(self, config: ModelConfig, num_blocks: int, block_size: int,
                 dtype: torch.dtype, device: str):
        shape = (num_blocks, block_size, config.num_key_value_heads, config.head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(config.num_hidden_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(config.num_hidden_layers)]
        self.block_size = block_size
        self.num_queries_per_kv = config.num_queries_per_kv
        self.device = device
        self.allocator = BlockAllocator(num_blocks)
        self.seqs: Dict[int, SequenceState] = {}
        self._step: List[Tuple[int, int]] = []  # (seq_id, num_new_tokens)
        self._slot_mapping = None
        self._batches: List[AttentionBatch] = []

    # ---- sequence lifecycle -------------------------------------------------

    def add_sequence(self, seq_id: int):
        if seq_id in self.seqs:
            raise ValueError(f"sequence {seq_id} already exists")
        self.seqs[seq_id] = SequenceState()

    def free_sequence(self, seq_id: int):
        for block in self.seqs.pop(seq_id).block_table:
            self.allocator.free(block)

    def blocks_needed(self, seq_id: int, num_new_tokens: int) -> int:
        """Extra blocks a sequence needs to grow by num_new_tokens."""
        seq = self.seqs[seq_id]
        total = -(-(seq.num_tokens + num_new_tokens) // self.block_size)  # ceil division
        return max(0, total - len(seq.block_table))

    def can_append(self, step: Sequence[Tuple[int, int]]) -> bool:
        return sum(self.blocks_needed(s, n) for s, n in step) <= self.allocator.num_free

    # ---- one forward pass ---------------------------------------------------

    def begin_step(self, step: Sequence[Tuple[int, int]]) -> torch.Tensor:
        """Reserve blocks for this pass and return the positions of its tokens.

        step: (seq_id, num_new_tokens) per sequence, in the same order as the
        tokens will be concatenated in input_ids.
        """
        if not self.can_append(step):
            raise OutOfBlocksError(f"need {sum(self.blocks_needed(s, n) for s, n in step)} blocks, "
                                   f"{self.allocator.num_free} free")
        positions, slots = [], []
        for seq_id, n in step:
            seq = self.seqs[seq_id]
            for _ in range(self.blocks_needed(seq_id, n)):
                seq.block_table.append(self.allocator.allocate())
            for pos in range(seq.num_tokens, seq.num_tokens + n):
                block = seq.block_table[pos // self.block_size]
                slots.append(block * self.block_size + pos % self.block_size)
                positions.append(pos)
        self._step = list(step)
        self._slot_mapping = torch.tensor(slots, dtype=torch.long, device=self.device)
        self._batches = self._build_batches(step)
        return torch.tensor(positions, dtype=torch.long, device=self.device)

    def _build_batches(self, step: Sequence[Tuple[int, int]]) -> List[AttentionBatch]:
        starts, offset = [], 0
        for _, n in step:
            starts.append(offset)
            offset += n
        decodes = [i for i, (_, n) in enumerate(step) if n == 1]
        prefills = [i for i, (_, n) in enumerate(step) if n > 1]
        return [self._build_batch([step[i] for i in group], [starts[i] for i in group])
                for group in (decodes, prefills) if group]

    def _build_batch(self, step: Sequence[Tuple[int, int]], starts: Sequence[int]) -> AttentionBatch:
        max_new = max(n for _, n in step)
        max_blocks = max(len(self.seqs[s].block_table) for s, _ in step)
        token_idx, padded_idx, tables, cached, context = [], [], [], [], []
        for row, ((seq_id, n), start) in enumerate(zip(step, starts)):
            seq = self.seqs[seq_id]
            token_idx.extend(range(start, start + n))
            padded_idx.extend(range(row * max_new, row * max_new + n))
            tables.append(seq.block_table + [0] * (max_blocks - len(seq.block_table)))
            cached.append(seq.num_tokens)
            context.append(seq.num_tokens + n)

        as_tensor = lambda x: torch.tensor(x, dtype=torch.long, device=self.device)
        cached, context = as_tensor(cached), as_tensor(context)
        key_pos = torch.arange(max_blocks * self.block_size, device=self.device)
        query_pos = cached[:, None] + torch.arange(max_new, device=self.device)  # [B, max_new]
        # New token i of a sequence sits at position cached + i and may see keys
        # up to there. Keys past the context are padding. A padded query row
        # (i >= n) can still see key 0, so it never softmaxes over nothing; its
        # output is dropped anyway.
        mask = (key_pos <= query_pos[..., None]) & (key_pos < context[:, None, None])
        mask = mask.repeat_interleave(self.num_queries_per_kv, dim=1)[:, None]
        return AttentionBatch(as_tensor(token_idx), as_tensor(padded_idx), as_tensor(tables), mask, max_new)

    def end_step(self):
        for seq_id, n in self._step:
            self.seqs[seq_id].num_tokens += n
        self._step = []
        self._slot_mapping = None
        self._batches = []

    def forward(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        k_cache, v_cache = self.k[layer_idx], self.v[layer_idx]
        # Write this pass's K/V into their slots. Viewing the cache as
        # [num_blocks * block_size, ...] makes a slot id a plain row index.
        k_cache.view(-1, *k_cache.shape[2:])[self._slot_mapping] = k
        v_cache.view(-1, *v_cache.shape[2:])[self._slot_mapping] = v

        out = torch.empty_like(q)
        _, num_heads, head_dim = q.shape
        num_kv_heads, group = k_cache.shape[2], self.num_queries_per_kv
        for b in self._batches:
            num_seqs = b.block_tables.shape[0]
            # One gather for every sequence: [B, max_blocks * block_size, kv_heads, head_dim].
            keys = k_cache[b.block_tables].flatten(1, 2).transpose(1, 2)
            values = v_cache[b.block_tables].flatten(1, 2).transpose(1, 2)

            q_pad = q.new_zeros(num_seqs * b.max_new, num_heads, head_dim)
            q_pad[b.padded_idx] = q[b.token_idx]
            # GQA without copying K/V: query head h reads KV head h // group, so
            # fold each KV head's group of query heads into the query-length
            # dim: [B, max_new, kv_heads * group, d] -> [B, kv_heads, max_new * group, d].
            q_pad = (q_pad.view(num_seqs, b.max_new, num_kv_heads, group, head_dim)
                     .permute(0, 2, 1, 3, 4).reshape(num_seqs, num_kv_heads, b.max_new * group, head_dim))
            o = F.scaled_dot_product_attention(q_pad, keys, values, attn_mask=b.mask)
            o = (o.view(num_seqs, num_kv_heads, b.max_new, group, head_dim)
                 .permute(0, 2, 1, 3, 4).reshape(num_seqs * b.max_new, num_heads, head_dim))
            out[b.token_idx] = o[b.padded_idx]
        return out

    # ---- introspection ------------------------------------------------------

    def memory_stats(self) -> Dict[str, int]:
        used_blocks = self.allocator.num_blocks - self.allocator.num_free
        tokens = sum(s.num_tokens for s in self.seqs.values())
        return {
            "num_blocks": self.allocator.num_blocks,
            "used_blocks": used_blocks,
            "free_blocks": self.allocator.num_free,
            "cached_tokens": tokens,
            # Slots allocated but not yet holding a token (partly filled last blocks).
            "wasted_slots": used_blocks * self.block_size - tokens,
        }


def bytes_per_token(config: ModelConfig, dtype: torch.dtype) -> int:
    """KV cache bytes for one token across all layers: 2 (K and V) x layers x kv_heads x head_dim."""
    elem = torch.tensor([], dtype=dtype).element_size()
    return 2 * config.num_hidden_layers * config.num_key_value_heads * config.head_dim * elem
