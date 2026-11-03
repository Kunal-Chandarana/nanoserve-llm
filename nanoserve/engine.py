"""
Continuous-batching Engine

Requests come and go at any time; every engine step builds a fresh batch from
whatever is running, plus as many waiting requests as fit. Nothing waits for
a batch to finish.

Each step, the scheduler fills a token budget (max_num_batched_tokens) in
this order, modeled on vLLM's V1 scheduler:

1. Running requests first, oldest first. A request still prefilling gets its
   next prompt chunk; a decoding request gets one token. If the KV cache has
   no block for it, the newest running request is preempted: its blocks are
   freed and it goes back to the front of the waiting queue. When it is
   rescheduled it recomputes its KV from prompt + output so far
   ("recompute" preemption; vLLM's default).
2. Then waiting requests, first come first served, while the token budget,
   max_num_seqs and free blocks allow. A prompt longer than the remaining
   budget is split across steps (chunked prefill), so one long prompt can't
   stall every decode in the batch.

A request samples a new token only in a step where its scheduled tokens reach
the end of everything it knows (the last prompt chunk, or a decode step).
"""

import itertools
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

import torch

from .cuda_graphs import DecodeGraphRunner
from .model import Qwen2Model
from .paged_attention import PagedKVCache


class Status(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Request:
    request_id: int
    prompt_ids: List[int]
    max_tokens: int
    temperature: float = 0.0
    eos_token_id: Optional[int] = None
    generator: Optional[torch.Generator] = None
    output_ids: List[int] = field(default_factory=list)
    status: Status = Status.WAITING
    num_computed_tokens: int = 0     # tokens whose KV is in the cache
    finish_reason: Optional[str] = None
    num_preemptions: int = 0
    arrival_time: float = field(default_factory=time.perf_counter)
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None

    @property
    def all_token_ids(self) -> List[int]:
        return self.prompt_ids + self.output_ids

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)


@dataclass
class StepOutput:
    request_id: int
    new_token_id: int
    finished: bool
    finish_reason: Optional[str]


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 2048
    num_blocks: int = 512
    block_size: int = 16


class Scheduler:
    def __init__(self, config: SchedulerConfig, cache: PagedKVCache):
        self.config = config
        self.cache = cache
        self.waiting: List[Request] = []
        self.running: List[Request] = []
        self.num_preemptions = 0

    def add(self, req: Request):
        self.waiting.append(req)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def _preempt(self, req: Request):
        self.running.remove(req)
        self.cache.free_sequence(req.request_id)
        req.status = Status.WAITING
        req.num_computed_tokens = 0
        req.num_preemptions += 1
        self.num_preemptions += 1
        self.waiting.insert(0, req)

    def schedule(self) -> List[Tuple[Request, int]]:
        budget = self.config.max_num_batched_tokens
        scheduled: List[Tuple[Request, int]] = []
        new_blocks = 0  # blocks claimed by requests already scheduled this step

        def fits(req: Request, n: int) -> bool:
            need = self.cache.blocks_needed(req.request_id, n)
            return new_blocks + need <= self.cache.allocator.num_free

        # 1. Running requests, oldest first.
        preempted = False
        i = 0
        while i < len(self.running) and budget > 0:
            req = self.running[i]
            n = min(req.num_tokens - req.num_computed_tokens, budget)
            while not fits(req, n):
                victim = self.running[-1]
                self._preempt(victim)
                preempted = True
                if victim is req:
                    break
            if req.status is not Status.RUNNING:
                break  # preempted itself; everything after it is gone too
            scheduled.append((req, n))
            new_blocks += self.cache.blocks_needed(req.request_id, n)
            budget -= n
            i += 1

        # 2. Waiting requests, FCFS, while they fit. No admissions in a step
        #    that had to preempt: memory is already tight.
        while (not preempted and self.waiting and budget > 0
               and len(self.running) < self.config.max_num_seqs):
            req = self.waiting[0]
            n = min(req.num_tokens - req.num_computed_tokens, budget)
            self.cache.add_sequence(req.request_id)
            if not fits(req, n):
                self.cache.free_sequence(req.request_id)
                break
            self.waiting.pop(0)
            req.status = Status.RUNNING
            self.running.append(req)
            scheduled.append((req, n))
            new_blocks += self.cache.blocks_needed(req.request_id, n)
            budget -= n
        return scheduled

    def finish(self, req: Request, reason: str):
        if req.status is Status.RUNNING:
            self.running.remove(req)
            self.cache.free_sequence(req.request_id)
        elif req.status is Status.WAITING:
            self.waiting.remove(req)
        req.status = Status.FINISHED
        req.finish_reason = reason
        req.finish_time = time.perf_counter()


class LLMEngine:
    def __init__(self, model: Qwen2Model, config: Optional[SchedulerConfig] = None,
                 cuda_graphs: Optional[bool] = None, attention_backend: str = "sdpa"):
        """cuda_graphs: replay decode steps from CUDA graphs. Defaults to on for
        CUDA, like vLLM without --enforce-eager. True on a CPU runs the same
        static-buffer path eagerly, which is how the tests exercise it.
        attention_backend: "sdpa" (default) or "triton"; see PagedKVCache."""
        self.model = model
        self.config = config or SchedulerConfig()
        weight = model.embed_tokens.weight
        self.device = weight.device
        self.cache = PagedKVCache(model.config, self.config.num_blocks, self.config.block_size,
                                  weight.dtype, weight.device, attention_backend)
        self.scheduler = Scheduler(self.config, self.cache)
        if cuda_graphs is None:
            cuda_graphs = self.device.type == "cuda"
        self.graphs = DecodeGraphRunner(model, self.cache, self.config.max_num_seqs) if cuda_graphs else None
        if self.graphs and self.graphs.capture:
            self.graphs.capture_all()
        self.requests: Dict[int, Request] = {}
        self._ids = itertools.count()
        # Per-step batch composition, for analysis and tests. Bounded so a
        # long-running server doesn't grow it forever.
        self.step_log: Deque[Dict] = deque(maxlen=10_000)

    def add_request(self, prompt_ids: List[int], max_tokens: int, temperature: float = 0.0,
                    eos_token_id: Optional[int] = None, seed: Optional[int] = None) -> int:
        capacity = self.config.num_blocks * self.config.block_size
        if len(prompt_ids) + max_tokens > capacity:
            raise ValueError(f"request needs up to {len(prompt_ids) + max_tokens} tokens of KV cache; "
                             f"the whole cache holds {capacity}")
        if not prompt_ids:
            raise ValueError("empty prompt")
        req = Request(next(self._ids), list(prompt_ids), max_tokens, temperature, eos_token_id,
                      torch.Generator().manual_seed(seed) if seed is not None else None)
        self.requests[req.request_id] = req
        self.scheduler.add(req)
        return req.request_id

    def abort_request(self, request_id: int):
        """Stop a request (e.g. its client disconnected) and free its blocks."""
        req = self.requests.get(request_id)
        if req is not None and req.status is not Status.FINISHED:
            self.scheduler.finish(req, "abort")

    def release(self, request_id: int) -> Optional[Request]:
        """Forget a finished request. A server calls this once the result is delivered."""
        return self.requests.pop(request_id, None)

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_work()

    def step(self) -> List[StepOutput]:
        # Wall-clock phases of the step, for serving-loop analysis (step_stats.py).
        # GPU work is asynchronous: "forward" is the time to launch it, and the
        # wait for it lands in "sample", where the token ids come back. A step
        # with nothing to sample (a mid-prompt chunk) doesn't wait, so its GPU
        # time shows up in the next step's "sample".
        t0 = time.perf_counter()
        batch = self.scheduler.schedule()
        if not batch:
            return []
        t1 = time.perf_counter()

        input_ids = []
        for req, n in batch:
            input_ids.extend(req.all_token_ids[req.num_computed_tokens:req.num_computed_tokens + n])
        step = [(req.request_id, n) for req, n in batch]
        t2 = time.perf_counter()
        graphed = bool(self.graphs and self.graphs.supports(step))
        if graphed:
            hidden = self.graphs.run(step, input_ids)
        else:
            positions = self.cache.begin_step(step)
            hidden = self.model(torch.tensor(input_ids, device=self.device), positions, self.cache)
        self.cache.end_step()
        t3 = time.perf_counter()

        # Which requests produce a token this step, and which hidden row each samples from.
        samplers, rows, offset = [], [], 0
        for req, n in batch:
            offset += n
            req.num_computed_tokens += n
            if req.num_computed_tokens == req.num_tokens:
                samplers.append(req)
                rows.append(offset - 1)
        log = {
            "num_seqs": len(batch),
            "num_tokens": len(input_ids),
            "prefill_tokens": sum(n for req, n in batch if n > 1 or not req.output_ids),
            "decode_seqs": sum(1 for req, n in batch if n == 1 and req.output_ids),
            "free_blocks": self.cache.allocator.num_free,
            "graphed": graphed,
            "num_sampled": len(samplers),
            "start": t0,
            "schedule_ms": 1000 * (t1 - t0),
            "prepare_ms": 1000 * (t2 - t1),
            "forward_ms": 1000 * (t3 - t2),
        }
        self.step_log.append(log)
        if not samplers:
            log["sample_ms"] = log["finish_ms"] = 0.0
            return []

        logits = self.model.compute_logits(hidden[torch.tensor(rows, device=self.device)])
        tokens = self._sample_batch(logits, samplers)
        t4 = time.perf_counter()
        log["sample_ms"] = 1000 * (t4 - t3)
        outputs = []
        now = time.perf_counter()
        for req, token in zip(samplers, tokens):
            req.output_ids.append(token)
            if req.first_token_time is None:
                req.first_token_time = now
            reason = None
            if token == req.eos_token_id:
                reason = "stop"
            elif len(req.output_ids) >= req.max_tokens:
                reason = "length"
            if reason:
                self.scheduler.finish(req, reason)
            outputs.append(StepOutput(req.request_id, token, reason is not None, reason))
        log["finish_ms"] = 1000 * (time.perf_counter() - t4)
        return outputs

    def _sample_batch(self, logits: torch.Tensor, reqs: List[Request]) -> List[int]:
        """Sample one token per row, on the logits' device.

        Copying the logits to the CPU to sample there moves [num_seqs, vocab]
        floats every step: 39 MB at batch 64 for Qwen's 151,936-token vocab,
        which was 81% of GPU time in a decode step. Here greedy rows are one
        batched argmax and unseeded random rows one batched multinomial, and
        only the token ids come back. Seeded requests keep their own CPU
        generator so a seed reproduces the same output on any device; only
        their rows are copied.
        """
        tokens = logits.argmax(-1)
        random_rows = [i for i, r in enumerate(reqs) if r.temperature > 0 and r.generator is None]
        if random_rows:
            idx = torch.tensor(random_rows, device=logits.device)
            temps = torch.tensor([reqs[i].temperature for i in random_rows], device=logits.device)
            probs = torch.softmax(logits[idx] / temps[:, None], dim=-1)
            tokens[idx] = torch.multinomial(probs, 1).squeeze(1)
        out = tokens.tolist()
        for i, req in enumerate(reqs):
            if req.temperature > 0 and req.generator is not None:
                out[i] = self._sample(logits[i].cpu(), req)
        return out

    @staticmethod
    def _sample(logits: torch.Tensor, req: Request) -> int:
        if req.temperature <= 0:
            return int(logits.argmax(-1))
        probs = torch.softmax(logits / req.temperature, dim=-1)
        return int(torch.multinomial(probs, 1, generator=req.generator))

    def generate(self, prompts: List[List[int]], max_tokens: int, **kwargs) -> List[List[int]]:
        """Run a fixed set of prompts to completion. Returns output ids per prompt."""
        ids = [self.add_request(p, max_tokens, **kwargs) for p in prompts]
        while self.has_unfinished_requests():
            self.step()
        return [self.requests[i].output_ids for i in ids]
