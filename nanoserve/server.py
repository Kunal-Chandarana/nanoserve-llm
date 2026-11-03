#!/usr/bin/env python3
"""
OpenAI-compatible Server

Serves an LLMEngine over /v1/completions and /v1/chat/completions, streaming
or not, so any OpenAI client -- and benchmarks/serving_benchmark.py -- can
talk to it.

Threading model: the engine runs in one background thread, because a forward
pass blocks for milliseconds to seconds and must not stall the event loop.
HTTP handlers live on the asyncio loop. They hand new requests and aborts to
the engine thread through a thread-safe inbox, and the engine thread pushes
each generated token back to the waiting handler's asyncio.Queue. Between
steps the engine thread drains the inbox, so a request that arrives
mid-generation joins the very next batch.

    python -m nanoserve.server --model Qwen/Qwen2.5-0.5B-Instruct --port 8000
    python benchmarks/serving_benchmark.py --base-url http://localhost:8000 --concurrency 1,4,8
"""

import argparse
import asyncio
import json
import queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .engine import LLMEngine, SchedulerConfig


# ---- engine thread ---------------------------------------------------------

@dataclass
class TokenEvent:
    token_id: Optional[int] = None
    finished: bool = False
    finish_reason: Optional[str] = None
    num_prompt_tokens: int = 0
    error: Optional[str] = None


def _put_all(items: List[tuple]):
    """Runs on the event loop: hand each request its event."""
    for events, event in items:
        events.put_nowait(event)


class EngineWorker:
    """Owns the engine and runs its step loop on a dedicated thread."""

    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self._inbox: "queue.Queue[tuple]" = queue.Queue()
        self._subscribers: Dict[int, tuple] = {}  # request_id -> (loop, asyncio.Queue)
        self._thread = threading.Thread(target=self._run, name="nanoserve-engine", daemon=True)
        self._stopped = threading.Event()
        self.num_aborted = 0  # requests cancelled before finishing (client went away)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stopped.set()
        self._inbox.put(("stop",))
        self._thread.join(timeout=5)

    async def submit(self, prompt_ids: List[int], max_tokens: int, temperature: float,
                     eos_token_id: Optional[int], seed: Optional[int]) -> "tuple[int, asyncio.Queue]":
        loop = asyncio.get_running_loop()
        events: asyncio.Queue = asyncio.Queue()
        accepted: asyncio.Future = loop.create_future()
        self._inbox.put(("add", (prompt_ids, max_tokens, temperature, eos_token_id, seed), loop, events, accepted))
        request_id = await accepted  # raises if the engine rejected the request
        return request_id, events

    def abort(self, request_id: int):
        self._inbox.put(("abort", request_id))

    def _deliver(self, request_id: int, event: TokenEvent):
        sub = self._subscribers.get(request_id)
        if sub:
            loop, events = sub
            loop.call_soon_threadsafe(events.put_nowait, event)

    def _handle(self, cmd: tuple):
        kind = cmd[0]
        if kind == "add":
            _, args, loop, events, accepted = cmd
            try:
                rid = self.engine.add_request(*args)
            except ValueError as e:
                loop.call_soon_threadsafe(accepted.set_exception, e)
                return
            self._subscribers[rid] = (loop, events)
            loop.call_soon_threadsafe(accepted.set_result, rid)
        elif kind == "abort":
            rid = cmd[1]
            req = self.engine.requests.get(rid)
            if req is not None and req.finish_reason is None:
                self.num_aborted += 1
            self.engine.abort_request(rid)
            self.engine.release(rid)
            self._subscribers.pop(rid, None)

    def _run(self):
        while not self._stopped.is_set():
            # Block for work only when idle; otherwise just drain what arrived.
            try:
                if not self.engine.has_unfinished_requests():
                    self._handle_or_stop(self._inbox.get())
                t_inbox = time.perf_counter()
                while True:
                    self._handle_or_stop(self._inbox.get_nowait())
            except queue.Empty:
                pass
            except StopIteration:
                return
            if not self.engine.has_unfinished_requests():
                continue
            t_step = time.perf_counter()
            last_log = self.engine.step_log[-1] if self.engine.step_log else None
            try:
                outputs = self.engine.step()
            except Exception as e:  # fail every in-flight request rather than hang them
                for rid in list(self._subscribers):
                    self._deliver(rid, TokenEvent(finished=True, finish_reason="error", error=repr(e)))
                    self.engine.abort_request(rid)
                    self.engine.release(rid)
                self._subscribers.clear()
                continue
            t_deliver = time.perf_counter()
            # One call_soon_threadsafe per event loop per step, not per token:
            # each call wakes the loop through its self-pipe and takes the GIL,
            # and at concurrency 64 the per-token version cost 1.8 ms a step.
            by_loop: Dict[asyncio.AbstractEventLoop, List[tuple]] = {}
            for out in outputs:
                req = self.engine.requests[out.request_id]
                sub = self._subscribers.get(out.request_id)
                if sub:
                    loop, events = sub
                    by_loop.setdefault(loop, []).append(
                        (events, TokenEvent(out.new_token_id, out.finished, out.finish_reason, len(req.prompt_ids))))
                if out.finished:
                    self._subscribers.pop(out.request_id, None)
                    self.engine.release(out.request_id)
            for loop, items in by_loop.items():
                loop.call_soon_threadsafe(_put_all, items)
            # The engine thread's time around the step, on the step's own log entry
            # (if it logged one: a step with nothing scheduled doesn't).
            if self.engine.step_log and self.engine.step_log[-1] is not last_log:
                log = self.engine.step_log[-1]
                log["inbox_ms"] = 1000 * (t_step - t_inbox)
                log["deliver_ms"] = 1000 * (time.perf_counter() - t_deliver)

    def _handle_or_stop(self, cmd: tuple):
        if cmd[0] == "stop":
            raise StopIteration
        self._handle(cmd)

    def stats(self) -> Dict[str, Any]:
        sched = self.engine.scheduler
        return {
            "running": len(sched.running),
            "waiting": len(sched.waiting),
            "num_preemptions": sched.num_preemptions,
            "num_aborted": self.num_aborted,
            **self.engine.cache.memory_stats(),
        }


# ---- incremental detokenization ---------------------------------------------

class Detokenizer:
    """Turns a growing list of token ids into text deltas.

    Decoding token by token is wrong for multi-byte characters (one emoji can
    span several tokens) and for tokenizers whose spacing depends on the
    previous token. Decoding the whole output every time is right but costs
    O(length) per token: 17% of serving wall time at concurrency 64, on the
    HTTP thread, holding the GIL the engine thread needs.

    So, as vLLM does, decode a short window instead. ids[prefix_offset:]
    starts at a token boundary that has already been turned into text, and
    ids[prefix_offset:read_offset] is what was emitted from that window last
    time. The new text is whatever the window decodes to beyond that. A
    trailing U+FFFD means a multi-byte character is still incomplete: hold it
    back until the next token completes it.
    """

    # Time spent decoding, across all requests, for /debug/stats. It runs on the
    # HTTP thread, which shares the GIL with the engine thread.
    total_ms = 0.0
    num_calls = 0

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ids: List[int] = []
        self.text = ""
        self.prefix_offset = 0  # window start
        self.read_offset = 0    # ids before this are already in self.text

    def add(self, token_id: int, final: bool = False) -> str:
        self.ids.append(token_id)
        return self.flush(final)

    def flush(self, final: bool) -> str:
        start = time.perf_counter()
        decode = self.tokenizer.decode
        prefix_text = decode(self.ids[self.prefix_offset:self.read_offset], skip_special_tokens=True)
        new_text = decode(self.ids[self.prefix_offset:], skip_special_tokens=True)
        Detokenizer.total_ms += 1000 * (time.perf_counter() - start)
        Detokenizer.num_calls += 1
        if not final and (len(new_text) <= len(prefix_text) or new_text.endswith("�")):
            return ""
        delta = new_text[len(prefix_text):]
        self.prefix_offset, self.read_offset = self.read_offset, len(self.ids)
        self.text += delta
        return delta


# ---- HTTP API ------------------------------------------------------------------

class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: Union[str, List[int]]
    max_tokens: int = 16
    temperature: float = 1.0
    seed: Optional[int] = None
    stream: bool = False
    stream_options: Optional[Dict[str, Any]] = None
    ignore_eos: bool = False  # vLLM extension, used by the benchmark


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: int = 256
    temperature: float = 1.0
    seed: Optional[int] = None
    stream: bool = False
    stream_options: Optional[Dict[str, Any]] = None
    ignore_eos: bool = False


def _sse(obj: Dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def build_app(worker: EngineWorker, tokenizer, model_name: str) -> FastAPI:
    app = FastAPI(title="nanoserve")

    @app.on_event("shutdown")
    def _shutdown():
        worker.stop()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics():
        return worker.stats()

    # Serving-loop analysis (benchmarks/step_stats.py): reset, run a load, fetch.
    @app.post("/debug/reset")
    async def debug_reset():
        worker.engine.step_log.clear()
        Detokenizer.total_ms, Detokenizer.num_calls = 0.0, 0
        return {"ok": True}

    @app.get("/debug/stats")
    async def debug_stats():
        # deque.copy() is atomic; iterating while the engine thread appends isn't.
        return {"steps": list(worker.engine.step_log.copy()),
                "detokenize_ms": Detokenizer.total_ms, "detokenize_calls": Detokenizer.num_calls}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "nanoserve"}]}

    async def run(prompt_ids: List[int], body) -> "tuple[int, asyncio.Queue]":
        eos = None if body.ignore_eos else tokenizer.eos_token_id
        return await worker.submit(prompt_ids, body.max_tokens, body.temperature, eos, body.seed)

    async def stream_events(request_id: int, events: asyncio.Queue) -> AsyncIterator[TokenEvent]:
        """Yield token events; abort the request if the consumer goes away early."""
        finished = False
        try:
            while not finished:
                event = await events.get()
                finished = event.finished
                yield event
        finally:
            if not finished:
                worker.abort(request_id)

    def usage(prompt_tokens: int, completion_tokens: int) -> Dict[str, int]:
        return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens}

    def error(status: int, message: str) -> JSONResponse:
        return JSONResponse({"error": {"message": message, "type": "invalid_request_error"}}, status_code=status)

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest, request: Request):
        prompt_ids = body.prompt if isinstance(body.prompt, list) else tokenizer.encode(body.prompt)
        try:
            rid, events = await run(prompt_ids, body)
        except ValueError as e:
            return error(400, str(e))
        cid, created = f"cmpl-{uuid.uuid4().hex[:16]}", int(time.time())

        def chunk(text: str, finish_reason: Optional[str]) -> Dict:
            return {"id": cid, "object": "text_completion", "created": created, "model": model_name,
                    "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}]}

        if body.stream:
            async def gen():
                detok, n_out = Detokenizer(tokenizer), 0
                async for ev in stream_events(rid, events):
                    if ev.error:
                        yield _sse({"error": {"message": ev.error}})
                        break
                    n_out += 1
                    yield _sse(chunk(detok.add(ev.token_id, ev.finished), ev.finish_reason))
                    if ev.finished and (body.stream_options or {}).get("include_usage"):
                        yield _sse({"id": cid, "object": "text_completion", "created": created,
                                    "model": model_name, "choices": [], "usage": usage(len(prompt_ids), n_out)})
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")

        detok, n_out, reason = Detokenizer(tokenizer), 0, None
        async for ev in stream_events(rid, events):
            if ev.error:
                return error(500, ev.error)
            detok.add(ev.token_id, ev.finished)
            n_out, reason = n_out + 1, ev.finish_reason
        out = chunk(detok.text, reason)
        out["usage"] = usage(len(prompt_ids), n_out)
        return out

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, request: Request):
        prompt_ids = tokenizer.apply_chat_template([m.model_dump() for m in body.messages],
                                                   add_generation_prompt=True)
        try:
            rid, events = await run(prompt_ids, body)
        except ValueError as e:
            return error(400, str(e))
        cid, created = f"chatcmpl-{uuid.uuid4().hex[:16]}", int(time.time())

        def chunk(delta: Dict, finish_reason: Optional[str]) -> Dict:
            return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}

        if body.stream:
            async def gen():
                yield _sse(chunk({"role": "assistant", "content": ""}, None))
                detok, n_out = Detokenizer(tokenizer), 0
                async for ev in stream_events(rid, events):
                    if ev.error:
                        yield _sse({"error": {"message": ev.error}})
                        break
                    n_out += 1
                    yield _sse(chunk({"content": detok.add(ev.token_id, ev.finished)}, ev.finish_reason))
                    if ev.finished and (body.stream_options or {}).get("include_usage"):
                        yield _sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                                    "model": model_name, "choices": [], "usage": usage(len(prompt_ids), n_out)})
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")

        detok, n_out, reason = Detokenizer(tokenizer), 0, None
        async for ev in stream_events(rid, events):
            if ev.error:
                return error(500, ev.error)
            detok.add(ev.token_id, ev.finished)
            n_out, reason = n_out + 1, ev.finish_reason
        return {"id": cid, "object": "chat.completion", "created": created, "model": model_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": detok.text},
                             "finish_reason": reason}],
                "usage": usage(len(prompt_ids), n_out)}

    return app


def main():
    import torch
    import uvicorn
    from transformers import AutoTokenizer

    from .loader import download, load_model
    from .paged_attention import bytes_per_token

    parser = argparse.ArgumentParser(description="nanoserve OpenAI-compatible server")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--kv-cache-gb", type=float, default=1.0, help="Memory for the paged KV cache")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Don't use CUDA graphs for decode (on by default on CUDA, as in vLLM)")
    parser.add_argument("--attention-backend", default="sdpa", choices=["sdpa", "triton"],
                        help="Decode attention: padded gather + SDPA, or the Triton paged kernel")
    parser.add_argument("--gil-switch-interval-ms", type=float, default=0.5,
                        help="Python's GIL switch interval (default in Python: 5 ms); see below")
    args = parser.parse_args()

    # The engine thread and the HTTP thread share one GIL. When the engine
    # thread releases it inside a blocking call, the HTTP thread can keep it
    # for up to the switch interval, 5 ms by default, while the engine waits
    # to launch the next step. Under torch 2.14 this cost ~6 ms per decode
    # step at concurrency 64 (3,357 -> 4,523 tok/s with 0.5 ms). vLLM avoids
    # the contention entirely by running the engine in its own process.
    sys.setswitchinterval(args.gil_switch_interval_ms / 1000)

    dtype = getattr(torch, args.dtype)
    model_dir = download(args.model)
    model = load_model(model_dir, dtype, args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    per_block = bytes_per_token(model.config, dtype) * args.block_size
    num_blocks = int(args.kv_cache_gb * 1024 ** 3 // per_block)
    config = SchedulerConfig(max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens,
                             num_blocks=num_blocks, block_size=args.block_size)
    print(f"KV cache: {num_blocks} blocks x {args.block_size} tokens = {num_blocks * args.block_size:,} tokens "
          f"({args.kv_cache_gb} GB)")

    start = time.perf_counter()
    engine = LLMEngine(model, config, cuda_graphs=False if args.enforce_eager else None,
                       attention_backend=args.attention_backend)
    print(f"Decode attention: {'triton' if engine.cache.use_triton else 'sdpa'}")
    if engine.graphs and engine.graphs.capture:
        print(f"Captured {len(engine.graphs.graphs)} decode CUDA graphs in {time.perf_counter() - start:.1f} s")
    worker = EngineWorker(engine)
    worker.start()
    uvicorn.run(build_app(worker, tokenizer, args.model), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
