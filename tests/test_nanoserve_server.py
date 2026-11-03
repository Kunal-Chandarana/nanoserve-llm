"""
Server tests: a real uvicorn server on a background thread, serving the tiny
random model through a character-level fake tokenizer, driven over HTTP.
"""

import asyncio
import json
import socket
import threading
import time

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
aiohttp = pytest.importorskip("aiohttp")
uvicorn = pytest.importorskip("uvicorn")

from nanoserve.engine import LLMEngine, SchedulerConfig
from nanoserve.generate import generate
from nanoserve.server import Detokenizer, EngineWorker, build_app

from test_nanoserve_model import tiny_pair


class CharTokenizer:
    """Token id = byte value; the tiny model's vocab (512) covers 0-255."""
    eos_token_id = 0

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids, skip_special_tokens=True):
        ids = [i for i in ids if i != 0 and i < 256]
        return bytes(ids).decode("utf-8", errors="replace")

    def apply_chat_template(self, messages, add_generation_prompt=True):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"
        return self.encode(text)


@pytest.fixture(scope="module")
def server():
    model = tiny_pair(True)[1]
    engine = LLMEngine(model, SchedulerConfig(max_num_seqs=4, num_blocks=64, block_size=4))
    worker = EngineWorker(engine)
    worker.start()
    app = build_app(worker, CharTokenizer(), "tiny")

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    while not srv.started:
        time.sleep(0.05)
    yield {"url": f"http://127.0.0.1:{port}", "model": model, "worker": worker}
    srv.should_exit = True
    thread.join(timeout=5)
    worker.stop()


def run(coro):
    return asyncio.run(coro)


async def post(url, path, body):
    async with aiohttp.ClientSession() as s:
        async with s.post(url + path, json=body) as r:
            return r.status, await r.json()


async def post_stream(url, path, body):
    chunks = []
    async with aiohttp.ClientSession() as s:
        async with s.post(url + path, json=body) as r:
            async for raw in r.content:
                line = raw.decode().strip()
                if line.startswith("data:") and line != "data: [DONE]":
                    chunks.append(json.loads(line[5:]))
    return chunks


def expected_ids(model, prompt_ids, n):
    return list(generate(model, prompt_ids, n))


def test_models_and_health(server):
    async def go():
        async with aiohttp.ClientSession() as s:
            async with s.get(server["url"] + "/v1/models") as r:
                assert (await r.json())["data"][0]["id"] == "tiny"
            async with s.get(server["url"] + "/health") as r:
                assert r.status == 200
    run(go())


def test_completion_matches_engine_output(server):
    prompt = [5, 9, 77, 300, 12]
    status, body = run(post(server["url"], "/v1/completions",
                            {"prompt": prompt, "max_tokens": 12, "temperature": 0, "ignore_eos": True}))
    assert status == 200
    assert body["usage"] == {"prompt_tokens": 5, "completion_tokens": 12, "total_tokens": 17}
    assert body["choices"][0]["text"] == CharTokenizer().decode(expected_ids(server["model"], prompt, 12))


def test_streaming_completion_with_usage(server):
    prompt = [3, 1, 4, 1, 5, 9, 2, 6]
    chunks = run(post_stream(server["url"], "/v1/completions",
                             {"prompt": prompt, "max_tokens": 10, "temperature": 0, "ignore_eos": True,
                              "stream": True, "stream_options": {"include_usage": True}}))
    text_chunks = [c for c in chunks if c["choices"]]
    assert len(text_chunks) == 10
    assert text_chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert chunks[-1]["usage"]["completion_tokens"] == 10
    streamed = "".join(c["choices"][0]["text"] for c in text_chunks)
    assert streamed == CharTokenizer().decode(expected_ids(server["model"], prompt, 10))


def test_chat_completion_streaming(server):
    chunks = run(post_stream(server["url"], "/v1/chat/completions",
                             {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 6,
                              "temperature": 0, "ignore_eos": True, "stream": True}))
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    prompt = CharTokenizer().apply_chat_template([{"role": "user", "content": "hi"}])
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert content == CharTokenizer().decode(expected_ids(server["model"], prompt, 6))


def test_concurrent_requests_are_batched_and_correct(server):
    prompts = [[i + 1, i + 2, i + 3] for i in range(8)]

    async def go():
        return await asyncio.gather(*[
            post(server["url"], "/v1/completions",
                 {"prompt": p, "max_tokens": 8, "temperature": 0, "ignore_eos": True}) for p in prompts])
    results = run(go())
    for p, (status, body) in zip(prompts, results):
        assert status == 200
        assert body["choices"][0]["text"] == CharTokenizer().decode(expected_ids(server["model"], p, 8))


def test_oversized_request_is_rejected(server):
    status, body = run(post(server["url"], "/v1/completions",
                            {"prompt": [1, 2, 3], "max_tokens": 10_000, "temperature": 0}))
    assert status == 400 and "KV cache" in body["error"]["message"]


def test_client_disconnect_aborts_request(server):
    aborted_before = server["worker"].num_aborted

    async def go():
        async with aiohttp.ClientSession() as s:
            async with s.post(server["url"] + "/v1/completions",
                              json={"prompt": [1, 2, 3], "max_tokens": 200, "temperature": 0,
                                    "ignore_eos": True, "stream": True}) as r:
                async for _ in r.content:
                    break  # read one chunk, then hang up
    run(go())
    deadline = time.time() + 5
    while time.time() < deadline:
        stats = server["worker"].stats()
        if stats["num_aborted"] > aborted_before and stats["running"] == 0:
            break
        time.sleep(0.05)
    # Cancelled, not merely finished on its own, and its blocks are back.
    assert stats["num_aborted"] == aborted_before + 1
    assert stats["running"] == 0 and stats["used_blocks"] == 0


def test_detokenizer_holds_back_partial_utf8():
    detok = Detokenizer(CharTokenizer())
    euro = "€".encode()  # 3 bytes
    assert detok.add(euro[0]) == ""
    assert detok.add(euro[1]) == ""
    assert detok.add(euro[2]) == "€"


def test_detokenizer_deltas_add_up_to_full_decode():
    """The windowed decode must emit exactly the full decode, split anywhere,
    including multi-byte characters that span tokens."""
    tok = CharTokenizer()
    text = "KV cache → 12 KiB/token, ünïcode 漢字 and emoji 🚀 in one stream."
    ids = tok.encode(text)
    detok = Detokenizer(tok)
    deltas = [detok.add(t, final=(i == len(ids) - 1)) for i, t in enumerate(ids)]
    assert "".join(deltas) == text == detok.text
    assert all("�" not in d for d in deltas)


@pytest.mark.skipif(not __import__("os").environ.get("NANOSERVE_REAL_MODEL"),
                    reason="set NANOSERVE_REAL_MODEL=1 to download a real tokenizer")
def test_detokenizer_matches_full_decode_with_real_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    text = ("Paged attention splits the KV cache into blocks.\n\n  Indented, 中文 text, "
            "emoji 🚀🎉, numbers 3.14159 and code: `x = y[:, None]`.")
    ids = tok.encode(text) + [tok.eos_token_id]
    detok = Detokenizer(tok)
    deltas = [detok.add(t, final=(i == len(ids) - 1)) for i, t in enumerate(ids)]
    assert "".join(deltas) == tok.decode(ids, skip_special_tokens=True)
