#!/usr/bin/env python3
"""
Serving-Loop Breakdown for nanoserve

profile_step.py times one decode step in isolation. Under real load the engine
also schedules, prefills new requests, streams tokens and detokenizes, and its
thread shares the GIL with the HTTP server. This script shows where the wall
time of a serving run actually goes, from the per-step log the server keeps.

    curl -X POST localhost:8000/debug/reset            # start clean
    python benchmarks/serving_benchmark.py ... --concurrency 64
    python benchmarks/step_stats.py --url http://localhost:8000 [--save stats.json]
    python benchmarks/step_stats.py --file stats.json   # re-analyze saved stats

Steps fall into four kinds:
- graphed: pure decode, replayed from a CUDA graph
- decode:  pure decode, run eagerly (no graphs, e.g. CPU or --enforce-eager)
- mixed:   decodes plus prefill chunks, run eagerly
- prefill: prefill only, run eagerly

"gap" is wall time between the end of one step's bookkeeping and the start of
the next step: the engine thread waiting for the GIL, or idle.
"""

import argparse
import json
import urllib.request
from collections import defaultdict
from typing import Dict, List

PHASES = ["inbox_ms", "schedule_ms", "prepare_ms", "forward_ms", "sample_ms", "finish_ms", "deliver_ms"]


KINDS = ("graphed", "decode", "mixed", "prefill")


def kind(step: Dict) -> str:
    if step.get("graphed"):
        return "graphed"
    if not step.get("prefill_tokens"):
        return "decode"
    return "mixed" if step.get("decode_seqs") else "prefill"


def summarize(stats: Dict) -> str:
    steps: List[Dict] = sorted(stats["steps"], key=lambda s: s["start"])
    if len(steps) < 2:
        return "not enough steps logged"
    busy = [sum(s.get(p, 0.0) for p in PHASES) for s in steps]
    # Each step's span runs from its inbox drain to the end of delivery; the
    # wall time covers the first step's start to the last step's end.
    starts = [s["start"] * 1000 - s.get("inbox_ms", 0.0) for s in steps]
    wall_ms = starts[-1] + busy[-1] - starts[0]
    gaps = [max(0.0, starts[i + 1] - (starts[i] + busy[i])) for i in range(len(steps) - 1)]
    sampled = sum(s.get("num_sampled", 0) for s in steps)

    lines = [f"{len(steps)} steps over {wall_ms / 1000:.2f} s: {len(steps) / wall_ms * 1000:.0f} steps/s, "
             f"{sampled / wall_ms * 1000:.0f} tokens/s sampled, mean {wall_ms / len(steps):.2f} ms per step", ""]

    by_kind = defaultdict(list)
    for s, b in zip(steps, busy):
        by_kind[kind(s)].append((s, b))
    header = f"{'kind':<9}{'steps':>7}{'share':>7}{'tokens':>8}" + "".join(f"{p[:-3]:>10}" for p in PHASES) + f"{'total':>9}"
    lines += ["Mean ms per step, by kind (tokens = tokens in the step):", header]
    for k in KINDS:
        rows = by_kind.get(k)
        if not rows:
            continue
        n = len(rows)
        line = f"{k:<9}{n:>7}{100 * n / len(steps):>6.0f}%{sum(s['num_tokens'] for s, _ in rows) / n:>8.0f}"
        line += "".join(f"{sum(s.get(p, 0.0) for s, _ in rows) / n:>10.2f}" for p in PHASES)
        lines.append(line + f"{sum(b for _, b in rows) / n:>9.2f}")

    lines += ["", "Where the wall time went:"]
    total_by_phase = {p: sum(s.get(p, 0.0) for s in steps) for p in PHASES}
    for p in PHASES:
        lines.append(f"  {p[:-3]:<10}{total_by_phase[p]:>10.0f} ms  {100 * total_by_phase[p] / wall_ms:>5.1f}%")
    gap = sum(gaps)
    lines.append(f"  {'gap':<10}{gap:>10.0f} ms  {100 * gap / wall_ms:>5.1f}%   (between steps: GIL or idle)")
    for k in KINDS:
        rows = by_kind.get(k, [])
        lines.append(f"  {k + ' steps':<16}{sum(b for _, b in rows):>10.0f} ms  "
                     f"{100 * sum(b for _, b in rows) / wall_ms:>5.1f}% of wall")

    calls = stats.get("detokenize_calls", 0)
    if calls:
        lines += ["", f"Detokenize (HTTP thread): {stats['detokenize_ms']:.0f} ms over {calls} calls, "
                      f"{stats['detokenize_ms'] / calls * 1000:.0f} us per call, "
                      f"{100 * stats['detokenize_ms'] / wall_ms:.1f}% of wall"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Break down a nanoserve serving run by step phase.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="nanoserve server, e.g. http://localhost:8000")
    src.add_argument("--file", help="stats JSON saved earlier with --save")
    parser.add_argument("--save", help="also write the raw stats JSON here")
    args = parser.parse_args()

    if args.url:
        with urllib.request.urlopen(args.url.rstrip("/") + "/debug/stats") as r:
            stats = json.load(r)
    else:
        with open(args.file) as f:
            stats = json.load(f)
    if args.save:
        with open(args.save, "w") as f:
            json.dump(stats, f)
    print(summarize(stats))


if __name__ == "__main__":
    main()
