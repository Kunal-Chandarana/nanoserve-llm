#!/usr/bin/env bash
# Where does nanoserve's serving time go? Starts the nanoserve server, and for
# each concurrency level: clears the step log, runs the serving benchmark, and
# saves the per-step breakdown (benchmarks/step_stats.py).
#
#   bash benchmarks/serving_breakdown.sh            # on a GPU, after compare_engines.sh installed deps
#   LEVELS=16,64 SERVER_ARGS="--enforce-eager" OUT=benchmarks/results/eager bash benchmarks/serving_breakdown.sh
#
# Settings (environment variables): MODEL, LEVELS (default 16,64), INPUT_LEN,
# OUTPUT_LEN, MAX_NUM_SEQS, DTYPE, DEVICE, KV_CACHE_GB, PYTHON, OUT, SERVER_ARGS.
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
LEVELS=${LEVELS:-16,64}
INPUT_LEN=${INPUT_LEN:-512}
OUTPUT_LEN=${OUTPUT_LEN:-128}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
DEVICE=${DEVICE:-cuda}
DTYPE=${DTYPE:-bfloat16}
KV_CACHE_GB=${KV_CACHE_GB:-8}
PYTHON=${PYTHON:-python3}
OUT=${OUT:-benchmarks/results/serving-breakdown}
SERVER_ARGS=${SERVER_ARGS:-}
PORT=8000
URL="http://localhost:$PORT"

cd "$(dirname "$0")/.."
mkdir -p "$OUT"

# shellcheck disable=SC2086
$PYTHON -u -m nanoserve.server --model "$MODEL" --port $PORT --device "$DEVICE" --dtype "$DTYPE" \
  --max-num-seqs "$MAX_NUM_SEQS" --kv-cache-gb "$KV_CACHE_GB" $SERVER_ARGS > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 600); do
  curl -sf "$URL/health" >/dev/null && break
  kill -0 $SERVER_PID 2>/dev/null || { echo "server exited; see $OUT/server.log"; exit 1; }
  sleep 1
done

for level in ${LEVELS//,/ }; do
  echo "== concurrency $level"
  curl -sf -X POST "$URL/debug/reset" >/dev/null
  $PYTHON benchmarks/serving_benchmark.py --base-url "$URL" --model "$MODEL" --tokenizer "$MODEL" \
    --input-len "$INPUT_LEN" --output-len "$OUTPUT_LEN" --concurrency "$level" --num-requests $((4 * level)) \
    --tag "breakdown-c$level" --output-dir "$OUT" | grep -E "^conc="
  $PYTHON benchmarks/step_stats.py --url "$URL" --save "$OUT/steps-c$level.json" | tee "$OUT/breakdown-c$level.txt"
done
