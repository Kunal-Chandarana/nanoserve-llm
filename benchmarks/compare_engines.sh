#!/usr/bin/env bash
# Head-to-head: vLLM vs nanoserve on the same model, GPU and workload.
#
# Run on a rented NVIDIA GPU machine (an L4 / A10G / A100 / H100 all work):
#   git clone https://github.com/Kunal-Chandarana/nanoserve-llm.git && cd nanoserve-llm
#   bash benchmarks/compare_engines.sh
#
# It installs vLLM, benchmarks `vllm serve`, benchmarks nanoserve with the
# same settings, profiles a nanoserve decode step, and writes everything to
# benchmarks/results/gpu/. Takes ~20-30 minutes. Copy that folder back and
# commit it.
#
# Settings (environment variables):
#   MODEL        model to serve            (default Qwen/Qwen2.5-0.5B-Instruct)
#   LEVELS       concurrency levels        (default 1,4,16,64)
#   INPUT_LEN    prompt tokens             (default 512)
#   OUTPUT_LEN   generated tokens          (default 128)
#   VENV         where vLLM gets installed (default ~/vllm-venv)
#   SKIP_INSTALL=1  use $PYTHON as is (e.g. PYTHON=~/vllm-venv/bin/python on a rerun)
#   SKIP_VLLM=1  only run nanoserve (also how to dry-run this script on a Mac:
#                SKIP_VLLM=1 SKIP_INSTALL=1 DEVICE=cpu DTYPE=float32 LEVELS=1,2 OUTPUT_LEN=8 INPUT_LEN=32)
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
LEVELS=${LEVELS:-1,4,16,64}
INPUT_LEN=${INPUT_LEN:-512}
OUTPUT_LEN=${OUTPUT_LEN:-128}
NUM_REQUESTS=${NUM_REQUESTS:-64}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
DEVICE=${DEVICE:-cuda}
DTYPE=${DTYPE:-bfloat16}
KV_CACHE_GB=${KV_CACHE_GB:-8}
PYTHON=${PYTHON:-python3}
VENV=${VENV:-$HOME/vllm-venv}
OUT=benchmarks/results/gpu
PORT=8000

cd "$(dirname "$0")/.."
mkdir -p "$OUT"
SERVER_PID=""

stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID"; wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap stop_server EXIT

wait_for_server() {
  echo "  waiting for server on :$PORT ..."
  for _ in $(seq 1 600); do
    if curl -sf "localhost:$PORT/health" >/dev/null; then return 0; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "  server exited; see its log"; return 1; fi
    sleep 1
  done
  echo "  server didn't start in 10 minutes"; return 1
}

run_benchmark() {
  local tag=$1
  $PYTHON benchmarks/serving_benchmark.py --base-url "http://localhost:$PORT" \
    --model "$MODEL" --tokenizer "$MODEL" \
    --input-len "$INPUT_LEN" --output-len "$OUTPUT_LEN" \
    --concurrency "$LEVELS" --num-requests "$NUM_REQUESTS" \
    --tag "$tag" --output-dir "$OUT"
}

if [[ "${SKIP_INSTALL:-0}" != 1 ]]; then
  # Into a fresh venv with uv: plain pip on a PyTorch image can backtrack for
  # half an hour trying to fit vLLM around the preinstalled torch, and Debian's
  # system Python refuses pip installs outright (PEP 668).
  echo "== Installing vLLM and benchmark dependencies into $VENV"
  $PYTHON -m pip install -q --break-system-packages uv || $PYTHON -m pip install -q uv
  $PYTHON -m uv venv -q "$VENV"
  $PYTHON -m uv pip install -q --python "$VENV/bin/python" vllm aiohttp matplotlib
  # Prebuilt FlashInfer kernels. Without them vLLM compiles its sampler at
  # startup, which fails on images with no nvcc.
  "$VENV/bin/vllm" download-kernels
  PYTHON="$VENV/bin/python"
fi

if [[ "$DEVICE" == cuda* ]]; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "$OUT/gpu.txt"
fi
echo "model=$MODEL levels=$LEVELS input=$INPUT_LEN output=$OUTPUT_LEN dtype=$DTYPE" | tee -a "$OUT/gpu.txt"

if [[ "${SKIP_VLLM:-0}" != 1 ]]; then
  echo "== vLLM"
  $PYTHON -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT --dtype "$DTYPE" \
    --max-num-seqs "$MAX_NUM_SEQS" --gpu-memory-utilization 0.85 \
    > "$OUT/vllm_server.log" 2>&1 &
  SERVER_PID=$!
  wait_for_server
  run_benchmark vllm
  stop_server
fi

echo "== nanoserve"
$PYTHON -u -m nanoserve.server --model "$MODEL" --port $PORT --device "$DEVICE" --dtype "$DTYPE" \
  --max-num-seqs "$MAX_NUM_SEQS" --kv-cache-gb "$KV_CACHE_GB" \
  > "$OUT/nanoserve_server.log" 2>&1 &
SERVER_PID=$!
wait_for_server
run_benchmark nanoserve
stop_server

echo "== Profiling a nanoserve decode step"
$PYTHON -m nanoserve.profile_step --model "$MODEL" --device "$DEVICE" --dtype "$DTYPE" \
  --batch 16 --prompt-len "$INPUT_LEN" --trace "$OUT/nanoserve_decode_trace.json" \
  2>&1 | grep -v Warning | tee "$OUT/nanoserve_profile.txt"

echo "== Results"
NANO=$(ls -t "$OUT"/nanoserve-*.json | head -1)
if [[ "${SKIP_VLLM:-0}" != 1 ]]; then
  VLLM=$(ls -t "$OUT"/vllm-*.json | head -1)
  $PYTHON benchmarks/plot_results.py "$VLLM" "$NANO" --out "$OUT/vllm_vs_nanoserve.png"
  $PYTHON benchmarks/compare_results.py "$VLLM" "$NANO" | tee "$OUT/comparison.txt"
else
  $PYTHON benchmarks/plot_results.py "$NANO" --out "$OUT/nanoserve.png"
fi
echo
echo "Done. Everything is in $OUT/ -- copy it back and commit it."
