#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ensure_bagel_perf
REPO_ROOT="$(bagel_repo_root)"
BENCH_DIR="${REPO_ROOT}/benchmarks/diffusion"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"
mkdir -p "${RESULTS_DIR}"

BASE_URL="${BASE_URL:-http://127.0.0.1:8091}"
MODEL="${MODEL:-/home/fq9hpsacuser01/models/BAGEL-7B-MoT}"
BACKEND="${BACKEND:-openai}"
DATASET="${DATASET:-random}"
TASK="${TASK:-t2i}"
NUM_PROMPTS="${NUM_PROMPTS:-5}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"
WARMUP_NUM_INFERENCE_STEPS="${WARMUP_NUM_INFERENCE_STEPS:-10}"
WIDTH="${WIDTH:-1024}"
HEIGHT="${HEIGHT:-1024}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
SEED="${SEED:-42}"
DATASET_PATH="${DATASET_PATH:-}"
TS="$(date -u +%Y%m%d_%H%M%S)"
OUTPUT_FILE="${OUTPUT_FILE:-${RESULTS_DIR}/t2i_official_benchmark_${TS}.json}"

cd "${BENCH_DIR}"
CMD=(
  python diffusion_benchmark_serving.py
  --base-url "${BASE_URL}"
  --model "${MODEL}"
  --backend "${BACKEND}"
  --dataset "${DATASET}"
  --task "${TASK}"
  --num-prompts "${NUM_PROMPTS}"
  --max-concurrency "${MAX_CONCURRENCY}"
  --request-rate "${REQUEST_RATE}"
  --warmup-requests "${WARMUP_REQUESTS}"
  --warmup-num-inference-steps "${WARMUP_NUM_INFERENCE_STEPS}"
  --width "${WIDTH}"
  --height "${HEIGHT}"
  --num-inference-steps "${NUM_INFERENCE_STEPS}"
  --seed "${SEED}"
  --output-file "${OUTPUT_FILE}"
)

if [[ -n "${DATASET_PATH}" ]]; then
  CMD+=(--dataset-path "${DATASET_PATH}")
fi

echo "[info] Official diffusion benchmark wrapper"
echo "[info] BASE_URL=${BASE_URL} MODEL=${MODEL}"
echo "[info] BACKEND=${BACKEND} DATASET=${DATASET} TASK=${TASK}"
echo "[info] NUM_PROMPTS=${NUM_PROMPTS} SIZE=${WIDTH}x${HEIGHT} STEPS=${NUM_INFERENCE_STEPS} SEED=${SEED}"
echo "[info] OUTPUT_FILE=${OUTPUT_FILE}"
echo "[info] CMD=${CMD[*]}"

"${CMD[@]}"
