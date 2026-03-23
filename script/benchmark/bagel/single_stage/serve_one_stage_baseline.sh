#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"

ensure_bagel_perf
REPO_ROOT="$(bagel_repo_root)"
cd "${REPO_ROOT}"

MODEL_PATH="${MODEL_PATH:-/home/fq9hpsacuser01/models/BAGEL-7B-MoT}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8091}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
STAGE_CONFIGS_PATH="${STAGE_CONFIGS_PATH:-${SCRIPT_DIR}/bagel_single_stage.yaml}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/../results}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/bagel_one_stage_${PORT}.log}"

mkdir -p "${LOG_DIR}"
VLLM_BIN="$(pick_vllm_bin)"

CMD=(
  "${VLLM_BIN}" serve "${MODEL_PATH}" --omni
  --host "${HOST}"
  --port "${PORT}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --stage-configs-path "${STAGE_CONFIGS_PATH}"
)

echo "[info] REPO_ROOT=${REPO_ROOT}"
echo "[info] MODEL_PATH=${MODEL_PATH}"
echo "[info] HOST=${HOST} PORT=${PORT}"
echo "[info] STAGE_CONFIGS_PATH=${STAGE_CONFIGS_PATH}"
echo "[info] LOG_FILE=${LOG_FILE}"
echo "[info] CMD=${CMD[*]}"

"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
