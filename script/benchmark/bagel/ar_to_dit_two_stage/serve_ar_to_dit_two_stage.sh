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
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
STAGE_CONFIGS_PATH="${STAGE_CONFIGS_PATH:-${SCRIPT_DIR}/bagel_ar_to_dit_two_stage.yaml}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/../results}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/bagel_ar_to_dit_two_stage_${PORT}.log}"

if [[ ! -f "${STAGE_CONFIGS_PATH}" ]]; then
  echo "[error] Missing split stage config: ${STAGE_CONFIGS_PATH}" >&2
  echo "[error] Materialize the AR->DiT two-stage yaml before using this wrapper." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
VLLM_BIN="$(pick_vllm_bin)"

CMD=(
  "${VLLM_BIN}" serve "${MODEL_PATH}" --omni
  --host "${HOST}"
  --port "${PORT}"
  --max-model-len "${MAX_MODEL_LEN}"
  --stage-configs-path "${STAGE_CONFIGS_PATH}"
)

if [[ -n "${GPU_MEMORY_UTILIZATION}" ]]; then
  CMD+=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
fi

echo "[info] REPO_ROOT=${REPO_ROOT}"
echo "[info] MODEL_PATH=${MODEL_PATH}"
echo "[info] HOST=${HOST} PORT=${PORT}"
echo "[info] GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-<stage-config-defaults>}"
echo "[info] STAGE_CONFIGS_PATH=${STAGE_CONFIGS_PATH}"
echo "[info] LOG_FILE=${LOG_FILE}"
echo "[info] CMD=${CMD[*]}"

"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
