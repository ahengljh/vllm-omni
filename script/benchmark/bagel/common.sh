#!/usr/bin/env bash
set -euo pipefail

ensure_bagel_perf() {
  local current_host
  current_host="$(hostname)"
  if [[ "${current_host}" != "Bagel-perf" ]]; then
    echo "[error] hostname mismatch: expected Bagel-perf, got ${current_host}" >&2
    #exit 1
  fi
}

bagel_benchmark_dir() {
  local script_source
  script_source="${BASH_SOURCE[1]}"
  cd "$(dirname "${script_source}")/.." && pwd
}

bagel_repo_root() {
  local bagel_dir
  bagel_dir="$(bagel_benchmark_dir)"
  cd "${bagel_dir}/../.." && pwd
}

pick_vllm_bin() {
  if command -v vllm-omni >/dev/null 2>&1; then
    echo "vllm-omni"
    return 0
  fi
  if command -v vllm >/dev/null 2>&1; then
    echo "vllm"
    return 0
  fi
  echo "[error] Cannot find vllm-omni or vllm in PATH." >&2
  return 1
}
