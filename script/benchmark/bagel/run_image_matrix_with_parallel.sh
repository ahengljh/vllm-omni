#!/usr/bin/env bash
#Example
# bash run_image_matrix.sh  --mode split --tp-size 2 --cfg-size 1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/benchmark/bagel/run_image_matrix_with_parallel.sh

This wrapper materializes repo-owned Bagel stage configs into /tmp and then
runs the existing matrix script with those derived YAMLs.

Current benchmark contract:
- official benchmark entry: benchmarks/diffusion/diffusion_benchmark_serving.py
- dataset: trace
- recommended backend on current main: vllm-omni

Common overrides:
  MATRIX_MODE=baseline|split|both
  TP_SIZE=1
  CFG_PARALLEL_SIZE=1
  RUN_LABEL=my_case
  OUTPUT_ROOT=/tmp/bagel_matrix_my_case

Mode-specific overrides:
  BASELINE_TP_SIZE=1
  BASELINE_CFG_PARALLEL_SIZE=1
  SPLIT_DIT_TP_SIZE=1
  SPLIT_DIT_CFG_PARALLEL_SIZE=1
  SPLIT_AR_TP_SIZE=1

GPU placement overrides:
  BASELINE_FIRST_GPU=0
  SPLIT_AR_FIRST_GPU=0
  SPLIT_DIT_FIRST_GPU=1

Source YAML overrides:
  BASELINE_SOURCE_STAGE_CONFIGS_PATH=scripts/benchmark/bagel/single_stage/bagel_single_stage.yaml
  SPLIT_SOURCE_STAGE_CONFIGS_PATH=vllm_omni/model_executor/stage_configs/bagel.yaml

Direct stage-config override still works and bypasses auto-generation:
  BASELINE_STAGE_CONFIGS_PATH=/tmp/already_materialized_baseline.yaml
  SPLIT_STAGE_CONFIGS_PATH=/tmp/already_materialized_split.yaml

Dry-run example:
  MATRIX_MODE=split TP_SIZE=2 CFG_PARALLEL_SIZE=1 DRY_RUN=1 \
  bash scripts/benchmark/bagel/run_image_matrix_with_parallel.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

info() {
  printf '[info] %s\n' "$*"
}

die() {
  printf '[error] %s\n' "$*" >&2
  exit 1
}

ensure_bagel_perf
REPO_ROOT="$(bagel_repo_root)"
cd "${REPO_ROOT}"

BRANCH="$(git branch --show-current)"
[[ "${BRANCH}" == "main" ]] || die "branch mismatch: expected main, got ${BRANCH}"

MATRIX_MODE="${MATRIX_MODE:-both}"
BACKEND="${BACKEND:-vllm-omni}"
RUN_LABEL="${RUN_LABEL:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/bagel_matrix_${RUN_LABEL}}"

TP_SIZE="${TP_SIZE:-1}"
CFG_PARALLEL_SIZE="${CFG_PARALLEL_SIZE:-1}"
BASELINE_TP_SIZE="${BASELINE_TP_SIZE:-${TP_SIZE}}"
BASELINE_CFG_PARALLEL_SIZE="${BASELINE_CFG_PARALLEL_SIZE:-${CFG_PARALLEL_SIZE}}"
SPLIT_DIT_TP_SIZE="${SPLIT_DIT_TP_SIZE:-${TP_SIZE}}"
SPLIT_DIT_CFG_PARALLEL_SIZE="${SPLIT_DIT_CFG_PARALLEL_SIZE:-${CFG_PARALLEL_SIZE}}"
SPLIT_AR_TP_SIZE="${SPLIT_AR_TP_SIZE:-1}"

BASELINE_FIRST_GPU="${BASELINE_FIRST_GPU:-0}"
SPLIT_AR_FIRST_GPU="${SPLIT_AR_FIRST_GPU:-0}"
SPLIT_DIT_FIRST_GPU="${SPLIT_DIT_FIRST_GPU:-1}"

BASELINE_SOURCE_STAGE_CONFIGS_PATH="${BASELINE_SOURCE_STAGE_CONFIGS_PATH:-${SCRIPT_DIR}/single_stage/bagel_single_stage.yaml}"
SPLIT_SOURCE_STAGE_CONFIGS_PATH="${SPLIT_SOURCE_STAGE_CONFIGS_PATH:-${REPO_ROOT}/vllm_omni/model_executor/stage_configs/bagel.yaml}"

BASELINE_STAGE_CONFIGS_PATH="${BASELINE_STAGE_CONFIGS_PATH:-}"
SPLIT_STAGE_CONFIGS_PATH="${SPLIT_STAGE_CONFIGS_PATH:-}"

export REPO_ROOT OUTPUT_ROOT MATRIX_MODE BACKEND
export BASELINE_SOURCE_STAGE_CONFIGS_PATH SPLIT_SOURCE_STAGE_CONFIGS_PATH
export BASELINE_STAGE_CONFIGS_PATH SPLIT_STAGE_CONFIGS_PATH
export BASELINE_TP_SIZE BASELINE_CFG_PARALLEL_SIZE SPLIT_DIT_TP_SIZE SPLIT_DIT_CFG_PARALLEL_SIZE SPLIT_AR_TP_SIZE
export BASELINE_FIRST_GPU SPLIT_AR_FIRST_GPU SPLIT_DIT_FIRST_GPU

mkdir -p "${OUTPUT_ROOT}/configs"

python3 - <<'PY'
import os
from pathlib import Path
import yaml


def device_csv(start: int, count: int) -> str:
    if count <= 0:
        raise ValueError(f"device count must be > 0, got {count}")
    return ",".join(str(start + i) for i in range(count))


def ensure_int(name: str) -> int:
    value = int(os.environ[name])
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


matrix_mode = os.environ["MATRIX_MODE"]
output_root = Path(os.environ["OUTPUT_ROOT"])
config_dir = output_root / "configs"
config_dir.mkdir(parents=True, exist_ok=True)

baseline_tp = ensure_int("BASELINE_TP_SIZE")
baseline_cfg = ensure_int("BASELINE_CFG_PARALLEL_SIZE")
split_dit_tp = ensure_int("SPLIT_DIT_TP_SIZE")
split_dit_cfg = ensure_int("SPLIT_DIT_CFG_PARALLEL_SIZE")
split_ar_tp = ensure_int("SPLIT_AR_TP_SIZE")

baseline_first_gpu = int(os.environ["BASELINE_FIRST_GPU"])
split_ar_first_gpu = int(os.environ["SPLIT_AR_FIRST_GPU"])
split_dit_first_gpu = int(os.environ["SPLIT_DIT_FIRST_GPU"])

if baseline_cfg not in (1, 2, 3):
    raise ValueError(f"BASELINE_CFG_PARALLEL_SIZE must be 1, 2, or 3, got {baseline_cfg}")
if split_dit_cfg not in (1, 2, 3):
    raise ValueError(f"SPLIT_DIT_CFG_PARALLEL_SIZE must be 1, 2, or 3, got {split_dit_cfg}")


def load_yaml(path_str: str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"missing source yaml: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f), path


def dump_yaml(obj, path: Path):
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


baseline_stage_override = os.environ.get("BASELINE_STAGE_CONFIGS_PATH", "").strip()
split_stage_override = os.environ.get("SPLIT_STAGE_CONFIGS_PATH", "").strip()

if matrix_mode in ("baseline", "both") and not baseline_stage_override:
    baseline_cfg_obj, baseline_src = load_yaml(os.environ["BASELINE_SOURCE_STAGE_CONFIGS_PATH"])
    stage = baseline_cfg_obj["stage_args"][0]
    if stage.get("stage_type") != "diffusion":
        raise ValueError(f"baseline source is not a single diffusion stage: {baseline_src}")

    total_devices = baseline_tp * baseline_cfg
    stage["runtime"]["devices"] = device_csv(baseline_first_gpu, total_devices)

    engine_args = stage.setdefault("engine_args", {})
    engine_args["num_gpus"] = total_devices
    if total_devices == 1:
        engine_args.pop("parallel_config", None)
        engine_args["tensor_parallel_size"] = 1
    else:
        engine_args.pop("tensor_parallel_size", None)
        engine_args["parallel_config"] = {
            "tensor_parallel_size": baseline_tp,
            "cfg_parallel_size": baseline_cfg,
        }

    out_path = config_dir / f"baseline_tp{baseline_tp}_cfg{baseline_cfg}.yaml"
    dump_yaml(baseline_cfg_obj, out_path)
    os.environ["BASELINE_STAGE_CONFIGS_PATH"] = str(out_path)

if matrix_mode in ("split", "both") and not split_stage_override:
    split_cfg_obj, split_src = load_yaml(os.environ["SPLIT_SOURCE_STAGE_CONFIGS_PATH"])
    stage0 = split_cfg_obj["stage_args"][0]
    stage1 = split_cfg_obj["stage_args"][1]
    if stage0.get("stage_type") != "llm" or stage1.get("stage_type") != "diffusion":
        raise ValueError(f"split source does not look like Bagel AR->DiT: {split_src}")

    stage0["runtime"]["devices"] = device_csv(split_ar_first_gpu, split_ar_tp)
    stage0_engine_args = stage0.setdefault("engine_args", {})
    stage0_engine_args["tensor_parallel_size"] = split_ar_tp

    dit_total_devices = split_dit_tp * split_dit_cfg
    stage1["runtime"]["devices"] = device_csv(split_dit_first_gpu, dit_total_devices)
    stage1_engine_args = stage1.setdefault("engine_args", {})
    stage1_engine_args.pop("tensor_parallel_size", None)
    stage1_engine_args["parallel_config"] = {
        "tensor_parallel_size": split_dit_tp,
        "cfg_parallel_size": split_dit_cfg,
    }

    out_path = config_dir / f"split_ar_tp{split_ar_tp}_dit_tp{split_dit_tp}_cfg{split_dit_cfg}.yaml"
    dump_yaml(split_cfg_obj, out_path)
    os.environ["SPLIT_STAGE_CONFIGS_PATH"] = str(out_path)

print(f"[info] dataset=trace")
print(f"[info] backend={os.environ['BACKEND']}")
print(f"[info] baseline_stage_configs_path={os.environ.get('BASELINE_STAGE_CONFIGS_PATH', '')}")
print(f"[info] split_stage_configs_path={os.environ.get('SPLIT_STAGE_CONFIGS_PATH', '')}")
print(
    "[info] cfg_parallel caveat: current official benchmark trace contract does not forward "
    "negative_prompt or true_cfg_scale, so cfg_parallel_size>1 changes the stage config but will not "
    "fully exercise CFG parallel until those request fields are added."
)
PY

if [[ -z "${BASELINE_STAGE_CONFIGS_PATH}" && ( "${MATRIX_MODE}" == "baseline" || "${MATRIX_MODE}" == "both" ) ]]; then
  BASELINE_STAGE_CONFIGS_PATH="${OUTPUT_ROOT}/configs/baseline_tp${BASELINE_TP_SIZE}_cfg${BASELINE_CFG_PARALLEL_SIZE}.yaml"
fi
if [[ -z "${SPLIT_STAGE_CONFIGS_PATH}" && ( "${MATRIX_MODE}" == "split" || "${MATRIX_MODE}" == "both" ) ]]; then
  SPLIT_STAGE_CONFIGS_PATH="${OUTPUT_ROOT}/configs/split_ar_tp${SPLIT_AR_TP_SIZE}_dit_tp${SPLIT_DIT_TP_SIZE}_cfg${SPLIT_DIT_CFG_PARALLEL_SIZE}.yaml"
fi

export BACKEND OUTPUT_ROOT BASELINE_STAGE_CONFIGS_PATH SPLIT_STAGE_CONFIGS_PATH

if [[ "${BACKEND}" != "vllm-omni" ]]; then
  info "warning: current main uses backend=vllm-omni as the Bagel-ready official path; got BACKEND=${BACKEND}"
fi

info "repo-owned baseline source=${BASELINE_SOURCE_STAGE_CONFIGS_PATH}"
info "repo-owned split source=${SPLIT_SOURCE_STAGE_CONFIGS_PATH}"
if [[ "${MATRIX_MODE}" == "baseline" || "${MATRIX_MODE}" == "both" || -n "${BASELINE_STAGE_CONFIGS_PATH}" ]]; then
  info "derived baseline yaml=${BASELINE_STAGE_CONFIGS_PATH}"
fi
if [[ "${MATRIX_MODE}" == "split" || "${MATRIX_MODE}" == "both" || -n "${SPLIT_STAGE_CONFIGS_PATH}" ]]; then
  info "derived split yaml=${SPLIT_STAGE_CONFIGS_PATH}"
fi
info "dataset=trace"

exec bash "${SCRIPT_DIR}/run_image_matrix.sh" "$@"
