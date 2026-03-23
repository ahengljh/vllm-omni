#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/benchmark/bagel/run_image_matrix.sh [options]

This script runs the Bagel image benchmark matrix inside the Bagel-perf
container. It automates:

- hostname / branch / repo checks
- process cleanup with plain ps -ef and explicit kill -9
- service launch and health polling
- trace contract generation
- official diffusion benchmark execution
- markdown matrix summary generation

Default behavior reproduces the matched 10-step image matrix:

- mode: both (baseline then split)
- tasks: t2i,i2i
- steps: 10
- concurrency: 1,2,4
- backend: vllm-omni
- dataset: trace
- output root: /tmp/<timestamp>/bagel_matrix_<mode>_tp<tp>_cfg<cfg>

Useful CLI options:

  --mode baseline|split|both
  --tasks t2i,i2i
  --steps 10,50
  --concurrency 1,2,4
  --num-requests 8
  --run-label 20260319_180000
  --output-root /tmp/custom_case

  --tp-size 1
  --cfg-size 1
  --baseline-tp-size 1
  --baseline-cfg-size 1
  --split-ar-tp-size 1
  --split-dit-tp-size 1
  --split-dit-cfg-size 1

  --baseline-first-gpu 0
  --split-ar-first-gpu 0
  --split-dit-first-gpu 1

  --compare-baseline-dir /tmp/bagel_matrix_prev/baseline
  --split-stage-configs-path /tmp/already_materialized_split.yaml
  --dry-run
  --keep-last-service

Examples:

  bash scripts/benchmark/bagel/run_image_matrix.sh     --mode baseline --tp-size 2 --cfg-size 1

  bash scripts/benchmark/bagel/run_image_matrix.sh     --mode split --tp-size 2 --cfg-size 1     --compare-baseline-dir /tmp/bagel_matrix_baseline_ref/baseline

Compatibility note:

- legacy environment variables still work, but CLI flags are now the primary
  entrypoint.
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

trim_csv() {
  local value="$1"
  value="${value// /}"
  printf '%s' "$value"
}

require_cli_value() {
  local opt="$1"
  local argc="$2"
  (( argc >= 2 )) || die "missing value for ${opt}"
}

parse_cli_args() {
  while (( $# > 0 )); do
    case "$1" in
      --mode)
        require_cli_value "$1" "$#"
        MATRIX_MODE="$2"
        shift 2
        ;;
      --tasks)
        require_cli_value "$1" "$#"
        TASKS_CSV="$2"
        shift 2
        ;;
      --steps)
        require_cli_value "$1" "$#"
        STEPS_CSV="$2"
        shift 2
        ;;
      --concurrency)
        require_cli_value "$1" "$#"
        CONCURRENCY_CSV="$2"
        shift 2
        ;;
      --num-requests)
        require_cli_value "$1" "$#"
        NUM_REQUESTS="$2"
        shift 2
        ;;
      --warmup-requests)
        require_cli_value "$1" "$#"
        WARMUP_REQUESTS="$2"
        shift 2
        ;;
      --warmup-steps)
        require_cli_value "$1" "$#"
        WARMUP_STEPS="$2"
        shift 2
        ;;
      --backend)
        require_cli_value "$1" "$#"
        BACKEND="$2"
        shift 2
        ;;
      --port)
        require_cli_value "$1" "$#"
        PORT="$2"
        shift 2
        ;;
      --run-label)
        require_cli_value "$1" "$#"
        RUN_LABEL="$2"
        shift 2
        ;;
      --output-root)
        require_cli_value "$1" "$#"
        OUTPUT_ROOT="$2"
        shift 2
        ;;
      --tp-size)
        require_cli_value "$1" "$#"
        TP_SIZE="$2"
        shift 2
        ;;
      --cfg-size)
        require_cli_value "$1" "$#"
        CFG_PARALLEL_SIZE="$2"
        shift 2
        ;;
      --baseline-tp-size)
        require_cli_value "$1" "$#"
        BASELINE_TP_SIZE="$2"
        shift 2
        ;;
      --baseline-cfg-size)
        require_cli_value "$1" "$#"
        BASELINE_CFG_PARALLEL_SIZE="$2"
        shift 2
        ;;
      --split-ar-tp-size)
        require_cli_value "$1" "$#"
        SPLIT_AR_TP_SIZE="$2"
        shift 2
        ;;
      --split-dit-tp-size)
        require_cli_value "$1" "$#"
        SPLIT_DIT_TP_SIZE="$2"
        shift 2
        ;;
      --split-dit-cfg-size)
        require_cli_value "$1" "$#"
        SPLIT_DIT_CFG_PARALLEL_SIZE="$2"
        shift 2
        ;;
      --baseline-first-gpu)
        require_cli_value "$1" "$#"
        BASELINE_FIRST_GPU="$2"
        shift 2
        ;;
      --split-ar-first-gpu)
        require_cli_value "$1" "$#"
        SPLIT_AR_FIRST_GPU="$2"
        shift 2
        ;;
      --split-dit-first-gpu)
        require_cli_value "$1" "$#"
        SPLIT_DIT_FIRST_GPU="$2"
        shift 2
        ;;
      --compare-baseline-dir)
        require_cli_value "$1" "$#"
        COMPARE_BASELINE_DIR="$2"
        shift 2
        ;;
      --baseline-stage-configs-path)
        require_cli_value "$1" "$#"
        BASELINE_STAGE_CONFIGS_PATH="$2"
        shift 2
        ;;
      --split-stage-configs-path)
        require_cli_value "$1" "$#"
        SPLIT_STAGE_CONFIGS_PATH="$2"
        shift 2
        ;;
      --dry-run)
        DRY_RUN=1
        shift
        ;;
      --keep-last-service)
        KEEP_LAST_SERVICE=1
        shift
        ;;
      --help|-h)
        usage
        exit 0
        ;;
      *)
        die "unknown argument: $1"
        ;;
    esac
  done
}

ensure_bagel_perf
REPO_ROOT="$(bagel_repo_root)"
cd "${REPO_ROOT}"

BRANCH="$(git branch --show-current)"
[[ "${BRANCH}" == "main" ]] || die "branch mismatch: expected main, got ${BRANCH}"

MATRIX_MODE="${MATRIX_MODE:-both}"
TASKS_CSV="${TASKS_CSV:-t2i,i2i}"
STEPS_CSV="${STEPS_CSV:-10}"
CONCURRENCY_CSV="${CONCURRENCY_CSV:-1,2,4}"
NUM_REQUESTS="${NUM_REQUESTS:-8}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
PORT="${PORT:-8091}"
BACKEND="${BACKEND:-vllm-omni}"
MODEL_PATH="${MODEL_PATH:-/home/fq9hpsacuser01/models/BAGEL-7B-MoT}"
WIDTH="${WIDTH:-1024}"
HEIGHT="${HEIGHT:-1024}"
SEED="${SEED:-42}"
RUN_LABEL="${RUN_LABEL:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
HEALTH_TIMEOUT_SEC="${HEALTH_TIMEOUT_SEC:-240}"
HEALTH_POLL_SEC="${HEALTH_POLL_SEC:-2}"
KEEP_LAST_SERVICE="${KEEP_LAST_SERVICE:-0}"
DRY_RUN="${DRY_RUN:-0}"

T2I_PROMPT="${T2I_PROMPT:-A cute cat wearing sunglasses}"
I2I_PROMPT="${I2I_PROMPT:-Transform this photo into a soft watercolor illustration while preserving the original composition, natural lighting, fur details, and face. Keep balanced exposure and realistic contrast.}"
I2I_IMAGE_PATH="${I2I_IMAGE_PATH:-${REPO_ROOT}/scripts/bagel/bagel_test.png}"

BASELINE_SERVE_SCRIPT="${BASELINE_SERVE_SCRIPT:-${SCRIPT_DIR}/single_stage/serve_one_stage_baseline.sh}"
BASELINE_SOURCE_STAGE_CONFIGS_PATH="${BASELINE_SOURCE_STAGE_CONFIGS_PATH:-${SCRIPT_DIR}/single_stage/bagel_single_stage.yaml}"
BASELINE_STAGE_CONFIGS_PATH="${BASELINE_STAGE_CONFIGS_PATH:-}"
BASELINE_MAX_MODEL_LEN="${BASELINE_MAX_MODEL_LEN:-8192}"
BASELINE_GPU_MEMORY_UTILIZATION="${BASELINE_GPU_MEMORY_UTILIZATION:-0.95}"
BASELINE_EXTRA_EXPORTS="${BASELINE_EXTRA_EXPORTS:-}"

SPLIT_SERVE_SCRIPT="${SPLIT_SERVE_SCRIPT:-${SCRIPT_DIR}/ar_to_dit_two_stage/serve_ar_to_dit_two_stage.sh}"
SPLIT_SOURCE_STAGE_CONFIGS_PATH="${SPLIT_SOURCE_STAGE_CONFIGS_PATH:-${REPO_ROOT}/vllm_omni/model_executor/stage_configs/bagel.yaml}"
SPLIT_STAGE_CONFIGS_PATH="${SPLIT_STAGE_CONFIGS_PATH:-}"
SPLIT_MAX_MODEL_LEN="${SPLIT_MAX_MODEL_LEN:-12288}"
SPLIT_GPU_MEMORY_UTILIZATION="${SPLIT_GPU_MEMORY_UTILIZATION:-0.95}"
SPLIT_EXTRA_EXPORTS="${SPLIT_EXTRA_EXPORTS:-FLASHINFER_DISABLE_VERSION_CHECK=1}"

TP_SIZE="${TP_SIZE:-1}"
CFG_PARALLEL_SIZE="${CFG_PARALLEL_SIZE:-1}"
BASELINE_TP_SIZE="${BASELINE_TP_SIZE:-}"
BASELINE_CFG_PARALLEL_SIZE="${BASELINE_CFG_PARALLEL_SIZE:-}"
SPLIT_DIT_TP_SIZE="${SPLIT_DIT_TP_SIZE:-}"
SPLIT_DIT_CFG_PARALLEL_SIZE="${SPLIT_DIT_CFG_PARALLEL_SIZE:-}"
SPLIT_AR_TP_SIZE="${SPLIT_AR_TP_SIZE:-}"

BASELINE_FIRST_GPU="${BASELINE_FIRST_GPU:-0}"
SPLIT_AR_FIRST_GPU="${SPLIT_AR_FIRST_GPU:-0}"
SPLIT_DIT_FIRST_GPU="${SPLIT_DIT_FIRST_GPU:-1}"

COMPARE_BASELINE_DIR="${COMPARE_BASELINE_DIR:-}"

parse_cli_args "$@"
TASKS_CSV="$(trim_csv "${TASKS_CSV}")"
STEPS_CSV="$(trim_csv "${STEPS_CSV}")"
CONCURRENCY_CSV="$(trim_csv "${CONCURRENCY_CSV}")"
BASELINE_TP_SIZE="${BASELINE_TP_SIZE:-${TP_SIZE}}"
BASELINE_CFG_PARALLEL_SIZE="${BASELINE_CFG_PARALLEL_SIZE:-${CFG_PARALLEL_SIZE}}"
SPLIT_DIT_TP_SIZE="${SPLIT_DIT_TP_SIZE:-${TP_SIZE}}"
SPLIT_DIT_CFG_PARALLEL_SIZE="${SPLIT_DIT_CFG_PARALLEL_SIZE:-${CFG_PARALLEL_SIZE}}"
SPLIT_AR_TP_SIZE="${SPLIT_AR_TP_SIZE:-1}"
if [[ -z "${OUTPUT_ROOT}" ]]; then
  OUTPUT_ROOT="/tmp/${RUN_LABEL}/bagel_matrix_${MATRIX_MODE}_tp${TP_SIZE}_cfg${CFG_PARALLEL_SIZE}"
fi

case "${MATRIX_MODE}" in
  baseline|split|both) ;;
  *) die "MATRIX_MODE must be one of: baseline, split, both" ;;
esac

IFS=',' read -r -a TASKS <<< "${TASKS_CSV}"
IFS=',' read -r -a STEPS <<< "${STEPS_CSV}"
IFS=',' read -r -a CONCURRENCIES <<< "${CONCURRENCY_CSV}"

show_config() {
  info "hostname=$(hostname)"
  info "repo=${REPO_ROOT}"
  info "branch=${BRANCH}"
  info "commit=$(git rev-parse HEAD)"
  info "mode=${MATRIX_MODE}"
  info "tasks=${TASKS_CSV}"
  info "steps=${STEPS_CSV}"
  info "concurrency=${CONCURRENCY_CSV}"
  info "output_root=${OUTPUT_ROOT}"
  info "backend=${BACKEND}"
  info "dataset=trace"
  info "baseline_source_stage_configs_path=${BASELINE_SOURCE_STAGE_CONFIGS_PATH}"
  info "split_source_stage_configs_path=${SPLIT_SOURCE_STAGE_CONFIGS_PATH}"
  if [[ -n "${BASELINE_STAGE_CONFIGS_PATH}" ]]; then
    info "baseline_stage_configs_path=${BASELINE_STAGE_CONFIGS_PATH}"
  fi
  if [[ -n "${SPLIT_STAGE_CONFIGS_PATH}" ]]; then
    info "split_stage_configs_path=${SPLIT_STAGE_CONFIGS_PATH}"
  fi
  info "tp_size=${TP_SIZE} cfg_parallel_size=${CFG_PARALLEL_SIZE}"
  info "baseline_parallel=tp${BASELINE_TP_SIZE},cfg${BASELINE_CFG_PARALLEL_SIZE},first_gpu=${BASELINE_FIRST_GPU}"
  info "split_parallel=ar_tp${SPLIT_AR_TP_SIZE},dit_tp${SPLIT_DIT_TP_SIZE},dit_cfg${SPLIT_DIT_CFG_PARALLEL_SIZE},ar_first_gpu=${SPLIT_AR_FIRST_GPU},dit_first_gpu=${SPLIT_DIT_FIRST_GPU}"
  info "port=${PORT}"
}

cleanup_processes() {
  info "process cleanup: ps -ef"
  ps -ef

  mapfile -t pids < <(
    ps -ef | awk '
      /benchmarks\/diffusion\/diffusion_benchmark_serving\.py/ {print $2}
      /serve_one_stage_baseline\.sh/ {print $2}
      /serve_ar_to_dit_two_stage\.sh/ {print $2}
      /\/usr\/local\/bin\/vllm-omni serve .*BAGEL-7B-MoT/ {print $2}
      /\/usr\/local\/bin\/vllm serve .*BAGEL-7B-MoT/ {print $2}
      /multiprocessing\.resource_tracker/ {print $2}
      /multiprocessing\.spawn/ {print $2}
      /tee .*bagel_(one_stage|ar_to_dit_two_stage|matrix)/ {print $2}
    ' | awk '!seen[$1]++'
  )

  if (( ${#pids[@]} > 0 )); then
    info "kill -9 ${pids[*]}"
    kill -9 "${pids[@]}" || true
  else
    info "nothing to kill"
  fi

  sleep 2
  info "process cleanup: ps -ef after kill"
  ps -ef
}

wait_for_health() {
  local name="$1"
  local deadline=$(( $(date +%s) + HEALTH_TIMEOUT_SEC ))

  while (( $(date +%s) < deadline )); do
    local code
    code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" || true)"
    if [[ "${code}" == "200" ]]; then
      info "${name} health=200"
      return 0
    fi
    sleep "${HEALTH_POLL_SEC}"
  done

  info "${name} launcher log tail"
  tail -n 40 "${OUTPUT_ROOT}/logs/${name}_launcher.out" || true
  die "${name} failed health check on port ${PORT}"
}

generate_contracts() {
  local contract_dir="${OUTPUT_ROOT}/contracts"
  mkdir -p "${contract_dir}"

  export CONTRACT_DIR="${contract_dir}"
  export STEPS_CSV NUM_REQUESTS T2I_PROMPT I2I_PROMPT I2I_IMAGE_PATH WIDTH HEIGHT SEED

  python3 - <<'PY'
import os

contract_dir = os.environ["CONTRACT_DIR"]
steps = [int(x) for x in os.environ["STEPS_CSV"].split(",") if x]
num_requests = int(os.environ["NUM_REQUESTS"])
width = int(os.environ["WIDTH"])
height = int(os.environ["HEIGHT"])
seed = int(os.environ["SEED"])
t2i_prompt = os.environ["T2I_PROMPT"]
i2i_prompt = os.environ["I2I_PROMPT"]
i2i_image_path = os.environ["I2I_IMAGE_PATH"]

os.makedirs(contract_dir, exist_ok=True)

for step in steps:
    t2i_path = os.path.join(contract_dir, f"t2i_{step}_{num_requests}req.trace")
    with open(t2i_path, "w", encoding="utf-8") as f:
        for _ in range(num_requests):
            f.write(
                "Request("
                f"prompt={t2i_prompt!r}, "
                f"width={width}, height={height}, "
                f"num_inference_steps={step}, seed={seed}"
                ")\n"
            )

    i2i_path = os.path.join(contract_dir, f"i2i_{step}_{num_requests}req.trace")
    with open(i2i_path, "w", encoding="utf-8") as f:
        for _ in range(num_requests):
            f.write(
                "Request("
                f"prompt={i2i_prompt!r}, "
                f"width={width}, height={height}, "
                f"num_inference_steps={step}, seed={seed}, "
                f"image_paths={[i2i_image_path]!r}"
                ")\n"
            )
PY
}

materialize_stage_configs() {
  export OUTPUT_ROOT MATRIX_MODE
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
PY

  if [[ -z "${BASELINE_STAGE_CONFIGS_PATH}" && ( "${MATRIX_MODE}" == "baseline" || "${MATRIX_MODE}" == "both" ) ]]; then
    BASELINE_STAGE_CONFIGS_PATH="${OUTPUT_ROOT}/configs/baseline_tp${BASELINE_TP_SIZE}_cfg${BASELINE_CFG_PARALLEL_SIZE}.yaml"
  fi
  if [[ -z "${SPLIT_STAGE_CONFIGS_PATH}" && ( "${MATRIX_MODE}" == "split" || "${MATRIX_MODE}" == "both" ) ]]; then
    SPLIT_STAGE_CONFIGS_PATH="${OUTPUT_ROOT}/configs/split_ar_tp${SPLIT_AR_TP_SIZE}_dit_tp${SPLIT_DIT_TP_SIZE}_cfg${SPLIT_DIT_CFG_PARALLEL_SIZE}.yaml"
  fi

  if (( BASELINE_CFG_PARALLEL_SIZE > 1 || SPLIT_DIT_CFG_PARALLEL_SIZE > 1 )); then
    info "cfg_parallel caveat: current official trace benchmark does not forward negative_prompt or true_cfg_scale; cfg_parallel_size>1 changes the stage config but will not fully exercise CFG parallel until those request fields are added"
  fi
}

launch_service() {
  local mode="$1"
  local serve_script
  local stage_configs_path
  local max_model_len
  local gpu_memory_utilization
  local extra_exports

  case "${mode}" in
    baseline)
      serve_script="${BASELINE_SERVE_SCRIPT}"
      stage_configs_path="${BASELINE_STAGE_CONFIGS_PATH}"
      max_model_len="${BASELINE_MAX_MODEL_LEN}"
      gpu_memory_utilization="${BASELINE_GPU_MEMORY_UTILIZATION}"
      extra_exports="${BASELINE_EXTRA_EXPORTS}"
      ;;
    split)
      serve_script="${SPLIT_SERVE_SCRIPT}"
      stage_configs_path="${SPLIT_STAGE_CONFIGS_PATH}"
      max_model_len="${SPLIT_MAX_MODEL_LEN}"
      gpu_memory_utilization="${SPLIT_GPU_MEMORY_UTILIZATION}"
      extra_exports="${SPLIT_EXTRA_EXPORTS}"
      ;;
    *)
      die "unknown mode for launch_service: ${mode}"
      ;;
  esac

  [[ -f "${serve_script}" ]] || die "missing serve script: ${serve_script}"
  [[ -f "${stage_configs_path}" ]] || die "missing stage configs path: ${stage_configs_path}"

  cleanup_processes

  local service_log_dir="${OUTPUT_ROOT}/logs/${mode}_service"
  local launcher_out="${OUTPUT_ROOT}/logs/${mode}_launcher.out"
  mkdir -p "${service_log_dir}"

  info "launching ${mode} service"
  (
    export MODEL_PATH PORT
    export LOG_DIR="${service_log_dir}"
    export STAGE_CONFIGS_PATH="${stage_configs_path}"
    export MAX_MODEL_LEN="${max_model_len}"
    export GPU_MEMORY_UTILIZATION="${gpu_memory_utilization}"
    if [[ -n "${extra_exports}" ]]; then
      eval "export ${extra_exports}"
    fi
    nohup bash "${serve_script}" > "${launcher_out}" 2>&1 &
    printf '%s\n' "$!" > "${OUTPUT_ROOT}/${mode}_launcher.pid"
  )

  wait_for_health "${mode}"
}

run_case() {
  local mode="$1"
  local task="$2"
  local steps="$3"
  local concurrency="$4"
  local result_dir="${OUTPUT_ROOT}/${mode}"
  local log_dir="${OUTPUT_ROOT}/logs/${mode}"
  local trace_path="${OUTPUT_ROOT}/contracts/${task}_${steps}_${NUM_REQUESTS}req.trace"
  local output_file="${result_dir}/${task}_${steps}_c${concurrency}_${BACKEND}.json"
  local log_file="${log_dir}/${task}_${steps}_c${concurrency}_${BACKEND}.log"

  mkdir -p "${result_dir}" "${log_dir}"
  [[ -f "${trace_path}" ]] || die "missing trace file: ${trace_path}"

  info "run ${mode} task=${task} steps=${steps} c=${concurrency}"
  python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
    --base-url "http://127.0.0.1:${PORT}" \
    --model "${MODEL_PATH}" \
    --backend "${BACKEND}" \
    --dataset trace \
    --dataset-path "${trace_path}" \
    --task "${task}" \
    --num-prompts "${NUM_REQUESTS}" \
    --max-concurrency "${concurrency}" \
    --request-rate inf \
    --warmup-requests "${WARMUP_REQUESTS}" \
    --warmup-num-inference-steps "${WARMUP_STEPS}" \
    --output-file "${output_file}" \
    --disable-tqdm 2>&1 | tee "${log_file}"
}

run_mode_matrix() {
  local mode="$1"
  launch_service "${mode}"

  local task
  local steps
  local concurrency
  for task in "${TASKS[@]}"; do
    for steps in "${STEPS[@]}"; do
      for concurrency in "${CONCURRENCIES[@]}"; do
        run_case "${mode}" "${task}" "${steps}" "${concurrency}"
      done
    done
  done
}

write_summary() {
  export OUTPUT_ROOT MATRIX_MODE BACKEND TASKS_CSV STEPS_CSV CONCURRENCY_CSV COMPARE_BASELINE_DIR

  python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_ROOT"])
mode = os.environ["MATRIX_MODE"]
backend = os.environ["BACKEND"]
tasks = [x for x in os.environ["TASKS_CSV"].split(",") if x]
steps_list = [int(x) for x in os.environ["STEPS_CSV"].split(",") if x]
concurrency_list = [int(x) for x in os.environ["CONCURRENCY_CSV"].split(",") if x]
compare_baseline_dir = os.environ.get("COMPARE_BASELINE_DIR", "").strip()


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt_float(value, digits=4):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def fmt_pct(value):
    if value is None:
        return "-"
    return f"{value:+.2f}%"


def single_mode_lines(label: str, result_dir: Path):
    lines = [
        f"# {label} Matrix",
        "",
        "| Task | Steps | Concurrency | Mean (s) | P99 (s) | QPS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in tasks:
        for steps in steps_list:
            for c in concurrency_list:
                data = load_json(result_dir / f"{task}_{steps}_c{c}_{backend}.json")
                if data is None:
                    lines.append(f"| {task} | {steps} | {c} | - | - | - |")
                    continue
                lines.append(
                    f"| {task} | {steps} | {c} | "
                    f"{fmt_float(data.get('latency_mean'))} | "
                    f"{fmt_float(data.get('latency_p99'))} | "
                    f"{fmt_float(data.get('throughput_qps'))} |"
                )
    lines.append("")
    return lines


def compare_lines(baseline_dir: Path, split_dir: Path):
    lines = [
        "# Matched Matrix",
        "",
        "| Task | Steps | Concurrency | Baseline Mean (s) | Split Mean (s) | Delta Mean | Baseline QPS | Split QPS | Delta QPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    tail_lines = [
        "",
        "| Task | Steps | Concurrency | Baseline P99 (s) | Split P99 (s) |",
        "|---|---:|---:|---:|---:|",
    ]

    for task in tasks:
        for steps in steps_list:
            for c in concurrency_list:
                base = load_json(baseline_dir / f"{task}_{steps}_c{c}_{backend}.json")
                split = load_json(split_dir / f"{task}_{steps}_c{c}_{backend}.json")
                if base is None or split is None:
                    lines.append(f"| {task} | {steps} | {c} | - | - | - | - | - | - |")
                    tail_lines.append(f"| {task} | {steps} | {c} | - | - |")
                    continue

                bm = base.get("latency_mean")
                sm = split.get("latency_mean")
                bq = base.get("throughput_qps")
                sq = split.get("throughput_qps")
                mean_delta = None if bm in (None, 0) else ((sm - bm) / bm) * 100.0
                qps_delta = None if bq in (None, 0) else ((sq - bq) / bq) * 100.0

                lines.append(
                    f"| {task} | {steps} | {c} | "
                    f"{fmt_float(bm)} | {fmt_float(sm)} | {fmt_pct(mean_delta)} | "
                    f"{fmt_float(bq)} | {fmt_float(sq)} | {fmt_pct(qps_delta)} |"
                )
                tail_lines.append(
                    f"| {task} | {steps} | {c} | "
                    f"{fmt_float(base.get('latency_p99'))} | "
                    f"{fmt_float(split.get('latency_p99'))} |"
                )

    lines.extend(tail_lines)
    lines.append("")
    return lines


summary_dir = root
summary_dir.mkdir(parents=True, exist_ok=True)

if mode == "both":
    baseline_dir = root / "baseline"
    split_dir = root / "split"
    baseline_md = summary_dir / "summary_baseline.md"
    split_md = summary_dir / "summary_split.md"
    compare_md = summary_dir / "summary_compare.md"

    baseline_text = "\n".join(single_mode_lines("Baseline", baseline_dir))
    split_text = "\n".join(single_mode_lines("Split", split_dir))
    compare_text = "\n".join(compare_lines(baseline_dir, split_dir))

    baseline_md.write_text(baseline_text + "\n", encoding="utf-8")
    split_md.write_text(split_text + "\n", encoding="utf-8")
    compare_md.write_text(compare_text + "\n", encoding="utf-8")

    print(compare_text)
    print(f"[info] summary file: {compare_md}")
elif mode == "baseline":
    baseline_dir = root / "baseline"
    baseline_md = summary_dir / "summary_baseline.md"
    baseline_text = "\n".join(single_mode_lines("Baseline", baseline_dir))
    baseline_md.write_text(baseline_text + "\n", encoding="utf-8")
    print(baseline_text)
    print(f"[info] summary file: {baseline_md}")
else:
    split_dir = root / "split"
    split_md = summary_dir / "summary_split.md"
    split_text = "\n".join(single_mode_lines("Split", split_dir))
    split_md.write_text(split_text + "\n", encoding="utf-8")
    print(split_text)
    print(f"[info] summary file: {split_md}")

    if compare_baseline_dir:
        compare_md = summary_dir / "summary_compare.md"
        compare_text = "\n".join(compare_lines(Path(compare_baseline_dir), split_dir))
        compare_md.write_text(compare_text + "\n", encoding="utf-8")
        print("")
        print(compare_text)
        print(f"[info] summary file: {compare_md}")
PY
}

main() {
  mkdir -p "${OUTPUT_ROOT}/logs"
  materialize_stage_configs
  show_config

  if (( DRY_RUN )); then
    info "dry-run only; exiting before cleanup / launch / benchmark"
    exit 0
  fi

  generate_contracts

  case "${MATRIX_MODE}" in
    both)
      run_mode_matrix baseline
      run_mode_matrix split
      ;;
    baseline)
      run_mode_matrix baseline
      ;;
    split)
      run_mode_matrix split
      ;;
  esac

  write_summary

  if (( KEEP_LAST_SERVICE )); then
    info "KEEP_LAST_SERVICE=1; leaving final service up"
  else
    cleanup_processes
  fi

  info "artifacts_root=${OUTPUT_ROOT}"
}

main "$@"
