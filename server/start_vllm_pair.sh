#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/server/vllm_logs}"
PID_DIR="${PID_DIR:-${LOG_DIR}/pids}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SLM_MODEL="${SLM_MODEL:-}"
LLM_MODEL="${LLM_MODEL:-}"
SLM_SERVED_MODEL_NAME="${SLM_SERVED_MODEL_NAME:-}"
LLM_SERVED_MODEL_NAME="${LLM_SERVED_MODEL_NAME:-}"

SLM_PORT="${SLM_PORT:-8101}"
LLM_PORT="${LLM_PORT:-8102}"
HOST="${HOST:-0.0.0.0}"
START_TIMEOUT="${START_TIMEOUT:-180}"

SLM_CUDA_DEVICE="${SLM_CUDA_DEVICE:-0}"
LLM_CUDA_DEVICE="${LLM_CUDA_DEVICE:-1}"
SLM_GPU_MEMORY_UTILIZATION="${SLM_GPU_MEMORY_UTILIZATION:-0.75}"
LLM_GPU_MEMORY_UTILIZATION="${LLM_GPU_MEMORY_UTILIZATION:-0.75}"
SLM_MAX_MODEL_LEN="${SLM_MAX_MODEL_LEN:-20000}"
LLM_MAX_MODEL_LEN="${LLM_MAX_MODEL_LEN:-20000}"

TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
API_KEY="${API_KEY:-}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
SLM_CHAT_TEMPLATE="${SLM_CHAT_TEMPLATE:-${CHAT_TEMPLATE}}"
LLM_CHAT_TEMPLATE="${LLM_CHAT_TEMPLATE:-${CHAT_TEMPLATE}}"

DTYPE="${DTYPE:-}"
SLM_DTYPE="${SLM_DTYPE:-${DTYPE}}"
LLM_DTYPE="${LLM_DTYPE:-${DTYPE}}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-}"
SLM_TENSOR_PARALLEL_SIZE="${SLM_TENSOR_PARALLEL_SIZE:-${TENSOR_PARALLEL_SIZE}}"
LLM_TENSOR_PARALLEL_SIZE="${LLM_TENSOR_PARALLEL_SIZE:-${TENSOR_PARALLEL_SIZE}}"

# DCRH needs raw pre-processor logits for entropy in the vLLM path.
LOGPROBS_MODE="${LOGPROBS_MODE:-raw_logits}"
MAX_LOGPROBS="${MAX_LOGPROBS:-20}"

SLM_EXTRA_ARGS="${SLM_EXTRA_ARGS:-}"
LLM_EXTRA_ARGS="${LLM_EXTRA_ARGS:-}"

mkdir -p "${LOG_DIR}" "${PID_DIR}"

require_model() {
  local role="$1"
  local model="$2"
  if [[ -z "${model}" ]]; then
    echo "${role}_MODEL is required" >&2
    exit 1
  fi
  if [[ ! -e "${model}" ]]; then
    echo "${role}_MODEL path does not exist: ${model}" >&2
    exit 1
  fi
}

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "PYTHON_BIN is not executable or on PATH: ${PYTHON_BIN}" >&2
  exit 1
fi

require_model "SLM" "${SLM_MODEL}"
require_model "LLM" "${LLM_MODEL}"

if [[ -z "${SLM_SERVED_MODEL_NAME}" ]]; then
  SLM_SERVED_MODEL_NAME="$(basename "${SLM_MODEL}")"
fi
if [[ -z "${LLM_SERVED_MODEL_NAME}" ]]; then
  LLM_SERVED_MODEL_NAME="$(basename "${LLM_MODEL}")"
fi

sanitize_name() {
  echo "$1" | tr '/:[:space:]' '___'
}

check_healthy() {
  local port="$1"
  curl -fsS -m 5 "http://127.0.0.1:${port}/health" >/dev/null 2>&1
}

start_one() {
  local role="$1"
  local model="$2"
  local served_name="$3"
  local port="$4"
  local cuda_device="$5"
  local gpu_memory="$6"
  local max_model_len="$7"
  local chat_template="$8"
  local dtype="$9"
  local tensor_parallel_size="${10}"
  local extra_args="${11}"

  local health_url="http://127.0.0.1:${port}/health"
  local models_url="http://127.0.0.1:${port}/v1/models"
  local existing_pid
  existing_pid="$(lsof -ti tcp:"${port}" || true)"

  if check_healthy "${port}"; then
    echo "${role} vLLM is already healthy on port ${port}"
    echo "${role}_health=${health_url}"
    echo "${role}_models=${models_url}"
    return 0
  fi

  if [[ -n "${existing_pid}" ]]; then
    if [[ "${FORCE_RESTART:-0}" != "1" ]]; then
      echo "Port ${port} is occupied by PID ${existing_pid}, but the health check failed." >&2
      echo "Set FORCE_RESTART=1 to kill that process and restart ${role} vLLM." >&2
      exit 1
    fi
    kill "${existing_pid}"
    sleep 2
  fi

  local timestamp
  timestamp="$(date +%Y%m%d_%H%M%S)"
  local safe_name
  safe_name="$(sanitize_name "${served_name}")"
  local log_file="${LOG_DIR}/${role}_${safe_name}_cuda${cuda_device}_port${port}_${timestamp}.log"
  local pid_file="${PID_DIR}/${role}.pid"

  local cmd=(
    "${PYTHON_BIN}" -u -m vllm.entrypoints.openai.api_server
    --model "${model}"
    --served-model-name "${served_name}"
    --port "${port}"
    --host "${HOST}"
    --gpu-memory-utilization "${gpu_memory}"
    --max-model-len "${max_model_len}"
    --logprobs-mode "${LOGPROBS_MODE}"
    --max-logprobs "${MAX_LOGPROBS}"
  )

  if [[ -n "${API_KEY}" ]]; then
    cmd+=(--api-key "${API_KEY}")
  fi
  if [[ -n "${tensor_parallel_size}" ]]; then
    cmd+=(--tensor-parallel-size "${tensor_parallel_size}")
  fi
  if [[ -n "${dtype}" ]]; then
    cmd+=(--dtype "${dtype}")
  fi
  if [[ -n "${chat_template}" ]]; then
    cmd+=(--chat-template "${chat_template}")
  fi
  if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
    cmd+=(--trust-remote-code)
  fi
  if [[ "${ENFORCE_EAGER}" == "1" ]]; then
    cmd+=(--enforce-eager)
  fi
  if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
    cmd+=(--enable-prefix-caching)
  fi
  if [[ -n "${extra_args}" ]]; then
    # shellcheck disable=SC2206
    local parsed_extra=( ${extra_args} )
    cmd+=("${parsed_extra[@]}")
  fi

  echo "Starting ${role} vLLM"
  echo "${role}_model=${model}"
  echo "${role}_served_model_name=${served_name}"
  echo "${role}_cuda_device=${cuda_device}"
  echo "${role}_port=${port}"
  echo "${role}_log=${log_file}"

  CUDA_VISIBLE_DEVICES="${cuda_device}" nohup "${cmd[@]}" >"${log_file}" 2>&1 &
  local server_pid=$!
  echo "${server_pid}" >"${pid_file}"

  for ((i = 1; i <= START_TIMEOUT; i++)); do
    if check_healthy "${port}"; then
      echo "${role} vLLM started successfully on port ${port}"
      echo "${role}_pid=${server_pid}"
      echo "${role}_pid_file=${pid_file}"
      echo "${role}_health=${health_url}"
      echo "${role}_models=${models_url}"
      return 0
    fi

    if ! kill -0 "${server_pid}" >/dev/null 2>&1; then
      echo "${role} vLLM exited before becoming healthy. Last log lines:" >&2
      tail -n 50 "${log_file}" >&2 || true
      exit 1
    fi

    sleep 1
  done

  echo "Timed out after ${START_TIMEOUT}s waiting for ${role} vLLM health check." >&2
  tail -n 50 "${log_file}" >&2 || true
  kill "${server_pid}" >/dev/null 2>&1 || true
  exit 1
}

start_one \
  "slm" \
  "${SLM_MODEL}" \
  "${SLM_SERVED_MODEL_NAME}" \
  "${SLM_PORT}" \
  "${SLM_CUDA_DEVICE}" \
  "${SLM_GPU_MEMORY_UTILIZATION}" \
  "${SLM_MAX_MODEL_LEN}" \
  "${SLM_CHAT_TEMPLATE}" \
  "${SLM_DTYPE}" \
  "${SLM_TENSOR_PARALLEL_SIZE}" \
  "${SLM_EXTRA_ARGS}"

start_one \
  "llm" \
  "${LLM_MODEL}" \
  "${LLM_SERVED_MODEL_NAME}" \
  "${LLM_PORT}" \
  "${LLM_CUDA_DEVICE}" \
  "${LLM_GPU_MEMORY_UTILIZATION}" \
  "${LLM_MAX_MODEL_LEN}" \
  "${LLM_CHAT_TEMPLATE}" \
  "${LLM_DTYPE}" \
  "${LLM_TENSOR_PARALLEL_SIZE}" \
  "${LLM_EXTRA_ARGS}"

echo "Both vLLM servers are healthy."
echo "SLM_BASE_URL=http://127.0.0.1:${SLM_PORT}/v1"
echo "LLM_BASE_URL=http://127.0.0.1:${LLM_PORT}/v1"
