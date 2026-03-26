#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

DEFAULT_PYTHON="${ROOT_DIR}/.mamba-tgmsd/bin/python"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" && -x "${DEFAULT_PYTHON}" ]]; then
  PYTHON_BIN="${DEFAULT_PYTHON}"
fi

MODE="${MODE:-full}"
AUTO_TUNE="${AUTO_TUNE:-1}"
PRUNE_UNUSED="${PRUNE_UNUSED:-1}"
PRUNE_GIT="${PRUNE_GIT:-0}"
AUTO_CLEANUP="${AUTO_CLEANUP:-1}"

CLEAN_LOG_DAYS="${CLEAN_LOG_DAYS:-7}"
CLEAN_PYCACHE="${CLEAN_PYCACHE:-1}"
KEEP_LATEST_CHECKPOINTS="${KEEP_LATEST_CHECKPOINTS:-0}"
EXTRA_CHECKPOINT_DIRS="${EXTRA_CHECKPOINT_DIRS:-}"
DISK_WARN_THRESHOLD_PCT="${DISK_WARN_THRESHOLD_PCT:-90}"

DATASET_DIR="${DATASET_DIR:-${ROOT_DIR}/ChEBI-20_data}"
PROMPT_SOURCE_SPLIT="${PROMPT_SOURCE_SPLIT:-test_pool90}"
EVAL_NUM_PROMPTS="${EVAL_NUM_PROMPTS:-128}"
NUM_SAMPLES_PER_PROMPT="${NUM_SAMPLES_PER_PROMPT:-8}"
OUTPUT="${OUTPUT:-${DATASET_DIR}/prompt_generated_large.tsv}"
PROMPT_FILE="${PROMPT_FILE:-}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-${ROOT_DIR}/tgm-dlm/checkpoints_sdvae_latent_crossattn}"
MODEL_PATH="${MODEL_PATH:-}"
TEXT_FUSION="${TEXT_FUSION:-crossattn}"
TEXT_ATTN_HEADS="${TEXT_ATTN_HEADS:-8}"

DEFAULT_INIT_CKPT="${ROOT_DIR}/tgm-dlm/checkpoints_sdvae_latent_chebi/PLAIN_model800000.pt"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
if [[ -z "${INIT_CHECKPOINT}" && -f "${DEFAULT_INIT_CKPT}" ]]; then
  INIT_CHECKPOINT="${DEFAULT_INIT_CKPT}"
fi

DEVICE="${DEVICE:-cuda}"
GPU_ID="${GPU_ID:-0}"
TRAIN_STEPS="${TRAIN_STEPS:-1200000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20000}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"

BATCH_SIZE="${BATCH_SIZE:-256}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-64}"
WORK_CHUNK_SIZE="${WORK_CHUNK_SIZE:-256}"
DECODE_BATCH_SIZE="${DECODE_BATCH_SIZE:-32}"
USE_FP16="${USE_FP16:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PIN_MEMORY="${PIN_MEMORY:-1}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-1}"

mkdir -p "${ROOT_DIR}/logs"

remove_path() {
  local path="$1"
  if [[ -e "${path}" ]]; then
    local size
    size="$(du -sh "${path}" 2>/dev/null | awk '{print $1}')"
    echo "[prune] remove ${path} (${size})"
    rm -r "${path}"
  fi
}

prune_irrelevant() {
  echo "[prune] start"
  remove_path "${ROOT_DIR}/tgm-dlm/scibert/flax_model.msgpack"
  remove_path "${ROOT_DIR}/tgm-dlm/bert-base-uncased"
  remove_path "${ROOT_DIR}/tgm-dlm/checkpoints"
  remove_path "${ROOT_DIR}/tgm-dlm/correction_checkpoints"
  remove_path "${ROOT_DIR}/sdvae/prog_vae"
  if [[ -d "${ROOT_DIR}/tgm-dlm/datasets" ]] && [[ -z "$(ls -A "${ROOT_DIR}/tgm-dlm/datasets")" ]]; then
    remove_path "${ROOT_DIR}/tgm-dlm/datasets"
  fi
  if [[ "${PRUNE_GIT}" == "1" ]]; then
    remove_path "${ROOT_DIR}/tgm-dlm/.git"
    remove_path "${ROOT_DIR}/sdvae/.git"
  fi
  echo "[prune] done"
}

status_report() {
  local used_pct=""
  used_pct="$(df -P "${ROOT_DIR}" | awk 'NR==2 {gsub(/%/, "", $5); print $5}')"

  echo "[status] root=${ROOT_DIR}"
  df -h "${ROOT_DIR}" | awk 'NR==1 || NR==2'

  for d in "${DATASET_DIR}" "${CHECKPOINT_PATH}" "${ROOT_DIR}/tgm-dlm/checkpoints_sdvae_latent_chebi" "${ROOT_DIR}/logs"; do
    if [[ -d "${d}" ]]; then
      local size
      size="$(du -sh "${d}" 2>/dev/null | awk '{print $1}')"
      echo "[status] size ${d}: ${size}"
    fi
  done

  if [[ -d "${CHECKPOINT_PATH}" ]]; then
    local count latest
    count="$(find "${CHECKPOINT_PATH}" -maxdepth 1 -type f -name 'PLAIN_model*.pt' | wc -l | tr -d '[:space:]')"
    latest="$(ls -1 "${CHECKPOINT_PATH}"/PLAIN_model*.pt 2>/dev/null | sort -V | tail -n 1 || true)"
    echo "[status] checkpoints model_count=${count} latest=${latest:-N/A}"
  fi

  if [[ "${used_pct}" =~ ^[0-9]+$ ]] && (( used_pct >= DISK_WARN_THRESHOLD_PCT )); then
    echo "[warn] disk usage is ${used_pct}% (>=${DISK_WARN_THRESHOLD_PCT}%). consider MODE=cleanup."
  fi
}

warn_if_disk_high() {
  local used_pct=""
  used_pct="$(df -P "${ROOT_DIR}" | awk 'NR==2 {gsub(/%/, "", $5); print $5}')"
  if [[ "${used_pct}" =~ ^[0-9]+$ ]] && (( used_pct >= DISK_WARN_THRESHOLD_PCT )); then
    echo "[warn] disk usage is ${used_pct}% (>=${DISK_WARN_THRESHOLD_PCT}%)."
    echo "[warn] run: MODE=cleanup KEEP_LATEST_CHECKPOINTS=40 ./07_full_pipeline.sh"
  fi
}

cleanup_checkpoint_dir() {
  local ckpt_dir="$1"
  local keep="$2"
  [[ -d "${ckpt_dir}" ]] || return 0
  [[ "${keep}" =~ ^[0-9]+$ ]] || return 0
  (( keep > 0 )) || return 0

  local -a models
  mapfile -t models < <(find "${ckpt_dir}" -maxdepth 1 -type f -name 'PLAIN_model*.pt' -printf '%f\n' | sort -V)
  local total="${#models[@]}"
  if (( total <= keep )); then
    echo "[cleanup] checkpoint keep=${keep}, current=${total}, skip ${ckpt_dir}"
    return 0
  fi

  local drop=$((total - keep))
  local i
  for ((i = 0; i < drop; i++)); do
    local model_file step
    model_file="${models[i]}"
    step="${model_file#PLAIN_model}"
    step="${step%.pt}"
    rm -f "${ckpt_dir}/PLAIN_model${step}.pt"
    rm -f "${ckpt_dir}/PLAIN_opt${step}.pt"
    rm -f "${ckpt_dir}/PLAIN_scaler${step}.pt"
    rm -f "${ckpt_dir}/opt${step}.pt"
    find "${ckpt_dir}" -maxdepth 1 -type f -name "PLAIN_ema_*_${step}.pt" -delete
    find "${ckpt_dir}" -maxdepth 1 -type f -name "ema_*_${step}.pt" -delete
  done
  echo "[cleanup] removed ${drop} old checkpoint step groups in ${ckpt_dir}"
}

cleanup_space() {
  echo "[cleanup] start"
  if [[ "${CLEAN_PYCACHE}" == "1" ]]; then
    find "${ROOT_DIR}" -type d -name "__pycache__" -prune -exec rm -rf {} +
  fi
  if [[ -d "${ROOT_DIR}/logs" && "${CLEAN_LOG_DAYS}" =~ ^[0-9]+$ && "${CLEAN_LOG_DAYS}" -gt 0 ]]; then
    find "${ROOT_DIR}/logs" -type f -name "*.log" -mtime +"${CLEAN_LOG_DAYS}" -delete
  fi
  if [[ "${KEEP_LATEST_CHECKPOINTS}" =~ ^[0-9]+$ ]] && (( KEEP_LATEST_CHECKPOINTS > 0 )); then
    cleanup_checkpoint_dir "${CHECKPOINT_PATH}" "${KEEP_LATEST_CHECKPOINTS}"
    if [[ -n "${EXTRA_CHECKPOINT_DIRS}" ]]; then
      IFS=',' read -r -a extra_dirs <<< "${EXTRA_CHECKPOINT_DIRS}"
      local d
      for d in "${extra_dirs[@]}"; do
        d="$(echo "${d}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        [[ -n "${d}" ]] || continue
        cleanup_checkpoint_dir "${d}" "${KEEP_LATEST_CHECKPOINTS}"
      done
    fi
  fi
  echo "[cleanup] done"
}

auto_tune() {
  if [[ "${AUTO_TUNE}" != "1" ]]; then
    return 0
  fi

  local mem_total=""
  if [[ "${DEVICE}" != "cpu" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    mem_total="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sed -n "$((GPU_ID + 1))p" | tr -d '[:space:]')"
  fi

  if [[ -n "${mem_total}" && "${mem_total}" =~ ^[0-9]+$ ]]; then
    if (( mem_total >= 79000 )); then
      BATCH_SIZE="1024"; GEN_BATCH_SIZE="128"; NUM_WORKERS="16"
    elif (( mem_total >= 47000 )); then
      BATCH_SIZE="512"; GEN_BATCH_SIZE="96"; NUM_WORKERS="12"
    elif (( mem_total >= 23000 )); then
      BATCH_SIZE="256"; GEN_BATCH_SIZE="64"; NUM_WORKERS="10"
    elif (( mem_total >= 15000 )); then
      BATCH_SIZE="128"; GEN_BATCH_SIZE="32"; NUM_WORKERS="8"
    else
      BATCH_SIZE="64"; GEN_BATCH_SIZE="16"; NUM_WORKERS="6"
    fi
    USE_FP16="1"
    PIN_MEMORY="1"
  else
    DEVICE="cpu"
    BATCH_SIZE="32"
    GEN_BATCH_SIZE="8"
    NUM_WORKERS="4"
    USE_FP16="0"
    PIN_MEMORY="0"
  fi

  echo "[tune] device=${DEVICE} gpu_id=${GPU_ID} mem_total_mb=${mem_total:-NA}"
  echo "[tune] train_bs=${BATCH_SIZE} gen_bs=${GEN_BATCH_SIZE} workers=${NUM_WORKERS} fp16=${USE_FP16}"
}

build_prompt_file() {
  if [[ -n "${PROMPT_FILE}" && -f "${PROMPT_FILE}" ]]; then
    echo "[prompt] use provided prompt file: ${PROMPT_FILE}"
    return 0
  fi

  local src_file="${DATASET_DIR}/${PROMPT_SOURCE_SPLIT}.txt"
  if [[ ! -f "${src_file}" ]]; then
    echo "[fatal] prompt source split file not found: ${src_file}" >&2
    exit 1
  fi

  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  PROMPT_FILE="${ROOT_DIR}/logs/auto_prompts_${PROMPT_SOURCE_SPLIT}_${EVAL_NUM_PROMPTS}_${ts}.txt"

  awk -F'\t' 'NR>1 && $2!="*" && length($3)>0 {print $3}' "${src_file}" \
    | awk '!seen[$0]++' \
    | shuf -n "${EVAL_NUM_PROMPTS}" > "${PROMPT_FILE}"

  local n
  n="$(wc -l < "${PROMPT_FILE}" | tr -d '[:space:]')"
  if [[ "${n}" == "0" ]]; then
    echo "[fatal] no prompt extracted from ${src_file}" >&2
    exit 1
  fi
  echo "[prompt] built ${PROMPT_FILE} (num_prompts=${n})"
}

run_prepare() {
  MODE=prepare \
  PYTHON_BIN="${PYTHON_BIN}" \
  DATASET_DIR="${DATASET_DIR}" \
  CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
  ./run_local_oneclick.sh
}

run_train() {
  MODE=train \
  PYTHON_BIN="${PYTHON_BIN}" \
  DATASET_DIR="${DATASET_DIR}" \
  CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
  DEVICE="${DEVICE}" \
  GPU_ID="${GPU_ID}" \
  LR_ANNEAL_STEPS="${TRAIN_STEPS}" \
  SAVE_INTERVAL="${SAVE_INTERVAL}" \
  LOG_INTERVAL="${LOG_INTERVAL}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  USE_FP16="${USE_FP16}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  PIN_MEMORY="${PIN_MEMORY}" \
  PREFETCH_FACTOR="${PREFETCH_FACTOR}" \
  PERSISTENT_WORKERS="${PERSISTENT_WORKERS}" \
  TEXT_FUSION="${TEXT_FUSION}" \
  TEXT_ATTN_HEADS="${TEXT_ATTN_HEADS}" \
  INIT_CHECKPOINT="${INIT_CHECKPOINT}" \
  ./run_local_oneclick.sh
}

run_generate() {
  build_prompt_file

  MODE=generate \
  PYTHON_BIN="${PYTHON_BIN}" \
  DATASET_DIR="${DATASET_DIR}" \
  CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
  MODEL_PATH="${MODEL_PATH}" \
  OUTPUT="${OUTPUT}" \
  PROMPT_FILE="${PROMPT_FILE}" \
  NUM_SAMPLES_PER_PROMPT="${NUM_SAMPLES_PER_PROMPT}" \
  GEN_BATCH_SIZE="${GEN_BATCH_SIZE}" \
  WORK_CHUNK_SIZE="${WORK_CHUNK_SIZE}" \
  DECODE_BATCH_SIZE="${DECODE_BATCH_SIZE}" \
  DEVICE="${DEVICE}" \
  GPU_ID="${GPU_ID}" \
  TEXT_FUSION="${TEXT_FUSION}" \
  TEXT_ATTN_HEADS="${TEXT_ATTN_HEADS}" \
  ./run_local_oneclick.sh
}

run_evaluate() {
  GENERATED_FILE="${OUTPUT}" \
  DATASET_DIR="${DATASET_DIR}" \
  ./05_evaluate.sh
}

auto_tune
warn_if_disk_high

case "${MODE}" in
  prune)
    prune_irrelevant
    ;;
  cleanup)
    cleanup_space
    ;;
  status)
    status_report
    ;;
  prepare)
    run_prepare
    ;;
  train)
    run_train
    ;;
  generate)
    run_generate
    ;;
  evaluate)
    run_evaluate
    ;;
  full)
    if [[ "${PRUNE_UNUSED}" == "1" ]]; then
      prune_irrelevant
    fi
    if [[ "${AUTO_CLEANUP}" == "1" ]]; then
      cleanup_space
    fi
    run_prepare
    run_train
    run_generate
    run_evaluate
    ;;
  *)
    echo "[fatal] MODE must be one of: full | prune | cleanup | status | prepare | train | generate | evaluate" >&2
    exit 1
    ;;
esac

echo "[done] mode=${MODE} output=${OUTPUT}"
