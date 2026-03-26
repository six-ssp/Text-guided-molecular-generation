#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

MODE="${MODE:-train}"
DEFAULT_PYTHON="${ROOT_DIR}/.mamba-tgmsd/bin/python"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${DEFAULT_PYTHON}"
fi
DATASET_DIR="${DATASET_DIR:-${ROOT_DIR}/ChEBI-20_data}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${ROOT_DIR}/tgm-dlm/checkpoints_sdvae_latent_chebi}"

DEVICE="${DEVICE:-cuda}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-16}"
WORK_CHUNK_SIZE="${WORK_CHUNK_SIZE:-256}"
DECODE_BATCH_SIZE="${DECODE_BATCH_SIZE:-32}"
NUM_SAMPLES_PER_PROMPT="${NUM_SAMPLES_PER_PROMPT:-8}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
LR_ANNEAL_STEPS="${LR_ANNEAL_STEPS:-800000}"
DIST_PORT="${DIST_PORT:-}"
USE_FP16="${USE_FP16:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PIN_MEMORY="${PIN_MEMORY:-1}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-1}"

KILL_STALE="${KILL_STALE:-1}"
PREPARE_IF_MISSING="${PREPARE_IF_MISSING:-1}"
CLEAN_UNUSED_SMILES="${CLEAN_UNUSED_SMILES:-0}"
AUTO_RESUME="${AUTO_RESUME:-1}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"

MODEL_PATH="${MODEL_PATH:-}"
PROMPT_FILE="${PROMPT_FILE:-${ROOT_DIR}/prompts_example.txt}"
OUTPUT="${OUTPUT:-${DATASET_DIR}/prompt_generated.tsv}"
TEXT_FUSION="${TEXT_FUSION:-pooled}"
TEXT_ATTN_HEADS="${TEXT_ATTN_HEADS:-8}"

MAX_HEAVY_ATOMS="${MAX_HEAVY_ATOMS:-65}"
TARGET_COVERAGE="${TARGET_COVERAGE:-0.9}"
TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE:-64}"
TEXT_SAVE_EVERY="${TEXT_SAVE_EVERY:-10}"
LATENT_CHUNK_SIZE="${LATENT_CHUNK_SIZE:-64}"
LATENT_SAVE_EVERY="${LATENT_SAVE_EVERY:-20}"
STATE_DTYPE="${STATE_DTYPE:-float16}"

SDVAE_ROOT="${SDVAE_ROOT:-${ROOT_DIR}/sdvae}"
SDVAE_SAVED_MODEL="${SDVAE_SAVED_MODEL:-${ROOT_DIR}/sdvae/dropbox/results/zinc/zinc_kl_avg.model}"
SDVAE_GRAMMAR_FILE="${SDVAE_GRAMMAR_FILE:-${ROOT_DIR}/sdvae/dropbox/context_free_grammars/mol_zinc.grammar}"

if [[ ! -d "${SDVAE_ROOT}/mol_vae/pytorch_eval" ]]; then
  SDVAE_ROOT="${ROOT_DIR}/sdvae"
fi
if [[ ! -f "${SDVAE_SAVED_MODEL}" ]]; then
  SDVAE_SAVED_MODEL="${ROOT_DIR}/sdvae/dropbox/results/zinc/zinc_kl_avg.model"
fi
if [[ ! -f "${SDVAE_GRAMMAR_FILE}" ]]; then
  SDVAE_GRAMMAR_FILE="${ROOT_DIR}/sdvae/dropbox/context_free_grammars/mol_zinc.grammar"
fi

mkdir -p "${ROOT_DIR}/logs"
mkdir -p "${CHECKPOINT_PATH}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[fatal] project python not found: ${PYTHON_BIN}" >&2
  echo "[fatal] repair .mamba-tgmsd or set PYTHON_BIN=/abs/path/to/python" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"

validate_python_env() {
  if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import torch
from rdkit import Chem
import transformers
assert Chem.MolFromSmiles("CCO") is not None
PY
  then
    echo "[fatal] python env is not usable: ${PYTHON_BIN}" >&2
    echo "[fatal] expected imports failed: torch rdkit transformers" >&2
    exit 1
  fi
}

find_latest_model() {
  ls -1 "${CHECKPOINT_PATH}"/PLAIN_model*.pt 2>/dev/null | sort -V | tail -n 1 || true
}

ensure_cuda_or_fallback_cpu() {
  if [[ "${DEVICE}" == "cpu" ]]; then
    USE_FP16=0
    PIN_MEMORY=0
    return 0
  fi
  if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import sys
import torch
sys.exit(0 if torch.cuda.is_available() else 1)
PY
  then
    return 0
  fi
  echo "[warn] CUDA not available, fallback to CPU."
  DEVICE="cpu"
  USE_FP16=0
  PIN_MEMORY=0
}

kill_stale_jobs() {
  [[ "${KILL_STALE}" == "1" ]] || return 0
  echo "[info] stopping stale project jobs..."
  pkill -f "${ROOT_DIR}/tgm-dlm/improved-diffusion/scripts/train_sdvae_latent.py" 2>/dev/null || true
  pkill -f "${ROOT_DIR}/tgm-dlm/improved-diffusion/scripts/process_text.py" 2>/dev/null || true
  pkill -f "${ROOT_DIR}/tgm-dlm/improved-diffusion/scripts/dump_sdvae_latents.py" 2>/dev/null || true
  pkill -f "${ROOT_DIR}/tgm-dlm/improved-diffusion/scripts/text_guided_generate.py" 2>/dev/null || true
  pkill -f "${ROOT_DIR}/03_train.sh" 2>/dev/null || true
  pkill -f "${ROOT_DIR}/04_generate.sh" 2>/dev/null || true
}

maybe_cleanup_unused_smiles() {
  [[ "${CLEAN_UNUSED_SMILES}" == "1" ]] || return 0
  local target="${ROOT_DIR}/tgm-dlm/datasets/SMILES"
  if [[ -d "${target}" ]]; then
    echo "[info] removing unused heavy directory: ${target}"
    rm -rf "${target}"
  fi
}

needs_prepare() {
  local required=(
    "${DATASET_DIR}/train_pool90.txt"
    "${DATASET_DIR}/validation_pool90.txt"
    "${DATASET_DIR}/test_pool90.txt"
    "${DATASET_DIR}/train_pool90_desc_states.pt"
    "${DATASET_DIR}/train_pool90_sdvae_latents.pt"
    "${DATASET_DIR}/validation_pool90_desc_states.pt"
    "${DATASET_DIR}/validation_pool90_sdvae_latents.pt"
    "${DATASET_DIR}/test_pool90_desc_states.pt"
    "${DATASET_DIR}/test_pool90_sdvae_latents.pt"
  )
  local f
  for f in "${required[@]}"; do
    [[ -f "${f}" ]] || return 0
  done
  return 1
}

do_prepare() {
  local log_file="${ROOT_DIR}/logs/oneclick_prepare_${timestamp}.log"
  echo "[run] prepare full data, log=${log_file}"
  {
    if [[ ! -f "${DATASET_DIR}/train_pool90.txt" || ! -f "${DATASET_DIR}/validation_pool90.txt" || ! -f "${DATASET_DIR}/test_pool90.txt" ]]; then
      echo "[prepare 1/3] build *_pool90.txt"
      "${PYTHON_BIN}" "${ROOT_DIR}/tgm-dlm/improved-diffusion/scripts/build_sdvae_pool90.py" \
        --dataset-dir "${DATASET_DIR}" \
        --splits train validation test \
        --output-suffix _pool90 \
        --max-heavy-atoms "${MAX_HEAVY_ATOMS}" \
        --target-coverage "${TARGET_COVERAGE}"
    else
      echo "[prepare 1/3] skip build (existing *_pool90.txt)"
    fi

    echo "[prepare 2/3] process text states (dtype=${STATE_DTYPE})"
    for split in train_pool90 validation_pool90 test_pool90; do
      "${PYTHON_BIN}" "${ROOT_DIR}/tgm-dlm/improved-diffusion/scripts/process_text.py" \
        -i "${split}" \
        --dataset-dir "${DATASET_DIR}" \
        --batch-size "${TEXT_BATCH_SIZE}" \
        --save-every "${TEXT_SAVE_EVERY}" \
        --state-dtype "${STATE_DTYPE}" \
        --device auto
    done

    echo "[prepare 3/3] dump sdvae latents"
    for split in train_pool90 validation_pool90 test_pool90; do
      "${PYTHON_BIN}" "${ROOT_DIR}/tgm-dlm/improved-diffusion/scripts/dump_sdvae_latents.py" \
        --split "${split}" \
        --dataset-dir "${DATASET_DIR}" \
        --chunk-size "${LATENT_CHUNK_SIZE}" \
        --save-every "${LATENT_SAVE_EVERY}" \
        --sdvae-root "${SDVAE_ROOT}" \
        --saved_model "${SDVAE_SAVED_MODEL}" \
        --grammar_file "${SDVAE_GRAMMAR_FILE}" \
        --mode auto
    done
  } 2>&1 | tee "${log_file}"
}

do_smoke() {
  local log_file="${ROOT_DIR}/logs/oneclick_smoke_${timestamp}.log"
  echo "[run] smoke demo, log=${log_file}"
  PYTHON_BIN="${PYTHON_BIN}" DATASET_DIR="${DATASET_DIR}" \
    ./01_smoke_demo.sh 2>&1 | tee "${log_file}"
}

do_train() {
  local log_file="${ROOT_DIR}/logs/oneclick_train_${timestamp}.log"
  local resume_checkpoint="${RESUME_CHECKPOINT:-}"
  if [[ -z "${resume_checkpoint}" && "${AUTO_RESUME}" == "1" ]]; then
    resume_checkpoint="$(find_latest_model)"
  fi
  if [[ -z "${DIST_PORT}" ]]; then
    DIST_PORT="$(shuf -i 20000-65000 -n 1)"
  fi
  echo "[run] train start, log=${log_file}"
  echo "[info] device=${DEVICE} gpu_id=${GPU_ID} dist_port=${DIST_PORT} batch_size=${BATCH_SIZE} use_fp16=${USE_FP16}"
  if [[ -n "${resume_checkpoint}" ]]; then
    echo "[info] resume from ${resume_checkpoint}"
  fi

  PYTHON_BIN="${PYTHON_BIN}" \
  DATASET_DIR="${DATASET_DIR}" \
  CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
  DEVICE="${DEVICE}" \
  GPU_ID="${GPU_ID}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  LOG_INTERVAL="${LOG_INTERVAL}" \
  SAVE_INTERVAL="${SAVE_INTERVAL}" \
  LR_ANNEAL_STEPS="${LR_ANNEAL_STEPS}" \
  DIST_PORT="${DIST_PORT}" \
  USE_FP16="${USE_FP16}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  PIN_MEMORY="${PIN_MEMORY}" \
  PREFETCH_FACTOR="${PREFETCH_FACTOR}" \
  PERSISTENT_WORKERS="${PERSISTENT_WORKERS}" \
  TEXT_FUSION="${TEXT_FUSION}" \
  TEXT_ATTN_HEADS="${TEXT_ATTN_HEADS}" \
  INIT_CHECKPOINT="${INIT_CHECKPOINT}" \
  RESUME_CHECKPOINT="${resume_checkpoint}" \
  ./03_train.sh 2>&1 | tee "${log_file}"
}

do_generate() {
  local log_file="${ROOT_DIR}/logs/oneclick_generate_${timestamp}.log"
  local selected_model="${MODEL_PATH}"
  if [[ -z "${selected_model}" ]]; then
    selected_model="$(find_latest_model)"
  fi
  if [[ -z "${selected_model}" ]]; then
    echo "[fatal] no checkpoint found under ${CHECKPOINT_PATH}. set MODEL_PATH=... and retry." >&2
    exit 1
  fi
  echo "[run] generate with model=${selected_model}, log=${log_file}"

  PYTHON_BIN="${PYTHON_BIN}" \
  DATASET_DIR="${DATASET_DIR}" \
  MODEL_PATH="${selected_model}" \
  OUTPUT="${OUTPUT}" \
  PROMPT_FILE="${PROMPT_FILE}" \
  NUM_SAMPLES_PER_PROMPT="${NUM_SAMPLES_PER_PROMPT}" \
  BATCH_SIZE="${GEN_BATCH_SIZE}" \
  WORK_CHUNK_SIZE="${WORK_CHUNK_SIZE}" \
  DECODE_BATCH_SIZE="${DECODE_BATCH_SIZE}" \
  DEVICE="${DEVICE}" \
  GPU_ID="${GPU_ID}" \
  TEXT_FUSION="${TEXT_FUSION}" \
  TEXT_ATTN_HEADS="${TEXT_ATTN_HEADS}" \
  SDVAE_ROOT="${SDVAE_ROOT}" \
  SDVAE_SAVED_MODEL="${SDVAE_SAVED_MODEL}" \
  SDVAE_GRAMMAR_FILE="${SDVAE_GRAMMAR_FILE}" \
  ./04_generate.sh 2>&1 | tee "${log_file}"
}

kill_stale_jobs
maybe_cleanup_unused_smiles
validate_python_env
ensure_cuda_or_fallback_cpu

case "${MODE}" in
  prepare)
    do_prepare
    ;;
  smoke)
    do_smoke
    ;;
  train)
    if [[ "${PREPARE_IF_MISSING}" == "1" ]] && needs_prepare; then
      do_prepare
    fi
    do_train
    ;;
  generate)
    do_generate
    ;;
  all)
    if [[ "${PREPARE_IF_MISSING}" == "1" ]] && needs_prepare; then
      do_prepare
    fi
    do_train
    do_generate
    ;;
  *)
    echo "[fatal] MODE must be one of: prepare | smoke | train | generate | all" >&2
    exit 1
    ;;
esac

echo "[done] mode=${MODE}"
