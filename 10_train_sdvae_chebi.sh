#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PYTHON="${ROOT_DIR}/.mamba-tgmsd/bin/python"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${DEFAULT_PYTHON}" ]]; then
    PYTHON_BIN="${DEFAULT_PYTHON}"
  else
    echo "[fatal] project python not found: ${DEFAULT_PYTHON}" >&2
    exit 1
  fi
fi

DATASET_DIR="${DATASET_DIR:-${ROOT_DIR}/ChEBI-20_data}"
SDVAE_ROOT="${SDVAE_ROOT:-${ROOT_DIR}/sdvae}"
SDVAE_GRAMMAR_FILE="${SDVAE_GRAMMAR_FILE:-${ROOT_DIR}/sdvae/dropbox/context_free_grammars/mol_zinc.grammar}"
if [[ ! -d "${SDVAE_ROOT}/mol_vae" ]]; then
  SDVAE_ROOT="${ROOT_DIR}/sdvae"
fi
if [[ ! -f "${SDVAE_GRAMMAR_FILE}" ]]; then
  SDVAE_GRAMMAR_FILE="${ROOT_DIR}/sdvae/dropbox/context_free_grammars/mol_zinc.grammar"
fi
SDVAE_DATA_DIR="${SDVAE_DATA_DIR:-${DATASET_DIR}/sdvae_chebi}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train_pool90}"
VALID_SPLIT="${VALID_SPLIT:-validation_pool90}"
TRAIN_DATA="${TRAIN_DATA:-${SDVAE_DATA_DIR}/${TRAIN_SPLIT}.pt}"
VALID_DATA="${VALID_DATA:-${SDVAE_DATA_DIR}/${VALID_SPLIT}.pt}"
BUILD_DATA="${BUILD_DATA:-1}"
TRAIN_LIMIT="${TRAIN_LIMIT:-0}"
VALID_LIMIT="${VALID_LIMIT:-0}"
BUILD_CHUNK_SIZE="${BUILD_CHUNK_SIZE:-512}"

SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/sdvae/dropbox/results/chebi_pool90}"
INIT_MODEL="${INIT_MODEL:-${ROOT_DIR}/sdvae/dropbox/results/zinc/zinc_kl_avg.model}"
FROM_SCRATCH="${FROM_SCRATCH:-0}"
if [[ "${FROM_SCRATCH}" == "1" ]]; then
  INIT_MODEL=""
fi

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.0003}"
KL_COEFF="${KL_COEFF:-1.0}"
EPS_STD="${EPS_STD:-0.01}"
GRAD_CLIP="${GRAD_CLIP:-5.0}"
DEVICE_MODE="${DEVICE_MODE:-auto}"
MAX_DECODE_STEPS="${MAX_DECODE_STEPS:-278}"
LATENT_DIM="${LATENT_DIM:-56}"
EVAL_BATCHES="${EVAL_BATCHES:-0}"

mkdir -p "${SDVAE_DATA_DIR}" "${SAVE_DIR}" "${ROOT_DIR}/logs"
cd "${ROOT_DIR}"

build_split() {
  local split="$1"
  local output="$2"
  local limit="$3"
  if [[ "${BUILD_DATA}" != "1" && -f "${output}" ]]; then
    return 0
  fi
  if [[ -f "${output}" && "${BUILD_DATA}" == "missing" ]]; then
    return 0
  fi
  echo "[sdvae-data] build split=${split} output=${output}"
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/build_sdvae_chebi_dataset.py" \
    --dataset-dir "${DATASET_DIR}" \
    --split "${split}" \
    --output "${output}" \
    --limit "${limit}" \
    --chunk-size "${BUILD_CHUNK_SIZE}" \
    --sdvae-root "${SDVAE_ROOT}" \
    --grammar-file "${SDVAE_GRAMMAR_FILE}" \
    --max-decode-steps "${MAX_DECODE_STEPS}" \
    --latent-dim "${LATENT_DIM}"
}

build_split "${TRAIN_SPLIT}" "${TRAIN_DATA}" "${TRAIN_LIMIT}"
build_split "${VALID_SPLIT}" "${VALID_DATA}" "${VALID_LIMIT}"

echo "[sdvae-train] train_data=${TRAIN_DATA}"
echo "[sdvae-train] valid_data=${VALID_DATA}"
echo "[sdvae-train] save_dir=${SAVE_DIR}"
if [[ -n "${INIT_MODEL}" ]]; then
  echo "[sdvae-train] init_model=${INIT_MODEL}"
else
  echo "[sdvae-train] init_model=<scratch>"
fi

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/train_sdvae_chebi.py" \
  --train-data "${TRAIN_DATA}" \
  --valid-data "${VALID_DATA}" \
  --save-dir "${SAVE_DIR}" \
  --init-model "${INIT_MODEL}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --learning-rate "${LEARNING_RATE}" \
  --kl-coeff "${KL_COEFF}" \
  --eps-std "${EPS_STD}" \
  --grad-clip "${GRAD_CLIP}" \
  --mode "${DEVICE_MODE}" \
  --sdvae-root "${SDVAE_ROOT}" \
  --grammar-file "${SDVAE_GRAMMAR_FILE}" \
  --max-decode-steps "${MAX_DECODE_STEPS}" \
  --latent-dim "${LATENT_DIM}" \
  --eval-batches "${EVAL_BATCHES}"
