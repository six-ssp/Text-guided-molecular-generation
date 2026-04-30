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
SDVAE_SAVED_MODEL="${SDVAE_SAVED_MODEL:-${ROOT_DIR}/sdvae/dropbox/results/chebi_pool90_scratch/epoch-best.model}"
SDVAE_GRAMMAR_FILE="${SDVAE_GRAMMAR_FILE:-${ROOT_DIR}/sdvae/dropbox/context_free_grammars/mol_zinc.grammar}"
if [[ ! -d "${SDVAE_ROOT}/mol_vae" ]]; then
  SDVAE_ROOT="${ROOT_DIR}/sdvae"
fi
if [[ ! -f "${SDVAE_GRAMMAR_FILE}" ]]; then
  SDVAE_GRAMMAR_FILE="${ROOT_DIR}/sdvae/dropbox/context_free_grammars/mol_zinc.grammar"
fi
SPLITS="${SPLITS:-train_pool90 validation_pool90 test_pool90}"
LATENT_TAG="${LATENT_TAG:-chebi}"
CHUNK_SIZE="${CHUNK_SIZE:-64}"
SAVE_EVERY="${SAVE_EVERY:-20}"
DEVICE_MODE="${DEVICE_MODE:-auto}"
MAX_DECODE_STEPS="${MAX_DECODE_STEPS:-278}"
LATENT_DIM="${LATENT_DIM:-56}"

cd "${ROOT_DIR}"
for split in ${SPLITS}; do
  output="${DATASET_DIR}/${split}_sdvae_${LATENT_TAG}_latents.pt"
  skipped="${DATASET_DIR}/${split}_sdvae_${LATENT_TAG}_skipped.txt"
  echo "[sdvae-latent] split=${split} model=${SDVAE_SAVED_MODEL}"
  "${PYTHON_BIN}" "${ROOT_DIR}/tgm-dlm/improved-diffusion/scripts/dump_sdvae_latents.py" \
    --split "${split}" \
    --dataset-dir "${DATASET_DIR}" \
    --output "${output}" \
    --skipped-output "${skipped}" \
    --chunk-size "${CHUNK_SIZE}" \
    --save-every "${SAVE_EVERY}" \
    --mode "${DEVICE_MODE}" \
    --sdvae-root "${SDVAE_ROOT}" \
    --saved_model "${SDVAE_SAVED_MODEL}" \
    --grammar_file "${SDVAE_GRAMMAR_FILE}" \
    --max_decode_steps "${MAX_DECODE_STEPS}" \
    --latent_dim "${LATENT_DIM}"
done
