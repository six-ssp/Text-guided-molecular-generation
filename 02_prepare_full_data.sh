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
    echo "[fatal] repair .mamba-tgmsd or set PYTHON_BIN=/abs/path/to/python" >&2
    exit 1
  fi
fi
if ! "${PYTHON_BIN}" -V >/dev/null 2>&1; then
  echo "[fatal] python is not runnable: ${PYTHON_BIN}" >&2
  exit 1
fi

DATASET_DIR="${DATASET_DIR:-${ROOT_DIR}/ChEBI-20_data}"
FORCE_CLEAN="${FORCE_CLEAN:-0}"
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

cd "${ROOT_DIR}"

if [[ "${FORCE_CLEAN}" == "1" || ! -f "${DATASET_DIR}/train_pool90.txt" || ! -f "${DATASET_DIR}/validation_pool90.txt" || ! -f "${DATASET_DIR}/test_pool90.txt" ]]; then
  echo "[1/3] clean ChEBI-20_data -> *_pool90.txt"
  "${PYTHON_BIN}" "${ROOT_DIR}/tgm-dlm/improved-diffusion/scripts/build_sdvae_pool90.py" \
    --dataset-dir "${DATASET_DIR}" \
    --splits train validation test \
    --output-suffix _pool90 \
    --max-heavy-atoms "${MAX_HEAVY_ATOMS}" \
    --target-coverage "${TARGET_COVERAGE}"
else
  echo "[1/3] skip clean (existing *_pool90.txt detected, FORCE_CLEAN=0)"
fi

echo "[2/3] process text states"
for SPLIT in train_pool90 validation_pool90 test_pool90; do
  "${PYTHON_BIN}" "${ROOT_DIR}/tgm-dlm/improved-diffusion/scripts/process_text.py" \
    -i "${SPLIT}" \
    --dataset-dir "${DATASET_DIR}" \
    --batch-size "${TEXT_BATCH_SIZE}" \
    --save-every "${TEXT_SAVE_EVERY}" \
    --state-dtype "${STATE_DTYPE}" \
    --device auto
done

echo "[3/3] dump sdvae latents"
for SPLIT in train_pool90 validation_pool90 test_pool90; do
  "${PYTHON_BIN}" "${ROOT_DIR}/tgm-dlm/improved-diffusion/scripts/dump_sdvae_latents.py" \
    --split "${SPLIT}" \
    --dataset-dir "${DATASET_DIR}" \
    --chunk-size "${LATENT_CHUNK_SIZE}" \
    --save-every "${LATENT_SAVE_EVERY}" \
    --sdvae-root "${SDVAE_ROOT}" \
    --saved_model "${SDVAE_SAVED_MODEL}" \
    --grammar_file "${SDVAE_GRAMMAR_FILE}" \
    --mode auto
done

echo "[done] prepare full data"
