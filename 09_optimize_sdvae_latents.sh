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
SPLIT="${SPLIT:-train_pool90}"
PROMPT_FILE="${PROMPT_FILE:-}"
LIMIT="${LIMIT:-0}"
OUTPUT_LATENTS="${OUTPUT_LATENTS:-${DATASET_DIR}/${SPLIT}_sdvae_latents_inverted.pt}"
SUMMARY_JSON="${SUMMARY_JSON:-${OUTPUT_LATENTS%.pt}.summary.json}"
DECODED_TSV="${DECODED_TSV:-${OUTPUT_LATENTS%.pt}.decoded.tsv}"
BATCH_SIZE="${BATCH_SIZE:-4}"
STEPS="${STEPS:-200}"
LR="${LR:-0.03}"
Z_L2="${Z_L2:-0.00001}"
GRAD_CLIP="${GRAD_CLIP:-5.0}"
DECODE_BATCH_SIZE="${DECODE_BATCH_SIZE:-2}"
RESUME="${RESUME:-1}"
DEVICE_MODE="${DEVICE_MODE:-auto}"
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

CMD=(
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/optimize_sdvae_latents.py"
  --dataset-dir "${DATASET_DIR}"
  --split "${SPLIT}"
  --limit "${LIMIT}"
  --output-latents "${OUTPUT_LATENTS}"
  --summary-json "${SUMMARY_JSON}"
  --decoded-tsv "${DECODED_TSV}"
  --batch-size "${BATCH_SIZE}"
  --steps "${STEPS}"
  --lr "${LR}"
  --z-l2 "${Z_L2}"
  --grad-clip "${GRAD_CLIP}"
  --decode-batch-size "${DECODE_BATCH_SIZE}"
  --mode "${DEVICE_MODE}"
  --sdvae-root "${SDVAE_ROOT}"
  --saved_model "${SDVAE_SAVED_MODEL}"
  --grammar_file "${SDVAE_GRAMMAR_FILE}"
)

if [[ -n "${PROMPT_FILE}" ]]; then
  CMD+=(--prompt-file "${PROMPT_FILE}")
fi
if [[ "${RESUME}" == "1" ]]; then
  CMD+=(--resume)
fi

cd "${ROOT_DIR}"
"${CMD[@]}"
