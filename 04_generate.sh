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
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${ROOT_DIR}/tgm-dlm/checkpoints_sdvae_latent_chebi}"
MODEL_PATH="${MODEL_PATH:-${CHECKPOINT_PATH}/PLAIN_model200000.pt}"
OUTPUT="${OUTPUT:-${DATASET_DIR}/prompt_generated.tsv}"
PROMPT_FILE="${PROMPT_FILE:-${ROOT_DIR}/prompts_example.txt}"

NUM_SAMPLES_PER_PROMPT="${NUM_SAMPLES_PER_PROMPT:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WORK_CHUNK_SIZE="${WORK_CHUNK_SIZE:-256}"
DECODE_BATCH_SIZE="${DECODE_BATCH_SIZE:-32}"
OVERSAMPLE_FACTOR="${OVERSAMPLE_FACTOR:-1}"
SELECT_VALID_UNIQUE="${SELECT_VALID_UNIQUE:-0}"
DECODE_RANDOM="${DECODE_RANDOM:-0}"
CANDIDATE_OUTPUT="${CANDIDATE_OUTPUT:-}"
RERANK_REFERENCE_FILE="${RERANK_REFERENCE_FILE:-}"
RERANK_METRIC="${RERANK_METRIC:-none}"
DEVICE="${DEVICE:-auto}"
GPU_ID="${GPU_ID:-0}"
TEXT_FUSION="${TEXT_FUSION:-pooled}"
TEXT_ATTN_HEADS="${TEXT_ATTN_HEADS:-8}"

SDVAE_ROOT="${SDVAE_ROOT:-${ROOT_DIR}/sdvae}"
SDVAE_MODEL="${SDVAE_SAVED_MODEL:-${ROOT_DIR}/sdvae/dropbox/results/zinc/zinc_kl_avg.model}"
SDVAE_GRAMMAR="${SDVAE_GRAMMAR_FILE:-${ROOT_DIR}/sdvae/dropbox/context_free_grammars/mol_zinc.grammar}"

if [[ ! -d "${SDVAE_ROOT}/mol_vae/pytorch_eval" ]]; then
  SDVAE_ROOT="${ROOT_DIR}/sdvae"
fi
if [[ ! -f "${SDVAE_MODEL}" ]]; then
  SDVAE_MODEL="${ROOT_DIR}/sdvae/dropbox/results/zinc/zinc_kl_avg.model"
fi
if [[ ! -f "${SDVAE_GRAMMAR}" ]]; then
  SDVAE_GRAMMAR="${ROOT_DIR}/sdvae/dropbox/context_free_grammars/mol_zinc.grammar"
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "model checkpoint not found: ${MODEL_PATH}" >&2
  echo "set MODEL_PATH=/abs/path/to/PLAIN_modelXXXXXX.pt and rerun" >&2
  exit 1
fi

cd "${ROOT_DIR}"

COMMON_ARGS=(
  --model-path "${MODEL_PATH}"
  --output "${OUTPUT}"
  --num-samples-per-prompt "${NUM_SAMPLES_PER_PROMPT}"
  --oversample-factor "${OVERSAMPLE_FACTOR}"
  --batch-size "${BATCH_SIZE}"
  --work-chunk-size "${WORK_CHUNK_SIZE}"
  --decode-batch-size "${DECODE_BATCH_SIZE}"
  --device "${DEVICE}"
  --gpu-id "${GPU_ID}"
  --text-fusion "${TEXT_FUSION}"
  --text-attn-heads "${TEXT_ATTN_HEADS}"
  --sdvae-root "${SDVAE_ROOT}"
  --saved_model "${SDVAE_MODEL}"
  --grammar_file "${SDVAE_GRAMMAR}"
  --rerank-metric "${RERANK_METRIC}"
)

if [[ "${SELECT_VALID_UNIQUE}" == "1" ]]; then
  COMMON_ARGS+=(--select-valid-unique)
fi
if [[ "${DECODE_RANDOM}" == "1" ]]; then
  COMMON_ARGS+=(--decode-random)
fi
if [[ -n "${CANDIDATE_OUTPUT}" ]]; then
  COMMON_ARGS+=(--candidate-output "${CANDIDATE_OUTPUT}")
fi
if [[ -n "${RERANK_REFERENCE_FILE}" ]]; then
  COMMON_ARGS+=(--rerank-reference-file "${RERANK_REFERENCE_FILE}")
fi

if [[ -f "${PROMPT_FILE}" ]]; then
  "${PYTHON_BIN}" "${ROOT_DIR}/tgm-dlm/improved-diffusion/scripts/text_guided_generate.py" \
    --prompt-file "${PROMPT_FILE}" \
    "${COMMON_ARGS[@]}"
else
  "${PYTHON_BIN}" "${ROOT_DIR}/tgm-dlm/improved-diffusion/scripts/text_guided_generate.py" \
    --prompt "The molecule is an aromatic amide." \
    --prompt "The molecule is a long-chain fatty alcohol." \
    "${COMMON_ARGS[@]}"
fi

echo "generated -> ${OUTPUT}"
