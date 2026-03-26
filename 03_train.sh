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
SPLIT="${SPLIT:-train_pool90}"
LATENT_FILE="${LATENT_FILE:-${DATASET_DIR}/${SPLIT}_sdvae_latents.pt}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${ROOT_DIR}/tgm-dlm/checkpoints_sdvae_latent_chebi}"

BATCH_SIZE="${BATCH_SIZE:-128}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
LR_ANNEAL_STEPS="${LR_ANNEAL_STEPS:-800000}"
DEVICE="${DEVICE:-auto}"
GPU_ID="${GPU_ID:-0}"
MODEL_CHANNELS="${MODEL_CHANNELS:-256}"
HIDDEN_SIZE="${HIDDEN_SIZE:-512}"
TEXT_FUSION="${TEXT_FUSION:-pooled}"
TEXT_ATTN_HEADS="${TEXT_ATTN_HEADS:-8}"
DIST_PORT="${DIST_PORT:-12145}"

RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
USE_FP16="${USE_FP16:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PIN_MEMORY="${PIN_MEMORY:-1}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-1}"

mkdir -p "${CHECKPOINT_PATH}"
cd "${ROOT_DIR}"

CMD=(
  "${PYTHON_BIN}" "${ROOT_DIR}/tgm-dlm/improved-diffusion/scripts/train_sdvae_latent.py"
  --dataset-dir "${DATASET_DIR}"
  --split "${SPLIT}"
  --latent-file "${LATENT_FILE}"
  --checkpoint-path "${CHECKPOINT_PATH}"
  --batch-size "${BATCH_SIZE}"
  --log-interval "${LOG_INTERVAL}"
  --save-interval "${SAVE_INTERVAL}"
  --lr-anneal-steps "${LR_ANNEAL_STEPS}"
  --device "${DEVICE}"
  --gpu-id "${GPU_ID}"
  --model-channels "${MODEL_CHANNELS}"
  --hidden-size "${HIDDEN_SIZE}"
  --text-fusion "${TEXT_FUSION}"
  --text-attn-heads "${TEXT_ATTN_HEADS}"
  --num-workers "${NUM_WORKERS}"
  --prefetch-factor "${PREFETCH_FACTOR}"
  --dist-port "${DIST_PORT}"
)

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  CMD+=(--resume-checkpoint "${RESUME_CHECKPOINT}")
fi

if [[ -n "${INIT_CHECKPOINT}" ]]; then
  CMD+=(--init-checkpoint "${INIT_CHECKPOINT}")
fi

if [[ "${USE_FP16}" == "1" ]]; then
  CMD+=(--use-fp16)
fi

if [[ "${PIN_MEMORY}" == "1" ]]; then
  CMD+=(--pin-memory)
fi

if [[ "${PERSISTENT_WORKERS}" == "1" && "${NUM_WORKERS}" -gt 0 ]]; then
  CMD+=(--persistent-workers)
fi

"${CMD[@]}"
