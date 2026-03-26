#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMPROVED_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TGM_ROOT="$(cd "${IMPROVED_DIR}/.." && pwd)"
PROJECT_ROOT="${TEXT2MOL_ROOT:-$(cd "${TGM_ROOT}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.mamba-tgmsd/bin/python}"
DATASET_DIR="${DATASET_DIR:-${TGM_ROOT}/datasets/SMILES}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${TGM_ROOT}/checkpoints_sdvae_latent_pool90}"
TRAIN_DEVICE="${TRAIN_DEVICE:-auto}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
LR_ANNEAL_STEPS="${LR_ANNEAL_STEPS:-200000}"
SPLIT_SEED="${SPLIT_SEED:-20260310}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
VAL_RATIO="${VAL_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.1}"

while pgrep -f "process_text.py -i pool90_all" >/dev/null || pgrep -f "dump_sdvae_latents.py --split pool90_all" >/dev/null; do
  sleep 30
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/resplit_pool90_noleak.py" \
  --dataset-dir "${DATASET_DIR}" \
  --source-split pool90_all \
  --train-split train_pool90 \
  --val-split validation_pool90 \
  --test-split test_pool90 \
  --train-ratio "${TRAIN_RATIO}" \
  --val-ratio "${VAL_RATIO}" \
  --test-ratio "${TEST_RATIO}" \
  --seed "${SPLIT_SEED}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_pool90_split_artifacts.py" \
  --dataset-dir "${DATASET_DIR}" \
  --source-split pool90_all \
  --target-splits train_pool90 validation_pool90 test_pool90

"${PYTHON_BIN}" "${SCRIPT_DIR}/train_sdvae_latent.py" \
  --dataset-dir "${DATASET_DIR}" \
  --split train_pool90 \
  --latent-file "${DATASET_DIR}/train_pool90_sdvae_latents.pt" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --device "${TRAIN_DEVICE}" \
  --gpu-id "${GPU_ID}" \
  --batch-size "${BATCH_SIZE}" \
  --save-interval "${SAVE_INTERVAL}" \
  --log-interval "${LOG_INTERVAL}" \
  --lr-anneal-steps "${LR_ANNEAL_STEPS}"
