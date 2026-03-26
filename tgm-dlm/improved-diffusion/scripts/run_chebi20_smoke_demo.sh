#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMPROVED_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TGM_ROOT="$(cd "${IMPROVED_DIR}/.." && pwd)"
PROJECT_ROOT="${TEXT2MOL_ROOT:-$(cd "${TGM_ROOT}/.." && pwd)}"
if [[ ! -d "${PROJECT_ROOT}/sdvae" || ! -d "${PROJECT_ROOT}/tgm-dlm" ]]; then
  PROJECT_ROOT="$(cd "${TGM_ROOT}/.." && pwd)"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/ChEBI-20_data}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${TGM_ROOT}/checkpoints_sdvae_latent_chebi_smoke}"

SMOKE_SPLIT="${SMOKE_SPLIT:-smoke_chebi20_pool90}"
SMOKE_ROWS="${SMOKE_ROWS:-64}"
MAX_HEAVY_ATOMS="${MAX_HEAVY_ATOMS:-65}"
TARGET_COVERAGE="${TARGET_COVERAGE:-0.9}"

SDVAE_ROOT="${SDVAE_ROOT:-${PROJECT_ROOT}/sdvae}"
SDVAE_MODEL="${SDVAE_SAVED_MODEL:-${PROJECT_ROOT}/sdvae/dropbox/results/zinc/zinc_kl_avg.model}"
SDVAE_GRAMMAR="${SDVAE_GRAMMAR_FILE:-${PROJECT_ROOT}/sdvae/dropbox/context_free_grammars/mol_zinc.grammar}"
if [[ ! -d "${SDVAE_ROOT}/mol_vae/pytorch_eval" ]]; then
  SDVAE_ROOT="${PROJECT_ROOT}/sdvae"
fi
if [[ ! -f "${SDVAE_MODEL}" ]]; then
  SDVAE_MODEL="${PROJECT_ROOT}/sdvae/dropbox/results/zinc/zinc_kl_avg.model"
fi
if [[ ! -f "${SDVAE_GRAMMAR}" ]]; then
  SDVAE_GRAMMAR="${PROJECT_ROOT}/sdvae/dropbox/context_free_grammars/mol_zinc.grammar"
fi

GEN_OUTPUT="${GEN_OUTPUT:-${DATASET_DIR}/${SMOKE_SPLIT}_generated.tsv}"

mkdir -p "${CHECKPOINT_PATH}"

cd "${PROJECT_ROOT}"

if [[ ! -f "${DATASET_DIR}/train_pool90.txt" || ! -f "${DATASET_DIR}/validation_pool90.txt" || ! -f "${DATASET_DIR}/test_pool90.txt" ]]; then
  echo "[1/7] clean ChEBI-20_data -> *_pool90.txt"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/build_sdvae_pool90.py" \
    --dataset-dir "${DATASET_DIR}" \
    --splits train validation test \
    --output-suffix _pool90 \
    --max-heavy-atoms "${MAX_HEAVY_ATOMS}" \
    --target-coverage "${TARGET_COVERAGE}"
else
  echo "[1/7] skip clean (existing *_pool90.txt detected)"
fi

echo "[2/7] build smoke split (${SMOKE_ROWS} rows)"
awk -v n="${SMOKE_ROWS}" 'NR==1 || NR<=n+1' "${DATASET_DIR}/train_pool90.txt" > "${DATASET_DIR}/${SMOKE_SPLIT}.txt"

echo "[3/7] process text"
"${PYTHON_BIN}" "${SCRIPT_DIR}/process_text.py" \
  -i "${SMOKE_SPLIT}" \
  --dataset-dir "${DATASET_DIR}" \
  --batch-size 32 \
  --save-every 1 \
  --device auto

echo "[4/7] dump sdvae latents"
"${PYTHON_BIN}" "${SCRIPT_DIR}/dump_sdvae_latents.py" \
  --split "${SMOKE_SPLIT}" \
  --dataset-dir "${DATASET_DIR}" \
  --chunk-size 32 \
  --save-every 1 \
  --sdvae-root "${SDVAE_ROOT}" \
  --saved_model "${SDVAE_MODEL}" \
  --grammar_file "${SDVAE_GRAMMAR}" \
  --mode auto

echo "[5/7] smoke train (8 steps)"
"${PYTHON_BIN}" "${SCRIPT_DIR}/train_sdvae_latent.py" \
  --dataset-dir "${DATASET_DIR}" \
  --split "${SMOKE_SPLIT}" \
  --latent-file "${DATASET_DIR}/${SMOKE_SPLIT}_sdvae_latents.pt" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --batch-size 16 \
  --log-interval 1 \
  --save-interval 5 \
  --lr-anneal-steps 8 \
  --device auto

echo "[6/7] text-guided generate"
"${PYTHON_BIN}" "${SCRIPT_DIR}/text_guided_generate.py" \
  --model-path "${CHECKPOINT_PATH}/PLAIN_model000008.pt" \
  --output "${GEN_OUTPUT}" \
  --prompt "The molecule is an aromatic amide." \
  --prompt "The molecule is a long-chain fatty alcohol." \
  --num-samples-per-prompt 2 \
  --batch-size 2 \
  --device auto \
  --sdvae-root "${SDVAE_ROOT}" \
  --saved_model "${SDVAE_MODEL}" \
  --grammar_file "${SDVAE_GRAMMAR}"

echo "[7/7] quick eval"
export GEN_OUTPUT
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import os
output = Path(os.environ["GEN_OUTPUT"])

with output.open() as f:
    rows = [line.rstrip("\n").split("\t") for i, line in enumerate(f) if i > 0]
valid = sum(1 for row in rows if len(row) >= 5 and row[4] == "1")
print({"samples": len(rows), "valid": valid, "valid_ratio": round(valid / len(rows), 4) if rows else 0.0})
PY

echo "smoke demo done -> ${GEN_OUTPUT}"
