#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMPROVED_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TGM_ROOT="$(cd "${IMPROVED_DIR}/.." && pwd)"
PROJECT_ROOT="${TEXT2MOL_ROOT:-$(cd "${TGM_ROOT}/.." && pwd)}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_DIR="${DATASET_DIR:-${TGM_ROOT}/datasets/SMILES}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${TGM_ROOT}/checkpoints_sdvae_latent_smoke}"
SMOKE_SPLIT="${SMOKE_SPLIT:-smoke_pool90}"
SMOKE_ROWS="${SMOKE_ROWS:-128}"

SDVAE_ROOT="${SDVAE_ROOT:-${PROJECT_ROOT}/sdvae}"
SDVAE_MODEL="${SDVAE_SAVED_MODEL:-${PROJECT_ROOT}/sdvae/dropbox/results/zinc/zinc_kl_avg.model}"
SDVAE_GRAMMAR="${SDVAE_GRAMMAR_FILE:-${PROJECT_ROOT}/sdvae/dropbox/context_free_grammars/mol_zinc.grammar}"
export DATASET_DIR SMOKE_SPLIT

mkdir -p "${CHECKPOINT_PATH}"

echo "[1/6] build smoke split (${SMOKE_ROWS} rows)"
awk -v n="${SMOKE_ROWS}" 'NR==1 || NR<=n+1' "${DATASET_DIR}/pool90_all.txt" > "${DATASET_DIR}/${SMOKE_SPLIT}.txt"

cd "${SCRIPT_DIR}"

echo "[2/6] process_text"
"${PYTHON_BIN}" process_text.py \
  -i "${SMOKE_SPLIT}" \
  --dataset-dir "${DATASET_DIR}" \
  --batch-size 32 \
  --save-every 2 \
  --device auto

echo "[3/6] dump_sdvae_latents"
"${PYTHON_BIN}" dump_sdvae_latents.py \
  --split "${SMOKE_SPLIT}" \
  --dataset-dir "${DATASET_DIR}" \
  --chunk-size 32 \
  --save-every 2 \
  --sdvae-root "${SDVAE_ROOT}" \
  --saved_model "${SDVAE_MODEL}" \
  --grammar_file "${SDVAE_GRAMMAR}" \
  --mode auto

echo "[4/6] train smoke model"
"${PYTHON_BIN}" train_sdvae_latent.py \
  --dataset-dir "${DATASET_DIR}" \
  --split "${SMOKE_SPLIT}" \
  --latent-file "${DATASET_DIR}/${SMOKE_SPLIT}_sdvae_latents.pt" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --batch-size 16 \
  --log-interval 1 \
  --save-interval 5 \
  --lr-anneal-steps 8 \
  --device auto

echo "[5/6] sample + decode"
MODEL_PATH="${CHECKPOINT_PATH}/PLAIN_model000008.pt"
"${PYTHON_BIN}" sample_sdvae_latent.py \
  --dataset-dir "${DATASET_DIR}" \
  --split "${SMOKE_SPLIT}" \
  --model-path "${MODEL_PATH}" \
  --output "${DATASET_DIR}/${SMOKE_SPLIT}_samples.txt" \
  --num-samples 16 \
  --batch-size 8 \
  --saved_model "${SDVAE_MODEL}" \
  --grammar_file "${SDVAE_GRAMMAR}" \
  --sdvae-root "${SDVAE_ROOT}" \
  --mode auto

echo "[6/6] quick eval (RDKit valid ratio)"
"${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path
from rdkit import Chem
p = Path(os.environ["DATASET_DIR"]) / f'{os.environ["SMOKE_SPLIT"]}_samples.txt'
if not p.exists():
    raise SystemExit(f'missing sample file: {p}')
rows = []
with p.open() as f:
    for line in f:
        cid, ref, pred, _latent = line.rstrip('\n').split('\t', 3)
        rows.append((cid, ref, pred))
valid = sum(1 for _,_,pred in rows if pred and Chem.MolFromSmiles(pred) is not None)
print({'samples': len(rows), 'valid': valid, 'valid_ratio': round(valid/len(rows),4) if rows else 0.0})
PY

echo "smoke run done"
