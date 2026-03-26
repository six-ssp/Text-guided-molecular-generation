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

export TEXT2MOL_ROOT="${TEXT2MOL_ROOT:-${ROOT_DIR}}"
export PYTHON_BIN
export DATASET_DIR="${DATASET_DIR:-${ROOT_DIR}/ChEBI-20_data}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-${ROOT_DIR}/tgm-dlm/checkpoints_sdvae_latent_chebi_smoke}"
export SDVAE_ROOT="${SDVAE_ROOT:-${ROOT_DIR}/sdvae}"
export SDVAE_SAVED_MODEL="${SDVAE_SAVED_MODEL:-${ROOT_DIR}/sdvae/dropbox/results/zinc/zinc_kl_avg.model}"
export SDVAE_GRAMMAR_FILE="${SDVAE_GRAMMAR_FILE:-${ROOT_DIR}/sdvae/dropbox/context_free_grammars/mol_zinc.grammar}"

cd "${ROOT_DIR}"
"${PYTHON_BIN}" "${ROOT_DIR}/tgm-dlm/improved-diffusion/scripts/run_chebi20_smoke_demo.sh"
