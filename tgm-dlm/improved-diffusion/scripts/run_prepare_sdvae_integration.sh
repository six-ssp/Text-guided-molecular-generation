#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SDVAE_MODEL="${SDVAE_MODEL:-/home/six_ssp/sdvae/dropbox/results/zinc/zinc_kl_avg.model}"
SDVAE_GRAMMAR="${SDVAE_GRAMMAR:-/home/six_ssp/sdvae/dropbox/context_free_grammars/mol_zinc.grammar}"
SDVAE_ROOT="${SDVAE_ROOT:-${ROOT_DIR}/../sdvae}"

cd "${SCRIPT_DIR}"

python process_text.py -i train_val_256
python process_text.py -i validation_256
python process_text.py -i test

python dump_sdvae_latents.py \
  --split train_val_256 \
  --sdvae-root "${SDVAE_ROOT}" \
  -saved_model "${SDVAE_MODEL}" \
  -grammar_file "${SDVAE_GRAMMAR}"

python dump_sdvae_latents.py \
  --split validation_256 \
  --sdvae-root "${SDVAE_ROOT}" \
  -saved_model "${SDVAE_MODEL}" \
  -grammar_file "${SDVAE_GRAMMAR}"

python dump_sdvae_latents.py \
  --split test \
  --sdvae-root "${SDVAE_ROOT}" \
  -saved_model "${SDVAE_MODEL}" \
  -grammar_file "${SDVAE_GRAMMAR}"
