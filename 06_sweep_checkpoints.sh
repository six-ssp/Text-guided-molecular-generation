#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${ROOT_DIR}/tgm-dlm/checkpoints_sdvae_latent_crossattn}"
START_STEP="${START_STEP:-0}"
END_STEP="${END_STEP:-99999999}"
STEP_STRIDE="${STEP_STRIDE:-20000}"
MAX_CHECKPOINTS="${MAX_CHECKPOINTS:-20}"
NUM_SAMPLES_PER_PROMPT="${NUM_SAMPLES_PER_PROMPT:-8}"
EVAL_NUM_PROMPTS="${EVAL_NUM_PROMPTS:-128}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-4}"
WORK_CHUNK_SIZE="${WORK_CHUNK_SIZE:-256}"
DECODE_BATCH_SIZE="${DECODE_BATCH_SIZE:-8}"
TEXT_FUSION="${TEXT_FUSION:-crossattn}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/logs/sweep_outputs}"

mkdir -p "${OUTPUT_DIR}"
SUMMARY_FILE="${OUTPUT_DIR}/sweep_summary_$(date +%Y%m%d_%H%M%S).tsv"
printf "step\tmodel_path\tgenerated_file\tvalid_ratio\tunique_ratio\tnovelty_ratio\tdiversity\n" > "${SUMMARY_FILE}"

mapfile -t MODELS < <(find "${CHECKPOINT_PATH}" -maxdepth 1 -type f -name 'PLAIN_model*.pt' | sort -V)

count=0
for model in "${MODELS[@]}"; do
  step="$(basename "${model}")"
  step="${step#PLAIN_model}"
  step="${step%.pt}"
  step_num=$((10#${step}))
  if (( step_num < START_STEP || step_num > END_STEP )); then
    continue
  fi
  if (( STEP_STRIDE > 0 )) && (( (step_num - START_STEP) % STEP_STRIDE != 0 )); then
    continue
  fi
  if (( count >= MAX_CHECKPOINTS )); then
    break
  fi

  out_file="${OUTPUT_DIR}/prompt_generated_${step}.tsv"
  echo "[sweep] step=${step_num} model=${model}"
  MODE=generate \
  KILL_STALE=0 \
  CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
  MODEL_PATH="${model}" \
  TEXT_FUSION="${TEXT_FUSION}" \
  EVAL_NUM_PROMPTS="${EVAL_NUM_PROMPTS}" \
  NUM_SAMPLES_PER_PROMPT="${NUM_SAMPLES_PER_PROMPT}" \
  GEN_BATCH_SIZE="${GEN_BATCH_SIZE}" \
  WORK_CHUNK_SIZE="${WORK_CHUNK_SIZE}" \
  DECODE_BATCH_SIZE="${DECODE_BATCH_SIZE}" \
  OUTPUT="${out_file}" \
  "${ROOT_DIR}/07_full_pipeline.sh"

  metrics="$(JSON_ONLY=1 GENERATED_FILE="${out_file}" "${ROOT_DIR}/05_evaluate.sh")"
  valid_ratio="$(echo "${metrics}" | "${ROOT_DIR}/.mamba-tgmsd/bin/python" - <<'PY'
import json, sys
data = json.loads(sys.stdin.read())
print(data["valid_ratio"])
PY
)"
  unique_ratio="$(echo "${metrics}" | "${ROOT_DIR}/.mamba-tgmsd/bin/python" - <<'PY'
import json, sys
data = json.loads(sys.stdin.read())
print(data["unique_ratio_among_valid"])
PY
)"
  novelty_ratio="$(echo "${metrics}" | "${ROOT_DIR}/.mamba-tgmsd/bin/python" - <<'PY'
import json, sys
data = json.loads(sys.stdin.read())
print(data["novelty_ratio_unique_valid"])
PY
)"
  diversity="$(echo "${metrics}" | "${ROOT_DIR}/.mamba-tgmsd/bin/python" - <<'PY'
import json, sys
data = json.loads(sys.stdin.read())
print(data["internal_diversity_unique_valid"])
PY
)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${step_num}" "${model}" "${out_file}" "${valid_ratio}" "${unique_ratio}" "${novelty_ratio}" "${diversity}" >> "${SUMMARY_FILE}"
  count=$((count + 1))
done

echo "[done] sweep summary -> ${SUMMARY_FILE}"
