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

INPUT="${INPUT:-${ROOT_DIR}/ChEBI-20_data/prompt_generated_large.tsv}"
OUTPUT="${OUTPUT:-${INPUT%.tsv}_legal.tsv}"
SUMMARY_JSON="${SUMMARY_JSON:-${OUTPUT}.summary.json}"
export INPUT OUTPUT SUMMARY_JSON

"${PYTHON_BIN}" - <<'PY'
import csv
import json
import os
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

input_path = Path(os.environ["INPUT"])
output_path = Path(os.environ["OUTPUT"])
summary_path = Path(os.environ["SUMMARY_JSON"])

if not input_path.exists():
    raise SystemExit(f"[fatal] input file not found: {input_path}")

with input_path.open() as f:
    reader = csv.DictReader(f, delimiter="\t")
    rows = list(reader)
    fieldnames = reader.fieldnames

kept = []
for row in rows:
    smi = (row.get("generated_smiles") or "").strip()
    if smi and Chem.MolFromSmiles(smi) is not None:
        kept.append(row)

with output_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(kept)

summary = {
    "input": str(input_path),
    "output": str(output_path),
    "total": len(rows),
    "kept": len(kept),
    "kept_ratio": round(len(kept) / len(rows), 4) if rows else 0.0,
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
