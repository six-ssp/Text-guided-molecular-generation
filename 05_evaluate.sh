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
INPUT="${INPUT:-}"
GENERATED_FILE="${GENERATED_FILE:-${INPUT:-${DATASET_DIR}/prompt_generated.tsv}}"
TRAIN_FILE="${TRAIN_FILE:-${DATASET_DIR}/train_pool90.txt}"
JSON_ONLY="${JSON_ONLY:-0}"
export GENERATED_FILE TRAIN_FILE JSON_ONLY

"${PYTHON_BIN}" - <<'PY'
import csv
import json
import os
import random
from pathlib import Path

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

generated_file = Path(os.environ["GENERATED_FILE"])
train_file = Path(os.environ["TRAIN_FILE"])
json_only = os.environ.get("JSON_ONLY", "0") == "1"

if not generated_file.exists():
    raise SystemExit(f"[fatal] generated file not found: {generated_file}")

def canon(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True), mol

rows = []
with generated_file.open() as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        rows.append(row)

train_set = set()
if train_file.exists():
    with train_file.open() as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            smi = parts[1].strip()
            if not smi or smi == "*":
                continue
            out = canon(smi)
            if out:
                train_set.add(out[0])

valid_canon = []
valid_mols = []
prompt_stats = {}
invalid_examples = []

for row in rows:
    smi = (row.get("generated_smiles") or "").strip()
    pid = row.get("prompt_id", "NA")
    if pid not in prompt_stats:
        prompt_stats[pid] = {"total": 0, "valid": 0, "canon": []}
    prompt_stats[pid]["total"] += 1

    out = canon(smi) if smi else None
    if out is None:
        if len(invalid_examples) < 5:
            invalid_examples.append(smi)
        continue
    c, mol = out
    valid_canon.append(c)
    valid_mols.append(mol)
    prompt_stats[pid]["valid"] += 1
    prompt_stats[pid]["canon"].append(c)

n = len(rows)
valid = len(valid_canon)
unique_valid_set = set(valid_canon)
unique_valid = len(unique_valid_set)

novel_unique = len([c for c in unique_valid_set if c not in train_set]) if train_set else None

diversity = None
if unique_valid >= 2:
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(c), 2, nBits=2048) for c in unique_valid_set]
    pairs = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            pairs.append((i, j))
    max_pairs = 20000
    if len(pairs) > max_pairs:
        random.seed(0)
        pairs = random.sample(pairs, max_pairs)
    dsum = 0.0
    for i, j in pairs:
        sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
        dsum += (1.0 - sim)
    diversity = dsum / len(pairs) if pairs else None

summary = {
    "generated_file": str(generated_file),
    "samples": n,
    "valid": valid,
    "valid_ratio": round(valid / n, 4) if n else 0.0,
    "unique_valid": unique_valid,
    "unique_ratio_among_valid": round(unique_valid / valid, 4) if valid else 0.0,
    "novel_unique_valid": novel_unique,
    "novelty_ratio_unique_valid": (
        round(novel_unique / unique_valid, 4)
        if (novel_unique is not None and unique_valid > 0)
        else None
    ),
    "internal_diversity_unique_valid": round(diversity, 4) if diversity is not None else None,
    "train_reference_size": len(train_set) if train_set else 0,
    "invalid_examples": invalid_examples,
}

if json_only:
    print(json.dumps(summary, ensure_ascii=False))
    raise SystemExit(0)

print(json.dumps(summary, ensure_ascii=False, indent=2))

if prompt_stats:
    print("\nper_prompt:")
    for pid in sorted(prompt_stats.keys(), key=lambda x: (x != "NA", x)):
        st = prompt_stats[pid]
        u = len(set(st["canon"]))
        print(
            json.dumps(
                {
                    "prompt_id": pid,
                    "total": st["total"],
                    "valid": st["valid"],
                    "valid_ratio": round(st["valid"] / st["total"], 4) if st["total"] else 0.0,
                    "unique_valid": u,
                    "unique_ratio_among_valid": round(u / st["valid"], 4) if st["valid"] else 0.0,
                },
                ensure_ascii=False,
            )
        )
PY
