#!/usr/bin/env python3
"""Evaluate text-guided molecule generation against ChEBI-style references."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MACCSkeys

RDLogger.DisableLog("rdApp.*")

try:
    from Levenshtein import distance as fast_levenshtein
except ImportError:
    fast_levenshtein = None


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "ChEBI-20_data"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generated",
        default=str(data_dir / "prompt_generated_testpool90_full_3021x8.tsv"),
        help="Generated TSV with prompt and generated_smiles columns.",
    )
    parser.add_argument(
        "--reference",
        default=str(data_dir / "test_pool90.txt"),
        help="Reference TSV with SMILES and description columns.",
    )
    parser.add_argument(
        "--json-out",
        default=str(data_dir / "text_guided_9metrics_full_3021x8.json"),
        help="Where to write JSON metrics.",
    )
    parser.add_argument(
        "--text2mol-json",
        default="",
        help="Optional JSON file containing a precomputed Text2Mol score.",
    )
    parser.add_argument(
        "--compute-fcd",
        action="store_true",
        help="Try to compute FCD if the optional fcd package is installed.",
    )
    return parser.parse_args()


def read_reference(path: Path) -> tuple[dict[str, str], list[str]]:
    desc_to_smiles: dict[str, str] = {}
    warnings: list[str] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            desc = (row.get("description") or "").strip()
            smi = (row.get("SMILES") or "").strip()
            if not desc or not smi or smi == "*":
                continue
            old = desc_to_smiles.get(desc)
            if old and old != smi:
                warnings.append(f"duplicate description with different SMILES: {desc[:80]}")
                continue
            desc_to_smiles[desc] = smi
    return desc_to_smiles, warnings


@lru_cache(maxsize=None)
def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    if not smiles:
        return None
    return Chem.MolFromSmiles(smiles)


@lru_cache(maxsize=None)
def canonical_from_smiles(smiles: str) -> str | None:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


@lru_cache(maxsize=None)
def inchi_from_smiles(smiles: str) -> str | None:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToInchi(mol)
    except Exception:
        return None


@lru_cache(maxsize=None)
def maccs_fp(smiles: str) -> Any:
    return MACCSkeys.GenMACCSKeys(mol_from_smiles(smiles))


@lru_cache(maxsize=None)
def rdk_fp(smiles: str) -> Any:
    return Chem.RDKFingerprint(mol_from_smiles(smiles))


@lru_cache(maxsize=None)
def morgan_fp(smiles: str) -> Any:
    return AllChem.GetMorganFingerprint(mol_from_smiles(smiles), 2)


def levenshtein(a: str, b: str) -> int:
    if fast_levenshtein is not None:
        return int(fast_levenshtein(a, b))
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (ca != cb)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def tanimoto_or_nan(fp_a: Any, fp_b: Any) -> float:
    try:
        return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))
    except Exception:
        return math.nan


def safe_mean(values: list[float]) -> float | None:
    arr = np.array([v for v in values if not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return None
    return float(arr.mean())


def build_records(generated_path: Path, desc_to_smiles: dict[str, str]) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    with generated_path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            prompt = (row.get("prompt") or "").strip()
            ref = desc_to_smiles.get(prompt)
            if ref is None:
                warnings.append(f"missing reference for prompt_id={row.get('prompt_id', 'NA')}")
                continue
            gen = (row.get("generated_smiles") or "").strip()
            ref_mol = mol_from_smiles(ref)
            gen_mol = mol_from_smiles(gen)
            records.append(
                {
                    "prompt_id": row.get("prompt_id", ""),
                    "sample_idx": row.get("sample_idx", ""),
                    "prompt": prompt,
                    "ref": ref,
                    "gen": gen,
                    "ref_mol": ref_mol,
                    "gen_mol": gen_mol,
                    "ref_canon": canonical_from_smiles(ref),
                    "gen_canon": canonical_from_smiles(gen),
                    "ref_inchi": inchi_from_smiles(ref),
                    "gen_inchi": inchi_from_smiles(gen),
                }
            )
    return records, warnings


def choose_best_by_prompt(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["prompt_id"])].append(record)

    best_records = []
    for _, items in sorted(grouped.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
        scored = []
        for item in items:
            score = -1.0
            if item["ref_mol"] is not None and item["gen_mol"] is not None:
                score = tanimoto_or_nan(morgan_fp(item["ref"]), morgan_fp(item["gen"]))
            scored.append((score if not math.isnan(score) else -1.0, item))
        best_records.append(max(scored, key=lambda x: x[0])[1])
    return best_records


def compute_metrics(records: list[dict[str, Any]], compute_fcd: bool, text2mol_json: Path | None) -> dict[str, Any]:
    total = len(records)
    valid_records = [r for r in records if r["gen_mol"] is not None]

    references = [[[c for c in r["ref"]]] for r in records]
    hypotheses = [[c for c in r["gen"]] for r in records]
    bleu = corpus_bleu(references, hypotheses) if records else 0.0

    levs = [levenshtein(r["gen"], r["ref"]) for r in records]
    exact_raw = sum(1 for r in records if r["gen"] == r["ref"])
    exact_canon = sum(
        1 for r in records if r["gen_canon"] is not None and r["gen_canon"] == r["ref_canon"]
    )
    exact_inchi = sum(
        1 for r in records if r["gen_inchi"] is not None and r["gen_inchi"] == r["ref_inchi"]
    )

    maccs_sims: list[float] = []
    rdk_sims: list[float] = []
    morgan_sims: list[float] = []
    for r in valid_records:
        if r["ref_mol"] is None:
            continue
        maccs_sims.append(tanimoto_or_nan(maccs_fp(r["ref"]), maccs_fp(r["gen"])))
        rdk_sims.append(tanimoto_or_nan(rdk_fp(r["ref"]), rdk_fp(r["gen"])))
        morgan_sims.append(tanimoto_or_nan(morgan_fp(r["ref"]), morgan_fp(r["gen"])))

    text2mol_score = None
    if text2mol_json and text2mol_json.exists():
        with text2mol_json.open() as f:
            loaded = json.load(f)
        text2mol_score = loaded.get("text2mol_score", loaded.get("Text2Mol", loaded.get("score")))

    fcd_score = None
    fcd_status = "not_requested"
    if compute_fcd:
        try:
            from fcd import canonical_smiles as fcd_canonical_smiles
            from fcd import get_fcd, load_ref_model

            model = load_ref_model()
            ref_smis = [r["ref"] for r in records]
            gen_smis = [r["gen"] if r["gen"] else "[]" for r in records]
            canon_ref = [s for s in fcd_canonical_smiles(ref_smis) if s is not None]
            canon_gen = [s for s in fcd_canonical_smiles(gen_smis) if s is not None]
            fcd_score = float(get_fcd(canon_ref, canon_gen, model))
            fcd_status = "computed"
        except Exception as exc:
            fcd_status = f"unavailable: {type(exc).__name__}: {exc}"

    return {
        "num_samples": total,
        "num_valid": len(valid_records),
        "validity": len(valid_records) / total if total else 0.0,
        "bleu": float(bleu),
        "exact_match_raw_smiles": exact_raw / total if total else 0.0,
        "exact_match_canonical_smiles": exact_canon / total if total else 0.0,
        "exact_match_inchi": exact_inchi / total if total else 0.0,
        "levenshtein_distance": float(np.mean(levs)) if levs else None,
        "maccs_fingerprint_similarity": safe_mean(maccs_sims),
        "rdkit_fingerprint_similarity": safe_mean(rdk_sims),
        "morgan_fingerprint_similarity": safe_mean(morgan_sims),
        "fcd": fcd_score,
        "fcd_status": fcd_status,
        "text2mol_score": text2mol_score,
        "text2mol_status": "loaded" if text2mol_score is not None else "not_computed_no_local_text2mol_model",
    }


def rounded(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: rounded(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rounded(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


def main() -> None:
    args = parse_args()
    generated = Path(args.generated)
    reference = Path(args.reference)
    json_out = Path(args.json_out)
    text2mol_json = Path(args.text2mol_json) if args.text2mol_json else None

    desc_to_smiles, ref_warnings = read_reference(reference)
    records, gen_warnings = build_records(generated, desc_to_smiles)
    best_records = choose_best_by_prompt(records)

    result = {
        "generated_file": str(generated),
        "reference_file": str(reference),
        "reference_prompts": len(desc_to_smiles),
        "matched_samples": len(records),
        "warnings": {
            "reference": ref_warnings[:20],
            "generated": gen_warnings[:20],
            "num_reference_warnings": len(ref_warnings),
            "num_generated_warnings": len(gen_warnings),
        },
        "all_samples": compute_metrics(records, args.compute_fcd, text2mol_json),
        "best_of_8_by_prompt_morgan": compute_metrics(best_records, args.compute_fcd, text2mol_json),
    }

    json_out.parent.mkdir(parents=True, exist_ok=True)
    with json_out.open("w") as f:
        json.dump(rounded(result), f, indent=2, ensure_ascii=False)

    print(json.dumps(rounded(result), indent=2, ensure_ascii=False))
    print(f"\nwrote {json_out}")


if __name__ == "__main__":
    main()
