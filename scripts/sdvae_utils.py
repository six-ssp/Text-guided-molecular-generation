#!/usr/bin/env python3
"""Shared helpers for local SD-VAE evaluation scripts."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MACCSkeys

RDLogger.DisableLog("rdApp.*")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def add_tgm_scripts_to_path(root: Path) -> None:
    script_dir = root / "tgm-dlm" / "improved-diffusion" / "scripts"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))


def load_proxy(root: Path, args: Any):
    add_tgm_scripts_to_path(root)
    from text_guided_generate import maybe_load_proxy

    class ProxyArgs:
        pass

    proxy_args = ProxyArgs()
    proxy_args.sdvae_root = str(Path(args.sdvae_root))
    proxy_args.saved_model = str(Path(args.saved_model))
    proxy_args.grammar_file = str(Path(args.grammar_file))
    proxy_args.mode = args.mode
    proxy_args.ae_type = args.ae_type
    proxy_args.encoder_type = args.encoder_type
    proxy_args.rnn_type = args.rnn_type
    proxy_args.max_decode_steps = args.max_decode_steps
    proxy_args.latent_dim_sdvae = args.latent_dim_sdvae
    return maybe_load_proxy(proxy_args)


def read_split_rows(dataset_dir: Path, split: str, prompt_file: Path | None = None, limit: int = 0) -> list[dict[str, Any]]:
    wanted_prompts = None
    if prompt_file is not None:
        wanted_prompts = {line.strip() for line in prompt_file.open() if line.strip()}

    rows: list[dict[str, Any]] = []
    split_path = dataset_dir / f"{split}.txt"
    with split_path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            smiles = (row.get("SMILES") or "").strip()
            desc = (row.get("description") or "").strip()
            if not smiles or smiles == "*" or not desc:
                continue
            if wanted_prompts is not None and desc not in wanted_prompts:
                continue
            rows.append({"cid": int(row["CID"]), "smiles": smiles, "description": desc})
            if limit and len(rows) >= limit:
                break
    return rows


def canonical_smiles(smiles: str) -> tuple[str | None, Any | None]:
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None, None
    return Chem.MolToSmiles(mol, canonical=True), mol


def levenshtein(a: str, b: str) -> int:
    try:
        from Levenshtein import distance

        return int(distance(a, b))
    except Exception:
        if a == b:
            return 0
        if len(a) < len(b):
            a, b = b, a
        previous = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            current = [i]
            for j, cb in enumerate(b, 1):
                current.append(min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
            previous = current
        return previous[-1]


def fingerprint_metrics(ref: str, pred: str) -> dict[str, float | None]:
    ref_canon, ref_mol = canonical_smiles(ref)
    pred_canon, pred_mol = canonical_smiles(pred)
    if ref_mol is None or pred_mol is None:
        return {"maccs": None, "rdkit": None, "morgan": None}
    return {
        "maccs": float(DataStructs.TanimotoSimilarity(MACCSkeys.GenMACCSKeys(ref_mol), MACCSkeys.GenMACCSKeys(pred_mol))),
        "rdkit": float(DataStructs.TanimotoSimilarity(Chem.RDKFingerprint(ref_mol), Chem.RDKFingerprint(pred_mol))),
        "morgan": float(DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(ref_mol, 2), AllChem.GetMorganFingerprint(pred_mol, 2))),
    }


def summarize_reconstruction(rows: list[dict[str, Any]], decoded: list[str]) -> dict[str, Any]:
    total = len(rows)
    valid = 0
    exact_raw = 0
    exact_canon = 0
    levs: list[int] = []
    maccs: list[float] = []
    rdk: list[float] = []
    morgan: list[float] = []

    for row, pred in zip(rows, decoded):
        ref = row["smiles"]
        ref_canon, ref_mol = canonical_smiles(ref)
        pred_canon, pred_mol = canonical_smiles(pred)
        if pred_mol is not None:
            valid += 1
        if pred == ref:
            exact_raw += 1
        if pred_canon is not None and pred_canon == ref_canon:
            exact_canon += 1
        levs.append(levenshtein(pred, ref))
        sims = fingerprint_metrics(ref, pred)
        if sims["maccs"] is not None:
            maccs.append(float(sims["maccs"]))
            rdk.append(float(sims["rdkit"]))
            morgan.append(float(sims["morgan"]))

    def mean(values: list[float | int]) -> float | None:
        return float(np.mean(values)) if values else None

    return {
        "samples": total,
        "valid": valid,
        "validity": valid / total if total else 0.0,
        "exact_match_raw_smiles": exact_raw / total if total else 0.0,
        "exact_match_canonical_smiles": exact_canon / total if total else 0.0,
        "levenshtein_distance": mean(levs),
        "maccs_fingerprint_similarity": mean(maccs),
        "rdkit_fingerprint_similarity": mean(rdk),
        "morgan_fingerprint_similarity": mean(morgan),
    }


def decode_latents(proxy: Any, latents: torch.Tensor, decode_batch_size: int, use_random: bool = False) -> list[str]:
    add_tgm_scripts_to_path(project_root())
    from text_guided_generate import decode_latents_in_chunks

    return decode_latents_in_chunks(
        proxy,
        latents.detach().cpu().numpy().astype(np.float32),
        decode_batch_size=decode_batch_size,
        use_random=use_random,
    )
