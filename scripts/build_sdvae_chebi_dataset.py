#!/usr/bin/env python3
"""Build SD-VAE training tensors from ChEBI split files.

The original SD-VAE data builder expects a plain SMILES file and writes HDF5.
This project keeps the ChEBI CID/description metadata and writes a torch .pt
file, avoiding an h5py dependency in the local environment.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(root / "ChEBI-20_data"))
    parser.add_argument("--split", default="train_pool90")
    parser.add_argument("--output", default="")
    parser.add_argument("--metadata-output", default="")
    parser.add_argument("--skipped-output", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--sdvae-root", default=str(root / "sdvae"))
    parser.add_argument("--grammar-file", default=str(root / "sdvae/dropbox/context_free_grammars/mol_zinc.grammar"))
    parser.add_argument("--max-decode-steps", type=int, default=278)
    parser.add_argument("--latent-dim", type=int, default=56)
    return parser.parse_args()


def configure_sdvae_imports(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    sdvae_root = Path(args.sdvae_root).resolve()
    eval_dir = sdvae_root / "mol_vae" / "pytorch_eval"
    common_dir = sdvae_root / "mol_vae" / "mol_common"
    decoder_dir = sdvae_root / "mol_vae" / "mol_decoder"
    cfg_dir = sdvae_root / "mol_vae" / "cfg_parser"
    for path in [common_dir, decoder_dir, cfg_dir, eval_dir]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    # SD-VAE modules read global cmd_args during import.
    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    sys.argv = [
        "build_sdvae_chebi_dataset",
        "-grammar_file",
        str(Path(args.grammar_file).resolve()),
        "-max_decode_steps",
        str(args.max_decode_steps),
        "-latent_dim",
        str(args.latent_dim),
        "-mode",
        "cpu",
        "-encoder_type",
        "cnn",
        "-ae_type",
        "vae",
        "-rnn_type",
        "gru",
    ]
    try:
        import os

        os.chdir(sdvae_root / "mol_vae" / "pytorch_eval")
        import cfg_parser as parser
        from mol_decoder import batch_make_att_masks
        from mol_tree import AnnotatedTree2MolTree
    finally:
        sys.argv = old_argv
        import os

        os.chdir(old_cwd)
    return parser, AnnotatedTree2MolTree, batch_make_att_masks


def read_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            smiles = (row.get("SMILES") or "").strip()
            desc = (row.get("description") or "").strip()
            if not smiles or smiles == "*" or not desc:
                continue
            rows.append({"cid": int(row["CID"]), "smiles": smiles, "description": desc})
            if limit and len(rows) >= limit:
                break
    return rows


def atomic_torch_save(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    split_path = dataset_dir / f"{args.split}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"split file not found: {split_path}")

    output = Path(args.output) if args.output else dataset_dir / "sdvae_chebi" / f"{args.split}.pt"
    metadata_output = Path(args.metadata_output) if args.metadata_output else output.with_suffix(".metadata.tsv")
    skipped_output = Path(args.skipped_output) if args.skipped_output else output.with_suffix(".skipped.tsv")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    skipped_output.parent.mkdir(parents=True, exist_ok=True)

    parser, annotated_tree_to_mol_tree, batch_make_att_masks = configure_sdvae_imports(args)
    grammar = parser.Grammar(str(Path(args.grammar_file).resolve()))
    rows = read_rows(split_path, args.limit)

    kept_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    x_chunks = []
    mask_chunks = []
    iterator = range(0, len(rows), max(1, args.chunk_size))
    if tqdm is not None:
        iterator = tqdm(iterator, total=(len(rows) + args.chunk_size - 1) // args.chunk_size, desc=f"sdvae-data:{args.split}", unit="chunk")

    for start in iterator:
        chunk_rows = rows[start : start + args.chunk_size]
        cfg_trees = []
        ok_rows = []
        for row in chunk_rows:
            try:
                parsed = parser.parse(row["smiles"], grammar)
                assert isinstance(parsed, list) and len(parsed) == 1
                cfg_trees.append(annotated_tree_to_mol_tree(parsed[0]))
                ok_rows.append(row)
            except Exception as exc:
                bad = dict(row)
                bad["error"] = f"{type(exc).__name__}: {exc}"
                skipped_rows.append(bad)

        if not cfg_trees:
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                onehot, masks = batch_make_att_masks(cfg_trees, dtype=np.uint8)
        except Exception:
            # Fall back to single-row processing to keep good molecules.
            for row, tree in zip(ok_rows, cfg_trees):
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        onehot, masks = batch_make_att_masks([tree], dtype=np.uint8)
                    x_chunks.append(torch.from_numpy(onehot))
                    mask_chunks.append(torch.from_numpy(masks))
                    kept_rows.append(row)
                except Exception as exc:
                    bad = dict(row)
                    bad["error"] = f"{type(exc).__name__}: {exc}"
                    skipped_rows.append(bad)
            continue

        x_chunks.append(torch.from_numpy(onehot))
        mask_chunks.append(torch.from_numpy(masks))
        kept_rows.extend(ok_rows)

    x = torch.cat(x_chunks, dim=0) if x_chunks else torch.empty((0, args.max_decode_steps, 0), dtype=torch.uint8)
    masks = torch.cat(mask_chunks, dim=0) if mask_chunks else torch.empty_like(x)
    payload = {
        "x": x,
        "masks": masks,
        "rows": kept_rows,
        "split": args.split,
        "source_file": str(split_path),
        "grammar_file": str(Path(args.grammar_file).resolve()),
        "max_decode_steps": args.max_decode_steps,
        "latent_dim": args.latent_dim,
    }
    atomic_torch_save(payload, output)

    with metadata_output.open("w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=["cid", "smiles", "description"])
        writer.writeheader()
        writer.writerows(kept_rows)

    with skipped_output.open("w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=["cid", "smiles", "description", "error"])
        writer.writeheader()
        writer.writerows(skipped_rows)

    summary = {
        "split": args.split,
        "source_rows": len(rows),
        "kept_rows": len(kept_rows),
        "skipped_rows": len(skipped_rows),
        "output": str(output),
        "metadata_output": str(metadata_output),
        "skipped_output": str(skipped_output),
        "x_shape": list(x.shape),
        "masks_shape": list(masks.shape),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
