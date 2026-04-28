#!/usr/bin/env python3
"""Measure the reconstruction ceiling of the frozen SD-VAE decoder."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from sdvae_utils import decode_latents, load_proxy, project_root, read_split_rows, summarize_reconstruction


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(root / "ChEBI-20_data"))
    parser.add_argument("--split", default="test_pool90")
    parser.add_argument("--latent-file", default="", help="Optional existing cid->latent .pt file. Defaults to <split>_sdvae_latents.pt.")
    parser.add_argument("--prompt-file", default="", help="Optional prompt file to evaluate only those descriptions.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-json", default=str(root / "ChEBI-20_data" / "sdvae_reconstruction_metrics.json"))
    parser.add_argument("--decoded-tsv", default="")
    parser.add_argument("--decode-batch-size", type=int, default=4)
    parser.add_argument("--decode-random", action="store_true")
    parser.add_argument("--sdvae-root", default=str(root / "sdvae"))
    parser.add_argument("--saved_model", default=str(root / "sdvae" / "dropbox/results/zinc/zinc_kl_avg.model"))
    parser.add_argument("--grammar_file", default=str(root / "sdvae" / "dropbox/context_free_grammars/mol_zinc.grammar"))
    parser.add_argument("--mode", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--ae_type", default="vae")
    parser.add_argument("--encoder_type", default="cnn")
    parser.add_argument("--rnn_type", default="gru")
    parser.add_argument("--max_decode_steps", type=int, default=278)
    parser.add_argument("--latent_dim_sdvae", type=int, default=56)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    dataset_dir = Path(args.dataset_dir)
    prompt_file = Path(args.prompt_file) if args.prompt_file else None
    rows = read_split_rows(dataset_dir, args.split, prompt_file=prompt_file, limit=args.limit)
    latent_path = Path(args.latent_file) if args.latent_file else dataset_dir / f"{args.split}_sdvae_latents.pt"
    latents = torch.load(latent_path, map_location="cpu")

    usable_rows = []
    z_rows = []
    missing_cids = []
    for row in rows:
        cid = row["cid"]
        if cid not in latents:
            missing_cids.append(cid)
            continue
        usable_rows.append(row)
        z_rows.append(torch.as_tensor(latents[cid]).float().view(-1))

    proxy = load_proxy(root, args)
    decoded = decode_latents(proxy, torch.stack(z_rows), args.decode_batch_size, use_random=args.decode_random) if z_rows else []
    metrics = summarize_reconstruction(usable_rows, decoded)
    metrics.update(
        {
            "split": args.split,
            "latent_file": str(latent_path),
            "requested_rows": len(rows),
            "usable_rows": len(usable_rows),
            "missing_latents": len(missing_cids),
            "missing_cid_examples": missing_cids[:20],
            "decode_random": bool(args.decode_random),
        }
    )

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")

    if args.decoded_tsv:
        decoded_path = Path(args.decoded_tsv)
        decoded_path.parent.mkdir(parents=True, exist_ok=True)
        with decoded_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=["CID", "reference_smiles", "decoded_smiles", "description"])
            writer.writeheader()
            for row, pred in zip(usable_rows, decoded):
                writer.writerow(
                    {
                        "CID": row["cid"],
                        "reference_smiles": row["smiles"],
                        "decoded_smiles": pred,
                        "description": row["description"],
                    }
                )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
