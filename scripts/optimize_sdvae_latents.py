#!/usr/bin/env python3
"""Optimize per-molecule SD-VAE latents against the frozen decoder.

This is a latent inversion step. It keeps the SD-VAE decoder fixed and updates
only z so the decoder assigns higher probability to the target grammar actions.
The resulting cid->latent file can be used by train_sdvae_latent.py via LATENT_FILE.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from sdvae_utils import decode_latents, load_proxy, project_root, read_split_rows, summarize_reconstruction


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(root / "ChEBI-20_data"))
    parser.add_argument("--split", default="train_pool90")
    parser.add_argument("--prompt-file", default="", help="Optional prompt file to optimize only matching descriptions.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-latents", default="", help="Defaults to <split>_sdvae_latents_inverted.pt")
    parser.add_argument("--decoded-tsv", default="", help="Optional TSV with deterministic decode of optimized latents.")
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--z-l2", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--save-every", type=int, default=64, help="Save after this many newly optimized rows; 0 saves every batch.")
    parser.add_argument("--decode-batch-size", type=int, default=2)
    parser.add_argument("--final-eval-limit", type=int, default=512, help="Decode only this many optimized rows for final summary; 0 decodes all rows.")
    parser.add_argument("--skip-final-decode", action="store_true", help="Skip final deterministic decode/summary; useful for long train split inversion.")
    parser.add_argument("--resume", action="store_true")
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


def import_sdvae_runtime() -> tuple[Any, Any]:
    from att_model_proxy import parse as parse_smiles
    from mol_decoder import batch_make_att_masks

    return parse_smiles, batch_make_att_masks


def prepare_batch(proxy: Any, rows: list[dict[str, Any]], parse_smiles: Any, batch_make_att_masks: Any) -> tuple[list[dict[str, Any]], torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, list[dict[str, Any]]]:
    if not rows:
        return [], None, None, None, []
    smiles = [row["smiles"] for row in rows]
    try:
        cfg_trees = parse_smiles(smiles, proxy.grammar)
        true_binary, rule_masks = batch_make_att_masks(
            cfg_trees,
            proxy.tree_decoder,
            proxy.onehot_walker,
            dtype=np.float32,
        )
        x_inputs = np.transpose(true_binary, [0, 2, 1]).astype(np.float32)
        proxy.ae.eval()
        with torch.no_grad():
            z_init, _ = proxy.ae.encoder(x_inputs)
        device = z_init.device
        true_t = torch.from_numpy(np.transpose(true_binary, [1, 0, 2])).float().to(device)
        mask_t = torch.from_numpy(np.transpose(rule_masks, [1, 0, 2])).float().to(device)
        return rows, z_init.detach(), true_t, mask_t, []
    except Exception as exc:
        if len(rows) == 1:
            bad = dict(rows[0])
            bad["error"] = f"{type(exc).__name__}: {exc}"
            return [], None, None, None, [bad]
        mid = len(rows) // 2
        left = prepare_batch(proxy, rows[:mid], parse_smiles, batch_make_att_masks)
        right = prepare_batch(proxy, rows[mid:], parse_smiles, batch_make_att_masks)
        ok_rows = left[0] + right[0]
        skipped = left[4] + right[4]
        if not ok_rows:
            return [], None, None, None, skipped
        z_parts = [x for x in (left[1], right[1]) if x is not None]
        true_parts = [x for x in (left[2], right[2]) if x is not None]
        mask_parts = [x for x in (left[3], right[3]) if x is not None]
        return ok_rows, torch.cat(z_parts, dim=0), torch.cat(true_parts, dim=1), torch.cat(mask_parts, dim=1), skipped


def masked_action_loss(logits: torch.Tensor, true_binary: torch.Tensor, rule_masks: torch.Tensor) -> torch.Tensor:
    target = true_binary.argmax(dim=-1)
    active = (rule_masks[..., -1] < 0.5).float()
    masked_logits = logits.masked_fill(rule_masks <= 0, -1e4)
    loss = F.cross_entropy(
        masked_logits.reshape(-1, masked_logits.shape[-1]),
        target.reshape(-1),
        reduction="none",
    ).view_as(active)
    denom = active.sum().clamp_min(1.0)
    return (loss * active).sum() / denom


def optimize_batch(proxy: Any, z_init: torch.Tensor, true_t: torch.Tensor, mask_t: torch.Tensor, steps: int, lr: float, z_l2: float, grad_clip: float) -> tuple[torch.Tensor, float]:
    for param in proxy.ae.parameters():
        param.requires_grad_(False)
    # cuDNN RNN backward requires training mode even when only z is updated.
    # Parameters stay frozen, so this does not fine-tune the SD-VAE decoder.
    proxy.ae.train()
    z = z_init.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    best_z = z.detach().clone()
    best_loss = float("inf")

    for _ in range(max(0, steps)):
        opt.zero_grad(set_to_none=True)
        logits = proxy.ae.state_decoder(z, true_t.shape[0])
        loss = masked_action_loss(logits, true_t, mask_t)
        if z_l2 > 0:
            loss = loss + z_l2 * (z - z_init).pow(2).mean()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([z], grad_clip)
        opt.step()
        value = float(loss.detach().cpu())
        if value < best_loss:
            best_loss = value
            best_z = z.detach().clone()

    if steps == 0:
        with torch.no_grad():
            best_loss = float(masked_action_loss(proxy.ae.state_decoder(z_init, true_t.shape[0]), true_t, mask_t).detach().cpu())
            best_z = z_init.detach().clone()
    return best_z.detach().cpu(), best_loss


def atomic_torch_save(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    root = project_root()
    dataset_dir = Path(args.dataset_dir)
    prompt_file = Path(args.prompt_file) if args.prompt_file else None
    output_latents = Path(args.output_latents) if args.output_latents else dataset_dir / f"{args.split}_sdvae_latents_inverted.pt"
    summary_json = Path(args.summary_json) if args.summary_json else output_latents.with_suffix(".summary.json")
    decoded_tsv = Path(args.decoded_tsv) if args.decoded_tsv else output_latents.with_suffix(".decoded.tsv")
    output_latents.parent.mkdir(parents=True, exist_ok=True)

    rows = read_split_rows(dataset_dir, args.split, prompt_file=prompt_file, limit=args.limit)
    proxy = load_proxy(root, args)
    parse_smiles, batch_make_att_masks = import_sdvae_runtime()

    optimized: dict[int, torch.Tensor] = {}
    if args.resume and output_latents.exists():
        optimized = torch.load(output_latents, map_location="cpu")
        print(f"resume from {output_latents}, loaded {len(optimized)} latents")

    skipped: list[dict[str, Any]] = []
    losses: list[float] = []
    total = len(rows)
    run_start = time.time()
    last_saved_count = len(optimized)
    for start in range(0, total, max(1, args.batch_size)):
        chunk = [row for row in rows[start : start + args.batch_size] if row["cid"] not in optimized]
        if not chunk:
            continue
        ok_rows, z_init, true_t, mask_t, bad = prepare_batch(proxy, chunk, parse_smiles, batch_make_att_masks)
        skipped.extend(bad)
        if not ok_rows or z_init is None or true_t is None or mask_t is None:
            continue
        z_best, loss = optimize_batch(proxy, z_init, true_t, mask_t, args.steps, args.lr, args.z_l2, args.grad_clip)
        for row, latent in zip(ok_rows, z_best):
            optimized[row["cid"]] = latent.float().cpu()
            losses.append(loss)
        done = min(start + args.batch_size, total)
        elapsed = max(time.time() - run_start, 1e-6)
        current_run_rows = max(len(optimized) - last_saved_count, 0)
        rows_per_min = len(losses) / elapsed * 60.0
        print(
            f"optimized {len(optimized)}/{total} rows | last_batch_loss={loss:.4f} "
            f"| skipped={len(skipped)} | run_rows_per_min={rows_per_min:.2f}",
            flush=True,
        )
        if args.save_every == 0 or len(optimized) - last_saved_count >= args.save_every:
            atomic_torch_save(optimized, output_latents)
            last_saved_count = len(optimized)

    atomic_torch_save(optimized, output_latents)

    eval_rows_all = [row for row in rows if row["cid"] in optimized]
    if args.skip_final_decode:
        eval_rows: list[dict[str, Any]] = []
        decoded: list[str] = []
        metrics = {
            "samples": 0,
            "valid": 0,
            "validity": None,
            "exact_match_raw_smiles": None,
            "exact_match_canonical_smiles": None,
            "levenshtein_distance": None,
            "maccs_fingerprint_similarity": None,
            "rdkit_fingerprint_similarity": None,
            "morgan_fingerprint_similarity": None,
            "final_decode_skipped": True,
        }
    else:
        eval_rows = eval_rows_all[: args.final_eval_limit] if args.final_eval_limit > 0 else eval_rows_all
        z_eval = torch.stack([optimized[row["cid"]].float().view(-1) for row in eval_rows]) if eval_rows else torch.empty(0, args.latent_dim_sdvae)
        decoded = decode_latents(proxy, z_eval, args.decode_batch_size, use_random=False) if eval_rows else []
        metrics = summarize_reconstruction(eval_rows, decoded)

    skipped_path = output_latents.with_suffix(".skipped.tsv")
    if skipped:
        with skipped_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=["cid", "smiles", "description", "error"])
            writer.writeheader()
            for row in skipped:
                writer.writerow(row)
        metrics["skipped_file"] = str(skipped_path)

    metrics.update(
        {
            "split": args.split,
            "output_latents": str(output_latents),
            "requested_rows": len(rows),
            "optimized_rows": len(optimized),
            "skipped_rows": len(skipped),
            "evaluated_rows": len(eval_rows),
            "final_eval_limit": args.final_eval_limit,
            "mean_optimization_loss": float(np.mean(losses)) if losses else None,
            "steps": args.steps,
            "lr": args.lr,
            "z_l2": args.z_l2,
        }
    )
    summary_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")

    if not args.skip_final_decode:
        with decoded_tsv.open("w", newline="") as f:
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=["CID", "reference_smiles", "decoded_smiles", "description"])
            writer.writeheader()
            for row, pred in zip(eval_rows, decoded):
                writer.writerow({"CID": row["cid"], "reference_smiles": row["smiles"], "decoded_smiles": pred, "description": row["description"]})

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
