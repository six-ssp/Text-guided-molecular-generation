#!/usr/bin/env python3
"""Train or fine-tune SD-VAE on ChEBI SD-VAE tensor dumps."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default=str(root / "ChEBI-20_data/sdvae_chebi/train_pool90.pt"))
    parser.add_argument("--valid-data", default=str(root / "ChEBI-20_data/sdvae_chebi/validation_pool90.pt"))
    parser.add_argument("--save-dir", default=str(root / "sdvae/dropbox/results/chebi_pool90"))
    parser.add_argument("--init-model", default=str(root / "sdvae/dropbox/results/zinc/zinc_kl_avg.model"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--kl-coeff", type=float, default=1.0)
    parser.add_argument("--eps-std", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=19260817)
    parser.add_argument("--mode", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--sdvae-root", default=str(root / "sdvae"))
    parser.add_argument("--grammar-file", default=str(root / "sdvae/dropbox/context_free_grammars/mol_zinc.grammar"))
    parser.add_argument("--max-decode-steps", type=int, default=278)
    parser.add_argument("--latent-dim", type=int, default=56)
    parser.add_argument("--ae-type", choices=["vae", "autoenc"], default="vae")
    parser.add_argument("--encoder-type", default="cnn")
    parser.add_argument("--rnn-type", default="gru")
    parser.add_argument("--loss-type", default="perplexity")
    parser.add_argument("--eval-batches", type=int, default=0, help="Limit validation batches for faster smoke runs; 0 means full validation.")
    return parser.parse_args()


def resolve_mode(mode: str) -> str:
    if mode == "auto":
        return "gpu" if torch.cuda.is_available() else "cpu"
    if mode == "gpu" and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; falling back to cpu")
        return "cpu"
    return mode


def configure_sdvae_imports(args: argparse.Namespace):
    sdvae_root = Path(args.sdvae_root).resolve()
    for rel in ["mol_vae/mol_common", "mol_vae/mol_vae", "mol_vae/mol_decoder", "mol_vae/mol_encoder", "mol_vae/cfg_parser"]:
        path = sdvae_root / rel
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    sys.argv = [
        "train_sdvae_chebi",
        "-mode",
        args.mode,
        "-save_dir",
        str(Path(args.save_dir).resolve()),
        "-saved_model",
        str(Path(args.init_model).resolve()) if args.init_model else "",
        "-grammar_file",
        str(Path(args.grammar_file).resolve()),
        "-encoder_type",
        args.encoder_type,
        "-ae_type",
        args.ae_type,
        "-rnn_type",
        args.rnn_type,
        "-loss_type",
        args.loss_type,
        "-max_decode_steps",
        str(args.max_decode_steps),
        "-batch_size",
        str(args.batch_size),
        "-latent_dim",
        str(args.latent_dim),
        "-num_epochs",
        str(args.epochs),
        "-learning_rate",
        str(args.learning_rate),
        "-kl_coeff",
        str(args.kl_coeff),
        "-eps_std",
        str(args.eps_std),
    ]
    try:
        import os

        os.chdir(sdvae_root / "mol_vae" / "pytorch_eval")
        from mol_vae import MolAutoEncoder, MolVAE
    finally:
        sys.argv = old_argv
        import os

        os.chdir(old_cwd)
    return MolAutoEncoder, MolVAE


def load_tensor_dump(path: Path) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    payload = torch.load(path, map_location="cpu")
    x = payload["x"].to(torch.uint8).contiguous()
    masks = payload["masks"].to(torch.uint8).contiguous()
    rows = payload.get("rows", [])
    if x.shape != masks.shape:
        raise ValueError(f"x/masks shape mismatch: {x.shape} vs {masks.shape}")
    return x, masks, rows


def batch_tensors(indices: list[int], x: torch.Tensor, masks: torch.Tensor, mode: str) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    x_np = x[indices].numpy().astype(np.float32)
    masks_np = masks[indices].numpy().astype(np.float32)
    true_binary_np = np.transpose(x_np, [1, 0, 2])
    rule_masks_np = np.transpose(masks_np, [1, 0, 2])
    x_inputs = np.transpose(true_binary_np, [1, 2, 0])
    true_binary = torch.from_numpy(true_binary_np)
    rule_masks = torch.from_numpy(rule_masks_np)
    if mode == "gpu":
        true_binary = true_binary.cuda(non_blocking=True)
        rule_masks = rule_masks.cuda(non_blocking=True)
    return x_inputs, true_binary, rule_masks


def masked_action_loss(logits: torch.Tensor, true_binary: torch.Tensor, rule_masks: torch.Tensor) -> torch.Tensor:
    target = true_binary.argmax(dim=-1)
    active = (rule_masks[..., -1] < 0.5).float()
    masked_logits = logits.masked_fill(rule_masks <= 0, -1e4)
    loss = F.cross_entropy(
        masked_logits.reshape(-1, masked_logits.shape[-1]),
        target.reshape(-1),
        reduction="none",
    ).view_as(active)
    return (loss * active).sum() / active.sum().clamp_min(1.0)


def model_loss(ae, x_inputs: np.ndarray, true_binary: torch.Tensor, rule_masks: torch.Tensor, ae_type: str, kl_coeff: float) -> tuple[torch.Tensor, float, float]:
    z_mean, z_log_var = ae.encoder(x_inputs)
    if ae_type == "vae":
        z = ae.reparameterize(z_mean, z_log_var)
    else:
        z = z_mean
    logits = ae.state_decoder(z, true_binary.shape[0])
    recon = masked_action_loss(logits, true_binary, rule_masks)
    kl = torch.zeros((), device=recon.device)
    if ae_type == "vae":
        kl_values = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - torch.exp(z_log_var), dim=-1)
        kl = kl_values.mean()
    loss = recon + kl_coeff * kl
    return loss, float(recon.detach().cpu()), float(kl.detach().cpu())


def run_epoch(name: str, ae, x: torch.Tensor, masks: torch.Tensor, args: argparse.Namespace, optimizer=None) -> dict[str, float]:
    is_train = optimizer is not None
    ae.train(is_train)
    indices = list(range(x.shape[0]))
    if is_train:
        random.shuffle(indices)
    batches = [indices[i : i + args.batch_size] for i in range(0, len(indices), args.batch_size)]
    if not is_train and args.eval_batches > 0:
        batches = batches[: args.eval_batches]
    iterator = batches
    if tqdm is not None:
        iterator = tqdm(batches, desc=name, unit="batch")

    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "n": 0}
    for batch in iterator:
        x_inputs, true_binary, rule_masks = batch_tensors(batch, x, masks, args.mode)
        with torch.set_grad_enabled(is_train):
            loss, recon, kl = model_loss(ae, x_inputs, true_binary, rule_masks, args.ae_type, args.kl_coeff)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(ae.parameters(), args.grad_clip)
            optimizer.step()
        batch_n = len(batch)
        totals["loss"] += float(loss.detach().cpu()) * batch_n
        totals["recon"] += recon * batch_n
        totals["kl"] += kl * batch_n
        totals["n"] += batch_n
        if tqdm is not None:
            iterator.set_postfix(loss=totals["loss"] / totals["n"], recon=totals["recon"] / totals["n"], kl=totals["kl"] / totals["n"])

    n = max(int(totals["n"]), 1)
    return {k: (v / n if k != "n" else v) for k, v in totals.items()}


def main() -> None:
    args = parse_args()
    args.mode = resolve_mode(args.mode)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    MolAutoEncoder, MolVAE = configure_sdvae_imports(args)
    ae = MolVAE() if args.ae_type == "vae" else MolAutoEncoder()
    if args.mode == "gpu":
        ae = ae.cuda()

    init_model = Path(args.init_model) if args.init_model else None
    if init_model and init_model.exists():
        print(f"loading init model: {init_model}")
        state = torch.load(init_model, map_location="cpu" if args.mode == "cpu" else None)
        ae.load_state_dict(state)
    elif init_model:
        print(f"[warn] init model not found, training from scratch: {init_model}")

    train_x, train_masks, train_rows = load_tensor_dump(Path(args.train_data))
    valid_x, valid_masks, valid_rows = load_tensor_dump(Path(args.valid_data))
    print(json.dumps({
        "train_data": args.train_data,
        "valid_data": args.valid_data,
        "train_rows": len(train_rows) or int(train_x.shape[0]),
        "valid_rows": len(valid_rows) or int(valid_x.shape[0]),
        "train_shape": list(train_x.shape),
        "valid_shape": list(valid_x.shape),
        "mode": args.mode,
        "init_model": str(init_model) if init_model else "",
        "save_dir": str(save_dir),
    }, indent=2))

    optimizer = optim.Adam(ae.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=3, min_lr=1e-5)
    best_valid = None
    history = []

    for epoch in range(args.epochs):
        train_metrics = run_epoch(f"train:{epoch}", ae, train_x, train_masks, args, optimizer=optimizer)
        valid_metrics = run_epoch(f"valid:{epoch}", ae, valid_x, valid_masks, args, optimizer=None)
        scheduler.step(valid_metrics["loss"])
        record = {"epoch": epoch, "train": train_metrics, "valid": valid_metrics, "lr": optimizer.param_groups[0]["lr"]}
        history.append(record)
        print(json.dumps(record, indent=2), flush=True)

        torch.save(ae.state_dict(), save_dir / f"epoch-{epoch}.model")
        if best_valid is None or valid_metrics["loss"] < best_valid:
            best_valid = valid_metrics["loss"]
            torch.save(ae.state_dict(), save_dir / "epoch-best.model")
            print(f"saved best model at epoch {epoch}: valid_loss={best_valid:.6f}", flush=True)
        (save_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")


if __name__ == "__main__":
    main()
