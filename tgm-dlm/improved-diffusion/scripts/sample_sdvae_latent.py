import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import set_seed
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from improved_diffusion import dist_util, logger
from improved_diffusion import gaussian_diffusion as gd
from improved_diffusion.latent_model import LatentConditionedMLP
from improved_diffusion.respace import SpacedDiffusion
from mydatasets import ChEBIdataset
from mytokenizers import regexTokenizer


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="../../datasets/SMILES")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", default="../../sdvae_latent_samples.txt")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--diffusion-steps", type=int, default=2000)
    parser.add_argument("--noise-schedule", default="sqrt")
    parser.add_argument("--latent-dim", type=int, default=56)
    parser.add_argument("--model-channels", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--text-fusion", choices=["pooled", "crossattn"], default="pooled")
    parser.add_argument("--text-attn-heads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=121)
    parser.add_argument("--use-ddim", action="store_true")
    parser.add_argument("--sdvae-root", default="../../../sdvae")
    parser.add_argument("-saved_model", "--saved_model", default=None)
    parser.add_argument("-grammar_file", "--grammar_file", default=None)
    parser.add_argument("-mode", "--mode", default="gpu")
    parser.add_argument("-ae_type", "--ae_type", default="vae")
    parser.add_argument("-encoder_type", "--encoder_type", default="cnn")
    parser.add_argument("-rnn_type", "--rnn_type", default="gru")
    parser.add_argument("-max_decode_steps", "--max_decode_steps", type=int, default=278)
    parser.add_argument("-latent_dim_sdvae", "--latent_dim_sdvae", type=int, default=56)
    return parser


SCRIPT_DIR = Path(__file__).resolve().parent


def maybe_load_proxy(args):
    if not args.saved_model or not args.grammar_file:
        return None
    sdvae_mode = args.mode
    if sdvae_mode == "auto":
        sdvae_mode = "gpu" if torch.cuda.is_available() else "cpu"
    sdvae_eval_dir = os.path.abspath(os.path.join(args.sdvae_root, "mol_vae", "pytorch_eval"))
    sdvae_argv = [
        "sample_sdvae_latent",
        "-saved_model",
        args.saved_model,
        "-grammar_file",
        args.grammar_file,
        "-mode",
        sdvae_mode,
        "-ae_type",
        args.ae_type,
        "-encoder_type",
        args.encoder_type,
        "-rnn_type",
        args.rnn_type,
        "-max_decode_steps",
        str(args.max_decode_steps),
        "-latent_dim",
        str(args.latent_dim_sdvae),
    ]
    old_argv = sys.argv[:]
    old_cwd = os.getcwd()
    try:
        os.chdir(sdvae_eval_dir)
        sys.argv = sdvae_argv
        sys.path.append(sdvae_eval_dir)
        from att_model_proxy import AttMolProxy
        sys.argv = old_argv
        return AttMolProxy()
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)


def main():
    args = create_argparser().parse_args()
    set_seed(args.seed)
    logger.configure()

    model = LatentConditionedMLP(
        latent_dim=args.latent_dim,
        model_channels=args.model_channels,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        text_fusion=args.text_fusion,
        text_attn_heads=args.text_attn_heads,
    )
    diffusion = SpacedDiffusion(
        use_timesteps=[i for i in range(args.diffusion_steps)],
        betas=gd.get_named_beta_schedule(args.noise_schedule, args.diffusion_steps),
        model_mean_type=gd.ModelMeanType.START_X,
        model_var_type=gd.ModelVarType.FIXED_LARGE,
        loss_type=gd.LossType.MSE,
        rescale_timesteps=True,
        model_arch="latent",
        training_mode="latent",
    )
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    model.eval()

    dataset = ChEBIdataset(
        dir=args.dataset_dir,
        smi_tokenizer=regexTokenizer(),
        split=args.split,
        replace_desc=False,
        load_state=True,
    )
    proxy = maybe_load_proxy(args)

    results = []
    num_done = 0
    sample_fn = diffusion.ddim_sample_loop if args.use_ddim else diffusion.p_sample_loop
    target = min(args.num_samples, len(dataset))
    pbar = tqdm(total=target, desc="sample latent", unit="mol") if tqdm is not None else None
    while num_done < target:
        idend = min(num_done + args.batch_size, args.num_samples, len(dataset))
        chunk = [dataset[i] for i in range(num_done, idend)]
        desc_state = torch.concat([row["desc_state"] for row in chunk], dim=0).to(dist_util.dev())
        desc_mask = torch.concat([row["desc_mask"] for row in chunk], dim=0).to(dist_util.dev())
        sample = sample_fn(
            model,
            (idend - num_done, args.latent_dim),
            clip_denoised=False,
            model_kwargs={},
            progress=True,
            desc=(desc_state, desc_mask),
        )
        sample = sample.detach().cpu().numpy()
        decoded = proxy.decode(sample, use_random=False) if proxy is not None else [""] * len(sample)
        for row, latent, smi in zip(chunk, sample, decoded):
            results.append((row["cid"], row["smiles"], smi, latent.tolist()))
        done_now = idend - num_done
        num_done = idend
        if pbar is not None:
            pbar.update(done_now)
    if pbar is not None:
        pbar.close()

    with open(args.output, "w") as f:
        for cid, ref_smi, pred_smi, latent in results:
            f.write(f"{cid}\t{ref_smi}\t{pred_smi}\t{latent}\n")

    print(f"saved samples to {args.output}")


if __name__ == "__main__":
    main()
