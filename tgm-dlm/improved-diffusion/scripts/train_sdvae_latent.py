import argparse
import os
import warnings
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from improved_diffusion import gaussian_diffusion as gd
from improved_diffusion import dist_util
from improved_diffusion.latent_model import LatentConditionedMLP
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.respace import SpacedDiffusion
from improved_diffusion.train_util import TrainLoop
from mydatasets import LatentChEBIDataset, get_latent_dataloader

warnings.filterwarnings("ignore")


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = Path(os.environ.get("TEXT2MOL_ROOT", SCRIPT_DIR.parents[2]))
DEFAULT_TGM_ROOT = (
    DEFAULT_PROJECT_ROOT / "tgm-dlm"
    if (DEFAULT_PROJECT_ROOT / "tgm-dlm").exists()
    else SCRIPT_DIR.parents[1]
)
DEFAULT_DATASET_DIR = DEFAULT_TGM_ROOT / "datasets" / "SMILES"
DEFAULT_CKPT_DIR = DEFAULT_TGM_ROOT / "checkpoints_sdvae_latent"


def resolve_path(path_str, preferred_base):
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    cwd_p = (Path.cwd() / p).resolve()
    if cwd_p.exists():
        return cwd_p
    return (preferred_base / p).resolve()


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--split", default="train_val_256")
    parser.add_argument("--latent-file", default=None)
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_CKPT_DIR))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--microbatch", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ema-rate", default="0.9999")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--fp16-scale-growth", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-anneal-steps", type=int, default=100000)
    parser.add_argument("--gradient-clipping", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=19991009)
    parser.add_argument("--diffusion-steps", type=int, default=2000)
    parser.add_argument("--noise-schedule", default="sqrt")
    parser.add_argument("--latent-dim", type=int, default=56)
    parser.add_argument("--model-channels", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--text-fusion", choices=["pooled", "crossattn"], default="pooled")
    parser.add_argument("--text-attn-heads", type=int, default=8)
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--mask-desc", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--dist-port", default="12145")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    return parser


def _resolve_dist_device_args(args, rank):
    gpu_id = args.gpu_id + rank
    if args.device == "cuda":
        return dict(backend="nccl", force_cuda=True, gpu_id=gpu_id)
    if args.device == "cpu":
        return dict(backend="gloo", force_cuda=False, gpu_id=gpu_id)
    return dict(backend=None, force_cuda=False, gpu_id=gpu_id)


def main_worker(rank, world_size, args):
    os.makedirs(args.checkpoint_path, exist_ok=True)
    dist_kwargs = _resolve_dist_device_args(args, rank)
    dist_util.setup_dist(rank, world_size, port=args.dist_port, **dist_kwargs)
    if rank == 0:
        print(
            f"dist setup done | device={dist_util.dev()} | backend={dist.get_backend()} "
            f"| torch_cuda_available={torch.cuda.is_available()} | world_size={world_size}"
        )

    model = LatentConditionedMLP(
        latent_dim=args.latent_dim,
        model_channels=args.model_channels,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        text_fusion=args.text_fusion,
        text_attn_heads=args.text_attn_heads,
    )
    if args.init_checkpoint and not args.resume_checkpoint:
        state_dict = torch.load(args.init_checkpoint, map_location="cpu")
        incompatible = model.load_state_dict(state_dict, strict=False)
        if rank == 0:
            print(
                f"init from {args.init_checkpoint} | "
                f"missing_keys={len(incompatible.missing_keys)} "
                f"unexpected_keys={len(incompatible.unexpected_keys)}"
            )
    elif args.init_checkpoint and args.resume_checkpoint and rank == 0:
        print("resume-checkpoint is set; ignore init-checkpoint.")
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
    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)
    dataset = LatentChEBIDataset(
        dir=args.dataset_dir,
        split=args.split,
        latent_file=args.latent_file,
        mask_desc=args.mask_desc,
    )
    dataloader = get_latent_dataloader(
        dataset,
        args.batch_size,
        rank,
        world_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=dataloader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=None,
        eval_interval=args.eval_interval,
    ).run_loop()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    args = create_argparser().parse_args()
    args.dataset_dir = str(resolve_path(args.dataset_dir, SCRIPT_DIR))
    args.checkpoint_path = str(resolve_path(args.checkpoint_path, SCRIPT_DIR))
    if args.latent_file:
        args.latent_file = str(resolve_path(args.latent_file, SCRIPT_DIR))
    if args.resume_checkpoint:
        args.resume_checkpoint = str(resolve_path(args.resume_checkpoint, SCRIPT_DIR))
    if args.init_checkpoint:
        args.init_checkpoint = str(resolve_path(args.init_checkpoint, SCRIPT_DIR))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    world_size = args.world_size
    if world_size == 1:
        main_worker(0, world_size, args)
    else:
        mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)
