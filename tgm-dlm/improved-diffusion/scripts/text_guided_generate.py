import argparse
import gc
import os
import sys
from pathlib import Path

import torch
from rdkit import Chem
from rdkit import RDLogger
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import set_seed

from improved_diffusion import gaussian_diffusion as gd
from improved_diffusion.latent_model import LatentConditionedMLP
from improved_diffusion.respace import SpacedDiffusion

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

RDLogger.DisableLog("rdApp.*")


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = Path(os.environ.get("TEXT2MOL_ROOT", SCRIPT_DIR.parents[2]))
DEFAULT_TGM_ROOT = (
    DEFAULT_PROJECT_ROOT / "tgm-dlm"
    if (DEFAULT_PROJECT_ROOT / "tgm-dlm").exists()
    else SCRIPT_DIR.parents[1]
)
DEFAULT_TEXT_MODEL = DEFAULT_TGM_ROOT / "scibert"
DEFAULT_OUTPUT = DEFAULT_TGM_ROOT / "text_guided_samples.txt"
DEFAULT_SDVAE_ROOT = DEFAULT_PROJECT_ROOT / "sdvae"
DEFAULT_SAVED_MODEL = Path(
    os.environ.get(
        "SDVAE_SAVED_MODEL",
        str(DEFAULT_SDVAE_ROOT / "dropbox/results/zinc/zinc_kl_avg.model"),
    )
)
DEFAULT_GRAMMAR_FILE = Path(
    os.environ.get(
        "SDVAE_GRAMMAR_FILE",
        str(DEFAULT_SDVAE_ROOT / "dropbox/context_free_grammars/mol_zinc.grammar"),
    )
)


def resolve_path(path_str, preferred_base):
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    return (Path.cwd() / p).resolve()


def resolve_device(mode, gpu_id):
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        torch.cuda.set_device(gpu_id)
        torch.empty(1, device=f"cuda:{gpu_id}")
        return torch.device(f"cuda:{gpu_id}")
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(gpu_id)
            torch.empty(1, device=f"cuda:{gpu_id}")
            return torch.device(f"cuda:{gpu_id}")
        except Exception:
            pass
    return torch.device("cpu")


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--num-samples-per-prompt", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--work-chunk-size", type=int, default=256)
    parser.add_argument("--decode-batch-size", type=int, default=32)
    parser.add_argument("--max-text-len", type=int, default=216)
    parser.add_argument("--text-model", default=str(DEFAULT_TEXT_MODEL))
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
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--sdvae-root", default=str(DEFAULT_SDVAE_ROOT))
    parser.add_argument("-saved_model", "--saved_model", default=str(DEFAULT_SAVED_MODEL))
    parser.add_argument("-grammar_file", "--grammar_file", default=str(DEFAULT_GRAMMAR_FILE))
    parser.add_argument("-mode", "--mode", default="auto", choices=["auto", "gpu", "cpu"])
    parser.add_argument("-ae_type", "--ae_type", default="vae")
    parser.add_argument("-encoder_type", "--encoder_type", default="cnn")
    parser.add_argument("-rnn_type", "--rnn_type", default="gru")
    parser.add_argument("-max_decode_steps", "--max_decode_steps", type=int, default=278)
    parser.add_argument("-latent_dim_sdvae", "--latent_dim_sdvae", type=int, default=56)
    return parser


def load_prompts(args):
    prompts = []
    for p in args.prompt:
        p = p.strip()
        if p:
            prompts.append(p)
    if args.prompt_file:
        prompt_file = resolve_path(args.prompt_file, SCRIPT_DIR)
        with open(prompt_file, "r") as f:
            for line in f:
                p = line.strip()
                if p:
                    prompts.append(p)
    if not prompts:
        raise ValueError("no prompt provided, use --prompt or --prompt-file")
    return prompts


def resolve_text_model_name(text_model):
    text_model_path = resolve_path(text_model, SCRIPT_DIR)
    if (text_model_path / "config.json").exists():
        return str(text_model_path)
    if os.path.exists(os.path.join(text_model, "config.json")):
        return text_model
    return "allenai/scibert_scivocab_uncased"


def build_text_encoder(model_name, device):
    tokz = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokz, model


def encode_text(prompts, tokz, model, batch_size, max_text_len, device):
    if not prompts:
        return torch.empty((0, max_text_len, model.config.hidden_size)), torch.empty((0, max_text_len), dtype=torch.long)

    states = []
    masks = []
    with torch.inference_mode():
        iterator = range(0, len(prompts), batch_size)
        if tqdm is not None:
            iterator = tqdm(iterator, total=(len(prompts) + batch_size - 1) // batch_size, desc="encode text", unit="batch")
        for start in iterator:
            end = min(start + batch_size, len(prompts))
            chunk = prompts[start:end]
            tok = tokz(
                chunk,
                max_length=max_text_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = tok["input_ids"].to(device)
            attn_mask = tok["attention_mask"].to(device)
            out = model(input_ids, attention_mask=attn_mask).last_hidden_state
            states.append(out.cpu())
            masks.append(attn_mask.cpu())
    return torch.cat(states, dim=0), torch.cat(masks, dim=0)


def maybe_load_proxy(args):
    sdvae_root = resolve_path(args.sdvae_root, SCRIPT_DIR)
    saved_model = resolve_path(args.saved_model, SCRIPT_DIR)
    grammar_file = resolve_path(args.grammar_file, SCRIPT_DIR)
    sdvae_eval_dir = sdvae_root / "mol_vae" / "pytorch_eval"

    if not saved_model.exists():
        raise FileNotFoundError(f"saved_model not found: {saved_model}")
    if not grammar_file.exists():
        raise FileNotFoundError(f"grammar_file not found: {grammar_file}")

    sdvae_mode = args.mode
    if sdvae_mode == "auto":
        sdvae_mode = "gpu" if torch.cuda.is_available() else "cpu"

    sdvae_argv = [
        "text_guided_generate",
        "-saved_model",
        str(saved_model),
        "-grammar_file",
        str(grammar_file),
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
        sys.path.append(str(sdvae_eval_dir))
        from att_model_proxy import AttMolProxy

        sys.argv = old_argv
        return AttMolProxy()
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)


def decode_latents_in_chunks(proxy, latents, decode_batch_size, use_random=False):
    total = len(latents)
    if total == 0:
        return []

    smiles = []
    chunk_size = max(1, int(decode_batch_size))
    idx = 0

    while idx < total:
        end = min(idx + chunk_size, total)
        chunk = latents[idx:end]
        try:
            smiles.extend(proxy.decode(chunk, use_random=use_random))
            idx = end
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" not in msg:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if chunk_size == 1:
                raise
            chunk_size = max(1, chunk_size // 2)
            print(f"[warn] SDVAE decode OOM, reduce decode_batch_size to {chunk_size} and retry...")

    return smiles


def main():
    args = create_argparser().parse_args()
    set_seed(args.seed)

    args.model_path = str(resolve_path(args.model_path, SCRIPT_DIR))
    args.output = str(resolve_path(args.output, SCRIPT_DIR))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    prompts = load_prompts(args)
    expanded = []
    for prompt_id, prompt in enumerate(prompts):
        for sample_idx in range(args.num_samples_per_prompt):
            expanded.append((prompt_id, sample_idx, prompt))
    total_samples = len(expanded)

    device = resolve_device(args.device, args.gpu_id)
    print(f"device={device}")

    text_model_name = resolve_text_model_name(args.text_model)
    print(f"text_model={text_model_name}")
    tokz, text_encoder = build_text_encoder(text_model_name, device)

    model = LatentConditionedMLP(
        latent_dim=args.latent_dim,
        model_channels=args.model_channels,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        text_fusion=args.text_fusion,
        text_attn_heads=args.text_attn_heads,
    )
    try:
        model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    except RuntimeError as e:
        raise RuntimeError(
            f"failed to load checkpoint {args.model_path}; "
            f"check --text-fusion/--text-attn-heads (current: {args.text_fusion}/{args.text_attn_heads})"
        ) from e
    model = model.to(device)
    model.eval()

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
    sample_fn = diffusion.ddim_sample_loop if args.use_ddim else diffusion.p_sample_loop

    proxy = maybe_load_proxy(args)
    valid = 0
    chunk_size = max(args.batch_size, args.work_chunk_size)
    outer_iter = range(0, total_samples, chunk_size)
    if tqdm is not None:
        outer_iter = tqdm(
            outer_iter,
            total=(total_samples + chunk_size - 1) // chunk_size,
            desc="generate chunks",
            unit="chunk",
        )

    with open(args.output, "w") as f:
        f.write("prompt_id\tsample_idx\tprompt\tgenerated_smiles\tis_valid\tlatent\n")
        for chunk_start in outer_iter:
            chunk_end = min(chunk_start + chunk_size, total_samples)
            chunk_meta = expanded[chunk_start:chunk_end]
            chunk_prompts = [x[2] for x in chunk_meta]

            desc_states, desc_masks = encode_text(
                chunk_prompts,
                tokz=tokz,
                model=text_encoder,
                batch_size=args.batch_size,
                max_text_len=args.max_text_len,
                device=device,
            )

            latent_chunks = []
            with torch.inference_mode():
                inner_iter = range(0, len(chunk_meta), args.batch_size)
                if tqdm is not None:
                    inner_iter = tqdm(
                        inner_iter,
                        total=(len(chunk_meta) + args.batch_size - 1) // args.batch_size,
                        desc=f"sample latent [{chunk_start}:{chunk_end}]",
                        unit="batch",
                        leave=False,
                    )
                for start in inner_iter:
                    end = min(start + args.batch_size, len(chunk_meta))
                    state_chunk = desc_states[start:end].to(device)
                    mask_chunk = desc_masks[start:end].to(device)
                    latent = sample_fn(
                        model,
                        (end - start, args.latent_dim),
                        clip_denoised=False,
                        model_kwargs={},
                        progress=False,
                        device=device,
                        desc=(state_chunk, mask_chunk),
                    )
                    latent_chunks.append(latent.cpu())

            latents = torch.cat(latent_chunks, dim=0).numpy()
            smiles = decode_latents_in_chunks(
                proxy,
                latents,
                decode_batch_size=args.decode_batch_size,
                use_random=False,
            )

            for (prompt_id, sample_idx, prompt), smi, latent in zip(chunk_meta, smiles, latents):
                is_valid = int(bool(smi) and Chem.MolFromSmiles(smi) is not None)
                valid += is_valid
                f.write(
                    f"{prompt_id}\t{sample_idx}\t{prompt}\t{smi}\t{is_valid}\t{latent.tolist()}\n"
                )

            f.flush()
            del desc_states, desc_masks, latent_chunks, latents, smiles
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    total = total_samples
    valid_ratio = (valid / total) if total else 0.0
    print(
        f"saved {total} samples to {args.output} | valid={valid} | valid_ratio={valid_ratio:.4f}"
    )


if __name__ == "__main__":
    main()
