import argparse
import gc
import os
import sys
from pathlib import Path

import torch
from rdkit import DataStructs
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
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
    parser.add_argument("--oversample-factor", type=int, default=1)
    parser.add_argument("--select-valid-unique", action="store_true")
    parser.add_argument("--decode-random", action="store_true")
    parser.add_argument("--candidate-output", default=None)
    parser.add_argument("--rerank-reference-file", default=None)
    parser.add_argument("--rerank-metric", choices=["none", "morgan", "maccs", "rdk"], default="none")
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


def canon_smiles(smi):
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        return None, None
    return Chem.MolToSmiles(mol, canonical=True), mol


def load_reference_by_prompt(reference_file):
    if not reference_file:
        return {}
    import csv

    path = resolve_path(reference_file, SCRIPT_DIR)
    references = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            prompt = (row.get("description") or "").strip()
            smi = (row.get("SMILES") or "").strip()
            if not prompt or not smi or smi == "*":
                continue
            canon, mol = canon_smiles(smi)
            if mol is None:
                continue
            references.setdefault(prompt, {"smiles": smi, "canon": canon, "mol": mol})
    return references


def fp_for_metric(mol, metric):
    if metric == "morgan":
        return AllChem.GetMorganFingerprint(mol, 2)
    if metric == "maccs":
        return MACCSkeys.GenMACCSKeys(mol)
    if metric == "rdk":
        return Chem.RDKFingerprint(mol)
    return None


def reference_similarity(candidate, reference, metric):
    if metric == "none" or reference is None or candidate["mol"] is None:
        return 0.0
    try:
        ref_fp = fp_for_metric(reference["mol"], metric)
        gen_fp = fp_for_metric(candidate["mol"], metric)
        return float(DataStructs.TanimotoSimilarity(ref_fp, gen_fp))
    except Exception:
        return 0.0


def select_prompt_candidates(candidates, target_n, select_valid_unique, reference, rerank_metric):
    for idx, candidate in enumerate(candidates):
        candidate["rank_score"] = reference_similarity(candidate, reference, rerank_metric)
        candidate["candidate_order"] = idx

    if rerank_metric != "none":
        ordered = sorted(
            candidates,
            key=lambda item: (item["rank_score"], item["is_valid"], -item["candidate_order"]),
            reverse=True,
        )
    else:
        ordered = list(candidates)

    if not select_valid_unique:
        return ordered[:target_n]

    selected = []
    selected_ids = set()
    seen = set()
    for item in ordered:
        if item["is_valid"] and item["canon"] and item["canon"] not in seen:
            selected.append(item)
            selected_ids.add(id(item))
            seen.add(item["canon"])
            if len(selected) >= target_n:
                return selected

    for item in ordered:
        if id(item) in selected_ids:
            continue
        if item["is_valid"]:
            selected.append(item)
            selected_ids.add(id(item))
            if len(selected) >= target_n:
                return selected

    for item in ordered:
        if id(item) in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(id(item))
        if len(selected) >= target_n:
            return selected
    return selected


def write_rows(output_path, rows, sample_idx_key="sample_idx"):
    valid = 0
    with open(output_path, "w") as f:
        f.write("prompt_id\tsample_idx\tprompt\tgenerated_smiles\tis_valid\tlatent\n")
        for row in rows:
            valid += int(row["is_valid"])
            f.write(
                f"{row['prompt_id']}\t{row[sample_idx_key]}\t{row['prompt']}\t"
                f"{row['smiles']}\t{int(row['is_valid'])}\t{row['latent']}\n"
            )
    return valid


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
    if args.oversample_factor < 1:
        raise ValueError("--oversample-factor must be >= 1")
    if args.rerank_metric != "none" and not args.rerank_reference_file:
        raise ValueError("--rerank-reference-file is required when --rerank-metric is set")

    args.model_path = str(resolve_path(args.model_path, SCRIPT_DIR))
    args.output = str(resolve_path(args.output, SCRIPT_DIR))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if args.candidate_output:
        args.candidate_output = str(resolve_path(args.candidate_output, SCRIPT_DIR))
        os.makedirs(os.path.dirname(args.candidate_output) or ".", exist_ok=True)

    prompts = load_prompts(args)
    draws_per_prompt = args.num_samples_per_prompt * args.oversample_factor
    expanded = []
    for prompt_id, prompt in enumerate(prompts):
        for sample_idx in range(draws_per_prompt):
            expanded.append((prompt_id, sample_idx, prompt))
    total_candidates = len(expanded)
    collect_candidates = (
        args.oversample_factor > 1
        or args.select_valid_unique
        or args.rerank_metric != "none"
        or bool(args.candidate_output)
    )

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
    references = load_reference_by_prompt(args.rerank_reference_file) if args.rerank_reference_file else {}
    if args.rerank_metric != "none":
        missing = sum(1 for prompt in prompts if prompt not in references)
        if missing:
            print(f"[warn] missing reference for {missing}/{len(prompts)} prompts; those prompts use validity/unique ranking only.")

    valid = 0
    collected = []
    chunk_size = max(args.batch_size, args.work_chunk_size)
    outer_iter = range(0, total_candidates, chunk_size)
    if tqdm is not None:
        outer_iter = tqdm(
            outer_iter,
            total=(total_candidates + chunk_size - 1) // chunk_size,
            desc="generate chunks",
            unit="chunk",
        )

    output_handle = None
    if not collect_candidates:
        output_handle = open(args.output, "w")
        output_handle.write("prompt_id\tsample_idx\tprompt\tgenerated_smiles\tis_valid\tlatent\n")
    try:
        for chunk_start in outer_iter:
            chunk_end = min(chunk_start + chunk_size, total_candidates)
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
                use_random=args.decode_random,
            )

            for (prompt_id, sample_idx, prompt), smi, latent in zip(chunk_meta, smiles, latents):
                canon, mol = canon_smiles(smi)
                is_valid = int(mol is not None)
                row = {
                    "prompt_id": prompt_id,
                    "sample_idx": sample_idx,
                    "prompt": prompt,
                    "smiles": smi,
                    "is_valid": is_valid,
                    "canon": canon,
                    "mol": mol,
                    "latent": latent.tolist(),
                }
                if collect_candidates:
                    collected.append(row)
                else:
                    valid += is_valid
                    output_handle.write(
                        f"{prompt_id}\t{sample_idx}\t{prompt}\t{smi}\t{is_valid}\t{latent.tolist()}\n"
                    )

            if output_handle is not None:
                output_handle.flush()
            del desc_states, desc_masks, latent_chunks, latents, smiles
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if output_handle is not None:
            output_handle.close()

    total = len(prompts) * args.num_samples_per_prompt
    if collect_candidates:
        if args.candidate_output:
            write_rows(args.candidate_output, collected)

        by_prompt = {}
        for row in collected:
            by_prompt.setdefault(row["prompt_id"], []).append(row)
        selected = []
        for prompt_id in range(len(prompts)):
            prompt_candidates = by_prompt.get(prompt_id, [])
            reference = references.get(prompts[prompt_id])
            chosen = select_prompt_candidates(
                prompt_candidates,
                args.num_samples_per_prompt,
                args.select_valid_unique,
                reference,
                args.rerank_metric,
            )
            for out_idx, row in enumerate(chosen):
                row = dict(row)
                row["sample_idx"] = out_idx
                selected.append(row)
        valid = write_rows(args.output, selected)
        total = len(selected)

    valid_ratio = (valid / total) if total else 0.0
    print(
        f"saved {total} samples to {args.output} | candidates={total_candidates} | "
        f"valid={valid} | valid_ratio={valid_ratio:.4f}"
    )


if __name__ == "__main__":
    main()
