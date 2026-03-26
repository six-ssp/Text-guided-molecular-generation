import argparse
import os
import sys
from pathlib import Path

import torch
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = Path(os.environ.get("TEXT2MOL_ROOT", SCRIPT_DIR.parents[2]))
DEFAULT_TGM_ROOT = (
    DEFAULT_PROJECT_ROOT / "tgm-dlm"
    if (DEFAULT_PROJECT_ROOT / "tgm-dlm").exists()
    else SCRIPT_DIR.parents[1]
)
DEFAULT_SDVAE_ROOT = DEFAULT_PROJECT_ROOT / "sdvae"
DEFAULT_DATASET_DIR = DEFAULT_TGM_ROOT / "datasets" / "SMILES"
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
    cwd_p = (Path.cwd() / p).resolve()
    if cwd_p.exists():
        return cwd_p
    return (preferred_base / p).resolve()


def load_smiles(dataset_file):
    rows = []
    with open(dataset_file, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            cid, smiles, desc = line.rstrip("\n").split("\t")
            if smiles != "*":
                rows.append((int(cid), smiles, desc))
    return rows


def atomic_torch_save(obj, path):
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def atomic_write_lines(lines, path):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        for line in lines:
            f.write(f"{line}\n")
    os.replace(tmp_path, path)


def encode_rows(proxy, rows, verbose_skips=False):
    if not rows:
        return [], []

    smiles = [row[1] for row in rows]
    try:
        latents = proxy.encode(smiles, use_random=False)
        return list(zip(rows, latents)), []
    except Exception as exc:
        if len(rows) == 1:
            row = rows[0]
            if verbose_skips:
                print(f"skip cid={row[0]} smiles={row[1]!r}: {exc}")
            return [], [row]

        mid = len(rows) // 2
        left_ok, left_bad = encode_rows(proxy, rows[:mid], verbose_skips=verbose_skips)
        right_ok, right_bad = encode_rows(proxy, rows[mid:], verbose_skips=verbose_skips)
        return left_ok + right_ok, left_bad + right_bad


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--skipped-output", default=None)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--verbose-skips", action="store_true")
    parser.add_argument("--sdvae-root", default=str(DEFAULT_SDVAE_ROOT))
    parser.add_argument("-saved_model", "--saved_model", default=str(DEFAULT_SAVED_MODEL))
    parser.add_argument("-grammar_file", "--grammar_file", default=str(DEFAULT_GRAMMAR_FILE))
    parser.add_argument("-mode", "--mode", default="auto", choices=["auto", "gpu", "cpu"])
    parser.add_argument("-ae_type", "--ae_type", default="vae")
    parser.add_argument("-encoder_type", "--encoder_type", default="cnn")
    parser.add_argument("-rnn_type", "--rnn_type", default="gru")
    parser.add_argument("-max_decode_steps", "--max_decode_steps", type=int, default=278)
    parser.add_argument("-latent_dim", "--latent_dim", type=int, default=56)
    return parser


def main():
    args = create_argparser().parse_args()
    dataset_dir = resolve_path(args.dataset_dir, SCRIPT_DIR)
    dataset_file = dataset_dir / f"{args.split}.txt"
    output_file = (
        resolve_path(args.output, SCRIPT_DIR)
        if args.output is not None
        else dataset_dir / f"{args.split}_sdvae_latents.pt"
    )
    skipped_output = (
        resolve_path(args.skipped_output, SCRIPT_DIR)
        if args.skipped_output is not None
        else dataset_dir / f"{args.split}_sdvae_skipped.txt"
    )
    sdvae_root = resolve_path(args.sdvae_root, SCRIPT_DIR)
    sdvae_eval_dir = sdvae_root / "mol_vae" / "pytorch_eval"
    saved_model = resolve_path(args.saved_model, SCRIPT_DIR)
    grammar_file = resolve_path(args.grammar_file, SCRIPT_DIR)
    if not saved_model.exists():
        raise FileNotFoundError(f"saved_model not found: {saved_model}")
    if not grammar_file.exists():
        raise FileNotFoundError(f"grammar_file not found: {grammar_file}")

    sdvae_mode = args.mode
    if sdvae_mode == "auto":
        try:
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
                torch.empty(1, device="cuda:0")
                sdvae_mode = "gpu"
            else:
                sdvae_mode = "cpu"
        except Exception:
            sdvae_mode = "cpu"
    print(f"dump_sdvae_latents mode={sdvae_mode}")

    sdvae_argv = [
        "dump_sdvae_latents",
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
        str(args.latent_dim),
    ]
    old_argv = sys.argv[:]
    old_cwd = os.getcwd()
    try:
        os.chdir(sdvae_eval_dir)
        sys.argv = sdvae_argv
        sys.path.append(str(sdvae_eval_dir))
        from att_model_proxy import AttMolProxy
        sys.argv = old_argv

        rows = load_smiles(str(dataset_file))
        proxy = AttMolProxy()
        if os.path.exists(output_file):
            latent_dump = torch.load(str(output_file), map_location="cpu")
            print(f"resume from {output_file}, loaded {len(latent_dump)} latents")
        else:
            latent_dump = {}
        if os.path.exists(skipped_output):
            with open(skipped_output, "r") as f:
                skipped_cids = {int(line.strip()) for line in f if line.strip()}
            print(f"resume from {skipped_output}, loaded {len(skipped_cids)} skipped cids")
        else:
            skipped_cids = set()

        total_chunks = (len(rows) + args.chunk_size - 1) // args.chunk_size
        chunk_starts = tqdm(
            range(0, len(rows), args.chunk_size),
            total=total_chunks,
            desc=f"latent:{args.split}",
            unit="chunk",
        )
        for chunk_idx, start in enumerate(chunk_starts):
            chunk = [
                row for row in rows[start : start + args.chunk_size]
                if row[0] not in latent_dump and row[0] not in skipped_cids
            ]
            encoded_rows, failed_rows = encode_rows(proxy, chunk, verbose_skips=args.verbose_skips)
            skipped_cids.update(row[0] for row in failed_rows)
            for (cid, _smiles, _desc), latent in encoded_rows:
                latent_dump[cid] = torch.tensor(latent).float()
            if hasattr(chunk_starts, "set_postfix"):
                chunk_starts.set_postfix(latents=len(latent_dump), skipped=len(skipped_cids))
            if chunk_idx % args.save_every == 0:
                atomic_torch_save(latent_dump, str(output_file))
                atomic_write_lines(sorted(skipped_cids), str(skipped_output))
                print(
                    f"checkpoint -> {output_file} ({len(latent_dump)} latents), "
                    f"{skipped_output} ({len(skipped_cids)} skipped)"
                )

        atomic_torch_save(latent_dump, str(output_file))
        atomic_write_lines(sorted(skipped_cids), str(skipped_output))
        print(f"saved latent dump to {output_file}")
        if skipped_cids:
            print(f"skipped {len(skipped_cids)} rows that failed sdvae parsing")
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
