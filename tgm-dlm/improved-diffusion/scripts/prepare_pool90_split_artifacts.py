import argparse
import os
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
DEFAULT_DATASET_DIR = DEFAULT_TGM_ROOT / "datasets" / "SMILES"


def resolve_path(path_str, preferred_base):
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    cwd_p = (Path.cwd() / p).resolve()
    if cwd_p.exists():
        return cwd_p
    return (preferred_base / p).resolve()


def atomic_torch_save(obj, path):
    tmp_path = str(path) + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def load_split_cids(split_file):
    cids = []
    with open(split_file, "r") as f:
        next(f)
        for line in f:
            cid = int(line.split("\t", 1)[0])
            cids.append(cid)
    return cids


def build_subset(source_map, cids, desc):
    subset = {}
    missing = 0
    for cid in tqdm(cids, desc=desc, unit="cid", leave=False):
        if cid in source_map:
            subset[cid] = source_map[cid]
        else:
            missing += 1
    return subset, missing


def check_no_overlap(split_to_cids):
    names = list(split_to_cids.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = names[i]
            b = names[j]
            overlap = split_to_cids[a] & split_to_cids[b]
            if overlap:
                raise RuntimeError(f"split leakage detected: {a} vs {b}, overlap={len(overlap)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--source-split", default="pool90_all")
    parser.add_argument(
        "--target-splits",
        nargs="+",
        default=["train_pool90", "validation_pool90", "test_pool90"],
    )
    args = parser.parse_args()

    dataset_dir = resolve_path(args.dataset_dir, SCRIPT_DIR)

    split_to_cids = {}
    for split in args.target_splits:
        split_file = dataset_dir / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"split file not found: {split_file}")
        split_to_cids[split] = set(load_split_cids(split_file))
    check_no_overlap(split_to_cids)

    source_desc_file = dataset_dir / f"{args.source_split}_desc_states.pt"
    source_latent_file = dataset_dir / f"{args.source_split}_sdvae_latents.pt"
    if not source_desc_file.exists():
        raise FileNotFoundError(f"source desc file not found: {source_desc_file}")
    if not source_latent_file.exists():
        raise FileNotFoundError(f"source latent file not found: {source_latent_file}")

    print(f"loading desc from {source_desc_file}")
    source_desc = torch.load(str(source_desc_file), map_location="cpu")
    print(f"loading latents from {source_latent_file}")
    source_latent = torch.load(str(source_latent_file), map_location="cpu")
    print(f"source sizes: desc={len(source_desc)}, latent={len(source_latent)}")

    for split in tqdm(args.target_splits, desc="split artifacts", unit="split"):
        cids = load_split_cids(dataset_dir / f"{split}.txt")
        desc_subset, desc_missing = build_subset(source_desc, cids, desc=f"desc:{split}")
        latent_subset, latent_missing = build_subset(source_latent, cids, desc=f"latent:{split}")

        desc_out = dataset_dir / f"{split}_desc_states.pt"
        latent_out = dataset_dir / f"{split}_sdvae_latents.pt"
        atomic_torch_save(desc_subset, desc_out)
        atomic_torch_save(latent_subset, latent_out)

        print(
            f"{split}: rows={len(cids)}, desc={len(desc_subset)} (missing={desc_missing}), "
            f"latent={len(latent_subset)} (missing={latent_missing})"
        )
        if desc_missing or latent_missing:
            print(f"warning: {split} has missing features; training will auto-filter unavailable rows.")


if __name__ == "__main__":
    main()
