import argparse
import os
import random
from pathlib import Path
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


def load_rows(split_file):
    with open(split_file, "r") as f:
        header = next(f)
        rows = []
        for line in f:
            cid, smiles, desc = line.rstrip("\n").split("\t")
            if smiles != "*":
                rows.append((int(cid), smiles, desc))
    return header, rows


def write_rows(path, header, rows):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        f.write(header)
        for cid, smiles, desc in rows:
            f.write(f"{cid}\t{smiles}\t{desc}\n")
    os.replace(tmp, path)


def smiles_overlap(rows_a, rows_b):
    smi_a = {row[1] for row in rows_a}
    smi_b = {row[1] for row in rows_b}
    return len(smi_a & smi_b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--source-split", default="pool90_all")
    parser.add_argument("--train-split", default="train_pool90")
    parser.add_argument("--val-split", default="validation_pool90")
    parser.add_argument("--test-split", default="test_pool90")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260310)
    args = parser.parse_args()

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError(f"ratios must sum to 1.0, got {total_ratio}")

    dataset_dir = resolve_path(args.dataset_dir, SCRIPT_DIR)
    src_file = dataset_dir / f"{args.source_split}.txt"
    header, rows = load_rows(src_file)
    total_rows = len(rows)
    if total_rows == 0:
        raise RuntimeError(f"no valid rows in {src_file}")

    smiles_to_rows = {}
    for row in tqdm(rows, desc="group smiles", unit="row"):
        smiles_to_rows.setdefault(row[1], []).append(row)

    grouped = list(smiles_to_rows.items())
    rnd = random.Random(args.seed)
    rnd.shuffle(grouped)

    target_train = int(total_rows * args.train_ratio)
    target_val = int(total_rows * args.val_ratio)

    train_rows, val_rows, test_rows = [], [], []
    for _smiles, group_rows in tqdm(grouped, desc="assign splits", unit="group"):
        if len(train_rows) < target_train:
            train_rows.extend(group_rows)
        elif len(val_rows) < target_val:
            val_rows.extend(group_rows)
        else:
            test_rows.extend(group_rows)

    out_train = dataset_dir / f"{args.train_split}.txt"
    out_val = dataset_dir / f"{args.val_split}.txt"
    out_test = dataset_dir / f"{args.test_split}.txt"
    write_rows(out_train, header, train_rows)
    write_rows(out_val, header, val_rows)
    write_rows(out_test, header, test_rows)

    print(
        f"rows: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}, "
        f"total={len(train_rows)+len(val_rows)+len(test_rows)}"
    )
    print(
        f"unique smiles: train={len({x[1] for x in train_rows})}, "
        f"val={len({x[1] for x in val_rows})}, test={len({x[1] for x in test_rows})}"
    )
    print(
        f"smiles overlap: train-val={smiles_overlap(train_rows, val_rows)}, "
        f"train-test={smiles_overlap(train_rows, test_rows)}, "
        f"val-test={smiles_overlap(val_rows, test_rows)}"
    )


if __name__ == "__main__":
    main()
