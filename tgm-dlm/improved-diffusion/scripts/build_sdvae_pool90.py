import argparse
import os
import sys
from collections import Counter
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

RDLogger.DisableLog("rdApp.*")


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = Path(os.environ.get("TEXT2MOL_ROOT", SCRIPT_DIR.parents[2]))
DEFAULT_TGM_ROOT = (
    DEFAULT_PROJECT_ROOT / "tgm-dlm"
    if (DEFAULT_PROJECT_ROOT / "tgm-dlm").exists()
    else SCRIPT_DIR.parents[1]
)
DEFAULT_DATASET_DIR = DEFAULT_TGM_ROOT / "datasets" / "SMILES"
DEFAULT_GRAMMAR_FILE = Path(
    os.environ.get(
        "SDVAE_GRAMMAR_FILE",
        str(DEFAULT_PROJECT_ROOT / "sdvae/dropbox/context_free_grammars/mol_zinc.grammar"),
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


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
    )
    parser.add_argument("--output-suffix", default="_pool90")
    parser.add_argument("--max-heavy-atoms", type=int, default=60)
    parser.add_argument("--max-smiles-len", type=int, default=180)
    parser.add_argument("--target-coverage", type=float, default=0.9)
    parser.add_argument("--grammar-file", default=str(DEFAULT_GRAMMAR_FILE))
    return parser


def load_rows(split_file):
    rows = []
    with open(split_file, "r") as f:
        header = next(f)
        for line in f:
            cid, smiles, desc = line.rstrip("\n").split("\t")
            if smiles != "*":
                rows.append((int(cid), smiles, desc))
    return header, rows


def normalize_smiles(smi, uncharger, max_heavy_atoms, max_smiles_len):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, "rdkit_parse_failed"
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=True)
    if frags:
        m = max(frags, key=lambda x: x.GetNumHeavyAtoms())
    for atom in m.GetAtoms():
        atom.SetIsotope(0)
    m = uncharger.uncharge(m)
    Chem.RemoveStereochemistry(m)
    try:
        Chem.Kekulize(m, clearAromaticFlags=True)
    except Exception:
        return None, "kekulize_failed"
    if m.GetNumHeavyAtoms() > max_heavy_atoms:
        return None, "too_many_heavy_atoms"
    norm = Chem.MolToSmiles(
        m,
        canonical=True,
        isomericSmiles=False,
        kekuleSmiles=True,
    )
    if len(norm) > max_smiles_len:
        return None, "too_long_smiles"
    return norm, None


def can_parse(smiles, grammar, parser_module):
    try:
        ts = parser_module.parse(smiles, grammar)
        return isinstance(ts, list) and len(ts) == 1
    except Exception:
        return False


def main():
    args = create_argparser().parse_args()
    dataset_dir = resolve_path(args.dataset_dir, SCRIPT_DIR)
    grammar_file = resolve_path(args.grammar_file, SCRIPT_DIR)
    if not grammar_file.exists():
        raise FileNotFoundError(f"grammar file not found: {grammar_file}")

    # Lazy import to keep script independent from cwd.
    sdvae_cfg_parser = resolve_path(
        "../../../sdvae/mol_vae/cfg_parser",
        SCRIPT_DIR,
    )
    sys.path.append(str(sdvae_cfg_parser))
    import cfg_parser as parser

    grammar = parser.Grammar(str(grammar_file))
    uncharger = rdMolStandardize.Uncharger()

    total_all = 0
    kept_all = 0
    for split in tqdm(args.splits, desc="pool90 splits", unit="split"):
        src_file = os.path.join(dataset_dir, f"{split}.txt")
        dst_split = f"{split}{args.output_suffix}"
        dst_file = os.path.join(dataset_dir, f"{dst_split}.txt")

        header, rows = load_rows(src_file)
        total = len(rows)
        kept = 0
        reasons = Counter()
        cache = {}

        with open(dst_file, "w") as out:
            out.write(header)
            for cid, smiles, desc in tqdm(
                rows,
                desc=f"filter:{split}",
                unit="row",
                leave=False,
            ):
                if smiles in cache:
                    out_smiles, reason = cache[smiles]
                else:
                    norm, reason = normalize_smiles(
                        smiles,
                        uncharger=uncharger,
                        max_heavy_atoms=args.max_heavy_atoms,
                        max_smiles_len=args.max_smiles_len,
                    )
                    if norm is None:
                        out_smiles, reason = None, reason
                    elif not can_parse(norm, grammar, parser):
                        out_smiles, reason = None, "grammar_parse_failed"
                    else:
                        out_smiles, reason = norm, None
                    cache[smiles] = (out_smiles, reason)

                if out_smiles is None:
                    reasons[reason] += 1
                    continue
                out.write(f"{cid}\t{out_smiles}\t{desc}\n")
                kept += 1

        cov = kept / total if total else 0.0
        print(f"{split} -> {dst_split}: kept={kept}/{total} ({cov:.4f})")
        print(f"drop_reasons({split})={dict(reasons)}")

        total_all += total
        kept_all += kept

    cov_all = kept_all / total_all if total_all else 0.0
    print(f"overall kept={kept_all}/{total_all} ({cov_all:.4f})")
    if cov_all < args.target_coverage:
        raise RuntimeError(
            f"coverage {cov_all:.4f} < target {args.target_coverage:.4f}; "
            "relax filters (max-heavy-atoms/max-smiles-len) and rerun"
        )


if __name__ == "__main__":
    main()
