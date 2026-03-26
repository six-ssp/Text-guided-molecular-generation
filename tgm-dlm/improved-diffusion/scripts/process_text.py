import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModel
from transformers import AutoTokenizer
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


SCRIPT_DIR = Path(__file__).resolve().parent
TGM_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATASET_DIR = TGM_ROOT / "datasets" / "SMILES"
DEFAULT_TEXT_MODEL = TGM_ROOT / "scibert"


def resolve_path(path_str, preferred_base):
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    cwd_p = (Path.cwd() / p).resolve()
    if cwd_p.exists():
        return cwd_p
    return (preferred_base / p).resolve()

def atomic_torch_save(obj, path):
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True)
parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
parser.add_argument("--text-model", default=str(DEFAULT_TEXT_MODEL))
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--save-every", type=int, default=20)
parser.add_argument("--output", default=None)
parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
parser.add_argument("--gpu-id", type=int, default=0)
parser.add_argument("--max-length", type=int, default=216)
parser.add_argument(
    "--state-dtype",
    choices=["float32", "float16", "bfloat16"],
    default="float32",
    help="dtype used to store text hidden states on disk",
)
args = parser.parse_args()

def resolve_device(mode, gpu_id):
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        try:
            torch.cuda.set_device(gpu_id)
            torch.empty(1, device=f"cuda:{gpu_id}")
            return torch.device(f"cuda:{gpu_id}")
        except Exception as exc:
            raise RuntimeError(
                f"--device cuda requested but CUDA init failed on gpu_id={gpu_id}: {exc}"
            ) from exc
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            torch.empty(1, device=f"cuda:{gpu_id}")
            return torch.device(f"cuda:{gpu_id}")
    except Exception:
        pass
    return torch.device("cpu")

split = args.input
dataset_dir = resolve_path(args.dataset_dir, SCRIPT_DIR)
dataset_file = dataset_dir / f"{split}.txt"
rows = []
with open(dataset_file, "r") as f:
    next(f)
    for line in f:
        cid, smiles, desc = line.rstrip("\n").split("\t")
        if smiles != "*":
            rows.append((int(cid), desc))

model_name = args.text_model
model_path = resolve_path(model_name, SCRIPT_DIR)
if model_path.exists():
    model_name = str(model_path)
if not os.path.exists(os.path.join(model_name, "config.json")):
    model_name = "allenai/scibert_scivocab_uncased"

model = AutoModel.from_pretrained(model_name)
tokz = AutoTokenizer.from_pretrained(model_name)
device = resolve_device(args.device, args.gpu_id)
print(f"process_text device={device}")

state_dtype = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}[args.state_dtype]

if args.output is not None:
    output_path = str(resolve_path(args.output, SCRIPT_DIR))
else:
    output_path = str(dataset_dir / f"{split}_desc_states.pt")
if os.path.exists(output_path):
    volume = torch.load(output_path, map_location="cpu")
    print(f"resume from {output_path}, loaded {len(volume)} items")
else:
    volume = {}


model = model.to(device)
model.eval()
num_batches = (len(rows) + args.batch_size - 1) // args.batch_size
with torch.inference_mode():
    for start in tqdm(
        range(0, len(rows), args.batch_size),
        total=num_batches,
        desc=f"text:{split}",
        unit="batch",
    ):
        batch = rows[start: min(start + args.batch_size, len(rows))]
        ids = [item[0] for item in batch]
        descs = [item[1] for item in batch]
        if all(cid in volume for cid in ids):
            if start % (args.batch_size * args.save_every) == 0:
                print(f"skip {start}/{len(rows)}")
            continue
        tok_op = tokz(
            descs,
            max_length=args.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        toked_desc = tok_op['input_ids']
        toked_desc_attentionmask = tok_op['attention_mask']
        lh = model(
            toked_desc.to(device),
            attention_mask=toked_desc_attentionmask.to(device),
        ).last_hidden_state.to('cpu').to(state_dtype)
        for offset, cid in enumerate(ids):
            volume[cid] = {
                'states': lh[offset:offset + 1],
                'mask': toked_desc_attentionmask[offset:offset + 1].to(torch.uint8),
            }
        batch_idx = start // args.batch_size
        if batch_idx % args.save_every == 0:
            print(f"save {start}/{len(rows)} -> {output_path}")
            atomic_torch_save(volume, output_path)


atomic_torch_save(volume, output_path)
print(f"finished {split}, saved {len(volume)} items to {output_path}")
