"""Download UIEB (standard 800/90 split) and build the semi-supervised split.

Source: HuggingFace mirror `Hikari0608/UIEB` (parquet, raw/gt pairs, 800 train / 90 val),
which follows the split used by Semi-UIR. Shards are streamed one at a time, every image
is resized so that its short side is 256 px, and the shard is deleted to save disk.

Output tree (default --root data/UIEB):
    labeled/{input,gt}/      first N_LABELED training pairs   (supervised branch)
    unlabeled/input/         remaining training inputs (unsupervised branch, GT never used for training)
    unlabeled_gt_analysis/   GT of the unlabeled images -- ONLY read by the pseudo-label
                             quality diagnostic (bank PSNR curve), never by the losses
    test/{input,gt}/         90 held-out pairs                 (evaluation only)

Usage:
    python data/prepare_uieb.py --root data/UIEB --n_labeled 200
"""
import argparse
import io
import os
import subprocess

import pyarrow.parquet as pq
from PIL import Image

REPO = "https://huggingface.co/datasets/Hikari0608/UIEB/resolve/main/data/"
SHARDS = {
    "train": ["train-00000-of-00003-400f34c28a61ba94.parquet",
              "train-00001-of-00003-5e4553c4feca8b3e.parquet",
              "train-00002-of-00003-8399d261352e3b11.parquet"],
    "test": ["val-00000-of-00001-ee22087fd151f704.parquet"],
}


def resize_short(img, short=256):
    w, h = img.size
    s = short / min(w, h)
    return img.resize((max(short, round(w * s)), max(short, round(h * s))), Image.BICUBIC)


def save(raw_bytes, path, short):
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    resize_short(img, short).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/UIEB")
    ap.add_argument("--n_labeled", type=int, default=200)
    ap.add_argument("--short", type=int, default=256)
    args = ap.parse_args()

    for d in ["labeled/input", "labeled/gt", "unlabeled/input", "unlabeled_gt_analysis", "test/input", "test/gt"]:
        os.makedirs(os.path.join(args.root, d), exist_ok=True)

    for split, shards in SHARDS.items():
        idx = 0
        for shard in shards:
            tmp = os.path.join(args.root, shard)
            marker = os.path.join(args.root, f".done_{shard}")
            if os.path.exists(marker):  # resume: shard already processed
                idx = int(open(marker).read())
                continue
            print(f"downloading {shard} ...", flush=True)
            subprocess.run(["curl", "-sSL", "--retry", "10", "--retry-all-errors", "-C", "-",
                            "--speed-limit", "10000", "--speed-time", "60", "-o", tmp, REPO + shard], check=True)
            table = pq.read_table(tmp, columns=["raw", "gt"]).to_pylist()
            for row in table:
                name = f"{idx:04d}.png"
                if split == "test":
                    save(row["raw"]["bytes"], os.path.join(args.root, "test/input", name), args.short)
                    save(row["gt"]["bytes"], os.path.join(args.root, "test/gt", name), args.short)
                elif idx < args.n_labeled:
                    save(row["raw"]["bytes"], os.path.join(args.root, "labeled/input", name), args.short)
                    save(row["gt"]["bytes"], os.path.join(args.root, "labeled/gt", name), args.short)
                else:
                    save(row["raw"]["bytes"], os.path.join(args.root, "unlabeled/input", name), args.short)
                    save(row["gt"]["bytes"], os.path.join(args.root, "unlabeled_gt_analysis", name), args.short)
                idx += 1
            del table
            os.remove(tmp)
            open(marker, "w").write(str(idx))
        print(f"{split}: {idx} images")

    for d in ["labeled/input", "unlabeled/input", "unlabeled_gt_analysis", "test/input"]:
        print(d, len(os.listdir(os.path.join(args.root, d))))


if __name__ == "__main__":
    main()
