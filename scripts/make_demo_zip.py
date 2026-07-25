"""Build a .zip of class folders to demo the upload + retrain flow.

    python -m scripts.make_demo_zip --per-class 25 --out demo_upload.zip

Pulls images from the CIFAR test split and lays them out as class_name/image.png,
which is exactly the layout the /upload endpoint expects. Use this in the video
demo so you are not hunting for images live.
"""
import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import load_cifar_split, load_label_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=25)
    ap.add_argument("--classes", type=int, default=4, help="how many classes to include")
    ap.add_argument("--out", default="demo_upload.zip")
    args = ap.parse_args()

    images, labels = load_cifar_split("test")
    names = load_label_names()
    rng = np.random.default_rng(7)
    chosen = rng.choice(len(names), size=min(args.classes, len(names)), replace=False)

    written = 0
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for ci in chosen:
            idx = np.where(labels == ci)[0][: args.per_class]
            for j, i in enumerate(idx):
                buf = io.BytesIO()
                Image.fromarray(images[i]).save(buf, format="PNG")
                zf.writestr(f"{names[ci]}/{names[ci]}_{j:03d}.png", buf.getvalue())
                written += 1

    print(f"Wrote {written} images across {len(chosen)} classes to {args.out}")
    print("Classes:", ", ".join(names[c] for c in chosen))


if __name__ == "__main__":
    main()
