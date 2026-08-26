#!/usr/bin/env python
"""Build a contact sheet + basic diversity stats from a folder of images.

Usage:
  python aug/inspect_gen.py --dir <folder> --out <sheet.png> [--n 64]
"""
import os
import glob
import argparse
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, "*.png")) + glob.glob(os.path.join(a.dir, "*.jpg")))
    files = files[: a.n]
    if not files:
        print("no images found in", a.dir)
        return

    imgs = []
    for f in files:
        im = Image.open(f).convert("RGB")
        imgs.append(np.asarray(im).astype(np.float32))
    side = imgs[0].shape[0]
    cols = int(np.ceil(np.sqrt(len(imgs))))
    rows = int(np.ceil(len(imgs) / cols))
    sheet = np.zeros((rows * side, cols * side, 3), dtype=np.uint8)
    for i, arr in enumerate(imgs):
        r, c = divmod(i, cols)
        sheet[r * side:(r + 1) * side, c * side:(c + 1) * side] = arr.astype(np.uint8)
    Image.fromarray(sheet).save(a.out)

    # Diversity / quality stats.
    arr = np.stack(imgs)  # [N, H, W, 3]
    global_mean = arr.mean()
    global_std = arr.std()
    per_img_std = arr.reshape(len(imgs), -1).std(axis=1)
    # Pairwise mean-abs-diff of downsampled thumbnails as a cheap diversity proxy.
    thumbs = np.stack([np.asarray(Image.fromarray(x.astype(np.uint8)).resize((16, 16)), dtype=np.float32).ravel()
                       for x in imgs])
    if len(thumbs) > 1:
        pairwise = np.abs(thumbs[None, :, :] - thumbs[:, None, :]).mean(axis=2)
        diversity = pairwise[np.triu_indices(len(thumbs), k=1)].mean()
    else:
        diversity = 0.0

    print(f"count={len(imgs)}  global_mean={global_mean:.1f}  global_std={global_std:.1f}")
    print(f"per_img_std mean={per_img_std.mean():.1f}  min={per_img_std.min():.1f}  max={per_img_std.max():.1f}")
    print(f"pairwise_diversity={diversity:.2f}  sheet={a.out}")


if __name__ == "__main__":
    main()
