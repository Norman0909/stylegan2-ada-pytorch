#!/usr/bin/env python
"""Build an augmented YOLO dataset: original train (hardlinked) + pasted samples.

Original images/labels are hardlinked (instant, no extra disk); pasted samples
are copied. Writes data_aug.yaml.

Usage:
  python aug/build_dataset.py --orig-images <images/train> --orig-labels <labels/train> \
      --pasted-images <exp>/pasted_full/images --pasted-labels <exp>/pasted_full/labels \
      --val-images <images/val> --val-labels <labels/val> --out <exp>/dataset \
      --names ball cube "human body" tyre "square cage" plane rov "circle cage" cylinder "metal bucket"
"""
import os
import glob
import shutil
import argparse

NAMES = [
    "ball", "cube", "human body", "tyre", "square cage",
    "plane", "rov", "circle cage", "cylinder", "metal bucket",
]


def link_or_copy(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)  # hardlink (same volume), instant + no disk
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def link_dir(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for f in glob.glob(os.path.join(src_dir, "*")):
        link_or_copy(f, os.path.join(dst_dir, os.path.basename(f)))


def copy_dir(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for f in glob.glob(os.path.join(src_dir, "*")):
        shutil.copy2(f, os.path.join(dst_dir, os.path.basename(f)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-images", required=True)
    ap.add_argument("--orig-labels", required=True)
    ap.add_argument("--pasted-images", required=True)
    ap.add_argument("--pasted-labels", required=True)
    ap.add_argument("--val-images", required=True)
    ap.add_argument("--val-labels", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    link_dir(a.orig_images, os.path.join(a.out, "images", "train"))
    link_dir(a.orig_labels, os.path.join(a.out, "labels", "train"))
    copy_dir(a.pasted_images, os.path.join(a.out, "images", "train"))
    copy_dir(a.pasted_labels, os.path.join(a.out, "labels", "train"))
    link_dir(a.val_images, os.path.join(a.out, "images", "val"))
    link_dir(a.val_labels, os.path.join(a.out, "labels", "val"))

    yaml = (
        f"# Augmented UATD dataset (see experiment dir).\n"
        f"path: {os.path.abspath(a.out)}\n"
        f"train: images/train\nval: images/val\n\nnc: {len(NAMES)}\nnames:\n"
    )
    for i, n in enumerate(NAMES):
        yaml += f"  {i}: {n}\n"
    with open(os.path.join(a.out, "data_aug.yaml"), "w") as f:
        f.write(yaml)

    n_tr_img = len(glob.glob(os.path.join(a.out, "images", "train", "*")))
    n_tr_lbl = len(glob.glob(os.path.join(a.out, "labels", "train", "*")))
    n_val = len(glob.glob(os.path.join(a.out, "images", "val", "*")))
    print(f"built {a.out}: train img={n_tr_img} lbl={n_tr_lbl} val={n_val}")


if __name__ == "__main__":
    main()
