#!/usr/bin/env python
"""Compare pasted object stats against the real class distribution.

Metrics: position (cx, cy), size (w, h in px), aspect ratio (w/h),
and object-region brightness (mean pixel).

Usage:
  python aug/compare_stats.py --real-labels <labels/train> --real-images <images/train> \
      --paste-labels <out/labels> --paste-images <out/images> --cls 9
"""
import os
import glob
import argparse
import numpy as np
from PIL import Image

CLASS_NAMES = [
    "ball", "cube", "human_body", "tyre", "square_cage",
    "plane", "rov", "circle_cage", "cylinder", "metal_bucket",
]


def find_image(images_dir, base):
    for ext in (".bmp", ".png", ".jpg", ".jpeg"):
        p = os.path.join(images_dir, base + ext)
        if os.path.exists(p):
            return p
    return None


def collect_real(labels_dir, images_dir, cls):
    recs = []
    for lf in glob.glob(os.path.join(labels_dir, "*.txt")):
        base = os.path.splitext(os.path.basename(lf))[0]
        img_path = find_image(images_dir, base)
        if img_path is None:
            continue
        with Image.open(img_path) as im:
            W, H = im.size
            arr = np.asarray(im.convert("L"))
        for line in open(lf):
            p = line.split()
            if len(p) < 5 or int(float(p[0])) != cls:
                continue
            cx, cy, w, h = (float(x) for x in p[1:5])
            x0 = int((cx - w / 2) * W); y0 = int((cy - h / 2) * H)
            x1 = int((cx + w / 2) * W); y1 = int((cy + h / 2) * H)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(W, x1), min(H, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = arr[y0:y1, x0:x1]
            recs.append(dict(cx=cx, cy=cy, w=w * W, h=h * H, bright=crop.mean()))
    return recs


def collect_pasted(labels_dir, images_dir):
    recs = []
    for lf in glob.glob(os.path.join(labels_dir, "*.txt")):
        lines = open(lf).read().strip().splitlines()
        if not lines:
            continue
        p = lines[-1].split()  # new object = last line
        base = os.path.splitext(os.path.basename(lf))[0]
        img_path = find_image(images_dir, base)
        if img_path is None:
            continue
        with Image.open(img_path) as im:
            W, H = im.size
            arr = np.asarray(im.convert("L"))
        cx, cy, w, h = (float(x) for x in p[1:5])
        x0 = int((cx - w / 2) * W); y0 = int((cy - h / 2) * H)
        x1 = int((cx + w / 2) * W); y1 = int((cy + h / 2) * H)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        crop = arr[y0:y1, x0:x1]
        recs.append(dict(cx=cx, cy=cy, w=w * W, h=h * H, bright=crop.mean()))
    return recs


def summarize(recs, name):
    cx = np.array([r["cx"] for r in recs]); cy = np.array([r["cy"] for r in recs])
    w = np.array([r["w"] for r in recs]); h = np.array([r["h"] for r in recs])
    aspect = w / np.maximum(h, 1)
    bright = np.array([r["bright"] for r in recs])
    def s(a):
        return f"med={np.median(a):.0f}  p5={np.percentile(a,5):.0f}  p95={np.percentile(a,95):.0f}"
    print(f"\n== {name} (n={len(recs)}) ==")
    print(f"cx (0左1右): mean={cx.mean():.3f} med={np.median(cx):.3f} std={cx.std():.3f}")
    print(f"cy (0上1下): mean={cy.mean():.3f} med={np.median(cy):.3f} std={cy.std():.3f}")
    print(f"w (px): {s(w)}")
    print(f"h (px): {s(h)}")
    print(f"aspect w/h: med={np.median(aspect):.2f} p5={np.percentile(aspect,5):.2f} p95={np.percentile(aspect,95):.2f}")
    print(f"bright (0-255): med={np.median(bright):.1f} mean={bright.mean():.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-labels", required=True)
    ap.add_argument("--real-images", required=True)
    ap.add_argument("--paste-labels", required=True)
    ap.add_argument("--paste-images", required=True)
    ap.add_argument("--cls", type=int, required=True)
    a = ap.parse_args()

    real = collect_real(a.real_labels, a.real_images, a.cls)
    pasted = collect_pasted(a.paste_labels, a.paste_images)
    print(f"class {a.cls} ({CLASS_NAMES[a.cls]})")
    summarize(real, "REAL")
    summarize(pasted, "PASTED (new object)")


if __name__ == "__main__":
    main()
