#!/usr/bin/env python
"""Extract object crops for a given class from UATD YOLO dataset.

Crops bbox region (+padding), pads to square, converts grayscale L -> RGB,
resizes to target size. Outputs PNG per instance for StyleGAN2-ADA training.
"""
import os
import glob
import argparse
from PIL import Image

CLASS_NAMES = [
    "ball", "cube", "human_body", "tyre", "square_cage",
    "plane", "rov", "circle_cage", "cylinder", "metal_bucket",
]


def crop_instances(labels_dir, images_dir, out_dir, cls_id, size=128, pad=1.6):
    os.makedirs(out_dir, exist_ok=True)
    label_files = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
    count = 0
    for lf in label_files:
        base = os.path.splitext(os.path.basename(lf))[0]
        img_path = None
        for ext in (".bmp", ".png", ".jpg", ".jpeg"):
            cand = os.path.join(images_dir, base + ext)
            if os.path.exists(cand):
                img_path = cand
                break
        if img_path is None:
            continue
        with open(lf) as f:
            lines = f.read().strip().splitlines()
        if not lines:
            continue
        img = Image.open(img_path).convert("L")
        W, H = img.size
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            if int(float(parts[0])) != cls_id:
                continue
            cx, cy, bw, bh = (float(x) for x in parts[1:5])
            px, py = cx * W, cy * H
            pw, ph = bw * W * pad, bh * H * pad
            x0 = max(0, int(px - pw / 2))
            y0 = max(0, int(py - ph / 2))
            x1 = min(W, int(px + pw / 2))
            y1 = min(H, int(py + ph / 2))
            if x1 <= x0 or y1 <= y0:
                continue
            crop = img.crop((x0, y0, x1, y1))
            # Pad to square with black (sonar background).
            cw, ch = crop.size
            side = max(cw, ch)
            square = Image.new("L", (side, side), 0)
            square.paste(crop, ((side - cw) // 2, (side - ch) // 2))
            square = square.resize((size, size), Image.LANCZOS)
            rgb = square.convert("RGB")
            out_name = f"{base}_{cls_id}_{count:05d}.png"
            rgb.save(os.path.join(out_dir, out_name))
            count += 1
    print(f"class {cls_id} ({CLASS_NAMES[cls_id]}): {count} crops -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cls", type=int, required=True)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--pad", type=float, default=1.6)
    a = ap.parse_args()
    crop_instances(a.labels, a.images, a.out, a.cls, a.size, a.pad)
