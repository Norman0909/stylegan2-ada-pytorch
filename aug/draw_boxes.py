#!/usr/bin/env python
"""Draw YOLO bounding boxes onto images and build a contact sheet.

Usage:
  python aug/draw_boxes.py --images <dir> --labels <dir> --out <sheet.png> \
      --n 25 --cols 5 --thumb 256
"""
import os
import glob
import argparse
from PIL import Image, ImageDraw

COLOR_NEW = (0, 255, 0)    # green: pasted new object
COLOR_ORIG = (255, 255, 255)  # white: original background objects


def load_label(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    for line in open(label_path):
        p = line.split()
        if len(p) < 5:
            continue
        cls = int(float(p[0]))
        cx, cy, w, h = (float(x) for x in p[1:5])
        boxes.append((cls, cx, cy, w, h))
    return boxes


def draw(img, boxes, new_last=True):
    rgb = img.convert("RGB")
    W, H = rgb.size
    d = ImageDraw.Draw(rgb)
    n = len(boxes)
    for i, (cls, cx, cy, w, h) in enumerate(boxes):
        x0 = (cx - w / 2) * W
        y0 = (cy - h / 2) * H
        x1 = (cx + w / 2) * W
        y1 = (cy + h / 2) * H
        color = COLOR_NEW if (new_last and i == n - 1) else COLOR_ORIG
        d.rectangle([x0, y0, x1, y1], outline=color, width=3)
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--thumb", type=int, default=256)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.images, "*.png")) +
                   glob.glob(os.path.join(a.images, "*.bmp")) +
                   glob.glob(os.path.join(a.images, "*.jpg")))[: a.n]

    thumbs = []
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        label_path = os.path.join(a.labels, base + ".txt")
        img = Image.open(f)
        boxes = load_label(label_path)
        drawn = draw(img, boxes)
        tw = a.thumb
        th = int(drawn.height * tw / drawn.width)
        thumbs.append(drawn.resize((tw, th)))

    cols = a.cols
    rows = (len(thumbs) + cols - 1) // cols
    pad = 4
    cell_h = max(t.height for t in thumbs)
    grid = Image.new("RGB", (cols * (a.thumb + pad) + pad, rows * (cell_h + pad) + pad), (30, 30, 30))
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        grid.paste(t, (pad + c * (a.thumb + pad), pad + r * (cell_h + pad)))
    grid.save(a.out)
    print(f"saved {a.out} ({len(thumbs)} images with boxes)")


if __name__ == "__main__":
    main()
