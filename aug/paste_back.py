#!/usr/bin/env python
"""Paste generated object crops back into sonar backgrounds as new samples.

For each generated 128x128 RGB crop:
  1. convert to grayscale, threshold to find the foreground blob
  2. extract the object bbox (+margin), apply random rotate/flip/scale
  3. alpha-paste into a random UATD background, with the object CENTER placed
     at a position sampled from the REAL class position distribution
     (if --pos-labels given), avoiding overlap with existing objects
  4. write a YOLO label = original background objects + the new pasted object
     (the new object is the LAST line)

Produces: <out>/images/*.png  and  <out>/labels/*.txt

Usage:
  python aug/paste_back.py --crops <dir> --backgrounds <images/train> \
      --pos-labels <labels/train> --out aug_out --cls 9 --seed 0
"""
import os
import glob
import json
import random
import argparse
import numpy as np
from PIL import Image

CLASS_NAMES = [
    "ball", "cube", "human_body", "tyre", "square_cage",
    "plane", "rov", "circle_cage", "cylinder", "metal_bucket",
]


def bbox_of_mask(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())  # x0,y0,x1,y1


def iou(a, b):
    """IoU of two boxes (x0, y0, x1, y1)."""
    ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def extract_object(gray_arr, thr=8, margin=4):
    mask = gray_arr > thr
    b = bbox_of_mask(mask)
    if b is None:
        return None, None
    x0, y0, x1, y1 = b
    H, W = gray_arr.shape
    x0 = max(0, x0 - margin); y0 = max(0, y0 - margin)
    x1 = min(W - 1, x1 + margin); y1 = min(H - 1, y1 + margin)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None, None
    return gray_arr[y0:y1 + 1, x0:x1 + 1], (x0, y0, x1, y1)


def transform_patch(patch, rng):
    img = Image.fromarray(patch)
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    ang = rng.uniform(-180, 180)
    img = img.rotate(ang, expand=True, fillcolor=0, resample=Image.BICUBIC)
    arr = np.asarray(img)
    b = bbox_of_mask(arr > 8)
    return arr, b, img


def load_label(path):
    boxes = []
    if not os.path.exists(path):
        return boxes
    for line in open(path):
        p = line.split()
        if len(p) < 5:
            continue
        boxes.append((int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return boxes


def sample_center(rng, real_centers):
    """Return normalized (cx, cy) sampled from real distribution or uniform."""
    if real_centers:
        cx, cy = rng.choice(real_centers)
        cx = min(1.0, max(0.0, cx + rng.gauss(0, 0.015)))
        cy = min(1.0, max(0.0, cy + rng.gauss(0, 0.015)))
    else:
        cx, cy = rng.random(), rng.random()
    return cx, cy


def paste_one(crop_path, bg, cls_id, rng, real_centers, orig_boxes, overlap_thr, max_tries,
              tw_min, tw_max, max_w, max_h):
    crop = Image.open(crop_path).convert("L")
    arr = np.asarray(crop)
    patch, _ = extract_object(arr)
    if patch is None:
        return None

    patch, b, pimg = transform_patch(patch, rng)
    if b is None:
        return None
    px0, py0, px1, py1 = b
    ph, pw = patch.shape

    # Random scale (target object width in pixels).
    target_w = rng.uniform(tw_min, tw_max)
    scale = target_w / max(1, (px1 - px0))
    new_w = int(pw * scale)
    new_h = int(ph * scale)
    if new_w < 3 or new_h < 3:
        return None
    pimg = pimg.resize((new_w, new_h), Image.LANCZOS)
    patch = np.asarray(pimg)
    mask = (patch > 8).astype(np.uint8) * 255
    b = bbox_of_mask(mask)
    if b is None:
        return None
    ox0, oy0, ox1, oy1 = b
    obj_cx = (ox0 + ox1) / 2.0
    obj_cy = (oy0 + oy1) / 2.0
    obj_w = ox1 - ox0 + 1
    obj_h = oy1 - oy0 + 1

    # Reject clearly oversized objects (generated blobs can be too large/square).
    if obj_w > max_w or obj_h > max_h:
        return None

    W, H = bg.size
    if new_w >= W or new_h >= H:
        return None

    # Original boxes in pixel coords for overlap checking.
    orig_px = []
    for cls, cx, cy, w, h in orig_boxes:
        x0 = (cx - w / 2) * W; y0 = (cy - h / 2) * H
        x1 = (cx + w / 2) * W; y1 = (cy + h / 2) * H
        orig_px.append((x0, y0, x1, y1))

    # Rejection sampling: find a position with no overlap.
    best = None
    best_iou = 1.0
    for _ in range(max_tries):
        cx, cy = sample_center(rng, real_centers)
        px, py = cx * W, cy * H
        x0 = int(px - obj_cx); y0 = int(py - obj_cy)
        x0 = max(0, min(W - new_w, x0))
        y0 = max(0, min(H - new_h, y0))
        obj_x0 = x0 + ox0; obj_y0 = y0 + oy0
        new_box = (obj_x0, obj_y0, obj_x0 + obj_w, obj_y0 + obj_h)
        mx = max([iou(new_box, o) for o in orig_px], default=0.0)
        if mx < overlap_thr:
            best = (x0, y0, new_box)
            best_iou = mx
            break
        if mx < best_iou:
            best_iou = mx
            best = (x0, y0, new_box)
    if best is None:
        return None
    x0, y0, new_box = best

    bg = bg.convert("L").copy()
    alpha = Image.fromarray(mask)
    bg.paste(pimg, (x0, y0), alpha)

    obj_x0, obj_y0, obj_x1, obj_y1 = new_box
    ncx = ((obj_x0 + obj_x1) / 2) / W
    ncy = ((obj_y0 + obj_y1) / 2) / H
    nw = (obj_x1 - obj_x0) / W
    nh = (obj_y1 - obj_y0) / H
    return bg, (cls_id, ncx, ncy, nw, nh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", required=True, help="dir of generated 128x128 crops")
    ap.add_argument("--backgrounds", required=True, help="dir of background images")
    ap.add_argument("--pos-labels", default=None, help="dir of real labels for position distribution")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cls", type=int, required=True)
    ap.add_argument("--overlap-threshold", type=float, default=0.1)
    ap.add_argument("--max-tries", type=int, default=40)
    ap.add_argument("--tw-min", type=float, default=60, help="min target object width (px)")
    ap.add_argument("--tw-max", type=float, default=120, help="max target object width (px)")
    ap.add_argument("--max-w", type=float, default=140, help="reject if object wider than this (px)")
    ap.add_argument("--max-h", type=float, default=130, help="reject if object taller than this (px)")
    ap.add_argument("--max", type=int, default=0, help="limit crops processed (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    crops = sorted(glob.glob(os.path.join(a.crops, "*.png")) + glob.glob(os.path.join(a.crops, "*.jpg")))
    if a.max:
        crops = crops[: a.max]

    bg_files = []
    for ext in ("*.bmp", "*.png", "*.jpg"):
        bg_files += sorted(glob.glob(os.path.join(a.backgrounds, ext)))
    if not bg_files:
        raise SystemExit("no backgrounds found")

    real_centers = None
    if a.pos_labels:
        real_centers = []
        for lf in glob.glob(os.path.join(a.pos_labels, "*.txt")):
            for line in open(lf):
                p = line.split()
                if len(p) >= 5 and int(float(p[0])) == a.cls:
                    real_centers.append((float(p[1]), float(p[2])))
        if not real_centers:
            raise SystemExit(f"no instances of class {a.cls} found in --pos-labels")

    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "labels"), exist_ok=True)

    done = 0
    skipped = 0
    manifest = []
    for ci, cp in enumerate(crops):
        bg_path = rng.choice(bg_files)
        bg = Image.open(bg_path)
        bg_base = os.path.splitext(os.path.basename(bg_path))[0]
        orig = load_label(os.path.join(a.pos_labels or a.backgrounds, bg_base + ".txt"))

        res = paste_one(cp, bg, a.cls, rng, real_centers, orig, a.overlap_threshold, a.max_tries,
                        a.tw_min, a.tw_max, a.max_w, a.max_h)
        if res is None:
            skipped += 1
            continue
        bg_out, new_box = res
        name = f"aug_{a.cls}_{done:06d}"
        bg_out.save(os.path.join(a.out, "images", name + ".png"))
        with open(os.path.join(a.out, "labels", name + ".txt"), "w") as f:
            for box in orig:
                f.write(" ".join(f"{v:.6f}" if i > 0 else str(v) for i, v in enumerate(box)) + "\n")
            cls, cx, cy, nw, nh = new_box
            f.write(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        manifest.append({"out": name, "bg": bg_base})
        done += 1

    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"class {a.cls} ({CLASS_NAMES[a.cls]}): {done} pasted, {skipped} skipped -> {a.out}")


if __name__ == "__main__":
    main()
