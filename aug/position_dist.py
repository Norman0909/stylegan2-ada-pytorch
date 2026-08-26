#!/usr/bin/env python
"""Position distribution of a class in the UATD train set.

Collects normalized center (cx, cy) of each instance of --cls, plots a 2D
heatmap and prints coarse region stats. Images have varying sizes, so all
positions are normalized to [0, 1].

Usage:
  python aug/position_dist.py --labels <labels/train> --cls 9 --out <heat.png>
"""
import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_NAMES = [
    "ball", "cube", "human_body", "tyre", "square_cage",
    "plane", "rov", "circle_cage", "cylinder", "metal_bucket",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--cls", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bins", type=int, default=12)
    a = ap.parse_args()

    pts = []
    for lf in sorted(glob.glob(os.path.join(a.labels, "*.txt"))):
        for line in open(lf):
            p = line.split()
            if len(p) < 5:
                continue
            if int(float(p[0])) != a.cls:
                continue
            cx, cy = float(p[1]), float(p[2])
            pts.append((cx, cy))
    pts = np.array(pts)
    if len(pts) == 0:
        raise SystemExit("no instances found for class", a.cls)

    cx, cy = pts[:, 0], pts[:, 1]
    print(f"class {a.cls} ({CLASS_NAMES[a.cls]}): n={len(pts)}")
    print(f"cx: mean={cx.mean():.3f} med={np.median(cx):.3f} std={cx.std():.3f}  (0=left 1=right)")
    print(f"cy: mean={cy.mean():.3f} med={np.median(cy):.3f} std={cy.std():.3f}  (0=top 1=bottom)")

    # Coarse 3x3 region occupancy (%).
    print("\n3x3 region occupancy (%):")
    print("        left   center  right")
    for ri, rlab in enumerate(["top", "mid ", "bot "]):
        row = []
        for ci in range(3):
            m = ((cx >= ci / 3) & (cx < (ci + 1) / 3) &
                 (cy >= ri / 3) & (cy < (ri + 1) / 3))
            row.append(100 * m.mean())
        print(f"{rlab}  " + "  ".join(f"{v:5.1f}" for v in row))

    # Heatmap.
    H, xedges, yedges = np.histogram2d(cx, cy, bins=a.bins, range=[[0, 1], [0, 1]])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(H.T, origin="upper", extent=[0, 1, 1, 0], cmap="hot", aspect="auto")
    ax.set_xlabel("cx (0=left 1=right)")
    ax.set_ylabel("cy (0=top 1=bottom)")
    ax.set_title(f"class {a.cls} ({CLASS_NAMES[a.cls]}) position heatmap, n={len(pts)}")
    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
