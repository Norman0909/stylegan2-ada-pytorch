#!/usr/bin/env python
"""Plot StyleGAN2-ADA training metrics from a run's stats.jsonl.

Usage:
  python aug/plot_metrics.py --run runs/00000-metal_bucket-auto1-kimg100 \
      --out runs/00000-metal_bucket-auto1-kimg100/metrics.png
"""
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ADA_TARGET = 0.6


def load(run_dir):
    path = os.path.join(run_dir, "stats.jsonl")
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # Tolerate a trailing line still being written.
            break
    return rows


def series(rows, key):
    xs, ys = [], []
    for r in rows:
        if "Progress/kimg" not in r:
            continue
        kx = r["Progress/kimg"]
        x = kx["mean"] if isinstance(kx, dict) else kx
        if key not in r:
            continue
        v = r[key]
        y = v["mean"] if isinstance(v, dict) else v
        if y is None or (isinstance(y, float) and (np.isnan(y) or np.isinf(y))):
            continue
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def smooth(x, y, w):
    if len(y) < 3:
        return x, y
    w = min(w, len(y))
    if w <= 1:
        return x, y
    kernel = np.ones(w) / w
    ys = np.convolve(y, kernel, mode="same")
    half = w // 2
    # Fix edge artifacts by clamping to the first/last valid convolved value.
    ys[:half] = ys[half]
    ys[-half:] = ys[-half - 1]
    return x, ys


def plot_metric(ax, rows, key, color, label, w=100, ylog=False, **kw):
    x, y = series(rows, key)
    if len(y) == 0:
        ax.text(0.5, 0.5, f"no data: {key}", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(key)
        return
    xs, ys = smooth(x, y, min(w, max(1, len(y))))
    ax.plot(xs, ys, color=color, lw=1.5, label=label, **kw)
    if ylog:
        ax.set_yscale("log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smooth", type=int, default=100)
    a = ap.parse_args()

    rows = load(a.run)
    if not rows:
        raise SystemExit(f"no rows in {a.run}/stats.jsonl")
    out = a.out or os.path.join(a.run, "metrics.png")
    w = a.smooth

    fig, axes = plt.subplots(3, 2, figsize=(13, 14))
    fig.suptitle(os.path.basename(a.run.rstrip("/\\")), fontsize=12)

    ax = axes[0, 0]
    plot_metric(ax, rows, "Loss/G/loss", "tab:blue", "G/loss", w)
    plot_metric(ax, rows, "Loss/D/loss", "tab:red", "D/loss", w)
    ax.set_title("Loss (lower = better)"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    plot_metric(ax, rows, "Loss/scores/real", "tab:green", "real", w)
    plot_metric(ax, rows, "Loss/scores/fake", "tab:orange", "fake", w)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_title("D logits (real>0, fake<0 = healthy)"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    plot_metric(ax, rows, "Loss/signs/real", "tab:green", "signs/real", w)
    plot_metric(ax, rows, "Loss/signs/fake", "tab:orange", "signs/fake", w)
    ax.axhline(ADA_TARGET, color="red", ls="--", lw=1, label=f"ada_target={ADA_TARGET}")
    ax.set_title("D logit signs (ADA driver)"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    plot_metric(ax, rows, "Progress/augment", "tab:purple", "augment p", w)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("ADA augmentation probability p (0=off, 1=max)"); ax.grid(alpha=0.3)

    ax = axes[2, 0]
    plot_metric(ax, rows, "Loss/r1_penalty", "tab:brown", "r1_penalty", w, ylog=True)
    ax.set_title("R1 penalty (log)"); ax.grid(alpha=0.3)

    ax = axes[2, 1]
    plot_metric(ax, rows, "Loss/pl_penalty", "tab:pink", "pl_penalty", w, ylog=True)
    ax.set_title("Path-length penalty (log)"); ax.grid(alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("kimg")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
