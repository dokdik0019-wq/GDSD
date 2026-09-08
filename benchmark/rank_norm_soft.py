#!/usr/bin/env python3
"""rank_norm_soft.py — per-image rank-normalize a soft-map directory.

Mimics the "rank normalization" step of the internal pipeline: replace each
pixel's strength by its percentile rank within that image (average ranks,
uniform in (0,1]). Zeros (non-edge pixels) rank together at the bottom, so
global thresholds become per-image relative quantiles.

Usage: rank_norm_soft.py <in_soft_dir(.npy)> <out_dir(.npy)> [<out_png_dir>]
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata


def rank_normalize(soft: np.ndarray) -> np.ndarray:
    """Map pixel values to their within-image percentile rank (0..1].  Flat
    (zero) regions share the lowest average rank, so they stay suppressed."""
    n = soft.size
    r = rankdata(soft.ravel(), method="average")  # 1..n
    return (r / n).reshape(soft.shape).astype(np.float32)


def main():
    in_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    png_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    out_dir.mkdir(parents=True, exist_ok=True)
    if png_dir is not None:
        png_dir.mkdir(parents=True, exist_ok=True)
        import cv2
    files = sorted(in_dir.glob("*.npy"))
    for f in files:
        soft = np.load(f)
        rn = rank_normalize(soft)
        np.save(out_dir / f.name, rn)
        if png_dir is not None:
            img8 = np.clip(rn * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(str(png_dir / f"{f.stem}.png"), img8)
    print(f"done rank-norm: {len(files)} -> {out_dir}")


if __name__ == "__main__":
    main()
