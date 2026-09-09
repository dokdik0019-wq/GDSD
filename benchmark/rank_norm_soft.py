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
    """Map nonzero pixel strengths to their within-image percentile rank;
    zero pixels stay exactly zero (as in the internal pipeline).

    Rank normalization must NOT give background pixels a nonzero value:
    ranking zeros together with edge pixels would spread background across
    the low end of (0,1] and make it fire at low thresholds, destroying
    precision.  Only the nonzero (candidate) pixels are ranked, then the
    result is placed back at their locations.
    """
    out = np.zeros(soft.shape, dtype=np.float32)
    flat = soft.ravel()
    nz = flat > 0
    n_nz = int(nz.sum())
    if n_nz == 0:
        return out
    r = rankdata(flat[nz], method="average")  # 1..n_nz
    out.ravel()[nz] = (r / n_nz).astype(np.float32)
    return out


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
