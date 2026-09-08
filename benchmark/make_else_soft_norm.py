#!/usr/bin/env python3
"""make_else_soft_norm.py — ELSE soft maps -> uint8 PNG for official BSDS eval.

Variants (normalized per-image by construction):
  else_r_norm    : R = sqrt(D^2+E^2),   R / R.max() * 255     (single-threshold ranking)
  else_nms_norm  : NMS(R), NMS(R).max() normalized            (double-threshold ranking)

Why per-image max normalization (NOT the raw ×255 clip used for GDSD/Canny):
the paper's quadratic fit uses adjacent-pixel distance 0.1, which scales the
first-order coefficients ~10x, so R routinely exceeds 1 on [0,1] images
(R.max ~ 4-5).  A plain clip(R*255) saturates a large fraction of strong
pixels at 255 and flattens the ranking the official linspace grid sees
(smoke: clip ODS 0.4949 vs norm ODS 0.5082 on val/20thr, AP 0.10 vs 0.48).
Per-image /max restores a full [0,1] span per image, which is what the
uint8-PNG convention assumes (255 = per-image strongest response).

Usage:
    python make_else_soft_norm.py [val|test] [else_r_norm|else_nms_norm|all]

Requires the else/ package on sys.path (quadratic_coeff_maps, nonmax_suppression).
"""
import os
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "else"))
from else_detector import nonmax_suppression          # noqa: E402
from quadratic_fit import quadratic_coeff_maps          # noqa: E402

ROOT = Path(os.environ.get("GDSD_BSDS_ROOT",
            "/Users/dookdik/Desktop/mywork /dookdik13/test3/BIDS-BSDS500-a04b7c6"))
OUT = Path(os.environ.get("GDSD_ELSE_SOFT_OUT",
            "/Users/dookdik/Desktop/GDSD-Paper/else-benchmark/soft"))
VARIANTS = {"else_r_norm": "r", "else_nms_norm": "n"}
SPLITS = ["val", "test"]


def build(split: str, variant: str) -> None:
    names = sorted(p.stem for p in (ROOT / "BSDS500" / "data" / "images" / split).glob("*.jpg"))
    outdir = OUT / variant / split
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = 0
    for name in names:
        png = outdir / f"{name}.png"
        if png.exists():
            continue
        img = np.asarray(
            Image.open(ROOT / "BSDS500" / "data" / "images" / split / f"{name}.jpg").convert("L"),
            dtype=np.float32,
        ) / 255.0
        A, B, C, D, E, F = quadratic_coeff_maps(img, radius=1, step=0.1)
        R = np.sqrt(D * D + E * E)
        if VARIANTS[variant] == "n":
            theta = np.arctan2(E, D)
            R = nonmax_suppression(R, theta)
        m = R.max()
        u8 = np.clip(R / (m if m > 0 else 1.0) * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(u8, mode="L").save(png)
        n += 1
    print(f"{split} {variant}: wrote {n} in {time.time()-t0:.0f}s -> {outdir}", flush=True)


def main():
    split_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    var_arg = sys.argv[2] if len(sys.argv) > 2 else "all"
    splits = SPLITS if split_arg == "all" else [split_arg]
    variants = list(VARIANTS) if var_arg == "all" else [var_arg]
    for split in splits:
        for variant in variants:
            build(split, variant)


if __name__ == "__main__":
    main()
