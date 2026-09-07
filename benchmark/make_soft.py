#!/usr/bin/env python3
"""make_soft.py — build a GDSD soft edge map from C++ feature files.

soft[y, x] = strength at zero-crossing pixels, 0 elsewhere.  The shape is
read from the actual image (data/images/<split>/<sid>.jpg), which avoids
the earlier hardcoded 481x321 transpose bug.

Usage:
    python make_soft.py <feat_dir> <out_dir> <split> [--strength response|gradient_magnitude]

    --strength response            GDSD v1: |R| at the zero crossing (AMM 2026 paper)
    --strength gradient_magnitude  GDSD v2: hypot(d, e) at the zero crossing (default)

Paths can be overridden with GDSD_BSDS_ROOT (directory containing
data/images/<split>/<sid>.jpg); the default is the current repo root.

Feature files expected next to <sid>.response.f64: <sid>.d.f64, <sid>.e.f64,
<sid>.zc.f64 (raw little-endian float64, h x w), produced by
gdsd_features_cli (see Makefile target build-benchmark).
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(os.environ.get("GDSD_BSDS_ROOT", Path(__file__).resolve().parent))
IMG_DIR = ROOT / "data" / "images"


def main():
    feat_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    split = sys.argv[3] if len(sys.argv) > 3 else "test"
    strength = "gradient_magnitude"
    if "--strength" in sys.argv:
        strength = sys.argv[sys.argv.index("--strength") + 1]
    if strength not in ("response", "gradient_magnitude"):
        raise SystemExit("--strength must be 'response' or 'gradient_magnitude'")
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(feat_dir.glob("*.response.f64"))
    n_ok = n_skip = 0
    for f in files:
        sid = f.name.replace(".response.f64", "")
        imgp = IMG_DIR / split / f"{sid}.jpg"
        if not imgp.exists():
            print(f"skip {sid}: no image")
            n_skip += 1
            continue
        img = cv2.imread(str(imgp), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"skip {sid}: cannot read image")
            n_skip += 1
            continue
        h, w = img.shape
        n = f.stat().st_size // 8
        if h * w != n:
            h, w = w, h  # try swapped orientation
        if h * w != n:
            print(f"skip {sid}: size mismatch {h}x{w} != {n}")
            n_skip += 1
            continue
        d = np.fromfile(feat_dir / f"{sid}.d.f64", dtype="<f8").reshape(h, w)
        e = np.fromfile(feat_dir / f"{sid}.e.f64", dtype="<f8").reshape(h, w)
        zc = np.fromfile(feat_dir / f"{sid}.zc.f64", dtype="<f8").reshape(h, w)
        gm = np.hypot(d, e)
        if strength == "response":
            resp = np.fromfile(f, dtype="<f8").reshape(h, w)
            soft = np.where(zc > 0.5, np.abs(resp), 0.0).astype(np.float32)
        else:
            soft = np.where(zc > 0.5, gm, 0.0).astype(np.float32)
        np.save(out_dir / f"{sid}.npy", soft)
        n_ok += 1
    print(f"done GDSD soft ({strength}): {n_ok} ok, {n_skip} skipped -> {out_dir}")


if __name__ == "__main__":
    main()
