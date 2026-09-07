#!/usr/bin/env python3
"""make_soft_pngs_u8.py — soft maps -> uint8 PNG (0-255) for official BSDS eval.

The official MATLAB suite loads boundary maps as PNG and divides by 255
(`pb = double(imread(inFile))/255`), then thresholds on a linspace grid
over [0,1].  The natural encoding for a soft map fed to that pipeline is
therefore a uint8 PNG whose values fill the 0-255 range (255 * raw value,
clipped).  Each image then occupies its full [0,1] scale after /255,
which is the convention used for classical edge detectors in the
literature (Canny & co. write 8-bit soft maps).

This is deliberately NOT a global-max normalization: per-image /255 via
the uint8 clip is what the official `imread/255` convention implies, and
it keeps every image's soft map comparable on the fixed threshold grid.

Usage:
    python make_soft_pngs_u8.py <npy_dir> <png_dir>
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import cv2


def main():
    npy_dir = Path(sys.argv[1])
    png_dir = Path(sys.argv[2])
    png_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(npy_dir.glob("*.npy"))
    if not files:
        print("no .npy")
        return
    # if source already looks like 0-255 (e.g. canny), use as-is; else scale 255
    probe = np.load(files[0]).astype(np.float32)
    already_u8_scale = float(probe.max()) > 200.0
    n_ok = 0
    for f in files:
        s = np.load(f).astype(np.float32)
        if already_u8_scale:
            img = np.clip(s, 0, 255).astype(np.uint8)
        else:
            img = np.clip(s * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(png_dir / f"{f.stem}.png"), img)
        n_ok += 1
    print(f"wrote {n_ok} uint8 PNGs (source scale {'u8' if already_u8_scale else 'raw*255'}) -> {png_dir}")


if __name__ == "__main__":
    main()
