#!/usr/bin/env python3
"""
make_softmaps.py — build classical-baseline soft edge-strength maps
(sobel / canny / LoG) at RAW scale (value * 255, clipped to 0..255).

IMPORTANT: never normalize per image — that would destroy the shared
global threshold grid the official protocol relies on.  Output is raw
float32 (.npy) after clipping; convert to uint8 PNG for the official
pr_eval with `make_soft_pngs_u8.py` (imread/255 convention).

Methods:
  sobel : gradient magnitude (Sobel 3x3) + NMS along the gradient
  canny : gradient magnitude on a Gaussian blur + Canny-style NMS
          (soft = strength BEFORE hysteresis)
  log   : |Laplacian of Gaussian| at zero crossings only
(gdsd soft maps come from gdsd_features_cli.cpp features — see make_soft.py)

Data location: images at <GDSD_BSDS_ROOT>/data/images/<split>/*.jpg
(the box layout ~/gdsd-bsds uses that root; on a machine with the BSDS
symlink layout point GDSD_BSDS_ROOT at the directory containing data/).

Usage: python make_softmaps.py <method> <split> <out_dir> [--sigma 1.4]
"""
import os
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(os.environ.get("GDSD_BSDS_ROOT",
                           str(Path(__file__).resolve().parent.parent)))
IMG_DIR = ROOT / "data" / "images"

def nms_dir(gmag, gx, gy):
    """Canny-style NMS along the quantized gradient direction (8 sectors)."""
    h, w = gmag.shape
    out = np.zeros_like(gmag)
    ang = np.degrees(np.arctan2(gy, gx)) % 180
    gp = np.pad(gmag, 1, mode="edge")
    # padded indexing: (y+1, x+1) = center
    for y in range(h):
        for x in range(w):
            a = ang[y, x]
            c = gmag[y, x]
            if (a < 22.5) or (a >= 157.5):
                # horizontal edge -> compare left-right
                d1, d2 = gp[y+1, x], gp[y+1, x+2]
            elif a < 67.5:
                # diagonal ↘ -> NE-SW
                d1, d2 = gp[y, x+2], gp[y+2, x]
            elif a < 112.5:
                # vertical edge -> compare up-down
                d1, d2 = gp[y, x+1], gp[y+2, x+1]
            else:
                # diagonal ↗ -> NW-SE
                d1, d2 = gp[y, x], gp[y+2, x+2]
            if c >= d1 and c >= d2:
                out[y, x] = c
    return out

def soft_sobel(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    return nms_dir(mag, gx, gy)

def soft_canny(gray, sigma):
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma) if sigma > 0 else gray
    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    return nms_dir(mag, gx, gy)

def soft_log(gray, sigma):
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma) if sigma > 0 else gray
    lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=1)
    lp = np.pad(lap, 1, mode="edge")
    h, w = lap.shape
    zc = np.zeros_like(lap)
    for y in range(h):
        for x in range(w):
            c = lap[y, x]
            if c == 0:
                continue
            sg = np.sign(c)
            nb = (lp[y, x+1], lp[y, x+2], lp[y+1, x], lp[y+2, x])
            if any(n != 0 and np.sign(n) != sg for n in nb):
                zc[y, x] = abs(c)
    return zc

def main():
    method = sys.argv[1]
    split = sys.argv[2]
    out_dir = Path(sys.argv[3])
    sigma = 1.4
    if "--sigma" in sys.argv:
        sigma = float(sys.argv[sys.argv.index("--sigma")+1])
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted((IMG_DIR / split).glob("*.jpg"))
    for i, f in enumerate(files):
        gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
        if method == "sobel":
            soft = soft_sobel(gray)
        elif method == "canny":
            soft = soft_canny(gray, sigma)
        elif method == "log":
            soft = soft_log(gray, sigma)
        else:
            raise SystemExit(f"unknown method {method}")
        # RAW scale: *255 then clip — NO per-image normalization
        soft8 = np.clip(soft * 255.0, 0, 255).astype(np.float32)
        np.save(out_dir / f"{f.stem}.npy", soft8)
        if (i + 1) % 50 == 0:
            print(f"  {method}: {i+1}/{len(files)}", flush=True)
    print(f"done {method} {split}: {len(files)} images -> {out_dir}")

if __name__ == "__main__":
    main()
