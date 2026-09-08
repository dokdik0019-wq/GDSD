#!/usr/bin/env python3
"""make_soft_hyst.py — GDSD v4 soft map: gm@ZC + Canny-style hysteresis.

v4 = v2 (gm@ZC) plus hysteresis on the zero-crossing support.  The strong
set is zero crossings whose gradient-side neighbour response exceeds Tp;
the weak set is zero crossings that satisfy the percentile-contrast test.
Weak pixels are kept only when they touch a strong pixel through an
8-connected component (Canny 1986 hysteresis).  Strength at surviving
pixels is the gradient magnitude, as in v2.

This addresses the known v2 weakness: GDSD edges shatter into isolated
pixels (median component size 1 px).  Hysteresis links weak-but-consistent
crossings to strong contours without the flood of the raw ZC set.

Input: same .f64 feature files as make_soft.py (from gdsd_features_cli):
  <feat_dir>/<sid>.response.f64/.d.f64/.e.f64/.zc.f64  (h x w, float64)
  image orientation read from data/images/<split>/<sid>.jpg
Output: <out_dir>/<sid>.npy  (float32 soft map)

Usage:
  python make_soft_hyst.py <feat_dir> <out_dir> <split> [percentile=90]
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import label

ROOT = Path(os.environ.get("GDSD_BSDS_ROOT", Path(__file__).resolve().parent))
IMG_DIR = ROOT / "data" / "images"


def hysteresis(edge_strong, edge_weak):
    """Canny-style: keep weak pixels 8-connected to a strong pixel."""
    mask = edge_strong | edge_weak
    lab, n = label(mask, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return edge_strong.copy()
    strong_ids = set(np.unique(lab[edge_strong]).tolist()) - {0}
    if not strong_ids:
        return edge_strong.copy()
    return np.isin(lab, list(strong_ids))


def soft_hyst_from_feats(sid, feat_dir, split, percentile=90.0):
    """Build v4 soft map for one image from C++ feature files."""
    imgp = IMG_DIR / split / f"{sid}.jpg"
    img = cv2.imread(str(imgp), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(imgp)
    h, w = img.shape

    resp_f = feat_dir / f"{sid}.response.f64"
    n = resp_f.stat().st_size // 8
    if h * w != n:
        h, w = w, h  # try swapped orientation
    if h * w != n:
        raise ValueError(f"size mismatch {h}x{w} != {n} for {sid}")

    resp = np.fromfile(resp_f, dtype="<f8").reshape(h, w)
    d = np.fromfile(feat_dir / f"{sid}.d.f64", dtype="<f8").reshape(h, w)
    e = np.fromfile(feat_dir / f"{sid}.e.f64", dtype="<f8").reshape(h, w)
    zc = np.fromfile(feat_dir / f"{sid}.zc.f64", dtype="<f8").reshape(h, w) > 0.5

    gm = np.hypot(d, e)
    valid = gm > 1e-2
    vals = resp[valid]
    if vals.size == 0:
        return np.zeros((h, w), dtype=np.float32)
    tp = float(np.percentile(vals, percentile))
    tlow = float(np.percentile(vals, 100.0 - percentile))

    # gradient-direction neighbours (same rule as detector/feature CLI)
    nb1 = np.zeros_like(resp)
    nb2 = np.zeros_like(resp)
    mag = np.hypot(d, e)
    sy = np.rint(e / np.where(mag > 1e-12, mag, 1.0)).astype(int)
    sx = np.rint(d / np.where(mag > 1e-12, mag, 1.0)).astype(int)
    zero = (sy == 0) & (sx == 0)
    y1 = np.clip(np.arange(h)[:, None] + sy, 0, h - 1)
    x1 = np.clip(np.arange(w)[None, :] + sx, 0, w - 1)
    y2 = np.clip(np.arange(h)[:, None] - sy, 0, h - 1)
    x2 = np.clip(np.arange(w)[None, :] - sx, 0, w - 1)
    nb1 = resp[y1, x1]
    nb2 = resp[y2, x2]
    nb1[zero] = 0.0
    nb2[zero] = 0.0

    zc_valid = valid & zc
    contrast = zc_valid & (((nb1 > tp) & (nb2 < tlow)) | ((nb2 > tp) & (nb1 < tlow)))
    strong = zc_valid & ((nb1 > tp) | (nb2 > tp))
    # weak = ALL contrast crossings (spike definition); hysteresis keeps those
    # in a component that also contains a strong pixel.
    weak = contrast

    keep = hysteresis(strong, weak)
    soft = np.where(keep, gm, 0.0).astype(np.float32)
    return soft


def main():
    feat_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    split = sys.argv[3] if len(sys.argv) > 3 else "test"
    percentile = float(sys.argv[4]) if len(sys.argv) > 4 else 90.0
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(feat_dir.glob("*.response.f64"))
    n_ok = n_skip = 0
    for f in files:
        sid = f.name.replace(".response.f64", "")
        try:
            soft = soft_hyst_from_feats(sid, feat_dir, split, percentile)
        except (FileNotFoundError, ValueError) as exc:
            print(f"skip {sid}: {exc}")
            n_skip += 1
            continue
        np.save(out_dir / f"{sid}.npy", soft)
        n_ok += 1
    print(f"done GDSD v4 hyst soft: {n_ok} ok, {n_skip} skipped -> {out_dir}")


if __name__ == "__main__":
    main()
