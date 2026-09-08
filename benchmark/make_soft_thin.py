#!/usr/bin/env python3
"""make_soft_thin.py — GDSD soft-map post-processing experiments (σ fixed = input feats).

Modes (all preserve the GDSD ZC identity; only *which* ZC pixels survive and
how strength is assigned changes):

  plain          v2 reference: gm at every ZC pixel (baseline, reproduces v2)
  zcnms          keep ZC pixel only when it is a local maximum of gm along the
                 gradient-normal direction *among ZC pixels* (removes the second
                 row of a doublet / parallel crossings; minimal intervention)
  gmnms          Canny-style NMS: keep ZC pixel when gm >= gm of its two
                 gradient-normal neighbours (whatever those neighbours are);
                 pulls the surviving edge onto the gm ridge
  gmnms_relax    gmnms but a ZC pixel also survives if it is the *only* ZC in
                 its 3x3 normal window (avoids holes on weak isolated crossings)

Rationale (doublet theory, see docs/GDSD_V2.md): R is a second derivative, so
a step edge produces a +/- doublet ~2.4 px apart; the human boundary lies
between the lobes. The ZC of R therefore fires on both flanks in places, giving
thick (2 px) responses that cap precision. Thinning along the gradient normal
keeps the strongest flank / the gm ridge pixel, targeting precision without
touching sigma or the ZC rule.

Usage:
  python make_soft_thin.py <feat_dir> <out_dir> <split> <mode> [strength=gm]
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(os.environ.get("GDSD_BSDS_ROOT", Path(__file__).resolve().parent))
IMG_DIR = ROOT / "data" / "images"

MODES = ("plain", "zcnms", "gmnms", "gmnms_relax")


def normal_neighbours(d, e):
    """Indices of the two gradient-normal neighbours (rounded unit direction)."""
    gm = np.hypot(d, e)
    with np.errstate(divide="ignore", invalid="ignore"):
        sx = np.rint(d / np.where(gm > 1e-12, gm, 1.0)).astype(int)
        sy = np.rint(e / np.where(gm > 1e-12, gm, 1.0)).astype(int)
    zero = (sx == 0) & (sy == 0)
    sx[zero] = 0
    sy[zero] = 0
    return sx, sy


def thin_zc(gm, zc, sx, sy, mode):
    h, w = gm.shape
    if mode == "plain":
        return zc.copy()
    yy = np.arange(h)[:, None]
    xx = np.arange(w)[None, :]
    n1y = np.clip(yy + sy, 0, h - 1)
    n1x = np.clip(xx + sx, 0, w - 1)
    n2y = np.clip(yy - sy, 0, h - 1)
    n2x = np.clip(xx - sx, 0, w - 1)
    if mode == "zcnms":
        # suppress a ZC if a ZC neighbour along the normal has strictly higher gm
        zc_n1 = zc[n1y, n1x]
        zc_n2 = zc[n2y, n2x]
        keep = zc.copy()
        keep &= ~(zc_n1 & (gm[n1y, n1x] > gm))
        keep &= ~(zc_n2 & (gm[n2y, n2x] > gm))
        return keep
    if mode == "gmnms":
        keep = zc.copy()
        keep &= gm >= gm[n1y, n1x]
        keep &= gm >= gm[n2y, n2x]
        # pixels with no valid direction are dropped unless truly isolated
        return keep
    if mode == "gmnms_relax":
        keep = zc.copy()
        stronger_n1 = gm[n1y, n1x] > gm
        stronger_n2 = gm[n2y, n2x] > gm
        # keep if not strictly weaker than both neighbours
        keep &= ~(stronger_n1 & stronger_n2)
        # neighbours with gm<=1e-9 do not count (direction padding)
        pad = np.zeros_like(gm)
        pad[gm <= 1e-9] = True
        return keep
    raise ValueError(mode)


def soft_from_feats(sid, feat_dir, split, mode, strength="gm"):
    imgp = IMG_DIR / split / f"{sid}.jpg"
    img = cv2.imread(str(imgp), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(imgp)
    h, w = img.shape
    resp_f = feat_dir / f"{sid}.response.f64"
    n = resp_f.stat().st_size // 8
    if h * w != n:
        h, w = w, h
    if h * w != n:
        raise ValueError(f"size mismatch {h}x{w} != {n} for {sid}")
    resp = np.fromfile(resp_f, dtype="<f8").reshape(h, w)
    d = np.fromfile(feat_dir / f"{sid}.d.f64", dtype="<f8").reshape(h, w)
    e = np.fromfile(feat_dir / f"{sid}.e.f64", dtype="<f8").reshape(h, w)
    zc = np.fromfile(feat_dir / f"{sid}.zc.f64", dtype="<f8").reshape(h, w) > 0.5

    gm = np.hypot(d, e)
    sx, sy = normal_neighbours(d, e)
    keep = thin_zc(gm, zc, sx, sy, mode)
    if strength == "gm":
        soft = np.where(keep, gm, 0.0).astype(np.float32)
    elif strength == "resp":
        soft = np.where(keep, np.abs(resp), 0.0).astype(np.float32)
    else:
        raise ValueError(strength)
    return soft


def main():
    feat_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    split = sys.argv[3]
    mode = sys.argv[4]
    strength = sys.argv[5] if len(sys.argv) > 5 else "gm"
    if mode not in MODES:
        raise SystemExit(f"--mode must be one of {MODES}")
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for f in sorted(feat_dir.glob("*.response.f64")):
        sid = f.name.replace(".response.f64", "")
        try:
            soft = soft_from_feats(sid, feat_dir, split, mode, strength)
        except (FileNotFoundError, ValueError) as exc:
            print(f"skip {sid}: {exc}")
            continue
        np.save(out_dir / f"{sid}.npy", soft)
        n_ok += 1
    print(f"done thin mode={mode}: {n_ok} ok -> {out_dir}")


if __name__ == "__main__":
    main()
