#!/usr/bin/env python3
"""haralick_facet.py — Haralick (1984) cubic facet edge detector + soft map generator

Reference: Haralick, "Digital Step Edges from Zero Crossing of Second Directional
Derivatives", IEEE TPAMI 6(1):58-68, 1984.
https://www.haralick.org/journals/04767475.pdf

Cubic facet fit on the 5x5 window (discrete orthogonal polynomial basis):
  f(x,y) = a0 + a1 x + a2 y + a3 x^2 + a4 xy + a5 y^2 + a6 x^3 + a7 x^2y + a8 xy^2 + a9 y^3
Each coefficient = a linear combination of the 5x5 patch -> convolution with a mask (fast)

Second directional derivative along the unit gradient (u,v) = (fx,fy)/|g| at the center:
  f''(0) = 2(a3 u^2 + a4 uv + a5 v^2)
  f'''(0) = 6(a6 u^3 + a7 u^2v + a8 uv^2 + a9 v^3)   (d/drho of f'')
Step edge when f'' has a negatively sloped zero crossing inside the pixel:
  f''(0) > 0, f'''(0) < 0, and |rho0| = |−f''/f'''| <= 1  (root inside the pixel)

Soft map (for the PR curve): strength = |g| = hypot(fx,fy) at edge pixels (like GDSD soft2)
Usage: python haralick_facet.py <img_dir> <out_dir>
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import cv2

def make_facet_masks(r=2):
    """Return 10 masks (2r+1)x(2r+1) — least-squares coefficient kernels.
    mask[k] matches np.pad(gray, r, 'edge') convolution exactly.
    Basis at offset (x,y) in {-r..r}^2: [1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3]"""
    coords = np.arange(-r, r + 1, dtype=np.float64)
    xs, ys = np.meshgrid(coords, coords)
    x = xs.ravel(); y = ys.ravel()
    Phi = np.stack([np.ones_like(x), x, y, x*x, x*y, y*y,
                    x**3, x*x*y, x*y*y, y**3], axis=1)  # (N,10)
    pinv = np.linalg.pinv(Phi)  # (10,N)
    masks = [pinv[k].reshape(2*r+1, 2*r+1) for k in range(10)]
    return masks

def haralick_soft(gray, masks, r=2, grad_thr=1e-3, rho_max=1.0, sigma=0.0,
                  direction="continuous", thinning="none"):
    """Vectorized Haralick facet edge → soft map (strength = gm at edge px).

    rho_max: allow the sub-pixel root |rho0| <= rho_max (Haralick 1984 uses
        rho_max=1, i.e. the root must lie inside the centre pixel; larger
        values admit roots in neighbouring pixels and change the candidate
        set — this is the sensitivity parameter noted in the v2 review).
    sigma: optional Gaussian blur (ksize auto-derived) before the facet fit.
    direction:
        "continuous" — Haralick 1984: second directional derivative along
            the *continuous* unit gradient (u,v) = (gx,gy)/|g|.
        "octant" — quantize the direction to one of the 8 octants first
            (step = round of the unit gradient, like the GDSD detector),
            then take f''/f''' along that quantized direction.  This is the
            ``oct_zero`` configuration (Haralick rule + GDSD octant
            direction) from the internal 2x2 ablation.
    thinning:
        "none" — keep every Haralick edge pixel (the raw 1984 rule).
        "zc_nms" — suppress an edge pixel when an *edge* neighbour along
            the gradient normal has strictly higher gm.  This is the same
            candidate-only NMS as GDSD's ``zcnms`` thinning, applied to the
            Haralick analytic-rule edge set — used to check whether that
            thinning is a generic gain or specific to GDSD's doublet.
    """
    if sigma > 0.0:
        gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma,
                                borderType=cv2.BORDER_REFLECT)
    h, w = gray.shape
    pad = cv2.copyMakeBorder(gray, r, r, r, r, cv2.BORDER_REPLICATE)
    # coefficient maps via filter2D (correlation = conv with kernel rotated; masks are symmetric)
    def cmap(k):
        return cv2.filter2D(pad, cv2.CV_64F, masks[k])[r:r+h, r:r+w]
    # precompute all 10
    c = [cmap(k) for k in range(10)]
    a0, a1, a2 = c[0], c[1], c[2]
    a3, a4, a5 = c[3], c[4], c[5]
    a6, a7, a8, a9 = c[6], c[7], c[8], c[9]

    gx = a1  # df/dx at center (a1 = coeff of x)
    gy = a2
    gm = np.hypot(gx, gy)
    valid = gm > grad_thr

    u = np.zeros_like(gm); v = np.zeros_like(gm)
    if direction == "continuous":
        u[valid] = gx[valid] / gm[valid]
        v[valid] = gy[valid] / gm[valid]
    elif direction == "octant":
        # quantize to nearest of 8 directions (round of the unit gradient)
        with np.errstate(divide="ignore", invalid="ignore"):
            sx = np.rint(gx / np.where(gm > 1e-12, gm, 1.0)).astype(int)
            sy = np.rint(gy / np.where(gm > 1e-12, gm, 1.0)).astype(int)
        ok = valid & ~((sx == 0) & (sy == 0))
        nrm = np.hypot(sx, sy)
        u[ok] = sx[ok] / nrm[ok]
        v[ok] = sy[ok] / nrm[ok]
    else:
        raise ValueError("direction must be 'continuous' or 'octant'")

    fpp = 2.0 * (a3 * u*u + a4 * u*v + a5 * v*v)
    fppp = 6.0 * (a6 * u**3 + a7 * u*u*v + a8 * u*v*v + a9 * v**3)

    # rho0 of f'' -> zero (sub-pixel, along (u,v)); root lies in the pixel when |rho0| <= rho_max
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.abs(fppp) > 1e-12
        rho0 = np.full_like(fpp, np.nan)
        rho0[denom] = -fpp[denom] / fppp[denom]

    edge = valid & (fpp > 0) & (fppp < 0) & (np.abs(rho0) <= rho_max)
    if thinning == "zc_nms":
        # candidate-only NMS along the gradient normal (same rule as GDSD
        # zcnms): suppress an edge pixel if an *edge* neighbour in the
        # +-(u,v) direction has strictly higher gm.
        h2, w2 = edge.shape
        # nearest of 8 directions from the unit normal (u,v)
        with np.errstate(divide="ignore", invalid="ignore"):
            mag = np.hypot(u, v)
            sx = np.rint(u / np.where(mag > 1e-12, mag, 1.0)).astype(int)
            sy = np.rint(v / np.where(mag > 1e-12, mag, 1.0)).astype(int)
        yy = np.arange(h2)[:, None]
        xx = np.arange(w2)[None, :]
        n1y = np.clip(yy + sy, 0, h2 - 1)
        n1x = np.clip(xx + sx, 0, w2 - 1)
        n2y = np.clip(yy - sy, 0, h2 - 1)
        n2x = np.clip(xx - sx, 0, w2 - 1)
        e1 = edge[n1y, n1x]
        e2 = edge[n2y, n2x]
        keep = edge.copy()
        keep &= ~(e1 & (gm[n1y, n1x] > gm))
        keep &= ~(e2 & (gm[n2y, n2x] > gm))
        edge = keep
    # Haralick: edge strength = |gradient| (soft map like GDSD2)
    strength = np.where(edge, gm, 0.0).astype(np.float32)
    return strength, {"edge_px": int(edge.sum()), "fpp_pos": int((fpp > 0).sum()),
                      "valid": int(valid.sum()), "gm": gm, "rho0": rho0, "edge": edge}

def main():
    img_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    rho_max = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    sigma = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    direction = sys.argv[5] if len(sys.argv) > 5 else "continuous"
    thinning = sys.argv[6] if len(sys.argv) > 6 else "none"
    out_dir.mkdir(parents=True, exist_ok=True)
    masks = make_facet_masks(2)
    imgs = sorted(img_dir.glob("*.jpg"))
    for f in imgs:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("skip", f); continue
        gray = img.astype(np.float64) / 255.0
        soft, info = haralick_soft(gray, masks, r=2, rho_max=rho_max,
                                   sigma=sigma, direction=direction,
                                   thinning=thinning)
        np.save(out_dir / f"{f.stem}.npy", soft)
    print(f"done Haralick facet (rho_max={rho_max}, sigma={sigma}, "
          f"dir={direction}, thin={thinning}): {len(imgs)} images -> {out_dir}")

if __name__ == "__main__":
    main()
