#!/usr/bin/env python3
"""make_v1_v2_figs.py — build comparison figures for docs/GDSD_V2.md.

Uses the real soft maps that produced the benchmark numbers:
  gdsd_v1fix = |R|@ZC (v1), gdsd2 = gm@ZC (v2), both sigma=1.4.
Figures:
  1. softmap grid: image | v1 | v2 on two example images (portrait + landscape)
  2. doublet schematic: 1-D profile across a real edge with R, |R|, gm,
     zero crossing and GT position (mechanism of the v2 choice)
  3. near-GT vs far-GT strength boxplot for |R| and gm (why gm ranks better)

Run on the box (has data + soft maps). Output PNGs to docs/figures/.
"""
import os
import sys
import glob

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/sc7308/gdsd-bsds"
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "figures")
os.makedirs(FIG, exist_ok=True)


def load_soft(sid, split="val"):
    img = cv2.imread(f"{ROOT}/data/images/{split}/{sid}.jpg", cv2.IMREAD_GRAYSCALE)
    v1 = np.load(f"{ROOT}/ods_ois_ap/soft/gdsd_v1fix/{split}/{sid}.npy")
    v2 = np.load(f"{ROOT}/ods_ois_ap/soft/gdsd2/{split}/{sid}.npy")
    return img, v1, v2


def norm(s):
    s = s.astype(np.float32)
    m = s.max() if s.max() > 0 else 1.0
    return s / m


def fig_softmap():
    pairs = [("101085", "val"), ("103070", "val")]  # portrait + landscape
    fig, axes = plt.subplots(len(pairs), 3, figsize=(13.5, 4.2 * len(pairs)))
    for r, (sid, split) in enumerate(pairs):
        img, v1, v2 = load_soft(sid, split)
        axes[r, 0].imshow(img, cmap="gray")
        axes[r, 0].set_title(f"image {sid}", fontsize=11)
        axes[r, 1].imshow(norm(v1), cmap="hot")
        axes[r, 1].set_title("v1: soft = |R| at ZC", fontsize=11)
        axes[r, 2].imshow(norm(v2), cmap="hot")
        axes[r, 2].set_title("v2: soft = gradient magnitude at ZC", fontsize=11)
        for c in range(3):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    fig.suptitle("GDSD soft edge maps — same zero-crossing support, different strength",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    out = os.path.join(FIG, "v1_vs_v2_softmaps.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_doublet_profile():
    """Real 1-D scan across a strong edge: intensity, R, |R|, gm, ZC, GT."""
    sid = "101085"
    img, v1, v2 = load_soft(sid)
    # a strong edge row/col: pick the brightest v2 segment, scan a row through it
    ys, xs = np.where(v2 > np.percentile(v2[v2 > 0], 99.5))
    if len(ys) == 0:
        return
    y0 = int(np.median(ys[:2000]))
    # scan a horizontal line in the middle band
    d2 = np.fromfile(f"{ROOT}/ods_ois_ap/feats/val/{sid}.d.f64", dtype="<f8").reshape(img.shape)
    e2 = np.fromfile(f"{ROOT}/ods_ois_ap/feats/val/{sid}.e.f64", dtype="<f8").reshape(img.shape)
    resp = np.fromfile(f"{ROOT}/ods_ois_ap/feats/val/{sid}.response.f64", dtype="<f8").reshape(img.shape)
    zc = np.fromfile(f"{ROOT}/ods_ois_ap/feats/val/{sid}.zc.f64", dtype="<f8").reshape(img.shape) > 0.5
    gm = np.hypot(d2, e2)

    # restrict to a segment with large gm variation
    row = img[y0]
    g = gm[y0]
    strong = np.where(g > np.percentile(g, 90))[0]
    if len(strong) < 2:
        return
    xa, xb = strong.min() - 8, strong.max() + 8
    xa, xb = max(xa, 0), min(xb, img.shape[1])
    xx = np.arange(xa, xb)

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(xx, (row[xa:xb].astype(float) - row[xa:xb].mean()) / 255.0,
            color="0.4", lw=1.6, label="intensity (centred)")
    ax.plot(xx, resp[y0, xa:xb], color="tab:blue", lw=1.8, label="response R = gradᵀH·grad")
    ax.plot(xx, np.abs(resp[y0, xa:xb]), color="tab:cyan", lw=1.2, ls="--",
            label="|R| (v1 strength)")
    ax.plot(xx, gm[y0, xa:xb], color="tab:red", lw=1.8, label="gradient magnitude (v2 strength)")
    zxp = xx[zc[y0, xa:xb]]
    if len(zxp):
        ax.axvline(zxp[0], color="k", ls=":", lw=1.2)
        ax.text(zxp[0] + 0.3, ax.get_ylim()[1] * 0.92, "ZC", fontsize=9, color="k")
    ax.set_xlabel("column (pixel)")
    ax.set_title("One scan line across an edge — second-derivative doublet, |R| near 0 at the "
                 "crossing while gm peaks at the edge", fontsize=11)
    ax.legend(fontsize=9, loc="best", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, "doublet_profile.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_near_gt_box():
    """Boxplot of |R| and gm at ZC pixels near GT vs far GT (val, 10 images)."""
    import scipy.ndimage as ndi
    from scipy.io import loadmat
    near_a, far_a, near_g, far_g = [], [], [], []
    for sid in sorted(os.listdir(f"{ROOT}/data/images/val"))[:10]:
        sid = sid[:-4]
        img = cv2.imread(f"{ROOT}/data/images/val/{sid}.jpg", cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        d = np.fromfile(f"{ROOT}/ods_ois_ap/feats/val/{sid}.d.f64", dtype="<f8").reshape(h, w)
        e = np.fromfile(f"{ROOT}/ods_ois_ap/feats/val/{sid}.e.f64", dtype="<f8").reshape(h, w)
        r = np.fromfile(f"{ROOT}/ods_ois_ap/feats/val/{sid}.response.f64", dtype="<f8").reshape(h, w)
        zc = np.fromfile(f"{ROOT}/ods_ois_ap/feats/val/{sid}.zc.f64", dtype="<f8").reshape(h, w) > 0.5
        gm = np.hypot(d, e)
        gt = loadmat(f"{ROOT}/data/groundTruth/val/{sid}.mat")["groundTruth"][0]
        bnd = np.zeros((h, w), bool)
        for g in gt:
            lab = g[0][0][0].astype(np.int32)
            bp = np.zeros((h, w), bool)
            bp[:-1, :] |= np.diff(lab, axis=0) != 0
            bp[:, :-1] |= np.diff(lab, axis=1) != 0
            bnd |= bp
        dist = ndi.distance_transform_edt(~bnd)
        nr = zc & (dist <= 2.5)
        fr = zc & (dist > 2.5)
        near_a.append(np.abs(r)[nr]); far_a.append(np.abs(r)[fr])
        near_g.append(gm[nr]); far_g.append(gm[fr])
    na = np.concatenate(near_a); fa = np.concatenate(far_a)
    ng = np.concatenate(near_g); fg = np.concatenate(far_g)

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    data = [na, fa, ng, fg]
    bp = ax.boxplot(data, showfliers=False, patch_artist=True,
                    medianprops=dict(color="black"))
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(["|R| near GT", "|R| far GT", "gm near GT", "gm far GT"])
    colors = ["#4C72B0", "#8FB0D9", "#C44E52", "#E8A2A6"]
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.8)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_ylabel("strength (log scale)")
    ax.set_title("ZC pixels near GT carry higher strength; separation is cleaner for gm",
                 fontsize=11)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = os.path.join(FIG, "near_gt_strength_boxplot.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_softmap()
    fig_doublet_profile()
    fig_near_gt_box()
    print("ALL_FIGS_DONE")
