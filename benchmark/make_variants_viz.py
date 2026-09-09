#!/usr/bin/env python3
"""viz_variants.py — collect per-threshold PR + best-threshold edge overlays
for every GDSD variant (val split), using the same official py-bsds500
matching as run_official_pr.py.  Outputs:
  viz_out/pr_curves.png        precision-recall curves per variant
  viz_out/edge_overlay_<sid>.png   GT vs variant edges at each variant's ODS thr
  viz_out/variant_summary.csv  ODS/OIS/AP per variant (cross-check w/ docs)
"""
from __future__ import annotations
import csv
import os
import sys
from pathlib import Path
import numpy as np
import cv2
from multiprocessing import get_context

_BSDS_ROOT = os.environ.get(
    "GDSD_BSDS_ROOT", "/Users/dookdik/gdsd-bsds-mac"
)
_PYBSDS = os.environ.get("GDSD_PYBSDS_PATH", "/tmp/py-bsds500")
sys.path.insert(0, _PYBSDS)
from bsds.bsds_dataset import BSDSDataset
from bsds import evaluate_boundaries

VARIANTS = {
    "v1 |R|@ZC s1.4": "viz_png/gdsd_v1_s14_val",
    "v2 gm@ZC s1.4": "viz_png/gdsd_v2_s14_val",
    "v2 gm@ZC s2.8": "viz_png/gdsd_v2_s28_val",
    "v3 WLS s1.4": "viz_png/wls_val",
    "v4 hyst s1.4": "png_v4_s14/val",
    "v4 hyst s2.8": "png_v4_s28/val",
    "v2+zcnms s2.8": "thin_val_results/png_s28_zcnms/val",
}
ROOT = Path("/Users/dookdik/gdsd-bsds-mac")
N_THR = 99
THR = np.linspace(1.0 / (N_THR + 1), 1.0 - 1.0 / (N_THR + 1), N_THR)


def load_pred(png_dir, sid):
    p = cv2.imread(str(Path(png_dir) / f"{sid}.png"), cv2.IMREAD_UNCHANGED)
    if p is None:
        raise FileNotFoundError(f"{png_dir}/{sid}.png")
    pred = (p.astype(np.float32) / 255.0) if p.dtype != np.uint16 else (p.astype(np.float32) / 65535.0)
    return pred


# ---- parallel worker state (fork) ----
_DS = None
_PNG_DIR = None


def _accum_one(nm):
    sid = Path(nm).name
    pred = load_pred(_PNG_DIR, sid)
    gt = _DS.boundaries(nm)
    tgt = gt[0].shape
    if pred.shape != tgt:
        pred = pred[: tgt[0], : tgt[1]]
        pred = np.pad(pred, [(0, tgt[0] - pred.shape[0]), (0, tgt[1] - pred.shape[1])], mode="constant")
    cr, sr, cp, sp, _ = evaluate_boundaries.evaluate_boundaries(
        pred, gt, thresholds=THR, apply_thinning=True)
    rec, prec, f1 = evaluate_boundaries.compute_rec_prec_f1(cr, sr, cp, sp)
    return cr, sr, cp, sp, float(f1.max())


def main():
    global _DS, _PNG_DIR
    ds = BSDSDataset(_BSDS_ROOT)
    _DS = ds
    names = list(ds.val_sample_names)   # e.g. "val/65033"
    os.makedirs("viz_out", exist_ok=True)
    os.makedirs("viz_out/png", exist_ok=True)

    summary = []
    curves = {}
    for label, png_dir in VARIANTS.items():
        png_dir_p = ROOT / png_dir
        n_ok = sum(1 for nm in names if (png_dir_p / f"{Path(nm).name}.png").exists())
        if n_ok < 100:
            print(f"[skip] {label}: only {n_ok}/100 pngs at {png_dir_p}")
            continue
        _PNG_DIR = png_dir_p
        with get_context("fork").Pool(processes=10) as pool:
            parts = pool.map(_accum_one, names)
        cr = sum(p[0] for p in parts)
        sr = sum(p[1] for p in parts)
        cp = sum(p[2] for p in parts)
        sp = sum(p[3] for p in parts)
        per_img_best = np.array([p[4] for p in parts])
        rec, prec, f1 = evaluate_boundaries.compute_rec_prec_f1(cr, sr, cp, sp)
        bi = int(np.argmax(f1))
        ods, ois = float(f1[bi]), float(per_img_best.mean())
        rec_u, ndx = np.unique(rec, return_index=True)
        prec_u = prec[ndx]
        prec_i = np.interp(np.arange(0, 1, 0.01), rec_u, prec_u, left=0.0, right=0.0)
        ap = float(prec_i.sum() * 0.01)
        summary.append((label, ods, ois, ap, float(THR[bi]), float(rec[bi]), float(prec[bi])))
        curves[label] = (rec, prec)
        print(f"{label:18s} ODS={ods:.4f} OIS={ois:.4f} AP={ap:.4f} @thr {THR[bi]:.3f}", flush=True)

    with open("viz_out/variant_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "ODS", "OIS", "AP", "best_thr", "R", "P"])
        w.writerows(summary)

    # ---- PR curves ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 6))
    for label, (rec, prec) in curves.items():
        ax.plot(rec, prec, lw=1.8, label=f"{label}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("GDSD variants — BSDS500 val, official pr_eval (99 thr)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig("viz_out/pr_curves.png", dpi=160)
    print("wrote viz_out/pr_curves.png")

    # ---- edge overlays on 2 images ----
    import matplotlib.pyplot as plt
    demo_sids = ["101085", "101087"]
    for sid in demo_sids:
        imgp = _BSDS_ROOT + "/BSDS500/data/images/val/" + sid + ".jpg"
        img = cv2.imread(imgp, cv2.IMREAD_GRAYSCALE)
        gt = ds.boundaries("val/" + sid)
        gt_sum = np.zeros(gt[0].shape, np.float32)
        for g in gt:
            gt_sum += g
        n = len(VARIANTS) + 1
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.1 * rows))
        axes = np.atleast_1d(axes).ravel()
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title("image", fontsize=8)
        axes[0].axis("off")
        for ax, (label, _) in zip(axes[1:], VARIANTS.items()):
            # find this variant's ODS threshold from summary
            row = next(r for r in summary if r[0] == label)
            thr = row[4]
            pred = load_pred(ROOT / VARIANTS[label], sid)
            mask = pred > thr
            ov = np.stack([img] * 3, axis=-1).copy()
            ov[mask] = [255, 80, 80]      # red = detector edge
            alpha = 0.55
            green = np.array([80, 220, 80])
            idx = gt_sum > 0
            ov[idx] = (alpha * green + (1 - alpha) * ov[idx]).astype(np.uint8)
            ax.imshow(ov)
            ax.set_title(f"{label}  ODS {row[1]:.4f}", fontsize=8)
            ax.axis("off")
        for ax in axes[n:]:
            ax.axis("off")
        fig.suptitle(f"val image {sid} — red: detector @ODS thr, green: human GT", fontsize=9)
        fig.tight_layout()
        fig.savefig(f"viz_out/edge_overlay_{sid}.png", dpi=150)
        plt.close(fig)
        print(f"wrote viz_out/edge_overlay_{sid}.png")
    print("DONE")


if __name__ == "__main__":
    main()
