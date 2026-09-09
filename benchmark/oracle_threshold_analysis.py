#!/usr/bin/env python3
"""oracle_threshold_analysis.py — per-image optimal threshold (oracle, GT-based)
for a soft-map PNG dir on val and/or test.  Outputs CSV:
  sid, optimal_thr, F1_at_optimal, plus soft-map stats (features usable by a
  GT-blind committee): n_zc, mean_gm_zc, med_gm_zc, p90_gm_zc, edge_density.

Usage:
  oracle_threshold_analysis.py <png_dir> <split> <out_csv>
"""
from __future__ import annotations
import csv
import os
import sys
from pathlib import Path
import numpy as np
import cv2
from multiprocessing import Pool, get_context

_BSDS_ROOT = os.environ.get("GDSD_BSDS_ROOT", str(Path.home() / "gdsd-bsds-mac"))
_PYBSDS = os.environ.get("GDSD_PYBSDS_PATH", "/tmp/py-bsds500")
sys.path.insert(0, _PYBSDS)
from bsds.bsds_dataset import BSDSDataset
from bsds import evaluate_boundaries

N_THR = 99
THR = np.linspace(1.0 / (N_THR + 1), 1.0 - 1.0 / (N_THR + 1), N_THR)

# module-level state for workers (fork)
_DS = None
_PNG_DIR = None


def _eval_one(nm):
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
    bi = int(np.argmax(f1))
    st = soft_stats(pred)
    return (sid, float(THR[bi]), float(f1[bi]), st["n_zc"], st["mean_gm_zc"],
            st["med_gm_zc"], st["p90_gm_zc"], st["edge_density"])


def load_pred(png_dir, sid):
    p = cv2.imread(str(Path(png_dir) / f"{sid}.png"), cv2.IMREAD_UNCHANGED)
    if p is None:
        raise FileNotFoundError(f"{png_dir}/{sid}.png")
    pred = (p.astype(np.float32) / 255.0) if p.dtype != np.uint16 else (p.astype(np.float32) / 65535.0)
    return pred


def soft_stats(pred):
    """GT-blind features from the soft map itself."""
    flat = pred[pred > 0]
    if flat.size == 0:
        return dict(n_zc=0, mean_gm_zc=0.0, med_gm_zc=0.0, p90_gm_zc=0.0,
                    edge_density=0.0)
    return dict(
        n_zc=int(flat.size),
        mean_gm_zc=float(flat.mean()),
        med_gm_zc=float(np.median(flat)),
        p90_gm_zc=float(np.percentile(flat, 90)),
        edge_density=float(flat.size) / pred.size,
    )


def main():
    global _DS, _PNG_DIR
    png_dir = Path(sys.argv[1])
    split = sys.argv[2]
    out_csv = sys.argv[3]
    _DS = BSDSDataset(_BSDS_ROOT)
    _PNG_DIR = png_dir
    names = list(_DS.val_sample_names if split == "val" else _DS.test_sample_names)

    chunks = [names[i::10] for i in range(10)]
    with get_context("fork").Pool(processes=10) as pool:
        flat = pool.map(_eval_one, names)
    rows = list(flat)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sid", "opt_thr", "f1_opt", "n_zc", "mean_gm_zc", "med_gm_zc",
                    "p90_gm_zc", "edge_density"])
        w.writerows(rows)

    t = np.array([r[1] for r in rows])
    f1 = np.array([r[2] for r in rows])
    print(f"=== {split}: {len(rows)} images ===")
    print(f"optimal thr: median {np.median(t):.4f}  mean {t.mean():.4f}  "
          f"std {t.std():.4f}  min {t.min():.4f}  max {t.max():.4f}  "
          f"p10 {np.percentile(t,10):.4f}  p90 {np.percentile(t,90):.4f}")
    print(f"OIS (mean per-image best F1): {f1.mean():.4f}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
