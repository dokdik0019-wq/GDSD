#!/usr/bin/env python3
"""run_official_pr.py — evaluate soft-map PNGs with the official py-bsds500
pr_evaluation pipeline (global linspace thresholds, interp AP).

Mirrors the official BSDS MATLAB suite: each method's soft maps are
PNG images (uint16, [0,1] after /65535); thresholds are a fixed grid
linspace over [0,1]; ODS = best F1 from accumulated counts, OIS = mean
per-image best F1, AP = area under interpolated PR.

Usage:
    python run_official_pr.py <method> <split> <png_dir> <n_thr>
    e.g.  python run_official_pr.py gdsd2 test ods_ois_ap/png_official/gdsd2/test 99
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

import numpy as np

_BSDS_ROOT = os.environ.get(
    "GDSD_BSDS_ROOT", str(Path(__file__).resolve().parent.parent)
)
_PYBSDS = os.environ.get("GDSD_PYBSDS_PATH", str(Path(__file__).resolve().parent / "py-bsds500"))
sys.path.insert(0, _PYBSDS)

from bsds.bsds_dataset import BSDSDataset
from bsds import evaluate_boundaries

from multiprocessing import Pool

# module-level state set by main() (needed by worker processes)
_DS = None
_PNG_DIR = None
_THRESHOLDS = None


def _load_pred(sample_name):
    import cv2
    p = cv2.imread(str(_PNG_DIR / f"{Path(sample_name).name}.png"), cv2.IMREAD_UNCHANGED)
    if p is None:
        raise FileNotFoundError(_PNG_DIR / f"{Path(sample_name).name}.png")
    if p.dtype == np.uint16:
        pred = p.astype(np.float32) / 65535.0
    else:
        pred = p.astype(np.float32) / 255.0
    bnds = _DS.boundaries(sample_name)
    tgt = bnds[0].shape
    if pred.shape != tgt:
        pred = pred[: tgt[0], : tgt[1]]
        pred = np.pad(pred, [(0, tgt[0] - pred.shape[0]), (0, tgt[1] - pred.shape[1])],
                      mode="constant")
    return pred


def shard_eval(names):
    """Evaluate one shard of images; returns accumulators + per-image best F1."""
    n_thr = len(_THRESHOLDS)
    count_r = np.zeros(n_thr)
    sum_r = np.zeros(n_thr)
    count_p = np.zeros(n_thr)
    sum_p = np.zeros(n_thr)
    per_img_best = []
    for sid in names:
        pred = _load_pred(sid)
        gt_b = _DS.boundaries(sid)
        cr, sr, cp, sp, used = evaluate_boundaries.evaluate_boundaries(
            pred, gt_b, thresholds=_THRESHOLDS, apply_thinning=True
        )
        count_r += cr; sum_r += sr; count_p += cp; sum_p += sp
        rec, prec, f1 = evaluate_boundaries.compute_rec_prec_f1(cr, sr, cp, sp)
        bi = int(np.argmax(f1))
        per_img_best.append(f1[bi])
    return count_r, sum_r, count_p, sum_p, np.array(per_img_best)


def main():
    method = sys.argv[1]
    split = sys.argv[2]
    png_dir = Path(sys.argv[3])
    n_thr = int(sys.argv[4]) if len(sys.argv) > 4 else 99

    global _DS, _PNG_DIR, _THRESHOLDS
    _DS = BSDSDataset(_BSDS_ROOT)
    _PNG_DIR = png_dir
    _THRESHOLDS = np.linspace(1.0 / (n_thr + 1), 1.0 - 1.0 / (n_thr + 1), n_thr)

    sample_names = _DS.val_sample_names if split == "val" else _DS.test_sample_names

    t0 = time.time()
    names = list(sample_names)
    chunks = [names[i::10] for i in range(10)]
    with Pool(processes=10) as pool:
        results = pool.map(shard_eval, chunks)

    count_r = sum(r[0] for r in results)
    sum_r = sum(r[1] for r in results)
    count_p = sum(r[2] for r in results)
    sum_p = sum(r[3] for r in results)
    per_img_best = np.concatenate([r[4] for r in results])

    rec, prec, f1 = evaluate_boundaries.compute_rec_prec_f1(count_r, sum_r, count_p, sum_p)
    best_i = int(np.argmax(f1))
    ods = float(f1[best_i])
    ois = float(per_img_best.mean()) if len(per_img_best) else 0.0

    # AP: official interpolation (np.interp over unique recall, 0..1 step 0.01)
    rec_unique, rec_unique_ndx = np.unique(rec, return_index=True)
    prec_unique = prec[rec_unique_ndx]
    if rec_unique.shape[0] > 1:
        prec_interp = np.interp(np.arange(0, 1, 0.01), rec_unique, prec_unique,
                                left=0.0, right=0.0)
        ap = float(prec_interp.sum() * 0.01)
    else:
        ap = 0.0

    print(f"\n=== {method} {split} (official pr_eval, {n_thr} thr) ===")
    print(f"ODS={ods:.4f}  OIS={ois:.4f}  AP={ap:.4f}  "
          f"(best thr {_THRESHOLDS[best_i]:.4f}, R={rec[best_i]:.3f} P={prec[best_i]:.3f})")
    print(f"time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
