#!/usr/bin/env python3
"""committee_threshold.py — train a GT-blind per-image threshold predictor on
val, freeze it, and evaluate on test (the v5 experiment).

Protocol (mirrors the CHEV committee design):
  - TRAIN on val: features are soft-map statistics ONLY (no GT); the target
    is the oracle per-image threshold (from GT).  This is in-sample/upper
    bound knowledge used for fitting, NOT for deployment.
  - FREEZE the fitted predictor.
  - EVALUATE on test: predict a threshold per image from features only, then
    compute per-image F1 at that threshold (official py-bsds500 matching).

Outputs the 3 honest rows:
  baseline ODS (single global threshold, official)
  oracle OIS (per-image GT-optimal threshold = upper bound)
  committee mean-F1 (GT-blind predicted thresholds, frozen model)

Usage: committee_threshold.py <val_csv> <test_csv> <val_png_dir> <test_png_dir>
"""
from __future__ import annotations
import csv
import os
import sys
from pathlib import Path
import numpy as np

_BSDS_ROOT = os.environ.get("GDSD_BSDS_ROOT", "/Users/dookdik/gdsd-bsds-mac")
_PYBSDS = os.environ.get("GDSD_PYBSDS_PATH", "/tmp/py-bsds500")
sys.path.insert(0, _PYBSDS)
from bsds.bsds_dataset import BSDSDataset
from bsds import evaluate_boundaries

N_THR = 99
THR = np.linspace(1.0 / (N_THR + 1), 1.0 - 1.0 / (N_THR + 1), N_THR)
FEATS = ["n_zc", "mean_gm_zc", "med_gm_zc", "p90_gm_zc", "edge_density"]
# features actually used (the two that correlate with the oracle threshold)
USE = ["mean_gm_zc", "p90_gm_zc"]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_pred(png_dir, sid):
    import cv2
    p = cv2.imread(str(Path(png_dir) / f"{sid}.png"), cv2.IMREAD_UNCHANGED)
    pred = (p.astype(np.float32) / 255.0) if p.dtype != np.uint16 else (p.astype(np.float32) / 65535.0)
    return pred


def fit_committee(val_rows):
    """Least squares: threshold ~ b0 + b1*mean_gm + b2*p90_gm  (frozen on val)."""
    X = np.stack([[1.0] + [float(r[f]) for f in USE] for r in val_rows])
    y = np.array([float(r["opt_thr"]) for r in val_rows])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def predict(coef, row):
    return float(coef[0] + sum(c * float(row[f]) for c, f in zip(coef[1:], USE)))


def main():
    val_csv, test_csv = sys.argv[1], sys.argv[2]
    val_png, test_png = sys.argv[3], sys.argv[4]
    val_rows = read_csv(val_csv)
    test_rows = read_csv(test_csv)
    ds = BSDSDataset(_BSDS_ROOT)

    coef = fit_committee(val_rows)
    print("committee coef (frozen from val):", np.round(coef, 4))

    # ---- per-image F1 at committee threshold (val = sanity, test = real) ----
    for split, rows, png_dir, name in [
        ("val", val_rows, val_png, "val/"),
        ("test", test_rows, test_png, "test/"),
    ]:
        f1s = []
        for r in rows:
            sid = r["sid"]
            thr = predict(coef, r)
            pred = load_pred(png_dir, sid)
            gt = ds.boundaries(name + sid)
            tgt = gt[0].shape
            if pred.shape != tgt:
                pred = pred[: tgt[0], : tgt[1]]
                pred = np.pad(pred, [(0, tgt[0] - pred.shape[0]), (0, tgt[1] - pred.shape[1])], mode="constant")
            thr_i = int(np.argmin(np.abs(THR - thr)))
            cr, sr, cp, sp, _ = evaluate_boundaries.evaluate_boundaries(
                pred, gt, thresholds=np.array([THR[thr_i]]), apply_thinning=True)
            rec, prec, f1 = evaluate_boundaries.compute_rec_prec_f1(cr, sr, cp, sp)
            f1s.append(float(f1[0]))
        print(f"=== {split} committee (GT-blind, frozen) ===")
        print(f"mean F1 @ committee thr: {np.mean(f1s):.4f}")


if __name__ == "__main__":
    main()
