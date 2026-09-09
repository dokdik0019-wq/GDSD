#!/usr/bin/env python3
"""eval_adaptive_thresholds.py — compare threshold strategies under ONE metric
(mean per-image F1, official matching), so comparisons are apples-to-apples.

Strategies:
  global      : single fixed threshold (the ODS-optimal pooled threshold)
  committee   : linear model frozen from val (mean_gm_zc, p90_gm_zc)
  oracle      : per-image GT-optimal threshold (upper bound = OIS mean)
  median      : fixed per-image threshold = val median opt_thr (0.15) — same as global if equal
  bin3        : committee bins by mean_gm_zc -> threshold per bin (fit on val, apply test)

Usage: eval_adaptive_thresholds.py <val_csv> <test_csv> <val_png> <test_png>
"""
from __future__ import annotations
import csv
import os
import sys
from pathlib import Path
import numpy as np
from multiprocessing import Pool, get_context

_BSDS_ROOT = os.environ.get("GDSD_BSDS_ROOT", str(Path.home() / "gdsd-bsds-mac"))
_PYBSDS = os.environ.get("GDSD_PYBSDS_PATH", "/tmp/py-bsds500")
sys.path.insert(0, _PYBSDS)
from bsds.bsds_dataset import BSDSDataset
from bsds import evaluate_boundaries

N_THR = 99
THR = np.linspace(1.0 / (N_THR + 1), 1.0 - 1.0 / (N_THR + 1), N_THR)
FEATS = ["mean_gm_zc", "p90_gm_zc"]

# worker globals (fork)
_DS = None
_PNG_DIR = None
_NAME_PREFIX = ""
_STRATEGY = "global"
_COEF = None
_GLOBAL_THR = 0.15
_BIN_EDGES = None
_BIN_THR = None


def _thr_for(row):
    """Compute this row's threshold for the active strategy (GT-blind)."""
    s = _STRATEGY
    if s == "global":
        return _GLOBAL_THR
    if s == "oracle":
        return float(row["opt_thr"])
    if s == "committee":
        return float(_COEF[0] + sum(c * float(row[f]) for c, f in zip(_COEF[1:], FEATS)))
    if s == "bin3":
        m = float(row["mean_gm_zc"])
        b = 2 if m >= _BIN_EDGES[1] else (1 if m >= _BIN_EDGES[0] else 0)
        return _BIN_THR[b]
    raise ValueError(s)


def _eval_one(row):
    sid = row["sid"]
    thr = _thr_for(row)
    pred = load_pred(_PNG_DIR, sid)
    gt = _DS.boundaries(_NAME_PREFIX + sid)
    tgt = gt[0].shape
    if pred.shape != tgt:
        pred = pred[: tgt[0], : tgt[1]]
        pred = np.pad(pred, [(0, tgt[0] - pred.shape[0]), (0, tgt[1] - pred.shape[1])], mode="constant")
    thr_i = int(np.argmin(np.abs(THR - thr)))
    cr, sr, cp, sp, _ = evaluate_boundaries.evaluate_boundaries(
        pred, gt, thresholds=np.array([THR[thr_i]]), apply_thinning=True)
    rec, prec, f1 = evaluate_boundaries.compute_rec_prec_f1(cr, sr, cp, sp)
    return float(f1[0])


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_pred(png_dir, sid):
    import cv2
    p = cv2.imread(str(Path(png_dir) / f"{sid}.png"), cv2.IMREAD_UNCHANGED)
    pred = (p.astype(np.float32) / 255.0) if p.dtype != np.uint16 else (p.astype(np.float32) / 65535.0)
    return pred


def eval_at_thresholds(rows, png_dir, name_prefix, ds, strategy,
                       coef=None, global_thr=0.15, bin_edges=None, bin_thr=None):
    """Evaluate one named strategy; mean per-image F1 (10 workers, fork)."""
    global _DS, _PNG_DIR, _NAME_PREFIX, _STRATEGY, _COEF, _GLOBAL_THR, _BIN_EDGES, _BIN_THR
    _DS, _PNG_DIR, _NAME_PREFIX, _STRATEGY = ds, png_dir, name_prefix, strategy
    _COEF, _GLOBAL_THR, _BIN_EDGES, _BIN_THR = coef, global_thr, bin_edges, bin_thr
    with get_context("fork").Pool(processes=10) as pool:
        f1s = pool.map(_eval_one, rows)
    return float(np.mean(f1s)), f1s


def main():
    val_csv, test_csv = sys.argv[1], sys.argv[2]
    val_png, test_png = sys.argv[3], sys.argv[4]
    val_rows = read_csv(val_csv)
    test_rows = read_csv(test_csv)
    ds = BSDSDataset(_BSDS_ROOT)

    # --- fit on val ---
    Xv = np.stack([[1.0] + [float(r[f]) for f in FEATS] for r in val_rows])
    yv = np.array([float(r["opt_thr"]) for r in val_rows])
    coef, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    print("committee coef:", np.round(coef, 4))

    # global ODS-optimal threshold from val opt (pooled, use median as proxy? no — use
    # the known ODS best thr: it equals the mode around val 0.15). We take it from
    # the val CSV median of opt_thr (fair fixed thr chosen on val).
    global_thr = float(np.median(yv))
    print(f"global fixed thr (val median opt): {global_thr:.4f}")

    # bin3 thresholds by mean_gm_zc terciles of the *val* distribution
    mg = np.array([float(r["mean_gm_zc"]) for r in val_rows])
    edges = np.quantile(mg, [1 / 3, 2 / 3])
    bin_thr = []
    for lo, hi in [(-np.inf, edges[0]), (edges[0], edges[1]), (edges[1], np.inf)]:
        idx = (mg >= lo) & (mg < hi) if hi != np.inf else (mg >= lo)
        bin_thr.append(float(np.mean(yv[idx])) if idx.sum() else global_thr)
    print(f"bin3 edges: {np.round(edges,4)}  thr/bin: {np.round(bin_thr,4)}")

    strategies = [
        ("global (fixed)", "global", dict(global_thr=global_thr)),
        ("committee linear", "committee", dict(coef=coef)),
        ("bin3 committee", "bin3", dict(bin_edges=edges, bin_thr=bin_thr)),
        ("oracle (upper)", "oracle", {}),
    ]

    for split, rows, png, prefix in [("val", val_rows, val_png, "val/"),
                                     ("test", test_rows, test_png, "test/")]:
        print(f"\n=== {split} (mean per-image F1) ===")
        for name, strat, kw in strategies:
            mean_f1, _ = eval_at_thresholds(rows, png, prefix, ds, strat, **kw)
            print(f"  {name:22s} {mean_f1:.4f}")


if __name__ == "__main__":
    main()
