#!/usr/bin/env python3
"""bsds_official_eval.py — evaluate soft maps with the official BSDS benchmark (py-bsds500)

Uses correspond_pixels (C++ CSA) + morphological thinning + per-human GT matching
identical to the official BSDS evaluation suite -> ODS/OIS/AP comparable to the literature

Usage: python bsds_official_eval.py <method> <split> [n_thresholds]
  <method> = folder name under ods_ois_ap/soft/ (haralick, gdsd2, canny, sobel, log)
  split = val | test
Output: ODS, OIS, AP
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import numpy as np
from multiprocessing import Pool

# Paths can be overridden via env vars so the scripts run both from a
# checked-out benchmark dir and from the research workspace:
#   GDSD_BSDS_ROOT   directory containing data/images, groundTruth (BSDS500)
#   GDSD_SOFT_ROOT   directory containing <method>/<split>/*.npy soft maps
#   GDSD_PYBSDS_PATH location of py-bsds500 (default: <repo>/benchmark/py-bsds500)
_HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("GDSD_BSDS_ROOT", _HERE))
SOFT = Path(os.environ.get("GDSD_SOFT_ROOT", _HERE / "soft"))
_PYBSDS = Path(os.environ.get("GDSD_PYBSDS_PATH", _HERE / "py-bsds500"))
sys.path.insert(0, str(_PYBSDS))

from bsds.bsds_dataset import BSDSDataset
from bsds import evaluate_boundaries

def compute_ap(count_r, sum_r, count_p, sum_p):
    """AP from the accumulators across thresholds (same as official: area under PR)."""
    R = count_r / (sum_r + (sum_r == 0))
    P = count_p / (sum_p + (sum_p == 0))
    # interpolated AP (P at recall >= L)
    recall_grid = np.linspace(0, 1, 101)
    ap = 0.0
    for L in recall_grid:
        idx = np.where(R >= L)[0]
        if len(idx):
            ap += P[idx].max() / len(recall_grid)
    return ap

def eval_one(sid, soft_dir, name_map, thresholds, ds):
    """Evaluate one image over thresholds → accumulators."""
    s = np.load(soft_dir / f"{sid}.npy").astype(np.float32)
    gt_key = name_map.get(sid, sid)
    bnds = ds.boundaries(gt_key)
    n = len(thresholds)
    crs = np.zeros(n); srs = np.zeros(n); cps = np.zeros(n); sps = np.zeros(n)
    img_f = []
    for ti, t in enumerate(thresholds):
        det = (s >= t).astype(np.uint8)
        if det.sum() == 0:
            continue
        cr, sr, cp, sp = evaluate_boundaries.evaluate_boundaries_bin(
            det, bnds, max_dist=0.0075, apply_thinning=True)
        crs[ti] = cr; srs[ti] = sr; cps[ti] = cp; sps[ti] = sp
        R = cr / (sr + (sr == 0)); P = cp / (sp + (sp == 0))
        F = 2*R*P/(R+P) if R+P else 0
        img_f.append(F)
    best_f = max(img_f) if img_f else 0.0
    return crs, srs, cps, sps, best_f

def main():
    method = sys.argv[1]
    split = sys.argv[2]
    n_thr = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    soft_dir = SOFT / method / split
    ids = sorted([p.stem for p in soft_dir.glob("*.npy")])
    if not ids:
        print("no .npy in", soft_dir); return

    ds = BSDSDataset(str(ROOT))
    sample_names = ds.val_sample_names if split == "val" else ds.test_sample_names
    name_map = {Path(n).name: n for n in sample_names}

    all_vals = []
    softs = {}
    for sid in ids:
        s = np.load(soft_dir / f"{sid}.npy").astype(np.float32)
        softs[sid] = s
        v = s[s > 0]
        if v.size:
            all_vals.append(v)
    if all_vals:
        allv = np.concatenate(all_vals)
        thresholds = np.percentile(allv, np.linspace(1, 99, n_thr))
    else:
        thresholds = np.linspace(0.01, 1.0, n_thr)

    t0 = time.time()
    count_r_all = np.zeros(n_thr); sum_r_all = np.zeros(n_thr)
    count_p_all = np.zeros(n_thr); sum_p_all = np.zeros(n_thr)
    per_img_best = []

    # parallel across images (CSA is CPU-bound; the box has 12 cores but other jobs run -> use 4)
    import functools
    work = [(sid, soft_dir, name_map, thresholds, ds) for sid in ids]
    with Pool(processes=4) as pool:
        results = pool.starmap(eval_one, work)
    for i, (crs, srs, cps, sps, bf) in enumerate(results):
        count_r_all += crs; sum_r_all += srs; count_p_all += cps; sum_p_all += sps
        per_img_best.append(bf)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(ids)} ({time.time()-t0:.0f}s)", flush=True)

    # ODS: best F from the pooled accumulators
    R_ods = count_r_all / (sum_r_all + (sum_r_all == 0))
    P_ods = count_p_all / (sum_p_all + (sum_p_all == 0))
    F_ods = 2*R_ods*P_ods/(R_ods+P_ods+1e-12)
    best_i = int(np.argmax(F_ods))
    ods = float(F_ods[best_i])
    ois = float(np.mean(per_img_best)) if per_img_best else 0.0
    # AP
    ap = compute_ap(count_r_all, sum_r_all, count_p_all, sum_p_all)

    print(f"\n=== {method} {split} (official BSDS, {n_thr} thr) ===")
    print(f"ODS={ods:.4f}  OIS={ois:.4f}  AP={ap:.4f}  (best thr idx {best_i}, R={R_ods[best_i]:.3f} P={P_ods[best_i]:.3f})")
    print(f"time {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
