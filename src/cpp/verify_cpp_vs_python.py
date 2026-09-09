#!/usr/bin/env python3
"""verify_cpp_vs_python.py — compare C++ (PNG) vs Python (npy) edge maps over the full set

Requires mismatch = 0 on every image for the C++ to count as exactly matching Python

Usage:
  python src/cpp/verify_cpp_vs_python.py [--py-dir DIR] [--cpp-dir DIR]
  default: output/edges (Python .npy) and output_cpp/edges (C++ .png) next to the repo
"""
import argparse, sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--py-dir", default=str(ROOT / "output" / "edges"))
    ap.add_argument("--cpp-dir", default=str(ROOT / "output_cpp" / "edges"))
    args = ap.parse_args()

    PY_EDGE = Path(args.py_dir)
    CPP_EDGE = Path(args.cpp_dir)

    if not PY_EDGE.exists() or not CPP_EDGE.exists():
        print(f"[SKIP] dirs not found: {PY_EDGE} or {CPP_EDGE}")
        return 2

    py_files = sorted(PY_EDGE.glob("*.npy"))
    total = len(py_files)
    mismatches = 0
    checked = 0
    for pyf in py_files:
        stem = pyf.stem  # e.g. train__100075
        cppf = CPP_EDGE / f"{stem}.png"
        if not cppf.exists():
            print(f"[MISSING] {cppf}"); mismatches += 1; continue
        py_edge = np.load(pyf).astype(np.uint8)
        cpp_img = cv2.imread(str(cppf), cv2.IMREAD_GRAYSCALE)
        if cpp_img is None:
            print(f"[BADPNG] {cppf}"); mismatches += 1; continue
        cpp_edge = (cpp_img > 0).astype(np.uint8)
        checked += 1
        if py_edge.shape != cpp_edge.shape:
            print(f"[SHAPE] {stem}: {py_edge.shape} vs {cpp_edge.shape}"); mismatches += 1; continue
        n = int((py_edge != cpp_edge).sum())
        if n > 0:
            mismatches += 1
            if mismatches <= 5:
                print(f"[DIFF] {stem}: mismatch={n} (py={int(py_edge.sum())}, cpp={int(cpp_edge.sum())})")
    print(f"\nchecked {checked}/{total} | mismatched {mismatches}")
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
