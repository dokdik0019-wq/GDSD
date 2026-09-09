#!/usr/bin/env python3
"""diff_test.py — differential test: C++ vs Python (gdsd.py) must match 100%

Procedure:
  1. Pick sample images (default: 3 from BSDS500 train)
  2. Run Python detect_edges -> edge map
  3. Run C++ gdsd_cli -> edge map
  4. Compare: identical? mismatch pixels? are Tp/Tlow close?

Data:
  - Requires BSDS500 images (fault-tolerant: if there is no data/ next to the repo,
    set the GDSD_BSDS_DATA env var to a data/ folder containing images/train)
  - Without any BSDS data, falls back to 3 synthetic (demo) images
"""
import os, subprocess, sys
from pathlib import Path
import numpy as np
import cv2

# repo root = src/cpp/.. (where gdsd.py lives)
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from gdsd import detect_edges

# locate data: env override -> data/ next to the repo
DATA_DIR = Path(os.environ.get("GDSD_BSDS_DATA", str(ROOT / "data")))
IMG_DIR = DATA_DIR / "images" / "train"
CPP_BIN = ROOT / "src" / "cpp" / "build" / "gdsd_cli"
SAMPLES = ["100075.jpg", "100080.jpg", "100098.jpg"]


def make_synthetic(name, path):
    """Build synthetic images (like demo.py) for the no-BSDS case"""
    img = np.zeros((256, 256), np.uint8)
    cv2.rectangle(img, (20, 20), (110, 110), 160, -1)
    cv2.line(img, (130, 35), (220, 215), 230, 4)
    cv2.circle(img, (190, 70), 30, 255, -1)
    cv2.putText(img, "GDSD", (40, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 190, 4, cv2.LINE_AA)
    rng = np.random.default_rng(0)
    noisy = np.clip(img.astype(np.float64) + rng.normal(0.0, 3.0, img.shape), 0, 255)
    cv2.imwrite(str(path), noisy.astype(np.uint8))
    return path


def run_cpp(img_path, out_path, pct=90.0, sigma=1.4):
    if not CPP_BIN.exists():
        return None, f"C++ binary not found: {CPP_BIN} (run make build-cpp)"
    # C++ writes PNG
    png_path = out_path.with_suffix(".png")
    r = subprocess.run([str(CPP_BIN), str(img_path), str(png_path),
                        str(pct), str(sigma), "0.01"],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return None, r.stderr.strip()
    edge = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    if edge is None:
        return None, "cannot read C++ png output"
    return (edge > 0).astype(np.uint8), r.stdout.strip()


def main() -> int:
    if not CPP_BIN.exists():
        print(f"[SKIP] build C++ first: make build-cpp (missing {CPP_BIN})")
        return 2

    # prepare sample paths
    use_synthetic = not IMG_DIR.exists()
    if use_synthetic:
        print("[INFO] BSDS data not found — using synthetic images")
        synth_dir = ROOT / "src" / "cpp" / ".tmp_diff"
        synth_dir.mkdir(parents=True, exist_ok=True)
        sample_paths = []
        for i, name in enumerate(SAMPLES):
            p = synth_dir / name
            if not p.exists():
                make_synthetic(name, p)
            sample_paths.append(p)
    else:
        sample_paths = []
        for name in SAMPLES:
            p = IMG_DIR / name
            if p.exists():
                sample_paths.append(p)

    if not sample_paths:
        print("[FAIL] no sample images available")
        return 1

    failures = 0
    for img_path in sample_paths:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        # Python
        res = detect_edges(img, percentile=90.0, sigma=1.4, gradient_threshold=1e-2)
        py_edge = res["edge"].astype(np.uint8)
        # C++
        cpp_out = img_path.parent / f"diff_{img_path.stem}"
        cpp_edge, cpp_log = run_cpp(img_path, cpp_out)
        if cpp_edge is None:
            print(f"[FAIL] {img_path.name}: C++ error: {cpp_log}"); failures += 1; continue
        # compare
        same_shape = py_edge.shape == cpp_edge.shape
        n_mismatch = int((py_edge != cpp_edge).sum()) if same_shape else -1
        n_edge_py = int(py_edge.sum())
        n_edge_cpp = int(cpp_edge.sum())
        ok = same_shape and n_mismatch == 0
        tp_cpp = cpp_log.split("Tp=")[1].split(" ")[0] if "Tp=" in cpp_log else "?"
        print(f"{img_path.name}: shape {py_edge.shape}=={cpp_edge.shape} ({same_shape}) "
              f"| mismatch={n_mismatch} | edge_px py={n_edge_py} cpp={n_edge_cpp} "
              f"| Tp py={res['Tp']:.6f} cpp={tp_cpp}")
        if not ok:
            failures += 1
    print(f"\n{'PASS' if failures == 0 else f'FAIL ({failures} failures)'}: "
          f"{len(sample_paths)} images diff-tested")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
