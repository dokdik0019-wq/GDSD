#!/usr/bin/env python3
"""GDSD edge detection demo.

Runs the GDSD edge detector on an input image (or on a generated
synthetic sample when no input is given) and saves the results:

    output/<name>_edge.png      binary edge map (white on black)
    output/<name>_overlay.png   input image with detected edges in red
    output/<name>_response.png  normalized GDSD response

Example:
    python demo.py                      # run on the generated sample
    python demo.py --input photo.jpg    # run on your own image
    python demo.py --percentile 95 --sigma 1.0
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from gdsd import detect_edges, make_demo_image


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = float(finite.min()), float(finite.max())
    if hi - lo < 1e-12:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr - lo) / (hi - lo) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _overlay(image: np.ndarray, edge: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    vis[edge > 0] = (0, 0, 255)  # red edges
    return vis


def main() -> None:
    parser = argparse.ArgumentParser(description="GDSD edge detection demo")
    parser.add_argument("--input", type=Path, default=None,
                        help="input image path (default: generated sample image)")
    parser.add_argument("--percentile", type=float, default=90.0,
                        help="response high threshold percentile (80-95)")
    parser.add_argument("--sigma", type=float, default=1.4,
                        help="Gaussian blur sigma before the fit")
    parser.add_argument("--gradient-threshold", type=float, default=1e-2,
                        help="minimum gradient magnitude to consider a pixel")
    parser.add_argument("--output", type=Path, default=Path("output"),
                        help="output directory")
    args = parser.parse_args()

    if args.input is None:
        image = make_demo_image()
        source_name = "sample"
        print(f"[INFO] no input image given, using generated sample ({image.shape[1]}x{image.shape[0]})")
    else:
        image = cv2.imread(str(args.input), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise SystemExit(f"[FAIL] cannot read image: {args.input}")
        source_name = args.input.stem

    t0 = time.perf_counter()
    result = detect_edges(
        image,
        percentile=args.percentile,
        sigma=args.sigma,
        gradient_threshold=args.gradient_threshold,
    )
    elapsed = time.perf_counter() - t0

    edge_map = result["edge"]
    edge = edge_map.astype(np.uint8) * 255
    response = _normalize_to_uint8(result["response"])

    args.output.mkdir(parents=True, exist_ok=True)
    edge_path = args.output / f"{source_name}_edge.png"
    overlay_path = args.output / f"{source_name}_overlay.png"
    response_path = args.output / f"{source_name}_response.png"
    cv2.imwrite(str(edge_path), edge)
    cv2.imwrite(str(overlay_path), _overlay(image, edge))
    cv2.imwrite(str(response_path), response)

    print(f"[OK] GDSD finished in {elapsed:.3f}s on {image.shape[1]}x{image.shape[0]}")
    print(f"     edge pixels : {int(edge_map.sum())}")
    print(f"     Tp          : {float(result['Tp']):.6f}")
    print(f"     Tlow        : {float(result['Tlow']):.6f}")
    print(f"     saved       : {edge_path}")
    print(f"                   {overlay_path}")
    print(f"                   {response_path}")


if __name__ == "__main__":
    main()
