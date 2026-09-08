# Reproduce the headline GDSD result (v2 σ=2.8 + zcnms thinning)

Latest headline on BSDS500 (official py-bsds500 pr_eval, 99 global
thresholds, uint8 PNG soft maps):

| split | ODS | OIS | AP |
|---|---|---|---|
| val (100) | 0.5916 | 0.6304 | 0.6032 |
| test (200) | **0.6106** | **0.6318** | **0.6119** |

## Fastest path — pure Python (no C++ needed)

The whole pipeline is in `gdsd.py`.  One call reproduces the headline soft
map (verified bit-identical to the C++ benchmark features on a BSDS image):

```python
import cv2
from gdsd import soft_edge_map

img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
soft = soft_edge_map(
    img,
    sigma=2.8,                     # wide scale (kernel auto-sized from σ)
    strength="gradient_magnitude", # v2: gm@ZC
    thinning="zcnms",              # keep stronger flank of the doublet
)
# soft[i,j] in [0, ~2]; 0 = not a kept zero crossing.
```

Requirements: `numpy`, `opencv-python-headless` (Python ≥3.9).

## Full BSDS500 protocol (reproduce the table)

1. Get data + matcher:
   - BSDS500 images/groundTruth (Berkeley BSR download).
   - py-bsds500: `git clone https://github.com/Britefury/py-bsds500` and
     build (`pip install cython`, `python setup.py build_ext --inplace`).
2. Make soft maps for every val/test image with the snippet above.
3. Convert to uint8 PNG (soft × 255, clip) — `benchmark/make_soft_pngs_u8.py`.
4. Evaluate: `benchmark/run_official_pr.py <name> <val|test> <png_dir> 99`
   (env `GDSD_BSDS_ROOT`, `GDSD_PYBSDS_PATH` point at data root and the
   py-bsds500 checkout).

Example (Linux/macOS, from repo root with data + py-bsds500 beside it):

```bash
python benchmark/make_soft_pngs_u8.py soft_s28zcnms/val png_s28zcnms/val
GDSD_BSDS_ROOT=. GDSD_PYBSDS_PATH=./py-bsds500 \
  python benchmark/run_official_pr.py gdsd_v2_s28_zcnms val png_s28zcnms/val 99
```

Expected val ODS ≈ 0.5916 (OIS 0.6304, AP 0.6032).  Timing for the soft-map
step only is ≈ 1–2 s/image in pure Python (vectorized NumPy); the C++
feature extractor (`make build-benchmark` + `gdsd_features_batch`) does all
200 test images in ~5 s if you need the fast path.

## What the variant is

- GDSD = facet quadratic fit → second directional derivative R along the
  gradient → zero crossings of R (with percentile-contrast gate).
- v1 scored crossings by |R| (R ≈ 0 at the true edge — poor ranking).
- v2 scores crossings by gradient magnitude gm = hypot(d,e).
- σ=2.8 (not 1.4) is the better single scale after the kernel-size fix
  (commit ca2ad97); σ is real now (kernel auto-sized).
- zcnms thinning: on a step edge the response is a ±doublet ~2.4 px apart,
  so the ZC rule fires on both flanks; the human boundary lies between the
  lobes.  Keeping only the stronger-gm flank raises precision.

See `docs/GDSD_V2.md`, `docs/BENCHMARK.md`, `docs/GDSD_VARIANTS.md` for the
full reasoning and the numbers of every variant tried.
