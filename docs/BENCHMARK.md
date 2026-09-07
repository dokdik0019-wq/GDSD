# BSDS500 benchmark

This document reports how the quantitative results in the README were
produced and how to reproduce them. **Do not compare numbers across
protocols** — the tables below use one protocol only (the official BSDS
evaluation), which is the only one comparable with the literature.

## Official BSDS500 protocol (py-bsds500)

- Images: BSDS500 splits `train` (200) / `val` (100) / `test` (200).
- Ground truth: all human segmentations (`groundTruth/*.mat`).
- Matching: `correspond_pixels` (C++ CSA implementation from
  py-bsds500), `max_dist = 0.0075` (fraction of image diagonal),
  morphological **thinning applied** to every thresholded edge map.
- Thresholds: 99 global thresholds per method; ODS/OIS/AP are computed
  from the accumulated counts, exactly as in the official BSDS suite.
- Runtime: roughly 1 h (val) to 2.5 h (test) per method on a 12-core
  machine.

The evaluation scripts live in this repository under `benchmark/` and
on the research box (`~/gdsd-bsds/`):

| Script | Purpose |
|---|---|
| `benchmark/gdsd_features_cli.cpp` | C++ feature extraction (response, gradient, zero-crossing) — output `.f64` files |
| `benchmark/make_soft.py` | builds the GDSD soft edge map from features |
| `benchmark/bsds_official_eval.py` | runs the official evaluation for one method/split |

Soft edge map definitions (what the paper/README call v1 and v2):

- **GDSD v1 (as in the AMM 2026 paper / thesis):** strength at a
  zero-crossing pixel is the absolute response `|R|`.
- **GDSD v2 (improved soft map):** strength at a zero-crossing pixel is
  the gradient magnitude `gm = hypot(d, e)`.  The diagnosis behind the
  change: on a step edge the response forms a +/− doublet and the human
  ground-truth contour sits at the |R| lobe rather than at the R = 0
  crossing, so ranking by gradient magnitude at the crossing gives a
  more precise soft map.  (Same binary detection; only the soft-map
  strength used for thresholding changes.)

## Results (official protocol, test split, 200 images)

| Method | ODS | OIS | AP |
|---|---|---|---|
| **GDSD v2 (gm@ZC, σ=1.4)** | **0.5917** | **0.6174** | **0.5762** |
| GDSD v2 (gm@ZC, σ=2.8) | 0.5917 | 0.6175 | 0.5762 |
| Canny (NMS soft map, σ=1.4) | 0.5741 | 0.6008 | 0.5523 |
| **GDSD v1 (\|R\|@ZC, σ=1.4)** | **0.5743** | **0.6031** | **0.5462** |
| Haralick facet (1984, cubic) | 0.5196 | 0.5523 | 0.4668 |

val split (100 images), for completeness:

| Method | ODS | OIS | AP |
|---|---|---|---|
| GDSD v2 σ=1.4 | 0.5680 | 0.6175 | 0.5618 |
| GDSD v2 σ=2.8 | 0.5724 | 0.6207 | 0.5691 |
| Canny | 0.5584 | 0.6046 | 0.5464 |
| **GDSD v1 (\|R\|@ZC)** | **0.5568** | **0.6084** | **0.5407** |
| Haralick | 0.4994 | 0.5516 | 0.4539 |

Note: σ=1.4 and σ=2.8 give the same test numbers because the official
evaluation thresholds each soft map over a global grid; the σ sweep on
`val` preferred 2.8 by a small margin.

## Internal protocol (thesis Table 4.4 / AMM 2026 §5.2)

The thesis and the AMM 2026 paper report an *internal* benchmark on the
radial square pattern with a one-pixel tolerance and precision /
recall / F-measure. Those numbers are computed by
`tests/verify_thesis_table44.py` and are **not** directly comparable
with the BSDS500 numbers above.

## Reproducing

```bash
# 1. features (C++) — needs OpenCV dev headers
make build-cpp
# 2. soft maps + official evaluation — needs py-bsds500 + BSDS500 data
python benchmark/make_soft.py ...   # see script usage
python benchmark/bsds_official_eval.py gdsd2 val 99
```
