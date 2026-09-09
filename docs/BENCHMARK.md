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
- Soft maps: uint8 PNG (255 × raw value, clipped) — loaded by the official
  pipeline as `imread/255`, so every image occupies its natural [0,1]
  scale (this is the convention used for classical edge detectors).
- Thresholds: a fixed global grid
  `linspace(1/(N+1), 1-1/(N+1), N)` over [0,1] with N=99, applied to
  every image of every method (not per-method percentiles).
- ODS = best F1 from counts accumulated over all images at the shared
  grid; OIS = mean of per-image best F1; AP = area under the
  precision-recall curve interpolated over recall in 0.01 steps —
  exactly the official BSDS suite definition.
- Runtime: roughly 1 h (val) to 2.5 h (test) per method on a 12-core
  machine.

The evaluation scripts live in this repository under `benchmark/` and
on the research box (`~/gdsd-bsds/`):

| Script | Purpose |
|---|---|
| `benchmark/gdsd_features_cli.cpp` | C++ feature extraction (response, gradient, zero-crossing) — output `.f64` files |
| `benchmark/make_soft.py` | builds the GDSD soft edge map from features (`--strength response` = v1, `gradient_magnitude` = v2) |
| `benchmark/make_soft_pngs_u8.py` | converts soft maps to uint8 PNG (×255, clip) — the official `imread/255` convention |
| `benchmark/run_official_pr.py` | runs the official `pr_evaluation` pipeline (PNG soft maps, global linspace thresholds, interp AP) |
| `benchmark/bsds_official_eval.py` | *(legacy)* earlier percentile-threshold variant; kept for reference — not the numbers in the tables |

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

**How to read these numbers.**  All results below were measured after the
Gaussian kernel-size fix (commit `ca2ad97`, 2026-09-08), described in the
next section.  Pre-fix code pinned the blur kernel to 5×5, so the σ
parameter never reached its labelled value for σ ≳ 1.4 (effective σ:
set 1.4 → 1.16, set 2.8 → 1.35); the old rows where σ=1.4 and σ=2.8 scored
identically (0.5917) were an artefact of that bug and are void.  Here each
labelled σ is the σ actually applied.

| Method | ODS | OIS | AP | exec time (200 img) |
|---|---|---|---|---|
| **GDSD v2 σ=2.8 + zcnms thinning** | **0.6106** | **0.6318** | **0.6119** | ~6 s |
| **GDSD v2 (gm@ZC, σ=2.8)** | **0.6070** | **0.6284** | **0.6085** | 6 s |
| Haralick facet, tuned (ρ=2.0, σ=2.8) | 0.5976 | 0.6218 | 0.5378 | 4 s |
| Haralick facet tuned + zc_nms thinning | 0.5994 | 0.6218 | 0.5375 | ~4 s |
| Haralick facet oct_zero (octant dir) | 0.5962 | 0.6255 | 0.5488 | ~4 s |
| Haralick tuned + rank-norm (zero-fix) | 0.6032 | 0.6341 | 0.5571 | ~4 s + rank |
| GDSD v2 (gm@ZC, σ=1.4) | 0.5913 | 0.6180 | 0.4948 | ~6 s |
| **GDSD v1 (\|R\|@ZC, σ=1.4)** | 0.5794 | 0.6055 | 0.3530 | ~6 s |
| Canny (NMS soft map, σ=1.4) | 0.5740 | 0.6008 | 0.4866 | 15 s |
| Haralick facet (1984, ρ≤1, no blur) | 0.5194 | 0.5513 | 0.4708 | ~4 s |
| ELSE (NMS, double) | 0.5143 | 0.5543 | 0.4711 | — |
| ELSE (R, single) | 0.5118 | 0.5592 | 0.4617 | — |
| Sobel (NMS soft map, σ=1.4) | 0.5101 | 0.5456 | 0.1636 | 16 s |

**Execution-time note.**  Column is detector-specific wall time for the
full 200-image test split (serial loop, single run) on the research box
**sc7308**: AMD Ryzen 5 5600G (6 cores / 12 threads, up to 4.47 GHz),
15 GB RAM, NVIDIA GTX 1660 SUPER 6 GB (not used — all timings are
CPU-only), Ubuntu 24.04 LTS.  GDSD = batch C++ features
(`gdsd_features_batch`, one process for all images — 5 s; σ1.4/σ2.8 same
cost) + `make_soft.py` (1 s) + uint8 PNG (<1 s); v1/v2 share the features
and differ only in the `make_soft.py` strength.  Haralick =
`haralick_facet.py` (4 s at ρ=2.0/σ=2.8; the untuned config is the same
code path).  Canny/Sobel = `make_softmaps.py` (15–16 s).  The official
pr_eval matching (~5 min on this box) is excluded — it is a fixed protocol
cost, identical for every method.  Raw timing log:
`timing_results/run.log` on the research box.

val split (100 images), for completeness:

| Method | ODS | OIS | AP |
|---|---|---|---|
| **GDSD v2 σ=2.8 + zcnms** | **0.5916** | **0.6304** | **0.6032** |
| **GDSD v2 σ=2.8** | **0.5881** | **0.6269** | **0.5998** |
| Haralick tuned (ρ=2.0, σ=2.8) | 0.5794 | 0.6199 | 0.5306 |
| Haralick tuned + zc_nms thinning | 0.5816 | 0.6201 | 0.5299 |
| Haralick oct_zero (octant dir) | 0.5776 | 0.6236 | 0.5408 |
| Haralick tuned + rank-norm (zero-fix) | 0.5908 | 0.6328 | 0.5537 |
| GDSD v2 σ=1.4 | 0.5724 | 0.6215 | 0.4722 |
| **GDSD v1 (\|R\|@ZC, σ=1.4)** | 0.5614 | 0.6108 | 0.3459 |
| Canny | 0.5584 | 0.6044 | 0.4675 |
| ELSE (NMS, double) | 0.5099 | 0.5602 | 0.4853 |
| ELSE (R, single) | 0.5071 | 0.5647 | 0.4784 |
| Haralick (1984, ρ≤1, no blur) | 0.4996 | 0.5507 | 0.4638 |
| Sobel | 0.4938 | 0.5457 | 0.1555 |

AP values follow the official definition (area under the real PR curve,
interpolated over recall) — classical detectors have low AP by nature,
which is why AP is rarely the headline metric for them.  The
rank-normalized Haralick row applies `rank_norm_soft.py` before the sweep
(its soft values are per-image ranks, not raw strengths), so it is
comparable with the other Haralick rows only through the ODS/OIS columns;
raw-sweep AP systematically favours gradient-magnitude scorers.

## What the kernel-size fix changed (commit `ca2ad97`, 2026-09-08)

The code base lets the caller choose the smoothing scale σ (the Gaussian
blur applied before the facet fit).  Before this commit, both `gdsd.py` and
`src/cpp/gdsd_cpp.hpp` fixed the Gaussian kernel at 5×5:

```python
cv2.GaussianBlur(image, (5, 5), sigmaX=sigma, ...)
```

A 5×5 window cannot represent a wide Gaussian: the achievable second moment
saturates near σ ≈ 1.38.  Measured effective σ (second moment of a blurred
delta):

| σ requested | effective with (5,5) | effective with auto kernel |
|---|---|---|
| 1.0 | 0.96 | 1.00 |
| 1.4 | 1.16 | 1.40 |
| 2.0 | 1.29 | 2.00 |
| 2.8 | 1.35 | 2.80 |
| 4.0 | 1.38 | 4.00 |

This is why pre-fix σ=1.4 and σ=2.8 rows were identical — the code could not
actually set σ=2.8.  The fix replaces the fixed size with `ksize=(0,0)` /
`cv::Size(0,0)`, letting OpenCV derive the kernel from σ.  Python/C++ parity
was re-verified (bit-identical on a BSDS image, maxdiff 0.0).  A regression
test (`test_sigma_kernel_not_saturated`) pins the behaviour.

Scale now has a real effect and σ=2.8 is the best single scale for GDSD v2
on both splits (val 0.5881 vs 0.5724 at σ=1.4; test 0.6070 vs 0.5913).
This matches the sensitivity the internal pipeline found (σ=2.8 favoured on
val); the reviewer's note that the loose scale knob was costing GDSD points
turned out to be correct.

## Haralick sensitivity (ρ/σ sweep, official protocol, val split)

Haralick (1984) facet edges are the zero crossings of the second
directional derivative with `f''>0, f'''<0` and a sub-pixel root inside
the centre pixel (`|ρ₀| ≤ 1`).  The generator lives in
[`benchmark/haralick_facet.py`](../benchmark/haralick_facet.py) — it takes
`rho_max` (root bound) and `sigma` (pre-fit blur) as parameters, so the
baseline can be tuned rather than quoted at a single fixed setting.

Val ODS over a 3×3 grid:

| ρ / σ | σ=0 (no blur) | σ=1.4 | σ=2.8 |
|---|---|---|---|
| ρ=0.5 | 0.4860 | 0.5014 | 0.4659 |
| ρ=1.0 | 0.4996 | 0.5526 | 0.5759 |
| ρ=2.0 | 0.5047 | 0.5634 | **0.5794** |

Best val config ρ=2.0/σ=2.8 → test **0.5976 / 0.6218 / 0.5378**.
Sensitivity is large (≈ +0.08 ODS from the untuned ρ≤1/no-blur row to the
tuned one), so any comparison against Haralick must state the configuration.
GDSD v2 at σ=2.8 beats even the tuned Haralick on both splits
(+0.009 ODS test; +0.009 val).

**Direction quantization (oct_zero).**  `haralick_facet.py` also accepts
`direction="octant"`: quantize the derivative direction to the nearest of
8 octants (GDSD-style) before evaluating f''/f'''.  This is the
configuration that ranks first in the internal rank-normalized pipeline
(reported there as 0.6115 test ODS).  Under the official protocol used in
this repo the octant direction scores essentially the same as the
continuous direction (ρ=2.0/σ=2.8): val 0.5776 vs 0.5794, test 0.5962 vs
0.5976 — direction quantization is nearly free in both protocols,
consistent with the Rayleigh prediction (response ≈ convex combination of
principal curvatures; median φ ≈ 15°, so octant error is negligible).
GDSD v2 σ2.8 + zcnms remains above both Haralick configurations under the
official protocol (+0.014 over oct_zero, +0.013 over continuous).

**Thinning fairness (zc_nms on Haralick).**  `haralick_facet.py` also
accepts `thinning="zc_nms"` — the same candidate-only NMS as GDSD's zcnms,
applied to the Haralick analytic-rule edge set.  This is the fairness
check: if thinning helps generic baselines, comparisons must give every
method the same post-processing.  It helps Haralick a little (val
0.5794→0.5816, test 0.5976→0.5994) but does not change the ordering —
GDSD v2 σ2.8 + zcnms still leads the equally-thinned Haralick by
+0.010 val / +0.011 test ODS.  All GDSD-vs-Haralick comparisons below use
the equally-thinned Haralick row when post-processing is the point.

**Rank normalization zero-mapping.**  `rank_norm_soft.py` maps zero pixels
to exactly zero and ranks only the nonzero candidates (a previous version
ranked all pixels including zeros, giving background a nonzero value that
destroyed precision at low thresholds — tuned Haralick under that buggy
transform scored val AP 0.4475; with the fix, val 0.5908 / 0.6328 / 0.5537
and test 0.6032 / 0.6341 / 0.5571).  Two readings: (1) for cross-protocol
AP comparisons, rank-normalized AP is only meaningful when zero stays zero;
(2) even with the corrected transform, rank normalization does not lift the
reference Haralick configuration to the 0.61 AP reported by the internal
pipeline — the remaining gap is attributable to that pipeline's matcher and
thinning conventions, not to the zero mapping.  Rank normalization is a
monotone per-image rescaling: it changes threshold-grid behaviour, never the
edge set or OIS ranking within an image.

## Experiment: GDSD v4 (hysteresis) — not better than v2 σ2.8 (2026-09-08)

**Hypothesis.**  v2 edges shatter into 1-px components; adding Canny-style
hysteresis (strong = ZC with a gradient-side neighbour above Tp; weak =
contrast crossings kept only when 8-connected to a strong pixel) would link
contours and raise precision.  Implemented as a soft-map post-filter in
[`benchmark/make_soft_hyst.py`](../benchmark/make_soft_hyst.py) — the
binary detector and the σ value are untouched; only the set of pixels that
receive a non-zero strength changes.

**Results (official pr_eval, 99 thr):**

| variant | val ODS | test ODS | test OIS | test AP |
|---|---|---|---|---|
| v2 σ=1.4 | 0.5724 | 0.5913 | 0.6180 | 0.4948 |
| v4 hyst σ=1.4 | 0.5881 | 0.5959 | 0.6068 | 0.3934 |
| v2 σ=2.8 | **0.5881** | **0.6070** | **0.6284** | **0.6085** |
| v4 hyst σ=2.8 | 0.5844 | 0.5834 | 0.5854 | 0.4191 |

**Verdict.**  Hysteresis helps at the *smaller* scale (+0.016 val ODS at
σ=1.4, recall preserved) but hurts at σ=2.8 (−0.024 test ODS): at the wide
scale the weak-ZC set is large and mostly unconnected, so hysteresis drops
real recall (R 0.569 vs 0.688 for v2 σ2.8).  v4 never beats v2 σ2.8, so it
is recorded as an experiment, not a release.

## Experiment: thinning (NMS on the ZC set) — zcnms confirmed on test (2026-09-08)

**Hypothesis.**  A step edge produces a ± doublet ~2.4 px apart; the ZC set
contains both flanks, capping precision.  Thinning along the gradient normal
(keep a ZC only when it is a local maximum of gm among ZC pixels / against
both normal neighbours) should remove the weaker flank.  Implemented in
[`benchmark/make_soft_thin.py`](../benchmark/make_soft_thin.py) (modes
`zcnms`, `gmnms`).  Same detector, same σ — ranking/post-processing only.

**Results (official pr_eval, 99 thr):**

| variant | val ODS | test ODS | test OIS | test AP |
|---|---|---|---|---|
| v2 σ=2.8 plain | 0.5881 | 0.6070 | 0.6284 | 0.6085 |
| **v2 σ=2.8 + zcnms** | **0.5916** | **0.6106** | **0.6318** | **0.6119** |
| v2 σ=2.8 + gmnms | 0.5870 | — (val worse, not run on test) | | |
| v2 σ=1.4 plain | 0.5724 | 0.5913 | 0.6180 | 0.4948 |
| v2 σ=1.4 + zcnms | 0.5770 | — | | |
| v2 σ=1.4 + gmnms | 0.5790 | — | | |

**Verdict.**  `zcnms` (suppress a ZC only when a *zero-crossing* neighbour
along the gradient normal has strictly higher gm) is the release candidate:
it improves every metric on val and test consistently (+0.0035 val /
+0.0036 test ODS).  `gmnms` (Canny-style suppression against any neighbour,
ZC or not) helps at σ=1.4 (+0.007 val) but hurts at σ=2.8 — at the wide
scale it removes too much real edge.  `zcnms` becomes the new headline GDSD
variant: v2 σ=2.8 + zcnms.

## Negative result: GDSD v3 (Gaussian-weighted LSQ) — rejected (2026-09-08)

**Hypothesis.**  Replacing the unweighted window least-squares fit with a
Gaussian-weighted fit (`w(r,c) = exp(-(r²+c²)/(2σ_w²))`,
`pinv_wls = (AᵀWA)⁻¹AᵀW`) would improve the local surface estimate and
therefore edge quality.  The rest of the pipeline was identical to v2
(`gm@ZC` soft map, official pr_eval).

**Setup.**  σ (blur) = 1.4, σ_w (fit weight) = 1.4 — same scale as the
blur so the window centre dominates.  Code:
`benchmark/gdsd_features_wls.cpp`.

**Provenance note.**  This run was extracted (2026-09-08 08:08) with the
pre-`ca2ad97` fixed 5×5 blur kernel — effective σ ≈ 1.16, as in the
pre-fix v1/v2 rows.  The comparison anchor below is therefore the pre-fix
v2 number (0.5917), which keeps the pair apples-to-apples; the σ-fix moved
v2 σ=1.4 by only +0.004 (0.5913), so the negative verdict is unaffected.
A post-fix v3 rerun was judged unnecessary for a rejected variant.

**Results (official pr_eval, 99 thr):**

| split | ODS | OIS | AP |
|---|---|---|---|
| val (100) | 0.5653 | 0.6158 | 0.4242 |
| test (200) | 0.5841 | 0.6120 | 0.4398 |

vs GDSD v2 on the same protocol: **−0.008 ODS test, −0.055 AP test**
(v2 = 0.5917 / 0.6176 / 0.4949).  The direction is consistent on val
(−0.003 ODS, −0.019 AP) and test, so it is not noise.

**Verdict.**  Rejected — WLS at σ_w = 1.4 does *not* improve over v2; it
is slightly worse.  v3 still beats v1 (+0.010 ODS test) and Canny
(+0.010), but the WLS change is not worth the extra complexity.  This is
consistent with the earlier WLS spike (weight *shape* moved F by
< 0.001): the ranking change in v2 (`gm@ZC`) addressed the actual
bottleneck (the second-derivative doublet), whereas the fit weighting
does not.  Only σ_w = 1.4 was benchmarked; a larger σ_w (2.0–2.5, the
value favoured in the earlier internal WLS experiment) was not run
because the effect direction was already clear and negative on both
splits.

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
# 2. soft maps (v1 |R|@ZC  OR  v2 gm@ZC) + official PNGs + pr_evaluation
python benchmark/make_soft.py feats/test soft/gdsd1/test test --strength response
python benchmark/make_soft_pngs_official.py soft/gdsd1/test soft_png/gdsd1/test
GDSD_BSDS_ROOT=. GDSD_PYBSDS_PATH=./py-bsds500 \
  python benchmark/run_official_pr.py gdsd1 test soft_png/gdsd1/test 99
```

See `benchmark/README.md` for the full workflow.
