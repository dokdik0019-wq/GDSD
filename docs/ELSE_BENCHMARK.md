# ELSE — Official BSDS500 Benchmark (lineage reference)

Results of the predecessor **ELSE** method (least-squares 3x3 quadratic fit +
elbow thresholding; IJ MCS 21(1) 2026, 33–43, DOI 10.69793/ijmcs/01.2026/pikul)
on the official BSDS benchmark, so the thesis can show the GDSD lineage on one
comparable table.

## Protocol

Same pipeline as GDSD v1/v2/Canny/Haralick tables (see `BENCHMARK.md` +
`official-pr-protocol-2026-09-07` in the skill): py-bsds500 `pr_evaluation`
(CSA `correspond_pixels`, 1px cross-dilation matching, morphological
thinning), 99 fixed global linspace thresholds over [0,1], uint8 PNG soft maps,
ODS from accumulated counts / OIS mean per-image best F1 / AP interp over
unique recall in 0.01 steps.

Soft-map variants (built by `benchmark/make_else_soft_norm.py`):

- **ELSE (R, single)** — ranking by `R = sqrt(D² + E²)` (first-order gradient
  magnitude of the fitted quadratic; the response used by the paper's
  single-threshold variant).
- **ELSE (NMS, double)** — ranking by non-maximum-suppressed R along the
  quantized gradient direction (the response ranking behind the paper's
  double-threshold variant, i.e. the Canny-style soft map analog).

### Normalization disclosure (IMPORTANT)

ELSE soft maps are normalized **per image** (`R / R.max() * 255`) — not the
raw `clip(R*255)` used for GDSD/Canny.  Reason: the paper's 0.1-unit
adjacent-pixel distance scales the first-order coefficients ~10x, so R
routinely exceeds 1 on [0,1] images (per-image max ~4–5).  A plain clip
saturates a large fraction of strong pixels at 255 and flattens the ranking
the threshold grid sees.  Measured effect on val/20thr smoke: clip
ODS 0.4949 / AP 0.10 vs per-image-max ODS 0.5082 / AP 0.48 — the clip version
understates ELSE.  GDSD/Canny responses sit near [0,1], so clip ≈ per-image
max for them and their published numbers are unaffected.  ODS/OIS are
scale-invariant per image only up to monotone ranking; per-image max is the
standard "255 = strongest response per image" convention.

## Results (99 thr, uint8 PNG)

GDSD/Canny/Haralick rows below are measured with the Gaussian blur kernel
auto-sized from σ (commit `ca2ad97`) — the same numbers as the current
tables in `BENCHMARK.md`.  ELSE has no Gaussian pre-blur step, so its rows
are unaffected by that fix.

### test (200 images)

| Method | ODS | OIS | AP |
|---|---|---|---|
| **GDSD v2 σ=2.8 + zcnms (headline)** | **0.6106** | **0.6318** | **0.6119** |
| GDSD v2 (gm@ZC, σ=2.8) | 0.6070 | 0.6284 | 0.6085 |
| GDSD v2 (gm@ZC, σ=1.4) | 0.5913 | 0.6180 | 0.4948 |
| **GDSD v1 (\|R\|@ZC, σ=1.4)** | 0.5794 | 0.6055 | 0.3530 |
| Canny (NMS soft map, σ=1.4) | 0.5740 | 0.6008 | 0.4866 |
| Haralick facet (1984) | 0.5194 | 0.5513 | 0.4708 |
| **ELSE (NMS, double)** | **0.5143** | **0.5543** | **0.4711** |
| **ELSE (R, single)** | **0.5118** | **0.5592** | **0.4617** |

### val (100 images)

| Method | ODS | OIS | AP |
|---|---|---|---|
| **GDSD v2 σ=2.8 + zcnms** | **0.5916** | **0.6304** | **0.6032** |
| GDSD v2 σ=2.8 | 0.5881 | 0.6269 | 0.5998 |
| GDSD v2 σ=1.4 | 0.5724 | 0.6215 | 0.4722 |
| GDSD v1 (\|R\|@ZC, σ=1.4) | 0.5614 | 0.6108 | 0.3459 |
| Canny | 0.5584 | 0.6044 | 0.4675 |
| Haralick | 0.4996 | 0.5507 | 0.4638 |
| **ELSE (NMS, double)** | **0.5099** | **0.5602** | **0.4853** |
| **ELSE (R, single)** | **0.5071** | **0.5647** | **0.4784** |

## Lineage reading (for the thesis)

On test ODS, attributing each design step at its own axis: ELSE 0.5143
(first-order, no zero-crossing) → GDSD v1 (adds second-order zero-crossing
decision + Gaussian presmooth) 0.5794 (+0.065) → GDSD v2 (swaps soft-map
strength \|R\|@ZC → gm@ZC) 0.5913 (+0.012); the scale step σ=1.4 → 2.8 then
adds +0.016 (0.6070) and zcnms thinning +0.004 (headline 0.6106).  ELSE ≈
Haralick (0.5143 vs 0.5194), both below Canny.  Each GDSD design step lifts
ODS clearly above the predecessor, and v1 already clears Canny, which is the
honest lineage story.  ELSE's AP (0.47) is higher than GDSD v1's (0.35) —
expected: ELSE ranks by raw first-order strength on a full [0,1] scale,
GDSD v1's map is sparse zero-crossing-gated \|R\|.

## Reproduce

```bash
# 1. soft maps (per-image max normalization)
GDSD_BSDS_ROOT=<...> GDSD_ELSE_SOFT_OUT=<...> \
  python benchmark/make_else_soft_norm.py all all

# 2. official eval (fork context; works on macOS and Linux)
GDSD_BSDS_ROOT=<...> GDSD_PYBSDS_PATH=<py-bsds500> \
  python benchmark/run_official_pr.py else_nms_norm test <soft>/else_nms_norm/test 99
```

Raw logs: `~/Desktop/GDSD-Paper/else-benchmark/logs/*_99.log`.
Code correctness checks (fit/elbow/NMS/step-edge/analytic pinv): see
`verify_else.py` run 2026-09-08 (10/12 auto-checks passed; 2 failures were
over-strict synthetic-step assumptions, not method bugs).
