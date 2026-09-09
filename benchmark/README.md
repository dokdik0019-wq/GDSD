# GDSD BSDS500 benchmark harness

This directory contains the code used to produce the BSDS500 numbers in
`docs/BENCHMARK.md` and the README.  The evaluation follows the official
BSDS protocol (py-bsds500: `correspond_pixels` CSA matching, per-human
ground-truth matching, morphological thinning, global thresholds).

## Layout

| File | Purpose |
|---|---|
| `gdsd_features_cli.cpp` | C++ feature extractor (uses `../src/cpp/gdsd_cpp.hpp` core); writes `.response/.d/.e/.zc.f64` per image |
| `make_soft.py` | Builds the GDSD soft edge map from features (`--strength response` = v1, `gradient_magnitude` = v2) |
| `make_softmaps.py` | Builds the classical baseline soft maps (sobel / canny / LoG) at raw scale — **no per-image normalization** |
| `make_soft_pngs_u8.py` | Converts float soft maps to uint8 PNG (x255, clip) — the official `imread/255` convention |
| `run_official_pr.py` | Runs the official py-bsds500 `pr_evaluation` pipeline (global linspace thresholds, interp AP) on soft-map PNGs |
| `bsds_official_eval.py` | *(legacy)* percentile-threshold variant; kept for reference |

## Prerequisites (not vendored)

- **BSDS500 data**: `data/images/{train,val,test}/*.jpg` and
  `data/groundTruth/{train,val,test}/*.mat` under this directory (or set
  `GDSD_BSDS_ROOT` to a directory that has them).
- **py-bsds500**: clone into `./py-bsds500` (or set `GDSD_PYBSDS_PATH`).
- OpenCV dev headers + a C++17 compiler for the feature extractor.

The data and py-bsds500 are third-party and are not committed to the
repository.

## Workflow

```bash
# 1. build the feature extractor
make build-benchmark            # -> benchmark/build/gdsd_features_cli

# 2. extract features for a split (σ=1.4, gate=0.01)
for f in data/images/test/*.jpg; do
  id=$(basename "$f" .jpg)
  [ -f feats/test/$id.response.f64 ] || \
    benchmark/build/gdsd_features_cli "$f" feats/test/$id 1.4 0.01
done

# 3. build soft maps (v1 |R|@ZC  OR  v2 gm@ZC)
python benchmark/make_soft.py feats/test soft/gdsd1/test test --strength response
python benchmark/make_soft.py feats/test soft/gdsd2/test test --strength gradient_magnitude

# 4. convert to official uint8 PNGs (raw x255 + clip — NOT global-max normalized;
#    per-image or global-max rescaling degenerates the fixed [0,1] threshold grid)
python benchmark/make_soft_pngs_u8.py soft/gdsd1/test soft_png/gdsd1/test
python benchmark/make_soft_pngs_u8.py soft/gdsd2/test soft_png/gdsd2/test

# (classical baselines instead: make_softmaps.py canny test soft/canny/test
#  -> then the same make_soft_pngs_u8.py conversion)

# 5. run official pr_evaluation (global linspace thresholds over [0,1], 99 thr)
GDSD_BSDS_ROOT=. GDSD_PYBSDS_PATH=./py-bsds500 \
  python benchmark/run_official_pr.py gdsd2 test soft_png/gdsd2/test 99
```

Each method/split takes roughly 1 h (val) to 2.5 h (test) on a 12-core
machine.  Run several methods in sequence with a wrapper script, or
parallelize the `run_official_pr.py` invocations (each uses 10 workers).

## Official protocol notes

The reported numbers use the **official BSDS evaluation pipeline**
(py-bsds500 `pr_evaluation`), which mirrors the MATLAB suite:

- soft maps stored as **uint8 PNG** with raw x255 + clip (the official
  `imread()/255` convention); heavy-tailed maps that exceed [0,1] natively
  (ELSE) use per-image max instead — see `docs/ELSE_BENCHMARK.md`;
- a **fixed global threshold grid** `linspace(1/(N+1), 1-1/(N+1), N)` over
  [0,1] — not per-method percentiles;
- ODS = best F1 from counts accumulated over all images at the shared grid;
- OIS = mean of per-image best F1;
- AP = area under the precision-recall curve, interpolated over recall in
  0.01 steps (as in the official suite).

## Soft map definitions

- **GDSD v1**: strength = `|R|` at zero-crossing pixels (matches the
  AMM 2026 paper / thesis soft map).
- **GDSD v2**: strength = gradient magnitude `gm = hypot(d,e)` at
  zero-crossing pixels.  The binary detection is unchanged; only the
  soft-map ranking used for threshold curves changes.

A pure-Python soft map is also available as `gdsd.soft_edge_map(image,
strength=...)`; the C++ path above is the reference implementation used
for the reported numbers.
