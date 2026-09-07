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
| `bsds_official_eval.py` | Runs the official evaluation for one method/split |

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

# 4. run official evaluation (99 global thresholds)
GDSD_SOFT_ROOT=soft python benchmark/bsds_official_eval.py gdsd2 test 99
```

Each method/split takes roughly 1 h (val) to 2.5 h (test) on a 12-core
machine.

## Soft map definitions

- **GDSD v1**: strength = `|R|` at zero-crossing pixels (matches the
  AMM 2026 paper / thesis soft map).
- **GDSD v2**: strength = gradient magnitude `gm = hypot(d,e)` at
  zero-crossing pixels.  The binary detection is unchanged; only the
  soft-map ranking used for threshold curves changes.

A pure-Python soft map is also available as `gdsd.soft_edge_map(image,
strength=...)`; the C++ path above is the reference implementation used
for the reported numbers.
