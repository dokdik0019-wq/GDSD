# GDSD v2 — soft map with gradient-magnitude strength at zero crossings (gm@ZC)

This document describes the **GDSD v2** soft edge map used in the BSDS500
tables, how it differs from the original GDSD soft map, and the evidence
behind the change.

## The two soft maps

GDSD's *detector* (blur → quadratic fit → response `R` → zero-crossing
mask) is **identical** in v1 and v2.  What changed is the *soft edge map*:
the real-valued strength assigned to each zero-crossing pixel when a
threshold is swept to build a precision-recall curve.

The GDSD response is the second directional derivative along the
gradient, scaled by the squared gradient magnitude:

```
R = 2 a d² + 2 c d e + 2 b e²        (a, b, c = quadratic fit coefficients,
                                      d, e = gradient components)
```

| Variant | Soft map strength at zero-crossing pixels | Origin |
|---|---|---|
| **v1 (|R|@ZC)** | `abs(R)` | original GDSD (AMM 2026 paper, thesis) |
| **v2 (gm@ZC)** | `hypot(d, e)` — the gradient magnitude | this work (2026) |

Both maps have the *same* support: the zero-crossing pixels.  They differ
only in the value carried at those pixels, which controls the ranking of
candidate edges when the map is thresholded.

Python API (`gdsd.soft_edge_map`):

```python
soft_v1 = soft_edge_map(image, strength="response")            # abs(R) @ ZC
soft_v2 = soft_edge_map(image, strength="gradient_magnitude")  # hypot(d,e) @ ZC (default)
```

## Why the change

An empirical diagnosis of the zero-crossing geometry on BSDS500
(200 test images, σ = 2.8) found that `R` is a *second* derivative, so a
single step edge produces a **+ / − doublet**:

- a real edge yields **two** zero crossings ~2.4 px apart (double-edge
  distance 2.39 px), instead of one thin contour;
- the integer zero crossing is offset **~0.5 px** from the true `R = 0`
  root;
- the human ground-truth contour lies **between the two |R| lobes**, not
  at the crossing itself (single-lobe variants that keep only one side of
  the doublet do *not* improve precision — GT matches either side).

Consequence for the soft map: ranking zero-crossing pixels by `abs(R)`
measures how close the *response doublet* is, whereas the quantity that
best reflects "is there an edge here" is the **gradient magnitude** at the
crossing.  Using `gm = hypot(d, e)` as the strength re-ranks the same
candidates by local contrast, which aligns better with the human contour.

This is a *soft-map* improvement: the binary edge map produced by
`detect_edges` is unchanged.

## Effect on the official benchmark (BSDS500, py-bsds500)

| Method (test, 200 images) | ODS | OIS | AP |
|---|---|---|---|
| **GDSD v2 (gm@ZC, σ=1.4)** | **0.5917** | **0.6176** | **0.4949** |
| GDSD v1 (|R|@ZC, σ=1.4) | 0.5743 | 0.6014 | 0.3142 |
| Canny (NMS soft map) | 0.5740 | 0.6008 | 0.4866 |
| Haralick facet (1984) | 0.5194 | 0.5513 | 0.4708 |

- **ODS +0.017** and **AP +0.18** over v1 on test (ΔODS +0.012 on val).
- With v1, GDSD sits at the same ODS level as Canny; with v2 it is
  clearly above Canny (+0.018 ODS test) and far above Haralick.
- σ = 1.4 and σ = 2.8 give essentially identical test numbers under the
  global-threshold protocol (val slightly prefers 2.8).

## Provenance and lineage

GDSD v2 is not a new detector family; it is an evaluation-focused
refinement of the authors' own GDSD method:

1. **Facet / polynomial-fit lineage** — fitting a local polynomial
   surface and taking derivatives of the fit (instead of finite
   differences) follows the facet-model tradition:
   Haralick (1984) detects edges at the zero crossing of the second
   directional derivative of a fitted surface; the quadratic-least-squares
   kernels used here are the standard 2-D Savitzky–Golay / facet kernels.
2. **GDSD itself** — the response `R = gradᵀ H grad` and the
   zero-crossing + percentile-contrast decision rule are the original
   method (AMM 2026 paper; doctoral thesis Ch. 3 §3.2).
3. **The v2 change** — using the gradient magnitude at zero crossings as
   the soft-map strength is motivated by the doublet diagnosis summarized
   above (full report: `GDSD_V2_ZC_DIAGNOSIS.md` in the research
   workspace; scripts in `benchmark/`).  The same diagnosis also explains
   why ranking by `abs(R)` under-reports precision for a second-order
   detector.

## References

- R. M. Haralick, "Digital step edges from zero crossing of second
  directional derivatives," IEEE Trans. Pattern Anal. Mach. Intell.,
  PAMI-6(1):58–68, 1984.
- T. Sinlapakorn, P. Puphasuk, J. Wetweerapong, "Edge detection using the
  gradient-direction second derivative of a local 2D quadratic
  approximation," Proc. 30th Annual Meeting in Mathematics and
  Conference in Number Theory and Applications 2026, pp. 449–458.
- The method is also described in the author's doctoral thesis,
  Chapter 3, Section 3.2.
- Benchmark protocol: P. Arbelaez et al., "Contour detection and
  hierarchical image segmentation," IEEE TPAMI 33(5):898–916, 2011;
  evaluated with the official BSDS500 suite (py-bsds500 port).

## Reproducing

See [`benchmark/README.md`](../benchmark/README.md) and
[`BENCHMARK.md`](BENCHMARK.md):

```bash
python benchmark/make_soft.py feats/test soft/gdsd2/test test --strength gradient_magnitude
python benchmark/make_soft_pngs_u8.py soft/gdsd2/test soft_png/gdsd2/test
GDSD_BSDS_ROOT=. GDSD_PYBSDS_PATH=./py-bsds500 \
  python benchmark/run_official_pr.py gdsd2 test soft_png/gdsd2/test 99
```
