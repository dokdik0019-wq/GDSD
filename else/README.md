# ELSE — Efficient Least-Squares Edge Detection (prior work)

ELSE is the **predecessor** of GDSD in the same research line: both methods
approximate the local intensity function by a 2D quadratic surface fitted
with least squares on a sliding neighborhood. GDSD then moved from the
first-order response to the second directional derivative (gradient-direction
second derivative) and to a 5x5 window with a zero-crossing decision rule.

This folder is a minimal, self-contained Python prototype matching the
published method:

> Sinlapakorn T, Limtrakul S, Wetweerapong J, Puphasuk P. Efficient image
> edge detection method using least squares and elbow technique.
> International Journal of Mathematics and Computer Science 2026;21(1):33–43.
> DOI: https://doi.org/10.69793/ijmcs/01.2026/pikul

## Method summary

1. Fit `S(x,y) = A x² + B y² + C x y + D x + E y + F` by least squares on each
   3x3 neighborhood (adjacent-pixel distance = 0.1 unit).
2. Edge response at the center pixel: `R = sqrt(D² + E²)` — the magnitude of
   the first-order gradient of the fitted surface.
3. **Single threshold**: pick the threshold from an *elbow* (knee) point of the
   graph of percentile position vs. response value
   (`elbow_threshold`), then keep `R >= R*`.
4. **Double threshold**: apply non-maximum suppression along the quantized
   gradient direction, find the elbow on the NMS-selected responses, then
   Canny-style hysteresis with high/low thresholds chosen 4 percentile steps
   apart.

The prototype differs from the paper only in implementation detail (edge
padding vs. border handling, exact elbow neighborhood step); the math and the
threshold-selection logic follow the paper.

## Files

| File | Purpose |
|---|---|
| `else_detector.py` | `else_single_threshold`, `else_double_threshold`, NMS + hysteresis helpers |
| `quadratic_fit.py` | shared 3x3 quadratic least-squares coefficient maps (`quadratic_coeff_maps`) |
| `utils.py` | grayscale load / uint8 save helpers |
| `demo.py` | CLI demo: `--image path --method else_single|else_double` |

## Run

```bash
pip install numpy scipy Pillow
python demo.py --image ../examples/sample_input.png --method else_single
python demo.py --image ../examples/sample_input.png --method else_double
```

Outputs are written to `output/` under this folder.

## Relation to GDSD

| | ELSE (this folder) | GDSD (repo root) |
|---|---|---|
| Fit window | 3x3, step 0.1 | 5x5, step 0.1 |
| Response | first-order `sqrt(D² + E²)` | second directional derivative `R = gradᵀ H grad` |
| Decision | threshold on R (elbow) | zero crossing along quantized gradient + percentile contrast |
| Variants | single / double (elbow + NMS + hysteresis) | v1 \|R\|@ZC, v2 gm@ZC soft maps |
| Published | IJ MCS 21(1) 2026, 33–43 | AMM 2026 proc. 449–458 (method); thesis Ch. 3 §3.2 |

The code here is the original prototype (May 2026) from
`~/Desktop/mythesis_2025/geometric_feature_detectors/`, kept as the lineage
reference. GDSD is the second-order successor evaluated on BSDS500 with the
official benchmark (see `docs/BENCHMARK.md`).
