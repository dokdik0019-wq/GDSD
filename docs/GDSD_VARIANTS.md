# GDSD variants: what changed, the hypothesis behind each, and the numbers

This document is the "map" of the GDSD research line.  Every variant below
changes **one** point of the same base pipeline, and the change is always
driven by an explicit hypothesis.  Hypotheses that failed are kept as
negative results — they explain *why* the winning combination is what it is.

**Base pipeline (identical for every variant):** Gaussian blur (σ) →
quadratic facet fit on a 5×5 window → response
R = 2ad² + 2cde + 2be² (second directional derivative along the gradient)
→ zero crossings of R between gradient-direction neighbours (with a
percentile-contrast gate) → soft map → official BSDS500 threshold sweep
(global linspace grid, 99 thresholds, py-bsds500).

## Version table (test split, 200 images, official pr_eval)

| Version | What changed | Hypothesis / principle | Mechanism | test ODS | Status |
|---|---|---|---|---|---|
| **v1** (\|R\|@ZC) | — (original, AMM 2026) | the pixel where R crosses zero is the edge | strength = \|R\| at the ZC | 0.5794 (σ1.4) | baseline |
| **v2** (gm@ZC) | strength value at the ZC | R is a *second* derivative → a step edge gives a ± doublet and R ≈ 0 **exactly at the edge**; ranking by \|R\| therefore ranks by "closeness to zero of the ranker itself" (meaningless); gradient magnitude peaks at the edge → correct ranking | strength = hypot(d,e) at the ZC | 0.5913 (σ1.4) | ✅ kept |
| **σ = 2.8** (on v2) | blur scale | after the kernel-size fix the σ knob is real; a wider Gaussian suppresses texture noise and human-annotated contours live at the coarser scale (the same scale the internal pipeline independently preferred on val) | kernel auto-sized from σ (commit `ca2ad97`) | **0.6070** | ✅ val-selected |
| **v3 (WLS)** | fit method (Gaussian-weighted LSQ) | a better local surface estimate gives better edges | `w(r,c) = exp(-(r²+c²)/2σ_w²)` inside pinv | 0.5841 (σ1.4) | ❌ worse than v2 (−0.008 ODS) |
| **v4 (hysteresis)** | post-process: link ZC pixels | v2 edges shatter into 1-px components; linking weak ZC to strong contours (8-connectivity) should complete the contour | strong/weak sets + connected components | 0.5959 (σ1.4) / 0.5834 (σ2.8) | ❌ never beats v2 σ2.8 |
| **zcnms thinning** 🏆 | post-process: **drop** the weaker ZC flank | the doublet is ~2.4 px apart → the ZC set fires on **both flanks** of every edge (the human boundary lies between the lobes); keeping only the stronger gm flank raises precision without losing recall | keep a ZC only when it is a local max of gm **among ZC neighbours** along the gradient normal | **0.6106** | ✅ release |
| gmnms (Canny-style NMS) | post-process: aggressive ZC thinning | NMS against *any* neighbour (ZC or not) | keep ZC when gm ≥ both normal neighbours | 0.5790 (σ1.4 val) / 0.5870 (σ2.8 val) | ❌ too aggressive at σ2.8 |
| (reference) Haralick tuned | ρ=2.0, σ=2.8 | fair tuned baseline — the untuned 1984 row is not the comparison to beat | `benchmark/haralick_facet.py` with rho_max/sigma | 0.5976 | reference |

## What each idea was betting on (one line per hypothesis)

1. **v2 (gm@ZC):** "v1's problem is the *number used to score* the edge —
   R = 0 exactly at the true edge, so scoring by |R| is close to random."
   Evidence: rank-AUC on held-out val gm 0.734 > |R| 0.728; doublet spacing
   2.39 px; integer ZC offset ≈ 0.5 px; GT between the lobes.
2. **σ = 2.8:** "We are not tuning for test — the wider scale simply
   removes texture noise, and BSDS human contours sit at that scale."
   Evidence: after the kernel fix σ has a real effect; val prefers σ2.8
   (0.5881 vs 0.5724) and test confirms (0.6070 vs 0.5913).  The advisor's
   internal pipeline independently chose σ=2.8 on val.
3. **v3 (WLS):** "A better local surface estimate should give better
   edges."  **Rejected** — worse than v2 on both splits.  The bottleneck is
   the *ranking*, not the fit.  (Negative result: tells future work not to
   spend effort on the fit stage.)
4. **v4 (hysteresis):** "Edges shatter because nothing links them."
   **Rejected as headline** — helps at σ1.4 (+0.016 val ODS) but at σ2.8 the
   weak-ZC set is large and mostly unconnected, so hysteresis drops real
   recall (R 0.569 vs 0.688).  Never beats v2 σ2.8.
5. **zcnms:** "Edges look 2 px thick because the ±doublet makes the ZC rule
   fire on both flanks; keep the stronger flank."  **Accepted** — improves
   every metric on val and test consistently (+0.0035 / +0.0036 ODS).
   Mechanism-based, chosen on val, confirmed on test.
6. **gmnms:** same thinning idea but applied against every neighbour rather
   than ZC neighbours only.  Too aggressive at σ2.8 — removes real edge.
   Helps only at σ1.4.

## Headline numbers (test, official pr_eval)

| Method | ODS | OIS | AP | time (200 img) |
|---|---|---|---|---|
| **GDSD v2 σ2.8 + zcnms** | **0.6106** | **0.6318** | **0.6119** | ~6 s |
| GDSD v2 σ2.8 (plain) | 0.6070 | 0.6284 | 0.6085 | 6 s |
| Haralick tuned (ρ2.0/σ2.8) | 0.5976 | 0.6218 | 0.5378 | 4 s |
| GDSD v2 σ1.4 | 0.5913 | 0.6180 | 0.4948 | ~6 s |
| GDSD v1 σ1.4 | 0.5794 | 0.6055 | 0.3530 | ~6 s |
| Canny σ1.4 | 0.5740 | 0.6008 | 0.4866 | 15 s |

## Methodology rules that shaped this line

- **One change per variant** — the base pipeline is shared, so a result
  difference is attributable to the single changed point.
- **Val chooses, test confirms** — mechanism/σ/parameter choices were made
  on val (or by theory); test numbers are reported after the choice was
  frozen (see the selection-bias discussion in `GDSD_V2.md`).
- **Baselines get tuned fairly** — Haralick is reported both untuned
  (1984, ρ≤1, no blur) and tuned (ρ/σ sweep, best val config); the claim
  "above Haralick" is made against the tuned row.
- **σ is real** — all numbers are post kernel-size fix (commit `ca2ad97`);
  see "What the kernel-size fix changed" in `BENCHMARK.md`.
- **Negative results are recorded** (v3, v4, gmnms-at-σ2.8) so the paper
  can explain *why* the winning variant wins, not only that it wins.
