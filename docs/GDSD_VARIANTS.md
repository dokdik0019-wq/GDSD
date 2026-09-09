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
| **v2** (gm@ZC) | strength value at the ZC | R is a *second* derivative → a step edge gives a ± doublet and R ≈ 0 **exactly at the edge**; ranking by \|R\| therefore ranks by "closeness to zero of the ranker itself" (meaningless); gradient magnitude peaks at the edge → correct ranking | strength = hypot(d,e) at the ZC | 0.5913 (σ1.4) | kept |
| **σ = 2.8** (on v2) | blur scale | after the kernel-size fix the σ knob is real; a wider Gaussian suppresses texture noise and human-annotated contours live at the coarser scale (the same scale the internal pipeline independently preferred on val) | kernel auto-sized from σ (commit `ca2ad97`) | **0.6070** | val-selected |
| **v3 (WLS)** | fit method (Gaussian-weighted LSQ) | a better local surface estimate gives better edges | `w(r,c) = exp(-(r²+c²)/2σ_w²)` inside pinv | 0.5841 (σ1.4) | rejected: worse than v2 (−0.008 ODS) |
| **v4 (hysteresis)** | post-process: link ZC pixels | v2 edges shatter into 1-px components; linking weak ZC to strong contours (8-connectivity) should complete the contour | strong/weak sets + connected components | 0.5959 (σ1.4) / 0.5834 (σ2.8) | rejected: never beats v2 σ2.8 |
| **zcnms thinning** (headline) | post-process: **drop** the weaker ZC flank | the doublet is ~2.4 px apart → the ZC set fires on **both flanks** of every edge (the human boundary lies between the lobes); keeping only the stronger gm flank raises precision without losing recall | keep a ZC only when it is a local max of gm **among ZC neighbours** along the gradient normal | **0.6106** | **release** |
| gmnms (Canny-style NMS) | post-process: aggressive ZC thinning | NMS against *any* neighbour (ZC or not) | keep ZC when gm ≥ both normal neighbours | 0.5790 (σ1.4 val) / 0.5870 (σ2.8 val) | rejected: too aggressive at σ2.8 |
| (reference) Haralick tuned | ρ=2.0, σ=2.8 | fair tuned baseline — the untuned 1984 row is not the comparison to beat | `benchmark/haralick_facet.py` with rho_max/sigma | 0.5976 | reference |
| **v5 (committee threshold)** | evaluation: per-image threshold from a GT-blind predictor | OIS (0.6318) is higher than ODS (0.6106) → a committee choosing each image's threshold should recover part of that gap (CHEV-style) | linear/bin3 predictor on soft-map stats (mean/p90 gm@ZC), frozen on val | 0.6038 mean-F1 (linear) | rejected: +0.003 only — ceiling is structural |
| (reference) Haralick oct_zero | Haralick zero rule + GDSD **octant direction** (ρ2.0/σ2.8) | the reference internal pipeline reports oct_zero as its top scorer; direction quantization should be ~free per the Rayleigh prediction | `haralick_facet.py direction="octant"` | 0.5962 | reference — ≈ continuous Haralick (0.5976) under official protocol |
| (reference) Haralick tuned + rank-norm | tuned Haralick, soft maps rank-normalized per image (zeros stay 0) | tests whether the internal pipeline's AP ≈ 0.61 is reproduced by rank normalization alone inside the official matcher | `benchmark/rank_norm_soft.py` (zero-mapping fixed, commit `62f3dbf`) | 0.6032 (AP 0.5571) | reference — rank-norm does **not** reach 0.61 AP in the official protocol; residual gap is the internal matcher/thinning |

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
| Haralick tuned + zc_nms thinning | 0.5994 | 0.6218 | 0.5375 | ~4 s |
| Haralick oct_zero (octant dir) | 0.5962 | 0.6255 | 0.5488 | ~4 s |
| GDSD v2 σ1.4 | 0.5913 | 0.6180 | 0.4948 | ~6 s |
| GDSD v1 σ1.4 | 0.5794 | 0.6055 | 0.3530 | ~6 s |
| Canny σ1.4 | 0.5740 | 0.6008 | 0.4866 | 15 s |

## Negative result: v5 (per-image committee threshold) — rejected (2026-09-08)

**Hypothesis (CHEV-style).**  OIS (0.6318 test) exceeds ODS (0.6106): the
per-image GT-optimal thresholds differ (std ≈ 0.07, range 0.03–0.55), so a
GT-blind committee predicting each image's threshold from soft-map
statistics should recover part of the OIS−ODS gap.

**Setup.**  Oracle analysis on val + test (`oracle_threshold_analysis.py`):
per-image optimal threshold + GT-blind features (n_zc, mean/med/p90
gm@ZC, edge density).  Predictors frozen on val only:
`committee_threshold.py` (linear) and `eval_adaptive_thresholds.py`
(global vs linear vs bin3 vs oracle, all under the *same* metric — mean
per-image F1, so ODS is not compared against a mean).

**Results (mean per-image F1, official matching):**

| strategy | val | test |
|---|---|---|
| global (fixed thr 0.15) | 0.5920 | 0.6009 |
| committee linear | 0.6010 | 0.6038 |
| bin3 committee | 0.5961 | 0.6030 |
| oracle (per-image GT-optimal) | 0.6304 | 0.6317 |

**Verdict.**  The oracle ceiling itself is small (OIS−ODS ≈ +0.03, test)
— this is *structural* for BSDS-type benchmarks (broad aligned optima,
monotone shifts cannot fix ranking errors, per-image optima are noisy,
precision is unobservable without GT).  A GT-blind predictor captures
~10% of that ceiling: committee linear +0.003, bin3 +0.002, and an
outlier-only rule (deviate only for the 4% of images needing high
thresholds) is bounded by ≤ +0.002.  All variants are inside the
per-image bootstrap noise of the benchmark — not worth the added
complexity or the protocol deviation (per-image thresholds are not the
standard ODS protocol).  Independent expert consult (Muse Spark)
corroborates: "if oracle OIS−ODS < 0.03, stop; achievable < 0.006".
**Rejected** — effort goes into soft-map ranking (local/spatial
adaptation such as zcnms), not per-image scalars.

## Visualizations (transparency artifacts)

All figures were produced by `benchmark/make_variants_viz.py` — an
independent re-evaluation with the official py-bsds500 matcher over the
val split (100 images, 99 global thresholds).  Every ODS/OIS/AP in the
CSV matches the benchmark tables in this repo exactly, which is the
consistency check that the reported numbers are reproducible rather than
hand-picked.

| Figure | What it shows |
|---|---|
| [`figures/pr_curves.png`](figures/pr_curves.png) | precision-recall curves of all 7 variants (v1, v2 σ1.4/σ2.8, v3 WLS, v4 σ1.4/σ2.8, v2+zcnms σ2.8) on val |
| [`figures/edge_overlay_101085.png`](figures/edge_overlay_101085.png) | example image: red = each variant's edges at its own ODS threshold, green = human ground truth |
| [`figures/edge_overlay_101087.png`](figures/edge_overlay_101087.png) | second example image, same layout |
| [`figures/variant_summary.csv`](figures/variant_summary.csv) | per-variant ODS/OIS/AP/best-thr/R/P (cross-check against tables above) |

## Release & versioning policy (2026-09-09)

Agreed in project review (2026-09-09): the release tag must
not embed benchmark numbers — numbers change as the protocol/baselines are
refined, and git tags are immutable.

- Current release: **`v2.0.0`** → https://github.com/dokdik0019-wq/GDSD/releases/tag/v2.0.0
- Benchmark numbers live in the **release notes** and in the docs tables
  of the repo, never in the tag name.
- An earlier tag `v2-zcnms-0.6106` (number in the name) was deleted and
  replaced by `v2.0.0`; the eval bundle asset (repo + py-bsds500 under
  `third_party/`, attribution in `BUNDLE_README.md`) is attached to the
  release for code-availability / reproducibility.
- Zenodo DOI can be minted from the release when the paper is submitted
  (GitHub-Zenodo integration archives the tag).

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
