# Stage A summary

**Goal**: load all four cubes + Brown v5 maps, compute the 8 default
parameters, validate against Brown 2020, produce per-cube parameter
maps, and surface anything that constrains downstream stages.

**Status**: complete (2026-04-21).

## What was done

1. Parsed the four `.lbl` files; geometry table at `00_cube_geometry.md`.
2. Loaded the four TER3 cubes + Brown v5 .npy maps; verified shapes
   and NaN distributions.
3. Computed VB 4 parameters (OLINDEX3, D2300, BD1900R2, BD2500_2)
   from scratch using Viviano-Beck 2014 formulae (`code/vb_parameters.py`).
4. Built per-cube 8-param feature stacks (Brown 4 + VB 4) at
   `outputs/feature_stacks/<cube>_stackA_8param.{npz,parquet}`.
5. Reproduced Brown 2020 centroid distribution on FRT00003E12 as a
   validation checkpoint - **PASSED** (mode 1.166 um, median 1.290,
   p95 1.386).
6. Produced per-cube parameter maps (display-flipped) +
   per-cube valid-pixel maps for A4FC.

## Headline findings

| cube | center lat/lon | n bands | valid pixels (Brown-strict) | valid % |
|---|---|---:|---:|---:|
| FRT00003E12 | 21.04N, 76.94E | 545 | 237 414 | 77.3 |
| FRT0000871C | 21.10N, 77.05E | 545 | 155 973 | 50.8 |
| FRT0000A4FC | 21.13N, 76.94E | 545 | **25 476** | **8.3** |
| FRT0000B438 | 21.19N, 76.95E | 545 | 120 812 | 39.3 |

- **A4FC valid-pixel geometry**: an arc-blob at one edge of the frame,
  not a uniform sample. v1 analyses had to use scaled HDBSCAN
  parameters; v2 dropped the Brown-strict mask and recovered ~287k
  valid pixels (full frame).
- **Brown 2020 reproduction PASSED** on 3E12 - the v5 .npy files match
  Brown's published centroid distribution within tolerance.
- **Cross-cube centroid heterogeneity**: medians vary by ~90 nm cube-
  to-cube; 871C is distributionally distinct (sharply peaked) compared
  to 3E12's broader bimodal distribution. This is real (not
  atmospheric - 871C has the lowest residuals despite the most
  blueshift), and it foreshadowed Stage C's chi-square result.
- **BD2500_2 column striping** - flagged as a known limitation.
  Doesn't break analysis but creates visible vertical artefacts in
  spatial maps. Stage D-equivalent work could try a stripe-corrected
  variant.

## AV-flagged decisions taken into Stage B

1. Centroid heterogeneity = working finding against Kremer's airfall.
2. Keep A4FC despite small valid fraction; scale per-cube parameters
   accordingly.
3. Add BD1900R, BD2100_2, MIN2295_2480 to the deferred Stage D
   candidate list.

## Outputs

- `outputs/feature_stacks/<cube>_stackA_8param.{npz,parquet}` (4 cubes)
- `outputs/figures/<cube>_stageA_*` (per-cube parameter maps,
  validation plots)
- `outputs/figures/FRT0000A4FC_valid_pixel_map.png`
- `outputs/reports/00_cube_geometry.md`
- `outputs/reports/01_stageA_validation.md` + `_note.md`
- `outputs/logs/<cube>_stackA_*.json` (per-cube run logs with md5s)

## What's NOT in Stage A

- 11-param VB extension (BD530_2, BD920_2, BDI1000VIS, LCPINDEX2,
  BD1400, BD2290, BD2355) - moved to Stage B.
- Any clustering or dimensionality reduction - moved to Stage B.
- Cross-cube statistical comparison - moved to Stage C.
