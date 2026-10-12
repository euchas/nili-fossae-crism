# v8 summary - MTR3 7-param GMM (5 VB + Brown centroid + asymmetry)

**Goal**: test whether re-introducing the two physically-interpretable
Brown band-fit parameters (centroid -> Fo number, asymmetry -> grain
size / saturation) into the v5-B alteration basis sharpens the
mineralogical class boundaries vs the "alteration only" v7-C run.

**Status**: complete (2026-04-29).

## Stack

| param | source | role |
|---|---|---|
| centroid | Brown v6 fit | composition (Fo number proxy) |
| asymmetry | Brown v6 fit | grain size / band saturation |
| OLINDEX3 | VB 2014 | olivine band depth |
| BD1400 | VB 2014 | OH overtone |
| BD1900R2 | VB 2014 | water/hydration |
| D2300 | VB 2014 | Fe/Mg-OH |
| BD2500_2 | VB 2014 | Mg-carbonate |

Brown maps loaded full-range from v6 .npy (no Brown 2020 Eq.2 gating
per project policy 2026-04-29). Pixel valid if all 7 params finite.

## Per-cube headline metrics

| cube | shape | valid pixels | cum-3 | cum-5 | wall |
|---|---|---:|---:|---:|---:|
| FRT00003E12 | 828 x 843 | 472,865 (67.7%) | 89.8% | 98.3% | ~1 min |
| FRT0000871C | 761 x 826 | 435,164 (69.2%) | 84.7% | 96.6% | ~1 min |
| FRT0000A4FC | 780 x 824 | 434,182 (67.6%) | 88.5% | 97.4% | ~1 min |
| FRT0000B438 | 823 x 830 | 467,661 (68.5%) | 88.3% | 97.1% | ~1 min |

PCA per-axis variance (rounded):

| cube | PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FRT00003E12 | 46.5% | 29.0% | 14.3% |  5.5% |  3.0% |  1.4% |  0.3% |
| FRT0000871C | 39.0% | 31.3% | 14.4% |  7.1% |  4.8% |  2.9% |  0.5% |
| FRT0000A4FC | 42.3% | 33.9% | 12.4% |  5.1% |  3.8% |  2.1% |  0.5% |
| FRT0000B438 | 48.3% | 26.4% | 13.7% |  4.8% |  3.9% |  2.4% |  0.5% |

PC1 is dominant (39-48%) — driven by olivine signal (centroid +
asymmetry + OLINDEX3 colinearity). PC2 captures the alteration
suite. PC1+PC2+PC3 covers 84-90% of variance; PC6 + PC7 carry
< 1.5% combined and are essentially noise axes.

## GMM cluster sizes (raw, per cube)

| cube | c0 | c1 | c2 | c3 | c4 | c5 |
|---|---:|---:|---:|---:|---:|---:|
| FRT00003E12 |  82714 | 106228 |  71479 |  31017 |  86734 |  94693 |
| FRT0000871C |  93613 |  27207 |  85017 |  28995 | 111099 |  89233 |
| FRT0000A4FC | 116379 |  15326 | 126114 |  48817 |  30233 |  97313 |
| FRT0000B438 | 123933 | 114064 |  48536 |  40375 |  56582 |  84171 |

Hungarian-aligned labels in `<cube>_v8_GMM_k6_aligned.tif`. All
clusters non-trivially populated (smallest 15326 on A4FC).

## v8 vs v7-C

Same cubes, same VB-parameter implementation, same GMM hyperparameters,
same Hungarian-alignment convention. The only difference is the input
basis:

- **v7-C** (5 params): OLINDEX3 + BD1400 + BD1900R2 + D2300 + BD2500_2
  -- "alteration only", Brown deliberately excluded.
- **v8** (7 params): same 5 + centroid + asymmetry. Brown band-fit
  parameters re-included to test AV's hypothesis that they sharpen
  the mineralogical class boundaries.

The 7-param basis spreads more variance into PC2 (now 26-34% vs v7-C's
24-32%) because centroid and asymmetry both correlate strongly with
OLINDEX3 along PC1. PC3-4 carry the alteration-vs-olivine
discrimination that v7-C had on PC1-2.

Compare the two side-by-side in ArcGIS to judge whether the v8
clustering aligns better with morphology / mineralogy than v7-C does.
Both sets of GeoTIFFs share the same coordinate system (MRO Mars
Equirectangular IAU 2000, 18 m/pixel) so they overlay directly.

## Outputs

```
outputs/_v8_mtr3_with_brown/
  <cube>/
    arrays/        6 .npy: param_stack (H,W,7), pixel_valid_mask,
                   PC_scores raw + whitened, GMM raw + aligned labels
    geotiffs/      16 GeoTIFFs: 7 PC raw + 7 PC whitened + 2 GMM (raw + aligned)
    logs/          1 v8 run log per cube
  reports/
    03_v8_alignment.json      Hungarian permutation + distance matrix
    v8_run_log_<TS>.json      top-level v8 run log
```

Total: 88 GeoTIFFs across 4 cubes (16 per cube + cross-cube alignment).

## Project policy: no Brown thresholding (2026-04-29)

Per AV 2026-04-29: from v8 onward, no output should gate on the
Brown 2020 Eq.2 quality threshold. Specifically:
- v6 Brown .npy / .tif: full-range, no clip (already this way)
- v7-B Brown RGB GeoTIFFs: alpha = finite(Brown 4), NOT eq.2 mask
  (re-rendered 2026-04-29 with full-range alpha)
- v8 GMM: pixel valid if all 7 params finite, no eq.2 gate
- All v8 GeoTIFFs: no Eq.2 transparency, full pixel range

The Brown Eq.2 mask is still saved in `<cube>_mask_map_v6.npy` for
diagnostic use, but it is not used as a downstream filter.

## Pointers

- Snapshot: `outputs/_v8_mtr3_with_brown/`
- Decisions log: `DECISIONS.md` (v8 entries + project-wide policy)
- Sister stage: `outputs/_v7_mtr3_map_projected/` (5-param pure-
  alteration version, also full-range)
- Brown RGB GeoTIFFs: `outputs/_v7_mtr3_map_projected/<cube>/brown/`
  (re-rendered with full-range alpha)
