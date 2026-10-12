# v9 summary - 3E12-seeded GMM (cross-cube hard prediction)

**Goal**: replace v7-C / v8 per-cube fit + Hungarian alignment with a
single GMM trained on FRT00003E12 and applied unchanged to the other
three cubes. Pixels that don't match any 3E12-defined class get
flagged "novel" rather than forced into the nearest cluster. By
construction, the same six class colors mean the same Gaussian on
every cube.

**Status**: complete (2026-04-29). Both runs produced. **Result is a
substantive negative finding** rather than the cleaner cross-cube map
the plan anticipated - see Section 4. Hardware: 350s wall total
(v9-A 183s, v9-B 167s).

## 1. Two parallel runs

| run | basis | dim | chi^2 critical (95%, df=d) |
|---|---|---:|---:|
| v9-A (control) | OLINDEX3, BD1400, BD1900R2, D2300, BD2500_2 | 5 | 11.07 |
| v9-B (primary) | v9-A + Brown centroid + asymmetry | 7 | 14.07 |

Both runs use the same pipeline. Differences in outcomes between v9-A
and v9-B trace to whether or not the two Brown band-fit parameters
are in the basis.

## 2. 3E12 reference fit

Fit on 472,883 (v9-A) / 472,865 (v9-B) valid pixels. Standardized,
PCA(whiten=True), then GaussianMixture(k=6, full covariance,
n_init=10, random_state=42, reg_covar=1e-5). All three transformations
persisted to `outputs/_v9_3e12_seeded/reference/3e12_v9{a,b}_{scaler,pca,gmm}.joblib`.

PCA per-axis variance (3E12 only):

| run | PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v9-A | 40.6% | 31.8% | 19.8% | 5.2% | 2.6% | - | - |
| v9-B | 46.5% | 29.0% | 14.3% | 5.5% | 3.0% | 1.4% | 0.3% |

GMM clusters were reordered by mafic-cap z-sum on the standardized
parameter space (sum of OLINDEX3, BD1900R2, D2300, BD2500_2 means).
Cluster 0 has the lowest alteration index (most mafic-cap-like),
cluster 5 the highest. The reordered means are persisted in the
saved GMM, so all downstream label outputs use cluster 0 = least
altered, 5 = most altered.

GMM cluster sizes on 3E12 after reordering:

| run | c0 | c1 | c2 | c3 | c4 | c5 |
|---|---:|---:|---:|---:|---:|---:|
| v9-A | 105066 | 129374 | 93797 | 52304 | 73898 | 18444 |
| v9-B | 114743 | 117040 | 104934 | 18720 | 77034 | 40394 |

## 3. Novelty rates per cube

Threshold rules: a pixel is novel if EITHER `Mahalanobis^2 > chi^2_{0.95, d}`
OR `max_posterior < 0.6`. Both are tunable post-hoc using the saved
`mahalanobis2.npy` and `max_posterior.npy`.

### v9-A (5 params)

| cube | n_valid | m^2 median | m^2 p95 | m^2>cv | p_max median | p<0.6 | novel |
|---|---:|---:|---:|---:|---:|---:|---:|
| FRT00003E12 | 472883 | 3.95 | 10.5 | **3.98%** | 0.822 | 20.70% | 23.72% |
| FRT0000871C | 435164 | 16.22 | 23.7 | 89.17% | 0.766 | 25.50% | 90.73% |
| FRT0000A4FC | 434206 | 21.51 | 32.8 | 97.15% | 0.754 | 27.70% | 97.80% |
| FRT0000B438 | 467849 | 30.05 | 47.1 | 99.04% | 0.991 | 5.09% | 99.07% |

### v9-B (7 params)

| cube | n_valid | m^2 median | m^2 p95 | m^2>cv | p_max median | p<0.6 | novel |
|---|---:|---:|---:|---:|---:|---:|---:|
| FRT00003E12 | 472865 | 6.02 | 14.1 | **5.11%** | 0.947 | 9.30% | 13.47% |
| FRT0000871C | 435164 | 18.54 | 27.1 | 85.05% | 0.944 | 10.26% | 85.95% |
| FRT0000A4FC | 434182 | 19.56 | 33.2 | 82.04% | 1.000 | 4.10% | 82.63% |
| FRT0000B438 | 467661 | 35.11 | 53.9 | 99.51% | 0.992 | 5.44% | 99.54% |

The chi^2 calibration is correct on 3E12 (3.98% v9-A and 5.11% v9-B,
both close to the expected 5%). The Mahalanobis^2 implementation is
not buggy. The novelty rates on the other cubes are real.

## 4. The negative finding

The plan anticipated ~5% novel on 3E12 and modest novel rates on the
other cubes (with 871C / A4FC / B438 cube-specific lithologies falling
out as coherent novel sub-populations). What we got instead:

- **3E12 itself is 13-24% novel.** Driven by `p_max < 0.6`, not by chi^2.
  In whitened 5- to 7-D PC space with k=6 full-covariance Gaussians, a
  large fraction of pixels naturally sit at cluster boundaries with
  posteriors split 0.4 / 0.4 / etc. The 0.6 threshold is too aggressive
  for this geometry. v9-B's higher dimensionality reduces the
  boundary-pixel population (9.3% < 0.6 vs v9-A's 20.7%) because there's
  more room for the clusters to spread out.
- **Non-reference cubes are 83-99.5% novel.** Driven primarily by
  `m^2 > chi^2_{0.95}`. Median m^2 on non-ref cubes is 16-35; well
  beyond the 11-14 critical values. This is not just a few outliers -
  the BULK of non-ref pixels are Mahalanobis-implausible under any
  3E12 cluster.
- **B438 is the worst.** 99% novel in both runs. v9-B is *worse* than
  v9-A on B438 (99.54% vs 99.07%) because adding Brown centroid +
  asymmetry adds two more cube-shifted dimensions.

### Why: the standardization gap

When 3E12's `StandardScaler` is applied to the other cubes, the
non-reference parameter distributions are systematically shifted off
zero. Standardized cube means (each column = parameter mean after
3E12-fit `(x - mu_3e12) / sigma_3e12`):

v9-A:

| cube | OLINDEX3 | BD1400 | BD1900R2 | D2300 | BD2500_2 |
|---|---:|---:|---:|---:|---:|
| 3E12 | +0.000 | +0.000 | +0.000 | +0.000 | -0.000 |
| 871C | -2.05 | -2.73 | +0.15 | -0.42 | +0.88 |
| A4FC | -1.61 | -3.29 | +0.63 | -0.04 | +1.55 |
| B438 | -1.23 | **-3.62** | **+2.23** | -0.20 | +1.72 |

v9-B (same 5 columns plus):

| cube | centroid | asymmetry |
|---|---:|---:|
| 3E12 | -0.000 | +0.000 |
| 871C | -0.97 | -1.03 |
| A4FC | -2.15 | -1.72 |
| B438 | -1.03 | -1.05 |

BD1400 in particular shifts -2.7 to -3.6 sigma across all three non-ref
cubes. That is large. Mean BD1400 on 3E12 is 0.0179 with scale 0.0052;
on non-ref cubes the mean is 0.0179 - 3.0 x 0.0052 ~ 0.002. An order
of magnitude weaker. BD1400 sits at the 1.4 micron OH overtone, which
is immediately adjacent to atmospheric water-vapor bands; small
differences in CAT/VOLCAT residuals between observations produce
exactly this kind of systematic per-cube offset.

OLINDEX3 also shifts -1 to -2 sigma. BD2500_2 shifts +1 to +1.7 sigma.
These are calibration / atmospheric residual differences that cannot
be distinguished from real mineralogical differences with the
information v9 has.

### Why: warm-start drift

For each non-reference cube, the warm-start sanity pass refit the GMM
with 3E12's parameters as init. If the cluster structure transferred
cleanly, the means should drift by less than 0.5 sigma_within_3e12
(plan's threshold). They don't:

| run | cube | min drift (sigma) | max drift (sigma) | flagged classes |
|---|---|---:|---:|---|
| v9-A | 871C | 2.87 | 4.66 | all 6 |
| v9-A | A4FC | 3.16 | 6.40 | all 6 |
| v9-A | B438 | **3.86** | **9.44** | all 6 |
| v9-B | 871C | 2.19 | 4.55 | all 6 |
| v9-B | A4FC | 3.65 | 6.84 | all 6 |
| v9-B | B438 | **4.36** | **9.68** | all 6 |

Every class on every non-ref cube drifts by more than the plan's flag
threshold of 0.5 sigma - by factors of 4 to 20. This is the cleanest
signal that 3E12's cluster structure is genuinely a poor fit for the
non-reference cubes; it is not "the same clusters with a few extra
boundary pixels" but a different distribution.

### Why: argmax pile-up on c3

Before the novelty filter is applied, on the non-ref cubes 60-95% of
the argmax-assigned pixels collapse onto a single cluster (c3 in both
runs):

- v9-A 871C: c3 = 109,643 of 222,541 argmax-labeled (49%)
- v9-A A4FC: c3 = 200,464 of 433k (46%)
- v9-A B438: c3 = **371,291 of 467k (79%)**
- v9-B A4FC: c3 = **353,054 of 434k (81%)**
- v9-B B438: c3 = 278,562 of 467k (60%)

On 3E12, c3 holds 18-52k pixels (4-11% of valid). On non-ref cubes it
becomes the dominant attractor by argmax. Then the chi^2 filter wipes
most of that pile, leaving the visible novel-pixel field. This is the
expected GMM failure mode for out-of-distribution input: pixels far
from any cluster center get assigned to whichever cluster is nominally
"closest" but the assignment is meaningless.

## 5. Outputs

```
outputs/_v9_3e12_seeded/
  reference/                      6 joblib (3 per run x 2 runs)
  v9{a,b}_{5,7}param/<cube>/
    arrays/    8 .npy   (param_stack, valid_mask, pc_scores,
                         gmm_labels, max_posterior, mahalanobis2,
                         novel_mask, novel_subclusters)
    geotiffs/  12 .tif  (5/7 PCs whitened + composite_map +
                         gmm_labels + max_posterior + mahalanobis2 +
                         novel_mask + novel_subclusters)
    logs/      1 cube-level log
  reports/
    v9_drift_check.json         per-class warm-start drift in sigma
    v9_novelty_stats.json       per-cube novelty + cluster sizes
    v9_palette.json             palette and threshold reference
    v9_run_log_<TS>.json        top-level run log + source MD5s
```

96 GeoTIFFs total (12 per cube x 4 cubes x 2 runs), 64 .npy, 6
joblib, 12 JSON. All GeoTIFFs in MTR3 source projection (MRO Mars
Equirectangular IAU 2000, 18 m/pixel).

The composite map for each cube uses encoding 0..5 = GMM class,
254 = novel (gray), 255 = invalid (transparent). On non-ref cubes
this means 80-99% of the cube reads gray.

## 6. Recommendations

The hard-prediction approach as specified does not produce a usable
cross-cube map - it exposes that 3E12's cluster basis does not
transfer to the other three cubes, primarily because of per-cube
parameter offsets that are dominated by atmospheric / calibration
residuals (BD1400, OLINDEX3, BD2500_2 all shift > 1 sigma) rather
than by surface composition.

Three viable paths from here, in order of how much they preserve the
v9 design intent:

1. **Hybrid: per-cube standardization, 3E12-fit PCA + GMM.** Apply each
   cube's own `StandardScaler` so the parameter means align across cubes
   (this is what v8 does and what removes the calibration shift), then
   project through 3E12's PCA basis and predict labels with 3E12's GMM.
   Cluster identities remain identical by construction (same Gaussians),
   but the standardization gap that drives the high m^2 disappears.
   Implementation: a 30-line edit to `stage_v9_3e12_seeded.py`. Recommended
   as the next iteration ("v9.1") if AV wants to keep the
   "single-cluster-basis" design.

2. **Accept v9 as a diagnostic.** Treat the high novelty rates as the
   answer: 3E12-defined classes only describe ~3% of B438, ~17% of A4FC,
   ~9-14% of 871C in absolute parameter space. Use the novel sub-cluster
   GeoTIFFs to characterize what each non-ref cube has that 3E12 doesn't.
   This is honest but not what the EPSL revision needs (the figure for
   the paper would be 80-99% gray on three of the four cubes).

3. **Revert to v8.** Per-cube fit + Hungarian alignment was imperfect
   (AV flagged class-flipping and over-/under-representation on
   particular cubes), but the alternative is what we found in v9: cubes
   are too different in absolute parameter space to share a 3E12-fit
   cluster basis without per-cube re-standardization. v8 plus targeted
   diagnostics on the failure modes (e.g. fix carbonate cluster on
   871C / A4FC by examining the centroids manually) is a defensible
   compromise.

My (Claude's) read: option 1 is the right next step. The negative
result tells us where the problem is (standardization, not cluster
geometry), and option 1 fixes that one issue without abandoning the
v9 design. Option 2 is the right framing if AV wants the result as-is
in the supplement. Option 3 is the fallback.

## 7. Project policy (unchanged from v8)

- No Brown 2020 Eq.2 gating in any output (still saved for diagnostic
  use only).
- Random seed 42 throughout. n_init=10. reg_covar=1e-5.
- Cluster ordering: 0 = lowest mafic-cap z-sum on 3E12, 5 = highest
  (matches v7-C / v8 convention).

## 8. Pointers

- Snapshot: `outputs/_v9_3e12_seeded/`
- Source: `code/stage_v9_3e12_seeded.py`
- Plan: `PLAN_v9.md` (project root)
- Predecessors: `outputs/_v7_mtr3_map_projected/`,
  `outputs/_v8_mtr3_with_brown/`
- Decisions log: `DECISIONS.md` (v9 entry below)
