# PLAN_v9_1.md - per-cube StandardScaler, no auto-reorder

Last updated 2026-04-29.

This is a delta from `PLAN_v9.md`. Read PLAN_v9.md first; only the
two changes below override it.

## Why

v9 (per PLAN_v9.md) used 3E12's StandardScaler for all four cubes.
That produced 80-99% novel pixels on 871C / A4FC / B438 because per-
cube parameter means were shifted -1 to -3.6 sigma against 3E12's
scaler - the worst offender being BD1400 at -2.7 to -3.6 sigma on
every non-ref cube. That shift pattern (1.4 micron OH overtone next
to atmospheric H2O) is consistent with CAT/VOLCAT atmospheric
correction residual, not surface mineralogy. See SUMMARY_v9.md
Section 4.

v9.1 fixes the standardization while keeping cluster definitions
identical-by-construction.

## Changes from PLAN_v9.md

### Change 1: per-cube StandardScaler

- **3E12**: fit StandardScaler, PCA(whiten=True), GMM(k=6) - same as
  v9. Save all three.
- **Other cubes (871C / A4FC / B438)**: fit a NEW StandardScaler on
  each cube's own valid pixels. Do NOT reuse 3E12's scaler. Then
  `pca.transform(...)` and `gmm.predict(...)` using the saved 3E12
  PCA and GMM. Do not refit PCA or GMM.

PCA `components_` and GMM `means_` / `covariances_` / `weights_` /
`precisions_cholesky_` stay locked to 3E12. That is what preserves
identical-clusters-by-construction. Only the standardization step
becomes per-cube.

### Change 2: drop the mafic-cap z-sum reordering

The v9 reorder summed olivine and alteration parameters together
(BD1900R2 + D2300 + BD2500_2 + OLINDEX3) so high olivine and high
carbonate looked identical to it; the resulting "0 = least altered,
5 = most altered" labeling didn't follow from the metric used.

v9.1 does not auto-reorder. After fitting GMM on 3E12, dump each
cluster's mean parameter vector in **original VB units** (un-
standardize using 3E12 scaler's `mean_` and `scale_`; un-whiten and
un-rotate using saved PCA `inverse_transform`). Print to the run log
so AV can assign mineralogical labels by inspection and write them
into `v9_1_palette.json` as the canonical mapping before the figure
is generated.

Reference labels (NOT applied automatically; for orientation only
when reading the printed means):

- Mafic cap: low OLINDEX3, low BD1900R2, low D2300, low BD2500_2
  (spectrally flat capping unit, not olivine)
- Olivine-rich: high OLINDEX3, low alteration params
- Mg-carbonate: high BD2500_2 + moderate BD1900R2, low D2300
- Fe/Mg-phyllosilicate: high D2300 + BD1900R2, moderate BD1400
- Mixed/other: assign by closest match

### Additional reporting requirement

After v9.1 completes, report the argmax cluster size distribution on
non-ref cubes BEFORE any novelty filter is applied. If per-cube
standardization works, no single cluster should hold > ~40% of
pixels on any cube. If c3 (or any cluster) still holds 60%+, flag
it - that means residual covariance-shape mismatch beyond what per-
cube standardization fixes, and we'll need to look at it before
trusting the map.

## What stays the same (from PLAN_v9.md)

- Two parallel runs: 7-param primary (now `v9_1b`) and 5-param control
  (now `v9_1a`).
- Hard prediction. No warm-start re-fit for the published map.
- Mahalanobis^2 + p_max < 0.6 novelty flag with the same chi^2 critical
  values (11.07 for d=5, 14.07 for d=7).
- Save mahalanobis2 and max_posterior as separate GeoTIFFs so
  thresholds remain tunable post-hoc.
- Novel-pixel sub-clustering (k=2) for the supplement.
- No Brown 2020 Eq.2 gating.
- Random seed 42, n_init 10, reg_covar 1e-5.
- Same MTR3 cube source, same VB parameter implementations, same
  Brown v6 maps.
- Cluster colors stay the existing 6-color tab10 subset (matches v7-C
  / v8); cluster IDs are now in sklearn's natural EM-converged order
  (deterministic on seed 42).
- Warm-start drift sanity pass still computed for diagnostic purposes
  (not used to relabel).

## Output layout

```
outputs/_v9_1_per_cube_scaler/
  reference/
    3e12_v9_1a_scaler.joblib   3e12_v9_1b_scaler.joblib
    3e12_v9_1a_pca.joblib      3e12_v9_1b_pca.joblib
    3e12_v9_1a_gmm.joblib      3e12_v9_1b_gmm.joblib
  v9_1a_5param/
    <cube>/
      arrays/    param_stack, valid_mask, pc_scores, gmm_labels,
                 max_posterior, mahalanobis2, novel_mask,
                 novel_subclusters, plus per_cube_scaler.joblib for
                 each non-ref cube
      geotiffs/  5 PC + composite + labels-only + max_p + m^2 +
                 novel_mask + subclusters
      logs/      v9_1a_<cube>_<TS>.json
  v9_1b_7param/
    <cube>/   (same layout, 7 PCs)
  reports/
    SUMMARY_v9_1.md
    v9_1_drift_check.json    warm-start drift, diagnostic only
    v9_1_novelty_stats.json  per-cube novel + argmax pile-up + cluster sizes
    v9_1_palette.json        cluster_id -> RGB + (AV-assigned) label
    v9_1_cluster_means_3e12.json  3E12 cluster means in original VB units
```

## Pointers

- Predecessor: `outputs/_v9_3e12_seeded/` (v9, single-scaler, exposed
  the standardization shift)
- Source: `code/stage_v9_1_per_cube_scaler.py` (to be written; reuses
  helpers from `code/stage_v9_3e12_seeded.py`)
- Decisions log: `DECISIONS.md`
