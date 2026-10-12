# PLAN_v9.md - 3E12-seeded GMM for cross-cube comparison

Last updated 2026-04-29.

## 0. Goal

Replace per-cube GMM + Hungarian alignment (v7-C, v8) with a single
GMM trained on FRT00003E12, applied **unchanged** to the other three
cubes. Pixels that don't match any 3E12-defined class are flagged as
"novel" rather than forced into the nearest cluster. Same six class
colors mean exactly the same Gaussian everywhere, by construction.

Fixes the failure modes seen in v8:
- Olivine class-flipping between cubes (PC1 was loaded by colinear
  olivine features and small per-cube differences pushed pixels
  around).
- 871C carbonate over-representation (carbonate Gaussian inflated
  to fit local population).
- A4FC carbonate loss to the graben-trough sulfate-bearing depression
  (sulfate-ish band depths at BD2500_2 hijacked the carbonate
  Gaussian on that cube).

Removes the need for Hungarian alignment entirely - class IDs are
identical across cubes by construction.

## 1. Two parallel runs

| run | basis | dim | rationale |
|---|---|---:|---|
| **v9-B** (primary) | v5-B + Brown centroid + asymmetry | 7 | matches v8; AV observed more structure and less noise; olivine is the science focus |
| v9-A (control) | v5-B alteration only | 5 | matches v7-C; cleanest alteration discrimination, no Brown colinearity in PC1; cross-check that v9-B's olivine clarity is real and not a basis artifact |

Both run end-to-end. v9-B is the figure; v9-A goes in the supplement
or as a sanity overlay.

## 2. Reference cube

**FRT00003E12** throughout. All three transformations (standardization,
PCA, GMM) are fit on its valid pixels alone and saved to disk. The
other three cubes (871C, A4FC, B438) only ever see the saved
transformations - they never participate in fitting.

## 3. Pipeline

### Reference cube (3E12) - fit and save

1. Build the param stack (5 for v9-A, 7 for v9-B). Same VB helpers as
   v7-C / v8. Same v6 Brown .npy as v8 for centroid + asymmetry. Mark
   pixel valid if all params finite.
2. Fit `StandardScaler` on valid pixels. Save `mean_`, `scale_`.
3. Fit `PCA(n_components=d, whiten=True)` on standardized scores
   (d=5 for v9-A, d=7 for v9-B). Save `components_`, `mean_`,
   `explained_variance_`, `whiten=True`.
4. Fit `GaussianMixture(n_components=6, covariance_type='full', n_init=10, random_state=42)`
   on whitened PC scores. Save `means_`, `covariances_`, `weights_`,
   `precisions_cholesky_`.

Persist via `joblib.dump` to
`outputs/_v9_3e12_seeded/reference/3e12_v9{a,b}_{scaler,pca,gmm}.joblib`.

### Non-reference cubes (871C, A4FC, B438) - hard predict

1. Build the same param stack with the same code path.
2. `scaler.transform(X)` - do NOT call `fit`.
3. `pca.transform(X_scaled)` - do NOT call `fit`. (sklearn PCA
   centers using the saved `mean_` automatically.)
4. Build a fresh `GaussianMixture` with the same `n_components=6` and
   `covariance_type='full'`. Assign `means_`, `covariances_`,
   `weights_`, `precisions_cholesky_` from the saved 3E12 GMM. Then
   call `predict` and `predict_proba`. Do NOT call `fit` or
   `fit_predict`.

This is hard-prediction. Cluster definitions are frozen.

### Sanity-check pass (parallel diagnostic, not for publication)

For each non-reference cube also run a warm-start re-fit: same
hyperparameters, but call `fit` with `means_init`, `precisions_init`,
`weights_init` set from the 3E12 GMM. Save the resulting means and
compare drift against 3E12's within-class sigma. Report in
`v9_drift_check.json`:
- per-class `||mu_warmstart - mu_3e12|| / sigma_within_3e12`
- if any class drifts > 0.5 sigma, flag it; the hard-prediction map is the
  one we publish, but the warm-start map is informative for the
  caveats paragraph.

## 4. Novelty flagging

For each pixel in each non-reference cube, in whitened PC space:

- **Mahalanobis^2** to assigned cluster: `m^2 = (x - mu_k)^T Sigma_k^-1 (x - mu_k)`,
  where k is the argmax-posterior class from `predict`.
- **Max posterior**: `p_max = max_k predict_proba(x)[k]`.

Flag a pixel as novel if EITHER:
- `m^2 > chi2_{0.95, d}`, where d=5 (v9-A) or d=7 (v9-B). Critical
  values: **11.07** for d=5, **14.07** for d=7. ("This pixel is
  implausible under its assigned cluster.")
- `p_max < 0.6`. ("This pixel is ambiguous between clusters." 0.6 is
  the starting threshold; tunable post-hoc.)

Save `mahalanobis2`, `max_posterior`, and the boolean `novel_mask` as
separate GeoTIFFs so thresholds can be retuned without re-running.

## 5. Map outputs

Per-cube composite map (the figure):
- 6 cluster colors, fixed palette defined once on 3E12 (e.g. olivine
  = orange, Mg-carbonate = green, Fe/Mg-phyllosilicate = blue, etc. -
  fix after looking at 3E12).
- **Neutral gray** for novel pixels (not a hot color - reads as "no
  claim" rather than "look at me").
- Transparent NoData for invalid (any param non-finite).

3E12 itself will have ~5% novel pixels (its own training distribution
clipped at 95% chi^2). That's fine and honest - it shows up as gray
salt-and-pepper on 3E12 too.

## 6. Novel sub-classes (supplement only)

For each non-reference cube, take pixels flagged as novel and fit a
small GMM (k=2 or 3) on those pixels alone, in the same whitened PC
space. This characterizes whatever 3E12 doesn't cover - e.g. A4FC's
sulfate-bearing graben depression should fall out as one or two
coherent sub-populations rather than getting absorbed into a
relabeled carbonate class.

Output as a **separate** "novel sub-classes" GeoTIFF per cube. Do
**not** merge into the main labels.

## 7. What NOT to do

- Do not re-fit the scaler, PCA, or GMM on the non-reference cubes.
- Do not Hungarian-align (no alignment needed; classes are by
  construction identical).
- Do not increase k on the full cube to "find more structure" -
  splitting existing classes adds noise without answering the
  question. Use the novel-pixel sub-clustering in Section 6 instead.
- Do not gate any output on Brown 2020 Eq.2 (project policy
  2026-04-29).
- Do not change the random seed (42) or `n_init` (10).
- Do not center-correct outside sklearn - `pca.transform()` already
  uses the saved `mean_`.

## 8. Output layout

```
outputs/_v9_3e12_seeded/
  reference/
    3e12_v9a_scaler.joblib   3e12_v9b_scaler.joblib
    3e12_v9a_pca.joblib      3e12_v9b_pca.joblib
    3e12_v9a_gmm.joblib      3e12_v9b_gmm.joblib
  v9a_5param/
    <cube>/
      arrays/    param_stack.npy, valid_mask.npy, pc_scores.npy,
                 gmm_labels.npy, max_posterior.npy, mahalanobis2.npy,
                 novel_mask.npy
      geotiffs/  5 PC + gmm_labels + max_posterior + mahalanobis2
                 + novel_mask + composite_map (labels with novel as gray)
                 [+ novel_subclusters from Section 6]
      logs/      v9a_<cube>_<TS>.json
  v9b_7param/
    <cube>/  (same layout, 7 PCs)
  reports/
    SUMMARY_v9.md
    v9_drift_check.json     warm-start drift per class per cube
    v9_novelty_stats.json   per-cube novel counts + thresholds used
    v9_palette.json         class -> color map fixed on 3E12
```

GeoTIFFs carry the source MTR3 projection (MRO Mars Equirectangular,
IAU 2000, 18 m/pixel) - same convention as v7 / v8. Drop directly
into ArcGIS Pro.

## 9. Methods paragraph (draft for the EPSL revision)

> Cluster definitions for the four CRISM MTR3 cubes were derived
> from the reference scene (FRT00003E12) alone. Standardization, a
> whitened principal-component projection (d=7: OLINDEX3, BD1400,
> BD1900R2, D2300, BD2500_2, plus the Brown 2020 1 micron centroid and
> asymmetry), and a Gaussian Mixture Model (k=6, full covariance,
> n_init=10, random seed 42) were fit sequentially on 3E12's valid
> pixels and persisted. The remaining three cubes (871C, A4FC, B438)
> were standardized and projected through the saved transformations
> without re-fitting; cluster labels and posterior probabilities were
> obtained from the saved GMM. Pixels with Mahalanobis^2 to their
> assigned cluster mean above the 95% chi^2 threshold for d=7 (14.07),
> or with maximum posterior below 0.6, were flagged as "novel" and
> rendered in neutral gray. This produces a cross-cube class
> assignment that is identical by construction for the six
> 3E12-defined classes, eliminating the per-cube re-fitting and
> Hungarian alignment used in earlier iterations and making
> cube-specific lithologies explicitly visible rather than absorbed
> into the nearest pre-existing class.

## 10. Pointers

- Predecessors: `outputs/_v7_mtr3_map_projected/` (v7-C 5-param,
  per-cube fit + Hungarian), `outputs/_v8_mtr3_with_brown/` (v8
  7-param, per-cube fit + Hungarian)
- Decisions log: `DECISIONS.md` (add v9 entry on first run)
- Source code: `code/stage_v9_3e12_seeded.py` (to be written)
- Reuse: `code/vb_parameters.py`, GeoTIFF writer from
  `code/1um_centroid_analysis_CRISM_MTR3_v1.py`, MTR3 loader path
  conventions from `code/stage_v7c_mtr3_pipeline.py` and
  `code/stage_v8_mtr3_with_brown.py`.
