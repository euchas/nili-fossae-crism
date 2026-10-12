# Stage B summary

**Goal**: extend to a 15-parameter feature stack covering VNIR/Fe,
olivine/pyroxene, OH/H2O/Fe-Mg-OH, and carbonate; run dimensionality
reduction and clustering per cube; produce per-cube reports.

**Status**: v1 complete (2026-04-21), v2 full-range complete
(2026-04-22).

## v1 vs v2 at a glance

| | v1 (Brown-strict mask) | v2 (full Brown range) |
|---|---|---|
| pixel mask | `brown_mask AND finite` | `finite` only |
| valid per cube | 25k - 237k (very uneven) | 287k each (uniform) |
| methods run | PCA, UMAP, HDBSCAN, k-means, GMM | GMM only |
| date | 2026-04-21 | 2026-04-22 |
| snapshot | `outputs/_v1_brown_strict/` | `outputs/_v2_full_range_gmm/` |

## Pipeline (both versions)

1. Build 15-param Stage B stack (`stage_b_build_stack.py`)
2. Per-cube `StandardScaler` -> 15-D mean-0/std-1 matrix
3. PCA on standardized matrix; first 5 PCs feed clustering
   (~84-91% cum variance)
4. (v1 only) UMAP sweep + HDBSCAN
5. (v1 only) k-means sweep [k=2..10] + adaptive pick + fixed k=6
6. GMM sweep (v1) or single fixed-k=6 fit (v2) on the 5-D PCA scores
7. `stage_b_visualize.py` - per-cube label maps, mean spectra,
   ratioed spectra (auto mafic-cap denominator), parameter-position
   overlays, transition maps
8. `stage_b_cross_cube.py` - 4-cube median-subtracted overlay,
   pairwise centroid KS + Wasserstein, PCA loadings comparison

## Key findings

### v1 (Brown-strict)
- **Adaptive-k methods disagree hard**: silhouette puts k-means at
  k=2-3; BIC puts GMM at k=9-10 (saturated). The fixed-k=6 layer is
  the only cross-cube comparable signal.
- **HDBSCAN is essentially dead on this data**: on 3 of 4 cubes one
  cluster captures 77-99.5% of pixels with fringe remnants; A4FC
  fragments into 123 clusters at 51% noise.
- **15-param PCA**: PC1 captures 35-58%, cum-5 ~84-90%. The
  expansion to 15 params from the Stage A 8 produced more independent
  axes (PC1 dropped from ~58% on the 8-param Stage A baseline).
- **Centroid distributions differ in shape, not just bulk offset** -
  the median-subtracted overlay shows distributional shape varies
  cube-to-cube even after removing per-cube median.

### v2 (full range)
- 15-param PCA PC1 = 34-46%, cum-5 = 84-91% (slightly less
  concentrated than v1 because the new low-centroid pixels add their
  own variance dimension).
- GMM k=6 cluster sizes are well-balanced on every cube (smallest
  cluster 0.8% on A4FC c5 to largest 27% on B438 c4).
- Auto mafic-cap denominators on A4FC and B438 now select genuine
  low-centroid pixel populations (centroid 1.04 um and 1.07 um) -
  the populations Brown's strict mask had hidden.

## Per-cube k=6 GMM cluster sizes (v2, raw labels before alignment)

| cube | c0 | c1 | c2 | c3 | c4 | c5 |
|---|---:|---:|---:|---:|---:|---:|
| FRT00003E12 | 38 617 | 16 741 | 43 245 | 73 739 | 46 982 | 67 476 |
| FRT0000871C | 58 695 | 72 953 | 49 825 | 51 007 | 49 995 |  4 325 |
| FRT0000A4FC | 74 140 | 54 972 |  2 410 | 43 771 | 35 138 | 76 369 |
| FRT0000B438 | 49 583 | 60 770 | 78 696 | 28 455 | 49 955 | 19 340 |

For Hungarian-aligned versions (c0=most-mafic-cap, c5=most-phyllosilicate),
see `03_stageC_alignment.json` and `02_stageB_per_cube_<cube>.md`.

## Outputs

### v1 (snapshot at `outputs/_v1_brown_strict/`)
- 67 embeddings (PCA 4/8/15 + 12 UMAP configs/cube + selected refits)
- 31 cluster_labels arrays
- 185 figures
- 7 stage-B markdown reports (per-cube + cross-cube)

### v2 (snapshot at `outputs/_v2_full_range_gmm/` + working `outputs/`)
- 12 PCA embeddings
- 12 GMM cluster_labels arrays
- ~40 stage-B figures (per-cube + cross-cube)
- 6 stage-B markdown reports

## Limitations

- Standardization is per-cube, so PCA axes are not directly
  comparable cube-to-cube. Cross-cube comparison goes via Stage C
  Hungarian alignment, not raw PC scores.
- Fixed k=6 is a comparability choice, not a per-cube optimum
  (BIC preferred 9-10 in v1).
- Brown 4 parameters are noisy on the new v2 low-centroid pixels
  (the asymmetric Gaussian fits noise where there's no real 1 um
  band) - the VB 11 carries the signal in those regions.
- BD2500_2 column striping carried over from Stage A (known).

## Pointers

- Pipeline detail: `code/REPRODUCE.md`
- Decisions log: `DECISIONS.md`
- Per-cube reports: `02_stageB_per_cube_<cube>.md`
- Cross-cube notes: `02_stageB_cross_cube_notes.md`
- Stage C synthesis: `SUMMARY_stageC.md`
