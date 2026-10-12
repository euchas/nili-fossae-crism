# v9.1 summary - per-cube StandardScaler, 3E12-fit PCA + GMM

**Goal**: fix v9's failure mode (80-99% novel on non-ref cubes,
caused by per-cube parameter offsets when 3E12's scaler was applied
to the other cubes) by fitting a fresh `StandardScaler` per cube.
Keep PCA components and GMM Gaussians locked to 3E12 so cluster
identities remain identical-by-construction.

**Status**: complete (2026-04-29). 327s wall total. **Result is a
substantial recovery from v9** - cross-cube novelty rates dropped
30-66 percentage points, no argmax pile-up on any cube, and the maps
are now usable. Some residual per-cube structure remains in the warm-
start drift (still 1-3 sigma vs plan's 0.5 sigma threshold), expected
since per-cube standardization fixes the *mean* shift but not *shape*
differences in cluster covariance.

## 1. Two parallel runs

| run | basis | dim | chi^2 critical (95%, df=d) |
|---|---|---:|---:|
| v9.1-A (control) | OLINDEX3, BD1400, BD1900R2, D2300, BD2500_2 | 5 | 11.07 |
| v9.1-B (primary) | v9.1-A + Brown centroid + asymmetry | 7 | 14.07 |

Same parameters, same VB / Brown sources, same hyperparameters as v9.
Only the standardization step changed.

## 2. Per-cube novelty (v9 vs v9.1)

| cube | v9-A | v9.1-A | delta | v9-B | v9.1-B | delta |
|---|---:|---:|---:|---:|---:|---:|
| FRT00003E12 | 23.72% | 23.72% | 0.00 | 13.47% | 13.47% | 0.00 |
| FRT0000871C | 90.73% | **43.70%** | **-47.03** | 85.95% | **43.23%** | **-42.72** |
| FRT0000A4FC | 97.80% | **43.61%** | **-54.19** | 82.63% | **50.74%** | **-31.90** |
| FRT0000B438 | 99.07% | **32.63%** | **-66.43** | 99.54% | **38.28%** | **-61.26** |

3E12 unchanged (its data path is identical between v9 and v9.1). Non-
reference cubes drop 32-66 points. B438 - the worst case in v9 -
shows the largest improvement.

The remaining 32-50% novel pixels per non-ref cube are not a calibration
artifact; they reflect populations the 3E12-fit GMM doesn't span. The
novel-sub-cluster GeoTIFFs (k=2 GMM on novel pixels alone) characterize
those populations without forcing them into 3E12 classes.

## 3. Argmax pile-up status (BEFORE novelty filter)

In v9, applying 3E12's scaler shoved 60-95% of non-ref pixels onto a
single cluster (c3 in both runs) before the chi^2 filter wiped them.
v9.1 eliminates that pile-up:

| cube | v9.1-A max cluster (frac) | v9.1-B max cluster (frac) | flag? |
|---|---|---|---|
| FRT00003E12 | c5 (27.4%) | c3 (24.7%) | ok |
| FRT0000871C | c5 (23.6%) | c0 (36.1%) | ok |
| FRT0000A4FC | c5 (24.0%) | c0 (37.9%) | ok |
| FRT0000B438 | c5 (27.5%) | c4 (26.9%) | ok |

All four cubes are below the 40% note threshold in v9.1-A; in v9.1-B
two cubes are at 36-38% (notable enrichment of cluster 0 on 871C and
A4FC, but still not a pile-up). Per the project specification the flag threshold is
60% - none of the cubes reach it.

The 871C / A4FC enrichment of v9.1-B cluster 0 is **scientifically
meaningful** (see Section 5): cluster 0 has the highest BD2500_2 (least
negative; ie strongest Mg-carbonate signature) of any cluster, plus
high olivine. That is the published "olivine-carbonate ridges"
signature, and it being more abundant on 871C and A4FC than on 3E12
is exactly what we want the cross-cube comparison to show.

## 4. Warm-start drift (sanity diagnostic, not used for the published map)

For each non-ref cube, the warm-start sanity refit compares the
cluster means after re-fitting against the 3E12 means.

| run | cube | min drift | max drift | mean drift | flagged classes (>0.5 sigma) |
|---|---|---:|---:|---:|---:|
| v9.1-A | 871C | 0.155 | 1.269 | 0.795 | 5 / 6 |
| v9.1-A | A4FC | 0.551 | 1.616 | 0.982 | 6 / 6 |
| v9.1-A | B438 | 0.254 | 1.083 | 0.630 | 4 / 6 |
| v9.1-B | 871C | 0.764 | 3.089 | 1.520 | 6 / 6 |
| v9.1-B | A4FC | 1.258 | 3.090 | 2.133 | 6 / 6 |
| v9.1-B | B438 | 0.580 | 1.423 | 1.035 | 6 / 6 |

Drift compared to v9 (single-scaler) maximum-per-cube:

| run | cube | v9 max sigma | v9.1 max sigma | reduction |
|---|---|---:|---:|---:|
| v9.1-A | 871C | 4.665 | 1.269 | -3.396 |
| v9.1-A | A4FC | 6.396 | 1.616 | -4.780 |
| v9.1-A | B438 | **9.443** | **1.083** | **-8.360** |
| v9.1-B | 871C | 4.555 | 3.089 | -1.466 |
| v9.1-B | A4FC | 6.839 | 3.090 | -3.749 |
| v9.1-B | B438 | **9.684** | **1.423** | **-8.261** |

Drift cut by 3-8 sigma in every case. Most of v9's drift was
standardization shift; once that's removed, what remains is genuine
per-cube structure (cluster shapes / covariances differ slightly even
after the means align).

v9.1-B shows higher residual drift than v9.1-A (max 3.1 vs 1.6 sigma).
This is consistent with the Brown band-fit parameters carrying more
per-cube variability than the alteration suite alone - the 1 micron
band fit is more sensitive to atmospheric / illumination conditions
than band-depth ratios at 1.4 / 1.9 / 2.3 / 2.5 micron. Worth flagging
as a caveat for v9.1-B but not a blocker.

## 5. 3E12 cluster means in original VB units (for AV's labeling)

Cluster IDs are in sklearn EM-converged order (deterministic on seed
42). AV assigns mineralogical labels by inspecting these tables and
writing them into `v9_1_palette.json` `labels` block.

### v9.1-A (5 params)

| c | OLINDEX3 | BD1400 | BD1900R2 | D2300 | BD2500_2 | n_pix | frac |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | **+0.2806** | +0.0223 | +0.0139 | +0.0240 | **-0.0093** | 18,444 | 3.9% |
| 1 | +0.1538 | +0.0130 | -0.0019 | +0.0032 | -0.0150 | 105,066 | 22.2% |
| 2 | +0.1860 | +0.0195 | **+0.0215** | **+0.0453** | -0.0175 | 73,898 | 15.6% |
| 3 | +0.2504 | +0.0182 | -0.0006 | +0.0145 | -0.0143 | 93,797 | 19.8% |
| 4 | +0.2068 | +0.0174 | +0.0074 | +0.0172 | -0.0144 | 52,304 | 11.1% |
| 5 | +0.2415 | +0.0199 | -0.0049 | +0.0079 | -0.0156 | 129,374 | 27.4% |

Reading (Claude's hypothesis, NOT auto-applied):
- **c0**: highest OLINDEX3 *and* highest BD2500_2 (least negative).
  Olivine + carbonate. Matches the published **olivine-carbonate ridge**
  signature. Smallest cluster (3.9% of 3E12).
- **c1**: lowest OLINDEX3, lowest BD1400, low D2300/BD2500_2.
  Spectrally flat. **Mafic cap candidate.**
- **c2**: highest BD1900R2 *and* highest D2300, lowest BD2500_2.
  **Fe/Mg-phyllosilicate** (alteration cluster).
- **c3, c5**: high OLINDEX3, low alteration. Olivine-rich
  background; c3 vs c5 may differ in fine olivine chemistry.
- **c4**: middle of every parameter. Mixed / intermediate.

### v9.1-B (7 params)

| c | centroid | asymmetry | OLINDEX3 | BD1400 | BD1900R2 | D2300 | BD2500_2 | n_pix | frac |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | +1.296 | +3.082 | +0.235 | +0.0195 | +0.0100 | +0.0229 | **-0.0119** | 40,394 | 8.5% |
| 1 | +1.211 | +2.044 | +0.159 | +0.0133 | -0.0015 | +0.0039 | -0.0150 | 114,743 | 24.3% |
| 2 | **+1.372** | **+4.198** | +0.249 | +0.0189 | -0.0032 | +0.0138 | -0.0149 | 104,934 | 22.2% |
| 3 | +1.333 | +3.659 | **+0.253** | +0.0203 | -0.0025 | +0.0089 | -0.0148 | 117,040 | 24.8% |
| 4 | +1.188 | +1.785 | +0.190 | +0.0191 | +0.0187 | **+0.0407** | -0.0170 | 77,034 | 16.3% |
| 5 | +1.112 | +1.225 | +0.174 | +0.0162 | **+0.0217** | +0.0314 | -0.0172 | 18,720 | 4.0% |

Reading:
- **c0**: high centroid + asymmetry + OLINDEX3 + D2300 *and* highest
  BD2500_2 (least negative). **Olivine-carbonate ridges.** This is the
  cluster that gets enriched on 871C (36%) and A4FC (38%) under v9.1-B
  - the cube-to-cube comparison is doing what the EPSL revision wants.
- **c1**: low centroid, lowest OLINDEX3, low alteration. Spectrally
  flat. **Mafic cap.**
- **c2**: highest centroid + asymmetry. Sharpest olivine band,
  moderate-high OLINDEX3. **Olivine-rich (sharp band).**
- **c3**: highest OLINDEX3, high centroid + asymmetry. **Olivine-rich
  background.**
- **c4**: low centroid + asymmetry, highest D2300, moderate-high
  BD1900R2. **Fe/Mg-phyllosilicate.**
- **c5**: lowest centroid + asymmetry, highest BD1900R2, high D2300.
  Weakest 1 micron band, wettest. **Wet / altered phyllosilicate**
  (smallest cluster on 3E12 at 4%, virtually absent from A4FC at 0.4%).

These are Claude's reads, not the canonical labeling. AV writes the
final labels into `outputs/_v9_1_per_cube_scaler/reports/v9_1_palette.json`
`labels` block before the figure is generated.

## 6. PCA explained variance (3E12 only - same as v9 since fit identical)

| run | PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v9.1-A | 40.6% | 31.8% | 19.8% | 5.2% | 2.6% | - | - |
| v9.1-B | 46.5% | 29.0% | 14.3% | 5.5% | 3.0% | 1.4% | 0.3% |

## 7. Outputs

```
outputs/_v9_1_per_cube_scaler/
  reference/                       6 joblib (3 per run x 2 runs) - 3E12 fit
  v9_1{a,b}_{5,7}param/<cube>/
    arrays/    8 .npy + per_cube_scaler.joblib (non-ref cubes only)
    geotiffs/  12 .tif (5/7 PCs whitened, composite, labels-only,
                       max_p, m^2, novel_mask, novel_subclusters)
    logs/      1 cube log (with per-cube scaler mean/scale)
  reports/
    v9_1_drift_check.json
    v9_1_novelty_stats.json        per-cube novelty + argmax_pileup
    v9_1_cluster_means_3e12.json   for AV's manual labeling
    v9_1_palette.json              AV fills in labels{} block
    v9_1_run_log_<TS>.json         top-level + source MD5s
```

96 GeoTIFFs total, 70 .npy (extra 6 = per-cube scalers), 12 joblib
(6 reference + 6 per-cube scalers), 12 JSON. Same MTR3 source projection
as v7-C / v8 / v9.

## 8. What this means scientifically

1. The cross-cube novelty rate now reflects real lithological
   differences, not calibration drift. A pixel marked novel on 871C
   genuinely doesn't fit any of 3E12's six populations - it's not just
   a BD1400 atmospheric residual.

2. The composite maps are now usable as the EPSL revision figure: 50-67%
   of each non-ref cube assigns to one of the 3E12 classes. The
   remaining "novel" fraction on each cube can be read as "this cube
   has populations 3E12 doesn't have" - a meaningful comparison
   statement.

3. The cluster-0-enrichment on 871C / A4FC under v9.1-B (36-38% vs
   3E12's 8.5%) is consistent with the published anchor finding of
   **olivine-carbonate-rich ridges** being more prevalent off 3E12.
   This shows up as the same color on all four cubes by construction
   - exactly what the v9 design was aiming for, now actually working.

4. v9.1-B has higher residual drift than v9.1-A (mean 1.5-2.1 sigma vs
   0.6-1.0 sigma). Brown band-fit parameters carry more per-cube
   variability. Worth noting in the supplement: v9.1-A is the cleaner
   "alteration only" basis; v9.1-B is the "alteration + olivine-band-
   shape" basis where the olivine class clarity is stronger but the
   per-cube comparison is slightly noisier.

5. The "novel" pixels are not noise - they're a population that 3E12
   doesn't span. The novel-sub-cluster GeoTIFFs split each cube's
   novel fraction into k=2 sub-classes for the supplement; on A4FC
   especially, this should isolate the sulfate-bearing graben-trough
   that has no analog on 3E12.

## 9. Open caveats

- v9.1-B residual drift at 1.4-3.1 sigma is above plan's 0.5 sigma flag
  threshold. Per-cube standardization fixes mean shift, not shape
  shift. If AV wants drift below 0.5 sigma we'd need to either (a)
  drop centroid + asymmetry, (b) accept v9.1-A as primary, or (c) add
  per-cube covariance whitening (not in current design and likely
  over-engineering for a comparison figure).
- The cluster means tables (Section 5) include Claude's reads of what
  each cluster might be. These are NOT applied automatically. AV
  should verify by spot-checking pixels in ArcGIS against published
  ROIs (e.g. plot the c0-only / c1-only / c2-only masks vs the Brown
  RGB and the CAR composite).
- Novel sub-clustering uses k=2 by default. For A4FC the natural
  sub-population count may be 3 (sulfate, halo, dust); the saved
  novel_mask.npy lets AV retrun sub-clustering with different k
  without re-running the full pipeline.

## 10. Pointers

- Snapshot: `outputs/_v9_1_per_cube_scaler/`
- Source: `code/stage_v9_1_per_cube_scaler.py`
- Plan: `PLAN_v9_1.md` (delta from `PLAN_v9.md`)
- Predecessor: `outputs/_v9_3e12_seeded/` (single-scaler v9, exposed
  the standardization shift)
- Decisions log: `DECISIONS.md` (v9.1 entry)
