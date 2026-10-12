# v4 summary - VB-only PCA + Brown RGB workflow

**Goal**: separate "what minerals are present" (VB-only PCA + GMM) from
"what kind of olivine, where" (Brown 4-param RGB at fixed physical
ranges); refine the parameter basis to drop redundancy and add ferric-
edge / HCP sensitivity.

**Status**: complete (2026-04-26).

## What changed from v2/v3

- **Brown 4 dropped from PCA**. Brown band-fit parameters are
  individually physically interpretable; including them in the PCA
  inflates "olivine band shape" variance and crowds out minor-phase
  axes.
- **BD2290 and BD2355 dropped from VB basis**. v2 loadings showed
  them loaded near-identically with D2300 on PC2 - three parameters
  for one feature.
- **SH600_2 added** (VB2014 Table 1). Independent ferric-edge
  sensitivity. Formula is INVERTED from band depth: positive for
  reflectance peaks. Per-cube medians +0.13 to +0.16 confirms ferric
  edge captured correctly.
- **HCPINDEX2 added** (VB2014 Table 1). Paired with LCPINDEX2 for
  full pyroxene story.
- **Net: 11-parameter VB-only basis** for PCA, Brown 4 handled
  independently as RGB + grayscale + scatter.

## v4 vs v3 vs v2/v1 at a glance

| | v1 (Brown-strict) | v2 (full range) | v3 (PC deep-dive) | v4 (VB-only PCA) |
|---|---|---|---|---|
| pixel mask | brown[mask] AND finite | finite only | finite only | finite only |
| PCA basis | 15 (Brown 4 + VB 11) | same 15 | same 15 | **11 (VB only)** |
| VB count | 11 | 11 | 11 | **11 (drop BD2290+BD2355, add SH600_2+HCPINDEX2)** |
| methods | PCA + UMAP + k-means + GMM + HDBSCAN | GMM only | GMM + targeted PC GMM | **GMM only on VB-PCA** |
| Brown handling | inside the PCA | inside the PCA | inside the PCA | **independent RGB workflow** |
| smallest cluster v2/v3-equivalent | varies | down to 2.4k (A4FC c5 in v2) | 2.4k | **12k (A4FC, well-balanced)** |
| snapshot | `_v1_brown_strict/` | `_v2_full_range_gmm/` | `_v3_pc_deep_dive/` | `_v4_vb_pca_brown_rgb/` |

## Key numbers per cube (v4)

| cube | PC1-5 cum | PCs needed for >=90% | GMM smallest | SH600_2 median | Brown post-residual |
|---|---:|---:|---:|---:|---:|
| FRT00003E12 | 87.3% | 6 | 14 828 | +0.152 | 225 543 |
| FRT0000871C | 82.4% | 7 | 20 453 | +0.149 | 148 174 |
| FRT0000A4FC | 86.4% | 6 | 12 739 | +0.134 |  24 202 |
| FRT0000B438 | 85.6% | 6 | 12 055 | +0.158 | 114 771 |

Reading:
- 871C alone needs 7 PCs to reach 90% (compositionally richest cube
  in the VB-only space).
- All four cubes have well-balanced k=6 partitions on whitened PCs -
  cleaner than v2/v3 where A4FC bottomed out at 2.4k pixels.
- A4FC's Brown-valid count is the v1 count (~25k) - most of the cube
  has centroid < 1.15 um and fails Brown 2020 eq. 2.

## Hungarian alignment quality (max centroid distance to 3E12)

- 871C: 2.65 (c5)
- A4FC: 2.75 (c4)
- B438: 2.39 (c1)

Same pattern as v3: bulk populations match well across cubes, the
most-extreme aligned clusters (~ phyllosilicate-rich tail) are weak
analogues of 3E12.

## Outputs

- `outputs/_v4_vb_pca_brown_rgb/<cube>/{arrays,figures,tables,logs}/`
  with 8 arrays + 11 figures + 1 CSV + 2 logs per cube
- `outputs/_v4_vb_pca_brown_rgb/figures/` with 6 cross-cube figures:
  - `comparison_aligned_k6_centroid_per_param_v4.png`
  - `comparison_all_cubes_median_subtracted_v4.png`
  - `comparison_PCA_loadings_4cube_v4.png`
  - `comparison_GMM_k6_4cube_v4.png`
  - `comparison_Brown_RGB_centroid_amp_hwhm_4cube_v4.png`
  - `comparison_Brown_RGB_centroid_amp_asym_4cube_v4.png`
- `outputs/_v4_vb_pca_brown_rgb/reports/03_v4_alignment.json`

## What this analysis is for

The v4 split is the architecture for the EPSL revision figure set:

- **Brown RGB 2x2 cross-cube panels** are the headline figures - at
  matched stretches, any color difference between cubes is a real
  compositional/textural difference, not a stretch artefact. These are
  the figures a reviewer can use to see whether the olivine
  composition + grain-size structure replicates across the four
  Horseshoe cubes.
- **GMM k=6 cross-cube panel** with aligned colors is the equivalent
  reviewer test for the broader VB mineralogy: same colors mean same
  populations across cubes, so a panel where cube N's c3 is large and
  cube M's c3 is small is direct evidence of compositional
  heterogeneity (not arbitrary cluster permutation noise).
- **Per-cube PCA loadings + scree + pair matrix** are the supporting
  evidence for what each PC means per cube; cross-cube loadings 2x2
  is the comparability check.

## What's still open

- v5 will produce PC scene back-projections, GMM-class ratio spectra,
  and lobe extraction from the v4 arrays. v4 saves all required
  inputs and does NOT preempt those outputs.
- Brown grain-size lines on the centroid-asymmetry scatter (Brown 2020
  Figure 16) - skipped, flagged for v4.1.
- Column-neutral ratios - flagged for v4.1.

## Pointers

- Decisions log: `DECISIONS.md` (v4 section at bottom)
- Reproduce: `code/REPRODUCE.md` (v4 section)
- VB2014 parameter source: `references/INDEX.md` (SH600_2 + HCPINDEX2 entries)
- Snapshot: `outputs/_v4_vb_pca_brown_rgb/` with its own README
