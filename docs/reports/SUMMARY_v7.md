# v7 summary - MTR3 map-projected products for ArcGIS (2026-04-29)

**Goal**: produce georeferenced Brown band-fit maps, Brown RGB
composites, and v5-B GMM cluster scenes from the four CRISM MTR3
cubes at the Horseshoe site, all as GeoTIFFs that drop directly into
ArcGIS Pro / QGIS.

**Status**: complete (2026-04-29). Three sub-stages:

- **v7-A** Brown 2020 fit on MTR3 cubes - 6 .npy + 6 GeoTIFFs + 2 PNGs
  per cube (centroid, amplitude, hwhm, asymmetry, residual, mask).
- **v7-B** Brown RGB GeoTIFF composites - 2 RGBA TIFFs per cube
  (centroid/amplitude/hwhm and centroid/amplitude/asymmetry) at v4.1
  matched stretches.
- **v7-C** v5-B GMM clustering port to MTR3 - 5 PC + 1 GMM_raw +
  1 GMM_aligned GeoTIFF per cube + arrays.

## Inputs

- 4 MTR3 I/F cubes: FRT00003E12, FRT0000871C, FRT0000A4FC, FRT0000B438
  (~1.2-1.4 GB each, BSQ float32, MRO Mars Equirectangular IAU 2000,
  18 m/pixel)
- VB parameter implementations from `code/vb_parameters.py` (same
  helpers used in v4/v5)
- v4.1 matched-stretch values for Brown RGB composites
- Hungarian-alignment convention from Stage C / v4 / v5

## Output inventory (per cube)

| location | count | what |
|---|---:|---|
| `outputs/_v7_mtr3_map_projected/<cube>/brown/` | 6 .npy + 8 .tif + 2 .png | v6 Brown maps + 2 RGB composites |
| `outputs/_v7_mtr3_map_projected/<cube>/geotiffs/` | 12 GeoTIFFs | 5 PC raw + 5 PC whitened + 2 GMM (raw + aligned) |
| `outputs/_v7_mtr3_map_projected/<cube>/arrays/` | 6 .npy | v7-C VB stack, PC scores, GMM labels, valid mask |
| `outputs/_v7_mtr3_map_projected/<cube>/logs/` | 1 .json | v7-C per-cube run log |

Plus top-level:
- `outputs/_v7_mtr3_map_projected/reports/03_v7c_alignment.json`
- `outputs/_v7_mtr3_map_projected/reports/v7c_run_log_<TS>.json`
- `outputs/_v7_mtr3_map_projected/README.md`

## Per-cube headline metrics

| cube | shape | Brown Eq.2 pass | v5-B valid | wall (Brown) | wall (v7-C) |
|---|---|---:|---:|---:|---:|
| FRT00003E12 | 828 x 843 | 63.9% (446k) | 67.7% (473k) | 53 min | ~30 s |
| FRT0000871C | 761 x 826 | 55.5% (349k) | 69.2% (435k) | 44 min | ~30 s |
| FRT0000A4FC | 780 x 824 |  8.8% (57k)  | 67.6% (434k) | 47 min | ~30 s |
| FRT0000B438 | 823 x 830 | 35.0% (239k) | 68.5% (468k) | 50 min | ~30 s |

A4FC's low Brown pass rate (8.8%) reflects its edge-of-unit position
(most pixels have sub-1.15 um centroids that fail Brown Eq.2). The
v5-B GMM still finds ~67% of pixels usable because VB summary
parameters don't require a meaningful 1 um band.

## GMM cluster sizes (raw, per cube)

| cube | c0 | c1 | c2 | c3 | c4 | c5 |
|---|---:|---:|---:|---:|---:|---:|
| FRT00003E12 |  18444 | 105066 |  73898 |  93797 |  52304 | 129374 |
| FRT0000871C |  92811 |  95928 |  48290 |  63781 | 123141 |  11213 |
| FRT0000A4FC | 109739 |  74405 |   6030 |  49513 |  75318 | 119201 |
| FRT0000B438 | 106727 | 130869 |  37072 |  64276 |  96245 |  32660 |

(Raw labels; Hungarian-aligned labels available in `<cube>_v7c_GMM_k6_aligned.tif`
and per-cube `<cube>_v7c_GMM_k6_labels_aligned.npy`.)

## Notes for the paper

- Brown 2020 method applied verbatim per the Supporting Information
  (Nelder-Mead simplex in apparent-absorbance / wavenumber space,
  no parameter bounds during fit, Eq.2 threshold post-hoc).
- Departures from Brown 2020: (i) parameter arrays retain all
  converged fits regardless of Eq.2; the mask is a separate flag
  rather than a filter on the underlying values; (ii) RGB composites
  use 4-cube-pooled p2/p98 stretches matched across cubes for direct
  comparison; (iii) ROI-driven extraction overlaid on geomorphology
  via the v6 ROI explorer (separate tool).
- v5-B GMM uses 5 VB summary parameters (OLINDEX3, BD1400, BD1900R2,
  D2300, BD2500_2) - the "pure alteration" basis. Brown band-fit
  parameters are deliberately excluded from the clustering input
  because they would inflate the variance carried by the 1 um band
  shape.
- All processing is reproducible from a fixed random seed (42) and
  pinned package environment.

## Caveats / known limits

- Cross-cube spatial registration is NOT done. Each cube is presented
  in its native MTR3 grid. Hungarian alignment of cluster IDs in
  feature space allows "is cluster 3 the same thing on cube X and
  cube Y" comparisons, but pixel-to-pixel spatial overlay is the
  user's responsibility.
- The v6 Brown fit on MTR3 was forced to use a non-contiguous BSQ
  view of the cube buffer because of system memory pressure. This
  slowed the per-pixel fit ~5-10x but did not affect the science -
  the fit results are bit-identical to a contiguous-RAM run.
- The v7-C MTR3 pipeline does NOT apply the project's display-flip
  convention (`apply_display_transform`). MTR3 cubes are already
  map-projected (north-up by IAU 2000 convention), so no flip is
  needed. TER3 outputs in earlier versions retain the flip.

## Pointers

- Snapshot README: `outputs/_v7_mtr3_map_projected/README.md`
- Decisions log: `DECISIONS.md` (v7 entries)
- Plan: `PLAN_v7.md`
- Source: `code/1um_centroid_analysis_CRISM_MTR3_v1.py` (v7-A),
  `code/v7_brown_rgb_geotiff.py` (v7-B),
  `code/stage_v7c_mtr3_pipeline.py` (v7-C)
- Methods documents (paper): methods_short.docx, methods_supplementary.docx
  (manuscript files, not part of this repository)
