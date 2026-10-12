# PLAN.md — project status and roadmap

Last updated 2026-04-27.

This file is the project's running roadmap. For per-stage detail see
the snapshot READMEs and the per-stage `outputs/reports/SUMMARY_*.md`
files. For methodological choices see `DECISIONS.md`. For run order
see `code/REPRODUCE.md`.

---

## Stage status

| stage | status | snapshot | summary |
|---|---|---|---|
| Stage A — geometry, validation, 8-param feature stack | **complete** (2026-04-21) | (no snapshot, in `outputs/`) | `outputs/reports/SUMMARY_stageA.md` |
| Stage B v1 — Brown-strict 15-param PCA + UMAP + HDBSCAN + k-means + GMM | **complete** (2026-04-21) | `outputs/_v1_brown_strict/` | `outputs/reports/SUMMARY_stageB.md` |
| Stage B v2 — full-range mask, GMM-only on the same 15-param basis | **complete** (2026-04-22) | `outputs/_v2_full_range_gmm/` | `outputs/reports/SUMMARY_stageB.md` |
| Stage C — cross-cube alignment + Kremer-vs-Valantinas test on Stage B | **complete** (2026-04-21, repeated 2026-04-22 on v2) | shared with B snapshots | `outputs/reports/SUMMARY_stageC.md` |
| v3 — PC deep-dive on the v2 stack (minor-phase isolation) | **complete** (2026-04-22) | `outputs/_v3_pc_deep_dive/` | `outputs/reports/SUMMARY_v3.md` |
| v4 — VB-only PCA + Brown 4 RGB separation, basis refinement | **complete** (2026-04-26) | `outputs/_v4_vb_pca_brown_rgb/` | `outputs/reports/SUMMARY_v4.md` and `_v4.../reports/04_v4_synthesis.md` |
| v4.1 — Brown RGB stretch retightening (data-driven matched) | **complete** (2026-04-26) | overwritten in v4 folder; v4 baseline preserved at `outputs/_v4_vb_pca_brown_rgb/_pre_v4p1_brown_restretch/` | DECISIONS.md "v4.1" entry |
| v5 — VB parameter pruning (v5-A 7-param + v5-B 5-param) + full-scene Brown RGB | **complete** (2026-04-27) | `outputs/_v5_param_pruning/` | `outputs/_v5_param_pruning/reports/SUMMARY_v5.md` |
| **v6 — interactive ROI explorer (ratio spectra + Brown parameter context)** | **built** (2026-04-28) | `outputs/_v6_interactive_explorer/` (per-session ROI saves) | `code/v6_roi_explorer.py` |
| v7-A — Brown 2020 fit on MTR3 cubes (v6 Brown .npy/.tif/.png) | **complete** (2026-04-29) | in cube dirs + `outputs/_v7_mtr3_map_projected/<cube>/brown/` | `code/1um_centroid_analysis_CRISM_MTR3_v1.py` |
| v7-B — Brown RGB GeoTIFF composites + snapshot | **complete** (2026-04-29) | `outputs/_v7_mtr3_map_projected/<cube>/brown/*_brown_rgb_*_v6.tif` | `code/v7_brown_rgb_geotiff.py` |
| v7-C — v5-B GMM ported to MTR3 (PC + GMM GeoTIFFs for ArcGIS) | **complete** (2026-04-29) | `outputs/_v7_mtr3_map_projected/<cube>/{arrays,geotiffs}/` | `code/stage_v7c_mtr3_pipeline.py`, `outputs/reports/SUMMARY_v7.md` |
| **v8 — 7-param MTR3 GMM (5 VB + Brown centroid + asymmetry)** | **complete** (2026-04-29) | `outputs/_v8_mtr3_with_brown/<cube>/{arrays,geotiffs}/` | `code/stage_v8_mtr3_with_brown.py`, `outputs/reports/SUMMARY_v8.md` |

## v6 - interactive ROI explorer (built 2026-04-28)

`code/v6_roi_explorer.py` - matplotlib-based interactive tool. Loads
one TER3 cube and opens **TWO live windows** that update together:

**Window 1 (Viewer)**:

| panel | what |
|---|---|
| left   | v5-B aligned k=6 GMM scene map - **drag rectangle here** |
| mid    | CAR carbonate composite (R=D2300, G=BD2500_2, B=BD1900R2) |
| right  | Brown RGB full scene (centroid / amplitude / asymmetry, v5 stretches) |
| bottom strip | status text (ROI counts) - keyboard-only, no buttons |

**Window 2 (Plots)**:

| panel | what |
|---|---|
| top    | live ratio spectrum (numerator ROI mean / denominator ROI mean) |
| bottom | Brown centroid-asymmetry hexbin (full-scene ~287k pixels) with **NUMERATOR ROI** pixels overplotted in red. Denominator NOT plotted (denominator is the ratio reference; only the science target appears on the parameter curve). |

**Paired rectangle ROIs** (drag rectangles, not lassos): each draw
adds a new ROI and the tool auto-alternates between target and mafic
per pair:

  rect 1 -> target1   rect 2 -> mafic1   (pair 1)
  rect 3 -> target2   rect 4 -> mafic2   (pair 2)
  ...

Each pair shares one color from the palette. Targets are drawn with
a solid black outline; mafics use a dashed black outline. Both have
a boxed text label centered on the ROI.

**Keyboard-only workflow** (no buttons - matplotlib widgets across
multi-window setups are flaky):
- Drag a rectangle on the GMM panel. Auto-cycle decides target vs
  mafic; you don't switch slots.
- `u` = undo last ROI.
- `c` = clear all ROIs.
- `s` = save per-ROI masks + per-pair ratio CSV + window screenshots
  to `outputs/_v6_interactive_explorer/<cube>/<TS>_*`. Save requires
  at least one COMPLETE pair (target+mafic of same pair index).
- `r` = force restretch FC + Brown panels (auto-fires on zoom anyway).
- Close either window to exit (the other follows).

**Dynamic stretch**: when you zoom or pan in the Viewer window
(matplotlib toolbar tools), the false color (Panel 2) and Brown RGB
(Panel 3) recompute their per-channel p2/p98 stretches from the
currently-visible region. The GMM panel (Panel 1) is categorical
and stays put. Panel titles update to show the active stretch
source (`visible region` vs `full scene`) and Brown channel bounds.

The viewer window suptitle always shows the next ROI label that
will be assigned (e.g. `next ROI = mafic2 (mafic)`).

Run from VS Code or terminal in project root:
```
python code/v6_roi_explorer.py                  # FRT00003E12 default
python code/v6_roi_explorer.py B438              # short form ok
```

Backend: tries QtAgg, then TkAgg, then default. If matplotlib falls
back to Agg the GUI won't appear; set MPLBACKEND env var.

Originally-planned v6 items (PC scene back-projections, batch
GMM-class ratio spectra, lobe extraction, column-neutral ratios,
Brown 2020 grain-size lines) shifted to deferred / batch-companion
to the interactive tool. The interactive workflow is more useful for
the EPSL revision figure-set choice; batch products can be generated
on demand from the saved ROI masks.

## v7 - map-projected products for GIS (planned)

After v6 ROI selection consolidates the spectra + cluster identities
to use in the paper, v7 will:

1. Read each cube's `.lbl` for projection geometry (planetographic
   lat/lon, map projection metadata).
2. Reproject the cluster maps + Brown RGBs + selected v6 ROI masks
   into IAU 2000 Mars equirectangular or polar stereographic.
3. Write GeoTIFFs that can be dropped directly into QGIS / ArcGIS /
   Pro for context inspection alongside HiRISE / CTX / MOLA.
4. Adds `rasterio` to the dep stack. Keep .npy as the pipeline-
   internal format; GeoTIFF as the publish/export format.

## Older PLAN content

The detailed Stage B v1 spec from 2026-04-21 (decisions on k-means k,
GMM k, UMAP config selection, ratioing denominator, parameter list)
is now subsumed by the snapshot READMEs and DECISIONS.md. Stage C
plan stays at `PLAN_stage_c.md`.

## Pointers

- `DECISIONS.md` - methodological decisions log (Stage A through v4.1)
- `code/REPRODUCE.md` - environment, run order, per-figure regeneration
- `outputs/INDEX.md` - snapshot folder map
- `references/INDEX.md` - reference library + per-parameter VB2014
  citation lines
- Per-stage one-pagers: `outputs/reports/SUMMARY_stage{A,B,C}.md`,
  `SUMMARY_v3.md`, `SUMMARY_v4.md`
- v4 synthesis: `outputs/_v4_vb_pca_brown_rgb/reports/04_v4_synthesis.md`
- v1, v2, v3, v4 snapshot READMEs at `outputs/_<snapshot>/README.md`
