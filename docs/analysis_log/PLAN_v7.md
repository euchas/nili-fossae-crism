# PLAN_v7.md - Map-projected (MTR3) products for ArcGIS

Last updated 2026-04-28. v7-A in progress (Brown fit running in background).

## 1. Scope

Three independent tracks, run in this order:

### v7-A (in progress) - Brown 2020 fit on MTR3 cubes -> v6 Brown products

Run `code/1um_centroid_analysis_CRISM_MTR3_v1.py` on the 4 MTR3 cubes
in `cubes/<lowercase_id>_07_if166j_mtr3/`. Produces, per cube:

- `<base>_centroid_map_v6.npy` (and matching .tif + .png)
- `<base>_amplitude_map_v6.npy` (and .tif)
- `<base>_hwhm_map_v6.npy` (and .tif)
- `<base>_asymmetry_map_v6.npy` (and .tif + .png)
- `<base>_residual_map_v6.npy` (and .tif)
- `<base>_mask_map_v6.npy` (and .tif) - Brown 2020 Eq.2 quality flag

Outputs land in the cube directory (matching the v5 naming convention).
GeoTIFFs carry the MTR3 projection (parsed from the ENVI header) so
they drop directly into ArcGIS Pro / QGIS.

**No clipping**: per the project specification, all valid Brown fits are saved
(centroid/amplitude/hwhm/asymmetry .npy contain every pixel where the
Nelder-Mead simplex converged). The Brown Eq.2 mask is a separate
flag, not a filter on the underlying arrays.

Wall-clock estimate: 1-4 hours per cube (per-pixel access on the
strided BSQ-derived view is ~5-10x slower than contiguous; ~700k
pixels per cube). Total 4-16 hours for all four.

### v7-B (next) - Snapshot v6 Brown outputs

Mirror the v6 .npy/.tif/.png files to
`outputs/_v7_mtr3_map_projected/<cube>/brown/` so v6 Brown is
preserved in the snapshot pattern. Write a SUMMARY_v7.md with:
- Per-cube fit pass rates (Brown Eq.2)
- Per-cube valid-fit pixel counts
- MTR3 projection metadata (CRS, pixel size)
- Side-by-side stats vs v5 TER3 Brown fits

### v7-C (planned) - Adapt v4/v5 VB+PCA+GMM pipeline to MTR3

The current v4/v5 pipeline reads TER3 detector-orientation cubes.
MTR3 cubes are map-projected with different shapes (lines/samples
vary per cube). Required changes:

1. **`crism_io.py`**: add MTR3 loader. The map-projected cubes
   require:
   - parsing ENVI header for projection metadata (already
     implemented in `1um_centroid_analysis_CRISM_MTR3_v1.py`'s
     `parse_envi_map_info`)
   - reading the I/F cube via `np.fromfile` (already implemented
     in the same script)
   - Brown maps loaded from the new v6 .npy files
   - per-cube spatial dimensions are NOT (480, 640) - the rest of
     the pipeline assumes that constant
2. **`config.py`**: add MTR3 cube directory pattern
   (`<lowercase_id>_07_if166j_mtr3` vs uppercase TER3) and a flag
   to choose which to load.
3. **`stage_b_build_stack.py`** etc.: adapt to per-cube
   variable shapes. Currently many functions assume `(H, W) =
   (480, 640)` - need to make this dynamic.
4. **`display.py` flip**: MTR3 cubes are already map-projected,
   so the FLIP_VERTICAL/FLIP_HORIZONTAL convention may not apply
   the same way. Check projection orientation.
5. **GeoTIFF write of cluster outputs**: at v7 end, the GMM /
   PCA scenes need to be written as GeoTIFFs (not just PNGs)
   so ArcGIS can ingest them. Reuse the `write_geotiff` function
   from `1um_centroid_analysis_CRISM_MTR3_v1.py`.

Requires AV's go before launching - v7-C is a substantial
refactor of v4/v5.

## 2. Output directory structure (planned)

```
outputs/_v7_mtr3_map_projected/
├── README.md                  - this stage's documentation
├── <cube>/
│   ├── brown/                 - copies of v6 Brown .npy/.tif/.png
│   │   ├── *_centroid_map_v6.npy
│   │   ├── *_amplitude_map_v6.npy
│   │   ├── *_hwhm_map_v6.npy
│   │   ├── *_asymmetry_map_v6.npy
│   │   ├── *_residual_map_v6.npy
│   │   ├── *_mask_map_v6.npy
│   │   └── *.tif (matching set, geo-referenced)
│   ├── arrays/                - v7-C VB stack + PCA + GMM (TBD)
│   ├── figures/
│   ├── tables/
│   └── logs/
├── figures/                   - v7-C cross-cube products (TBD)
└── reports/
    ├── 04_v7_brown_fit_pass_rates.json
    └── SUMMARY_v7.md
```

## 3. v6_roi_explorer.py is unchanged
Per AV's instruction: the v6 interactive ROI explorer
(`code/v6_roi_explorer.py`) stays as-is. It reads TER3 cubes via
`crism_io.load_cube_by_id()` and uses the v5 .npy Brown maps.
v7 is independent.

## 4. Naming reconciliation
- "v6" refers to two things:
  - **v6 Brown products**: the .npy/.tif files from the new MTR3
    Brown fit, named `*_<param>_map_v6.npy` matching the v5
    naming convention. This is a Brown FIT version, not a stage.
  - **v6 ROI explorer**: the interactive tool from the previous
    session, in `code/v6_roi_explorer.py`. This is an ANALYSIS
    stage version.
- "v7" is the new analysis stage covering MTR3-based map-projected
  products for ArcGIS. Output snapshot at
  `outputs/_v7_mtr3_map_projected/`.
