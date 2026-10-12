# PLAN — contact-alteration test with the geologic map and DEMs

Date: 2026-08-28. Follows `PLAN_revision_contact.md` (2D distance-to-contact
test, done; see `outputs/revision/distance_to_contact/REPORT.md`). That test
left two things open:

1. Is the near-contact hydration gradient real alteration or unresolved clay
   inside "olivine" pixels at the dissected unit margin?
2. Mandon et al. 2020 argue that carbonate distributed over tens of metres of
   section rules out a thin basal contact zone. Is alteration at Horseshoe
   confined to the base of the olivine unit, or distributed through it?

Both are now testable with the assets uploaded to `Nili Planum/`.

## Inputs (inventoried 2026-08-28)

| asset | path (under `Nili Planum/CaSSIS-CRISM-Nili_Fossae/`) | notes |
|---|---|---|
| Geologic map (Fig. 4a) | `CRISM site Creta Vulcanica/Nili fossae mapping.gdb`, feature dataset `CaSSIS_units` | Clay_unit 165 polys / 158 km2; Mafic_cap 89 / 76 km2; olivine_dunes; graben_materials; `Olivine_unit` (root) = 467 km2 study outline; `Intrusions` 53 points (IDs 1-56 = Supp. Table 4). CRS: Equirectangular, clon 180, std parallel 20, R 3393833.26 m = same sphere/parallel as the MTR3 GeoTIFFs (X offset exactly 10,019,200 m). Readable directly with pyogrio (installed). |
| HiRISE DTMs (1 m) | `DTMs/DTEEC_006712_2020_005855_2020_A01.IMG` (Fig. 6/7 site), `DTMs/DTEEC_002888_2025_002176_2025_A01.IMG` (north), `DTMs/Long ridges site/DTEEC_008993_2020_009626_2020_A01.IMG` | PDS (UA) products. `.IGwx` sidecars carry the manual alignment shift (e.g. 006712: dX -77 m, dY -104 m) - must be applied by hand, GDAL ignores them. Orthos: `PSP_006712_2020_RED_A_01_ORTHO.JP2`, `PSP_002176_2025_RED_A_01_ORTHO.JP2`, `PSP_002888_2025_IRB_A_01_ORTHO.JP2`. |
| CTX DEM (20 m) | `Isidis Project/Isidis Project.gdb` raster `P03_002176_2024_XI_22N283W__P05_002888_2025_XI_22N283W_dtm_interpolate` | Correct CRS (Mars 2015 sphere, equirect clon 0); readable via `OpenFileGDB:` in rasterio. The `.tif` twins in `DTMs/CTX/` have broken or LOCAL_CS georeferencing - do not use. Covers the whole map area. |
| Controlled CTX mosaic (6 m) | USGS Robbins controlled CTX mosaic, tile MC13 (6 m/px, 16-bit) | Robbins mosaic; the frame the CaSSIS-HiRISE basemap and therefore the geologic map were aligned to. |
| CRISM MTR3 grid, GMM labels, params | `outputs/_v9_1_per_cube_scaler/...`, `outputs/_v8_mtr3_with_brown/...` | as in the 2D test |
| Phase-correlation engine | the author's CaSSIS coregistration utility `coreg_cassis.py` (ported here as `code/contact_coreg_lib.py`) | `coarse_lock`, `tile_offsets`, `fit_offsets`; rasterio + skimage, both available |

Not found in the upload: the PANCO pansharpened HiRISE-CaSSIS tiles. Not
needed for this plan.

## Stages

### Stage 0 - Frame check (must come first)
Everything below bins by tens of metres, so the CRISM-to-map registration
has to be measured, not assumed.
- For each cube: phase-correlate a CRISM VNIR band (~0.75 um) against the
  Robbins CTX mosaic resampled to the MTR3 grid, using the Z: engine
  (`tile_offsets` + confidence gate). Record dx, dy, confidence per cube.
  Apply as a geotransform shift only if |offset| > 0.5 px and the fit is
  confident; otherwise leave the PDS georeferencing.
- Same for each HiRISE ortho (with the `.IGwx` shift applied) against the
  CTX mosaic, to confirm the manual alignment.
- Output: `coreg_offsets.json`, QC PNGs.

### Stage 1 - Sub-pixel clay fraction (settles question 1)
- Rasterise Clay_unit, Mafic_cap, olivine_dunes, graben_materials at 3 m
  in the MTR3 CRS, block-average to 18 m -> per-pixel area fractions
  f_clay, f_cap, f_dune, f_graben for each cube grid. Olivine-unit fraction
  = Olivine_unit outline minus the others.
- Among pixels with f_oliv >= 0.9 (map says olivine, at most 10% anything
  else) repeat the distance-to-contact profiles of the 2D test. If the
  D2300 / BD1900R2 gradient over 100-400 m survives with f_clay = 0, it is
  not map-scale clay mixing. Also regress D2300 on f_clay (all olivine-unit
  pixels) and report the residual gradient with distance.
- Cross-check the GMM class map against the geologic map (confusion matrix
  clay/cap/olivine/dune) - a useful supplement table in its own right.

### Stage 2 - Height above the basal contact (settles question 2)
- Extract the clay/olivine contact as the shared boundary of Clay_unit and
  the olivine-unit polygon (map lines, not CRISM pixels).
- Sample contact elevation from the CTX DEM (whole map) and from the HiRISE
  DTMs (three areas), build a contact surface by natural-neighbour /
  IDW interpolation of contact-line elevations within 1 km, and compute
  h = z_pixel - z_contact for every olivine-unit pixel.
- Profiles of D2300, BD1900R2, BD2500_2, centroid vs h (bins 5 m with
  HiRISE, 10 m with CTX). Also the olivine-unit thickness map where the cap
  is present (z at cap base - z_contact), to put a number on the
  aureole-scaling argument (Einsele 1982; Aarnes 2010).
- Caveats to carry: CTX DEM vertical noise (~5-10 m), so the CTX result is a
  coarse check and the HiRISE result the real one; the contact surface is
  only defined near mapped contacts, so restrict to pixels within ~1 km of
  a contact line.

### Stage 3 - 3D distance
- Combine horizontal distance and h into a 3D distance to the contact
  surface; redo the 2D profiles with it. Expected to sharpen, not change,
  the Stage 1 result.

### Stage 4 (optional, HiRISE only) - Ridge margins
- For the intrusion points inside the three HiRISE DTM footprints, extract
  topographic profiles perpendicular to the local ridge trend (from the
  ridge polylines / dike map) and compare flank elevation at 5-30 m from
  the olivine core with far-field clay at 50-150 m. Indurated flanks are a
  morphological, CRISM-independent signal of altered/cemented contacts
  (thermal or fluid - stated as ambiguous).
- Report as exploratory; no claims beyond "present / not present".

### Stage 5 - Figures and report
- `outputs/revision/contact_3d/REPORT.md`; supplement-ready figures:
  (a) profiles vs distance at f_clay = 0, (b) profiles vs height above
  contact, (c) contact/height map on the HiRISE DTM for the Fig. 6 site.

## Reproducibility
Scripts in `code/contact_coreg_frame.py`, `code/contact_geomap_fractions.py`,
`code/contact_height_above_contact.py`, `code/contact_ridge_margins.py`;
seed 42; JSON logs with input md5s next to every output. No source data is
modified; derived rasters go to `outputs/revision/contact_3d/`.

## Runtime estimate
Stage 0 ~10 min; Stage 1 ~10 min; Stage 2 ~20 min (HiRISE warps dominate);
Stage 3 ~2 min; Stage 4 ~10 min. Nothing needs the cloud.

## Decision points for AV
1. Which HiRISE DTM alignment to trust: the `.IGwx` shift as-is, or
   re-measure against the CTX mosaic in Stage 0 (I propose: re-measure and
   report both; use the measured one).
2. Whether the `Olivine_unit` outline minus other units is the intended
   olivine-unit definition for the map (as the Methods text implies).
3. Whether to include Stage 4.
