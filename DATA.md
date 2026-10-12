# Data

## 1. Derived products (Zenodo, CC-BY-4.0)

Record: **[doi:10.5281/zenodo.23163697](https://doi.org/10.5281/zenodo.23163697)**.
Unzip every section into one folder, `data/` in the repo root by default, or set
`NILI_DATA_DIR`. Each section has its own `README.txt`, a `MANIFEST.sha256` and
build logs in `_logs/`.

| Section | Contents | Built by |
|---|---|---|
| `A_geologic_map` | Hand-drawn geologic map of the Horseshoe site: 20 layers in `Horseshoe_geologic_map.gpkg` (units, ridges, flow fronts, intrusions, dike map, profiles); same data as shapefiles and as the original ArcGIS File GDB | `code/release/export_geologic_map.py` |
| `B_ArcGIS_project` | The ArcGIS Pro project used for the paper figures, with layers repointed to this package where possible | see section README |
| `C_Brown2020_olivine_1um_analysis` | Brown et al. (2020) 1 µm band fits (centroid, asymmetry, amplitude, HWHM, mask, residual): TER3 sensor geometry (v5, .npy) and MTR3 map-projected (v6, GeoTIFF); ROI and class tables | `code/release/build_crism_sections.py` |
| `D_CRISM_band_parameters` | Viviano-Beck (2014) parameter maps (ArcGIS-computed per observation, and the Python-computed MTR3 stack), PCA/GMM maps, TER3 feature stacks | `code/release/build_crism_sections.py` |
| `E_contact_metamorphism` | All outputs of the contact-metamorphism tests (the section's `outputs/` folder is `outputs/revision/` in this repository) and the Plebani-chain ratioed cubes and carbonate maps they read | `code/release/build_crism_sections.py` |
| `F_pansharpened_CaSSIS_HiRISE` | 26 CaSSIS colour scenes pansharpened with HiRISE (0.25-0.29 m), bands NIR/RED/PAN/BLU | `code/release/convert_pansharpened.py` |
| `G_HiRISE_DEMs_Tullo` | Two HiRISE stereo DEMs and orthoimages produced by A. Tullo | `code/release/convert_dems.py` |
| `H_CTX_DEM` | CTX stereo DEM P03_002176 / P05_002888, cleaned and interpolated versions | `code/release/export_ctx_dem.py` |

### Encoding notes

- **Pansharpened scenes (F)** are stored as uint16: `I/F = DN * 2e-5`, nodata
  `DN = 65535`. The scale factor is in the GeoTIFF metadata. Maximum
  quantisation error is 1e-5 I/F, far below CaSSIS noise. The original
  float32 files are not archived. Use `config.read_reflectance()` to read
  them as float I/F.
- Reduced-resolution reads (`out_shape=` / overviews) of any raster depend on
  the overview pyramid. The archived overviews (average, levels 2-64) differ
  from the ones used during the original analysis, so decimated figure
  panels can differ slightly in appearance. Full-resolution values are unchanged.
- **HiRISE DEMs (G)** were recompressed losslessly (bit-identical values).
  `PSP_006712_2020_PSP_005855_2020-*` files carry a local CRS; their
  georeferencing comes from the accompanying `.tfwx` world files (see
  `code/contact_dtm_registry.py`).
- All other files are copied unchanged.

## 2. Public data to download (not redistributed)

Put these in `data/pds_downloads/` (or set `NILI_PDS_DIR`), keeping the
original PDS file names. CRISM products for the Plebani chain and carbonate
analysis also go under `pds_downloads/` in their PDS folder structure.

**CRISM** (PDS Geosciences Node, https://pds-geosciences.wustl.edu/missions/mro/crism.htm):
FRT00003E12, FRT0000871C, FRT0000A4FC, FRT0000B438. Products used: TER3 and
MTR3 (`*_07_if166j_*`), TRR3 + DDR (`*_07_if166l_trr3`, `*_07_de166l_ddr1`).
TER3 and MTR3 cubes go in `cubes/` (or `NILI_CUBES_DIR`), as
`cubes/<ID>_07_IF166J_TER3/` and `cubes/<id>_07_if166j_mtr3/`.

**HiRISE** (PDS Imaging Node / https://www.uahirise.org/dtm/):
DTEEC_006712_2020_005855_2020_A01, DTEEC_002888_2025_002176_2025_A01,
DTEEC_008993_2020_009626_2020_A01 and their orthoimages
PSP_006712_2020_RED_A_01_ORTHO, PSP_002176_2025_RED_A_01_ORTHO,
PSP_009626_2020_RED_C_01_ORTHO. HiRISE images used for pansharpening:
PSP_002888_2025, PSP_005855_2020, PSP_007569_2020, ESP_026992_2025,
ESP_080901_2025, ESP_081745_2025, ESP_084738_2025, ESP_084382_2025,
ESP_087639_2025, ESP_087349_2025.

**CaSSIS** (ESA Planetary Science Archive, https://psa.esa.int):
MY35_013481_158_0, MY36_014433_022_0, MY37_022904_024_3, MY37_024813_159_0,
MY37_025124_157_0, MY37_025236_157_0, MY37_025554_021_0, MY37_027047_023_1,
MY37_027358_024_0.

**CTX**: the global CTX mosaic tile MC13 (Robbins et al.) is read remotely by
`code/contact_coreg_frame.py`.

## 3. Environment variables

| Variable | Default | Points to |
|---|---|---|
| `NILI_DATA_DIR` | `<repo>/data` | Unzipped Zenodo record |
| `NILI_PDS_DIR` | `$NILI_DATA_DIR/pds_downloads` | Public PDS/PSA products |
| `NILI_CUBES_DIR` | `<repo>/cubes` | CRISM TER3/MTR3 cubes and Brown v5 maps |
| `NILI_PANSHARP_DIR` | `$NILI_DATA_DIR/F_pansharpened_CaSSIS_HiRISE` | Pansharpened scenes |
| `NILI_PLEBANI_RATIO_DIR`, `NILI_PLEBANI_CARB_DIR` | inside section E | Plebani-chain inputs |
| `NILI_BROWN_ROI_DIR` | inside section C | Brown ROI/class tables |
