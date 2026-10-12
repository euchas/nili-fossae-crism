# Reproducing the Nili Fossae CRISM reanalysis

Environment and run order, so any result in `outputs/` can be regenerated.

## Environment

- Python 3.14.0 (see `python-version.txt`)
- Package versions in `requirements.txt` (the exact versions used for the paper are noted there)
- Random seed 42 (config.py `RANDOM_SEED`)
- Platform tested: Windows 11

Install:

```
python -m pip install -r requirements.txt
```

## Inputs (not in repo)

Everything not in this repository is either in the Zenodo record
(doi:10.5281/zenodo.23163697) or public PDS/PSA data. See `../DATA.md` for
the list and for the environment variables that point `config.py` at them.
The CRISM TER3/MTR3 cubes and Brown v5 maps go under `cubes/` (see
`config.CUBES_DIR`); the Brown v5 `.npy` maps are in Zenodo section C.

## Run order

Each script is idempotent, writes to `outputs/<category>/<cube>_*.{npz,npy,json,png}`,
and logs parameters to `outputs/logs/<cube>_stage*_<TIMESTAMP>.json` alongside
the md5 of every source file the run depended on.

### Stage A — per-cube VB parameters + sanity checks

```
cd code
python stage_a_build_feature_stack.py         # 15-param stack (also writes 8-param baseline)
python stage_a_validation_3e12.py             # Brown 2020 centroid reproduction check
python stage_a_sanity_plots.py                # spatial parameter maps
python stage_a_a4fc_valid_map.py              # A4FC valid-pixel geometry
```

Outputs: `outputs/feature_stacks/*.npz`, `outputs/figures/*stageA*`, `outputs/reports/01_stageA_*`.

### Stage B — dim-reduction + clustering on 15-param stack

```
python stage_b_build_stack.py                 # standardize + joint-valid mask (15-param)
python stage_b_pca.py                         # PCA on 4/8/15-param stacks
python stage_b_umap.py                        # 12-config UMAP sweep + selected refits (~1 hr total)
python stage_b_cluster_pca.py                 # k-means + GMM on PCA 5-D (adaptive + fixed k=6)
python stage_b_cluster_hdbscan.py             # HDBSCAN on selected UMAP embeddings
python stage_b_visualize.py                   # label maps, mean spectra, ratioed spectra, param positions
python stage_b_cross_cube.py                  # 4-cube diagnostics
python stage_b_report.py                      # per-cube + cross-cube markdown reports
```

Outputs: `outputs/embeddings/`, `outputs/cluster_labels/`, `outputs/figures/*stageB*`, `outputs/reports/02_stageB_*`.

### v3 — PC deep-dive on the v2 GMM stack (2026-04-22)

```
python stage_v3_per_cube.py        # all four cubes
python stage_v3_cross_cube.py      # cross-cube lobe spectra
```

Outputs land in `outputs/{figures,logs,reports,cluster_labels}/` with `_v3` suffix and snapshot at `outputs/_v3_pc_deep_dive/`.

### v4 — VB-only PCA + Brown band-fit RGB workflow (2026-04-26)

Drops Brown 4 from the PCA basis; drops BD2290 / BD2355 (redundant with D2300);
adds SH600_2 (independent ferric edge) and HCPINDEX2 (paired with LCPINDEX2).
Brown 4 is processed independently as RGB composites + grayscale maps.

```
python stage_v4_vb_pca_and_brown_rgb.py        # all four cubes + cross-cube
```

Outputs: `outputs/_v4_vb_pca_brown_rgb/<cube>/{arrays,figures,tables,logs}/`
plus top-level `outputs/_v4_vb_pca_brown_rgb/{figures,reports}/`.

Wall clock ~3 minutes total (much faster than v1/v2 because UMAP/HDBSCAN
are skipped and Brown is processed independently of the PCA fit).

### v4.1 — Brown RGB stretch retightening (2026-04-26)

Replaces v4's Brown 2020 eq.2 validity bounds with 4-cube-pooled
p2/p98 stretches so the colormaps are fully used on real data:
centroid 1.15-1.40, amplitude 0.10-0.37, HWHM 0.18-0.30,
asymmetry 1.25-5.00. Each Brown figure carries a footnote
explaining the data-driven matched-stretch convention. v4 baseline
preserved at `outputs/_v4_vb_pca_brown_rgb/_pre_v4p1_brown_restretch/`.

```
python stage_v4_vb_pca_and_brown_rgb.py --brown-only
```

Skips TER3 load + VB compute + PCA + GMM and just rebuilds Brown
figures from the cached `brown_valid_mask_v4.npy` arrays. Wall
clock ~30 s.

### Contact metamorphism and band-centre tests (2026-08/09, revision)

Plans: `PLAN_revision_contact.md`, `PLAN_revision_contact_3D.md`,
`PLAN_revision_2300_zonation.md`. Results: `outputs/revision/**/REPORT.md`
(text copies in `docs/contact_reports/`; full outputs in Zenodo section E,
whose `outputs/` folder unpacks to `outputs/revision/` in this repository). Inputs: geologic map
(section A), HiRISE DEMs (section G and PDS), CTX DEM (section H), Plebani-chain
ratioed cubes and carbonate maps (section E), MTR3 products from the v7/v8 runs.

Contact geometry (PLAN_revision_contact_3D.md stages):

```
python contact_coreg_frame.py              # stage 0: CRISM/CTX/HiRISE frame check
python contact_new_dem_coreg.py            # stage 0: frame check for A. Tullo's DEMs
python contact_dtm_registry.py             # registry of usable HiRISE DEMs (imported by later steps)
python contact_geomap_fractions.py         # stage 1a: sub-pixel unit fractions on the CRISM grid
python contact_distance_to_contact.py      # distance-to-contact test
python contact_fraction_profiles.py        # stage 1b: distance-to-contact profiles
python contact_height_above_contact.py     # stages 2-3: height above the basal contact
python contact_ridge_margins.py            # stage 4: HiRISE-scale ridge-margin profiles
python contact_block_stats.py              # post-verification statistics
python contact_contact_fringe.py           # stacked DTM profiles across the clay/olivine contact
python contact_robustness_90m.py           # near-band robustness check
```

2.3 / 1.4 um band-centre campaign:

```
python contact_band_center_map.py          # tier 0 per-pixel 2.3 um band-centre maps
python contact_band_center_modes.py
python contact_band_center_plebani.py      # replication on Plebani bland-ratioed spectra
python contact_band_center_1400.py
python contact_band_center_1400_plebani.py
python contact_band_matched_filter.py
python contact_oh_matched_filter.py
python contact_band_combinations.py
python contact_combo_subpop.py
python contact_combo_amplitude_control.py
python contact_combo_stacks.py
python contact_combo_mode_map.py
python contact_mode_block_test.py
python contact_mode_elevation.py
python contact_carbonate_patches.py
python contact_carbonate_fringe_overlap.py
```

The remaining `contact_*.py` scripts are diagnostics and figure variants;
each states its purpose in its docstring and writes a JSON log next to its
outputs naming the inputs it read.

Paper and supplement figures:

```
python contact_paper_fig7.py               # Figure 7 (intrusions at site 1)
python contact_paper_fig_site1.py          # four-panel site figure
python contact_fringe_gallery_v4_cassis.py # contact-margin gallery with pansharpened CaSSIS
python supplement_fig_bandcenter.py
python supplement_fig_carbonate.py
python supplement_fig_elevation.py
python supplement_final_figs_3E12.py
```

### Zenodo package

`code/release/` rebuilds the Zenodo record from the working copy
(`convert_pansharpened.py`, `convert_dems.py`, `export_ctx_dem.py`,
`export_geologic_map.py`, `build_crism_sections.py`; locations in
`release_config.py`).

### Overrides (optional)

- Mafic-cap denominator (display-only): `outputs/reports/denominator_overrides.json`
  e.g. `{"FRT0000A4FC/kmeans_k6_fixed": 3}` then rerun `stage_b_visualize.py`.

## Determinism

- All random-seeded operations use `RANDOM_SEED=42` (config.py).
- UMAP, k-means, GMM, HDBSCAN all take the seed. Reruns on the same
  input data reproduce bit-identical embeddings and labels on the same
  platform / numpy / umap-learn versions.
- Each log JSON records the source-file md5s used for that run — to
  diagnose drift, compare the `source_code_md5` field against
  `git log -- <file>` or local checksums.

## Regenerating a single figure

Every figure is written by exactly one script. To regenerate only one cube
or one method, pass the cube id as an argv:

```
python stage_b_visualize.py FRT00003E12
python stage_b_cluster_pca.py FRT0000871C
```

## Data provenance

- Brown v5 maps: pre-computed by Brown 2020 methodology, validated against
  Brown 2020 published results for FRT00003E12 (Stage A validation, PASS).
- VB parameters: computed from scratch on joined cubes using Viviano-Beck
  2014 formulae (code/vb_parameters.py). No CRISM-team browse products
  were used.
