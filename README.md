# nili-fossae-crism

[![Paper](https://img.shields.io/badge/paper-10.1016%2Fj.epsl.2026.120390-blue)](https://doi.org/10.1016/j.epsl.2026.120390)
[![Data](https://zenodo.org/badge/DOI/10.5281/zenodo.23163697.svg)](https://doi.org/10.5281/zenodo.23163697)
[![License: MIT](https://img.shields.io/badge/code-MIT-green.svg)](LICENSE)

Analysis code for *Valantinas, A., Mustard, J. F., Tullo, A., Bramble, M. S.
and Head, J. W. (2026), "An exposed magmatic feeder system for the regional
olivine unit in Nili Fossae, Mars", Earth and Planetary Science Letters 696,
120390*, [doi:10.1016/j.epsl.2026.120390](https://doi.org/10.1016/j.epsl.2026.120390):
olivine-rich ridges and intrusion-like features at the "Horseshoe" site in
western Nili Fossae, from CRISM spectroscopy, CaSSIS/HiRISE pansharpened
imagery and HiRISE/CTX topography.

**Data:** the companion dataset (geologic map, ArcGIS session, CRISM
parameter maps, 1 µm olivine band fits, contact-metamorphism outputs, 26
pansharpened CaSSIS-HiRISE scenes, HiRISE and CTX DEMs; 30 GB, CC BY 4.0) is
on Zenodo: **[doi:10.5281/zenodo.23163697](https://doi.org/10.5281/zenodo.23163697)**.
Source data are on the NASA PDS / ESA PSA; see `DATA.md`.

## Setup

```
git clone https://github.com/euchas/nili-fossae-crism
cd nili-fossae-crism
pip install -r requirements.txt
```

Unpack inputs beside `code/` (or point the environment variables in
`DATA.md` elsewhere):

```
nili-fossae-crism/
  code/                     (this repo)
  cubes/                    CRISM TER3 / MTR3 cubes        <- PDS
                            + Brown v5 maps                <- Zenodo section C
  data/A_geologic_map/ ...  all Zenodo sections, unzipped  <- Zenodo record
  data/pds_downloads/       HiRISE DTMs/orthos, CRISM TRR3/DDR, CaSSIS  <- PDS / PSA
  outputs/                  pipeline outputs (created)
```

All paths are set in `code/config.py`. Run scripts from `code/`. Developed
with Python 3.14 on Windows 11; random seed 42 throughout. Step-by-step run
order: `code/REPRODUCE.md`.

## Pipeline

```
CRISM TER3 / MTR3 (PDS)
  ├─ 1um_centroid_analysis_CRISM_{TER3,MTR3}_v1.py   1 µm olivine band fit (Brown et al., 2020)
  ├─ vb_parameters.py, stage_a_build_feature_stack.py
  │                                                  Viviano-Beck (2014) parameters
  └─ stage_b_*.py, stage_v3..v9_*.py                 per-cube PCA / UMAP / HDBSCAN / GMM
       ├─ stage_v7c_mtr3_pipeline.py, stage_v8_mtr3_with_brown.py
       │                                             map-projected (MTR3) classes
       └─ v9_1_brown_scatter.py                      band-shape space (Fig. 9)
          v9_1_ratio_spectra.py, v9_library_vs_crism.py
                                                     ratio spectra, library comparison

contact metamorphism (geologic map + DEMs + CRISM)
  contact_coreg_frame.py, contact_dtm_registry.py    CRISM / CTX / HiRISE frame check
  contact_geomap_fractions.py                        sub-pixel unit fractions
  contact_distance_to_contact.py, contact_height_above_contact.py
                                                     alteration vs distance / height
  contact_ridge_margins.py, contact_contact_fringe.py
                                                     HiRISE-scale margin profiles
  contact_band_center_*.py, contact_combo_*.py       2.3 / 1.4 µm band-centre tests
  contact_carbonate_patches.py                       carbonate patches vs the contact

figures
  map figures (Figs. 1-8)      ArcGIS Pro layouts, Zenodo section B
  supplement_fig_*.py          supplement figures
  contact_paper_fig7.py, contact_paper_fig_site1.py, contact_fringe_gallery_v4_cassis.py
                               figure panels rebuilt in the project frame

packaging
  release/                     builds the Zenodo record
```

`code/archive/` keeps superseded exploratory scripts for provenance.
`docs/` holds the scientific context (`CONTEXT.md`), the methodological
decision log (`DECISIONS.md`), the analysis plans as they were written
(`analysis_log/`), the text reports of every analysis stage (`reports/`) and
the reports and run logs of the contact-metamorphism tests added during the
revision of the paper (`contact_reports/`; their full outputs are Zenodo
section E).

## AI assistance

Much of the analysis code was written with an AI coding assistant (Claude,
Anthropic) under the direction of the authors, who reviewed the code and
results. `docs/DECISIONS.md` and `docs/analysis_log/` record the
methodological choices as they were made.

## Citing

Please cite the paper and, for the data, the Zenodo record:

> Valantinas, A., Mustard, J. F., Tullo, A., Bramble, M. S., Head, J. W. (2026).
> An exposed magmatic feeder system for the regional olivine unit in Nili
> Fossae, Mars. *Earth and Planetary Science Letters* 696, 120390.
> https://doi.org/10.1016/j.epsl.2026.120390

Machine-readable metadata for this repository are in `CITATION.cff`.
Code: MIT licence. Data: CC BY 4.0.
