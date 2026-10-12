# Project context — Horseshoe site reanalysis

**Investigator:** Adomas (AV) Valantinas
**Date:** April 2026
**Scope:** Tier 1 — exploratory reanalysis of the olivine unit using modern
unsupervised methods across four independent CRISM observations.

---

## 1. The site

The "Horseshoe" is a horseshoe-shaped exposure of olivine-bearing bedrock in
western Nili Fossae. It has been the type locality for Mars olivine-carbonate-
phyllosilicate mineralogy since the first CRISM observations, and has been
interpreted in five distinct ways by different authors. The site sits upslope
and west of Jezero crater; the Perseverance rover has been exploring related
lithologies since 2021.

## 2. The four cubes

Four CRISM FRT (full-resolution targeted) observations cover different but
partially overlapping swaths of the Horseshoe, acquired on different MRO
orbits and therefore under different atmospheric and illumination conditions.

| Cube ID          | Role in this analysis                        |
|------------------|-----------------------------------------------|
| FRT00003E12      | Validation cube. The most-studied CRISM cube at this site. |
| FRT0000871C      | Independent replication                        |
| FRT0000A4FC      | Independent replication                        |
| FRT0000B438      | Independent replication                        |

FRT00003E12 is the cube used by Mustard 2009, Ehlmann 2009, Brown 2010,
Viviano 2013, Brown 2020, and Kremer 2019. Any cluster structure found in
3E12 can be cross-referenced against these published interpretations.

The cubes are **unprojected** (raw detector/sensor coordinates). Cross-cube
pixel-matching is therefore NOT possible without MRDR projection — so the
replication test must be **distributional** (does the same cluster structure
emerge in all four cubes?), not pixel-wise.

## 3. Five competing hypotheses for the olivine unit

The analysis should be framed against these five published interpretations.
They are not mutually exclusive in all cases, but some are in direct conflict.

1. **Primary igneous lava flow** (Mustard 2007, 2009)
   Regional Mg-rich olivine-bearing volcanic deposit, possibly ultramafic.

2. **Serpentinization** (Ehlmann 2009)
   Partial aqueous alteration: olivine + H₂O → serpentine + magnetite + H₂.
   Predicts OH signature near 2.32 µm coupled with the 1 µm olivine band.

3. **Hydrothermal clay-carbonate alteration** (Brown 2010)
   Proposes that the Mg/Fe-carbonate + phyllosilicate (possibly talc)
   assemblage associated with the olivine unit formed via a single
   hydrothermal event, with the argillic (phyllosilicate-rich) and
   propylitic (carbonate-rich) zones reflecting different temperatures in
   the same fluid system, analogous to the Archean Warrawoona Group in
   Western Australia. Evidence: asymmetric-Gaussian fits of the 2.3 and
   2.5 µm carbonate bands + Earth-analog mineralogy. Distinct from
   Ehlmann's serpentinization in process and predicted assemblage, though
   the two hypotheses are not mutually exclusive. NB: this is about the
   *2.3/2.5 µm carbonate band shape*, not the *1 µm olivine band* — the
   asymmetric-Gaussian fitting of the 1 µm olivine band is a separate
   later methodology (Brown et al. 2020) that produced the `v5` maps in
   this project and is a tool, not a hypothesis.

4. **Compositional transition via talc** (Viviano 2013)
   Smooth mineralogical gradient from phyllosilicate-bearing bedrock into the
   olivine unit, with talc as the intermediate phase. Implies a continuous
   gradient rather than a sharp contact.

5. **Impact ejecta / airfall from Isidis** (Kremer 2019)
   CRITICAL. The olivine unit is a draped airfall deposit, NOT in-situ igneous
   material. This hypothesis is directly incompatible with any interpretation
   that invokes in-situ igneous emplacement, including the intrusive
   magmatism framing of Valantinas et al. (2026, EPSL).

   Predictions of Kremer's hypothesis (derive the specifics by reading the
   paper — these are guidelines):
   - Homogeneous olivine composition across the unit
   - Draping contact with underlying bedrock (no gradational boundary)
   - No systematic relationship between olivine parameters and local
     topography beyond draping

Valantinas et al. (2026, EPSL) propose **intrusive magmatism** for
olivine-bearing ridges within the unit — which is directly incompatible with
Kremer. Ruling in or out the airfall hypothesis is the single most important
scientific question this analysis can address.

## 4. Prior findings from my own analysis (anchor points)

Three findings now anchor the cross-cube tests. The first two are from
earlier PCA + interactive class extraction on FRT00003E12; the third is
a Stage A result from the current four-cube reanalysis.

1. **Ridge-associated pixels are olivine-carbonate rich** — they occupy a
   distinct region of the PC1-PC2 space with simultaneously elevated OLINDEX3
   and carbonate-band parameters.

2. **Potential alteration halos around the olivine unit** — transition zones
   at the margins of the olivine unit show intermediate parameter values
   between "clean olivine" and "phyllosilicate-bearing bedrock." Compatible
   with (but not proof of) Viviano 2013's talc-transition framing.

3. **Cross-cube centroid variability (Stage A, April 2026)** — the median
   1 µm olivine band centroid shifts by ~90 nm across the four FRT
   observations of the Horseshoe (871C 1.198 µm, A4FC 1.243 µm,
   B438 1.268 µm, 3E12 1.290 µm). Residuals of the Brown v5 fit are
   *lowest* in 871C, which is the most blueshifted cube, so the offset
   is not attributable to atmospheric state or fit quality. Indicates
   real compositional and/or grain-size heterogeneity within the unit.
   This is inconsistent with the strict form of Kremer 2019's
   compositional-homogeneity prediction, though substrate-dependent
   grain-size variation within an airfall deposit cannot be excluded by
   centroid alone.

Findings 1 and 2 should be **tested for replication** in cubes 871C, A4FC,
B438 in Stage C/E. Finding 3 should be dissected in Stage D/E by matching
sub-populations across cubes and asking whether the centroid heterogeneity
survives on matched populations — distinguishing "different parts of the
unit look different" from "the same mineralogical population has different
centroids in different cubes."

## 5. Data products available in this project

Per cube, in `cubes/<CUBE_ID>_07_IF166J_TER3/`:

- **Joined reflectance cube:** `*_if166j_ter3.img` + `.hdr` + `.lbl`
  TER3-corrected, VNIR+IR joined. This is what spectral analysis runs on.
- **Brown v5 parameter maps** (asymmetric Gaussian fits of the 1 µm olivine
  band; Brown et al. 2020 methodology):
  - `*_centroid_map_v5.npy` — band center (related to Fo#)
  - `*_asymmetry_map_v5.npy` — band asymmetry (grain size / composition)
  - `*_amplitude_map_v5.npy` — band depth
  - `*_hwhm_map_v5.npy` — half-width half-maximum
  - `*_mask_map_v5.npy` — valid-pixel mask
  - `*_residual_map_v5.npy` — fit quality

**Provenance note:** the Brown v5 maps have been validated against Brown 2020
published results for FRT00003E12. They are considered trusted inputs and
should NOT be recomputed.

**Viviano-Beck browse products were deliberately deleted** from the cube
folders. The analysis must compute its own spectral parameters from the
joined cubes, to avoid anchoring on CRISM-team RGB composite interpretations.

## 6. Reference library

Papers consulted (the PDFs are not part of this repository):
- Valantinas et al. (2026, EPSL) — the hypothesis being tested
- `Kremer_2019_olivine_unit_airfall_origin.pdf` — the strongest competitor
- `Ehlmann_2009_*.pdf`, `Brown_2010_*.pdf`, `Viviano_2013_*.pdf` — the other
  published hypotheses
- `Viviano-Beck_2014_CRISM_spectral_parameters.pdf` — parameter definitions
- `Brown_2020_olivine_asymmetric_gaussian.pdf` — method for existing v5 maps

## 7. Prior code

See `code/` for existing scripts. The most relevant is:

- `PCA_CRISM_analysis_v2.py` — full end-to-end PCA workflow for a single
  cube: loads TER3 cube, loads Brown v5 maps, computes OLINDEX3, D2300,
  BD1900R2, BD2500_2, builds an 8D feature stack, runs PCA, supports
  interactive rectangle-based class definition on PC1-PC2, extracts mean
  I/F spectra per class, supports ratioing. This is the starting point.
  Extend it; do NOT rewrite it from scratch.

- `1um_centroid_analysis_CRISM_TER3_v1.py` / `_MTR3_v1.py` — Brown asymmetric
  Gaussian fitting (produced the v5 .npy maps). Reference only; not to be re-run.

- `crism_spectra_extractor_joined.py` — interactive spectral extractor with
  ROI lasso and polynomial continuum-removed band fitting.

## 8. Success criteria for Tier 1

At the end of this exploration, you should be able to report:

1. **Which parameter combinations** best separate the pixel populations in the
   feature space? Is the existing 8-parameter stack (Brown 4 + VB 4) optimal,
   or does adding BD2290, SINDEX2, LCPINDEX2, BD1400, or MIN2295_2480 improve
   cluster separation?

2. **How many populations** of pixels are present in each cube? Does the
   number replicate across cubes?

3. **Where in feature space** does each population sit? For each population,
   what's its Brown centroid / asymmetry signature, and what does its mean
   ratioed I/F spectrum look like between 0.4 and 2.6 µm?

4. **Where in the cube** does each population sit? Spatially coherent blobs?
   Linear features? Halos around other populations?

5. **Are the two anchor findings replicated** across all four cubes?
   (Olivine-carbonate ridges; alteration halos at unit margins.)

6. **Are any of the five published hypotheses falsified or strongly supported**
   by the four-cube evidence? Specifically — can the Kremer airfall hypothesis
   be distinguished from the in-situ igneous hypotheses?

The deliverable is a markdown report in `outputs/reports/` with figures.
