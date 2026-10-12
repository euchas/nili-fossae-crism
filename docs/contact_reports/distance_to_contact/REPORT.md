# NOTE (2026-08-30): superseded in part

The map-and-DEM-controlled reanalysis (`outputs/revision/contact_3d/REPORT.md`)
supersedes the inference in this report. In particular: the gradient is
confirmed at block level only in FRT00003E12 and FRT0000B438 (871C/A4FC are
inconclusive, not negative); the cap contact is NOT a clean null control
(mirror-signed D2300 gradient); pixel-level p-values here ignore spatial
autocorrelation; and the CRISM MTR3 georeferencing is offset 300-360 m from
the controlled CTX frame. The distance transform here used spectral classes,
not the geologic map.

# Distance-to-contact test of olivine alteration at Horseshoe

Date: 2026-08-28. Scripts: `code/contact_distance_to_contact.py`,
`code/contact_band_center_2300.py`. Plan: `PLAN_revision_contact.md`.
Run logs: `run_log.json`, `run_log_band_center.json`.

## 1. Question

Is alteration/carbonate in the olivine unit concentrated at the contact
with the clay basement, or pervasive across the unit? Section 5.1 of the
submitted manuscript stated that carbonates are concentrated along the
contact with the clay basement, citing Viviano et al. (2013), but had not
tested this on our own data. This test asks, in four independent CRISM observations, whether
alteration indicators in olivine-bearing pixels depend on distance from
the nearest Fe/Mg-phyllosilicate pixel.

## 2. Method (summary)

Map-projected MTR3 grid, 18 m/px. Classes from the v9.1-B GMM (fit on
FRT00003E12, applied to all cubes with per-cube standardization):
clay = c4+c5, mafic cap = c1, olivine-bearing = c0+c2+c3, "altered
olivine" = c0 (highest D2300 and BD2500_2 of the olivine classes,
intermediate 1 um band shape). Euclidean distance transform to the
nearest clay pixel and to the nearest cap pixel. Olivine pixels within
90 m of the scene edge, or within 90 m of the *other* contact, are
excluded from each profile. Bins <= 36 m (1-2 px) are treated as
sub-pixel-mixing-affected and not used for inference. Parameters from
the v8 MTR3 stack (centroid, asymmetry, OLINDEX3, BD1400, BD1900R2,
D2300, BD2500_2). Three variants: all valid pixels; excluding "novel"
pixels; GMM-free olivine definition (OLINDEX3 above the per-cube 60th
percentile). All three give the same profiles (`distance_stats.csv`).

## 3. Results

### 3.1 The 2.3 um and 1.9 um features decay with distance from the clay contact in all four cubes

Median D2300 of olivine-bearing pixels by distance from the clay contact
(m), all-valid variant:

| distance | 3E12 | 871C | A4FC | B438 |
|---|---|---|---|---|
| 54 | 0.0229 | 0.0155 | 0.0261 | 0.0181 |
| 55-90 | 0.0201 | 0.0142 | 0.0239 | 0.0153 |
| 90-150 | 0.0156 | 0.0121 | 0.0198 | 0.0103 |
| 150-250 | 0.0131 | 0.0101 | 0.0154 | 0.0067 |
| 250-400 | 0.0113 | 0.0087 | 0.0129 | 0.0047 |
| 400-700 | 0.0086 | 0.0073 | 0.0126 | 0.0034 |

Spearman rho (distance vs D2300, 36-1200 m): -0.35, -0.46, -0.54, -0.49.
BD1900R2 behaves the same way (rho -0.40, -0.31, -0.57, -0.44). Near
(36-150 m) vs far (>400 m) rank-biserial effect sizes for D2300 are
-0.55 to -0.69 in every cube. The decline is monotonic over 100-400 m,
i.e. 5-20 pixels, far beyond the 1-2 px sub-pixel mixing zone.

The 1 um centroid rises with distance from clay (3E12: 1.32 um at 54 m
to 1.37 um beyond 400 m), i.e. the olivine band is most saturated in the
interior of the exposures. The same centroid trend exists vs. distance
from the cap, so it is not specific to the clay contact.

### 3.2 No comparable structure at the mafic-cap contact

Against distance from the cap (olivine pixels > 90 m from clay), D2300
and BD1900R2 are flat or *increase* with distance (the far-from-cap
pixels are, on average, nearer to clay). Near-cap olivine does not carry
elevated hydration. The clay-contact gradient is therefore not a generic
edge or resampling effect.

### 3.3 The 2.5 um carbonate band is NOT contact-concentrated

Median BD2500_2 vs distance from clay: 3E12 shows a weak maximum at
55-150 m (-0.0132 vs -0.0152 beyond 400 m; rank-biserial -0.35); B438 a
marginal near-contact enhancement (-0.10); 871C none (+0.10); A4FC the
opposite (+0.11; carbonate band slightly stronger away from clay). On
the specific point of *carbonate*, a "pervasive, not contact-related"
description is consistent with our data: BD2500_2 is weak everywhere and shows no
replicated relationship to the clay contact. The manuscript sentence
attributing contact-concentrated carbonate to Viviano et al. (2013)
should be removed or rewritten; Viviano's Fig. 8 (FRT00003E12) actually
documents carbonate decreasing and talc increasing *down-section* toward
the basement, which is a different statement.

### 3.4 The "altered olivine" class (c0) forms a halo in 3E12 and B438 but is pervasive in 871C and A4FC

Fraction of olivine-bearing pixels in c0 at 90-150 / 250-400 / 400-700 m:
3E12 0.21 / 0.05 / 0.02; B438 0.21 / 0.11 / 0.13; 871C 0.60 / 0.47 / 0.48;
A4FC 0.89 / 0.89 / 0.91. In 3E12 the near/far ratio is 20. In 871C and
A4FC the c0 class covers most olivine exposures regardless of distance.
Part of this is a known cross-cube artefact (871C/A4FC have globally
blueshifted 1 um centroids, Stage A; c0 is partly defined by a lower
centroid, so a 3E12-fit GMM over-assigns c0 there), which is why the
continuous D2300/BD1900R2 profiles (which replicate in all four cubes)
are the more reliable result. But it is also a caution: "halo" is the
right description for 3E12 and B438, not obviously for 871C and A4FC.

### 3.5 Is the near-contact 2.3/1.9 um signal smectite admixture or an alteration phase? Inconclusive at CRISM SNR

Two discriminants were tried (`band_center_2300.csv`,
`excess_ratio_1900_2300.csv`):

- 2.3 um band centre (continuum 2.21-2.38 um, quadratic-refined minimum).
  Interior clay: 2.307-2.313 um. Olivine 36-90 m from clay: 2.311-2.318
  um. Far olivine and cap: 2.317-2.324 um, but with band depths of
  0.007-0.010 that are at the residual/noise floor, so those "centres"
  are not mineralogical. The near-contact shift relative to clay is
  +2.7 to +5 nm, less than one CRISM channel (6.55 nm) and in the
  direction expected for *either* a serpentine/talc-like phase *or*
  dilution of the clay band into the background residual. Not
  diagnostic.
- BD1900R2 / D2300 excess ratio. Relative to a cap baseline the 1.9 um
  excess vanishes by 150-400 m while a 2.3 um excess persists (all four
  cubes), which would argue for a 2.3-without-1.9 phase; relative to a
  far-olivine baseline the near-contact ratio is at or above the clay
  ratio, which is what mixing predicts. The answer flips with the
  baseline because the excess depths are 0.002-0.005, comparable to the
  olivine-band-induced bias in BD1900R2. Not diagnostic.

Conclusion: CRISM band shape cannot, at this SNR and sampling, separate
"Fe/Mg-smectite present at sub-class-map scale within the olivine margin
zone" from "a hydrated Mg-phyllosilicate (saponite/talc/serpentine)
alteration phase in the olivine". Note that the expected alteration
products at this site (Brown 2010; Viviano 2013; the manuscript's own
Supp. Fig. 8 "talc/saponite mixture") are themselves hydrated Mg-OH
phases, so this ambiguity is intrinsic, not a failure of the test.

### 3.6 Elevation

c0 fraction and BD2500_2 vs MOLA elevation (CRISM DDR) show no
consistent trend beyond the fact that the highest-elevation olivine
pixels in 3E12 and B438 are also the ones nearest the clay. MOLA at the
18 m grid cannot address Mandon et al. (2020)'s tens-of-metres vertical
argument; that needs HiRISE DEM + outcrop-scale mapping.

## 4. What this does and does not establish

Established (replicated in 4 cubes, 3 pixel-selection variants):
1. Hydration (1.9 um) and Mg-OH/carbonate (2.3 um) signatures in
   olivine-bearing pixels are spatially structured with respect to the
   clay contact, decaying monotonically over 100-400 m. They are not
   "pervasive and unrelated to contacts" at Horseshoe.
2. No such structure exists at the mafic-cap contact.
3. The 2.5 um carbonate band does not share this structure.

Not established:
4. Whether the gradient records alteration of the olivine or the
   geometry of the unit margin (thin, dissected olivine cover over clay
   near the mapped contact, with unmapped clay windows at 18 m scale).
   Discriminating these needs the 0.25 m/px HiRISE-CaSSIS geologic map
   rasterised to the CRISM grid: regress D2300 on sub-pixel clay
   fraction, then test whether the distance dependence survives at
   fixed clay fraction. This is a few hours in ArcGIS + one script.
5. Temperature or timing. A contact-concentrated hydration halo is
   equally consistent with (a) syn-emplacement reaction between hot
   olivine-bearing magma and pore water in the smectite, and (b) later
   groundwater ponding on the clay aquiclude at the base of the unit.
   The CRISM data do not constrain temperature.

## 5. Outputs

- `distance_profiles.csv`, `distance_stats.csv`, `elevation_profiles.csv`
- `band_center_2300.csv`, `excess_ratio_1900_2300.csv`
- `fig_distance_profiles_{all_valid,excl_novel,olindex3_gt_p60}.{png,pdf}`
- `fig_contact_maps.{png,pdf}`, `fig_elevation_*.{png,pdf}`,
  `fig_band_center_2300.{png,pdf}`
- `run_log.json`, `run_log_band_center.json` (parameters, input md5s)
