# Contact-alteration test with the geologic map and DEMs (Stages 0-5)

Date: 2026-08-30 (v2; v1 was revised after an adversarial
verification pass - see section 7 for what changed and why). Plan:
`PLAN_revision_contact_3D.md`. Predecessor (2D, spectral-class-based, now
partly superseded): `outputs/revision/distance_to_contact/REPORT.md`.
Scripts: `code/contact_coreg_lib.py`, `contact_coreg_frame.py`,
`contact_geomap_fractions.py`, `contact_fraction_profiles.py`,
`contact_height_above_contact.py`, `contact_ridge_margins.py`,
`contact_block_stats.py`. Every output folder has a `run_log.json`.

Statistical convention: effect sizes are Cliff's delta with POSITIVE =
near-contact values higher; confidence intervals are 95 % block bootstraps
(504 m blocks, 1000 draws, seed 42). Pixel-level p-values are not quoted:
the effective sample size is ~1 % of the pixel count (variance inflation
~60-100x measured by block bootstrap).

## 1. Stage 0 - frame check

Tile-based phase correlation (high-pass 110 m, edge-NCC gate 0.3,
confidence-weighted median; adapted from the CaSSIS/M3 utility; sign
convention verified by a synthetic self-test AND independently by
brute-force NCC and by a HiRISE-ortho route) of each CRISM MTR3 cube
(749 nm) and each HiRISE ortho (6 m) against the USGS Robbins controlled
CTX mosaic (MC13, 6 m).

| target | dx (m) | dy (m) | MAD x/y (m) | good tiles |
|---|---:|---:|---|---|
| FRT00003E12 | -363.6 | +54.0 | 49 / 20 | 99/99 |
| FRT0000871C | -358.2 | +106.2 | 47 / 23 | 92/92 |
| FRT0000A4FC | -304.2 | -14.4 | 50 / 18 | 97/97 |
| FRT0000B438 | -325.8 | -61.2 | 65 / 23 | 95/98 |
| HiRISE DTM 006712/005855 | -75.7 | -105.9 | 3.6/5.4 | manual .IGwx: -78.4/-101.8; residual 3/-4 m |
| HiRISE DTM 002888/002176 | -107.1 | -123.4 | 6.1/4.2 | manual: -113.1/-112.1; residual 6/-11 m |
| HiRISE DTM 008993/009626 | -21.8 | +9.7 | 5.4/8.5 | at 72.25 E, outside all cubes; unused |

All four MTR3 cubes sit 300-360 m west (17-20 px) of the controlled CTX
frame, with cube-specific N-S offsets. Verified three ways: (i) the
pipeline reproduces the manual HiRISE world-file shifts to 3-11 m; (ii) an
independent integer-NCC and a CRISM-vs-HiRISE-ortho route give the same
offsets to ~2 px; (iii) a wrong sphere/datum handling would produce ~3 km
or ~14 km errors, not 360 m, and PROJ round-trips the two Mars sphere CRSs
at 0.0 m. Applying the shift raises map-vs-GMM class agreement in every
cube and every unit (3E12 clay 0.59->0.79, cap 0.71->0.82, oliv 0.79->0.89,
dune 0.65->0.79; B438 0.55->0.65 / 0.53->0.65 / 0.76->0.86 / 0.72->0.80;
871C clay 0.29->0.42, cap 0.12->0.18, oliv 0.80->0.90; A4FC 0.25->0.28 /
0.18->0.20 / 0.82->0.90, dune 0.68->0.90 on n=335). Absolute clay/cap
agreement in 871C/A4FC stays low - the 3E12-fit GMM classifies those cubes
poorly (known v9.1 issue), which is also why class-based tests are weakest
there.

The shifts are corrections to the georeferencing of
`outputs/_v8_mtr3_with_brown/<cube>/geotiffs/*.tif` (UL corners as in
`frame/coreg_offsets.json`); the ENVI vs PDS-label readings of the MTR3
grid itself differ by +-0.5 px, and within-scene MAD is ~3 px, so +-1-2
CRISM px stays unresolved everywhere below.

Consequence for the manuscript: every CRISM-on-HiRISE/CaSSIS overlay at
Horseshoe (Fig. 4d, Fig. 6c) is misplaced by ~20 CRISM pixels unless a
comparable shift was applied in ArcGIS.

## 2. Stage 1 - map-controlled distance profiles

Setup: unit polygons rasterised at 3 m (stacking dune > cap > clay >
graben > olivine remainder), averaged to 18 m on the shifted grid; contact
defined from the map (>= 50 % mapped clay), olivine population = map-pure
pixels (>= 90 % olivine unit, zero mapped clay, not GMM-clay, > 90 m from
scene edge and from the other contact). Map covers 3E12 fully; 871C, A4FC,
B438 partly.

Median D2300 vs distance from the mapped clay contact (both <= 36 m bins
are sub-pixel-mixing-affected; 54 and 55-90 m are at the registration
limit):

| distance (m) | 3E12 | 871C | A4FC | B438 |
|---|---|---|---|---|
| 18* | 0.0164 | 0.0111 | 0.0206 | 0.0134 |
| 36* | 0.0157 | 0.0112 | 0.0192 | 0.0126 |
| 54 | 0.0153 | 0.0111 | 0.0184 | 0.0119 |
| 55-90 | 0.0145 | 0.0110 | 0.0171 | 0.0114 |
| 90-150 | 0.0148 | 0.0107 | 0.0148 | 0.0103 |
| 150-250 | 0.0139 | 0.0107 | 0.0137 | 0.0081 |
| 250-400 | 0.0132 | 0.0111 | 0.0131 | 0.0072 |
| 400-700 | 0.0116 | 0.0112 | 0.0151 | 0.0067 |
| 700-1200 | 0.0111 | - | - | 0.0052 |

Block-level near (36-150 m) minus far (> 400 m) differences
(`stage5_stats/block_bootstrap_near_far.csv`):

| cube | contact | D2300 diff [CI] | delta | BD1900R2 diff [CI] | delta | BD2500_2 diff [CI] |
|---|---|---|---|---|---|---|
| 3E12 | clay | +0.0035 [+0.0023,+0.0047] | +0.23 | +0.0051 [+0.0042,+0.0061] | +0.44 | +0.0012 [+0.0009,+0.0015] |
| 3E12 | cap | -0.0039 [-0.0049,-0.0028] | -0.29 | -0.0003 [-0.0012,+0.0005] | -0.00 | -0.0002 [-0.0005,+0.0002] |
| B438 | clay | +0.0048 [+0.0036,+0.0059] | +0.41 | +0.0053 [+0.0038,+0.0066] | +0.39 | +0.0001 [-0.0004,+0.0005] |
| B438 | cap | -0.0015 [-0.0024,-0.0005] | -0.15 | -0.0012 [-0.0025,-0.0000] | -0.11 | +0.0004 [+0.0001,+0.0008] |
| 871C | clay | -0.0003 [-0.0036,+0.0015] | -0.04 | +0.0010 [-0.0012,+0.0023] | +0.13 | -0.0007 [-0.0012,-0.0001] |
| 871C | cap | +0.0016 [+0.0006,+0.0026] | +0.24 | +0.0022 [+0.0014,+0.0029] | +0.35 | +0.0000 [-0.0004,+0.0004] |
| A4FC | clay | +0.0011 [-0.0015,+0.0040] | +0.10 | +0.0024 [+0.0007,+0.0040] | +0.26 | -0.0003 [-0.0009,+0.0003] |
| A4FC | cap | +0.0020 [+0.0000,+0.0039] | +0.20 | +0.0021 [+0.0009,+0.0031] | +0.25 | +0.0000 [-0.0005,+0.0005] |

Within-block paired Wilcoxon supports the 3E12 and B438 clay-contact
differences (p 0.0016 / 0.0065 on 66 / 32 paired blocks). Overlap split:
3E12 and B438 image ~60 % common ground, but the clay-contact D2300
difference holds separately inside and outside the overlap in both cubes
(B438 outside: +0.0056 [+0.0034,+0.0073]; 3E12 outside: +0.0033
[+0.0017,+0.0048]).

Reading, with the verification's corrections built in:
- In the area the map covers well (northern Horseshoe; cubes 3E12 and
  B438, partly overlapping), map-pure olivine pixels are ~30-40 % higher
  in D2300 and BD1900R2 within 150 m of the mapped clay contact than
  beyond 400 m, declining over 100-700 m. The effect is medium
  (delta 0.2-0.4), robust at block level, and present independently in
  the non-overlapping parts of both cubes.
- 871C and A4FC are INCONCLUSIVE for the clay contact (few independent
  far-field blocks under the mapped area; CIs straddle zero and include
  the 3E12 effect size), not negative.
- The cap contact is NOT a null control: D2300 shows a mirror-signed
  gradient toward the cap in 3E12 (-0.0039) and B438 (-0.0015). Under a
  margin-contamination reading, map-pure olivine pixels inherit the
  spectral character of the neighbouring unit over ~100-250 m at BOTH
  contacts (hydrated at clay, anhydrous at cap). Under an alteration
  reading, the olivine is hydrated near the water-bearing unit and not
  near the anhydrous one; the cap D2300 mirror follows from dilution
  either way (BD1900R2 shows no cap gradient in 3E12, but the cap-olivine
  BD1900R2 contrast is near zero, so mixing predicts that too - it does
  not discriminate).
- Partial correlations (verification): controlling elevation, 1 um
  centroid, brightness roughly halves the clay-contact Spearman
  (3E12 -0.19 -> -0.10; B438 -0.30 -> -0.17). About half the gradient is
  entangled with margin covariates; a contact-specific residual remains.
- BD2500_2 declines toward far-field in 3E12 (+0.0012 near-far, CI
  excludes zero) with ~the same relative effect as D2300; flat in the
  other cubes. BD2500_2 is negative everywhere (no resolved 2.5 um band),
  so this is a continuum-slope change, not a carbonate detection - but
  "no relationship to the contact anywhere" is NOT a correct summary for
  3E12.
- Implied sub-map-scale clay if the whole gradient were unmapped smectite
  cover (linear mixing, per-cube clay endmembers D2300 0.040 / 0.024):
  ~9 % areal for 3E12, ~20-30 % for B438. Metre-scale patches adding to
  20-30 % of area inside 1:1000-mapped "olivine" polygons would be hard
  to miss, which weighs against pure unmapped-cover in B438.

## 3. Stages 2-3 - vertical geometry (corrected)

The v1 conclusion ("onlap against clay highs", median h < 0) was an IDW
artefact: interpolating contact elevations over 1 km across a regional
slope makes distant pixels look "below the contact". Verified replacement,
the contact-step test (`stage5_stats/contact_step.csv`): pairing the clay
side and the olivine side of every mapped contact cell,

| grid | band | n pairs | median z(clay)-z(oliv) | oliv higher >2 m | clay higher >2 m |
|---|---|---:|---:|---|---|
| CTX 18 m, 3E12 | 18-36 m | 5777 | -2.5 m | 53 % | 23 % |
| CTX 18 m, 3E12 | 54-108 m | 1124 | -10.5 m | 66 % | 24 % |
| CTX 18 m, B438 | 18-36 m | 4333 | -2.0 m | 50 % | 28 % |
| CTX 18 m, 871C | 18-36 m | 2853 | -0.9 m | 43 % | 20 % |
| CTX 18 m, A4FC | 18-36 m | 4593 | -0.1 m | 35 % | 23 % |
| HiRISE 006712, 3 m | 6-24 m | 19732 | -0.7 m | 35 % | 18 % |
| HiRISE 006712, 3 m | 30-60 m | 3253 | -3.3 m | 55 % | 28 % |
| HiRISE 002888, 3 m | 6-24 m | 4794 | -1.4 m | 45 % | 22 % |
| HiRISE 002888, 3 m | 30-60 m | 996 | -6.4 m | 65 % | 31 % |

The olivine side stands HIGHER than the clay side at the mapped contact
(by ~1 m at the contact, 3-12 m tens of metres away), in the median and
by pair counts, on both DEM sources - as expected for an olivine unit
overlying the clay basement. Clay-higher contacts exist (~20-30 % of
pairs) but are the minority.

Height-above-contact profiling beyond ~150 m from a contact is not
usable (IDW extrapolation); within its usable range the h-profiles say
only that the near-contact D2300 excess is confined to |h| <~ 15-20 m of
the contact elevation and is symmetric in |h| - it tracks relief, not
stratigraphic depth. The "thickness" estimate from lateral contacts is
not usable (negative in two cubes, 0-5 m in the others, all within its
own noise). We therefore make NO claim about vertical confinement in
either direction, and Mandon et al.'s tens-of-metres argument is neither
confirmed nor contradicted by these data.

## 4. Stage 4 - HiRISE profiles across catalogued intrusions (exploratory)

21 catalogued intrusion points inside the 006712/005855 DTM. Swath-median
profiles perpendicular to the local ridge strike, far-field detrended:
nine broad ridges 6-15 m high, ~50-100 m wide (sites 30-35, 39, 40, 52);
five in shallow troughs with no positive relief (36-38, 41, 48); the rest
at DTM noise level. Orthoimage brightness peaks on the flanks at a few
sites (30, 32, 35, 52) but no systematic indurated-flank step at 20-40 m.
Not diagnostic either way. Site-level metrics and galleries in
`stage4_ridges/`.

## 5. Bottom line

1. Registration: CRISM MTR3 at Horseshoe needs a ~300-360 m westward
   shift to sit in the CTX/HiRISE frame. Fact, three-way verified;
   affects manuscript overlays.
2. A clay-contact-associated enhancement of 2.3 and 1.9 um band depths in
   map-pure olivine-unit pixels exists in the northern map area (3E12 and
   B438, including their non-overlapping parts), amplitude ~0.004-0.005,
   decaying over 100-700 m, robust at block level. 871C/A4FC are
   inconclusive under the mapped area. About half the signal survives
   control for elevation/centroid/brightness covariates.
3. The same population shows a mirror-signed D2300 gradient at the cap
   contact, so "olivine pixels inherit the neighbouring unit's spectral
   character near every margin" (sub-map-scale cover, regolith transport)
   remains a viable alternative to in-situ alteration. CRISM cannot
   separate these; the HiRISE IRB colour fraction inside map-pure pixels
   vs distance is the next discriminating test and the data for it are on
   disk.
4. The 2.5 um carbonate band is not detected in the olivine unit at all
   (BD2500_2 < 0 everywhere); its continuum slope shows the contact
   gradient only in 3E12. No carbonate-at-the-contact claim can be made.
5. The olivine unit stands above the clay at the mapped contact
   (contact-step test, both DEMs) - consistent with the manuscript's
   stratigraphy; the earlier "onlap against clay highs" reading is
   withdrawn.
6. No claim about temperature, timing, vertical confinement, unit
   thickness, or indurated margins is supported by these data.

## 6. Summary statement

"In the northern part of the mapped area, covered by two overlapping CRISM
observations, the 2.3 and 1.9 um band depths of olivine-unit pixels that
contain no mapped clay are ~30-40 % higher within 150 m of the mapped
clay-basement contact than beyond 400 m, declining over 100-700 m
(block-bootstrap 95 % CIs exclude zero; the pattern holds in the
non-overlapping parts of both observations). A mirror-image decline of the
2.3 um parameter occurs toward the anhydrous mafic-cap contact, as
expected for sub-pixel mixing at any unit margin, and about half of the
clay-contact gradient is entangled with margin covariates; the remainder
is consistent with, but not unique evidence for, alteration of the olivine
adjacent to the water-bearing basement. These data do not constrain the
temperature or timing of that alteration."

## 6b. Stage 6 - Plebani-classifier carbonate patches vs the clay contact

A separate archive-wide carbonate project had already run
the carbonate-only Plebani et al. (2022) crism_ml HBM ensemble (classes
Ca/Fe CO3 and MgCO3, threshold 0.5, >= 5 px regions, version
carb-plebani-1.0, 2026-07-17) on the TRR3 cubes of FRT00003E12, FRT0000871C
and FRT0000B438 (A4FC's TRR3 was not in that archive), with a 3E12
validation against a ground-truth mask (region-filtered lift 45.7). 3E12
also passes all three of that project's independent carbonate methods
(clustering, rule-based detector, Plebani).

`code/contact_carbonate_patches.py` projects the region-filtered carbonate
pixels through the sensor-space DDR lat/lon into the shifted CRISM grid and
measures their distance to the mapped clay contact
(`stage6_carbonate/`):

| cube | carb px on map | patches | patch median dist | background median | within 150 m (carb vs bg) | patch-level KS p |
|---|---:|---:|---:|---:|---|---|
| 3E12 | 762 | 126 | 143 m | 364 m | 52 % vs 20 % | 8e-13 |
| B438 | 766 | 103 | 36 m | 268 m | 72 % vs 30 % | 4e-26 |
| 871C | 327 (of 1354; map covers a corner) | 22 | 150 m | 144 m | 50 % vs 52 % | 0.88 |
| A4FC | 90 (of 567; map covers the west only) | 19 | 0 m | 130 m | on-map sample boundary-dominated | 2e-05 (not meaningful, see below) |

A4FC (added 2026-08-30 after downloading its TRR3 from PDS and running the
same carb-plebani-1.0 pass: 22 regions, 567 px, max MgCO3 0.82): its two
largest carbonate clusters lie in the unmapped eastern part of the scene;
the 19 on-map patches hug the map boundary and mapped clay (median
distance 0 m, 53 % of centroids on mapped clay), so the on-map distance
statistic is boundary-dominated and A4FC is INCONCLUSIVE for this test,
like 871C - though 95 % of its patches still fall in GMM class c0.

Context: 75-91 % of patch centroids sit on map-pure olivine unit; 82-100 %
fall in the GMM altered-olivine class (c0), cross-validating the
clustering; their BD2500_2 (median -0.003 to -0.001) is the extreme tail of
the olivine-unit distribution (background median -0.007 to -0.014) - i.e.
carbonate IS present as discrete patches (~0.3-0.5 % of the scene) even
though the median olivine pixel shows no resolved 2.5 um band. The map
figure for 3E12 shows the patches tracing the olivine-unit margins around
the clay mesas and the horseshoe rim; a minority (~10-20 %) sit on mapped
clay or cap at 18 m scale (thin covers or residual registration).

This changes section 5 point 4: carbonate patches exist and, in the two
northern cubes, are concentrated toward the clay contact (2-7x closer than
the olivine-unit background); 871C is flat on the small mapped corner.
Caveats: patch-level tests still assume patch independence (patches
cluster along contacts); the same margin-covariate ambiguity as for the
hydration gradient applies; and enrichment toward the contact is equally
consistent with contact-driven alteration and with erosional exposure of a
basal carbonate-bearing level at the unit margins.

## 6c. Stage 7 - contact "fringes" in the HiRISE DTMs

Motivated by AV's recollection of subtle uplifted, fold-like fringes along
the olivine/clay contact. `code/contact_contact_fringe.py` stacks 17,610
contact-normal 1 m DTM profiles (+-100 m, +-6 m swath, ~10 m spacing along
every mapped clay/olivine contact in the two Horseshoe DTMs), detrending
each side on its 40-100 m far field (`stage7_fringe/`).

Result: the clay surface systematically rolls up toward the contact -
median detrended residual reaches +1.2 m (006712 DTM) and +1.7 m (002888)
over the last ~25-40 m, far above the 0.3-0.8 m far-field rms; the olivine
side is a flat bench with no bump. Per-profile: 68 % have a clay-side bump
>= 1 m within 2-30 m of the line (median amplitude 2.0-2.4 m, median
position -5 to -10 m). Present in 4 of 5 spatial clusters of the northern
DTM contact.

Discrimination: bump amplitude correlates with the local olivine step
height (Spearman 0.26-0.45), so part of the population is scarp-related
(talus / erosional shoulder). But at FLAT olivine edges (|step| <= 0.5 m;
n = 1,765 profiles) the clay-side bump persists (median 1.7-2.0 m, 64-69 %
>= 1 m, peak at ~-12 m), where neither talus nor a protective caprock
shoulder is available; the flat-edge galleries show discrete raised rims
2-4 m high and ~10-30 m wide that drop off on both sides
(`fig_fringe_flatedge_*.png`). This is topographic evidence for a
deformed and/or indurated clay margin along the contact - consistent with
forceful intrusive emplacement (marginal folding) or with fluid
alteration/cementation of the wall rock; not distinguishable from the DTM
alone. DTM matching artefacts are unlikely at this amplitude and coherence
(0.3 m far-field rms in the northern DTM). This is the morphological
counterpart the manuscript's contact story was missing; candidate for a
main-text or supplement figure.

v2 update (after AV flagged TIN-interpolation facets in the first
amplitude-ranked 006712 gallery): a facet detector (median |Laplacian| of
the 1 m DTM in a 60 m box; interpolation fill is piecewise planar,
threshold 0.05 m) flags only 0.8 % of the 006712 sites and none in 002888,
and the population statistics are unchanged after excluding them
(flat-edge clay-side bump medians 1.71 / 2.02 m; 64-69 % >= 1 m). The
amplitude ranking of the first gallery had sorted fill areas to the top;
the v2 galleries (`fig_gallery_v2_{flat,strong}_*.png`) exclude them, add
a CTX 6 m column and overlay the hand-drawn clay contact on both
thumbnails, with panels labelled P (002888 flat-edge), Q (006712
flat-edge), R/S (strongest). Per-site noise metric in
`fringe_metrics_v2.csv`; noise-filtered stats in `summary_v2.json`.

## 6d. Stage 8 - automated slope-break lineament scan (project specification)

`code/contact_slope_break_scan.py`: within 100 m of the mapped clay
contact, detect continuous lineaments of abrupt slope excess (2 m grid,
slope >= 12 deg AND >= 7 deg above the 50 m local median, TIN-fill and
DTM-footprint-edge exclusion, skeleton length >= 100 m, width <= 25 m).

- Northern DTM (002888): 19 lineaments, 2.8 km total, median length
  142 m, median distance to the contact 22 m, 47 % parallel to the local
  contact within 30 deg. The best (ids 1366, 1149, 4265, 4871, 6191) are
  crisp scarp-face lineaments 120-230 m long running along the contact -
  the automated counterpart of the stacked-profile fringe.
  (`stage8_slopebreaks/fig_overview_*`, `fig_thumbs_*`,
  `slope_break_lineaments.csv`.)
- Southern DTM (006712): NOT usable. Its clay unit is pervasively
  TIN-interpolated (stereo-correlation fill), and facet-crease edges
  mimic slope-break lineaments; all 1,129 detections there are junk and
  are retained in the CSV only for the record. Corollary: the Stage 7
  profile statistics for 006712 should be treated with more caution than
  the 0.8 % site-level flag suggested (the per-site 60 m-box median
  |Laplacian| under-detects mixed real/fill boxes); the 002888 DTM is the
  trustworthy source, and its flat-edge fringe statistics (median 2.0 m,
  69 % >= 1 m) stand on clean ground.

## 7. Changes from v1 after adversarial verification

- Withdrawn: "onlap against clay highs / olivine in the lows" (IDW
  artefact; replaced by the contact-step test with the opposite result).
- Withdrawn: "no counterpart at the cap contact" (mirror-signed D2300
  gradient exists) and "carbonate flat" for 3E12.
- Withdrawn: "absent in 871C / weak in A4FC" -> inconclusive; "2 of 4
  cubes" -> "the northern map area, in both cubes covering it (~60 %
  overlap), including their non-overlapping parts".
- Replaced pixel-level p-values with block bootstrap CIs and within-block
  paired tests; effect sizes now standard-sign Cliff's delta.
- Fixed: cap exclusion missing from the height population; unused
  `others` mask in contact_pixels(); QC quiver sign; Olivine_unit CRS in
  the HiRISE contact-step; documented which georeferencing the Stage 0
  shifts correct.
- Downgraded: thickness and vertical-confinement statements to "not
  usable / no claim".
