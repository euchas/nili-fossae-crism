# Joint 1.4 / 2.3 (/ 2.5) um detection combinations - phase discrimination

Run 2026-09-07, Plebani bland-ratioed chain, all four cubes.
Scripts: `contact_band_combinations.py` (class maps, joint histograms),
`contact_combo_amplitude_control.py` (dilution control),
`contact_combo_subpop.py` (sub-population split, rim vs interior),
`contact_band_matched_filter.py 2500` (2.5 um per-pixel run),
`contact_combo_stacks.py` (stacked spectra of the sub-populations).
Products in `combinations/` and `matched_filter/2500/`.

## 1. Combination classes (`combo_counts.csv`, `fig_combo_map_{cube}`)

Per-pixel classes from the 1.4 um OH and 2.3 um metal-OH matched-filter
detections (same sensor pixels): neither / 1.4-only / 2.3-only / both.

| unit | 2.3-only | both | share of 2.3 detections that also have 1.4 |
|---|---|---|---|
| clay map-pure (3E12/871C/A4FC/B438) | 26 / 22 / 21 / 22 % | 20 / 7 / 14 / 6 % | **43 / 23 / 39 / 22 %** |
| olivine map-pure | 10.3 / 7.5 / 9.0 / 2.5 % | 0.4 / 0.2 / 1.2 / 0.1 % | **3.5 / 2.3 / 11.8 / 3.5 %** |
| cap map-pure | ~1 % | ~0 % | - |

Map pattern (3E12): 'both' fills the CORES of the clay polygons; '2.3-only'
forms the polygon rims AND the broad western olivine-unit swath.

## 2. Joint band-centre histogram ('both' pixels, 4 nm bins, `fig_combo_hist2d`)

One cluster only, in every cube and both units: (1.400-1.410, 2.296-2.300
um) - the Fe/Mg smectite locus. No population at the saponite, talc,
serpentine or chlorite loci. There is exactly ONE hydrated-phyllosilicate
phase in these scenes; the olivine unit's abundant '2.3-only' phase is
something else.

## 3. Dilution control (`combo_amplitude_control.csv`, `fig_combo_amplitude_control`)

Objection: '2.3-only' could be diluted smectite whose weaker 1.4 um band
fell below threshold (the polygon rims show exactly this). Test: 1.4 um
co-detection rate and median 1.4 um SNR at MATCHED 2.3 um amplitude.

3E12, 2.3 um amplitude 1.6-2.2 %: clay co-detects 1.4 um in 37 % (median
1.4 SNR 7.9) vs olivine 5.5 % (SNR 2.9); at 2.2-3.0 %: 55 % vs 10 %; at
3.0-4.5 %: 72 % vs 22 %. 871C and B438: a 5-15x gap in every bin. A4FC:
the gap closes at high amplitude (62 % vs 46 %) - A4FC's olivine-unit
detections behave like diluted clay, consistent with it being the cube
with the most inferred sub-pixel clay mixing.
=> In 3E12 / 871C / B438 the olivine-unit 2.3 um phase has far less OH per
unit of metal-OH band than the basement smectite. It is not diluted clay.

## 4. Sub-populations (`combo_subpopulations.csv`) and stacks (`combo_stack_metrics.csv`, `fig_combo_stacks_{cube}`)

The olivine '2.3-only' centre distribution is bimodal (3E12, 871C, A4FC):
modes at ~2.298-2.300 um and ~2.312-2.315 um (valley 2.305). Split there:

| group (3E12) | n | median dist. to clay | within 90 m | 1.4 SNR med | STACK 2.3 centre / depth | STACK 2.5 centre / depth | STACK 1.9 depth |
|---|---|---|---|---|---|---|---|
| olivine 2.3-only, < 2.305 | 5,639 | 285 m | 15 % | 2.9 | 2.293 / 1.7 % | **2.507 / 0.61 %** | 1.1 % |
| olivine 2.3-only, >= 2.305 | 5,432 | 285 m | 12 % | 1.2 | **2.318** / 1.2 % | 2.502 / 0.32 % | **-0.6 % (none)** |
| olivine both | 404 | 219 m | 28 % | 11.3 | 2.296 / 2.7 % | **2.506 / 1.57 %** | 2.6 % |
| clay 2.3-only | 13,240 | 0 | 100 % | 5.9 | 2.300 / 2.0 % | 2.515 / 0.20 % | 1.1 % |
| clay both | 10,165 | 0 | 100 % | 13.2 | 2.299 / 2.9 % | 2.506 / 0.21 % | 1.7 % |

The clay stacks are the 2.5 um control: 0.1-0.2 % depth is the systematic
floor. Against it: olivine '< 2.305' carries a 2.5 um partner at 0.61 %
(3E12), 1.00 % (B438), 0.45 % (871C) - 3-5x the floor - and at 0.22 % in
A4FC (floor; the mixing cube). 'Olivine both' carries the strongest 2.5 um
band of all (1.0-1.6 %). The '>= 2.305' group has a weak-to-floor 2.5 um
band and NO 1.9 um water in any cube (bd1900 -0.6 to +0.08 %).

Per-pixel 2.5 um detections (`matched_filter/2500`, thresholds SNR 18-20)
sit at the floor in every unit (0.2-0.6 %): the 2.5 um band is not
resolvable per pixel, consistent with the ledger's BD2500_2 < 0 result;
the stacks, not the per-pixel run, carry this test.

## 5. Provisional phase assignments (feature-space first; labels provisional per project rule)

- BASEMENT CLAY: 2.30 + 1.41 + 1.9 um, no 2.5 um -> Fe/Mg smectite. One
  phase, all cubes.
- OLIVINE UNIT, population A ('2.3-only', centre ~2.30, weak 1.4, 2.5 um
  partner, some 1.9): Mg-carbonate-bearing (magnesite-like), with a minor
  hydrated component. Interior of the unit (median 285 m from clay in
  3E12), the broader halo of the stage-6 carbonate patches ('olivine
  both' = the carbonate-rich cores, 2.5 um 1.0-1.6 %).
- OLIVINE UNIT, population B ('2.3-only', centre 2.31-2.32, no 1.9, weak
  1.39, 2.5 at/near floor): an anhydrous Mg-OH phase of the
  serpentine/talc family - the Ehlmann et al. (2009, Fig. 17D)
  "serpentinized olivine" signature (2.32 um, no 1.9 um). The 3E12 stack
  centre (2.318) favours serpentine; 871C/B438 (2.306) sit nearer talc;
  the ~13 nm inter-chain wavelength caveat forbids choosing.
- A4FC's olivine-unit detections are dominated by diluted basement clay
  (rim geography, clay-like co-detection, no 2.5 um).

## 6. What this changes

The olivine unit's 2.3 um signal is now resolved into named phases WITHIN
one processing chain, with the basement smectite as the internal control
at every step: it is carbonate + a serpentine/talc-family Mg-OH phase, not
sub-pixel clay contamination (3 of 4 cubes). This is the shape-verified,
three-band version of the 'contact-associated alteration' observation and
directly answers the 'you cannot decipher' class of objection.

## Caveats (must accompany any use)

- Single chain (Plebani bland-ratioed TRR3). The MTR3-chain unit-wide
  2.32 um signature and the dual-chain carbonate patches are consistent
  with it, but the three-band discrimination itself has not been
  replicated in a second chain.
- Stacks are descriptive; the 2.5 um floor is set by the clay control
  (0.1-0.2 %), not by a formal CI. Block-level statistics are required
  before any of this enters the manuscript (project convention).
- Per-pixel class maps carry n_eff ~1 %; residual column striping exists
  in the ratioed chain.
- Absolute band centres carry the ~13 nm inter-chain offset: serpentine vs
  talc is not decidable from centre alone.
- No spatial-gradient claim is made here; the distance-to-contact test on
  these classes (block level) is the natural next step.

## 7. Mode map, block-level distance test, and elevation test (2026-09-07 addendum)

Scripts: `contact_combo_mode_map.py` (fig_mode_map_{cube}, mode_map_stats.csv),
`contact_mode_block_test.py` (fig_mode_block_test, mode_block_bins.csv,
mode_block_contrast.csv), `contact_mode_elevation.py` (fig_mode_elevation,
mode_elevation.csv). Classes: A carbonate-mode ('2.3 only', < 2.305 um),
B serpentine/talc-mode ('2.3 only', >= 2.305 um), C both bands. NOTE: the
class names are stack-level inferences (sec. 4); no pixel is 2.5 um-detected.

MODE MAP (per pixel): in 3E12, B forms the NW interior swath (median 275 m
from clay, 16 % within 90 m), A sits nearer the basement (234 m, 25 %), C
hugs the contacts (51 m, 67 %). The ordering C < A < B in median distance
holds in all four cubes, but the A-B difference is small (40 m in 3E12).

BLOCK TEST (504 m blocks, 1000 draws, seed 42; near 36-150 vs far > 400 m;
robustness near 90-150 m): NO class shows a near/far fraction contrast
whose CI excludes zero in 3E12, 871C or B438 (3E12: A +1.6 % [-0.7, +4.2],
B +1.9 % [-0.2, +4.2], C +0.2 % [-0.05, +0.55]). In A4FC, B is FAR-enriched
(-6.1 % [-11.1, -0.7]) and C near-enriched (+1.1 % [+0.4, +2.0]) - the
rim-mixing signature. Verdict: the per-pixel median ordering does not
reach block-level significance; no contact-ordered phase zonation can be
claimed for the diffuse classes A and B. The stage-6 result (classifier
patch cores 2-7x closer to the contact, patch-level KS) concerns the
strongest carbonate cores and is not contradicted - class C is
near-enriched ~1.8x in 3E12 with a CI just touching zero.

ELEVATION TEST (z_rel = z - mean of olivine-unit ground within 270 m;
matched distance bins; block-bootstrap medians):
- 3E12, 90-250 m: CTX control +0.08 m [-0.14, +0.27] vs carbonate-mode
  -0.45 m [-0.96, -0.03], serpentine-mode +0.29 m [+0.07, +0.56], both-bands
  -1.36 m [-2.58, -0.26]. HiRISE (002888, clean): control -0.21 [-0.65,
  +0.25] vs carbonate-mode -1.28 m [-2.15, -0.72]; > 250 m: control -0.84
  vs carbonate-mode -1.93 m [-2.33, -0.68]. Height above the IDW contact
  surface (<= 150 m): control +0.19 vs carbonate-mode -1.36 m [-3.41, -0.01].
  => In 3E12 the carbonate-bearing pixels sit 0.5 m (CTX) to 1.1-1.3 m
  (HiRISE) LOWER than adjacent olivine ground at the same distance from
  the basement, in both DEMs; the serpentine/talc-mode sits at or slightly
  above local ground.
- 871C: near the contact carbonate-mode is ~2 m lower than control on the
  contact-surface metric (-1.20 vs +0.90); interior bins mixed/ null.
- A4FC, B438: no consistent difference (A4FC = mixing cube; B438 HiRISE
  counts too small).

READING. The diffuse carbonate occupies local topographic LOWS within the
olivine unit and is not block-significantly concentrated toward the
basement contact. That pattern fits erosional exposure of a lower
(basal?) carbonate-bearing level - windows - or carbonate hosted in hollows,
and does not fit a resistant carbonate cap or a contact-fed halo for the
diffuse component; the contact-hugging cores (class C, stage-6 patches)
remain compatible with both a contact-fed and a basal-exposure origin,
since the contact is where the base of the unit crops out. The
serpentine/talc-family phase is interior, at/above local ground, and not
contact-concentrated (far-enriched in A4FC): a pervasive property of the
unit, consistent with the MTR3-chain unit-wide 2.32 um signature - i.e.
serpentinization here is NOT contact-controlled.

IMPLICATION FOR THE SUPPLEMENT. The supplement's stated alternative -
"equally consistent with erosional exposure of a basal carbonate-bearing
level at the unit margins" - now has positive topographic support for the
diffuse carbonate in 3E12; the phrasing can be strengthened in that
direction once the block-level numbers are cited. Single chain, descriptive
stacks for the phase labels; the topographic and block statistics above
are formal (block bootstrap).

## 8. Topographic profiles through the hottest patches (2026-09-07, 3E12 only)

Script `contact_hotspot_profiles.py`; products `fig_hotspot_profiles_{carbonate,serpentine}_FRT00003E12`,
`hotspot_profiles_FRT00003E12.csv`, `run_log_hotspot_profiles.json`.

Why: the sec. 7 elevation contrasts (0.5-1.3 m) are small against their 95 %
block-bootstrap CIs, and absolute elevations cannot be compared directly -
the olivine unit spans 20-140 m of relief within any 1 km profile, so a
metre-scale class effect is invisible without local detrending. The
old-school check is a profile through each of the hottest patches so the
patch and the adjacent non-altered olivine are read off the same line.

Method: hotspot = local maximum of the class density field (fraction of
olivine pixels within ~90 m carrying the class; carbonate = classes A + C,
serpentine = class B), interior (> 90 m from clay), >= 600 m apart, top 5
per class; patch = connected region around the peak with density >= half
the peak (<= 180 m radius). Two orthogonal profiles per patch (principal
axis and across), extended 250 m into the unit each side, sampled every
3 m on a HiRISE DEM mosaic (PDS 002888 clean > ADI 027691 > ADI 006712 >
PDS 006712, measured shifts) with the CTX 18 m DEM where no HiRISE DEM
exists. Reference = olivine-unit ground 30-130 m beyond the patch edge on
each side; 'below chord' = patch mean minus the straight line joining the
two flank means (negative = hollow, positive = crest). Chord is NaN where
one flank lies outside the olivine unit.

HiRISE coverage of 3E12 is partial: only carbonate hotspots 3 (002888) and
4 (027691, along-axis) are on HiRISE; all serpentine hotspots (NW/W/SW
interior) are CTX only.

Carbonate hotspots (density 0.94-1.00):
| # | location | dist. clay | DEM | geometry | below chord (along / across) |
|---|---|---|---|---|---|
| 1 | NE olivine island | 141 m | CTX | shallow saddle behind the rim of a knob that stands ~10 m above the clay | -3.7 m / n.a. (knob) |
| 2 | NE olivine island | 163 m | CTX | bench on a 30 m slope; across: 10 m knob crest carrying the both-bands pixels | -0.2 (min -4.9) / n.a. (knob) |
| 3 | SW | 314 m | HiRISE | bench on a gentle slope; across: 70 m valley wall below clay | +0.3 (min -1.1) / n.a. |
| 4 | E | 420 m | HiRISE | 15 m deep, ~100 m wide trough; carbonate-mode fills floor and walls, both-bands on the rim | -2.9 (min -15.3) / +29 (slope) |
| 5 | E | 467 m | CTX | 16 m knob crest, both-bands on the crest | +9.9 / n.a. |

Serpentine/talc hotspots (density 0.88-0.97, all CTX):
| # | location | dist. clay | geometry | below chord (along / across) |
|---|---|---|---|---|
| 1 | NW | 524 m | hill shoulder / crest | +3.3 / +4.8 |
| 2 | SW | 568 m | ridge crest; across: planar slope | +1.1 / -2.6 |
| 3 | N interior | 1025 m | knob crest | +0.9 / +2.4 |
| 4 | W | 205 m | planar slope | +1.3 / n.a. |
| 5 | SW | 491 m | slope shoulder | +6.1 / -1.6 |

READING.
- Carbonate: no consistent landform. Two of five hottest patches occupy
  hollows (a saddle, a 15 m trough), two sit on benches, one caps a knob;
  the NE hotspots are whole olivine islands (knobs 10 m above the clay)
  whose entire surface is carbonate-mode. The sec. 7 'carbonate 0.5-1.3 m
  lower' result is therefore a small AVERAGE tendency, not a patch-scale
  landform, and the 'erosional windows in hollows' picture is NOT
  supported at the hottest patches. The one clean hollow (hotspot 4, on
  HiRISE) is real and worth a look, but it is one of five.
- Serpentine/talc-mode: systematically on local highs - crests and
  shoulders in 4 of 5 patches, positive chord offset in 7 of 9 evaluable
  profiles (median +1.3 m, up to +6 m), never a hollow. Stronger than the
  block statistic (+0.3 m) suggested. Consistent with a pervasive unit
  property expressed best where the unit is well exposed (wind-swept
  crests, least cover), or with the serpentinized material being the
  upper preserved level; it is not a contact or a hollow phenomenon.
- Together: carbonate and serpentine-mode differ in where they are
  expressed (carbonate mixed/lows-leaning, serpentine highs), but neither
  shows the geometry a contact-fed halo would give, and the carbonate
  topographic evidence for erosional windows is weaker than sec. 7 read
  it. Supplement wording should stay at 'consistent with' for the
  erosional alternative, not 'supported by'.

Caveats: CTX 18 m posting for 7 of 10 patches (100-300 m landforms are
resolved; < 5 m relief is not reliable); patches picked by class density
on a per-pixel map with n_eff ~1 %; single chain for the class labels.

### 8b. All four cubes (2026-09-08)

Same script, `fig_hotspot_profiles_{class}_{cube}(_abs)` and
`hotspot_profiles_{cube}.csv` for 871C, A4FC, B438 (absolute-elevation
versions `_abs` for every cube). HiRISE coverage: 871C entirely on Adriano's
ASP 006712 DEM (0.8 m); A4FC 6 of 20 profiles on HiRISE (006712 + 027691);
B438 8 of 20 (002888 + 027691); 3E12 3 of 20. Caveats: 871C/A4FC hotspots
lie in the small mapped corner/strip of those cubes; B438 serpentine
patches are weak (density 0.29-0.47, 38-89 cells); A4FC is the mixing cube.

Tally of profiles by chord offset (low <= -1 m, high >= +1 m; 10 profiles
per class per cube, NaN where one flank is off the olivine unit):

| class | cube | low | flat | high | n.a. | median chord | HiRISE profiles |
|---|---|---|---|---|---|---|---|
| carbonate | 3E12 | 2 | 2 | 2 | 4 | +0.0 m | 3 |
| carbonate | 871C | 4 | 3 | 2 | 1 | -0.6 m | 10 |
| carbonate | A4FC | 3 | 2 | 4 | 1 | -0.2 m | 2 |
| carbonate | B438 | 4 | 2 | 3 | 1 | -0.8 m | 4 |
| **carbonate, all** | | **13** | **9** | **11** | 7 | **-0.2 m** | 19 |
| serpentine | 3E12 | 2 | 1 | 6 | 1 | +1.3 m | 0 |
| serpentine | 871C | 2 | 1 | 3 | 4 | +0.8 m | 10 |
| serpentine | A4FC | 5 | 2 | 1 | 2 | -1.1 m | 4 |
| serpentine | B438 | 1 | 1 | 7 | 1 | +1.4 m | 4 |
| **serpentine, all** | | **10** | **5** | **17** | 8 | **+1.1 m** | 18 |

Geometries seen: carbonate hotspots are crests (3E12-5, 871C-2, B438-1,
A4FC-5 knob), benches (3E12-2/3, 871C-3/5, B438-2), valley floors or
troughs (3E12-4 on HiRISE, B438-3 at the base of a 175 m slope, 871C-1 a
hollow) and plain slopes, in roughly equal numbers in every cube. The
serpentine hotspots are crests, plateau tops and shoulders in 3E12 (6/9)
and B438 (7/9), highs by a small margin in 871C (3 high, 2 hollow incl. a
10 m trough at 871C-5), and reversed in A4FC (5 low / 1 high; the cube
whose olivine-unit 2.3 um detections behave like diluted clay).

VERDICT (four cubes). (1) The diffuse carbonate class has no preferred
landform anywhere: hollow, bench, slope and crest in comparable numbers,
median chord offset -0.2 m. The sec. 7 'carbonate 0.5-1.3 m lower' block
result is confirmed as a weak average tendency and nothing more; the
erosional-window geometry is not a property of the carbonate patches.
(2) The serpentine/talc-mode class prefers local highs in the three
non-mixing cubes (16 high vs 5 low of 27 evaluable profiles; median
+1.1 m overall), reversed only in A4FC. This is the one landform
association in the campaign, and it argues for the serpentine-family
signature being a property of the best-exposed, least-mantled olivine
surfaces (or of an upper preserved level), not of the contact or of
hollows. Neither class shows the geometry a contact-fed halo would give.
