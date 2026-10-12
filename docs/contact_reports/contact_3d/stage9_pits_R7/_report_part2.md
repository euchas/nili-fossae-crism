# Stage 9, part 2 - Artifact assessment and verdict on the pit catalogue

Date: 2026-08-31. Continues `_report_part1.md` (detection and catalogue). This part
reports the artifact diagnostics (scripts 11-13, added 2026-08-31), the verdict from
an adversarial verification review (5 claims tested, all survived with two wording
refinements), and the corrections this forces on part 1. New scripts and parameter
logs in `scripts/` and `params_11..13_*.json`.

## 3. Additional diagnostics

### 3.1 Phase-randomized noise null (`scripts/11_noise_null.py`)

Five surrogate DEMs (seeds 42-46) with the window's own amplitude spectrum
(detrended, Hermitian-symmetric random phases, variance rescaled, trend restored)
were run through the identical detector as `01_detect_pits.py`.

| depth class (m) | observed | null mean (range) |
|---|---|---|
| 0.3-0.5 | 830 | 800.4 |
| 0.5-1 | 1,745 | 3,195.8 |
| 1-2 | 400 | 1,113.2 |
| 2-3 | 41 | 62.6 |
| >= 3 | 101 | 8.2 (3-12) |

Only the >= 3 m class exceeds the null (about 12x). Every shallower class is matched
or over-produced by structureless surrogates. An independent inverted-DEM control
(closed peaks instead of closed depressions, same pipeline) agrees: 8 peaks >= 3 m
vs 101 pits, near-parity at low prominence (2,653 vs 3,117; identical median depth
0.62 vs 0.63 m). Caveat: real landforms also contribute spectral power, so the null
is an upper bound on artifact counts - but for the sub-metre classes the bound is
already at or above the observed count.

### 3.2 Depth-stratified diagnostics (`scripts/12_depth_split.py`)

- Phase locking to the instrumental row ripples: the sub-metre classes are weakly but
  significantly locked to the 3.407 m ripple (Rayleigh p = 1.3e-3 and 2.2e-9;
  R = 0.089-0.107, roughly a 20 percent density modulation), while a 20,000-point
  random control is uniform (p = 0.69). Classes >= 1 m show no 3.4 m locking
  (p >= 0.23), but verification found the 1-2 m class locked to the secondary
  10.222 m ripple (R = 0.127, p = 1.6e-3) - an instrumental fingerprint at the same
  evidential standard.
- Orthoimage contrast is significant only for the sub-metre classes (-0.68 DN
  p = 0.015; -0.55 DN p = 1.3e-4); the >= 3 m class shows no aggregate darkening
  (p = 0.49). 78/101 deep pits have dark centres vs 42 percent in a radius-matched
  random-aperture null (binomial p ~ 4e-13) - real, but causally ambiguous, because
  shadowed low-contrast patches are exactly where the stereo matcher fails and
  creates spurious minima. The 25 cm ortho is rectified on the audited DTM itself
  and is not an independent dataset.
- Exclusive-area unit density ratios (olivine/clay): 2.31, 2.27, 2.18 for the
  sub-2 m classes; 0.15 (2-3 m) and 0.33 (>= 3 m) - the contrast inverts with depth.
- Spatial confinement: a contiguous box (window rows 550-1050, cols 50-950; 31.6
  percent of valid area, containing the steep shadowed clay-unit scarp) holds 98
  percent of >= 3 m pits, 100 percent of >= 5 m pits, and 94.9 percent of 2-5 m
  pits, vs 16.9 percent of sub-metre pits. 86/101 deep pits lie in Clay_unit at
  median slope 23.7 deg, strongly clustered (Clark-Evans R = 0.45). 125 catalogue
  pits have physically implausible depth/diameter > 0.5 (extreme: pit 1664, 7.37 m
  deep, 3.4 m wide).

### 3.3 Deep-pit chains and coalescence (`scripts/13_deep_pit_chains.py`)

Tested AV's visual hypothesis (2026-08-31) that the >= 3 m pits are the real
population and merge into graben-like troughs near R7.

- Chaining: at the 3 m cut, 7 chains of >= 3 members at <= 30 m linkage vs 2.5
  expected from random same-size subsets of the full catalogue (p = 0.008); largest
  chain 22 members, 187 m, PCA linearity 0.92, azimuth 105 deg. Per-pit elongation
  azimuths cluster at 89 deg (R = 0.534), within 5 deg of the local trough axis
  (94 deg) and the E-W contact trend. The 2 m cut is not above the null (p = 0.18).
- Coalescence: fails. Scanning the fill-depth threshold from 0.30 down to 0.02 m,
  the 26 deep pits in the test box never merge (25 components remain); the thalweg
  crosses a ~20 m transverse rise and half the deep pits sit above the box's
  minimum-elevation path (median 2.2 m). A co-linear catena of discrete closed
  basins, not a continuous trough. (A breached, along-axis-draining trough would
  also fail the merge test; the transverse high argues against one here.)
- Reinterpretation forced by 3.2: the chain analysis' null (random subsets of the
  whole catalogue) tests spatial organization, not reality. Since 98 percent of the
  deep pits are confined to the E-W-trending scarp zone, "chains aligned with the
  contact trend" and "detections strung along the scarp face" are observationally
  identical. The chain result therefore demonstrates organization but cannot
  discriminate landforms from correlation blunders that concentrate on the same
  scarp.

## 4. Verdict

The n = 3,117 catalogue is not defensible as a landform population.

- Sub-metre classes (n = 2,575; 83 percent): statistically indistinguishable from
  detector noise (null-matched counts, ripple phase locking). Report only as an
  uncertified census at the noise floor.
- 1-2 m (n = 400) and 2-3 m (n = 41): within the noise null's count reach; the 1-2 m
  class carries a 10.2 m ripple fingerprint. Defensible only morphologically,
  per-pit, not statistically.
- >= 3 m (n = 101): the only class with a count excess over both nulls (~12x). This
  proves phase-coherent, spatially organized structure - but coherence is not
  reality: the class is confined to the most artifact-prone terrain (steep, shadowed
  scarp), where localized correlation blunders would produce the same excess over a
  phase-randomized null. Status: ~100 coherent candidate structures requiring
  individual stereo-pair vetting, not confirmed pits.
- Individually secure today: only pit 567 (the 273 m enclosed basin - a
  mapping-scale landform, its own class, n = 1). The flagship resolved forms
  (pits 1403, 1414) lie inside the suspect zone and also need vetting.

Conservative publication cut (if any list appears in supplementary material): three
explicitly separated tiers - Tier 1: pit 567 alone; Tier 2 (n = 212): depth >= 1 m,
diameter >= 5 m, depth/diameter <= 0.5, tin_frac = 0, outside the suspect zone,
labelled "candidate closed depressions, individually unvetted"; Tier 3 (~103 in-zone
deep detections): held back entirely pending vetting. Never quote n = 3,117,
2,207 per km2, or median depth 0.63 m as landform statistics.

## 5. Corrections to part 1 and to scripts

- Banding-trough statistic (61.2 percent, p = 6e-42) is partly self-circular; the
  fill-DEM control gives 56.8 percent (p = 3e-18) and surrogate nulls give 64.6-66.8
  percent - non-diagnostic in either direction. Do not use.
- Clark-Evans R = 1.02 must be quoted against the mask-corrected null
  (1.011 +/- 0.009), i.e. z ~ 1.1: consistent with randomness, and affirmative
  evidence against dike-controlled pit chains at catalogue scale.
- The published unit-density comparison (2,184 vs 1,587 per km2) was
  superset-vs-subset (Clay_unit is nested in Olivine_unit); use exclusive areas
  only, and only after the geologic-map registration is verified.
- The zero-TIN claim is vacuous: the detector flags zero pixels window-wide and was
  never validated on known fill (do that on DTEEC_006712).
- Script defects to fix before reuse: overview/zoom figures outline the pre-cut
  component set (~5,200) rather than the reported 3,117; `08_density_units.py`
  rewrites `pits_catalog.csv` in place; the CTX test weighting is dominated by
  pit 567 (88 percent of weight); `07_stack.py` uses min-vs-median rather than an
  annulus statistic for chip darkness.

## 6. Relation to the contact-alteration argument

Decouple the pits from the contact-alteration argument entirely. "Deep pits line the contact" is
inadmissible: the contact is topographically the scarp, the scarp is where the
matcher fails, and the entire deep tail sits there. The catalogue-scale spatial
randomness (corrected Clark-Evans) is itself evidence against dike-controlled pit
chains; and even real metre-scale pits could not be primary degassing features
surviving ~3.5 Gyr - they would be geologically young regardless of origin. If
mentioned at all, present the screened candidates as a young, erosion-controlled
surface texture, explicitly neutral to the contact-alteration case. The stage's
value is protective: the artifact battery was run, overclaiming was
declined, and the results that do survive elsewhere (block-level hydration gradient,
carbonate-patch concentration, clay-side fringe) are the stronger for it.

## 7. Next steps

1. Decisive test: cross-DTM recurrence. Neither of Adriano's DEMs reaches the R7
   window (ADI_027691's north edge falls ~420 m short in y with no x overlap;
   ADI_006712 is ~14 km south) - so either request an independent ASP re-reduction
   of the PSP_002888/002176 pair with different correlation parameters, or run the
   detector-recurrence comparison in the ADI_027691 / PDS_002888 swath overlap south
   of R7 (tests the detector's artifact rate on the same terrain type, though not
   the R7 candidates themselves).
2. Manual stereo-pair and ortho vetting of the 101 >= 3 m pits and the 83 in-zone
   conservative-cut pits (including 1403 and 1414).
3. Illumination-phase test for pits >= 5 m diameter: sun-azimuth-oriented ortho
   stack; real depressions shade the sunward inner wall systematically, matcher
   blunders carry no illumination phase.
4. One-off verification of the geologic-map gdb registration to the controlled
   frame before quoting any unit-density statistic.
5. Spatially variable noise null (local highpass-RMS-scaled) to bound the 1-2 m
   class artifact fraction.
6. Only if a genetic reading is ever pursued: ridge-strike alignment of vetted deep
   pits vs the Pascuzzo ridge classes, and overlap vs the Stage 6 carbonate patches,
   each against a proper random null.
