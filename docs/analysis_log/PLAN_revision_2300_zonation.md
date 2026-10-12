# PLAN: 2.3 um band-position variability and zonation test (band_center v2)

Status: PROPOSED 2026-09-01. Awaiting AV's approval. No compute run yet.

## Question

Is there detectable 2.3 um band-POSITION/shape variability across each of the four
Horseshoe cubes, and specifically an ordered mineral zonation
(nontronite ~2287-2290 nm -> Fe/Mg smectite ~2300 -> saponite/talc/magnesite
~2310-2313 -> serpentine ~2320 -> Mg-chlorite ~2330 -> Fe-chlorite/prehnite
~2350-2360) with distance from the mapped olivine/clay-basement contact?
This is the "ordered zonation" criterion of the contact-metamorphism test
framework (the single strongest orbital criterion for a thermal aureole).

## Why the existing test does not answer it

`code/contact_band_center_2300.py` (2026-08-28) was a six-population median
contrast, not a variability study:

- Per-pixel argmin + 3-point quadratic on continuum-removed UNRATIOED I/F;
  medians per population (interior clay, interior cap, olivine in 4 distance
  bands). Result: near-contact olivine 2311-2318 nm vs interior clay
  2307-2313 nm, shift +2.7 to +5.2 nm ("< 1 channel" of 6.55 nm).
- The shift carries NO confidence interval (the only bootstrap in that script
  is on the BD1900R2/D2300 excess ratio). "< 1 channel" is a difference of
  medians, not a bounded measurement.
- Contact defined from GMM spectral classes on the UNSHIFTED PDS frame; both
  the geologic-map contact and the 300-360 m coregistration fix postdate it.
- No band-centre map, no spatial-coherence statistics, no zonation-sequence
  test, no doublet handling; search window 2265-2350 nm clips the chlorite
  range; far-field "centres" (2317-2324 nm) sit on 0.007-0.010 depths at the
  residual floor and were rightly called non-mineralogical.
- Declared failure mode was ATTRIBUTION (smectite dilution vs new Mg-OH phase
  both predict the direction), not detection.

The zonation question is therefore genuinely untested.

## Sensitivity ladder (from Brown 2010 / Viviano 2013 / VB2014 / Ehlmann 2009)

- Sampling 6.6 nm/channel, FWHM ~14 nm. Adjacent phases in the ladder are
  ~10 nm (1.5 ch) apart; full sequence span 40-60 nm.
- Saponite/talc/magnesite are 0-3 nm apart at 2.3 um: NEVER separable by
  position (Brown 2010 says so explicitly). Attribution there rests on
  secondaries: 1.39 um sharpness/position, 2.10-2.12 um (serpentine-only),
  2.35 um septum (talc, Viviano), 2.39 um, 2.50-2.51 um.
- Ehlmann 2009 quotes +/-10 nm single-spectrum band-centre error; Brown 2010
  reached 1 nm centroids only by asymmetric-Gaussian fitting with per-column
  CDR WA wavelengths. We have NO per-column wavelengths on disk; the MTRDR
  products carry EMPIRICAL_SMILE_CORR_FLAG = ON, so only the RESIDUAL smile
  needs handling - empirically, via the clay basement as internal standard
  against DDR-projected detector column.
- Per-pixel centres at our band depths (0.005-0.02) are noise-dominated
  (sigma ~10-30 nm). Per-pixel maps cannot answer the question. Stack-then-fit
  can: 504 m block stacks give ~1.5-2.5 nm; distance-bin stacks give 95% CI
  half-widths ~1.5-3 nm (0.25-0.45 ch) IF the post-correction smile residual
  is <= 1-1.5 nm. That resolves the 10 nm per-phase steps at 3-5x the limit.

## Tier 0 - simple first look (AV's point: 70 nm is ~11 channels, trivially resolvable when bands are deep)

Before any of the machinery below: extend `contact_band_center_2300.py` to
SAVE the per-pixel band-centre map it already computes (it only saved
histograms), on the corrected frame with the map-based contact, in two depth
tiers:

- depth >= 0.015 ("strong band" tier): at these depths single-pixel centres
  are good to a few nm, so any well-developed zonation at the 10-70 nm scale
  (e.g. a chlorite/serpentine zone with a real band) is visible by eye in the
  map. Also plot centre-vs-distance for this tier directly.
- depth 0.005-0.015 ("weak band" tier): displayed separately and labelled
  noise-dominated (sigma 10-30 nm/pixel), for context only.

Cost: ~1-2 h of work, minutes of compute. If the strong-band tier shows
coherent shifted-band zones near the contact, the question is largely answered
and Tier 2 shrinks to quantification. If the strong-band tier is empty or flat
inside the olivine unit (likely - the old test's accepted-pixel IQR was
5-10 nm and medians sat within one channel), then the signal, if any, lives in
the weak-band sub-channel regime, which is what the rest of this plan is for.

## Tier 1 gates + Tier 2 full design (merged: Design B skeleton + Design A instruments, per adversarial review)

Only needed if Tier 0's strong-band tier is empty/flat. Two cheap GATES first,
one afternoon, before the full pipeline:

- GATE 1 (falsifier, ~10 min, 3E12 only): block-stack interior clay (the
  deepest, most uniform 2.3 um band in the scene), fit centres, regress on
  DDR-projected detector column; report column-trend peak-to-peak and
  post-detrend block RMS vs a per-pixel Monte Carlo noise floor. If
  post-detrend RMS > ~2-3 nm, the sub-channel budget is fiction: only the
  >= 1.5-channel variance/coherence question survives, and we take the
  sensitivity-limited-null exit before touching an olivine pixel.
- GATE 2 (mixing explanation, nearly free): two-endmember mixing trajectory -
  stack interior clay and far-field olivine, form areal and single-scattering-
  albedo mixtures, push through the identical fit. If the curve reproduces the
  prior +2.7 to +5 nm near-contact shift at matched depth, the old signal is
  explained by sub-pixel mixing with no new phase, and the migration test is
  moot (variance question + secondaries still run).

Then, per cube (never concatenated), seed 42, block statistics only:

- S0 build: memmap windows 2.10-2.60 + 1.30-1.50 um; dual variants
  (V1 unratioed, V2 cap-median ratioed); masks/distances via
  `contact_block_stats.cube_setup` on the CTX-shifted frame; 504 m blocks.
- S1 Jezero-style milestone: per-pixel Monte Carlo noise floor (NOT split-half:
  MTR3 resampling duplicates 34.6% of vertical neighbour pairs, measured, so
  split-half floors are biased low); robust variogram + Moran's I on block
  medians. STOP/GO statistic = olivine coherence IN EXCESS of the identically
  processed interior-clay reference (raw coherence would pass on smile alone).
  STOP in all 4 cubes -> honest null: "no coherent 2.3 um band-position
  variability above a measured X nm block-level floor".
- S2 calibration: TER3 sensor-space clay wavelength standard vs detector
  column (via DDR, same machinery as the Plebani pass); isotonic depth-bias
  curve b(depth) from interior clay (the documented weak-band drift to
  2317-2324 nm); 100-translation fake-contact ensemble (0.7-3.0 km) as the
  empirical null; injection-ramp calibration {1.5, 3.0, 6.6, 13.2 nm} run per
  depth regime - the smallest recovered ramp IS the quoted detection limit.
- S3 migration test: distance bins (36-90 flagged, 90-150, 150-400, >400 m)
  from the map contact on the corrected frame; median-stack ratioed spectra;
  Brown-form asymmetric-Gaussian fits (continuum 2200/2384 nm, apparent
  absorbance, 32 jittered restarts; 9 fixed-model continuum variants as the
  systematic; doublet only via delta-BIC >= 10, reported as model preference);
  paired 1000-draw block bootstrap for joint CIs; Jonckheere-Terpstra ordered
  trend; depth-matched rerun. A POSITIVE additionally requires the near-contact
  (depth, centre, asymmetry) locus to depart, at CI level, from the GATE-2
  mixing trajectory - both prior designs would otherwise pass on pure mixing.
- S4 secondaries + shape classes: 1.39/2.10/2.35/2.39/2.51 um measured on the
  SAME stacks (consistency matrix per candidate phase; chi asymmetry in Brown's
  centroid frame, compared only to Brown Table 1 loci). Descriptive
  PCA/GMM/HDBSCAN on shape-normalized continuum-removed segments, clay-anchored
  class EXCLUDED from any arrangement statistic.
- S5 cross-cube: structural comparison only (signs, orderings, class ladders);
  871C/A4FC blueshift forbids absolute cross-cube centres. Zonation language
  requires >= 3/4 cubes same-sign with all controls null.

Pre-specified honest exits: sensitivity-limited null with the measured floor;
"depth-matching collapses the shift -> smectite-band dilution (mixing, not new
phase)"; "migration detected, phase unattributed" (saponite/talc/magnesite
degeneracy); "structured heterogeneity, not contact-ordered".

## Compute

Gates: ~15 min. Full pipeline: ~2.5-4 h wall-clock for all four cubes,
single-core numpy/scipy/sklearn/hdbscan, < 1 GB new disk. Development
~1-1.5 days.

## Items needing AV's explicit sign-off

1. Approve this plan overall.
2. Primary claim-bearing contrast: propose 90-150 m vs > 400 m (registration-
   robust), a change from the project's 36-150 m convention - the 36-90 m bin
   stays computed but flagged, never claim-bearing.
3. New stack-level spectral scalars (2.35 um septum, BD2390-like, BD2250-like,
   2.51 um fit) - stack readouts only, no per-pixel maps; adjacent to but not
   on the approved Stage D list (BD2290, BD1400, SINDEX2, LCPINDEX2,
   MIN2295_2480).
4. Escalation branch: if interior clay itself shows a real non-column-aligned
   spatial trend, the internal-standard assumption fails - that is a finding
   needing a decision, not something to silently correct through.

## Hypothesis framing (for the eventual report)

Kremer airfall predicts a spatially homogeneous band-position field; intrusive
zonation predicts monotonic contact-ordered migration with phase-consistent
secondaries. Positive templates: Ehlmann 2009 Fig 17 "partially altered
olivine" at 2.32 um (serpentinized-olivine class) vs Viviano 2013 carbonation
(talc+magnesite colocated, serpentine consumed) - two different predicted
centroid populations, separable at the 1-1.5 channel level on stacks.
