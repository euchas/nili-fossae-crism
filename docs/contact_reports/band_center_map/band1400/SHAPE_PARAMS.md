# Shape-selective 1.4 um parameters: options and prototype results

Prototype run 2026-09-04, `code/contact_shape_param_prototype.py`, on the
two bland-ratioed ROI spectra of FRT00003E12
(`plebani/roi_spectra_FRT00003E12.csv`): the olivine-unit ROI (smooth 1 um
wing, truth = NO 1.4 um band) and the smectite ROI (truth = real OH band at
~1.39 um). Metrics: `plebani/shape_param_prototype_metrics.csv`,
`plebani/shape_param_injection.csv`; figure
`plebani/fig_shape_param_prototype.png`.

## Why fixed-wavelength band depths fail here

A band-depth parameter measures the displacement of the spectrum below a
CHORD drawn between two shoulders. Over the olivine unit the 1.3-1.5 um
region is a smooth concave-up curve (the 1 um band's long-wavelength
recovery), so any chord manufactures a sag with no absorption present. The
sag maximum falls mid-window (~1.43-1.44 um), which is exactly what the
band-centre fit was reporting. Shape information is what separates the two
cases, so the parameter has to use it.

## Results: false positive (olivine ROI) vs true positive (smectite ROI)

Stacked-spectrum 1-sigma noise: 0.0017 (olivine ROI), 0.0007 (smectite).

| method | olivine ROI (truth: nothing) | smectite ROI (truth: band) | verdict |
|---|---|---|---|
| M0 linear continuum + minimum (current) | depth 0.030 at 1.434 um | depth 0.015 at 1.388 um | FAILS - large false positive |
| M1 quadratic continuum on band-free shoulders | depth -0.0066 | depth 0.0117 at 1.388 um | passes, but biased (see below) |
| M2 convex-hull continuum | depth 0.068, FWHM 230 nm | depth 0.017, FWHM 59 nm | FAILS on depth; the WIDTH separates them |
| M3 Savitzky-Golay 2nd derivative, ~25 nm scale | SNR 1.2 | SNR 12.4 at 1.388 um | PASSES |
| M4 background x Gaussian, delta-BIC | amp 0.0007, FWHM pinned at bound, delta-BIC -10.4 | amp 0.0111, centre 1.394 um, FWHM 34 nm, delta-BIC +57 | PASSES, and returns shape |

## Sensitivity: can a real band still be found on the wing?

Gaussian OH band (FWHM 25 nm at 1.390 um) injected onto the olivine ROI:

| injected depth | M1 depth | M2 hull depth | M3 SNR | M4 amp | M4 delta-BIC | M4 centre / FWHM |
|---|---|---|---|---|---|---|
| 0.0% | -0.66% | 6.83% | 1.2 | 0.07% | -10.4 | (rejected) |
| 0.4% | -0.28% | 6.92% | 2.5 | 0.87% | -5.7 | (rejected) |
| 0.7% | 0.01% | 6.99% | 3.7 | 0.54% | +2.9 | 1.388 / 18 nm |
| 1.0% | 0.29% | 7.22% | 5.0 | 0.82% | +16.7 | 1.389 / 21 nm |
| 2.0% | 1.24% | 8.15% | 9.0 | 1.81% | +57.1 | 1.389 / 23 nm |
| 3.0% | 2.19% | 9.07% | 13.4 | 2.81% | +84.6 | 1.390 / 24 nm |

Detection limits on a stacked ROI spectrum: ~0.5-0.7% band depth (M3, at
SNR 3), ~0.8-1.0% (M4, at delta-BIC 10). Above ~1% M4 recovers the centre
to 1-2 nm and the width to a few nm. M2's hull depth is swamped by its
6.8% baseline offset; M1 carries a -0.66% offset because the wing is
steeper than quadratic, so its zero point is wing-strength dependent.

## Literature precedent (literature audit, 2026-09-04)

- DECISIVE, SAME CUBE: Brown (2020) fits the olivine 1 um band at
  FRT00003E12 with a 0.7/1.75 um continuum and reports a centroid of
  1.412 um (asymmetry 4.875), a long-wavelength tail at ~1.43 um, and a
  population "further redshifted into the 1.4-1.5 um centroid range"; he
  notes that saturation "causes the band shape to flatten between 1 and
  1.5 um". Our olivine-unit "1.43-1.45 um band centres" ARE Brown's
  olivine centroid on the same data. This is publishable as a caution.
- Mandon et al. (2020) build their olivine index from band depths at
  1.08, 1.26, 1.37 AND 1.47 um (continuum 1.63/2.53 um), gated on
  convexity at 1.26 / 1.34 um. Peer-reviewed practice therefore treats
  1.37 and 1.47 um as INSIDE the olivine band and 1.34 um convexity as a
  POSITIVE olivine test - our left anchor (1.316-1.342) sits on it.
- Viviano-Beck et al. (2014) revised LCPINDEX2/HCPINDEX2 to be "less
  subject to false positives from other spectral end-members or mixtures
  that display convexity between ~1.3 and 1.5 um" - the CRISM team names
  this exact trap, and their remedy was moving anchors OUT of the window,
  which is impossible for a 1.4 um parameter. Note also that BD1300
  carries the caveat "Fe-Olivine can be > 0" while BD1400's caveat cell
  is EMPTY, and that OLINDEX3 gives R1330 its LARGEST weight (0.4) -
  i.e. VB2014 treats 1330 nm as deeply absorbing while a 1.33/1.52 chord
  treats it as unabsorbed shoulder. Mutually contradictory premises, and
  a quantitative prediction of the observed rho 0.56-0.83.
- MGM (Sunshine et al. 1990) is the canonical shape-fitting form, but
  Brown notes it "will not handle band saturation resulting from large
  grain sizes that are present in the CRISM dataset" - so a local
  narrow-band model (M4) is preferable to a full MGM deconvolution here.
- Convex-hull removal (Clark & Roush 1984) was predicted a priori to fail
  for this problem - over a concave-up segment the upper hull IS the
  chord - which matches the measured M2 result exactly.

## Recommendation (hardened after adversarial methods review)

1. PRIMARY - M4 (background x Gaussian, model comparison), with SIX
   mandatory hardenings. As specified in the prototype it would produce a
   badly false-positive map:
   - THRESHOLD MUST BE CALIBRATED EMPIRICALLY, NOT FROM A TABLE. Under
     ISRF-correlated noise (14 nm ISRF at 6.6 nm sampling) the simulated
     NULL delta-BIC has median +6 to +15 and 99th percentile +40 to +68,
     so a textbook "delta-BIC > 10" flags more than half of pure-noise
     spectra. Set the threshold at the 99th percentile of the identical
     estimator run in band-free windows (1.31, 1.33, 1.56, 1.58 um), per
     detector column. (The prototype's stacked-ROI delta-BIC of -10.4 for
     olivine and +57 for smectite is a high-SNR case and does not license
     the threshold for lower-SNR spectra.)
   - Convolve the Gaussian with the instrument response, or bound the
     OBSERVED FWHM to 20-40 nm; 10 nm is unphysical below the 14 nm ISRF
     and merely licenses fitting single-channel spikes.
   - Fix the background polynomial order a priori (order selection
     changes recovered depth by 11-29%).
   - Charge a look-elsewhere penalty for the ~5 independent centre
     positions in 1.37-1.43 um.
   - Quote the SHOULDER/WINDOW JITTER ENVELOPE as the error bar, not the
     fit covariance: at these depths the statistical error (~0.0014) is
     2-3x SMALLER than background-misspecification bias (0.002-0.004).
   - Report non-detections alongside detections.
2. INDEPENDENT CONFIRMATION - M3 (matched-scale 2nd derivative), with two
   restrictions: bland-ratioed chain ONLY (every usable SG kernel reaches
   the verified 1.441 um CO2 residual - a 7-point kernel measuring at
   1.420 um reaches 1.4398 um - so on unratioed I/F it would report a
   scene-wide false detection), and BLOCK statistic only (per-pixel SNR
   for a 0.005 band is 0.8-2.4). It fails differently from M4, which is
   the point of running both.
3. DISPLAY / CROSS-CHECK ONLY - M1, with its right shoulder moved to
   1.50-1.58 um to clear the CO2 residuals. Band-free bias -0.002 to
   -0.004 and a shoulder-jitter envelope of 0.0046-0.0081 make it
   unusable as a standalone map parameter at our signal levels.
4. REJECT OUTRIGHT - M2. Over a concave-up wing the upper hull degenerates
   to the M0 chord, and it adds a one-sided noise bias (+0.004) that
   scales with local SNR, i.e. an albedo-correlated artifact that would
   trace the geology and be mistaken for it. The hull WIDTH remains a
   legitimate qualitative flag but nothing more.
5. RETIRE - M0 and, in the olivine unit, BD1400. The M0 failure is
   predictable from anchor geometry alone: for a locally quadratic wing
   the chord-removed minimum lands at the arithmetic midpoint of the two
   anchor centroids, (1.329 + 1.521)/2 = 1.4250 um, which is the observed
   1.43-1.45 um pile-up; the search window then quantizes it onto the
   1.441/1.477 um CO2 residual channels.

## CORRECTION to the "clay is genuine OH" framing

The earlier statement that 1.4 um depth over the clay unit is genuine
OH/H2O while only olivine is wing-driven is TOO CLEAN. In
`wing_check.csv` the map-pure CLAY unit also correlates with OLINDEX3 at
rho 0.599 (3E12), 0.637 (A4FC), 0.581 (B438); in 3E12 clay that EXCEEDS
its correlation with BD1900R2 (0.136). Only 871C behaves as advertised
(0.172 vs 0.523). Likely subpixel olivine mixing or shared continuum
slope. What survives: the bland-ratioed clay ROI carries a directly
measured, shape-verified band (M4 amplitude 1.11%, centre 1.394 um, FWHM
34 nm, delta-BIC +57) - a measurement, not a correlation. Any new
parameter must be validated for OLINDEX3-orthogonality in BOTH units.

## Mandatory validation gate before any map is trusted

(i) Null-window calibration per detector column (threshold = 99th
percentile of the estimator in band-free windows). (ii) Injection-recovery
with ISRF-convolved Gaussians at 0.003-0.020 amplitude on REAL map-pure
olivine spectra; report the completeness limit, not the smallest visible
value. (iii) ORTHOGONALITY as the pass/fail criterion: rho(metric,
OLINDEX3) and rho(metric, 1 um centroid) must fall to ~0 within map-pure
olivine (from 0.56-0.83 and 0.55-0.78) while rho(metric, BD1900R2) rises
in clay. (iv) Negative control: the anhydrous cap must return null.
(v) Column stratification to expose smile/striping. (vi) Dual-chain
replication.

## Claim ceiling for any resulting product

Even a fully hardened M4 map cannot support a hydration map of the olivine
unit. The honest deliverables are: (a) a quantified 3-sigma UPPER LIMIT on
narrow (20-40 nm) OH band amplitude in block-median map-pure olivine
spectra, realistically ~0.004-0.006 once the jitter envelope is folded in
- a non-detection with a number attached; (b) a positive control in the
clay basement at 1.389-1.392 um on the ratioed chain, itself re-verified
under the new method; and (c) NO per-pixel detections, no spatial gradient
below ~1 km, and no mineral identification from band centre at all - the
~2-channel (13 nm) inter-chain wavelength offset equals the 1.39 vs 1.41 um
separation the identification would rest on.

## Caveats before any map is made from this

- These limits are for STACKED spectra (n ~ 230 sensor pixels, 1-sigma
  0.0017). Per-pixel noise is several times larger, so a per-pixel M4 map
  would have a detection limit several times worse; the natural product is
  a per-BLOCK (504 m) or per-ROI map, consistent with project convention.
- M4's polynomial background can absorb a genuinely BROAD band (> ~60 nm);
  the method is specific to narrow OH features by construction. State that
  limitation with any null result.
- Residual column striping in the ratioed chain will feed the second
  derivative; M3 must be run after the striping screen or on block stacks.
- Same treatment is warranted at 2.3 um before any future band-depth
  claim there, though the olivine wing is far weaker at that wavelength.
