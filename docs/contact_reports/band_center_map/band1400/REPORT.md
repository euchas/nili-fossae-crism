# 1.4 um band-centre campaign (2026-09-04)

Scripts: `contact_band_center_1400.py` (MTR3, two runs: full 1.36-1.49 um
window, then restricted 1.36-1.425 um), `contact_band1400_diagnostic.py`
(unit-median spectra), `contact_band_center_1400_plebani.py` (bland-
ratioed TRR3 chain, full window). Outputs under band1400/ and
band1400/plebani/.

## Findings

1. THE MTR3 1.4 um WINDOW IS SYSTEMATICS-LIMITED FOR BAND POSITION.
   The full-window minimum search locks onto two fixed wavelengths in
   every unit of every cube - 1.4404-1.4414 um and ~1.475-1.480 um - with
   sub-nanometre IQRs and depths up to 0.04 even on the anhydrous cap.
   The unit-median diagnostic (fig_diagnostic_unit_spectra) shows these
   as sharp dips common to clay, cap, and olivine (in 871C the three
   units are near-identical across the whole window): volcano-scan CO2
   residuals / instrument systematics, not mineralogy. Consequence:
   prehnite (1.48 um) CANNOT be tested in the MTR3 chain (the ~1.477
   residual sits on it), and the first-run "prehnite tail" counts were
   artifacts. Even the restricted window hits a common ~1.389 dip in
   871C. This also explains part of the known cross-cube BD1400
   standardization offsets (>2.7 sigma), and BD1400's long anchor
   (~1.467 um) sits on the 1.477 residual wing.
2. REAL UNIT-DEPENDENT DEPTH STRUCTURE EXISTS UNDERNEATH (AV's BD1400
   memory is correct) - BUT IT IS THE OLIVINE 1 um BAND WING, NOT
   HYDRATION (wing_check.csv, 2026-09-04). The olivine unit has the
   deepest broad 1.37-1.45 um depression in every cube (e.g. 3E12 p50
   0.033 vs cap 0.019), and that depth correlates with OLINDEX3
   (Spearman rho 0.56-0.83) and the 1 um centroid (0.55-0.78) but
   near-zero or negatively with BD1900R2 (-0.16 to +0.11): a straight
   continuum across the concave long-wavelength wing of the olivine 1 um
   composite band manufactures the "depression". In the clay unit the
   hydration correlations rise (D2300 0.30-0.71, BD1900R2 up to 0.62) -
   BUT NOT CLEANLY (corrected 2026-09-04): clay also correlates with
   OLINDEX3 at 0.58-0.64 in 3E12/A4FC/B438, exceeding its BD1900R2
   correlation in 3E12 (0.599 vs 0.136); only 871C is a clean hydration
   case. The defensible positive control is the DIRECT shape-verified
   measurement on the bland-ratioed clay ROI (centre 1.394 um, FWHM
   34 nm, delta-BIC +57; SHAPE_PARAMS.md), not the depth correlations.
   Consequences:
   (a) 1.4 um DEPTH in the olivine unit is an olivine-abundance proxy,
   not a hydration measure - this also contaminates BD1400 there;
   (b) the earlier reading of the olivine-unit ~1.390 um minimum as
   serpentine/talc OH is WITHDRAWN as a depth claim (the sharp position
   may still contain an OH contribution, but it rides on the wing and
   is not separable here); (c) in the bland-ratioed chain the olivine
   unit's 1.43-1.45 um "centres" are the wing-curvature residual
   minimum, not an absorption band.
3. PLEBANI CHAIN: the column ratio cancels the fixed dips (no sub-nm
   pile-ups; distributions broaden to ~30 nm IQR). Clay retains a
   genuine unit-specific 1.389-1.392 um OH peak in all four cubes; the
   olivine unit's differential minima are broad (1.42-1.45) because the
   bland denominators are drawn largely from olivine-unit-like scene
   pixels, cancelling its own shared band. Distance profiles are
   non-monotonic (inverted-U, +-3-5 nm) - no coherent contact trend.
4. PREHNITE: TESTED, NEGATIVE. In the ratioed chain (the only legitimate
   window onto 1.48 um) there is no discrete 1.48 um population in any
   unit; the fraction of strong-tier pixels with minima >= 1.46 um is
   unit-independent (3E12: olivine 12%, cap 13%; same pattern in the
   other cubes) - a noise/residual tail, not a phase - and shows no
   contact association. Combined with the empty 2.33-2.36 um range, the
   temperature-diagnostic-phase criterion (supplement criterion B) is
   now backed by a dedicated search. Supplement updated accordingly.

## Ledger

- KEPT: criterion-B prehnite search negative (dual statement: 2.33-2.36
  empty in both chains; 1.48 no discrete population, ratioed chain).
  Clay 1.389-1.392 um OH peak (ratioed chain) is real mineralogy. The
  ratioed-chain 1.4 um map is a genuine UNIT DISCRIMINATOR (clay
  1.39-1.41 vs olivine-unit 1.43-1.45), but the two sides mean
  different things: clay = OH band position; olivine = 1 um wing
  curvature (see finding 2).
- DO NOT USE: any 1.4 um band-POSITION map from the MTR3 chain; the
  first-run 1.44/1.48 "populations"; 1.4 um positional distance trends
  in either chain; 1.4 um DEPTH in the olivine unit as a hydration
  measure (it is an olivine-abundance/wing proxy - applies to BD1400
  there too).
