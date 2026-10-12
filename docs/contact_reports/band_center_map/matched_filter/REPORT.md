# Matched-filter band detection at 2.3 and 1.9 um (Plebani chain)

Run 2026-09-05, `code/contact_band_matched_filter.py {2300|1900}`; same
estimator and validation as the 1.4 um reference run
(`oh_filter/REPORT.md`). Products per band under `matched_filter/{band}/`:
`fig_{band}_{cube}.png/pdf`, `{cube}_centre_det.npy`, `{cube}_snr.npy`,
`{cube}_fit_sensor.npz`, `unit_summary.csv`, `column_stratification.csv`,
`run_log.json`.

## 2.3 um (metal-OH): validated, and a new spatial result

Configuration: fit 2.14-2.48 um, signal centres 2.280-2.350, FWHM
25/35/45/60 nm, nulls 2.175-2.225 and 2.400-2.450 (may hold weak real
features -> conservative threshold). Empirical thresholds SNR 9.5-10.6.

| test | result |
|---|---|
| cap negative control (detection fraction) | 0.8% / 1.0% / 2.8% / 1.2% - at the floor (A4FC cap is small and clay-adjacent) |
| rho(amplitude, OLINDEX3), map-pure olivine | +0.01 / +0.23 / +0.00 / +0.05 - no wing leakage |
| rho(amplitude, D2300), map-pure clay | 0.71 / 0.70 / 0.79 / 0.70 - the filter measures what D2300 measures where D2300 is valid |
| clay positive control | 46% / 29% / 35% / 28% detected, centre 2.300 um, amplitude 1.2-2.0% |
| olivine map-pure | **10.7% / 7.6% / 10.2% / 2.6%** detected - well above the floor in three cubes; centre 2.304-2.306 um (one channel longward of clay), amplitude 0.9-1.7%; rho(amp, D2300) 0.43-0.52 |

Unlike 1.4 um, the olivine unit carries genuine, shape-verified 2.3 um
detections in ~10% of its map-pure pixels - and they are NOT randomly
placed. In 3E12 (`fig_2300_FRT00003E12.png`) they form a broad, non-stripy
swath through the WESTERN olivine unit (between the western clay polygons
and the Horseshoe rim, extending several hundred metres from mapped
clay), plus fringes around several clay polygons and the NW polygons; the
eastern/central olivine interior is empty (SNR 0-2). This is the same
geography as the verified block-level D2300 hydration gradient and the
carbonate-patch concentration, now in a form that cannot be a
continuum-curvature artifact.

Candidate explanations, NOT yet discriminated:
(a) hydrated alteration of the olivine unit (smectite/serpentine-side,
    2.30-2.32 um) - the contact-associated alteration hypothesis;
(b) Mg-carbonate (2.30 + 2.50 um): the filter at 2.30 with 60 nm FWHM also
    responds to the carbonate 2.3 band, and the Plebani MgCO3 patches line
    the western Horseshoe rim - the 2.50 um partner would discriminate;
(c) sub-pixel clay contamination: predicts detections hugging polygon
    edges, but the western swath extends well beyond the 36-90 m
    mixing zone, so it cannot explain the swath alone.
Because the bland denominators are largely olivine-like, this map shows
what differs from the LOCAL background; the unit-wide 2.32 um component
found in the MTR3 chain is partly cancelled here by construction.

## 1.9 um (H2O): estimator not suitable as configured - zero detections

Configuration: fit 1.74-2.10 um, signal 1.895-1.945, FWHM 40-90 nm, nulls
1.755-1.795 and 1.805-1.845 (both on the short side to avoid the 2.0 um CO2
region). Empirical thresholds came out at SNR 32-42 (vs 8-11 for the
narrow bands) and NO pixel passes in any unit - including the clay
basement, which certainly has a 1.9 um band (rho(amp, BD1900R2) = 0.58-0.67
in clay shows the filter does measure it).

Diagnosis: the null windows sit on the flattening olivine 1 um wing, and a
40-90 nm Gaussian template correlates strongly with residual background
curvature there, so the null max-SNR is enormous. This is exactly the
broad-template / background-curvature degeneracy the methods review
warned about: the narrow-band matched filter is the wrong instrument for
a band 2-3x wider than its own continuum model can separate. The result
is a design failure of the null, not evidence about 1.9 um hydration.
Options if a shape-verified 1.9 um product is wanted: (i) a null built
from phase-randomized residuals or shuffled templates instead of adjacent
spectral windows; (ii) a physically constrained background (olivine wing
model) rather than a free cubic; (iii) keep BD1900R2 for 1.9 um, whose
close anchors (1.85/2.07) are far less wing-affected than at 1.4 um and
which is already block-verified in the contact ledger.

## Next steps (proposed, not run)

1. Block-level (504 m, block bootstrap, seed 42) near/far contrast of the
   2.3 um detection fraction and amplitude in map-pure olivine vs
   distance to the mapped clay contact - the shape-verified version of the
   contact-halo hydration test.
2. Overlap of olivine-unit 2.3 um detections with the Plebani MgCO3
   patches, and a 2.50 um partner-band check on the detected pixels, to
   separate (a) from (b).
3. Stripe screen by detector column before any count-based statement.
