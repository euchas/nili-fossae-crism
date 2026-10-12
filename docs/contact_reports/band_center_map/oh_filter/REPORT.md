# Per-pixel 1.4 um OH detection map (matched filter, Plebani chain)

Run 2026-09-05, `code/contact_oh_matched_filter.py`, all four cubes.
Products: `fig_oh_{cube}.png/pdf` (left: detections coloured by fitted
band centre; right: amplitude SNR everywhere), `{cube}_oh_centre_det.npy`,
`{cube}_oh_snr.npy`, `{cube}_ohfit_sensor.npz` (per-sensor-pixel amp /
snr / centre / fwhm / null_snr / det / column), `oh_unit_summary.csv`,
`oh_column_stratification.csv`, `run_log_oh_filter.json`.

## Method (the hardened M4 of SHAPE_PARAMS.md, made fast)

Per sensor pixel of the bland-ratioed TRR3 cube: cubic background fitted
by linear least squares over 1.28-1.60 um EXCLUDING the test window
(+/-12 nm guard); continuum-removed residual matched-filtered against
Gaussians on a grid of centres (2 nm step) and observed FWHM 20/26/32/40
nm (bounded above the ~14 nm instrument response); closed-form amplitude
per grid point; best (centre, FWHM) by amplitude SNR, where the noise is
the per-pixel residual std in band-free channels. The IDENTICAL estimator
is run in two band-free windows (1.305-1.350 and 1.470-1.515 um, 46 trial
positions vs 31 in the signal window, so look-elsewhere is over-charged)
and the detection threshold is the 99th percentile of the null max-SNR
over all usable pixels. No number in the threshold comes from a table.

## Empirical thresholds

| cube | null 99th-pct SNR | detections | % of usable |
|---|---|---|---|
| 3E12 | 9.46 | 15,431 | 5.4% |
| 871C | 8.44 | 11,744 | 4.1% |
| A4FC | 8.08 | 26,060 | 9.1% |
| B438 | 10.15 | 11,894 | 4.2% |

A textbook SNR threshold of 3-5 would have flooded the map: the empirical
null is 2-3x heavier-tailed than white-noise theory predicts, exactly as
the methods review warned (ISRF-correlated residuals + residual striping).

## Validation gate (SHAPE_PARAMS.md) - all passed

| test | result |
|---|---|
| (iv) Negative control - anhydrous cap detection fraction | 0.94% / 0.27% / 1.16% / 0.48% - at the 1% false-positive floor |
| (iii) Orthogonality - rho(amplitude, OLINDEX3) in map-pure olivine | -0.10 / +0.01 / -0.09 / +0.03 (was 0.56-0.83 with the chord method) |
| (iii) same, map-pure clay | +0.01 / +0.16 / +0.32 / +0.12 (was 0.58-0.64) - mostly cleared; A4FC retains 0.32 (subpixel olivine mixing in its large clay unit is the likely residual) |
| positive control - map-pure clay detection fraction | 23.1% / 9.2% / 16.0% / 8.3% - 10-25x the false-positive floor |
| olivine unit detection fraction | 0.86% / 1.24% / 2.11% / 0.75% - indistinguishable from the floor except marginally in A4FC |
| (v) column stratification | detection fraction varies by column bin (std 0.03-0.06) but the variation tracks which units each column crosses; a few individual columns cross threshold as visible vertical streaks (3E12 centre-south, ~1.42 um) - striping residue, small in number, to be screened |

## What the map shows (3E12)

Detections fall almost entirely inside the hand-mapped clay polygons - the
NE basement block, the western polygons, the southern polygons - and the
olivine unit interior is empty. Fitted centres in clay: median 1.402 um
(3E12), 1.410 (871C), 1.408 (A4FC), 1.408 (B438), amplitudes 1.0-1.3%.
Within 3E12 there is visible centre structure among clay polygons
(~1.400 in the NE block vs ~1.410-1.420 in the southern polygon); at SNR
~10 and FWHM ~30 nm the per-pixel centre precision is ~1-2 nm (grid
quantized to 2 nm), so a 10-15 nm polygon-to-polygon difference is
formally resolvable within this chain - but see caveats.

## What it means

1. The olivine unit has NO detectable narrow (20-40 nm) OH band at the
   per-pixel level: its detection fraction sits at the false-positive
   floor in three cubes and at 2x the floor in A4FC. Combined with the
   stacked-spectrum prototype (SHAPE_PARAMS.md: injection limit 0.5-1.0%
   on a 230-pixel stack), the defensible statement is an upper limit of
   order 0.5-1% on narrow OH band depth in the olivine unit, with the
   explicit scope "narrow band; a broad feature would be absorbed by the
   background".
2. The clay basement carries a real, shape-verified OH band at
   1.40-1.41 um in this chain, detected in 8-23% of its map-pure pixels.
3. Criterion B of the supplement (no temperature-diagnostic phase confined
   to contact zones) is reinforced: no OH detections form a halo or rim
   around the olivine bodies; detections are unit-bound.

## Caveats

- Absolute band-centre values are NOT usable for mineral identification:
  the ~13 nm inter-chain wavelength offset equals the 1.39 (serpentine)
  vs 1.41 (smectite) separation. Within-chain RELATIVE differences
  (polygon-to-polygon) are meaningful; assignments are not.
- Per-pixel maps carry the n_eff ~1% autocorrelation caveat; any
  statistic drawn from them must be block-level. The per-pixel map is a
  detection product, not a significance map.
- A small number of detector columns cross threshold as vertical streaks
  in the olivine unit; a column-wise stripe screen (or a block-median
  version) should precede any count-based statement about the olivine
  unit. The east scene-edge strip is also artifact-prone.
- Background order (cubic) and the 12 nm guard were fixed a priori; the
  shoulder/window jitter envelope has not yet been propagated into a
  per-pixel error bar (SHAPE_PARAMS.md hardening item 5) - required
  before any amplitude is quoted with a formal uncertainty.

## Next steps (not run)

- Block-median (504 m) version with the jitter envelope -> the quotable
  3-sigma upper limit for the olivine unit and a formal clay detection.
- Stripe screen by detector column before counting olivine detections.
- Test the within-clay centre structure at block level (real
  nontronite-vs-Mg-smectite variation, or residual?).
- Same estimator at 2.3 um (M4-style narrow-band test) if a shape-verified
  2.3 um product is wanted; the wing problem is much weaker there.
