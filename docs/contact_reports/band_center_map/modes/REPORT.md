# Mode-split analysis of the 2.3 um band-centre maps

Run 2026-09-01, `code/contact_band_center_modes.py`, on the Tier 0 arrays.
Thresholds 2.308 / 2.311 / 2.314 um (2.308 and 2.314 fall between CRISM
channel centres 2.3046 / 2.3112 / 2.3178; 2.311 splits the middle pile).
Strong tier only (depth >= 0.015). Patch cross-check is pixel-level and
exploratory, NOT block statistics.

## 1. Channel comb vs real bimodality (fig_histograms_v2)

The jagged Tier 0 histograms were an aliasing artifact: 3.3 nm bins against
the 6.6 nm channel comb. With channel-aligned bins + 2.5 nm KDE:

- 3E12: olivine = ONE mode at ~2.317, cleanly offset from clay (~2.306-2.310
  broad). No internal olivine bimodality; small ~2.330 shoulder remains.
- 871C: genuinely bimodal (~2.305 and ~2.317) in BOTH units - and the split
  is spatially coherent (below), so it is not sampling noise.
- A4FC: broad, weakly two-humped olivine; clay single ~2.310.
- B438: olivine = clay-like peak ~2.311 plus a heavy long shoulder; cap and
  weak tier carry striping - treat B438 fine structure with caution.

A caution that applies throughout: a phase whose true centre falls midway
between channels (e.g. 2.313-2.314) splits its pixels between the adjacent
channel piles, so mode FRACTIONS partly re-express the local mean centre.
Spatial coherence, not the fraction itself, is the evidence of structure.

## 2. Olivine-unit mode split vs distance from mapped clay

Short-mode fraction (th 2.311), 36-90 m bin -> >400 m bin:

- 3E12: 0.25 -> 0.13 (monotonic decay); short-mode median distance 184 m vs
  long-mode 270 m.
- 871C: 0.23 -> 0.11; A4FC: 0.48 -> 0.23 (strongest decay; near-contact
  olivine shares the clay mode).
- B438: FLAT (0.38 -> 0.41), and short mode sits FARTHER than long
  (166 vs 145 m). Its short-mode pixels are unit-wide salt-and-pepper with
  stripe traces - consistent with distributed sub-pixel clay cover (the
  20-30 percent inferred by the mixing calculation) and/or striping, NOT
  adjacency mixing.

Reading: in 3E12/871C/A4FC the short-mode olivine pixels form rims near clay
and patches near carbonate detections - the arrangement predicted by
sub-pixel contamination, not by a distinct interior phase.

## 3. Carbonate-patch cross-check (plebani_tail_check.csv)

The Plebani MgCO3 patches ARE visible in the band-centre field - as
SHORT-centre anomalies:

- Median centre within 36 m of a patch vs unit median: 3E12 2.3098 vs
  2.3174; 871C 2.3146 vs 2.3179; A4FC 2.3101 vs 2.3115; B438 2.3100 vs
  2.3119. Mg-carbonate's 2.3 um band sits at 2.30-2.31, so an independent
  ML classifier and our independent band fit agree - a useful
  cross-validation of both products.
- Consequently the >= 2.325 um long tail is ANTI-associated with patches
  (enrichment within 54/108 m: 0.007/0.024 in 3E12, 0.13/0.07 in 871C,
  0.19/0.23 in B438). The 3E12 ~2.330 bump is NOT the MgCO3 patches.
  Untested candidates: Fe/Ca carbonate (2.33-2.34 um bands; the catalogue
  carries a cafeco3_max channel not yet used), chlorite-range material, or
  artifacts (especially in B438, n_tail=3,664 with striping).

## 4. New observation: two spatially coherent clay provinces

The basement clay is not one spectral population. In 3E12 the large eastern
basement block is short-mode (2.30-2.31, classic Fe/Mg smectite) while the
western clay polygons - the ones adjacent to the olivine unit - are
long-mode dominated (>= 2.311, toward 2.316). 871C shows the same
polygon-scale province structure (its real KDE bimodality is this), B438
partially (NE block long-mode, west/central shorter). The pattern is
organized by polygon/region, not by image column, so simple striping does
not explain it.

Candidate explanations (untested):
(a) real basement heterogeneity - a saponite/talc/serpentine-leaning
    province vs a nontronite/Fe-smectite province (cf. Viviano 2013
    talc-bearing basement);
(b) sub-pixel olivine-unit spillover into mapped clay polygons near the
    unit (f >= 0.5 display; but the effect persists in f >= 0.9 map-pure
    clay: long-mode fraction 0.63-0.70 at th 2.311 in 3E12);
(c) depth/dust differences between provinces interacting with the
    depth-centre bias.
If (a) survives testing, the long-centre clay province lies along the
olivine-unit contact in 3E12 - the first band-POSITION structure with a
contact-relevant geography. That makes discriminating (a) from (b)/(c) the
highest-value follow-up: stacked ratioed spectra of the two provinces
(secondary bands 1.39 / 1.9 / 2.39 / 2.5 um) at f_clay >= 0.95, f_oliv = 0.

## Outputs

`mode_fraction_vs_distance.csv`, `mode_split_summary.csv`,
`plebani_tail_check.csv`, `fig_modes_{cube}.png/pdf`,
`fig_short_fraction_vs_distance.png/pdf`, `fig_units_{cube}.png/pdf`
(unit-constrained centre maps), `fig_histograms_v2.png/pdf`,
`run_log_modes.json`.
