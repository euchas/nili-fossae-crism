# Clay provinces, 2.33 um tail attribution, and unit-internal anomaly ratios

Run 2026-09-01: `code/contact_clay_provinces.py` (cap-ratioed province and
tail stacks, cafeco3 correlation) and `code/contact_anomaly_ratios.py`
(Jezero-style unit-internal ratios: anomaly-pixel stacks / median of the
whole unit). All stack-level, descriptive, no CIs; formal claims need the
Tier 1 block machinery. Data products: MTR3 I/F throughout; Plebani patch
catalogue (TRR3 sensor-space chain) reused for locations only.

## 1. Clay provinces are a real spectral facies split

Connected clay components (f_clay >= 0.5) classified by median strong-tier
centre at 2.311 um; stacks from f_clay >= 0.95, f_oliv = 0 pixels only.

- 3E12: short province = the large EASTERN basement block (4 components,
  9.0k pure px), stacked centre 2.3029; long province = the rest,
  including all clay adjacent to the olivine unit (15 components, 82k px),
  stacked centre 2.3101. On high-SNR stacks the two provinces differ by
  ~7 nm (about one channel) - this is not pixel noise.
- B438 repeats the split (2.3046 vs 2.3111); A4FC classifies uniformly
  short (2.3085); 871C uniformly long at component scale (2.3090) - its
  KDE bimodality lives BELOW component scale (mixed within components).
- Character of the long province (3E12, B438): DEEPER 2.3 band (e.g.
  0.037 vs 0.029 in 3E12 - so not the weak-band drift artifact), stronger
  2.35 septum (0.019 vs 0.013) and 2.39 um band (0.010 vs 0.007), 1.39 um
  centre shifted short (1.408 vs 1.417), similar 1.9 um at province scale.
  Feature-space reading: Mg-phyllosilicate-leaning (saponite/talc-like)
  facies vs an Fe-smectite/nontronite-leaning facies (2.303, 1.417+).
- Unit-internal deciles (f >= 0.5 pixels) sharpen this: high-centre clay
  pixels are LESS hydrated than the clay median (bd1900 -0.007 in 3E12,
  -0.005 A4FC, -0.003 B438) with a strong red-slope difference; low-centre
  pixels add a 2.29-2.30 nontronite-like differential band. Caveat: the
  decile sets use f >= 0.5, so some sub-pixel olivine mixing can
  contribute to the high-centre end; the f >= 0.95 province stacks are the
  cleaner statement.
- Geography (3E12): the long/Mg-leaning province includes the clay
  polygons along the olivine-unit contact; the Fe-smectite province is the
  eastern block away from the unit. Contact-relevant arrangement -
  hypothesis for the revision, pending formal stats and the TER3/TRR3
  replication.

## 2. The >= 2.325 um tail is a real, ANHYDROUS 2.332 um sub-population

Unit-internal ratios (tail stack / olivine-unit median), all four cubes:

| cube | n_tail | differential band centre | bd1900 excess |
|---|---|---|---|
| 3E12 | 5,121 | 2.3320 (depth 0.009) | -0.0004 (none) |
| 871C | 808 | 2.3328 (0.011) | +0.0006 (none) |
| A4FC | 588 | 2.3314 (0.009) | -0.0006 (none) |
| B438 | 3,664 | 2.3321 (0.008) | +0.0009 (none) |

A sharp ~2.332 um excess absorption with zero added hydration, replicated
in every cube. Not the MgCO3 patches (anti-associated, see modes REPORT).
Attribution: Fe/Ca carbonate (2.33 + 2.53 um, anhydrous) is the leading
candidate, but the 2.53 um partner is only suggestive: present in the
cap-ratioed stacks of 3E12 (centre 2.527) and 871C (2.524) at depths
0.002-0.003, NOT robust in the unit-internal ratios (fitted 2.5 centres
scatter 2.48-2.54). Fe-chlorite fits poorly (predicts 2.35 and some
hydration). Provisional label: "anhydrous 2.33 um phase, possibly
Fe/Ca-bearing carbonate"; needs the TER3/TRR3 replication and a spatial
map of tail pixels.

Plebani cafeco3_max channel: no on-map patch reaches 0.3 (max 0.248) - the
classifier detects no CaFe carbonate AT the MgCO3 patches, and the
catalogue cannot test for it elsewhere (patch detection is MgCO3-driven).
Patch-local centre vs cafeco3_max correlations are weak/mixed (rho +0.15
3E12, +0.46 871C n=20, -0.22 B438; naive p-values, patch level).

## 3. Olivine low-centre anomalies = hydrated smectite-like additive

Low-decile olivine pixels (centre <= ~2.304) vs unit median add a
2.29-2.30 um band WITH 1.9 um hydration (bd1900 +0.013 in 3E12) and a
1.42-1.43 um feature - the clay-contamination/carbonate signature, spatially
the near-contact rims (modes REPORT sec 2). Consistent with the mixing
reading of the near-contact centre shifts.

## Outputs

provinces/: `clay_province_metrics.csv`, `tail_stack_metrics.csv`,
`cafeco3_correlation.csv`, `fig_provinces_{cube}.png/pdf`,
`fig_tail_{cube}.png/pdf`, `run_log_provinces.json`.
anomaly_ratios/: `anomaly_ratio_metrics.csv`, `fig_anomaly_{cube}.png/pdf`,
`run_log_anomaly_ratios.json`.

## 4. Where the high-decile pixels fall (2026-09-01 addendum)

`contact_anomaly_pixel_maps.py` -> anomaly_ratios/fig_highdecile_map_{cube}.
In 3E12 the olivine high-decile/tail pixels are NOT random: they form a
dense, spatially coherent field in the NW olivine unit - north/above the
Horseshoe feature - well away from the MgCO3 patch chains (visual
confirmation of the anti-association), plus sparser southern clusters.
N-S-trending pixel strings near the EASTERN scene edge are flagged as
possible column/edge artefacts and should not be interpreted. Clay
high-decile pixels concentrate in the same polygons as the long province
(internally consistent). The NW field is a mappable target for a
region-specific stack and HiRISE/CaSSIS context.

The anomaly-ratio figures were regenerated with a labelled 2.3 um zoom
panel (ticks 2.30-2.35, reference line at 2.30): the olivine high-decile
differential dip bottoms at 2.33 exactly; the low decile at 2.30; the clay
high-decile shows a relative PEAK at 2.31 (shallower 2.3 band than the
clay median, with the band displaced longward).

## Next steps (agreed direction)

- Replicate the whole band-centre chain on independent starting data:
  TER3 sensor space and/or the Plebani TRR3 pipeline (AV, 2026-09-01).
- Spatial map + block-level statistics for the 2.332 um tail population.
- Formal block CIs for the province contrasts (Tier 1 machinery) before
  any of this enters the manuscript.
