# Tier 0: per-pixel 2.3 um band-centre map, all four cubes

Run 2026-09-01, `code/contact_band_center_map.py` (deterministic, no RNG).
Extends `contact_band_center_2300.py`: full-grid centre/depth/edge-flag arrays
saved; map-based contact on the CTX-corrected frame; search window widened to
2.265-2.355 um; argmin at a search-window edge excluded from the strong tier.
Tiers: strong = continuum-removed depth >= 0.015 (single-pixel centres ~3-8 nm;
bin medians of >= 10^3 pixels limited by systematics, not noise); weak =
0.005-0.015 (noise-dominated, context only; known drift toward ~2.318-2.320).

## Headline numbers (strong tier, median [q25-q75])

Unit medians (um):

| cube | olivine map-pure | clay map-pure | cap map-pure | strong px in pure olivine |
|---|---|---|---|---|
| 3E12 | 2.3177 [2.3136-2.3195] | 2.3137 [2.3074-2.3163] | 2.3155 (n=5,400 only) | 59,975 (32%) |
| 871C | 2.3181 [2.3120-2.3198] | 2.3166 [2.3064-2.3179] | 2.3183 (n=2,435) | 9,054 (19%) |
| A4FC | 2.3124 [2.3066-2.3173] | 2.3105 [2.3061-2.3144] | 2.3114 (n=3,471) | 15,653 (35%) |
| B438 | 2.3121 [2.3068-2.3202] | 2.3117 [2.3086-2.3159] | 2.3114 (n=3,262) | 21,581 (23%) |

Distance profile, map-pure olivine, strong tier (median centre, um):

| cube | 36-90 m | 90-150 | 150-250 | 250-400 | >400 | far minus near |
|---|---|---|---|---|---|---|
| 3E12 | 2.3168 | 2.3173 | 2.3179 | 2.3182 | 2.3183 | +1.5 nm |
| 871C | 2.3179 | 2.3180 | 2.3185 | 2.3186 | 2.3193 | +1.4 nm |
| A4FC | 2.3112 | 2.3129 | 2.3161 | 2.3171 | 2.3170 | +5.8 nm |
| B438 | 2.3121 | 2.3131 | 2.3119 | 2.3117 | 2.3118 | -0.3 nm |

## Reading

1. NO ordered zonation at any resolvable scale. The strong-band tier is well
   populated (9k-60k pixels in the pure olivine unit per cube), so this is not
   a sensitivity-limited null at the >= 1-channel scale: there is simply no
   chlorite-range (2.33-2.35 um) population, no long-centre zone hugging the
   contact, and no multi-phase spatial ladder in any cube. Histogram support
   is confined to 2.30-2.325 um everywhere (minor 3E12 olivine bump at ~2.330,
   ~2 percent of pixels, unlocalized - noted for follow-up, not interpreted).
2. Where distance trends exist (3E12 +1.5 nm, 871C +1.4, A4FC +5.8), the
   centre moves TOWARD the clay value near the contact and rises to a common
   ~2.317-2.319 um far-field. This is the direction predicted by sub-pixel
   smectite mixing (and, degenerately, by the weak-band depth-centre drift,
   since band depth also declines away from the contact). It is the OPPOSITE
   of thermal-aureole zonation, which would put the longer-centre phases
   (serpentine/talc/chlorite) closest to the contact.
3. The olivine unit's strong-band phase sits at ~2.316-2.320 um - one channel
   longward of the clay basement (2.310-2.316) - throughout the unit, far from
   the contact included. In feature space this is the Ehlmann et al. (2009,
   Fig. 17D) "serpentinized olivine" 2.32 um signature, and it is UNIT-WIDE,
   not contact-concentrated.
4. B438 is the outlier: its strong-tier olivine centres sit at the clay value
   (~2.312) at ALL distances, and its strong pixels cluster around clay
   outcrops on the map - consistent with the previously inferred 20-30 percent
   sub-pixel clay cover in this cube. B438 also shows the worst column
   striping (weak tier); its fine structure should not be interpreted.

## Caveats

- Depth-centre covariance: the far-field drift toward ~2.318-2.320 um is the
  known weak-band attractor; even at depth >= 0.015 the +1.5 nm trends in
  3E12/871C are within the plausible bias amplitude. The A4FC +5.8 nm trend is
  larger but shares the bias/mixing direction. None of the trends can be
  attributed without the depth-matched and mixing-curve tests (Tier 1 gates).
- Channel quantization (6.6 nm) shapes the histogram peaks; sub-channel
  structure within 2.31-2.32 um is not resolved per-pixel.
- Cross-cube absolute centres are not comparable (871C/A4FC calibration
  offsets); only within-cube patterns are read here.

## Implication for the zonation criterion

The ordered-zonation test for contact metamorphism is NEGATIVE at every scale
this map resolves (>= 1 channel, >= 1 pixel). The 2.3 um band-position field
shows a unit-wide serpentinized-olivine-like signature plus near-contact
shifts in the mixing direction only. This is consistent with regional
alteration of the unit, and
distinguishable from a thermal aureole, which is not observed. The surviving
contact evidence remains the hydration-DEPTH gradient and carbonate-patch
concentration, with their documented ambiguity (alteration vs cover/mixing).

## Next steps (not run; need approval)

- GATE 2 (mixing curve) is now the highest-value cheap test: the observed
  centre-vs-distance pattern is its exact prediction. If clay+olivine mixing
  reproduces the profiles at matched depth, the band-position story closes.
- Depth-matched contrast (part of Tier 1) adjudicates the A4FC +5.8 nm trend.
- The 3E12 ~2.330 um minor population deserves a spatial look (is it the
  Plebani carbonate patches?).

Outputs: `{cube}_bc230_center.npy/_depth.npy/_edgeflag.npy`,
`band_center_map_summary.csv`, `band_center_map_profile.csv`,
`fig_map_{cube}.png/pdf`, `fig_center_vs_distance.png/pdf`,
`fig_histograms_strong.png/pdf`, `run_log_band_center_map.json`.
