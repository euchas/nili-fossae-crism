# 01 — Stage A validation report

Generated 2026-04-20 UTC from `code/stage_a_build_feature_stack.py`,
`code/stage_a_sanity_plots.py`, and `code/stage_a_validation_3e12.py`.
All artefacts under `outputs/feature_stacks/`, `outputs/figures/`,
`outputs/logs/`.

## What Stage A produced

Per-cube 8-parameter feature stacks (Brown v5 centroid/asymmetry/amplitude/
hwhm + VB OLINDEX3/D2300/BD1900R2/BD2500_2) saved as both dense `.npz` and
tidy `.parquet`, plus JSON run logs that record random seed, MD5s of the
source modules, per-parameter NaN fractions, and path provenance.

## Brown 2020 reproduction checkpoint (FRT00003E12)

**PASS on all three conditions.**

| condition | expected | observed |
|---|---|---|
| centroid mode | 1.15–1.30 µm | **1.166 µm** |
| centroid median | ≥ 1.15 µm | **1.290 µm** |
| centroid 95th percentile | ≥ 1.30 µm | **1.386 µm** |

The centroid-vs-asymmetry hexbin
(`FRT00003E12_brown2020_centroid_asymmetry.png`) reproduces the Brown
2020 Figure 7b locus: a dense main body at (centroid ≈ 1.17–1.25 µm,
asymmetry ≈ 1.5–3.5) and a diffuse long-wavelength tail extending to
(centroid ≈ 1.40 µm, asymmetry ≈ 7+) corresponding to the coarse-grained
dune population Brown 2020 flagged with a red arrow in their Figure 7c.
The v5 maps therefore reflect the published Brown 2020 fitting and are
trustworthy for downstream analysis.

> *Process note: my first draft of the validation used expected range
> 1.04–1.14 µm, which reflects textbook forsteritic olivine band centres
> rather than Brown 2020's published distribution. Brown 2020 explicitly
> restricts their analysis to centroid ≥ 1.15 µm. I corrected the
> checkpoint thresholds before reporting; commit the corrected script,
> not the original.*

## Valid-pixel yield per cube

| cube_id      | Ls    | n_valid | valid % | residual median | centroid median (µm) |
|--------------|-------|--------:|--------:|-----------------:|----------------------:|
| FRT00003E12  | 166°  | 237 414 |   77.3  | 0.00894          | 1.290                |
| FRT0000871C  | 337°  | 155 973 |   50.8  | 0.00516          | 1.198                |
| FRT0000A4FC  |  42°  |  25 476 |    8.3  | 0.00956          | 1.243                |
| FRT0000B438  |  91°  | 120 812 |   39.3  | 0.00921          | 1.268                |

`valid %` is the fraction of the 480 × 640 frame that survives the AND of
the Brown v5 mask, finite Brown parameters, and finite VB parameters. The
gating criterion is effectively the Brown v5 mask: Brown-valid and
VB-finite almost perfectly agree (within 1 pixel in B438; exact everywhere
else), so everything excluded by the v5 mask is excluded from Stage A.

**A4FC only contributes 25 k valid pixels** (≈ 10× fewer than 3E12). Looking
at `FRT0000A4FC_stackA_param_maps.png`, the olivine unit is a small patch
in the centre-left of the frame; most of the cube's footprint is either
off-unit or masked by Brown. For cross-cube distributional tests, A4FC
has commensurately less statistical power and will be flagged in every
four-cube plot.

**871C has the lowest residuals of the four cubes** (median 0.00516 vs.
0.00894–0.00956 for the others). That's informative: despite 871C being
the Ls ≈ 337° dust-storm-season cube, its 1 µm band fits are cleaner,
not worse, than the other three. The residual histogram
(`FRT0000871C_stackA_residual.png`) is unimodal with no bad-fit
secondary mode, so its distributional differences from 3E12 are not
explained by atmospheric contamination.

## Parameter distributions across cubes

See `all_cubes_stackA_distributions.png` (all eight parameters overlaid)
and `all_cubes_centroid_hist.png` (centroid alone with the Kremer
homogeneity framing).

**The four cubes do *not* share a common centroid distribution.** They
span roughly:

- **871C:** sharply unimodal, peaked at 1.166 µm (the low edge of Brown
  2020's analysis window), with a long right-tail out to ~1.35 µm;
- **3E12:** bimodal — a minor peak near 1.17 µm and a dominant peak at
  1.36 µm (the coarse-dune population Brown 2020 already noted);
- **B438:** broad unimodal distribution centred near 1.30 µm;
- **A4FC:** nearly flat in 1.18–1.32 µm, with thin statistics.

Per-cube histograms (`<cube>_stackA_histograms.png`) also show bimodal
asymmetry, amplitude, and HWHM distributions in 3E12 and B438 that are
absent or weaker in 871C. This bimodality is very likely the regional
olivine unit bedrock (less redshifted) vs. dune / coarse-grained olivine
(more redshifted) split documented in Brown 2020 Figure 7.

**First-pass implication for the Kremer test.** Kremer 2019's airfall
hypothesis predicts olivine compositional homogeneity across the
~70 000 km² outcrop. The naive centroid overlay shows the four cubes
are *not* samples of the same centroid distribution. Two caveats before
treating that as evidence against Kremer:

1. The four cubes **do not all cover the same spatial footprint of the
   olivine unit**. Some of the inter-cube difference is simply "which
   part of the unit each cube observed" — e.g. 3E12 contains the
   Horseshoe's prominent olivine-rich dune field; 871C sits slightly
   NW and has far less dune coverage. The regional OU bedrock population
   *might* still be homogeneous after dune pixels are separated. This
   is a Stage C / D question (cluster first, compare populations
   across cubes later).
2. Kremer's prediction applies to the *olivine composition itself* (i.e.
   Fo#), not to the 1 µm band centroid directly. The Brown 2020 centroid
   is a joint function of Fo#, grain size, and band saturation. Coarse
   ejecta grading by fallout distance could produce a draped deposit
   with homogeneous Fo# but spatially varying grain size — which would
   shift centroids even under airfall. This does not save Kremer
   outright, but it does mean the centroid shift alone is not a
   knockout test.

Both caveats argue the four-cube comparison must be done on matched
sub-populations, and must distinguish Fo#-driven shifts from grain-size
shifts. The existing 8-parameter stack cannot do that alone; adding
BD2290 / BD1400 / LCPINDEX2 in Stage D may help.

## Parameter redundancy within the 8-parameter stack

`FRT00003E12_stackA_correlation.png` shows that in 3E12:

- the four Brown parameters (centroid, asymmetry, amplitude, hwhm) have
  pairwise |Spearman ρ| ≥ 0.89 — they are essentially four views of the
  same "olivine redshift intensity" axis;
- OLINDEX3 correlates strongly with the Brown parameters
  (ρ = +0.75 with centroid, +0.89 with amplitude) — again, olivine
  strength;
- D2300 and BD1900R2 are correlated with each other (+0.60) but only
  weakly with the olivine parameters — they load hydration / OH;
- BD2500_2 is near-orthogonal to everything else (|ρ| ≤ 0.15) —
  carbonate loads its own axis.

The 8-parameter stack therefore encodes roughly **three** independent
information axes: olivine strength, hydration, carbonate. The same
correlation structure holds in the other three cubes (figures available
but not reproduced here). This is a strong argument for the Stage D
12-parameter extension to bring in parameters that are spectrally
orthogonal to this 3-axis basis (SINDEX2 for sulfate convexity,
BD2290 for Mg/Fe-OH narrow band, BD1400 for structural OH + serpentine,
LCPINDEX2 for low-Ca pyroxene), so PCA / UMAP are not starting from a
rank-3-ish feature matrix.

## Residual distributions

See `<cube>_stackA_residual.png`. All four cubes have unimodal residual
distributions. No cube shows a bad-fit secondary mode. Residual medians
are 0.005 (871C) to 0.010 (A4FC); the differences do not track
atmospheric season (871C is the dust-season cube but has the cleanest
fits). I'd read that as Brown v5 having already done a good job of
rejecting noisy pixels through the mask — there is no population of
"valid-mask but high-residual" pixels that needs a post-hoc cutoff at
Stage B. **Recommend: keep the Stage A plan of threshold-late.** We can
revisit residuals if Stage B/C clustering reveals a residual-driven
artefact.

## Spatial parameter maps

`<cube>_stackA_param_maps.png` for all four cubes show coherent spatial
structure matching the expected olivine-unit geometry (central bright OU
outcrop in 3E12 consistent with the Horseshoe; extended OU in 871C and
B438; small olivine patches in A4FC). D2300 and BD1900R2 highlight the
phyllosilicate basement around the OU as expected. BD2500_2 is
dominated by near-zero values with modest positive regions — carbonate
is sparse and localised, consistent with Ehlmann 2009 / Brown 2010 /
Viviano 2013 characterisations.

**Display convention (added after Stage A review).** Cubes are stored
on disk in CRISM detector orientation, which is upside-down and mirrored
relative to standard Mars-map view. All spatial figures in
`outputs/figures/` are now rendered through
`display.apply_display_transform` — `FLIP_VERTICAL = True`,
`FLIP_HORIZONTAL = True` — matching the existing
PCA_CRISM_analysis_v2.py convention. Data on disk is unchanged; the flip
is display-only.

**Mask convention for parameter-map displays (added after Stage A
review).** Previously all eight parameter panels used `joint_valid`
(Brown AND VB). For display this was wrong for the four VB parameters:
OLINDEX3, D2300, BD1900R2, and BD2500_2 do not depend on the 1 µm
olivine fit succeeding, so applying the Brown mask discarded real
signal about hydration, Fe/Mg-OH, and carbonate outside the olivine
unit. The sanity plots now use:
- Brown 4 (centroid, asymmetry, amplitude, hwhm): `brown_mask & finite(param)`;
- VB 4 (OLINDEX3, D2300, BD1900R2, BD2500_2): `finite(param)` only.
The combined `joint_valid` mask is still used for histograms,
correlation matrices, and every feature-matrix column that will feed
PCA / UMAP / clustering in Stage B (where each pixel must have every
feature defined).

### Known limitation — BD2500_2 column striping

The BD2500_2 map shows visible column striping in all four cubes — fine
vertical stripes at the detector-column frequency. This is residual
detector noise in the TER3 joined product rather than a flaw in the
parameter computation (BD2500_2 is a simple three-channel band depth at
2.364/2.480/2.570 µm; noise in any of the three anchors shows through
directly). Not a bug, and no action needed for Stage B, but worth noting
for any future per-column or dune-wavelength-scale analysis:
- cross-cube BD2500_2 medians should be treated with a systematic
  uncertainty at least as large as the striping amplitude;
- any analysis that correlates BD2500_2 with linear spatial features
  running along the detector column direction (i.e. north–south in
  detector coordinates, which maps to the display's vertical axis after
  flip) may be partially confounded by the stripes.
BD1900R2 and the Brown parameters do not show the same striping at
visible amplitude.

## Items flagged for your decision before Stage B

1. **Cross-cube comparability strategy.** Raw centroid distributions
   clearly differ across cubes. Two non-exclusive options:
   - **(a)** Per-cube independent clustering (Stage B–C as planned),
     then match clusters across cubes by parameter-space proximity
     (Stage E). Feature matrices are **not** standardised across cubes.
   - **(b)** Same as (a), but also produce a "cube-id-conditioned"
     diagnostic: for each cube, re-compute centroid-median-subtracted
     values and plot those distributions too. If the mean-subtracted
     distributions *do* collapse to a common shape, that would support
     a common underlying Fo# with atmospheric / illumination offsets;
     if they still diverge, that's a stronger statement about real
     mineralogical variability. (b) is an extra plot, not a different
     analysis; adds maybe 5 minutes of work.
   
   My preference: (a) plus (b) as a diagnostic. Flag if disagree.

2. **A4FC inclusion.** With only 25 k valid pixels, A4FC will be
   marginal for any density-based clustering (HDBSCAN `min_cluster_size`
   scales with total points). Options:
   - keep A4FC in Stage B with a scaled `min_cluster_size`;
   - defer A4FC to Stage E's cross-cube comparison only and do the
     per-cube cluster development on the other three;
   - drop A4FC entirely.
   I'd prefer the first: keep it but flag the reduced statistical power
   in every cross-cube plot.

3. **Parameter redundancy.** The PCA in the original script will put
   ~70 % of its variance on a single "olivine" axis. That's fine as a
   diagnostic but a waste for clustering. For Stage B I'd like to run
   PCA on both the 8-parameter stack (as a baseline) and a **4-parameter
   reduced stack** (one representative from each information axis:
   centroid, OLINDEX3, D2300, BD2500_2) to see if dropping the
   redundant shape parameters improves UMAP separation. This is cheap
   and adds one extra figure per cube.

4. **FRT0000871C olivine population.** 871C is systematically
   blueshifted (median centroid 1.198 µm) compared to 3E12 (1.290 µm).
   If this reflects a real difference in Nili Fossae olivine
   composition/grain size between the two sites, it's an interesting
   finding. Before Stage B I'd appreciate context: do you know whether
   the 871C footprint contains a distinct sub-region of the olivine
   unit (less-altered bedrock, higher-Mg forsterite exposure, fewer
   dunes), or is this the first time you've seen a cube-level difference
   of this magnitude?

## Status

- Stage A executed; all planned outputs produced. **Stopping here per
  your instruction.** No Stage B work will start until you've reviewed
  the figures.
- Feature-stacks and logs are ready to be consumed by the next stage's
  scripts without re-loading cubes.
- CONTEXT.md updated to decouple Brown 2010 (hydrothermal hypothesis)
  from Brown 2020 (v5 map methodology).
