# Plebani-chain replication of the 2.3 um band-centre campaign - CLOSING VERDICT

Run 2026-09-01: `code/contact_band_center_plebani.py` +
`code/contact_plebani_figures.py`. Replication chain: TRR3 sensor-space
I/F ratioed per detector COLUMN by the crism_ml bland-pixel method (cached
stage-6 products, `plebani_inputs/ratioed/*_ratioed_bland.npz` in Zenodo
section E),
same band-centre fit as Tier 0, projected to the MTR3 grid by the stage-6
DDR recipe (per-cell median). The column-wise ratio cancels
column-correlated detector artefacts by construction, making this the
decisive test of the MTR3-chain (unratioed) findings.

## Verdict: no band-position claim survives dual-chain replication,
## except the negative zonation result, which is now final.

1. WITHDRAWN - the 3E12 NW high-decile/tail field. AV's diagnosis
   (detector noise: straight, slightly rotated N-S pattern) is confirmed.
   Under the column-cancelling ratio the 3E12 tail collapses from 5,121 to
   1,165 pixels and the dense NW field VANISHES (fig_tail_compare_3E12);
   the survivors are thin N-S strings and scene-edge clusters - residual
   artefacts themselves. In B438 (3,664 -> 1,156) the surviving pixels are
   overtly stripe-organized. 871C/A4FC tails collapse to 156/109 px.
2. WITHDRAWN - the "anhydrous 2.33 um phase, possibly Fe/Ca carbonate"
   lead (provinces/REPORT.md sec 2). The new-chain tail stacks still fit a
   ~2.33 differential band, but they are built on stripe/edge-selected
   pixels; with the spatial coherence gone and n collapsed, the feature is
   attributed to residual column noise, not mineralogy. Do not resurrect.
3. NOT REPLICATED - the smooth monotonic centre-vs-distance trends of the
   MTR3 chain (fig_profile_compare): the Plebani-chain profiles are
   non-monotonic in all four cubes (3E12 rises then DROPS at > 400 m;
   871C/A4FC reverse on small far-field n; B438 humped). There is no
   usable band-position gradient toward the contact in either direction -
   which also retires the band-centre version of the mixing gradient
   (the hydration-DEPTH gradient of the contact ledger is a separate,
   block-verified result and is untouched by this).
4. NOT CARRIED FORWARD - the two clay provinces. The absolute
   classification does not transfer (a chain-level wavelength offset of
   ~12 nm shortward - TRR3 nominal vs MTRDR-corrected calibration - puts
   every component below the 2.311 threshold), and the geographic split
   was not re-established chain-internally. The province observation is
   chain-sensitive at minimum; treat as unverified and do not use.
5. FINAL AND CHAIN-ROBUST - the negative ordered-zonation verdict. BOTH
   chains, all four cubes: 2.3 um band-position support is confined to
   ~2.29-2.325 um, with NO chlorite-range population, no long-centre zone
   along the contact, and no multi-phase spatial ladder. The strongest
   orbital criterion for contact metamorphism is observationally negative
   at >= 1 channel / >= 1 pixel scale in two independent processing
   chains. This is the one publishable statement from the band-centre
   campaign.
6. STILL STANDING (dual-chain by construction) - the carbonate patches as
   short-centre anomalies: patches come from the Plebani chain, the
   short-centre association was measured in the MTR3 chain.

## Methods lesson

The two chains disagree on absolute wavelength by ~2 channels and on band
depth (the bland ratio removes the unit-shared band component: strong-tier
counts drop ~7x). Absolute single-chain band-centre claims at this site
are fragile; only within-chain relative structure that REPLICATES across
chains counts. This mirrors and extends the DECISIONS 2026-08-28 caution.

## Outputs

fig_map_{cube}, fig_highdecile_map_{cube}, fig_tail_compare_{cube},
fig_anomaly_{cube}, fig_profile_compare, fig_histograms (.png/.pdf);
arrays {cube}_bcP_center_{strong,weak}.npy, _bcP_depth.npy, _bcP_colmap.npy,
{cube}_bcP_sensor.npz, {cube}_stacks.npz; CSVs summary/profile/
mode_fraction/provinces/anomaly_metrics; run_log_plebani.json,
run_log_figures.json.
