# Decisions log

A running record of methodological choices made over the course of the
project. Each entry: what was decided, when, why, and what was
considered/rejected. Update when a new significant choice is made.

"AV" in the entries below is A. Valantinas, the investigator who directed the
analysis and reviewed each result (see the AI-assistance note in the README).

## Mask / pixel-population choices

### Brown v5 mask: strict (v1) vs full range (v2)
- **v1 (2026-04-21)**: `joint_valid_15 = brown["mask"] & finite(Brown 4) &
  finite(VB 11)`. Brown's mask is essentially `centroid >= 1.15 um` plus
  a residual-quality cut. Per-cube valid fractions: 8.3% (A4FC) -
  77.3% (3E12).
  - Why: matched Brown 2020 protocol; ensures every clustered pixel has
    a meaningful 1 um band fit.
  - Trade-off: A4FC is dominated by sub-1.15 pixels; the mafic cap and
    phyllosilicate populations are excluded.
- **v2 (2026-04-22)**: `joint_valid_15 = finite(Brown 4) & finite(VB 11)`.
  Drops the centroid threshold. All four cubes now ~287k pixels (93.4%
  of frame).
  - Why: AV's ask - the low-centroid pixels carry the mafic-cap and
    phyllosilicate signal we want to see. Sub-1.15 centroids include
    pyroxene-dominated, low-band-depth basaltic-cap, and phyllosilicate-
    dominated pixels (where the asymmetric Gaussian fit is noise but VB
    parameters are real).
  - Trade-off: Brown 4 params are noise on the new pixels; the VB 11
    carries the real signal in those regions. PCA absorbs this fine.
- **Toggle**: `STRICT_BROWN_MASK` constant in `code/stage_b_build_stack.py`.
  Set True to reproduce v1.

### Display orientation
- **2026-04-21**: All spatial figures flipped via `apply_display_transform`
  (`FLIP_VERTICAL=True, FLIP_HORIZONTAL=True` in `code/display.py`).
  - Why: AV's preference - "easier to see" map-view vs detector
    orientation.

### VB parameter masks in display
- **2026-04-21**: Brown v5 mask applies only to the 4 Brown params for
  display. VB-only params (BD1900R2, BD2500_2, D2300, OLINDEX3) use
  finite-only masking.
  - Why: AV flagged that BD1900R2 and BD2500_2 are independent of the
    1 um fit; masking them with brown_mask was incorrect.

## Feature-stack composition

### Stage A: 8-param baseline
Brown 4 (centroid, asymmetry, amplitude, hwhm) + VB 4 (OLINDEX3, D2300,
BD1900R2, BD2500_2). Justified by the existing PCA_CRISM_analysis_v2
script and Brown 2020 + Viviano-Beck 2014 sufficiency for first-pass
mineralogy.

### Stage B: 15-param expansion (AV's selection 2026-04-21)
Brown 4 + nine VB required (BD530_2, BD920_2, BDI1000VIS, OLINDEX3,
LCPINDEX2, BD1400, BD1900R2, D2300, BD2500_2) + two exploratory
(BD2290, BD2355).
- Why: Stage A 8-param showed ~3 independent axes (|rho|>=0.89 within
  Brown 4); needed more independent dimensions to see fine structure.
  AV directly chose the param list and folded what would have been
  Stage D's "controlled comparison" into Stage B itself.

### Stage D candidates (deferred, per AV)
BD1900R, BD2100_2, MIN2295_2480 plus the originals - reserved for
sensitivity testing if v2 results need stress-testing.

## Standardization

### Per-cube StandardScaler
- All Stage B PCA and clustering use `StandardScaler.fit_transform` per
  cube before PCA.
  - Why: removes per-cube atmospheric/illumination offsets so PCA
    discovers structure within each cube on comparable scales.
  - Trade-off: removes any genuine cross-cube absolute differences;
    cross-cube comparison is structural, not absolute.

## Dimensionality reduction

### PCA: 5 PCs to clustering
- All clustering operates on the **first 5 PCs** of the standardized
  15-param stack (~84-91% cumulative variance).
  - Why: balances signal preservation against the curse of dimensionality
    for GMM full-covariance and HDBSCAN.

### UMAP sweep (Stage B v1, skipped in v2)
- 12 configs per cube: n_neighbors in {15, 30, 50, 100} x
  min_dist in {0.0, 0.1, 0.3} on a 30k-pixel subsample, then full-cube
  refit for the selected config(s).
  - Selection: heuristic (n100, m0.0) AND validity-index pick (DBCV
    via HDBSCAN).
  - **Skipped in v2** per AV's instruction (focus on GMM only).

## Clustering choices

### Methods (v1)
k-means + GMM + HDBSCAN. k-means and GMM on PCA 5-D, HDBSCAN on UMAP
2-D.

### k=6 fixed across cubes (AV's choice 2026-04-21)
For cross-cube comparability we run an additional k=6 run alongside
the adaptive picks (k-means by silhouette, GMM by BIC).
- Why: silhouette pushed k-means to k=2-3 (trivial mafic-cap-vs-rest
  splits); BIC pushed GMM to k=9-10 (saturated at sweep top, no
  natural optimum). k=6 gives balanced clusters comparable across
  cubes; adaptive picks are reported but not used as the cross-cube
  reference.

### v2 simplification (2026-04-22)
GMM k=6 fixed only. k-means, UMAP, HDBSCAN skipped in the full-range
re-run because (a) v1 showed k-means silhouette + HDBSCAN are
unstable on this data and (b) AV wants to focus on the soft-
clustering / transition-pixel readout that GMM gives.

### HDBSCAN min_cluster_size (v1)
Per-cube scaled: 3E12=150, 871C=100, A4FC=15, B438=75. A4FC's small
valid-pixel count required a much smaller min_cluster_size, but the
result was 123 clusters with 51% noise - HDBSCAN was effectively
unusable on the v1 A4FC.

## Mafic-cap denominator (display-only ratio target)

### Auto-selector
Score each cluster by z-sum across {BD1900R2, D2300, BD2500_2,
OLINDEX3}; pick the lowest. Tie-break on LCPINDEX2 (some pyroxene
signature preferred).
- Why: AV's rule - the "mafic cap" unit is low in hydration AND low
  in olivine index AND low in carbonate.
- Override mechanism: `outputs/reports/denominator_overrides.json`
  keyed `<cube>/<method>: <raw_cluster_id>`.
- **Critical scope**: ratio is for display only - never used for
  clustering, never used for parameter computation.

## Cross-cube alignment (Stage C)

### Reference cube: FRT00003E12
- Why: largest valid fraction (v1 77%, v2 93%); Brown 2020 validated
  centroid distribution; widest natural distribution to act as
  alignment anchor.

### Hungarian on standardized 15-param centroids
- Cluster centroids computed in the per-cube standardized 15-param
  space; pairwise euclidean distance; `scipy.optimize.linear_sum_assignment`.
- Why: euclidean on standardized centroids is the same space k-means
  optimizes; cosine would de-emphasize magnitude differences which we
  want to keep for compositional matching.

### Cluster ordering: c0 = most mafic-cap-like
- After alignment, the 6 aligned labels are ordered by mafic-cap
  z-sum on the reference cube. c0 = lowest hydration+olivine+carbonate;
  c5 = highest.
- Why: gives a stable, interpretable color order for cross-cube figures.

## Statistical tests

### Mixing-fraction homogeneity: chi-square
- 4-cube x 6-cluster contingency table; `scipy.stats.chi2_contingency`
  with Bonferroni correction over the number of methods tested.
- Critical chi^2 at p=0.001, dof=15: 37.7. Both v1 and v2 returned
  values orders of magnitude above (v1 ~16k, v2 ~113k for GMM).

### Connected-component aspect ratio: principal-axis ratio
- Per cluster per cube, label connected components, compute coordinate
  PCA, take sqrt(eigenvalue ratio). Skip components < 20 pixels.
- Reading: aspect > 2 = elongated (fracture/dike-like); aspect ~1 =
  blob.
- Caveat: baseline is ~2 across all clusters at 18 m/pixel (cube
  geometry), so this is a relative measure within a cube, not
  absolute.

## Random seed
**Always 42**, set via `RANDOM_SEED` in `code/config.py`. Used in
KMeans, GMM, UMAP, HDBSCAN-driven validity sweeps, and any subsample
operation. Reruns on the same input produce bit-identical outputs on
the same platform / numpy / sklearn version.

## Reproducibility
- `requirements.txt` (pinned via `pip freeze`)
- `code/python-version.txt` (3.14.0)
- `code/REPRODUCE.md` (run order + per-figure regeneration)
- Every analysis script logs its `source_code_md5` for every file it
  depends on, into the JSON log alongside the output.
- Two snapshot folders preserve the v1 and v2 final states:
  `outputs/_v1_brown_strict/`, `outputs/_v2_full_range_gmm/`.

## v4 (2026-04-26) - VB-only PCA + Brown RGB separation

### Drop Brown 4 from the PCA basis
- **Why**: Brown band-fit parameters (centroid, asymmetry, amplitude,
  HWHM) are individually physically interpretable - centroid maps to
  Fo number, HWHM to grain size - and don't need PCA to extract their
  information. Including them in the PCA inflates the variance attributable
  to "olivine band shape" and crowds out minor-phase axes.
- **Outcome**: v4 splits the workflow into a VB-only PCA (mineral
  composition question) and a Brown RGB display (olivine grain-size /
  composition question), both independently presented at matched stretches.

### Drop BD2290 and BD2355 from the parameter basis
- **Why**: v2 loadings showed all three (D2300, BD2290, BD2355) loaded
  near-identically on PC2 - they all measure the same Fe/Mg-OH absorption.
  Three parameters for one feature inflates PC2's variance and crowds
  out minor-phase axes.
- **Outcome**: D2300 alone carries the 2.3 um signal in v4. The
  redundant pair stays available in `compute_extended_vb_stack` for
  any future stage that wants the full v2 stack back.

### Add SH600_2
- **Why**: Independent ferric-edge sensitivity to complement BD530_2
  (nanophase Fe3+) and BD920_2 (crystalline Fe3+/Fe2+). Not present
  in v2 or v3 stacks.
- **Implementation**: VB2014 Table 1 verbatim. Formula is INVERTED
  from band depth: `SH600_2 = 1 - (a*R533 + b*R716) / R600`. Positive
  for reflectance peaks (ferric edge); kernel widths 5/5/5.
- **Sanity check**: per-cube SH600_2 median should be positive on
  ferric-bearing terrain; flagged in run log if median falls below
  -0.05.

### Add HCPINDEX2
- **Why**: Paired with LCPINDEX2 for full pyroxene story.
- **Implementation**: VB2014 Table 1 verbatim. Equal-weighted mean of
  band depths at {2.120, 2.140, 2.230, 2.250, 2.430, 2.460} um with
  continuum anchored at R1690 / R2530. Generic
  `_band_depth_multi_pt(centres, anchors, ...)` helper added to
  `vb_parameters.py` for this and any future multi-channel parameter.

### v4 alignment space
- Hungarian alignment runs on **per-cube standardized 11-param VB-only
  centroids** (not the 15-param Brown+VB space used in v3 Stage C).
  Brown is excluded from alignment because it is excluded from the PCA.
- Brown values appear in cross-cube comparison plot 5.2 only as
  derived per-cluster medians on already-aligned clusters.
- Cluster identities therefore differ from v3 Stage C - this is
  expected and correct, not a regression.

### Brown median computation in cross-cube panel 5.2
- Per-cluster Brown medians use the intersection (cluster ∩ Brown
  eq.2 ∩ residual <= p95) — same gate as the Brown RGB / scatter plots.
- Hollow markers if a cluster has fewer than 100 Brown-valid pixels
  in a given cube (warning that the median is noisy).
- Missing point if zero Brown-valid pixels.
- Per-cluster Brown-valid counts logged in the per-cube run log.

## v4.1 (2026-04-26) - Data-driven matched stretches for Brown RGB

### Switch from eq.2 validity bounds to 4-cube p2/p98 stretches
v4 used Brown 2020 eq. 2 validity bounds as RGB / grayscale stretches:
centroid 1.15-1.50 um, amplitude 0.10-1.00, HWHM 0.05-0.30 um, asymmetry
0.80-9.00. Inspection per-cube percentiles showed the actual data fills
only a fraction of these ranges (amplitude p98 maxes at 0.37 across cubes;
HWHM p2 minimum is 0.18; asymmetry p98 maxes at 4.79). This left 40-60%
of each colormap unused on real data, so subtle inter-cube differences
were hard to read - especially on A4FC where the data sit in a narrow
slice.

v4.1 retightens to four-cube-pooled p2-p98 spans:
- centroid: **1.15 -> 1.40 um** (1.15 = eq. 2 floor; covers all cubes' p98)
- amplitude: **0.10 -> 0.37** (lower bound at the eq. 2 floor, not p2,
  so the stretch range still has a clean "any pixel surviving the
  filter is in the colormap" interpretation; upper covers 3E12's p98)
- HWHM: **0.18 -> 0.30 um** (covers all cubes' span)
- asymmetry: **1.25 -> 5.00** (covers most pixels; values >5 saturate
  to pure blue per Brown 2020 band-saturation regime - documented in
  caption)

**Why**: per-cube percentile stretches were considered but rejected
because they would break cross-cube color correspondence ("identical
color = identical parameter value"). The chosen approach is the
four-cube-pooled equivalent and preserves the matched-stretch property.

**What changed**:
- `BROWN_STRETCH` dict in `code/stage_v4_vb_pca_and_brown_rgb.py`
  updated with new bounds.
- All 4 per-cube `Brown_RGB_centroid_amp_hwhm_v4.png`,
  `Brown_RGB_centroid_amp_asym_v4.png`, and 4 grayscale Brown maps
  regenerated (16 figures + 8 RGB = 24 per-cube; plus 2 cross-cube
  = 26 total).
- Centroid-asymmetry hexbin scatter X-axis kept at 1.05-1.50 so the
  eq. 2 cutoff at 1.15 stays visually obvious as a vertical edge in
  the density.
- Figure titles now show new stretches explicitly
  (e.g. `centroid: 1.15-1.40 | amplitude: 0.10-0.37 | HWHM: 0.18-0.30`).
- Each figure carries a footnote: "Stretches are data-driven, matched
  across cubes (4-cube pooled p2/p98; amplitude lower bound at the
  Brown 2020 eq. 2 floor of 0.10). Identical color = identical
  parameter value across cubes. Brown 2020 eq. 2 + residual<=p95
  applied as a separate transparency mask."
- RGB B + grayscale asymmetry figures additionally carry: "Asymmetry
  > 5.0 (band-saturation regime per Brown 2020) clips to pure blue."

**How to reproduce v4.1 only** (no full pipeline rerun):
```
python code/stage_v4_vb_pca_and_brown_rgb.py --brown-only
```
This skips TER3 cube load, VB compute, PCA, and GMM, and just rebuilds
the Brown figures using the cached `brown_valid_mask_v4.npy` arrays.
Wall-clock ~30 s.

**Pre-restretch v4 figures preserved at**
`outputs/_v4_vb_pca_brown_rgb/_pre_v4p1_brown_restretch/` (30 PNGs:
26 in scope + 4 unchanged centroid-asymmetry scatters for completeness)
so the v4 baseline is recoverable for direct before/after comparison
without rerunning the pipeline.

## v5 (2026-04-27) - VB parameter pruning + full-scene Brown RGB

### Drop four high-loading VB params: SH600_2, HCPINDEX2, BDI1000VIS, LCPINDEX2

Two stack variants tested in parallel as a sensitivity comparison:
- **v5-A** (7 params): drops the four. Keeps BD530_2, BD920_2,
  OLINDEX3, BD1400, BD1900R2, D2300, BD2500_2.
- **v5-B** (5 params): v5-A minus BD530_2 + BD920_2 = pure alteration
  suite (OLINDEX3 + 1.4 + 1.9 + 2.3 + 2.5 um band depths).

**Three rationales (AV: "all three are reasonable" 2026-04-27):**

1. **Suspected noise / atmospheric residual.** No flagging in v4 run
   logs but worth checking - if these channels are noisy on this
   dataset, they would inflate PCs without contributing real
   structure.
2. **Reducing VNIR-region cross-talk to clarify the alteration
   story.** SH600_2 + BD530_2 + BD920_2 + BDI1000VIS + LCPINDEX2 +
   HCPINDEX2 + OLINDEX3 (7 of 11 v4 params) all touch the < 2 um
   region. Dropping six of them (v5-B; OLINDEX3 retained) leaves the
   1.4 / 1.9 / 2.3 / 2.5 um OH+carbonate suite to dominate the PCA.
   v5 = "what does cluster structure look like if alteration
   mineralogy carries the weight?"
3. **Reducing collinearity in the ferric edge.** SH600_2 + BD530_2 +
   BD920_2 measure overlapping spectral regions. v5-A keeps
   BD530_2 + BD920_2; v5-B keeps neither.

**Caveat for the log**: the four parameters dropped in v5-A carry
21-30% of the squared loadings in v4 PC1-5 across cubes -- so this
is a substantive basis change, not "trim the redundant noise". The
v4 entry justified ADDING SH600_2 + HCPINDEX2; v5 reverses that
deliberately to test the alteration-only basis. Both v4 and v5
remain available as snapshot folders.

**Results** (see `outputs/_v5_param_pruning/reports/SUMMARY_v5.md`):
- v5-A PC1-5 cum reaches 90% on every cube (was 82-87% in v4).
- v5-B PC1-5 = 100% by construction (5-D space).
- v5-B Hungarian alignment max distance to 3E12 reference: 1.60
  (vs v4's 2.75). Tightest cross-cube alignment of any version.
- GMM BIC drops on v5-B vs v5-A on 3 of 4 cubes (fair comparison;
  same GMM model dimensionality of k=6 on 5-D whitened input).

### Full-scene Brown RGB (no eq.2 mask)

v4.1 Brown RGB applied the eq.2 + residual filter as a transparency
mask, leaving 24k-225k pixels per cube. v5 regenerates the same
RGB (centroid / amplitude / asymmetry) without the mask, on all
finite-Brown pixels (~287k per cube), with two stretch sets
produced:

- **v4.1 stretches** (centroid 1.15-1.40, amplitude 0.10-0.37,
  asymmetry 1.25-5.00): direct comparability to v4.1 eq.2-masked
  panels. Full-scene pixels outside this range clip.
- **v5 stretches** (centroid 1.014-1.379, amplitude 0.116-0.335,
  asymmetry 0.525-4.439): 4-cube pooled p2/p98 on the full-scene
  Brown-finite pixel set. Reflects the wider distribution that the
  eq.2 mask hid - especially asymmetry 0.525 vs 1.25 picks up the
  low-asymmetry mafic-cap pixels.

**Why two stretches**: the v4.1 stretch is the matched-comparison
control (does v4.1 still look right when shown the whole scene); the
v5 stretch is the fully-data-driven view (uses the colormap
optimally on the larger pixel set).

## v6 (2026-04-28) - Interactive ROI explorer

### Build a single-cube interactive tool, not a batch product
v6 was originally planned as batch PC back-projections + GMM-class
ratio spectra + lobe extraction. AV redirected to an interactive
ROI explorer because the EPSL revision needs candidate ROIs picked
by eye (not by automated thresholds), then their ratio spectra
inspected, then their position on the Brown centroid-asymmetry curve
verified - all in one workflow.

### matplotlib + LassoSelector chosen over napari, plotly, streamlit
- napari: more polished ROI tools but adds 100+ MB dep tree (Qt,
  vispy, magicgui).
- plotly + Dash: web-app deployment overhead, ROI drawing is decent
  but slower than native lasso.
- streamlit: cleanest deploy story but ROI / lasso UX is limited.
- matplotlib LassoSelector: no new deps, native to the existing
  pipeline, lasso-then-callback is the simplest workflow that
  matches the spec.

### Inputs reused, no re-computation
- v5-B aligned k=6 GMM labels
  (`outputs/_v5_param_pruning/<cube>/arrays/<cube>_v5b_GMM_k6_labels_aligned.npy`)
- Brown maps via `crism_io.find_cube_and_brown_maps`
- TER3 cube via `crism_io.load_cube_by_id` (slow ~30s; loaded once
  per session)

### Brown mask: full scene, NOT v4 eq.2
The Brown valid mask in v6 is built fresh from
`finite(centroid) AND finite(amplitude) AND finite(hwhm) AND finite(asymmetry)`
on the loaded Brown maps - the entire ~287k-pixel range per cube,
NOT the v4 eq.2-restricted set (~25k-225k). Reason: ROI pixels with
sub-1.15-um centroids would silently disappear from the Panel 5
scatter overlay if we used the v4 eq.2 mask, defeating the point of
the interactive workflow. The Brown 2020 eq. 2 floor at 1.15 um is
shown as a vertical dotted line on the scatter for reference, with
the sub-1.15 region shaded grey + labeled "below eq.2 (<1.15)".

### Brown RGB stretches: v5 full-scene values
v6 uses the v5 4-cube-pooled p2/p98 stretches (centroid 1.014-1.379,
amplitude 0.116-0.335, asymmetry 0.525-4.439) on the full-scene
pixel set, matching the v5 full-scene Brown RGB convention rather
than the v4.1 eq.2-restricted stretches. Same reason: the v4.1
stretches assume sub-1.15 pixels don't exist, which is wrong for
the full-scene view.

### Scatter axis bounds extended
Centroid X-axis 0.90-1.50 um (was 1.05-1.50 in v4 scatter); asymmetry
Y-axis 0.30-10.0 (was 0.80-9.00). Extension covers the genuine data
range across all four cubes (centroid 0.91-1.47, asymmetry 0.20-13.63
from the Brown .npy arrays).

### Display orientation handled at load time
All spatial arrays (cube, GMM labels, Brown maps) are display-flipped
once at load via `apply_display_transform` (FLIP_VERTICAL =
FLIP_HORIZONTAL = True). Lasso coords directly index the flipped
arrays. Saved ROI masks are un-flipped back to detector orientation
for archival.

### Paired-rectangle ROI workflow (replaces lasso + slot toggle)
2026-04-28 second revision: dropped lasso in favor of
RectangleSelector and replaced the n/d slot-toggle with auto-
alternation. Each rectangle is appended to a single ROI list; the
tool decides target vs mafic from the count:

  rect 1 -> target1   rect 2 -> mafic1   (pair 1)
  rect 3 -> target2   rect 4 -> mafic2   (pair 2)
  ...

Rationale:
- Rectangles are faster to draw than lassos and sufficient for the
  EPSL ROI workflow (AV's call).
- Auto-alternation removes the slot-toggle keystrokes and matches
  the natural target+reference pattern of CRISM ratio analysis.
- Each pair shares one color from PAIR_PALETTE so the visual
  association is unambiguous.
- Targets get solid black outline + 0.50 fill alpha; mafics get
  dashed black outline + 0.20 fill alpha so target/mafic are
  visually distinguishable within a pair without needing two
  palettes.
- Each ROI carries the rectangle bbox in (x_lo, y_lo, x_hi, y_hi)
  form (display orientation) for archival in the meta JSON.

Ratio computation:
- Per pair: `mean(target_i) / mean(mafic_i)`. One curve per
  COMPLETE pair (both halves drawn) on the ratio plot, colored to
  match the pair.
- Brown scatter overlays target pixels only (mafic = ratio
  reference, not science); each target colored by its pair color.

Save requires at least one complete pair. Half-completed pairs
(target with no mafic, or vice versa) are kept in the in-memory
list but skipped in CSV output.

### Output schema per ROI session
On 's' save, files per timestamp under
`outputs/_v6_interactive_explorer/<cube>/`:
- `<cube>_<label>_mask_<TS>.npy` per ROI (detector orientation,
  bool). One file per target + one per mafic.
- `<cube>_ratio_spectra_<TS>.csv` (single CSV with columns:
  wavelength_um, target1_mean, mafic1_mean, target1_over_mafic1,
  target2_mean, mafic2_mean, target2_over_mafic2, ...).
- `<cube>_viewer_<TS>.png` (Viewer window screenshot)
- `<cube>_plots_<TS>.png` (Plots window screenshot)
- `<cube>_meta_<TS>.json` (cube id + per-ROI labels/kinds/pair_idx/
  colors/pixel counts/bbox + complete-pair list + label source +
  Brown stretches + false-color band assignments + scatter axis
  bounds + ratio convention note)

### Undo
Pressing 'u' removes the most recent ROI from the single ROI list.
Better than Clear for iterative refinement; especially useful with
the paired-rectangle workflow when you mis-click a rectangle.

### Dynamic stretch on FC + Brown panels (zoom-aware)
The false color (Panel 2) and Brown RGB (Panel 3) panels recompute
per-channel p2/p98 stretches from the currently-visible region every
time the user zooms or pans. Implementation:
- Raw channels stored in `self._fc_raw` and `self._brown_raw`.
- Two callbacks (`xlim_changed` + `ylim_changed`) on `ax_gmm` (which
  is the sharex/sharey master for all three viewer panels) feed a
  single coalesced restretch via `_schedule_restretch` -> 10ms
  deferred `_restretch_now`. Coalescing avoids duplicate work when a
  zoom event fires both signals.
- `_restretch_now` reads the visible slice via `_visible_slice`,
  recomputes per-channel p2/p98 from those pixels only, and
  re-normalizes the FULL image using the visible-region bounds.
  Pixels outside the zoom box clip - this is the desired behavior;
  zoom-out triggers a fresh restretch that recovers full-scene
  dynamic range.
- Panel titles update on every restretch to show the active stretch
  source (`visible region` vs `full scene`) and the current Brown
  per-channel bounds (centroid / amplitude / asymmetry).
- The GMM panel intentionally has no dynamic stretch - it's a
  categorical map; only its lasso rectangle drives the zoom event.

Manual trigger: pressing 'r' forces a recompute. Useful if the
deferred `after()` call fails to fire on a non-Tk backend.

### Keyboard-only UI (no matplotlib Buttons)
Initial v6 had Button widgets for [N: Numerator], [D: Denominator],
[Undo], [Clear], [Save]. Removed 2026-04-28 because matplotlib
Button widgets were causing the tool to break in AV's runtime
(known issue: Buttons across multi-window setups have stale event
state, fail to redraw, and can hang the figure event loop). Replaced
with keyboard shortcuts (n, d, u, c, s) which use the same handlers
and have none of those bugs.

The viewer window suptitle always shows the active slot and the
next ROI label that will be assigned, so the user always knows what
keystroke to expect for the next draw. Status strip across the
bottom of the viewer shows ROI counts per slot.

### Two-window layout (separated viewer + plots)
v6 opens TWO matplotlib figures:
- **Window 1 (Viewer)**: 1x3 grid of GMM / false color / Brown RGB
  + button strip + status text. Lasso lives here.
- **Window 2 (Plots)**: 2x1 grid of ratio spectrum (top) + Brown
  scatter (bottom).

Updates from a lasso event refresh both windows' canvases. Closing
either window calls `plt.close('all')` so the second window goes
with it (clean exit).

### Panel 2 = CAR carbonate composite (parameter-based, not band-based)
2026-04-28: replaced the original band-based false color
(R=2.529 / G=1.506 / B=1.080 um) with a CAR-style carbonate composite
built from VB parameters:
- R = D2300 (Fe/Mg-OH; carbonate left shoulder + smectite overlap)
- G = BD2500_2 (2.5 um Mg-carbonate doublet)
- B = BD1900R2 (1.9 um water of hydration; visually anti-correlates
  with carbonate, distinguishes carbonate from clay+water)

Channels read from the cached `<cube>_VB_stack_v4.npy` (no recompute,
no TER3 reload for the composite). Per-channel p2/p98 stretch is
computed dynamically from the visible region per the existing zoom-
restretch hook.

CAR variants in the literature: Pelkey 2007 used D2300 / BD2500 /
BDCARB; VB2014 CAR_2 uses MIN2295_2480 / MIN2345_2537 / BD2500_2.
We use a third combination because BDCARB / MIN2295_2480 / MIN2345_2537
are not in our v4 VB basis. Documented in panel title:
`CAR composite R=D2300 G=BD2500_2 B=BD1900R2`. If the science needs
the canonical Pelkey or VB2014 CAR, BDCARB / MIN2295_2480 / MIN2345_2537
need to be added to the VB compute module first.

### Linked zoom/pan across Viewer panels
The three Viewer-window panels (GMM, false color, Brown RGB) share
both x and y axes via matplotlib's `sharex`/`sharey`. Zooming or
panning on any one panel mirrors the view on the other two
automatically. The Plots-window panels (ratio spectrum, Brown
centroid-asymmetry) intentionally do NOT share axes with anything -
their coordinates are wavelength vs ratio (Panel A) and centroid vs
asymmetry (Panel B), unrelated to detector pixel coordinates.

### Brown scatter shows numerator only
Per spec: only numerator ROI pixels are overplotted on the Brown
centroid-asymmetry hexbin. Denominator pixels are the ratio
reference (typically a "neutral" / mafic-cap region) and not the
science target whose composition you're interrogating, so plotting
them on the parameter curve would be misleading. If a denominator
view is ever needed, the saved denominator mask can be re-plotted
post-hoc from the saved .npy file.

## v7 (2026-04-29) - Map-projected MTR3 Brown + GMM products for ArcGIS

### MTR3 cube loader uses np.fromfile + transposed view (no spectral.SpyFile.load)
The `spectral` library's default loader allocates the cube ~3-5x in
RAM (array.array intermediate -> ndarray -> np.array copy). Hit
MemoryError on every MTR3 cube (~1.2 GB on disk). Replaced with
`np.fromfile` reading the IMG file directly into a single 1.2 GB
float32 buffer, exposed as a transposed view of the BSQ data
(no second allocation). Per-pixel access is then strided
(cache-unfriendly) but correct; cost is ~5-10x slower per-pixel
fit, no precision/correctness penalty. v7-A wall clock: ~50 min
per cube vs ~10 min that contiguous access would give.

If RAM permits in future runs, swap to a contiguous read by
adding `np.ascontiguousarray(cube)` after the transpose - that
allocates a second 1.2 GB block but eliminates the strided-access
penalty. Documented in `code/1um_centroid_analysis_CRISM_MTR3_v1.py`
near the loader.

### v7-C drops the display-flip convention
TER3 cubes are in detector orientation, so v1-v6 apply
`apply_display_transform` (FLIP_VERTICAL = FLIP_HORIZONTAL = True)
on every spatial display. MTR3 cubes are already map-projected
(MRO Mars Equirectangular, IAU 2000, north-up). The v7-C pipeline
does NOT apply the flip. The v6 ROI explorer (TER3-only) keeps the
flip as before; nothing in v6 was modified.

### v7-C uses v5-B basis only
Per AV 2026-04-29: only the 5-parameter "pure alteration" basis
(OLINDEX3, BD1400, BD1900R2, D2300, BD2500_2) is ported to MTR3.
v4 (11-param) and v5-A (7-param) are not re-run on MTR3. The
methods documents were updated to drop the v4 / v5-A discussion -
only v5-B is described.

### v7-C GeoTIFF outputs: 5 PC scores + GMM raw + GMM aligned per cube
Per AV 2026-04-29: GeoTIFFs are produced for the 5 PC scores
(raw float32) plus 5 PC scores (whitened float32, useful for
GMM input replication) plus 1 GMM raw uint8 categorical and 1
GMM Hungarian-aligned uint8 categorical. Per-cube total: 12
GeoTIFFs. The 5 VB parameter rasters were NOT written as
individual GeoTIFFs; the .npy stack is preserved in `arrays/`
for downstream use. If individual VB GeoTIFFs are needed later,
a small loop over `write_float_geotiff` in
`stage_v7c_mtr3_pipeline.py` will produce them.

### Brown RGB GeoTIFF composites
Per AV 2026-04-28: produced as 4-band uint8 RGBA GeoTIFFs with
v4.1 matched stretches (centroid 1.15-1.40, amplitude 0.10-0.37,
hwhm 0.18-0.30, asymmetry 1.25-5.00). Two composites per cube
(centroid/amplitude/hwhm and centroid/amplitude/asymmetry). The
4th band is the alpha channel = Brown 2020 eq.2 quality mask;
pixels failing eq.2 are transparent (alpha = 0). Embedded color
profile = "RGB", `compress=deflate`. ArcGIS Pro picks them up
as RGB composites by default.

### Methods docs scope: PCA dropped, v5-B only
Per AV 2026-04-29: PCA in v5-B is rank-5 -> 5 PCs, full rank
preserved, no dimensionality reduction. PCA is a numerical
preconditioner for GMM (decorrelation + whitening) and is not
strictly necessary - GMM with full covariance can absorb the
correlation structure natively. The methods docs were rewritten
to describe v5-B as "z-scored, decorrelated and whitened, then
GMM-clustered", omitting the PCA-as-DR framing. The
supplementary doc retains a brief mention that PCA is a
preconditioner, not a feature-engineering step.

## v8 (2026-04-29) - MTR3 7-param GMM (Brown back into the basis)

### Re-include Brown centroid + asymmetry in the GMM input space
v4-v7 deliberately excluded all Brown band-fit parameters from the
PCA + GMM pipeline (rationale: physically interpretable on their
own; would otherwise inflate the variance attributable to olivine
band shape). Per AV 2026-04-29: re-add the two Brown parameters
that carry independent compositional and textural information --
**centroid** (-> Fo number proxy) and **asymmetry** (-> grain size
/ band-saturation regime) -- to test whether they sharpen the
mineralogical class boundaries.

v8 stack: centroid + asymmetry + OLINDEX3 + BD1400 + BD1900R2 +
D2300 + BD2500_2 = 7 parameters. Same MTR3 cube source as v7-C,
same per-cube standardize -> PCA -> GMM(k=6) -> Hungarian align
pipeline. Outputs at `outputs/_v8_mtr3_with_brown/`.

amplitude and HWHM are NOT re-added -- amplitude duplicates much of
OLINDEX3, HWHM duplicates much of asymmetry. Two-Brown-param
addition is the smallest meaningful test of the "Brown sharpens
GMM classes" hypothesis.

### PROJECT-WIDE POLICY (2026-04-29): no Brown 2020 Eq.2 gating in outputs
From now on, no downstream product should clip on the Brown 2020
Eq.2 quality threshold. The Brown Eq.2 mask is still saved
(`<cube>_mask_map_v6.npy`) for diagnostic / reproducibility
purposes, but it is NOT applied as a transparency or selection
filter on:
- Brown .npy parameter arrays (already full-range)
- Brown RGB GeoTIFF composites (alpha = `finite(Brown 4)` only)
- v8 GMM input mask (pixel valid if all 7 params finite)
- Any future output

Specifically: **v7-B Brown RGB GeoTIFFs were re-rendered 2026-04-29**
with full-range alpha and 4-cube-pooled p2/p98 stretches:
centroid 1.02-1.39, amplitude 0.115-0.334, hwhm 0.188-0.320,
asymmetry 0.58-4.55. The earlier v4.1 stretches (centroid 1.15-1.40
etc.) clipped sub-1.15 mafic-cap pixels to dark and are no longer
appropriate for the full-range workflow.

## v9 (2026-04-29) - 3E12-seeded GMM (negative result on cross-cube transfer)

### Hard-predict on a frozen 3E12 GMM in place of v8 per-cube fit + Hungarian
Per AV 2026-04-29: replace v7-C / v8 per-cube standardize-PCA-GMM-then-
Hungarian-align with a single GMM fit on FRT00003E12 only, applied
unchanged to the other three cubes. Saved scaler / PCA(whiten=True) /
GMM(k=6, n_init=10) at `outputs/_v9_3e12_seeded/reference/`. Two parallel
basis variants:
- v9-A (5 params) - alteration only (matches v7-C basis)
- v9-B (7 params) - v9-A + Brown centroid + asymmetry (matches v8 basis)

Cluster reordering: clusters re-permuted on 3E12 by ascending mafic-cap
z-sum (BD1900R2 + D2300 + BD2500_2 + OLINDEX3) so cluster 0 = least
altered, cluster 5 = most altered. Same ordering convention as v7-C / v8.

### Novelty thresholds: chi^2 + max-posterior union
Per pixel in non-ref cubes: novel if Mahalanobis^2 > chi^2_{0.95, d} OR
max_posterior < 0.6. d = 5 -> 11.07; d = 7 -> 14.07. The two arrays
saved separately so thresholds are tunable post-hoc without rerunning.
Mahalanobis^2 computed in whitened PC space using GMM
`precisions_cholesky_`: m^2 = ||(x - mu_k)^T L_k||^2.

### Negative finding: cluster basis does not transfer
3E12 self-novelty: 23.7% (v9-A), 13.5% (v9-B). Plan expected ~5%.
chi^2 fraction is correctly 4-5%; the inflation is from p_max < 0.6 on
boundary pixels (20.7% v9-A, 9.3% v9-B). The 0.6 threshold is too
aggressive in 5-D, marginal in 7-D.

Non-ref cube novelty: 86-99.5%. Median Mahalanobis^2 on non-ref cubes
is 16-35 (vs 4-6 on 3E12). Warm-start drift sanity refit moves cluster
means by 2.2-9.7 sigma_within_3e12 (plan threshold: 0.5 sigma) on every
class on every non-ref cube. Per-cube standardized parameter means are
shifted -1 to -3.6 sigma on OLINDEX3, BD1400, centroid, asymmetry; +0.9
to +2.2 sigma on BD2500_2 and BD1900R2. BD1400 is the worst offender,
shifting > 2.7 sigma on all non-ref cubes - consistent with atmospheric /
calibration residual (1.4 micron OH overtone is adjacent to atmospheric
H2O bands). These offsets cannot be distinguished from real surface
differences with the information v9 has.

Conclusion: as specified, v9 is a diagnostic (3E12 doesn't span the
cube space) rather than a usable cross-cube map (would render 80-99%
gray on three of the four cubes).

### Recommendation: v9.1 hybrid (per-cube standardization, 3E12 PCA + GMM)
Three options were considered:
1. v9.1 hybrid - apply each cube's own StandardScaler before projecting
   through 3E12's saved PCA + GMM. Removes the per-cube standardization
   shift while preserving identical-cluster-basis-by-construction.
   Implementation: ~30-line edit to stage_v9_3e12_seeded.py.
2. Accept v9 as-is and use the novel sub-cluster GeoTIFFs to describe
   what each non-ref cube has that 3E12 doesn't. Honest but produces
   ~80-99% gray maps for the EPSL figure.
3. Revert to v8 per-cube fit + Hungarian alignment.

Pending AV's call on which path to take.

### Output layout
Same as v7-C / v8 conventions. Per cube: 12 GeoTIFFs (5 or 7 whitened
PCs, composite_map with 6 colors + gray-novel + transparent-invalid,
gmm_labels-only, max_posterior float, mahalanobis2 float, novel_mask
binary, novel_subclusters categorical), 8 .npy, 1 cube log. Top-level:
3 reports JSON (drift_check, novelty_stats, palette) + run log + 6
joblib (3 transformations x 2 runs). 96 GeoTIFFs total.

## v9.1 (2026-04-29) - per-cube StandardScaler, 3E12-fit PCA + GMM

### Per-cube standardization, frozen 3E12 PCA + GMM
Per AV's call after v9 negative finding: each cube gets a fresh
StandardScaler fit on its own valid pixels; PCA and GMM stay locked to
3E12. This preserves identical-clusters-by-construction (same Gaussian
parameters everywhere) while removing the per-cube parameter mean shifts
that drove v9's 80-99% novel-pixel rates. See PLAN_v9_1.md for the spec
delta from PLAN_v9.md.

Implementation: `code/stage_v9_1_per_cube_scaler.py`. Reuses helpers
from `stage_v9_3e12_seeded.py` and overrides `fit_reference` (no
reorder, dump cluster means in original VB units to log) and
`predict_cube` (per-cube scaler.fit_transform, then 3E12 pca.transform
and gmm.predict_proba). Per-cube scalers persisted to
`<cube>/arrays/<cube>_v9_1{a,b}_per_cube_scaler.joblib` for
reproducibility.

### No mafic-cap z-sum reordering
v9's reorder summed olivine and alteration parameters together so high
olivine and high carbonate were indistinguishable; the resulting "0 =
least altered, 5 = most altered" labeling didn't follow from the metric
used. v9.1 keeps cluster IDs in sklearn EM-converged order
(deterministic on seed 42) and dumps cluster means in original VB units
to `v9_1_cluster_means_3e12.json` for AV to label by inspection. The
palette JSON (`v9_1_palette.json`) has a `labels` block that starts as
all-null; AV fills it in before the figure is generated.

### Argmax pile-up reporting (added per the project specification)
Per-cube argmax cluster size distribution is computed BEFORE the
novelty filter and reported in the per-cube log + aggregate JSON. Flag
threshold > 60% (residual covariance mismatch); note threshold > 40%
(modest concentration). v9.1 results: all 8 cube-runs (4 cubes x 2
basis variants) come in below 40% on v9.1-A and below 38% on v9.1-B -
no flags or notes. The pile-up failure mode v9 exhibited (60-95% of
non-ref pixels piling onto one cluster) is gone.

### Result: substantial recovery from v9
Novelty reduction (v9 -> v9.1):
- v9-A: 871C 91% -> 44%, A4FC 98% -> 44%, B438 99% -> 33%
- v9-B: 871C 86% -> 43%, A4FC 83% -> 51%, B438 99.5% -> 38%
- 3E12 self-novelty unchanged (its data path is identical between
  v9 and v9.1)

Drift reduction (warm-start re-fit max sigma):
- v9-A B438: 9.44 -> 1.08
- v9-B B438: 9.68 -> 1.42
- All non-ref cubes: drift cut by 1.5 - 8.4 sigma vs v9

Residual drift remains 0.6 - 3.1 sigma per class (above plan's 0.5
sigma flag threshold but no longer dominated by standardization shift).
v9.1-B has higher residual drift than v9.1-A because the Brown 1 micron
band-fit parameters carry more per-cube variability than the alteration
band-depth ratios.

### Cluster-0 enrichment on 871C / A4FC under v9.1-B is meaningful
v9.1-B 3E12 cluster 0 has the highest BD2500_2 (least negative; ie
strongest Mg-carbonate signature) of any cluster, plus high olivine
band depth. It holds 8.5% of 3E12 valid pixels. Under v9.1-B it holds
36% of 871C and 38% of A4FC valid pixels - a 4x enrichment. This is
consistent with the published "olivine-carbonate ridges" anchor finding
being more prevalent off 3E12, and it shows up as the same color on
all four cubes by construction. This is what the v9 design was aiming
for, now actually working.

### v9.1-B still has higher residual drift (caveat for the EPSL figure)
Per-cube mean drift v9.1-A vs v9.1-B:
- 871C: 0.80 vs 1.52 sigma
- A4FC: 0.98 vs 2.13 sigma
- B438: 0.63 vs 1.04 sigma

If the EPSL figure needs the cleanest cross-cube comparison, v9.1-A is
the safer primary. v9.1-B is preferable for olivine class clarity but
should be paired with a caveat paragraph noting the higher per-cube
parameter variability that adding centroid + asymmetry introduces.
This is a AV call.

### Cluster mean tables persisted in original VB units
`outputs/_v9_1_per_cube_scaler/reports/v9_1_cluster_means_3e12.json`
holds the 3E12 cluster means inverse-transformed through the saved
PCA(whiten=True) and StandardScaler back to original VB units (not
standardized z-scores). Format: `cluster_means_original_units_3e12`
shape (k=6, n_features=5 or 7). This is what AV inspects to assign
mineralogical labels.

### Output layout
Same conventions as v7-C / v8 / v9. Per cube: 12 GeoTIFFs (5/7 whitened
PCs, composite_map, gmm_labels-only, max_posterior, mahalanobis2,
novel_mask, novel_subclusters), 8 .npy + per_cube_scaler.joblib (non-ref
cubes), 1 cube log. Top-level: 4 reports JSON (drift_check,
novelty_stats, cluster_means_3e12, palette) + run log + 6 reference
joblib (3 per run x 2 runs). 96 GeoTIFFs total, 70 .npy, 12 joblib,
12 JSON.

## Open / deferred
- Stage D parameter sensitivity (BD1900R, BD2100_2, MIN2295_2480 vs
  current 15-param) - reserved.
- Lab-spectral-library matching to discriminate intrusive-magmatism
  vs post-depositional-alteration (the v1+v2 chi-square rejects
  airfall but cannot distinguish those two).
- HiRISE / CTX morphology overlap with the intrusion-candidate
  cluster - not in scope for this analysis pass.
- Brown 2020 grain-size lines (70 um, 500 um, 1 mm) on the
  centroid-asymmetry scatter (v4 §6.4). Skipped in v4/v5 - requires
  digitizing Figure 16 or implementing the Shkuratov radiative-
  transfer model. Flagged for v6 if needed.
- Column-neutral ratios (deferred from v3 spec) - flagged for v6.
- Initialise `git` for the project. Currently no version control;
  v4.1 / v5 used the existing snapshot-folder pattern. AV flagged
  this should happen before the next major iteration.
- v6: PC scene back-projections, GMM-class ratio spectra, lobe
  extraction. v4 + v5 save the input arrays per
  `outputs/_v4_vb_pca_brown_rgb/<cube>/arrays/` and
  `outputs/_v5_param_pruning/<cube>/arrays/` so v6 doesn't recompute.

## Revision analyses (2026-08-28)

### Distance-to-contact test on the MTR3 grid
- Revision question (Aug 2026): is alteration in the olivine unit
  concentrated at the clay contact, as the submitted manuscript inferred,
  or pervasive across the unit? Tested directly with
  `code/contact_distance_to_contact.py` + `code/contact_band_center_2300.py`
  (see `PLAN_revision_contact.md`, `outputs/revision/distance_to_contact/REPORT.md`).
- Grid: MTR3 18 m/px (not TER3) so distances are metres and the CRISM DDR
  MOLA elevation band drops on without resampling.
- Classes: v9.1-B argmax labels incl. "novel" pixels (robustness variants:
  excl. novel; OLINDEX3 > per-cube p60). Clay = c4+c5, cap = c1,
  olivine = c0+c2+c3. Bins <= 36 m flagged as sub-pixel mixing.
- Result: D2300 and BD1900R2 in olivine pixels decay monotonically over
  100-400 m from the clay contact in all four cubes; no analogous structure
  at the cap contact; BD2500_2 (carbonate) is NOT contact-concentrated.
  Band-centre and 1.9/2.3 ratio tests cannot separate smectite admixture
  from a hydrated Mg-OH alteration phase (shift < 1 channel; baseline-
  dependent ratio). Decision: describe the result as a contact-associated
  hydration gradient, not as carbonate concentration and not as evidence
  of temperature.
- Rejected: using the c0 class fraction as the headline metric (3E12-fit
  GMM over-assigns c0 on the blueshifted cubes 871C/A4FC).

### Contact test with geologic map and DEMs (2026-08-30)
- Frame: all four MTR3 cubes are offset 300-360 m W (17-20 px) and up to
  +-100 m N-S from the Robbins controlled CTX mosaic (phase correlation,
  `code/contact_coreg_frame.py`; HiRISE manual .IGwx shifts reproduced to
  3-11 m). Decision: apply per-cube constant shifts to the CRISM grid for
  every map/DEM comparison (`geomap/frame_ctx`); PDS frame kept as a
  variant. Validated by map-vs-GMM class agreement rising in all cubes.
- Geologic map: `Nili fossae mapping.gdb/CaSSIS_units`; olivine unit =
  `Olivine_unit` outline minus clay/cap/dune/graben, stacking order
  dune > cap > clay > graben (AV, 2026-08-30). Rasterised at 3 m -> 18 m
  fractions. Pixels outside the outline are unmapped, not olivine.
- Contact defined from the map (>= 50 % mapped clay), not the spectral
  class map; olivine population = >= 90 % mapped olivine with zero mapped
  clay. Result: hydration gradient survives in 3E12/B438 at ~half
  amplitude, weak in A4FC, absent in 871C; carbonate band flat.
- Height above contact: interpolated from lateral map contacts; geometry is
  onlap against clay highs (median h < 0 in 3E12/B438), so h is not a
  stratigraphic height and the thickness estimate is not usable. Report
  only the fixed-distance low-h vs high-h comparison.
- Stage 4 ridge profiles: exploratory, not diagnostic; kept in the outputs.
- Long-ridges HiRISE DTM (72.25 E) is outside all cubes; excluded.

### Adversarial verification of the contact-3D analysis (2026-08-30)
- An adversarial verification pass confirmed the coregistration (sign
  convention, no datum artefact, independent NCC & HiRISE-ortho routes agree
  to ~2 px) and the Stage 1 tables (within 0.6%), and found real errors:
  (1) Stage 2 "onlap against clay highs / negative h" was an IDW
  extrapolation artefact - at the contact itself the olivine side is HIGHER
  than clay (paired transects); (2) the cap contact is a mirror-signed
  gradient, not a null control; (3) pixel-level p-values invalid
  (n_eff ~1% of n); (4) 871C/A4FC inconclusive rather than negative;
  (5) "carbonate flat" wrong for 3E12 (BD2500_2 declines like D2300, but is
  negative everywhere = continuum-slope, not a carbonate band).
- Fixes applied: contact_pixels others-mask bug, cap exclusion in the height
  population, QC quiver sign; new `code/contact_block_stats.py` (block
  bootstrap 504 m / seed 42, correctly-signed Cliff's delta, contact-step
  paired test, overlap split). Decision: all inference is reported at block
  level; pixel p-values dropped.
- Block-level outcome: clay-contact gradient robust in 3E12 (D2300 +0.0035
  CI [+0.0023,+0.0047]; BD1900R2 +0.0051 [+0.0042,+0.0061]) and B438
  (+0.0048 [+0.0036,+0.0059]), holds inside AND outside their ~60% overlap;
  871C/A4FC CIs straddle zero.

### Carbonate patches via the Plebani archive pass (2026-08-30)
- Reused the archive-wide carbonate-only Plebani (crism_ml) maps
  (carb-plebani-1.0; 3E12/871C/B438 done 2026-07-17; A4FC TRR3 absent).
  Projected region pixels through sensor DDR lat/lon onto the shifted
  CRISM grid (same per-cube Stage 0 correction as the MTR3 product).
- Result: discrete MgCO3 patches (~0.3-0.5% of scene) concentrated toward
  the mapped clay contact in 3E12 (143 vs 364 m) and B438 (36 vs 268 m),
  patch-level KS p 8e-13 / 4e-26; flat in 871C's small mapped corner.
  82-100% of patches fall in GMM class c0; BD2500_2 at patches is the
  distribution tail - explains "no band at the median pixel".
- Interpretation guard: contact enrichment is consistent with both
  contact-driven alteration and erosional exposure of a basal
  carbonate-bearing level at unit margins; stated in REPORT 6b.

### Contact fringes and A4FC carbonate completion (2026-08-30)
- Stage 7 (`code/contact_contact_fringe.py`): stacked 17,610 contact-normal
  1 m HiRISE profiles across the mapped clay/olivine contact. Clay side
  rolls up 1.2-1.7 m (median) within ~25-40 m of the line; 68% of profiles
  have >=1 m clay-side bumps. Key control: bumps persist at FLAT olivine
  edges (|step|<=0.5 m, median 1.7-2.0 m, peak ~-12 m) where talus and
  erosional-shoulder explanations fail -> deformed/indurated clay margin.
  Ranking by raw amplitude surfaces scarps; the flat-edge ranking
  (`contact_fringe_gallery_flat.py`) is the diagnostic gallery.
- A4FC carbonate gap: TRR3 downloaded from PDS (mrocr_2102/trdr/2008/2008_068,
  found via ODE REST; the trdrYYYY guess and volumes 2104-2110 were wrong)
  into the archive project's raw TRR3 store for its Plebani pipeline;
  sensor DDR was already in the Nili Planum upload.

### Fringe gallery v2 (2026-08-31, after AV's review)
- AV identified TIN-interpolation facets (stereo-fill) in the first
  amplitude-ranked 006712 gallery. Added a facet detector: median
  |Laplacian| of the 1 m DTM over a 60 m box, threshold 0.05 m
  (fill is piecewise planar -> ~0). Only 0.8% of 006712 sites and 0% of
  002888 sites are fill; population stats unchanged after exclusion.
- v2 galleries: CTX 6 m column (streamed Robbins mosaic) + hillshade,
  hand-drawn clay contact overlain on both, panels labelled P/Q (flat-edge)
  and R/S (strongest, noise-filtered) for AV's per-panel feedback.
  `code/contact_fringe_gallery_v2.py`.

### Slope-break scan and DTM quality verdict (2026-08-31)
- Stage 8 (`code/contact_slope_break_scan.py`): automated detection of
  continuous slope-break lineaments within 100 m of the mapped contact
  (slope >= 12 deg, +7 deg over 50 m median, fill+footprint-edge masks,
  L >= 100 m, w <= 25 m). Two failure modes fixed during QC: median-filled
  nodata created fake 88-deg cliffs at footprint edges (15 m dilated
  exclusion added); matplotlib hillshade OOM (overview decimated 3x).
- Verdict: 002888 catalogue is clean (19 lineaments, best ids 1366, 1149,
  4265, 4871, 6191); 006712 catalogue is junk (pervasive TIN fill in its
  clay unit; facet creases mimic lineaments). DECISION: all DTM-based
  fine-scale claims rest on 002888; 006712 used only for swath-median
  profile stacking, with a stated caution.

### New-DEM registry and coregistration (2026-08-31)
- `code/contact_dtm_registry.py`: single registry for all usable DEMs.
  Adriano's ASP products replace/extend the PDS set: ADI_006712 (0.8 m,
  LOCAL_CS + .tfwx worldfile, CRS assigned Mars-2015-sphere clon 0) is the
  new PRIMARY for the south (holes, not TIN facets); ADI_027691
  (ESP_027691/026992, geoid-aligned) is a new third area. PDS DTEECs kept
  for cross-checks with their Stage 0 shifts.
- `code/contact_new_dem_coreg.py` DRG-vs-Robbins-CTX offsets (applied on
  top of file/worldfile georef): ADI_006712 dx -3.4, dy +23.0 m (268
  tiles, MAD 2-3 m); ADI_027691 dx -30.1, dy -201.6 m (119 tiles, MAD
  1-3 m, conf 0.80, uniform quiver - clean rigid translation; the
  "geoid-aligned" georef is ~200 m off the controlled frame N-S). First
  run OOM'd in the masked full-frame phase correlation (10206x3456
  complex128); fixed by block-decimated coarse lock (`coarse_lock
  max_dim`, opt-in, Stage 0 behaviour unchanged; self-test extended).
- Footprints: neither ADI DEM reaches the R7 pit window (027691's north
  edge ~420 m short in y, no x overlap; 006712 ~14 km south) - the pit
  cross-DTM recurrence test needs an ASP re-reduction of 002888/002176 or
  must use the 027691/002888 swath overlap south of R7.

### Stage 9 pit catalogue verdict (2026-08-31)
- Adversarial verification review (two new diagnostics; every claim
  checked independently; full report
  `outputs/revision/contact_3d/stage9_pits_R7/_report_part2.md`).
- Phase-randomized noise null (script 11, seeds 42-46, identical
  detector): sub-metre classes matched/over-produced by structureless
  surrogates (830 obs vs 800 null; 1745 vs 3196); 1-2 m and 2-3 m within
  null reach (400 vs 1113; 41 vs 63); ONLY >= 3 m exceeds it (101 vs 8.2,
  ~12x; inverted-DEM control agrees, 8 peaks).
- Depth split (script 12): sub-metre pits phase-locked to the 3.407 m row
  ripple (p up to 2.2e-9); 1-2 m locked to the 10.222 m ripple (p=1.6e-3);
  98% of >= 3 m pits confined to the steep shadowed clay-scarp zone (31.6%
  of area; 86/101 in Clay_unit, median slope 23.7 deg); dark centres
  (78/101 vs 42% null) causally ambiguous (shadow causes matcher failure);
  ortho is DTM-rectified, not independent; olivine/clay density contrast
  inverts with depth (2.3x sub-2 m -> 0.15-0.33 deep).
- Chain test of AV's graben hypothesis (script 13): >= 3 m pits DO chain
  (7 chains vs 2.5 null, p=0.008; 22-member 187 m chain az 105 deg) and
  elongations align with the contact trend (89 vs 94 deg), but never
  coalesce (25 components at t=0.02 m; 20 m transverse thalweg rise) - a
  catena of discrete basins, not a graben. Caveat: alignment is
  observationally identical to blunders strung along the E-W scarp.
- DECISION: the n=3117 catalogue is NOT publishable as a landform
  population; never quote n=3117 / 2207 per km2 / median 0.63 m. The
  >= 3 m class is ~100 coherent CANDIDATE structures pending per-pit
  stereo vetting. Only pit 567 (273 m basin) is individually secure.
  Corrections adopted: banding-trough stat non-diagnostic (de-circularized
  56.8%, surrogates 64.6-66.8%); Clark-Evans vs mask-corrected null
  (1.011 +/- 0.009); unit densities exclusive-area only; zero-TIN claim
  vacuous.
- DECISION: decouple the pits from the contact-metamorphism argument
  entirely ("near the contact" = "on the scarp" = where the matcher
  fails; corrected Clark-Evans randomness argues against dike-controlled
  chains; even real pits would be young). Value is protective: the
  artifact battery documents that we declined to overclaim.

### CaSSIS pansharpened gallery (v4) stopped (2026-09-01)
- The 2026-08-31 session ended in a machine crash while
  `contact_fringe_gallery_v4_cassis.py` (written 01:43) was reading the
  PANCO GSA tiles from Dropbox (GSA12 7.9 GB, GSA10 1.0 GB). Zero outputs
  were written: no QC PNGs, no gallery figure, no params JSON in
  stage7_fringe. Nothing to salvage; a re-run would start from scratch.
- DECISION (AV, 2026-09-01): stop the pansharpened-CaSSIS panel work.
  The script stays parked in `code/` unrun. The v3 HiRISE-only gallery
  (`stage7_fringe/fig_gallery_v3_selected.png`) remains the current
  contact-margin figure.

### Tier 0 band-centre map run (2026-09-01)
- AV approved the Tier 0 per-pixel 2.3 um band-centre map
  (PLAN_revision_2300_zonation.md); `code/contact_band_center_map.py`
  ran on all four cubes (map-based contact, CTX-corrected frame,
  strong tier depth >= 0.015). Full report:
  outputs/revision/band_center_map/REPORT.md.
- Findings: no ordered mineral zonation at any resolvable scale
  (no 2.33-2.35 um population, no contact-hugging long-centre zone);
  olivine-unit strong-band centres sit at ~2.316-2.320 um (Ehlmann
  serpentinized-olivine-like) UNIT-WIDE; distance trends (+1.4 to
  +5.8 nm, far minus near) run in the smectite-mixing/depth-bias
  direction, opposite to thermal zonation; B438 strong tier sits at
  the clay value at all distances (consistent with its inferred
  20-30% sub-pixel clay cover).
- DECISION: the ordered-zonation criterion for contact metamorphism is
  observationally negative at >= 1-channel scale; do not claim zonation
  in the revised manuscript. Attribution of the small distance trends
  (mixing vs depth bias vs real) awaits the Tier 1 gates (mixing curve,
  depth matching) - not yet approved to run.

### Band-centre campaign closed: Plebani-chain replication (2026-09-01)
- AV judged the 3E12 NW high-decile/tail field detector noise (straight,
  slightly rotated N-S). Decisive test run: full band-centre battery
  recomputed on the cached crism_ml bland-COLUMN-ratioed TRR3 spectra
  (Zenodo section E, `plebani_inputs/ratioed/`) and projected to the MTR3 grid
  (code/contact_band_center_plebani.py + contact_plebani_figures.py;
  verdict report outputs/revision/band_center_map/plebani/REPORT.md).
- CONFIRMED ARTIFACT: the NW field vanishes under the column-cancelling
  ratio (3E12 tail 5,121 -> 1,165 px, survivors are stripe/edge strings;
  B438 survivors overtly stripe-organized).
- DECISION: WITHDRAW the "anhydrous 2.33 um phase / possible Fe-Ca
  carbonate" lead; do not resurrect. The clay two-province split is
  chain-sensitive and NOT carried forward. The monotonic centre-vs-
  distance trends do not replicate; no band-position gradient claim of
  any kind. The hydration-DEPTH gradient ledger (block stats) is a
  separate result and stands unchanged.
- KEPT (final, chain-robust): NO ordered mineral zonation in the 2.3 um
  band-position field in either processing chain, any cube - the
  ordered-zonation criterion for contact metamorphism is observationally
  negative at >= 1 channel / >= 1 pixel scale. Carbonate patches as
  short-centre anomalies also stand (dual-chain by construction).
- Methods rule going forward: the two chains differ by ~2 channels in
  absolute wavelength and ~7x in strong-tier counts; only within-chain
  relative structure that replicates across chains is claimable.

### v10 revision package drafted (2026-09-02)
- Created manuscript_v10/: supplement_S_contact_metamorphism.md (A/B/C
  contact-metamorphism test framework: zonation negative dual-chain,
  no contact-confined T-diagnostic phase, raised margins present but
  non-diagnostic; carbonate patch concentration + block hydration
  gradients with full caveats), three supplement figures +
  robustness_near90.csv
  (90-150 m near band: all four conclusive-cube hydration contrasts
  stay significant, e.g. 3E12 D2300 +0.0034 [+0.0023,+0.0048]).
- The supplement passed an adversarial verification pass (numbers vs
  ledgers, withdrawn-claims compliance, completeness); fixes applied:
  Table 1 row reworded to null discrimination among emplacement
  end-members; the "bins below 90 m excluded" misstatement replaced by
  the actual convention (<=36 m mixing-affected, 36-90 m registration-
  limited) backed by the new robustness CSV; band-centre shift claims
  restricted to what each chain shows.

### 1.4 um band-centre campaign (2026-09-04)
- Dedicated 1.4 um battery run per AV's request (maps, histograms,
  centre-vs-distance, all four cubes). Full report:
  outputs/revision/band_center_map/band1400/REPORT.md.
- FINDING: the MTR3 1.4 um window is systematics-limited for band
  POSITION - fixed-wavelength minima at ~1.441 and ~1.477 um in every
  unit incl. the anhydrous cap (volcano-scan CO2 residuals; unit-median
  diagnostic figure). Prehnite (1.48 um) is untestable in that chain;
  BD1400's ~1.467 um anchor sits on the residual wing. AV's recalled
  BD1400 spatial structure is nonetheless real: unit-dependent broad
  1.37-1.45 um depth (olivine unit deepest in all cubes) underlies the
  residuals.
- PLEBANI chain (residuals cancelled): clay keeps a genuine
  1.389-1.392 um OH peak; olivine differential minima broad; no coherent
  distance trends. PREHNITE: no discrete 1.48 um population, >=1.46 um
  tail fractions unit-independent (12% olivine vs 13% cap, 3E12) = noise
  tail. Criterion B is now a TESTED negative; supplement updated (the
  "dedicated 1.48 search not performed" caveat is
  replaced by the tested result).
- DO NOT USE: any MTR3-chain 1.4 um band-position map; first-run
  1.44/1.48 "populations"; 1.4 um positional distance trends.

### Shape-selective band parameters prototyped (2026-09-04)
- AV's point: a fixed-wavelength band DEPTH measures displacement from a
  chord, so it cannot tell an absorption from continuum curvature - the
  reason BD-type parameters mislead over the olivine 1 um wing. Five
  methods prototyped on the two 3E12 ROI spectra
  (code/contact_shape_param_prototype.py; report
  outputs/revision/band_center_map/band1400/SHAPE_PARAMS.md).
- RESULT on the olivine ROI (truth: no 1.4 um band): linear-continuum
  depth 0.030 (false positive), convex-hull depth 0.068 (worse),
  quadratic-continuum -0.007 (ok but wing-dependent offset), 2nd-derivative
  SNR 1.2 (correct null), Gaussian-vs-background delta-BIC -10 (correct
  null). On the smectite ROI all shape-aware methods recover the real
  1.388-1.394 um band (delta-BIC +57, FWHM 34 nm).
- Injection test (25 nm Gaussian onto the olivine wing): detection limit
  ~0.5-0.7% depth (2nd derivative, SNR 3) and ~0.8-1.0% (model comparison,
  delta-BIC 10), with centre recovered to 1-2 nm above 1%.
- DECISION: for any future 1.4 um work use M4 (background x Gaussian with
  delta-BIC, plus FWHM 15-40 nm and centre 1.37-1.42 gates) as the primary
  parameter and M3 (matched-scale 2nd derivative) as screen/confirmation;
  hull WIDTH as a secondary flag; retire linear-continuum band depth and
  BD1400 in the olivine unit. Limits quoted are for STACKED spectra - the
  natural product is a block-level (504 m) map, not per-pixel.
- HARDENED after an adversarial methods review and literature audit
  (2026-09-04): (a) a textbook delta-BIC > 10 threshold is INVALID -
  under ISRF-correlated noise the null delta-BIC median is +6 to +15 and
  99th pct +40 to +68, so thresholds must be calibrated empirically on
  band-free windows (1.31/1.33/1.56/1.58 um) per detector column;
  (b) the Gaussian must be ISRF-convolved or its observed FWHM bounded
  to 20-40 nm; (c) error bars = shoulder/window jitter envelope
  (0.005-0.008), NOT fit covariance - bias exceeds statistical error by
  2-5x; (d) M3 (2nd derivative) is restricted to the bland-ratioed chain
  and block statistics (SG kernels reach the 1.441 um CO2 residual;
  per-pixel SNR 0.8-2.4); (e) M2 hull REJECTED outright (over a
  concave-up wing the upper hull IS the chord, plus a +0.004
  albedo-correlated noise bias). Claim ceiling: an upper limit on narrow
  OH amplitude (~0.004-0.006) in block-median olivine spectra, NOT a
  hydration map; no per-pixel detections; no mineral ID from band centre
  (the ~13 nm inter-chain offset equals the 1.39-vs-1.41 separation).
- CORRECTION to an earlier statement: "clay 1.4 um depth is genuine OH"
  is too clean - map-pure clay also correlates with OLINDEX3 at
  0.58-0.64 (3E12/A4FC/B438), exceeding BD1900R2 in 3E12 (0.599 vs
  0.136); only 871C is clean. The defensible positive control is the
  direct shape-verified band on the bland-ratioed clay ROI.
- LITERATURE (citable): Brown 2020 fits FRT00003E12's olivine 1 um band
  at centroid 1.412 um with a tail at ~1.43 um and a population
  redshifted into 1.4-1.5 um - our olivine "1.43-1.45 um band centres"
  ARE his olivine centroid, same cube. Mandon 2020's olivine index uses
  band depths at 1.37 and 1.47 um and treats 1.34 um convexity as a
  positive olivine test (our left anchor sits on it). VB2014 revised
  LCPINDEX2/HCPINDEX2 for "false positives from ... convexity between
  ~1.3 and 1.5 um" and flags BD1300 with "Fe-Olivine can be > 0" while
  BD1400's caveat cell is EMPTY - a documented gap this project can fill.

### Per-pixel 1.4 um OH detection map built and validated (2026-09-05)
- AV's design (fit a Gaussian, map detected pixels colour-coded by band
  centre, on the Plebani-ratioed products) implemented as a matched
  filter with empirically calibrated threshold
  (code/contact_oh_matched_filter.py; report
  outputs/revision/band_center_map/oh_filter/REPORT.md).
- Empirical null thresholds SNR 8.1-10.2 (a textbook 3-5 would have
  flooded the map). VALIDATION GATE PASSED: cap negative control at the
  1% false-positive floor (0.3-1.2%); rho(amplitude, OLINDEX3) in
  map-pure olivine -0.10 to +0.03 (was 0.56-0.83 with the chord method);
  clay positive control detected in 8-23% of pixels at 1.40-1.41 um,
  amplitude 1.0-1.3%; olivine unit at the floor (0.8-2.1%).
- DECISION: the olivine unit has no detectable narrow OH band per pixel;
  quotable as an upper limit of order 0.5-1% (with the stacked-spectrum
  prototype). Detections are unit-bound (clay polygons), no halos around
  olivine bodies - reinforces supplement criterion B. Absolute centres
  are NOT for mineral ID (inter-chain offset ~13 nm = the 1.39/1.41
  separation); within-chain polygon-to-polygon centre differences are
  formally resolvable but untested at block level.
- Before any count-based olivine statement: column stripe screen; before
  any quoted amplitude uncertainty: jitter-envelope propagation.

### Matched filter extended to 2.3 and 1.9 um (2026-09-05)
- code/contact_band_matched_filter.py; report
  outputs/revision/band_center_map/matched_filter/REPORT.md.
- 2.3 um VALIDATED (cap at floor, rho(amp,OLINDEX3) in olivine ~0,
  rho(amp,D2300) in clay 0.70-0.79, clay detected 28-46% at 2.300 um).
  NEW: the map-pure olivine unit carries shape-verified 2.3 um detections
  in ~10% of pixels (3E12 10.7%, 871C 7.6%, A4FC 10.2%, B438 2.6%),
  centre 2.304-2.306 um, amplitude ~1%, forming a broad non-stripy swath
  in the WESTERN olivine unit of 3E12 (same geography as the block-level
  D2300 gradient and the carbonate patches). Candidate explanations
  (hydrated alteration / Mg-carbonate / sub-pixel clay) NOT yet
  discriminated; the swath extends beyond the mixing zone, so (c) alone
  cannot explain it. Proposed next: block-level near/far contrast of the
  detection fraction (the shape-verified contact test) + MgCO3 patch
  overlap + 2.50 um partner check.
- 1.9 um: estimator NOT suitable as configured - null thresholds SNR
  32-42, zero detections in any unit including clay (which has the
  band: rho(amp,BD1900R2)=0.58-0.67). Broad 40-90 nm templates degenerate
  with background curvature on the olivine-wing null windows. DECISION:
  keep BD1900R2 for 1.9 um (close anchors, block-verified); do not
  interpret the 1.9 um matched-filter run as a null result.

### Unified unit-contour style on all map figures (2026-09-05)
- AV's request: thick, colour-coded unit outlines. Shared helper
  code/figstyle.py (draw_units: olivine red #d62728, phyllosilicate blue
  #1f77b4, mafic cap dark dashed, all on a white casing; unit_legend).
- code/refigure_maps.py regenerated 36 map figures from saved arrays
  (Tier 0, band1400, plebani 2.3/1.4 incl. gap-filled, oh_filter,
  matched_filter/2300, high-decile maps) to the SAME filenames. 15
  generating scripts patched to call draw_units (compile-checked; no
  hand-drawn unit contours remain); stale "black = clay" captions fixed;
  supplement_fig_carbonate / _bandcenter, overlay_check, anomaly pixel
  maps and plebani_figures re-run. Carbonate patch markers changed to
  violet to avoid clashing with the red olivine outline.

### Joint 1.4/2.3/2.5 um combination analysis (2026-09-07)
- AV asked for the 1.4 x 2.3 um overlap and its combinations; built as
  per-pixel classes (neither / 1.4-only / 2.3-only / both) on the Plebani
  chain plus joint centre histograms, an amplitude-matched dilution
  control, a bimodal sub-population split, a 2.5 um matched-filter run
  and stacked spectra. Report: outputs/revision/band_center_map/
  combinations/REPORT.md.
- RESULTS: (i) only ONE hydrated-phyllosilicate phase exists in all
  scenes - Fe/Mg smectite at (1.40-1.41, 2.30 um), no 2.5 um band;
  (ii) at matched 2.3 um amplitude the olivine unit co-detects 1.4 um
  5-15x less often than clay (3E12/871C/B438) -> its 2.3 um phase is NOT
  diluted clay; A4FC is the exception (mixing cube); (iii) the olivine
  '2.3-only' population is bimodal: a ~2.30 um mode whose stack carries a
  2.5 um carbonate partner (0.45-1.0 % vs clay floor 0.1-0.2 %) =
  Mg-carbonate-bearing, and a 2.31-2.32 um mode with NO 1.9 um water and
  2.5 at floor = anhydrous serpentine/talc-family Mg-OH phase (Ehlmann
  2009 'serpentinized olivine'); both interior to the unit in 3E12
  (median 285 m from clay). 'Olivine both' pixels = carbonate cores
  (2.5 um 1.0-1.6 %).
- DECISION: provisional phase labels adopted (smectite basement;
  carbonate + serpentine/talc-family olivine-unit phases); usable in the
  manuscript only after (a) block-level statistics and (b) a second-chain
  check; absolute centres never used to choose serpentine vs talc. The
  per-pixel 2.5 um run is at the floor everywhere and is NOT a null
  result - the 2.5 test lives on stacks.

### Phase-class block test and elevation test (2026-09-07)
- Block-level near/far test on the three olivine 2.3 um classes
  (carbonate-mode / serpentine-talc-mode / both-bands): NO CI-excluding
  contrast in 3E12, 871C, B438; A4FC serpentine-mode far-enriched and
  both-bands near-enriched (mixing). DECISION: the per-pixel distance
  ordering of the classes is NOT claimable; no phase zonation toward the
  contact. Stage-6 patch-core concentration stands (different product).
- Elevation test (local relative elevation, matched distance bins, CTX +
  HiRISE, block bootstrap): in 3E12 carbonate-mode pixels sit 0.5 m (CTX)
  to 1.1-1.3 m (HiRISE) LOWER than adjacent olivine ground at 90-250 m
  and > 250 m, cores lower still, serpentine-mode at/above ground; 871C
  partially consistent near the contact; A4FC/B438 null. DECISION: the
  diffuse carbonate occupies local lows -> supports erosional exposure of a
  lower carbonate-bearing level (or hollow-hosted carbonate) over a
  contact-fed halo for the diffuse component; the supplement may
  strengthen the erosional-exposure alternative accordingly. The
  serpentine/talc-family phase is pervasive/interior, not contact-
  controlled. Report: combinations/REPORT.md sec. 7.

### Hotspot topographic profiles, 3E12 (2026-09-07)
- AV's objection to the sec. 7 elevation result: contrasts of ~1 m against
  wide CIs; absolute elevations useless (20-140 m of relief per km of
  profile). Response: old-school profiles (two orthogonal, HiRISE mosaic
  002888 + ADI 027691, CTX fallback) through the 5 hottest interior
  carbonate and 5 hottest serpentine-mode patches; patch elevation vs the
  chord between the flanking olivine ground. Script
  contact_hotspot_profiles.py; REPORT.md sec. 8.
- RESULT: carbonate hotspots show NO consistent landform - 2 hollows (one a
  15 m trough on HiRISE), 2 benches, 1 knob crest; the NE hotspots are
  whole olivine knobs standing 10 m above the clay. Serpentine-mode
  hotspots sit on crests/shoulders (positive chord offset in 7 of 9,
  median +1.3 m, up to +6 m), never in hollows; all on CTX.
- DECISION: the sec. 7 'carbonate 0.5-1.3 m lower' is downgraded to a
  small average tendency, not a landform; the erosional-window reading is
  NOT supported at the patch scale and the supplement wording stays at
  'consistent with', not 'supported by'. New kept observation: the
  serpentine/talc-mode signature is expressed on local highs - a pervasive
  unit property, not contact- or hollow-related. HiRISE coverage of the
  3E12 interior is partial (no DTM over the NW/W serpentine field).
- 2026-09-08 extension to all four cubes (+ absolute-elevation versions):
  carbonate hotspots have NO preferred landform in any cube (13 low / 9
  flat / 11 high of 40 profiles, median chord -0.2 m). Serpentine-mode
  hotspots prefer local highs in 3E12, B438 and (weakly) 871C - 16 high vs
  5 low - and reverse in A4FC (mixing cube). DECISION: 'serpentine-mode on
  local highs' is a kept, three-cube observation (CTX + HiRISE); the
  carbonate elevation tendency is not to be used as evidence for
  erosional windows. REPORT.md sec. 8b.

### Spectrum-site panels, alignment check (2026-09-08)
- AV's two spectrum sites: site 1 = 77.1370253 E, 21.9842103 N; site 2 =
  77.1467583 E, 22.3070500 N. Script contact_site_panels.py -> outputs/
  revision/sites/ (fig_{site}_panels, {site}_layers.npz, run_log_{site}.json).
  500 x 500 m boxes, 0.5 m grid, CTX Robbins frame; panels (a) CTX + DEM
  contours, (b) DEM hillshade, (c) DEM colour + contours, (d) HiRISE RED
  ortho (CaSSIS pansharpened panel to be added by AV).
- Site 1 is covered by the 006712 pair (Adriano's 0.8 m ASP DEM = primary;
  PDS DTEEC cross-check; PDS 0.25 m RED ortho). Local residual of the
  frame-shifted HiRISE products vs CTX in a 2 km window: ADI DEM/DRG
  -8.4/+2.4 m, PDS ortho -11.4/+6.0 m (MAD ~1 m, conf 0.81) -> the frame
  shift leaves a ~10 m local residual (probably a small rotation/scale of
  the HiRISE georef over its 20 km strip). DECISION: apply the measured
  local residual as a documented site-local refinement (LOCAL_REFINE=True,
  values in the run log); after it the layers agree at the ~1 m level.
  Site 1 sits on the east flank/crest of a ~5 m, ~100 m knob at the edge of
  a plateau that drops ~25 m eastward.
- Site 2 is NOT covered by any HiRISE DTM on disk (1.3 km E of the 002888
  DTM, 1.6 km N of the 027691 DEM) nor by a HiRISE ortho; only the CTX 20 m
  DEM exists there. In CTX it sits on a sharp NE-SW bright lineament/ridge
  crest that the CTX DEM cannot resolve. Profiles at site 2 need a HiRISE
  DTM (check PDS for a stereo pair) or a CaSSIS DTM.

### Paper site figure regenerated in the project frame (2026-09-09)
- contact_paper_fig_site1.py -> outputs/revision/sites/fig_paper_site1(.png/.pdf)
  and fig_paper_site1_slope (5-panel variant), paper_fig_site1_layers.npz,
  run_log_paper_fig_site1.json. Window = site1_wide (900 x 900 m, the
  footprint of AV's QGIS panel located by template matching). Panels:
  CaSSIS NIR-PAN-BLU colour / geologic map (paper colours: olivine #e07820,
  clay #2880a8; mafic cap grey) / OLINDEX3 (A4FC v8 stack, 18 m) / HiRISE
  DEM hillshade + elevation (+ slope). Contours off; no profile line; the
  'next figure' box (site1 300 x 270 m fracture-band window) only in (a).
- REGISTRATION (all to the CTX Robbins frame): CaSSIS cube
  MY37_025236_157_0.cubeit.cub file georef is OFF BY +83 / -1159 m (1.16 km
  N-S) - measured by phase correlation of PAN vs CTX in 3 km and 6 km
  windows (639-766 good tiles, MAD ~2 px), residual after correction 0.3 px
  (~2 m). Any CaSSIS cube in 'CaSSIS cubes/' must be registered before use;
  never trust its .cub georef. HiRISE ADI_006712: frame shift + site-local
  residual (-9/+2 m). CRISM A4FC: frame shift -304/-14 m (MAD 50/18 m at
  18 m/px, the weakest link). Geologic map: CTX frame by construction
  (validated indirectly by the stage-2 contact-step test).
- OPEN: the gdb has a Mafic_cap unit over the NW quarter of the window,
  which AV's paper panel (b) shows as clay; the cap colour of the paper's
  Fig. 4a is unknown (grey used). CaSSIS pansharpened product not found on
  disk (nothing new under 'CaSSIS cubes/'); the 4 m cube stands in.
- 2026-09-09 revision (AV): Mafic_cap painted as clay (CAP_AS_CLAY); window
  780 x 780 m, centre 285 m W / 305 m S of the spectrum (pansharpened strip
  fills the view; SITES['site1_wide'] updated -> the wide DEM figure moves too);
  no spectrum marker; olivine outline in white on (c) and (d); A4 full-width
  layout (7.3 in, inset colourbars, 16/14/13 pt letters/labels/scale bar);
  OLINDEX shown black = 0. Panel (c) now defaults to OLINDEX2 (paper index)
  computed by contact_olindex_compare.py from the A4FC MTR3 cube; OLINDEX3
  via --param. WHY: same cube, same 7-channel kernels, no spatial filtering,
  yet OLINDEX3 (1.21-1.33 um vs a 1.75-1.86 continuum) is much noisier than
  OLINDEX2 (1.08-1.47 um vs a 1.75-2.40 continuum) - r = 0.86 pixelwise; the
  fresh OLINDEX3 equals the v8 stack exactly (max diff 0), so the earlier
  'interpolated' look was the percentile stretch (black at 0.09), not
  smoothing. Pansharpened CaSSIS (cubes/GSA_MY36_014433_022_0_nodatacorr.tif,
  0.29 m) registers to CTX at -11/+19 m (MAD 1/4 m); its east edge lies 172 m
  E of the spectrum point.
- OLINDEX2 vs OLINDEX3 anatomy (contact_olindex_anatomy.py, fig_olindex_anatomy_
  FRT0000A4FC, fig_olindex_raw_pixels_FRT0000A4FC): both maps are raw 18 m MTR3
  pixels (nearest, no spatial filter). In the site-1 window OLINDEX2 has
  pixel noise 0.0037 vs spatial signal range 0.114 (S/N 31); OLINDEX3 noise
  0.0079 vs range 0.071 (S/N 9) - 3.5x worse, because its five samples sit
  in a 0.12 um cluster at 1.21-1.33 um and its continuum is extrapolated
  0.5 um back from a 0.11 um baseline (1.75-1.862), while OLINDEX2 samples
  the whole wing to 1.47 um (0.4 weight at 1.47) against a 0.65 um baseline.
  Note: within this window the map-pure olivine-unit and clay-unit MEDIAN
  spectra have the same 1 um band and wing (indices 0.143 vs 0.139); the
  units differ only beyond 1.75 um (clay: 1.9/2.3 um bands, steeper slope).
  The index contrast in the window is carried by specific bright patches,
  not by the mapped units as a whole.
- 2026-09-09 BUG (fixed): contour overlays drawn with `ax.contour(Z, extent=(0, nx,
  ny, 0))` were mirrored top-to-bottom (contour puts row 0 at the extent's
  bottom, imshow at its top). Affected: the olivine outlines in
  fig_paper_site1 (c)/(d) and the DEM contours of the earlier
  fig_site1/site2_panels versions (contours now off). NOT affected: any
  figure using figstyle.draw_units or contour without extent (all
  band-centre-campaign maps, hotspot maps). Fix: contour(XC, YC, Z) with
  pixel-centre coordinates. Alignment conclusions rest on phase correlation
  and stand; retract only the visual 'knob inside the contour ring' remark
  from the first site-1 panel. Map paint rule for the paper figure: olivine
  base, clay polygons (with holes) on top, Mafic_cap layer not drawn ->
  olivine 49 % / clay 51 % of the window.
- 2026-09-09 CRISM site-local registration (contact_crism_local_residual.py):
  A4FC MTR3 vs CTX in 4/6 km windows around site 1, after the frame shift,
  for 749 (S detector), 1330, 1750, 2200 nm (L detector): residual dx 0 to
  +9 m (poorly constrained, MAD 13-31 m), dy -11 to -14 m (MAD 2 m) - all
  four bands agree, so S and L are co-registered and the georef is good to
  ~0.7 px here. Applied to the paper figure as +3.6/-14.0 m (mean of 4).
  Cross-correlating OLINDEX2 against the mapped olivine fraction prefers
  36 m south (NCC 0.13 -> 0.26): the extra ~1 px beyond the albedo residual
  is an index-vs-map offset, not georeferencing; NOT applied (would be
  map-matching). AV's estimate of ~50 m N-S is thus ~14 m registration +
  ~20 m spectral/map offset + 18 m quantisation.
- 2026-09-09 paper figure (c): AV chose candidate '-18 m' (one CRISM pixel
  south, check_crism_shift_candidates.png) as a manual adjustment on top of
  the frame + site-local shifts; it coincides with the NCC optimum of the
  index vs the mapped olivine unit (0.264 at -18 m vs 0.235 at 0). Declared
  in the caption. Unit labels now placed at the deepest interior point of
  the largest patch (distance transform), clear of the next-figure box.

### Paper Fig. 7 regenerated (2026-09-09)
- contact_paper_fig7.py -> outputs/revision/sites/fig_paper_fig7(.png/.pdf),
  paper_fig7_layers.npz, fig_paper_fig7_caption.txt, run_log_fig_paper_fig7.json.
  Window = SITES['site1'], widened to 380 x 270 m (centre 214 m W / 290 m S of
  the spectrum) so intrusion 1 fits; the Fig. 6 'next figure' box follows.
  (a) CaSSIS pansharpened 0.25 m with the intrusion-1 box and the intrusion-2
  transect A-A'; (b) HiRISE slope; (c) intrusion-1 map from 'dike map.shp'
  (41 blocks) + 'amorphous matrix.shp' + gdb clay; (d) A-A' elevation profile.
- Intrusion-2 transect = AV's red line on the slope panel, located by template
  matching his screenshot (NCC 0.95): 70 m long, azimuth ~72 deg, at x 214-280 /
  y 49-71 m of the 380 m window; sampled +-30 m beyond. Relief 3.3 m: ~1 m
  ridge at A, 1 m trough, 1.5 m step up at 55-60 m (the N-S slope lineation).
- REGISTRATION FINDING: 'dike map.shp' and 'amorphous matrix.shp' were digitised
  on the pansharpened product in its FILE georeference; they sit ~20 m S of the
  bright blocks unless the CaSSIS-to-CTX correction (-12/+20 m) is applied
  (check_fig7_polygon_registration.png). Applied to both; gdb units untouched
  (they were consistent with CTX/HiRISE). Any other shapefile in 'CRISM site
  Creta Vulcanica' digitised on that product needs the same shift.
- 2026-09-09 MAP FRAME FINDING: the gdb units themselves (Clay_unit etc.) were
  also digitised on the pansharpened product's file frame - at their file
  position the clay outlines cut through the fractured olivine band, shifted
  by the CaSSIS correction (-12/+19 m) they follow the unit margins
  (check_fig7_clay_registration.png). Applied to the gdb units in Fig. 6 and
  Fig. 7 (MAP_IN_CASSIS_FRAME). Consequence: the earlier 'manual -18 m'
  CRISM adjustment in Fig. 6 was compensating this map offset; with the map
  corrected the index-vs-map NCC peaks at the MEASURED CRISM position
  (0.250 at 0 m) -> manual shift removed. Fig. 7 (c): matrix trimmed to the
  olivine unit (outline minus clay), box padding 2 m so the dike/matrix map
  fills the panel; unit labels at fixed positions in (a)/(b).
  OPEN (AV's call): whether the whole gdb carries this ~20 m offset relative
  to the CTX/HiRISE frame. Block-level CRISM statistics are insensitive to
  ~1 px, but the HiRISE-scale contact tests (stage 2 contact step, stage 3
  fringe, stage 8 lineaments) used the map at 3 m and would move by 20 m;
  the offset was measured only in the site-1 window and other map areas were
  probably drawn on other CaSSIS products with their own georef errors.
- 2026-09-09 FINAL: manuscript_v10/figures/final/Fig6.{png,pdf} (= fig_paper_site1
  run 10:40 UTC, corrected map, no manual shift; NOT the 10:26 fig_paper_site1.png)
  and Fig7.{png,pdf} (= fig_paper_fig7 run 10:50 UTC, panel (c) 87 x 62 m at the
  panel-(a) aspect), with captions, run logs and a README (md5s). Regenerate via
  the two scripts; do not hand-edit the copies.
- 2026-09-09 Fig. 6 final updated: green box on (c) (GREEN_BOX_M, 228 x 58 m at
  x 544-772 / y 408-466 m of the window, located from AV's screenshot by
  template matching); manuscript_v10/figures/final/Fig6.* refreshed (run 11:00 UTC).
- 2026-09-10 Fig. 6 (c) display stretch changed from [0, p99.5] to [p2, p99.5] =
  0.09-0.25 (OLINDEX_STRETCH_PCT) for olivine/clay contrast at AV's request;
  Fig6_FINAL.* refreshed. Caption states the stretch.
- 2026-09-10 Fig. 6 (c) display stretch changed from [0, p99.5] to [p2, p99.5] =
  0.09-0.25 (OLINDEX_STRETCH_PCT) for olivine/clay contrast at AV's request;
  Fig6_FINAL.* refreshed. Caption states the stretch.
- 2026-09-10 Supplement figures finalised (code/supplement_final_figs_3E12.py ->
  manuscript_v10/figures/final/FigS_carbonate_FINAL.*, FigS_bandcentre_FINAL.*,
  captions, FigS_FINAL_run_log.json). 3E12 only, A4 width, geologic units as
  black line textures (hatch/stipple) or line styles (solid/dashed/dotted).
  FigS_carbonate: 126 on-map patches, median 143 m vs olivine-unit background
  360 m, KS D 0.33 p 1.8e-12 (patch-level). FigS_bandcentre: 1.4 um OH centre
  map (n 28937), 2.3 um centre map (n 74685), joint classes (1.4-only 6187 /
  2.3-only 51935 / both 22750), olivine-unit modes (carbonate 11889 /
  serpentine-talc 11051 / cores 2153). Unit contours masked to the mapped
  area (no map-edge lines).
- 2026-09-10 FigS_bandcentre revised (AV): panels (c)+(d) replaced by one
  three-phase map - Fe/Mg smectite (1.4 + 2.3 um, n 22750), serpentinized
  olivine (olivine-unit 2.3-only >= 2.305 um, no 1.9 um, n 11051), Mg-carbonate
  (crism_ml 2.3 + 2.5 um classifier pixels projected from the sensor grid,
  n 753, dilated 1 px for visibility). '1.4 only', 'carbonate-mode' and
  'carbonate cores' dropped from the figure (still in combinations/REPORT.md).
- 2026-09-10 FigS_carbonate reduced to the map only (CDF/KS panel dropped; the
  numbers - patch median 143 m vs background 360 m, KS D 0.33 p 1.8e-12 - stay
  in the caption text and run log). FigS_bandcentre legend: 'serpentinized
  olivine (2.3 um only)'.
- 2026-09-10 supplement figures revised again (AV): FigS_carbonate = geologic
  sketch + MgCO3 patches (blue) + serpentinized-olivine patches (red circles =
  connected components >= 5 px of the olivine-unit 2.3-only >= 2.305 um class;
  333 of 1118 components, 85 % of the pixels); FigS_bandcentre = two rows,
  one column (1.4 um and 2.3 um centre maps), phase panel dropped.
