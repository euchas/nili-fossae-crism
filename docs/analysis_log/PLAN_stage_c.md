# Stage C plan — cross-cube cluster matching and anchor-finding test

## Scope and rationale

Stage B produced per-cube cluster labels. Stage C aligns them across the four
cubes and tests the central scientific question: is the olivine unit
compositionally homogeneous across the Horseshoe site (airfall prediction,
Kremer 2019) or structurally heterogeneous (intrusive prediction, Valantinas
2026)?

Reference layer: **fixed-k=6 k-means on 15-param PCA** — the only partition
with balanced clusters and cross-cube comparability. Cross-validated against
**fixed-k=6 GMM** as a soft-clustering sanity check.

No new heavy compute. All inputs already on disk from Stage B.
Expected wall clock < 10 minutes total.

## Steps

### C-1. Cross-cube cluster alignment (Hungarian)

For both `kmeans_k6_fixed` and `gmm_k6_fixed`:

1. Load per-cube cluster labels and the 15-param standardized stack.
2. Compute each cluster's centroid in the 15-param standardized space.
3. Reference cube = **FRT00003E12** (largest valid fraction, Brown-validated).
4. For each of the other three cubes, solve the assignment problem
   (scipy.optimize.linear_sum_assignment) on the 6x6 euclidean-distance
   matrix between that cube's centroids and 3E12's centroids.
5. Relabel the other three cubes so "cluster k" across all four cubes
   points to the same aligned population.
6. Sort the aligned labels so label 0 = most mafic-cap-like cluster in 3E12
   (lowest z-sum over {BD1900R2, D2300, BD2500_2, OLINDEX3}), label 5 =
   most opposite. This gives a stable, interpretable color-order for the
   cross-cube figures.

Output: `outputs/cluster_labels/<cube>_kmeans_k6_aligned.npy`,
`<cube>_gmm_k6_aligned.npy`, and `outputs/reports/03_stageC_alignment.json`
listing the assignment permutations + per-cluster centroid distances.

### C-2. Compositional homogeneity test

For the aligned kmeans_k6 labels:

1. **Mixing fractions** per cube — 4x6 matrix of pixel-share per aligned
   cluster. Kremer (airfall) predicts near-equal fractions; Valantinas
   (intrusive) predicts cube-dependent fractions.
2. **Aligned-centroid comparison** — per parameter, how much does each
   aligned cluster's median shift cube-to-cube? Kremer predicts small
   shifts; Valantinas predicts systematic shifts.
3. **Compositional distance** — pairwise euclidean between aligned
   centroids across cubes (e.g. 3E12-cluster-0 vs 871C-cluster-0). Report
   as a 6-cluster x 6-pair matrix. If the same aligned cluster differs
   more across cubes than adjacent clusters within one cube, alignment is
   weak and the homogeneity hypothesis is in trouble.

Outputs:
- `outputs/figures/all_cubes_mixing_fractions.png` (stacked bar, 4 cubes)
- `outputs/figures/all_cubes_aligned_centroid_comparison.png`
  (15 panels, one per param, 6 aligned clusters x 4 cubes)
- `outputs/reports/03_stageC_homogeneity_test.md`

### C-3. Anchor-finding tests

From CONTEXT.md Section 4, two anchor findings to test.

1. **Olivine-carbonate-rich ridges** — for each cube, find aligned cluster(s)
   with high OLINDEX3 AND high BD2500_2 (top quartile on both). Does a
   single consistent aligned-label satisfy this across all four cubes?
   Are the corresponding pixels spatially coherent within each cube
   (ridge-like geometry)?
2. **Alteration halos** — look for aligned clusters with intermediate
   BD1900R2 + intermediate OLINDEX3 values that sit spatially between
   olivine-rich and phyllosilicate-rich clusters. For each cube, compute
   per-cluster median values and a "halo score" = closeness to the
   midpoint between olivine-rich and phyllosilicate-rich cluster centroids
   in 15-param space. Flag any aligned cluster scoring as halo on more
   than one cube.

Outputs:
- `outputs/figures/olivine_carbonate_anchor_maps.png` (4 cubes side-by-side,
  candidate cluster highlighted)
- `outputs/figures/alteration_halo_candidates.png` (same)
- `outputs/reports/03_stageC_anchor_findings.md`

### C-4. Cross-method consistency

Confusion matrix between kmeans_k6_fixed aligned labels and gmm_k6_fixed
aligned labels, per cube. If kmeans and GMM agree on cluster structure,
confidence in the fixed-k=6 layer goes up; if they disagree, the k=6 choice
is fragile and the Stage C conclusions need caveats.

Output: `outputs/figures/all_cubes_kmeans_gmm_confusion.png` and a summary
table in `03_stageC_kremer_vs_valantinas.md`.

### C-5. Synthesis report

`outputs/reports/03_stageC_kremer_vs_valantinas.md` — pulls together C-2,
C-3, C-4 into a single writeup framed explicitly around the airfall vs.
intrusive-magmatism question. Does NOT claim confirmation; reports what's
consistent with / distinguishable from each hypothesis, with the specific
evidence. Includes a "what could still be wrong" section.

## Decisions to surface before launching

1. **Reference cube for alignment.** Proposing FRT00003E12 (largest
   n_pixels_joint_valid_15 at 237k, Brown-validated, broadest centroid
   distribution). Alternative is A4FC if we want to anchor on the
   most-restrictive valid geometry, but its 25k pixels and narrow spatial
   coverage make it less representative. **Recommendation: 3E12.**

2. **Alignment metric.** Proposing euclidean distance in the standardized
   15-param space (same space kmeans optimizes). Cosine would de-emphasize
   magnitude differences — useful if we wanted shape-only matching, but
   for cross-cube *compositional* matching euclidean is the honest choice.
   **Recommendation: euclidean on standardized 15-param centroids.**

3. **Which cluster layer to emphasize.** Proposing kmeans_k6_fixed as the
   primary, gmm_k6_fixed as the cross-check (via confusion matrix). The
   adaptive-k runs and the HDBSCAN runs are too unstable cross-cube to be
   the basis for this comparison. **Recommendation: kmeans_k6_fixed primary.**

4. **Anchor cluster detection threshold.** For the olivine-carbonate test,
   "top quartile" (>= 75th percentile on OLINDEX3 AND on BD2500_2) is a
   defensible choice but arbitrary. Alternative: cluster-median-based
   ranking (pick the aligned cluster that maximizes OLINDEX3 median +
   BD2500_2 median z-sum). **Recommendation: use the cluster-median
   ranking for selection, then visualize with the top-quartile pixel mask
   for spatial coherence.**

5. **Mixing-fraction test threshold.** Kremer prediction is "near-equal
   fractions across cubes." Near-equal is operationalized as chi-square
   test on the 4x6 contingency table; reject null if p < 0.001 after
   Bonferroni. **Recommendation: chi-square, with caveat that the null
   requires large-enough expected counts — A4FC's smaller sample will
   have wider confidence bounds.**

## Risks and caveats

- Alignment via Hungarian on centroids assumes the 6 clusters are
  structurally analogous across cubes. If a cube is missing a population
  (e.g. A4FC's edge geometry may not contain a phyllosilicate-rich region
  at all), Hungarian will still produce a one-to-one mapping but the
  matched cluster may be a weak analogue. Mitigation: report per-pair
  centroid distance alongside the permutation; flag weak matches.
- 871C is distributionally distinct from the other cubes on centroid
  (Stage B finding). If that drives the alignment to look like
  "populations with matching labels that differ substantially in
  composition", the right conclusion is that homogeneity fails, not that
  alignment is wrong.
- GMM fixed-k=6 on cubes where BIC preferred k=10 means the 6-component
  fit is suboptimal for GMM — the confusion-matrix agreement with kmeans
  may be artificially low as a result. Caveat in the writeup.

## Go/no-go

Awaiting approval on decisions 1-5 above, then launching a single
orchestration script `stage_c_align_and_test.py`. After it finishes,
reading the three markdown reports + the 6 figures and expecting
pushback on any statement that sounds like "confirms hypothesis X."
