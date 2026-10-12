# Stage C summary

**Goal**: align the per-cube k=6 cluster labels into a common reference
frame, then test the central scientific question - is the olivine unit
compositionally homogeneous across cubes (Kremer 2019 airfall) or
heterogeneous (Valantinas 2026 intrusive magmatism)?

**Status**: v1 complete (2026-04-21, k-means + GMM), v2 complete
(2026-04-22, GMM only).

## Pipeline

1. Hungarian alignment of fixed-k=6 cluster centroids to a reference
   cube (FRT00003E12), in standardized 15-param space.
2. Re-order aligned labels so c0 = most-mafic-cap-like (lowest
   hydration+olivine+carbonate z-sum), c5 = most-phyllosilicate-like.
3. Compositional homogeneity test: chi-square on the 4-cube x 6-cluster
   contingency table of mixing fractions, Bonferroni-corrected.
4. Anchor-finding tests:
   - Olivine-carbonate ridges: aligned cluster maximizing z(OLINDEX3) +
     z(BD2500_2)
   - Altered-olivine intrusion candidate: aligned cluster maximizing
     z(OLINDEX3) + z(BD1900R2) + z(D2300)
5. Intrusion-geometry test: connected-component median aspect ratio of
   the intrusion candidate per cube.
6. (v1 only) k-means vs GMM confusion matrix on aligned labels.

## Headline result

**Cross-cube compositional homogeneity is rejected** under both v1 and
v2.

| metric | v1 (Brown-strict) | v2 (full range) |
|---|---:|---:|
| chi^2 (k-means, dof=15) | 25 980 | n/a (skipped in v2) |
| chi^2 (GMM, dof=15) | 15 918 | **112 568** |
| critical chi^2 at p=0.001 | 37.7 | 37.7 |
| p_bonferroni | ~0 | ~0 |

The v2 number is ~7x v1 because the previously-masked low-centroid
pixels are not uniformly distributed across cubes. Including them
makes the heterogeneity louder, not quieter.

## v2 mixing fractions (% per aligned cluster, GMM)

| cube | c0 | c1 | c2 | c3 | c4 | c5 |
|---|---:|---:|---:|---:|---:|---:|
| FRT00003E12 | 25.7 | 16.4 | 23.5 | 13.5 | 15.1 |  5.8 |
| FRT0000871C | 17.4 | 17.4 | 20.5 | 25.4 | 17.8 |  1.5 |
| FRT0000A4FC | 15.3 | 26.6 | 19.2 | 12.3 | 25.9 |  0.8 |
| FRT0000B438 |  6.7 | 17.4 | 17.3 | 21.2 | 27.4 |  9.9 |

Spread per cluster: c0 4x range, c4 1.8x, **c5 12x** (B438 has
9.9% phyllosilicate-rich pixels; A4FC only 0.8%).

## Hungarian alignment quality (v2)

Per-cube euclidean distance to FRT00003E12 in the standardized 15-param
space. Larger distance = the matched cluster is a weaker analogue of
3E12's reference cluster.

| cube | mean dist | max dist (cluster) |
|---|---:|---|
| FRT0000871C | 2.14 | 3.80 (c5) |
| FRT0000A4FC | 2.30 | 2.78 (c1) |
| FRT0000B438 | 1.84 | 2.19 (c0, c5) |

A4FC has the largest average distance - even after Hungarian matching,
A4FC's clusters are uniformly further from 3E12's references than
871C or B438's are. Cross-cube end-members do not perfectly match.

## Anchor-finding (v2 GMM)

- Both anchor candidates (intrusion + olivine-carbonate) collapse onto
  the same aligned cluster c5.
- Intrusion candidate fractions per cube: 5.8 / 1.5 / 0.8 / 9.9%.

## Intrusion-geometry test - inconclusive standalone

Median connected-component aspect ratio of c5 per cube: 2.30 / 1.83 /
2.02 / 2.24. Baseline aspect ~2 across all 6 aligned clusters on all 4
cubes - elongation ~2 is a baseline feature of mixed spatial patterns
at 18 m/pixel, not a fracture-specific signature. The aspect-ratio
test on its own cannot distinguish fracture-controlled emplacement
from layered deposition.

## What this analysis can / cannot say

- **Can say**: draped airfall (Kremer 2019) is inconsistent with the
  data. Mixing fractions differ substantially cube-to-cube, aligned-
  cluster centroids shift cube-to-cube, and Hungarian distances show
  the matched populations are not strict compositional analogues.
- **Cannot say**: intrusive magmatism vs post-depositional fluid
  alteration. Both predict heterogeneous mixing and elongated alteration
  geometry; CRISM at 18 m/pixel cannot resolve the meter-scale dike
  contacts that would morphologically discriminate them. Lab-spectral
  library matching and HiRISE/CTX morphology overlap are needed for
  the next step.

## Outputs

- `03_stageC_homogeneity_test.md` + `.json` - chi-square + mixing-
  fraction tables + per-cube/per-cluster centroid spread
- `03_stageC_anchor_findings.md` + `.json` - intrusion + olivine-
  carbonate cluster identities + per-cluster geometry
- `03_stageC_alignment.json` - Hungarian permutations + distance
  matrices per cube per method
- `03_stageC_kremer_vs_valantinas.md` - synthesis writeup
- `03_stageC_confusion.json` (v1 only) - k-means vs GMM agreement
- Figures: `all_cubes_aligned_k6_labels_*.png`,
  `all_cubes_mixing_fractions.png`,
  `all_cubes_aligned_centroid_comparison.png`,
  `intrusion_candidate_maps.png`, `all_cubes_kmeans_gmm_confusion.png`
  (v1 only)

## Pointers

- Decisions log: `DECISIONS.md`
- Stage C plan: `PLAN_stage_c.md`
- Stage B summary: `SUMMARY_stageB.md`
- v1 snapshot: `outputs/_v1_brown_strict/reports/03_stageC_*`
- v2 snapshot: `outputs/_v2_full_range_gmm/reports/03_stageC_*`
