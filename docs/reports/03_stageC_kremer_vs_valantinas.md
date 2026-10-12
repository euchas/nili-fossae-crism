# Stage C synthesis: Kremer 2019 vs Valantinas 2026

Two competing hypotheses for the olivine unit at the Horseshoe site: Kremer 2019 argues it is draped pyroclastic airfall; Valantinas 2026 argues intrusive magmatism. Stage C tests both against cross-cube cluster structure using fixed-k=6 GMM.

## Summary of evidence

### Mixing-fraction homogeneity (chi-square)
| method | chi^2 | p_raw | p_bonf |
|---|---:|---:|---:|
| gmm | 112568.1 | 0.00e+00 | 0.00e+00 |

### Intrusion geometry (gmm_pca15_k6_fixed)
Intrusion candidate = aligned cluster **c5**.
| cube | pix | n_comp | median_aspect |
|---|---:|---:|---:|
| FRT00003E12 | 16741 | 105 | 2.30 |
| FRT0000871C | 4325 | 29 | 1.83 |
| FRT0000A4FC | 2410 | 7 | 2.02 |
| FRT0000B438 | 28455 | 119 | 2.24 |

## Verdict on each hypothesis

The chi-square homogeneity test rejects cube-to-cube uniformity of mixing fractions at p_bonferroni effectively 0. This is direct evidence against Kremer 2019's prediction that the olivine unit is compositionally uniform across the Horseshoe site - the aligned populations are not in matching proportions across cubes. Draped airfall is not what the spectral data show.

The intrusion-geometry test (median connected-component aspect ratio of the altered-olivine cluster) is a weak standalone discriminator: all six aligned clusters on all four cubes have median aspect in the 1.8-2.6 range, so elongation ~2 is a baseline feature of mixed spatial patterns at 18 m/pixel rather than a fracture-specific signature. Aspect-ratio evidence supports fracture-controlled emplacement only where the intrusion-candidate cluster is clearly elevated above its cube's other clusters - check the per-cluster geometry table in `03_stageC_anchor_findings.md` to see which cubes show this lift.

What this analysis cannot do: distinguish intrusive magmatism (the olivine unit was emplaced as dikes/sills) from post-depositional fluid alteration of a pre-existing olivine body. Both predict heterogeneous compositional mixing and elongated alteration geometry. The rejection of draped airfall is the strong statement this analysis can make. Further discrimination needs lab-spectral library matching (Stage D candidate) and HiRISE/CTX morphological overlap, not more unsupervised clustering.

## Known limitations
- GMM fixed-k=6 is a cross-cube comparability choice, not a per-cube optimum. BIC preferred a higher k on every cube in v1, so the six components are coarser than the data supports per cube.
- The Hungarian alignment assumes each cube contains analogues of all six reference populations. On cubes where a population is weakly represented, the matched cluster is a weak analogue; see `03_stageC_alignment.json` for per-cube centroid distances to the reference.
