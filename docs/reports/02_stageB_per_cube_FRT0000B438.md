# Stage B per-cube report — FRT0000B438

Generated from logs and summary JSONs. Figures in `outputs/figures/`.

## Feature stack

- joint_valid_15 pixels: **286799** (93.4% of frame)
- BDI1000VIS NaN loss vs baseline: 0.00% (flag: False)

### Per-parameter NaN fractions

| param | NaN frac |
|---|---:|
| centroid | 0.0664 |
| asymmetry | 0.0664 |
| amplitude | 0.0664 |
| hwhm | 0.0664 |
| BD530_2 | 0.0664 |
| BD920_2 | 0.0664 |
| BDI1000VIS | 0.0664 |
| OLINDEX3 | 0.0664 |
| LCPINDEX2 | 0.0664 |
| BD1400 | 0.0664 |
| BD1900R2 | 0.0664 |
| D2300 | 0.0664 |
| BD2290 | 0.0664 |
| BD2355 | 0.0664 |
| BD2500_2 | 0.0664 |

## PCA (15-param)

- PC1=46.2%, PC2=18.4%, PC3=9.2%, PC4=7.2%, PC5=5.8%  → cum5=86.7%
- 8-param baseline: PC1=60.8%, PC2=16.8%, PC3=12.1%

Figures: `*_pca15_variance.png`, `*_pca15_loadings.png`, `*_pca15_density.png`, `*_pc1_pc2_maps.png`.

## UMAP

- heuristic pick: **n100_m0.0**
- validity_index pick: **n15_m0.0**
- protocols agree: **False**
- HDBSCAN min_cluster_size used for validity: 75

Validity scores per config:

- n15_m0.0: 0.025
- n15_m0.1: -0.629
- n15_m0.3: -0.312
- n30_m0.0: -0.016
- n30_m0.1: -0.723
- n30_m0.3: -0.654
- n50_m0.0: -0.506
- n50_m0.1: -0.536
- n50_m0.3: -0.253
- n100_m0.0: -0.620
- n100_m0.1: -0.174
- n100_m0.3: -0.443

Figure: `*_umap_sweep.png`, `*_umap_selected_density_*.png`.

## Clustering — k-means + GMM (PCA 5-D input)

- k-means adaptive k (silhouette): **2**
- GMM adaptive k (BIC): **10**
- Fixed k across cubes: **6**

Figures: `*_kmeans_silhouette_sweep.png`, `*_gmm_bic_sweep.png`.

## Clustering — HDBSCAN (UMAP 2-D input)

- min_cluster_size: **75**
- [heuristic] clusters=2, noise=0.4%
- [validity] clusters=2, noise=0.1%

## Per-method cluster summary

Mean spectra + label maps + param-position figures in `outputs/figures/`. Ratioed-spectra figures use the auto-selected mafic-cap denominator (display-only, never used for clustering).

| method | n_clusters | mafic denom (auto) | denom source |
|---|---:|---:|---|
| gmm_k6_fixed | 6 | c5 | auto |

## Next-stage notes

- For Stage C cluster-matching across cubes, the fixed-k=6 labels provide the nominally-comparable reference.
- If any method's auto-picked mafic denominator looks wrong (check the ratioed spectra visually), override via `outputs/reports/denominator_overrides.json` and rerun `stage_b_visualize.py <cube_id>`.
