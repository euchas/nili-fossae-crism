# Stage B per-cube report — FRT00003E12

Generated from logs and summary JSONs. Figures in `outputs/figures/`.

## Feature stack

- joint_valid_15 pixels: **286800** (93.4% of frame)
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

- PC1=46.0%, PC2=23.2%, PC3=8.2%, PC4=6.8%, PC5=6.2%  → cum5=90.4%
- 8-param baseline: PC1=59.5%, PC2=22.2%, PC3=12.3%

Figures: `*_pca15_variance.png`, `*_pca15_loadings.png`, `*_pca15_density.png`, `*_pc1_pc2_maps.png`.

## UMAP

- heuristic pick: **n100_m0.0**
- validity_index pick: **n50_m0.1**
- protocols agree: **False**
- HDBSCAN min_cluster_size used for validity: 150

Validity scores per config:

- n15_m0.0: -0.440
- n15_m0.1: -0.424
- n15_m0.3: -inf
- n30_m0.0: -0.450
- n30_m0.1: -0.514
- n30_m0.3: -inf
- n50_m0.0: -0.462
- n50_m0.1: -0.247
- n50_m0.3: -0.596
- n100_m0.0: -0.511
- n100_m0.1: -0.632
- n100_m0.3: -inf

Figure: `*_umap_sweep.png`, `*_umap_selected_density_*.png`.

## Clustering — k-means + GMM (PCA 5-D input)

- k-means adaptive k (silhouette): **3**
- GMM adaptive k (BIC): **10**
- Fixed k across cubes: **6**

Figures: `*_kmeans_silhouette_sweep.png`, `*_gmm_bic_sweep.png`.

## Clustering — HDBSCAN (UMAP 2-D input)

- min_cluster_size: **150**
- [heuristic] clusters=12, noise=8.7%
- [validity] clusters=3, noise=3.1%

## Per-method cluster summary

Mean spectra + label maps + param-position figures in `outputs/figures/`. Ratioed-spectra figures use the auto-selected mafic-cap denominator (display-only, never used for clustering).

| method | n_clusters | mafic denom (auto) | denom source |
|---|---:|---:|---|
| gmm_k6_fixed | 6 | c3 | auto |

## Next-stage notes

- For Stage C cluster-matching across cubes, the fixed-k=6 labels provide the nominally-comparable reference.
- If any method's auto-picked mafic denominator looks wrong (check the ratioed spectra visually), override via `outputs/reports/denominator_overrides.json` and rerun `stage_b_visualize.py <cube_id>`.
