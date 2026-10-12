# Stage B per-cube report — FRT0000871C

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

- PC1=35.2%, PC2=23.2%, PC3=9.7%, PC4=9.2%, PC5=6.2%  → cum5=83.5%
- 8-param baseline: PC1=50.9%, PC2=22.1%, PC3=12.7%

Figures: `*_pca15_variance.png`, `*_pca15_loadings.png`, `*_pca15_density.png`, `*_pc1_pc2_maps.png`.

## UMAP

- heuristic pick: **n100_m0.0**
- validity_index pick: **n50_m0.3**
- protocols agree: **False**
- HDBSCAN min_cluster_size used for validity: 100

Validity scores per config:

- n15_m0.0: -0.667
- n15_m0.1: -0.638
- n15_m0.3: -0.454
- n30_m0.0: -0.505
- n30_m0.1: -0.596
- n30_m0.3: -0.543
- n50_m0.0: -0.482
- n50_m0.1: -0.535
- n50_m0.3: -0.428
- n100_m0.0: -0.680
- n100_m0.1: -inf
- n100_m0.3: -0.565

Figure: `*_umap_sweep.png`, `*_umap_selected_density_*.png`.

## Clustering — k-means + GMM (PCA 5-D input)

- k-means adaptive k (silhouette): **2**
- GMM adaptive k (BIC): **10**
- Fixed k across cubes: **6**

Figures: `*_kmeans_silhouette_sweep.png`, `*_gmm_bic_sweep.png`.

## Clustering — HDBSCAN (UMAP 2-D input)

- min_cluster_size: **100**
- [heuristic] clusters=2, noise=0.0%
- [validity] clusters=10, noise=6.9%

## Per-method cluster summary

Mean spectra + label maps + param-position figures in `outputs/figures/`. Ratioed-spectra figures use the auto-selected mafic-cap denominator (display-only, never used for clustering).

| method | n_clusters | mafic denom (auto) | denom source |
|---|---:|---:|---|
| gmm_k6_fixed | 6 | c4 | auto |

## Next-stage notes

- For Stage C cluster-matching across cubes, the fixed-k=6 labels provide the nominally-comparable reference.
- If any method's auto-picked mafic denominator looks wrong (check the ratioed spectra visually), override via `outputs/reports/denominator_overrides.json` and rerun `stage_b_visualize.py <cube_id>`.
