# Stage B per-cube report — FRT0000A4FC

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

- PC1=34.0%, PC2=25.1%, PC3=10.4%, PC4=8.2%, PC5=6.9%  → cum5=84.6%
- 8-param baseline: PC1=44.9%, PC2=22.7%, PC3=13.7%

Figures: `*_pca15_variance.png`, `*_pca15_loadings.png`, `*_pca15_density.png`, `*_pc1_pc2_maps.png`.

## UMAP

- heuristic pick: **n100_m0.0**
- validity_index pick: **n100_m0.0**
- protocols agree: **True**
- HDBSCAN min_cluster_size used for validity: 15

Validity scores per config:

- n15_m0.0: -0.719
- n15_m0.1: -0.666
- n15_m0.3: -0.783
- n30_m0.0: -0.717
- n30_m0.1: -0.640
- n30_m0.3: -0.729
- n50_m0.0: -0.736
- n50_m0.1: -0.722
- n50_m0.3: -0.637
- n100_m0.0: -0.062
- n100_m0.1: -0.723
- n100_m0.3: -0.689

Figure: `*_umap_sweep.png`, `*_umap_selected_density_*.png`.

## Clustering — k-means + GMM (PCA 5-D input)

- k-means adaptive k (silhouette): **2**
- GMM adaptive k (BIC): **9**
- Fixed k across cubes: **6**

Figures: `*_kmeans_silhouette_sweep.png`, `*_gmm_bic_sweep.png`.

## Clustering — HDBSCAN (UMAP 2-D input)

- min_cluster_size: **15**
- [heuristic] clusters=123, noise=51.3%

## Per-method cluster summary

Mean spectra + label maps + param-position figures in `outputs/figures/`. Ratioed-spectra figures use the auto-selected mafic-cap denominator (display-only, never used for clustering).

| method | n_clusters | mafic denom (auto) | denom source |
|---|---:|---:|---|
| gmm_k6_fixed | 6 | c3 | auto |

## Next-stage notes

- For Stage C cluster-matching across cubes, the fixed-k=6 labels provide the nominally-comparable reference.
- If any method's auto-picked mafic denominator looks wrong (check the ratioed spectra visually), override via `outputs/reports/denominator_overrides.json` and rerun `stage_b_visualize.py <cube_id>`.
