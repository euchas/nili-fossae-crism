# Stage B cross-cube notes

Plots: `all_cubes_median_subtracted.png`, `all_cubes_centroid_ks_wasserstein.png`, `all_cubes_pca15_loadings_compare.png`. Per-cube deliverables under `02_stageB_per_cube_<cube>.md`.

## Centroid pairwise distances

From `02_stageB_centroid_pairwise.csv`:

```
metric,a,b,value
ks,FRT00003E12,FRT0000871C,0.440402
wasserstein,FRT00003E12,FRT0000871C,0.071034
ks,FRT00003E12,FRT0000A4FC,0.304596
wasserstein,FRT00003E12,FRT0000A4FC,0.035152
ks,FRT00003E12,FRT0000B438,0.156962
wasserstein,FRT00003E12,FRT0000B438,0.016144
ks,FRT0000871C,FRT0000A4FC,0.293706
wasserstein,FRT0000871C,FRT0000A4FC,0.036297
ks,FRT0000871C,FRT0000B438,0.389190
wasserstein,FRT0000871C,FRT0000B438,0.058540
ks,FRT0000A4FC,FRT0000B438,0.181265
wasserstein,FRT0000A4FC,FRT0000B438,0.022252

```

## Per-cube method outcomes

| cube | km_adaptive_k | gmm_adaptive_k | hdbscan_heuristic k | noise% |
|---|---:|---:|---:|---:|
| FRT00003E12 | 3 | 10 | 12 | 8.7% |
| FRT0000871C | 2 | 10 | 2 | 0.0% |
| FRT0000A4FC | 2 | 9 | 123 | 51.3% |
| FRT0000B438 | 2 | 10 | 2 | 0.4% |

## Open questions for review

- Does the fixed-k=6 labelling produce the same six interpretable populations across cubes, or do some cubes over- or under-split? See per-cube `*_kmeans_k6_fixed_labels.png` side-by-side.
- Where does the mafic-cap cluster sit in PC1-PC2 on each cube? The automatic denominator should fall in the same region on each; if A4FC's auto-pick is inconsistent, flag for override.
- GMM transition maps (`*_gmm_k*_transition.png`) mark pixels with high second-best posterior - are those spatially coincident with OU boundaries as the Viviano 2013 talc-transition framing predicts?
