# Stage 9 - Pit-like depressions around fringe-site R7: characterization and artifact assessment

Date: 2026-08-30
Site: R7, (x, y) = (2219.2, 1320691.4) in the shifted DTM frame
(HiRISE DTM DTEEC_002888_2025_002176_2025, georeferencing shifted by (-107.1, -123.4) m
to the controlled-CTX frame). Window analysed: 1.51 x 1.51 km centred on R7
(1500 x 1500 px at 1.0086 m/px), valid DTM area 1.413 km^2 (61.7 % of the window;
the rest is off the DTM swath edge).

All outputs are in `outputs/revision/contact_3d/stage9_pits_R7/`. Scripts in
`scripts/` (random seed 42 where applicable; per-step parameters logged in
`params_*.json`). No input data were modified.

## 1. What was done

1. Closed-depression detection on the DTM window: border-seeded morphological
   reconstruction fill (`skimage.morphology.reconstruction`, erosion method);
   depth = fill - z; threshold depth >= 0.3 m; 8-connected components; components
   smaller than 3 px discarded (`scripts/01_detect_pits.py`).
2. Artifact checks: TIN-fill facet mask from a windowed |Laplacian| metric;
   2-D power spectrum of the detrended DEM and autocorrelation of the pit-centre
   point pattern; elevation profiles of representative pits; 0.25 m orthoimage
   brightness test (per-pit centre vs annulus, plus a 1,888-pit stack);
   CTX 6 m dark-spot correlation with a +-30 m shift search (scripts 02 to 07).
3. Spatial distribution: 100-m binned density map, KDE peak, nearest-neighbour
   statistics, membership in the mapped Clay/Olivine units from the gdb
   (scripts 08, 10).
4. Figures: overview, histograms, 4-panel zoom at the densest cluster, plus one
   diagnostic figure per artifact check.

## 2. Catalogue (`pits_catalog.csv`, n = 3,117)

| quantity | value |
|---|---|
| raw components (>= 0.3 m deep) | 5,204 (2,087 of 1-2 px discarded) |
| catalogued pits (>= 3 px) | 3,117 |
| overall density | 2,207 per km^2 |
| 100-m-bin density | median 1,573; p90 4,832; max 7,078 per km^2 at (1917, 1320893) |
| max depth | median 0.63 m; p90 1.21 m; max 7.37 m |
| equivalent diameter | median 3.2 m; p90 6.2 m; max 19.3 m (plus one 273-m enclosed basin, pit 567) |
| counts vs depth | >=0.5 m: 2,287; >=1 m: 542; >=2 m: 142; >=3 m: 101; >=5 m: 25 |
| nearest-neighbour spacing | mean 10.9 m; median 9.4 m |
| Clark-Evans ratio | 1.02 (indistinguishable from spatial randomness) |
| shape (3-20 m class) | minor/major axis median 0.58 (irregular, not circular) |

Depth and diameter histograms: `fig_histograms.png`. Overview map: `fig_overview.png`.
Zoom at the densest cluster: `fig_zoom_densest_cluster.png`.
