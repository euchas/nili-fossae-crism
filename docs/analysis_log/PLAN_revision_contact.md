# PLAN — revision test: is olivine alteration concentrated at the clay contact?

Date: 2026-08-28. Motivation: during revision the question arose whether
carbonate/alteration in the olivine unit is pervasive rather than
concentrated at the contact with other units, which would weaken the
contact-related alteration inference. The manuscript (Sec. 5.1) asserted
contact concentration, citing Viviano et al. 2013 but not our own data. This
is a direct, cheap, quantitative test.

## Question

Among olivine-bearing pixels, do carbonate/hydration indicators vary with
distance from the nearest Fe/Mg-phyllosilicate pixel? Control: the same
indicators vs. distance from the mafic-cap contact (an equally sharp
contact with no water-bearing host). Bonus: the same indicators vs. MOLA
elevation from the CRISM DDR (Mandon 2020's argument is vertical).

## Inputs (all map-projected MTR3, 18 m/px, no resampling)

- Class labels: `outputs/_v9_1_per_cube_scaler/v9_1b_7param/<CUBE>/arrays/
  <CUBE>_v9_1b_gmm_labels.npy` (argmax of the 3E12-fit 7-param GMM;
  c0 olivine-carbonate, c1 mafic cap, c2/c3 olivine, c4/c5 phyllosilicate).
  `_novel_mask.npy` used for a robustness re-run.
- Parameters: `outputs/_v8_mtr3_with_brown/<CUBE>/arrays/<CUBE>_v8_param_stack.npy`
  (centroid, asymmetry, OLINDEX3, BD1400, BD1900R2, D2300, BD2500_2).
- Elevation: CRISM DDR `cubes/<cube>_07_if166j_mtr3/<cube>_07_de166l_mtr1.img`
  band 10 (MOLA elevation), same grid.

## Method

1. Masks: clay = c4|c5; cap = c1; olivine-bearing = c0|c2|c3; carbonate = c0.
2. Euclidean distance transform (scipy, sampling 18 m) to nearest clay pixel
   and to nearest cap pixel; distance to nearest invalid pixel used to drop
   olivine pixels within 90 m of the scene edge / nodata.
3. Bin olivine-bearing pixels by distance (18, 36, 54, 90, 150, 250, 400,
   700, 1200, >1200 m). Per bin: n, fraction of c0, median and IQR of
   BD2500_2, D2300, BD1900R2, OLINDEX3, centroid, asymmetry.
   Bins <= 36 m are flagged as sub-pixel-mixing-affected and not used for
   inference.
4. Clay-distance profile uses olivine pixels > 90 m from the cap; cap-distance
   control uses olivine pixels > 90 m from clay.
5. Statistics: Spearman rho (distance vs. BD2500_2, D2300) on pixels 36-1200 m;
   near (36-150 m) vs far (>400 m) c0-fraction ratio with Wilson 95% CI.
6. Elevation: c0 fraction and BD2500_2 in 25 m elevation bins per cube.
7. Robustness: repeat 3-5 excluding novel pixels; repeat with a GMM-free
   olivine definition (OLINDEX3 above the per-cube 60th percentile).

## Outputs

`outputs/revision/distance_to_contact/`: CSV of binned stats, JSON run log
(parameters, input md5s), `fig_distance_profiles.{png,pdf}`,
`fig_contact_maps.{png,pdf}`, `fig_elevation.{png,pdf}`, `REPORT.md`.

## What it can and cannot show

Can: whether alteration indicators are spatially structured relative to the
clay contact in four independent observations, beyond the 1-2 px mixing zone.
Cannot: temperature or timing. A contact-concentrated signal is consistent
with both syn-emplacement (heat + basement water) and post-emplacement
groundwater ponding on the clay aquiclude. A flat profile would argue for
pervasive alteration.
