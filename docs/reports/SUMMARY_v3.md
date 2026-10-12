# v3 PC deep-dive summary

**Goal**: drill into per-cube PC structure to surface minor phases
(carbonate, ferric, redox) that PC1 (olivine, universal) and PC2
(Fe/Mg smectite, universal) don't discriminate. Built on the v2
full-range PCA / GMM stack.

**Status**: complete (2026-04-22).

## What was produced

- 200 per-cube figures + 3 cross-cube figures (203 total `_v3` PNGs)
- 4 cluster-diagnostics CSVs (k=6 GMM)
- 20 new GMM cluster-label arrays (whitened k=6/8/10/12 + targeted
  minor-PC GMM per cube)
- 5 run logs with mode-shred warnings, BIC curves, and
  carbonate/ferric candidate flags
- Snapshot at `outputs/_v3_pc_deep_dive/`

## Headline findings

### 1. k=6 GMM is too coarse for minor-phase work
On every one of the four cubes, the top-5% and bottom-5% of PC3, PC4,
or PC5 spans **>=3 GMM classes** under the standard k=6 fit. The
mode-shred-warning count per cube is uniformly 6 (one for each PC's
high or low extreme that gets shredded). This is the quantitative
confirmation that PC1 vs PC2 alone misses the minor-phase signal -
exactly the project specification premise.

### 2. Higher-k GMM isolates carbonate + ferric clusters

| cube | k=8 carbonate | k=8 ferric | k=10 carbonate | k=10 ferric | k=12 carbonate | k=12 ferric |
|---|---|---|---|---|---|---|
| FRT00003E12 | c5 | - | c8 | c0 | c9 (+ferric) | c9, c10 |
| FRT0000871C | c2 | c1 | c2 | c7 | c7 | c9 |
| FRT0000A4FC | c0 | c7 | c0 | c5, c9 | c11 | c1, c7, c9 |
| FRT0000B438 | c2 | c4 | c5 | c6 | c6 | c5 |

A cluster is flagged as carbonate-candidate when its mean BD2500_2 is
> scene mean + 1 sd; ferric-candidate when mean BD530_2 is > scene mean
+ 1 sd. These flags match the per-cube PC3-5 predictions AV laid
out.

### 3. Targeted minor-PC GMM (Section 4c) is the cleanest signal
- **B438** GMM on {PC3, PC4, PC5}, k=4: bulk + redox + carbonate +
  ferric, cleanly separated.
- **A4FC** GMM on {PC3, PC5}, k=4 (PC4 skipped because it is
  band-width, not compositional): carbonate + ferric + redox + bulk.
- **871C** GMM on {PC3, PC4, PC5}, k=4: bulk + carbonate + ferric +
  redox.
- **3E12** GMM on {PC3, PC4, PC5}, k=5: ferric-carbonate coupling
  generates one extra intermediate class.

See `<cube>_gmm_minorPC_v3.png` for spatial maps overlaid on BD2500_2
(carbonate) and BD530_2 (ferric) reference parameters.

### 4. Per-cube PC1-5 explained variance (v2 full-range)

| cube | PC1 | PC2 | PC3 | PC4 | PC5 | cum-5 |
|---|---:|---:|---:|---:|---:|---:|
| FRT00003E12 | 46.0% | 23.2% | 8.2% | 6.8% | 6.2% | 90.4% |
| FRT0000871C | 35.2% | 23.2% | 9.7% | 9.2% | 6.2% | 83.5% |
| FRT0000A4FC | 34.0% | 25.1% | 10.4% | 8.2% | 6.9% | 84.6% |
| FRT0000B438 | 46.2% | 18.4% | 9.2% | 7.2% | 5.8% | 86.7% |

Cubes with smaller PC1 (871C, A4FC) have correspondingly more variance
in PC3-5, which is why those cubes show stronger minor-phase
structure on the targeted PC GMM.

### 5. Lobe spectra extracted from TER3 (atmospherically corrected)
Per the project specification the spectra are pulled directly from TER3 cubes via the
PCA score map (row, col) - no re-projection or VS correction needed.
Annotated with 1.0 / 1.4 / 1.9 / 2.3 / 2.5 um absorption guides and
shaded residual CO2 region (1.95-2.05 um) since VS correction is
never perfect.

Lobe D for A4FC, 871C, 3E12 was defined on the PC pair that shows the
cleanest minor-phase separation per the project specification: PC3 vs PC5 for A4FC
and 871C; PC3 vs PC4 for 3E12. To refine after visual inspection,
edit `LOBE_DEFS` in `code/stage_v3_common.py` and rerun
`stage_v3_per_cube.py <cube>` (or just Section 6).

## What's still open

- **Lobe boundary refinement**: starting bounds were the v3 spec
  values; visual inspection of the new pair plots may justify
  tightening or relocating Lobe D for A4FC / 871C / 3E12.
- **Column-neutral ratios** (deferred to v3.1 per spec).
- **Minor-PC interpretation per individual cluster** (e.g.
  3E12 k=12 c9 carbonate+ferric co-association) needs AV's eye on the
  spatial maps to confirm it's a real morphological feature vs a
  statistical artefact.

## Pointers

- Per-cube PC interpretations + lobe definitions: `code/stage_v3_common.py`
- Per-cube driver: `code/stage_v3_per_cube.py`
- Cross-cube driver: `code/stage_v3_cross_cube.py`
- Snapshot: `outputs/_v3_pc_deep_dive/` (with its own README)
- Raw run logs: `outputs/logs/<cube>_v3_per_cube_*.json` (mode-shred
  warnings + BIC curves + candidate flags)
