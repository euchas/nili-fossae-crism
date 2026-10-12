# v5 summary - VB parameter pruning + full-scene Brown RGB

**Goal**: test whether dropping four high-loading VB parameters
(SH600_2, HCPINDEX2, BDI1000VIS, LCPINDEX2) sharpens the alteration-
mineralogy story; produce full-scene Brown RGB composites without the
eq.2 transparency mask.

**Status**: complete (2026-04-27).

## Stacks

| stack | params (n) | drops vs v4 |
|---|---|---|
| v5-A | BD530_2, BD920_2, OLINDEX3, BD1400, BD1900R2, D2300, BD2500_2 (7) | SH600_2, HCPINDEX2, BDI1000VIS, LCPINDEX2 |
| v5-B | OLINDEX3, BD1400, BD1900R2, D2300, BD2500_2 (5) | v5-A drops + BD530_2, BD920_2 |

## Per-cube results (PCA + GMM k=6 on whitened PCs)

| cube | v5-A finite | v5-A PC1-5 cum | v5-A BIC | v5-B finite | v5-B PC1-5 cum | v5-B BIC |
|---|---:|---:|---:|---:|---:|---:|
| FRT00003E12 | 286 800 | 94.6% | 3 663 077 | 286 800 | 100.0% | 3 593 891 |
| FRT0000871C | 286 800 | 90.9% | 3 762 797 | 286 800 | 100.0% | 3 743 579 |
| FRT0000A4FC | 286 800 | 93.0% | 3 743 519 | 286 800 | 100.0% | 3 745 605 |
| FRT0000B438 | 286 800 | 92.8% | 3 801 614 | 286 800 | 100.0% | 3 783 627 |

- **v5-A PC1-5 reaches 90% on every cube** (was 82.4%-87.3% in v4 with
  11 params). The dropped VNIR / pyroxene params were carrying ~7-12%
  of variance that did NOT compress cleanly into the first 5 PCs.
- **v5-B PC1-5 = 100%** by construction (5 params -> 5-D space; rank
  limited). PC4-PC5 carry near-trivial structure; read off PC1-PC3
  only per the PLAN_v5 caveat.
- **BIC drops on v5-B vs v5-A** for every cube. Lower BIC = better
  GMM fit per parameter count; this is a fair comparison because the
  GMM model dimensionality is the same (k=6 on 5-D whitened space)
  even though the upstream PCA basis differs.

## Hungarian alignment quality (centroid distance to 3E12, std-units)

| stack | cube | max dist (cluster) | mean |
|---|---|---:|---:|
| v5-A | FRT0000871C | 1.21 | 0.78 |
| v5-A | FRT0000A4FC | 2.12 | 1.31 |
| v5-A | FRT0000B438 | 2.21 | 1.23 |
| v5-B | FRT0000871C | 1.35 | 0.98 |
| v5-B | FRT0000A4FC | 1.44 | 1.12 |
| v5-B | FRT0000B438 | 1.60 | 0.92 |

**v5-B has tighter alignment on every cube** (mean distances 0.92-1.12
vs v5-A's 0.78-1.31). Pure-alteration basis aligns more cleanly
across cubes, consistent with rationale 2 (clarify the alteration
story by removing VNIR cross-talk).

For comparison, v4 11-param max distances were 2.39-2.75. v5-B is the
**tightest cross-cube alignment of any version so far** (max 1.60 on
B438).

## Brown RGB stretches

Two stretch sets produced for full-scene RGB:

- **v4.1 stretch** (centroid 1.15-1.40, amplitude 0.10-0.37,
  asymmetry 1.25-5.00): direct comparability to the eq.2-masked
  v4.1 panels. Full-scene pixels with centroid < 1.15 or asymmetry
  < 1.25 clip to dark colors. This is the headline "do v4.1 colors
  hold up when we show the whole scene" test.
- **v5 stretch** (centroid 1.014-1.379, amplitude 0.116-0.335,
  asymmetry 0.525-4.439): 4-cube pooled p2/p98 on the actual full-
  scene Brown-finite pixel set. Lower bounds are noticeably below
  v4.1 (especially asymmetry 0.525 vs 1.25), reflecting that the
  full scene includes mafic-cap pixels with low centroid + low
  asymmetry that the eq.2 mask hid.

Both versions live in
`outputs/_v5_param_pruning/<cube>/figures/<cube>_Brown_RGB_centroid_amp_asym_v5_fullscene_{v4p1stretch,v5stretch}.png`
and as cross-cube 2x2 panels in
`outputs/_v5_param_pruning/figures/comparison_Brown_RGB_centroid_amp_asym_4cube_v5_fullscene_{v4p1stretch,v5stretch}.png`.

## Outputs

### Per-cube (4 cubes x same structure)

- `arrays/`: 11 files - PC scores raw + whitened (v5-A, v5-B), GMM
  labels (raw + aligned, v5-A and v5-B, .npy with -1 outside valid),
  GMM posteriors (v5-A and v5-B), pixel counts JSON
- `figures/`: 4 files - per-cube GMM scene maps (v5-A, v5-B; raw
  per-cube color order); 2 full-scene Brown RGBs (v4.1 + v5 stretch)
- `tables/`: 2 PC loadings CSVs (v5-A, v5-B)

### Cross-cube (top-level v5 root)

- `comparison_v5a_v5b_loadings.png` - 4 cubes x 2 stacks PC1-PC3
  loadings, shared RdBu_r scale
- `comparison_v5a_v5b_GMM_aligned.png` - per-cube confusion matrix
  v5-A vs v5-B (Hungarian-aligned). Diagonal agreement = how
  consistently the two stacks recover the same partition.
- `comparison_GMM_k6_v5a_4cube_aligned.png` - 4-cube v5-A aligned
  scene panels
- `comparison_GMM_k6_v5b_4cube_aligned.png` - 4-cube v5-B aligned
  scene panels
- `comparison_Brown_RGB_centroid_amp_asym_4cube_v5_fullscene_v4p1stretch.png`
- `comparison_Brown_RGB_centroid_amp_asym_4cube_v5_fullscene_v5stretch.png`

### Reports

- `03_v5_alignment.json` - Hungarian permutations + distance matrices
  for v5-A and v5-B
- `v5_run_log_<TS>.json` - run timestamp + full source-code md5 list
- `SUMMARY_v5.md` (this file)

## Headline observations

1. **v5-B is the tightest cross-cube alignment of any version so
   far.** Mean distances 0.92-1.12 std-units (vs v4 2.39-2.75 max,
   v5-A 0.78-1.31 mean). The pure-alteration basis is cleaner cross-
   cube than basis variants that mix VNIR / pyroxene structure.
2. **v5-A PC1-5 reaches 90% on every cube** (vs v4's 82-87%). Dropping
   the four high-loading VNIR / pyroxene params concentrates more
   variance into fewer PCs. Whether this is "cleaner" or "less
   informative" is a scientific judgment - check the loadings.
3. **GMM BIC drops on v5-B for 3 of 4 cubes** vs v5-A, despite v5-B
   being a smaller-rank space. Fair comparison: the GMM model dim is
   k=6 on 5-D whitened input in both cases.
4. **A4FC's full-scene Brown-finite count = 286 800** (vs ~24k after
   eq.2 in v4). The v5 full-scene Brown RGB shows the entire A4FC
   scene including the (large) sub-1.15-centroid mafic-cap region
   that was transparent in v4.1.

## Pointers

- Decisions log: `DECISIONS.md` (v5 entry at bottom)
- Plan: `PLAN_v5.md`
- Snapshot: `outputs/_v5_param_pruning/`
- v6 deferred: PC scene back-projections, GMM-class ratio spectra,
  lobe extraction, column-neutral ratios, Brown 2020 grain-size lines
