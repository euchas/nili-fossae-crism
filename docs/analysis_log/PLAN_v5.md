# PLAN_v5.md - VB parameter pruning + full-scene Brown RGB

Last updated 2026-04-27. Pre-launch; awaiting AV's approval on the
two open decisions in §4 below before any compute runs.

## 1. Scope

Targeted refinement of v4.1:

- Drop four VB parameters (SH600_2, HCPINDEX2, BDI1000VIS, LCPINDEX2)
  from the PCA + GMM input.
- Run two PCA + GMM passes — 7-param (v5-A) and 5-param (v5-B, drops
  BD530_2 + BD920_2) — as a sensitivity comparison.
- Regenerate Brown RGB composites for the full scene (no Brown eq. 2
  transparency mask).

Pixel masking is v4-style finite-only.

**Out of scope** (deferred to v6):
PC scene back-projections, GMM-class ratio spectra, lobe extraction,
column-neutral ratios, Brown 2020 grain-size lines on the
centroid-asymmetry hexbin.

## 2. Inputs - already on disk

- `outputs/_v4_vb_pca_brown_rgb/<cube>/arrays/<cube>_VB_stack_v4.npy`
  - reindex columns to v5-A and v5-B; do NOT recompute VB parameters
    from TER3.
- `outputs/_v4_vb_pca_brown_rgb/<cube>/arrays/<cube>_brown_*_v4.npy`
  for the full-scene Brown RGB regeneration.
  (Note: filename was `<cube>_vb_param_stack_v4.npy` in original
  draft - corrected to `<cube>_VB_stack_v4.npy` to match disk.)

## 3. Per-cube and cross-cube outputs

### Per-cube (4 cubes)

| File | Contents |
|---|---|
| `<cube>_v5a_PC_scores.npy` | PCA scores, v5-A stack, shape (n_valid, 5) |
| `<cube>_v5b_PC_scores.npy` | as above for v5-B |
| `<cube>_v5a_GMM_k6_labels.npy` | GMM cluster labels, v5-A; -1 for non-finite pixels |
| `<cube>_v5b_GMM_k6_labels.npy` | as above for v5-B |
| `<cube>_v5a_GMM_k6_proba.npy` | GMM soft-cluster posteriors, v5-A, shape (n_valid, 6) |
| `<cube>_v5b_GMM_k6_proba.npy` | as above for v5-B |
| `<cube>_v5a_loadings.csv` | PC loadings table |
| `<cube>_v5b_loadings.csv` | as above for v5-B |
| `<cube>_v5_pixel_counts.json` | finite-pixel counts per stack |
| `<cube>_GMM_k6_v5a_map.png` | spatial GMM map per cube (per-cube colour order; aligned version below) |
| `<cube>_GMM_k6_v5b_map.png` | as above for v5-B |
| `<cube>_Brown_RGB_centroid_amp_asym_v5_fullscene_v4p1stretch.png` | full-scene Brown RGB, v4.1 stretches |
| `<cube>_Brown_RGB_centroid_amp_asym_v5_fullscene_v5stretch.png` | full-scene Brown RGB, v5 4-cube p2/p98 |

### Cross-cube

| File | Contents |
|---|---|
| `comparison_Brown_RGB_centroid_amp_asym_4cube_v5_fullscene_v4p1stretch.png` | v5 analogue of the v4.1 cross-cube panel, v4.1 stretches |
| `comparison_Brown_RGB_centroid_amp_asym_4cube_v5_fullscene_v5stretch.png` | as above with full-scene stretches |
| `comparison_v5a_v5b_loadings.png` | PC1-PC3 loadings, 4 cubes x 2 stacks |
| `comparison_v5a_v5b_GMM_aligned.png` | confusion matrices per cube + Hungarian alignment v5-A vs v5-B |
| `comparison_GMM_k6_v5a_4cube_aligned.png` | aligned spatial GMM maps, 4 cubes side-by-side, v5-A |
| `comparison_GMM_k6_v5b_4cube_aligned.png` | as above for v5-B |
| `SUMMARY_v5.md` | one-pager: stack, finite-pixel counts, PC variance shares, GMM cluster sizes, headline observations + caveats |

## 4. Open decisions (resolve before launch)

### 4.1 Brown RGB stretch for full scene

**Recommendation**: produce both — v4.1-stretch (direct comparability)
AND v5-stretch (4-cube pooled p2/p98 on all finite-Brown pixels).
Marginal cost ~30 s wall-clock per pass.

### 4.2 Rationale for the four-parameter drop

Need AV's actual reason for DECISIONS.md. Currently a placeholder:
the v4 entry justified *adding* SH600_2 + HCPINDEX2; v5 reverses that
and the log shouldn't be ambiguous about why.

### 4.3 GMM k

**Recommendation**: keep k=6 (cross-version comparability). Re-sweep
in v6 if needed.

## 5. Risks / caveats

- **v5 vs v4 PC comparison is structural only** — same pixel set
  (finite-only), but different parameter basis, so PCs are different
  axes. Don't cross-plot v4 PC1 against v5 PC1 expecting a 1:1.
- **v5-B 5-param stack is rank-limited** — PC4-PC5 will share
  near-trivial variance shares. Read off PC1-PC3 only.
- **Hungarian alignment of v5-A vs v5-B GMM** is across two
  *different* feature spaces. The alignment is a label-correspondence
  heuristic, not a guarantee that "v5-A cluster k" and "v5-B cluster k"
  are the same population. Confusion-matrix diagonal dominance is the
  honest test.

## 6. Run order (post-approval)

1. Implement v5 stack functions in
   `code/stage_v5_param_pruning.py` — copy from v4 stage script; do
   not modify v4 in place.
2. Per-cube load (cached arrays from v4 snapshot) + finite-only
   masking + log finite-pixel counts. ~3 min total.
3. PCA + GMM k=6 per stack per cube (8 model fits total). ~10 min.
4. Cross-cube Hungarian alignment per stack on per-cube standardized
   centroids (same convention as v4 alignment, just on v5-A 7-param
   and v5-B 5-param spaces). ~2 min.
5. Per-cube and aligned cross-cube GMM maps. ~5 min.
6. Brown RGB regeneration full-scene, both stretch options. ~1 min.
7. Comparison figures + `SUMMARY_v5.md`. ~15 min.
8. Snapshot the run to `outputs/_v5_param_pruning/`.

Total wall clock: ~35-40 min on the workstation.

## 7. Pointers

- v4.1 baseline retained at `outputs/_v4_vb_pca_brown_rgb/`
- Pre-v4.1 stretches: `outputs/_v4_vb_pca_brown_rgb/_pre_v4p1_brown_restretch/`
- Updated DECISIONS.md "v5 (2026-04-27)" entry
- v5 snapshot: `outputs/_v5_param_pruning/`
- v6 deferred items: PC back-projections, ratio spectra, lobes,
  column-neutral ratios, Brown 2020 grain-size lines
