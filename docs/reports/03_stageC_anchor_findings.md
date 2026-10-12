# Stage C anchor findings

Two anchor tests from CONTEXT.md Section 4, plus the intrusion-geometry test motivated by AV's visual read of B438 and A4FC.

## gmm_pca15_k6_fixed

- Altered-olivine / intrusion candidate: aligned cluster **c5**
- Olivine-carbonate ridges candidate: aligned cluster **c5**

### Per-cube geometry (primary candidate clusters)
| cube | intr_pix | intr_frac | intr_n_comp | intr_med_aspect | olc_pix | olc_n_comp | olc_med_aspect |
|---|---:|---:|---:|---:|---:|---:|---:|
| FRT00003E12 | 16741 | 5.8% | 105 | 2.30 | 16741 | 105 | 2.30 |
| FRT0000871C | 4325 | 1.5% | 29 | 1.83 | 4325 | 29 | 1.83 |
| FRT0000A4FC | 2410 | 0.8% | 7 | 2.02 | 2410 | 7 | 2.02 |
| FRT0000B438 | 28455 | 9.9% | 119 | 2.24 | 28455 | 119 | 2.24 |

### All-cluster geometry (n_components, median aspect ratio)
| cube | c0 | c1 | c2 | c3 | c4 | c5 |
|---|---|---|---|---|---|---|
| FRT00003E12 | 77/1.99 | 219/2.40 | 129/2.14 | 82/2.10 | 45/2.12 | 105/2.30 |
| FRT0000871C | 40/1.77 | 83/2.10 | 151/1.92 | 129/2.07 | 81/1.94 | 29/1.83 |
| FRT0000A4FC | 80/1.83 | 123/1.82 | 122/1.96 | 63/1.94 | 155/1.87 | 7/2.02 |
| FRT0000B438 | 50/2.35 | 126/1.84 | 184/2.05 | 55/2.08 | 67/2.25 | 119/2.24 |

Reading: high `med_aspect` (>>2) = elongated components = fracture-like / dike-like.  Low `med_aspect` (~1) = blob-like / layered-deposit-like.
