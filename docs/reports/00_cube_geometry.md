# 00 — Cube geometry and metadata inventory

Generated from the four `.lbl` files in `cubes/*/`. No cube data was loaded for
this report (metadata only, as required by the setup task).

## Metadata available in the PDS3 `.lbl` files

| cube_id     | obs_date (UTC)         | orbit | Ls (deg) | solar_dist (km) | lines × samples | bands (total / valid) |
|-------------|------------------------|-------|----------|-----------------|-----------------|-----------------------|
| FRT00003E12 | 2007-01-13T00:13:51    | 2176  | 165.80   | 224 224 448     | 480 × 640       | 545 / 489             |
| FRT0000871C | 2007-10-26T16:22:59    | 5855  | 337.22   | 224 542 484     | 480 × 640       | 545 / 489             |
| FRT0000A4FC | 2008-03-08T06:08:31    | 7569  | 42.12    | 246 050 488     | 480 × 640       | 545 / 489             |
| FRT0000B438 | 2008-06-28T06:06:07    | 9006  | 91.34    | 247 646 851     | 480 × 640       | 545 / 489             |

All four cubes are TER3-processed joined products (PRODUCT_TYPE = TARGETED_RDR,
MRO:SENSOR_ID = J). Per the `.lbl`, all four have:

- `MRO:PHOTOMETRIC_CORR_FLAG    = ON`
- `MRO:ATMOSPHERIC_CORR_FLAG    = ON`
- `MRO:EMPIRICAL_SMILE_CORR_FLAG = ON`
- `MRO:SENSOR_SPACE_TRANSFORM_FLAG = ON`
- `MRO:RATIO_SHIFT_CORR_FLAG    = ON`
- `MRO:EMPIRICAL_GEOM_NORM_FLAG = ON`

The BBL (Bad Band List) in each `.hdr` flags 56 channels out of 545 as invalid
(typically detector-junction + 2.65–2.75 µm CO₂ blackout + VNIR/IR overlap).
All four cubes report the same valid-band count (489), so raw band indexing
is consistent across them.

## Metadata NOT in the `.lbl` files

The `.lbl` files do **not** contain the requested fields:

- center latitude / longitude
- solar incidence angle
- emission angle
- phase angle

These are per-pixel quantities stored in the associated DDR (Derived Data
Record) files (e.g. `*_DE166L_DDR1.IMG`, `*_DE166S_DDR1.IMG`) — referenced in
`SOURCE_PRODUCT_ID` inside each `.lbl` but **not** present in
`cubes/<CUBE_ID>_07_IF166J_TER3/`. There is also no scene-level summary row in
the `.lbl` (CRISM TER labels are unusually sparse — they carry only start/stop
time, Ls, solar distance, cube dimensions, and processing flags).

**Implications for Stage A**

- Cross-cube parameter distributions will differ not only from surface
  mineralogy but also from seasonal atmospheric state. The four cubes span
  Ls ∈ {42°, 91°, 166°, 337°} — i.e. early N. spring, early N. summer, late N.
  summer, and mid N. winter. FRT0000871C at Ls = 337° is close to dust-storm
  season; FRT00003E12 at Ls = 166° sits near aphelion with cold-trap water
  ice cloud activity possible over Nili Fossae. The project rules already mandate
  per-cube distributional comparison rather than absolute values; this table
  quantifies why.
- The four cubes were acquired over a 17-month window on four different MRO
  orbits (2176, 5855, 7569, 9006). Genuinely independent observations.

**Open question for AV (flagged in PLAN.md):** do you want me to recover
the missing geometry — center lat/lon, incidence, emission, phase — by
(a) reading the DDR `.IMG` files from PDS / your Dropbox if a copy exists,
(b) looking them up from CRISM PDS nightly-orbit tables,
or (c) leaving them out for Stage A and adding later if needed? Stage A
parameter computation does not strictly require them (TER3 has already been
photometrically + atmospherically corrected), but they are useful for
later seasonal/atmospheric-residual diagnostics.

## Data integrity — files present per cube

| cube_id     | .img | .hdr | .lbl | centroid_v5 | asymmetry_v5 | amplitude_v5 | hwhm_v5 | mask_v5 | residual_v5 |
|-------------|:----:|:----:|:----:|:-----------:|:------------:|:------------:|:-------:|:-------:|:-----------:|
| FRT00003E12 | yes  | yes  | yes  | yes         | yes          | yes          | yes     | yes     | yes         |
| FRT0000871C | yes  | yes  | yes  | yes         | yes          | yes          | yes     | yes     | yes         |
| FRT0000A4FC | yes  | yes  | yes  | yes         | yes          | yes          | yes     | yes     | yes         |
| FRT0000B438 | yes  | yes  | yes  | yes         | yes          | yes          | yes     | yes     | yes         |

`871C` additionally carries a stray `frt0000871c_07_in166j_ter3.hdr` file
(note `in` not `if` — this is an orphan ENVI header without a matching
`.img`, likely leftover from a separate L-only product). It does not
interfere with the standard `_if166j_ter3` glob.

No obvious mismatches; all four cubes report identical 480 × 640 × 545 cube
shape, matching the Brown v5 `.npy` maps (480 × 640) reported by the existing
`PCA_CRISM_analysis_v2.py`. Actual load + shape verification is deferred to
Stage A.
