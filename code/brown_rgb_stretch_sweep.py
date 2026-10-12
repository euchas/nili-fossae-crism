"""Brown RGB stretch sweep - 3 stretch sets x 2 composites x 4 cubes.

Produces 24 RGBA GeoTIFFs at outputs/brown_rgb/stretched/stretch_sweep/
so the choice of stretch range can be evaluated visually side-by-side.

Stretches:
  p5p95     4-cube-pooled p5  / p95  - saturates ~10% of pixels
  p10p90    4-cube-pooled p10 / p90  - saturates ~20%, more contrast
  physical  matches v4.1 literature-motivated bounds:
              centroid  1.15 -> 1.40  (Brown 2020 reliability)
              amplitude 0.10 -> 0.37  (Brown 2020 eq.2 floor at 0.10)
              hwhm      0.18 -> 0.30  (covers all cubes' span)
              asymmetry 1.25 -> 5.00  (Brown 2020 saturation regime)

Composites:
  centroid_amp_asym   R=centroid G=amplitude B=asymmetry  -> olivine = bright magenta
  hwhm_centroid_asym  R=hwhm     G=centroid  B=asymmetry  -> olivine = cyan

Filenames: <cube>_brown_RGB_<composite>_<stretch>.tif
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio

from config import CUBE_IDS, OUTPUTS_DIR
from stage_v7c_mtr3_pipeline import MTR3_CUBE_DIRS, CUBES_DIR, parse_envi_geo
from brown_rgb_unstretched import load_brown_param, make_transform, make_crs
from brown_rgb_stretched import stretch_band, write_rgba_geotiff

# 4-cube-pooled percentiles (from brown_rgb_pixel_stats.py output)
STRETCH_SETS = {
    "p5p95": {
        "centroid":  (1.040, 1.371),
        "amplitude": (0.123, 0.309),
        "asymmetry": (0.667, 4.201),
        "hwhm":      (0.196, 0.309),
    },
    "p10p90": {
        "centroid":  (1.055, 1.349),
        "amplitude": (0.132, 0.279),
        "asymmetry": (0.755, 3.762),
        "hwhm":      (0.206, 0.299),
    },
    "physical": {
        "centroid":  (1.150, 1.400),
        "amplitude": (0.100, 0.370),
        "asymmetry": (1.250, 5.000),
        "hwhm":      (0.180, 0.300),
    },
}

COMPOSITES = {
    "centroid_amp_asym":  ("centroid", "amplitude", "asymmetry"),
    "hwhm_centroid_asym": ("hwhm",     "centroid",  "asymmetry"),
}

SNAPSHOT = OUTPUTS_DIR / "brown_rgb" / "stretched" / "stretch_sweep"


def process_cube(cube_id: str) -> dict:
    print(f"\n=== {cube_id} ===")
    dir_name = MTR3_CUBE_DIRS[cube_id]
    hdr_path = CUBES_DIR / dir_name / f"{dir_name}.hdr"
    geo = parse_envi_geo(hdr_path)

    needed = sorted({p for names in COMPOSITES.values() for p in names})
    bands_raw = {p: load_brown_param(cube_id, p) for p in needed}
    H, W = bands_raw[needed[0]].shape

    alpha_bool = np.ones((H, W), dtype=bool)
    for p in needed:
        alpha_bool &= np.isfinite(bands_raw[p])
    alpha = (alpha_bool.astype(np.uint8) * 255)

    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    out_files = []
    for stretch_name, stretch in STRETCH_SETS.items():
        for comp_name, names in COMPOSITES.items():
            rgba = np.zeros((4, H, W), dtype=np.uint8)
            for i, p in enumerate(names):
                lo, hi = stretch[p]
                rgba[i] = stretch_band(bands_raw[p], lo, hi)
            rgba[3] = alpha
            out_path = (SNAPSHOT
                        / f"{cube_id}_brown_RGB_{comp_name}_{stretch_name}.tif")
            desc = (f"Brown RGB {comp_name} ({stretch_name} stretch): "
                    f"R={names[0]}[{stretch[names[0]][0]:.3f},"
                    f"{stretch[names[0]][1]:.3f}] "
                    f"G={names[1]}[{stretch[names[1]][0]:.3f},"
                    f"{stretch[names[1]][1]:.3f}] "
                    f"B={names[2]}[{stretch[names[2]][0]:.3f},"
                    f"{stretch[names[2]][1]:.3f}]; "
                    "alpha = finite-mask, no Eq.2 mask.")
            write_rgba_geotiff(rgba, out_path, geo, desc)
            out_files.append(out_path.name)
    print(f"  alpha-finite pixels: {int(alpha_bool.sum())}/{H*W}; "
          f"wrote {len(out_files)} files")
    return {"cube_id": cube_id, "shape": [H, W],
            "n_alpha_finite": int(alpha_bool.sum()),
            "out_files": out_files}


def main(argv: list[str]) -> None:
    t0 = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    targets = list(CUBE_IDS) if len(argv) <= 1 else argv[1:]

    summaries = {}
    for cube_id in targets:
        summaries[cube_id] = process_cube(cube_id)

    log = {
        "stage": "brown_rgb_stretch_sweep",
        "timestamp_utc": timestamp,
        "snapshot": str(SNAPSHOT),
        "stretch_sets": STRETCH_SETS,
        "composites": {tag: list(names) for tag, names in COMPOSITES.items()},
        "filename_pattern": "<cube>_brown_RGB_<composite>_<stretch>.tif",
        "encoding": ("4-band uint8 RGBA, photometric=rgb, ALPHA=YES, "
                     "alpha = finite-mask of all 3 input bands; no Eq.2 "
                     "mask. Real low-end pixels stay opaque (visible)."),
        "per_cube": summaries,
        "wall_seconds": round(time.time() - t0, 2),
    }
    log_path = SNAPSHOT / f"brown_rgb_stretch_sweep_log_{timestamp}.json"
    log_path.write_text(json.dumps(log, indent=2, default=str),
                        encoding="utf-8")
    print(f"\nwrote {log_path}")
    print(f"done in {round(time.time()-t0, 2)}s; output at {SNAPSHOT}")


if __name__ == "__main__":
    main(sys.argv)
