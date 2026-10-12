"""Brown RGB composites with 4-cube-pooled p2-p98 stretch baked in.

For each MTR3 cube, write a 4-band uint8 RGBA GeoTIFF with the same
matched stretch across all four cubes. Same composite as the
unstretched run (R=centroid, G=amplitude, B=asymmetry), but linearly
mapped to [0, 255] using a single fixed range per band so that
identical color = identical parameter value across cubes when
mosaicked.

Stretches are 4-cube-pooled p2 to p98 from
brown_rgb_pixel_stats.py:
  R centroid:   1.022 -> 1.387
  G amplitude:  0.115 -> 0.334
  B asymmetry:  0.577 -> 4.548

Pixels where any input Brown param was NaN get alpha=0; all other
pixels get alpha=255. The alpha band cleanly separates fit-failure
(transparent) from real-data-that-happens-to-stretch-to-(0,0,0)
(opaque black). No nodata=0 trick - that loses legitimate low-end
pixels at the stretch tail.

No Eq.2 mask per project policy 2026-04-29.

Output: outputs/brown_rgb/stretched/
        <cube>_brown_RGB_centroid_amp_asym_stretched.tif
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

STRETCH = {
    # "Physical" stretches anchored on Brown 2020 reliability bounds:
    #   centroid  >= 1.15 um is the eq.2 reliability floor; 1.40 um is
    #              the upper bound on Fo-number-proxy centroid shifts.
    #   amplitude >= 0.10 is the eq.2 lower bound; 0.37 = data p98.
    #   hwhm      0.18 - 0.30 um covers all four cubes' span.
    #   asymmetry 1.25 - 5.00; >5 = saturation regime per Brown 2020.
    "centroid":  (1.150, 1.400),
    "amplitude": (0.100, 0.370),
    "asymmetry": (1.250, 5.000),
    "hwhm":      (0.180, 0.300),
}
COMPOSITES = {
    # Canonical recipe: R=centroid, G=amplitude, B=asymmetry.
    # Original Brown 2020-style channel order, restored after a
    # short experiment with the swapped (amp, centroid, asym)
    # variant. With physical stretches anchored on the eq.2 bounds,
    # olivine-rich pixels read as bright magenta-white and sub-1.15
    # centroid (illumination-tail) pixels clip to dark blue/cyan.
    "centroid_amp_asym":  ("centroid", "amplitude", "asymmetry"),
}
SNAPSHOT = OUTPUTS_DIR / "brown_rgb" / "stretched"


def stretch_band(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Linear [lo, hi] -> [0, 255] uint8, clipped, NaN -> 0."""
    out = (arr - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    out = (out * 255.0).astype(np.float32)
    out[~np.isfinite(arr)] = 0.0
    return out.astype(np.uint8)


def write_rgba_geotiff(rgba: np.ndarray, path: Path, geo: dict | None,
                        description: str) -> None:
    """rgba shape: (4, H, W) uint8. ColorInterp tagged R, G, B, alpha."""
    _, h, w = rgba.shape
    profile = {
        "driver": "GTiff", "dtype": "uint8",
        "width": w, "height": h, "count": 4,
        "crs": make_crs(geo), "transform": make_transform(geo),
        "compress": "deflate",
        "photometric": "rgb",
        "interleave": "pixel",
        "tiled": True, "blockxsize": 256, "blockysize": 256,
        "ALPHA": "YES",
    }
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(rgba, indexes=[1, 2, 3, 4])
        dst.colorinterp = [
            rasterio.enums.ColorInterp.red,
            rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.blue,
            rasterio.enums.ColorInterp.alpha,
        ]
        dst.update_tags(description=description)
        dst.set_band_description(1, "centroid (R)")
        dst.set_band_description(2, "amplitude (G)")
        dst.set_band_description(3, "asymmetry (B)")
        dst.set_band_description(4, "alpha (finite-mask)")


def process_cube(cube_id: str) -> dict:
    print(f"\n=== {cube_id} ===")
    dir_name = MTR3_CUBE_DIRS[cube_id]
    hdr_path = CUBES_DIR / dir_name / f"{dir_name}.hdr"
    geo = parse_envi_geo(hdr_path)

    # Load every band that any composite needs (deduped).
    needed = sorted({p for names in COMPOSITES.values() for p in names})
    bands_raw = {p: load_brown_param(cube_id, p) for p in needed}
    H, W = bands_raw[needed[0]].shape

    # Brown fits succeed or fail jointly -> single alpha applies to
    # every composite (built from union of needed bands).
    alpha_bool = np.ones((H, W), dtype=bool)
    for p in needed:
        alpha_bool &= np.isfinite(bands_raw[p])
    alpha = (alpha_bool.astype(np.uint8) * 255)

    out_dir = SNAPSHOT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_files = []
    for tag, names in COMPOSITES.items():
        rgba = np.zeros((4, H, W), dtype=np.uint8)
        for i, p in enumerate(names):
            lo, hi = STRETCH[p]
            rgba[i] = stretch_band(bands_raw[p], lo, hi)
        rgba[3] = alpha
        out_path = out_dir / f"{cube_id}_brown_RGB_{tag}_stretched.tif"
        desc = (f"Brown RGB R={names[0]} G={names[1]} B={names[2]}, "
                f"4-cube-pooled p2-p98 stretch: "
                f"R[{STRETCH[names[0]][0]:.3f},{STRETCH[names[0]][1]:.3f}] "
                f"G[{STRETCH[names[1]][0]:.3f},{STRETCH[names[1]][1]:.3f}] "
                f"B[{STRETCH[names[2]][0]:.3f},{STRETCH[names[2]][1]:.3f}]; "
                "alpha = finite-mask. No Eq.2 mask.")
        write_rgba_geotiff(rgba, out_path, geo, desc)
        out_files.append(out_path.name)
        print(f"  wrote {out_path.name}  (4-band uint8 RGBA)")
    print(f"  alpha-finite pixels: {int(alpha_bool.sum())}/{H*W}")
    return {
        "cube_id": cube_id,
        "shape": [H, W],
        "n_alpha_finite": int(alpha_bool.sum()),
        "out_files": out_files,
        "composites": {tag: list(names) for tag, names in COMPOSITES.items()},
    }


def main(argv: list[str]) -> None:
    t0 = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    targets = list(CUBE_IDS) if len(argv) <= 1 else argv[1:]
    bad = [t for t in targets if t not in CUBE_IDS]
    if bad:
        raise SystemExit(f"unknown cube id(s): {bad}; known: {CUBE_IDS}")

    summaries = {}
    for cube_id in targets:
        summaries[cube_id] = process_cube(cube_id)

    log = {
        "stage": "brown_rgb_stretched",
        "timestamp_utc": timestamp,
        "snapshot": str(SNAPSHOT),
        "stretch_per_band": {p: list(v) for p, v in STRETCH.items()},
        "stretch_basis": "4-cube-pooled p2-p98 from brown_rgb_pixel_stats.py",
        "composites": {tag: list(names) for tag, names in COMPOSITES.items()},
        "encoding": ("4-band uint8 RGBA, photometric=rgb, alpha = "
                     "finite-mask of all 3 input bands; no Eq.2 mask. "
                     "Alpha (not nodata=0) used so real pixels at the "
                     "stretch low-end stay visible."),
        "per_cube": summaries,
        "wall_seconds": round(time.time() - t0, 2),
    }
    log_path = SNAPSHOT / f"brown_rgb_stretched_log_{timestamp}.json"
    log_path.write_text(json.dumps(log, indent=2, default=str),
                        encoding="utf-8")
    print(f"\nwrote {log_path}")
    print(f"done in {round(time.time()-t0, 2)}s; output at {SNAPSHOT}")


if __name__ == "__main__":
    main(sys.argv)
