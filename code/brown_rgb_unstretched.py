"""Brown parameter RGB composites - unstretched float32 GeoTIFFs.

For each MTR3 cube write two 3-band GeoTIFFs:
  *_brown_RGB_centroid_amp_hwhm.tif   bands = [centroid, amplitude, hwhm]
  *_brown_RGB_centroid_amp_asym.tif   bands = [centroid, amplitude, asymmetry]

Each band carries the raw Brown v6 fit value with NaN as NoData. No
per-band stretch, no Eq.2 mask. ArcGIS Pro applies user-controlled
stretches at display time. Per project policy 2026-04-29: no Brown
2020 Eq.2 gating.

Output: outputs/_brown_rgb_unstretched/<cube>/...
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from config import CUBE_IDS, OUTPUTS_DIR
from stage_v7c_mtr3_pipeline import MTR3_CUBE_DIRS, CUBES_DIR, parse_envi_geo

BROWN_PARAMS = ("centroid", "amplitude", "hwhm", "asymmetry")
COMPOSITES = {
    # Original recipe: olivine reads bright magenta-white because all
    # three Brown bands are HIGH for olivine.
    "centroid_amp_hwhm": ("centroid", "amplitude", "hwhm"),
    "centroid_amp_asym": ("centroid", "amplitude", "asymmetry"),
    # Olivine-cool recipe: HWHM is LOW for olivine (sharp 1 um band)
    # so it goes in R; centroid (HIGH for olivine) goes in G; asymmetry
    # (HIGH for olivine) goes in B. Olivine-rich pixels read CYAN
    # (green+blue), altered/diffuse-fit pixels read RED. Closest you
    # can get to "olivine = green" using only Brown params.
    "hwhm_centroid_asym": ("hwhm", "centroid", "asymmetry"),
}
SNAPSHOT = OUTPUTS_DIR / "brown_rgb" / "unstretched"


def load_brown_param(cube_id: str, name: str) -> np.ndarray:
    dir_name = MTR3_CUBE_DIRS[cube_id]
    path = CUBES_DIR / dir_name / f"{dir_name}_{name}_map_v6.npy"
    if not path.exists():
        raise SystemExit(f"missing Brown v6 .npy: {path}")
    return np.load(path).astype(np.float32)


def make_transform(geo: dict | None) -> Affine:
    if geo and geo.get("geotransform"):
        gt = geo["geotransform"]
        return Affine(gt[1], gt[2], gt[0], gt[4], gt[5], gt[3])
    return Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)


def make_crs(geo: dict | None):
    if geo and geo.get("wkt"):
        try:
            return CRS.from_wkt(geo["wkt"])
        except Exception:
            return None
    return None


def write_multiband_float(stack: np.ndarray, band_names: tuple[str, ...],
                           path: Path, geo: dict | None,
                           description: str) -> None:
    """stack shape: (n_bands, H, W) float32. NoData = NaN."""
    n_bands, h, w = stack.shape
    profile = {
        "driver": "GTiff", "dtype": "float32",
        "width": w, "height": h, "count": n_bands,
        "crs": make_crs(geo), "transform": make_transform(geo),
        "nodata": float("nan"), "compress": "deflate",
        "photometric": "minisblack",
        "interleave": "pixel",
    }
    with rasterio.open(str(path), "w", **profile) as dst:
        for i in range(n_bands):
            dst.write(stack[i], i + 1)
            dst.set_band_description(i + 1, band_names[i])
        dst.update_tags(description=description)


def process_cube(cube_id: str) -> dict:
    print(f"\n=== {cube_id} ===")
    dir_name = MTR3_CUBE_DIRS[cube_id]
    hdr_path = CUBES_DIR / dir_name / f"{dir_name}.hdr"
    geo = parse_envi_geo(hdr_path)

    bands = {p: load_brown_param(cube_id, p) for p in BROWN_PARAMS}
    H, W = bands["centroid"].shape
    summary = {"cube_id": cube_id, "shape": [H, W], "geo_proj_name":
               geo["proj_name"] if geo else None, "band_stats": {}}
    for p, arr in bands.items():
        finite = np.isfinite(arr)
        n_fin = int(finite.sum())
        if n_fin > 0:
            summary["band_stats"][p] = {
                "n_finite": n_fin,
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "p2": float(np.nanpercentile(arr, 2)),
                "p98": float(np.nanpercentile(arr, 98)),
                "median": float(np.nanmedian(arr)),
            }
        print(f"  {p}: shape {arr.shape}, n_finite={n_fin}, "
              f"range [{summary['band_stats'][p]['min']:.4f}, "
              f"{summary['band_stats'][p]['max']:.4f}], "
              f"p2/p98 [{summary['band_stats'][p]['p2']:.4f}, "
              f"{summary['band_stats'][p]['p98']:.4f}]")

    out_dir = SNAPSHOT
    out_dir.mkdir(parents=True, exist_ok=True)

    composite_files = []
    for tag, names in COMPOSITES.items():
        stack = np.stack([bands[n] for n in names], axis=0)
        out_path = out_dir / f"{cube_id}_brown_RGB_{tag}.tif"
        desc = (f"Brown v6 RGB composite (UNSTRETCHED float32) - "
                f"R={names[0]} G={names[1]} B={names[2]}; NoData=NaN; "
                "no Eq.2 mask. Apply per-band stretch at display.")
        write_multiband_float(stack, names, out_path, geo, desc)
        composite_files.append(out_path.name)
        print(f"  wrote {out_path.name}  ({stack.shape}, float32)")

    summary["composite_files"] = composite_files
    summary["composites"] = {tag: list(names) for tag, names in COMPOSITES.items()}
    return summary


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
        "stage": "brown_rgb_unstretched",
        "timestamp_utc": timestamp,
        "snapshot": str(SNAPSHOT),
        "composites": {tag: list(names) for tag, names in COMPOSITES.items()},
        "encoding": ("3-band float32, NoData=NaN, no per-band stretch, "
                     "no Brown Eq.2 mask. ArcGIS user applies stretch."),
        "per_cube": summaries,
        "wall_seconds": round(time.time() - t0, 2),
    }
    log_path = SNAPSHOT / f"brown_rgb_unstretched_log_{timestamp}.json"
    log_path.write_text(json.dumps(log, indent=2, default=str),
                        encoding="utf-8")
    print(f"\nwrote {log_path}")
    print(f"done in {round(time.time()-t0, 2)}s; output at {SNAPSHOT}")


if __name__ == "__main__":
    main(sys.argv)
