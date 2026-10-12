"""v7 - Generate Brown RGB GeoTIFF composites from v6 MTR3 Brown maps.

Reads the v6 Brown .npy files written by 1um_centroid_analysis_CRISM_MTR3_v1.py,
applies matched-stretch normalization, and writes two 3-band RGBA GeoTIFFs
per cube (composite A: centroid/amplitude/hwhm; composite B: centroid/
amplitude/asymmetry) carrying the MTR3 projection from the source ENVI
header. Drop-in for ArcGIS Pro / QGIS.

Run: python code/v7_brown_rgb_geotiff.py
     python code/v7_brown_rgb_geotiff.py FRT00003E12   # one cube
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from config import CUBES_DIR, OUTPUTS_DIR

SNAPSHOT_DIR = OUTPUTS_DIR / "_v7_mtr3_map_projected"
VERSION = "v6"

# Cube IDs to process (lowercase MTR3 dir form)
MTR3_CUBE_DIRS = {
    "FRT00003E12": "frt00003e12_07_if166j_mtr3",
    "FRT0000871C": "frt0000871c_07_if166j_mtr3",
    "FRT0000A4FC": "frt0000a4fc_07_if166j_mtr3",
    "FRT0000B438": "frt0000b438_07_if166j_mtr3",
}

# Full-range stretches: 4-cube-pooled p2/p98 across all v6 MTR3
# Brown-finite pixels (no Brown Eq.2 gating per project policy
# 2026-04-29: "no Brown thresholding from now on; full range").
BROWN_STRETCH = {
    "centroid": (1.02, 1.39),
    "amplitude": (0.115, 0.334),
    "hwhm": (0.188, 0.320),
    "asymmetry": (0.58, 4.55),
}

# Two RGB composites per cube
COMPOSITES = {
    "centroid_amp_hwhm": ("centroid", "amplitude", "hwhm"),
    "centroid_amp_asym": ("centroid", "amplitude", "asymmetry"),
}


def parse_envi_geo(hdr_path: Path) -> dict | None:
    """Extract geotransform + WKT from an ENVI .hdr file."""
    content = hdr_path.read_text()
    map_match = re.search(r'map\s+info\s*=\s*\{([^}]+)\}',
                          content, re.IGNORECASE | re.DOTALL)
    if not map_match:
        return None
    parts = [p.strip() for p in map_match.group(1).split(',')]
    if len(parts) < 7:
        return None
    try:
        proj_name = parts[0]
        x_ref = float(parts[1]); y_ref = float(parts[2])
        x_coord = float(parts[3]); y_coord = float(parts[4])
        x_pix = float(parts[5]); y_pix = float(parts[6])
    except (ValueError, IndexError):
        return None
    x_origin = x_coord - (x_ref - 0.5) * x_pix
    y_origin = y_coord + (y_ref - 0.5) * y_pix
    geotransform = (x_origin, x_pix, 0.0, y_origin, 0.0, -y_pix)
    cs_match = re.search(r'coordinate\s+system\s+string\s*=\s*\{([^}]+)\}',
                         content, re.IGNORECASE | re.DOTALL)
    wkt = cs_match.group(1).strip() if cs_match else None
    return {"proj_name": proj_name, "geotransform": geotransform,
            "wkt": wkt, "x_pixel_size": x_pix, "y_pixel_size": y_pix}


def percentile_norm(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Stretch to [0, 1] using fixed range. NaN passes through."""
    out = np.full_like(arr, np.nan, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any() or hi - lo < 1e-9:
        out[finite] = 0.5
        return out
    out[finite] = np.clip((arr[finite] - lo) / (hi - lo), 0, 1)
    return out


def build_rgba_uint8(maps: dict, mask: np.ndarray,
                     channels: tuple) -> np.ndarray:
    """Build (4, H, W) RGBA uint8 array. Alpha=255 where mask is True."""
    h, w = mask.shape
    rgba = np.zeros((4, h, w), dtype=np.uint8)
    for i, name in enumerate(channels):
        lo, hi = BROWN_STRETCH[name]
        chan = percentile_norm(np.where(mask, maps[name], np.nan), lo, hi)
        chan_u8 = np.where(np.isfinite(chan), (chan * 255).astype(np.uint8), 0)
        rgba[i] = chan_u8
    rgba[3] = (mask & np.isfinite(maps[channels[0]])
               & np.isfinite(maps[channels[1]])
               & np.isfinite(maps[channels[2]])).astype(np.uint8) * 255
    return rgba


def write_rgba_geotiff(rgba: np.ndarray, out_path: Path,
                       geo: dict | None, description: str = "") -> None:
    """Write a 4-band RGBA GeoTIFF (uint8, deflate compressed)."""
    n_bands, h, w = rgba.shape
    if geo and geo.get("geotransform"):
        gt = geo["geotransform"]
        transform = Affine(gt[1], gt[2], gt[0], gt[4], gt[5], gt[3])
    else:
        transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
    crs = None
    if geo and geo.get("wkt"):
        try:
            crs = CRS.from_wkt(geo["wkt"])
        except Exception:
            crs = None
    profile = {
        "driver": "GTiff", "dtype": "uint8",
        "width": w, "height": h, "count": n_bands,
        "crs": crs, "transform": transform,
        "compress": "deflate", "photometric": "RGB",
        "alpha": "yes",
    }
    with rasterio.open(str(out_path), "w", **profile) as dst:
        for i in range(n_bands):
            dst.write(rgba[i], i + 1)
        dst.set_band_description(1, "R: " + description.split(" / ")[0])
        dst.set_band_description(2, "G: " + description.split(" / ")[1])
        dst.set_band_description(3, "B: " + description.split(" / ")[2])
        dst.set_band_description(4, "Alpha (Brown 2020 eq.2 mask)")
        dst.update_tags(STRETCH_DESCRIPTION=description)


def process_cube(cube_id: str) -> dict:
    """Generate the two RGB GeoTIFFs for one cube. Returns metadata."""
    dir_name = MTR3_CUBE_DIRS[cube_id]
    cube_dir = CUBES_DIR / dir_name
    base = dir_name  # e.g. "frt00003e12_07_if166j_mtr3"
    print(f"\n=== {cube_id} ({base}) ===")

    # Find Brown v6 .npy files
    npy_paths = {}
    for param in ("centroid", "amplitude", "hwhm", "asymmetry"):
        p = cube_dir / f"{base}_{param}_map_{VERSION}.npy"
        if not p.exists():
            print(f"  MISSING {p.name}")
            return {"cube_id": cube_id, "status": "missing_inputs",
                    "missing": str(p)}
        npy_paths[param] = p

    print(f"  loading v6 .npy maps...")
    maps = {p: np.load(npy_paths[p]) for p in
            ("centroid", "amplitude", "hwhm", "asymmetry")}
    # Full-range alpha: any pixel with finite Brown 4 (NO eq.2 gating).
    mask = (np.isfinite(maps["centroid"]) & np.isfinite(maps["amplitude"])
            & np.isfinite(maps["hwhm"]) & np.isfinite(maps["asymmetry"]))
    print(f"  shape: {mask.shape}, finite Brown 4: {int(mask.sum())} / {mask.size} "
          f"({100 * mask.sum() / mask.size:.1f}%) [no Eq.2 gating]")

    # Parse projection from the source ENVI header
    hdr = cube_dir / f"{base}.hdr"
    geo = parse_envi_geo(hdr)
    if geo:
        print(f"  proj: {geo['proj_name']}, "
              f"pixel: {geo['x_pixel_size']:.6f} x {geo['y_pixel_size']:.6f}")
    else:
        print(f"  WARNING no map info in {hdr.name}; pixel-coords only")

    # Snapshot dirs
    snap_brown = SNAPSHOT_DIR / cube_id / "brown"
    snap_brown.mkdir(parents=True, exist_ok=True)

    written = []
    for tag, channels in COMPOSITES.items():
        rgba = build_rgba_uint8(maps, mask, channels)
        desc = " / ".join(
            f"{c} (stretch {BROWN_STRETCH[c][0]}-{BROWN_STRETCH[c][1]})"
            for c in channels
        )
        out_name = f"{base}_brown_rgb_{tag}_{VERSION}.tif"
        # Write to cube dir (alongside v6 maps)
        out_cube = cube_dir / out_name
        write_rgba_geotiff(rgba, out_cube, geo, desc)
        # Mirror to snapshot
        out_snap = snap_brown / out_name
        write_rgba_geotiff(rgba, out_snap, geo, desc)
        written.append(out_name)
        print(f"  wrote {out_name}")

    return {"cube_id": cube_id, "status": "ok",
            "written": written, "shape": list(mask.shape),
            "n_mask_pass": int(mask.sum()),
            "proj_name": geo["proj_name"] if geo else None}


def main(argv: list[str]) -> None:
    targets = argv[1:] if len(argv) > 1 else list(MTR3_CUBE_DIRS)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for cube_id in targets:
        if cube_id not in MTR3_CUBE_DIRS:
            print(f"SKIP unknown cube: {cube_id}")
            continue
        try:
            summary.append(process_cube(cube_id))
        except Exception as exc:
            print(f"ERROR processing {cube_id}: {exc}")
            summary.append({"cube_id": cube_id, "status": "error",
                            "error": str(exc)})

    print("\n=== summary ===")
    for s in summary:
        print(f"  {s['cube_id']}: {s.get('status')}")


if __name__ == "__main__":
    main(sys.argv)
