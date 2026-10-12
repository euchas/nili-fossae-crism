"""Stitch the four cube Brown RGB composites into a single mosaic.

Direct paste, first-wins. No histogram matching. No alpha blending.
The four input TIFs already carry the same matched physical stretch
(R=centroid 1.15-1.40, G=amplitude 0.10-0.37, B=asymmetry 1.25-5.00),
so simple overlap paste is what's needed.

Inputs: outputs/brown_rgb/stretched/<cube>_brown_RGB_centroid_amp_asym_stretched.tif
Outputs: outputs/brown_rgb/mosaic/
  4cube_brown_RGB_centroid_amp_asym_mosaic.tif       direct mosaic
  4cube_brown_RGB_centroid_amp_asym_mosaic.png       preview render
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CUBE_IDS, OUTPUTS_DIR

INPUT_DIR = OUTPUTS_DIR / "brown_rgb" / "stretched"
SNAPSHOT = OUTPUTS_DIR / "brown_rgb" / "mosaic"
COMPOSITE = "centroid_amp_asym"
PAINT_ORDER = ["FRT00003E12", "FRT0000B438", "FRT0000A4FC", "FRT0000871C"]


def main() -> None:
    t0 = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    SNAPSHOT.mkdir(parents=True, exist_ok=True)

    sources = {}
    for cube_id in CUBE_IDS:
        p = INPUT_DIR / f"{cube_id}_brown_RGB_{COMPOSITE}_stretched.tif"
        if not p.exists():
            raise SystemExit(f"missing {p}; run brown_rgb_stretched.py first")
        s = rasterio.open(p)
        sources[cube_id] = {
            "path": p, "src": s, "rgba": s.read(),
            "transform": s.transform, "bounds": s.bounds, "crs": s.crs,
        }
    res_x = sources[CUBE_IDS[0]]["transform"].a
    res_y = -sources[CUBE_IDS[0]]["transform"].e
    crs = sources[CUBE_IDS[0]]["crs"]
    for c in CUBE_IDS[1:]:
        s = sources[c]
        assert abs(s["transform"].a - res_x) < 1e-6
        assert abs(-s["transform"].e - res_y) < 1e-6
        assert s["crs"] == crs

    # Union bounds, snapped to source pixel grid
    bounds = [sources[c]["bounds"] for c in CUBE_IDS]
    x_min = min(b.left for b in bounds)
    x_max = max(b.right for b in bounds)
    y_min = min(b.bottom for b in bounds)
    y_max = max(b.top for b in bounds)
    grid_x = sources[CUBE_IDS[0]]["transform"].c
    grid_y = sources[CUBE_IDS[0]]["transform"].f
    x_min = grid_x + np.floor((x_min - grid_x) / res_x) * res_x
    x_max = grid_x + np.ceil((x_max - grid_x) / res_x) * res_x
    y_max = grid_y + np.ceil((y_max - grid_y) / res_y) * res_y
    y_min = grid_y - np.ceil((grid_y - y_min) / res_y) * res_y
    W = int(round((x_max - x_min) / res_x))
    H = int(round((y_max - y_min) / res_y))
    out_transform = Affine(res_x, 0.0, x_min, 0.0, -res_y, y_max)
    print(f"  mosaic: {H}x{W}  x={x_min:.1f}..{x_max:.1f}  "
          f"y={y_min:.1f}..{y_max:.1f}")

    # Paint each cube in PAINT_ORDER, first wins.
    mosaic = np.zeros((4, H, W), dtype=np.uint8)
    for cube_id in PAINT_ORDER:
        cube_rgba = sources[cube_id]["rgba"]
        b = sources[cube_id]["bounds"]
        col_off = int(round((b.left - x_min) / res_x))
        row_off = int(round((y_max - b.top) / res_y))
        h, w = cube_rgba.shape[1], cube_rgba.shape[2]
        sl_r = slice(row_off, row_off + h)
        sl_c = slice(col_off, col_off + w)
        cube_alpha = (cube_rgba[3] == 255)
        write_mask = (mosaic[3, sl_r, sl_c] == 0) & cube_alpha
        for c in range(4):
            band_slice = mosaic[c, sl_r, sl_c]
            band_slice[write_mask] = cube_rgba[c][write_mask]
        print(f"  painted {cube_id}  offset=({row_off},{col_off})  "
              f"wrote {int(write_mask.sum())} pixels")

    n_data = int((mosaic[3] == 255).sum())
    print(f"  total data pixels in mosaic: {n_data}")

    out_meta = {
        "driver": "GTiff", "dtype": "uint8",
        "height": H, "width": W, "count": 4,
        "crs": crs, "transform": out_transform,
        "compress": "deflate", "photometric": "rgb",
        "interleave": "pixel",
        "tiled": True, "blockxsize": 256, "blockysize": 256,
        "ALPHA": "YES",
    }
    out_tif = SNAPSHOT / f"4cube_brown_RGB_{COMPOSITE}_mosaic.tif"
    with rasterio.open(out_tif, "w", **out_meta) as dst:
        dst.write(mosaic, indexes=[1, 2, 3, 4])
        dst.colorinterp = [
            rasterio.enums.ColorInterp.red,
            rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.blue,
            rasterio.enums.ColorInterp.alpha,
        ]
        dst.update_tags(description=(
            "4-cube mosaic of Brown RGB R=centroid G=amplitude B=asymmetry, "
            "physical stretch (1.15-1.40, 0.10-0.37, 1.25-5.00). Direct "
            "paint, first-wins paint order: " + ", ".join(PAINT_ORDER) + "."
        ))
    print(f"  wrote {out_tif.name}")

    rgba_disp = np.transpose(mosaic, (1, 2, 0))
    fig_w = 12
    aspect = H / W
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * aspect))
    ax.imshow(rgba_disp,
              extent=[x_min, x_max, y_min, y_max], origin="upper",
              interpolation="nearest")
    ax.set_xlabel("Easting (m, IAU 2000 Equirectangular)")
    ax.set_ylabel("Northing (m, IAU 2000 Equirectangular)")
    ax.set_title("4-cube Brown RGB mosaic | "
                  "R=centroid (1.15-1.40), G=amplitude (0.10-0.37), "
                  "B=asymmetry (1.25-5.00)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    out_png = SNAPSHOT / f"4cube_brown_RGB_{COMPOSITE}_mosaic.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_png.name}")

    log = {
        "stage": "brown_rgb_mosaic",
        "timestamp_utc": timestamp,
        "composite": COMPOSITE,
        "paint_order": PAINT_ORDER,
        "merge_method": "direct_paint_first_wins",
        "stretch_per_band": {"centroid": [1.15, 1.40],
                              "amplitude": [0.10, 0.37],
                              "asymmetry": [1.25, 5.00]},
        "out_tif": str(out_tif),
        "out_png": str(out_png),
        "mosaic_shape_HW": [H, W],
        "n_data_pixels": n_data,
        "wall_seconds": round(time.time() - t0, 2),
    }
    log_path = SNAPSHOT / f"brown_rgb_mosaic_log_{timestamp}.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"  wrote {log_path.name}")
    print(f"\ndone in {round(time.time()-t0, 2)}s; output at {SNAPSHOT}")

    for c in CUBE_IDS:
        sources[c]["src"].close()


if __name__ == "__main__":
    main()
