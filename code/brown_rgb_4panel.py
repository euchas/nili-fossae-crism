"""Render the four cube Brown RGB composites as a 2x2 four-panel figure
on an A4 page. No axes, just cube IDs as titles. PNG + PDF output for
direct paper inclusion.

Inputs: outputs/brown_rgb/stretched/<cube>_brown_RGB_centroid_amp_asym_stretched.tif
Outputs: outputs/brown_rgb/figure_4panel/
  4cube_brown_RGB_centroid_amp_asym_4panel_A4.{png,pdf}
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CUBE_IDS, OUTPUTS_DIR

INPUT_DIR = OUTPUTS_DIR / "brown_rgb" / "stretched"
SNAPSHOT = OUTPUTS_DIR / "brown_rgb" / "figure_4panel"
COMPOSITE = "centroid_amp_asym"

# A4 portrait width preserved; height shrunk to match 2 rows of square
# cubes (avoids the "lots of white space" you get from aspect='equal'
# inside an over-tall axes). The saved file is then bbox-tight, so
# the resulting image inserts cleanly on an A4 page with normal
# manuscript margins.
A4_PORTRAIT = (8.27, 8.6)

# 2x2 grid layout - cube_id -> (row, col)
GRID = {
    "FRT00003E12": (0, 0),
    "FRT0000B438": (0, 1),
    "FRT0000A4FC": (1, 0),
    "FRT0000871C": (1, 1),
}


def main() -> None:
    t0 = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    SNAPSHOT.mkdir(parents=True, exist_ok=True)

    rgbas = {}
    for cube_id in CUBE_IDS:
        p = INPUT_DIR / f"{cube_id}_brown_RGB_{COMPOSITE}_stretched.tif"
        if not p.exists():
            raise SystemExit(f"missing {p}; run brown_rgb_stretched.py first")
        with rasterio.open(p) as src:
            rgbas[cube_id] = src.read()  # (4, H, W) uint8 RGBA
        print(f"  loaded {p.name}  shape={rgbas[cube_id].shape}")

    fig, axes = plt.subplots(2, 2, figsize=A4_PORTRAIT,
                              facecolor="white")
    for cube_id, (r, c) in GRID.items():
        ax = axes[r, c]
        rgba = rgbas[cube_id]
        rgba_disp = np.transpose(rgba, (1, 2, 0))  # (H, W, 4)
        ax.imshow(rgba_disp, interpolation="nearest", origin="upper")
        ax.set_title(cube_id, fontsize=11, pad=4,
                     fontfamily="DejaVu Sans")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_aspect("equal")

    fig.subplots_adjust(left=0.005, right=0.995, top=0.965, bottom=0.005,
                        wspace=0.02, hspace=0.02)

    out_png = SNAPSHOT / f"4cube_brown_RGB_{COMPOSITE}_4panel_A4.png"
    out_pdf = SNAPSHOT / f"4cube_brown_RGB_{COMPOSITE}_4panel_A4.pdf"
    fig.savefig(out_png, dpi=300, facecolor="white",
                bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_pdf, facecolor="white",
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  wrote {out_png.name}")
    print(f"  wrote {out_pdf.name}")

    log = {
        "stage": "brown_rgb_4panel",
        "timestamp_utc": timestamp,
        "composite": COMPOSITE,
        "layout": "2x2 grid on A4 portrait",
        "page_size_inches": list(A4_PORTRAIT),
        "panel_order": [[c for c, p in GRID.items() if p == (r, col)][0]
                         for r in range(2) for col in range(2)],
        "stretch_per_band": {"centroid": [1.15, 1.40],
                              "amplitude": [0.10, 0.37],
                              "asymmetry": [1.25, 5.00]},
        "out_png": str(out_png),
        "out_pdf": str(out_pdf),
        "wall_seconds": round(time.time() - t0, 2),
    }
    log_path = SNAPSHOT / f"brown_rgb_4panel_log_{timestamp}.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"  wrote {log_path.name}")
    print(f"\ndone in {round(time.time()-t0, 2)}s; output at {SNAPSHOT}")


if __name__ == "__main__":
    main()
