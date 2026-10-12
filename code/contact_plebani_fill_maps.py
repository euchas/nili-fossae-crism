"""Gap-filled Plebani-chain 1.4 um band-centre maps (nearest-sample fill).

The stage-6 forward projection leaves unfilled 18 m cells wherever the
local sensor pixel scale exceeds the grid spacing (aliasing stripes). This
script inverts the lookup: every map cell takes the value of the NEAREST
projected sensor sample within FILL_TOL_M (no interpolation of values -
each cell holds one real measurement), and a 3x3 median variant is added
for display only. Reuses the fit and projection of
contact_band_center_1400_plebani.py. Deterministic.

Usage:  python code/contact_plebani_fill_maps.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import transform as rtransform
from scipy.spatial import cKDTree
from scipy.ndimage import generic_filter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import CUBE_IDS  # noqa: E402
from contact_distance_to_contact import load_cube, masks_from_labels, md5  # noqa: E402
from contact_band_center_map import unit_masks  # noqa: E402
from contact_carbonate_patches import CUBES, GEOMAP, FRAME, LONLAT, load_ddr_lonlat  # noqa: E402
from contact_band_center_plebani import ZRAT  # noqa: E402
from contact_band_center_1400_plebani import (  # noqa: E402
    CMAP_RANGE, PHASE_TICKS, STRONG_DEPTH, fit_rows,
)
from figstyle import draw_units, label_panels  # noqa: E402

OUT = ROOT / "outputs" / "revision" / "band_center_map" / "band1400" / "plebani"
FILL_TOL_M = 30.0


def med3(a: np.ndarray) -> np.ndarray:
    def f(w):
        v = w[np.isfinite(w)]
        return np.median(v) if v.size >= 3 else np.nan
    return generic_filter(a, f, size=3, mode="constant", cval=np.nan)


def main() -> None:
    t0 = datetime.now(timezone.utc)
    offsets = json.loads(FRAME.read_text())["crism"]
    inputs = {}
    for cube in CUBE_IDS:
        stem = CUBES[cube]["stem"]
        z = np.load(ZRAT / f"{stem}_ratioed_bland.npz")
        if2 = z["if2"].astype(np.float32)
        wav = z["bands"].astype(np.float64)
        wav = wav / 1000.0 if wav.max() > 100 else wav
        cen_s, dep_s, edg_s = fit_rows(wav, if2)
        del if2, z
        inputs[cube] = {"ratioed_npz": str(ZRAT / f"{stem}_ratioed_bland.npz")}

        c = load_cube(cube)
        m = masks_from_labels(c["labels"])
        units, d, fr = unit_masks(cube, m)
        lon, lat = load_ddr_lonlat(CUBES[cube]["ddr"])
        lon, lat = lon.ravel(), lat.ravel()
        fz = np.load(GEOMAP / f"{cube}_fractions.npz")
        tr = Affine.from_gdal(*fz["transform"])
        with rasterio.open(sorted((ROOT / "outputs" / "_v8_mtr3_with_brown" / cube / "geotiffs").glob("*.tif"))[0]) as ds:
            crs = ds.crs
        o = offsets[cube]
        okp = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(cen_s)
        X, Y = rtransform(LONLAT, crs, lon[okp].astype(float), lat[okp].astype(float))
        xs = np.asarray(X) + o["dx_m"]; ys = np.asarray(Y) + o["dy_m"]
        tree = cKDTree(np.column_stack([xs, ys]))

        H, W = c["labels"].shape
        cc, rr = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        gx, gy = tr * (cc.ravel(), rr.ravel())
        dist, idx = tree.query(np.column_stack([gx, gy]), distance_upper_bound=FILL_TOL_M)
        hit = np.isfinite(dist)
        cen_src, dep_src, edg_src = cen_s[okp], dep_s[okp], edg_s[okp]
        cen_f = np.full(H * W, np.nan, np.float32)
        dep_f = np.full(H * W, np.nan, np.float32)
        edg_f = np.zeros(H * W, bool)
        cen_f[hit] = cen_src[idx[hit]]
        dep_f[hit] = dep_src[idx[hit]]
        edg_f[hit] = edg_src[idx[hit]]
        cen_f = cen_f.reshape(H, W); dep_f = dep_f.reshape(H, W); edg_f = edg_f.reshape(H, W)
        cen_f[~m["valid"]] = np.nan
        strong_f = np.isfinite(cen_f) & (dep_f >= STRONG_DEPTH) & ~edg_f
        r_strong = np.where(strong_f, cen_f, np.nan)
        np.save(OUT / f"{cube}_bc140P_center_strong_filled.npy", r_strong)
        r_smooth = med3(r_strong)
        r_smooth[~m["valid"]] = np.nan
        np.save(OUT / f"{cube}_bc140P_center_strong_filled_med3.npy", r_smooth)
        print(cube, f"filled_cells={int(np.isfinite(r_strong).sum())} "
              f"(was {int(np.isfinite(np.load(OUT / f'{cube}_bc140P_center_strong.npy')).sum())})")

        fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharex=True, sharey=True)
        for ax, r, title in zip(axes, (r_strong, r_smooth),
                                (f"gap-filled (nearest sensor sample <= {FILL_TOL_M:.0f} m)",
                                 "3x3 median smoothed (display only)")):
            bg = np.where(m["valid"], 0.92, np.nan)
            ax.imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            im = ax.imshow(r, cmap="viridis", vmin=CMAP_RANGE[0], vmax=CMAP_RANGE[1],
                           interpolation="nearest")
            draw_units(ax, fr, units=("clay", "cap"))
            ax.set_title(f"{cube} - {title}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
        cbar.set_ticks([t for t, _ in PHASE_TICKS])
        cbar.set_ticklabels([lab for _, lab in PHASE_TICKS]); cbar.ax.tick_params(labelsize=8)
        fig.suptitle(f"{cube}: 1.4 um band centre, Plebani chain, strong tier "
                     f"(depth >= {STRONG_DEPTH}) - gap-filled", fontsize=11)
        label_panels(fig)
        fig.savefig(OUT / f"fig_map_filled_{cube}.png", dpi=200, bbox_inches="tight")
        fig.savefig(OUT / f"fig_map_filled_{cube}.pdf", bbox_inches="tight")
        plt.close(fig)

    log = {"script": Path(__file__).name, "script_md5": md5(Path(__file__)),
           "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
           "fill_tol_m": FILL_TOL_M,
           "note": "nearest-sample fill (one real measurement per cell, no value interpolation); "
                   "med3 variant is display-only", "inputs": inputs}
    (OUT / "run_log_fill_maps.json").write_text(json.dumps(log, indent=2, default=str))
    print("done ->", OUT)


if __name__ == "__main__":
    main()
