"""Map of the two olivine-unit 2.3 um modes (carbonate-mode vs serpentine-mode).

Olivine-unit pixels (map fraction >= 0.5, no mapped clay) with a 2.3 um
matched-filter detection are classed as:
  A  '2.3 only', centre <  2.305 um  - carbonate-mode (2.30 + 2.50 partner on stacks)
  B  '2.3 only', centre >= 2.305 um  - serpentine/talc-mode (no 1.9 um water)
  C  'both' (1.4 + 2.3)              - carbonate cores with a hydrated admixture
Panel (a): class map with unit outlines; panel (b): distance to mapped clay
per class. Plebani chain. Deterministic.

Usage:  python code/contact_combo_mode_map.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.warp import transform as rtransform
from scipy.spatial import cKDTree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import CUBE_IDS  # noqa: E402
from contact_distance_to_contact import load_cube, masks_from_labels, md5  # noqa: E402
from contact_band_center_map import unit_masks  # noqa: E402
from contact_carbonate_patches import CUBES, GEOMAP, FRAME, LONLAT, load_ddr_lonlat  # noqa: E402
from figstyle import draw_units, label_panels, safe_savefig, unit_legend  # noqa: E402

BC = ROOT / "outputs" / "revision" / "band_center_map"
OH, MF23 = BC / "oh_filter", BC / "matched_filter" / "2300"
OUT = BC / "combinations"
SPLIT_UM = 2.305
FILL_TOL_M = 30.0
CLASSES = {1: ("carbonate-mode (2.3 only, < 2.305 um)", "#ff7f0e"),
           2: ("serpentine/talc-mode (2.3 only, >= 2.305 um)", "#8c564b"),
           3: ("carbonate cores (1.4 + 2.3)", "#2ca02c")}
DIST_EDGES = np.arange(0, 720, 36)


def project(cube, c):
    lon, lat = load_ddr_lonlat(CUBES[cube]["ddr"])
    lon, lat = lon.ravel(), lat.ravel()
    fz = np.load(GEOMAP / f"{cube}_fractions.npz")
    tr = Affine.from_gdal(*fz["transform"])
    with rasterio.open(sorted((ROOT / "outputs" / "_v8_mtr3_with_brown" / cube / "geotiffs").glob("*.tif"))[0]) as ds:
        crs = ds.crs
    o = json.loads(FRAME.read_text())["crism"][cube]
    okp = np.isfinite(lon) & np.isfinite(lat)
    xs = np.full(lon.shape, np.nan); ys = np.full(lon.shape, np.nan)
    X, Y = rtransform(LONLAT, crs, lon[okp].astype(float), lat[okp].astype(float))
    xs[okp] = np.asarray(X) + o["dx_m"]; ys[okp] = np.asarray(Y) + o["dy_m"]
    colf, rowf = ~tr * (xs, ys)
    return xs, ys, np.floor(rowf), np.floor(colf), tr


def main() -> None:
    t0 = datetime.now(timezone.utc)
    rows = []
    for cube in CUBE_IDS:
        a = np.load(OH / f"{cube}_ohfit_sensor.npz"); b = np.load(MF23 / f"{cube}_fit_sensor.npz")
        det14 = a["det"].astype(bool); det23, cen23 = b["det"].astype(bool), b["centre"].astype(float)
        c = load_cube(cube); m = masks_from_labels(c["labels"]); units, d, fr = unit_masks(cube, m)
        xs, ys, gr, gc, tr = project(cube, c)
        H, W = c["labels"].shape
        inside = np.isfinite(xs) & (gr >= 0) & (gr < H) & (gc >= 0) & (gc < W)
        gri = np.where(inside, gr, 0).astype(int); gci = np.where(inside, gc, 0).astype(int)
        f_oliv = np.where(inside, fr["oliv"][gri, gci], np.nan)
        f_clay = np.where(inside, fr["clay"][gri, gci], np.nan)
        dclay = np.where(inside, d["clay"][gri, gci], np.nan)
        mvalid = np.where(inside, m["valid"][gri, gci], False)
        oliv = inside & mvalid & (f_oliv >= 0.5) & (f_clay < 0.5) & det23
        cls = np.zeros(det23.size, np.int8)
        cls[oliv & ~det14 & (cen23 < SPLIT_UM)] = 1
        cls[oliv & ~det14 & (cen23 >= SPLIT_UM)] = 2
        cls[oliv & det14] = 3
        for k, (lab, _) in CLASSES.items():
            sel = cls == k
            rows.append({"cube": cube, "class": lab, "n": int(sel.sum()),
                         "dist_clay_med_m": float(np.nanmedian(dclay[sel])) if sel.any() else np.nan,
                         "frac_within_90m": float(np.nanmean(dclay[sel] <= 90)) if sel.any() else np.nan,
                         "frac_beyond_250m": float(np.nanmean(dclay[sel] > 250)) if sel.any() else np.nan})

        usable = inside & mvalid
        tree = cKDTree(np.column_stack([xs[usable], ys[usable]]))
        cc, rr = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        gx, gy = tr * (cc.ravel(), rr.ravel())
        dist, idx = tree.query(np.column_stack([gx, gy]), distance_upper_bound=FILL_TOL_M)
        hit = np.isfinite(dist)
        cmap_full = np.zeros(H * W, np.int8)
        cmap_full[hit] = cls[usable][idx[hit]]
        cmap_full = cmap_full.reshape(H, W)
        np.save(OUT / f"{cube}_mode_class.npy", cmap_full)

        fig = plt.figure(figsize=(15, 7))
        gs = fig.add_gridspec(1, 2, width_ratios=(1.35, 0.85), wspace=0.15)
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(np.where(m["valid"], 0.93, np.nan), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        lay = np.where(cmap_full > 0, cmap_full.astype(float), np.nan)
        ax.imshow(lay, cmap=ListedColormap([CLASSES[k][1] for k in (1, 2, 3)]), vmin=1, vmax=3,
                  interpolation="nearest")
        draw_units(ax, fr)
        ax.set_xticks([]); ax.set_yticks([])
        counts = {k: int((cls == k).sum()) for k in CLASSES}
        handles = [Patch(color=CLASSES[k][1], label=f"{CLASSES[k][0]}  n={counts[k]}") for k in (1, 2, 3)]
        leg1 = ax.legend(handles=handles, loc="lower right", fontsize=8.5, frameon=True, framealpha=0.9,
                         title="olivine-unit 2.3 um detections")
        ax.add_artist(leg1)
        unit_legend(ax, loc="lower left")
        ax.set_title(f"{cube}: olivine-unit 2.3 um phases (Plebani chain, per-pixel classes)", fontsize=10)

        ax2 = fig.add_subplot(gs[0, 1])
        for k, (lab, col) in CLASSES.items():
            v = dclay[cls == k]
            v = v[np.isfinite(v)]
            if v.size >= 30:
                ax2.hist(v, bins=DIST_EDGES, histtype="step", lw=2.0, color=col, density=True,
                         label=f"{lab.split(' (')[0]} (median {np.median(v):.0f} m)")
        ax2.axvline(90, color="0.6", lw=0.9, ls="--")
        ax2.annotate("90 m: mixing /\nregistration limit", (95, ax2.get_ylim()[1] * 0.9), fontsize=8, color="0.4")
        ax2.set_xlabel("distance from mapped clay (m)"); ax2.set_ylabel("density")
        ax2.set_title("where each class sits relative to the basement", fontsize=10)
        ax2.legend(fontsize=8, frameon=False); ax2.grid(True, color="0.95")
        ax2.spines[["top", "right"]].set_visible(False)
        label_panels(fig)
        safe_savefig(fig, OUT / f"fig_mode_map_{cube}", bbox_inches="tight")
        plt.close(fig)
        print(cube, counts)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "mode_map_stats.csv", index=False)
    print(df.round(3).to_string())
    log = {"script": Path(__file__).name, "script_md5": md5(Path(__file__)),
           "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
           "split_um": SPLIT_UM, "fill_tol_m": FILL_TOL_M, "deterministic": True}
    (OUT / "run_log_mode_map.json").write_text(json.dumps(log, indent=2, default=str))


if __name__ == "__main__":
    main()
