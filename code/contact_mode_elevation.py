"""Elevation test for the olivine-unit 2.3 um phase classes (erosional-window check).

Hypothesis: if the carbonate-bearing pixels are a basal carbonate level
exposed by erosion, they should sit LOWER than the surrounding olivine-unit
ground. Metric: local relative elevation z_rel = z - mean(z of olivine-unit
cells within a 15x15-cell (270 m) window), so that the general step down
toward the clay contact does not masquerade as a window; compared per class
against the olivine-unit control (map-pure olivine with usable fits and no
2.3 um detection), within distance-to-clay bins. DEMs: CTX (18 m, all
cubes) and the HiRISE DTM covering each cube (measured shifts applied, as
in stage 2). Block-bootstrap medians (504 m blocks, 1000 draws, seed 42).
Secondary: stage-2 height above the IDW clay-contact surface, within 150 m
of clay only (beyond that the IDW is untrusted). Deterministic.

Usage:  python code/contact_mode_elevation.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import transform as rtransform
from scipy.ndimage import uniform_filter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import CUBE_IDS  # noqa: E402
from contact_distance_to_contact import PALETTE, load_cube, masks_from_labels, md5  # noqa: E402
from contact_band_center_map import unit_masks  # noqa: E402
from contact_carbonate_patches import CUBES, GEOMAP, FRAME, LONLAT, load_ddr_lonlat  # noqa: E402
from contact_height_above_contact import CTX_DEM, HIRISE, warp_dem  # noqa: E402
from figstyle import label_panels, safe_savefig  # noqa: E402

BC = ROOT / "outputs" / "revision" / "band_center_map"
OH, MF23 = BC / "oh_filter", BC / "matched_filter" / "2300"
S2 = ROOT / "outputs" / "revision" / "contact_3d" / "stage2_height"
OUT = BC / "combinations"
SPLIT_UM = 2.305
WIN_PX = 15                      # 270 m local reference window
BLOCK_PX, NBOOT = 28, 1000
RNG = np.random.default_rng(42)
DBINS = ((36.1, 90.0), (90.0, 250.0), (250.0, np.inf))
DLABS = ("36-90 m", "90-250 m", "> 250 m")
CLASSES = {1: ("carbonate-mode", "#ff7f0e"), 2: ("serpentine/talc-mode", "#8c564b"),
           3: ("both bands", "#2ca02c"), 0: ("olivine control (no 2.3 det.)", "0.45")}
HIRISE_FOR = {"FRT00003E12": "DTEEC_002888_2025_002176_2025_A01", "FRT0000B438": "DTEEC_002888_2025_002176_2025_A01",
              "FRT0000871C": "DTEEC_006712_2020_005855_2020_A01", "FRT0000A4FC": "DTEEC_006712_2020_005855_2020_A01"}


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
    return xs, np.floor(rowf), np.floor(colf), tr, crs


def local_relief(z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    zz = np.where(mask & np.isfinite(z), z, 0.0)
    mm = (mask & np.isfinite(z)).astype(float)
    num = uniform_filter(zz, WIN_PX, mode="constant"); den = uniform_filter(mm, WIN_PX, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        ref = np.where(den > 0.15, num / den, np.nan)
    return z - ref


def block_boot_med(v, bid, w=None):
    blocks = np.unique(bid)
    per = {b: v[bid == b] for b in blocks}
    stat = float(np.median(v))
    if blocks.size < 5:
        return stat, np.nan, np.nan, int(blocks.size)
    boots = []
    for _ in range(NBOOT):
        pick = RNG.choice(blocks, blocks.size, replace=True)
        s = np.concatenate([per[b] for b in pick])
        if s.size > 20:
            boots.append(np.median(s))
    return stat, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), int(blocks.size)


def main() -> None:
    t0 = datetime.now(timezone.utc)
    offsets = json.loads(FRAME.read_text())
    rows = []
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    for j, cube in enumerate(CUBE_IDS):
        a = np.load(OH / f"{cube}_ohfit_sensor.npz"); b = np.load(MF23 / f"{cube}_fit_sensor.npz")
        det14, snr14 = a["det"].astype(bool), a["snr"].astype(float)
        det23, cen23, snr23 = b["det"].astype(bool), b["centre"].astype(float), b["snr"].astype(float)
        c = load_cube(cube); m = masks_from_labels(c["labels"]); units, d, fr = unit_masks(cube, m)
        xs, gr, gc, tr, crs = project(cube, c)
        H, W = c["labels"].shape
        inside = np.isfinite(xs) & (gr >= 0) & (gr < H) & (gc >= 0) & (gc < W)
        gri = np.where(inside, gr, 0).astype(int); gci = np.where(inside, gc, 0).astype(int)
        pure = np.where(inside, units["olivine map-pure"][gri, gci], False)
        dclay = np.where(inside, d["clay"][gri, gci], np.nan)
        dcap = np.where(inside, d["cap"][gri, gci], np.nan)
        mvalid = np.where(inside, m["valid"][gri, gci], False)
        universe = inside & mvalid & pure & (dcap > 90) & np.isfinite(snr14) & np.isfinite(snr23)
        bid = (gri // BLOCK_PX) * 100000 + (gci // BLOCK_PX)
        cls = np.zeros(det23.size, np.int8)
        cls[det23 & ~det14 & (cen23 < SPLIT_UM)] = 1
        cls[det23 & ~det14 & (cen23 >= SPLIT_UM)] = 2
        cls[det23 & det14] = 3

        oliv_cells = m["valid"] & (fr["oliv"] >= 0.5)
        dems = {"ctx": warp_dem(CTX_DEM, (0.0, 0.0), crs, tr, (H, W), Resampling.bilinear)}
        dtm = HIRISE_FOR[cube]
        mres = offsets["hirise"][dtm]["measured"]
        zh = warp_dem(str(HIRISE[dtm]), (mres["dx_m"], mres["dy_m"]), crs, tr, (H, W), Resampling.average)
        if np.isfinite(zh[oliv_cells]).sum() > 2000:
            dems["hirise"] = zh
        # secondary: stage-2 height above contact (band 1), trusted within 150 m only
        hpath = S2 / f"{cube}_ctx_h_above_contact.tif"
        h_above = rasterio.open(hpath).read(1) if hpath.exists() else None

        for di, (dem_name, z) in enumerate(dems.items()):
            zrel_map = local_relief(z, oliv_cells)
            zrel = np.where(inside, zrel_map[gri, gci], np.nan)
            ax = axes[di, j]
            pos = 0
            for (lo, hi), dl in zip(DBINS, DLABS):
                for k in (0, 1, 2, 3):
                    sel = universe & (cls == k) & (dclay > lo) & (dclay <= hi) & np.isfinite(zrel)
                    n = int(sel.sum())
                    if n < 30:
                        continue
                    med, clo, chi, nb = block_boot_med(zrel[sel], bid[sel])
                    rows.append({"cube": cube, "dem": dem_name, "dist_bin": dl, "class": CLASSES[k][0], "n": n,
                                 "zrel_med_m": med, "ci_lo": clo, "ci_hi": chi, "n_blocks": nb})
                    ax.errorbar(pos + k * 0.2, med, yerr=[[med - clo], [chi - med]] if np.isfinite(clo) else None,
                                fmt="o", color=CLASSES[k][1], ms=6, capsize=3,
                                label=CLASSES[k][0] if (pos == 0) else None)
                pos += 1.2
            ax.axhline(0, color="0.8", lw=0.8)
            ax.set_xticks([0.3 + 1.2 * i for i in range(len(DLABS))]); ax.set_xticklabels(DLABS, fontsize=8)
            ax.set_title(f"{cube} - {dem_name.upper()} DEM", fontsize=10, color=PALETTE[cube])
            ax.set_ylabel("local relative elevation (m)"); ax.grid(True, color="0.94")
            ax.spines[["top", "right"]].set_visible(False)
            if j == 0:
                ax.legend(fontsize=7.5, frameon=False, loc="lower left")
            if h_above is not None and dem_name == "ctx":
                ha = np.where(inside, h_above[gri, gci], np.nan)
                for k in (0, 1, 2, 3):
                    sel = universe & (cls == k) & (dclay <= 150) & np.isfinite(ha)
                    if sel.sum() >= 30:
                        med, clo, chi, nb = block_boot_med(ha[sel], bid[sel])
                        rows.append({"cube": cube, "dem": "ctx_h_above_contact(<=150m)", "dist_bin": "<=150 m",
                                     "class": CLASSES[k][0], "n": int(sel.sum()), "zrel_med_m": med,
                                     "ci_lo": clo, "ci_hi": chi, "n_blocks": nb})
        if "hirise" not in dems:
            axes[1, j].set_title(f"{cube} - HiRISE: no coverage", fontsize=10); axes[1, j].axis("off")
        print(cube, "done", list(dems))
    df = pd.DataFrame(rows); df.to_csv(OUT / "mode_elevation.csv", index=False)
    print(df.round(2).to_string())
    fig.suptitle("Are the carbonate-mode pixels topographically LOW (erosional windows)? "
                 "Local relative elevation vs surrounding olivine-unit ground (270 m window),\n"
                 "per phase class and distance-to-clay bin; block-bootstrap medians with 95% CIs", fontsize=10.5)
    fig.tight_layout()
    label_panels(fig)
    safe_savefig(fig, OUT / "fig_mode_elevation")
    plt.close(fig)
    log = {"script": Path(__file__).name, "script_md5": md5(Path(__file__)),
           "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
           "win_px": WIN_PX, "block_px": BLOCK_PX, "nboot": NBOOT, "seed": 42, "dbins": [list(b) for b in DBINS],
           "hirise_for": HIRISE_FOR, "note": "z_rel = z - local olivine-unit mean (270 m); h_above_contact secondary <=150 m"}
    (OUT / "run_log_mode_elevation.json").write_text(json.dumps(log, indent=2, default=str))


if __name__ == "__main__":
    main()
