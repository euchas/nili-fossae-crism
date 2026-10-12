"""ROI spectra: olivine 'elevated 1.4 um centre' vs smectite 'low centre'.

Auto-selects two homogeneous square ROIs on the gap-filled Plebani 1.4 um
map of FRT00003E12 (deterministic rule, logged): one in map-pure olivine
with smooth centres in 1.420-1.455 um (>=200 m from mapped clay), one in
map-pure clay with centres in 1.380-1.405 um. Median-stacks the bland-
ratioed sensor spectra inside each ROI and plots both together: ROI
location map, full-range spectra, and a 1.30-1.55 um zoom with the fit's
linear continuum drawn, so the olivine 1 um wing curvature and the clay
OH band can be compared directly. Deterministic.

Usage:  python code/contact_roi_spectra_1400.py
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
from scipy.ndimage import uniform_filter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from contact_distance_to_contact import load_cube, masks_from_labels, md5  # noqa: E402
from contact_band_center_map import unit_masks  # noqa: E402
from contact_carbonate_patches import CUBES, GEOMAP, FRAME, LONLAT, load_ddr_lonlat  # noqa: E402
from contact_band_center_plebani import ZRAT  # noqa: E402
from contact_band_center_1400_plebani import LEFT_ANCHOR, RIGHT_ANCHOR  # noqa: E402
from figstyle import draw_units, label_panels  # noqa: E402

CUBE = "FRT00003E12"
OUT = ROOT / "outputs" / "revision" / "band_center_map" / "band1400" / "plebani"
BOX = 24                       # ROI side in 18 m cells (~430 m)
ROI_RULES = {
    "olivine ROI": {"frac_key": "oliv", "range_um": (1.420, 1.455), "min_dist_clay_m": 200.0,
                    "color": "#D55E00"},
    "smectite ROI": {"frac_key": "clay", "range_um": (1.380, 1.405), "min_dist_clay_m": None,
                     "color": "#0072B2"},
}


def pick_roi(filled: np.ndarray, ok_cells: np.ndarray, rng_um: tuple) -> tuple[int, int]:
    """Top-left cell of the BOX x BOX window maximizing in-range fraction,
    tie-broken by centre homogeneity (lowest std)."""
    inr = (np.isfinite(filled) & (filled >= rng_um[0]) & (filled <= rng_um[1]) & ok_cells).astype(float)
    frac = uniform_filter(inr, size=BOX, mode="constant", cval=0.0)
    v = np.where(np.isfinite(filled) & ok_cells, filled, 0.0)
    v2 = uniform_filter(v * inr, size=BOX, mode="constant", cval=0.0)
    v2sq = uniform_filter((v * inr) ** 2, size=BOX, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = v2 / frac
        std = np.sqrt(np.maximum(v2sq / frac - mean ** 2, 0.0))
    score = frac - 5.0 * np.where(np.isfinite(std), std, 1.0)
    score[frac < 0.5] = -np.inf
    r, c = np.unravel_index(np.argmax(score), score.shape)
    return int(r - BOX // 2), int(c - BOX // 2)


def main() -> None:
    t0 = datetime.now(timezone.utc)
    stem = CUBES[CUBE]["stem"]
    z = np.load(ZRAT / f"{stem}_ratioed_bland.npz")
    if2 = z["if2"].astype(np.float32)
    wav = z["bands"].astype(np.float64)
    wav = wav / 1000.0 if wav.max() > 100 else wav

    c = load_cube(CUBE)
    m = masks_from_labels(c["labels"])
    units, d, fr = unit_masks(CUBE, m)
    filled = np.load(OUT / f"{CUBE}_bc140P_center_strong_filled.npy")
    H, W = filled.shape

    lon, lat = load_ddr_lonlat(CUBES[CUBE]["ddr"])
    lon, lat = lon.ravel(), lat.ravel()
    fz = np.load(GEOMAP / f"{CUBE}_fractions.npz")
    tr = Affine.from_gdal(*fz["transform"])
    with rasterio.open(sorted((ROOT / "outputs" / "_v8_mtr3_with_brown" / CUBE / "geotiffs").glob("*.tif"))[0]) as ds:
        crs = ds.crs
    o = json.loads(FRAME.read_text())["crism"][CUBE]
    okp = np.isfinite(lon) & np.isfinite(lat)
    X, Y = rtransform(LONLAT, crs, lon[okp].astype(float), lat[okp].astype(float))
    xs = np.asarray(X) + o["dx_m"]; ys = np.asarray(Y) + o["dy_m"]
    colf, rowf = ~tr * (xs, ys)
    sgr = np.full(lon.shape, -1); sgc = np.full(lon.shape, -1)
    sgr[okp] = np.floor(rowf); sgc[okp] = np.floor(colf)

    rois, spectra = {}, {}
    for name, rule in ROI_RULES.items():
        ok = m["valid"] & (fr[rule["frac_key"]] >= 0.9)
        if rule["min_dist_clay_m"]:
            ok &= d["clay"] > rule["min_dist_clay_m"]
        r0, c0 = pick_roi(filled, ok, rule["range_um"])
        rois[name] = (r0, c0)
        sel = (sgr >= r0) & (sgr < r0 + BOX) & (sgc >= c0) & (sgc < c0 + BOX)
        cellok = np.zeros((H, W), bool)
        cellok[r0:r0 + BOX, c0:c0 + BOX] = ok[r0:r0 + BOX, c0:c0 + BOX]
        sel &= np.where((sgr >= 0) & (sgc >= 0),
                        cellok[np.clip(sgr, 0, H - 1), np.clip(sgc, 0, W - 1)], False)
        sp = np.nanmedian(if2[sel], axis=0)
        spectra[name] = (sp, int(sel.sum()))
        cen_cells = filled[r0:r0 + BOX, c0:c0 + BOX]
        print(name, f"box at (row {r0}, col {c0}) size {BOX}; n_sensor_px={int(sel.sum())}; "
              f"cell centres med={np.nanmedian(cen_cells):.4f}")

    pd.DataFrame({"wavelength_um": wav,
                  **{k: v[0] for k, v in spectra.items()}}).to_csv(
        OUT / "roi_spectra_FRT00003E12.csv", index=False)

    fig = plt.figure(figsize=(16, 5.8))
    gs = fig.add_gridspec(1, 3, width_ratios=(0.9, 1.1, 1.0), wspace=0.25)
    ax0 = fig.add_subplot(gs[0, 0])
    bg = np.where(m["valid"], 0.92, np.nan)
    ax0.imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax0.imshow(filled, cmap="viridis", vmin=1.375, vmax=1.485, interpolation="nearest")
    draw_units(ax0, fr, units=("clay",))
    for name, rule in ROI_RULES.items():
        r0, c0 = rois[name]
        ax0.add_patch(Rectangle((c0, r0), BOX, BOX, fill=False, edgecolor=rule["color"], lw=2.0))
    ax0.set_title(f"{CUBE}: filled 1.4 um centre map + ROIs", fontsize=10)
    ax0.set_xticks([]); ax0.set_yticks([])

    ax1 = fig.add_subplot(gs[0, 1])
    for name, rule in ROI_RULES.items():
        sp, n = spectra[name]
        ax1.plot(wav, sp / np.nanmedian(sp), lw=1.5, color=rule["color"], label=f"{name} (n={n})")
    for f in (1.39, 1.9, 2.31):
        ax1.axvline(f, color="0.92", lw=0.8)
    ax1.set_xlim(1.02, 2.6)
    ax1.set_xlabel("wavelength (um)"); ax1.set_ylabel("bland-ratioed I/F (normalized)")
    ax1.set_title("ROI median spectra, full range", fontsize=10)
    ax1.legend(fontsize=9, frameon=False)
    ax1.grid(True, color="0.96"); ax1.spines[["top", "right"]].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 2])
    zs = (wav >= 1.28) & (wav <= 1.57)
    off = 0.0
    for name, rule in ROI_RULES.items():
        sp, n = spectra[name]
        y = sp / np.nanmedian(sp[zs]) + off
        ax2.plot(wav[zs], y[zs], lw=1.8, color=rule["color"], label=name)
        la = (wav >= LEFT_ANCHOR[0]) & (wav <= LEFT_ANCHOR[1])
        ra = (wav >= RIGHT_ANCHOR[0]) & (wav <= RIGHT_ANCHOR[1])
        xl, xr = wav[la].mean(), wav[ra].mean()
        yl = np.nanmean(y[la]); yr = np.nanmean(y[ra])
        ax2.plot([xl, xr], [yl, yr], color=rule["color"], lw=1.0, ls=":")
        off += 0.04
    for f, lab in ((1.385, "1.385"), (1.41, "1.41"), (1.44, "1.44"), (1.48, "1.48")):
        ax2.axvline(f, color="0.9", lw=0.7)
        ax2.annotate(lab, (f, ax2.get_ylim()[1]), fontsize=7, ha="center", color="0.45",
                     annotation_clip=False)
    ax2.set_xlim(1.28, 1.57)
    ax2.set_xlabel("wavelength (um)")
    ax2.set_title("zoom 1.3-1.55 um with fit continuum (dotted)", fontsize=10)
    ax2.legend(fontsize=9, frameon=False)
    ax2.grid(True, color="0.96"); ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Bland-ratioed ROI spectra: olivine-unit 'elevated centre' vs smectite 'low centre'",
                 fontsize=11.5)
    label_panels(fig)
    fig.savefig(OUT / "fig_roi_spectra_FRT00003E12.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig_roi_spectra_FRT00003E12.pdf", bbox_inches="tight")
    plt.close(fig)

    log = {"script": Path(__file__).name, "script_md5": md5(Path(__file__)),
           "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
           "cube": CUBE, "box_cells": BOX,
           "rois": {k: {"row0": rois[k][0], "col0": rois[k][1], **{kk: str(vv) for kk, vv in v.items()}}
                    for k, v in ROI_RULES.items()},
           "deterministic": True}
    (OUT / "run_log_roi_spectra.json").write_text(json.dumps(log, indent=2, default=str))
    print("done ->", OUT)


if __name__ == "__main__":
    main()
