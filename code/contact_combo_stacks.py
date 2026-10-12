"""Stacked spectra of the 1.4/2.3 um combination sub-populations (2.5 um test).

Stacks the bland-ratioed spectra of: olivine-unit '2.3 only' pixels split at
2.305 um, olivine 'both', clay '2.3 only', clay 'both'; plots 2.15-2.62 um
raw and divided by the unit median (unit-internal, Jezero style); measures
the 2.3 and 2.5 um bands on each stack. Mg-carbonate predicts a 2.50-2.51
um partner band on the olivine '2.3 only' stacks; a diluted smectite
predicts none and a 1.4/1.9 um signature instead. Deterministic.

Usage:  python code/contact_combo_stacks.py
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import CUBE_IDS  # noqa: E402
from contact_distance_to_contact import PALETTE, load_cube, masks_from_labels, md5  # noqa: E402
from contact_band_center_map import unit_masks  # noqa: E402
from contact_carbonate_patches import CUBES, GEOMAP, FRAME, LONLAT, load_ddr_lonlat  # noqa: E402
from contact_band_center_plebani import ZRAT  # noqa: E402
from contact_clay_provinces import band_metrics  # noqa: E402
from figstyle import label_panels, safe_savefig  # noqa: E402

BC = ROOT / "outputs" / "revision" / "band_center_map"
OH, MF23 = BC / "oh_filter", BC / "matched_filter" / "2300"
OUT = BC / "combinations"
SPLIT_UM = 2.305
GROUPS = ["olivine 2.3-only <2.305", "olivine 2.3-only >=2.305", "olivine both", "clay 2.3-only", "clay both"]
COLS = {"olivine 2.3-only <2.305": "#ff7f0e", "olivine 2.3-only >=2.305": "#8c564b",
        "olivine both": "#2ca02c", "clay 2.3-only": "#9467bd", "clay both": "#1f77b4"}


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
    return xs, np.floor(rowf), np.floor(colf)


def main() -> None:
    t0 = datetime.now(timezone.utc)
    rows, stacks_all = [], {}
    for cube in CUBE_IDS:
        a = np.load(OH / f"{cube}_ohfit_sensor.npz"); b = np.load(MF23 / f"{cube}_fit_sensor.npz")
        det14 = a["det"].astype(bool); det23, cen23 = b["det"].astype(bool), b["centre"].astype(float)
        c = load_cube(cube); m = masks_from_labels(c["labels"]); units, d, fr = unit_masks(cube, m)
        xs, gr, gc = project(cube, c)
        H, W = c["labels"].shape
        inside = np.isfinite(xs) & (gr >= 0) & (gr < H) & (gc >= 0) & (gc < W)
        gri = np.where(inside, gr, 0).astype(int); gci = np.where(inside, gc, 0).astype(int)
        f_clay = np.where(inside, fr["clay"][gri, gci], np.nan)
        f_oliv = np.where(inside, fr["oliv"][gri, gci], np.nan)
        mvalid = np.where(inside, m["valid"][gri, gci], False)
        oliv_unit = inside & mvalid & (f_oliv >= 0.5)
        clay_unit = inside & mvalid & (f_clay >= 0.5)
        oliv = inside & mvalid & det23 & np.where(inside, units["olivine map-pure"][gri, gci], False)
        clay = inside & mvalid & det23 & (f_clay >= 0.9)
        groups = {"olivine 2.3-only <2.305": oliv & ~det14 & (cen23 < SPLIT_UM),
                  "olivine 2.3-only >=2.305": oliv & ~det14 & (cen23 >= SPLIT_UM),
                  "olivine both": oliv & det14, "clay 2.3-only": clay & ~det14, "clay both": clay & det14}

        z = np.load(ZRAT / f"{CUBES[cube]['stem']}_ratioed_bland.npz")
        if2 = z["if2"].astype(np.float32); wav = z["bands"].astype(np.float64)
        wav = wav / 1000.0 if wav.max() > 100 else wav
        den = {"olivine": np.nanmedian(if2[oliv_unit], axis=0), "clay": np.nanmedian(if2[clay_unit], axis=0)}
        stacks = {}
        for g, sel in groups.items():
            n = int(sel.sum())
            if n < 30:
                continue
            sp = np.nanmedian(if2[sel], axis=0)
            unit = "olivine" if g.startswith("olivine") else "clay"
            r = sp / den[unit]
            stacks[g] = (sp, r, n)
            row = {"cube": cube, "group": g, "n": n}
            row.update({f"raw_{k}": v for k, v in band_metrics(wav, sp).items()})
            row.update({f"rel_{k}": v for k, v in band_metrics(wav, r).items()})
            rows.append(row)
        del if2, z
        stacks_all[cube] = (wav, stacks)
        print(cube, {g: v[2] for g, v in stacks.items()})

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
        for ax, which, title in zip(axes, (0, 1), ("bland-ratioed stack (normalized + offset)",
                                                    "stack / own-unit median (normalized + offset)")):
            off = 0.0
            for g in GROUPS:
                if g not in stacks:
                    continue
                y = stacks[g][which]
                sel = (wav >= 2.15) & (wav <= 2.62)
                yn = y / np.nanmedian(y[sel]) + off
                ax.plot(wav[sel], yn[sel], color=COLS[g], lw=1.6, label=f"{g} (n={stacks[g][2]})")
                off += 0.03
            for f, lab in ((2.300, "2.30"), (2.313, "2.31"), (2.322, "2.32"), (2.50, "2.50"), (2.53, "2.53")):
                ax.axvline(f, color="0.88", lw=0.8)
                ax.annotate(lab, (f, ax.get_ylim()[1]), fontsize=7, ha="center", color="0.45", annotation_clip=False)
            ax.set_xlim(2.15, 2.62); ax.set_xlabel("wavelength (um)"); ax.set_title(title, fontsize=10)
            ax.legend(fontsize=7.5, frameon=False); ax.grid(True, color="0.96")
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"{cube}: stacked spectra of the 1.4/2.3 um combination sub-populations - "
                     "does the olivine '2.3 only' phase carry the 2.5 um carbonate partner?", fontsize=10.5)
        fig.tight_layout()
        label_panels(fig)
        safe_savefig(fig, OUT / f"fig_combo_stacks_{cube}")
        plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "combo_stack_metrics.csv", index=False)
    cols = ["cube", "group", "n", "raw_bc2300_um", "raw_bd2300", "raw_bc2500_um", "raw_bd2500",
            "rel_bc2300_um", "rel_bd2300", "rel_bc2500_um", "rel_bd2500", "raw_bd1900", "raw_bd1390"]
    print(df[cols].round(4).to_string())
    log = {"script": Path(__file__).name, "script_md5": md5(Path(__file__)),
           "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
           "split_um": SPLIT_UM, "deterministic": True}
    (OUT / "run_log_stacks.json").write_text(json.dumps(log, indent=2, default=str))


if __name__ == "__main__":
    main()
