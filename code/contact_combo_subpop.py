"""Sub-populations of the olivine-unit '2.3 um only' detections.

Splits olivine-unit pixels detected at 2.3 um but not at 1.4 um by fitted
2.3 um centre (< 2.305 um: magnesite/smectite position; >= 2.305 um: Brown
2010's CRISM carbonate centroid 2.312-2.315 / talc position) and compares:
distance to mapped clay (rim vs interior), 1.4 um co-detection and SNR,
2.3 um amplitude, and the 2.5 um carbonate partner-band detection rate
(matched_filter/2500, if present). Clay-unit and 'both' pixels are given
for reference. Deterministic.

Usage:  python code/contact_combo_subpop.py
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import CUBE_IDS  # noqa: E402
from contact_distance_to_contact import load_cube, masks_from_labels, md5  # noqa: E402
from contact_band_center_map import unit_masks  # noqa: E402
from contact_carbonate_patches import CUBES, GEOMAP, FRAME, LONLAT, load_ddr_lonlat  # noqa: E402

BC = ROOT / "outputs" / "revision" / "band_center_map"
OH, MF23, MF25 = BC / "oh_filter", BC / "matched_filter" / "2300", BC / "matched_filter" / "2500"
OUT = BC / "combinations"
SPLIT_UM = 2.305
RIM_M = 90.0


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
    rows = []
    have25 = MF25.exists() and all((MF25 / f"{c}_fit_sensor.npz").exists() for c in CUBE_IDS)
    for cube in CUBE_IDS:
        a = np.load(OH / f"{cube}_ohfit_sensor.npz"); b = np.load(MF23 / f"{cube}_fit_sensor.npz")
        det14, snr14 = a["det"].astype(bool), a["snr"].astype(float)
        det23, cen23, amp23 = b["det"].astype(bool), b["centre"].astype(float), b["amp"].astype(float)
        det25 = np.load(MF25 / f"{cube}_fit_sensor.npz")["det"].astype(bool) if have25 else None
        c = load_cube(cube); m = masks_from_labels(c["labels"]); units, d, fr = unit_masks(cube, m)
        xs, gr, gc = project(cube, c)
        H, W = c["labels"].shape
        inside = np.isfinite(xs) & (gr >= 0) & (gr < H) & (gc >= 0) & (gc < W)
        gri = np.where(inside, gr, 0).astype(int); gci = np.where(inside, gc, 0).astype(int)
        f_clay = np.where(inside, fr["clay"][gri, gci], np.nan)
        dclay = np.where(inside, d["clay"][gri, gci], np.nan)
        mvalid = np.where(inside, m["valid"][gri, gci], False)
        usable = inside & mvalid & det23 & np.isfinite(snr14)
        oliv = usable & np.where(inside, units["olivine map-pure"][gri, gci], False)
        clay = usable & (f_clay >= 0.9)
        groups = {
            "olivine 2.3-only, centre < 2.305": oliv & ~det14 & (cen23 < SPLIT_UM),
            "olivine 2.3-only, centre >= 2.305": oliv & ~det14 & (cen23 >= SPLIT_UM),
            "olivine both": oliv & det14,
            "clay 2.3-only": clay & ~det14,
            "clay both": clay & det14,
        }
        for g, sel in groups.items():
            n = int(sel.sum())
            row = {"cube": cube, "group": g, "n": n}
            if n >= 30:
                row.update({"centre23_med": float(np.median(cen23[sel])),
                            "amp23_med": float(np.median(amp23[sel])),
                            "dist_clay_med_m": float(np.nanmedian(dclay[sel])),
                            "frac_within_90m_of_clay": float(np.nanmean(dclay[sel] <= RIM_M)),
                            "snr14_med": float(np.median(snr14[sel])),
                            "codet14_frac": float(det14[sel].mean()),
                            "det25_frac": float(det25[sel].mean()) if have25 else np.nan})
            rows.append(row)
        print(cube, "done")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "combo_subpopulations.csv", index=False)
    print(df.round(4).to_string())
    log = {"script": Path(__file__).name, "script_md5": md5(Path(__file__)),
           "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
           "split_um": SPLIT_UM, "rim_m": RIM_M, "have_2500": bool(have25), "deterministic": True}
    (OUT / "run_log_subpop.json").write_text(json.dumps(log, indent=2, default=str))


if __name__ == "__main__":
    main()
