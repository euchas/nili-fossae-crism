"""2.3 um band centres on Plebani bland-ratioed spectra (replication chain).

Independent replication of the Tier 0 band-centre analysis on a different
processing chain: sensor-space TRR3 spectra ratioed per detector COLUMN by
the crism_ml bland-pixel method (cached by the stage-6 campaign at
config.PLEBANI_RATIO_DIR/<stem>_ratioed_bland.npz). The
column-wise ratio cancels column-correlated detector artefacts, so any
feature of the MTR3 chain that vanishes here (the NW high-decile field,
the 2.33 um tail) is instrumental; anything that survives is real.

Same fit as Tier 0 (anchors 2.200-2.225 / 2.365-2.395 um, search
2.265-2.355 um, 3-point quadratic, edge flag; no despike in either chain).
Outputs sensor-space and map-projected centre/depth arrays, unit-sampled
summaries, decile/tail stacks (unit-internal ratios), and CSVs mirroring
the Tier 0 battery. Deterministic.

Usage:  python code/contact_band_center_plebani.py
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
from scipy.ndimage import label as cc_label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import CUBE_IDS  # noqa: E402
from contact_distance_to_contact import PIX_M, load_cube, masks_from_labels, md5  # noqa: E402
from contact_band_center_map import (  # noqa: E402
    DIST_EDGES, DIST_LABELS, LEFT_ANCHOR, RIGHT_ANCHOR, SEARCH, STRONG_DEPTH, WEAK_DEPTH, unit_masks,
)
from contact_band_center_modes import MODE_MAP_THRESH, LONG_TAIL  # noqa: E402
from contact_carbonate_patches import CUBES, GEOMAP, FRAME, LONLAT, load_ddr_lonlat  # noqa: E402
from contact_clay_provinces import band_metrics  # noqa: E402

from config import PLEBANI_RATIO_DIR as ZRAT  # noqa: E402
OUT = ROOT / "outputs" / "revision" / "band_center_map" / "plebani"
OUT.mkdir(parents=True, exist_ok=True)

DECILES = (0.10, 0.90)
TAIL_MIN_N = 200
MIN_COMPONENT_STRONG = 100


def fit_rows(wav: np.ndarray, spec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tier 0 band-centre fit on (n, nw) spectra; returns centre, depth, edge."""
    la = (wav >= LEFT_ANCHOR[0]) & (wav <= LEFT_ANCHOR[1])
    ra = (wav >= RIGHT_ANCHOR[0]) & (wav <= RIGHT_ANCHOR[1])
    rl, rr = np.nanmean(spec[:, la], axis=1), np.nanmean(spec[:, ra], axis=1)
    wl, wr = wav[la].mean(), wav[ra].mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        cr = spec / (rl[:, None] + (rr - rl)[:, None] * (wav[None, :] - wl) / (wr - wl))
    sw = np.where((wav >= SEARCH[0]) & (wav <= SEARCH[1]))[0]
    sub = cr[:, sw]
    ok = np.isfinite(sub).all(axis=1) & (np.abs(sub) > 1e-8).any(axis=1)
    center = np.full(spec.shape[0], np.nan, np.float32)
    depth = np.full(spec.shape[0], np.nan, np.float32)
    edge = np.zeros(spec.shape[0], bool)
    if not ok.any():
        return center, depth, edge
    s = sub[ok]
    i = np.argmin(s, axis=1)
    edge[ok] = (i == 0) | (i == s.shape[1] - 1)
    depth[ok] = (1.0 - s[np.arange(s.shape[0]), i]).astype(np.float32)
    ic = np.clip(i, 1, s.shape[1] - 2)
    y0 = s[np.arange(len(ic)), ic - 1]; y1 = s[np.arange(len(ic)), ic]; y2 = s[np.arange(len(ic)), ic + 1]
    x0, x1 = wav[sw[ic - 1]], wav[sw[ic]]
    den = y0 - 2 * y1 + y2
    with np.errstate(divide="ignore", invalid="ignore"):
        shift = 0.5 * (y0 - y2) / den * (x1 - x0)
    shift = np.where(np.isfinite(shift) & (np.abs(shift) <= (x1 - x0)), shift, 0.0)
    center[ok] = (x1 + shift).astype(np.float32)
    return center, depth, edge


def rasterize_median(vals: np.ndarray, gr: np.ndarray, gc: np.ndarray, shape) -> np.ndarray:
    out = np.full(shape, np.nan, np.float32)
    df = pd.DataFrame({"cell": gr.astype(np.int64) * shape[1] + gc, "v": vals})
    med = df.groupby("cell")["v"].median()
    out.ravel()[med.index.to_numpy()] = med.to_numpy(np.float32)
    return out


def main() -> None:
    t0 = datetime.now(timezone.utc)
    offsets = json.loads(FRAME.read_text())["crism"]
    sum_rows, prof_rows, mode_rows, prov_rows, stack_rows, inputs = [], [], [], [], [], {}
    for cube in CUBE_IDS:
        stem = CUBES[cube]["stem"]
        z = np.load(ZRAT / f"{stem}_ratioed_bland.npz")
        if2 = z["if2"].astype(np.float32)
        H_s, W_s = [int(v) for v in z["im_shape"]]
        wav = z["bands"].astype(np.float64)
        wav = wav / 1000.0 if wav.max() > 100 else wav
        inputs[cube] = {"ratioed_npz": str(ZRAT / f"{stem}_ratioed_bland.npz"),
                        "npz_size_bytes": (ZRAT / f"{stem}_ratioed_bland.npz").stat().st_size}
        cen_s, dep_s, edg_s = fit_rows(wav, if2)
        scol = (np.arange(if2.shape[0]) % W_s).astype(np.int16)

        # project sensor pixels to the MTR3 grid (stage-6 recipe)
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
        okp = np.isfinite(lon) & np.isfinite(lat)
        xs = np.full(lon.shape, np.nan); ys = np.full(lon.shape, np.nan)
        X, Y = rtransform(LONLAT, crs, lon[okp].astype(float), lat[okp].astype(float))
        xs[okp] = np.asarray(X) + o["dx_m"]; ys[okp] = np.asarray(Y) + o["dy_m"]
        colf, rowf = ~tr * (xs, ys)
        gc = np.floor(colf); gr = np.floor(rowf)
        H, W = c["labels"].shape
        inside = okp & (gr >= 0) & (gr < H) & (gc >= 0) & (gc < W)
        gri = np.where(inside, gr, 0).astype(int); gci = np.where(inside, gc, 0).astype(int)

        # sample map-grid context at each sensor pixel
        samp = lambda a: np.where(inside, a[gri, gci], np.nan)
        f_oliv, f_clay, f_cap = samp(fr["oliv"]), samp(fr["clay"]), samp(fr["cap"])
        pure = np.where(inside, units["olivine map-pure"][gri, gci], False)
        dclay, dcap = samp(d["clay"]), samp(d["cap"])
        mvalid = np.where(inside, m["valid"][gri, gci], False)

        good = np.isfinite(cen_s) & inside & mvalid
        strong = good & (dep_s >= STRONG_DEPTH) & ~edg_s
        weak = good & (dep_s >= WEAK_DEPTH) & (dep_s < STRONG_DEPTH)

        # map-grid rasters (per-cell median of strong-tier values; weak separately)
        for tier, tm in (("strong", strong), ("weak", weak)):
            r = rasterize_median(cen_s[tm], gr[tm].astype(int), gc[tm].astype(int), (H, W))
            np.save(OUT / f"{cube}_bcP_center_{tier}.npy", r)
        np.save(OUT / f"{cube}_bcP_depth.npy",
                rasterize_median(dep_s[good], gr[good].astype(int), gc[good].astype(int), (H, W)))
        np.save(OUT / f"{cube}_bcP_colmap.npy",
                rasterize_median(scol[good].astype(np.float32), gr[good].astype(int), gc[good].astype(int), (H, W)))
        np.savez_compressed(OUT / f"{cube}_bcP_sensor.npz", center=cen_s, depth=dep_s, edge=edg_s,
                            scol=scol, im_shape=(H_s, W_s))

        # unit summaries + distance profile + mode fractions (sensor-pixel level)
        unit_defs = {"olivine map-pure": pure & good,
                     "clay map-pure": good & (f_clay >= 0.9),
                     "cap map-pure": good & (f_cap >= 0.9)}
        for uname, um in unit_defs.items():
            for tier, tm in (("strong", strong), ("weak", weak)):
                v = cen_s[um & tm]
                sum_rows.append({"cube": cube, "unit": uname, "tier": tier, "n": int(v.size),
                                 "center_med_um": float(np.median(v)) if v.size else np.nan,
                                 "center_q25": float(np.percentile(v, 25)) if v.size else np.nan,
                                 "center_q75": float(np.percentile(v, 75)) if v.size else np.nan})
        sel = pure & strong & (dcap > 90)
        for (lo, hi), lab in zip(zip(DIST_EDGES[:-1], DIST_EDGES[1:]), DIST_LABELS):
            bm = sel & (dclay > lo) & (dclay <= hi)
            v = cen_s[bm]
            prof_rows.append({"cube": cube, "dist_bin_m": lab, "n": int(v.size),
                              "center_med_um": float(np.median(v)) if v.size else np.nan})
            mode_rows.append({"cube": cube, "dist_bin_m": lab, "n": int(v.size),
                              "frac_short": float((v < MODE_MAP_THRESH).mean()) if v.size else np.nan})

        # clay provinces on the projected strong-tier raster
        rstrong = np.load(OUT / f"{cube}_bcP_center_strong.npy")
        comp, ncomp = cc_label(fr["clay"] >= 0.5)
        for k in range(1, ncomp + 1):
            cm = (comp == k) & np.isfinite(rstrong)
            if int(cm.sum()) < MIN_COMPONENT_STRONG:
                continue
            med = float(np.median(rstrong[cm]))
            prov_rows.append({"cube": cube, "component": k, "n_cells": int(cm.sum()),
                              "center_med_um": med,
                              "class": "short" if med < MODE_MAP_THRESH else "long"})

        # decile/tail stacks with unit-internal ratios (on the bland-ratioed spectra)
        stacks = {}
        for uname, fu in (("olivine", f_oliv), ("clay", f_clay)):
            unit_px = good & (fu >= 0.5)
            selu = unit_px & strong
            if selu.sum() < 1000:
                continue
            qlo, qhi = np.quantile(cen_s[selu], DECILES)
            den = np.nanmedian(if2[unit_px], axis=0)
            sets = {f"low decile (<= {qlo:.4f})": selu & (cen_s <= qlo),
                    f"high decile (>= {qhi:.4f})": selu & (cen_s >= qhi)}
            if uname == "olivine" and (selu & (cen_s >= LONG_TAIL)).sum() >= TAIL_MIN_N:
                sets[f"tail (>= {LONG_TAIL})"] = selu & (cen_s >= LONG_TAIL)
            for name, mask in sets.items():
                r = np.nanmedian(if2[mask], axis=0) / den
                row = {"cube": cube, "unit": uname, "set": name, "n_px": int(mask.sum())}
                row.update(band_metrics(wav, r))
                stack_rows.append(row)
                stacks[f"{uname}|{name}"] = r
        np.savez_compressed(OUT / f"{cube}_stacks.npz", wav=wav,
                            **{k: v for k, v in stacks.items()})
        print(cube, f"good={int(good.sum())} strong={int(strong.sum())} "
              f"strong_pure_oliv={int((pure & strong).sum())} tail={int((good & strong & (f_oliv >= 0.5) & (cen_s >= LONG_TAIL)).sum())}")
        del if2, z

    pd.DataFrame(sum_rows).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(prof_rows).to_csv(OUT / "profile.csv", index=False)
    pd.DataFrame(mode_rows).to_csv(OUT / "mode_fraction.csv", index=False)
    pd.DataFrame(prov_rows).to_csv(OUT / "provinces.csv", index=False)
    pd.DataFrame(stack_rows).to_csv(OUT / "anomaly_metrics.csv", index=False)
    for name in ("summary", "profile", "mode_fraction", "provinces", "anomaly_metrics"):
        print(f"--- {name}")
        print(pd.read_csv(OUT / f"{name}.csv").round(4).to_string())

    log = {"script": Path(__file__).name, "script_md5": md5(Path(__file__)),
           "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
           "chain": "TRR3 sensor space, crism_ml bland-pixel column ratio (cached if2), no despike",
           "fit": {"left_anchor": LEFT_ANCHOR, "right_anchor": RIGHT_ANCHOR, "search": SEARCH,
                   "strong_depth": STRONG_DEPTH, "weak_depth": WEAK_DEPTH},
           "deciles": DECILES, "tail_um": LONG_TAIL, "mode_threshold_um": MODE_MAP_THRESH,
           "projection": "DDR lonlat -> MTR3 CRS + coreg shift -> fractions affine, per-cell median",
           "deterministic": True, "inputs": inputs}
    (OUT / "run_log_plebani.json").write_text(json.dumps(log, indent=2, default=str))
    print("done ->", OUT)


if __name__ == "__main__":
    main()
