"""Stages 2-3 of PLAN_revision_contact_3D.md: height above the basal contact.

For every map-pure olivine pixel (>= 90 % olivine unit, no mapped clay) on
the CTX-frame CRISM grid, estimate the elevation of the clay/olivine
contact beneath it by inverse-distance interpolation of DEM elevations
sampled along the mapped contact, and compute h = z - z_contact. Do this
with the CTX DEM (20 m, whole map) and with the two Horseshoe HiRISE DTMs
(1 m, measured Stage 0 shifts applied). Profile the alteration parameters
against h, against the 3D distance to the contact, and estimate the unit
thickness where the mafic cap is present.

Usage: python code/contact_height_above_contact.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject
from scipy import stats
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.spatial import cKDTree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import CUBE_IDS  # noqa: E402
from contact_distance_to_contact import PALETTE, PIX_M, V8_PARAMS, load_cube, masks_from_labels  # noqa: E402

from config import CTX_DEM, HIRISE_PDS_DIR  # noqa: E402
HIRISE = {"DTEEC_006712_2020_005855_2020_A01": HIRISE_PDS_DIR / "DTEEC_006712_2020_005855_2020_A01.IMG",
          "DTEEC_002888_2025_002176_2025_A01": HIRISE_PDS_DIR / "DTEEC_002888_2025_002176_2025_A01.IMG"}
GEOMAP = ROOT / "outputs" / "revision" / "contact_3d" / "geomap" / "frame_ctx"
FRAME = ROOT / "outputs" / "revision" / "contact_3d" / "frame" / "coreg_offsets.json"
OUT = ROOT / "outputs" / "revision" / "contact_3d" / "stage2_height"
OUT.mkdir(parents=True, exist_ok=True)
CONTACT_FRAC = 0.5
OLIV_MIN, K_NEAR, R_MAX_M, MIN_PTS = 0.9, 25, 1000.0, 6
H_EDGES = np.array([-np.inf, -10, -5, 0, 5, 10, 15, 20, 30, 40, 60, 80, 120, np.inf])
H_LABELS = ["<-10", "-10..-5", "-5..0", "0..5", "5..10", "10..15", "15..20", "20..30", "30..40", "40..60", "60..80", "80..120", ">120"]
D3_EDGES = np.array([0, 18.1, 36.1, 54.1, 90, 150, 250, 400, 700, 1200, np.inf])
D3_LABELS = ["18", "36", "54", "55-90", "90-150", "150-250", "250-400", "400-700", "700-1200", ">1200"]
METRICS = ("D2300", "BD1900R2", "BD2500_2", "centroid", "OLINDEX3")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def warp_dem(src_path: str, src_shift: tuple[float, float], crs, tr: Affine, shape, resampling) -> np.ndarray:
    with rasterio.open(src_path) as ds:
        src = ds.read(1).astype(np.float32)
        nd = ds.nodata
        if nd is not None:
            src[src == nd] = np.nan
        src[(src < -1e30) | (src > 1e30)] = np.nan
        src_tr = Affine.translation(*src_shift) * ds.transform
        dst = np.full(shape, np.nan, np.float32)
        reproject(src, dst, src_transform=src_tr, src_crs=ds.crs, src_nodata=np.nan,
                  dst_transform=tr, dst_crs=crs, dst_nodata=np.nan, resampling=resampling)
    return dst


def contact_pixels(fr: dict, a: str, b: str) -> np.ndarray:
    """Pixels of unit a with a 4-neighbour of unit b, excluding places touched by other units."""
    A, B = fr[a] >= CONTACT_FRAC, fr[b] >= CONTACT_FRAC
    others = sum(fr[u] for u in ("clay", "cap", "dune", "graben") if u not in (a, b)) >= 0.1
    st = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    return ((A & binary_dilation(B, st)) | (B & binary_dilation(A, st))) & ~others


def idw_surface(z: np.ndarray, pts_mask: np.ndarray, query_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """IDW (p=2) of z at pts_mask pixels, evaluated at query_mask pixels (k nearest within R_MAX_M)."""
    pr, pc = np.nonzero(pts_mask & np.isfinite(z))
    out = np.full(z.shape, np.nan, np.float32)
    npts = np.zeros(z.shape, np.int16)
    if pr.size < MIN_PTS:
        return out, npts
    tree = cKDTree(np.column_stack([pr, pc]) * PIX_M)
    qr, qc = np.nonzero(query_mask)
    d, i = tree.query(np.column_stack([qr, qc]) * PIX_M, k=min(K_NEAR, pr.size), distance_upper_bound=R_MAX_M)
    d = np.atleast_2d(d); i = np.atleast_2d(i)
    ok = np.isfinite(d)
    n = ok.sum(axis=1)
    w = np.where(ok, 1.0 / np.maximum(d, PIX_M / 2) ** 2, 0.0)
    zz = np.where(ok, z[pr[np.clip(i, 0, pr.size - 1)], pc[np.clip(i, 0, pr.size - 1)]], 0.0)
    est = (w * zz).sum(axis=1) / np.where(w.sum(axis=1) > 0, w.sum(axis=1), np.nan)
    est[n < MIN_PTS] = np.nan
    out[qr, qc] = est
    npts[qr, qc] = n
    return out, npts


def binned(values: dict, key: np.ndarray, edges, labels, carb: np.ndarray, extra: dict) -> list[dict]:
    idx = np.digitize(key, edges) - 1
    rows = []
    for b, lab in enumerate(labels):
        s = idx == b
        n = int(s.sum())
        if n < 50:
            continue
        row = dict(extra, bin=lab, bin_idx=b, n=n, key_med=float(np.median(key[s])), carb_frac=float(carb[s].mean()))
        for p, v in values.items():
            vv = v[s][np.isfinite(v[s])]
            row[f"{p}_med"] = float(np.median(vv)) if vv.size else np.nan
            row[f"{p}_q25"] = float(np.percentile(vv, 25)) if vv.size else np.nan
            row[f"{p}_q75"] = float(np.percentile(vv, 75)) if vv.size else np.nan
        rows.append(row)
    return rows


def main() -> None:
    t0 = datetime.now(timezone.utc)
    offsets = json.loads(FRAME.read_text())
    rows_h, rows_d3, stats_rows, thick_rows = [], [], [], []
    for cube in CUBE_IDS:
        c = load_cube(cube)
        m = masks_from_labels(c["labels"])
        fz = np.load(GEOMAP / f"{cube}_fractions.npz")
        fr = {k: fz[k] for k in ("clay", "cap", "dune", "graben", "oliv", "outline")}
        tr = Affine.from_gdal(*fz["transform"])
        with rasterio.open(sorted((ROOT / "outputs" / "_v8_mtr3_with_brown" / cube / "geotiffs").glob("*.tif"))[0]) as ds:
            crs = ds.crs
        shape = c["labels"].shape
        mapped = fr["outline"] >= 0.99
        pure = m["valid"] & mapped & (fr["oliv"] >= OLIV_MIN) & (fr["clay"] == 0) & ~m["clay"]
        edge = distance_transform_edt(m["valid"] & mapped, sampling=(PIX_M, PIX_M)) > 90
        d_cap = distance_transform_edt(~(fr["cap"] >= CONTACT_FRAC), sampling=(PIX_M, PIX_M))
        pure &= edge & (d_cap > 90)          # same cap exclusion as Stage 1
        d_clay = distance_transform_edt(~(fr["clay"] >= CONTACT_FRAC), sampling=(PIX_M, PIX_M))
        cont_clay = contact_pixels(fr, "clay", "oliv")
        cont_cap = contact_pixels(fr, "cap", "oliv")

        dems = {"ctx": warp_dem(CTX_DEM, (0.0, 0.0), crs, tr, shape, Resampling.bilinear)}
        for dtm, path in HIRISE.items():
            mres = offsets["hirise"][dtm]["measured"]
            z = warp_dem(str(path), (mres["dx_m"], mres["dy_m"]), crs, tr, shape, Resampling.average)
            if np.isfinite(z[pure]).sum() > 2000:
                dems[dtm] = z
        vals = {p: c["params"][..., V8_PARAMS.index(p)] for p in METRICS}
        for dem_name, z in dems.items():
            zc, npts = idw_surface(z, cont_clay, pure)
            h = z - zc
            sel = pure & np.isfinite(h)
            if sel.sum() < 500:
                continue
            hh = h[sel]; dd = d_clay[sel]
            d3 = np.hypot(dd, hh)
            extra = {"cube": cube, "dem": dem_name}
            rows_h += binned({p: vals[p][sel] for p in METRICS}, hh, H_EDGES, H_LABELS, m["carb"][sel], extra)
            rows_d3 += binned({p: vals[p][sel] for p in METRICS}, d3, D3_EDGES, D3_LABELS, m["carb"][sel], extra)
            st = dict(extra, n=int(sel.sum()), h_med=float(np.median(hh)), h_p10=float(np.percentile(hh, 10)), h_p90=float(np.percentile(hh, 90)),
                      spearman_h_vs_dist=float(stats.spearmanr(hh, dd)[0]))
            for p in ("D2300", "BD1900R2", "BD2500_2", "centroid"):
                v = vals[p][sel]; ok = np.isfinite(v)
                st[f"spearman_{p}_h"] = float(stats.spearmanr(hh[ok], v[ok])[0])
                st[f"spearman_{p}_d3"] = float(stats.spearmanr(d3[ok], v[ok])[0])
                st[f"spearman_{p}_dist"] = float(stats.spearmanr(dd[ok], v[ok])[0])
                # partial: does h explain D2300 beyond horizontal distance? compare low vs high h at fixed distance band 90-400 m
                band = (dd > 90) & (dd <= 400) & ok
                if band.sum() > 200:
                    lo, hi = v[band & (hh <= 10)], v[band & (hh > 20)]
                    if lo.size > 50 and hi.size > 50:
                        st[f"{p}_h<=10_med_90-400m"] = float(np.median(lo)); st[f"{p}_h>20_med_90-400m"] = float(np.median(hi))
                        st[f"{p}_mwu_p_h_90-400m"] = float(stats.mannwhitneyu(lo, hi).pvalue)
            stats_rows.append(st)
            # unit thickness where the cap contact is nearby
            zcap, ncap = idw_surface(z, cont_cap, pure)
            T = zcap - zc
            tsel = sel & np.isfinite(T)
            if tsel.sum() > 200:
                thick_rows.append(dict(extra, n=int(tsel.sum()), T_med=float(np.median(T[tsel])), T_p10=float(np.percentile(T[tsel], 10)),
                                       T_p90=float(np.percentile(T[tsel], 90))))
            # rasters for ArcGIS
            with rasterio.open(OUT / f"{cube}_{dem_name}_h_above_contact.tif", "w", driver="GTiff", width=shape[1], height=shape[0], count=3,
                               dtype="float32", crs=crs, transform=tr, nodata=np.nan, compress="deflate") as dst:
                dst.write(np.where(sel, h, np.nan).astype(np.float32), 1); dst.set_band_description(1, "h_above_clay_contact_m")
                dst.write(np.where(np.isfinite(zc), zc, np.nan).astype(np.float32), 2); dst.set_band_description(2, "contact_elevation_m")
                dst.write(np.where(tsel, T, np.nan).astype(np.float32), 3); dst.set_band_description(3, "thickness_capbase_minus_contact_m")
            print(cube, dem_name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in st.items() if k in ("n", "h_med", "h_p10", "h_p90", "spearman_h_vs_dist", "spearman_D2300_h", "spearman_BD1900R2_h", "spearman_D2300_dist")})

    ph = pd.DataFrame(rows_h); ph.to_csv(OUT / "profiles_vs_height.csv", index=False)
    p3 = pd.DataFrame(rows_d3); p3.to_csv(OUT / "profiles_vs_3d_distance.csv", index=False)
    pd.DataFrame(stats_rows).to_csv(OUT / "height_stats.csv", index=False)
    pd.DataFrame(thick_rows).to_csv(OUT / "unit_thickness.csv", index=False)

    panels = [("carb_frac", "fraction in altered-olivine class (c0)"), ("D2300_med", "D2300 median"), ("BD1900R2_med", "BD1900R2 median"),
              ("BD2500_2_med", "BD2500_2 median"), ("centroid_med", "1 um centroid median (um)")]
    for dem_group, title, fname in [(["ctx"], "CTX DEM (20 m)", "fig_profiles_vs_height_ctx"),
                                    (list(HIRISE), "HiRISE DTMs (1 m, averaged to 18 m)", "fig_profiles_vs_height_hirise")]:
        sub = ph[ph.dem.isin(dem_group)]
        if sub.empty:
            continue
        fig, axes = plt.subplots(len(panels), 1, figsize=(7, 13), sharex=True)
        for ax, (key, ylabel) in zip(axes, panels):
            for (cube, dem), s in sub.groupby(["cube", "dem"]):
                s = s.sort_values("bin_idx")
                ax.plot(s.bin_idx, s[key], "-o", ms=4, lw=1.6, color=PALETTE[cube], label=f"{cube} ({dem[:12]})" if key == "carb_frac" else None)
            ax.set_ylabel(ylabel, fontsize=9); ax.grid(True, color="0.92"); ax.spines[["top", "right"]].set_visible(False)
            ax.axvline(2.5, color="0.6", lw=0.8, ls="--")
        axes[0].legend(fontsize=7, frameon=False)
        axes[-1].set_xticks(range(len(H_LABELS))); axes[-1].set_xticklabels(H_LABELS, rotation=45, ha="right", fontsize=8)
        axes[-1].set_xlabel("height above interpolated clay/olivine contact (m)")
        fig.suptitle(f"Map-pure olivine pixels vs. height above the basal contact - {title}", fontsize=11)
        fig.tight_layout(); fig.savefig(OUT / f"{fname}.png", dpi=200); fig.savefig(OUT / f"{fname}.pdf"); plt.close(fig)

    log = {"script_md5": md5(Path(__file__)), "ctx_dem": CTX_DEM, "hirise": {k: v.name for k, v in HIRISE.items()},
           "hirise_shifts_used": {k: offsets["hirise"][k]["measured"] for k in HIRISE}, "contact_frac": CONTACT_FRAC, "oliv_min": OLIV_MIN,
           "idw": {"k": K_NEAR, "r_max_m": R_MAX_M, "min_pts": MIN_PTS}, "h_edges": H_EDGES.tolist(),
           "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat()}
    (OUT / "run_log.json").write_text(json.dumps(log, indent=2, default=str))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
