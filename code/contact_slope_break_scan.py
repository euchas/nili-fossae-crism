"""Automated slope-break lineament scan in a buffer around the mapped
clay/olivine contact.

the project specification: within ~100 m of the hand-drawn outlines, find SHARP, ABRUPT
slope changes that are CONTINUOUS (lineaments), on the 1 m HiRISE DTMs.

Method per DTM:
1. Slope map from the 1 m DTM after a 2 m Gaussian smooth.
2. TIN-fill mask (windowed mean |Laplacian| < 0.02 m, dilated 10 m) -
   fill facet edges would fake exactly this signal, so they are excluded.
3. Candidate pixels: slope >= SLOPE_MIN AND slope excess over the local
   50 m median slope >= EXCESS_MIN (sharp relative to surroundings).
4. Restrict to the +-BUFFER_M buffer of the mapped clay boundary.
5. Connected components (8-conn); keep components with skeleton length
   >= LEN_MIN and mean width (area/length) <= WID_MAX -> "lineaments".
6. Catalogue: length, width, mean/max slope, centroid (DTM frame +
   lon/lat), median distance to the contact line, orientation vs the
   local contact orientation. Overview map + thumbnails of the longest.

Outputs: outputs/revision/contact_3d/stage8_slopebreaks/

Usage: python code/contact_slope_break_scan.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.warp import transform as rtransform
from scipy import ndimage as ndi

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from contact_contact_fringe import HIRISE  # noqa: E402
from contact_fringe_gallery_v2 import clay_boundaries  # noqa: E402

OUT = ROOT / "outputs" / "revision" / "contact_3d" / "stage8_slopebreaks"
OUT.mkdir(parents=True, exist_ok=True)
FRAME = ROOT / "outputs" / "revision" / "contact_3d" / "frame" / "coreg_offsets.json"
WORK_RES = 2.0            # m/px working grid (2x decimation of the 1 m DTM)
SMOOTH_M = 2.0
BUFFER_M = 100.0
SLOPE_MIN = 12.0          # deg
EXCESS_MIN = 7.0          # deg above local 50 m median slope
LEN_MIN = 100.0           # m skeleton length for "continuous"
WID_MAX = 25.0            # m mean width
FILL_LAP = 0.02
LONLAT = "+proj=longlat +R=3393833.26 +no_defs"
N_THUMBS = 12


def skeleton_length(mask: np.ndarray, res: float) -> float:
    """Approximate length: count of skeleton pixels x res (8-conn diagonal corr ~1.12)."""
    from skimage.morphology import skeletonize
    sk = skeletonize(mask)
    return float(sk.sum()) * res * 1.12, sk


def main() -> None:
    t0 = datetime.now(timezone.utc)
    offsets = json.loads(FRAME.read_text())["hirise"]
    all_rows = []
    for dtm, path in HIRISE.items():
        o = offsets[dtm]["measured"]
        with rasterio.open(path) as ds:
            f = max(1, int(round(WORK_RES / abs(ds.transform.a))))
            z = ds.read(1, out_shape=(ds.height // f, ds.width // f), resampling=rasterio.enums.Resampling.average).astype(np.float32)
            nd = ds.nodata
            if nd is not None:
                z[z == nd] = np.nan
            z[(z < -1e30) | (z > 1e30)] = np.nan
            tr = Affine.translation(o["dx_m"], o["dy_m"]) * ds.transform * Affine.scale(f)
            dcrs = ds.crs
        res = abs(tr.a)
        zs = ndi.gaussian_filter(np.nan_to_num(z, nan=np.nanmedian(z)), SMOOTH_M / res)
        gy, gx = np.gradient(zs, res)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        slope[~np.isfinite(z)] = np.nan
        # fill mask
        lap = np.abs(ndi.laplace(np.nan_to_num(z, nan=0.0)))
        lap[~np.isfinite(z)] = np.nan
        fill = ndi.uniform_filter(np.nan_to_num(lap, nan=0.3), size=int(15 / res)) < FILL_LAP
        fill = ndi.binary_dilation(fill, iterations=int(10 / res))
        # local background slope (50 m median approximated by two-pass percentile filter on coarse grid)
        bg = ndi.median_filter(np.nan_to_num(slope, nan=0.0), size=int(50 / res))
        edge = ndi.binary_dilation(~np.isfinite(z), iterations=int(15 / res))   # median-filled nodata creates fake cliffs
        sharp = (slope >= SLOPE_MIN) & ((slope - bg) >= EXCESS_MIN) & ~fill & ~edge & np.isfinite(slope)
        # buffer around the mapped clay boundary
        lines = clay_boundaries(dcrs)
        buf = shapely.buffer(lines, BUFFER_M)
        bufmask = rasterize([(buf, 1)], out_shape=z.shape, transform=tr, fill=0, dtype="uint8").astype(bool)
        cand = sharp & bufmask
        # distance to the contact line (for the catalogue)
        linemask = rasterize([(lines, 1)], out_shape=z.shape, transform=tr, fill=0, dtype="uint8", all_touched=True).astype(bool)
        dline = ndi.distance_transform_edt(~linemask, sampling=(res, res))
        lab, n = ndi.label(cand, structure=np.ones((3, 3), bool))
        print(dtm, "candidate px:", int(cand.sum()), "components:", n)
        objs = ndi.find_objects(lab)
        rows = []
        for cid in range(1, n + 1):
            sl_ = objs[cid - 1]
            m = lab[sl_] == cid
            area_m2 = m.sum() * res * res
            if area_m2 < LEN_MIN * 2:      # cheap pre-filter
                continue
            try:
                length, sk = skeleton_length(m, res)
            except Exception:
                continue
            if length < LEN_MIN:
                continue
            width = area_m2 / length
            if width > WID_MAX:
                continue
            rr, cc = np.nonzero(m)
            rr = rr + sl_[0].start; cc = cc + sl_[1].start
            xw, yw = tr * (cc + 0.5, rr + 0.5)
            cx, cy = float(np.median(xw)), float(np.median(yw))
            lon, lat = rtransform(dcrs, LONLAT, [cx], [cy])
            # orientation of the component (PCA) vs local contact orientation
            xy = np.column_stack([xw, yw]) - [cx, cy]
            w_, v_ = np.linalg.eigh(np.cov(xy.T))
            comp_ang = float(np.degrees(np.arctan2(v_[1, -1], v_[0, -1])) % 180)
            from shapely.ops import nearest_points
            near = nearest_points(shapely.Point(cx, cy), lines)[1]
            seg = shapely.intersection(lines, shapely.buffer(shapely.Point(near.x, near.y), 40))
            cont_ang = np.nan
            if not seg.is_empty:
                geoms = seg.geoms if hasattr(seg, "geoms") else [seg]
                pts = np.concatenate([np.asarray(g.coords) for g in geoms if hasattr(g, "coords") and len(g.coords) > 1]) if geoms else None
                if pts is not None and len(pts) > 2:
                    pxy = pts - pts.mean(axis=0)
                    _, vv = np.linalg.eigh(np.cov(pxy.T))
                    cont_ang = float(np.degrees(np.arctan2(vv[1, -1], vv[0, -1])) % 180)
            dang = np.nan if np.isnan(cont_ang) else float(min(abs(comp_ang - cont_ang), 180 - abs(comp_ang - cont_ang)))
            rows.append({"dtm": dtm, "id": cid, "x_m": cx, "y_m": cy, "lon_e": lon[0], "lat_n": lat[0],
                         "length_m": round(length, 1), "width_m": round(width, 1), "area_m2": round(area_m2, 1),
                         "slope_mean": float(np.nanmean(np.where(lab == cid, slope, np.nan)[sl_])),
                         "slope_max": float(np.nanmax(np.where(lab == cid, slope, np.nan)[sl_])),
                         "dist_to_contact_med_m": float(np.median(dline[rr, cc])),
                         "angle_vs_contact_deg": dang})
        rows.sort(key=lambda r: -r["length_m"])
        all_rows += rows
        print(dtm, "lineaments kept:", len(rows))
        # overview figure (display decimated 3x -> ~6 m to keep memory low; detection stays at 2 m)
        DEC = 3
        ls = LightSource(azdeg=300, altdeg=30)
        zd = np.where(np.isfinite(z), z, np.nanmedian(z))[::DEC, ::DEC].astype(np.float32)
        hs = ls.hillshade(zd, vert_exag=2, dx=res * DEC, dy=res * DEC)
        del zd
        keepmask = np.isin(lab, [r["id"] for r in rows])
        keep_d = ndi.maximum_filter(keepmask, size=DEC)[::DEC, ::DEC]
        buf_d = bufmask[::DEC, ::DEC]
        fig, ax = plt.subplots(figsize=(9, 9 * hs.shape[0] / hs.shape[1]))
        ax.imshow(hs, cmap="gray")
        ax.imshow(np.ma.masked_where(~buf_d, np.ones_like(hs)), cmap="Blues", alpha=0.12, vmin=0, vmax=2)
        ax.imshow(np.ma.masked_where(~keep_d, np.ones_like(hs)), cmap="autumn", alpha=0.9, vmin=0, vmax=2)
        for r in rows[:30]:
            invt = ~tr
            c_, r_ = invt * (r["x_m"], r["y_m"])
            ax.annotate(str(r["id"]), (c_ / DEC, r_ / DEC), color="#00c8ff", fontsize=6)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{dtm}: continuous slope-break lineaments (red) within {BUFFER_M:.0f} m of the mapped contact (blue tint)\n"
                     f"slope >= {SLOPE_MIN:.0f} deg, excess >= {EXCESS_MIN:.0f} deg, length >= {LEN_MIN:.0f} m, width <= {WID_MAX:.0f} m, fill masked", fontsize=9)
        fig.tight_layout(); fig.savefig(OUT / f"fig_overview_{dtm}.png", dpi=170); plt.close(fig)
        # thumbnails of the longest
        top = rows[:N_THUMBS]
        if top:
            ncol = 4; nrow = int(np.ceil(len(top) / ncol))
            fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4.4 * nrow), squeeze=False)
            for r, ax in zip(top, axs.ravel()):
                invt = ~tr
                c_, r_ = invt * (r["x_m"], r["y_m"])
                h = int(150 / res)
                s0r, s0c = max(0, int(r_) - h), max(0, int(c_) - h)
                sub = np.s_[s0r:int(r_) + h, s0c:int(c_) + h]
                zt = np.where(np.isfinite(z[sub]), z[sub], np.nanmedian(z[sub])).astype(np.float32)
                ax.imshow(ls.hillshade(zt, vert_exag=2, dx=res, dy=res), cmap="gray")
                km = lab[sub] == r["id"]
                ax.imshow(np.ma.masked_where(~km, np.ones_like(zt)), cmap="autumn", alpha=0.85, vmin=0, vmax=2)
                ax.set_title(f"id {r['id']}: L={r['length_m']:.0f} m, w={r['width_m']:.0f} m, slope {r['slope_mean']:.0f}/{r['slope_max']:.0f} deg,\n"
                             f"d_contact {r['dist_to_contact_med_m']:.0f} m, angle vs contact {r['angle_vs_contact_deg']:.0f} deg", fontsize=7)
                ax.set_xticks([]); ax.set_yticks([])
            for ax in axs.ravel()[len(top):]:
                ax.set_visible(False)
            fig.suptitle(f"{dtm}: longest slope-break lineaments near the contact", fontsize=10)
            fig.tight_layout(); fig.savefig(OUT / f"fig_thumbs_{dtm}.png", dpi=150); plt.close(fig)

    pd.DataFrame(all_rows).to_csv(OUT / "slope_break_lineaments.csv", index=False)
    (OUT / "run_log.json").write_text(json.dumps({"work_res_m": WORK_RES, "buffer_m": BUFFER_M, "slope_min_deg": SLOPE_MIN,
        "excess_min_deg": EXCESS_MIN, "len_min_m": LEN_MIN, "wid_max_m": WID_MAX, "fill_lap": FILL_LAP,
        "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print("wrote", OUT, "| total lineaments:", len(all_rows))


if __name__ == "__main__":
    main()
