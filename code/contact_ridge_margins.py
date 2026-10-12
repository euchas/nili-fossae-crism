"""Stage 4 of PLAN_revision_contact_3D.md: HiRISE-scale profiles across
olivine-cored ridges (exploratory).

For every catalogued intrusion point inside the DTEEC_006712_2020_005855_2020
DTM (measured Stage 0 shift applied), take the local strike from the nearest
mapped ridge polyline (Raised_ridges / Ruptured_ridges; PCA of local highs
if none within 60 m), extract a swath-averaged elevation and orthoimage
brightness profile perpendicular to the ridge (+-150 m, 1 m step, swath
+-10 m along strike), detrend against the far field (80-150 m both sides),
and report core height, flank elevation (20-40 m from the axis) and the
bright-core half-width. Produces a gallery figure and a metrics CSV.

Usage: python code/contact_ridge_margins.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
import rasterio
import shapely
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import transform as rtransform
from rasterio.windows import from_bounds
from scipy.ndimage import map_coordinates

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
from config import GEOMAP_SOURCE as GDB, HIRISE_PDS_DIR  # noqa: E402
DTM_ID = "DTEEC_006712_2020_005855_2020_A01"
DTM = HIRISE_PDS_DIR / f"{DTM_ID}.IMG"
ORTHO = HIRISE_PDS_DIR / "PSP_006712_2020_RED_A_01_ORTHO.JP2"
FRAME = ROOT / "outputs" / "revision" / "contact_3d" / "frame" / "coreg_offsets.json"
OUT = ROOT / "outputs" / "revision" / "contact_3d" / "stage4_ridges"
OUT.mkdir(parents=True, exist_ok=True)
HALF_M, STEP_M, SWATH_M, WIN_M = 150.0, 1.0, 10.0, 220.0
FAR = (80.0, 150.0)
FLANK = (20.0, 40.0)
CORE = 15.0
LINE_SEARCH_M = 60.0


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def reproj(geoms, src: CRS, dst: CRS):
    return [shapely.transform(g, lambda xy: np.column_stack(rtransform(src, dst, xy[:, 0], xy[:, 1]))) for g in geoms]


def read_window(path, crs, transform_shift: Affine, cx: float, cy: float, res_out: float):
    """Read a WIN_M window around (cx, cy) (coordinates in the shifted frame) resampled to res_out."""
    with rasterio.open(path) as ds:
        tr = transform_shift * ds.transform
        inv = ~tr
        c0, r0 = inv * (cx - WIN_M, cy + WIN_M)
        c1, r1 = inv * (cx + WIN_M, cy - WIN_M)
        win = rasterio.windows.Window(int(np.floor(c0)), int(np.floor(r0)), int(np.ceil(c1 - c0)), int(np.ceil(r1 - r0)))
        f = max(1, int(round(res_out / abs(ds.transform.a))))
        out_shape = (max(1, win.height // f), max(1, win.width // f))
        arr = ds.read(1, window=win, out_shape=out_shape, resampling=Resampling.average, boundless=True, fill_value=np.nan).astype(np.float32)
        nd = ds.nodata
        if nd is not None:
            arr[arr == nd] = np.nan
        bad = (arr < -1e30) | (arr > 1e30)
        if "ORTHO" in str(path):
            bad |= arr == 0                      # JP2 orthos use 0 as fill
        arr[bad] = np.nan
        wtr = rasterio.windows.transform(win, tr) * Affine.scale(f)
    return arr, wtr


def sample(arr: np.ndarray, wtr: Affine, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    inv = ~wtr
    cols, rows = inv * (xs, ys)
    return map_coordinates(arr, [rows, cols], order=1, mode="constant", cval=np.nan)


def strike_from_lines(pt, lines) -> tuple[float | None, float]:
    """Direction (radians) of the nearest polyline segment within LINE_SEARCH_M."""
    best, bd = None, np.inf
    for ln in lines:
        d = shapely.distance(pt, ln)
        if d < bd:
            best, bd = ln, d
    if best is None or bd > LINE_SEARCH_M:
        return None, bd
    parts = best.geoms if hasattr(best, "geoms") else [best]
    seg_best, sd = None, np.inf
    for part in parts:
        xy = np.asarray(part.coords)
        for i in range(len(xy) - 1):
            seg = shapely.LineString([xy[i], xy[i + 1]])
            d = shapely.distance(pt, seg)
            if d < sd:
                sd, seg_best = d, xy[i + 1] - xy[i]
    return float(np.arctan2(seg_best[1], seg_best[0])), bd


def strike_from_pca(z: np.ndarray, wtr: Affine, cx: float, cy: float) -> float:
    inv = ~wtr
    c, r = inv * (cx, cy)
    rr, cc = np.mgrid[0:z.shape[0], 0:z.shape[1]]
    res = abs(wtr.a)
    dist = np.hypot((rr - r) * res, (cc - c) * res)
    loc = (dist < 60) & np.isfinite(z)
    zz = z[loc]
    hi = loc.copy(); hi[loc] = zz > np.nanpercentile(zz, 75)
    pr, pc = np.nonzero(hi)
    xy = np.column_stack([pc * res, -pr * res]); xy -= xy.mean(axis=0)
    w, v = np.linalg.eigh(np.cov(xy.T))
    return float(np.arctan2(v[1, -1], v[0, -1]))


def main() -> None:
    t0 = datetime.now(timezone.utc)
    off = json.loads(FRAME.read_text())["hirise"][DTM_ID]["measured"]
    shift = Affine.translation(off["dx_m"], off["dy_m"])
    with rasterio.open(DTM) as ds:
        dcrs = ds.crs; b = ds.bounds
    box = shapely.box(b.left + off["dx_m"], b.bottom + off["dy_m"], b.right + off["dx_m"], b.top + off["dy_m"])
    mcrs = CRS.from_wkt(pyogrio.read_info(GDB, layer="Intrusions")["crs"])
    _, _, g, f = pyogrio.raw.read(GDB, layer="Intrusions")
    pts = reproj(shapely.force_2d(shapely.from_wkb(g)), mcrs, dcrs); ids = [int(x) for x in f[2]]
    lines = []
    for lyr in ("Raised_ridges", "Ruptured_ridges"):
        _, _, gl, _ = pyogrio.raw.read(GDB, layer=lyr)
        lines += reproj(shapely.force_2d(shapely.from_wkb(gl)), mcrs, dcrs)
    sites = [(i, p) for i, p in zip(ids, pts) if box.contains(p)]
    sites.sort()
    x = np.arange(-HALF_M, HALF_M + STEP_M, STEP_M)
    along = np.arange(-SWATH_M, SWATH_M + 1, 2.0)
    rows, profiles = [], {}
    for sid, pt in sites:
        cx, cy = pt.x, pt.y
        z, ztr = read_window(DTM, dcrs, shift, cx, cy, 1.0)
        o, otr = read_window(ORTHO, dcrs, shift, cx, cy, 1.0)
        theta, dline = strike_from_lines(pt, lines)
        src = "ridge polyline"
        if theta is None:
            theta = strike_from_pca(z, ztr, cx, cy); src = "PCA of local highs"
        nx, ny = -np.sin(theta), np.cos(theta)              # unit normal to strike
        sx, sy = np.cos(theta), np.sin(theta)
        Z = np.full((along.size, x.size), np.nan); O = np.full_like(Z, np.nan)
        for k, a in enumerate(along):
            xs = cx + a * sx + x * nx; ys = cy + a * sy + x * ny
            Z[k] = sample(z, ztr, xs, ys); O[k] = sample(o, otr, xs, ys)
        zp = np.nanmedian(Z, axis=0); op = np.nanmedian(O, axis=0)
        far = (np.abs(x) >= FAR[0]) & (np.abs(x) <= FAR[1]) & np.isfinite(zp)
        if far.sum() < 40:
            continue
        coef = np.polyfit(x[far], zp[far], 1)
        rel = zp - np.polyval(coef, x)
        far_rms = float(np.std(rel[far]))
        core = np.abs(x) <= CORE
        fl_l = (x <= -FLANK[0]) & (x >= -FLANK[1]); fl_r = (x >= FLANK[0]) & (x <= FLANK[1])
        ofar = np.nanmedian(op[far]); osd = np.nanstd(op[far])
        bright = np.isfinite(op) & (op > ofar + 1.5 * osd)
        # bright-core half width: extent of contiguous bright pixels around x = 0
        hw = 0.0
        if bright[np.argmin(np.abs(x))]:
            i0 = np.argmin(np.abs(x)); l = i0; r = i0
            while l > 0 and bright[l - 1]: l -= 1
            while r < x.size - 1 and bright[r + 1]: r += 1
            hw = float((x[r] - x[l]) / 2)
        rows.append({"site_id": sid, "x": cx, "y": cy, "strike_deg": float(np.degrees(theta)) % 180, "strike_source": src,
                     "dist_to_ridge_line_m": float(dline), "core_rel_max_m": float(np.nanmax(rel[core])), "core_rel_mean_m": float(np.nanmean(rel[core])),
                     "flank_left_rel_mean_m": float(np.nanmean(rel[fl_l])), "flank_right_rel_mean_m": float(np.nanmean(rel[fl_r])),
                     "far_rms_m": far_rms, "bright_core_halfwidth_m": hw, "ortho_core_over_far": float(np.nanmedian(op[core]) / ofar) if ofar else np.nan})
        profiles[sid] = (x, rel, op / ofar if ofar else op, theta, z, ztr, o, otr, (cx, cy))
        print(sid, src, f"strike {np.degrees(theta) % 180:.0f} deg; core +{np.nanmax(rel[core]):.1f} m; flanks L {np.nanmean(rel[fl_l]):+.1f} R {np.nanmean(rel[fl_r]):+.1f} m; far rms {far_rms:.1f} m; bright hw {hw:.0f} m")

    df = pd.DataFrame(rows); df.to_csv(OUT / "ridge_profiles_metrics.csv", index=False)
    # gallery: profiles
    n = len(profiles); ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.2 * nrow), sharex=True)
    for ax, (sid, (x, rel, on, theta, *_)) in zip(axes.ravel(), profiles.items()):
        ax.plot(x, rel, color="#0072B2", lw=1.5, label="elevation (detrended)")
        ax2 = ax.twinx(); ax2.plot(x, on, color="#E69F00", lw=1.0, alpha=0.8, label="ortho brightness / far")
        ax2.set_ylim(0.6, 1.8); ax2.tick_params(labelsize=7)
        ax.axvspan(-CORE, CORE, color="0.92", zorder=0)
        for s in (-1, 1):
            ax.axvspan(s * FLANK[0], s * FLANK[1], color="#fde9c9", zorder=0)
        ax.set_title(f"site {sid}  strike {np.degrees(theta) % 180:.0f} deg", fontsize=9); ax.grid(True, color="0.93"); ax.tick_params(labelsize=7)
        ax.set_ylabel("rel. elev. (m)", fontsize=7)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("distance from ridge axis (m)", fontsize=8)
    fig.suptitle("Swath-median HiRISE DTM profiles across catalogued intrusions (grey: core +-15 m; orange: flank bands 20-40 m); brightness in orange line", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig_ridge_profiles_gallery.png", dpi=150); fig.savefig(OUT / "fig_ridge_profiles_gallery.pdf"); plt.close(fig)
    # gallery: context maps (ortho with profile line)
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 4.6 * nrow))
    for ax, (sid, (x, rel, on, theta, z, ztr, o, otr, (cx, cy))) in zip(axes.ravel(), profiles.items()):
        ext = (otr.c, otr.c + otr.a * o.shape[1], otr.f + otr.e * o.shape[0], otr.f)
        ax.imshow(o, cmap="gray", extent=ext, vmin=np.nanpercentile(o, 2), vmax=np.nanpercentile(o, 98))
        nx, ny = -np.sin(theta), np.cos(theta)
        ax.plot([cx - HALF_M * nx, cx + HALF_M * nx], [cy - HALF_M * ny, cy + HALF_M * ny], color="#D55E00", lw=1.2)
        ax.plot(cx, cy, "o", color="#0072B2", ms=4); ax.set_title(f"site {sid}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle("HiRISE RED orthoimage context (0.25 m, shown at 1 m) with profile line", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "fig_ridge_context_gallery.png", dpi=120); plt.close(fig)
    log = {"script_md5": md5(Path(__file__)), "dtm": str(DTM.relative_to(ROOT)), "ortho": str(ORTHO.relative_to(ROOT)), "shift_applied": off,
           "params": {"half_m": HALF_M, "swath_m": SWATH_M, "far": FAR, "flank": FLANK, "core": CORE, "line_search_m": LINE_SEARCH_M},
           "n_sites": n, "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat()}
    (OUT / "run_log.json").write_text(json.dumps(log, indent=2, default=str))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
