"""Alignment panels for the two spectrum sites: CTX image vs HiRISE DEM.

For each site a 500 x 500 m box centred on the given lon/lat (planetocentric,
east-positive) is built on a fine grid in the project working CRS (the MTR3
GeoTIFF CRS = the frame the Robbins controlled CTX mosaic lives in). Layers:
- CTX: Robbins 6 m controlled mosaic, streamed via /vsicurl (the frame
  reference; no shift).
- DEM: Adriano's 0.8 m ASP DEM of the PSP_006712/005855 pair (worldfile
  georef + measured CTX shift), PDS DTEEC as a cross-check; where no HiRISE
  DEM covers the site the CTX 20 m DEM is used and labelled as such.
- HiRISE RED: PDS 0.25 m ortho with the measured CTX shift.
Panels: (a) CTX with DEM contours, (b) DEM hillshade, (c) DEM colour +
contours over hillshade, (d) HiRISE RED ortho (or placeholder). A CaSSIS
pansharpened panel is to be added later. Warped layers are saved as .npz
for the profile step. Deterministic.

Usage:  python code/contact_site_panels.py [site ...]
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds, transform as win_transform
from pyproj import CRS as PCRS, Transformer

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
from contact_coreg_frame import CTX_URL, ctx_on_grid  # noqa: E402
from contact_coreg_lib import coarse_lock, fit_offsets, highpass, tile_offsets  # noqa: E402
from contact_dtm_registry import registry, open_meta, FRAME  # noqa: E402
from contact_height_above_contact import CTX_DEM, md5  # noqa: E402
from figstyle import safe_savefig  # noqa: E402

SITES = {"site1": {"lon": 77.1370253, "lat": 21.9842103, "label": "site 1 (21.984 N, 77.137 E)",
                   # AV's crop of the polygonal-fracture band (2026-09-08), located in the previous
                   # 500 x 400 m box by template matching (NCC 0.98): 300 x 273 m window centred
                   # 174 m W, 290 m S of the spectrum point; no CTX panel -> 2 x 2 layout
                   # 2026-09-09: widened 60 m to the west so the Fig. 7 intrusion-1 polygons (dike map.shp /
                   # amorphous matrix.shp, CaSSIS-corrected) fall fully inside -> 380 x 270 m, centre 214 m W, 290 m S
                   "center_offset_m": (-214.0, -290.0), "box_wh_m": (380.0, 270.0),
                   "panels": ("hs", "resid", "slope", "ortho")},
         # paper-figure window: the QGIS panel footprint located in our DEM by template matching
         # (NCC 0.78, scale plateau 840-920 m, centre 225 m W / 305 m S of the spectrum); then
         # (AV, 2026-09-09) zoomed in to 780 m and pushed 60 m W so the pansharpened CaSSIS
         # strip (whose east edge lies 172 m E of the spectrum) fills the whole view
         "site1_wide": {"lon": 77.1370253, "lat": 21.9842103, "label": "site 1 wide (21.984 N, 77.137 E)",
                        "center_offset_m": (-285.0, -305.0), "box_wh_m": (780.0, 780.0),
                        "panels": ("hs_color", "slope"), "scalebar_m": 150.0},
         "site2": {"lon": 77.1467583, "lat": 22.3070500, "label": "site 2 (22.307 N, 77.147 E)",
                   # zoomed out (AV, 2026-09-08): 1200 m box on a 2 m grid (only CTX 6 m / CTX DEM 20 m exist here)
                   "box_wh_m": (1200.0, 1200.0), "grid_m": 2.0, "panels": ("ctx", "hs", "resid", "slope")}}
DEFAULT_PANELS = ("ctx", "hs", "slope", "resid", "ortho", "cassis")
ORTHO_GRID_M = 0.25            # HiRISE RED ortho is read on its own finer grid for display
BOX_M = 500.0
GRID_M = 0.5
OUT = ROOT / "outputs" / "revision" / "sites"
OUT.mkdir(parents=True, exist_ok=True)
WORK_TIF = sorted((ROOT / "outputs" / "_v8_mtr3_with_brown" / "FRT00003E12" / "geotiffs").glob("*.tif"))[0]
DEM_PRIORITY = ("ADI_006712", "PDS_002888", "ADI_027691", "PDS_006712")
ORTHO_PRIORITY = ("PDS_006712", "PDS_002888", "ADI_027691", "ADI_006712")
MARGIN_M = 60.0
HS_ALT, HS_VEXAG, HS_AZS = 30.0, 2.0, (315.0, 45.0, 0.0)   # low sun, 2x vertical exaggeration, multi-azimuth
HS_SMOOTH_M = 1.5              # Gaussian sigma before hillshading (suppresses ASP correlation striping)
RESID_CONTOUR_M = 1.0          # contour interval on the plane-detrended elevation panel
SLOPE_SMOOTH_M = 2.0           # Gaussian sigma before the slope computation
CONTOURS = False               # contour overlays off everywhere (AV, 2026-09-09)


def working_crs():
    with rasterio.open(WORK_TIF) as ds:
        return ds.crs


def lonlat_to_xy(lon, lat, crs):
    geo = PCRS.from_proj4(crs.to_proj4()).geodetic_crs
    return Transformer.from_crs(geo, crs, always_xy=True).transform(lon, lat)


def covers(entry, which, x, y, crs) -> bool:
    path, tr, scrs, nod = open_meta(entry, which)
    with rasterio.open(path) as ds:
        w, h = ds.width, ds.height
    sx, sy = transform_bounds(crs, scrs, x, y, x, y)[:2]
    col, row = ~tr * (sx, sy)
    return 0 <= col < w and 0 <= row < h


def read_on_grid(path, src_tr, src_crs, nodata, dst_crs, dst_tr, dst_shape, resampling) -> np.ndarray:
    H, W = dst_shape
    left, top = dst_tr * (0, 0); right, bottom = dst_tr * (W, H)
    b = transform_bounds(dst_crs, src_crs, min(left, right), min(top, bottom), max(left, right), max(top, bottom), densify_pts=21)
    win = from_bounds(b[0] - MARGIN_M, b[1] - MARGIN_M, b[2] + MARGIN_M, b[3] + MARGIN_M, transform=src_tr).round_offsets().round_lengths()
    with rasterio.open(path) as ds:
        win = win.intersection(rasterio.windows.Window(0, 0, ds.width, ds.height))
        src = ds.read(1, window=win).astype(np.float32)
        nd = ds.nodata if nodata is None else nodata
    if nd is not None:
        src[src == nd] = np.nan
    src[(src < -1e30) | (src > 1e30)] = np.nan
    dst = np.full(dst_shape, np.nan, np.float32)
    reproject(src, dst, src_transform=win_transform(win, src_tr), src_crs=src_crs, src_nodata=np.nan,
              dst_transform=dst_tr, dst_crs=dst_crs, dst_nodata=np.nan, resampling=resampling)
    return dst


def hillshade(z, cell, az=315.0, alt=45.0, vert_exag=1.0):
    ls = LightSource(azdeg=az, altdeg=alt)
    zz = np.where(np.isfinite(z), z, np.nanmean(z))
    return ls.hillshade(zz, vert_exag=vert_exag, dx=cell, dy=cell)


def multi_hillshade(z, cell, alt=HS_ALT, vert_exag=HS_VEXAG, azs=HS_AZS, smooth_m=None):
    """Mean of hillshades from several azimuths, contrast-stretched to the 1-99 % range."""
    from scipy.ndimage import gaussian_filter
    sm = HS_SMOOTH_M if smooth_m is None else smooth_m
    zs = gaussian_filter(np.where(np.isfinite(z), z, np.nanmean(z)), sm / cell) if sm > 0 else z
    h = np.mean([hillshade(zs, cell, az, alt, vert_exag) for az in azs], axis=0)
    lo, hi = np.nanpercentile(h, (1, 99))
    return np.clip((h - lo) / max(hi - lo, 1e-6), 0, 1)


def detrend_plane(z):
    """Residual elevation after removing the best-fit plane (local structure only)."""
    H, W = z.shape
    rr, cc = np.mgrid[0:H, 0:W]
    ok = np.isfinite(z)
    A = np.column_stack([rr[ok], cc[ok], np.ones(ok.sum())])
    coef, *_ = np.linalg.lstsq(A, z[ok], rcond=None)
    return z - (coef[0] * rr + coef[1] * cc + coef[2])


def contour_interval(z):
    rng = np.nanpercentile(z, 99) - np.nanpercentile(z, 1)
    raw = rng / 12.0
    for step in (0.5, 1, 2, 5, 10, 20):
        if raw <= step:
            return step
    return 50


def slope_deg(z, cell, smooth_m=None):
    """Slope in degrees after a light Gaussian smoothing (suppresses stereo-matcher speckle)."""
    from scipy.ndimage import gaussian_filter
    s = SLOPE_SMOOTH_M if smooth_m is None else smooth_m
    zz = np.where(np.isfinite(z), z, np.nanmean(z))
    zs = gaussian_filter(zz, s / cell) if s > 0 else zz
    gy, gx = np.gradient(zs, cell)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def scalebar(ax, nx, ny, cell, length_m=100.0, color="white"):
    px = length_m / cell
    x0, y0 = nx * 0.05, ny * 0.94
    ax.plot([x0, x0 + px], [y0, y0], color=color, lw=3, solid_capstyle="butt")
    ax.text(x0 + px / 2, y0 - ny * 0.015, f"{length_m:.0f} m", color=color, ha="center", va="bottom", fontsize=8, fontweight="bold")


CHECK_BOX_M, CHECK_RES_M, CHECK_HP_M = 2000.0, 6.0, 110.0
LOCAL_REFINE = True            # apply the measured local residual on top of the frame shift


def local_residual(xc, yc, crs, entry, which) -> dict:
    """Residual CTX-vs-HiRISE offset (after the registry shift) in a 2 km window, 6 m grid."""
    n = int(CHECK_BOX_M / CHECK_RES_M)
    tr6 = Affine(CHECK_RES_M, 0, xc - CHECK_BOX_M / 2, 0, -CHECK_RES_M, yc + CHECK_BOX_M / 2)
    ref, refv = ctx_on_grid(crs, tr6, (n, n))
    path, str_, scrs, nod = open_meta(entry, which)
    mov = read_on_grid(path, str_, scrs, nod, crs, tr6, (n, n), Resampling.average)
    mov[mov <= 0] = np.nan
    movv = np.isfinite(mov)
    sig = CHECK_HP_M / CHECK_RES_M
    rh, mh = highpass(ref, refv, sig), highpass(mov, movv, sig)
    coarse = coarse_lock(rh, mh, refv, movv)
    fit = fit_offsets(tile_offsets(rh, mh, refv, movv, coarse, 64, 32))
    if fit.get("ok"):
        fit["residual_dx_m"] = fit["dcol"] * CHECK_RES_M; fit["residual_dy_m"] = -fit["drow"] * CHECK_RES_M
        fit["residual_mad_m"] = [fit["mad_col"] * CHECK_RES_M, fit["mad_row"] * CHECK_RES_M]
    fit["coarse_px"] = coarse
    return fit


def run_site(name: str) -> None:
    t0 = datetime.now(timezone.utc)
    site = SITES[name]; crs = working_crs()
    xs, ys = lonlat_to_xy(site["lon"], site["lat"], crs)            # spectrum location
    ox, oy = site.get("center_offset_m", (0.0, 0.0))
    xc, yc = xs + ox, ys + oy                                        # scene centre
    bw, bh = site.get("box_wh_m", (BOX_M, BOX_M))
    g = float(site.get("grid_m", GRID_M))
    nx, ny = int(round(bw / g)), int(round(bh / g))
    tr = Affine(g, 0, xc - bw / 2, 0, -g, yc + bh / 2)
    shape = (ny, nx)
    reg = registry(); log = {"site": name, **site, "x_spectrum_m": xs, "y_spectrum_m": ys, "x_center_m": xc, "y_center_m": yc,
                             "crs": crs.to_string(), "grid_m": g, "box_wh_m": (bw, bh)}
    pc, pr = ~tr * (xs, ys)                                          # spectrum location in grid pixels

    ctx, _ = ctx_on_grid(crs, tr, shape)
    layers = {"ctx": ctx}
    dem_key = next((k for k in DEM_PRIORITY if covers(reg[k], "dem", xc, yc, crs)), None)
    ortho_key = next((k for k in ORTHO_PRIORITY if covers(reg[k], "ortho", xc, yc, crs)), None)

    # local residual alignment of each HiRISE product vs CTX (registry shift already applied);
    # the residual is applied as a site-local refinement when LOCAL_REFINE is on
    checks, refine = {}, {}
    for key in {k for k in (dem_key, ortho_key) if k}:
        if reg[key]["ortho"].exists():
            fit = local_residual(xc, yc, crs, reg[key], "ortho")
            checks[key] = fit
            if LOCAL_REFINE and fit.get("ok") and fit["conf_median"] >= 0.5:
                refine[key] = (fit["residual_dx_m"], fit["residual_dy_m"])
            print(f"  CTX vs {key} ortho: residual dx {fit.get('residual_dx_m', float('nan')):+.1f} m, dy {fit.get('residual_dy_m', float('nan')):+.1f} m "
                  f"(MAD {fit.get('residual_mad_m')}, n_good {fit.get('n_good')}, conf {fit.get('conf_median', float('nan')):.2f})")
    log["residual_checks"] = checks; log["local_refine_applied_m"] = refine

    def shifted(key, which):
        path, str_, scrs, nod = open_meta(reg[key], which)
        return path, Affine.translation(*refine.get(key, (0.0, 0.0))) * str_, scrs, nod

    def shift_txt(key):
        s = reg[key]["shift"]; r = refine.get(key, (0.0, 0.0))
        return f"frame shift {s[0]:+.1f}/{s[1]:+.1f} m, local refinement {r[0]:+.1f}/{r[1]:+.1f} m"

    if dem_key:
        path, str_, scrs, nod = shifted(dem_key, "dem")
        dem = read_on_grid(path, str_, scrs, nod, crs, tr, shape, Resampling.bilinear)
        dem_label = f"HiRISE DEM {dem_key} ({reg[dem_key]['quality']})\n{shift_txt(dem_key)}"
        dem_cell_native = abs(str_.a)
        alt_key = next((k for k in DEM_PRIORITY if k != dem_key and covers(reg[k], "dem", xc, yc, crs)), None)
        if alt_key:
            p2, t2, c2, n2 = open_meta(reg[alt_key], "dem")
            layers["dem_alt"] = read_on_grid(p2, t2, c2, n2, crs, tr, shape, Resampling.bilinear)
            log["dem_alt"] = alt_key
    else:
        with rasterio.open(CTX_DEM) as ds:
            dem = read_on_grid(CTX_DEM, ds.transform, ds.crs, ds.nodata, crs, tr, shape, Resampling.bilinear)
        dem_label = "CTX 20 m DEM (no HiRISE DTM covers this site)"; dem_cell_native = 20.0
    layers["dem"] = dem; log["dem"] = dem_key or "CTX_DEM"; log["dem_label"] = dem_label

    if ortho_key:
        path, str_, scrs, nod = shifted(ortho_key, "ortho")
        fo = max(1, int(round(g / ORTHO_GRID_M)))
        tr_o = tr * Affine.scale(1.0 / fo)
        ortho = read_on_grid(path, str_, scrs, nod, crs, tr_o, (ny * fo, nx * fo), Resampling.average)
        ortho[ortho <= 0] = np.nan
        layers["ortho"] = ortho; log["ortho_grid_m"] = ORTHO_GRID_M
        ortho_label = f"HiRISE RED ortho {ortho_key} ({abs(str_.a):.2f} m)\n{shift_txt(ortho_key)}"
    else:
        ortho = None; ortho_label = "no HiRISE ortho covers this site"
    log["ortho"] = ortho_key; log["ortho_label"] = ortho_label

    np.savez_compressed(OUT / f"{name}_layers.npz", transform=np.array(tr.to_gdal()), **{k: v for k, v in layers.items() if v is not None})

    # ---- figure
    hs_sm = max(HS_SMOOTH_M, dem_cell_native)          # never smooth less than one native DEM cell
    sl_sm = max(SLOPE_SMOOTH_M, dem_cell_native)
    hs = multi_hillshade(dem, g, smooth_m=hs_sm)
    resid = detrend_plane(dem)
    ci = contour_interval(dem)
    lv = np.arange(np.floor(np.nanmin(dem) / ci) * ci, np.nanmax(dem) + ci, ci)
    cir = max(RESID_CONTOUR_M, contour_interval(resid))   # 1 m unless the local relief is large
    lvr = np.arange(np.floor(np.nanmin(resid) / cir) * cir, np.nanmax(resid) + cir, cir)
    slope = slope_deg(dem, g, smooth_m=sl_sm)
    ext = (0, nx, ny, 0)
    XC, YC = np.arange(nx) + 0.5, np.arange(ny) + 0.5     # pixel centres for contour(): row 0 must sit at the TOP like imshow
    panels = site.get("panels", DEFAULT_PANELS)
    ncol = len(panels) if len(panels) <= 2 else (2 if len(panels) <= 4 else 3)
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, (6.2 * ny / nx + 1.0) * nrow), squeeze=False)
    letters = "abcdefgh"
    dem_short = dem_label.splitlines()[0].split(" (")[0]
    hs_txt = f"az {'/'.join(f'{a_:.0f}' for a_ in HS_AZS)}, alt {HS_ALT:.0f}, {HS_VEXAG:.0f}x, {hs_sm:g} m pre-smoothing"
    for ax, key, let in zip(axes.ravel(), panels, letters):
        if key == "ctx":
            p1, p99 = np.nanpercentile(ctx, (1, 99))
            ax.imshow(ctx, cmap="gray", vmin=p1, vmax=p99, extent=ext, interpolation="bilinear")
            if CONTOURS:
                ax.contour(XC, YC, dem, levels=lv, colors="#ffd700", linewidths=0.5, alpha=0.8)
            ax.set_title(f"({let}) CTX 6 m controlled mosaic (frame reference)" + (f"\n+ DEM contours every {ci:g} m" if CONTOURS else ""), fontsize=10)
        elif key == "hs":
            ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=ext)
            ax.set_title(f"({let}) multi-azimuth hillshade of {dem_short}\n({hs_txt}; native cell {dem_cell_native:.2f} m)", fontsize=9)
        elif key == "hs_color":
            ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=ext)
            v1, v99 = np.nanpercentile(dem, (0.5, 99.5))
            im = ax.imshow(dem, cmap="turbo", alpha=0.55, extent=ext, vmin=v1, vmax=v99)
            fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label="elevation (m, DEM datum)")
            ax.set_title(f"({let}) {dem_label}\nhillshade ({hs_txt}) + elevation colour", fontsize=8.5)
        elif key == "resid":
            ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=ext)
            vr = np.nanpercentile(np.abs(resid), 98)
            im = ax.imshow(resid, cmap="RdBu_r", alpha=0.6, extent=ext, vmin=-vr, vmax=vr)
            if CONTOURS:
                ax.contour(XC, YC, resid, levels=lvr, colors="k", linewidths=0.35, alpha=0.7)
            fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label="elevation residual from best-fit plane (m)")
            ax.set_title(f"({let}) {dem_label}\nplane-detrended elevation" + (f", contours every {cir:g} m" if CONTOURS else "") +
                         f" (raw range {np.nanmin(dem):.0f} to {np.nanmax(dem):.0f} m)", fontsize=8.5)
        elif key == "slope":
            smax = np.nanpercentile(slope, 98)
            im = ax.imshow(slope, cmap="magma", vmin=0, vmax=smax, extent=ext)
            fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label="slope (deg)")
            if CONTOURS:
                ax.contour(XC, YC, resid, levels=lvr, colors="w", linewidths=0.3, alpha=0.5)
            ax.set_title(f"({let}) slope of the {dem_short} ({sl_sm:g} m Gaussian pre-smoothing)" +
                         (f"\n+ detrended-elevation contours every {cir:g} m" if CONTOURS else ""), fontsize=9)
        elif key == "ortho":
            if ortho is not None:
                q1, q99 = np.nanpercentile(ortho, (1, 99))
                ax.imshow(ortho, cmap="gray", vmin=q1, vmax=q99, extent=ext, interpolation="bilinear")
                if CONTOURS:
                    ax.contour(XC, YC, dem, levels=lv, colors="#ffd700", linewidths=0.5, alpha=0.8)
                ax.set_title(f"({let}) {ortho_label}" + (f"\n+ DEM contours every {ci:g} m" if CONTOURS else ""), fontsize=8.5)
            else:
                ax.text(0.5, 0.5, "HiRISE RED\n(none covers this site)", ha="center", va="center", fontsize=12, transform=ax.transAxes)
                ax.set_title(f"({let}) placeholder", fontsize=10)
        elif key == "cassis":
            ax.text(0.5, 0.5, "CaSSIS pansharpened\n(to be added)", ha="center", va="center", fontsize=12, transform=ax.transAxes)
            ax.set_title(f"({let}) placeholder", fontsize=10)
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    axes = axes.ravel()[:len(panels)]
    arm = 25 / g
    inside = 0 <= pc < nx and 0 <= pr < ny
    for ax in axes:
        if inside:
            ax.plot([pc - arm, pc + arm], [pr, pr], color="red", lw=1.2)
            ax.plot([pc, pc], [pr - arm, pr + arm], color="red", lw=1.2)
        else:                                                        # spectrum location off the box: arrow toward it
            ec, er = np.clip(pc, 0, nx), np.clip(pr, 0, ny)
            dist = np.hypot(pc - ec, pr - er) * g
            ux, uy = (pc - ec) / max(np.hypot(pc - ec, pr - er), 1e-9), (pr - er) / max(np.hypot(pc - ec, pr - er), 1e-9)
            L = 70 / g
            ax.annotate("", xy=(ec, er), xytext=(ec - ux * L, er - uy * L),
                        arrowprops=dict(arrowstyle="-|>", color="red", lw=1.5))
            ax.text(ec - ux * L * 1.15, er - uy * L * 1.15, f"spectrum\n{dist:.0f} m", color="red", ha="center", va="center",
                    fontsize=8, fontweight="bold", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6))
        ax.set_xlim(0, nx); ax.set_ylim(ny, 0)
        ax.set_xticks([]); ax.set_yticks([])
        scalebar(ax, nx, ny, g, length_m=site.get("scalebar_m", 100.0 if bw <= 600 else 200.0))
        ax.annotate("N", xy=(nx * 0.94, ny * 0.06), xytext=(nx * 0.94, ny * 0.16), color="white", ha="center", fontsize=9, fontweight="bold",
                    arrowprops=dict(arrowstyle="-|>", color="white", lw=1.5))
    res_txt = "; ".join(f"{k}: residual vs CTX {v.get('residual_dx_m', float('nan')):+.1f}/{v.get('residual_dy_m', float('nan')):+.1f} m "
                        f"(MAD {v.get('residual_mad_m', [float('nan')] * 2)[0]:.1f}/{v.get('residual_mad_m', [float('nan')] * 2)[1]:.1f} m)"
                        for k, v in checks.items()) or "no HiRISE product to check"
    off_txt = f" (centre {abs(ox):.0f} m {'W' if ox < 0 else 'E'}, {abs(oy):.0f} m {'S' if oy < 0 else 'N'} of the spectrum)" if (ox or oy) else ""
    fig.suptitle(f"{site['label']}: {bw:.0f} x {bh:.0f} m box{off_txt}, grid north up, red cross = spectrum location; "
                 f"all layers in the CTX (Robbins) frame\n{res_txt}"
                 f"{' - refined locally' if refine else ''}", fontsize=10)
    fig.tight_layout()
    safe_savefig(fig, OUT / f"fig_{name}_panels", bbox_inches="tight")
    plt.close(fig)
    log.update({"contour_interval_m": ci, "dem_range_m": [float(np.nanmin(dem)), float(np.nanmax(dem))],
                "ctx_url": CTX_URL, "script": Path(__file__).name, "script_md5": md5(Path(__file__)),
                "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat()})
    (OUT / f"run_log_{name}.json").write_text(json.dumps(log, indent=2, default=str))
    print(name, "dem:", dem_key or "CTX_DEM", "ortho:", ortho_key, "range", np.nanmin(dem), np.nanmax(dem), "ci", ci)


def main() -> None:
    for s in (sys.argv[1:] or list(SITES)):
        run_site(s)


if __name__ == "__main__":
    main()
