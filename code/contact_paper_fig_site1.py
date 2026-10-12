"""Regenerate the paper's four-panel site figure (CaSSIS colour / geologic map /
OLINDEX / HiRISE DEM) in the project's own frame, A4 full-page-width layout.

Window: contact_site_panels.SITES['site1_wide'] (780 x 780 m, centred 285 m W
and 305 m S of the site-1 spectrum point; the QGIS panel footprint located by
template matching, then zoomed and shifted so the pansharpened CaSSIS strip
fills the view). All layers in the CTX (Robbins) frame, 0.5 m grid, north up.
(a) CaSSIS pansharpened NIR-PAN-BLU (GSA_MY36_014433_022_0, 0.29 m), PAN
    registered to CTX by phase correlation; white box = the next figure
    (the 300 x 270 m fracture-band window).
(b) Geologic map from the mapping gdb: olivine unit orange as the base,
    clay basement blue on top; Mafic_cap not drawn (AV, 2026-09-09).
(c) OLINDEX2 (default) or OLINDEX3 (--param OLINDEX3) computed by
    contact_olindex_compare.py from the FRT0000A4FC MTR3 cube, CRISM-to-CTX
    frame shift applied, native 18 m pixels, black = 0; olivine outline white.
(d) HiRISE DEM (ADI_006712) hillshade + elevation colour, frame shift and
    site-local refinement; olivine outline white.
Panel letters and unit labels in the corners; technical details go to the
caption file fig_paper_site1_caption.txt and the run log. Deterministic.

Usage:  python code/contact_paper_fig_site1.py [--param OLINDEX2|OLINDEX3] [--slope]
"""
from __future__ import annotations

import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import shapely
import shapely.affinity
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy.ndimage import label as cc_label, center_of_mass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from contact_site_panels import (SITES, working_crs, lonlat_to_xy, read_on_grid, multi_hillshade, slope_deg,  # noqa: E402
                                  local_residual, HS_SMOOTH_M, SLOPE_SMOOTH_M)
from contact_coreg_frame import ctx_on_grid  # noqa: E402
from contact_coreg_lib import coarse_lock, fit_offsets, highpass, tile_offsets  # noqa: E402
from contact_dtm_registry import registry, open_meta  # noqa: E402
from contact_geomap_fractions import load_units  # noqa: E402
from contact_height_above_contact import md5  # noqa: E402
from figstyle import safe_savefig  # noqa: E402

OUT = ROOT / "outputs" / "revision" / "sites"
WIDE, CLOSE = SITES["site1_wide"], SITES["site1"]
GRID_M = 0.5
from config import CASSIS_RAW_DIR, pansharp_file  # noqa: E402
CASSIS_CUBE = pansharp_file("GSA_MY36_014433_022_0_nodatacorr.tif")      # HiRISE-pansharpened CaSSIS, 0.29 m
CASSIS_FILL = CASSIS_RAW_DIR / "MY37_025236_157_0.cubeit.cub"
CASSIS_RGB = ("NIR", "PAN", "BLU")
CRISM_CUBE = "FRT0000A4FC"
FRAME_JSON = ROOT / "outputs" / "revision" / "contact_3d" / "frame" / "coreg_offsets.json"
UNIT_COLOURS = {"oliv": "#e07820", "clay": "#2880a8"}          # (b) of the paper figure
DRAW_CAP = False                                               # Mafic_cap not drawn: olivine shows through (AV, 2026-09-09)
CHECK_BOX_M, CHECK_RES_M, CHECK_HP_M = 3000.0, 6.0, 110.0
MANUAL_CRISM_SHIFT_M = (0.0, 0.0)       # was -18 m while the map sat in the pansharpened file frame; with the map corrected (MAP_IN_CASSIS_FRAME) no manual shift is needed
MAP_IN_CASSIS_FRAME = True
OLINDEX_STRETCH_PCT = (2.0, 99.5)       # display stretch of (c): 2nd-99.5th percentile in the window (AV, 2026-09-10: more olivine/clay contrast)
GREEN_BOX_M = ((544.0, 408.0), (772.0, 466.0))   # (c): AV's box (screenshot located by template matching, 2026-09-09), window metres from W / N edge              # gdb units were digitised on the pansharpened product: apply its CaSSIS-to-CTX correction (check_fig7_clay_registration.png)
# --- A4 full-width layout
FIG_W_IN = 7.3
FS_LETTER, FS_UNIT, FS_SCALE, FS_CB = 16, 14, 13, 11
STROKE = [pe.withStroke(linewidth=3, foreground="black")]


def residual_vs_ctx(path, band, xc, yc, crs) -> dict:
    """Phase-correlation residual of a raster band vs the CTX mosaic (6 m, 3 km window)."""
    n = int(CHECK_BOX_M / CHECK_RES_M)
    tr6 = Affine(CHECK_RES_M, 0, xc - CHECK_BOX_M / 2, 0, -CHECK_RES_M, yc + CHECK_BOX_M / 2)
    ref, refv = ctx_on_grid(crs, tr6, (n, n))
    with rasterio.open(path) as ds:
        src_tr, src_crs, nod = ds.transform, ds.crs, ds.nodata
    mov = read_band_on_grid(path, band, src_tr, src_crs, nod, crs, tr6, (n, n), Resampling.average,
                            decimate=max(1, int(CHECK_RES_M / abs(src_tr.a) / 2)))
    movv = np.isfinite(mov) & (mov > 0)
    sig = CHECK_HP_M / CHECK_RES_M
    rh, mh = highpass(ref, refv, sig), highpass(np.where(movv, mov, np.nan), movv, sig)
    coarse = coarse_lock(rh, mh, refv, movv)
    fit = fit_offsets(tile_offsets(rh, mh, refv, movv, coarse, 64, 32))
    if fit.get("ok"):
        fit["dx_m"] = fit["dcol"] * CHECK_RES_M; fit["dy_m"] = -fit["drow"] * CHECK_RES_M
        fit["mad_m"] = [fit["mad_col"] * CHECK_RES_M, fit["mad_row"] * CHECK_RES_M]
    fit["coarse_px"] = coarse
    return fit


def read_band_on_grid(path, band, src_tr, src_crs, nodata, dst_crs, dst_tr, dst_shape, resampling, decimate=1):
    """read_on_grid for a specific band index (1-based); decimate > 1 reads through overviews."""
    from rasterio.warp import reproject, transform_bounds
    from rasterio.windows import from_bounds, transform as win_transform
    H, W = dst_shape
    left, top = dst_tr * (0, 0); right, bottom = dst_tr * (W, H)
    b = transform_bounds(dst_crs, src_crs, min(left, right), min(top, bottom), max(left, right), max(top, bottom), densify_pts=21)
    win = from_bounds(b[0] - 60, b[1] - 60, b[2] + 60, b[3] + 60, transform=src_tr).round_offsets().round_lengths()
    with rasterio.open(path) as ds:
        win = win.intersection(rasterio.windows.Window(0, 0, ds.width, ds.height))
        oshape = (max(1, int(win.height // decimate)), max(1, int(win.width // decimate)))
        src = ds.read(band, window=win, out_shape=oshape, resampling=Resampling.average).astype(np.float32)
        scale = ds.scales[band - 1] if ds.dtypes[0] == "uint16" else 1.0
    if nodata is not None:
        src[src == nodata] = np.nan
    src *= np.float32(scale)
    src[(src < -1e30) | (src > 1e30)] = np.nan
    wtr = win_transform(win, src_tr) * Affine.scale(win.width / oshape[1], win.height / oshape[0])
    dst = np.full(dst_shape, np.nan, np.float32)
    reproject(src, dst, src_transform=wtr, src_crs=src_crs, src_nodata=np.nan,
              dst_transform=dst_tr, dst_crs=dst_crs, dst_nodata=np.nan, resampling=resampling)
    return dst


def stretch(a, lo=2, hi=98):
    p1, p2 = np.nanpercentile(a, (lo, hi))
    return np.clip((a - p1) / max(p2 - p1, 1e-9), 0, 1)


def unit_labels(ax, lab, colour="white", avoid=None):
    """'Olivine' / 'Clay' at the deepest interior point (max distance to any boundary or the frame edge)
    of the largest patch of each unit, so the label never sits on a margin."""
    from scipy.ndimage import distance_transform_edt
    for code, text in ((1, "Olivine"), (3, "Clay")):
        m = lab == code
        if m.mean() < 0.02:
            continue
        cc, _ = cc_label(m)
        big = np.argmax(np.bincount(cc.ravel())[1:]) + 1
        patch = cc == big
        if avoid is not None:
            patch = patch & ~avoid                            # e.g. keep clear of the next-figure box
        patch = np.pad(patch, 1)                              # frame edge counts as a boundary
        d = distance_transform_edt(patch)[1:-1, 1:-1]
        r0, c0 = np.unravel_index(np.argmax(d), d.shape)
        ax.text(c0, r0, text, color=colour, ha="center", va="center", fontsize=FS_UNIT, fontweight="bold", path_effects=STROKE)


def inset_cbar(fig, ax, im, label):
    """Horizontal colourbar inside the panel (bottom right), white text with a black stroke."""
    cax = ax.inset_axes([0.56, 0.055, 0.40, 0.035])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=FS_CB - 1, colors="white", length=2, pad=1)
    for t in cb.ax.get_xticklabels():
        t.set_path_effects(STROKE); t.set_fontweight("bold")
    cb.outline.set_edgecolor("white"); cb.outline.set_linewidth(0.8)
    cax.set_title(label, color="white", fontsize=FS_CB, fontweight="bold", pad=2, path_effects=STROKE)
    return cb


def scalebar_big(ax, nx, ny, cell, length_m, color="white"):
    px = length_m / cell
    x0, y0 = nx * 0.05, ny * 0.95
    ax.plot([x0, x0 + px], [y0, y0], color=color, lw=4, solid_capstyle="butt",
            path_effects=[pe.withStroke(linewidth=6, foreground="black")] if color == "white" else None)
    ax.text(x0 + px / 2, y0 - ny * 0.015, f"{length_m:.0f} m", color=color, ha="center", va="bottom", fontsize=FS_SCALE, fontweight="bold",
            path_effects=STROKE if color == "white" else None)


def main() -> None:
    args = sys.argv[1:]
    param = args[args.index("--param") + 1] if "--param" in args else "OLINDEX2"
    want_slope = "--slope" in args
    t0 = datetime.now(timezone.utc)
    crs = working_crs()
    xs, ys = lonlat_to_xy(WIDE["lon"], WIDE["lat"], crs)
    ox, oy = WIDE["center_offset_m"]; xc, yc = xs + ox, ys + oy
    bw, bh = WIDE["box_wh_m"]
    nx, ny = int(round(bw / GRID_M)), int(round(bh / GRID_M))
    tr = Affine(GRID_M, 0, xc - bw / 2, 0, -GRID_M, yc + bh / 2); shape = (ny, nx)
    ext = (0, nx, ny, 0)
    XC, YC = np.arange(nx) + 0.5, np.arange(ny) + 0.5     # pixel centres; contour(XC, YC, Z) aligns with imshow(extent=ext)
    log = {"window": WIDE, "grid_m": GRID_M, "x_center_m": xc, "y_center_m": yc, "param": param}

    # ---- (a) CaSSIS pansharpened, registered to CTX
    with rasterio.open(CASSIS_CUBE) as ds:
        desc = list(ds.descriptions); c_tr, c_crs, c_nod = ds.transform, ds.crs, ds.nodata
    fit = residual_vs_ctx(CASSIS_CUBE, desc.index("PAN") + 1, xc, yc, crs)
    shift = (fit["dx_m"], fit["dy_m"]) if (fit.get("ok") and fit["conf_median"] >= 0.4) else (0.0, 0.0)
    print(f"CaSSIS {CASSIS_CUBE.name} PAN vs CTX: dx {fit.get('dx_m', float('nan')):+.1f} m dy {fit.get('dy_m', float('nan')):+.1f} m "
          f"(MAD {fit.get('mad_m')}, n_good {fit.get('n_good')}, conf {fit.get('conf_median', float('nan')):.2f}) -> applied {shift}")
    log["cassis"] = {"file": CASSIS_CUBE.name, "bands_rgb": CASSIS_RGB, "fit": fit, "shift_applied_m": shift}
    c_tr_eff = Affine.translation(*shift) * c_tr
    bands = [read_band_on_grid(CASSIS_CUBE, desc.index(b) + 1, c_tr_eff, c_crs, c_nod, crs, tr, shape, Resampling.bilinear) for b in CASSIS_RGB]
    gap = ~np.isfinite(np.dstack(bands)).all(-1)
    log["cassis"]["nodata_fraction_in_window"] = float(gap.mean())
    if gap.any():                                    # strip edge inside the window: fill from the registered 4 m cube, tone-matched
        with rasterio.open(CASSIS_FILL) as ds:
            fdesc = list(ds.descriptions); f_tr, f_crs, f_nod = ds.transform, ds.crs, ds.nodata
        ffit = residual_vs_ctx(CASSIS_FILL, fdesc.index("PAN") + 1, xc, yc, crs)
        fshift = (ffit["dx_m"], ffit["dy_m"]) if (ffit.get("ok") and ffit["conf_median"] >= 0.4) else (0.0, 0.0)
        f_tr_eff = Affine.translation(*fshift) * f_tr
        for i, b in enumerate(CASSIS_RGB):
            fb = read_band_on_grid(CASSIS_FILL, fdesc.index(b) + 1, f_tr_eff, f_crs, f_nod, crs, tr, shape, Resampling.bilinear)
            ov = np.isfinite(bands[i]) & np.isfinite(fb)
            a_, b_ = np.polyfit(fb[ov], bands[i][ov], 1)
            bands[i] = np.where(np.isfinite(bands[i]), bands[i], a_ * fb + b_)
        log["cassis"]["fill"] = {"file": CASSIS_FILL.name, "fit": ffit, "shift_applied_m": fshift}
        print(f"  strip edge: {gap.mean() * 100:.1f} % of the window filled from {CASSIS_FILL.name} (shift {fshift})")
    rgb = np.dstack([stretch(b) for b in bands])

    # ---- (b) geologic map: olivine unit as the base, clay polygons (with their holes) painted on top;
    #      the Mafic_cap layer is left out so the olivine beneath it shows (paper convention)
    units = load_units(crs)
    if MAP_IN_CASSIS_FRAME:
        units = {k: [shapely.affinity.translate(g, *shift) for g in v] for k, v in units.items()}
    log["geomap_shift_m"] = shift if MAP_IN_CASSIS_FRAME else (0.0, 0.0)
    codes = {"oliv": 1, "graben": 2, "cap": 4, "clay": 3, "dune": 5}
    src_key = {"oliv": "outline", "graben": "graben", "clay": "clay", "cap": "cap", "dune": "dune"}
    order = ("oliv", "graben") + (("cap",) if DRAW_CAP else ()) + ("clay", "dune")
    shapes = [(g, codes[k]) for k in order for g in units[src_key[k]]]
    lab = rasterize(shapes, out_shape=shape, transform=tr, fill=0, dtype="uint8", all_touched=False)
    geo_rgb = np.ones((ny, nx, 3)) * 0.93
    geo_rgb[lab == 1] = matplotlib.colors.to_rgb(UNIT_COLOURS["oliv"])
    geo_rgb[lab == 3] = matplotlib.colors.to_rgb(UNIT_COLOURS["clay"])
    geo_rgb[lab == 2] = matplotlib.colors.to_rgb(UNIT_COLOURS["clay"])       # graben materials (none in this window)
    geo_rgb[lab == 4] = 0.35                                                  # cap, only if DRAW_CAP
    olv = (lab == 1).astype(float)
    log["geomap_fractions"] = {"oliv": float((lab == 1).mean()), "clay": float((lab == 3).mean()),
                               "other": float(((lab != 1) & (lab != 3)).mean())}

    # ---- (c) OLINDEX from the fresh MTR3 computation, CRISM->CTX shift applied
    arr_path = OUT / f"{CRISM_CUBE}_{param}_mtr3.npy"
    if not arr_path.exists():
        raise SystemExit(f"{arr_path} missing - run code/contact_olindex_compare.py first")
    oli = np.load(arr_path).astype(np.float32)
    gt = sorted(glob.glob(str(ROOT / "outputs" / "_v8_mtr3_with_brown" / CRISM_CUBE / "geotiffs" / "*.tif")))[0]
    with rasterio.open(gt) as ds:
        m_tr, m_crs = ds.transform, ds.crs
    off = json.loads(FRAME_JSON.read_text())["crism"][CRISM_CUBE]
    # site-local residual of the MTR3 georeference vs CTX (contact_crism_local_residual.py, mean over the
    # bands measured; S and L detectors agree), applied on top of the frame shift
    loc = [json.loads(f.read_text())["applied"] for f in sorted(OUT.glob("crism_local_residual_site1_wide_*nm.json"))]
    loc = [l for l in loc if l]
    crism_local = (float(np.mean([l["dx_m"] for l in loc])), float(np.mean([l["dy_m"] for l in loc]))) if loc else (0.0, 0.0)
    m_tr_eff = Affine.translation(off["dx_m"] + crism_local[0] + MANUAL_CRISM_SHIFT_M[0],
                                  off["dy_m"] + crism_local[1] + MANUAL_CRISM_SHIFT_M[1]) * m_tr
    print(f"CRISM {CRISM_CUBE}: frame shift {off['dx_m']:+.1f}/{off['dy_m']:+.1f} m + site-local {crism_local[0]:+.1f}/{crism_local[1]:+.1f} m "
          f"(from {len(loc)} bands) + manual {MANUAL_CRISM_SHIFT_M[0]:+.0f}/{MANUAL_CRISM_SHIFT_M[1]:+.0f} m")
    tmp = OUT / f"_tmp_{CRISM_CUBE}_{param}.tif"
    with rasterio.open(tmp, "w", driver="GTiff", width=oli.shape[1], height=oli.shape[0], count=1, dtype="float32",
                       crs=m_crs, transform=m_tr_eff, nodata=np.nan) as dst:
        dst.write(oli, 1)
    oli_g = read_on_grid(str(tmp), m_tr_eff, m_crs, None, crs, tr, shape, Resampling.nearest)
    tmp.unlink(missing_ok=True)
    vmin, vmax = (float(np.nanpercentile(oli_g, q)) for q in OLINDEX_STRETCH_PCT)
    log["olindex"] = {"param": param, "cube": CRISM_CUBE, "crism_frame_shift_m": (off["dx_m"], off["dy_m"]), "crism_local_shift_m": crism_local, "crism_manual_shift_m": MANUAL_CRISM_SHIFT_M,
                      "range_in_window": [float(np.nanmin(oli_g)), float(np.nanmax(oli_g))], "vmin_shown": vmin, "vmax_shown": vmax}

    # ---- (d) HiRISE DEM
    reg = registry()
    fitd = local_residual(xc, yc, crs, reg["ADI_006712"], "ortho")
    refine = (fitd["residual_dx_m"], fitd["residual_dy_m"]) if (fitd.get("ok") and fitd["conf_median"] >= 0.5) else (0.0, 0.0)
    path, d_tr, d_crs, d_nod = open_meta(reg["ADI_006712"], "dem")
    dem = read_on_grid(path, Affine.translation(*refine) * d_tr, d_crs, d_nod, crs, tr, shape, Resampling.bilinear)
    hs = multi_hillshade(dem, GRID_M, smooth_m=max(HS_SMOOTH_M, abs(d_tr.a)))
    slope = slope_deg(dem, GRID_M, smooth_m=max(SLOPE_SMOOTH_M, abs(d_tr.a)))
    log["dem"] = {"key": "ADI_006712", "frame_shift_m": reg["ADI_006712"]["shift"], "local_refine_m": refine, "residual_fit": fitd}

    np.savez_compressed(OUT / "paper_fig_site1_layers.npz", transform=np.array(tr.to_gdal()), cassis_rgb=rgb.astype(np.float32),
                        geomap_codes=lab, olindex=oli_g, dem=dem)

    # ---- figure (A4 full width)
    panels = ["cassis", "geomap", "olindex", "dem"] + (["slope"] if want_slope else [])
    ncol = 2 if len(panels) <= 4 else 3
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(FIG_W_IN, FIG_W_IN * nrow / ncol), squeeze=False)
    axes_flat = axes.ravel()
    cx2, cy2 = xs + CLOSE["center_offset_m"][0], ys + CLOSE["center_offset_m"][1]
    bw2, bh2 = CLOSE["box_wh_m"]
    bc, br = ~tr * (cx2 - bw2 / 2, cy2 + bh2 / 2)
    for ax, key, let in zip(axes_flat, panels, "abcde"):
        dark = key == "geomap"
        if key == "cassis":
            ax.imshow(rgb, extent=ext, interpolation="bilinear")
            if gap.any():
                ax.contour(XC, YC, gap.astype(float), levels=[0.5], colors="white", linewidths=0.6, linestyles="dotted")
            ax.add_patch(Rectangle((bc, br), bw2 / GRID_M, bh2 / GRID_M, fill=False, ec="white", lw=1.4))
            avoid = np.zeros(lab.shape, bool)
            r0, r1 = int(max(br - 40 / GRID_M, 0)), int(min(br + (bh2 + 40) / GRID_M, ny))
            c0, c1 = int(max(bc - 40 / GRID_M, 0)), int(min(bc + (bw2 + 40) / GRID_M, nx))
            avoid[r0:r1, c0:c1] = True
            unit_labels(ax, lab, avoid=avoid)
        elif key == "geomap":
            ax.imshow(geo_rgb, extent=ext, interpolation="nearest")
            unit_labels(ax, lab)
        elif key == "olindex":
            im = ax.imshow(oli_g, cmap="gray", vmin=vmin, vmax=vmax, extent=ext, interpolation="nearest")
            ax.contour(XC, YC, olv, levels=[0.5], colors="white", linewidths=0.9)
            (gx0, gy0), (gx1, gy1) = GREEN_BOX_M
            ax.add_patch(Rectangle((gx0 / GRID_M, gy0 / GRID_M), (gx1 - gx0) / GRID_M, (gy1 - gy0) / GRID_M, fill=False, ec="#00e64d", lw=2.4, zorder=6))
            inset_cbar(fig, ax, im, param)
            unit_labels(ax, lab)
        elif key == "dem":
            ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=ext)
            v1, v99 = np.nanpercentile(dem, (0.5, 99.5))
            im = ax.imshow(dem, cmap="turbo", alpha=0.55, extent=ext, vmin=v1, vmax=v99)
            ax.contour(XC, YC, olv, levels=[0.5], colors="white", linewidths=0.9)
            inset_cbar(fig, ax, im, "elevation (m)")
            unit_labels(ax, lab)
        elif key == "slope":
            im = ax.imshow(slope, cmap="magma", vmin=0, vmax=np.nanpercentile(slope, 98), extent=ext)
            ax.contour(XC, YC, olv, levels=[0.5], colors="white", linewidths=0.9)
            inset_cbar(fig, ax, im, "slope (deg)")
        ax.text(0.03, 0.97, let, transform=ax.transAxes, ha="left", va="top", fontsize=FS_LETTER, fontweight="bold",
                color="black" if dark else "white", path_effects=None if dark else STROKE)
        ax.set_xlim(0, nx); ax.set_ylim(ny, 0); ax.set_xticks([]); ax.set_yticks([])
        scalebar_big(ax, nx, ny, GRID_M, 150.0, color="black" if dark else "white")
    for ax in axes_flat[len(panels):]:
        ax.set_visible(False)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.04, hspace=0.04)
    stem = "fig_paper_site1" + ("_slope" if want_slope else "") + ("" if param == "OLINDEX2" else f"_{param}")
    safe_savefig(fig, OUT / stem, dpi=300, pdf_dpi=600, bbox_inches="tight")
    plt.close(fig)

    caption = (
        f"Site 1, {bw:.0f} x {bh:.0f} m window, north up, all layers co-registered to the controlled CTX (Robbins) mosaic. "
        f"(a) CaSSIS colour (NIR-PAN-BLU) pansharpened with HiRISE ({CASSIS_CUBE.name.split('_nodata')[0]}, 0.29 m); "
        f"white box = the next figure ({bw2:.0f} x {bh2:.0f} m). (b) Geologic map (olivine unit orange, clay-bearing basement blue). "
        f"(c) {param} olivine index from CRISM {CRISM_CUBE} (MTR3, 18 m), greyscale stretched {vmin:.2f} (black) to {vmax:.2f} (white); white outline = mapped olivine unit; green box = {GREEN_BOX_M[1][0]-GREEN_BOX_M[0][0]:.0f} x {GREEN_BOX_M[1][1]-GREEN_BOX_M[0][1]:.0f} m region of interest. "
        f"(d) HiRISE stereo DEM (PSP_006712/PSP_005855, 0.8 m) hillshade with elevation colour; white outline as in (c). Scale bars 150 m.\n"
        f"Registration: CaSSIS shift {shift[0]:+.1f}/{shift[1]:+.1f} m (tile MAD {fit.get('mad_m', [0, 0])[0]:.0f}/{fit.get('mad_m', [0, 0])[1]:.0f} m); "
        f"HiRISE frame shift {reg['ADI_006712']['shift'][0]:+.1f}/{reg['ADI_006712']['shift'][1]:+.1f} m + local {refine[0]:+.1f}/{refine[1]:+.1f} m; "
        f"CRISM frame shift {off['dx_m']:+.1f}/{off['dy_m']:+.1f} m (tile MAD {off['dx_mad_m']:.0f}/{off['dy_mad_m']:.0f} m) + site-local residual {crism_local[0]:+.1f}/{crism_local[1]:+.1f} m (749/1330/1750/2200 nm bands vs CTX, tile MAD 2 m in y) ; geologic map digitised on the pansharpened product and moved by the same CaSSIS correction ({shift[0]:+.1f}/{shift[1]:+.1f} m) into the CTX frame" + (f"; manual CRISM adjustment {MANUAL_CRISM_SHIFT_M[1]:+.0f} m" if any(MANUAL_CRISM_SHIFT_M) else "") + "."
        + (f" The easternmost {gap.mean() * bw:.0f} m of (a) lie beyond the pansharpened strip and are filled from the 4 m CaSSIS cube (dotted seam)." if gap.any() else "")
    )
    (OUT / f"{stem}_caption.txt").write_text(caption, encoding="utf-8")
    log.update({"script": Path(__file__).name, "script_md5": md5(Path(__file__)), "started_utc": t0.isoformat(),
                "finished_utc": datetime.now(timezone.utc).isoformat(), "caption": caption})
    (OUT / f"run_log_{stem}.json").write_text(json.dumps(log, indent=2, default=str))
    print("geomap fractions in window:", {k: round(v, 3) for k, v in log["geomap_fractions"].items()}, "| olindex vmax", round(vmax, 3))


if __name__ == "__main__":
    main()
