"""Paper Figure 7 regenerated in the project frame: the two intrusions at site 1.

Window: contact_site_panels.SITES['site1'] (380 x 270 m; the white box of
Fig. 6). All layers in the CTX (Robbins) frame, north up.
(a) CaSSIS pansharpened NIR-PAN-BLU (0.25 m display grid, PAN registered to
    CTX); white box = intrusion 1 (bounds of 'dike map.shp' + 'amorphous
    matrix.shp'); red line = intrusion 2 profile (AV's transect, located
    from his screenshot of the slope panel by template matching).
(b) Slope of the HiRISE DEM (ADI_006712, 0.8 m; frame shift + site-local
    residual), same annotations.
(c) Geologic map of intrusion 1: olivine blocks ('dike map.shp', orange),
    amorphous matrix ('amorphous matrix.shp', black), clay basement (gdb
    Clay_unit, blue), on the intrusion-1 box.
(d) Elevation profile of intrusion 2 along the red line (HiRISE DEM sampled
    every 0.5 m), drawn extent marked, optional extension each side.
A4 full-width layout. Deterministic.

Usage:  python code/contact_paper_fig7.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyogrio
import rasterio
import shapely
import shapely.affinity
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.warp import transform as rtransform
from scipy.ndimage import map_coordinates

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from contact_site_panels import (SITES, working_crs, lonlat_to_xy, read_on_grid, slope_deg, local_residual,  # noqa: E402
                                  SLOPE_SMOOTH_M)
from contact_paper_fig_site1 import (residual_vs_ctx, read_band_on_grid, stretch, scalebar_big, inset_cbar,  # noqa: E402
                                      CASSIS_CUBE, CASSIS_RGB, STROKE, FS_LETTER, FS_UNIT, FS_CB, FIG_W_IN)
from contact_dtm_registry import registry, open_meta  # noqa: E402
from contact_geomap_fractions import load_units  # noqa: E402
from contact_height_above_contact import md5  # noqa: E402
from figstyle import safe_savefig  # noqa: E402

OUT = ROOT / "outputs" / "revision" / "sites"
WIN = SITES["site1"]
GRID_M, ORTHO_GRID_M = 0.5, 0.25
from config import GEOMAP_SHP_DIR as SHP  # noqa: E402
DIKES, MATRIX = SHP / "dike map.shp", SHP / "amorphous matrix.shp"
COL = {"oliv": "#e07820", "clay": "#2880a8", "matrix": "#101010", "other": "#d9d9d9"}
# intrusion-2 transect from AV's screenshot, metres from the W / N edge of the PREVIOUS 300 x 270 m window
LINE_OLD_WIN = ((133.8, 71.1), (200.0, 49.0))
OLD_WIN_SHIFT_W_M = 80.0          # the window was widened 80 m to the west afterwards (300 -> 380 m)
PROFILE_EXTEND_M = 30.0           # extra length sampled beyond the drawn line, each side
BOX_PAD_M = 2.0                   # tight: the dike/matrix map fills panel (c)
CROP_LEFT_M, CROP_RIGHT_M = 26.0, 3.0   # cut the olivine/matrix demarcation at the W end and the sliver at the E end (AV, 2026-09-09)


def load_shp(path, crs):
    info = pyogrio.read_info(path); _, _, geoms, _ = pyogrio.raw.read(path)
    src = CRS.from_wkt(info["crs"])
    return [shapely.make_valid(shapely.transform(shapely.force_2d(g), lambda xy: np.column_stack(rtransform(src, crs, xy[:, 0], xy[:, 1]))))
            for g in shapely.from_wkb(geoms)]


def main() -> None:
    t0 = datetime.now(timezone.utc)
    crs = working_crs()
    xs, ys = lonlat_to_xy(WIN["lon"], WIN["lat"], crs)
    xc, yc = xs + WIN["center_offset_m"][0], ys + WIN["center_offset_m"][1]
    bw, bh = WIN["box_wh_m"]
    nx, ny = int(round(bw / GRID_M)), int(round(bh / GRID_M))
    tr = Affine(GRID_M, 0, xc - bw / 2, 0, -GRID_M, yc + bh / 2); shape = (ny, nx)
    x0, y0 = tr.c, tr.f                                   # window top-left in map metres
    log = {"window": WIN, "grid_m": GRID_M}

    # ---- CaSSIS pansharpened at 0.25 m, registered
    with rasterio.open(CASSIS_CUBE) as ds:
        desc = list(ds.descriptions); c_tr, c_crs, c_nod = ds.transform, ds.crs, ds.nodata
    fit = residual_vs_ctx(CASSIS_CUBE, desc.index("PAN") + 1, xc, yc, crs)
    shift = (fit["dx_m"], fit["dy_m"]) if (fit.get("ok") and fit["conf_median"] >= 0.4) else (0.0, 0.0)
    c_tr_eff = Affine.translation(*shift) * c_tr
    fo = int(round(GRID_M / ORTHO_GRID_M)); tr_o = tr * Affine.scale(1.0 / fo)
    rgb = np.dstack([stretch(read_band_on_grid(CASSIS_CUBE, desc.index(b) + 1, c_tr_eff, c_crs, c_nod, crs, tr_o, (ny * fo, nx * fo), Resampling.bilinear))
                     for b in CASSIS_RGB])
    log["cassis"] = {"file": CASSIS_CUBE.name, "shift_m": shift, "fit": fit}
    print(f"CaSSIS shift {shift[0]:+.1f}/{shift[1]:+.1f} m (MAD {fit.get('mad_m')}, conf {fit.get('conf_median', 0):.2f})")

    # ---- HiRISE DEM + slope
    reg = registry()
    fitd = local_residual(xc, yc, crs, reg["ADI_006712"], "ortho")
    refine = (fitd["residual_dx_m"], fitd["residual_dy_m"]) if (fitd.get("ok") and fitd["conf_median"] >= 0.5) else (0.0, 0.0)
    path, d_tr, d_crs, d_nod = open_meta(reg["ADI_006712"], "dem")
    dem = read_on_grid(path, Affine.translation(*refine) * d_tr, d_crs, d_nod, crs, tr, shape, Resampling.bilinear)
    slope = slope_deg(dem, GRID_M, smooth_m=max(SLOPE_SMOOTH_M, abs(d_tr.a)))
    log["dem"] = {"key": "ADI_006712", "frame_shift_m": reg["ADI_006712"]["shift"], "local_refine_m": refine}
    print(f"HiRISE local refinement {refine[0]:+.1f}/{refine[1]:+.1f} m")

    # ---- intrusion 1: dike / matrix polygons, clay from the gdb
    # the dike / matrix shapefiles were digitised on the pansharpened product in its FILE georeference
    # (check_fig7_polygon_registration.png): apply the same CaSSIS-to-CTX correction to them
    dikes = [shapely.affinity.translate(g, *shift) for g in load_shp(DIKES, crs)]
    matrix = [shapely.affinity.translate(g, *shift) for g in load_shp(MATRIX, crs)]
    log["dike_matrix_shift_m"] = shift
    units = {k: [shapely.affinity.translate(g, *shift) for g in v] for k, v in load_units(crs).items()}   # gdb: same frame (check_fig7_clay_registration.png)
    log["geomap_shift_m"] = shift
    # panel-(c) extent: the trimmed matrix edge-to-edge in x (no padding), BOX_PAD_M in y (AV, 2026-09-09)
    oliv_only = shapely.unary_union(units["outline"]).difference(shapely.unary_union(units["clay"]))
    mtrim = shapely.unary_union([g.intersection(oliv_only) for g in matrix])
    b = shapely.unary_union([mtrim] + dikes).bounds
    bx0, bx1 = b[0] + CROP_LEFT_M, b[2] - CROP_RIGHT_M
    # same aspect ratio as panel (a) (AV, 2026-09-09): height = width / (bw/bh); vertical position chosen to
    # keep the largest possible share of the trimmed matrix inside the box
    hh = (bx1 - bx0) / (bw / bh)
    cands = np.arange(b[1] + hh / 2, b[3] - hh / 2 + 0.01, 0.5)
    def inside_area(ycen):
        box = shapely.box(bx0, ycen - hh / 2, bx1, ycen + hh / 2)
        return mtrim.intersection(box).area
    ycen = max(cands, key=inside_area) if cands.size else 0.5 * (b[1] + b[3])
    by1, by0 = ycen - hh / 2, ycen + hh / 2
    log["intrusion1_matrix_share_in_box"] = float(inside_area(ycen) / mtrim.area)
    box_px = ((bx0 - x0) / GRID_M, (y0 - by0) / GRID_M, (bx1 - bx0) / GRID_M, (by0 - by1) / GRID_M)   # col, row, w, h
    log["intrusion1_box_m"] = {"x_from_W": bx0 - x0, "x_to": bx1 - x0, "y_from_N": y0 - by0, "y_to": y0 - by1}
    # intrusion-1 map raster at 0.1 m
    g1 = 0.1
    n1x, n1y = int(round((bx1 - bx0) / g1)), int(round((by0 - by1) / g1))
    tr1 = Affine(g1, 0, bx0, 0, -g1, by0)
    # matrix trimmed by the unit map: it exists only inside the olivine unit (outline minus clay); dikes on top
    oliv_only = shapely.unary_union(units["outline"]).difference(shapely.unary_union(units["clay"]))
    matrix = [g.intersection(oliv_only) for g in matrix]
    matrix = [g for g in matrix if not g.is_empty]
    shapes = [(g, 1) for g in units["outline"]] + [(g, 3) for g in units["clay"]] + [(g, 4) for g in matrix] + [(g, 5) for g in dikes]
    lab1 = rasterize(shapes, out_shape=(n1y, n1x), transform=tr1, fill=0, dtype="uint8")
    geo1 = np.ones((n1y, n1x, 3)) * matplotlib.colors.to_rgb(COL["other"])
    for code, key in ((1, "oliv"), (3, "clay"), (4, "matrix"), (5, "oliv")):
        geo1[lab1 == code] = matplotlib.colors.to_rgb(COL[key])
    log["intrusion1_fractions"] = {k: float((lab1 == c).mean()) for k, c in (("olivine_unit", 1), ("clay", 3), ("matrix", 4), ("dike_blocks", 5))}

    # ---- intrusion 2 profile
    (ax_, ay_), (bx_, by_) = LINE_OLD_WIN
    P1 = np.array([ax_ + OLD_WIN_SHIFT_W_M, ay_]); P2 = np.array([bx_ + OLD_WIN_SHIFT_W_M, by_])   # window metres (x from W, y from N)
    if P1[0] > P2[0]:
        P1, P2 = P2, P1                                                 # run W -> E
    u = (P2 - P1) / np.linalg.norm(P2 - P1); L = np.linalg.norm(P2 - P1)
    s = np.arange(-PROFILE_EXTEND_M, L + PROFILE_EXTEND_M + GRID_M / 2, GRID_M)
    pts = P1[None, :] + s[:, None] * u[None, :]
    zprof = map_coordinates(np.where(np.isfinite(dem), dem, np.nanmean(dem)), [pts[:, 1] / GRID_M - 0.5, pts[:, 0] / GRID_M - 0.5], order=1)
    sprof = map_coordinates(slope, [pts[:, 1] / GRID_M - 0.5, pts[:, 0] / GRID_M - 0.5], order=1)
    log["intrusion2_profile"] = {"P1_window_m": P1.tolist(), "P2_window_m": P2.tolist(), "length_m": float(L),
                                 "azimuth_deg": float(np.degrees(np.arctan2(u[0], -u[1])) % 360), "extend_m": PROFILE_EXTEND_M,
                                 "z_range_m": [float(np.nanmin(zprof)), float(np.nanmax(zprof))]}
    np.savez_compressed(OUT / "paper_fig7_layers.npz", transform=np.array(tr.to_gdal()), cassis_rgb=rgb.astype(np.float32), dem=dem, slope=slope,
                        geo1_codes=lab1, geo1_transform=np.array(tr1.to_gdal()), profile_s=s, profile_z=zprof, profile_slope=sprof)

    # ---- figure
    ext = (0, nx, ny, 0)
    # manual layout: image panels with the window aspect, (d) with room for its axis labels
    asp = bh / bw
    pw = 0.475; ph_in = FIG_W_IN * pw * asp
    fig_h = 2 * ph_in + 0.75
    fig = plt.figure(figsize=(FIG_W_IN, fig_h))
    ph = ph_in / fig_h
    a = fig.add_axes([0.01, 1 - 0.01 - ph, pw, ph]); b_ = fig.add_axes([0.515, 1 - 0.01 - ph, pw, ph])
    c = fig.add_axes([0.01, 0.01 + 0.05, pw, ph]); d = fig.add_axes([0.60, 0.01 + 0.12, 0.385, ph - 0.10])
    a.imshow(rgb, extent=ext, interpolation="bilinear")
    sm = np.nanpercentile(slope, 98)
    im = b_.imshow(slope, cmap="magma", vmin=0, vmax=sm, extent=ext)
    cb = inset_cbar(fig, b_, im, "slope (deg)")
    cb.ax.set_position([0.56, 0.12, 0.40, 0.035])                                        # raised (AV, 2026-09-09)
    for ax, let in ((a, "a"), (b_, "b")):
        ax.add_patch(Rectangle((box_px[0], box_px[1]), box_px[2], box_px[3], fill=False, ec="white", lw=1.4))
        ax.text(box_px[0] + box_px[2] / 2, box_px[1] - 6 / GRID_M, "Intrusion 1", color="white", ha="center", va="bottom", fontsize=FS_UNIT - 3,
                fontweight="bold", path_effects=STROKE)
        ax.plot([P1[0] / GRID_M, P2[0] / GRID_M], [P1[1] / GRID_M, P2[1] / GRID_M], color="red", lw=1.8)
        ax.text(P1[0] / GRID_M - 4 / GRID_M, P1[1] / GRID_M, "A", color="red", ha="right", va="center", fontsize=FS_UNIT - 3, fontweight="bold", path_effects=STROKE)
        ax.text(P2[0] / GRID_M + 4 / GRID_M, P2[1] / GRID_M, "A'", color="red", ha="left", va="center", fontsize=FS_UNIT - 3, fontweight="bold", path_effects=STROKE)
        ax.text((P1[0] + P2[0]) / 2 / GRID_M, min(P1[1], P2[1]) / GRID_M - 16 / GRID_M, "Intrusion 2", color="red", ha="center", va="bottom",
                fontsize=FS_UNIT - 3, fontweight="bold", path_effects=STROKE)
        ax.text(0.03, 0.97, let, transform=ax.transAxes, ha="left", va="top", fontsize=FS_LETTER, fontweight="bold", color="white", path_effects=STROKE)
        if let == "a":                                                                    # unit labels only in (a) (AV, 2026-09-09)
            for (fx, fy), text in (((0.45, 0.72), "Clay"), ((0.84, 0.86 - 50.0 / bh), "Olivine")):   # olivine 50 m further north
                ax.text(fx, 1 - fy, text, transform=ax.transAxes, ha="center", va="center", fontsize=FS_UNIT - 2, fontweight="bold", color="white", path_effects=STROKE)
        ax.set_xlim(0, nx); ax.set_ylim(ny, 0); ax.set_xticks([]); ax.set_yticks([])
        scalebar_big(ax, nx, ny, GRID_M, 50.0, color="white")
    # (c) intrusion-1 map
    c.imshow(geo1, extent=(0, n1x, n1y, 0), interpolation="nearest")
    for key, text in (("matrix", "Matrix"), ("clay", "Clay")):
        code = {"matrix": 4, "clay": 3}[key]; m = lab1 == code
        if m.any():
            from scipy.ndimage import distance_transform_edt
            dd = distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1]; r0, c0 = np.unravel_index(np.argmax(dd), dd.shape)
            c.text(c0, r0, text, color="white", ha="center", va="center", fontsize=FS_UNIT - 3, fontweight="bold", path_effects=STROKE)
    if (lab1 == 5).any():                                   # 'Olivine' on the largest block that sits well inside the panel
        inner = [g for g in dikes if bx0 + 12 < g.centroid.x < bx1 - 12 and by1 + 8 < g.centroid.y < by0 - 8]
        big = max(inner or dikes, key=lambda g: g.area); cx_, cy_ = big.centroid.x, big.centroid.y
        c.text((cx_ - bx0) / g1, (by0 - cy_) / g1, "Olivine", color="white", ha="center", va="center", fontsize=FS_UNIT - 3, fontweight="bold", path_effects=STROKE)
    c.text(0.03, 0.97, "c", transform=c.transAxes, ha="left", va="top", fontsize=FS_LETTER, fontweight="bold", color="white", path_effects=STROKE)
    c.set_xlim(0, n1x); c.set_ylim(n1y, 0); c.set_xticks([]); c.set_yticks([])
    scalebar_big(c, n1x, n1y, g1, 10.0, color="white")
    # (d) profile
    d.plot(s, zprof, color="k", lw=1.6)
    d.axvspan(0, L, color="red", alpha=0.08)
    d.axvline(0, color="red", lw=0.8, ls="--"); d.axvline(L, color="red", lw=0.8, ls="--")
    d.text(0, np.nanmax(zprof), "A", color="red", ha="center", va="bottom", fontsize=FS_UNIT - 3, fontweight="bold")
    d.text(L, np.nanmax(zprof), "A'", color="red", ha="center", va="bottom", fontsize=FS_UNIT - 3, fontweight="bold")
    d.set_xlabel("distance along A-A' (m)", fontsize=FS_CB + 1); d.set_ylabel("elevation (m)", fontsize=FS_CB + 1)
    d.tick_params(labelsize=FS_CB)
    d.grid(True, color="0.92"); d.spines[["top", "right"]].set_visible(False)
    d.text(0.03, 0.97, "d", transform=d.transAxes, ha="left", va="top", fontsize=FS_LETTER, fontweight="bold")
    bbox = d.get_position(); ax_w_in, ax_h_in = bbox.width * FIG_W_IN, bbox.height * fig_h
    ve = ((s[-1] - s[0]) / ax_w_in) / (np.ptp(d.get_ylim()) / ax_h_in)
    d.set_title("Intrusion 2", fontsize=FS_CB + 1, fontweight="bold")                     # DEM source and V.E. go in the caption (AV)
    log["intrusion2_profile"]["vertical_exaggeration"] = float(ve)
    safe_savefig(fig, OUT / "fig_paper_fig7", dpi=300, pdf_dpi=600, bbox_inches="tight")
    plt.close(fig)
    caption = (f"Fig. 7 draft. (a) CaSSIS pansharpened colour (NIR-PAN-BLU, 0.25 m) of the Fig. 6 box ({bw:.0f} x {bh:.0f} m); white box = intrusion 1, red line A-A' = intrusion 2 transect "
               f"({L:.0f} m, azimuth {log['intrusion2_profile']['azimuth_deg']:.0f} deg). (b) Slope of the HiRISE stereo DEM (PSP_006712/005855, 0.8 m). "
               f"(c) Geologic map of intrusion 1 ({bx1-bx0:.0f} x {by0-by1:.0f} m): olivine blocks orange, amorphous matrix black, clay basement blue. "
               f"(d) Elevation along A-A' sampled every {GRID_M} m from the HiRISE stereo DEM (PSP_006712/PSP_005855, 0.8 m), vertical exaggeration {ve:.0f}:1 as drawn (drawn transect shaded, {PROFILE_EXTEND_M:.0f} m extension each side). "
               f"Registration: CaSSIS {shift[0]:+.1f}/{shift[1]:+.1f} m, HiRISE local {refine[0]:+.1f}/{refine[1]:+.1f} m on top of the frame shifts; the dike/matrix polygons and the gdb units (all digitised on the pansharpened product) carry the same CaSSIS correction.")
    (OUT / "fig_paper_fig7_caption.txt").write_text(caption, encoding="utf-8")
    log.update({"script": Path(__file__).name, "script_md5": md5(Path(__file__)), "caption": caption,
                "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat()})
    (OUT / "run_log_fig_paper_fig7.json").write_text(json.dumps(log, indent=2, default=str))
    print("intrusion-1 box (window m):", {k: round(v) for k, v in log["intrusion1_box_m"].items()}, "| profile z range", [round(v, 2) for v in log["intrusion2_profile"]["z_range_m"]])
    print("intrusion-1 map fractions:", {k: round(v, 3) for k, v in log["intrusion1_fractions"].items()})


if __name__ == "__main__":
    main()
