"""Topographic profiles through the hottest 2.3 um phase patches (FRT00003E12).

Old-school check of the elevation result. 'Hotspot' = local maximum of the
class density field (fraction of valid map-pure olivine pixels within ~90 m
that carry the class), restricted to the unit interior (> 90 m from the
clay contact); the patch is the connected region around the peak where the
density stays above half the peak value (capped at 250 m radius). For the
top-N hotspots of the carbonate class (carbonate-mode + both-bands) and of
the serpentine/talc-mode class, two orthogonal elevation profiles (along
the patch's principal axis and across it) are drawn through the peak,
extended 250 m into the surrounding olivine unit on each side. Elevation
comes from a HiRISE DEM mosaic (registry priority PDS_002888 clean >
ADI_027691 > ADI_006712 > PDS_006712, measured shifts applied, 3 m warped
grid) with the CTX 18 m DEM as fallback where no HiRISE DEM exists.
Profile samples are coloured by the 18 m class raster; the patch extent is
shaded; patch-minus-flank elevation is reported after removing a linear
trend along the profile (flank = other-olivine samples > 30 m outside the
patch). Deterministic.

Usage:  python code/contact_hotspot_profiles.py [CUBE ...]   (default: all four cubes)
"""
from __future__ import annotations

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
from scipy.ndimage import gaussian_filter, label as cc_label, map_coordinates, maximum_filter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from contact_distance_to_contact import load_cube, masks_from_labels, md5, PIX_M  # noqa: E402
from contact_band_center_map import unit_masks  # noqa: E402
from contact_carbonate_patches import GEOMAP  # noqa: E402
from contact_height_above_contact import CTX_DEM, warp_dem  # noqa: E402
from contact_dtm_registry import registry, open_meta  # noqa: E402
from figstyle import draw_units, safe_savefig  # noqa: E402

CUBES = ("FRT00003E12", "FRT0000871C", "FRT0000A4FC", "FRT0000B438")
DEM_PRIORITY = ("PDS_002888", "ADI_027691", "ADI_006712", "PDS_006712")
BC = ROOT / "outputs" / "revision" / "band_center_map"
OUT = BC / "combinations"
N_TOP = 5
SIGMA_PX = 2.5                 # density kernel, 45 m sigma (~90 m FWHM)
PEAK_WIN_PX = 15               # local-maximum window (270 m)
MIN_PEAK_FRAC = 0.25
MIN_SEP_M = 600.0
PATCH_MAX_R_M = 180.0
MIN_DIST_CLAY_M = 90.0
FLANK_M = 250.0                # profile extension beyond the patch, each side
FLANK_GAP_M = 30.0             # near-flank reference window: 30-130 m beyond the patch edge
FLANK_NEAR_M = 130.0
FINE = 6                       # 18 m / 6 = 3 m sampling grid
STEP_M = 3.0
CLASS_STYLE = {0: ("other olivine", "0.55"), 1: ("carbonate-mode", "#ff7f0e"),
               2: ("serpentine/talc-mode", "#8c564b"), 3: ("both bands", "#2ca02c")}
TARGETS = {"carbonate": ((1, 3), "#ff7f0e"), "serpentine": ((2,), "#8c564b")}


def sample(arr, tr, x, y, order):
    col, row = ~tr * (x, y)
    return map_coordinates(arr, [row, col], order=order, mode="constant", cval=np.nan)


def hirise_mosaic(crs, tr, shape) -> tuple[np.ndarray, dict]:
    reg = registry(); z = np.full(shape, np.nan, np.float32); cov = {}
    for key in DEM_PRIORITY:
        path, src_tr, src_crs, nod = open_meta(reg[key], "dem")
        with rasterio.open(path) as ds:
            src = ds.read(1).astype(np.float32)
        if nod is not None:
            src[src == nod] = np.nan
        src[(src < -1e30) | (src > 1e30)] = np.nan
        dst = np.full(shape, np.nan, np.float32)
        reproject(src, dst, src_transform=src_tr, src_crs=src_crs, src_nodata=np.nan,
                  dst_transform=tr, dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
        fill = np.isnan(z) & np.isfinite(dst)
        z[fill] = dst[fill]; cov[key] = int(fill.sum())
        del src, dst
    return z, cov


def find_hotspots(dens, interior):
    peaks = (dens == maximum_filter(dens, size=PEAK_WIN_PX)) & interior & (dens >= MIN_PEAK_FRAC)
    pr, pc = np.nonzero(peaks)
    order = np.argsort(dens[pr, pc])[::-1]
    picked = []
    for i in order:
        r, c = pr[i], pc[i]
        if all(np.hypot(r - r2, c - c2) * PIX_M >= MIN_SEP_M for r2, c2 in picked):
            picked.append((r, c))
        if len(picked) == N_TOP:
            break
    return picked


def patch_region(dens, r, c, interior):
    above = (dens >= 0.5 * dens[r, c]) & interior
    lab, _ = cc_label(above, structure=np.ones((3, 3)))
    reg = lab == lab[r, c]
    rr, cc = np.indices(dens.shape)
    reg &= np.hypot(rr - r, cc - c) * PIX_M <= PATCH_MAX_R_M
    return reg


def run_cube(CUBE: str) -> None:
    t0 = datetime.now(timezone.utc)
    c = load_cube(CUBE); m = masks_from_labels(c["labels"]); units, d, fr = unit_masks(CUBE, m)
    cls = np.load(OUT / f"{CUBE}_mode_class.npy").astype(int)
    fz = np.load(GEOMAP / f"{CUBE}_fractions.npz"); tr = Affine.from_gdal(*fz["transform"])
    with rasterio.open(sorted((ROOT / "outputs" / "_v8_mtr3_with_brown" / CUBE / "geotiffs").glob("*.tif"))[0]) as ds:
        crs = ds.crs
    H, W = cls.shape
    tr_fine = tr * Affine.scale(1.0 / FINE); shape_fine = (H * FINE, W * FINE)
    z_hi, cov = hirise_mosaic(crs, tr_fine, shape_fine)
    z_ctx = warp_dem(CTX_DEM, (0.0, 0.0), crs, tr_fine, shape_fine, Resampling.bilinear)
    hi_cov18 = np.isfinite(z_hi[FINE // 2::FINE, FINE // 2::FINE]).astype(float)
    print("HiRISE mosaic fill (3 m cells):", cov)

    olv = (fr["oliv"] >= 0.5) & m["valid"]
    interior = olv & (d["clay"] > MIN_DIST_CLAY_M)
    den = gaussian_filter(olv.astype(float), SIGMA_PX)
    rows = []
    for (tname, (kset, tcol)), absolute in [(t, a) for t in TARGETS.items() for a in (False, True)]:
        ind = (np.isin(cls, kset) & olv).astype(float)
        dens = np.where(den > 0.2, gaussian_filter(ind, SIGMA_PX) / np.maximum(den, 1e-6), 0.0)
        picks = find_hotspots(dens, interior)
        fig = plt.figure(figsize=(24, 10))
        gs = fig.add_gridspec(2, 6, width_ratios=(2.0, 1, 1, 1, 1, 1))
        axm = fig.add_subplot(gs[:, 0])
        axm.imshow(np.where(m["valid"], 0.93, np.nan), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(f"dens_{tname}", ["#ffffff", tcol, "#000000"])
        im = axm.imshow(np.where(olv, dens, np.nan), cmap=cmap, vmin=0, vmax=1.0, interpolation="nearest")
        fig.colorbar(im, ax=axm, fraction=0.035, pad=0.01, label=f"{tname} class density")
        axm.contour(hi_cov18, levels=[0.5], colors="#9467bd", linewidths=1.4, linestyles="dotted")
        draw_units(axm, fr, units=("clay", "cap"))
        axm.set_xticks([]); axm.set_yticks([])
        axm.set_title(f"(a) {tname} class density, {CUBE}\n(fraction of olivine pixels within ~90 m; "
                      f"dotted violet = HiRISE DEM coverage;\nsolid line = along-axis profile, dashed = across)", fontsize=9)
        letters = "bcdefghijklmnop"
        for pi, (r, c0) in enumerate(picks):
            reg = patch_region(dens, r, c0, interior)
            rr, cc = np.nonzero(reg)
            pts = np.column_stack([cc - c0, rr - r]).astype(float)
            if pts.shape[0] >= 3:
                _, _, vt = np.linalg.svd(pts - pts.mean(0), full_matrices=False); ux, uy = vt[0]
            else:
                ux, uy = 1.0, 0.0
            x0, y0 = tr * (c0 + 0.5, r + 0.5)
            for li, (vx, vy, lname) in enumerate(((ux, uy, "along axis"), (-uy, ux, "across axis"))):
                ext = max(np.abs(pts @ np.array([vx, vy])).max() * PIX_M, PIX_M)
                half = ext + FLANK_M
                s = np.arange(-half, half + STEP_M, STEP_M)
                xs = x0 + s * vx; ys = y0 - s * vy
                z_h = sample(z_hi, tr_fine, xs, ys, 1); z_c = sample(z_ctx, tr_fine, xs, ys, 1)
                use_h = np.isfinite(z_h).mean() > 0.8
                z = z_h if use_h else z_c; dem_name = "HiRISE" if use_h else "CTX 18 m"
                kcls = sample(cls.astype(float), tr, xs, ys, 0)
                foliv = sample(fr["oliv"].astype(float), tr, xs, ys, 0)
                inreg = sample(reg.astype(float), tr, xs, ys, 0) > 0.5
                ok = np.isfinite(z)
                fl_ok = ok & ~inreg & (foliv >= 0.5)          # any olivine-unit ground outside the patch
                left = fl_ok & (s < -ext - FLANK_GAP_M) & (s > -ext - FLANK_NEAR_M)
                right = fl_ok & (s > ext + FLANK_GAP_M) & (s < ext + FLANK_NEAR_M)
                flank = left | right
                inside = ok & inreg & np.isin(kcls, kset)
                dz_near = dz_chord = dz_chord_min = np.nan; chord = None
                if inside.sum() > 2 and flank.sum() > 5:
                    dz_near = float(np.mean(z[inside]) - np.mean(z[flank]))
                if inside.sum() > 2 and left.sum() >= 5 and right.sum() >= 5:
                    sl, zl = np.mean(s[left]), np.mean(z[left]); sr, zr = np.mean(s[right]), np.mean(z[right])
                    chord = (sl, zl, sr, zr)
                    zc = zl + (zr - zl) * (s - sl) / (sr - sl)
                    dz_chord = float(np.mean(z[inside] - zc[inside])); dz_chord_min = float(np.min(z[inside] - zc[inside]))
                if not absolute:
                    rows.append({"cube": CUBE, "class": tname, "hotspot": pi + 1, "profile": lname, "row": int(r), "col": int(c0),
                                 "peak_density": float(dens[r, c0]), "patch_cells": int(reg.sum()),
                                 "dist_clay_m": float(d["clay"][r, c0]), "dem": dem_name, "half_len_m": float(ext),
                                 "dz_patch_minus_nearflank_m": dz_near, "dz_chord_m": dz_chord, "dz_chord_min_m": dz_chord_min,
                                 "n_in": int(inside.sum()), "n_left": int(left.sum()), "n_right": int(right.sum())})
                axm.plot([c0 + vx * (-half) / PIX_M, c0 + vx * half / PIX_M],
                         [r - vy * (-half) / PIX_M, r - vy * half / PIX_M], color="k" if li == 0 else "0.35",
                         lw=1.0, ls="-" if li == 0 else "--")
                ax = fig.add_subplot(gs[li, 1 + pi])
                ref = 0.0 if absolute else (np.nanmean(z[flank]) if flank.any() else np.nanmean(z))
                zz = z - ref
                ax.plot(s, zz, color="0.75", lw=1.0, zorder=1)
                for kk, (klab, kcol) in CLASS_STYLE.items():
                    sel = ok & (kcls == kk) & (foliv >= 0.5)
                    if sel.any():
                        ax.scatter(s[sel], zz[sel], s=9, color=kcol, zorder=3, label=klab if (pi == 0 and li == 0) else None)
                non = ok & (foliv < 0.5)
                if non.any():
                    ax.scatter(s[non], zz[non], s=9, color="#1f77b4", marker="x", zorder=3,
                               label="non-olivine (clay/cap)" if (pi == 0 and li == 0) else None)
                ax.axvspan(-ext, ext, color=tcol, alpha=0.12, zorder=0)
                for sel_f in (left, right):
                    if sel_f.any():
                        ax.axvspan(s[sel_f].min(), s[sel_f].max(), color="0.5", alpha=0.10, zorder=0)
                if not absolute:
                    ax.axhline(0, color="0.85", lw=0.8)
                if chord is not None:
                    ax.plot([chord[0], chord[2]], [chord[1] - ref, chord[3] - ref], color="k", lw=0.9, ls=":", zorder=2)
                ax.text(0.012, 0.985, f"({letters[pi * 2 + li]})", transform=ax.transAxes, ha="left", va="top",
                        fontsize=11, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85), zorder=20)
                ax.set_title(f"hotspot {pi + 1}, {lname}: {d['clay'][r, c0]:.0f} m from clay, {dem_name}\n"
                             f"patch - near flank {dz_near:+.1f} m; below chord {dz_chord:+.1f} m (min {dz_chord_min:+.1f})", fontsize=8)
                ax.set_xlabel(f"distance {lname} (m)")
                ax.set_ylabel("elevation (m, DEM datum)" if absolute else "elevation - flank mean (m)")
                if absolute:
                    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.0f"))
                ax.grid(True, color="0.95"); ax.spines[["top", "right"]].set_visible(False)
                if pi == 0 and li == 0:
                    ax.legend(fontsize=7, frameon=False, loc="best")
            axm.annotate(f"{pi + 1}", (c0, r), xytext=(8, 8), textcoords="offset points", fontsize=9, fontweight="bold",
                         bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="k", lw=0.6))
        fig.suptitle(f"Topographic profiles through the {len(picks)} hottest interior {tname} patches, {CUBE}"
                     f"{' - absolute elevation' if absolute else ''} "
                     f"(coloured shading = patch; grey shading = near-flank reference windows, olivine-unit ground "
                     f"{FLANK_GAP_M:.0f}-{FLANK_NEAR_M:.0f} m beyond the patch; dotted = chord between the two flanks)", fontsize=11)
        fig.tight_layout()
        safe_savefig(fig, OUT / f"fig_hotspot_profiles_{tname}_{CUBE}{'_abs' if absolute else ''}", bbox_inches="tight")
        plt.close(fig)
        if not absolute:
            print(tname, "hotspots:", [(int(r), int(c0), round(float(dens[r, c0]), 2)) for r, c0 in picks])

    df = pd.DataFrame(rows); df.to_csv(OUT / f"hotspot_profiles_{CUBE}.csv", index=False)
    print(df.round(2).to_string())
    log = {"script": Path(__file__).name, "script_md5": md5(Path(__file__)),
           "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
           "cube": CUBE, "dem_priority": DEM_PRIORITY, "hirise_fill_cells": cov, "n_top": N_TOP,
           "sigma_px": SIGMA_PX, "peak_win_px": PEAK_WIN_PX, "min_peak_frac": MIN_PEAK_FRAC, "min_sep_m": MIN_SEP_M,
           "patch_max_r_m": PATCH_MAX_R_M, "min_dist_clay_m": MIN_DIST_CLAY_M, "flank_m": FLANK_M,
           "flank_gap_m": FLANK_GAP_M, "flank_near_m": FLANK_NEAR_M,
           "fine_factor": FINE, "step_m": STEP_M, "deterministic": True}
    (OUT / f"run_log_hotspot_profiles_{CUBE}.json").write_text(json.dumps(log, indent=2, default=str))


def main() -> None:
    for cube in (sys.argv[1:] or CUBES):
        run_cube(cube)


if __name__ == "__main__":
    main()
