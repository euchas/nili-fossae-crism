"""Fringe gallery v4: the selected 002888 contact-margin sites with a HiRISE
RED column and AV's PANCO (GSA) CaSSIS-pansharpened column, all coregistered.

Requested by AV (2026-08-31) to look at the raised sharp contact in colour.
- Sites: the v3 selection (`stage7_fringe/selected_sites.csv`) restricted to
  DTEEC_002888 (006712 excluded per the Stage 8 DTM verdict).
- PANCO tiles (config.PANSHARP_DIR,
  the only Dropbox folder authorised): GSA Gram-Schmidt pansharpened CaSSIS
  (bands NIR/RED/PAN/BLU, I/F floats). GSA12_MY37_025124 covers the southern
  seven sites at 0.25 m; GSA10_Mosaic027025 covers P4 at 0.5 m. Their frame is
  clon-180 equirectangular WITHOUT the cos(20) easting scale of the DTM frame;
  the conversion is applied as an exact affine, then a residual offset per tile
  is MEASURED against the PSP_002176 RED ortho (tile phase correlation, 1 m)
  and applied. Displayed composite: NPB (R=NIR 939, G=PAN 678, B=BLU 494 nm).
Outputs: stage7_fringe/fig_gallery_v4_cassis_ps.png/.pdf, params_gallery_v4.json,
per-tile QC PNGs.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import shapely
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from contact_contact_fringe import HIRISE, X  # noqa: E402
from contact_coreg_frame import qc_figure, register  # noqa: E402
from contact_fringe_gallery_v2 import clay_boundaries, draw_lines, hillshade  # noqa: E402

import os  # noqa: E402

os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "FALSE"   # coreg_frame sets EMPTY_DIR; keep sidecars visible

OUT = ROOT / "outputs" / "revision" / "contact_3d" / "stage7_fringe"
FRAME = ROOT / "outputs" / "revision" / "contact_3d" / "frame" / "coreg_offsets.json"
from config import HIRISE_PDS_DIR, pansharp_file, read_reflectance  # noqa: E402
RED_ORTHO = HIRISE_PDS_DIR / "PSP_002176_2025_RED_A_01_ORTHO.JP2"
TILES = {"GSA12": pansharp_file("GSA12_MY37_025124_157_0-013.tif"),
         "GSA10": pansharp_file("GSA10_Mosaic027025.tif")}
NPB = [1, 3, 4]                     # NIR, PAN, BLU band indices (1-based)
DTM_KEY = "DTEEC_002888_2025_002176_2025_A01"
RSPH = 3393833.2607584
DEG = RSPH * math.pi / 180.0
COS20 = math.cos(math.radians(20.0))
XOFF0 = (77.06 - 180.0) * DEG       # shifted-frame lon origin -> GSA clon-180, no cos scale
S_INV = Affine(COS20, 0, -XOFF0 * COS20, 0, 1, 0)   # GSA frame -> shifted frame
BOX_M = 120.0
CHIP_RES = 0.25
SEED = 42


def gsa_window(tile: Path, x0: float, x1: float, y0: float, y1: float,
               corr: Affine, bands=NPB, res: float | None = None) -> tuple[np.ndarray, Affine]:
    """Read bands of a GSA tile over a shifted-frame box, optionally decimated to `res`.

    Returns (n, h, w) array and its shifted-frame transform incl. `corr`."""
    gx0, gx1 = (x0 / COS20 + XOFF0, x1 / COS20 + XOFF0)
    with rasterio.open(tile) as ds:
        inv = ~ds.transform
        c0, r0 = inv * (gx0, y1)
        c1, r1 = inv * (gx1, y0)
        win = Window(int(c0) - 2, int(r0) - 2, int(c1 - c0) + 4, int(r1 - r0) + 4)
        f = max(1, int(round(res / ds.res[0]))) if res else 1
        out = (len(bands), int(win.height // f), int(win.width // f))
        a = read_reflectance(ds, bands, window=win, out_shape=out, resampling=Resampling.average,
                    boundless=True, fill_value=np.nan).astype(np.float32)
        t_g = (ds.transform * Affine.translation(win.col_off, win.row_off)
               * Affine.scale(win.width / out[2], win.height / out[1]))
    a[a <= 0] = np.nan
    return a, corr * (S_INV * t_g)


def resample_to(src: np.ndarray, src_tr: Affine, dst_tr: Affine, shape,
                method=Resampling.bilinear) -> np.ndarray:
    dst = np.full(shape, np.nan, np.float32)
    reproject(src, dst, src_transform=src_tr, src_crs="EPSG:4326", src_nodata=np.nan,
              dst_transform=dst_tr, dst_crs="EPSG:4326", dst_nodata=np.nan, resampling=method)
    return dst


def read_red_window(red_ds, red_tr: Affine, x0: float, x1: float, y0: float, y1: float,
                    res: float) -> tuple[np.ndarray, Affine]:
    """Windowed, decimated read of the RED ortho (whole-frame decode overflows openjpeg)."""
    inv = ~red_tr
    c0, r0 = inv * (x0, y1)
    c1, r1 = inv * (x1, y0)
    win = Window(int(c0), int(r0), int(c1 - c0), int(r1 - r0))
    f = max(1, int(round(res / red_ds.res[0])))
    out = (int(win.height // f), int(win.width // f))
    a = red_ds.read(1, window=win, out_shape=out, resampling=Resampling.average,
                    boundless=True, fill_value=0).astype(np.float32)
    a[a <= 0] = np.nan
    tr = red_tr * Affine.translation(win.col_off, win.row_off) * Affine.scale(win.width / out[1], win.height / out[0])
    return a, tr


def measure_tile_offset(tile_key: str, sites: list, red_ds, red_tr: Affine) -> dict:
    xs = [m.x_m for m in sites]; ys = [m.y_m for m in sites]
    x0, x1 = min(xs) - 400, max(xs) + 400
    y0, y1 = min(ys) - 400, max(ys) + 400
    n1c, n1r = int(x1 - x0), int(y1 - y0)
    tr1 = Affine(1.0, 0, x0, 0, -1.0, y1)
    red_win, red_win_tr = read_red_window(red_ds, red_tr, x0, x1, y0, y1, 1.0)
    ref = resample_to(red_win, red_win_tr, tr1, (n1r, n1c), Resampling.average)
    a, t = gsa_window(TILES[tile_key], x0, x1, y0, y1, Affine.identity(), bands=[3], res=1.0)
    mov = resample_to(a[0], t, tr1, (n1r, n1c))            # PAN band at 1 m
    fit, tiles, rh, mh = register(ref, np.isfinite(ref), mov, np.isfinite(mov), 1.0,
                                  tile=256, stride=128,
                                  label=f"{tile_key} PAN vs RED ortho (1 m)", coarse_max_dim=2048)
    assert fit.get("ok"), (tile_key, fit)
    qc_figure(rh, mh, tiles, fit, fit["label"], OUT / f"fig_gallery_v4_{tile_key}_qc.png")
    print(f"{tile_key}: dx={fit['dx_m']:+.1f} m dy={fit['dy_m']:+.1f} m "
          f"(MAD {fit['dx_mad_m']:.1f}/{fit['dy_mad_m']:.1f}, n_good {fit['n_good']}, "
          f"conf {fit['conf_median']:.2f})")
    return fit


def main() -> None:
    t0 = datetime.now(timezone.utc)
    o = json.loads(FRAME.read_text())["hirise"][DTM_KEY]["measured"]

    sites = pd.read_csv(OUT / "selected_sites.csv")
    sites = sites[sites.dtm == DTM_KEY].reset_index(drop=True)
    met = pd.read_csv(OUT / "fringe_metrics_v2.csv")
    met = met[met.dtm == DTM_KEY]
    rows = []
    for _, s in sites.iterrows():
        dd = np.hypot(met.x_m - s.x_dtm_frame_m, met.y_m - s.y_dtm_frame_m)
        m = met.loc[dd.idxmin()]
        assert dd.min() < 1.0, (s.panel, dd.min())
        tile = "GSA10" if s.panel == "P4" else "GSA12"
        rows.append((s.panel, tile, m))
    print("sites:", [(p, t) for p, t, _ in rows])

    red_ds0 = rasterio.open(RED_ORTHO)
    red_tr = Affine.translation(o["dx_m"], o["dy_m"]) * red_ds0.transform

    fits = {}
    for tk in ("GSA12", "GSA10"):
        group = [m for _, t, m in rows if t == tk]
        fits[tk] = measure_tile_offset(tk, group, red_ds0, red_tr)
    red_ds0.close()
    corr = {tk: Affine.translation(fits[tk]["dx_m"], fits[tk]["dy_m"]) for tk in fits}

    with rasterio.open(HIRISE[DTM_KEY]) as dtm_ds:
        z_tr = Affine.translation(o["dx_m"], o["dy_m"]) * dtm_ds.transform
        dcrs = dtm_ds.crs
        z_nod = dtm_ds.nodata
        lines = clay_boundaries(dcrs)
        s_npz = np.load(OUT / f"{DTM_KEY}_profiles.npz")
        red_ds = rasterio.open(RED_ORTHO)
        n_ps = int(2 * BOX_M / CHIP_RES)
        fig, axes = plt.subplots(len(rows), 4, figsize=(16.5, 3.0 * len(rows)), squeeze=False)
        for ri, ((panel, tk, m), (axr, axps, axm, axp)) in enumerate(zip(rows, axes), 1):
            ext = (m.x_m - BOX_M, m.x_m + BOX_M, m.y_m - BOX_M, m.y_m + BOX_M)
            box = shapely.box(ext[0], ext[2], ext[1], ext[3])
            tr_ps = Affine(CHIP_RES, 0, ext[0], 0, -CHIP_RES, ext[3])
            # HiRISE RED
            invr = ~red_tr
            c0, r0 = invr * (ext[0], ext[3]); c1, r1 = invr * (ext[1], ext[2])
            win = Window(int(c0), int(r0), int(c1 - c0), int(r1 - r0))
            red = red_ds.read(1, window=win, boundless=True, fill_value=0).astype(np.float32)
            red[red <= 0] = np.nan
            v = np.nanpercentile(red, [2, 98])
            axr.imshow(red, cmap="gray", extent=ext, vmin=v[0], vmax=v[1])
            draw_lines(axr, lines, box)
            axr.set_ylabel(panel, fontsize=12, rotation=0, labelpad=18, va="center", fontweight="bold")
            # PANCO GSA pansharpened NPB
            a, t = gsa_window(TILES[tk], ext[0] - 10, ext[1] + 10, ext[2] - 10, ext[3] + 10, corr[tk])
            rgb = np.zeros((n_ps, n_ps, 3), np.float32)
            ok = True
            for bi in range(3):
                b = resample_to(a[bi], t, tr_ps, (n_ps, n_ps))
                good = np.isfinite(b)
                ok &= good.mean() > 0.3
                if good.any():
                    p1, p2 = np.nanpercentile(b, [2, 98])
                    rgb[..., bi] = np.clip((b - p1) / max(p2 - p1, 1e-9), 0, 1)
            axps.imshow(rgb, extent=ext, interpolation="nearest")
            if not ok:
                axps.text(0.5, 0.5, "outside PANCO tile", transform=axps.transAxes,
                          color="white", ha="center", fontsize=8)
            draw_lines(axps, lines, box)
            # hillshade
            invz = ~z_tr
            c0, r0 = invz * (ext[0], ext[3]); c1, r1 = invz * (ext[1], ext[2])
            win = Window(int(c0), int(r0), int(c1 - c0), int(r1 - r0))
            zz = dtm_ds.read(1, window=win, boundless=True, fill_value=np.nan).astype(np.float32)
            zz[(zz < -1e30) | (zz == z_nod)] = np.nan
            axm.imshow(hillshade(zz, abs(z_tr.a)), cmap="gray", extent=ext)
            draw_lines(axm, lines, box)
            axm.set_xlabel(f"bump {m.bump_clay_max_m:.1f} m at x={m.bump_clay_x_m:.0f} m; "
                           f"step {m.step_m:+.1f} m", fontsize=7)
            for ax in (axr, axps, axm):
                ax.plot(m.x_m, m.y_m, "o", mfc="none", mec="#0072B2", ms=9, mew=1.5)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            # profile
            prof = s_npz["profs"][int(m.idx)]
            rel = prof - np.nanmedian(prof[np.abs(X) <= 2])
            axp.plot(X, rel, color="#0072B2", lw=1.5)
            axp.axvline(0, color="0.4", lw=0.8)
            axp.axvline(m.bump_clay_x_m, color="#D55E00", lw=0.8, ls="--")
            axp.set_xlabel("m from contact (clay < 0)", fontsize=8)
            axp.grid(True, color="0.93")
            ylim = max(5.0, 1.5 * m.bump_clay_max_m)
            axp.set_ylim(-ylim, ylim); axp.tick_params(labelsize=7)
            if ri == 1:
                axr.set_title("HiRISE RED 25 cm + contact", fontsize=9)
                axps.set_title("PANCO GSA pansharpened, NPB (R=939, G=678, B=494 nm)", fontsize=9)
                axm.set_title("HiRISE 1 m hillshade + contact", fontsize=9)
                axp.set_title("contact-normal profile", fontsize=9)
        offs = ", ".join(f"{k} dx={fits[k]['dx_m']:+.1f}/dy={fits[k]['dy_m']:+.1f} m" for k in fits)
        fig.suptitle(f"{DTM_KEY} selected fringe sites; orange = mapped clay contact (hand-drawn); "
                     f"PANCO offsets measured vs RED ortho: {offs}", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / "fig_gallery_v4_cassis_ps.png", dpi=150)
        fig.savefig(OUT / "fig_gallery_v4_cassis_ps.pdf")
        plt.close(fig)
        red_ds.close()

    (OUT / "params_gallery_v4.json").write_text(json.dumps(dict(
        panels=[(p, t) for p, t, _ in rows],
        panco_tiles={k: str(v) for k, v in TILES.items()}, npb_bands=NPB,
        frame=dict(clon_shift=77.06, clon_gsa=180.0, cos20_easting=True, r_sphere_m=RSPH),
        measured_offsets={k: {kk: fits[k][kk] for kk in
                              ("dx_m", "dy_m", "dx_mad_m", "dy_mad_m", "n_good", "conf_median")}
                          for k in fits},
        chip_res_m=CHIP_RES, box_half_m=BOX_M, seed=SEED,
        started_utc=t0.isoformat(), finished_utc=datetime.now(timezone.utc).isoformat()), indent=2))
    print("wrote", OUT / "fig_gallery_v4_cassis_ps.png")


if __name__ == "__main__":
    main()
