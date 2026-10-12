"""Stage 0 of PLAN_revision_contact_3D.md: frame check against the Robbins
controlled CTX mosaic (USGS, 6 m, streamed via /vsicurl/).

A. Each CRISM MTR3 cube (VNIR band ~749 nm on the 18 m grid) vs CTX.
B. Each HiRISE orthoimage (decimated to 6 m) vs CTX, with and without the
   manual world-file shift (.IGwx / .LLwx sidecars written in ArcGIS).

Outputs (outputs/revision/contact_3d/frame/): coreg_offsets.json, per-target
tile CSVs and QC PNGs, and the CTX mosaic resampled onto each MTR3 grid.

Usage: python code/contact_coreg_frame.py [--skip-hirise] [--skip-crism]
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject, transform_bounds

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import CUBE_IDS  # noqa: E402
from contact_coreg_lib import coarse_lock, fit_offsets, highpass, self_test, tile_offsets  # noqa: E402

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
CTX_URL = ("/vsicurl/https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/Mars/"
           "Mars_MRO_CTX_Equi_Mosaics_Robbins/6mpp/MC13_6mpp.16bit.tif")
OUT = ROOT / "outputs" / "revision" / "contact_3d" / "frame"
OUT.mkdir(parents=True, exist_ok=True)
from config import HIRISE_PDS_DIR  # noqa: E402
NP_DTMS = HIRISE_PDS_DIR
CRISM_BAND_NM = 749.0
HP_SCALE_M = 110.0           # high-pass sigma in metres (same physical scale at 18 m and 6 m)
CTX_MARGIN_M = 1500.0
HIRISE = {                   # DTM -> ortho used for matching
    "DTEEC_006712_2020_005855_2020_A01": NP_DTMS / "PSP_006712_2020_RED_A_01_ORTHO.JP2",
    "DTEEC_002888_2025_002176_2025_A01": NP_DTMS / "PSP_002176_2025_RED_A_01_ORTHO.JP2",
    "DTEEC_008993_2020_009626_2020_A01": NP_DTMS / "Long ridges site" / "PSP_009626_2020_RED_C_01_ORTHO.JP2",
}
DTM_DIRS = {"DTEEC_006712_2020_005855_2020_A01": NP_DTMS, "DTEEC_002888_2025_002176_2025_A01": NP_DTMS,
            "DTEEC_008993_2020_009626_2020_A01": NP_DTMS / "Long ridges site"}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_worldfile(path: Path) -> Affine | None:
    """ESRI world file (A, D, B, E, C, F; C/F = centre of the UL pixel) -> corner-based Affine."""
    if not path.exists():
        return None
    a, d, b, e, c, f = [float(x) for x in path.read_text().split()[:6]]
    return Affine(a, b, c - a / 2 - b / 2, d, e, f - d / 2 - e / 2)


def ctx_on_grid(dst_crs, dst_transform, dst_shape) -> tuple[np.ndarray, np.ndarray]:
    """Stream the CTX window covering the grid and resample it (area average) onto the grid."""
    with rasterio.open(CTX_URL) as ctx:
        H, W = dst_shape
        left, top = dst_transform * (0, 0)
        right, bottom = dst_transform * (W, H)
        b = transform_bounds(dst_crs, ctx.crs, min(left, right), min(top, bottom), max(left, right), max(top, bottom), densify_pts=21)
        win = ctx.window(b[0] - CTX_MARGIN_M, b[1] - CTX_MARGIN_M, b[2] + CTX_MARGIN_M, b[3] + CTX_MARGIN_M).round_offsets().round_lengths()
        src = ctx.read(1, window=win).astype(np.float32)
        src_nd = ctx.nodata
        src_tr = ctx.window_transform(win)
        src[src == src_nd] = np.nan
        dst = np.full(dst_shape, np.nan, np.float32)
        reproject(src, dst, src_transform=src_tr, src_crs=ctx.crs, src_nodata=np.nan,
                  dst_transform=dst_transform, dst_crs=dst_crs, dst_nodata=np.nan, resampling=Resampling.average)
        return dst, np.isfinite(dst)


def register(ref: np.ndarray, ref_valid: np.ndarray, mov: np.ndarray, mov_valid: np.ndarray, res_m: float,
             tile: int, stride: int, label: str, coarse_max_dim: int | None = None) -> tuple[dict, list[dict], np.ndarray, np.ndarray]:
    sigma = HP_SCALE_M / res_m
    rh = highpass(ref, ref_valid, sigma)
    mh = highpass(mov, mov_valid, sigma)
    coarse = coarse_lock(rh, mh, ref_valid, mov_valid, max_dim=coarse_max_dim)
    tiles = tile_offsets(rh, mh, ref_valid, mov_valid, coarse, tile, stride)
    fit = fit_offsets(tiles)
    fit.update({"coarse_drow": coarse[0], "coarse_dcol": coarse[1], "res_m": res_m, "tile_px": tile, "stride_px": stride,
                "hp_sigma_px": sigma, "label": label})
    if fit["ok"]:
        # moving content must shift by (+drow,+dcol) px -> georeferencing correction of the moving raster
        fit["dx_m"] = fit["dcol"] * res_m
        fit["dy_m"] = -fit["drow"] * res_m
        fit["dx_mad_m"] = fit["mad_col"] * res_m
        fit["dy_mad_m"] = fit["mad_row"] * res_m
    return fit, tiles, rh, mh


def qc_figure(rh, mh, tiles, fit, title, out_png) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for ax, img, t in zip(axes[:2], (rh, mh), ("reference (CTX, high-pass)", "moving (high-pass)")):
        ax.imshow(img, cmap="gray", vmin=-2.5, vmax=2.5); ax.set_title(t, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    ax = axes[2]
    if tiles:
        r = np.array([t["row"] for t in tiles]); c = np.array([t["col"] for t in tiles])
        dr = np.array([t["drow"] for t in tiles]); dc = np.array([t["dcol"] for t in tiles]); cf = np.array([t["conf"] for t in tiles])
        ax.imshow(rh, cmap="gray", vmin=-2.5, vmax=2.5, alpha=0.4)
        # axis is image-oriented (row increases downward), so the row component is passed unnegated
        q = ax.quiver(c, r, dc, dr, cf, cmap="viridis", scale=None, scale_units="xy", angles="xy", width=0.004, clim=(0, 0.8))
        fig.colorbar(q, ax=ax, fraction=0.04, label="edge-NCC confidence")
    ax.set_xticks([]); ax.set_yticks([])
    s = (f"drow={fit['drow']:+.2f} px, dcol={fit['dcol']:+.2f} px  (dx={fit['dx_m']:+.1f} m, dy={fit['dy_m']:+.1f} m); "
         f"n_good={fit['n_good']}/{fit['n_tiles']}, conf_med={fit['conf_median']:.2f}") if fit.get("ok") else "fit failed"
    ax.set_title("per-tile offsets (moving -> reference)", fontsize=10)
    fig.suptitle(f"{title}\n{s}", fontsize=10)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def write_tiles(tiles: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["row", "col", "drow", "dcol", "conf", "conf_unshifted"]); w.writeheader(); w.writerows(tiles)


def crism_band(cube: str) -> tuple[np.ndarray, np.ndarray, object, Affine, dict]:
    low = cube.lower()
    hdr = ROOT / "cubes" / f"{low}_07_if166j_mtr3" / f"{low}_07_if166j_mtr3.hdr"
    txt = hdr.read_text()
    get = lambda k: int(re.search(rf"^{k}\s*=\s*(\d+)", txt, re.M | re.I).group(1))
    ns, nl, nb = get("samples"), get("lines"), get("bands")
    wav = np.array([float(x) for x in re.search(r"wavelength\s*=\s*\{([^}]*)\}", txt, re.S | re.I).group(1).split(",")])
    bi = int(np.argmin(np.abs(wav - CRISM_BAND_NM)))
    mm = np.memmap(hdr.with_suffix(".img"), dtype="<f4", mode="r", shape=(nb, nl, ns))
    band = np.array(mm[bi]).astype(np.float32)
    band[band >= 65535.0] = np.nan
    gt = sorted(glob.glob(str(ROOT / "outputs" / "_v8_mtr3_with_brown" / cube / "geotiffs" / "*.tif")))[0]
    with rasterio.open(gt) as ds:
        assert ds.shape == band.shape, (ds.shape, band.shape)
        crs, tr = ds.crs, ds.transform
    return band, np.isfinite(band), crs, tr, {"band_index": bi, "band_nm": float(wav[bi]), "grid_from": str(Path(gt).relative_to(ROOT))}


def run_crism(results: dict) -> None:
    for cube in CUBE_IDS:
        band, valid, crs, tr, meta = crism_band(cube)
        ctx, ctxv = ctx_on_grid(crs, tr, band.shape)
        with rasterio.open(OUT / f"{cube}_ctx_on_mtr3.tif", "w", driver="GTiff", width=band.shape[1], height=band.shape[0], count=1,
                           dtype="float32", crs=crs, transform=tr, nodata=np.nan, compress="deflate") as dst:
            dst.write(ctx.astype(np.float32), 1)
        fit, tiles, rh, mh = register(ctx, ctxv, band, valid, abs(tr.a), tile=128, stride=64, label=f"{cube} MTR3 {meta['band_nm']:.0f} nm vs CTX")
        fit.update(meta)
        write_tiles(tiles, OUT / f"{cube}_tiles.csv")
        qc_figure(rh, mh, tiles, fit, fit["label"], OUT / f"{cube}_qc.png")
        results["crism"][cube] = fit
        print(cube, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in fit.items() if k in ("ok", "drow", "dcol", "dx_m", "dy_m", "dx_mad_m", "dy_mad_m", "n_good", "n_tiles", "conf_median")})


def run_hirise(results: dict) -> None:
    for dtm, ortho in HIRISE.items():
        with rasterio.open(DTM_DIRS[dtm] / f"{dtm}.IMG") as dtm_ds:
            crs = dtm_ds.crs          # ortho JP2 headers can carry a malformed CRS; the DTM label is authoritative
        with rasterio.open(ortho) as ds:
            res0 = abs(ds.transform.a)
            f = max(1, int(round(6.0 / res0)))
            out_shape = (ds.height // f, ds.width // f)
            img = ds.read(1, out_shape=out_shape, resampling=Resampling.average).astype(np.float32)
            tr0 = ds.transform * Affine.scale(f)
            nd = ds.nodata
        img[(img == 0) | (img == nd) | ~np.isfinite(img)] = np.nan
        valid = np.isfinite(img)
        res_m = abs(tr0.a)
        # manual world-file shift (sidecar of the ortho or of the DTM; both describe the same grid shift)
        sidecars = [ortho.with_suffix(".LLwx"), ortho.with_suffix(".J2wx"), DTM_DIRS[dtm] / f"{dtm}.IGwx"]
        manual = None
        for sc in sidecars:
            wf = read_worldfile(sc)
            if wf is not None:
                src = sc
                with rasterio.open(DTM_DIRS[dtm] / f"{dtm}.IMG" if sc.suffix == ".IGwx" else ortho) as ref_ds:
                    t_label = ref_ds.transform
                manual = {"sidecar": str(sc.relative_to(ROOT)), "dx_m": wf.c - t_label.c, "dy_m": wf.f - t_label.f}
                break
        ctx, ctxv = ctx_on_grid(crs, tr0, img.shape)
        fit, tiles, rh, mh = register(ctx, ctxv, img, valid, res_m, tile=256, stride=128, label=f"{ortho.name} (6 m) vs CTX, label georef")
        write_tiles(tiles, OUT / f"{dtm}_tiles.csv")
        qc_figure(rh, mh, tiles, fit, fit["label"], OUT / f"{dtm}_qc.png")
        rec = {"ortho": str(ortho.relative_to(ROOT)), "ortho_native_res_m": res0, "decimation": f, "measured": fit, "manual_worldfile": manual}
        if manual and fit.get("ok"):
            rec["residual_after_manual_m"] = {"dx": fit["dx_m"] - manual["dx_m"], "dy": fit["dy_m"] - manual["dy_m"]}
        results["hirise"][dtm] = rec
        print(dtm, "measured:", {k: round(fit[k], 1) for k in ("dx_m", "dy_m", "dx_mad_m", "dy_mad_m")} if fit.get("ok") else "FAILED",
              "| manual:", None if manual is None else {k: round(manual[k], 1) for k in ("dx_m", "dy_m")},
              "| residual:", rec.get("residual_after_manual_m"))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-crism", action="store_true"); ap.add_argument("--skip-hirise", action="store_true")
    a = ap.parse_args()
    self_test()
    t0 = datetime.now(timezone.utc)
    results = {"crism": {}, "hirise": {}, "ctx_url": CTX_URL, "hp_scale_m": HP_SCALE_M, "crism_band_nm": CRISM_BAND_NM,
               "sign_convention": "moving content shifts by (+drow,+dcol) px to match CTX; correction UL_X += dx_m, UL_Y += dy_m",
               "started_utc": t0.isoformat(), "script_md5": md5(Path(__file__))}
    prev = OUT / "coreg_offsets.json"
    if prev.exists():                      # keep sections from an earlier partial run when a part is skipped
        old = json.loads(prev.read_text())
        results["crism"], results["hirise"] = old.get("crism", {}), old.get("hirise", {})
    if not a.skip_crism:
        run_crism(results)
    if not a.skip_hirise:
        run_hirise(results)
    results["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (OUT / "coreg_offsets.json").write_text(json.dumps(results, indent=2, default=str))
    print("wrote", OUT / "coreg_offsets.json")


if __name__ == "__main__":
    main()
