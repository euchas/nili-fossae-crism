"""Site-local residual of the CRISM MTR3 georeference vs CTX (after the frame shift).

The frame-level CRISM-vs-CTX shift (coreg_offsets.json) is a cube-wide fit
with 50/18 m tile scatter. For a 780 m figure window that scatter is a
visible misregistration, so the residual is re-measured locally: the MTR3
749 nm band (frame shift applied) vs the Robbins CTX mosaic on an 18 m grid
in a window around the site, same high-pass / coarse-lock / tile fit as the
frame stage. The result is written to outputs/revision/sites/crism_local_
residual.json and used by contact_paper_fig_site1.py as an additional
translation of the CRISM layer. Deterministic.

Usage:  python code/contact_crism_local_residual.py [site1_wide] [band_nm]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from stage_v7c_mtr3_pipeline import load_mtr3  # noqa: E402
from contact_site_panels import SITES, working_crs, lonlat_to_xy, read_on_grid  # noqa: E402
from contact_coreg_frame import ctx_on_grid  # noqa: E402
from contact_coreg_lib import coarse_lock, fit_offsets, highpass, tile_offsets  # noqa: E402
from contact_height_above_contact import md5  # noqa: E402

CUBE = "FRT0000A4FC"
HDR = ROOT / "cubes" / f"{CUBE.lower()}_07_if166j_mtr3" / f"{CUBE.lower()}_07_if166j_mtr3.hdr"
FRAME_JSON = ROOT / "outputs" / "revision" / "contact_3d" / "frame" / "coreg_offsets.json"
OUT = ROOT / "outputs" / "revision" / "sites"
BAND_NM = float(sys.argv[2]) if len(sys.argv) > 2 else 749.0     # S detector 749; L detector e.g. 1330 / 1750
RES_M, HP_M = 18.0, 110.0
WINDOWS_M = (4000.0, 6000.0)          # two window sizes as a consistency check
TILE, STRIDE = 64, 32


def main() -> None:
    site = sys.argv[1] if len(sys.argv) > 1 else "site1_wide"
    t0 = datetime.now(timezone.utc)
    crs = working_crs(); S = SITES[site]
    xs, ys = lonlat_to_xy(S["lon"], S["lat"], crs)
    xc, yc = xs + S.get("center_offset_m", (0, 0))[0], ys + S.get("center_offset_m", (0, 0))[1]
    cube, wav, meta = load_mtr3(HDR)
    bi = int(np.abs(wav * 1000 - BAND_NM).argmin())
    band = cube[bi].astype(np.float32); del cube
    gt = sorted((ROOT / "outputs" / "_v8_mtr3_with_brown" / CUBE / "geotiffs").glob("*.tif"))[0]
    with rasterio.open(gt) as ds:
        m_tr, m_crs = ds.transform, ds.crs
    off = json.loads(FRAME_JSON.read_text())["crism"][CUBE]
    m_tr_eff = Affine.translation(off["dx_m"], off["dy_m"]) * m_tr
    tmp = OUT / f"_tmp_{CUBE}_749.tif"
    with rasterio.open(tmp, "w", driver="GTiff", width=band.shape[1], height=band.shape[0], count=1, dtype="float32",
                       crs=m_crs, transform=m_tr_eff, nodata=np.nan) as dst:
        dst.write(band, 1)
    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for W in WINDOWS_M:
        n = int(W / RES_M); tr = Affine(RES_M, 0, xc - W / 2, 0, -RES_M, yc + W / 2)
        ref, refv = ctx_on_grid(crs, tr, (n, n))
        mov = read_on_grid(str(tmp), m_tr_eff, m_crs, None, crs, tr, (n, n), Resampling.bilinear)
        movv = np.isfinite(mov)
        sig = HP_M / RES_M
        rh, mh = highpass(ref, refv, sig), highpass(np.where(movv, mov, np.nan), movv, sig)
        coarse = coarse_lock(rh, mh, refv, movv)
        fit = fit_offsets(tile_offsets(rh, mh, refv, movv, coarse, TILE, STRIDE))
        if fit.get("ok"):
            fit["residual_dx_m"] = fit["dcol"] * RES_M; fit["residual_dy_m"] = -fit["drow"] * RES_M
            fit["residual_mad_m"] = [fit["mad_col"] * RES_M, fit["mad_row"] * RES_M]
        fit["coarse_px"] = coarse; fit["window_m"] = W
        results[f"{W:.0f}"] = fit
        print(f"window {W:.0f} m: coarse {coarse} px; residual dx {fit.get('residual_dx_m', float('nan')):+.1f} m, dy {fit.get('residual_dy_m', float('nan')):+.1f} m "
              f"(MAD {fit.get('residual_mad_m')}, n_good {fit.get('n_good')}, conf {fit.get('conf_median', float('nan')):.2f})")
        if W == WINDOWS_M[0]:
            for ax, img, t in zip(axes[:2], (rh, mh), ("CTX high-pass (18 m grid)", f"CRISM {BAND_NM:.0f} nm high-pass, frame shift applied")):
                ax.imshow(img, cmap="gray", vmin=-2.5, vmax=2.5); ax.set_title(t, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    tmp.unlink(missing_ok=True)
    ax = axes[2]
    for k, f in results.items():
        if f.get("ok"):
            ax.errorbar(f["residual_dx_m"], f["residual_dy_m"], xerr=f["residual_mad_m"][0], yerr=f["residual_mad_m"][1], fmt="o", capsize=3, label=f"{k} m window (n_good {f['n_good']}, conf {f['conf_median']:.2f})")
    ax.axhline(0, color="0.8"); ax.axvline(0, color="0.8"); ax.set_xlabel("residual dx (m, apply to CRISM)"); ax.set_ylabel("residual dy (m)")
    ax.set_title("site-local CRISM residual after the frame shift", fontsize=10); ax.legend(fontsize=8); ax.grid(True, color="0.95"); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(OUT / f"fig_crism_local_residual_{site}_{BAND_NM:.0f}nm.png", dpi=150)
    # choose: the 6 km fit if ok and confident, else the 4 km one
    pick = next((results[k] for k in (f"{WINDOWS_M[1]:.0f}", f"{WINDOWS_M[0]:.0f}") if results[k].get("ok") and results[k]["conf_median"] >= 0.3), None)
    out = {"cube": CUBE, "site": site, "band_nm": BAND_NM, "frame_shift_m": (off["dx_m"], off["dy_m"]), "fits": results,
           "applied": {"dx_m": pick["residual_dx_m"], "dy_m": pick["residual_dy_m"], "window_m": pick["window_m"]} if pick else None,
           "script": Path(__file__).name, "script_md5": md5(Path(__file__)), "started_utc": t0.isoformat(),
           "finished_utc": datetime.now(timezone.utc).isoformat()}
    (OUT / f"crism_local_residual_{site}_{BAND_NM:.0f}nm.json").write_text(json.dumps(out, indent=2, default=str))
    print("applied:", out["applied"])


if __name__ == "__main__":
    main()
