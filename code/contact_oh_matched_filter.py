"""Per-pixel 1.4 um OH band detection by matched filter, Plebani chain.

For every sensor pixel of the bland-ratioed (crism_ml) cubes:
  1. fit a cubic background over 1.28-1.60 um, EXCLUDING the test window
     (same treatment for the signal window and for the two null windows);
  2. matched-filter the continuum-removed residual with Gaussians on a grid
     of centres and observed FWHM 20-40 nm (bounded at/above the ~14 nm
     instrument response, per methods review);
  3. record the best amplitude, centre, FWHM and amplitude SNR;
  4. repeat identically in two BAND-FREE windows to build an empirical null
     distribution - the detection threshold is the 99th percentile of that
     null (look-elsewhere included, since the null uses more trial
     positions than the signal window), NOT a value from a table.

Products: per-pixel detection maps colour-coded by fitted band centre,
block-median (504 m) maps as the statistically defensible version, and the
validation battery (cap negative control, OLINDEX3 orthogonality, per-column
stratification). Deterministic.

Usage:  python code/contact_oh_matched_filter.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.warp import transform as rtransform
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import CUBE_IDS  # noqa: E402
from contact_distance_to_contact import PALETTE, V8_PARAMS, load_cube, masks_from_labels, md5  # noqa: E402
from contact_band_center_map import unit_masks  # noqa: E402
from contact_carbonate_patches import CUBES, GEOMAP, FRAME, LONLAT, load_ddr_lonlat  # noqa: E402
from contact_band_center_plebani import ZRAT  # noqa: E402
from figstyle import draw_units, label_panels  # noqa: E402

OUT = ROOT / "outputs" / "revision" / "band_center_map" / "oh_filter"
OUT.mkdir(parents=True, exist_ok=True)

FIT_WIN = (1.28, 1.60)
SIGNAL_WIN = (1.370, 1.430)
NULL_WINS = ((1.305, 1.350), (1.470, 1.515))
GUARD_NM = 12.0                    # background exclusion half-margin around a test window
BG_ORDER = 3                       # fixed a priori
CENTRE_STEP = 0.002
FWHM_NM = (20.0, 26.0, 32.0, 40.0)
NULL_PCT = 99.0
FILL_TOL_M = 30.0
BLOCK_PX = 28                      # 504 m


def design(wav_fit, exclude):
    use = ~exclude
    A = np.vander(wav_fit[use], BG_ORDER + 1)
    return np.linalg.pinv(A), np.vander(wav_fit, BG_ORDER + 1), use


def grid_matrix(wav_fit, win):
    centres = np.arange(win[0], win[1] + 1e-9, CENTRE_STEP)
    cols, meta = [], []
    for c in centres:
        for f in FWHM_NM:
            s = f / 2.3548 / 1000.0
            g = np.exp(-0.5 * ((wav_fit - c) / s) ** 2)
            cols.append(g)
            meta.append((c, f))
    G = np.stack(cols, axis=1)
    return G, np.array(meta)


def run_window(Y, wav_fit, win, noise_mask):
    """Background fit excluding the window, then matched filter inside it."""
    ex = (wav_fit >= win[0] - GUARD_NM / 1000.0) & (wav_fit <= win[1] + GUARD_NM / 1000.0)
    pinv, Afull, use = design(wav_fit, ex)
    coef = Y[:, use] @ pinv.T
    bg = coef @ Afull.T
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 1.0 - Y / bg
    r[~np.isfinite(r)] = 0.0
    sigma = np.std(r[:, noise_mask], axis=1)
    G, meta = grid_matrix(wav_fit, win)
    gg = np.sum(G * G, axis=0)
    amp = (r @ G) / gg                      # (n_pix, n_grid) closed-form amplitude
    snr = amp * np.sqrt(gg)[None, :] / np.maximum(sigma, 1e-9)[:, None]
    best = np.argmax(snr, axis=1)
    idx = np.arange(Y.shape[0])
    return {"amp": amp[idx, best], "snr": snr[idx, best],
            "centre": meta[best, 0], "fwhm": meta[best, 1], "sigma": sigma}


def fit_cube(if2, wav):
    fw = (wav >= FIT_WIN[0]) & (wav <= FIT_WIN[1])
    wav_fit = wav[fw]
    Y = if2[:, fw].astype(np.float64)
    ok = np.isfinite(Y).all(axis=1) & (np.abs(Y).sum(axis=1) > 1e-6)
    noise_mask = np.ones(wav_fit.size, bool)
    for w in (SIGNAL_WIN,) + NULL_WINS:
        noise_mask &= ~((wav_fit >= w[0] - GUARD_NM / 1000.0) & (wav_fit <= w[1] + GUARD_NM / 1000.0))
    res = {"ok": ok}
    sig = run_window(Y[ok], wav_fit, SIGNAL_WIN, noise_mask)
    nulls = [run_window(Y[ok], wav_fit, w, noise_mask) for w in NULL_WINS]
    null_snr = np.maximum(nulls[0]["snr"], nulls[1]["snr"])
    res.update({k: v for k, v in sig.items()})
    res["null_snr"] = null_snr
    return res


def project(cube, c, fr):
    lon, lat = load_ddr_lonlat(CUBES[cube]["ddr"])
    lon, lat = lon.ravel(), lat.ravel()
    fz = np.load(GEOMAP / f"{cube}_fractions.npz")
    tr = Affine.from_gdal(*fz["transform"])
    with rasterio.open(sorted((ROOT / "outputs" / "_v8_mtr3_with_brown" / cube / "geotiffs").glob("*.tif"))[0]) as ds:
        crs = ds.crs
    o = json.loads(FRAME.read_text())["crism"][cube]
    okp = np.isfinite(lon) & np.isfinite(lat)
    xs = np.full(lon.shape, np.nan); ys = np.full(lon.shape, np.nan)
    X, Y = rtransform(LONLAT, crs, lon[okp].astype(float), lat[okp].astype(float))
    xs[okp] = np.asarray(X) + o["dx_m"]; ys[okp] = np.asarray(Y) + o["dy_m"]
    colf, rowf = ~tr * (xs, ys)
    return xs, ys, np.floor(rowf), np.floor(colf), tr


def main() -> None:
    t0 = datetime.now(timezone.utc)
    summary, colrows, inputs = [], [], {}
    for cube in CUBE_IDS:
        stem = CUBES[cube]["stem"]
        z = np.load(ZRAT / f"{stem}_ratioed_bland.npz")
        if2 = z["if2"].astype(np.float32)
        wav = z["bands"].astype(np.float64)
        wav = wav / 1000.0 if wav.max() > 100 else wav
        W_s = int(z["im_shape"][1])
        inputs[cube] = {"ratioed_npz": str(ZRAT / f"{stem}_ratioed_bland.npz")}
        res = fit_cube(if2, wav)
        del if2, z
        ok = res["ok"]
        n_ok = int(ok.sum())

        c = load_cube(cube)
        m = masks_from_labels(c["labels"])
        units, d, fr = unit_masks(cube, m)
        xs, ys, gr, gc, tr = project(cube, c, fr)
        H, W = c["labels"].shape
        inside = np.isfinite(xs) & (gr >= 0) & (gr < H) & (gc >= 0) & (gc < W)
        gri = np.where(inside, gr, 0).astype(int); gci = np.where(inside, gc, 0).astype(int)
        samp = lambda a: np.where(inside, a[gri, gci], np.nan)
        f_oliv, f_clay, f_cap = samp(fr["oliv"]), samp(fr["clay"]), samp(fr["cap"])
        olind = samp(c["params"][..., V8_PARAMS.index("OLINDEX3")])
        mvalid = np.where(inside, m["valid"][gri, gci], False)

        def expand(v, fill=np.nan):
            a = np.full(ok.size, fill, float)
            a[ok] = v
            return a
        amp, snr, cen, fw, nsnr = (expand(res[k]) for k in ("amp", "snr", "centre", "fwhm", "null_snr"))
        scol = (np.arange(ok.size) % W_s).astype(np.int16)
        usable = ok & inside & mvalid

        thr = float(np.nanpercentile(nsnr[usable], NULL_PCT))
        det = usable & (snr >= thr) & (cen >= SIGNAL_WIN[0] + 1e-9) & (cen <= SIGNAL_WIN[1] - 1e-9)
        np.savez_compressed(OUT / f"{cube}_ohfit_sensor.npz", amp=amp.astype(np.float32),
                            snr=snr.astype(np.float32), centre=cen.astype(np.float32),
                            fwhm=fw.astype(np.float32), null_snr=nsnr.astype(np.float32),
                            det=det, scol=scol)

        # unit statistics + orthogonality + negative control
        unit_defs = {"olivine map-pure": usable & units["olivine map-pure"][gri, gci] & inside,
                     "clay map-pure": usable & (f_clay >= 0.9),
                     "cap map-pure": usable & (f_cap >= 0.9)}
        for uname, um in unit_defs.items():
            n = int(um.sum())
            dfrac = float((um & det).sum() / n) if n else np.nan
            good = um & np.isfinite(olind) & np.isfinite(amp)
            rho = float(spearmanr(amp[good], olind[good]).statistic) if good.sum() > 200 else np.nan
            summary.append({"cube": cube, "unit": uname, "n": n, "n_det": int((um & det).sum()),
                            "det_frac": dfrac, "thr_snr": thr,
                            "amp_med_det": float(np.median(amp[um & det])) if (um & det).any() else np.nan,
                            "centre_med_det": float(np.median(cen[um & det])) if (um & det).any() else np.nan,
                            "rho_amp_OLINDEX3": rho})
        for col in range(0, W_s, 32):
            sel = usable & (scol >= col) & (scol < col + 32)
            if sel.sum() > 200:
                colrows.append({"cube": cube, "col_bin": col,
                                "thr_snr": float(np.nanpercentile(nsnr[sel], NULL_PCT)),
                                "det_frac": float((sel & det).sum() / sel.sum())})

        # nearest-sample fill to the map grid
        src = np.column_stack([xs[usable], ys[usable]])
        tree = cKDTree(src)
        cc, rr = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        gx, gy = tr * (cc.ravel(), rr.ravel())
        dist, idx = tree.query(np.column_stack([gx, gy]), distance_upper_bound=FILL_TOL_M)
        hit = np.isfinite(dist)
        cen_u, det_u, snr_u = cen[usable], det[usable], snr[usable]
        cen_map = np.full(H * W, np.nan); det_map = np.zeros(H * W, bool); snr_map = np.full(H * W, np.nan)
        cen_map[hit] = cen_u[idx[hit]]; det_map[hit] = det_u[idx[hit]]; snr_map[hit] = snr_u[idx[hit]]
        cen_map = cen_map.reshape(H, W); det_map = det_map.reshape(H, W); snr_map = snr_map.reshape(H, W)
        det_map &= m["valid"]
        np.save(OUT / f"{cube}_oh_centre_det.npy", np.where(det_map, cen_map, np.nan))
        np.save(OUT / f"{cube}_oh_snr.npy", np.where(m["valid"], snr_map, np.nan))
        print(cube, f"n_ok={n_ok} thr_SNR={thr:.2f} det_px={int(det.sum())} "
              f"({100 * det.sum() / max(usable.sum(), 1):.2f}% of usable)")

        fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.5), sharex=True, sharey=True)
        bg = np.where(m["valid"], 0.93, np.nan)
        axes[0].imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        im = axes[0].imshow(np.where(det_map, cen_map, np.nan), cmap="plasma",
                            vmin=SIGNAL_WIN[0], vmax=SIGNAL_WIN[1], interpolation="nearest")
        axes[0].set_title(f"OH detections (SNR >= {thr:.1f}), coloured by fitted centre", fontsize=10)
        cb = fig.colorbar(im, ax=axes[0], shrink=0.8, pad=0.02)
        cb.set_label("fitted 1.4 um band centre (um)", fontsize=8.5)
        axes[1].imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        im2 = axes[1].imshow(snr_map, cmap="viridis", vmin=0, vmax=max(6.0, thr * 1.5),
                             interpolation="nearest")
        axes[1].set_title("amplitude SNR (all usable pixels)", fontsize=10)
        fig.colorbar(im2, ax=axes[1], shrink=0.8, pad=0.02)
        for ax in axes:
            draw_units(ax, fr, units=("clay", "cap"))
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{cube}: 1.4 um OH matched-filter detection, Plebani bland-ratioed chain\n"
                     f"threshold = {NULL_PCT:.0f}th percentile of the band-free null distribution",
                     fontsize=11)
        fig.tight_layout()
        label_panels(fig)
        fig.savefig(OUT / f"fig_oh_{cube}.png", dpi=200, bbox_inches="tight")
        fig.savefig(OUT / f"fig_oh_{cube}.pdf", bbox_inches="tight")
        plt.close(fig)

    sdf = pd.DataFrame(summary); sdf.to_csv(OUT / "oh_unit_summary.csv", index=False)
    cdf = pd.DataFrame(colrows); cdf.to_csv(OUT / "oh_column_stratification.csv", index=False)
    print(sdf.round(4).to_string())
    print(cdf.groupby("cube").det_frac.describe().round(4).to_string())

    log = {"script": Path(__file__).name, "script_md5": md5(Path(__file__)),
           "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
           "fit_win": FIT_WIN, "signal_win": SIGNAL_WIN, "null_wins": NULL_WINS,
           "bg_order": BG_ORDER, "guard_nm": GUARD_NM, "centre_step_um": CENTRE_STEP,
           "fwhm_nm": FWHM_NM, "null_percentile": NULL_PCT, "fill_tol_m": FILL_TOL_M,
           "threshold_note": "empirical: 99th pct of max-SNR in two band-free windows "
                             "(more trial positions than the signal window, so conservative)",
           "deterministic": True, "inputs": inputs}
    (OUT / "run_log_oh_filter.json").write_text(json.dumps(log, indent=2, default=str))
    print("done ->", OUT)


if __name__ == "__main__":
    main()
