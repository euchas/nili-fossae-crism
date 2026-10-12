"""Per-pixel Gaussian matched-filter band detection, Plebani chain, any band.

Generalization of contact_oh_matched_filter.py (whose 1.4 um run stands as
the reference). Same estimator: cubic background excluding the test window,
closed-form Gaussian amplitude on a (centre, FWHM) grid, amplitude SNR
against the pixel's own band-free residual noise, and an EMPIRICAL detection
threshold = 99th percentile of the identical estimator run in band-free
null windows (more trial positions than the signal window, so
look-elsewhere is over-charged). Per-band configuration below.

Usage:  python code/contact_band_matched_filter.py 2300
        python code/contact_band_matched_filter.py 1900
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
from contact_distance_to_contact import V8_PARAMS, load_cube, masks_from_labels, md5  # noqa: E402
from contact_band_center_map import unit_masks  # noqa: E402
from contact_carbonate_patches import CUBES, GEOMAP, FRAME, LONLAT, load_ddr_lonlat  # noqa: E402
from contact_band_center_plebani import ZRAT  # noqa: E402
from figstyle import draw_units, label_panels  # noqa: E402

BANDS = {
    "2300": {
        "label": "2.3 um metal-OH", "fit_win": (2.14, 2.48), "signal_win": (2.280, 2.350),
        "null_wins": ((2.175, 2.225), (2.400, 2.450)), "guard_nm": 15.0,
        "fwhm_nm": (25.0, 35.0, 45.0, 60.0), "ref_param": "D2300",
        "note": "null windows may hold weak real features (Al-OH 2.21, talc 2.39 tail) -> "
                "threshold is conservative; olivine-unit shared 2.32 band partly cancels "
                "against olivine-like bland denominators",
    },
    "2500": {
        "label": "2.5 um carbonate", "fit_win": (2.36, 2.64), "signal_win": (2.480, 2.540),
        "null_wins": ((2.395, 2.440), (2.575, 2.620)), "guard_nm": 15.0,
        "fwhm_nm": (30.0, 40.0, 55.0, 70.0), "ref_param": "BD2500_2",
        "note": "Mg-carbonate 2.50-2.51 um partner of the 2.30 um band (Ca/Fe carbonate 2.53); "
                "short null may hold the 2.39/2.46 um talc bands -> conservative threshold",
    },
    "1900": {
        "label": "1.9 um H2O", "fit_win": (1.74, 2.10), "signal_win": (1.895, 1.945),
        "null_wins": ((1.755, 1.795), (1.805, 1.845)), "guard_nm": 15.0,
        "fwhm_nm": (40.0, 55.0, 70.0, 90.0), "ref_param": "BD1900R2",
        "note": "broad band; both nulls on the short side to avoid the 2.0 um CO2 residual "
                "region; fit window truncated at 2.10 um for the same reason",
    },
}
CENTRE_STEP = 0.002
NULL_PCT = 99.0
FILL_TOL_M = 30.0
BG_ORDER = 3


def design(wav_fit, exclude):
    use = ~exclude
    A = np.vander(wav_fit[use], BG_ORDER + 1)
    return np.linalg.pinv(A), np.vander(wav_fit, BG_ORDER + 1), use


def grid_matrix(wav_fit, win, fwhm_nm):
    centres = np.arange(win[0], win[1] + 1e-9, CENTRE_STEP)
    cols, meta = [], []
    for c in centres:
        for f in fwhm_nm:
            s = f / 2.3548 / 1000.0
            cols.append(np.exp(-0.5 * ((wav_fit - c) / s) ** 2))
            meta.append((c, f))
    return np.stack(cols, axis=1), np.array(meta)


def run_window(Y, wav_fit, win, noise_mask, guard_um, fwhm_nm):
    ex = (wav_fit >= win[0] - guard_um) & (wav_fit <= win[1] + guard_um)
    pinv, Afull, use = design(wav_fit, ex)
    bg = (Y[:, use] @ pinv.T) @ Afull.T
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 1.0 - Y / bg
    r[~np.isfinite(r)] = 0.0
    sigma = np.std(r[:, noise_mask], axis=1)
    G, meta = grid_matrix(wav_fit, win, fwhm_nm)
    gg = np.sum(G * G, axis=0)
    amp = (r @ G) / gg
    snr = amp * np.sqrt(gg)[None, :] / np.maximum(sigma, 1e-9)[:, None]
    best = np.argmax(snr, axis=1)
    idx = np.arange(Y.shape[0])
    return {"amp": amp[idx, best], "snr": snr[idx, best],
            "centre": meta[best, 0], "fwhm": meta[best, 1], "sigma": sigma}


def fit_cube(if2, wav, cfg):
    fw = (wav >= cfg["fit_win"][0]) & (wav <= cfg["fit_win"][1])
    wav_fit = wav[fw]
    Y = if2[:, fw].astype(np.float64)
    ok = np.isfinite(Y).all(axis=1) & (np.abs(Y).sum(axis=1) > 1e-6)
    g = cfg["guard_nm"] / 1000.0
    noise_mask = np.ones(wav_fit.size, bool)
    for w in (cfg["signal_win"],) + tuple(cfg["null_wins"]):
        noise_mask &= ~((wav_fit >= w[0] - g) & (wav_fit <= w[1] + g))
    sig = run_window(Y[ok], wav_fit, cfg["signal_win"], noise_mask, g, cfg["fwhm_nm"])
    nulls = [run_window(Y[ok], wav_fit, w, noise_mask, g, cfg["fwhm_nm"]) for w in cfg["null_wins"]]
    sig["null_snr"] = np.maximum.reduce([n["snr"] for n in nulls])
    sig["ok"] = ok
    return sig


def project(cube, c):
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


def main(band: str) -> None:
    cfg = BANDS[band]
    out = ROOT / "outputs" / "revision" / "band_center_map" / "matched_filter" / band
    out.mkdir(parents=True, exist_ok=True)
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
        res = fit_cube(if2, wav, cfg)
        del if2, z
        ok = res["ok"]

        c = load_cube(cube)
        m = masks_from_labels(c["labels"])
        units, d, fr = unit_masks(cube, m)
        xs, ys, gr, gc, tr = project(cube, c)
        H, W = c["labels"].shape
        inside = np.isfinite(xs) & (gr >= 0) & (gr < H) & (gc >= 0) & (gc < W)
        gri = np.where(inside, gr, 0).astype(int); gci = np.where(inside, gc, 0).astype(int)
        samp = lambda a: np.where(inside, a[gri, gci], np.nan)
        f_clay, f_cap = samp(fr["clay"]), samp(fr["cap"])
        olind = samp(c["params"][..., V8_PARAMS.index("OLINDEX3")])
        refp = samp(c["params"][..., V8_PARAMS.index(cfg["ref_param"])])
        mvalid = np.where(inside, m["valid"][gri, gci], False)

        def expand(v):
            a = np.full(ok.size, np.nan)
            a[ok] = v
            return a
        amp, snr, cen, fw, nsnr = (expand(res[k]) for k in ("amp", "snr", "centre", "fwhm", "null_snr"))
        scol = (np.arange(ok.size) % W_s).astype(np.int16)
        usable = ok & inside & mvalid
        thr = float(np.nanpercentile(nsnr[usable], NULL_PCT))
        sw = cfg["signal_win"]
        det = usable & (snr >= thr) & (cen > sw[0] + 1e-9) & (cen < sw[1] - 1e-9)
        np.savez_compressed(out / f"{cube}_fit_sensor.npz", amp=amp.astype(np.float32),
                            snr=snr.astype(np.float32), centre=cen.astype(np.float32),
                            fwhm=fw.astype(np.float32), null_snr=nsnr.astype(np.float32),
                            det=det, scol=scol)

        unit_defs = {"olivine map-pure": usable & np.where(inside, units["olivine map-pure"][gri, gci], False),
                     "clay map-pure": usable & (f_clay >= 0.9),
                     "cap map-pure": usable & (f_cap >= 0.9)}
        for uname, um in unit_defs.items():
            n = int(um.sum()); dd = um & det
            good = um & np.isfinite(olind) & np.isfinite(amp)
            goodr = um & np.isfinite(refp) & np.isfinite(amp)
            summary.append({"cube": cube, "unit": uname, "n": n, "n_det": int(dd.sum()),
                            "det_frac": float(dd.sum() / n) if n else np.nan, "thr_snr": thr,
                            "amp_med_det": float(np.median(amp[dd])) if dd.any() else np.nan,
                            "centre_med_det": float(np.median(cen[dd])) if dd.any() else np.nan,
                            "fwhm_med_det": float(np.median(fw[dd])) if dd.any() else np.nan,
                            "rho_amp_OLINDEX3": float(spearmanr(amp[good], olind[good]).statistic) if good.sum() > 200 else np.nan,
                            f"rho_amp_{cfg['ref_param']}": float(spearmanr(amp[goodr], refp[goodr]).statistic) if goodr.sum() > 200 else np.nan})
        for col in range(0, W_s, 32):
            sel = usable & (scol >= col) & (scol < col + 32)
            if sel.sum() > 200:
                colrows.append({"cube": cube, "col_bin": col,
                                "thr_snr": float(np.nanpercentile(nsnr[sel], NULL_PCT)),
                                "det_frac": float((sel & det).sum() / sel.sum())})

        tree = cKDTree(np.column_stack([xs[usable], ys[usable]]))
        cc, rr = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        gx, gy = tr * (cc.ravel(), rr.ravel())
        dist, idx = tree.query(np.column_stack([gx, gy]), distance_upper_bound=FILL_TOL_M)
        hit = np.isfinite(dist)
        cen_u, det_u, snr_u = cen[usable], det[usable], snr[usable]
        cen_map = np.full(H * W, np.nan); det_map = np.zeros(H * W, bool); snr_map = np.full(H * W, np.nan)
        cen_map[hit] = cen_u[idx[hit]]; det_map[hit] = det_u[idx[hit]]; snr_map[hit] = snr_u[idx[hit]]
        cen_map = cen_map.reshape(H, W); det_map = det_map.reshape(H, W) & m["valid"]
        snr_map = snr_map.reshape(H, W)
        np.save(out / f"{cube}_centre_det.npy", np.where(det_map, cen_map, np.nan))
        np.save(out / f"{cube}_snr.npy", np.where(m["valid"], snr_map, np.nan))
        print(cube, f"thr_SNR={thr:.2f} det_px={int(det.sum())} ({100 * det.sum() / max(usable.sum(), 1):.2f}% of usable)")

        fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.5), sharex=True, sharey=True)
        bg = np.where(m["valid"], 0.93, np.nan)
        axes[0].imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        im = axes[0].imshow(np.where(det_map, cen_map, np.nan), cmap="plasma",
                            vmin=sw[0], vmax=sw[1], interpolation="nearest")
        axes[0].set_title(f"{cfg['label']} detections (SNR >= {thr:.1f}), coloured by fitted centre", fontsize=10)
        cb = fig.colorbar(im, ax=axes[0], shrink=0.8, pad=0.02)
        cb.set_label("fitted band centre (um)", fontsize=8.5)
        axes[1].imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        im2 = axes[1].imshow(snr_map, cmap="viridis", vmin=0, vmax=max(6.0, thr * 1.5), interpolation="nearest")
        axes[1].set_title("amplitude SNR (all usable pixels)", fontsize=10)
        fig.colorbar(im2, ax=axes[1], shrink=0.8, pad=0.02)
        for ax in axes:
            draw_units(ax, fr, units=("clay", "cap"))
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{cube}: {cfg['label']} matched-filter detection, Plebani bland-ratioed chain\n"
                     f"threshold = {NULL_PCT:.0f}th percentile of the band-free null distribution", fontsize=11)
        fig.tight_layout()
        label_panels(fig)
        fig.savefig(out / f"fig_{band}_{cube}.png", dpi=200, bbox_inches="tight")
        fig.savefig(out / f"fig_{band}_{cube}.pdf", bbox_inches="tight")
        plt.close(fig)

    sdf = pd.DataFrame(summary); sdf.to_csv(out / "unit_summary.csv", index=False)
    pd.DataFrame(colrows).to_csv(out / "column_stratification.csv", index=False)
    print(sdf.round(4).to_string())
    log = {"script": Path(__file__).name, "script_md5": md5(Path(__file__)), "band": band,
           "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
           "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
           "centre_step_um": CENTRE_STEP, "null_percentile": NULL_PCT, "bg_order": BG_ORDER,
           "fill_tol_m": FILL_TOL_M, "deterministic": True, "inputs": inputs}
    (out / "run_log.json").write_text(json.dumps(log, indent=2, default=str))
    print("done ->", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2300")
