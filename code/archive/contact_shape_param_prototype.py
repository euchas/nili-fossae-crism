"""Prototype: shape-selective 1.4 um band parameters vs the olivine 1 um wing.

Compares five ways of measuring a possible ~1.39 um OH band on the two ROI
spectra extracted by contact_roi_spectra_1400.py (olivine-unit ROI = smooth
1 um wing, no real band; smectite ROI = genuine OH band):

  M0  linear continuum + band minimum          (current method, known broken)
  M1  quadratic continuum on band-free shoulders, depth at 1.385-1.41
  M2  convex-hull (upper hull) continuum removal, depth + width
  M3  Savitzky-Golay 2nd derivative matched to a ~25 nm feature, SNR
  M4  model comparison: polynomial background vs background x Gaussian,
      delta-BIC, with fitted amplitude / centre / FWHM

Two tests per method:
  (a) FALSE POSITIVE - what does it report on the olivine ROI (truth: nothing)?
  (b) SENSITIVITY - inject a Gaussian OH band (FWHM 25 nm at 1.39 um) of known
      depth onto the olivine ROI and find the smallest depth recovered.

Deterministic; no RNG. Prototype only - no maps are produced.

Usage:  python code/contact_shape_param_prototype.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from contact_distance_to_contact import md5  # noqa: E402

OUT = ROOT / "outputs" / "revision" / "band_center_map" / "band1400" / "plebani"
CSV = OUT / "roi_spectra_FRT00003E12.csv"

FIT_WIN = (1.30, 1.52)          # region the parameters operate on
SHOULDERS = ((1.28, 1.35), (1.45, 1.55))
BAND_WIN = (1.375, 1.420)       # where a real Mg-OH / H2O overtone would sit
BAND_MEAS = (1.383, 1.400)      # depth measurement window
NOISE_WINS = ((1.30, 1.35), (1.50, 1.55))
FWHM_NM = 25.0
SIGMA0 = FWHM_NM / 2.3548 / 1000.0
INJECT = (0.0, 0.002, 0.004, 0.007, 0.010, 0.015, 0.020, 0.030)
DBIC_THRESH = 10.0
SNR_THRESH = 3.0


def _win(x, w):
    return (x >= w[0]) & (x <= w[1])


def noise_level(x, y):
    """Per-channel high-frequency noise of the stacked spectrum."""
    sm = savgol_filter(y, 9, 3)
    res = y - sm
    sel = _win(x, NOISE_WINS[0]) | _win(x, NOISE_WINS[1])
    return float(np.std(res[sel]))


def m0_linear(x, y):
    la, ra = _win(x, (1.316, 1.342)), _win(x, (1.505, 1.535))
    rl, rr = np.mean(y[la]), np.mean(y[ra])
    wl, wr = x[la].mean(), x[ra].mean()
    cont = rl + (rr - rl) * (x - wl) / (wr - wl)
    cr = y / cont
    sw = _win(x, (1.360, 1.490))
    i = int(np.argmin(cr[sw]))
    return {"depth": float(1.0 - cr[sw][i]), "center_um": float(x[sw][i])}, cont


def m1_quadratic(x, y):
    sel = _win(x, SHOULDERS[0]) | _win(x, SHOULDERS[1])
    coef = np.polyfit(x[sel], y[sel], 2)
    cont = np.polyval(coef, x)
    cr = y / cont
    d = float(1.0 - np.median(cr[_win(x, BAND_MEAS)]))
    bw = _win(x, BAND_WIN)
    i = int(np.argmin(cr[bw]))
    return {"depth": d, "center_um": float(x[bw][i]), "min_depth": float(1.0 - cr[bw][i])}, cont


def upper_hull(x, y):
    pts = list(zip(x.tolist(), y.tolist()))
    hull = []
    for p in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            cross = (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1)
            if cross >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    hx = np.array([p[0] for p in hull]); hy = np.array([p[1] for p in hull])
    return np.interp(x, hx, hy)


def m2_hull(x, y):
    cont = upper_hull(x, y)
    cr = y / cont
    bw = _win(x, BAND_WIN)
    i = int(np.argmin(cr[bw]))
    depth = float(1.0 - cr[bw][i])
    half = 1.0 - depth / 2.0
    idx = np.where(bw)[0]
    j = idx[i]
    lo = j
    while lo > 0 and cr[lo] < half:
        lo -= 1
    hi = j
    while hi < len(cr) - 1 and cr[hi] < half:
        hi += 1
    return {"depth": depth, "center_um": float(x[j]),
            "fwhm_nm": float((x[hi] - x[lo]) * 1000.0)}, cont


def m3_deriv(x, y):
    dl = float(np.median(np.diff(x)))
    d2 = savgol_filter(y / np.median(y), 7, 3, deriv=2, delta=dl)
    sel = _win(x, NOISE_WINS[0]) | _win(x, NOISE_WINS[1])
    sd = float(np.std(d2[sel]))
    bw = _win(x, BAND_WIN)
    peak = float(np.max(d2[bw]))
    i = int(np.argmax(d2[bw]))
    return {"d2_peak": peak, "d2_noise": sd, "snr": peak / sd if sd > 0 else np.nan,
            "center_um": float(x[bw][i])}, d2


def m4_model(x, y):
    sel = _win(x, FIT_WIN)
    xs, ys = x[sel], y[sel]
    n = xs.size

    def bg_resid(c):
        return np.polyval(c, xs) - ys

    c0 = np.polyfit(xs, ys, 3)
    rb = least_squares(bg_resid, c0)
    rss_bg = float(np.sum(rb.fun ** 2))

    def full_resid(p):
        a, cen, sig = p[0], p[1], p[2]
        bg = np.polyval(p[3:], xs)
        return bg * (1.0 - a * np.exp(-0.5 * ((xs - cen) / sig) ** 2)) - ys

    p0 = np.concatenate([[0.01, 1.390, SIGMA0], rb.x])
    lo = np.concatenate([[0.0, 1.370, 4.0 / 2.3548 / 1000.0], -np.inf * np.ones(4)])
    hi = np.concatenate([[0.5, 1.430, 40.0 / 2.3548 / 1000.0], np.inf * np.ones(4)])
    rf = least_squares(full_resid, p0, bounds=(lo, hi))
    rss_full = float(np.sum(rf.fun ** 2))
    bic = lambda rss, k: n * np.log(max(rss, 1e-30) / n) + k * np.log(n)
    dbic = bic(rss_bg, 4) - bic(rss_full, 7)
    return {"amp": float(rf.x[0]), "center_um": float(rf.x[1]),
            "fwhm_nm": float(rf.x[2] * 2.3548 * 1000.0), "delta_bic": float(dbic)}, (xs, rf, rb)


def run_all(x, y):
    r0, c0 = m0_linear(x, y)
    r1, c1 = m1_quadratic(x, y)
    r2, c2 = m2_hull(x, y)
    r3, d2 = m3_deriv(x, y)
    r4, _ = m4_model(x, y)
    return {"M0": r0, "M1": r1, "M2": r2, "M3": r3, "M4": r4}, (c0, c1, c2, d2)


def main() -> None:
    t0 = datetime.now(timezone.utc)
    df = pd.read_csv(CSV)
    x = df.wavelength_um.to_numpy()
    keep = _win(x, (1.26, 1.60)) & np.isfinite(df["olivine ROI"]) & np.isfinite(df["smectite ROI"])
    x = x[keep]
    spec = {"olivine ROI": df["olivine ROI"].to_numpy()[keep],
            "smectite ROI": df["smectite ROI"].to_numpy()[keep]}

    rows, conts = [], {}
    for name, y in spec.items():
        res, cc = run_all(x, y)
        conts[name] = (y, cc)
        nz = noise_level(x, y / np.median(y))
        for meth, vals in res.items():
            rows.append({"spectrum": name, "method": meth, "noise_1sigma": nz,
                         **{k: v for k, v in vals.items()}})
        print(name, "noise", round(nz, 5), json.dumps({k: {kk: round(vv, 5) for kk, vv in v.items()}
                                                       for k, v in res.items()}))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "shape_param_prototype_metrics.csv", index=False)

    # injection test on the olivine ROI
    y_ol = spec["olivine ROI"]
    inj_rows = []
    for a in INJECT:
        y = y_ol * (1.0 - a * np.exp(-0.5 * ((x - 1.390) / SIGMA0) ** 2))
        res, _ = run_all(x, y)
        inj_rows.append({"injected_depth": a,
                         "M1_depth": res["M1"]["depth"], "M2_depth": res["M2"]["depth"],
                         "M3_snr": res["M3"]["snr"], "M4_amp": res["M4"]["amp"],
                         "M4_dbic": res["M4"]["delta_bic"], "M4_center": res["M4"]["center_um"],
                         "M4_fwhm_nm": res["M4"]["fwhm_nm"]})
    inj = pd.DataFrame(inj_rows)
    inj.to_csv(OUT / "shape_param_injection.csv", index=False)
    print(inj.round(4).to_string())

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    ax = axes[0, 0]
    for name, col in (("olivine ROI", "#D55E00"), ("smectite ROI", "#0072B2")):
        y, (c0, c1, c2, _) = conts[name]
        yn = y / np.median(y)
        ax.plot(x, yn, color=col, lw=1.8, label=name)
        ax.plot(x, c0 / np.median(y), color=col, lw=0.9, ls=":", label="linear (M0)" if name == "olivine ROI" else None)
        ax.plot(x, c1 / np.median(y), color=col, lw=0.9, ls="--", label="quadratic (M1)" if name == "olivine ROI" else None)
        ax.plot(x, c2 / np.median(y), color=col, lw=0.9, ls="-.", label="hull (M2)" if name == "olivine ROI" else None)
    ax.set_title("(a) spectra and candidate continua", fontsize=10)
    ax.set_xlabel("wavelength (um)"); ax.set_ylabel("normalized ratioed I/F")
    ax.legend(fontsize=7.5, frameon=False); ax.grid(True, color="0.96")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[0, 1]
    for name, col in (("olivine ROI", "#D55E00"), ("smectite ROI", "#0072B2")):
        y, (c0, c1, c2, _) = conts[name]
        ax.plot(x, y / c1, color=col, lw=1.8, label=f"{name} / quadratic")
        ax.plot(x, y / c2, color=col, lw=1.0, ls="-.", label=f"{name} / hull")
    ax.axhline(1.0, color="0.8", lw=0.8)
    ax.axvspan(*BAND_MEAS, color="0.93", zorder=0)
    ax.set_ylim(0.975, 1.006)
    ax.set_title("(b) continuum-removed (curved continua)", fontsize=10)
    ax.set_xlabel("wavelength (um)"); ax.set_ylabel("continuum-removed")
    ax.legend(fontsize=7.5, frameon=False); ax.grid(True, color="0.96")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1, 0]
    for name, col in (("olivine ROI", "#D55E00"), ("smectite ROI", "#0072B2")):
        y, (_, _, _, d2) = conts[name]
        ax.plot(x, d2, color=col, lw=1.6, label=name)
    ax.axvspan(*BAND_WIN, color="0.93", zorder=0)
    ax.axhline(0, color="0.8", lw=0.8)
    ax.set_title("(c) 2nd derivative, matched scale (M3)", fontsize=10)
    ax.set_xlabel("wavelength (um)"); ax.set_ylabel("d2 I/F / dl2")
    ax.legend(fontsize=8, frameon=False); ax.grid(True, color="0.96")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1, 1]
    ax.plot(inj.injected_depth * 100, inj.M1_depth * 100, "-o", color="#009E73", ms=4, label="M1 quadratic depth")
    ax.plot(inj.injected_depth * 100, inj.M2_depth * 100, "-s", color="#CC79A7", ms=4, label="M2 hull depth")
    ax.plot(inj.injected_depth * 100, inj.M4_amp * 100, "-^", color="#0072B2", ms=4, label="M4 fitted amplitude")
    ax.plot([0, max(INJECT) * 100], [0, max(INJECT) * 100], color="0.8", lw=0.8, ls="--", label="1:1")
    ax.set_xlabel("injected band depth (%)"); ax.set_ylabel("recovered depth / amplitude (%)")
    ax2 = ax.twinx()
    ax2.plot(inj.injected_depth * 100, inj.M4_dbic, ":", color="0.35", lw=1.4, label="M4 delta-BIC (right)")
    ax2.axhline(DBIC_THRESH, color="0.6", lw=0.8, ls=":")
    ax2.set_ylabel("delta-BIC"); ax2.set_yscale("symlog")
    ax.set_title("(d) injection recovery on the olivine ROI", fontsize=10)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, frameon=False, loc="upper left")
    ax.grid(True, color="0.96"); ax.spines[["top"]].set_visible(False)

    fig.suptitle("Shape-selective 1.4 um parameters: false positives on the olivine wing "
                 "and sensitivity to a real 25 nm OH band", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_shape_param_prototype.png", dpi=200)
    fig.savefig(OUT / "fig_shape_param_prototype.pdf")
    plt.close(fig)

    log = {"script": Path(__file__).name, "script_md5": md5(Path(__file__)),
           "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
           "fit_win": FIT_WIN, "shoulders": SHOULDERS, "band_win": BAND_WIN,
           "band_meas": BAND_MEAS, "fwhm_nm": FWHM_NM, "inject": list(INJECT),
           "dbic_thresh": DBIC_THRESH, "snr_thresh": SNR_THRESH,
           "input_csv_md5": md5(CSV), "deterministic": True}
    (OUT / "run_log_shape_param_prototype.json").write_text(json.dumps(log, indent=2, default=str))
    print("done ->", OUT)


if __name__ == "__main__":
    main()
