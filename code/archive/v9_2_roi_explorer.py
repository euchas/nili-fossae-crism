"""v9.2 ROI Explorer - BD2330 / LWS / VB-RGB viewer with 2.3 um band fit.

Three viewer panels (replacing the four-panel v9.1 layout):
  Panel 1   BD2330 parameter map (single-band, inferno colormap).
            BD2330 = 1 - R2330 / (a*R2250 + b*R2400), a=0.467, b=0.533,
            5-channel kernel per anchor (Viviano-Beck 2014 convention).
  Panel 2   LWS index map (single-band, viridis).
            LWS = BD2330 / (D2300 + 1e-6); no D2300 floor / clipping
            (full pixel range rendered).
  Panel 3   VB RGB composite: R = BD2330, G = OLINDEX3, B = BD1900_2.

Plots window adds a 2.3 um band fit per (target, mafic) pair:
  - Brown 2006 asymmetric Gaussian fit to the continuum-removed
    ratio spectrum in the 2.20-2.45 um window. Free parameters
    alpha, lambda_0, sigma, chi.
  - Linear continuum between the 2.20 and 2.45 um shoulders.
  - Reports lambda_0, chi, AND lambda_min (numerical argmin of the
    fitted continuum-removed reflectance) for direct comparison
    with Ehlmann et al. band positions.
  - Fit overlay drawn on the ratio panel; residuals on a third
    sub-panel below.
  - Per-pair fit results saved to <cube>_band_2300_fits.csv in the
    session directory.

Outputs land in the same directory tree as v9.1 (so accumulated
ROIs across both tools live alongside each other):
    outputs/spectra sample locations/<cube>/<name>/...

Run:
    python code/v9_2_roi_explorer.py FRT00003E12
    python code/v9_2_roi_explorer.py 871C            # short form
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
for _bk in ("QtAgg", "TkAgg", "Qt5Agg"):
    try:
        matplotlib.use(_bk)
        break
    except (ImportError, ValueError):
        continue

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
import scipy.optimize as opt

plt.rcParams.update({
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 12,
})

from config import CUBE_IDS, OUTPUTS_DIR
from crism_io import load_cube_by_id
from vb_parameters import (
    calc_olindex3, calc_d2300, calc_bd2330, calc_bd1900_2,
)

# =============================================================================
# Constants
# =============================================================================
PAIR_PALETTE = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e",
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                "#bcbd22", "#17becf"]

# Single-band heatmap colormaps + diagnostic stretches.
BD2330_CMAP = "inferno"
LWS_CMAP = "viridis"
LWS_EPS = 1e-6            # denominator divide-by-zero guard (no D2300 floor)

# Fixed Panel-3 RGB composite per spec.
PANEL3_CHANNELS = ("BD2330", "OLINDEX3", "BD1900_2")

# 2.3 um band fit configuration.
FIT_SHOULDER_LO = 2.20
FIT_SHOULDER_HI = 2.45

# Brown RGB stretches (carry over from v9.1; for the scatter plot context only).
BROWN_STRETCH_FULLSCENE = {
    "centroid":  (1.014, 1.379),
    "amplitude": (0.116, 0.335),
    "asymmetry": (0.525, 4.439),
}
SCATTER_XLIM = (0.90, 1.50)
SCATTER_YLIM = (0.30, 10.0)
ANNOTATION_BANDS = (1.00, 1.40, 1.90, 2.30, 2.50)
CO2_RESIDUAL = (1.95, 2.05)

SAMPLES_ROOT = OUTPUTS_DIR / "spectra sample locations"


# =============================================================================
# Helpers
# =============================================================================
ROIS_CSV_HEADER = [
    "session_ts", "roi_label", "kind", "pair_idx", "color", "mode",
    "n_pixels", "n_brown_finite",
    "centroid_mean", "centroid_std",
    "amplitude_mean", "amplitude_std",
    "hwhm_mean", "hwhm_std",
    "asymmetry_mean", "asymmetry_std",
    "bbox_x_lo", "bbox_y_lo", "bbox_x_hi", "bbox_y_hi",
    "mask_file",
]

FIT_CSV_HEADER = [
    "session_ts", "pair_idx", "target_label", "mafic_label", "color",
    "n_target_px", "n_mafic_px",
    "alpha", "lambda_0", "sigma", "chi",
    "lambda_min", "rms_residual", "n_fit_points",
    "shoulder_lo_um", "shoulder_hi_um",
]


def sanitize_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return ""
    return re.sub(r"[^A-Za-z0-9_-]", "_", name.replace(" ", "_")).strip("_")


def resolve_cube_id(arg: str | None) -> str:
    if not arg:
        return "FRT00003E12"
    a = arg.upper().strip()
    if a in CUBE_IDS:
        return a
    for cid in CUBE_IDS:
        if cid.endswith(a):
            return cid
    raise SystemExit(f"unknown cube {arg!r}; known: {CUBE_IDS}")


def percentile_norm(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any() or hi - lo < 1e-9:
        out[finite] = 0.5
        return out
    out[finite] = np.clip((arr[finite] - lo) / (hi - lo), 0, 1)
    return out


def percentile_clim(arr: np.ndarray) -> tuple[float, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size < 10:
        return 0.0, 1.0
    return tuple(float(v) for v in np.percentile(finite, [2, 98]))


def annotate_spectrum(ax) -> None:
    ax.axvspan(*CO2_RESIDUAL, color="#cccccc", alpha=0.3, zorder=0)
    for wl in ANNOTATION_BANDS:
        ax.axvline(wl, color="#aa4444", lw=0.8, ls="--", alpha=0.6, zorder=0)


# =============================================================================
# 2.3 um asymmetric Gaussian fit
# =============================================================================
def asym_gaussian(lam: np.ndarray, alpha: float, lam0: float,
                   sigma: float, chi: float) -> np.ndarray:
    """Brown (2006/2020) asymmetric Gaussian in absorbance / wavelength space:
        A(lam) = alpha * exp(-((lam - lam0) /
                                (sigma * (1 + sgn(lam-lam0)*chi)))**2)
    """
    denom = sigma * (1.0 + np.sign(lam - lam0) * chi)
    denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    return alpha * np.exp(-((lam - lam0) / denom) ** 2)


def fit_2300_band(wav: np.ndarray, ratio: np.ndarray,
                    shoulder_lo: float = FIT_SHOULDER_LO,
                    shoulder_hi: float = FIT_SHOULDER_HI) -> dict | None:
    """Fit Brown asymmetric Gaussian to the continuum-removed 2.3 um
    band of `ratio` (target_over_mafic).

    Returns a dict with fit params and diagnostics, or None if the fit
    failed / not enough points.
    """
    mask = ((wav >= shoulder_lo - 0.005) & (wav <= shoulder_hi + 0.005)
            & np.isfinite(ratio))
    if mask.sum() < 8:
        return None
    w = wav[mask]
    r = ratio[mask]
    if not (np.isfinite(r).any()):
        return None

    # Linear continuum: take averages of the closest points to each shoulder.
    near_lo = (np.abs(w - shoulder_lo) < 0.012)
    near_hi = (np.abs(w - shoulder_hi) < 0.012)
    if near_lo.sum() < 1 or near_hi.sum() < 1:
        return None
    r_lo = float(np.nanmean(r[near_lo]))
    r_hi = float(np.nanmean(r[near_hi]))
    if not (np.isfinite(r_lo) and np.isfinite(r_hi)):
        return None
    cont = r_lo + (r_hi - r_lo) * (w - shoulder_lo) / (shoulder_hi - shoulder_lo)

    # Continuum-removed reflectance and apparent absorbance.
    r_cr = np.where(cont != 0, r / cont, np.nan)
    r_cr_safe = np.clip(r_cr, 1e-6, None)
    a_obs = -np.log(r_cr_safe)   # natural log absorbance (sign convention: positive in absorption)

    # Restrict to fit window (already mostly there, tight bounds for fit only).
    fit_mask = ((w >= shoulder_lo) & (w <= shoulder_hi)
                 & np.isfinite(a_obs))
    if fit_mask.sum() < 6:
        return None
    w_fit = w[fit_mask]
    a_fit_obs = a_obs[fit_mask]

    # Initial guesses
    alpha0 = max(float(a_fit_obs.max()), 1e-3)
    lam0_0 = float(w_fit[np.argmax(a_fit_obs)])
    sigma0 = 0.05
    chi0 = 0.0
    bounds = ([1e-4, shoulder_lo, 0.005, -0.95],
              [2.0,  shoulder_hi, 0.30,   0.95])

    try:
        popt, _ = opt.curve_fit(
            asym_gaussian, w_fit, a_fit_obs,
            p0=[alpha0, lam0_0, sigma0, chi0],
            bounds=bounds, maxfev=5000,
        )
    except Exception as e:
        return {"error": str(e)}

    alpha, lam0, sigma, chi = (float(v) for v in popt)
    a_pred = asym_gaussian(w_fit, alpha, lam0, sigma, chi)
    residual = a_fit_obs - a_pred
    rms = float(np.sqrt(np.mean(residual ** 2)))

    # lambda_min: numerical argmin of the fitted continuum-removed reflectance
    # (= argmax of the fitted absorbance) on a fine grid inside the window.
    lam_grid = np.linspace(shoulder_lo, shoulder_hi, 4001)
    a_grid = asym_gaussian(lam_grid, alpha, lam0, sigma, chi)
    lam_min = float(lam_grid[np.argmax(a_grid)])

    # Fit curves at the original window for plotting overlay.
    cont_full = r_lo + (r_hi - r_lo) * (w - shoulder_lo) / (shoulder_hi - shoulder_lo)
    a_full_pred = asym_gaussian(w, alpha, lam0, sigma, chi)
    r_cr_pred = np.exp(-a_full_pred)
    ratio_fit_overlay = r_cr_pred * cont_full   # back to ratio space

    return {
        "alpha": alpha, "lambda_0": lam0, "sigma": sigma, "chi": chi,
        "lambda_min": lam_min, "rms_residual": rms,
        "n_fit_points": int(fit_mask.sum()),
        "shoulder_lo_um": shoulder_lo, "shoulder_hi_um": shoulder_hi,
        "r_shoulder_lo": r_lo, "r_shoulder_hi": r_hi,
        # window arrays (for plotting)
        "fit_wav": w, "fit_ratio_obs": r,
        "fit_ratio_overlay": ratio_fit_overlay,
        "fit_residual": (r - ratio_fit_overlay),
    }


# =============================================================================
# Explorer
# =============================================================================
class RoiExplorer:
    def __init__(self, cube_id: str):
        self.cube_id = cube_id
        self.session_ts = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ")
        self.rois: list[dict] = []
        self.pending_pixels: list[tuple[int, int]] = []
        self.mode: str = "rect"
        self._pending_artists: list = []
        self._restretch_pending = False
        self.session_name: str | None = None
        self.name_dir: Path | None = None
        self.session_dir: Path | None = None
        self.rois_csv: Path | None = None
        self.spectra_csv: Path | None = None
        self.fits_csv: Path | None = None
        self._load_data()
        self._build_figure()

    def _next_kind_and_label(self) -> tuple[str, str, int, str]:
        n = len(self.rois)
        is_target = (n % 2 == 0)
        kind = "target" if is_target else "mafic"
        pair_idx = n // 2 + 1
        color = PAIR_PALETTE[(pair_idx - 1) % len(PAIR_PALETTE)]
        label = f"{self.cube_id}_{kind}{pair_idx}"
        return kind, label, pair_idx, color

    def _targets(self) -> list[dict]:
        return [r for r in self.rois if r["kind"] == "target"]

    def _mafics(self) -> list[dict]:
        return [r for r in self.rois if r["kind"] == "mafic"]

    def _pairs(self) -> list[tuple[dict, dict]]:
        targets = {r["pair_idx"]: r for r in self._targets()}
        mafics = {r["pair_idx"]: r for r in self._mafics()}
        return [(targets[i], mafics[i])
                for i in sorted(set(targets) & set(mafics))]

    # ---- Data loading ----
    def _load_data(self) -> None:
        print(f"[load] {self.cube_id}: TER3 cube...")
        t0 = time.time()
        cube, wav, brown, _ = load_cube_by_id(self.cube_id)
        self.wav = wav.astype(np.float32)
        self.cube = cube[:, ::-1, ::-1].copy()
        self.brown = {k: v[::-1, ::-1].copy() for k, v in brown.items()}
        self.shape = self.cube.shape[1:]
        print(f"  cube {self.cube.shape} in {time.time()-t0:.1f}s")

        self.brown_valid = (np.isfinite(self.brown["centroid"])
                            & np.isfinite(self.brown["amplitude"])
                            & np.isfinite(self.brown["hwhm"])
                            & np.isfinite(self.brown["asymmetry"]))

        # Compute the four VB params we need fresh from the cube.
        print("  computing BD2330 ...")
        self._bd2330 = calc_bd2330(self.cube, self.wav).astype(np.float32)
        print("  computing D2300 ...")
        self._d2300 = calc_d2300(self.cube, self.wav).astype(np.float32)
        print("  computing OLINDEX3 ...")
        self._olindex3 = calc_olindex3(self.cube, self.wav).astype(np.float32)
        print("  computing BD1900_2 ...")
        self._bd1900_2 = calc_bd1900_2(self.cube, self.wav).astype(np.float32)

        # LWS = BD2330 / (D2300 + eps). No D2300 floor / clipping per AV
        # 2026-05-01 - render the full range; downstream sees what falls
        # out where D2300 ~ 0 (large or NaN values).
        with np.errstate(divide="ignore", invalid="ignore"):
            lws = self._bd2330 / (self._d2300 + LWS_EPS)
        self._lws = lws.astype(np.float32)

        # Default percentile clim for the heatmap panels.
        self._bd2330_clim = percentile_clim(self._bd2330)
        self._lws_clim = percentile_clim(self._lws)
        print(f"  BD2330 p2/p98 = {self._bd2330_clim[0]:.4f}, "
              f"{self._bd2330_clim[1]:.4f}")
        print(f"  LWS    p2/p98 = {self._lws_clim[0]:.4f}, "
              f"{self._lws_clim[1]:.4f}")

        # Panel 3 RGB raw channels.
        self._panel3_raw = {
            "R": self._bd2330,
            "G": self._olindex3,
            "B": self._bd1900_2,
        }
        self.panel3_rgb, _ = self._compute_rgb(self._panel3_raw, None)

    def _setup_output_dir_lazy(self) -> None:
        # Same lazy materialization as v9.1, plus fits_csv path.
        if self.session_name is None:
            try:
                raw = input("Name for this session "
                             "(e.g. clays / fe-mg-oh / ridges): ").strip()
            except EOFError:
                raw = ""
            slug = sanitize_name(raw)
            if not slug:
                slug = "session"
                print(f"[name] empty input; using fallback '{slug}'.")
            self.session_name = slug
            print(f"[name] using '{self.session_name}'")
        cube_root = SAMPLES_ROOT / self.cube_id
        self.name_dir = cube_root / self.session_name
        self.session_dir = self.name_dir / "sessions" / self.session_ts
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "masks").mkdir(parents=True, exist_ok=True)
        self.rois_csv = (self.name_dir
                          / f"{self.cube_id}_{self.session_name}_rois.csv")
        if not self.rois_csv.exists():
            with open(self.rois_csv, "w", newline="") as f:
                csv.writer(f).writerow(ROIS_CSV_HEADER)
        self.spectra_csv = (self.session_dir
                              / f"{self.cube_id}_spectra.csv")
        self.fits_csv = (self.session_dir
                          / f"{self.cube_id}_band_2300_fits.csv")
        print(f"  name root -> {self.name_dir}")
        print(f"  session   -> {self.session_dir}")

    # ---- RGB / single-band rendering ----
    def _visible_slice(self) -> tuple[slice, slice] | None:
        try:
            xlim = self.ax_bd2330.get_xlim()
            ylim = self.ax_bd2330.get_ylim()
        except Exception:
            return None
        h, w = self.shape
        x_lo = max(0, int(np.floor(min(xlim))))
        x_hi = min(w, int(np.ceil(max(xlim))) + 1)
        y_lo = max(0, int(np.floor(min(ylim))))
        y_hi = min(h, int(np.ceil(max(ylim))) + 1)
        if x_hi <= x_lo or y_hi <= y_lo:
            return None
        return slice(y_lo, y_hi), slice(x_lo, x_hi)

    def _compute_rgb(self, raw: dict, view) -> tuple[np.ndarray, dict]:
        h, w = self.shape
        out = np.zeros((h, w, 3), dtype=np.float32)
        bounds = {}
        for k, name in enumerate(("R", "G", "B")):
            full = raw[name]
            sample = full[view] if view is not None else full
            finite = sample[np.isfinite(sample)]
            if finite.size < 10:
                lo, hi = 0.0, 1.0
            else:
                lo, hi = np.percentile(finite, [2, 98])
            bounds[name] = (float(lo), float(hi))
            denom = max(hi - lo, 1e-9)
            chan = np.clip((full - lo) / denom, 0, 1).astype(np.float32)
            chan[np.isnan(full)] = 0.0
            out[..., k] = chan
        return out, bounds

    def _band_clim(self, arr: np.ndarray, view,
                    fallback: tuple[float, float]) -> tuple[float, float]:
        sample = arr[view] if view is not None else arr
        finite = sample[np.isfinite(sample)]
        if finite.size < 10:
            return fallback
        return tuple(float(v) for v in np.percentile(finite, [2, 98]))

    # ---- Figure layout ----
    def _build_figure(self) -> None:
        plt.rcdefaults()
        plt.rcParams.update({
            "axes.titlesize": 13, "axes.labelsize": 12,
            "legend.fontsize": 10, "xtick.labelsize": 10,
            "ytick.labelsize": 10, "figure.titlesize": 12,
        })
        # ---- Window 1: Viewer (BD2330 + LWS + RGB) ----
        self.fig_viewer = plt.figure(
            figsize=(18, 7),
            num=f"v9.2 Viewer - {self.cube_id}",
        )
        gs_v = self.fig_viewer.add_gridspec(
            1, 3, wspace=0.07,
            left=0.02, right=0.995, top=0.91, bottom=0.16,
        )
        self.ax_bd2330 = self.fig_viewer.add_subplot(gs_v[0, 0])
        self.ax_lws = self.fig_viewer.add_subplot(
            gs_v[0, 1], sharex=self.ax_bd2330, sharey=self.ax_bd2330)
        self.ax_panel3 = self.fig_viewer.add_subplot(
            gs_v[0, 2], sharex=self.ax_bd2330, sharey=self.ax_bd2330)

        # ---- Window 2: Plots (ratio + 2.3 fit residuals + Brown scatter) ----
        self.fig_plots = plt.figure(
            figsize=(12, 10),
            num=f"v9.2 Plots - {self.cube_id}",
        )
        gs_p = self.fig_plots.add_gridspec(
            3, 1, height_ratios=[3, 1.5, 3], hspace=0.45,
            left=0.09, right=0.97, top=0.95, bottom=0.06,
        )
        self.ax_ratio = self.fig_plots.add_subplot(gs_p[0, 0])
        self.ax_resid = self.fig_plots.add_subplot(gs_p[1, 0])
        self.ax_scatter = self.fig_plots.add_subplot(gs_p[2, 0])

        self._draw_static_panels()
        self._draw_status_strip()
        self._update_dynamic()

        # Rectangle selectors on ALL three viewer panels.
        self._rects = []
        for ax in (self.ax_bd2330, self.ax_lws, self.ax_panel3):
            rect = RectangleSelector(
                ax, onselect=self._on_rect,
                useblit=True, button=[1],
                minspanx=2, minspany=2, spancoords="data",
                interactive=False,
                state_modifier_keys=dict(
                    move=" ", clear="escape", square="shift",
                    center="ctrl", rotate="never",
                ),
            )
            if not hasattr(rect, "_corner_order"):
                rect._corner_order = []
            self._rects.append(rect)
        self.fig_viewer.canvas.mpl_connect(
            "button_press_event", self._on_click)
        self.ax_bd2330.callbacks.connect("ylim_changed",
                                          self._schedule_restretch)
        self.ax_bd2330.callbacks.connect("xlim_changed",
                                          self._schedule_restretch)
        self.fig_viewer.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig_plots.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig_viewer.canvas.mpl_connect("close_event", self._on_close)
        self.fig_plots.canvas.mpl_connect("close_event", self._on_close)
        self._closed = False
        self._refresh_titles()

    def _refresh_titles(self) -> None:
        kind, label, _, _ = self._next_kind_and_label()
        self.fig_viewer.suptitle(
            f"v9.2 Viewer  -  {self.cube_id}  |  "
            f"mode = {self.mode.upper()}  |  next ROI = {label} ({kind})  |  "
            f"keys: m=mode  enter=finalize  u=undo  c=clear  s=save  r=restretch",
            fontsize=11,
        )
        self.fig_plots.suptitle(
            f"v9.2 Plots  -  {self.cube_id}  "
            "(ratio + 2.3 um asym Gauss fit residuals + Brown scatter)",
            fontsize=11,
        )

    def _draw_static_panels(self) -> None:
        # Panel 1: BD2330
        lo, hi = self._bd2330_clim
        self._bd2330_imshow = self.ax_bd2330.imshow(
            self._bd2330, cmap=BD2330_CMAP,
            vmin=lo, vmax=hi, interpolation="nearest")
        self._bd2330_base_title = "BD2330 (long-shifted Mg/Fe-OH band depth)"
        self.ax_bd2330.set_title(
            f"{self._bd2330_base_title}  [dynamic stretch]",
            fontsize=11, fontweight="bold")
        self.ax_bd2330.set_xticks([]); self.ax_bd2330.set_yticks([])

        # Panel 2: LWS
        lo, hi = self._lws_clim
        self._lws_imshow = self.ax_lws.imshow(
            self._lws, cmap=LWS_CMAP,
            vmin=lo, vmax=hi, interpolation="nearest")
        self._lws_base_title = "LWS = BD2330 / (D2300+1e-6)"
        self.ax_lws.set_title(
            f"{self._lws_base_title}  [dynamic stretch]",
            fontsize=11, fontweight="bold")
        self.ax_lws.set_xticks([]); self.ax_lws.set_yticks([])

        # Panel 3: RGB
        self._panel3_imshow = self.ax_panel3.imshow(
            self.panel3_rgb, interpolation="nearest")
        self._panel3_base_title = (f"VB RGB  R={PANEL3_CHANNELS[0]}  "
                                    f"G={PANEL3_CHANNELS[1]}  "
                                    f"B={PANEL3_CHANNELS[2]}")
        self.ax_panel3.set_title(
            f"{self._panel3_base_title}  [dynamic stretch]",
            fontsize=11, fontweight="bold")
        self.ax_panel3.set_xticks([]); self.ax_panel3.set_yticks([])

        # Lock autoscale off after imshow set the bounds.
        for ax in (self.ax_bd2330, self.ax_lws, self.ax_panel3):
            ax.set_autoscale_on(False)

        # Brown scatter background (Fo zones).
        self.ax_scatter.axvspan(0.90, 1.15, alpha=0.05, color="grey")
        self.ax_scatter.axvspan(1.15, 1.20, alpha=0.04, color="green")
        self.ax_scatter.axvspan(1.20, 1.30, alpha=0.04, color="orange")
        self.ax_scatter.axvspan(1.30, 1.50, alpha=0.04, color="red")
        self.ax_scatter.axvline(1.15, color="#444444", lw=0.6, ls=":",
                                 alpha=0.6)
        self._scatter_static_texts = []
        for x_, ytext, label_, c in [
            (1.025, 9.5, "below eq.2\n(<1.15)", "#555555"),
            (1.175, 9.5, "high Fo\n(>66)", "green"),
            (1.25, 9.5, "med Fo\n(40-66)", "darkorange"),
            (1.40, 9.5, "low Fo\n(<40)", "red"),
        ]:
            t = self.ax_scatter.text(x_, ytext, label_, ha="center", va="top",
                                       fontsize=10, color=c, alpha=0.85)
            self._scatter_static_texts.append(t)
        self.ax_scatter.set_xlim(*SCATTER_XLIM)
        self.ax_scatter.set_ylim(*SCATTER_YLIM)
        self.ax_scatter.set_xlabel("centroid (um)")
        self.ax_scatter.set_ylabel("asymmetry")
        self.ax_scatter.grid(alpha=0.2)

    def _draw_status_strip(self) -> None:
        self.status_ax = self.fig_viewer.add_axes(
            [0.02, 0.025, 0.96, 0.10])
        self.status_ax.set_xticks([]); self.status_ax.set_yticks([])
        for spine in self.status_ax.spines.values():
            spine.set_visible(False)
        self._status_txt = self.status_ax.text(
            0.0, 0.5, "", fontsize=10, transform=self.status_ax.transAxes,
            va="center", ha="left", family="monospace",
        )

    # ---- Mean spectrum, Brown stats, fit ----
    def _mean_spectrum(self, mask: np.ndarray) -> np.ndarray:
        if mask is None or not mask.any():
            return np.full(self.cube.shape[0], np.nan, dtype=np.float32)
        with np.errstate(invalid="ignore"):
            return np.nanmean(self.cube[:, mask], axis=1)

    def _compute_brown_stats(self, mask: np.ndarray) -> dict:
        sub = mask & self.brown_valid
        n_fin = int(sub.sum())
        out: dict = {"n_finite": n_fin}
        for name in ("centroid", "amplitude", "hwhm", "asymmetry"):
            if n_fin > 0:
                v = self.brown[name][sub]
                out[f"{name}_mean"] = float(np.nanmean(v))
                out[f"{name}_std"] = float(np.nanstd(v))
            else:
                out[f"{name}_mean"] = float("nan")
                out[f"{name}_std"] = float("nan")
        return out

    def _ratio_for_pair(self, tgt: dict, maf: dict) -> np.ndarray:
        st = self._mean_spectrum(tgt["mask"])
        sm = self._mean_spectrum(maf["mask"])
        return st / np.where(np.abs(sm) > 1e-6, sm, np.nan)

    # ---- Selectors ----
    def _on_rect(self, eclick, erelease) -> None:
        if self.mode != "rect":
            return
        if eclick.xdata is None or erelease.xdata is None:
            return
        x_lo, x_hi = sorted([int(round(eclick.xdata)),
                               int(round(erelease.xdata))])
        y_lo, y_hi = sorted([int(round(eclick.ydata)),
                               int(round(erelease.ydata))])
        h, w = self.shape
        x_lo = max(0, x_lo); x_hi = min(w - 1, x_hi)
        y_lo = max(0, y_lo); y_hi = min(h - 1, y_hi)
        if x_hi < x_lo or y_hi < y_lo:
            return
        mask = np.zeros(self.shape, dtype=bool)
        mask[y_lo:y_hi + 1, x_lo:x_hi + 1] = True
        self._add_roi(mask, mode="rect")

    def _on_click(self, event) -> None:
        if self.mode != "pixel" or event.button != 1:
            return
        if event.inaxes not in (self.ax_bd2330, self.ax_lws, self.ax_panel3):
            return
        tb = getattr(self.fig_viewer.canvas, "toolbar", None)
        if tb is not None and getattr(tb, "mode", "") != "":
            return
        if event.xdata is None or event.ydata is None:
            return
        x = int(round(event.xdata)); y = int(round(event.ydata))
        h, w = self.shape
        if not (0 <= x < w and 0 <= y < h):
            return
        coord = (y, x)
        if coord in self.pending_pixels:
            self.pending_pixels.remove(coord)
        else:
            self.pending_pixels.append(coord)
        self._update_pending_overlay()
        kind, label, _, _ = self._next_kind_and_label()
        self._status_txt.set_text(
            f"  mode={self.mode.upper():<6}  rois={len(self.rois):>2}  "
            f"complete pairs={len(self._pairs()):>2}  "
            f"pending pixels={len(self.pending_pixels):>3}  "
            f"next: {label} ({kind})"
        )
        self.fig_viewer.canvas.draw_idle()

    def _finalize_pending(self) -> None:
        if not self.pending_pixels:
            return
        mask = np.zeros(self.shape, dtype=bool)
        for y, x in self.pending_pixels:
            mask[y, x] = True
        self._add_roi(mask, mode="pixel")
        self.pending_pixels = []
        self._update_pending_overlay()

    def _add_roi(self, mask: np.ndarray, mode: str) -> None:
        kind, label, pair_idx, color = self._next_kind_and_label()
        brown = self._compute_brown_stats(mask)
        ys, xs = np.where(mask)
        bbox = (int(xs.min()), int(ys.min()),
                int(xs.max()), int(ys.max())) if len(xs) else (-1, -1, -1, -1)
        roi = {
            "label": label, "kind": kind, "pair_idx": pair_idx,
            "color": color, "mode": mode, "mask": mask,
            "brown": brown, "bbox": bbox,
            "added_ts": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "_saved": False,
        }
        self.rois.append(roi)
        n_px = int(mask.sum())
        cm = brown["centroid_mean"]
        cm_str = f"{cm:.3f}" if np.isfinite(cm) else "nan"
        print(f"[roi] {label}: {n_px} px  mode={mode}  "
              f"centroid_mean={cm_str}  (unsaved; will write at close or 's')")
        self._update_dynamic()

    # ---- Mode + key handling ----
    def _set_mode(self, new_mode: str) -> None:
        if new_mode == self.mode:
            return
        xlim = self.ax_bd2330.get_xlim()
        ylim = self.ax_bd2330.get_ylim()
        if self.mode == "pixel" and self.pending_pixels:
            print(f"[mode] auto-finalizing {len(self.pending_pixels)} "
                  "pending pixel(s)")
            self._finalize_pending()
        self.mode = new_mode
        for rect in self._rects:
            rect.set_active(new_mode == "rect")
        print(f"[mode] -> {self.mode}")
        self._update_pending_overlay()
        self._update_dynamic()
        self.ax_bd2330.set_xlim(xlim)
        self.ax_bd2330.set_ylim(ylim)
        self.fig_viewer.canvas.draw_idle()

    def _on_key(self, event) -> None:
        if event.key == "m":
            self._set_mode("pixel" if self.mode == "rect" else "rect")
        elif event.key in ("enter", "return"):
            if self.mode == "pixel":
                self._finalize_pending()
        elif event.key == "u":
            self._undo()
        elif event.key == "c":
            self._clear()
        elif event.key == "s":
            self._save_session()
        elif event.key == "r":
            self._restretch_now()

    def _undo(self) -> None:
        if self.mode == "pixel" and self.pending_pixels:
            popped = self.pending_pixels.pop()
            print(f"[undo] removed pending pixel {popped}")
            self._update_pending_overlay()
            return
        if not self.rois:
            print("[undo] nothing to remove")
            return
        popped = self.rois.pop()
        print(f"[undo] removed {popped['label']} (CSV row remains)")
        self._update_dynamic()

    def _clear(self) -> None:
        self.rois.clear()
        self.pending_pixels.clear()
        print("[clear] cleared in-memory ROIs + pending pixels")
        self._update_pending_overlay()
        self._update_dynamic()

    # ---- Pending pixel overlay ----
    def _update_pending_overlay(self) -> None:
        for art in self._pending_artists:
            try: art.remove()
            except Exception: pass
        self._pending_artists = []
        if self.mode != "pixel" or not self.pending_pixels:
            self.fig_viewer.canvas.draw_idle()
            return
        ys = [y for y, _ in self.pending_pixels]
        xs = [x for _, x in self.pending_pixels]
        for ax in (self.ax_bd2330, self.ax_lws, self.ax_panel3):
            sc = ax.scatter(xs, ys, s=40, marker="x",
                              c="cyan", linewidths=1.6, zorder=10)
            self._pending_artists.append(sc)
        self.fig_viewer.canvas.draw_idle()

    # ---- Dynamic update ----
    def _update_dynamic(self) -> None:
        for attr in ("viewer_artists", "scatter_artists",
                      "ratio_lines", "resid_lines", "fit_overlays"):
            arts = getattr(self, attr, [])
            for art in arts:
                try: art.remove()
                except Exception: pass
            setattr(self, attr, [])
        # Defensive sweep for stale text on the scatter axis.
        static_ids = set(id(t) for t in
                          getattr(self, "_scatter_static_texts", []))
        for t in list(self.ax_scatter.texts):
            if id(t) not in static_ids:
                try: t.remove()
                except Exception: pass

        # Status text
        n_rois = len(self.rois)
        n_pairs = len(self._pairs())
        n_pending = len(self.pending_pixels)
        next_kind, next_label, _, _ = self._next_kind_and_label()
        self._status_txt.set_text(
            f"  mode={self.mode.upper():<6}  rois={n_rois:>2}  "
            f"complete pairs={n_pairs:>2}  pending pixels={n_pending:>3}  "
            f"next: {next_label} ({next_kind})"
        )
        self._refresh_titles()

        # ROI rectangles + labels on all three viewer panels.
        viewer_axes = (self.ax_bd2330, self.ax_lws, self.ax_panel3)
        for roi in self.rois:
            mask = roi["mask"]
            if not mask.any():
                continue
            color = roi["color"]; label = roi["label"]
            ys, xs = np.where(mask)
            ls = "-" if roi["kind"] == "target" else "--"
            lw = 1.6 if roi["kind"] == "target" else 1.4
            x_lo, x_hi = int(xs.min()), int(xs.max())
            y_lo, y_hi = int(ys.min()), int(ys.max())
            for ax in viewer_axes:
                rect = Rectangle(
                    (x_lo - 0.5, y_lo - 0.5),
                    x_hi - x_lo + 1, y_hi - y_lo + 1,
                    fill=False, edgecolor="white", linewidth=lw,
                    linestyle=ls, zorder=6,
                )
                ax.add_patch(rect)
                self.viewer_artists.append(rect)
                cx = (x_lo + x_hi) / 2.0
                cy = y_hi + 1.0
                txt = ax.text(
                    cx, cy, label,
                    color="black", ha="center", va="top",
                    fontsize=11, fontweight="bold", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              alpha=0.95, edgecolor="black", lw=1.0),
                )
                self.viewer_artists.append(txt)
            if roi["mode"] == "pixel":
                for ax in viewer_axes:
                    sc = ax.scatter(xs, ys, s=20, c=color, marker="o",
                                      alpha=0.85, edgecolors="black",
                                      linewidths=0.5, zorder=8)
                    self.viewer_artists.append(sc)

        # Brown scatter: target ROI means w/ 1-sigma error bars.
        for roi in self._targets():
            br = roi["brown"]
            if not (np.isfinite(br["centroid_mean"])
                    and np.isfinite(br["asymmetry_mean"])):
                continue
            xerr = (br["centroid_std"]
                    if np.isfinite(br["centroid_std"]) else 0.0)
            yerr = (br["asymmetry_std"]
                    if np.isfinite(br["asymmetry_std"]) else 0.0)
            err = self.ax_scatter.errorbar(
                br["centroid_mean"], br["asymmetry_mean"],
                xerr=xerr, yerr=yerr, fmt="o", color=roi["color"],
                ecolor="black", elinewidth=0.8,
                capsize=3, capthick=0.8,
                markersize=10, markeredgecolor="black",
                markeredgewidth=1.0, zorder=5,
                label=f"{roi['label']} (n={int(roi['mask'].sum())})",
            )
            for child in err.get_children():
                self.scatter_artists.append(child)
            txt = self.ax_scatter.text(
                br["centroid_mean"], br["asymmetry_mean"],
                "  " + roi["label"],
                color="black", ha="left", va="center",
                fontsize=10, fontweight="bold", zorder=6,
            )
            self.scatter_artists.append(txt)
        self.ax_scatter.set_title(
            f"Brown centroid vs asymmetry  (ROI means +/- 1 sigma; "
            f"{len(self._targets())} target ROI/s)",
            fontsize=12, fontweight="bold",
        )
        leg = self.ax_scatter.get_legend()
        if leg is not None:
            leg.remove()
        if self._targets():
            self.ax_scatter.legend(fontsize=9, loc="upper right")

        # Ratio + 2.3 fit overlay + residuals
        self.ax_ratio.clear()
        self.ax_resid.clear()
        self.ax_ratio.set_xlabel("wavelength (um)")
        self.ax_ratio.set_ylabel("ratio")
        self.ax_ratio.set_xlim(0.4, 2.6)
        self.ax_resid.set_xlabel("wavelength (um)")
        self.ax_resid.set_ylabel("residual")
        self.ax_resid.set_xlim(FIT_SHOULDER_LO - 0.02, FIT_SHOULDER_HI + 0.02)
        self.ax_resid.axhline(0.0, color="black", lw=0.6, alpha=0.5)
        pairs = self._pairs()
        if pairs:
            self.ax_ratio.axhline(1.0, color="#888888", ls=":", lw=0.7)
            for tgt, maf in pairs:
                ratio = self._ratio_for_pair(tgt, maf)
                line, = self.ax_ratio.plot(
                    self.wav, ratio, color=tgt["color"], lw=1.7,
                    label=f"{tgt['label']} / {maf['label']}",
                )
                self.ratio_lines.append(line)
                # Fit
                fit = fit_2300_band(self.wav, ratio)
                if fit is not None and "alpha" in fit:
                    overlay, = self.ax_ratio.plot(
                        fit["fit_wav"], fit["fit_ratio_overlay"],
                        color=tgt["color"], lw=2.4, alpha=0.55,
                        linestyle="--", zorder=5,
                    )
                    self.fit_overlays.append(overlay)
                    rl, = self.ax_resid.plot(
                        fit["fit_wav"], fit["fit_residual"],
                        color=tgt["color"], lw=1.4,
                        label=(f"{tgt['label']}: lam_min="
                               f"{fit['lambda_min']:.3f}  chi="
                               f"{fit['chi']:+.2f}"),
                    )
                    self.resid_lines.append(rl)
                    # Vertical guide at lambda_min on the ratio plot
                    guide = self.ax_ratio.axvline(
                        fit["lambda_min"], color=tgt["color"],
                        ls=":", lw=0.9, alpha=0.7,
                    )
                    self.fit_overlays.append(guide)
                    txt = self.ax_ratio.text(
                        fit["lambda_min"] + 0.005, 1.001,
                        f"{fit['lambda_min']:.3f}",
                        color=tgt["color"], fontsize=8,
                        ha="left", va="bottom",
                    )
                    self.fit_overlays.append(txt)
            self.ax_ratio.legend(fontsize=10, loc="lower right",
                                  framealpha=0.92)
            self.ax_ratio.set_title(
                f"Ratio spectra ({len(pairs)} complete pair/s) "
                "with 2.3 um asym Gauss fit overlay",
                fontsize=12, fontweight="bold",
            )
            self.ax_resid.legend(fontsize=8, loc="upper right",
                                  framealpha=0.85)
            self.ax_resid.set_title(
                f"2.3 um band fit residuals  "
                f"(window {FIT_SHOULDER_LO:.2f}-{FIT_SHOULDER_HI:.2f} um)",
                fontsize=11,
            )
            annotate_spectrum(self.ax_ratio)
        else:
            self.ax_ratio.set_title(
                "Ratio spectra - draw target then mafic to populate",
                fontsize=12, fontweight="bold",
            )
            self.ax_ratio.text(
                0.5, 0.5,
                "Draw rectangles or click pixels.\n"
                "Auto-cycle: roi 1 -> target1, roi 2 -> mafic1,\n"
                "roi 3 -> target2, roi 4 -> mafic2, ...\n\n"
                "m = mode  enter = finalize  u/c = undo/clear  s = save",
                ha="center", va="center", transform=self.ax_ratio.transAxes,
                color="#888888", fontsize=11,
            )
            self.ax_resid.text(
                0.5, 0.5,
                "Residuals appear once at least one (target, mafic)\n"
                "pair is complete and the 2.3 um fit converges.",
                ha="center", va="center", transform=self.ax_resid.transAxes,
                color="#888888", fontsize=10,
            )
            self.ax_resid.set_title(
                f"2.3 um band fit residuals  "
                f"(window {FIT_SHOULDER_LO:.2f}-{FIT_SHOULDER_HI:.2f} um)",
                fontsize=11,
            )

        self.fig_viewer.canvas.draw_idle()
        self.fig_plots.canvas.draw_idle()

    # ---- Restretch ----
    def _schedule_restretch(self, _ax) -> None:
        if self._restretch_pending:
            return
        self._restretch_pending = True
        try:
            self.fig_viewer.canvas.manager.window.after(
                10, self._restretch_now)
        except Exception:
            self._restretch_now()

    def _restretch_now(self) -> None:
        self._restretch_pending = False
        view = self._visible_slice()
        # BD2330
        lo, hi = self._band_clim(self._bd2330, view, self._bd2330_clim)
        self._bd2330_imshow.set_clim(lo, hi)
        # LWS
        lo2, hi2 = self._band_clim(self._lws, view, self._lws_clim)
        self._lws_imshow.set_clim(lo2, hi2)
        # Panel-3 RGB
        rgb, _ = self._compute_rgb(self._panel3_raw, view)
        self.panel3_rgb = rgb
        self._panel3_imshow.set_data(rgb)
        scope = "visible region" if view is not None else "full scene"
        self.ax_bd2330.set_title(
            f"{self._bd2330_base_title}  [{scope} p2-p98: "
            f"{lo:.3f}-{hi:.3f}]", fontsize=10)
        self.ax_lws.set_title(
            f"{self._lws_base_title}  [{scope} p2-p98: "
            f"{lo2:.3f}-{hi2:.3f}]", fontsize=10)
        self.ax_panel3.set_title(
            f"{self._panel3_base_title}  [{scope} p2-p98]",
            fontsize=10)
        self.fig_viewer.canvas.draw_idle()

    # ---- Save ----
    def _save_session(self) -> None:
        if not self.rois:
            print("[save] no ROIs to save.")
            return
        self._setup_output_dir_lazy()
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        # 1. Per-ROI mask + rois.csv rows
        new_rows = []
        for roi in self.rois:
            if roi.get("_saved"):
                continue
            short = f"{roi['kind']}{roi['pair_idx']}"
            mask_path = (self.session_dir / "masks"
                          / f"{self.cube_id}_{roi['added_ts']}_{short}.npy")
            np.save(mask_path, roi["mask"][::-1, ::-1])
            new_rows.append((roi, mask_path))
            roi["_saved"] = True
        if new_rows:
            with open(self.rois_csv, "a", newline="") as f:
                w = csv.writer(f)
                for roi, mask_path in new_rows:
                    rel_path = (mask_path.relative_to(self.name_dir)
                                  .as_posix())
                    w.writerow([
                        self.session_ts, roi["label"], roi["kind"],
                        str(roi["pair_idx"]), roi["color"], roi["mode"],
                        str(int(roi["mask"].sum())),
                        str(int(roi["brown"]["n_finite"])),
                        f"{roi['brown']['centroid_mean']:.6f}",
                        f"{roi['brown']['centroid_std']:.6f}",
                        f"{roi['brown']['amplitude_mean']:.6f}",
                        f"{roi['brown']['amplitude_std']:.6f}",
                        f"{roi['brown']['hwhm_mean']:.6f}",
                        f"{roi['brown']['hwhm_std']:.6f}",
                        f"{roi['brown']['asymmetry_mean']:.6f}",
                        f"{roi['brown']['asymmetry_std']:.6f}",
                        *(str(v) for v in roi["bbox"]),
                        rel_path,
                    ])

        # 2. Per-session spectra CSV
        self._save_spectra_csv()

        # 3. Per-pair 2.3 um fit results CSV
        self._save_fits_csv()

        # 4. Screenshots + meta
        self.fig_viewer.savefig(
            self.session_dir / f"{self.cube_id}_viewer.png",
            dpi=150, bbox_inches="tight")
        self.fig_plots.savefig(
            self.session_dir / f"{self.cube_id}_plots.png",
            dpi=150, bbox_inches="tight")
        meta = {
            "cube_id": self.cube_id,
            "tool_version": "v9.2",
            "session_ts": self.session_ts,
            "session_name": self.session_name,
            "saved_ts": ts,
            "n_rois_in_session": len(self.rois),
            "rois": [
                {"label": r["label"], "kind": r["kind"],
                 "pair_idx": r["pair_idx"], "mode": r["mode"],
                 "color": r["color"],
                 "n_pixels": int(r["mask"].sum()),
                 "bbox_xlyhxh": list(r["bbox"]),
                 "brown": r["brown"]}
                for r in self.rois
            ],
            "complete_pairs": [
                {"pair_idx": tgt["pair_idx"],
                 "target": tgt["label"], "mafic": maf["label"]}
                for tgt, maf in self._pairs()
            ],
            "panels": {
                "panel1": "BD2330 (single-band, "
                           f"cmap={BD2330_CMAP})",
                "panel2": ("LWS = BD2330 / (D2300+1e-6); "
                           f"no D2300 floor; cmap={LWS_CMAP}"),
                "panel3": (f"VB RGB R={PANEL3_CHANNELS[0]} "
                            f"G={PANEL3_CHANNELS[1]} "
                            f"B={PANEL3_CHANNELS[2]}"),
            },
            "fit_2300_window_um": [FIT_SHOULDER_LO, FIT_SHOULDER_HI],
            "rois_csv": str(self.rois_csv),
            "spectra_csv": str(self.spectra_csv),
            "fits_csv": str(self.fits_csv),
        }
        meta_path = self.session_dir / f"{self.cube_id}_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[save] {len(new_rows)} new mask/CSV row(s); spectra + "
              f"fits + screenshots + meta -> {self.session_dir}")

    def _save_spectra_csv(self) -> None:
        if self.spectra_csv is None:
            return
        rois = self.rois
        if not rois:
            return
        wav = self.wav
        cols = ["wavelength_um"]
        spectra_per_roi = []
        for roi in rois:
            cols.append(f"{roi['label']}_raw_mean")
            spectra_per_roi.append(self._mean_spectrum(roi["mask"]))
        pairs = self._pairs()
        ratios = []
        for tgt, maf in pairs:
            cols.append(f"{tgt['label']}_over_{maf['label']}")
            ratios.append(self._ratio_for_pair(tgt, maf))
        with open(self.spectra_csv, "w", newline="") as f:
            w_csv = csv.writer(f)
            w_csv.writerow(cols)
            for i in range(len(wav)):
                row = [f"{wav[i]:.6f}"]
                for s in spectra_per_roi:
                    row.append(f"{s[i]:.6f}")
                for r in ratios:
                    row.append(f"{r[i]:.6f}")
                w_csv.writerow(row)

    def _save_fits_csv(self) -> None:
        if self.fits_csv is None:
            return
        pairs = self._pairs()
        if not pairs:
            return
        with open(self.fits_csv, "w", newline="") as f:
            w_csv = csv.writer(f)
            w_csv.writerow(FIT_CSV_HEADER)
            for tgt, maf in pairs:
                ratio = self._ratio_for_pair(tgt, maf)
                fit = fit_2300_band(self.wav, ratio)
                if fit is None or "alpha" not in fit:
                    w_csv.writerow([
                        self.session_ts, str(tgt["pair_idx"]),
                        tgt["label"], maf["label"], tgt["color"],
                        str(int(tgt["mask"].sum())),
                        str(int(maf["mask"].sum())),
                        "nan", "nan", "nan", "nan",
                        "nan", "nan", "0",
                        f"{FIT_SHOULDER_LO}", f"{FIT_SHOULDER_HI}",
                    ])
                    continue
                w_csv.writerow([
                    self.session_ts, str(tgt["pair_idx"]),
                    tgt["label"], maf["label"], tgt["color"],
                    str(int(tgt["mask"].sum())),
                    str(int(maf["mask"].sum())),
                    f"{fit['alpha']:.6f}",
                    f"{fit['lambda_0']:.6f}",
                    f"{fit['sigma']:.6f}",
                    f"{fit['chi']:+.6f}",
                    f"{fit['lambda_min']:.6f}",
                    f"{fit['rms_residual']:.6f}",
                    str(fit["n_fit_points"]),
                    f"{fit['shoulder_lo_um']}",
                    f"{fit['shoulder_hi_um']}",
                ])

    def _on_close(self, _evt) -> None:
        if self._closed:
            return
        self._closed = True
        if not self.rois:
            print("[close] no ROIs to save.")
            plt.close("all")
            return
        n_unsaved = sum(1 for r in self.rois if not r.get("_saved"))
        if n_unsaved == 0:
            print(f"[close] all {len(self.rois)} ROIs already persisted; "
                  "writing fresh screenshots + meta + spectra + fits only.")
            try:
                self._save_session()
            except Exception as e:
                print(f"[close] save failed: {e}")
            plt.close("all")
            return
        print()
        print("=" * 60)
        print(f"Closing session for {self.cube_id}: "
              f"{n_unsaved} unsaved ROI(s) (of {len(self.rois)} total).")
        try:
            ans = input("Save session? [Y/n]: ").strip().lower()
        except EOFError:
            ans = "y"
        if ans in ("", "y", "yes"):
            try:
                self._save_session()
                print("[close] session saved.")
            except Exception as e:
                print(f"[close] save failed: {e}")
        else:
            print("[close] discarded session (no files written).")
        plt.close("all")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cube", nargs="?", default="FRT00003E12",
                        help="cube id (full FRT0000XXXX or short like 871C)")
    args = parser.parse_args()
    cube_id = resolve_cube_id(args.cube)
    print(f"=== v9.2 ROI Explorer (BD2330 / LWS / VB RGB): {cube_id} ===")
    print(f"backend: {matplotlib.get_backend()}")
    explorer = RoiExplorer(cube_id)
    plt.show()


if __name__ == "__main__":
    main()
