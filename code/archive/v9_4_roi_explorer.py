"""v9.4 ROI Contrast-Scan Tool with denominator ratioing.

Same workflow as v9.3 (CAR + BD2330/OLINDEX3/BD1900_2 RGB panels,
polygon-based ROI selection, 2.3 um BD-center contrast sweep) but adds
a third polygon (DENOMINATOR / bland region) used for spectral
ratioing.

Per-cube workflow:
  1. Load cube, compute both composites.
  2. User draws TARGET polygon (intrusion) on the right panel.
  3. User draws BACKGROUND polygon (smectite reference).
  4. User draws DENOMINATOR polygon (bland region).
  5. Spectra plot now shows FOUR continuum-removed curves over
     2.0 - 2.6 um:
       - target              (raw, continuum-removed)
       - background          (raw, continuum-removed)
       - target / denominator     (ratioed, continuum-removed)
       - background / denominator (ratioed, continuum-removed)
     Band minima are annotated for all four.
  6. BD-center contrast sweep (unchanged from v9.3): target vs
     background per pixel, anchors 2.250 / 2.400 um, kernel 5,
     centers 2.300 - 2.360 um in 2 nm steps. Contrast curve plotted
     and peak printed.
  7. Saves polygons (3), masks (3), spectra (raw + 2 ratios), scan
     results, summary JSON, and screenshots under
     outputs/spectra sample locations/<cube>/contrast_scan/<TS>/.

Keys:
  r       reset polygons (start over from target)
  s       re-save the current scan to a new timestamped dir
  esc     cancel the polygon currently being drawn

Run:
  python code/v9_4_roi_explorer.py FRT00003E12
  python code/v9_4_roi_explorer.py 871C
"""

from __future__ import annotations

import argparse
import csv
import json
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
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path as MplPath
from matplotlib.widgets import PolygonSelector
import numpy as np

from config import CUBE_IDS, OUTPUTS_DIR
from crism_io import load_cube_by_id
from vb_parameters import (
    calc_olindex3, calc_d2300, calc_bd1900r2, calc_bd2500_2,
    calc_bd1900_2, calc_bd2330,
)

# =============================================================================
# Constants
# =============================================================================
PANEL_LEFT_NAME = "CAR"
PANEL_LEFT_CHANNELS = {"R": "D2300", "G": "BD2500_2", "B": "BD1900R2"}
PANEL_RIGHT_NAME = "BD2330_OLINDEX3_BD1900_2"
PANEL_RIGHT_CHANNELS = {"R": "BD2330", "G": "OLINDEX3", "B": "BD1900_2"}

SCAN_LO_ANCHOR = 2.250
SCAN_HI_ANCHOR = 2.400
SCAN_KERNEL = 5
SCAN_CENTERS = np.round(np.arange(2.300, 2.361, 0.002), 4)

SPEC_LO = 2.00
SPEC_HI = 2.60

ROI_COLORS = {
    "target":      "#ffd400",   # yellow
    "background":  "#e7298a",   # magenta
    "denominator": "#00cccc",   # cyan
}

SAMPLES_ROOT = OUTPUTS_DIR / "spectra sample locations"


# =============================================================================
# Helpers
# =============================================================================
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


def stretch_channel(arr: np.ndarray, lo_p: float = 2.0,
                     hi_p: float = 98.0) -> tuple[np.ndarray, float, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size < 10:
        return np.full_like(arr, 0.0, dtype=np.float32), 0.0, 1.0
    lo, hi = np.percentile(finite, [lo_p, hi_p])
    denom = max(hi - lo, 1e-9)
    out = np.clip((arr - lo) / denom, 0, 1).astype(np.float32)
    out[np.isnan(arr)] = 0.0
    return out, float(lo), float(hi)


def kernel(cube: np.ndarray, wav: np.ndarray, target_um: float,
            width: int = 5) -> np.ndarray:
    idx = int(np.argmin(np.abs(wav - target_um)))
    hw = width // 2
    lo = max(0, idx - hw)
    hi = min(len(wav), idx + hw + 1)
    with np.errstate(invalid="ignore"):
        return np.nanmedian(cube[lo:hi, :, :], axis=0)


def vert_to_mask(verts: list[tuple[float, float]],
                   shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    points = np.column_stack([xx.ravel(), yy.ravel()])
    path = MplPath(np.asarray(verts))
    return path.contains_points(points).reshape(h, w)


def make_poly_selector(ax, onselect):
    common = dict(useblit=True)
    try:
        return PolygonSelector(ax, onselect,
                                 props=dict(color="white", linewidth=2,
                                            linestyle="-"),
                                 handle_props=dict(markerfacecolor="white",
                                                    markeredgecolor="black"),
                                 **common)
    except TypeError:
        return PolygonSelector(
            ax, onselect,
            lineprops=dict(color="white", linewidth=2),
            **common,
        )


def continuum_remove(spec: np.ndarray, wav: np.ndarray,
                     lo_um: float, hi_um: float
                     ) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Continuum-remove `spec` over [lo_um, hi_um] via a linear
    interpolation between the wavelength-mean reflectances at the two
    endpoints (5-channel mean per shoulder). Returns
    (window_wav, R_cr, R_lo, R_hi)."""
    win = (wav >= lo_um - 0.005) & (wav <= hi_um + 0.005)
    w = wav[win]
    r = spec[win]
    idx_lo = int(np.argmin(np.abs(wav - lo_um)))
    idx_hi = int(np.argmin(np.abs(wav - hi_um)))
    hw = 2
    with np.errstate(invalid="ignore"):
        r_lo = float(np.nanmean(spec[max(0, idx_lo - hw):idx_lo + hw + 1]))
        r_hi = float(np.nanmean(spec[max(0, idx_hi - hw):idx_hi + hw + 1]))
    cont = r_lo + (r_hi - r_lo) * (w - lo_um) / (hi_um - lo_um)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_cr = np.where(cont != 0, r / cont, np.nan)
    return w, r_cr, r_lo, r_hi


# =============================================================================
# Tool
# =============================================================================
class ContrastScanTool:
    POLY_ORDER = ("target", "background", "denominator")

    def __init__(self, cube_id: str):
        self.cube_id = cube_id
        self.session_ts = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ")
        self._role: str = "target"
        self._verts: dict[str, list] = {}
        self._masks: dict[str, np.ndarray] = {}
        self._poly_overlays: list = []
        self._scan_done = False
        self._load_data()
        self._build_figures()
        self._init_polygon_selector()

    # ---- Load + composites ----
    def _load_data(self) -> None:
        print(f"[load] {self.cube_id} cube...")
        t0 = time.time()
        cube, wav, brown, _ = load_cube_by_id(self.cube_id)
        self.cube = cube[:, ::-1, ::-1].copy()
        self.wav = wav.astype(np.float32)
        self.shape = self.cube.shape[1:]
        print(f"  cube {self.cube.shape} wav {wav[0]:.3f}-{wav[-1]:.3f} um "
              f"({time.time()-t0:.1f}s)")

        print("  computing CAR channels (D2300, BD2500_2, BD1900R2)...")
        d2300 = calc_d2300(self.cube, self.wav).astype(np.float32)
        bd2500 = calc_bd2500_2(self.cube, self.wav).astype(np.float32)
        bd1900r2 = calc_bd1900r2(self.cube, self.wav).astype(np.float32)
        self._car_raw = {"R": d2300, "G": bd2500, "B": bd1900r2}

        print("  computing BD2330 / OLINDEX3 / BD1900_2 channels...")
        bd2330 = calc_bd2330(self.cube, self.wav).astype(np.float32)
        olindex3 = calc_olindex3(self.cube, self.wav).astype(np.float32)
        bd1900_2 = calc_bd1900_2(self.cube, self.wav).astype(np.float32)
        self._panel_right_raw = {"R": bd2330, "G": olindex3, "B": bd1900_2}

        h, w = self.shape
        self._car_rgb = np.zeros((h, w, 3), dtype=np.float32)
        self._panel_right_rgb = np.zeros((h, w, 3), dtype=np.float32)
        for k, ch in enumerate(("R", "G", "B")):
            self._car_rgb[..., k], *_ = stretch_channel(self._car_raw[ch])
            self._panel_right_rgb[..., k], *_ = stretch_channel(
                self._panel_right_raw[ch])

        self._R_lo_anchor = kernel(self.cube, self.wav,
                                     SCAN_LO_ANCHOR, SCAN_KERNEL)
        self._R_hi_anchor = kernel(self.cube, self.wav,
                                     SCAN_HI_ANCHOR, SCAN_KERNEL)
        print("  done.")

    # ---- Figures ----
    def _build_figures(self) -> None:
        plt.rcdefaults()
        plt.rcParams.update({
            "axes.titlesize": 13, "axes.labelsize": 12,
            "legend.fontsize": 10, "xtick.labelsize": 10,
            "ytick.labelsize": 10, "figure.titlesize": 12,
        })
        # Window 1: viewer
        self.fig_viewer = plt.figure(
            figsize=(14, 7), num=f"v9.4 Viewer - {self.cube_id}")
        gs = self.fig_viewer.add_gridspec(
            1, 2, wspace=0.06,
            left=0.02, right=0.99, top=0.92, bottom=0.13,
        )
        self.ax_car = self.fig_viewer.add_subplot(gs[0, 0])
        self.ax_panel_right = self.fig_viewer.add_subplot(
            gs[0, 1], sharex=self.ax_car, sharey=self.ax_car)
        self.ax_car.imshow(self._car_rgb, interpolation="nearest")
        self.ax_panel_right.imshow(self._panel_right_rgb,
                                    interpolation="nearest")
        ch_l = PANEL_LEFT_CHANNELS
        ch_r = PANEL_RIGHT_CHANNELS
        self.ax_car.set_title(f"CAR  R={ch_l['R']}  G={ch_l['G']}  "
                                f"B={ch_l['B']}",
                                fontsize=11, fontweight="bold")
        self.ax_panel_right.set_title(
            f"VB RGB  R={ch_r['R']}  G={ch_r['G']}  B={ch_r['B']}  "
            "(draw polygons here)",
            fontsize=11, fontweight="bold")
        for ax in (self.ax_car, self.ax_panel_right):
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_autoscale_on(False)
        self.status_ax = self.fig_viewer.add_axes([0.02, 0.02, 0.96, 0.07])
        self.status_ax.set_xticks([]); self.status_ax.set_yticks([])
        for spine in self.status_ax.spines.values():
            spine.set_visible(False)
        self._status = self.status_ax.text(
            0.0, 0.5, "", fontsize=11, transform=self.status_ax.transAxes,
            va="center", ha="left", family="monospace",
        )
        self._update_status()

        # Window 2: analysis (spectra + contrast)
        self.fig_plots = plt.figure(
            figsize=(11, 9), num=f"v9.4 Analysis - {self.cube_id}")
        gs2 = self.fig_plots.add_gridspec(
            2, 1, height_ratios=[1.2, 1], hspace=0.32,
            left=0.10, right=0.97, top=0.95, bottom=0.07,
        )
        self.ax_spectra = self.fig_plots.add_subplot(gs2[0, 0])
        self.ax_curve = self.fig_plots.add_subplot(gs2[1, 0])
        self.ax_spectra.set_xlabel(r"wavelength ($\mu$m)")
        self.ax_spectra.set_ylabel("continuum-removed reflectance")
        self.ax_spectra.set_xlim(SPEC_LO, SPEC_HI)
        self.ax_spectra.set_title(
            "Continuum-removed mean spectra "
            "(target / background / target-over-denom / "
            "background-over-denom) - draw all three polygons to populate",
            fontsize=11)
        self.ax_curve.set_xlabel(r"BD center wavelength ($\mu$m)")
        self.ax_curve.set_ylabel(r"contrast = mean(BD)$_T$ - mean(BD)$_B$")
        self.ax_curve.set_xlim(SCAN_CENTERS[0] - 0.002,
                                SCAN_CENTERS[-1] + 0.002)
        self.ax_curve.set_title(
            "BD-center contrast scan (target vs background; "
            "denominator not used here)", fontsize=11)
        for a in (self.ax_spectra, self.ax_curve):
            a.grid(True, alpha=0.3)

        self.fig_viewer.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig_plots.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig_viewer.canvas.mpl_connect(
            "close_event", lambda evt: plt.close("all"))
        self.fig_plots.canvas.mpl_connect(
            "close_event", lambda evt: plt.close("all"))

    def _update_status(self) -> None:
        order_msg = "  ".join(
            f"{i+1}/{len(self.POLY_ORDER)}={r.upper()}"
            for i, r in enumerate(self.POLY_ORDER))
        if self._role in self.POLY_ORDER:
            step = self.POLY_ORDER.index(self._role) + 1
            msg = (f"[{step}/{len(self.POLY_ORDER)}] Drawing "
                   f"{self._role.upper()} polygon on right panel: click "
                   "vertices, close by clicking first vertex.   "
                   f"({order_msg})")
        else:
            msg = ("Done. Press 'r' to redraw polygons, 's' to re-save.")
        self._status.set_text(msg)
        self.fig_viewer.canvas.draw_idle()

    # ---- Polygon selection ----
    def _init_polygon_selector(self) -> None:
        self._ps = make_poly_selector(self.ax_panel_right,
                                       self._on_poly_done)

    def _on_poly_done(self, vertices) -> None:
        verts = [(float(v[0]), float(v[1])) for v in vertices]
        if len(verts) < 3:
            print("[poly] need >= 3 vertices.")
            return
        if self._role not in self.POLY_ORDER:
            return
        role = self._role
        self._verts[role] = verts
        self._masks[role] = vert_to_mask(verts, self.shape)
        n = int(self._masks[role].sum())
        print(f"[poly] {role}: {n} pixels")
        self._draw_polygon_overlay(verts, role.upper(), ROI_COLORS[role])
        try:
            self._ps.disconnect_events()
        except Exception:
            pass
        # Advance to next role
        next_idx = self.POLY_ORDER.index(role) + 1
        if next_idx < len(self.POLY_ORDER):
            self._role = self.POLY_ORDER[next_idx]
            self._init_polygon_selector()
            self._update_status()
        else:
            self._role = "scan"
            self._update_status()
            self._run_scan_and_save()

    def _draw_polygon_overlay(self, verts, label: str, color: str) -> None:
        for ax in (self.ax_car, self.ax_panel_right):
            poly = MplPolygon(verts, fill=False, edgecolor=color,
                               linewidth=2.0, zorder=10)
            ax.add_patch(poly)
            cx = float(np.mean([v[0] for v in verts]))
            cy = float(np.mean([v[1] for v in verts]))
            txt = ax.text(cx, cy, label, color="black",
                            ha="center", va="center", fontsize=11,
                            fontweight="bold", zorder=11,
                            bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor="white", edgecolor=color,
                                      lw=1.5, alpha=0.95))
            self._poly_overlays.append(poly)
            self._poly_overlays.append(txt)
        self.fig_viewer.canvas.draw_idle()

    # ---- Scan ----
    def _mean_bd_for_center(self, center: float,
                              mask_t: np.ndarray, mask_b: np.ndarray
                              ) -> tuple[float, float, float]:
        a = (SCAN_HI_ANCHOR - center) / (SCAN_HI_ANCHOR - SCAN_LO_ANCHOR)
        b = (center - SCAN_LO_ANCHOR) / (SCAN_HI_ANCHOR - SCAN_LO_ANCHOR)
        cont = a * self._R_lo_anchor + b * self._R_hi_anchor
        R_c = kernel(self.cube, self.wav, center, SCAN_KERNEL)
        with np.errstate(divide="ignore", invalid="ignore"):
            bd = np.where(cont != 0, 1.0 - R_c / cont, np.nan)
        with np.errstate(invalid="ignore"):
            bd_t = float(np.nanmean(bd[mask_t]))
            bd_b = float(np.nanmean(bd[mask_b]))
        return bd_t, bd_b, bd_t - bd_b

    def _run_scan_and_save(self) -> None:
        m_t = self._masks.get("target")
        m_b = self._masks.get("background")
        m_d = self._masks.get("denominator")
        if m_t is None or m_b is None or m_d is None:
            return
        n_t = int(m_t.sum()); n_b = int(m_b.sum()); n_d = int(m_d.sum())
        print(f"[scan] target n={n_t}  background n={n_b}  denominator n={n_d}")

        # 1. Mean spectra (raw reflectances)
        with np.errstate(invalid="ignore"):
            spec_t = np.nanmean(self.cube[:, m_t], axis=1)
            spec_b = np.nanmean(self.cube[:, m_b], axis=1)
            spec_d = np.nanmean(self.cube[:, m_d], axis=1)

        # 2. Continuum-remove the raw spectra (existing v9.3 behavior)
        w_t, r_cr_t, r_lo_t, r_hi_t = continuum_remove(spec_t, self.wav,
                                                         SPEC_LO, SPEC_HI)
        w_b, r_cr_b, r_lo_b, r_hi_b = continuum_remove(spec_b, self.wav,
                                                         SPEC_LO, SPEC_HI)

        # 3. Ratioed spectra: target/denom and background/denom; then continuum-remove
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_td = np.where(np.abs(spec_d) > 1e-6, spec_t / spec_d, np.nan)
            ratio_bd = np.where(np.abs(spec_d) > 1e-6, spec_b / spec_d, np.nan)
        w_td, r_cr_td, *_ = continuum_remove(ratio_td, self.wav,
                                              SPEC_LO, SPEC_HI)
        w_bd, r_cr_bd, *_ = continuum_remove(ratio_bd, self.wav,
                                              SPEC_LO, SPEC_HI)

        # 4. Band minima for each curve
        def argmin_safe(arr):
            if not np.isfinite(arr).any():
                return None
            i = int(np.nanargmin(arr))
            return i
        i_t  = argmin_safe(r_cr_t)
        i_b  = argmin_safe(r_cr_b)
        i_td = argmin_safe(r_cr_td)
        i_bd = argmin_safe(r_cr_bd)
        lam_t  = float(w_t[i_t])  if i_t  is not None else float("nan")
        lam_b  = float(w_b[i_b])  if i_b  is not None else float("nan")
        lam_td = float(w_td[i_td]) if i_td is not None else float("nan")
        lam_bd = float(w_bd[i_bd]) if i_bd is not None else float("nan")
        rcr_t  = float(r_cr_t[i_t])  if i_t  is not None else float("nan")
        rcr_b  = float(r_cr_b[i_b])  if i_b  is not None else float("nan")
        rcr_td = float(r_cr_td[i_td]) if i_td is not None else float("nan")
        rcr_bd = float(r_cr_bd[i_bd]) if i_bd is not None else float("nan")
        for label, lam, rcr in [
            ("target",            lam_t, rcr_t),
            ("background",        lam_b, rcr_b),
            ("target/denom",      lam_td, rcr_td),
            ("background/denom",  lam_bd, rcr_bd),
        ]:
            if np.isfinite(lam):
                print(f"[scan] {label:<18s} band min  "
                      f"lambda={lam:.4f} um  R_cr={rcr:.4f}")

        # 5. BD-center sweep (target vs background, unchanged from v9.3)
        bd_t = np.empty_like(SCAN_CENTERS, dtype=np.float64)
        bd_b = np.empty_like(SCAN_CENTERS, dtype=np.float64)
        contrast = np.empty_like(SCAN_CENTERS, dtype=np.float64)
        for i, c in enumerate(SCAN_CENTERS):
            bdt, bdb, dlt = self._mean_bd_for_center(float(c), m_t, m_b)
            bd_t[i] = bdt
            bd_b[i] = bdb
            contrast[i] = dlt
        peak_idx = int(np.nanargmax(contrast))
        peak_lambda = float(SCAN_CENTERS[peak_idx])
        peak_contrast = float(contrast[peak_idx])
        print(f"[scan] PEAK CONTRAST at lambda = {peak_lambda:.4f} um "
              f"with contrast = {peak_contrast:.4f}")
        print(f"       (target BD = {bd_t[peak_idx]:.4f}, "
              f"background BD = {bd_b[peak_idx]:.4f})")

        # 6. Plot + save
        self._plot_results(
            w_t, r_cr_t, w_b, r_cr_b, w_td, r_cr_td, w_bd, r_cr_bd,
            (lam_t, rcr_t), (lam_b, rcr_b),
            (lam_td, rcr_td), (lam_bd, rcr_bd),
            bd_t, bd_b, contrast, peak_lambda, peak_contrast,
        )
        outdir = self._save_outputs(
            spec_t, spec_b, spec_d,
            ratio_td, ratio_bd,
            w_t, r_cr_t, w_b, r_cr_b, w_td, r_cr_td, w_bd, r_cr_bd,
            (lam_t, lam_b, lam_td, lam_bd),
            bd_t, bd_b, contrast,
            peak_lambda, peak_contrast, n_t, n_b, n_d,
        )
        print(f"[save] -> {outdir}")
        self._scan_done = True
        self._update_status()

    # ---- Plot ----
    def _plot_results(self,
                       w_t, r_cr_t, w_b, r_cr_b,
                       w_td, r_cr_td, w_bd, r_cr_bd,
                       min_t, min_b, min_td, min_bd,
                       bd_t, bd_b, contrast,
                       peak_lambda, peak_contrast) -> None:
        # Spectra panel
        self.ax_spectra.clear()
        self.ax_spectra.set_xlabel(r"wavelength ($\mu$m)")
        self.ax_spectra.set_ylabel("continuum-removed reflectance")
        self.ax_spectra.set_xlim(SPEC_LO, SPEC_HI)
        self.ax_spectra.axhline(1.0, color="black", lw=0.5, alpha=0.4)
        for w_ref in (2.30, 2.33, 2.40, 2.50):
            self.ax_spectra.axvline(w_ref, color="#aa4444", lw=0.5,
                                      ls="--", alpha=0.4, zorder=0)

        c_t = ROI_COLORS["target"]
        c_b = ROI_COLORS["background"]
        # Raw continuum-removed (solid, lighter weight to give visual room
        # for the ratioed curves which are the primary scientific result).
        self.ax_spectra.plot(w_t, r_cr_t, color=c_t, lw=1.4, alpha=0.85,
                              ls="-",
                              label=f"target  (min {min_t[0]:.4f})")
        self.ax_spectra.plot(w_b, r_cr_b, color=c_b, lw=1.4, alpha=0.85,
                              ls="-",
                              label=f"background  (min {min_b[0]:.4f})")
        # Ratioed (dashed, bolder)
        self.ax_spectra.plot(w_td, r_cr_td, color=c_t, lw=2.2, ls="--",
                              label=f"target / denom  "
                                    f"(min {min_td[0]:.4f})")
        self.ax_spectra.plot(w_bd, r_cr_bd, color=c_b, lw=2.2, ls="--",
                              label=f"background / denom  "
                                    f"(min {min_bd[0]:.4f})")

        # Band-min markers
        for (lam, rcr), color, marker in [
            (min_t,  c_t, "o"),
            (min_b,  c_b, "o"),
            (min_td, c_t, "D"),
            (min_bd, c_b, "D"),
        ]:
            if np.isfinite(lam) and np.isfinite(rcr):
                self.ax_spectra.scatter([lam], [rcr], color=color,
                                          marker=marker, s=70,
                                          edgecolor="black", linewidth=0.6,
                                          zorder=5)
        n_t = int(self._masks["target"].sum())
        n_b = int(self._masks["background"].sum())
        n_d = int(self._masks["denominator"].sum())
        self.ax_spectra.set_title(
            "Continuum-removed mean spectra (2.0-2.6 um shoulders) "
            f"|  target n={n_t}  background n={n_b}  denominator n={n_d}",
            fontsize=11)
        self.ax_spectra.legend(fontsize=9, loc="lower right",
                                framealpha=0.92)
        self.ax_spectra.grid(True, alpha=0.3)

        # Contrast curve panel (target vs background; denominator NOT used)
        self.ax_curve.clear()
        self.ax_curve.set_xlabel(r"BD center wavelength ($\mu$m)")
        self.ax_curve.set_ylabel(r"contrast = mean(BD)$_T$ - mean(BD)$_B$")
        self.ax_curve.axhline(0.0, color="black", lw=0.5, alpha=0.4)
        self.ax_curve.plot(SCAN_CENTERS, contrast, "-o",
                             color="#1f77b4", lw=1.6, ms=4,
                             label="contrast")
        self.ax_curve.scatter([peak_lambda], [peak_contrast],
                                color="#d62728", s=130, zorder=5,
                                edgecolor="black", linewidth=0.8,
                                label=f"peak {peak_lambda:.4f} um  "
                                       f"contrast {peak_contrast:.4f}")
        self.ax_curve.axvline(peak_lambda, color="#d62728", ls="--",
                                lw=0.8, alpha=0.5)
        self.ax_curve.legend(fontsize=10, loc="best")
        self.ax_curve.set_title(
            f"BD-center contrast scan  "
            f"(centers {SCAN_CENTERS[0]:.3f}-{SCAN_CENTERS[-1]:.3f} um, "
            f"step {(SCAN_CENTERS[1]-SCAN_CENTERS[0])*1000:.0f} nm; "
            f"anchors {SCAN_LO_ANCHOR:.3f}/{SCAN_HI_ANCHOR:.3f}, "
            f"kernel {SCAN_KERNEL})",
            fontsize=10)
        self.ax_curve.grid(True, alpha=0.3)

        self.fig_plots.canvas.draw_idle()

    # ---- Save ----
    def _save_outputs(self,
                       spec_t, spec_b, spec_d,
                       ratio_td, ratio_bd,
                       w_t, r_cr_t, w_b, r_cr_b,
                       w_td, r_cr_td, w_bd, r_cr_bd,
                       lams,
                       bd_t, bd_b, contrast,
                       peak_lambda, peak_contrast,
                       n_t, n_b, n_d) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        outdir = (SAMPLES_ROOT / self.cube_id / "contrast_scan" / ts)
        outdir.mkdir(parents=True, exist_ok=True)

        # Polygons (3)
        poly_payload = {
            "cube_id": self.cube_id,
            "session_ts": self.session_ts,
            "saved_ts": ts,
            "tool_version": "v9.4",
            "polygon_order": list(self.POLY_ORDER),
            "target_vertices_xy": self._verts.get("target"),
            "background_vertices_xy": self._verts.get("background"),
            "denominator_vertices_xy": self._verts.get("denominator"),
            "shape_HW": list(self.shape),
            "display_orientation": ("flipped vertically AND horizontally "
                                     "vs detector; coords above are in "
                                     "display orientation"),
        }
        (outdir / "polygons.json").write_text(
            json.dumps(poly_payload, indent=2), encoding="utf-8")

        # Masks (un-flipped to detector orientation on disk).
        np.save(outdir / "target_mask.npy",
                self._masks["target"][::-1, ::-1])
        np.save(outdir / "background_mask.npy",
                self._masks["background"][::-1, ::-1])
        np.save(outdir / "denominator_mask.npy",
                self._masks["denominator"][::-1, ::-1])

        # Mean spectra CSV (raw + continuum-removed + ratios + cr-ratios).
        mask_window = ((self.wav >= SPEC_LO - 0.005)
                        & (self.wav <= SPEC_HI + 0.005))

        def fill(buf):
            full = np.full_like(self.wav, np.nan, dtype=np.float64)
            full[mask_window] = buf
            return full

        r_cr_t_full  = fill(r_cr_t)
        r_cr_b_full  = fill(r_cr_b)
        r_cr_td_full = fill(r_cr_td)
        r_cr_bd_full = fill(r_cr_bd)

        with open(outdir / "mean_spectra.csv", "w", newline="") as f:
            w_csv = csv.writer(f)
            w_csv.writerow([
                "wavelength_um",
                "target_R", "background_R", "denominator_R",
                "target_R_cr_2p0_2p6", "background_R_cr_2p0_2p6",
                "target_over_denominator_R",
                "background_over_denominator_R",
                "target_over_denominator_R_cr_2p0_2p6",
                "background_over_denominator_R_cr_2p0_2p6",
            ])
            for i in range(len(self.wav)):
                row = [
                    f"{self.wav[i]:.6f}",
                    f"{spec_t[i]:.6f}",
                    f"{spec_b[i]:.6f}",
                    f"{spec_d[i]:.6f}",
                    (f"{r_cr_t_full[i]:.6f}"
                     if np.isfinite(r_cr_t_full[i]) else ""),
                    (f"{r_cr_b_full[i]:.6f}"
                     if np.isfinite(r_cr_b_full[i]) else ""),
                    (f"{ratio_td[i]:.6f}"
                     if np.isfinite(ratio_td[i]) else ""),
                    (f"{ratio_bd[i]:.6f}"
                     if np.isfinite(ratio_bd[i]) else ""),
                    (f"{r_cr_td_full[i]:.6f}"
                     if np.isfinite(r_cr_td_full[i]) else ""),
                    (f"{r_cr_bd_full[i]:.6f}"
                     if np.isfinite(r_cr_bd_full[i]) else ""),
                ]
                w_csv.writerow(row)

        # Scan results CSV
        with open(outdir / "scan_results.csv", "w", newline="") as f:
            w_csv = csv.writer(f)
            w_csv.writerow(["bd_center_um", "target_bd_mean",
                             "background_bd_mean", "contrast"])
            for i, c in enumerate(SCAN_CENTERS):
                w_csv.writerow([f"{c:.4f}",
                                  f"{bd_t[i]:.6f}",
                                  f"{bd_b[i]:.6f}",
                                  f"{contrast[i]:.6f}"])

        # Summary JSON
        lam_t, lam_b, lam_td, lam_bd = lams
        summary = {
            "cube_id": self.cube_id,
            "session_ts": self.session_ts,
            "saved_ts": ts,
            "tool_version": "v9.4",
            "n_target_pixels": int(n_t),
            "n_background_pixels": int(n_b),
            "n_denominator_pixels": int(n_d),
            "scan": {
                "centers_um": [float(c) for c in SCAN_CENTERS],
                "lo_anchor_um": SCAN_LO_ANCHOR,
                "hi_anchor_um": SCAN_HI_ANCHOR,
                "kernel": SCAN_KERNEL,
            },
            "spec_continuum_window_um": [SPEC_LO, SPEC_HI],
            "band_minima_um": {
                "target":           lam_t,
                "background":       lam_b,
                "target_over_denom":     lam_td,
                "background_over_denom": lam_bd,
            },
            "peak_contrast": {
                "bd_center_um": float(peak_lambda),
                "contrast": float(peak_contrast),
                "target_bd_mean": float(bd_t[int(np.nanargmax(contrast))]),
                "background_bd_mean": float(
                    bd_b[int(np.nanargmax(contrast))]),
            },
        }
        (outdir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")

        try:
            self.fig_viewer.savefig(
                outdir / "viewer.png", dpi=150,
                bbox_inches="tight", facecolor="white")
            self.fig_plots.savefig(
                outdir / "plots.png", dpi=150,
                bbox_inches="tight", facecolor="white")
        except Exception as e:
            print(f"[save] screenshot failed: {e}")

        return outdir

    # ---- Reset / re-save ----
    def _reset(self) -> None:
        for art in self._poly_overlays:
            try: art.remove()
            except Exception: pass
        self._poly_overlays = []
        self._verts.clear()
        self._masks.clear()
        self._role = "target"
        self._scan_done = False
        try:
            self._ps.disconnect_events()
        except Exception:
            pass
        self._init_polygon_selector()
        self.ax_spectra.clear()
        self.ax_curve.clear()
        self.ax_spectra.set_title(
            "Continuum-removed mean spectra (target / background / "
            "target-over-denom / background-over-denom) - draw all "
            "three polygons to populate", fontsize=11)
        self.ax_curve.set_title(
            "BD-center contrast scan (target vs background; "
            "denominator not used here)", fontsize=11)
        for a in (self.ax_spectra, self.ax_curve):
            a.grid(True, alpha=0.3)
        self.fig_plots.canvas.draw_idle()
        self.fig_viewer.canvas.draw_idle()
        self._update_status()
        print("[reset] polygons cleared; draw target again.")

    def _on_key(self, event) -> None:
        if event.key == "r":
            self._reset()
        elif event.key == "s":
            if not self._scan_done:
                print("[save] scan not yet completed; nothing to save.")
                return
            self._run_scan_and_save()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cube", nargs="?", default="FRT00003E12",
                        help="cube id (full FRT0000XXXX or short like 871C)")
    args = parser.parse_args()
    cube_id = resolve_cube_id(args.cube)
    print(f"=== v9.4 Contrast-Scan + Ratio Tool: {cube_id} ===")
    print(f"backend: {matplotlib.get_backend()}")
    tool = ContrastScanTool(cube_id)
    plt.show()


if __name__ == "__main__":
    main()
