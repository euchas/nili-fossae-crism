"""v9.3 ROI Contrast-Scan Tool.

Two-panel viewer (CAR + BD2330/OLINDEX3/BD1900_2 RGB) with polygon-
based ROI selection and an automatic 2.3 um band-center contrast sweep.

Workflow per cube:
  1. Load the cube and compute the two reference composites.
  2. User draws TARGET polygon (intrusion) on the BD2330 RGB panel.
     The polygon outline mirrors onto the CAR panel for context.
  3. User draws BACKGROUND polygon (smectite reference); same
     mirroring.
  4. Tool plots continuum-removed mean spectra of both ROIs over the
     2.0 - 2.6 um range with band minima annotated.
  5. Tool runs the BD-center sweep:
       bd_center in {2.300, 2.302, ..., 2.360} (2 nm steps),
       anchors fixed at 2.250 and 2.400 um, 5-channel kernels,
       contrast = mean(BD_target) - mean(BD_background).
  6. Plots contrast vs bd_center, annotates the peak, prints the
     peak wavelength + contrast value.
  7. Saves polygons, masks, spectra, scan results, and summary JSON
     under outputs/spectra sample locations/<cube>/contrast_scan/<TS>/.

Keys:
  r       reset polygons (start over from target)
  s       manually re-save the current scan to a new timestamped dir
  esc     cancel the polygon currently being drawn

Run:
  python code/v9_3_roi_explorer.py FRT00003E12
  python code/v9_3_roi_explorer.py 871C
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

# BD-center sweep configuration (project specification)
SCAN_LO_ANCHOR = 2.250
SCAN_HI_ANCHOR = 2.400
SCAN_KERNEL = 5
SCAN_CENTERS = np.round(np.arange(2.300, 2.361, 0.002), 4)

# Sanity-check spectrum range (continuum removed via shoulders at SPEC_LO/HI)
SPEC_LO = 2.00
SPEC_HI = 2.60

ROI_COLORS = {"target": "#ffd400", "background": "#e7298a"}

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
    """Median over `width` channels closest to `target_um`."""
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
    """Cross-version PolygonSelector constructor."""
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


# =============================================================================
# Tool
# =============================================================================
class ContrastScanTool:
    def __init__(self, cube_id: str):
        self.cube_id = cube_id
        self.session_ts = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ")
        self._role: str = "target"
        self._target_verts: list[tuple[float, float]] | None = None
        self._target_mask: np.ndarray | None = None
        self._background_verts: list[tuple[float, float]] | None = None
        self._background_mask: np.ndarray | None = None
        self._poly_overlays: list = []
        self._scan_done = False
        self._load_data()
        self._build_figures()
        self._init_polygon_selector()

    # ---- Loading + composites ----
    def _load_data(self) -> None:
        print(f"[load] {self.cube_id} cube...")
        t0 = time.time()
        cube, wav, brown, _ = load_cube_by_id(self.cube_id)
        # Display flip H + W (consistent with v9.1/v9.2).
        self.cube = cube[:, ::-1, ::-1].copy()
        self.wav = wav.astype(np.float32)
        self.shape = self.cube.shape[1:]
        print(f"  cube {self.cube.shape} wav {wav[0]:.3f}-{wav[-1]:.3f} um  "
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

        # Pre-build display RGBs (full-scene p2/p98 stretch per channel).
        h, w = self.shape
        self._car_rgb = np.zeros((h, w, 3), dtype=np.float32)
        self._panel_right_rgb = np.zeros((h, w, 3), dtype=np.float32)
        for k, ch in enumerate(("R", "G", "B")):
            self._car_rgb[..., k], *_ = stretch_channel(self._car_raw[ch])
            self._panel_right_rgb[..., k], *_ = stretch_channel(
                self._panel_right_raw[ch])

        # Pre-cache the BD anchors the scan reuses on every iteration.
        self._R_lo_anchor = kernel(self.cube, self.wav,
                                     SCAN_LO_ANCHOR, SCAN_KERNEL)
        self._R_hi_anchor = kernel(self.cube, self.wav,
                                     SCAN_HI_ANCHOR, SCAN_KERNEL)
        print("  done.")

    # ---- Figure layout ----
    def _build_figures(self) -> None:
        plt.rcdefaults()
        plt.rcParams.update({
            "axes.titlesize": 13, "axes.labelsize": 12,
            "legend.fontsize": 10, "xtick.labelsize": 10,
            "ytick.labelsize": 10, "figure.titlesize": 12,
        })

        # Window 1: viewer with two synced panels.
        self.fig_viewer = plt.figure(
            figsize=(14, 7), num=f"v9.3 Viewer - {self.cube_id}")
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
        # Status strip at the bottom of the viewer.
        self.status_ax = self.fig_viewer.add_axes([0.02, 0.02, 0.96, 0.07])
        self.status_ax.set_xticks([]); self.status_ax.set_yticks([])
        for spine in self.status_ax.spines.values():
            spine.set_visible(False)
        self._status = self.status_ax.text(
            0.0, 0.5, "", fontsize=11, transform=self.status_ax.transAxes,
            va="center", ha="left", family="monospace",
        )
        self._update_status()

        # Window 2: analysis (spectra + contrast curve)
        self.fig_plots = plt.figure(
            figsize=(11, 8.5), num=f"v9.3 Analysis - {self.cube_id}")
        gs2 = self.fig_plots.add_gridspec(
            2, 1, height_ratios=[1.1, 1], hspace=0.32,
            left=0.10, right=0.97, top=0.95, bottom=0.07,
        )
        self.ax_spectra = self.fig_plots.add_subplot(gs2[0, 0])
        self.ax_curve = self.fig_plots.add_subplot(gs2[1, 0])
        self.ax_spectra.set_xlabel(r"wavelength ($\mu$m)")
        self.ax_spectra.set_ylabel("continuum-removed reflectance")
        self.ax_spectra.set_xlim(SPEC_LO, SPEC_HI)
        self.ax_spectra.set_title(
            "Continuum-removed mean spectra (target vs background) - "
            "draw both polygons to populate", fontsize=11)
        self.ax_curve.set_xlabel(r"BD center wavelength ($\mu$m)")
        self.ax_curve.set_ylabel(r"contrast = mean(BD)$_T$ - mean(BD)$_B$")
        self.ax_curve.set_xlim(SCAN_CENTERS[0] - 0.002,
                                SCAN_CENTERS[-1] + 0.002)
        self.ax_curve.set_title(
            "BD-center contrast scan - populates after both polygons drawn",
            fontsize=11)
        for a in (self.ax_spectra, self.ax_curve):
            a.grid(True, alpha=0.3)

        self.fig_viewer.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig_plots.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig_viewer.canvas.mpl_connect(
            "close_event", lambda evt: plt.close("all"))
        self.fig_plots.canvas.mpl_connect(
            "close_event", lambda evt: plt.close("all"))

    def _update_status(self) -> None:
        if self._role == "target":
            msg = ("[1/2] Drawing TARGET polygon on right panel: click "
                   "vertices, close by clicking first vertex (or Esc to "
                   "cancel current polygon).")
        elif self._role == "background":
            msg = ("[2/2] Drawing BACKGROUND polygon: click vertices, "
                   "close by clicking first vertex.")
        else:
            msg = ("Done. Press 'r' to reset polygons and rescan, or "
                   "'s' to re-save the current results.")
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
        if self._role == "target":
            self._target_verts = verts
            self._target_mask = vert_to_mask(verts, self.shape)
            print(f"[poly] target: {int(self._target_mask.sum())} pixels")
            self._draw_polygon_overlay(verts, "TARGET",
                                        ROI_COLORS["target"])
            try:
                self._ps.disconnect_events()
            except Exception:
                pass
            self._role = "background"
            self._init_polygon_selector()
            self._update_status()
        elif self._role == "background":
            self._background_verts = verts
            self._background_mask = vert_to_mask(verts, self.shape)
            n = int(self._background_mask.sum())
            print(f"[poly] background: {n} pixels")
            self._draw_polygon_overlay(verts, "BACKGROUND",
                                        ROI_COLORS["background"])
            try:
                self._ps.disconnect_events()
            except Exception:
                pass
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
        """Compute BD map for the given center and return mean over each ROI."""
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

    def _continuum_remove_2026(self, spec: np.ndarray
                                 ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Continuum-remove `spec` over [SPEC_LO, SPEC_HI] via a linear
        interpolation between the wavelength-mean reflectances at the
        two endpoints. Returns (window_wav, R_cr, R_lo, R_hi)."""
        win = (self.wav >= SPEC_LO - 0.005) & (self.wav <= SPEC_HI + 0.005)
        w = self.wav[win]
        r = spec[win]
        # 5-channel mean at each shoulder
        idx_lo = int(np.argmin(np.abs(self.wav - SPEC_LO)))
        idx_hi = int(np.argmin(np.abs(self.wav - SPEC_HI)))
        hw = 2
        with np.errstate(invalid="ignore"):
            r_lo = float(np.nanmean(spec[max(0, idx_lo - hw):idx_lo + hw + 1]))
            r_hi = float(np.nanmean(spec[max(0, idx_hi - hw):idx_hi + hw + 1]))
        cont = r_lo + (r_hi - r_lo) * (w - SPEC_LO) / (SPEC_HI - SPEC_LO)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_cr = np.where(cont != 0, r / cont, np.nan)
        return w, r_cr, r_lo, r_hi

    def _run_scan_and_save(self) -> None:
        if self._target_mask is None or self._background_mask is None:
            return
        n_t = int(self._target_mask.sum())
        n_b = int(self._background_mask.sum())
        print(f"[scan] target n={n_t}  background n={n_b}")

        # 1. Continuum-removed mean spectra (sanity check)
        with np.errstate(invalid="ignore"):
            spec_t = np.nanmean(self.cube[:, self._target_mask], axis=1)
            spec_b = np.nanmean(self.cube[:, self._background_mask], axis=1)
        w_t, r_cr_t, r_lo_t, r_hi_t = self._continuum_remove_2026(spec_t)
        w_b, r_cr_b, r_lo_b, r_hi_b = self._continuum_remove_2026(spec_b)
        # Find band minima (argmin of continuum-removed reflectance)
        min_idx_t = int(np.nanargmin(r_cr_t))
        min_idx_b = int(np.nanargmin(r_cr_b))
        lam_min_t = float(w_t[min_idx_t])
        lam_min_b = float(w_b[min_idx_b])
        print(f"[scan] target band minimum at "
              f"{lam_min_t:.4f} um (R_cr = {r_cr_t[min_idx_t]:.4f})")
        print(f"[scan] background band minimum at "
              f"{lam_min_b:.4f} um (R_cr = {r_cr_b[min_idx_b]:.4f})")

        # 2. BD center sweep
        bd_t = np.empty_like(SCAN_CENTERS, dtype=np.float64)
        bd_b = np.empty_like(SCAN_CENTERS, dtype=np.float64)
        contrast = np.empty_like(SCAN_CENTERS, dtype=np.float64)
        for i, c in enumerate(SCAN_CENTERS):
            bdt, bdb, dlt = self._mean_bd_for_center(
                float(c), self._target_mask, self._background_mask)
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

        # 3. Plot
        self._plot_results(
            w_t, r_cr_t, w_b, r_cr_b, lam_min_t, lam_min_b,
            r_cr_t[min_idx_t], r_cr_b[min_idx_b],
            bd_t, bd_b, contrast, peak_lambda, peak_contrast,
        )

        # 4. Save
        outdir = self._save_outputs(
            spec_t, spec_b, w_t, r_cr_t, w_b, r_cr_b,
            lam_min_t, lam_min_b,
            bd_t, bd_b, contrast,
            peak_lambda, peak_contrast, n_t, n_b,
        )
        print(f"[save] -> {outdir}")
        self._scan_done = True
        self._update_status()

    # ---- Plotting ----
    def _plot_results(self, w_t, r_cr_t, w_b, r_cr_b,
                       lam_min_t, lam_min_b, rcr_min_t, rcr_min_b,
                       bd_t, bd_b, contrast,
                       peak_lambda, peak_contrast) -> None:
        # Spectra panel
        self.ax_spectra.clear()
        self.ax_spectra.set_xlabel(r"wavelength ($\mu$m)")
        self.ax_spectra.set_ylabel("continuum-removed reflectance")
        self.ax_spectra.set_xlim(SPEC_LO, SPEC_HI)
        self.ax_spectra.axhline(1.0, color="black", lw=0.5, alpha=0.4)
        # Reference vertical lines
        for w_ref, lbl in [(2.30, "2.30"), (2.33, "2.33"),
                            (2.40, "2.40"), (2.50, "2.50")]:
            self.ax_spectra.axvline(w_ref, color="#aa4444", lw=0.5,
                                      ls="--", alpha=0.4, zorder=0)
        self.ax_spectra.plot(w_t, r_cr_t, color=ROI_COLORS["target"],
                              lw=1.8, label="target")
        self.ax_spectra.plot(w_b, r_cr_b, color=ROI_COLORS["background"],
                              lw=1.8, label="background")
        # Annotate band minima
        self.ax_spectra.scatter([lam_min_t], [rcr_min_t],
                                  color=ROI_COLORS["target"],
                                  s=80, zorder=5,
                                  edgecolor="black", linewidth=0.6)
        self.ax_spectra.annotate(
            f"target min\n{lam_min_t:.4f}", xy=(lam_min_t, rcr_min_t),
            xytext=(8, -12), textcoords="offset points",
            color=ROI_COLORS["target"], fontsize=9,
            arrowprops=dict(arrowstyle="-", color=ROI_COLORS["target"],
                              lw=0.6))
        self.ax_spectra.scatter([lam_min_b], [rcr_min_b],
                                  color=ROI_COLORS["background"],
                                  s=80, zorder=5,
                                  edgecolor="black", linewidth=0.6)
        self.ax_spectra.annotate(
            f"background min\n{lam_min_b:.4f}", xy=(lam_min_b, rcr_min_b),
            xytext=(8, 14), textcoords="offset points",
            color=ROI_COLORS["background"], fontsize=9,
            arrowprops=dict(arrowstyle="-", color=ROI_COLORS["background"],
                              lw=0.6))
        self.ax_spectra.set_title(
            "Continuum-removed mean spectra "
            f"(2.0-2.6 um shoulders; target n={int(self._target_mask.sum())}, "
            f"background n={int(self._background_mask.sum())})",
            fontsize=11)
        self.ax_spectra.legend(fontsize=10, loc="lower right")
        self.ax_spectra.grid(True, alpha=0.3)

        # Contrast curve panel
        self.ax_curve.clear()
        self.ax_curve.set_xlabel(r"BD center wavelength ($\mu$m)")
        self.ax_curve.set_ylabel(r"contrast = mean(BD)$_T$ - mean(BD)$_B$")
        self.ax_curve.axhline(0.0, color="black", lw=0.5, alpha=0.4)
        self.ax_curve.plot(SCAN_CENTERS, contrast, "-o",
                             color="#1f77b4", lw=1.6, ms=4,
                             label="contrast")
        # Peak
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

    # ---- Saving ----
    def _save_outputs(self, spec_t, spec_b, w_t, r_cr_t, w_b, r_cr_b,
                       lam_min_t, lam_min_b,
                       bd_t, bd_b, contrast,
                       peak_lambda, peak_contrast,
                       n_t, n_b) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        outdir = (SAMPLES_ROOT / self.cube_id / "contrast_scan" / ts)
        outdir.mkdir(parents=True, exist_ok=True)

        # Polygons
        poly_payload = {
            "cube_id": self.cube_id,
            "session_ts": self.session_ts,
            "saved_ts": ts,
            "target_vertices_xy": self._target_verts,
            "background_vertices_xy": self._background_verts,
            "shape_HW": list(self.shape),
            "display_orientation": ("flipped vertically AND horizontally "
                                     "vs detector; coords above are in "
                                     "display orientation"),
        }
        (outdir / "polygons.json").write_text(
            json.dumps(poly_payload, indent=2), encoding="utf-8")

        # Masks (un-flipped to detector orientation on disk).
        np.save(outdir / "target_mask.npy",
                self._target_mask[::-1, ::-1])
        np.save(outdir / "background_mask.npy",
                self._background_mask[::-1, ::-1])

        # Mean spectra CSV (full wavelength range; continuum-removed in window)
        with open(outdir / "mean_spectra.csv", "w", newline="") as f:
            w_csv = csv.writer(f)
            w_csv.writerow(["wavelength_um",
                             "target_R", "background_R",
                             "target_R_cr_2p0_2p6",
                             "background_R_cr_2p0_2p6"])
            r_cr_t_full = np.full_like(self.wav, np.nan, dtype=np.float64)
            r_cr_b_full = np.full_like(self.wav, np.nan, dtype=np.float64)
            mask_window = ((self.wav >= SPEC_LO - 0.005)
                            & (self.wav <= SPEC_HI + 0.005))
            r_cr_t_full[mask_window] = r_cr_t
            r_cr_b_full[mask_window] = r_cr_b
            for i in range(len(self.wav)):
                w_csv.writerow([
                    f"{self.wav[i]:.6f}",
                    f"{spec_t[i]:.6f}", f"{spec_b[i]:.6f}",
                    f"{r_cr_t_full[i]:.6f}" if np.isfinite(r_cr_t_full[i]) else "",
                    f"{r_cr_b_full[i]:.6f}" if np.isfinite(r_cr_b_full[i]) else "",
                ])

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
        summary = {
            "cube_id": self.cube_id,
            "session_ts": self.session_ts,
            "saved_ts": ts,
            "n_target_pixels": int(n_t),
            "n_background_pixels": int(n_b),
            "scan": {
                "centers_um": [float(c) for c in SCAN_CENTERS],
                "lo_anchor_um": SCAN_LO_ANCHOR,
                "hi_anchor_um": SCAN_HI_ANCHOR,
                "kernel": SCAN_KERNEL,
            },
            "spec_continuum_window_um": [SPEC_LO, SPEC_HI],
            "band_minimum_target_um": float(lam_min_t),
            "band_minimum_background_um": float(lam_min_b),
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

        # Screenshots
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
        self._target_verts = self._target_mask = None
        self._background_verts = self._background_mask = None
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
            "Continuum-removed mean spectra (target vs background) - "
            "draw both polygons to populate", fontsize=11)
        self.ax_curve.set_title(
            "BD-center contrast scan - populates after both polygons drawn",
            fontsize=11)
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
            # Re-run scan and save in a fresh timestamped dir.
            self._run_scan_and_save()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cube", nargs="?", default="FRT00003E12",
                        help="cube id (full FRT0000XXXX or short like 871C)")
    args = parser.parse_args()
    cube_id = resolve_cube_id(args.cube)
    print(f"=== v9.3 Contrast-Scan Tool: {cube_id} ===")
    print(f"backend: {matplotlib.get_backend()}")
    tool = ContrastScanTool(cube_id)
    plt.show()


if __name__ == "__main__":
    main()
