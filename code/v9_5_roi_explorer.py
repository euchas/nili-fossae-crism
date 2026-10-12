"""v9.5 ROI Contrast-Scan Tool with three contrast variants + R_chlor panel.

Same workflow as v9.4 (CAR + BD2330/OLINDEX3/BD1900_2 RGB panels +
R_chlor heatmap; three polygons: target, background, denominator;
4-curve continuum-removed spectrum plot) but the contrast scan now computes THREE
variants of the discrimination metric:

  plain       contrast(lam) = BD_T(lam) - BD_B(lam)
              -- combines depth + position differences.

  normalized  contrast(lam) = BD_T(lam)/max(BD_T)
                                  - BD_B(lam)/max(BD_B)
              -- scales each ROI's BD curve to its own peak,
                 isolating band-position separation. Zero crossing
                 localizes the position offset.

  ratio       contrast(lam) = BD_T(lam)/BD_T(2.30)
                                  - BD_B(lam)/BD_B(2.30)
              -- scales by the BD at the 2.30 um anchor (first scan
                 center), measuring band shape relative to a fixed
                 wavelength reference.

Analysis figure now has THREE panels (top to bottom):
  1. Continuum-removed mean spectra (4 curves, as in v9.4).
  2. Three contrast variants overlaid (single panel) with peak
     |contrast| annotated for each variant.
  3. Raw BD_T(lam) and BD_B(lam) curves (not differenced) so the
     per-ROI band-depth peaks are directly visible.

After the scan, a summary table is printed to stdout:

  variant         peak_lambda (um)   contrast at peak
  plain           ...                ...
  normalized      ...                ...
  ratio           ...                ...

Outputs (in the same place as v9.4):
  outputs/spectra sample locations/<cube>/contrast_scan/<TS>/
    polygons.json, target/background/denominator_mask.npy,
    mean_spectra.csv,
    scan_results.csv      now: bd_center_um, target_bd_mean,
                          background_bd_mean, contrast_plain,
                          contrast_normalized, contrast_ratio
    summary.json          now includes per-variant peak info
    viewer.png, plots.png

Keys: r reset, s re-save, esc cancel polygon being drawn.

Run:
  python code/v9_5_roi_explorer.py FRT00003E12
  python code/v9_5_roi_explorer.py 871C
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
    calc_bd1900_2, calc_bd2330, calc_r_chlor,
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
SCAN_RATIO_REF_LAMBDA = 2.300   # BD reference for the 'ratio' variant

SPEC_LO = 2.00
SPEC_HI = 2.60

# R_chlor heatmap (third viewer panel; 2-98 % percentile stretch).
RCHLOR_CMAP = "viridis"
RCHLOR_BD_FLOOR = 0.005   # mask R_chlor where BD(2.302) below this

ROI_COLORS = {
    "target":      "#ffd400",
    "background":  "#e7298a",
    "denominator": "#00cccc",
}
CONTRAST_COLORS = {
    "plain":      "#1f77b4",
    "normalized": "#ff7f0e",
    "ratio":      "#2ca02c",
}
CONTRAST_LINESTYLES = {
    "plain":      "-",
    "normalized": "--",
    "ratio":      "-.",
}
VARIANT_ORDER = ("plain", "normalized", "ratio")

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


def safe_argmax_abs(centers: np.ndarray, contrast: np.ndarray
                     ) -> tuple[float, float]:
    """Return (peak_lambda, signed_contrast_at_peak) for the wavelength
    of maximum |contrast|. NaN-safe; returns (nan, nan) if contrast is
    all-NaN."""
    abs_c = np.abs(contrast)
    if not np.isfinite(abs_c).any():
        return float("nan"), float("nan")
    i = int(np.nanargmax(abs_c))
    return float(centers[i]), float(contrast[i])


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

    # ---- Load ----
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

        # R_chlor panel: BD(2.328) / BD(2.302) ratio with floor mask.
        print("  computing R_chlor = BD(2.328) / BD(2.302) ...")
        self._rchlor = calc_r_chlor(self.cube, self.wav,
                                     bd_floor=RCHLOR_BD_FLOOR)
        rch_finite = self._rchlor[np.isfinite(self._rchlor)]
        if rch_finite.size > 10:
            self._rchlor_clim = tuple(float(v)
                                        for v in np.percentile(rch_finite, [2, 98]))
        else:
            self._rchlor_clim = (0.0, 1.0)
        print(f"  R_chlor p2/p98 = {self._rchlor_clim[0]:.4f}, "
              f"{self._rchlor_clim[1]:.4f}  "
              f"(masked {(~np.isfinite(self._rchlor)).sum()} px)")

        self._R_lo_anchor = kernel(self.cube, self.wav,
                                     SCAN_LO_ANCHOR, SCAN_KERNEL)
        self._R_hi_anchor = kernel(self.cube, self.wav,
                                     SCAN_HI_ANCHOR, SCAN_KERNEL)
        print("  done.")

    # ---- Figures ----
    def _build_figures(self) -> None:
        plt.rcdefaults()
        plt.rcParams.update({
            "axes.titlesize": 12, "axes.labelsize": 11,
            "legend.fontsize": 9, "xtick.labelsize": 10,
            "ytick.labelsize": 10, "figure.titlesize": 12,
        })
        # Window 1: viewer (CAR + BD2330 RGB + R_chlor)
        self.fig_viewer = plt.figure(
            figsize=(20, 6.5), num=f"v9.5 Viewer - {self.cube_id}")
        gs = self.fig_viewer.add_gridspec(
            1, 3, wspace=0.06,
            left=0.02, right=0.99, top=0.92, bottom=0.13,
        )
        self.ax_car = self.fig_viewer.add_subplot(gs[0, 0])
        self.ax_panel_right = self.fig_viewer.add_subplot(
            gs[0, 1], sharex=self.ax_car, sharey=self.ax_car)
        self.ax_rchlor = self.fig_viewer.add_subplot(
            gs[0, 2], sharex=self.ax_car, sharey=self.ax_car)
        self.ax_car.imshow(self._car_rgb, interpolation="nearest")
        self.ax_panel_right.imshow(self._panel_right_rgb,
                                    interpolation="nearest")
        rch_lo, rch_hi = self._rchlor_clim
        self.ax_rchlor.imshow(self._rchlor, cmap=RCHLOR_CMAP,
                                vmin=rch_lo, vmax=rch_hi,
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
        self.ax_rchlor.set_title(
            f"R_chlor = BD(2.328)/BD(2.302)  "
            f"[BD(2.302) >= {RCHLOR_BD_FLOOR}]  "
            f"clim {rch_lo:.2f} - {rch_hi:.2f}",
            fontsize=11, fontweight="bold")
        for ax in (self.ax_car, self.ax_panel_right, self.ax_rchlor):
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

        # Window 2: 3-row analysis (spectra | contrast variants | raw BD)
        self.fig_plots = plt.figure(
            figsize=(11, 11.5), num=f"v9.5 Analysis - {self.cube_id}")
        gs2 = self.fig_plots.add_gridspec(
            3, 1, height_ratios=[1.3, 1.1, 0.9], hspace=0.42,
            left=0.10, right=0.97, top=0.96, bottom=0.06,
        )
        self.ax_spectra = self.fig_plots.add_subplot(gs2[0, 0])
        self.ax_contrast = self.fig_plots.add_subplot(gs2[1, 0])
        self.ax_rawbd = self.fig_plots.add_subplot(
            gs2[2, 0], sharex=self.ax_contrast)

        # Spectra panel placeholders
        self.ax_spectra.set_xlabel(r"wavelength ($\mu$m)")
        self.ax_spectra.set_ylabel("continuum-removed reflectance")
        self.ax_spectra.set_xlim(SPEC_LO, SPEC_HI)
        self.ax_spectra.set_title(
            "Continuum-removed mean spectra "
            "(target / background / target-over-denom / "
            "background-over-denom) - draw all three polygons to populate",
            fontsize=11)
        # Contrast panel placeholder
        self.ax_contrast.set_xlabel(r"BD center wavelength ($\mu$m)")
        self.ax_contrast.set_ylabel("contrast")
        self.ax_contrast.set_xlim(SCAN_CENTERS[0] - 0.002,
                                    SCAN_CENTERS[-1] + 0.002)
        self.ax_contrast.set_title(
            "BD-center contrast scan - 3 variants (plain / normalized / "
            "ratio); populates after scan", fontsize=11)
        # Raw BD panel placeholder
        self.ax_rawbd.set_xlabel(r"BD center wavelength ($\mu$m)")
        self.ax_rawbd.set_ylabel(r"mean BD")
        self.ax_rawbd.set_title(
            "Raw per-ROI BD response (target vs background) - shows "
            "individual band-depth peaks", fontsize=11)
        for a in (self.ax_spectra, self.ax_contrast, self.ax_rawbd):
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
        for ax in (self.ax_car, self.ax_panel_right, self.ax_rchlor):
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
                              ) -> tuple[float, float]:
        a = (SCAN_HI_ANCHOR - center) / (SCAN_HI_ANCHOR - SCAN_LO_ANCHOR)
        b = (center - SCAN_LO_ANCHOR) / (SCAN_HI_ANCHOR - SCAN_LO_ANCHOR)
        cont = a * self._R_lo_anchor + b * self._R_hi_anchor
        R_c = kernel(self.cube, self.wav, center, SCAN_KERNEL)
        with np.errstate(divide="ignore", invalid="ignore"):
            bd = np.where(cont != 0, 1.0 - R_c / cont, np.nan)
        with np.errstate(invalid="ignore"):
            bd_t = float(np.nanmean(bd[mask_t]))
            bd_b = float(np.nanmean(bd[mask_b]))
        return bd_t, bd_b

    def _run_scan_and_save(self) -> None:
        m_t = self._masks.get("target")
        m_b = self._masks.get("background")
        m_d = self._masks.get("denominator")
        if m_t is None or m_b is None or m_d is None:
            return
        n_t = int(m_t.sum()); n_b = int(m_b.sum()); n_d = int(m_d.sum())
        print(f"[scan] target n={n_t}  background n={n_b}  "
              f"denominator n={n_d}")

        # 1. Mean spectra
        with np.errstate(invalid="ignore"):
            spec_t = np.nanmean(self.cube[:, m_t], axis=1)
            spec_b = np.nanmean(self.cube[:, m_b], axis=1)
            spec_d = np.nanmean(self.cube[:, m_d], axis=1)
        # 2. Continuum-remove raw + ratioed spectra
        w_t, r_cr_t, *_ = continuum_remove(spec_t, self.wav, SPEC_LO, SPEC_HI)
        w_b, r_cr_b, *_ = continuum_remove(spec_b, self.wav, SPEC_LO, SPEC_HI)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_td = np.where(np.abs(spec_d) > 1e-6,
                                 spec_t / spec_d, np.nan)
            ratio_bd = np.where(np.abs(spec_d) > 1e-6,
                                 spec_b / spec_d, np.nan)
        w_td, r_cr_td, *_ = continuum_remove(ratio_td, self.wav,
                                              SPEC_LO, SPEC_HI)
        w_bd, r_cr_bd, *_ = continuum_remove(ratio_bd, self.wav,
                                              SPEC_LO, SPEC_HI)

        # 3. Band minima
        def argmin_safe(arr):
            return int(np.nanargmin(arr)) if np.isfinite(arr).any() else None
        idxs = (argmin_safe(r_cr_t), argmin_safe(r_cr_b),
                argmin_safe(r_cr_td), argmin_safe(r_cr_bd))
        lams = []
        rcrs = []
        for arr_w, arr_r, i in zip(
                (w_t, w_b, w_td, w_bd),
                (r_cr_t, r_cr_b, r_cr_td, r_cr_bd), idxs):
            lams.append(float(arr_w[i]) if i is not None else float("nan"))
            rcrs.append(float(arr_r[i]) if i is not None else float("nan"))
        lam_t, lam_b, lam_td, lam_bd = lams
        rcr_t, rcr_b, rcr_td, rcr_bd = rcrs

        # 4. BD-center sweep
        bd_t = np.empty_like(SCAN_CENTERS, dtype=np.float64)
        bd_b = np.empty_like(SCAN_CENTERS, dtype=np.float64)
        for i, c in enumerate(SCAN_CENTERS):
            bd_t[i], bd_b[i] = self._mean_bd_for_center(float(c), m_t, m_b)

        # 5. Three contrast variants
        contrast_plain = bd_t - bd_b

        # Normalized: each curve scaled to its own peak BD over the scan.
        with np.errstate(divide="ignore", invalid="ignore"):
            t_max = float(np.nanmax(bd_t)) if np.isfinite(bd_t).any() else np.nan
            b_max = float(np.nanmax(bd_b)) if np.isfinite(bd_b).any() else np.nan
            bd_t_norm = (bd_t / t_max) if (np.isfinite(t_max)
                                              and abs(t_max) > 1e-9) \
                          else np.full_like(bd_t, np.nan)
            bd_b_norm = (bd_b / b_max) if (np.isfinite(b_max)
                                              and abs(b_max) > 1e-9) \
                          else np.full_like(bd_b, np.nan)
        contrast_normalized = bd_t_norm - bd_b_norm

        # Ratio: each curve divided by its BD value at the 2.30 um anchor.
        ref_idx = int(np.argmin(np.abs(SCAN_CENTERS - SCAN_RATIO_REF_LAMBDA)))
        bd_t_ref = float(bd_t[ref_idx])
        bd_b_ref = float(bd_b[ref_idx])
        with np.errstate(divide="ignore", invalid="ignore"):
            bd_t_ratio = (bd_t / bd_t_ref) if abs(bd_t_ref) > 1e-9 \
                          else np.full_like(bd_t, np.nan)
            bd_b_ratio = (bd_b / bd_b_ref) if abs(bd_b_ref) > 1e-9 \
                          else np.full_like(bd_b, np.nan)
        contrast_ratio = bd_t_ratio - bd_b_ratio

        contrasts = {
            "plain":      contrast_plain,
            "normalized": contrast_normalized,
            "ratio":      contrast_ratio,
        }

        # 6. Peak |contrast| per variant + summary table to stdout
        peaks = {
            v: safe_argmax_abs(SCAN_CENTERS, contrasts[v])
            for v in VARIANT_ORDER
        }
        print()
        print(f"  variant         peak_lambda (um)   contrast at peak")
        for v in VARIANT_ORDER:
            lam_p, c_p = peaks[v]
            print(f"  {v:<14s}  {lam_p:>16.4f}   {c_p:>+16.6f}")
        print()
        print(f"[scan] BD_T peak: lambda = "
              f"{SCAN_CENTERS[int(np.nanargmax(bd_t))]:.4f} um  "
              f"BD = {np.nanmax(bd_t):.4f}")
        print(f"[scan] BD_B peak: lambda = "
              f"{SCAN_CENTERS[int(np.nanargmax(bd_b))]:.4f} um  "
              f"BD = {np.nanmax(bd_b):.4f}")
        print(f"[scan] BD_T(2.30) = {bd_t_ref:.4f}   "
              f"BD_B(2.30) = {bd_b_ref:.4f}")

        # 7. Plot + save
        self._plot_results(
            w_t, r_cr_t, w_b, r_cr_b, w_td, r_cr_td, w_bd, r_cr_bd,
            (lam_t, rcr_t), (lam_b, rcr_b),
            (lam_td, rcr_td), (lam_bd, rcr_bd),
            bd_t, bd_b, contrasts, peaks,
        )
        outdir = self._save_outputs(
            spec_t, spec_b, spec_d,
            ratio_td, ratio_bd,
            w_t, r_cr_t, w_b, r_cr_b, w_td, r_cr_td, w_bd, r_cr_bd,
            (lam_t, lam_b, lam_td, lam_bd),
            bd_t, bd_b, contrasts, peaks,
            n_t, n_b, n_d,
        )
        print(f"[save] -> {outdir}")
        self._scan_done = True
        self._update_status()

    # ---- Plot ----
    def _plot_results(self,
                       w_t, r_cr_t, w_b, r_cr_b,
                       w_td, r_cr_td, w_bd, r_cr_bd,
                       min_t, min_b, min_td, min_bd,
                       bd_t, bd_b, contrasts, peaks) -> None:
        # ---- Spectra panel ----
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
        self.ax_spectra.plot(w_t, r_cr_t, color=c_t, lw=1.4, alpha=0.85,
                              ls="-",
                              label=f"target  (min {min_t[0]:.4f})")
        self.ax_spectra.plot(w_b, r_cr_b, color=c_b, lw=1.4, alpha=0.85,
                              ls="-",
                              label=f"background  (min {min_b[0]:.4f})")
        self.ax_spectra.plot(w_td, r_cr_td, color=c_t, lw=2.2, ls="--",
                              label=f"target / denom  "
                                    f"(min {min_td[0]:.4f})")
        self.ax_spectra.plot(w_bd, r_cr_bd, color=c_b, lw=2.2, ls="--",
                              label=f"background / denom  "
                                    f"(min {min_bd[0]:.4f})")
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

        # ---- Contrast variants panel ----
        self.ax_contrast.clear()
        self.ax_contrast.set_xlabel(r"BD center wavelength ($\mu$m)")
        self.ax_contrast.set_ylabel("contrast")
        self.ax_contrast.set_xlim(SCAN_CENTERS[0] - 0.002,
                                    SCAN_CENTERS[-1] + 0.002)
        self.ax_contrast.axhline(0.0, color="black", lw=0.5, alpha=0.4)
        for v in VARIANT_ORDER:
            color = CONTRAST_COLORS[v]
            ls = CONTRAST_LINESTYLES[v]
            lam_p, c_p = peaks[v]
            self.ax_contrast.plot(
                SCAN_CENTERS, contrasts[v],
                color=color, lw=2.0, ls=ls, marker="o", ms=3.5,
                label=(f"{v}  (peak {lam_p:.4f}, "
                       f"contrast {c_p:+.4f})"),
            )
            if np.isfinite(lam_p) and np.isfinite(c_p):
                self.ax_contrast.axvline(lam_p, color=color, ls=":",
                                           lw=0.9, alpha=0.65)
                self.ax_contrast.scatter([lam_p], [c_p], color=color,
                                           s=110, zorder=5,
                                           edgecolor="black", linewidth=0.7)
        self.ax_contrast.set_title(
            f"Three contrast variants  "
            f"(centers {SCAN_CENTERS[0]:.3f}-{SCAN_CENTERS[-1]:.3f} um, "
            f"step {(SCAN_CENTERS[1]-SCAN_CENTERS[0])*1000:.0f} nm; "
            f"anchors {SCAN_LO_ANCHOR:.3f}/{SCAN_HI_ANCHOR:.3f}, "
            f"kernel {SCAN_KERNEL})",
            fontsize=10)
        self.ax_contrast.legend(fontsize=9, loc="best", framealpha=0.92)
        self.ax_contrast.grid(True, alpha=0.3)

        # ---- Raw BD panel ----
        self.ax_rawbd.clear()
        self.ax_rawbd.set_xlabel(r"BD center wavelength ($\mu$m)")
        self.ax_rawbd.set_ylabel("mean BD")
        self.ax_rawbd.set_xlim(SCAN_CENTERS[0] - 0.002,
                                 SCAN_CENTERS[-1] + 0.002)
        self.ax_rawbd.axhline(0.0, color="black", lw=0.5, alpha=0.4)
        # Per-ROI peaks
        if np.isfinite(bd_t).any():
            i_t = int(np.nanargmax(bd_t))
            lam_t_peak = float(SCAN_CENTERS[i_t])
            bd_t_peak = float(bd_t[i_t])
        else:
            lam_t_peak, bd_t_peak = float("nan"), float("nan")
        if np.isfinite(bd_b).any():
            i_b = int(np.nanargmax(bd_b))
            lam_b_peak = float(SCAN_CENTERS[i_b])
            bd_b_peak = float(bd_b[i_b])
        else:
            lam_b_peak, bd_b_peak = float("nan"), float("nan")
        self.ax_rawbd.plot(
            SCAN_CENTERS, bd_t, color=ROI_COLORS["target"],
            lw=2.0, marker="o", ms=3.5,
            label=(f"target BD  (peak {lam_t_peak:.4f}, "
                   f"max {bd_t_peak:.4f})"))
        self.ax_rawbd.plot(
            SCAN_CENTERS, bd_b, color=ROI_COLORS["background"],
            lw=2.0, marker="o", ms=3.5,
            label=(f"background BD  (peak {lam_b_peak:.4f}, "
                   f"max {bd_b_peak:.4f})"))
        if np.isfinite(lam_t_peak):
            self.ax_rawbd.scatter([lam_t_peak], [bd_t_peak],
                                    color=ROI_COLORS["target"], s=110,
                                    edgecolor="black", linewidth=0.7,
                                    zorder=5)
            self.ax_rawbd.axvline(lam_t_peak, color=ROI_COLORS["target"],
                                    ls=":", lw=0.7, alpha=0.55)
        if np.isfinite(lam_b_peak):
            self.ax_rawbd.scatter([lam_b_peak], [bd_b_peak],
                                    color=ROI_COLORS["background"], s=110,
                                    edgecolor="black", linewidth=0.7,
                                    zorder=5)
            self.ax_rawbd.axvline(lam_b_peak,
                                    color=ROI_COLORS["background"],
                                    ls=":", lw=0.7, alpha=0.55)
        self.ax_rawbd.set_title(
            "Raw per-ROI BD response  (target vs background, "
            "not differenced)", fontsize=10)
        self.ax_rawbd.legend(fontsize=9, loc="best", framealpha=0.92)
        self.ax_rawbd.grid(True, alpha=0.3)

        self.fig_plots.canvas.draw_idle()

    # ---- Save ----
    def _save_outputs(self,
                       spec_t, spec_b, spec_d,
                       ratio_td, ratio_bd,
                       w_t, r_cr_t, w_b, r_cr_b,
                       w_td, r_cr_td, w_bd, r_cr_bd,
                       lams,
                       bd_t, bd_b, contrasts, peaks,
                       n_t, n_b, n_d) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        outdir = (SAMPLES_ROOT / self.cube_id / "contrast_scan" / ts)
        outdir.mkdir(parents=True, exist_ok=True)

        poly_payload = {
            "cube_id": self.cube_id,
            "session_ts": self.session_ts,
            "saved_ts": ts,
            "tool_version": "v9.5",
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

        np.save(outdir / "target_mask.npy",
                self._masks["target"][::-1, ::-1])
        np.save(outdir / "background_mask.npy",
                self._masks["background"][::-1, ::-1])
        np.save(outdir / "denominator_mask.npy",
                self._masks["denominator"][::-1, ::-1])

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
                w_csv.writerow([
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
                ])

        # Scan results CSV (extended schema)
        with open(outdir / "scan_results.csv", "w", newline="") as f:
            w_csv = csv.writer(f)
            w_csv.writerow([
                "bd_center_um", "target_bd_mean", "background_bd_mean",
                "contrast_plain", "contrast_normalized", "contrast_ratio",
            ])
            for i, c in enumerate(SCAN_CENTERS):
                w_csv.writerow([
                    f"{c:.4f}",
                    f"{bd_t[i]:.6f}", f"{bd_b[i]:.6f}",
                    f"{contrasts['plain'][i]:.6f}",
                    f"{contrasts['normalized'][i]:.6f}",
                    f"{contrasts['ratio'][i]:.6f}",
                ])

        # Summary JSON
        lam_t, lam_b, lam_td, lam_bd = lams
        # raw BD peaks
        i_t = int(np.nanargmax(bd_t)) if np.isfinite(bd_t).any() else 0
        i_b = int(np.nanargmax(bd_b)) if np.isfinite(bd_b).any() else 0
        ref_idx = int(np.argmin(np.abs(SCAN_CENTERS - SCAN_RATIO_REF_LAMBDA)))

        peak_payload = {}
        for v in VARIANT_ORDER:
            lam_p, c_p = peaks[v]
            peak_payload[v] = {
                "bd_center_um": lam_p,
                "contrast": c_p,
                "abs_contrast": float(abs(c_p)) if np.isfinite(c_p) else float("nan"),
            }

        # R_chlor stats per ROI (target / background / denominator),
        # computed within each polygon mask (NaN-safe mean).
        rchlor_stats = {}
        for role, mask in (("target", self._masks["target"]),
                            ("background", self._masks["background"]),
                            ("denominator", self._masks["denominator"])):
            sub = self._rchlor[mask]
            with np.errstate(invalid="ignore"):
                rchlor_stats[role] = {
                    "mean": float(np.nanmean(sub))
                              if np.isfinite(sub).any() else float("nan"),
                    "median": float(np.nanmedian(sub))
                              if np.isfinite(sub).any() else float("nan"),
                    "std": float(np.nanstd(sub))
                              if np.isfinite(sub).any() else float("nan"),
                    "n_finite": int(np.isfinite(sub).sum()),
                    "n_pixels": int(mask.sum()),
                }

        summary = {
            "cube_id": self.cube_id,
            "session_ts": self.session_ts,
            "saved_ts": ts,
            "tool_version": "v9.5",
            "n_target_pixels": int(n_t),
            "n_background_pixels": int(n_b),
            "n_denominator_pixels": int(n_d),
            "scan": {
                "centers_um": [float(c) for c in SCAN_CENTERS],
                "lo_anchor_um": SCAN_LO_ANCHOR,
                "hi_anchor_um": SCAN_HI_ANCHOR,
                "kernel": SCAN_KERNEL,
                "ratio_ref_lambda_um": SCAN_RATIO_REF_LAMBDA,
            },
            "spec_continuum_window_um": [SPEC_LO, SPEC_HI],
            "band_minima_um": {
                "target":           lam_t,
                "background":       lam_b,
                "target_over_denom":     lam_td,
                "background_over_denom": lam_bd,
            },
            "raw_bd_peaks": {
                "target":     {"bd_center_um": float(SCAN_CENTERS[i_t]),
                                "bd": float(bd_t[i_t])},
                "background": {"bd_center_um": float(SCAN_CENTERS[i_b]),
                                "bd": float(bd_b[i_b])},
                "target_at_2.30":     float(bd_t[ref_idx]),
                "background_at_2.30": float(bd_b[ref_idx]),
            },
            "peak_contrast_per_variant": peak_payload,
            "r_chlor": {
                "definition": ("BD(2.328) / BD(2.302), "
                                f"masked where BD(2.302) < {RCHLOR_BD_FLOOR}"),
                "scene_p2_p98": list(self._rchlor_clim),
                "per_roi_stats": rchlor_stats,
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
        for ax in (self.ax_spectra, self.ax_contrast, self.ax_rawbd):
            ax.clear()
        self.ax_spectra.set_title(
            "Continuum-removed mean spectra - draw all three "
            "polygons to populate", fontsize=11)
        self.ax_contrast.set_title(
            "Three contrast variants - populates after scan",
            fontsize=11)
        self.ax_rawbd.set_title(
            "Raw per-ROI BD response", fontsize=11)
        for a in (self.ax_spectra, self.ax_contrast, self.ax_rawbd):
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
    print(f"=== v9.5 Contrast-Scan Tool (3 variants): {cube_id} ===")
    print(f"backend: {matplotlib.get_backend()}")
    tool = ContrastScanTool(cube_id)
    plt.show()


if __name__ == "__main__":
    main()
