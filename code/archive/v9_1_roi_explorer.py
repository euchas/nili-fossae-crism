"""v9.1 ROI Explorer - CAR + custom VB RGB + OLINDEX3 + Brown RGB.

Replaces code/v6_roi_explorer.py for the v9.1 workflow. Differences
from v6:
  - Viewer is FOUR panels: CAR composite (carbonates) + a configurable
    second VB RGB (default olv-fe-hyd: R=OLINDEX3 G=D2300 B=BD1900R2)
    + OLINDEX3 (single-band inferno) + Brown RGB (GMM panel removed;
    cluster context is not needed at ROI-pick time).
  - Rectangle / pixel mode toggle (key 'm'). In pixel mode each left-
    click toggles a single pixel in/out of the in-progress ROI; press
    'enter' to finalize. In rect mode drag a rectangle (same as v6).
  - Brown scatter shows ONLY the per-target-ROI mean as one labeled
    marker; no full-scene background hexbin (per request).
  - No auto-save while picking. Data is persisted ONLY when the
    session is explicitly saved: either by pressing 's' mid-session
    (manual save) or on window close, where a terminal prompt
    confirms save / discard. Rectangle selectors are wired on every
    panel (CAR, OLINDEX3, Brown RGB) so dragging works on any of
    them; pixel-mode left-clicks also accept any panel. The
    rois.csv accumulates across sessions, so iterating cube-by-cube
    yields one canonical CSV per cube.

Auto-alternation pair convention is unchanged from v6:
    rect/pixel-finalize 1 -> target1   2 -> mafic1   (pair 1)
    rect/pixel-finalize 3 -> target2   4 -> mafic2   (pair 2)
    ...

Outputs (under outputs/spectra sample locations/<cube>/<name>/):
    <name> is a free-text label entered at save time (e.g. carbonates,
    olivine, ridges). Each name has its own folder with its own
    cumulative CSV, so ROIs from different geomorphic / mineralogic
    themes never mix.

    <cube>_<name>_rois.csv                cumulative across all
                                          sessions of this <name>;
                                          one row per ROI; has a
                                          session_ts column. Rows
                                          append; never overwritten.
    sessions/<session_ts>/
      <cube>_spectra.csv                  this session's raw + ratio
      <cube>_meta.json
      <cube>_viewer.png
      <cube>_plots.png
      masks/<cube>_<roi_ts>_<short>.npy   one mask per ROI
                                          (detector orientation).
                                          short = kind + pair_idx.

    Each new launch creates a new sessions/<session_ts>/ subdir under
    whatever <name> the user enters at save; previous sessions remain
    untouched.

Run:
    python code/v9_1_roi_explorer.py FRT00003E12
    python code/v9_1_roi_explorer.py B438       # short form OK

Panel-2 composite (olv-fe-hyd by default; Panel-1 stays as CAR):
    python code/v9_1_roi_explorer.py B438 --composite olv-fe-hyd
    python code/v9_1_roi_explorer.py B438 --composite OLINDEX3,BD1900R2,D2300
    python code/v9_1_roi_explorer.py B438 --composite mafic        # OLINDEX3, LCPINDEX2, HCPINDEX2
    python code/v9_1_roi_explorer.py B438 --composite iron         # BD530_2, SH600_2, BD920_2

Available VB params (from v4 stack): BD530_2, SH600_2, BD920_2,
BDI1000VIS, OLINDEX3, LCPINDEX2, HCPINDEX2, BD1400, BD1900R2, D2300,
BD2500_2.

Keyboard:
    m       toggle rectangle <-> pixel mode (works on every panel)
    enter   finalize pending pixel ROI (pixel mode only)
    u       undo last finalized ROI (or, in pixel mode, last pending pixel)
    c       clear all in-memory ROIs + pending pixels (rois.csv on
             disk is unaffected)
    s       manual save: write masks + rois.csv rows + spectra CSV +
             screenshots + meta JSON for any unsaved ROIs
    r       force re-stretch of viewer panels (zoom-aware)
    Close window: terminal asks save / discard for any unsaved ROIs.
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
from vb_parameters import V4_VB_PARAM_ORDER

# One color per (target, mafic) pair.
PAIR_PALETTE = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e",
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                "#bcbd22", "#17becf"]

# Panel 1 = always CAR (carbonate). Hardcoded.
CAR_CHANNELS = {"R": "D2300", "G": "BD2500_2", "B": "BD1900R2"}

# Panel 2 = a SECOND VB-parameter RGB composite, generated on the fly
# from the cached v4 VB stack. Pick any preset by name on the CLI, or
# override with three comma-separated VB param names.
PANEL2_PRESETS = {
    "olv-fe-hyd": {"R": "OLINDEX3", "G": "D2300",     "B": "BD1900R2"},   # default: olivine + 2.3 + 1.9
    "olv-hyd-fe": {"R": "OLINDEX3", "G": "BD1900R2",  "B": "D2300"},      # olivine + 1.9 + 2.3
    "car":        {"R": "D2300",    "G": "BD2500_2",  "B": "BD1900R2"},   # same as panel 1
    "mafic":      {"R": "OLINDEX3", "G": "LCPINDEX2", "B": "HCPINDEX2"},  # olivine + LCP + HCP
    "iron":       {"R": "BD530_2",  "G": "SH600_2",   "B": "BD920_2"},    # ferric edge
}
DEFAULT_PANEL2_COMPOSITE = "olv-fe-hyd"
# Legacy alias for any external callers
PANEL1_PRESETS = PANEL2_PRESETS

# Single-band olivine map (replacing the NIR false-color panel).
OLINDEX_PARAM = "OLINDEX3"
OLINDEX_CMAP = "inferno"

# Brown RGB stretches: 4-cube-pooled p2/p98 from the brown_rgb work.
BROWN_STRETCH_FULLSCENE = {
    "centroid":  (1.014, 1.379),
    "amplitude": (0.116, 0.335),
    "asymmetry": (0.525, 4.439),
}

# Scatter axis bounds covering the full data span across all 4 cubes.
SCATTER_XLIM = (0.90, 1.50)
SCATTER_YLIM = (0.30, 10.0)

ANNOTATION_BANDS = (1.00, 1.40, 1.90, 2.30, 2.50)
CO2_RESIDUAL = (1.95, 2.05)

V9_1_ROOT = OUTPUTS_DIR / "_v9_1_per_cube_scaler"
V4_ROOT = OUTPUTS_DIR / "_v4_vb_pca_brown_rgb"
SAMPLES_ROOT = OUTPUTS_DIR / "spectra sample locations"


def percentile_norm(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any() or hi - lo < 1e-9:
        out[finite] = 0.5
        return out
    out[finite] = np.clip((arr[finite] - lo) / (hi - lo), 0, 1)
    return out


def annotate_spectrum(ax) -> None:
    ax.axvspan(*CO2_RESIDUAL, color="#cccccc", alpha=0.3, zorder=0)
    for wl in ANNOTATION_BANDS:
        ax.axvline(wl, color="#aa4444", lw=0.8, ls="--", alpha=0.6, zorder=0)


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


# =============================================================================
# Explorer
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


def sanitize_name(name: str) -> str:
    """Filename-safe slug from a user-provided session name."""
    name = (name or "").strip()
    if not name:
        return ""
    return re.sub(r"[^A-Za-z0-9_-]", "_", name.replace(" ", "_")).strip("_")


def parse_composite_arg(arg: str | None) -> dict:
    """Resolve --composite into an {'R':..,'G':..,'B':..} dict for Panel 2.

    Accepts either a preset name (one of PANEL2_PRESETS) or a comma-
    separated triplet of VB param names from V4_VB_PARAM_ORDER, e.g.
    'OLINDEX3,BD1900R2,D2300'."""
    if not arg:
        return dict(PANEL2_PRESETS[DEFAULT_PANEL2_COMPOSITE])
    key = arg.lower()
    if key in PANEL2_PRESETS:
        return dict(PANEL2_PRESETS[key])
    parts = [p.strip() for p in arg.split(",")]
    if len(parts) != 3:
        raise SystemExit(
            f"--composite must be a preset name "
            f"({', '.join(PANEL2_PRESETS)}) or three comma-separated VB "
            f"param names; got {arg!r}"
        )
    available = set(V4_VB_PARAM_ORDER)
    bad = [p for p in parts if p not in available]
    if bad:
        raise SystemExit(
            f"unknown VB param(s) {bad}; available in v4 stack: "
            f"{list(V4_VB_PARAM_ORDER)}"
        )
    return {"R": parts[0], "G": parts[1], "B": parts[2]}


class RoiExplorer:
    def __init__(self, cube_id: str,
                  panel2_channels: dict | None = None):
        self.cube_id = cube_id
        self.session_ts = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ")
        self.rois: list[dict] = []
        self.pending_pixels: list[tuple[int, int]] = []
        self.mode: str = "rect"
        self._pending_artists: list = []
        self._restretch_pending = False
        # Panel 1 is always CAR; panel 2 takes whatever the user passes.
        self._panel1_channels = dict(CAR_CHANNELS)
        self._panel2_channels = (panel2_channels
                                  or dict(PANEL2_PRESETS[DEFAULT_PANEL2_COMPOSITE]))
        # Output paths are NOT created until save (after the user
        # supplies a session name at the close prompt or via 's').
        self.session_name: str | None = None
        self.name_dir: Path | None = None
        self.session_dir: Path | None = None
        self.rois_csv: Path | None = None
        self.spectra_csv: Path | None = None
        self._load_data()
        self._build_figure()

    # ---- ROI bookkeeping ----
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
        # Display flip H + W. The cube is (B, H, W).
        self.wav = wav.astype(np.float32)
        self.cube = cube[:, ::-1, ::-1].copy()
        self.brown = {k: v[::-1, ::-1].copy() for k, v in brown.items()}
        self.shape = self.cube.shape[1:]
        print(f"  cube {self.cube.shape} in {time.time()-t0:.1f}s")

        # Brown valid: every pixel with all 4 Brown params finite.
        self.brown_valid = (np.isfinite(self.brown["centroid"])
                            & np.isfinite(self.brown["amplitude"])
                            & np.isfinite(self.brown["hwhm"])
                            & np.isfinite(self.brown["asymmetry"]))

        # CAR composite: load v4 VB stack, pick 3 channels.
        v4_vb_path = (V4_ROOT / self.cube_id / "arrays"
                      / f"{self.cube_id}_VB_stack_v4.npy")
        if not v4_vb_path.exists():
            raise SystemExit(
                f"missing v4 VB stack at {v4_vb_path}. "
                "Run stage_v4_vb_pca_and_brown_rgb.py first."
            )
        vb_stack = np.load(v4_vb_path)[::-1, ::-1, :].copy()
        vb_idx = {p: i for i, p in enumerate(V4_VB_PARAM_ORDER)}
        # Panel 1: CAR
        self._car_raw = {
            chan: vb_stack[..., vb_idx[self._panel1_channels[chan]]].astype(np.float32)
            for chan in ("R", "G", "B")
        }
        ch1 = self._panel1_channels
        print(f"  Panel-1 (CAR):     R={ch1['R']}  G={ch1['G']}  B={ch1['B']}")
        self.car_rgb, _ = self._compute_rgb(self._car_raw, None)
        # Panel 2: configurable VB RGB
        self._vbrgb_raw = {
            chan: vb_stack[..., vb_idx[self._panel2_channels[chan]]].astype(np.float32)
            for chan in ("R", "G", "B")
        }
        ch2 = self._panel2_channels
        print(f"  Panel-2 (VB RGB):  R={ch2['R']}  G={ch2['G']}  B={ch2['B']}")
        self.vbrgb_rgb, _ = self._compute_rgb(self._vbrgb_raw, None)

        # OLINDEX3 single-band (replaces v6's NIR false-color panel).
        if OLINDEX_PARAM not in vb_idx:
            raise SystemExit(
                f"{OLINDEX_PARAM} not in v4 VB stack; available: "
                f"{list(vb_idx)}"
            )
        self._olindex_raw = (
            vb_stack[..., vb_idx[OLINDEX_PARAM]].astype(np.float32)
        )
        finite = self._olindex_raw[np.isfinite(self._olindex_raw)]
        if finite.size > 10:
            self._olindex_default_clim = tuple(
                float(v) for v in np.percentile(finite, [2, 98]))
        else:
            self._olindex_default_clim = (0.0, 1.0)
        print(f"  OLINDEX3 default p2/p98 = "
              f"{self._olindex_default_clim[0]:.4f} - "
              f"{self._olindex_default_clim[1]:.4f}")

        # Brown raw channels for restretch.
        bv = self.brown_valid
        self._brown_raw = {
            name: np.where(bv, self.brown[name].astype(np.float32), np.nan)
            for name in ("centroid", "amplitude", "asymmetry")
        }
        self.brown_rgb, _ = self._compute_brown_rgba(
            None, BROWN_STRETCH_FULLSCENE)

    def _ensure_output_paths(self) -> None:
        """Materialize the on-disk paths for this session's <name>.
        Idempotent - subsequent calls in the same launch are no-ops
        because session_name was set on the first save.
        """
        if self.session_name is None:
            try:
                raw = input(
                    "Name for this session "
                    "(e.g. carbonates / olivine / ridges): ").strip()
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
        self.spectra_csv = self.session_dir / f"{self.cube_id}_spectra.csv"
        print(f"  name root -> {self.name_dir}")
        print(f"  session   -> {self.session_dir}")

    def _visible_slice(self) -> tuple[slice, slice] | None:
        try:
            xlim = self.ax_car.get_xlim()
            ylim = self.ax_car.get_ylim()
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

    def _compute_rgb(self, raw_channels: dict,
                       view) -> tuple[np.ndarray, dict]:
        """Generic 3-band RGB stretcher used for CAR and NIR FC. raw_channels
        is a dict mapping 'R'/'G'/'B' -> (H, W) float32 array."""
        h, w = self.shape
        out = np.zeros((h, w, 3), dtype=np.float32)
        bounds = {}
        for k, name in enumerate(("R", "G", "B")):
            full = raw_channels[name]
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

    def _compute_brown_rgba(self, view, default_stretch
                              ) -> tuple[np.ndarray, dict]:
        h, w = self.shape
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        bounds = {}
        for k, name in enumerate(("centroid", "amplitude", "asymmetry")):
            full = self._brown_raw[name]
            if view is not None:
                sample = full[view]
                finite = sample[np.isfinite(sample)]
                if finite.size < 10:
                    lo, hi = (default_stretch[name] if default_stretch
                              else (0.0, 1.0))
                else:
                    lo, hi = np.percentile(finite, [2, 98])
                lo, hi = float(lo), float(hi)
            elif default_stretch is not None:
                lo, hi = default_stretch[name]
            else:
                finite = full[np.isfinite(full)]
                lo, hi = (np.percentile(finite, [2, 98])
                          if finite.size >= 10 else (0.0, 1.0))
                lo, hi = float(lo), float(hi)
            bounds[name] = (lo, hi)
            rgba[..., k] = percentile_norm(full, lo, hi)
        rgba[..., 3] = self.brown_valid.astype(np.float32)
        nan_b = (np.isnan(rgba[..., 0]) | np.isnan(rgba[..., 1])
                 | np.isnan(rgba[..., 2]))
        for k in range(3):
            rgba[..., k] = np.where(nan_b, 0.0, rgba[..., k])
        rgba[nan_b, 3] = 0.0
        return rgba, bounds

    # ---- Figure construction ----
    def _build_figure(self) -> None:
        plt.rcdefaults()
        plt.rcParams.update({
            "axes.titlesize": 13, "axes.labelsize": 12,
            "legend.fontsize": 10, "xtick.labelsize": 10,
            "ytick.labelsize": 10, "figure.titlesize": 12,
        })

        # ---- Window 1: Viewer (CAR + VB RGB + OLINDEX3 + Brown RGB) ----
        self.fig_viewer = plt.figure(
            figsize=(22, 6.5),
            num=f"v9.1 Viewer - {self.cube_id}",
        )
        gs_v = self.fig_viewer.add_gridspec(
            1, 4, wspace=0.07,
            left=0.02, right=0.995, top=0.91, bottom=0.16,
        )
        self.ax_car = self.fig_viewer.add_subplot(gs_v[0, 0])
        self.ax_vbrgb = self.fig_viewer.add_subplot(
            gs_v[0, 1], sharex=self.ax_car, sharey=self.ax_car)
        self.ax_olindex = self.fig_viewer.add_subplot(
            gs_v[0, 2], sharex=self.ax_car, sharey=self.ax_car)
        self.ax_brown = self.fig_viewer.add_subplot(
            gs_v[0, 3], sharex=self.ax_car, sharey=self.ax_car)

        # ---- Window 2: Plots (ratio + Brown ROI-mean scatter) ----
        self.fig_plots = plt.figure(
            figsize=(12, 8.5),
            num=f"v9.1 Plots - {self.cube_id}",
        )
        gs_p = self.fig_plots.add_gridspec(
            2, 1, height_ratios=[1, 1.1], hspace=0.30,
            left=0.08, right=0.97, top=0.95, bottom=0.07,
        )
        self.ax_ratio = self.fig_plots.add_subplot(gs_p[0, 0])
        self.ax_scatter = self.fig_plots.add_subplot(gs_p[1, 0])

        self._draw_static_panels()
        self._draw_status_strip()
        self._update_dynamic()

        # Rectangle selectors on ALL four viewer panels so the user
        # can drag a rectangle on any panel they prefer. The onselect
        # callback is the same; axes share x/y so the resulting pixel
        # bbox is identical regardless of source panel.
        self._rects = []
        for ax in (self.ax_car, self.ax_vbrgb,
                    self.ax_olindex, self.ax_brown):
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

        # Click handler (for pixel mode)
        self.fig_viewer.canvas.mpl_connect(
            "button_press_event", self._on_click)
        # Zoom restretch
        self.ax_car.callbacks.connect("ylim_changed", self._schedule_restretch)
        self.ax_car.callbacks.connect("xlim_changed", self._schedule_restretch)
        # Keys on both windows
        self.fig_viewer.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig_plots.canvas.mpl_connect("key_press_event", self._on_key)
        # Closing either closes both AND saves session metadata.
        self.fig_viewer.canvas.mpl_connect("close_event", self._on_close)
        self.fig_plots.canvas.mpl_connect("close_event", self._on_close)
        self._closed = False
        self._refresh_titles()

    def _refresh_titles(self) -> None:
        kind, label, _, _ = self._next_kind_and_label()
        self.fig_viewer.suptitle(
            f"v9.1 Viewer  -  {self.cube_id}  |  "
            f"mode = {self.mode.upper()}  |  next ROI = {label} ({kind})  |  "
            f"keys: m=mode  enter=finalize  u=undo  c=clear  s=save  r=restretch",
            fontsize=11,
        )
        self.fig_plots.suptitle(
            f"v9.1 Plots  -  {self.cube_id}  "
            f"(ratio = target_i / mafic_i ; scatter = per-target ROI mean)",
            fontsize=11,
        )

    def _draw_static_panels(self) -> None:
        self._car_imshow = self.ax_car.imshow(
            self.car_rgb, interpolation="nearest")
        ch1 = self._panel1_channels
        self._car_base_title = (
            f"CAR  R={ch1['R']}  G={ch1['G']}  B={ch1['B']}"
        )
        self.ax_car.set_title(self._car_base_title + "  [dynamic stretch]",
                                fontsize=11, fontweight="bold")
        self.ax_car.set_xticks([]); self.ax_car.set_yticks([])

        self._vbrgb_imshow = self.ax_vbrgb.imshow(
            self.vbrgb_rgb, interpolation="nearest")
        ch2 = self._panel2_channels
        self._vbrgb_base_title = (
            f"VB RGB  R={ch2['R']}  G={ch2['G']}  B={ch2['B']}"
        )
        self.ax_vbrgb.set_title(
            self._vbrgb_base_title + "  [dynamic stretch]",
            fontsize=11, fontweight="bold")
        self.ax_vbrgb.set_xticks([]); self.ax_vbrgb.set_yticks([])

        self._olindex_imshow = self.ax_olindex.imshow(
            self._olindex_raw,
            cmap=OLINDEX_CMAP,
            vmin=self._olindex_default_clim[0],
            vmax=self._olindex_default_clim[1],
            interpolation="nearest",
        )
        self._olindex_base_title = "OLINDEX3 (olivine 1 um band depth)"
        self.ax_olindex.set_title(
            f"{self._olindex_base_title}  [dynamic stretch]",
            fontsize=11, fontweight="bold")
        self.ax_olindex.set_xticks([]); self.ax_olindex.set_yticks([])

        self._brown_imshow = self.ax_brown.imshow(
            self.brown_rgb, interpolation="nearest")
        self._brown_base_title = (
            "Brown RGB  centroid / amplitude / asymmetry"
        )
        self.ax_brown.set_title(
            self._brown_base_title + "  [dynamic stretch]",
            fontsize=11, fontweight="bold")
        self.ax_brown.set_xticks([]); self.ax_brown.set_yticks([])

        # Now that imshow has fitted the axis limits to the image extent,
        # lock autoscale off on the viewer panels so subsequent
        # add_patch / scatter calls (ROI overlays, pending-pixel markers)
        # plus RectangleSelector.set_active(False) cannot resize the view.
        for ax in (self.ax_car, self.ax_vbrgb,
                    self.ax_olindex, self.ax_brown):
            ax.set_autoscale_on(False)

        # Brown scatter background (no hexbin per spec); just Fo zone shading.
        self.ax_scatter.axvspan(0.90, 1.15, alpha=0.05, color="grey")
        self.ax_scatter.axvspan(1.15, 1.20, alpha=0.04, color="green")
        self.ax_scatter.axvspan(1.20, 1.30, alpha=0.04, color="orange")
        self.ax_scatter.axvspan(1.30, 1.50, alpha=0.04, color="red")
        self.ax_scatter.axvline(1.15, color="#444444", lw=0.6, ls=":",
                                 alpha=0.6)
        # Track the static Fo-zone text artists so dynamic cleanup can
        # sweep any other Text on this axis (defensive against stale
        # ROI labels that art.remove() failed to delete in the
        # interactive matplotlib loop).
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

    # ---- Dynamic update ----
    def _mean_spectrum(self, mask: np.ndarray) -> np.ndarray:
        if mask is None or not mask.any():
            return np.full(self.cube.shape[0], np.nan, dtype=np.float32)
        with np.errstate(invalid="ignore"):
            return np.nanmean(self.cube[:, mask], axis=1)

    def _update_pending_overlay(self) -> None:
        # Clear previous pending markers
        for art in self._pending_artists:
            try: art.remove()
            except Exception: pass
        self._pending_artists = []
        if self.mode != "pixel" or not self.pending_pixels:
            self.fig_viewer.canvas.draw_idle()
            return
        ys = [y for y, _ in self.pending_pixels]
        xs = [x for _, x in self.pending_pixels]
        for ax in (self.ax_car, self.ax_vbrgb,
                    self.ax_olindex, self.ax_brown):
            # 'x' marker is unfilled; thick cyan stroke contrasts with
            # the inferno colormap (warm-toned) and the dark VB / Brown
            # backgrounds.
            sc = ax.scatter(xs, ys, s=40, marker="x",
                              c="cyan", linewidths=1.6, zorder=10)
            self._pending_artists.append(sc)
        self.fig_viewer.canvas.draw_idle()

    def _update_dynamic(self) -> None:
        # Clear previous overlay artists.
        for attr in ("viewer_artists", "scatter_artists", "ratio_lines"):
            arts = getattr(self, attr, [])
            for art in arts:
                try: art.remove()
                except Exception: pass
            setattr(self, attr, [])

        # Defensive sweep: matplotlib's interactive backends sometimes
        # leave Text artists behind even after .remove() if the prior
        # draw was mid-flush. Walk ax_scatter.texts and drop anything
        # that isn't one of the static Fo-zone labels we drew once at
        # startup.
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

        # ROI rectangles + labels on all four viewer panels
        viewer_axes = (self.ax_car, self.ax_vbrgb,
                        self.ax_olindex, self.ax_brown)
        for roi in self.rois:
            mask = roi["mask"]; color = roi["color"]; label = roi["label"]
            if not mask.any():
                continue
            ys, xs = np.where(mask)
            ls = "-" if roi["kind"] == "target" else "--"
            lw = 1.6 if roi["kind"] == "target" else 1.4
            x_lo, x_hi = int(xs.min()), int(xs.max())
            y_lo, y_hi = int(ys.min()), int(ys.max())
            for ax in viewer_axes:
                # Rectangle outline of mask bbox
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
            # For pixel-mode ROIs, also mark the actual pixels
            if roi["mode"] == "pixel":
                for ax in viewer_axes:
                    sc = ax.scatter(xs, ys, s=20, c=color, marker="o",
                                      alpha=0.85, edgecolors="black",
                                      linewidths=0.5, zorder=8)
                    self.viewer_artists.append(sc)

        # Brown scatter: ROI-MEAN of each TARGET ROI as a labeled marker
        # with 1-sigma error bars on centroid (x) and asymmetry (y).
        # The bars show within-ROI pixel-value spread (sample std), not
        # the standard error of the mean - so they convey ROI
        # heterogeneity. SEM = std / sqrt(n) is computable post-hoc
        # from the rois.csv columns if AV prefers that convention.
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
                xerr=xerr, yerr=yerr,
                fmt="o", color=roi["color"],
                ecolor="black", elinewidth=0.8,
                capsize=3, capthick=0.8,
                markersize=10, markeredgecolor="black",
                markeredgewidth=1.0,
                zorder=5,
                label=f"{roi['label']} (n={int(roi['mask'].sum())})",
            )
            for child in err.get_children():
                self.scatter_artists.append(child)
            txt = self.ax_scatter.text(
                br["centroid_mean"], br["asymmetry_mean"], "  " + roi["label"],
                color="black", ha="left", va="center",
                fontsize=10, fontweight="bold", zorder=6,
            )
            self.scatter_artists.append(txt)
        self.ax_scatter.set_title(
            f"Brown centroid vs asymmetry  "
            f"(ROI means +/- 1 sigma; "
            f"{len(self._targets())} target ROI/s)",
            fontsize=12, fontweight="bold",
        )
        leg = self.ax_scatter.get_legend()
        if leg is not None:
            leg.remove()
        if self._targets():
            self.ax_scatter.legend(fontsize=9, loc="upper right")

        # Ratio spectra: one curve per complete pair
        self.ax_ratio.clear()
        self.ax_ratio.set_xlabel("wavelength (um)")
        self.ax_ratio.set_ylabel("ratio")
        self.ax_ratio.set_xlim(0.4, 2.6)
        pairs = self._pairs()
        if pairs:
            self.ax_ratio.axhline(1.0, color="#888888", ls=":", lw=0.7)
            for tgt, maf in pairs:
                spec_t = self._mean_spectrum(tgt["mask"])
                spec_m = self._mean_spectrum(maf["mask"])
                ratio = spec_t / np.where(np.abs(spec_m) > 1e-6,
                                            spec_m, np.nan)
                line, = self.ax_ratio.plot(
                    self.wav, ratio, color=tgt["color"], lw=1.7,
                    label=f"{tgt['label']} / {maf['label']}",
                )
                self.ratio_lines.append(line)
            self.ax_ratio.legend(fontsize=10, loc="lower right",
                                  framealpha=0.92)
            self.ax_ratio.set_title(
                f"Ratio spectra ({len(pairs)} complete pair/s)",
                fontsize=12, fontweight="bold",
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
                "m = toggle mode (rect / pixel)\n"
                "enter = finalize pending pixel ROI\n"
                "u = undo  |  c = clear  |  s = save",
                ha="center", va="center", transform=self.ax_ratio.transAxes,
                color="#888888", fontsize=11,
            )

        self.fig_viewer.canvas.draw_idle()
        self.fig_plots.canvas.draw_idle()

    # ---- Brown stats per ROI ----
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
            print("[roi] empty rectangle; ignored")
            return
        mask = np.zeros(self.shape, dtype=bool)
        mask[y_lo:y_hi + 1, x_lo:x_hi + 1] = True
        self._add_roi(mask, mode="rect")

    def _on_click(self, event) -> None:
        if self.mode != "pixel":
            return
        if event.button != 1:
            return
        if event.inaxes not in (self.ax_car, self.ax_vbrgb,
                                  self.ax_olindex, self.ax_brown):
            return
        # Skip if matplotlib's nav toolbar is in zoom / pan mode - those
        # consume the same left-click but the user expects them to NOT
        # add a pixel to the ROI.
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
        # Toggle pixel: click a selected pixel again to remove it
        if coord in self.pending_pixels:
            self.pending_pixels.remove(coord)
        else:
            self.pending_pixels.append(coord)
        self._update_pending_overlay()
        # Status update
        next_kind, next_label, _, _ = self._next_kind_and_label()
        self._status_txt.set_text(
            f"  mode={self.mode.upper():<6}  rois={len(self.rois):>2}  "
            f"complete pairs={len(self._pairs()):>2}  "
            f"pending pixels={len(self.pending_pixels):>3}  "
            f"next: {next_label} ({next_kind})"
        )
        self.fig_viewer.canvas.draw_idle()

    def _finalize_pending(self) -> None:
        if not self.pending_pixels:
            print("[finalize] no pending pixels")
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
        print(f"[roi] {label}: {n_px} px  mode={mode}  centroid_mean={cm_str}  "
              "(unsaved; will write at close or 's')")
        # No I/O here - data is only persisted at session close (or 's' key).
        self._update_dynamic()

    def _append_rois_csv(self, roi: dict, mask_path: Path) -> None:
        row = [
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
            mask_path.name,
        ]
        with open(self.rois_csv, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _save_spectra_csv(self) -> None:
        # Always re-emit the per-session spectra CSV from current state.
        # Per-ROI raw mean spectra for ALL ROIs + per-pair ratios.
        if self.spectra_csv is None:
            # paths not yet materialized; nothing to write.
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
            spec_t = self._mean_spectrum(tgt["mask"])
            spec_m = self._mean_spectrum(maf["mask"])
            r = spec_t / np.where(np.abs(spec_m) > 1e-6, spec_m, np.nan)
            ratios.append(r)
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

    # ---- Mode + key handling ----
    def _set_mode(self, new_mode: str) -> None:
        if new_mode == self.mode:
            return
        # Save view limits so the mode switch never changes zoom.
        xlim = self.ax_car.get_xlim()
        ylim = self.ax_car.get_ylim()
        # Leaving pixel mode with pending pixels: finalize them
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
        # Restore view (axes are shared, so one set propagates)
        self.ax_car.set_xlim(xlim)
        self.ax_car.set_ylim(ylim)
        self.fig_viewer.canvas.draw_idle()

    def _on_key(self, event) -> None:
        if event.key == "m":
            self._set_mode("pixel" if self.mode == "rect" else "rect")
        elif event.key == "enter" or event.key == "return":
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
        # In pixel mode with pending pixels: pop the last pending click
        if self.mode == "pixel" and self.pending_pixels:
            popped = self.pending_pixels.pop()
            print(f"[undo] removed pending pixel {popped}")
            self._update_pending_overlay()
            return
        if not self.rois:
            print("[undo] nothing to remove")
            return
        popped = self.rois.pop()
        print(f"[undo] removed {popped['label']} (CSV row remains; "
              "edit rois.csv to remove the historical record)")
        self._save_spectra_csv()
        self._update_dynamic()

    def _clear(self) -> None:
        self.rois.clear()
        self.pending_pixels.clear()
        print("[clear] cleared in-memory ROIs + pending pixels  "
              "(rois.csv unchanged; edit manually if needed)")
        self._update_pending_overlay()
        self._update_dynamic()

    # ---- Session save (writes everything; called only on 's' or close) ----
    def _save_session(self) -> None:
        if not self.rois:
            print("[save] no ROIs to save.")
            return
        # Resolve output paths on first save (prompt for the name slug).
        self._ensure_output_paths()
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        # 1. Mask files + rois.csv rows for any ROI not yet persisted.
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
            roi["_mask_path"] = mask_path
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

        # 2. Per-session spectra CSV (re-written each save call)
        self._save_spectra_csv()

        # 3. Screenshots + meta inside the session subdir
        self.fig_viewer.savefig(
            self.session_dir / f"{self.cube_id}_viewer.png",
            dpi=150, bbox_inches="tight")
        self.fig_plots.savefig(
            self.session_dir / f"{self.cube_id}_plots.png",
            dpi=150, bbox_inches="tight")
        meta = {
            "cube_id": self.cube_id,
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
            "panel1_composite_car": dict(self._panel1_channels),
            "panel2_composite_vbrgb": dict(self._panel2_channels),
            "car_channels": dict(self._panel1_channels),  # legacy alias
            "olindex_param": OLINDEX_PARAM,
            "olindex_cmap": OLINDEX_CMAP,
            "olindex_default_clim_p2p98": list(self._olindex_default_clim),
            "brown_stretch_fullscene": BROWN_STRETCH_FULLSCENE,
            "scatter_xlim": list(SCATTER_XLIM),
            "scatter_ylim": list(SCATTER_YLIM),
            "rois_csv": str(self.rois_csv),
            "spectra_csv": str(self.spectra_csv),
        }
        meta_path = self.session_dir / f"{self.cube_id}_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[save] {len(new_rows)} new mask/CSV row(s); spectra + "
              f"screenshots + meta written to {self.session_dir}")

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
                  "writing fresh screenshots + meta only.")
            try:
                self._save_session()
            except Exception as e:
                print(f"[close] save failed: {e}")
            plt.close("all")
            return
        # Terminal prompt for new (unsaved) ROIs.
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
        car_rgb, _ = self._compute_rgb(self._car_raw, view)
        self.car_rgb = car_rgb
        self._car_imshow.set_data(car_rgb)

        vb_rgb, _ = self._compute_rgb(self._vbrgb_raw, view)
        self.vbrgb_rgb = vb_rgb
        self._vbrgb_imshow.set_data(vb_rgb)

        # OLINDEX3 single-band: update vmin/vmax (clim) on zoom.
        sample = (self._olindex_raw[view] if view is not None
                  else self._olindex_raw)
        finite = sample[np.isfinite(sample)]
        if finite.size > 10:
            ol_lo, ol_hi = np.percentile(finite, [2, 98])
        else:
            ol_lo, ol_hi = self._olindex_default_clim
        self._olindex_imshow.set_clim(float(ol_lo), float(ol_hi))

        rgba, br_bounds = self._compute_brown_rgba(
            view, BROWN_STRETCH_FULLSCENE)
        self.brown_rgb = rgba
        self._brown_imshow.set_data(rgba)
        scope = "visible region" if view is not None else "full scene"
        self.ax_car.set_title(
            f"{self._car_base_title}  [{scope} p2-p98]",
            fontsize=10,
        )
        self.ax_vbrgb.set_title(
            f"{self._vbrgb_base_title}  [{scope} p2-p98]",
            fontsize=10,
        )
        self.ax_olindex.set_title(
            f"{self._olindex_base_title}  [{scope} p2-p98: "
            f"{ol_lo:.3f} - {ol_hi:.3f}]",
            fontsize=10,
        )
        br_lbl = " / ".join(
            f"{name[:4]}: {br_bounds[name][0]:.2f}-{br_bounds[name][1]:.2f}"
            for name in ("centroid", "amplitude", "asymmetry")
        )
        self.ax_brown.set_title(
            f"{self._brown_base_title}  [{scope} p2-p98]\n{br_lbl}",
            fontsize=9,
        )
        self.fig_viewer.canvas.draw_idle()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cube", nargs="?", default="FRT00003E12",
                        help="cube id (full FRT0000XXXX or short)")
    parser.add_argument("--composite", default=None,
                        help=("Panel-2 RGB composite. Either a preset name "
                              f"({', '.join(PANEL2_PRESETS)}) or three "
                              "comma-separated VB param names "
                              "(e.g. OLINDEX3,BD1900R2,D2300). "
                              f"Default: {DEFAULT_PANEL2_COMPOSITE}. "
                              "Panel 1 is always CAR."))
    args = parser.parse_args()
    cube_id = resolve_cube_id(args.cube)
    panel2 = parse_composite_arg(args.composite)
    print(f"=== v9.1 ROI Explorer: {cube_id} ===")
    print(f"backend: {matplotlib.get_backend()}")
    explorer = RoiExplorer(cube_id, panel2_channels=panel2)
    plt.show()


if __name__ == "__main__":
    main()
