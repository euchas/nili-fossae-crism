"""v6 - Interactive ROI explorer for ratio spectra + Brown parameter context.

TWO windows. Viewer + Plots open side-by-side, both update live.

PAIRED RECTANGLE ROIs: drag rectangles on the GMM panel. The tool
auto-alternates between target (science) and mafic (reference) per pair:

  rect 1  -> target1   rect 2 -> mafic1   (pair 1)
  rect 3  -> target2   rect 4 -> mafic2   (pair 2)
  rect 5  -> target3   rect 6 -> mafic3   (pair 3)
  ...

Each pair shares one color (vivid target + same color for mafic). All
ROIs are outlined in black with a boxed text label centered on them.

Window 1 (Viewer):
  Panel 1 (left)   : v5-B aligned k=6 GMM scene map - drag rectangle here
  Panel 2 (mid)    : CRISM false-color composite (R=2.529, G=1.506, B=1.080 um)
  Panel 3 (right)  : Brown RGB full scene (centroid/amplitude/asymmetry, v5 stretches)
  Bottom strip     : status text (ROI counts + pair list)

Window 2 (Plots):
  Panel A (top)    : ratio spectra - one curve per pair, colored to
                     match the pair: target_i mean / mafic_i mean.
  Panel B (bottom) : Brown centroid-asymmetry hexbin with each TARGET ROI's
                     pixels color-coded and labeled. Mafic pixels are NOT
                     plotted (mafic = ratio reference; the science target
                     goes on the parameter curve).

Workflow (keyboard-only - no buttons because matplotlib widgets are
flaky across multi-window setups):
  1. Default slot at startup is NUMERATOR.
  2. Lasso-draw on Panel 1 -> auto-labels num1, num2, num3, ...
  3. Press 'd' to switch to DENOMINATOR slot. Subsequent lassos
     become den1, den2, ...
  4. Press 'n' to switch back to numerators (more num4, num5, ...).
  5. Both windows update automatically.
  6. 'u' undoes the last ROI from the active slot.
  7. 'c' clears all ROIs.
  8. 's' saves per-ROI masks + ratio CSV + window screenshots.
  9. Close either window to exit (the other follows).

Run from VS Code or terminal in the project root:

    python code/v6_roi_explorer.py                  # FRT00003E12 by default
    python code/v6_roi_explorer.py FRT0000B438       # any other cube id
    python code/v6_roi_explorer.py B438             # short form also works

Backends: tries QtAgg, then TkAgg, then default. Both are typically
present in a Python install; if neither works, set MPLBACKEND env var.

Alternatives considered: napari (heavier dep, polished ROI tools) and
plotly+dash (web app, more setup). The matplotlib RectangleSelector
path keeps the existing stack and is fastest to launch. (The earlier
LassoSelector iteration was dropped 2026-04-28 - rectangles are
faster to draw, less error-prone, and sufficient for the EPSL workflow.)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
# Try interactive backends in priority order.
for _bk in ("QtAgg", "TkAgg", "Qt5Agg"):
    try:
        matplotlib.use(_bk)
        break
    except (ImportError, ValueError):
        continue

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector

# Larger font defaults so panel titles + axis labels stay readable
# when printed at A4 page width.
plt.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.titlesize": 14,
})

from config import CUBE_IDS, OUTPUTS_DIR, cube_dir
from crism_io import find_cube_and_brown_maps, load_cube_by_id
from display import apply_display_transform
from vb_parameters import V4_VB_PARAM_ORDER

# --- v6 constants ---
GMM_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c",
               "#d62728", "#9467bd", "#8c564b"]
NODATA_GREY = "#dddddd"

# One color per (target, mafic) pair. target_i and mafic_i share the same
# color from this palette so the visual association is unambiguous.
PAIR_PALETTE = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e",
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                "#bcbd22", "#17becf"]

# CAR (carbonate) composite per CRISM convention -- VB-parameter-based,
# not band-based. R highlights Fe/Mg-OH (carbonate left shoulder), G the
# 2.5 um Mg-carbonate doublet, B the 1.9 um water of hydration. Loaded
# from the cached v4 VB stack (no recompute).
CAR_CHANNELS = {"R": "D2300", "G": "BD2500_2", "B": "BD1900R2"}

# Brown RGB stretches: v5 full-scene 4-cube pooled p2/p98 (same data-driven
# stretch used in v5 full-scene Brown RGBs).  These cover sub-1.15-centroid
# mafic-cap pixels that the v4.1 eq.2-restricted stretch clipped.
BROWN_STRETCH_FULLSCENE = {
    "centroid": (1.014, 1.379),
    "amplitude": (0.116, 0.335),
    "asymmetry": (0.525, 4.439),
}

# Scatter axis bounds set to cover the full data span across all 4 cubes
# (centroid 0.91-1.47, asymmetry 0.20-13.63 from the Brown .npy files).
SCATTER_XLIM = (0.90, 1.50)
SCATTER_YLIM = (0.30, 10.0)

ANNOTATION_BANDS = {
    1.00: "1.0 (olivine/pyx)",
    1.40: "1.4 (OH/H2O)",
    1.90: "1.9 (H2O)",
    2.30: "2.3 (Fe-Mg-OH)",
    2.50: "2.5 (carbonate)",
}
CO2_RESIDUAL = (1.95, 2.05)

V5_ROOT = OUTPUTS_DIR / "_v5_param_pruning"
V6_OUT = OUTPUTS_DIR / "_v6_interactive_explorer"


def percentile_norm(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any() or hi - lo < 1e-9:
        out[finite] = 0.5
        return out
    out[finite] = np.clip((arr[finite] - lo) / (hi - lo), 0, 1)
    return out


def percentile_stretch(arr: np.ndarray, lo: float = 2, hi: float = 98):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return -1.0, 1.0
    return float(np.percentile(finite, lo)), float(np.percentile(finite, hi))


def annotate_spectrum(ax, fontsize: int = 10) -> None:
    """Vertical DASHED lines at standard absorption wavelengths + CO2
    residual shading. No text labels per v6 spec (AV: 'no labels for
    these lines'). fontsize argument retained for compat but unused."""
    ax.axvspan(CO2_RESIDUAL[0], CO2_RESIDUAL[1], color="#cccccc",
               alpha=0.3, zorder=0)
    for wl in ANNOTATION_BANDS:
        ax.axvline(wl, color="#aa4444", lw=0.8, ls="--", alpha=0.6, zorder=0)


def resolve_cube_id(arg: str | None) -> str:
    if not arg:
        return "FRT00003E12"
    a = arg.upper().strip()
    if a in CUBE_IDS:
        return a
    # Allow short form like "3E12", "871C", "A4FC", "B438"
    for cid in CUBE_IDS:
        if cid.endswith(a):
            return cid
    raise SystemExit(f"unknown cube {arg!r}; known: {CUBE_IDS}")


# =============================================================================
# Main explorer
# =============================================================================
class RoiExplorer:
    def __init__(self, cube_id: str):
        self.cube_id = cube_id
        self.timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        # Single ROI list. Each item: {"label": "target1"|"mafic1"|...,
        #                              "kind": "target"|"mafic",
        #                              "pair_idx": 1-indexed,
        #                              "color": "#xxxxxx",
        #                              "mask": (H,W) bool}
        # Auto-alternation: even-indexed (0,2,4) -> target; odd (1,3,5) -> mafic.
        self.rois: list[dict] = []
        self._load_data()
        self._build_figure()

    def _next_kind_and_label(self) -> tuple[str, str, int, str]:
        """Return (kind, label, pair_idx, color) for the next ROI."""
        n = len(self.rois)
        is_target = (n % 2 == 0)
        kind = "target" if is_target else "mafic"
        pair_idx = n // 2 + 1
        color = PAIR_PALETTE[(pair_idx - 1) % len(PAIR_PALETTE)]
        label = f"{kind}{pair_idx}"
        return kind, label, pair_idx, color

    def _targets(self) -> list[dict]:
        return [r for r in self.rois if r["kind"] == "target"]

    def _mafics(self) -> list[dict]:
        return [r for r in self.rois if r["kind"] == "mafic"]

    def _pairs(self) -> list[tuple[dict, dict]]:
        """Return list of (target, mafic) pairs that have BOTH halves drawn."""
        out = []
        targets = {r["pair_idx"]: r for r in self._targets()}
        mafics = {r["pair_idx"]: r for r in self._mafics()}
        for idx in sorted(set(targets) & set(mafics)):
            out.append((targets[idx], mafics[idx]))
        return out

    # ---- Data loading ----
    def _load_data(self) -> None:
        print(f"[load] {self.cube_id}: TER3 cube...")
        t0 = time.time()
        cube, wav, brown, _ = load_cube_by_id(self.cube_id)
        # Display-flip everything once at load. The cube is (B, H, W) -> flip H, W.
        # GMM + Brown are (H, W) -> flip H, W.
        self.wav = wav.astype(np.float32)
        self.cube = cube[:, ::-1, ::-1].copy()  # bands x H x W in display orientation
        self.brown = {k: v[::-1, ::-1].copy() for k, v in brown.items()}
        self.shape = self.cube.shape[1:]
        print(f"  cube loaded {self.cube.shape} in {time.time()-t0:.1f}s")

        # GMM v5-B aligned labels (preferred). Fall back to unaligned if missing.
        v5b_aligned = (V5_ROOT / self.cube_id / "arrays"
                       / f"{self.cube_id}_v5b_GMM_k6_labels_aligned.npy")
        v5b_raw = (V5_ROOT / self.cube_id / "arrays"
                   / f"{self.cube_id}_v5b_GMM_k6_labels.npy")
        labels_path = v5b_aligned if v5b_aligned.exists() else v5b_raw
        if not labels_path.exists():
            raise SystemExit(
                f"no v5-B GMM labels found at {labels_path.parent}. "
                "Run stage_v5_param_pruning.py first."
            )
        self.labels_path = labels_path
        labels = np.load(labels_path)
        self.gmm_labels = labels[::-1, ::-1].copy()
        print(f"  loaded {labels_path.name}")

        # FULL-SCENE Brown mask: every pixel with all 4 Brown params finite.
        # This is the entire range of the Brown .npy outputs (~287k per cube),
        # NOT the v4 eq.2-restricted set (~25k-225k per cube). Required so
        # ROI pixels with sub-1.15 centroids appear on the scatter overlay
        # rather than silently disappearing.
        self.brown_valid = (np.isfinite(self.brown["centroid"])
                            & np.isfinite(self.brown["amplitude"])
                            & np.isfinite(self.brown["hwhm"])
                            & np.isfinite(self.brown["asymmetry"]))
        n_full = int(self.brown_valid.sum())
        print(f"  full-scene Brown-finite pixels: {n_full}")

        # CAR (carbonate) composite: load the v4 VB stack and pick the
        # three channel parameters per CAR_CHANNELS. The v4 stack is
        # detector-oriented; flip H+W to match display orientation.
        v4_vb_path = (OUTPUTS_DIR / "_v4_vb_pca_brown_rgb" / self.cube_id
                      / "arrays" / f"{self.cube_id}_VB_stack_v4.npy")
        if not v4_vb_path.exists():
            raise SystemExit(
                f"missing v4 VB stack at {v4_vb_path}. "
                "run stage_v4_vb_pca_and_brown_rgb.py first."
            )
        vb_stack = np.load(v4_vb_path)[::-1, ::-1, :].copy()
        vb_idx = {p: i for i, p in enumerate(V4_VB_PARAM_ORDER)}
        self._fc_raw = {
            chan: vb_stack[..., vb_idx[CAR_CHANNELS[chan]]].astype(np.float32)
            for chan in ("R", "G", "B")
        }
        print(f"  loaded CAR channels: "
              + ", ".join(f"{c}={CAR_CHANNELS[c]}" for c in "RGB"))
        # Initial render uses full-scene stretch
        self.false_color, _ = self._compute_fc_rgb(None)

        # Brown RGB - keep raw channels masked by brown_valid for restretch
        bv = self.brown_valid
        self._brown_raw = {
            name: np.where(bv, self.brown[name].astype(np.float32), np.nan)
            for name in ("centroid", "amplitude", "asymmetry")
        }
        # Initial render uses v5 full-scene stretches
        self.brown_rgb, _ = self._compute_brown_rgba(
            None, BROWN_STRETCH_FULLSCENE)

    @staticmethod
    def _stretch_channel(arr: np.ndarray, lo_p: float = 2, hi_p: float = 98) -> np.ndarray:
        finite = np.isfinite(arr)
        out = np.full_like(arr, np.nan, dtype=np.float32)
        if not finite.any():
            return out
        a, b = np.percentile(arr[finite], [lo_p, hi_p])
        if b - a < 1e-9:
            out[finite] = 0.5
            return out
        out[finite] = np.clip((arr[finite] - a) / (b - a), 0, 1)
        return out

    def _visible_slice(self) -> tuple[slice, slice] | None:
        """Return (y_slice, x_slice) in detector-array coords for what's
        currently visible in the GMM panel. Returns None if axis limits
        are not yet initialized."""
        try:
            xlim = self.ax_gmm.get_xlim()
            ylim = self.ax_gmm.get_ylim()
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

    def _compute_fc_rgb(self, view: tuple[slice, slice] | None
                         ) -> tuple[np.ndarray, dict]:
        """Build the full (H,W,3) false-color RGB. If `view` is given, use
        only that visible region to compute per-channel p2/p98 stretches;
        otherwise use the whole scene. The full-scene RGB is always returned
        (clipping outside-view pixels)."""
        h, w = self.shape
        out = np.zeros((h, w, 3), dtype=np.float32)
        bounds = {}
        for k, name in enumerate(("R", "G", "B")):
            full = self._fc_raw[name]
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

    def _compute_brown_rgba(self, view: tuple[slice, slice] | None,
                             default_stretch: dict | None
                             ) -> tuple[np.ndarray, dict]:
        """Build the full (H,W,4) Brown RGBA. If `view` is given, recompute
        per-channel p2/p98 from the visible region; if `view` is None and
        `default_stretch` is given, use the default stretches; if both are
        None, fall back to full-scene p2/p98."""
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
        # Alpha = brown valid (so non-Brown pixels stay transparent black)
        rgba[..., 3] = self.brown_valid.astype(np.float32)
        nan_b = (np.isnan(rgba[..., 0]) | np.isnan(rgba[..., 1])
                 | np.isnan(rgba[..., 2]))
        for k in range(3):
            rgba[..., k] = np.where(nan_b, 0.0, rgba[..., k])
        rgba[nan_b, 3] = 0.0
        return rgba, bounds

    # ---- Figure layout (TWO WINDOWS) ----
    def _build_figure(self) -> None:
        plt.rcdefaults()

        # ---- Window 1: Viewer (3 maps + button strip) ----
        self.fig_viewer = plt.figure(
            figsize=(19, 7.5),
            num=f"v6 Viewer - {self.cube_id}",
        )
        gs_v = self.fig_viewer.add_gridspec(
            1, 3, wspace=0.10,
            left=0.02, right=0.99, top=0.92, bottom=0.14,
        )
        # Link x/y limits across the three Viewer panels: zoom/pan on any
        # one of GMM / false color / Brown RGB will mirror to the others.
        self.ax_gmm = self.fig_viewer.add_subplot(gs_v[0, 0])
        self.ax_fc = self.fig_viewer.add_subplot(
            gs_v[0, 1], sharex=self.ax_gmm, sharey=self.ax_gmm)
        self.ax_brown = self.fig_viewer.add_subplot(
            gs_v[0, 2], sharex=self.ax_gmm, sharey=self.ax_gmm)

        # ---- Window 2: Plots (ratio + Brown scatter) ----
        self.fig_plots = plt.figure(
            figsize=(13, 9),
            num=f"v6 Plots - {self.cube_id}",
        )
        gs_p = self.fig_plots.add_gridspec(
            2, 1, height_ratios=[1, 1.1], hspace=0.28,
            left=0.08, right=0.97, top=0.95, bottom=0.07,
        )
        self.ax_ratio = self.fig_plots.add_subplot(gs_p[0, 0])
        self.ax_scatter = self.fig_plots.add_subplot(gs_p[1, 0])

        self._draw_static_panels()
        self._draw_status_strip()
        self._reset_dynamic_overlays()
        self._update_dynamic()

        # Rectangle selector on the GMM panel (viewer window).
        # Override state_modifier_keys to disable matplotlib's built-in
        # 'rotate' modifier (default key 'r'). With interactive=False the
        # selector never initializes `_corner_order`, so any state-toggle
        # via 'r' makes _onmove crash with AttributeError. Setting
        # rotate='never' frees up 'r' for our restretch shortcut and
        # avoids the bug entirely.
        self.rect = RectangleSelector(
            self.ax_gmm, onselect=self._on_rect,
            useblit=True, button=[1],
            minspanx=2, minspany=2, spancoords="data",
            interactive=False,
            state_modifier_keys=dict(
                move=" ", clear="escape", square="shift",
                center="ctrl", rotate="never",
            ),
        )
        # Belt-and-suspenders: matplotlib still calls _onmove which checks
        # `self._corner_order`. With interactive=False this attribute is
        # never set; initialize it to an empty list to silence future
        # matplotlib regressions.
        if not hasattr(self.rect, "_corner_order"):
            self.rect._corner_order = []
        # Dynamic stretch on FC + Brown when GMM zoom/pan changes.
        # ax_gmm shares x/y with ax_fc and ax_brown so all three panels
        # have identical visible bounds; we only need to listen on one.
        self._restretch_pending = False
        self.ax_gmm.callbacks.connect("ylim_changed", self._schedule_restretch)
        self.ax_gmm.callbacks.connect("xlim_changed", self._schedule_restretch)
        # Key handlers on both windows so shortcuts work whichever has focus
        self.fig_viewer.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig_plots.canvas.mpl_connect("key_press_event", self._on_key)
        # Closing either window closes both
        self.fig_viewer.canvas.mpl_connect("close_event",
                                            lambda evt: plt.close("all"))
        self.fig_plots.canvas.mpl_connect("close_event",
                                           lambda evt: plt.close("all"))
        self.fig_viewer.suptitle(self._make_suptitle(), fontsize=11)
        self.fig_plots.suptitle(
            f"v6 Plots  -  {self.cube_id}  "
            f"(targets only on Brown scatter; mafic = ratio reference)",
            fontsize=11,
        )

    def _make_suptitle(self) -> str:
        kind, label, _, _ = self._next_kind_and_label()
        return (
            f"v6 Viewer  -  {self.cube_id}  |  "
            f"next ROI = {label} ({kind})  |  "
            f"drag rect on GMM  |  zoom restretches FC + Brown  |  "
            f"keys: u=undo c=clear s=save r=restretch"
        )

    def _draw_static_panels(self) -> None:
        # Panel 1: GMM
        gmm_disp = np.where(self.gmm_labels >= 0,
                            self.gmm_labels.astype(np.float32), np.nan)
        cmap = ListedColormap(GMM_PALETTE[:6])
        cmap.set_bad(NODATA_GREY)
        self.ax_gmm.imshow(gmm_disp, cmap=cmap, vmin=0, vmax=5,
                           interpolation="nearest")
        self.ax_gmm.set_title(
            f"GMM k=6 aligned (v5-B)",
            fontsize=14, fontweight="bold",
        )
        self.ax_gmm.set_xticks([]); self.ax_gmm.set_yticks([])

        # Panel 2: CAR carbonate composite (dynamic stretch on zoom)
        self._fc_imshow = self.ax_fc.imshow(self.false_color,
                                             interpolation="nearest")
        self._fc_base_title = (
            f"CAR composite  R={CAR_CHANNELS['R']}  "
            f"G={CAR_CHANNELS['G']}  B={CAR_CHANNELS['B']}"
        )
        self.ax_fc.set_title(self._fc_base_title + "  [dynamic stretch]",
                             fontsize=14, fontweight="bold")
        self.ax_fc.set_xticks([]); self.ax_fc.set_yticks([])

        # Panel 3: Brown RGB (dynamic stretch — recomputes on zoom)
        self._brown_imshow = self.ax_brown.imshow(self.brown_rgb,
                                                   interpolation="nearest")
        self._brown_base_title = (
            "Brown RGB  centroid / amplitude / asymmetry"
        )
        self.ax_brown.set_title(self._brown_base_title + "  [dynamic stretch]",
                                fontsize=14, fontweight="bold")
        self.ax_brown.set_xticks([]); self.ax_brown.set_yticks([])

        # Panel 5: Brown centroid-asymmetry hexbin (FULL-SCENE Brown-finite)
        cv = self.brown["centroid"][self.brown_valid]
        av = self.brown["asymmetry"][self.brown_valid]
        self.ax_scatter.hexbin(
            cv, av, gridsize=80, cmap="Greys", mincnt=1,
            norm=LogNorm(),
            extent=(SCATTER_XLIM[0], SCATTER_XLIM[1],
                    SCATTER_YLIM[0], SCATTER_YLIM[1]),
            alpha=0.85,
        )
        self.ax_scatter.set_xlim(*SCATTER_XLIM)
        self.ax_scatter.set_ylim(*SCATTER_YLIM)
        self.ax_scatter.set_xlabel("centroid (um)", fontsize=13)
        self.ax_scatter.set_ylabel("asymmetry", fontsize=13)
        self.ax_scatter.set_title(
            f"Brown centroid vs asymmetry  ({self.brown_valid.sum()} "
            "Brown-finite pixels, full scene)",
            fontsize=14, fontweight="bold")
        # Brown 2020 Fo-group regions (apply only above the eq.2 floor of 1.15)
        self.ax_scatter.axvspan(0.90, 1.15, alpha=0.05, color="grey")
        self.ax_scatter.axvspan(1.15, 1.20, alpha=0.04, color="green")
        self.ax_scatter.axvspan(1.20, 1.30, alpha=0.04, color="orange")
        self.ax_scatter.axvspan(1.30, 1.50, alpha=0.04, color="red")
        self.ax_scatter.axvline(1.15, color="#444444", lw=0.6, ls=":",
                                 alpha=0.6)
        self.ax_scatter.text(1.025, 9.5, "below eq.2\n(<1.15)", ha="center",
                             va="top", fontsize=10, color="#555555", alpha=0.9)
        self.ax_scatter.text(1.175, 9.5, "high Fo\n(>66)", ha="center",
                             va="top", fontsize=10, color="green", alpha=0.85)
        self.ax_scatter.text(1.25, 9.5, "med Fo\n(40-66)", ha="center",
                             va="top", fontsize=10, color="darkorange", alpha=0.85)
        self.ax_scatter.text(1.40, 9.5, "low Fo\n(<40)", ha="center",
                             va="top", fontsize=10, color="red", alpha=0.85)
        self.ax_scatter.grid(alpha=0.2)

    def _draw_status_strip(self) -> None:
        # Single text strip across the bottom of the viewer with current
        # ROI counts. No buttons - keyboard shortcuts only (matplotlib
        # Buttons across multi-window setups are flaky).
        self.status_ax = self.fig_viewer.add_axes([0.02, 0.025, 0.96, 0.07])
        self.status_ax.set_xticks([]); self.status_ax.set_yticks([])
        for spine in self.status_ax.spines.values():
            spine.set_visible(False)
        self._status_txt = self.status_ax.text(
            0.0, 0.5, "", fontsize=10, transform=self.status_ax.transAxes,
            va="center", ha="left", family="monospace",
        )

    def _reset_dynamic_overlays(self) -> None:
        # Lists of artists per category so we can blow them away on update
        self.gmm_artists: list = []  # filled scatter, contour, label, on GMM
        self.scatter_artists: list = []  # per-ROI scatter on Brown scatter
        self.ratio_lines: list = []  # per-numerator ratio curves

    # ---- Event handlers ----
    def _on_rect(self, eclick, erelease) -> None:
        if eclick.xdata is None or erelease.xdata is None:
            return
        x0, y0 = float(eclick.xdata), float(eclick.ydata)
        x1, y1 = float(erelease.xdata), float(erelease.ydata)
        x_lo, x_hi = sorted([int(round(x0)), int(round(x1))])
        y_lo, y_hi = sorted([int(round(y0)), int(round(y1))])
        h, w = self.shape
        x_lo = max(0, x_lo); x_hi = min(w - 1, x_hi)
        y_lo = max(0, y_lo); y_hi = min(h - 1, y_hi)
        if x_hi < x_lo or y_hi < y_lo:
            print("[roi] empty rectangle; ignoring")
            return
        mask = np.zeros(self.shape, dtype=bool)
        mask[y_lo:y_hi + 1, x_lo:x_hi + 1] = True
        n = int(mask.sum())
        if n == 0:
            print("[roi] empty rectangle; ignoring")
            return
        kind, label, pair_idx, color = self._next_kind_and_label()
        roi = {"label": label, "kind": kind, "pair_idx": pair_idx,
               "color": color, "mask": mask,
               "bbox": (x_lo, y_lo, x_hi, y_hi)}
        self.rois.append(roi)
        print(f"[roi] {label}: {n} px  color={color}  bbox=({x_lo},{y_lo})-({x_hi},{y_hi})")
        self._update_dynamic()

    def _on_key(self, event) -> None:
        if event.key == "u":
            self._undo_last()
        elif event.key == "c":
            self._clear()
        elif event.key == "s":
            self._save()
        elif event.key == "r":
            self._restretch_now()  # manual restretch trigger

    def _schedule_restretch(self, _ax) -> None:
        """xlim_changed and ylim_changed both fire on a single zoom; coalesce
        them so we recompute the stretch only once per zoom event."""
        if self._restretch_pending:
            return
        self._restretch_pending = True
        # Defer to next event-loop tick so both x and y limits have settled
        try:
            self.fig_viewer.canvas.manager.window.after(
                10, self._restretch_now)
        except Exception:
            # Backend without a Tk-style after(); just call directly.
            self._restretch_now()

    def _restretch_now(self) -> None:
        self._restretch_pending = False
        view = self._visible_slice()
        # FC
        rgb, fc_bounds = self._compute_fc_rgb(view)
        self.false_color = rgb
        self._fc_imshow.set_data(rgb)
        # Brown
        rgba, br_bounds = self._compute_brown_rgba(
            view, BROWN_STRETCH_FULLSCENE)
        self.brown_rgb = rgba
        self._brown_imshow.set_data(rgba)
        # Update titles to reflect the active stretch source
        scope = "visible region" if view is not None else "full scene"
        self.ax_fc.set_title(
            f"{self._fc_base_title}  [stretch: {scope} p2-p98]",
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

    def _undo_last(self) -> None:
        if not self.rois:
            print("[undo] no ROIs to remove")
            return
        popped = self.rois.pop()
        print(f"[undo] removed {popped['label']}")
        self._update_dynamic()

    def _clear(self) -> None:
        self.rois.clear()
        self._update_dynamic()
        print("[roi] cleared all ROIs")

    def _save(self) -> None:
        pairs = self._pairs()
        if not pairs:
            print("[save] need at least one COMPLETE pair "
                  "(target + mafic of same index)")
            return
        out = V6_OUT / self.cube_id
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        # Per-ROI mask saves (un-flipped to detector orientation)
        for roi in self.rois:
            mask_det = roi["mask"][::-1, ::-1]
            np.save(out / f"{self.cube_id}_{roi['label']}_mask_{ts}.npy",
                    mask_det)

        # Per-pair ratio spectra
        per_pair = []
        for tgt, maf in pairs:
            spec_t = self._mean_spectrum(tgt["mask"])
            spec_m = self._mean_spectrum(maf["mask"])
            spec_m_safe = np.where(np.abs(spec_m) > 1e-6, spec_m, np.nan)
            ratio = spec_t / spec_m_safe
            per_pair.append((tgt["pair_idx"], tgt["label"], maf["label"],
                             spec_t, spec_m, ratio))

        # CSV: wav, [target_i_mean, mafic_i_mean, ratio_i] per pair
        cols = ["wavelength_um"]
        for idx, t_lbl, m_lbl, _, _, _ in per_pair:
            cols += [f"{t_lbl}_mean", f"{m_lbl}_mean",
                     f"{t_lbl}_over_{m_lbl}"]
        with open(out / f"{self.cube_id}_ratio_spectra_{ts}.csv", "w") as f:
            f.write(",".join(cols) + "\n")
            for i, w_ in enumerate(self.wav):
                row = [f"{w_:.6f}"]
                for _, _, _, st, sm, r in per_pair:
                    row += [f"{st[i]:.6f}", f"{sm[i]:.6f}", f"{r[i]:.6f}"]
                f.write(",".join(row) + "\n")

        # Screenshots - both windows
        self.fig_viewer.savefig(out / f"{self.cube_id}_viewer_{ts}.png",
                                 dpi=150, bbox_inches="tight")
        self.fig_plots.savefig(out / f"{self.cube_id}_plots_{ts}.png",
                                dpi=150, bbox_inches="tight")
        # Metadata
        meta = {
            "cube_id": self.cube_id,
            "timestamp_utc": ts,
            "rois": [
                {"label": r["label"], "kind": r["kind"],
                 "pair_idx": r["pair_idx"], "color": r["color"],
                 "pixel_count": int(r["mask"].sum()),
                 "bbox_xlyhxhxh": list(r["bbox"])}
                for r in self.rois
            ],
            "complete_pairs": [
                {"pair_idx": idx, "target": t_lbl, "mafic": m_lbl}
                for idx, t_lbl, m_lbl, _, _, _ in per_pair
            ],
            "labels_source": str(self.labels_path),
            "car_channels": dict(CAR_CHANNELS),
            "brown_mask": "full_scene_finite_brown_4 (no eq.2)",
            "brown_stretch_fullscene": BROWN_STRETCH_FULLSCENE,
            "scatter_xlim": list(SCATTER_XLIM),
            "scatter_ylim": list(SCATTER_YLIM),
            "ratio_convention": (
                "Per-pair: mean(target_i) / mean(mafic_i). "
                "Pairs with index in both target and mafic lists are saved."
            ),
        }
        (out / f"{self.cube_id}_meta_{ts}.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        n_files = len(self.rois) + 4  # masks + 2 PNGs + 1 CSV + 1 JSON
        print(f"[save] wrote {n_files} files for {len(per_pair)} complete pair(s) -> {out}")

    # ---- Dynamic update ----
    def _mean_spectrum(self, mask: np.ndarray) -> np.ndarray:
        # mask is in display orientation; cube is also display-flipped
        if mask is None or not mask.any():
            return np.full(self.cube.shape[0], np.nan, dtype=np.float32)
        # Suppress all-NaN slice warnings from BBL bands
        with np.errstate(invalid="ignore"):
            spec = np.nanmean(self.cube[:, mask], axis=1)
        return spec

    def _update_dynamic(self) -> None:
        # Clear previous overlay artists (across all 3 viewer panels +
        # the Brown scatter)
        for art in self.gmm_artists:
            try: art.remove()
            except Exception: pass
        self.gmm_artists = []
        for art in self.scatter_artists:
            try: art.remove()
            except Exception: pass
        self.scatter_artists = []
        for line in self.ratio_lines:
            try: line.remove()
            except Exception: pass
        self.ratio_lines = []

        # Status text
        targets = self._targets()
        mafics = self._mafics()
        pairs = self._pairs()
        next_kind, next_label, _, _ = self._next_kind_and_label()
        status = (
            f"  ROIs: {len(self.rois):>2}     "
            f"complete pairs: {len(pairs)}     "
            f"targets: {len(targets)}  mafics: {len(mafics)}     "
            f"next: {next_label} ({next_kind})"
        )
        self._status_txt.set_text(status)
        self.fig_viewer.suptitle(self._make_suptitle(), fontsize=12)

        # Per-ROI overlays on ALL THREE viewer panels: rectangle outline +
        # white-filled label box positioned BELOW the rectangle. Filled
        # colored pixels (the data overlay) only on the GMM panel since
        # FC and Brown panels carry their own data.
        viewer_axes = (self.ax_gmm, self.ax_fc, self.ax_brown)
        for roi in self.rois:
            mask = roi["mask"]; color = roi["color"]; label = roi["label"]
            if not mask.any():
                continue
            x_lo, y_lo, x_hi, y_hi = roi["bbox"]
            width = x_hi - x_lo + 1; height = y_hi - y_lo + 1
            ls = "-" if roi["kind"] == "target" else "--"
            lw = 1.6 if roi["kind"] == "target" else 1.4

            # GMM only: fill the actual ROI pixels with a faint color so
            # you can see which GMM cluster they fall into. Skip on FC and
            # Brown — those panels show their own continuous data.
            ys, xs = np.where(mask)
            fill_alpha = 0.40 if roi["kind"] == "target" else 0.15
            sc = self.ax_gmm.scatter(
                xs, ys, s=4, c=color, alpha=fill_alpha, marker=".",
                edgecolors="none", zorder=5,
            )
            self.gmm_artists.append(sc)

            # Rectangle outline + label on all three viewer panels.
            # Outlines are WHITE per spec so they read on any underlying
            # panel (GMM colored cells, FC dark areas, Brown blacks).
            for ax in viewer_axes:
                rect = Rectangle(
                    (x_lo - 0.5, y_lo - 0.5), width, height,
                    fill=False, edgecolor="white", linewidth=lw,
                    linestyle=ls, zorder=6,
                )
                ax.add_patch(rect)
                self.gmm_artists.append(rect)
                # Label DIRECTLY below the rectangle (1 px gap so the
                # white box is flush with the bottom edge).
                cx = (x_lo + x_hi) / 2.0
                cy = y_hi + 1.0
                txt = ax.text(
                    cx, cy, label,
                    color="black", ha="center", va="top",
                    fontsize=12, fontweight="bold", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              alpha=0.95, edgecolor="black", lw=1.0),
                )
                self.gmm_artists.append(txt)

        # Brown scatter: TARGETS ONLY (mafic = ratio reference, not science)
        for roi in targets:
            sub = roi["mask"] & self.brown_valid
            if not sub.any():
                continue
            sc = self.ax_scatter.scatter(
                self.brown["centroid"][sub], self.brown["asymmetry"][sub],
                s=18, c=roi["color"], alpha=0.80, edgecolors="black",
                linewidths=0.5,
                label=f"{roi['label']} (n={int(sub.sum())})", zorder=4,
            )
            self.scatter_artists.append(sc)
            # Label the cluster of points on the scatter at its centroid
            cx = float(np.nanmedian(self.brown["centroid"][sub]))
            cy = float(np.nanmedian(self.brown["asymmetry"][sub]))
            txt = self.ax_scatter.text(
                cx, cy, roi["label"],
                color="black", ha="center", va="center",
                fontsize=11, fontweight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          alpha=0.92, edgecolor="black", lw=0.8),
            )
            self.scatter_artists.append(txt)
        leg = self.ax_scatter.get_legend()
        if leg is not None:
            leg.remove()
        if targets:
            self.ax_scatter.legend(fontsize=10, loc="upper right")

        # Ratio spectra: one curve per COMPLETE pair (target_i / mafic_i)
        self.ax_ratio.clear()
        self.ax_ratio.set_xlabel("wavelength (um)", fontsize=13)
        self.ax_ratio.set_ylabel("ratio", fontsize=13)
        self.ax_ratio.set_xlim(0.4, 2.6)
        # No grid per spec; only the dashed absorption-band lines remain.
        if pairs:
            self.ax_ratio.axhline(1.0, color="#888888", ls=":", lw=0.7)
            for tgt, maf in pairs:
                spec_t = self._mean_spectrum(tgt["mask"])
                spec_m = self._mean_spectrum(maf["mask"])
                ratio = spec_t / np.where(np.abs(spec_m) > 1e-6, spec_m, np.nan)
                line, = self.ax_ratio.plot(
                    self.wav, ratio, color=tgt["color"], lw=1.8,
                    label=f"{tgt['label']} / {maf['label']}",
                )
                self.ratio_lines.append(line)
            # Legend at BOTTOM-RIGHT per spec
            self.ax_ratio.legend(fontsize=11, loc="lower right",
                                  framealpha=0.92)
            self.ax_ratio.set_title(
                f"Ratio spectra ({len(pairs)} complete pair/s)",
                fontsize=14, fontweight="bold",
            )
            annotate_spectrum(self.ax_ratio)
        else:
            self.ax_ratio.set_title(
                "Ratio spectra - draw a target then a mafic rectangle to populate",
                fontsize=14, fontweight="bold",
            )
            self.ax_ratio.text(
                0.5, 0.5,
                "Drag rectangles on the GMM map.  Cycle is automatic:\n"
                "  rect 1 -> target1   rect 2 -> mafic1   (pair 1)\n"
                "  rect 3 -> target2   rect 4 -> mafic2   (pair 2)\n"
                "  ...\n"
                "u = undo last,  c = clear all,  s = save",
                ha="center", va="center", transform=self.ax_ratio.transAxes,
                color="#888888", fontsize=12,
            )

        # Refresh both windows
        self.fig_viewer.canvas.draw_idle()
        self.fig_plots.canvas.draw_idle()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cube", nargs="?", default="FRT00003E12",
                        help="cube id (full FRT0000XXXX or short XXXX)")
    args = parser.parse_args()
    cube_id = resolve_cube_id(args.cube)
    print(f"=== v6 ROI Explorer: {cube_id} ===")
    print(f"backend: {matplotlib.get_backend()}")
    explorer = RoiExplorer(cube_id)
    plt.show()


if __name__ == "__main__":
    main()
