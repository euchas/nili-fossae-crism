"""v9.1 Ratioed spectra grouped by geological class, multi-cube overlay.

Reads every outputs/spectra sample locations/<cube>/<class>/sessions/<TS>/<cube>_spectra.csv,
extracts the target_over_mafic columns, and renders a 2-column grid of
panels (one panel per class). Within each panel, every ROI's ratio
spectrum is drawn as a thin line colored by cube; a thick black mean
+/- 1-sigma envelope summarizes the class.

Usage:
  python code/v9_1_ratio_spectra.py
  python code/v9_1_ratio_spectra.py --classes olivine carbonates
  python code/v9_1_ratio_spectra.py --ymin 0.7 --ymax 1.7
  python code/v9_1_ratio_spectra.py --xmin 0.4 --xmax 2.6

Outputs:
  outputs/spectra sample locations/figures/ratio_spectra_by_class.png
  outputs/spectra sample locations/figures/ratio_spectra_by_class.pdf
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from config import OUTPUTS_DIR

SAMPLES_ROOT = OUTPUTS_DIR / "spectra sample locations"
FIGURE_DIR = SAMPLES_ROOT / "figures"

CLASS_COLORS = {
    "olivine":         "#1b9e77",
    "olivine_dunes":   "#66c2a5",
    "ol-carb":         "#d95f02",
    "carbonates":      "#d95f02",
    "ridge":           "#7570b3",
    "discovery_ridge": "#7570b3",
    "intrusions":      "#e7298a",
    "clays":           "#66a61e",
    "nontronite":      "#e6ab02",
    "saponite":        "#a6761d",
}
DEFAULT_CLASS_COLOR = "#444444"

EXCLUDE_CUBES = {"TEST_CUBE"}
EXCLUDE_CLASSES = {"session"}

CUBE_COLORS = {
    "FRT00003E12": "#377eb8",
    "FRT0000871C": "#4daf4a",
    "FRT0000A4FC": "#984ea3",
    "FRT0000B438": "#ff7f00",
}
DEFAULT_CUBE_COLOR = "#888888"

REF_BANDS = [
    (1.05, "1 µm\n(ol/pyx)"),
    (1.40, "OH"),
    (1.90, "H2O"),
    (2.30, "Fe/Mg-OH"),
    (2.50, "CO3"),
]
CO2_RESIDUAL = (1.95, 2.05)

USER_STYLE_PATHS = [str(Path(__file__).parent / "my_style.mplstyle.txt")]


def setup_style():
    for p in USER_STYLE_PATHS:
        if os.path.exists(p):
            try:
                plt.style.use(p)
                print(f"[style] using {p}")
                return
            except Exception as e:
                print(f"[style] {p} found but failed: {e}")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 300,
    })


def discover_class_spectra(classes_filter: list[str] | None
                            ) -> dict[str, list[dict]]:
    """Walk the cube/class layout and pull every target_over_mafic column
    from every per-session spectra CSV. Returns {class_name: [spec_dict, ...]}.
    Each spec_dict has cube, roi (target label), wavelength, ratio.
    """
    by_class: dict[str, list[dict]] = {}
    for cube_dir in sorted(SAMPLES_ROOT.iterdir()):
        if not cube_dir.is_dir() or cube_dir.name == "figures":
            continue
        cube_id = cube_dir.name
        if cube_id in EXCLUDE_CUBES and not classes_filter:
            continue
        for cls_dir in sorted(cube_dir.iterdir()):
            if not cls_dir.is_dir() or cls_dir.name == "sessions":
                continue
            cls = cls_dir.name
            if classes_filter and cls not in classes_filter:
                continue
            if not classes_filter and cls in EXCLUDE_CLASSES:
                continue
            sessions = cls_dir / "sessions"
            if not sessions.exists():
                continue
            for sess_dir in sorted(sessions.iterdir()):
                spec_csv = sess_dir / f"{cube_id}_spectra.csv"
                if not spec_csv.exists():
                    continue
                df = pd.read_csv(spec_csv)
                wav = df["wavelength_um"].to_numpy()
                # Match columns of the form '<target>_over_<mafic>'
                ratio_cols = [c for c in df.columns if "_over_" in c]
                for col in ratio_cols:
                    target_label = col.split("_over_")[0]
                    by_class.setdefault(cls, []).append({
                        "cube": cube_id,
                        "class": cls,
                        "roi": target_label,
                        "wavelength": wav,
                        "ratio": df[col].to_numpy(),
                        "session": sess_dir.name,
                    })
                if ratio_cols:
                    print(f"  {spec_csv.relative_to(SAMPLES_ROOT)} -> "
                          f"{len(ratio_cols)} ratios")
    if not by_class:
        raise SystemExit("no ratio columns found in any session spectra "
                          "CSV; nothing to plot")
    return by_class


def common_grid(spectra: list[dict]) -> np.ndarray:
    """Pick the longest wavelength array and assume all share it (TER3
    pipeline produces identical wavelengths per cube)."""
    return max(spectra, key=lambda s: len(s["wavelength"]))["wavelength"]


def annotate_panel(ax, xmin: float, xmax: float, ymax: float) -> None:
    # CO2 residual shading
    if CO2_RESIDUAL[0] < xmax and CO2_RESIDUAL[1] > xmin:
        ax.axvspan(*CO2_RESIDUAL, color="#cccccc", alpha=0.3, zorder=0)
    for w, _label in REF_BANDS:
        if xmin <= w <= xmax:
            ax.axvline(w, color="#aa4444", lw=0.6, ls="--",
                       alpha=0.5, zorder=0)
    ax.axhline(1.0, color="black", lw=0.5, alpha=0.5, zorder=0)


def annotate_band_labels(ax, xmin: float, xmax: float) -> None:
    """Top-of-panel band tags (only on first panel)."""
    ymin, ymax = ax.get_ylim()
    y = ymax - 0.02 * (ymax - ymin)
    for w, label in REF_BANDS:
        if xmin <= w <= xmax:
            ax.text(w, y, label, fontsize=7, color="0.35",
                    ha="center", va="top",
                    bbox=dict(boxstyle="round,pad=0.18",
                              facecolor="white", edgecolor="none",
                              alpha=0.7))


def render(by_class: dict[str, list[dict]],
            xmin: float, xmax: float,
            ymin: float, ymax: float,
            class_order: list[str] | None = None) -> dict:
    if class_order is None:
        class_order = sorted(by_class)
    n = len(class_order)
    ncols = 2
    nrows = math.ceil(n / ncols)

    fig_w = 13
    fig_h = 4.6 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                              sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    cubes_present_overall = sorted({s["cube"] for cls in by_class
                                      for s in by_class[cls]})

    for idx, cls in enumerate(class_order):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        spectra = by_class.get(cls, [])
        cls_color = CLASS_COLORS.get(cls, DEFAULT_CLASS_COLOR)
        if not spectra:
            ax.text(0.5, 0.5, f"no data\nfor {cls}",
                    ha="center", va="center", transform=ax.transAxes,
                    color="0.5")
            ax.set_title(cls, fontsize=11, color=cls_color,
                          fontweight="bold")
            continue

        # Per-ROI thin lines colored by cube
        for s in spectra:
            cube_color = CUBE_COLORS.get(s["cube"], DEFAULT_CUBE_COLOR)
            ax.plot(s["wavelength"], s["ratio"],
                     color=cube_color, lw=0.7, alpha=0.55, zorder=2)

        # Mean +/- 1 sigma across all ROIs in the class (across all cubes)
        wav = common_grid(spectra)
        stack = np.array([s["ratio"] if len(s["ratio"]) == len(wav)
                           else np.full(len(wav), np.nan)
                           for s in spectra])
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(stack, axis=0)
            std = np.nanstd(stack, axis=0)
        ax.fill_between(wav, mean - std, mean + std,
                         color=cls_color, alpha=0.20, zorder=1)
        ax.plot(wav, mean, color=cls_color, lw=2.0, zorder=4,
                 label=f"mean (n={len(spectra)})")

        annotate_panel(ax, xmin, xmax, ymax)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        cubes_in_panel = sorted({s["cube"] for s in spectra})
        ax.set_title(
            f"{cls.upper()}  ({len(spectra)} ROIs, "
            f"{len(cubes_in_panel)} cube{'s' if len(cubes_in_panel)>1 else ''})",
            fontsize=11, color=cls_color, fontweight="bold",
        )
        if r == nrows - 1:
            ax.set_xlabel(r"Wavelength ($\mu$m)")
        if c == 0:
            ax.set_ylabel("Ratio (target / mafic)")
        ax.grid(True, alpha=0.25)

    # Hide unused axes (for odd number of classes)
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    # Band labels on first panel
    if n > 0:
        annotate_band_labels(axes[0, 0], xmin, xmax)

    # Combined legend at the figure level: cube colors
    cube_handles = [
        Line2D([0], [0], color=CUBE_COLORS.get(c, DEFAULT_CUBE_COLOR),
               lw=2.0, label=c)
        for c in cubes_present_overall
    ]
    fig.legend(handles=cube_handles, title="Cube",
               loc="lower center", ncol=len(cube_handles),
               bbox_to_anchor=(0.5, -0.01),
               fontsize=9, title_fontsize=10, frameon=False)

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_png = FIGURE_DIR / "ratio_spectra_by_class.png"
    out_pdf = FIGURE_DIR / "ratio_spectra_by_class.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_png.relative_to(OUTPUTS_DIR.parent)}")
    print(f"  wrote {out_pdf.relative_to(OUTPUTS_DIR.parent)}")
    return {"out_png": str(out_png), "out_pdf": str(out_pdf),
            "n_classes": n}


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--classes", nargs="+", default=None,
                   help="restrict to these class names (and order)")
    p.add_argument("--xmin", type=float, default=0.4)
    p.add_argument("--xmax", type=float, default=2.6)
    p.add_argument("--ymin", type=float, default=0.8)
    p.add_argument("--ymax", type=float, default=1.6)
    args = p.parse_args(argv[1:])

    setup_style()
    print(f"=== ratioed spectra grouped by class ===")
    print(f"reading from: {SAMPLES_ROOT}")
    by_class = discover_class_spectra(args.classes)
    render(by_class,
           xmin=args.xmin, xmax=args.xmax,
           ymin=args.ymin, ymax=args.ymax,
           class_order=args.classes)


if __name__ == "__main__":
    main(sys.argv)
