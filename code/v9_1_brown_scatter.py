"""v9.1 Brown centroid vs asymmetry scatter, multi-class multi-cube.

Reads every outputs/spectra sample locations/<cube>/<class>/<cube>_<class>_rois.csv
and plots TARGET ROIs as one labeled marker per ROI with 1-sigma error
bars on centroid (x) and asymmetry (y). Color codes class, marker shape
codes cube. Background tinted by Brown 2020 Fo-number ranges.

Configuration:
  CLASS_DISPLAY    legend ordering and underscore-stripped display names
  EXCLUDE_ROWS     (cube, class) tuples to drop entirely (e.g. known-bad)
  COLLAPSE_TO_MEAN class names to collapse into a single mean point

Usage:
  python code/v9_1_brown_scatter.py
  python code/v9_1_brown_scatter.py --classes olivine carbonates
  python code/v9_1_brown_scatter.py --include-mafic
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedLocator, MultipleLocator
import numpy as np
import pandas as pd

from config import OUTPUTS_DIR

SAMPLES_ROOT = OUTPUTS_DIR / "spectra sample locations"
FIGURE_DIR = SAMPLES_ROOT / "figures"

# Local style copy (my_style.mplstyle.txt).
LOCAL_STYLE = Path(__file__).parent / "my_style.mplstyle.txt"
USER_STYLE_PATHS = [LOCAL_STYLE]

# Class colors. discovery_ridge -> black (pops out vs orange carbonates),
# intrusions -> dark grey, olivine_dunes -> intense light green.
# Saponite + nontronite merged into a single 'fe-mg-phyllosilicate' class
# coloured with nontronite's yellow-ochre (AV: 2026-05-01).
CLASS_COLORS = {
    "olivine":              "#1b9e77",   # teal-green
    "olivine_dunes":        "#7cb342",   # intense light green
    "ol-carb":              "#d95f02",   # orange
    "carbonates":           "#d95f02",   # orange
    "ridge":                "#000000",   # black
    "discovery_ridge":      "#000000",   # black
    "intrusions":           "#555555",   # dark grey
    "saponite":             "#a6761d",   # warm brown (legacy; merged below)
    "nontronite":           "#e6ab02",   # yellow-ochre (legacy; merged below)
    "fe-mg-phyllosilicate": "#e6ab02",   # merged; uses nontronite color
    "clays":                "#66a61e",   # olive
}
DEFAULT_COLOR = "#666666"

# Saponite + nontronite -> 'fe-mg-phyllosilicate'. Set to {} to skip.
MERGE_CLASSES = {
    "saponite":   "fe-mg-phyllosilicate",
    "nontronite": "fe-mg-phyllosilicate",
}

CUBE_MARKERS = {
    "FRT00003E12": "o",
    "FRT0000871C": "D",
    "FRT0000A4FC": "s",
    "FRT0000B438": "^",
}
DEFAULT_MARKER = "X"

# Class ordering + display names for the legend (post-merge order).
CLASS_DISPLAY_ORDER = [
    "olivine",
    "olivine_dunes",
    "ol-carb",
    "carbonates",
    "fe-mg-phyllosilicate",   # merged saponite + nontronite
    "ridge",
    "discovery_ridge",
    "intrusions",
    "clays",
]


# Display names for the legend.
CLASS_DISPLAY_NAMES = {
    "olivine":              "Olivine Unit",
    "olivine_dunes":        "Olivine Dunes",
    "ol-carb":              "Olivine-carb",
    "carbonates":           "Olivine-carb",
    "ridge":                "Ridge core",
    "discovery_ridge":      "Ridge core",
    "intrusions":           "Intrusions",
    "saponite":             "Saponite",
    "nontronite":           "Nontronite",
    "fe-mg-phyllosilicate": "Fe/Mg-Phyll.",
    "clays":                "Clays",
}


def display_name(cls: str) -> str:
    return CLASS_DISPLAY_NAMES.get(cls, cls.replace("_", " ").title())


# Rows to drop entirely (AV flagged 871C olivine_dunes as a misclick).
EXCLUDE_ROWS = {
    ("FRT0000871C", "olivine_dunes"),
}
EXCLUDE_CUBES = {"TEST_CUBE"}
EXCLUDE_CLASSES = {"session"}

# Classes to collapse into a single (mean) point for visualisation.
COLLAPSE_TO_MEAN = {"discovery_ridge"}

# Marker styling - all classes use the default size now.
MARKER_SIZE_DEFAULT = 12
HIGHLIGHT_CLASSES: set[str] = set()

# Brown 2020 Med Fo box: drawn as a single Rectangle (border only, no
# fill) around the Fo 40-66 region. The asymmetry bounds 3-6 fall in
# Brown 2020's medium-to-coarse grain-size band, with the upper edge
# (chi ~ 5-7) approaching the 1 mm grain-size isoline; the box label
# captures both the Fo range and that grain-size endpoint.
MED_FO_BOX = {
    "x_lo": 1.27, "x_hi": 1.43,
    "y_lo": 3.0,  "y_hi": 6.0,
    "edgecolor": "black", "linewidth": 1.6, "linestyle": "--",
    "label": "Med Fo  (Fo 40-66, ~1 mm grain size)",
}


def setup_style():
    for p in USER_STYLE_PATHS:
        if p.exists():
            try:
                plt.style.use(str(p))
                print(f"[style] using {p}")
                return
            except Exception as e:
                print(f"[style] {p} found but failed: {e}")
    print("[style] no style file found; using matplotlib defaults")


def discover_rois(classes_filter: list[str] | None = None) -> pd.DataFrame:
    rows = []
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
            if not classes_filter and (cube_id, cls) in EXCLUDE_ROWS:
                print(f"  [excluded] {cube_id}/{cls}")
                continue
            csv_path = cls_dir / f"{cube_id}_{cls}_rois.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            df["cube"] = cube_id
            df["class"] = cls
            rows.append(df)
            print(f"  {csv_path.relative_to(SAMPLES_ROOT)}  -> "
                  f"{len(df)} ROIs ({(df['kind']=='target').sum()} targets)")
    if not rows:
        raise SystemExit("no ROI CSVs found under "
                          f"{SAMPLES_ROOT}; nothing to plot")
    full = pd.concat(rows, ignore_index=True)
    # Merge classes per MERGE_CLASSES (e.g. saponite + nontronite ->
    # fe-mg-phyllosilicate) unless the user explicitly requested one of
    # the source classes via --classes.
    if MERGE_CLASSES:
        if classes_filter and any(c in MERGE_CLASSES
                                    for c in classes_filter):
            print("  [merge] skipped because --classes references a "
                  "merged source class explicitly")
        else:
            for src, dst in MERGE_CLASSES.items():
                n = (full["class"] == src).sum()
                if n:
                    print(f"  [merge] {src} ({n} ROIs) -> {dst}")
            full["class"] = full["class"].replace(MERGE_CLASSES)
    return full


def collapse_class_to_mean(df: pd.DataFrame) -> pd.DataFrame:
    """For classes in COLLAPSE_TO_MEAN, replace the per-ROI rows with a
    single synthesized row. Uses pixel-count-weighted means and POOLED
    within-ROI std (sqrt(sum(n_i * sigma_i^2) / sum(n_i))) so the error
    bars represent the heterogeneity of the underlying pixels rather
    than the (often tiny) across-ROI variation. Without pooling, three
    consistent ROIs would yield near-zero error bars and the marker
    looks like a point with no uncertainty.
    """
    def pooled_std(stds, ns):
        s = np.asarray(stds, dtype=float)
        n = np.asarray(ns, dtype=float)
        if n.sum() == 0:
            return float("nan")
        return float(np.sqrt(np.sum(n * s ** 2) / n.sum()))

    out_rows = []
    for cls, sub in df.groupby("class"):
        if cls in COLLAPSE_TO_MEAN and len(sub) >= 2:
            tgt = sub[sub["kind"] == "target"]
            if tgt.empty:
                out_rows.append(sub)
                continue
            ns = tgt["n_pixels"].to_numpy()
            row = {
                "cube": tgt["cube"].mode().iat[0],
                "class": cls,
                "kind": "target",
                "n_pixels": int(ns.sum()),
                "centroid_mean":  float(np.average(tgt["centroid_mean"], weights=ns)),
                "centroid_std":   pooled_std(tgt["centroid_std"], ns),
                "amplitude_mean": float(np.average(tgt["amplitude_mean"], weights=ns)),
                "amplitude_std":  pooled_std(tgt["amplitude_std"], ns),
                "hwhm_mean":      float(np.average(tgt["hwhm_mean"], weights=ns)),
                "hwhm_std":       pooled_std(tgt["hwhm_std"], ns),
                "asymmetry_mean": float(np.average(tgt["asymmetry_mean"], weights=ns)),
                "asymmetry_std":  pooled_std(tgt["asymmetry_std"], ns),
                "n_collapsed":    int(len(tgt)),
            }
            print(f"  [collapse] {cls}: averaged {len(tgt)} target ROIs "
                  f"({int(ns.sum())} px) -> "
                  f"centroid {row['centroid_mean']:.3f} +/- "
                  f"{row['centroid_std']:.3f}  (pooled), asym "
                  f"{row['asymmetry_mean']:.3f} +/- "
                  f"{row['asymmetry_std']:.3f}")
            out_rows.append(pd.DataFrame([row]))
        else:
            out_rows.append(sub)
    return pd.concat(out_rows, ignore_index=True)


def annotate_fo_zones(ax, xmin: float, xmax: float, ymin: float,
                       ymax: float) -> None:
    """No-op. Med Fo box, '~1 mm grain size' label, and the
    'increasing alteration' arrow were removed at AV's request
    (2026-05-01) - he prefers to add any annotations himself in
    post-processing. The function stays in place so the call site in
    render() doesn't need editing if annotations come back later."""
    return


def build_legend(ax, df: pd.DataFrame) -> None:
    classes_present = [c for c in CLASS_DISPLAY_ORDER
                       if c in set(df["class"])]
    cubes_present = sorted(df["cube"].unique())
    # Class legend = stacked rectangles, no spacing between rows.
    class_handles = [
        Patch(facecolor=CLASS_COLORS.get(c, DEFAULT_COLOR),
              edgecolor="black", linewidth=0.5,
              label=display_name(c))
        for c in classes_present
    ]
    leg1 = ax.legend(
        handles=class_handles, title="Class",
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        fontsize=14, title_fontsize=15, frameon=True,
        handlelength=2.2, handleheight=1.6,
        labelspacing=0.0,        # no gap between rows
        borderpad=0.4, handletextpad=0.6,
    )
    ax.add_artist(leg1)

    # Cube legend = marker symbols (these are still shapes, not colors).
    cube_handles = [
        Line2D([0], [0], marker=CUBE_MARKERS.get(c, DEFAULT_MARKER),
               color="w", markerfacecolor="lightgray",
               markeredgecolor="black", markeredgewidth=0.6,
               markersize=12, label=c)
        for c in cubes_present
    ]
    ax.legend(
        handles=cube_handles, title="Cube",
        loc="upper left", bbox_to_anchor=(1.02, 0.40),
        fontsize=14, title_fontsize=15, frameon=True,
    )


def render(df: pd.DataFrame, include_mafic: bool = False) -> dict:
    if not include_mafic:
        df = df[df["kind"] == "target"].copy()
    df = collapse_class_to_mean(df)

    # Square axes layout so the data sits naturally centred in the
    # panel (AV 2026-05-01). Figure is slightly wider than tall to
    # leave room for the right-side legend after set_box_aspect(1).
    fig, ax = plt.subplots(figsize=(11, 9))

    # Draw points + error bars per ROI (or per collapsed group).
    for _, row in df.iterrows():
        cls = row["class"]
        color = CLASS_COLORS.get(cls, DEFAULT_COLOR)
        marker = CUBE_MARKERS.get(row["cube"], DEFAULT_MARKER)
        x = row["centroid_mean"]
        y = row["asymmetry_mean"]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xerr = (row["centroid_std"]
                if np.isfinite(row["centroid_std"]) else 0.0)
        yerr = (row["asymmetry_std"]
                if np.isfinite(row["asymmetry_std"]) else 0.0)

        # Error bars: lighter, no caps - matches AV's dashboard style.
        ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                     fmt="none", ecolor=color, elinewidth=1.0,
                     alpha=0.50, capsize=0, zorder=2)

        # Marker
        ax.plot(x, y, marker=marker, color=color,
                markersize=MARKER_SIZE_DEFAULT,
                markeredgecolor="black", markeredgewidth=0.6,
                linestyle="None",
                alpha=0.9 if row["kind"] == "target" else 0.45,
                zorder=4)

    # Axis labels using unicode symbols (no mathtext blow-up).
    ax.set_xlabel("1 µm band centroid \u03BB\u2080 (µm)")
    ax.set_ylabel("Band asymmetry \u03C7")
    ax.grid(True, alpha=0.25)

    # Data-driven symmetric padding on both axes so the points sit
    # naturally centred inside the square box. Tick LABELS are still
    # restricted to {1.0, 1.1, 1.2, 1.3, 1.4} per AV's earlier spec;
    # the limits give equal breathing room on either side of the data
    # extent.
    cent_lo = (df["centroid_mean"]
                - df["centroid_std"].fillna(0.0)).min()
    cent_hi = (df["centroid_mean"]
                + df["centroid_std"].fillna(0.0)).max()
    asym_lo = (df["asymmetry_mean"]
                - df["asymmetry_std"].fillna(0.0)).min()
    asym_hi = (df["asymmetry_mean"]
                + df["asymmetry_std"].fillna(0.0)).max()
    x_span = max(float(cent_hi - cent_lo), 1e-6)
    y_span = max(float(asym_hi - asym_lo), 1e-6)
    xpad = 0.10 * x_span
    ypad = 0.10 * y_span
    xmin = float(cent_lo) - xpad
    xmax = float(cent_hi) + xpad
    ymin = float(asym_lo) - ypad
    ymax = float(asym_hi) + ypad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_box_aspect(1)
    ax.xaxis.set_major_locator(
        FixedLocator([1.0, 1.1, 1.2, 1.3, 1.4]))

    # Backgrounds + Fo labels at top
    annotate_fo_zones(ax, xmin, xmax, ymin, ymax)

    build_legend(ax, df)
    plt.tight_layout(rect=[0, 0, 0.84, 1])

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    base = "brown_scatter_centroid_asymmetry"
    out_png = FIGURE_DIR / f"{base}.png"
    out_pdf = FIGURE_DIR / f"{base}.pdf"
    out_svg = FIGURE_DIR / f"{base}.svg"
    fig.savefig(out_png, dpi=300, bbox_inches="tight",
                facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_png.relative_to(OUTPUTS_DIR.parent)}")
    print(f"  wrote {out_pdf.relative_to(OUTPUTS_DIR.parent)}")
    print(f"  wrote {out_svg.relative_to(OUTPUTS_DIR.parent)}")
    return {"out_png": str(out_png), "out_pdf": str(out_pdf),
            "out_svg": str(out_svg),
            "n_rows_plotted": int(len(df))}


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--classes", nargs="+", default=None)
    p.add_argument("--include-mafic", action="store_true")
    args = p.parse_args(argv[1:])

    setup_style()
    print("=== Brown centroid-asymmetry scatter ===")
    print(f"reading from: {SAMPLES_ROOT}")
    df = discover_rois(args.classes)
    render(df, include_mafic=args.include_mafic)


if __name__ == "__main__":
    main(sys.argv)
