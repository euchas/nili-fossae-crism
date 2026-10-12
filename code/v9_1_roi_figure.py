"""v9.1 ROI figure - render all saved ROI sampling locations on a cube
background, on a single PNG/PDF for the supplement.

Reads:
  outputs/spectra sample locations/<cube>/<name>/<cube>_<name>_rois.csv
  ... and the mask .npy files referenced by each row's mask_file column.

Output (in the same <cube>/<name>/ folder):
  <cube>_<name>_locations.png       300 dpi
  <cube>_<name>_locations.pdf       vector

Usage:
  python code/v9_1_roi_figure.py FRT00003E12 olivine
  python code/v9_1_roi_figure.py FRT00003E12 olivine --bg car
  python code/v9_1_roi_figure.py FRT00003E12 olivine --labels labels.csv

--bg picks the background:
  olindex (default)   single-band OLINDEX3, inferno colormap
  car                 CAR carbonate composite (R=D2300 G=BD2500_2 B=BD1900R2)
  brown               Brown RGB (R=centroid G=amplitude B=asymmetry)

--labels takes a 2-column CSV (roi_label,display_label) that overrides
the default labels for any matching rows. Rows not listed in the
override CSV use the default short label (cube_id stripped from the
front, so 'FRT00003E12_target1' -> 'target1'). This is the editable
label mechanism: just open the CSV, edit one column, re-run.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from config import CUBE_IDS, OUTPUTS_DIR
from crism_io import load_cube_by_id
from vb_parameters import V4_VB_PARAM_ORDER

PAIR_PALETTE = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e",
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                "#bcbd22", "#17becf"]
SAMPLES_ROOT = OUTPUTS_DIR / "spectra sample locations"
V4_ROOT = OUTPUTS_DIR / "_v4_vb_pca_brown_rgb"

CAR_CHANNELS = {"R": "D2300", "G": "BD2500_2", "B": "BD1900R2"}
BROWN_STRETCH_FULLSCENE = {
    "centroid":  (1.014, 1.379),
    "amplitude": (0.116, 0.335),
    "asymmetry": (0.525, 4.439),
}


def resolve_cube_id(arg: str) -> str:
    a = arg.upper().strip()
    if a in CUBE_IDS:
        return a
    for cid in CUBE_IDS:
        if cid.endswith(a):
            return cid
    raise SystemExit(f"unknown cube {arg!r}; known: {CUBE_IDS}")


def load_rois(rois_csv: Path) -> list[dict]:
    rows = []
    with open(rois_csv, newline="") as f:
        for r in csv.DictReader(f):
            r["pair_idx"] = int(r["pair_idx"])
            r["n_pixels"] = int(r["n_pixels"])
            for c in ("bbox_x_lo", "bbox_y_lo", "bbox_x_hi", "bbox_y_hi"):
                r[c] = int(r[c])
            for c in ("centroid_mean", "centroid_std",
                      "amplitude_mean", "amplitude_std",
                      "hwhm_mean", "hwhm_std",
                      "asymmetry_mean", "asymmetry_std"):
                r[c] = float(r[c])
            rows.append(r)
    return rows


def load_label_overrides(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if "roi_label" in r and "display_label" in r:
                out[r["roi_label"]] = r["display_label"]
    return out


def short_label(roi_label: str, cube_id: str) -> str:
    """Strip the leading <cube_id>_ if present, otherwise return as-is."""
    if roi_label.startswith(cube_id + "_"):
        return roi_label[len(cube_id) + 1:]
    return roi_label


def build_olindex_background(cube_id: str) -> tuple[np.ndarray, dict]:
    v4_vb_path = V4_ROOT / cube_id / "arrays" / f"{cube_id}_VB_stack_v4.npy"
    if not v4_vb_path.exists():
        raise SystemExit(f"missing v4 VB stack: {v4_vb_path}")
    stack = np.load(v4_vb_path)[::-1, ::-1, :].copy()
    idx = V4_VB_PARAM_ORDER.index("OLINDEX3")
    arr = stack[..., idx].astype(np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size > 10:
        vmin, vmax = (float(v) for v in np.percentile(finite, [2, 98]))
    else:
        vmin, vmax = 0.0, 1.0
    return arr, {"vmin": vmin, "vmax": vmax}


def build_car_background(cube_id: str) -> np.ndarray:
    v4_vb_path = V4_ROOT / cube_id / "arrays" / f"{cube_id}_VB_stack_v4.npy"
    stack = np.load(v4_vb_path)[::-1, ::-1, :].copy()
    h, w, _ = stack.shape
    out = np.zeros((h, w, 3), dtype=np.float32)
    for k, name in enumerate(("R", "G", "B")):
        idx = V4_VB_PARAM_ORDER.index(CAR_CHANNELS[name])
        ch = stack[..., idx].astype(np.float32)
        finite = ch[np.isfinite(ch)]
        if finite.size > 10:
            lo, hi = np.percentile(finite, [2, 98])
        else:
            lo, hi = 0.0, 1.0
        out[..., k] = np.clip((ch - lo) / max(hi - lo, 1e-9), 0, 1)
        out[..., k][np.isnan(ch)] = 0.0
    return out


def build_brown_background(cube_id: str) -> np.ndarray:
    """Build Brown RGB from the cached Brown v5 .npy maps."""
    _, _, brown, _ = load_cube_by_id(cube_id)
    H, W = brown["centroid"].shape
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    valid = (np.isfinite(brown["centroid"])
             & np.isfinite(brown["amplitude"])
             & np.isfinite(brown["asymmetry"])
             & np.isfinite(brown["hwhm"]))
    for k, name in enumerate(("centroid", "amplitude", "asymmetry")):
        full = np.where(valid, brown[name].astype(np.float32), np.nan)
        lo, hi = BROWN_STRETCH_FULLSCENE[name]
        chan = np.clip((full - lo) / (hi - lo), 0, 1).astype(np.float32)
        chan[np.isnan(full)] = 0.0
        rgba[..., k] = chan
    rgba[..., 3] = valid.astype(np.float32)
    rgba = rgba[::-1, ::-1, :]  # display flip to match the explorer
    return rgba


def render(cube_id: str, name: str, bg: str = "olindex",
            labels_csv: Path | None = None) -> dict:
    name_dir = SAMPLES_ROOT / cube_id / name
    rois_csv = name_dir / f"{cube_id}_{name}_rois.csv"
    if not rois_csv.exists():
        raise SystemExit(f"no rois CSV at {rois_csv}")
    rows = load_rois(rois_csv)
    if not rows:
        raise SystemExit(f"rois CSV is empty: {rois_csv}")
    overrides = load_label_overrides(labels_csv)

    print(f"  rois csv: {rois_csv}")
    print(f"  found {len(rows)} ROI(s)")
    if overrides:
        print(f"  loaded {len(overrides)} label override(s)")

    # Build background
    if bg == "olindex":
        bg_arr, bg_kw = build_olindex_background(cube_id)
        H, W = bg_arr.shape
    elif bg == "car":
        bg_arr = build_car_background(cube_id)
        H, W = bg_arr.shape[:2]
        bg_kw = {}
    elif bg == "brown":
        bg_arr = build_brown_background(cube_id)
        H, W = bg_arr.shape[:2]
        bg_kw = {}
    else:
        raise SystemExit(f"unknown --bg {bg!r}")

    # Figure
    fig, ax = plt.subplots(figsize=(11, 9), facecolor="white")
    if bg == "olindex":
        ax.imshow(bg_arr, cmap="inferno", interpolation="nearest", **bg_kw)
    else:
        ax.imshow(bg_arr, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Outlines + labels per ROI
    for r in rows:
        color = PAIR_PALETTE[(r["pair_idx"] - 1) % len(PAIR_PALETTE)]
        x_lo, y_lo = r["bbox_x_lo"], r["bbox_y_lo"]
        x_hi, y_hi = r["bbox_x_hi"], r["bbox_y_hi"]
        width = x_hi - x_lo + 1
        height = y_hi - y_lo + 1
        ls = "-" if r["kind"] == "target" else "--"
        lw = 1.6 if r["kind"] == "target" else 1.4
        rect = Rectangle(
            (x_lo - 0.5, y_lo - 0.5), width, height,
            fill=False, edgecolor="white", linewidth=lw, linestyle=ls,
            zorder=4,
        )
        ax.add_patch(rect)
        # If pixel-mode ROI (often very small), also splash the actual pixels.
        if r["mode"] == "pixel":
            mask_path = name_dir / r["mask_file"]
            if mask_path.exists():
                mask = np.load(mask_path)[::-1, ::-1]
                ys, xs = np.where(mask)
                ax.scatter(xs, ys, s=20, c=color, marker="o",
                            alpha=0.9, edgecolors="black",
                            linewidths=0.5, zorder=5)
        # Label
        label = overrides.get(r["roi_label"], short_label(r["roi_label"],
                                                            cube_id))
        cx = (x_lo + x_hi) / 2.0
        cy = y_hi + 1.0
        ax.text(
            cx, cy, label,
            color="black", ha="center", va="top",
            fontsize=10, fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      alpha=0.95, edgecolor="black", lw=0.8),
        )

    bg_title = {
        "olindex": "OLINDEX3 (olivine 1 um band depth)",
        "car": "CAR composite (R=D2300, G=BD2500_2, B=BD1900R2)",
        "brown": "Brown RGB (R=centroid, G=amplitude, B=asymmetry)",
    }[bg]
    ax.set_title(
        f"{cube_id}  -  {name} ROI sampling locations  |  {bg_title}",
        fontsize=11,
    )

    plt.tight_layout()
    out_png = name_dir / f"{cube_id}_{name}_locations.png"
    out_pdf = name_dir / f"{cube_id}_{name}_locations.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight",
                facecolor="white", pad_inches=0.05)
    fig.savefig(out_pdf, bbox_inches="tight",
                facecolor="white", pad_inches=0.05)
    plt.close(fig)
    print(f"  wrote {out_png.name}")
    print(f"  wrote {out_pdf.name}")
    return {"out_png": str(out_png), "out_pdf": str(out_pdf),
            "n_rois": len(rows)}


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cube", help="cube id (full FRT0000XXXX or short)")
    parser.add_argument("name", help="session name slug "
                                       "(e.g. carbonates, olivine, ridges)")
    parser.add_argument("--bg", default="olindex",
                        choices=["olindex", "car", "brown"],
                        help="background image (default: olindex)")
    parser.add_argument("--labels", default=None,
                        help="optional CSV with roi_label,display_label "
                             "columns to override the default short labels")
    args = parser.parse_args(argv[1:])
    cube_id = resolve_cube_id(args.cube)
    labels_path = Path(args.labels) if args.labels else None
    if labels_path is not None and not labels_path.exists():
        raise SystemExit(f"labels CSV not found: {labels_path}")
    print(f"=== ROI figure: {cube_id} / {args.name} (bg={args.bg}) ===")
    render(cube_id, args.name, bg=args.bg, labels_csv=labels_path)


if __name__ == "__main__":
    main(sys.argv)
