"""Package the three publication plots into a standalone, cloud-friendly
folder.

For each of:
  - Brown centroid-asymmetry scatter (v9_1_brown_scatter.py)
  - 2-panel library vs CRISM (v9_library_vs_crism.py)
  - Supplement single-panel ratio spectra by class
    (v9_supplement_class_ratios.py)

this builder

  1. Pre-computes the spectra once (splice-correction, ratio averaging,
     library lookup), so the standalone scripts never need to touch the
     CRISM cubes, masks, or any project-internal helper modules.

  2. Dumps the resulting numerical data as plain CSVs into
     cloud_package/data/.

  3. Writes 3 self-contained Python scripts into cloud_package/ that
     only depend on numpy, pandas, matplotlib, and the bundled CSVs.
     Each script resolves paths via Path(__file__).parent so it can be
     run from any working directory.

The output folder cloud_package/ is intended to be uploaded to cloud
storage as a portable set of one-click figure regenerators.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import OUTPUTS_DIR
from crism_io import load_cube_by_id
from v9_per_pair_inspector import (
    load_pairs, apply_splice_correction, DEFAULT_SPLICE_UM,
)
# Reuse the brown-scatter module for its merge / exclude rules so the
# packaged CSV exactly matches what the in-project script plots.
import v9_1_brown_scatter as bs

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_ROOT = OUTPUTS_DIR / "spectra sample locations"
PKG_ROOT = PROJECT_ROOT / "cloud_package"
DATA_DIR = PKG_ROOT / "data"
STYLE_DIR = PKG_ROOT / "style"

LIBRARY_REFLECTANCE = "s07_HY07_Olivine_KI3188_Fo51_lt60um_BECKb_AREF.txt"
LIBRARY_WAVELENGTH = "s07_HY07_Wavelength_HyMap2007_124ch_(microns).txt"

LIBRARY_VS_CRISM_PAIRS = {
    "ridge_3E12":     ("FRT00003E12", "discovery_ridge", "pair", 1),
    "olivine_3E12":   ("FRT00003E12", "olivine",         "average", None),
    "intrusion_A4FC": ("FRT0000A4FC", "intrusions",      "average", None),
}

SUPPLEMENT_CLASSES = ["olivine_dunes", "carbonates", "saponite", "nontronite"]


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STYLE_DIR.mkdir(parents=True, exist_ok=True)


def copy_style() -> None:
    src = Path(__file__).parent / "my_style.mplstyle.txt"
    dst = STYLE_DIR / "my_style.mplstyle.txt"
    shutil.copyfile(src, dst)
    print(f"  copied style -> {dst.relative_to(PKG_ROOT)}")


def build_brown_rois_csv() -> None:
    """Run the existing discover_rois() so merge + exclude rules match
    the in-project plot exactly, then dump as a single CSV."""
    df = bs.discover_rois(classes_filter=None)
    out = DATA_DIR / "brown_rois.csv"
    df.to_csv(out, index=False)
    print(f"  wrote {out.relative_to(PKG_ROOT)}  ({len(df)} rows)")


def _ratio_for_pair(cube: np.ndarray, t_mask: np.ndarray,
                     m_mask: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        t_spec = np.nanmean(cube[:, t_mask], axis=1)
        m_spec = np.nanmean(cube[:, m_mask], axis=1)
    return t_spec / np.where(np.abs(m_spec) > 1e-6, m_spec, np.nan)


def crism_pair_ratio(cube_id: str, name: str, pair_idx: int
                       ) -> tuple[np.ndarray, np.ndarray]:
    pairs = load_pairs(cube_id, name)
    pair = next((p for p in pairs if p[0] == pair_idx), None)
    if pair is None:
        raise SystemExit(f"pair {pair_idx} not found in {cube_id}/{name}")
    _, t_info, m_info = pair
    cube, wav, _, _ = load_cube_by_id(cube_id)
    cube = cube[:, ::-1, ::-1].copy()
    wav = wav.astype(np.float32)
    t_mask = np.load(t_info["mask_path"])[::-1, ::-1]
    m_mask = np.load(m_info["mask_path"])[::-1, ::-1]
    ratio = _ratio_for_pair(cube, t_mask, m_mask)
    ratio_c, _, _ = apply_splice_correction(wav, ratio,
                                              splice_um=DEFAULT_SPLICE_UM)
    return wav, ratio_c


def crism_class_average(cube_id: str, name: str
                          ) -> tuple[np.ndarray, np.ndarray, int]:
    pairs = load_pairs(cube_id, name)
    if not pairs:
        raise SystemExit(f"no pairs for {cube_id}/{name}")
    cube, wav, _, _ = load_cube_by_id(cube_id)
    cube = cube[:, ::-1, ::-1].copy()
    wav = wav.astype(np.float32)
    stack = []
    for idx, t_info, m_info in pairs:
        t_mask = np.load(t_info["mask_path"])[::-1, ::-1]
        m_mask = np.load(m_info["mask_path"])[::-1, ::-1]
        ratio = _ratio_for_pair(cube, t_mask, m_mask)
        ratio_c, _, _ = apply_splice_correction(
            wav, ratio, splice_um=DEFAULT_SPLICE_UM)
        stack.append(ratio_c)
    arr = np.array(stack)
    with np.errstate(invalid="ignore"):
        mean_ratio = np.nanmean(arr, axis=0)
    return wav, mean_ratio, len(pairs)


def build_library_vs_crism_csvs() -> None:
    """3 splice-corrected CRISM ratios + library spectrum, both as CSV."""
    common_wav: np.ndarray | None = None
    cols: dict[str, np.ndarray] = {}
    for label, (cube_id, name, mode, pidx) in LIBRARY_VS_CRISM_PAIRS.items():
        if mode == "pair":
            wav, ratio = crism_pair_ratio(cube_id, name, pidx)
        else:
            wav, ratio, n_pairs = crism_class_average(cube_id, name)
            print(f"    {label}: averaged {n_pairs} pair(s)")
        if common_wav is None:
            common_wav = wav
        elif not np.allclose(common_wav, wav):
            raise SystemExit("CRISM wavelength axes differ across cubes")
        cols[label] = ratio
    out = DATA_DIR / "crism_ratios.csv"
    df = pd.DataFrame({"wavelength_um": common_wav, **cols})
    df.to_csv(out, index=False)
    print(f"  wrote {out.relative_to(PKG_ROOT)}  "
          f"({len(df)} bands x {len(cols)} ratios)")

    # Library spectrum: parse wav + reflectance text files, save as one
    # CSV with a header label embedded as a sidecar JSON.
    refl_lines = (SAMPLES_ROOT / LIBRARY_REFLECTANCE).read_text().splitlines()
    header = refl_lines[0].strip()
    refl = np.array([float(line.strip()) for line in refl_lines[1:]
                      if line.strip()], dtype=np.float64)
    wav_lines = (SAMPLES_ROOT / LIBRARY_WAVELENGTH).read_text().splitlines()
    wav_vals: list[float] = []
    for line in wav_lines:
        s = line.strip()
        if not s:
            continue
        try:
            wav_vals.append(float(s))
        except ValueError:
            continue
    lib_wav = np.array(wav_vals, dtype=np.float64)
    if len(lib_wav) != len(refl):
        print(f"  [warn] library wav count {len(lib_wav)} != refl "
              f"count {len(refl)}")
        n = min(len(lib_wav), len(refl))
        lib_wav = lib_wav[:n]
        refl = refl[:n]
    # Trim header for legend display (same logic as the in-project plot).
    short = header.split(":", 1)[1].strip() if ":" in header else header
    for tag in (" BECKb AREF", " BECK AREF", " AREF", " BECKb"):
        if short.endswith(tag):
            short = short[: -len(tag)].rstrip()
            break
    short = " ".join(short.split())

    lib_csv = DATA_DIR / "library_olivine.csv"
    pd.DataFrame({"wavelength_um": lib_wav,
                  "reflectance":   refl}).to_csv(lib_csv, index=False)
    meta = {"raw_header": header, "legend_label": short[:70],
            "n_bands": int(len(lib_wav))}
    (DATA_DIR / "library_olivine_meta.json").write_text(
        json.dumps(meta, indent=2))
    print(f"  wrote {lib_csv.relative_to(PKG_ROOT)} (+ meta json)")


def build_class_mean_ratios_csv() -> None:
    """Splice-correct every (target/mafic) pair across every cube for
    each class in SUPPLEMENT_CLASSES, mean across pairs, save as CSV.

    Mirrors collect_class_means() in v9_supplement_class_ratios.py.
    """
    common_wav: np.ndarray | None = None
    by_class: dict[str, list[np.ndarray]] = {c: [] for c in SUPPLEMENT_CLASSES}

    for cube_dir in sorted(SAMPLES_ROOT.iterdir()):
        if not cube_dir.is_dir():
            continue
        cube_id = cube_dir.name
        if cube_id == "figures":
            continue
        present = [c for c in SUPPLEMENT_CLASSES if (cube_dir / c).is_dir()]
        if not present:
            continue
        print(f"  loading {cube_id}...")
        cube, wav, _, _ = load_cube_by_id(cube_id)
        cube = cube[:, ::-1, ::-1].copy()
        wav = wav.astype(np.float32)
        if common_wav is None:
            common_wav = wav
        elif not np.allclose(common_wav, wav):
            raise SystemExit("CRISM wavelength axes differ across cubes")
        for cls in present:
            try:
                pairs = load_pairs(cube_id, cls)
            except SystemExit:
                continue
            if not pairs:
                continue
            for _, t_info, m_info in pairs:
                t_mask = np.load(t_info["mask_path"])[::-1, ::-1]
                m_mask = np.load(m_info["mask_path"])[::-1, ::-1]
                ratio = _ratio_for_pair(cube, t_mask, m_mask)
                ratio_c, _, _ = apply_splice_correction(
                    wav, ratio, splice_um=DEFAULT_SPLICE_UM)
                by_class[cls].append(ratio_c)
            print(f"    {cls}: {len(pairs)} pair(s)")

    counts: dict[str, int] = {}
    means: dict[str, np.ndarray] = {}
    for cls in SUPPLEMENT_CLASSES:
        if not by_class[cls]:
            print(f"  [warn] class '{cls}' has no pairs")
            continue
        arr = np.array(by_class[cls])
        with np.errstate(invalid="ignore"):
            means[cls] = np.nanmean(arr, axis=0)
        counts[cls] = len(by_class[cls])
    if common_wav is None:
        raise SystemExit("no usable cubes")
    out = DATA_DIR / "class_mean_ratios.csv"
    df = pd.DataFrame({"wavelength_um": common_wav,
                        **{c: means[c] for c in SUPPLEMENT_CLASSES
                           if c in means}})
    df.to_csv(out, index=False)
    print(f"  wrote {out.relative_to(PKG_ROOT)}  "
          f"({len(df)} bands x {len(means)} classes)")
    (DATA_DIR / "class_mean_ratios_counts.json").write_text(
        json.dumps(counts, indent=2))
    print(f"  wrote class_mean_ratios_counts.json")


# ----- Standalone plot scripts written into cloud_package/ ---------------

PLOT_BROWN = r'''"""Standalone Brown centroid-asymmetry scatter.

Reads cloud_package/data/brown_rois.csv and renders the publication
figure. No project-internal imports - just numpy, pandas, matplotlib.

Run from anywhere:
  python plot_brown_scatter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_CSV = HERE / "data" / "brown_rois.csv"
STYLE_FILE = HERE / "style" / "my_style.mplstyle.txt"
FIGURE_DIR = HERE / "figures"

CLASS_COLORS = {
    "olivine":              "#1b9e77",
    "olivine_dunes":        "#7cb342",
    "ol-carb":              "#d95f02",
    "carbonates":           "#d95f02",
    "ridge":                "#000000",
    "discovery_ridge":      "#000000",
    "intrusions":           "#555555",
    "saponite":             "#a6761d",
    "nontronite":           "#e6ab02",
    "fe-mg-phyllosilicate": "#e6ab02",
    "clays":                "#66a61e",
}
DEFAULT_COLOR = "#666666"

CUBE_MARKERS = {
    "FRT00003E12": "o",
    "FRT0000871C": "D",
    "FRT0000A4FC": "s",
    "FRT0000B438": "^",
}
DEFAULT_MARKER = "X"

CLASS_DISPLAY_ORDER = [
    "olivine", "olivine_dunes", "ol-carb", "carbonates",
    "fe-mg-phyllosilicate", "ridge", "discovery_ridge",
    "intrusions", "clays",
]
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

COLLAPSE_TO_MEAN = {"discovery_ridge"}
MARKER_SIZE_DEFAULT = 12


def display_name(cls):
    return CLASS_DISPLAY_NAMES.get(cls, cls.replace("_", " ").title())


def collapse_class_to_mean(df):
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
                "cube":           tgt["cube"].mode().iat[0],
                "class":          cls,
                "kind":           "target",
                "n_pixels":       int(ns.sum()),
                "centroid_mean":  float(np.average(tgt["centroid_mean"], weights=ns)),
                "centroid_std":   pooled_std(tgt["centroid_std"], ns),
                "amplitude_mean": float(np.average(tgt["amplitude_mean"], weights=ns)),
                "amplitude_std":  pooled_std(tgt["amplitude_std"], ns),
                "hwhm_mean":      float(np.average(tgt["hwhm_mean"], weights=ns)),
                "hwhm_std":       pooled_std(tgt["hwhm_std"], ns),
                "asymmetry_mean": float(np.average(tgt["asymmetry_mean"], weights=ns)),
                "asymmetry_std":  pooled_std(tgt["asymmetry_std"], ns),
            }
            out_rows.append(pd.DataFrame([row]))
        else:
            out_rows.append(sub)
    return pd.concat(out_rows, ignore_index=True)


def build_legend(ax, df):
    classes_present = [c for c in CLASS_DISPLAY_ORDER
                       if c in set(df["class"])]
    cubes_present = sorted(df["cube"].unique())
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
        labelspacing=0.0, borderpad=0.4, handletextpad=0.6,
    )
    ax.add_artist(leg1)
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


def main():
    if STYLE_FILE.exists():
        try:
            plt.style.use(str(STYLE_FILE))
        except Exception:
            pass
    if not DATA_CSV.exists():
        raise SystemExit(f"missing data file: {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)
    df = df[df["kind"] == "target"].copy()
    df = collapse_class_to_mean(df)

    fig, ax = plt.subplots(figsize=(11, 9))

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
        ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                     fmt="none", ecolor=color, elinewidth=1.0,
                     alpha=0.50, capsize=0, zorder=2)
        ax.plot(x, y, marker=marker, color=color,
                markersize=MARKER_SIZE_DEFAULT,
                markeredgecolor="black", markeredgewidth=0.6,
                linestyle="None",
                alpha=0.9 if row["kind"] == "target" else 0.45,
                zorder=4)

    ax.set_xlabel("1 \u00b5m band centroid \u03BB\u2080 (\u00b5m)")
    ax.set_ylabel("Band asymmetry \u03C7")
    ax.grid(True, alpha=0.25)

    cent_lo = (df["centroid_mean"] - df["centroid_std"].fillna(0.0)).min()
    cent_hi = (df["centroid_mean"] + df["centroid_std"].fillna(0.0)).max()
    asym_lo = (df["asymmetry_mean"] - df["asymmetry_std"].fillna(0.0)).min()
    asym_hi = (df["asymmetry_mean"] + df["asymmetry_std"].fillna(0.0)).max()
    x_span = max(float(cent_hi - cent_lo), 1e-6)
    y_span = max(float(asym_hi - asym_lo), 1e-6)
    xpad = 0.10 * x_span
    ypad = 0.10 * y_span
    ax.set_xlim(float(cent_lo) - xpad, float(cent_hi) + xpad)
    ax.set_ylim(float(asym_lo) - ypad, float(asym_hi) + ypad)
    ax.set_box_aspect(1)
    ax.xaxis.set_major_locator(FixedLocator([1.0, 1.1, 1.2, 1.3, 1.4]))

    build_legend(ax, df)
    plt.tight_layout(rect=[0, 0, 0.84, 1])

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    base = "brown_scatter_centroid_asymmetry"
    for ext in ("png", "pdf", "svg"):
        out = FIGURE_DIR / f"{base}.{ext}"
        kw = {"dpi": 300} if ext == "png" else {}
        fig.savefig(out, bbox_inches="tight", facecolor="white", **kw)
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
'''


PLOT_LIBVCRISM = r'''"""Standalone two-panel figure: CRISM ratios vs library olivine spectrum.

Reads the pre-computed splice-corrected ratios from
cloud_package/data/crism_ratios.csv and the library spectrum from
cloud_package/data/library_olivine.csv.

Run from anywhere:
  python plot_library_vs_crism.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
STYLE_FILE = HERE / "style" / "my_style.mplstyle.txt"
FIGURE_DIR = HERE / "figures"

CRISM_CSV = DATA_DIR / "crism_ratios.csv"
LIB_CSV = DATA_DIR / "library_olivine.csv"
LIB_META = DATA_DIR / "library_olivine_meta.json"

XMIN, XMAX = 0.4, 2.6

# Per-curve scaling so the CRISM ratios separate vertically.
SCALES = {
    "ridge_3E12":     0.6,
    "olivine_3E12":   0.75,
    "intrusion_A4FC": 0.8,
}
LABELS = {
    "ridge_3E12":     "Ridge core-3E12",
    "olivine_3E12":   "Olivine-3E12",
    "intrusion_A4FC": "Intrusion-A4FC",
}
COLORS = {
    "ridge_3E12":     "#d62728",
    "olivine_3E12":   "#1b9e77",
    "intrusion_A4FC": "#e7298a",
}


def main():
    if STYLE_FILE.exists():
        try:
            plt.style.use(str(STYLE_FILE))
        except Exception:
            pass
    # Bump sizes so the figure stays readable at A4 print size.
    plt.rcParams.update({
        "axes.labelsize":  22,
        "axes.titlesize":  24,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 18,
    })
    for p in (CRISM_CSV, LIB_CSV):
        if not p.exists():
            raise SystemExit(f"missing data file: {p}")

    crism = pd.read_csv(CRISM_CSV)
    crism_wav = crism["wavelength_um"].to_numpy()
    lib = pd.read_csv(LIB_CSV)
    lib_wav = lib["wavelength_um"].to_numpy()
    lib_refl = lib["reflectance"].to_numpy()
    lib_label = "Library olivine"
    if LIB_META.exists():
        try:
            lib_label = json.loads(LIB_META.read_text()).get(
                "legend_label", lib_label)
        except Exception:
            pass

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 7.5))

    # Panel A
    scaled_arrays = []
    for key in ("ridge_3E12", "olivine_3E12", "intrusion_A4FC"):
        y = crism[key].to_numpy() * SCALES[key]
        axA.plot(crism_wav, y, color=COLORS[key], lw=2.0,
                  label=LABELS[key])
        scaled_arrays.append(y)
    axA.set_xlim(XMIN, XMAX)
    axA.set_xlabel(r"Wavelength ($\mu$m)")
    axA.set_ylabel("Ratioed reflectance (scaled and offset for clarity)")
    axA.set_title("(a) CRISM ratio spectra", loc="left")
    axA.legend(loc="best", framealpha=0.92)
    all_y = np.concatenate(scaled_arrays)
    finite = all_y[np.isfinite(all_y)]
    if finite.size > 0:
        y_lo = float(finite.min())
        y_hi = float(finite.max())
        pad = 0.05 * (y_hi - y_lo)
        axA.set_ylim(y_lo - pad, y_hi + pad)
    axA.xaxis.set_major_locator(MultipleLocator(0.5))
    axA.minorticks_off()
    axA.grid(True, which="major", alpha=0.3, linewidth=0.6)
    axA.set_box_aspect(1)

    # Panel B
    axB.plot(lib_wav, lib_refl, color="black", lw=2.0,
              label=lib_label[:70])
    axB.set_xlim(XMIN, XMAX)
    axB.set_xlabel(r"Wavelength ($\mu$m)")
    axB.set_ylabel("Reflectance")
    axB.set_title("(b) Library spectrum", loc="left")
    axB.legend(loc="best", framealpha=0.92)
    axB.xaxis.set_major_locator(MultipleLocator(0.5))
    axB.minorticks_off()
    axB.grid(True, which="major", alpha=0.3, linewidth=0.6)
    axB.set_box_aspect(1)

    plt.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    base = "ridge_olivine_intrusion_vs_olivine_library"
    for ext in ("png", "pdf", "svg"):
        out = FIGURE_DIR / f"{base}.{ext}"
        kw = {"dpi": 300} if ext == "png" else {}
        fig.savefig(out, bbox_inches="tight", facecolor="white",
                    pad_inches=0.05, **kw)
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
'''


PLOT_SUPPLEMENT = r'''"""Standalone supplement figure: ratio spectra by class, single panel.

Reads cloud_package/data/class_mean_ratios.csv and renders one
splice-corrected mean ratio per class, stacked with additive offsets.

Run from anywhere:
  python plot_supplement_class_ratios.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
STYLE_FILE = HERE / "style" / "my_style.mplstyle.txt"
FIGURE_DIR = HERE / "figures"

DATA_CSV = DATA_DIR / "class_mean_ratios.csv"
COUNTS_JSON = DATA_DIR / "class_mean_ratios_counts.json"

CLASS_ORDER = ["olivine_dunes", "carbonates", "saponite", "nontronite"]
CLASS_DISPLAY = {
    "olivine_dunes": "Olivine dunes",
    "carbonates":    "Olivine-carb",
    "saponite":      "Fe/Mg-phyllosilicate 1",
    "nontronite":    "Fe/Mg-phyllosilicate 2",
}
CLASS_COLORS = {
    "olivine_dunes": "#7cb342",
    "carbonates":    "#d95f02",
    "saponite":      "#a6761d",
    "nontronite":    "#e6ab02",
}
CLASS_SCALE = {
    "olivine_dunes": 1.0,
    "carbonates":    1.0,
    "saponite":      1.0,
    "nontronite":    1.0,
}

XMIN, XMAX = 0.4, 2.6
OFFSET_STEP = 0.5
REF_LINES_UM = (1.40, 1.90, 2.30, 2.50)


def main():
    if STYLE_FILE.exists():
        try:
            plt.style.use(str(STYLE_FILE))
        except Exception:
            pass
    if not DATA_CSV.exists():
        raise SystemExit(f"missing data file: {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)
    wav = df["wavelength_um"].to_numpy()
    means = {c: df[c].to_numpy() for c in CLASS_ORDER if c in df.columns}
    counts = {}
    if COUNTS_JSON.exists():
        try:
            counts = json.loads(COUNTS_JSON.read_text())
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(8.5, 11.5))

    for w_ref in REF_LINES_UM:
        if XMIN <= w_ref <= XMAX:
            ax.axvline(w_ref, color="0.45", lw=0.7, ls="--",
                        alpha=0.55, zorder=0)

    offsets = {c: -OFFSET_STEP * i for i, c in enumerate(CLASS_ORDER)}

    def label_for(cls):
        s = CLASS_SCALE.get(cls, 1.0)
        if abs(s - 1.0) < 1e-3:
            return CLASS_DISPLAY[cls]
        return f"{CLASS_DISPLAY[cls]} (\u00d7{s:g})"

    for cls in CLASS_ORDER:
        if cls not in means:
            continue
        y = means[cls] * CLASS_SCALE.get(cls, 1.0) + offsets[cls]
        ax.plot(wav, y, color=CLASS_COLORS[cls], lw=2.0,
                 label=label_for(cls))

    ax.set_xlim(XMIN, XMAX)
    ax.set_xlabel(r"Wavelength ($\mu$m)")
    scaled_any = any(abs(s - 1.0) > 1e-3 for s in CLASS_SCALE.values())
    if abs(OFFSET_STEP) < 1e-9 and not scaled_any:
        ax.set_ylabel("Ratioed reflectance")
    else:
        ax.set_ylabel("Ratioed reflectance (scaled and offset for clarity)")
    ax.legend(loc="best", framealpha=0.92)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.minorticks_off()
    ax.grid(True, which="major", alpha=0.3, linewidth=0.6)

    all_y = []
    for cls in CLASS_ORDER:
        if cls in means:
            all_y.append(means[cls] * CLASS_SCALE.get(cls, 1.0) + offsets[cls])
    if all_y:
        flat = np.concatenate(all_y)
        finite = flat[np.isfinite(flat)]
        if finite.size > 0:
            y_lo = float(finite.min())
            y_hi = float(finite.max())
            pad = 0.05 * (y_hi - y_lo)
            ax.set_ylim(y_lo - pad, y_hi + pad)

    plt.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    base = "supplement_ratio_spectra_classes"
    for ext in ("png", "svg"):
        out = FIGURE_DIR / f"{base}.{ext}"
        kw = {"dpi": 300} if ext == "png" else {}
        fig.savefig(out, bbox_inches="tight", facecolor="white",
                    pad_inches=0.05, **kw)
        print(f"wrote {out}")
    plt.close(fig)
    if counts:
        for cls in CLASS_ORDER:
            if cls in counts:
                print(f"  {cls:<14s} -> {CLASS_DISPLAY[cls]:<22s}  "
                      f"averaged over {counts[cls]} pair(s)")


if __name__ == "__main__":
    main()
'''


README_TEXT = """# Cloud-portable plot package

Three standalone Python scripts that regenerate the publication figures
from pre-computed CSVs. No CRISM cubes, ROI masks, or project-internal
modules required.

## Layout

    cloud_package/
      data/
        brown_rois.csv                 - per-ROI Brown band-fit results
        crism_ratios.csv               - 3 splice-corrected CRISM ratios
        library_olivine.csv            - HyMap library olivine spectrum
        library_olivine_meta.json      - library legend label / header
        class_mean_ratios.csv          - mean splice-corrected ratio per class
        class_mean_ratios_counts.json  - n_pairs averaged per class
      style/
        my_style.mplstyle.txt          - matplotlib style sheet
      figures/                         - created on first run
      plot_brown_scatter.py
      plot_library_vs_crism.py
      plot_supplement_class_ratios.py

## Dependencies

    python >= 3.10
    numpy
    pandas
    matplotlib

## Usage

From anywhere:

    python /path/to/cloud_package/plot_brown_scatter.py
    python /path/to/cloud_package/plot_library_vs_crism.py
    python /path/to/cloud_package/plot_supplement_class_ratios.py

Each script writes PNG/PDF/SVG (or PNG/SVG for the supplement) into
`cloud_package/figures/`.
"""


def write_standalone_scripts() -> None:
    (PKG_ROOT / "plot_brown_scatter.py").write_text(PLOT_BROWN, encoding="utf-8")
    (PKG_ROOT / "plot_library_vs_crism.py").write_text(PLOT_LIBVCRISM, encoding="utf-8")
    (PKG_ROOT / "plot_supplement_class_ratios.py").write_text(PLOT_SUPPLEMENT, encoding="utf-8")
    (PKG_ROOT / "README.md").write_text(README_TEXT, encoding="utf-8")
    for name in ("plot_brown_scatter.py", "plot_library_vs_crism.py",
                  "plot_supplement_class_ratios.py", "README.md"):
        print(f"  wrote {name}")


def main(argv: list[str]) -> None:
    print("=== Building cloud_package/ ===")
    ensure_dirs()
    copy_style()
    print("[1/4] brown ROI CSV...")
    build_brown_rois_csv()
    print("[2/4] library + 3 CRISM ratios...")
    build_library_vs_crism_csvs()
    print("[3/4] class mean ratios...")
    build_class_mean_ratios_csv()
    print("[4/4] standalone scripts + README...")
    write_standalone_scripts()
    print()
    print(f"done -> {PKG_ROOT}")


if __name__ == "__main__":
    main(sys.argv)
