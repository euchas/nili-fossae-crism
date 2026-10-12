"""Supplement figure - ratio spectra by class, single panel.

Plots one splice-corrected average ratio spectrum per class on a single
elongated A4 panel. The four classes (carbonates, olivine_dunes,
saponite, nontronite) are renamed for the legend and offset vertically
for clarity. Discovery_ridge is intentionally excluded.

Class -> legend label mapping:
  carbonates    -> Carbonate
  olivine_dunes -> Olivine dunes
  saponite      -> Fe/Mg-phyllosilicate 1
  nontronite    -> Fe/Mg-phyllosilicate 2

For each class, every (target, mafic) pair across every cube is loaded,
splice-corrected at ~1 um (S/L detector join), then averaged. The
averaged curves are stacked with additive offsets so they don't pile
on top of each other.

Style + font are inherited from code/my_style.mplstyle.txt (same as
the main library-vs-CRISM figure).

Output: outputs/spectra sample locations/figures/
  supplement_ratio_spectra_classes.png
  supplement_ratio_spectra_classes.svg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

from config import OUTPUTS_DIR
from crism_io import load_cube_by_id
from v9_per_pair_inspector import (
    load_pairs, apply_splice_correction, DEFAULT_SPLICE_UM,
)

SAMPLES_ROOT = OUTPUTS_DIR / "spectra sample locations"
FIGURE_DIR = SAMPLES_ROOT / "figures"
USER_STYLE = Path(__file__).parent / "my_style.mplstyle.txt"

# Order from top to bottom (visual stack).
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

# Per-class multiplicative scale applied BEFORE the additive offset.
# All 1.0 by default (AV: 2026-05-01 - drop the phyllo x2 amplifier
# and just rely on the offset stack to spread the curves vertically).
CLASS_SCALE = {
    "olivine_dunes": 1.0,
    "carbonates":    1.0,
    "saponite":      1.0,
    "nontronite":    1.0,
}

EXCLUDE_CUBES = {"TEST_CUBE"}
DEFAULT_OFFSET_STEP = 0.5    # additive offset between consecutive curves

# Reference vertical lines at diagnostic absorption wavelengths.
REF_LINES_UM = (1.40, 1.90, 2.30, 2.50)


def setup_style():
    if USER_STYLE.exists():
        try:
            plt.style.use(str(USER_STYLE))
            print(f"[style] using {USER_STYLE}")
            return
        except Exception:
            pass
    plt.rcParams.update({
        "axes.titlesize": 12, "axes.labelsize": 12, "legend.fontsize": 10,
    })


def _ratio_for_pair(cube, t_mask, m_mask):
    with np.errstate(invalid="ignore"):
        t_spec = np.nanmean(cube[:, t_mask], axis=1)
        m_spec = np.nanmean(cube[:, m_mask], axis=1)
    return t_spec / np.where(np.abs(m_spec) > 1e-6, m_spec, np.nan)


def collect_class_means(splice_um: float = DEFAULT_SPLICE_UM
                          ) -> tuple[np.ndarray, dict[str, np.ndarray],
                                      dict[str, int]]:
    """For each class in CLASS_ORDER, splice-correct every pair across
    every cube and return the unweighted mean ratio.

    Returns (wav, {class: mean_ratio}, {class: n_pairs}).
    """
    common_wav = None
    by_class: dict[str, list[np.ndarray]] = {c: [] for c in CLASS_ORDER}

    for cube_dir in sorted(SAMPLES_ROOT.iterdir()):
        if not cube_dir.is_dir():
            continue
        cube_id = cube_dir.name
        if cube_id in EXCLUDE_CUBES or cube_id == "figures":
            continue
        # Skip if this cube has none of the classes we want.
        present = [c for c in CLASS_ORDER if (cube_dir / c).is_dir()]
        if not present:
            continue
        print(f"  loading {cube_id} cube...")
        cube, wav, _, _ = load_cube_by_id(cube_id)
        cube = cube[:, ::-1, ::-1].copy()
        wav = wav.astype(np.float32)
        if common_wav is None:
            common_wav = wav
        for cls in present:
            try:
                pairs = load_pairs(cube_id, cls)
            except SystemExit:
                continue
            if not pairs:
                continue
            for idx, t_info, m_info in pairs:
                t_mask = np.load(t_info["mask_path"])[::-1, ::-1]
                m_mask = np.load(m_info["mask_path"])[::-1, ::-1]
                ratio = _ratio_for_pair(cube, t_mask, m_mask)
                ratio_c, _, _ = apply_splice_correction(
                    wav, ratio, splice_um=splice_um)
                by_class[cls].append(ratio_c)
            print(f"    {cls}: collected {len(pairs)} pair(s)")

    means: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for cls in CLASS_ORDER:
        if not by_class[cls]:
            print(f"  [warn] class '{cls}' has no pairs across all cubes")
            continue
        arr = np.array(by_class[cls])
        with np.errstate(invalid="ignore"):
            means[cls] = np.nanmean(arr, axis=0)
        counts[cls] = len(by_class[cls])
    if common_wav is None:
        raise SystemExit("no usable cubes / classes found")
    return common_wav, means, counts


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xmin", type=float, default=0.4)
    parser.add_argument("--xmax", type=float, default=2.6)
    parser.add_argument("--offset-step", type=float,
                        default=DEFAULT_OFFSET_STEP,
                        help=f"additive offset between consecutive curves "
                             f"(default {DEFAULT_OFFSET_STEP}; pass 0.0 "
                             "to plot original values on shared y-axis)")
    parser.add_argument("--splice", type=float, default=DEFAULT_SPLICE_UM)
    parser.add_argument("--figsize", nargs=2, type=float,
                        default=[8.5, 11.5],
                        metavar=("W", "H"),
                        help="figure size in inches; default 8.5x11.5 "
                             "(elongated portrait, A4-ish)")
    args = parser.parse_args(argv[1:])

    setup_style()
    print("=== Supplement: ratio spectra by class (single panel) ===")
    wav, means, counts = collect_class_means(splice_um=args.splice)

    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    # Reference vertical lines at diagnostic band positions; drawn
    # before the spectra so they sit underneath.
    for w_ref in REF_LINES_UM:
        if args.xmin <= w_ref <= args.xmax:
            ax.axvline(w_ref, color="0.45", lw=0.7, ls="--",
                        alpha=0.55, zorder=0)

    # Additive offset stack + per-class multiplicative scale.
    offsets: dict[str, float] = {}
    for i, cls in enumerate(CLASS_ORDER):
        offsets[cls] = -args.offset_step * i

    def _label(cls: str) -> str:
        s = CLASS_SCALE.get(cls, 1.0)
        if abs(s - 1.0) < 1e-3:
            return CLASS_DISPLAY[cls]
        return f"{CLASS_DISPLAY[cls]} (\u00d7{s:g})"

    for cls in CLASS_ORDER:
        if cls not in means:
            continue
        scale = CLASS_SCALE.get(cls, 1.0)
        y = means[cls] * scale + offsets[cls]
        ax.plot(wav, y, color=CLASS_COLORS[cls], lw=2.0,
                 label=_label(cls))

    ax.set_xlim(args.xmin, args.xmax)
    ax.set_xlabel(r"Wavelength ($\mu$m)")
    scaled_any = any(abs(s - 1.0) > 1e-3 for s in CLASS_SCALE.values())
    if abs(args.offset_step) < 1e-9 and not scaled_any:
        ax.set_ylabel("Ratioed reflectance")
    else:
        ax.set_ylabel("Ratioed reflectance (scaled and offset for clarity)")
    ax.legend(loc="best", framealpha=0.92)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.minorticks_off()
    ax.grid(True, which="major", alpha=0.3, linewidth=0.6)

    # Tighter Y limits based on the actual scaled+offset data.
    all_y = []
    for cls in CLASS_ORDER:
        if cls in means:
            scale = CLASS_SCALE.get(cls, 1.0)
            all_y.append(means[cls] * scale + offsets[cls])
    if all_y:
        flat = np.concatenate(all_y)
        finite = flat[np.isfinite(flat)]
        if finite.size > 0:
            y_lo = float(finite.min()); y_hi = float(finite.max())
            pad = 0.05 * (y_hi - y_lo)
            ax.set_ylim(y_lo - pad, y_hi + pad)

    plt.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    base = "supplement_ratio_spectra_classes"
    out_png = FIGURE_DIR / f"{base}.png"
    out_svg = FIGURE_DIR / f"{base}.svg"
    fig.savefig(out_png, dpi=300, bbox_inches="tight",
                facecolor="white", pad_inches=0.05)
    fig.savefig(out_svg, bbox_inches="tight",
                facecolor="white", pad_inches=0.05)
    plt.close(fig)
    print()
    for cls in CLASS_ORDER:
        if cls in counts:
            print(f"  {cls:<14s} -> {CLASS_DISPLAY[cls]:<22s}  "
                  f"averaged over {counts[cls]} pair(s)")
    print()
    print(f"wrote {out_png}")
    print(f"wrote {out_svg}")


if __name__ == "__main__":
    main(sys.argv)
