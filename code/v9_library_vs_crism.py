"""Two-panel figure: splice-corrected CRISM ratios vs a library spectrum.

Panel A (CRISM, all splice-corrected at ~1 um):
  - 3E12 discovery_ridge pair 1 (target1 / mafic1)
  - 3E12 olivine class average (mean of all pair ratios)
  - <other cube> intrusions class average (default FRT0000A4FC)

Panel B:
  - Library reflectance from
    outputs/spectra sample locations/s07_HY07_Olivine_KI3188_Fo51_lt60um_BECKb_AREF.txt
    (+ paired wavelength file in the same dir).

Output: outputs/spectra sample locations/figures/
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

from config import OUTPUTS_DIR, CUBE_IDS
from crism_io import load_cube_by_id
from v9_per_pair_inspector import (
    load_pairs, apply_splice_correction,
    DEFAULT_SPLICE_UM, DEFAULT_SPLICE_AVG,
)

SAMPLES_ROOT = OUTPUTS_DIR / "spectra sample locations"
FIGURE_DIR = SAMPLES_ROOT / "figures"

LIBRARY_REFLECTANCE = "s07_HY07_Olivine_KI3188_Fo51_lt60um_BECKb_AREF.txt"
LIBRARY_WAVELENGTH = "s07_HY07_Wavelength_HyMap2007_124ch_(microns).txt"

USER_STYLE = Path(__file__).parent / "my_style.mplstyle.txt"

REF_LINES_UM = [1.00, 1.40, 1.90, 2.30, 2.50]
CO2_RESIDUAL = (1.95, 2.05)


def setup_style():
    if USER_STYLE.exists():
        try:
            plt.style.use(str(USER_STYLE))
            print(f"[style] using {USER_STYLE}")
            return
        except Exception:
            pass
    plt.rcParams.update({
        "axes.titlesize": 12, "axes.labelsize": 12,
        "legend.fontsize": 10,
    })


def resolve_cube_id(arg: str | None) -> str:
    if not arg:
        return "FRT00003E12"
    a = arg.upper().strip()
    if a in CUBE_IDS:
        return a
    for cid in CUBE_IDS:
        if cid.endswith(a):
            return cid
    raise SystemExit(f"unknown cube {arg!r}")


def load_library_spectrum() -> tuple[np.ndarray, np.ndarray, str]:
    refl_path = SAMPLES_ROOT / LIBRARY_REFLECTANCE
    wav_path = SAMPLES_ROOT / LIBRARY_WAVELENGTH
    if not refl_path.exists():
        raise SystemExit(f"missing library reflectance: {refl_path}")
    if not wav_path.exists():
        raise SystemExit(f"missing library wavelength: {wav_path}")
    with open(refl_path) as f:
        lines = f.readlines()
    header = lines[0].strip()
    refl = np.array(
        [float(line.strip()) for line in lines[1:]
         if line.strip()], dtype=np.float64,
    )
    with open(wav_path) as f:
        wav_lines = f.readlines()
    # The wavelength file may or may not have a header. Try reading
    # all lines as floats; skip non-numeric lines.
    wav_vals: list[float] = []
    for line in wav_lines:
        s = line.strip()
        if not s:
            continue
        try:
            wav_vals.append(float(s))
        except ValueError:
            # First line is a header, skip.
            continue
    wav = np.array(wav_vals, dtype=np.float64)
    if len(wav) != len(refl):
        print(f"[lib] WARNING: wavelength count {len(wav)} != "
              f"reflectance count {len(refl)}")
    return wav, refl, header


def _ratio_for_pair(cube: np.ndarray, t_mask: np.ndarray,
                     m_mask: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        t_spec = np.nanmean(cube[:, t_mask], axis=1)
        m_spec = np.nanmean(cube[:, m_mask], axis=1)
    return t_spec / np.where(np.abs(m_spec) > 1e-6, m_spec, np.nan)


def crism_pair_ratio(cube_id: str, name: str, pair_idx: int,
                       splice_um: float = DEFAULT_SPLICE_UM
                       ) -> tuple[np.ndarray, np.ndarray, float]:
    """Splice-corrected ratio for one specific pair."""
    pairs = load_pairs(cube_id, name)
    pair = next((p for p in pairs if p[0] == pair_idx), None)
    if pair is None:
        raise SystemExit(f"pair {pair_idx} not found in {cube_id}/{name}")
    _, t_info, m_info = pair
    print(f"  {cube_id}/{name} pair {pair_idx}: "
          f"{t_info['label']} (n={t_info['n_pixels']}) / "
          f"{m_info['label']} (n={m_info['n_pixels']})")
    cube, wav, _, _ = load_cube_by_id(cube_id)
    cube = cube[:, ::-1, ::-1].copy()
    wav = wav.astype(np.float32)
    t_mask = np.load(t_info["mask_path"])[::-1, ::-1]
    m_mask = np.load(m_info["mask_path"])[::-1, ::-1]
    ratio = _ratio_for_pair(cube, t_mask, m_mask)
    ratio_c, scale, splice_idx = apply_splice_correction(
        wav, ratio, splice_um=splice_um)
    print(f"    splice scale = {scale:.4f}  at band {splice_idx} "
          f"(wav {wav[splice_idx]:.4f} um)")
    return wav, ratio_c, scale


def crism_class_average(cube_id: str, name: str,
                          splice_um: float = DEFAULT_SPLICE_UM
                          ) -> tuple[np.ndarray, np.ndarray, int, list[float]]:
    """Mean splice-corrected target/mafic ratio across all pairs in
    <cube>/<name>. Returns (wav, mean_ratio, n_pairs, [scales])."""
    pairs = load_pairs(cube_id, name)
    if not pairs:
        raise SystemExit(f"no pairs for {cube_id}/{name}")
    cube, wav, _, _ = load_cube_by_id(cube_id)
    cube = cube[:, ::-1, ::-1].copy()
    wav = wav.astype(np.float32)
    print(f"  {cube_id}/{name}: averaging {len(pairs)} pair(s)")
    stack = []
    scales = []
    for idx, t_info, m_info in pairs:
        t_mask = np.load(t_info["mask_path"])[::-1, ::-1]
        m_mask = np.load(m_info["mask_path"])[::-1, ::-1]
        ratio = _ratio_for_pair(cube, t_mask, m_mask)
        ratio_c, scale, _ = apply_splice_correction(
            wav, ratio, splice_um=splice_um)
        stack.append(ratio_c)
        scales.append(float(scale))
        print(f"    pair {idx}: {t_info['label']} / {m_info['label']}  "
              f"splice scale = {scale:.4f}")
    arr = np.array(stack)
    with np.errstate(invalid="ignore"):
        mean_ratio = np.nanmean(arr, axis=0)
    return wav, mean_ratio, len(pairs), scales


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ridge-cube", default="FRT00003E12")
    parser.add_argument("--ridge-pair", type=int, default=1)
    parser.add_argument("--olivine-cube", default="FRT00003E12")
    parser.add_argument("--intrusions-cube", default="FRT0000A4FC")
    parser.add_argument("--xmin", type=float, default=0.4)
    parser.add_argument("--xmax", type=float, default=2.6)
    parser.add_argument("--splice", type=float, default=DEFAULT_SPLICE_UM)
    parser.add_argument("--ridge-scale", type=float, default=0.6,
                        help="multiplicative scale on the ridge ratio "
                             "(default 0.6)")
    parser.add_argument("--olivine-scale", type=float, default=0.75,
                        help="multiplicative scale on the olivine "
                             "ratio (default 0.75; reduced so the "
                             "olivine band depth doesn't dominate)")
    parser.add_argument("--intrusions-scale", type=float, default=0.8,
                        help="multiplicative scale on the intrusions "
                             "ratio (default 0.8)")
    args = parser.parse_args(argv[1:])

    setup_style()
    # Bump sizes on top of the project style so the figure stays
    # readable when sized to a single A4 page (AV 2026-05-01).
    plt.rcParams.update({
        "axes.labelsize":  22,
        "axes.titlesize":  24,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 18,
    })
    print("=== Library vs CRISM comparison ===")

    # Library
    lib_wav, lib_refl, lib_header = load_library_spectrum()
    print(f"[library] header: {lib_header}")
    print(f"[library] {len(lib_refl)} bands, "
          f"{lib_wav.min():.3f}-{lib_wav.max():.3f} um")

    # CRISM 1: 3E12 discovery_ridge pair N
    print()
    print("[1] 3E12 discovery_ridge pair 1...")
    cube_ridge = resolve_cube_id(args.ridge_cube)
    w1, r1, scale1 = crism_pair_ratio(
        cube_ridge, "discovery_ridge", args.ridge_pair,
        splice_um=args.splice)

    # CRISM 2: 3E12 olivine class average
    print()
    print("[2] 3E12 olivine class average...")
    cube_olv = resolve_cube_id(args.olivine_cube)
    w2, r2, n2, scales2 = crism_class_average(
        cube_olv, "olivine", splice_um=args.splice)

    # CRISM 3: A4FC intrusions class average
    print()
    print(f"[3] {args.intrusions_cube} intrusions class average...")
    cube_intr = resolve_cube_id(args.intrusions_cube)
    w3, r3, n3, scales3 = crism_class_average(
        cube_intr, "intrusions", splice_um=args.splice)

    # All three CRISM curves should share the same wavelength axis
    # (same instrument, same TER3 pipeline).
    assert np.allclose(w1, w2) and np.allclose(w2, w3), (
        "CRISM wavelength arrays differ across cubes - unexpected.")
    crism_wav = w1

    # Apply per-curve scaling so the three CRISM ratios separate
    # vertically instead of piling on each other near 1.0.
    s_ridge = float(args.ridge_scale)
    s_olv   = float(args.olivine_scale)
    s_intr  = float(args.intrusions_scale)
    r1_s = r1 * s_ridge
    r2_s = r2 * s_olv
    r3_s = r3 * s_intr

    # Plot - both panels rendered as visual squares via set_box_aspect.
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(15, 7.5),
    )

    # ---- Panel A: CRISM ratios (scaled) ----
    axA.plot(crism_wav, r1_s, color="#d62728", lw=2.0,
             label="Ridge core-3E12")
    axA.plot(crism_wav, r2_s, color="#1b9e77", lw=2.0,
             label="Olivine-3E12")
    axA.plot(crism_wav, r3_s, color="#e7298a", lw=2.0,
             label="Intrusion-A4FC")
    axA.set_xlim(args.xmin, args.xmax)
    axA.set_xlabel(r"Wavelength ($\mu$m)")
    axA.set_ylabel("Ratioed reflectance (scaled and offset for clarity)")
    axA.set_title("(a) CRISM ratio spectra", loc="left")
    axA.legend(loc="best", framealpha=0.92)

    # Restretch y after scaling so the limits track the actual data
    # extent (not whatever auto-scale picked before scaling was applied).
    all_y = np.concatenate([r1_s, r2_s, r3_s])
    finite = all_y[np.isfinite(all_y)]
    if finite.size > 0:
        y_lo = float(finite.min())
        y_hi = float(finite.max())
        pad = 0.05 * (y_hi - y_lo)
        axA.set_ylim(y_lo - pad, y_hi + pad)

    # Coarser grid: major-only at 0.5 um (x) and the natural y locator,
    # minor ticks off so the panel reads cleanly at print size.
    axA.xaxis.set_major_locator(MultipleLocator(0.5))
    axA.minorticks_off()
    axA.grid(True, which="major", alpha=0.3, linewidth=0.6)
    axA.set_box_aspect(1)

    # ---- Panel B: library ----
    # Pull a short library label from the header (after "Record=NNNN:")
    # and strip the trailing "BECKb AREF" descriptor.
    lib_short = lib_header
    if ":" in lib_header:
        lib_short = lib_header.split(":", 1)[1].strip()
    for tag in (" BECKb AREF", " BECK AREF", " AREF", " BECKb"):
        if lib_short.endswith(tag):
            lib_short = lib_short[: -len(tag)].rstrip()
            break
    lib_short = " ".join(lib_short.split())
    axB.plot(lib_wav, lib_refl, color="black", lw=2.0,
             label=lib_short[:70])
    axB.set_xlim(args.xmin, args.xmax)
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
    out_png = FIGURE_DIR / f"{base}.png"
    out_pdf = FIGURE_DIR / f"{base}.pdf"
    out_svg = FIGURE_DIR / f"{base}.svg"
    fig.savefig(out_png, dpi=300, bbox_inches="tight",
                facecolor="white", pad_inches=0.05)
    fig.savefig(out_pdf, bbox_inches="tight",
                facecolor="white", pad_inches=0.05)
    fig.savefig(out_svg, bbox_inches="tight",
                facecolor="white", pad_inches=0.05)
    plt.close(fig)
    print()
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")
    print(f"wrote {out_svg}")


if __name__ == "__main__":
    main(sys.argv)
