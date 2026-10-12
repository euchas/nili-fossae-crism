"""Per-pair ratio inspector with detector-splice correction at 1 um.

For a given cube + class folder, walks the cumulative
<cube>_<class>_rois.csv, pairs every (target, mafic) by pair_idx,
loads the saved boolean masks, and re-computes the ratio
(target_mean / mafic_mean) directly from the cube. Each pair gets
its own panel.

The CRISM S detector (~0.36-1.02 um) and L detector (~1.00-3.92 um)
join near 1.00 um. In a target/mafic ratio, residual calibration
mismatches between the two detectors propagate as a visible step at
the splice. This script:

  - finds the wavelength band closest to --splice (default 1.000 um)
  - averages the ratio over `--splice-avg` bands on each side
  - scales the L-detector segment by left_avg / right_avg so the
    ratio is continuous at the splice

Both raw (light grey) and splice-corrected (red) curves are plotted
so the correction is visible and verifiable.

Usage:
  python code/v9_per_pair_inspector.py FRT00003E12 discovery_ridge
  python code/v9_per_pair_inspector.py 871C carbonates
  python code/v9_per_pair_inspector.py FRT00003E12 olivine --no-splice
  python code/v9_per_pair_inspector.py FRT00003E12 discovery_ridge --splice 1.023

Outputs (in outputs/spectra sample locations/<cube>/<name>/):
  <cube>_<name>_per_pair_ratios.png     2-col grid, one panel per pair
  <cube>_<name>_per_pair_ratios.pdf
  <cube>_<name>_per_pair_ratios.csv     wavelength + raw and corrected
                                         ratio per pair
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

from config import OUTPUTS_DIR, CUBE_IDS
from crism_io import load_cube_by_id

SAMPLES_ROOT = OUTPUTS_DIR / "spectra sample locations"

# Optional: load AV's local style if present (sans-serif, larger fonts).
USER_STYLE = Path(__file__).parent / "my_style.mplstyle.txt"

# Default detector splice wavelength (CRISM S/L join near ~1 um).
DEFAULT_SPLICE_UM = 1.000
DEFAULT_SPLICE_AVG = 5

# Reference absorption-band lines for context.
REF_LINES_UM = [1.00, 1.40, 1.90, 2.30, 2.50]
CO2_RESIDUAL = (1.95, 2.05)


def setup_style() -> None:
    if USER_STYLE.exists():
        try:
            plt.style.use(str(USER_STYLE))
            print(f"[style] using {USER_STYLE}")
            return
        except Exception as e:
            print(f"[style] {USER_STYLE} found but failed: {e}")
    plt.rcParams.update({
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
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
    raise SystemExit(f"unknown cube {arg!r}; known: {CUBE_IDS}")


def load_pairs(cube_id: str, name: str) -> list[tuple[int, dict, dict]]:
    """Return [(pair_idx, target_info, mafic_info), ...] from the
    cumulative rois.csv, one entry per complete (target, mafic) pair."""
    name_dir = SAMPLES_ROOT / cube_id / name
    rois_csv = name_dir / f"{cube_id}_{name}_rois.csv"
    if not rois_csv.exists():
        raise SystemExit(f"no rois CSV at {rois_csv}")
    by_idx: dict[int, dict] = {}
    with open(rois_csv) as f:
        for row in csv.DictReader(f):
            try:
                idx = int(row["pair_idx"])
            except (KeyError, ValueError):
                continue
            entry = {
                "label": row["roi_label"],
                "session_ts": row["session_ts"],
                "n_pixels": int(row["n_pixels"]),
                "mask_path": name_dir / row["mask_file"],
                "color": row.get("color", "#888888"),
            }
            by_idx.setdefault(idx, {})[row["kind"]] = entry
    out = []
    for idx in sorted(by_idx):
        p = by_idx[idx]
        if "target" in p and "mafic" in p:
            out.append((idx, p["target"], p["mafic"]))
    return out


def apply_splice_correction(wav: np.ndarray, ratio: np.ndarray,
                              splice_um: float = DEFAULT_SPLICE_UM,
                              n_avg: int = DEFAULT_SPLICE_AVG
                              ) -> tuple[np.ndarray, float, int]:
    """Scale the L-detector segment (lambda > splice_um) by
    last_S_avg / first_L_avg so the ratio is continuous at the splice.

    The CRISM TER3 join near 1 um has BBL-flagged bands (NaN) on both
    sides of the splice. A naive `splice_idx +/- n` window often falls
    entirely inside the BBL gap and returns NaN. Instead, walk outward
    from the splice on each side and average the FIRST `n_avg` finite
    bands encountered.
    """
    splice_idx = int(np.argmin(np.abs(wav - splice_um)))
    finite = np.isfinite(ratio)
    if not finite.any():
        return ratio.copy(), 1.0, splice_idx

    # Walk left from splice_idx-1 toward 0, gather up to n_avg finite values.
    left_vals = []
    for i in range(splice_idx - 1, -1, -1):
        if finite[i]:
            left_vals.append(ratio[i])
            if len(left_vals) >= n_avg:
                break
    # Walk right from splice_idx forward, gather up to n_avg finite values.
    right_vals = []
    for i in range(splice_idx, len(ratio)):
        if finite[i]:
            right_vals.append(ratio[i])
            if len(right_vals) >= n_avg:
                break
    if not left_vals or not right_vals:
        return ratio.copy(), 1.0, splice_idx
    left = float(np.mean(left_vals))
    right = float(np.mean(right_vals))
    if not (np.isfinite(left) and np.isfinite(right)
            and abs(right) > 1e-6):
        return ratio.copy(), 1.0, splice_idx
    scale = left / right
    out = ratio.copy()
    out[splice_idx:] *= scale
    return out, scale, splice_idx


def annotate_ref_lines(ax, xmin: float, xmax: float) -> None:
    if CO2_RESIDUAL[0] < xmax and CO2_RESIDUAL[1] > xmin:
        ax.axvspan(*CO2_RESIDUAL, color="#cccccc", alpha=0.25, zorder=0)
    for w in REF_LINES_UM:
        if xmin <= w <= xmax:
            ax.axvline(w, color="#aa4444", lw=0.5, ls="--",
                        alpha=0.45, zorder=0)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cube", help="cube id (full or short, e.g. 3E12)")
    parser.add_argument("name", help="class folder name (e.g. discovery_ridge)")
    parser.add_argument("--splice", type=float, default=DEFAULT_SPLICE_UM,
                        help=f"detector splice wavelength (um); "
                             f"default {DEFAULT_SPLICE_UM}")
    parser.add_argument("--splice-avg", type=int, default=DEFAULT_SPLICE_AVG,
                        help=f"bands averaged on each side of the splice "
                             f"for the scale factor; default "
                             f"{DEFAULT_SPLICE_AVG}")
    parser.add_argument("--no-splice", action="store_true",
                        help="skip the detector splice correction")
    parser.add_argument("--xmin", type=float, default=0.4)
    parser.add_argument("--xmax", type=float, default=2.6)
    args = parser.parse_args(argv[1:])

    setup_style()
    cube_id = resolve_cube_id(args.cube)
    print(f"=== Per-pair ratio inspector: {cube_id} / {args.name} ===")

    pairs = load_pairs(cube_id, args.name)
    print(f"  found {len(pairs)} complete (target, mafic) pairs")
    if not pairs:
        return
    for idx, t, m in pairs:
        print(f"    pair {idx}: {t['label']} (n={t['n_pixels']}) / "
              f"{m['label']} (n={m['n_pixels']})  "
              f"-> {t['mask_path'].name}, {m['mask_path'].name}")

    print(f"  loading {cube_id} cube...")
    cube, wav, _, _ = load_cube_by_id(cube_id)
    cube = cube[:, ::-1, ::-1].copy()
    wav = wav.astype(np.float32)

    n = len(pairs)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(15, 4.6 * nrows),
                              sharex=True, sharey=False)
    axes = np.atleast_1d(axes).flatten()

    # CSV output
    out_dir = SAMPLES_ROOT / cube_id / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{cube_id}_{args.name}_per_pair_ratios.csv"
    csv_rows: list[dict] = []   # one dict per wavelength
    for i, w in enumerate(wav):
        csv_rows.append({"wavelength_um": float(w)})

    splice_idx_global = None
    for i, (idx, t_info, m_info) in enumerate(pairs):
        t_mask = np.load(t_info["mask_path"])[::-1, ::-1]
        m_mask = np.load(m_info["mask_path"])[::-1, ::-1]
        with np.errstate(invalid="ignore"):
            t_spec = np.nanmean(cube[:, t_mask], axis=1)
            m_spec = np.nanmean(cube[:, m_mask], axis=1)
        ratio = t_spec / np.where(np.abs(m_spec) > 1e-6, m_spec, np.nan)

        if args.no_splice:
            ratio_c = ratio.copy()
            scale = 1.0
            splice_idx = int(np.argmin(np.abs(wav - args.splice)))
        else:
            ratio_c, scale, splice_idx = apply_splice_correction(
                wav, ratio, splice_um=args.splice,
                n_avg=args.splice_avg)
        splice_idx_global = splice_idx

        for j, w in enumerate(wav):
            csv_rows[j][f"pair{idx}_raw"] = (
                f"{ratio[j]:.6f}" if np.isfinite(ratio[j]) else "")
            csv_rows[j][f"pair{idx}_corrected"] = (
                f"{ratio_c[j]:.6f}" if np.isfinite(ratio_c[j]) else "")

        ax = axes[i]
        if not args.no_splice:
            ax.plot(wav, ratio, color="lightgray", lw=1.0, alpha=0.85,
                     label=f"raw ratio")
        ax.plot(wav, ratio_c, color="#d62728", lw=1.6,
                 label=("splice-corrected" if not args.no_splice
                        else "ratio"))
        ax.axvline(wav[splice_idx], color="black", lw=0.6, ls=":",
                    alpha=0.7,
                    label=(f"splice ({wav[splice_idx]:.4f} um)"
                           if i == 0 else None))
        ax.axhline(1.0, color="#888888", lw=0.5, ls=":", alpha=0.55)
        annotate_ref_lines(ax, args.xmin, args.xmax)
        ax.set_xlim(args.xmin, args.xmax)
        title = (f"Pair {idx}: {t_info['label']} / {m_info['label']}\n"
                 f"target n={t_info['n_pixels']}, "
                 f"mafic n={m_info['n_pixels']}")
        if not args.no_splice:
            title += f"  |  splice scale = {scale:.4f}"
        ax.set_title(title, fontsize=10)
        if (i // ncols) == nrows - 1:
            ax.set_xlabel(r"wavelength ($\mu$m)")
        ax.set_ylabel("ratio (target / mafic)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best", framealpha=0.92)

    for j in range(n, nrows * ncols):
        axes[j].set_visible(False)

    if not args.no_splice:
        fig.suptitle(
            f"{cube_id} - {args.name}: per-pair target/mafic ratios "
            f"with S/L detector-splice correction at "
            f"{args.splice:.4f} um (avg {args.splice_avg} bands/side)",
            fontsize=11, y=1.0)
    else:
        fig.suptitle(
            f"{cube_id} - {args.name}: per-pair target/mafic ratios "
            "(no splice correction)",
            fontsize=11, y=1.0)
    plt.tight_layout()

    out_png = out_dir / f"{cube_id}_{args.name}_per_pair_ratios.png"
    out_pdf = out_dir / f"{cube_id}_{args.name}_per_pair_ratios.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight",
                facecolor="white", pad_inches=0.05)
    fig.savefig(out_pdf, bbox_inches="tight",
                facecolor="white", pad_inches=0.05)
    plt.close(fig)
    print(f"  wrote {out_png}")
    print(f"  wrote {out_pdf}")

    # CSV
    if csv_rows:
        cols = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w_csv = csv.DictWriter(f, fieldnames=cols)
            w_csv.writeheader()
            w_csv.writerows(csv_rows)
        print(f"  wrote {csv_path}")


if __name__ == "__main__":
    main(sys.argv)
