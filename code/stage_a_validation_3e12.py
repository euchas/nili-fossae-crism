"""Stage A — Brown 2020 reproduction checkpoint for FRT00003E12.

Brown 2020 Section 2.5 explicitly restricts their analysis to pixels whose
1 µm olivine centroid is ≥ 1.15 µm ("we have eliminated highly Mg-rich
olivine or fine-grained olivine spectra. This is done intentionally to
focus the study on redshifted and saturated olivine signatures."). Their
Figure 7b shows FRT00003E12 centroids clustering roughly between 1.15 and
1.40 µm with a density peak around 1.15–1.25 µm; the longest-wavelength
tail reaches ~1.4 µm (red dunes noted by the large red arrow in their
Figure 7c). This is a different beast from the textbook forsteritic
1.04–1.08 µm band center — Nili Fossae olivine is unusually coarse and
Fa-enriched, and the 1 µm band is partly saturated.

The reproduction check here is therefore:
  - modal centroid for FRT00003E12 should fall in 1.15–1.30 µm;
  - 95th percentile should extend to ≥ 1.30 µm (long-wavelength tail
    corresponding to Brown 2020's most-redshifted population);
  - median should sit above 1.15 µm.
If the v5 maps load and the centroid distribution hits all three, the
Brown fit is behaving as published and the rest of the pipeline can
proceed. Otherwise we have a calibration problem and everything downstream
should pause.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from config import out_dir

CUBE_ID = "FRT00003E12"

# Expected centroid range from Brown 2020 Figure 7 (FRT00003E12).
EXPECTED_MODE_RANGE_UM = (1.15, 1.30)
MIN_P95_UM = 1.30
MIN_MEDIAN_UM = 1.15


def main() -> None:
    npz = out_dir("feature_stacks") / f"{CUBE_ID}_stackA_8param.npz"
    if not npz.exists():
        raise SystemExit(
            f"Missing {npz}; run stage_a_build_feature_stack.py first."
        )
    data = np.load(npz, allow_pickle=True)
    valid = data["joint_valid"].astype(bool)
    cent = data["centroid"][valid]
    asym = data["asymmetry"][valid]
    mask = np.isfinite(cent) & np.isfinite(asym)
    cent = cent[mask]
    asym = asym[mask]

    # Mode via histogram peak with 0.001 µm bin.
    bins = np.arange(0.98, 1.22, 0.001)
    hist, edges = np.histogram(cent, bins=bins)
    if hist.sum() == 0:
        raise SystemExit("No valid centroid values to check.")
    mode_idx = int(np.argmax(hist))
    mode_um = 0.5 * (edges[mode_idx] + edges[mode_idx + 1])
    median_um = float(np.median(cent))
    q05, q50, q95 = (float(x) for x in np.quantile(cent, [0.05, 0.5, 0.95]))
    n = int(cent.size)

    mode_ok = EXPECTED_MODE_RANGE_UM[0] <= mode_um <= EXPECTED_MODE_RANGE_UM[1]
    median_ok = median_um >= MIN_MEDIAN_UM
    tail_ok = q95 >= MIN_P95_UM
    all_ok = mode_ok and median_ok and tail_ok
    status = "PASS" if all_ok else "FAIL"
    within = all_ok

    fig, (ax_h, ax_2d) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    ax_h.hist(cent, bins=bins, color="#3366aa", edgecolor="none")
    ax_h.axvspan(*EXPECTED_MODE_RANGE_UM, color="green", alpha=0.15,
                 label=f"Brown 2020 expected mode range {EXPECTED_MODE_RANGE_UM}")
    ax_h.axvline(mode_um, color="red", ls="--", lw=1.1,
                 label=f"observed mode {mode_um:.3f} µm")
    ax_h.axvline(median_um, color="black", ls=":", lw=1.0,
                 label=f"median {median_um:.3f} µm")
    ax_h.set_xlabel("1 µm olivine centroid (µm)")
    ax_h.set_ylabel("valid pixels")
    ax_h.set_title(f"{CUBE_ID} centroid distribution — {status}")
    ax_h.legend(fontsize=8)

    hb = ax_2d.hexbin(cent, asym, gridsize=90, cmap="viridis",
                      mincnt=1, norm=LogNorm())
    ax_2d.set_xlabel("centroid (µm)")
    ax_2d.set_ylabel("asymmetry")
    ax_2d.set_title(f"{CUBE_ID} centroid vs asymmetry (n={n:,})")
    plt.colorbar(hb, ax=ax_2d, shrink=0.8, label="pixels (log)")

    stem = out_dir("figures") / f"{CUBE_ID}_brown2020_centroid_asymmetry"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {stem}.png")
    print(
        f"{CUBE_ID} centroid mode={mode_um:.3f} µm (expected {EXPECTED_MODE_RANGE_UM}) -> {status}"
    )
    print(
        f"  median={median_um:.3f}  p05={q05:.3f}  p95={q95:.3f}  n_valid={n}"
    )

    note_path = out_dir("reports") / "01_stageA_validation_note.md"
    body = (
        "## FRT00003E12 centroid checkpoint (Brown 2020 reproduction)\n\n"
        "Brown 2020 focuses on the redshifted Nili Fossae olivine population "
        "and restricts their analysis to centroid >= 1.15 um. Three "
        "conditions expected for a PASS:\n\n"
        f"- Mode within {EXPECTED_MODE_RANGE_UM[0]}-{EXPECTED_MODE_RANGE_UM[1]} um: "
        f"observed {mode_um:.3f} -> **{'OK' if mode_ok else 'FAIL'}**\n"
        f"- Median >= {MIN_MEDIAN_UM} um: observed {median_um:.3f} -> "
        f"**{'OK' if median_ok else 'FAIL'}**\n"
        f"- 95th percentile >= {MIN_P95_UM} um: observed {q95:.3f} -> "
        f"**{'OK' if tail_ok else 'FAIL'}**\n\n"
        f"Overall status: **{status}**.\n\n"
        f"n_valid pixels: {n}. p05={q05:.3f}; p95={q95:.3f}.\n\n"
        "See `outputs/figures/FRT00003E12_brown2020_centroid_asymmetry.png`.\n"
    )
    note_path.write_text(body, encoding="utf-8")
    print(f"saved {note_path}")


if __name__ == "__main__":
    main()
