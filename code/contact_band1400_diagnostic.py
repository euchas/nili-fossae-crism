"""Diagnostic: unit-median continuum-removed spectra, 1.30-1.55 um, MTR3.

Tests whether the 1.441 / 1.477 um band-minimum pile-ups are geological or
a scene-wide systematic: stacks the median I/F spectrum of each map-pure
unit (olivine / clay / cap) per cube, removes the same linear continuum
used by the band-centre fit, and overlays the three units. A mineral OH
band should differ between units (deep in clay, absent in cap); a
volcano-scan/instrument residual appears in all units at the same
wavelength. Deterministic.

Usage:  python code/contact_band1400_diagnostic.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import CUBE_IDS  # noqa: E402
from contact_distance_to_contact import PALETTE, load_cube, masks_from_labels, md5  # noqa: E402
from contact_band_center_map import unit_masks  # noqa: E402
from contact_band_center_1400 import LEFT_ANCHOR, RIGHT_ANCHOR, read_window  # noqa: E402
from figstyle import label_panels  # noqa: E402

OUT = ROOT / "outputs" / "revision" / "band_center_map" / "band1400"
OUT.mkdir(parents=True, exist_ok=True)
MARKS = (1.385, 1.410, 1.441, 1.477)


def main() -> None:
    t0 = datetime.now(timezone.utc)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    style = {"clay map-pure": "#7f7f7f", "cap map-pure": "#2b2b2b", "olivine map-pure": "#d55e00"}
    for ax, cube in zip(axes.ravel(), CUBE_IDS):
        c = load_cube(cube)
        m = masks_from_labels(c["labels"])
        units, d, fr = unit_masks(cube, m)
        wav, block = read_window(cube)
        la = (wav >= LEFT_ANCHOR[0]) & (wav <= LEFT_ANCHOR[1])
        ra = (wav >= RIGHT_ANCHOR[0]) & (wav <= RIGHT_ANCHOR[1])
        wl, wr = wav[la].mean(), wav[ra].mean()
        for uname, col in style.items():
            sp = np.nanmedian(block[:, units[uname]], axis=1)
            rl, rgt = np.nanmean(sp[la]), np.nanmean(sp[ra])
            cr = sp / (rl + (rgt - rl) * (wav - wl) / (wr - wl))
            ax.plot(wav, cr, color=col, lw=1.7, label=f"{uname} (n={int(units[uname].sum())})")
        for xv in MARKS:
            ax.axvline(xv, color="0.88", lw=0.8, ls=":" if xv > 1.43 else "-")
            ax.annotate(f"{xv:.3f}", (xv, 1.003), fontsize=7, ha="center", color="0.45",
                        annotation_clip=False)
        ax.set_title(cube, fontsize=10, color=PALETTE[cube])
        ax.grid(True, color="0.96"); ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
        del block
        print(cube, "done")
    for ax in axes[1]:
        ax.set_xlabel("wavelength (um)")
    for ax in axes[:, 0]:
        ax.set_ylabel("continuum-removed median I/F")
    fig.suptitle("Unit-median continuum-removed spectra, 1.30-1.55 um (MTR3 I/F)\n"
                 "unit-independent dips at fixed wavelengths = scene-wide systematic, not mineralogy",
                 fontsize=11)
    fig.tight_layout()
    label_panels(fig)
    fig.savefig(OUT / "fig_diagnostic_unit_spectra.png", dpi=200)
    fig.savefig(OUT / "fig_diagnostic_unit_spectra.pdf")
    plt.close(fig)

    log = {"script": Path(__file__).name, "script_md5": md5(Path(__file__)),
           "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat()}
    (OUT / "run_log_diagnostic.json").write_text(json.dumps(log, indent=2, default=str))
    print("done ->", OUT)


if __name__ == "__main__":
    main()
