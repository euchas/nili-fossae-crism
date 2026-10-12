"""Maps of the band-centre anomaly pixels (high decile and 2.33 um tail).

Companion to contact_anomaly_ratios.py: shows WHERE the strong-tier
high-decile (top 10 percent of 2.3 um centre) pixels fall in each unit,
over the hand-mapped geologic contours (clay solid, cap dashed, olivine
unit outline thin grey), with the Plebani carbonate patches for reference.
Deterministic.

Usage:  python code/contact_anomaly_pixel_maps.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import CUBE_IDS  # noqa: E402
from contact_distance_to_contact import load_cube, masks_from_labels, md5  # noqa: E402
from contact_band_center_map import OUT as T0_OUT, STRONG_DEPTH, unit_masks  # noqa: E402
from contact_band_center_modes import LONG_TAIL, PATCHES, patch_rowcol  # noqa: E402
from contact_anomaly_ratios import DECILES, OUT as AR_OUT  # noqa: E402
from figstyle import draw_units, label_panels  # noqa: E402

HI_COL, TAIL_COL = "#E69F00", "#CC3311"


def contours(ax, fr) -> None:
    draw_units(ax, fr, units=("oliv", "clay", "cap"))


def main() -> None:
    t0 = datetime.now(timezone.utc)
    patches_all = pd.read_csv(PATCHES)
    inputs = {}
    for cube in CUBE_IDS:
        c = load_cube(cube)
        m = masks_from_labels(c["labels"])
        units, d, fr = unit_masks(cube, m)
        cen = np.load(T0_OUT / f"{cube}_bc230_center.npy")
        dep = np.load(T0_OUT / f"{cube}_bc230_depth.npy")
        edg = np.load(T0_OUT / f"{cube}_bc230_edgeflag.npy")
        strong = np.isfinite(cen) & (dep >= STRONG_DEPTH) & ~edg
        inputs[cube] = {"center_md5": md5(T0_OUT / f"{cube}_bc230_center.npy")}

        fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharex=True, sharey=True)
        thresholds = {}
        for ax, (uname, fkey) in zip(axes, (("olivine", "oliv"), ("clay", "clay"))):
            unit = m["valid"] & (fr[fkey] >= 0.5)
            sel = unit & strong
            qhi = float(np.quantile(cen[sel], DECILES[1])) if sel.sum() >= 1000 else np.nan
            thresholds[uname] = qhi
            bg = np.where(np.isfinite(cen), 0.93, np.nan)
            ax.imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            hi = sel & (cen >= qhi) if np.isfinite(qhi) else np.zeros_like(sel)
            tail = sel & (cen >= LONG_TAIL)
            lay = np.full(cen.shape, np.nan)
            lay[hi] = 0
            if uname == "olivine":
                lay[tail] = 1
            ax.imshow(lay, cmap=matplotlib.colors.ListedColormap([HI_COL, TAIL_COL]),
                      vmin=0, vmax=1, interpolation="nearest")
            contours(ax, fr)
            n_hi, n_tail = int(hi.sum()), int(tail.sum())
            title = f"{uname}: high decile >= {qhi:.4f} um (n={n_hi})"
            if uname == "olivine":
                title += f"; dark red = tail >= {LONG_TAIL} (n={n_tail})"
                pdfc = patches_all[(patches_all.cube == cube) & patches_all.on_map]
                if len(pdfc):
                    pr, pc = patch_rowcol(cube, pdfc, cen.shape)
                    ax.plot(pc, pr, "x", color="magenta", ms=4, mew=1.0, label="MgCO3 patch")
                    ax.legend(fontsize=8, loc="lower right")
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{cube}: strong-tier band-centre high-decile pixels "
                     "(red = olivine unit, blue = phyllosilicate basement, dashed = mafic cap)", fontsize=11)
        fig.tight_layout()
        label_panels(fig)
        fig.savefig(AR_OUT / f"fig_highdecile_map_{cube}.png", dpi=200, bbox_inches="tight")
        fig.savefig(AR_OUT / f"fig_highdecile_map_{cube}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(cube, "done", thresholds)

    log = {"script": Path(__file__).name, "script_md5": md5(Path(__file__)),
           "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
           "decile": DECILES[1], "tail_um": LONG_TAIL, "strong_depth": STRONG_DEPTH,
           "inputs": inputs}
    (AR_OUT / "run_log_pixel_maps.json").write_text(json.dumps(log, indent=2, default=str))
    print("done ->", AR_OUT)


if __name__ == "__main__":
    main()
