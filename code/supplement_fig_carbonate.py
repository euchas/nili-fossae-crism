"""Supplement figure: carbonate patch distribution vs the mapped clay contact.

Panels: (a) FRT00003E12 and (b) FRT0000B438 patch maps over the hand-mapped
geologic contours (patch symbols scaled by patch area); (c) patch median
distance to the mapped clay unit vs the map-pure olivine background median,
all four cubes, with 871C/A4FC flagged inconclusive (map coverage).
Patch catalogue: stage-6 Plebani crism_ml detections (TRR3 chain).
Deterministic.

Usage:  python code/supplement_fig_carbonate.py
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
from contact_distance_to_contact import PALETTE, load_cube, masks_from_labels, md5  # noqa: E402
from contact_band_center_map import unit_masks  # noqa: E402
from contact_band_center_modes import PATCHES, patch_rowcol  # noqa: E402
from figstyle import draw_units  # noqa: E402

OUT = ROOT / "manuscript_v10" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MAP_CUBES = ("FRT00003E12", "FRT0000B438")
INCONCLUSIVE = {"FRT0000871C": "map covers corner", "FRT0000A4FC": "boundary-dominated"}
PATCH_COL = "#9400D3"   # violet: distinct from the red olivine / blue clay unit contours


def main() -> None:
    t0 = datetime.now(timezone.utc)
    patches_all = pd.read_csv(PATCHES)
    stats = []
    grids = {}
    for cube in CUBE_IDS:
        c = load_cube(cube)
        m = masks_from_labels(c["labels"])
        units, d, fr = unit_masks(cube, m)
        pdfc = patches_all[(patches_all.cube == cube) & patches_all.on_map].reset_index(drop=True)
        bg = d["clay"][units["olivine map-pure"] & (d["cap"] > 90)]
        stats.append({"cube": cube, "n_patches": int(len(pdfc)),
                      "patch_med_m": float(pdfc.dist_to_clay_contact_m.median()),
                      "bg_med_m": float(np.median(bg)),
                      "inconclusive": INCONCLUSIVE.get(cube, "")})
        if cube in MAP_CUBES:
            pr, pc = patch_rowcol(cube, pdfc, c["labels"].shape)
            grids[cube] = {"fr": fr, "valid": m["valid"], "pr": pr, "pc": pc,
                           "npx": pdfc.n_px.to_numpy()[:len(pr)]}
        print(cube, stats[-1])

    sdf = pd.DataFrame(stats)
    sdf.to_csv(OUT / "fig_S_carbonate_stats.csv", index=False)

    fig = plt.figure(figsize=(15.5, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=(1.15, 1.15, 0.75), wspace=0.08)
    for i, cube in enumerate(MAP_CUBES):
        g = grids[cube]
        ax = fig.add_subplot(gs[0, i])
        bg = np.where(g["valid"], 0.93, np.nan)
        ax.imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        draw_units(ax, g["fr"], units=("oliv", "clay", "cap"))
        ax.scatter(g["pc"], g["pr"], s=10 + 6 * np.sqrt(g["npx"]), facecolors="none",
                   edgecolors=PATCH_COL, linewidths=1.1)
        ax.set_title(f"({'ab'[i]}) {cube}: MgCO3 detections (n={len(g['pr'])} patches)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[0, 2])
    ys = np.arange(len(CUBE_IDS))[::-1]
    for y, row in zip(ys, sdf.itertuples()):
        alpha = 1.0 if not row.inconclusive else 0.4
        ax.plot([row.patch_med_m, row.bg_med_m], [y, y], color="0.7", lw=1.6, alpha=alpha, zorder=1)
        ax.scatter([row.patch_med_m], [y], s=70, color=PATCH_COL, alpha=alpha, zorder=2,
                   label="carbonate patches" if y == ys[0] else None)
        ax.scatter([row.bg_med_m], [y], s=70, color="0.35", alpha=alpha, zorder=2,
                   label="olivine-unit background" if y == ys[0] else None)
        note = f" ({row.inconclusive})" if row.inconclusive else ""
        ax.annotate(f"{row.cube.replace('FRT0000', '').replace('FRT', '')}{note}",
                    (0.02, y - 0.33), xycoords=("axes fraction", "data"), fontsize=8.5)
    ax.set_ylim(-0.8, len(CUBE_IDS) - 0.4)
    ax.set_yticks([]); ax.set_xlabel("median distance to mapped clay unit (m)", fontsize=9)
    ax.set_title("(c) patch vs background distance", fontsize=10, pad=14)
    ax.legend(fontsize=8, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.88))
    ax.grid(True, axis="x", color="0.93"); ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("Carbonate detections concentrate toward the clay-basement contact "
                 "(red = olivine unit, blue = phyllosilicate basement, dashed = mafic cap)", fontsize=11)
    fig.savefig(OUT / "fig_S_carbonate.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig_S_carbonate.pdf", bbox_inches="tight")
    plt.close(fig)

    log = {"script": Path(__file__).name, "script_md5": md5(Path(__file__)),
           "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
           "patches_csv_md5": md5(PATCHES), "stats": stats}
    (OUT / "run_log_fig_S_carbonate.json").write_text(json.dumps(log, indent=2, default=str))
    print("done ->", OUT)


if __name__ == "__main__":
    main()
