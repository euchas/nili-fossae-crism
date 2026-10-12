"""Mode-split analysis of the Tier 0 2.3 um band-centre maps.

Digs into the bimodal band-centre distributions (short mode ~2.306 um vs
long mode ~2.316-2.320 um) seen in the strong-band tier:

1. Mode split at several thresholds: spatial maps of short- vs long-mode
   pixels per unit, and short-mode fraction vs distance from the mapped
   clay unit (map-pure olivine, strong tier). Short mode hugging clay
   outcrops favours sub-pixel clay contamination; interior patches favour
   a real second phase.
2. Unit-constrained band-centre maps (olivine / clay / cap at map
   fraction >= 0.5) to remove cross-unit spillover from the display.
3. Channel-aligned histograms + smooth density curves (the 3.3 nm bins of
   the Tier 0 figure alias the 6.6 nm channel comb).
4. Plebani carbonate-patch cross-check: are long-tail centres
   (>= 2.325 um) enriched near the stage6 carbonate patches?
   (Exploratory pixel-level counts, not block statistics.)

Requires contact_band_center_map.py outputs. Seed 42 (KDE subsample).

Usage:  python code/contact_band_center_modes.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.stats import gaussian_kde

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import CUBE_IDS  # noqa: E402
from contact_distance_to_contact import PALETTE, PIX_M, load_cube, masks_from_labels, md5  # noqa: E402
from contact_band_center_map import (  # noqa: E402
    CMAP_RANGE, DIST_EDGES, DIST_LABELS, OUT as T0_OUT, PHASE_TICKS, SEARCH, STRONG_DEPTH, unit_masks,
)
from contact_band_center_2300 import read_window  # noqa: E402
from figstyle import draw_units, label_panels  # noqa: E402

OUT = T0_OUT / "modes"
OUT.mkdir(parents=True, exist_ok=True)
PATCHES = ROOT / "outputs" / "revision" / "contact_3d" / "stage6_carbonate" / "carbonate_patches.csv"
GEOMAP = ROOT / "outputs" / "revision" / "contact_3d" / "geomap" / "frame_ctx"

THRESHOLDS = (2.308, 2.311, 2.314)   # antimode sweep; 2.308/2.314 sit between channel centres
MODE_MAP_THRESH = 2.311
LONG_TAIL = 2.325                    # candidate carbonate/chlorite tail
PATCH_RADII_M = (54.0, 108.0)
KDE_MAX_N = 50_000
KDE_BW_NM = 2.5
RNG = np.random.default_rng(42)
SHORT_COL, LONG_COL = "#0072B2", "#D55E00"


def load_tier0(cube: str) -> dict:
    return {k: np.load(T0_OUT / f"{cube}_bc230_{k}.npy") for k in ("center", "depth", "edgeflag")}


def patch_rowcol(cube: str, df: pd.DataFrame, shape: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Invert the fractions-raster GDAL transform: map metres -> row/col."""
    t = np.load(GEOMAP / f"{cube}_fractions.npz")["transform"]  # (c, a, b, f, d, e)
    c0, a, b, f0, dd, e = [float(v) for v in t]
    A = np.array([[a, b], [dd, e]])
    inv = np.linalg.inv(A)
    xy = np.stack([df.x_m.to_numpy() - c0, df.y_m.to_numpy() - f0])
    colrow = inv @ xy
    col, row = np.floor(colrow[0]).astype(int), np.floor(colrow[1]).astype(int)
    ok = (row >= 0) & (row < shape[0]) & (col >= 0) & (col < shape[1])
    return row[ok], col[ok]


def main() -> None:
    t0 = datetime.now(timezone.utc)
    wav, _, _ = read_window(CUBE_IDS[0])
    chan = wav[(wav >= SEARCH[0]) & (wav <= SEARCH[1])]
    ch_edges = np.concatenate([[chan[0] - np.diff(chan).mean() / 2],
                               0.5 * (chan[1:] + chan[:-1]),
                               [chan[-1] + np.diff(chan).mean() / 2]])
    patches_all = pd.read_csv(PATCHES)
    frac_rows, patch_rows, clay_rows, inputs = [], [], [], {}
    grids = {}
    for cube in CUBE_IDS:
        c = load_cube(cube)
        m = masks_from_labels(c["labels"])
        units, d, fr = unit_masks(cube, m)
        t = load_tier0(cube)
        inputs[cube] = {"center_md5": md5(T0_OUT / f"{cube}_bc230_center.npy")}
        strong = np.isfinite(t["center"]) & (t["depth"] >= STRONG_DEPTH) & ~t["edgeflag"]
        cen = t["center"]
        grids[cube] = {"cen": cen, "strong": strong, "fr": fr, "units": units, "d": d}

        # 1. short-mode fraction vs distance from mapped clay, per threshold
        sel = units["olivine map-pure"] & strong & (d["cap"] > 90)
        dist = d["clay"]
        for th in THRESHOLDS:
            short = cen < th
            for (lo, hi), lab in zip(zip(DIST_EDGES[:-1], DIST_EDGES[1:]), DIST_LABELS):
                bm = sel & (dist > lo) & (dist <= hi)
                n = int(bm.sum())
                frac_rows.append({"cube": cube, "threshold_um": th, "dist_bin_m": lab, "n": n,
                                  "n_short": int((bm & short).sum()),
                                  "frac_short": float((bm & short).sum() / n) if n else np.nan})
            ds, dl = dist[sel & short], dist[sel & ~short]
            clay_rows.append({"cube": cube, "unit": "olivine map-pure", "threshold_um": th,
                              "n_short": int(ds.size), "n_long": int(dl.size),
                              "dist_med_short_m": float(np.median(ds)) if ds.size else np.nan,
                              "dist_med_long_m": float(np.median(dl)) if dl.size else np.nan,
                              "frac_short_unit": float(ds.size / (ds.size + dl.size)) if ds.size + dl.size else np.nan})
            cs = units["clay map-pure"] & strong
            clay_rows.append({"cube": cube, "unit": "clay map-pure", "threshold_um": th,
                              "n_short": int((cs & short).sum()), "n_long": int((cs & ~short).sum()),
                              "dist_med_short_m": np.nan, "dist_med_long_m": np.nan,
                              "frac_short_unit": float((cs & short).sum() / cs.sum()) if cs.sum() else np.nan})

        # 4. Plebani patch cross-check (pixel-level, exploratory)
        pdfc = patches_all[(patches_all.cube == cube) & patches_all.on_map]
        if len(pdfc):
            pr, pc = patch_rowcol(cube, pdfc, cen.shape)
            pmask = np.zeros(cen.shape, bool)
            pmask[pr, pc] = True
            dpatch = distance_transform_edt(~pmask, sampling=(PIX_M, PIX_M))
            oliv_disp = m["valid"] & (fr["oliv"] >= 0.5) & strong
            tail = oliv_disp & (cen >= LONG_TAIL)
            row = {"cube": cube, "n_patches_on_map": int(len(pdfc)), "n_patch_px_in_grid": int(pmask.sum()),
                   "n_strong_oliv": int(oliv_disp.sum()), "n_tail": int(tail.sum()),
                   "center_med_near_patch_um": float(np.median(cen[oliv_disp & (dpatch <= 36)]))
                   if (oliv_disp & (dpatch <= 36)).any() else np.nan,
                   "center_med_unit_um": float(np.median(cen[oliv_disp]))}
            for r in PATCH_RADII_M:
                f_tail = float((tail & (dpatch <= r)).sum() / tail.sum()) if tail.any() else np.nan
                f_all = float((oliv_disp & (dpatch <= r)).sum() / oliv_disp.sum())
                row[f"frac_tail_within_{int(r)}m"] = f_tail
                row[f"frac_unit_within_{int(r)}m"] = f_all
                row[f"enrichment_{int(r)}m"] = f_tail / f_all if f_all else np.nan
            patch_rows.append(row)
            grids[cube]["dpatch"] = dpatch
            grids[cube]["pmask_rc"] = (pr, pc)
        print(cube, "done")

    fdf = pd.DataFrame(frac_rows); fdf.to_csv(OUT / "mode_fraction_vs_distance.csv", index=False)
    cdf = pd.DataFrame(clay_rows); cdf.to_csv(OUT / "mode_split_summary.csv", index=False)
    pdf2 = pd.DataFrame(patch_rows); pdf2.to_csv(OUT / "plebani_tail_check.csv", index=False)
    print(fdf.round(4).to_string()); print(cdf.round(4).to_string()); print(pdf2.round(4).to_string())

    # Fig A: mode maps (threshold 2.311) - olivine and clay units
    for cube in CUBE_IDS:
        g = grids[cube]
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharex=True, sharey=True)
        for ax, (uname, fkey) in zip(axes, (("olivine", "oliv"), ("clay", "clay"))):
            um = (g["fr"][fkey] >= 0.5) & g["strong"]
            bg = np.where(np.isfinite(g["cen"]), 0.92, np.nan)
            ax.imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            mode = np.full(g["cen"].shape, np.nan)
            mode[um & (g["cen"] < MODE_MAP_THRESH)] = 0
            mode[um & (g["cen"] >= MODE_MAP_THRESH)] = 1
            ax.imshow(mode, cmap=matplotlib.colors.ListedColormap([SHORT_COL, LONG_COL]),
                      vmin=0, vmax=1, interpolation="nearest")
            draw_units(ax, g["fr"], units=("clay",))
            if "pmask_rc" in g and uname == "olivine":
                pr, pc = g["pmask_rc"]
                ax.plot(pc, pr, "x", color="magenta", ms=4, mew=1.0, label="carbonate patch")
                ax.legend(fontsize=8, loc="lower right")
            ax.set_title(f"{uname} unit (f >= 0.5), strong tier", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{cube}: band-centre mode split at {MODE_MAP_THRESH} um "
                     f"(blue = short mode < {MODE_MAP_THRESH}, orange = long mode; "
                     "blue = mapped clay outline)", fontsize=11)
        fig.tight_layout()
        label_panels(fig)
        fig.savefig(OUT / f"fig_modes_{cube}.png", dpi=200, bbox_inches="tight")
        fig.savefig(OUT / f"fig_modes_{cube}.pdf", bbox_inches="tight")
        plt.close(fig)

    # Fig B: short-mode fraction vs distance, all thresholds
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
    xs = np.arange(len(DIST_LABELS))
    for ax, cube in zip(axes.ravel(), CUBE_IDS):
        for th, ls in zip(THRESHOLDS, (":", "-", "--")):
            sub = fdf[(fdf.cube == cube) & (fdf.threshold_um == th)]
            ax.plot(xs, sub.frac_short, marker="o", ls=ls, color=PALETTE[cube], lw=1.8, ms=4,
                    label=f"th {th:.3f} um")
        ax.set_title(cube, fontsize=10, color=PALETTE[cube])
        ax.set_xticks(xs); ax.set_xticklabels(DIST_LABELS)
        ax.grid(True, color="0.94"); ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
    for ax in axes[1]:
        ax.set_xlabel("distance from mapped clay (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("short-mode fraction")
    fig.suptitle("Map-pure olivine, strong tier: fraction of short-mode pixels vs distance from mapped clay",
                 fontsize=11)
    fig.tight_layout()
    label_panels(fig)
    fig.savefig(OUT / "fig_short_fraction_vs_distance.png", dpi=200)
    fig.savefig(OUT / "fig_short_fraction_vs_distance.pdf")
    plt.close(fig)

    # Fig C: unit-constrained band-centre maps (strong tier)
    for cube in CUBE_IDS:
        g = grids[cube]
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.0), sharex=True, sharey=True)
        for ax, (uname, fkey) in zip(axes, (("olivine", "oliv"), ("clay", "clay"), ("cap", "cap"))):
            um = (g["fr"][fkey] >= 0.5) & g["strong"]
            bg = np.where(np.isfinite(g["cen"]), 0.92, np.nan)
            ax.imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            im = ax.imshow(np.where(um, g["cen"], np.nan), cmap="viridis",
                           vmin=CMAP_RANGE[0], vmax=CMAP_RANGE[1], interpolation="nearest")
            draw_units(ax, g["fr"], units=("clay",))
            ax.set_title(f"{uname} unit only (f >= 0.5), n={int(um.sum())}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.015)
        cbar.set_ticks([t for t, _ in PHASE_TICKS])
        cbar.set_ticklabels([lab for _, lab in PHASE_TICKS]); cbar.ax.tick_params(labelsize=8)
        fig.suptitle(f"{cube}: unit-constrained 2.3 um band-centre maps, strong tier "
                     f"(depth >= {STRONG_DEPTH})", fontsize=11)
        label_panels(fig)
        fig.savefig(OUT / f"fig_units_{cube}.png", dpi=200, bbox_inches="tight")
        fig.savefig(OUT / f"fig_units_{cube}.pdf", bbox_inches="tight")
        plt.close(fig)

    # Fig D: channel-aligned histograms + KDE
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    style = {"clay map-pure": "#7f7f7f", "cap map-pure": "#2b2b2b", "olivine map-pure": "#d55e00"}
    grid_x = np.linspace(SEARCH[0], SEARCH[1], 400)
    for ax, cube in zip(axes.ravel(), CUBE_IDS):
        g = grids[cube]
        for uname, col in style.items():
            v = g["cen"][g["units"][uname] & g["strong"]]
            if v.size < 200:
                continue
            h, _ = np.histogram(v, bins=ch_edges, density=True)
            ax.stairs(h, ch_edges, color=col, alpha=0.30, fill=True)
            vs = v if v.size <= KDE_MAX_N else RNG.choice(v, KDE_MAX_N, replace=False)
            kde = gaussian_kde(vs, bw_method=(KDE_BW_NM / 1000.0) / vs.std())
            ax.plot(grid_x, kde(grid_x), color=col, lw=1.8, label=f"{uname} (n={v.size})")
        for xv, _ in PHASE_TICKS[:-1]:
            ax.axvline(xv, color="0.88", lw=0.7, zorder=0)
        ax.set_title(cube, fontsize=10, color=PALETTE[cube])
        ax.grid(True, color="0.96"); ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, frameon=False)
        ax.set_xlim(2.28, SEARCH[1])
    for ax in axes[1]:
        ax.set_xlabel("2.3 um band centre (um)")
    for ax in axes[:, 0]:
        ax.set_ylabel("density")
    fig.suptitle("Strong tier: channel-aligned histograms (bars, one bin per CRISM channel) "
                 f"with {KDE_BW_NM:.1f} nm KDE (lines)", fontsize=11)
    fig.tight_layout()
    label_panels(fig)
    fig.savefig(OUT / "fig_histograms_v2.png", dpi=200)
    fig.savefig(OUT / "fig_histograms_v2.pdf")
    plt.close(fig)

    log = {"script": Path(__file__).name, "script_md5": md5(Path(__file__)),
           "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
           "thresholds_um": THRESHOLDS, "mode_map_threshold_um": MODE_MAP_THRESH,
           "long_tail_um": LONG_TAIL, "patch_radii_m": PATCH_RADII_M, "strong_depth": STRONG_DEPTH,
           "kde": {"max_n": KDE_MAX_N, "bw_nm": KDE_BW_NM, "seed": 42},
           "patches_csv_md5": md5(PATCHES), "inputs": inputs,
           "note": "patch cross-check is pixel-level and exploratory, not block statistics"}
    (OUT / "run_log_modes.json").write_text(json.dumps(log, indent=2, default=str))
    print("done ->", OUT)


if __name__ == "__main__":
    main()
