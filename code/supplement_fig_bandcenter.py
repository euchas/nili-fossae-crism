"""Supplement figure: no ordered 2.3 um band-position zonation at the contact.

Panels: (a) FRT00003E12 strong-tier band-centre map (MTR3 chain) - no
chlorite-range zones, no contact-following band-position structure;
(b) band-centre distributions by map unit (KDE, strong tier, all cubes as
thin lines + 3E12 bold); (c) centre vs distance from the mapped clay
contact in BOTH processing chains (MTR3 unratioed, grey; TRR3
bland-column-ratioed, colour) with the ~10 nm adjacent-phase step for
scale - shifts are far below one phase step and not reproducible between
chains. Deterministic (seed 42 for KDE subsample).

Usage:  python code/supplement_fig_bandcenter.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import CUBE_IDS  # noqa: E402
from contact_distance_to_contact import PALETTE, load_cube, masks_from_labels, md5  # noqa: E402
from contact_band_center_map import (  # noqa: E402
    CMAP_RANGE, DIST_LABELS, OUT as T0_OUT, PHASE_TICKS, STRONG_DEPTH, unit_masks,
)
from figstyle import draw_units  # noqa: E402

PLE = T0_OUT / "plebani"
OUT = ROOT / "manuscript_v10" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(42)
FOCUS = "FRT00003E12"


def main() -> None:
    t0 = datetime.now(timezone.utc)
    fig = plt.figure(figsize=(15.5, 6.0))
    gs = fig.add_gridspec(1, 3, width_ratios=(1.2, 0.85, 0.85), wspace=0.22)

    # (a) 3E12 strong-tier centre map
    c = load_cube(FOCUS)
    m = masks_from_labels(c["labels"])
    units, d, fr = unit_masks(FOCUS, m)
    cen = np.load(T0_OUT / f"{FOCUS}_bc230_center.npy")
    dep = np.load(T0_OUT / f"{FOCUS}_bc230_depth.npy")
    edg = np.load(T0_OUT / f"{FOCUS}_bc230_edgeflag.npy")
    strong = np.isfinite(cen) & (dep >= STRONG_DEPTH) & ~edg
    ax = fig.add_subplot(gs[0, 0])
    bg = np.where(np.isfinite(cen), 0.93, np.nan)
    ax.imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    im = ax.imshow(np.where(strong, cen, np.nan), cmap="viridis",
                   vmin=CMAP_RANGE[0], vmax=CMAP_RANGE[1], interpolation="nearest")
    draw_units(ax, fr, units=("clay", "cap"))
    ax.set_title(f"(a) {FOCUS}: 2.3 um band centre, strong tier", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_ticks([t for t, _ in PHASE_TICKS])
    cbar.set_ticklabels([lab for _, lab in PHASE_TICKS])
    cbar.ax.tick_params(labelsize=7.5)

    # (b) KDE distributions per unit
    ax = fig.add_subplot(gs[0, 1])
    grid_x = np.linspace(2.283, 2.35, 400)
    style = {"clay map-pure": "#7f7f7f", "olivine map-pure": "#d55e00"}
    for cube in CUBE_IDS:
        cc = load_cube(cube)
        mm = masks_from_labels(cc["labels"])
        uu, dd_, _ = unit_masks(cube, mm)
        cen_i = np.load(T0_OUT / f"{cube}_bc230_center.npy")
        dep_i = np.load(T0_OUT / f"{cube}_bc230_depth.npy")
        edg_i = np.load(T0_OUT / f"{cube}_bc230_edgeflag.npy")
        st = np.isfinite(cen_i) & (dep_i >= STRONG_DEPTH) & ~edg_i
        for uname, col in style.items():
            v = cen_i[uu[uname] & st]
            if v.size < 500:
                continue
            vs = v if v.size <= 50000 else RNG.choice(v, 50000, replace=False)
            kde = gaussian_kde(vs, bw_method=0.0025 / vs.std())
            bold = cube == FOCUS
            ax.plot(grid_x, kde(grid_x), color=col, lw=2.2 if bold else 0.9,
                    alpha=1.0 if bold else 0.45,
                    label=(uname.replace(" map-pure", "") if bold else None))
    for xv in (2.33, 2.35):
        ax.axvline(xv, color="0.85", lw=0.8, ls=":")
    ax.annotate("chlorite /\nFe-chlorite range\n(not observed)", (2.336, 0.75),
                xycoords=("data", "axes fraction"), fontsize=8, color="0.4")
    ax.set_xlabel("2.3 um band centre (um)", fontsize=9)
    ax.set_ylabel("density", fontsize=9)
    ax.set_title("(b) band-centre distributions, all cubes", fontsize=10)
    ax.legend(fontsize=8.5, frameon=False)
    ax.grid(True, color="0.95"); ax.spines[["top", "right"]].set_visible(False)

    # (c) dual-chain distance profiles
    ax = fig.add_subplot(gs[0, 2])
    prof_old = pd.read_csv(T0_OUT / "band_center_map_profile.csv")
    prof_new = pd.read_csv(PLE / "profile.csv")
    xs = np.arange(len(DIST_LABELS))
    for cube in CUBE_IDS:
        po = prof_old[(prof_old.cube == cube) & (prof_old.tier == "strong")]
        pn = prof_new[prof_new.cube == cube]
        rel_o = (po.center_med_um.to_numpy() - po.center_med_um.to_numpy()[-1]) * 1000
        rel_n = (pn.center_med_um.to_numpy() - pn.center_med_um.to_numpy()[-1]) * 1000
        ax.plot(xs, rel_o, "-", color="0.65", lw=1.1)
        ax.plot(xs, rel_n, "-o", color=PALETTE[cube], lw=1.6, ms=3.5)
    ax.axhspan(-10, 10, color="0.94", zorder=0)
    ax.annotate("one adjacent-phase step (+/-10 nm)", (0.03, 8.2), fontsize=8, color="0.45")
    ax.axhline(0, color="0.8", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(DIST_LABELS, fontsize=8)
    ax.set_xlabel("distance from mapped clay contact (m)", fontsize=9)
    ax.set_ylabel("band-centre shift vs far field (nm)", fontsize=9)
    ax.set_title("(c) centre vs distance, two processing chains\n(grey = MTR3, colour = bland-ratioed TRR3)",
                 fontsize=9.5)
    ax.grid(True, color="0.95"); ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("No ordered 2.3 um band-position zonation toward the basement contact", fontsize=11.5)
    fig.savefig(OUT / "fig_S_bandcenter.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig_S_bandcenter.pdf", bbox_inches="tight")
    plt.close(fig)

    log = {"script": Path(__file__).name, "script_md5": md5(Path(__file__)),
           "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat()}
    (OUT / "run_log_fig_S_bandcenter.json").write_text(json.dumps(log, indent=2, default=str))
    print("done ->", OUT)


if __name__ == "__main__":
    main()
