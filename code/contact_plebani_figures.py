"""Figures for the Plebani-ratioed band-centre replication.

Regenerates the Tier 0 figure battery from the contact_band_center_plebani
outputs and adds the verdict figure: old chain (MTR3, unratioed) vs new
chain (TRR3, bland-column-ratioed) tail maps side by side. Deterministic.

Usage:  python code/contact_plebani_figures.py
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
    CMAP_RANGE, DIST_LABELS, OUT as T0_OUT, PHASE_TICKS, SEARCH, STRONG_DEPTH, unit_masks,
)
from contact_band_center_modes import LONG_TAIL, MODE_MAP_THRESH, PATCHES, patch_rowcol  # noqa: E402
from figstyle import draw_units, label_panels  # noqa: E402

PLE = T0_OUT / "plebani"
OUT = PLE
HI_COL, TAIL_COL = "#E69F00", "#CC3311"
ZOOM_UM = (2.27, 2.40)
ZOOM_TICKS = (2.30, 2.31, 2.32, 2.33, 2.34, 2.35)
RNG = np.random.default_rng(42)


def contours(ax, fr) -> None:
    draw_units(ax, fr, units=("oliv", "clay", "cap"))


def main() -> None:
    t0 = datetime.now(timezone.utc)
    patches_all = pd.read_csv(PATCHES)
    prof_new = pd.read_csv(PLE / "profile.csv")
    prof_old = pd.read_csv(T0_OUT / "band_center_map_profile.csv")
    for cube in CUBE_IDS:
        c = load_cube(cube)
        m = masks_from_labels(c["labels"])
        units, d, fr = unit_masks(cube, m)
        rs = np.load(PLE / f"{cube}_bcP_center_strong.npy")
        rw = np.load(PLE / f"{cube}_bcP_center_weak.npy")
        # old-chain tail for comparison
        cen0 = np.load(T0_OUT / f"{cube}_bc230_center.npy")
        dep0 = np.load(T0_OUT / f"{cube}_bc230_depth.npy")
        edg0 = np.load(T0_OUT / f"{cube}_bc230_edgeflag.npy")
        strong0 = np.isfinite(cen0) & (dep0 >= STRONG_DEPTH) & ~edg0
        oliv_cells = m["valid"] & (fr["oliv"] >= 0.5)

        # 1. centre maps
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharex=True, sharey=True)
        for ax, r, title in zip(axes, (rs, rw),
                                ("strong tier (cell median, depth >= 0.015)",
                                 "weak tier (noise-dominated)")):
            bg = np.where(m["valid"], 0.92, np.nan)
            ax.imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            im = ax.imshow(r, cmap="viridis", vmin=CMAP_RANGE[0], vmax=CMAP_RANGE[1],
                           interpolation="nearest")
            draw_units(ax, fr, units=("clay", "cap"))
            ax.set_title(f"{cube} - {title}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
        cbar.set_ticks([t for t, _ in PHASE_TICKS])
        cbar.set_ticklabels([lab for _, lab in PHASE_TICKS]); cbar.ax.tick_params(labelsize=8)
        fig.suptitle(f"{cube}: 2.3 um band centre, PLEBANI BLAND-RATIOED chain", fontsize=11)
        label_panels(fig)
        fig.savefig(OUT / f"fig_map_{cube}.png", dpi=200, bbox_inches="tight")
        fig.savefig(OUT / f"fig_map_{cube}.pdf", bbox_inches="tight")
        plt.close(fig)

        # 2. high-decile + tail map (cell level) with contours and patches
        selc = oliv_cells & np.isfinite(rs)
        if selc.sum() > 500:
            qhi = float(np.quantile(rs[selc], 0.90))
            hi = selc & (rs >= qhi)
            tail = selc & (rs >= LONG_TAIL)
            fig, ax = plt.subplots(figsize=(8.5, 8))
            bg = np.where(m["valid"], 0.93, np.nan)
            ax.imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            lay = np.full(rs.shape, np.nan)
            lay[hi] = 0; lay[tail] = 1
            ax.imshow(lay, cmap=matplotlib.colors.ListedColormap([HI_COL, TAIL_COL]),
                      vmin=0, vmax=1, interpolation="nearest")
            contours(ax, fr)
            pdfc = patches_all[(patches_all.cube == cube) & patches_all.on_map]
            if len(pdfc):
                pr, pc = patch_rowcol(cube, pdfc, rs.shape)
                ax.plot(pc, pr, "x", color="magenta", ms=4, mew=1.0, label="MgCO3 patch")
                ax.legend(fontsize=8, loc="lower right")
            ax.set_title(f"{cube} olivine, Plebani chain: high decile >= {qhi:.4f} "
                         f"(n={int(hi.sum())}); dark red = tail >= {LONG_TAIL} (n={int(tail.sum())})",
                         fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            fig.tight_layout()
            label_panels(fig)
            fig.savefig(OUT / f"fig_highdecile_map_{cube}.png", dpi=200, bbox_inches="tight")
            fig.savefig(OUT / f"fig_highdecile_map_{cube}.pdf", bbox_inches="tight")
            plt.close(fig)

        # 3. VERDICT: old-chain vs new-chain tail maps
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharex=True, sharey=True)
        for ax, tail_mask, title in zip(
                axes,
                (oliv_cells & strong0 & (cen0 >= LONG_TAIL), oliv_cells & np.isfinite(rs) & (rs >= LONG_TAIL)),
                ("OLD chain: MTR3, unratioed", "NEW chain: TRR3, bland-column-ratioed")):
            bg = np.where(m["valid"], 0.93, np.nan)
            ax.imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            lay = np.where(tail_mask, 1.0, np.nan)
            ax.imshow(lay, cmap=matplotlib.colors.ListedColormap([TAIL_COL]),
                      vmin=0, vmax=1, interpolation="nearest")
            contours(ax, fr)
            ax.set_title(f"{title} (n={int(tail_mask.sum())})", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{cube}: 2.33 um tail (centre >= {LONG_TAIL}, strong tier) - chain comparison",
                     fontsize=11)
        fig.tight_layout()
        label_panels(fig)
        fig.savefig(OUT / f"fig_tail_compare_{cube}.png", dpi=200, bbox_inches="tight")
        fig.savefig(OUT / f"fig_tail_compare_{cube}.pdf", bbox_inches="tight")
        plt.close(fig)

        # 4. anomaly stacks with zoom
        sz = np.load(PLE / f"{cube}_stacks.npz")
        wav = sz["wav"]
        names = [k for k in sz.files if k != "wav"]
        by_unit = {"olivine": [k for k in names if k.startswith("olivine|")],
                   "clay": [k for k in names if k.startswith("clay|")]}
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
        for j, uname in enumerate(("olivine", "clay")):
            ax_full, ax_zoom = axes[0, j], axes[1, j]
            keys = by_unit[uname]
            cols = {}
            for k in keys:
                lab = k.split("|", 1)[1]
                cols[k] = "#0072B2" if lab.startswith("low") else ("#CC79A7" if lab.startswith("tail") else "#D55E00")
            curves = {}
            for k in keys:
                y = sz[k]
                curves[k] = y / np.nanmedian(y)
                ls = "--" if "tail" in k else "-"
                ax_full.plot(wav, curves[k], lw=1.7, color=cols[k], ls=ls, label=k.split("|", 1)[1])
                ax_zoom.plot(wav, curves[k], lw=1.9, color=cols[k], ls=ls)
            ax_full.set_xlim(1.3, 2.62)
            ax_full.set_title(f"{uname}: anomaly / unit-median (bland-ratioed spectra)", fontsize=10)
            ax_full.grid(True, color="0.96"); ax_full.spines[["top", "right"]].set_visible(False)
            ax_full.legend(fontsize=8, frameon=False)
            ax_zoom.set_xlim(ZOOM_UM)
            ax_zoom.axvline(2.30, color="0.35", lw=1.0, zorder=0)
            ax_zoom.set_xticks(ZOOM_TICKS)
            ax_zoom.set_xticklabels([f"{t:.2f}" for t in ZOOM_TICKS], fontsize=9)
            for t in ZOOM_TICKS[1:]:
                ax_zoom.axvline(t, color="0.90", lw=0.7, zorder=0)
            zs = (wav >= ZOOM_UM[0]) & (wav <= ZOOM_UM[1])
            if curves:
                ylo = min(np.nanmin(y[zs]) for y in curves.values())
                yhi = max(np.nanmax(y[zs]) for y in curves.values())
                pad = 0.15 * (yhi - ylo)
                ax_zoom.set_ylim(ylo - pad, yhi + pad)
            ax_zoom.set_xlabel("wavelength (um)")
            ax_zoom.set_title("zoom 2.3 um region", fontsize=9)
            ax_zoom.spines[["top", "right"]].set_visible(False)
        for ax in axes[:, 0]:
            ax.set_ylabel("ratioed I/F (normalized)")
        fig.suptitle(f"{cube}: unit-internal anomaly ratios, PLEBANI chain", fontsize=11)
        fig.tight_layout()
        label_panels(fig)
        fig.savefig(OUT / f"fig_anomaly_{cube}.png", dpi=200, bbox_inches="tight")
        fig.savefig(OUT / f"fig_anomaly_{cube}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(cube, "figures done")

    # 5. profile comparison old vs new
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    xs = np.arange(len(DIST_LABELS))
    for ax, cube in zip(axes.ravel(), CUBE_IDS):
        po = prof_old[(prof_old.cube == cube) & (prof_old.tier == "strong")]
        pn = prof_new[prof_new.cube == cube]
        ax.plot(xs, po.center_med_um, "-o", color="0.6", lw=1.6, ms=4,
                label=f"old chain (n={int(po.n.sum())})")
        ax.plot(xs, pn.center_med_um, "-o", color=PALETTE[cube], lw=2.0, ms=5,
                label=f"Plebani chain (n={int(pn.n.sum())})")
        ax.set_title(cube, fontsize=10, color=PALETTE[cube])
        ax.set_xticks(xs); ax.set_xticklabels(DIST_LABELS)
        ax.grid(True, color="0.94"); ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
    for ax in axes[1]:
        ax.set_xlabel("distance from mapped clay (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("2.3 um band centre (um)")
    fig.suptitle("Map-pure olivine, strong tier: centre vs distance - chain comparison", fontsize=11)
    fig.tight_layout()
    label_panels(fig)
    fig.savefig(OUT / "fig_profile_compare.png", dpi=200)
    fig.savefig(OUT / "fig_profile_compare.pdf")
    plt.close(fig)

    # 6. KDE histograms per unit (cell level)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    style = {"clay": ("clay", "#7f7f7f"), "cap": ("cap", "#2b2b2b"), "oliv": ("olivine", "#d55e00")}
    grid_x = np.linspace(SEARCH[0], SEARCH[1], 400)
    for ax, cube in zip(axes.ravel(), CUBE_IDS):
        c = load_cube(cube)
        m = masks_from_labels(c["labels"])
        units, d, fr = unit_masks(cube, m)
        rs = np.load(PLE / f"{cube}_bcP_center_strong.npy")
        for fkey, (lab, col) in style.items():
            v = rs[m["valid"] & (fr[fkey] >= 0.9) & np.isfinite(rs)]
            if v.size < 200:
                continue
            vs = v if v.size <= 50000 else RNG.choice(v, 50000, replace=False)
            kde = gaussian_kde(vs, bw_method=0.0025 / vs.std())
            ax.plot(grid_x, kde(grid_x), color=col, lw=1.8, label=f"{lab} (n={v.size})")
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
    fig.suptitle("Plebani chain, strong tier (cell medians): band-centre KDE by map unit", fontsize=11)
    fig.tight_layout()
    label_panels(fig)
    fig.savefig(OUT / "fig_histograms.png", dpi=200)
    fig.savefig(OUT / "fig_histograms.pdf")
    plt.close(fig)

    log = {"script": Path(__file__).name, "script_md5": md5(Path(__file__)),
           "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat()}
    (OUT / "run_log_figures.json").write_text(json.dumps(log, indent=2, default=str))
    print("done ->", OUT)


if __name__ == "__main__":
    main()
