"""Regenerate all map figures from saved arrays with the shared unit style.

Rebuilds (without reloading any cube data from Z:) every band-centre /
detection map figure produced in the band-centre campaign, drawing the
geologic units with figstyle.draw_units (olivine red, clay blue, cap dark
dashed, white casing). Writes to the SAME filenames as the originating
scripts. Deterministic.

Usage:  python code/refigure_maps.py
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
from contact_band_center_map import (  # noqa: E402
    CMAP_RANGE as CM23, OUT as T0, PHASE_TICKS as PT23, STRONG_DEPTH as SD23, WEAK_DEPTH as WD23,
    unit_masks,
)
from contact_band_center_1400 import (  # noqa: E402
    CMAP_RANGE as CM14, PHASE_TICKS as PT14, STRONG_DEPTH as SD14, WEAK_DEPTH as WD14,
)
from contact_band_center_1400_plebani import (  # noqa: E402
    CMAP_RANGE as CM14P, PHASE_TICKS as PT14P, STRONG_DEPTH as SD14P,
)
from contact_band_center_modes import LONG_TAIL, PATCHES, patch_rowcol  # noqa: E402
from contact_anomaly_ratios import DECILES, OUT as AR_OUT  # noqa: E402
from figstyle import draw_units, label_panels, unit_legend  # noqa: E402

B14 = T0 / "band1400"
PLE23 = T0 / "plebani"
PLE14 = B14 / "plebani"
OH = T0 / "oh_filter"
MF23 = T0 / "matched_filter" / "2300"
HI_COL, TAIL_COL = "#E69F00", "#CC3311"


LOCKED = []


def save(fig, path: Path) -> None:
    """Label panels, then save png+pdf; if a file is locked (open in a viewer), write *_new instead."""
    label_panels(fig)
    for ext in (".png", ".pdf"):
        target = path.with_suffix(ext)
        try:
            fig.savefig(target, dpi=200 if ext == ".png" else None, bbox_inches="tight")
        except OSError:
            alt = path.with_name(path.name + "_new").with_suffix(ext)
            fig.savefig(alt, dpi=200 if ext == ".png" else None, bbox_inches="tight")
            LOCKED.append(str(target))


def base(ax, valid):
    ax.imshow(np.where(valid, 0.93, np.nan), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])


def two_panel(valid, fr, layers, titles, suptitle, cmap_range, ticks, cbar_label, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharex=True, sharey=True)
    im = None
    for ax, lay, title in zip(axes, layers, titles):
        base(ax, valid)
        im = ax.imshow(lay, cmap="viridis", vmin=cmap_range[0], vmax=cmap_range[1], interpolation="nearest")
        draw_units(ax, fr)
        ax.set_title(title, fontsize=10)
    unit_legend(axes[0])
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_ticks([t for t, _ in ticks]); cbar.set_ticklabels([lab for _, lab in ticks])
    cbar.ax.tick_params(labelsize=8); cbar.set_label(cbar_label, fontsize=9)
    fig.suptitle(suptitle, fontsize=11)
    save(fig, path)
    plt.close(fig)


def detection_panel(valid, fr, cen_det, snr, thr, label, win, suptitle, path):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.5), sharex=True, sharey=True)
    base(axes[0], valid)
    im = axes[0].imshow(cen_det, cmap="plasma", vmin=win[0], vmax=win[1], interpolation="nearest")
    axes[0].set_title(f"{label} detections (SNR >= {thr:.1f}), coloured by fitted centre", fontsize=10)
    cb = fig.colorbar(im, ax=axes[0], shrink=0.8, pad=0.02); cb.set_label("fitted band centre (um)", fontsize=8.5)
    base(axes[1], valid)
    im2 = axes[1].imshow(snr, cmap="viridis", vmin=0, vmax=max(6.0, thr * 1.5), interpolation="nearest")
    axes[1].set_title("amplitude SNR (all usable pixels)", fontsize=10)
    fig.colorbar(im2, ax=axes[1], shrink=0.8, pad=0.02)
    for ax in axes:
        draw_units(ax, fr)
    unit_legend(axes[0])
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    save(fig, path)
    plt.close(fig)


def main() -> None:
    t0 = datetime.now(timezone.utc)
    oh_sum = pd.read_csv(OH / "oh_unit_summary.csv")
    mf_sum = pd.read_csv(MF23 / "unit_summary.csv")
    patches_all = pd.read_csv(PATCHES)
    made = []
    for cube in CUBE_IDS:
        c = load_cube(cube)
        m = masks_from_labels(c["labels"])
        units, d, fr = unit_masks(cube, m)
        valid = m["valid"]

        # Tier 0 2.3 um (MTR3)
        cen = np.load(T0 / f"{cube}_bc230_center.npy"); dep = np.load(T0 / f"{cube}_bc230_depth.npy")
        edg = np.load(T0 / f"{cube}_bc230_edgeflag.npy")
        strong = np.isfinite(cen) & (dep >= SD23) & ~edg
        weak = np.isfinite(cen) & (dep >= WD23) & (dep < SD23)
        two_panel(valid, fr, (np.where(strong, cen, np.nan), np.where(weak, cen, np.nan)),
                  (f"{cube} - strong band (depth >= {SD23})", f"{cube} - weak band ({WD23}-{SD23}, noise-dominated)"),
                  f"{cube}: 2.3 um band-centre map (MTR3 chain)", CM23, PT23, "2.3 um band centre (um)",
                  T0 / f"fig_map_{cube}")
        made.append(f"fig_map_{cube}")

        # high-decile map (Tier 0 arrays)
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharex=True, sharey=True)
        for ax, (uname, fkey) in zip(axes, (("olivine", "oliv"), ("clay", "clay"))):
            unit = valid & (fr[fkey] >= 0.5); sel = unit & strong
            qhi = float(np.quantile(cen[sel], DECILES[1])) if sel.sum() >= 1000 else np.nan
            base(ax, valid)
            hi = sel & (cen >= qhi) if np.isfinite(qhi) else np.zeros_like(sel)
            tail = sel & (cen >= LONG_TAIL)
            lay = np.full(cen.shape, np.nan); lay[hi] = 0
            if uname == "olivine":
                lay[tail] = 1
            ax.imshow(lay, cmap=matplotlib.colors.ListedColormap([HI_COL, TAIL_COL]), vmin=0, vmax=1,
                      interpolation="nearest")
            draw_units(ax, fr)
            title = f"{uname}: high decile >= {qhi:.4f} um (n={int(hi.sum())})"
            if uname == "olivine":
                title += f"; dark red = tail >= {LONG_TAIL} (n={int(tail.sum())})"
                pdfc = patches_all[(patches_all.cube == cube) & patches_all.on_map]
                if len(pdfc):
                    pr, pc = patch_rowcol(cube, pdfc, cen.shape)
                    ax.plot(pc, pr, "x", color="magenta", ms=4, mew=1.0, label="MgCO3 patch")
                    ax.legend(loc="lower right", fontsize=8)
            ax.set_title(title, fontsize=9)
        unit_legend(axes[1])
        fig.suptitle(f"{cube}: strong-tier band-centre high-decile pixels (MTR3 chain; withdrawn NW field "
                     "shown for the record)", fontsize=10.5)
        fig.tight_layout()
        save(fig, AR_OUT / f"fig_highdecile_map_{cube}")
        plt.close(fig)
        made.append(f"fig_highdecile_map_{cube}")

        # MTR3 1.4 um (restricted window)
        cen14 = np.load(B14 / f"{cube}_bc140_center.npy"); dep14 = np.load(B14 / f"{cube}_bc140_depth.npy")
        edg14 = np.load(B14 / f"{cube}_bc140_edgeflag.npy")
        s14 = np.isfinite(cen14) & (dep14 >= SD14) & ~edg14
        w14 = np.isfinite(cen14) & (dep14 >= WD14) & (dep14 < SD14)
        two_panel(valid, fr, (np.where(s14, cen14, np.nan), np.where(w14, cen14, np.nan)),
                  (f"{cube} - strong band (depth >= {SD14})", f"{cube} - weak band ({WD14}-{SD14}, noise-dominated)"),
                  f"{cube}: 1.4 um band-centre map (MTR3 chain, window 1.36-1.425 um)", CM14, PT14,
                  "1.4 um band centre (um)", B14 / f"fig_map_{cube}")
        made.append(f"band1400/fig_map_{cube}")

        # Plebani 2.3 um strong/weak + filled
        rs = np.load(PLE23 / f"{cube}_bcP_center_strong.npy"); rw = np.load(PLE23 / f"{cube}_bcP_center_weak.npy")
        two_panel(valid, fr, (rs, rw),
                  (f"{cube} - strong tier (cell median, depth >= {SD23})", "weak tier (noise-dominated)"),
                  f"{cube}: 2.3 um band centre, PLEBANI BLAND-RATIOED chain", CM23, PT23,
                  "2.3 um band centre (um)", PLE23 / f"fig_map_{cube}")
        rf = np.load(PLE23 / f"{cube}_bcP_center_strong_filled.npy")
        rm = np.load(PLE23 / f"{cube}_bcP_center_strong_filled_med3.npy")
        two_panel(valid, fr, (rf, rm),
                  ("gap-filled (nearest sensor sample <= 30 m)", "3x3 median smoothed (display only)"),
                  f"{cube}: 2.3 um band centre, Plebani chain, strong tier - gap-filled", CM23, PT23,
                  "2.3 um band centre (um)", PLE23 / f"fig_map_filled_{cube}")
        made += [f"plebani/fig_map_{cube}", f"plebani/fig_map_filled_{cube}"]

        # Plebani 1.4 um strong/weak + filled
        rs14 = np.load(PLE14 / f"{cube}_bc140P_center_strong.npy"); rw14 = np.load(PLE14 / f"{cube}_bc140P_center_weak.npy")
        two_panel(valid, fr, (rs14, rw14), (f"{cube} - strong tier", f"{cube} - weak tier"),
                  f"{cube}: 1.4 um band centre, PLEBANI BLAND-RATIOED chain (strong >= {SD14P})", CM14P, PT14P,
                  "1.4 um band centre (um)", PLE14 / f"fig_map_{cube}")
        rf14 = np.load(PLE14 / f"{cube}_bc140P_center_strong_filled.npy")
        rm14 = np.load(PLE14 / f"{cube}_bc140P_center_strong_filled_med3.npy")
        two_panel(valid, fr, (rf14, rm14),
                  ("gap-filled (nearest sensor sample <= 30 m)", "3x3 median smoothed (display only)"),
                  f"{cube}: 1.4 um band centre, Plebani chain, strong tier - gap-filled", CM14P, PT14P,
                  "1.4 um band centre (um)", PLE14 / f"fig_map_filled_{cube}")
        made += [f"band1400/plebani/fig_map_{cube}", f"band1400/plebani/fig_map_filled_{cube}"]

        # matched-filter detection maps
        thr = float(oh_sum[oh_sum.cube == cube].thr_snr.iloc[0])
        detection_panel(valid, fr, np.load(OH / f"{cube}_oh_centre_det.npy"), np.load(OH / f"{cube}_oh_snr.npy"),
                        thr, "OH", (1.370, 1.430),
                        f"{cube}: 1.4 um OH matched-filter detection, Plebani bland-ratioed chain\n"
                        "threshold = 99th percentile of the band-free null distribution", OH / f"fig_oh_{cube}")
        thr = float(mf_sum[mf_sum.cube == cube].thr_snr.iloc[0])
        detection_panel(valid, fr, np.load(MF23 / f"{cube}_centre_det.npy"), np.load(MF23 / f"{cube}_snr.npy"),
                        thr, "2.3 um metal-OH", (2.280, 2.350),
                        f"{cube}: 2.3 um metal-OH matched-filter detection, Plebani bland-ratioed chain\n"
                        "threshold = 99th percentile of the band-free null distribution", MF23 / f"fig_2300_{cube}")
        made += [f"oh_filter/fig_oh_{cube}", f"matched_filter/2300/fig_2300_{cube}"]
        print(cube, "done")

    log = {"script": Path(__file__).name, "script_md5": md5(Path(__file__)),
           "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
           "style": "figstyle.draw_units: olivine red, clay blue, cap dark dashed, white casing",
           "figures": made, "locked_written_as_new": LOCKED}
    (T0 / "run_log_refigure_maps.json").write_text(json.dumps(log, indent=2, default=str))
    print("done:", len(made), "figures; locked ->", LOCKED)


if __name__ == "__main__":
    main()
