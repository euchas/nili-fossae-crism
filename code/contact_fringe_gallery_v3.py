"""Final fringe figure: AV's selected panels, with a slope-map column.

Selected (2026-08-31): P4, P9, P10, P11 (002888 flat-edge), Q12 (006712
flat-edge), R2, R3, R8, R12 (002888 strongest), S5 (006712 strongest).
Reproduces the v2 panel selection deterministically (same ranking code and
inputs), then renders 4 columns: CTX 6 m + contact | HiRISE hillshade +
contact | slope map (fill masked) + contact | profile. Also writes
selected_sites.csv with lon/lat and frame coordinates for ArcGIS.

Usage: python code/contact_fringe_gallery_v3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
import rasterio
import shapely
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import transform as rtransform
from scipy import ndimage as ndi

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from contact_contact_fringe import HIRISE, X  # noqa: E402
from contact_coreg_frame import ctx_on_grid  # noqa: E402
from contact_fringe_gallery_v2 import NOISE_THR, PREFIX, clay_boundaries, draw_lines, hillshade  # noqa: E402

OUT = ROOT / "outputs" / "revision" / "contact_3d" / "stage7_fringe"
FRAME = ROOT / "outputs" / "revision" / "contact_3d" / "frame" / "coreg_offsets.json"
BOX_M = 120.0
N_SHOW = 12
SELECTED = ["P4", "P9", "P10", "P11", "Q12", "R2", "R3", "R8", "R12", "S5"]
LONLAT = "+proj=longlat +R=3393833.26 +no_defs"


def build_keep_lists(d: pd.DataFrame) -> dict[str, pd.Series]:
    """Reproduce the v2 selection exactly; returns label -> site row."""
    label_map = {}
    for (dtm, kind), pre in PREFIX.items():
        g = d[(d.dtm == dtm) & (d.noise_medlap_m >= NOISE_THR)].dropna(subset=["bump_clay_max_m", "far_rms_clay_m"])
        if kind == "flat":
            g = g[(g.step_m.abs() <= 0.5) & g.bump_clay_max_m.between(1.0, 4.0)]
            g = g.assign(rank=g.bump_clay_max_m / g.far_rms_clay_m)
        else:
            g = g[g.bump_clay_max_m.between(1.0, 8.0)]
            g = g.assign(rank=g.bump_clay_max_m)
        g = g.sort_values("rank", ascending=False)
        keep = []
        for _, r in g.iterrows():
            if all(np.hypot(r.x_m - k.x_m, r.y_m - k.y_m) > 150 for k in keep):
                keep.append(r)
            if len(keep) >= N_SHOW:
                break
        for i, r in enumerate(keep, 1):
            label_map[f"{pre}{i}"] = r
    return label_map


def slope_deg(z: np.ndarray, res: float) -> np.ndarray:
    gy, gx = np.gradient(z.astype(np.float32), res)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def fill_mask(z: np.ndarray, res: float) -> np.ndarray:
    """TIN-fill detector on a crop: windowed mean |laplacian| below threshold."""
    lap = np.abs(ndi.laplace(np.nan_to_num(z, nan=0.0)))
    lap[~np.isfinite(z)] = np.nan
    m = ndi.uniform_filter(np.nan_to_num(lap, nan=0.2), size=int(15 / res))
    return m < 0.02


def main() -> None:
    offsets = json.loads(FRAME.read_text())["hirise"]
    d = pd.read_csv(OUT / "fringe_metrics_v2.csv")
    label_map = build_keep_lists(d)
    missing = [s for s in SELECTED if s not in label_map]
    if missing:
        print("WARNING: could not reproduce panels:", missing)
    sel = [(s, label_map[s]) for s in SELECTED if s in label_map]

    lines_cache, prof_cache = {}, {}
    rows_csv = []
    fig, axes = plt.subplots(len(sel), 4, figsize=(17.5, 2.9 * len(sel)), squeeze=False)
    for (lab, r), axrow in zip(sel, axes):
        dtm = r.dtm
        path = HIRISE[dtm]
        o = offsets[dtm]["measured"]
        with rasterio.open(path) as ds:
            tr1 = Affine.translation(o["dx_m"], o["dy_m"]) * ds.transform
            dcrs = ds.crs
            res1 = abs(tr1.a)
            inv = ~tr1
            c0, r0 = inv * (r.x_m - BOX_M, r.y_m + BOX_M); c1, r1 = inv * (r.x_m + BOX_M, r.y_m - BOX_M)
            win = rasterio.windows.Window(int(c0), int(r0), int(c1 - c0), int(r1 - r0))
            zz = ds.read(1, window=win, boundless=True, fill_value=np.nan).astype(np.float32)
            zz[(zz < -1e30) | (zz == ds.nodata)] = np.nan
        if dtm not in lines_cache:
            lines_cache[dtm] = clay_boundaries(dcrs)
            prof_cache[dtm] = np.load(OUT / f"{dtm}_profiles.npz")
        lines = lines_cache[dtm]
        box = shapely.box(r.x_m - BOX_M, r.y_m - BOX_M, r.x_m + BOX_M, r.y_m + BOX_M)
        ext = (r.x_m - BOX_M, r.x_m + BOX_M, r.y_m - BOX_M, r.y_m + BOX_M)
        axc, axm, axs, axp = axrow
        # CTX
        n3 = int(2 * BOX_M / 3.0)
        tr_box = Affine(3.0, 0, r.x_m - BOX_M, 0, -3.0, r.y_m + BOX_M)
        ctx, _ = ctx_on_grid(dcrs, tr_box, (n3, n3))
        axc.imshow(ctx, cmap="gray", extent=ext, vmin=np.nanpercentile(ctx, 2), vmax=np.nanpercentile(ctx, 98))
        draw_lines(axc, lines, box)
        axc.plot(r.x_m, r.y_m, "o", mfc="none", mec="#0072B2", ms=9, mew=1.5)
        axc.set_ylabel(lab, fontsize=13, rotation=0, labelpad=20, va="center", fontweight="bold")
        # hillshade
        axm.imshow(hillshade(zz, res1), cmap="gray", extent=ext)
        draw_lines(axm, lines, box)
        axm.plot(r.x_m, r.y_m, "o", mfc="none", mec="#0072B2", ms=9, mew=1.5)
        axm.set_xlabel(f"bump {r.bump_clay_max_m:.1f} m at x={r.bump_clay_x_m:.0f} m; step {r.step_m:+.1f} m", fontsize=7)
        # slope (fill masked)
        sl = slope_deg(zz, res1)
        sl[fill_mask(zz, res1)] = np.nan
        im = axs.imshow(sl, cmap="magma", extent=ext, vmin=0, vmax=20)
        draw_lines(axs, lines, box, color="#00c800")
        axs.plot(r.x_m, r.y_m, "o", mfc="none", mec="w", ms=9, mew=1.5)
        # profile
        prof = prof_cache[dtm]["profs"][int(r.idx)]
        rel = prof - np.nanmedian(prof[np.abs(X) <= 2])
        axp.plot(X, rel, color="#0072B2", lw=1.5)
        axp.axvline(0, color="0.4", lw=0.8); axp.axvline(r.bump_clay_x_m, color="#D55E00", lw=0.8, ls="--")
        axp.set_xlabel("m from contact (clay < 0)", fontsize=8); axp.grid(True, color="0.93")
        axp.set_ylim(-max(5, 1.3 * abs(r.bump_clay_max_m)), max(5, 1.3 * abs(r.bump_clay_max_m)))
        axp.tick_params(labelsize=7)
        for ax in (axc, axm, axs):
            ax.set_xticks([]); ax.set_yticks([])
        lon, lat = rtransform(dcrs, LONLAT, [r.x_m], [r.y_m])
        rows_csv.append({"panel": lab, "dtm": dtm, "x_dtm_frame_m": r.x_m, "y_dtm_frame_m": r.y_m,
                         "lon_e": lon[0], "lat_n": lat[0], "bump_m": r.bump_clay_max_m, "bump_x_m": r.bump_clay_x_m,
                         "step_m": r.step_m, "noise_medlap_m": r.noise_medlap_m, "strike_deg": r.strike})
    axes[0, 0].set_title("CTX 6 m + mapped contact", fontsize=9)
    axes[0, 1].set_title("HiRISE 1 m hillshade + contact", fontsize=9)
    axes[0, 2].set_title("slope 0-20 deg (fill masked) + contact", fontsize=9)
    axes[0, 3].set_title("swath-median profile", fontsize=9)
    fig.suptitle("Selected contact-fringe candidates (AV, 2026-08-31); orange/green = hand-drawn clay contact", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_gallery_v3_selected.png", dpi=150)
    fig.savefig(OUT / "fig_gallery_v3_selected.pdf")
    plt.close(fig)
    pd.DataFrame(rows_csv).to_csv(OUT / "selected_sites.csv", index=False)
    print("panels:", [s for s, _ in sel]); print("wrote", OUT / "fig_gallery_v3_selected.png")


if __name__ == "__main__":
    main()
