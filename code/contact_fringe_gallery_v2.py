"""Fringe galleries v2, after AV's review of v1.

Changes vs v1:
- Facet/TIN-fill detector: median |Laplacian| of the 1 m DTM in a 60 m box
  around each site (stereo-correlation fill is piecewise planar -> ~0;
  real 1 m terrain noise -> >~0.05 m). Stored per site; artefact sites are
  excluded from galleries and the population statistics are recomputed
  with and without them.
- Panels have three columns: CTX 6 m crop (streamed Robbins mosaic, the
  frame the geologic map was drawn in), HiRISE hillshade, profile; the
  mapped clay contact is overlain on both thumbnails (hand-drawn, so it
  may sit off the texture - that is visible now).
- Panels labelled P1.. (002888 flat-edge), Q1.. (006712 flat-edge),
  R1.. (002888 strongest, noise-filtered), S1.. (006712 strongest).

Usage: python code/contact_fringe_gallery_v2.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
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
from contact_contact_fringe import GDB, HIRISE, X  # noqa: E402
from contact_coreg_frame import ctx_on_grid  # noqa: E402

OUT = ROOT / "outputs" / "revision" / "contact_3d" / "stage7_fringe"
FRAME = ROOT / "outputs" / "revision" / "contact_3d" / "frame" / "coreg_offsets.json"
BOX_M = 120.0                  # half-size of thumbnails
NOISE_BOX_M = 30.0             # half-size of the facet-metric box
NOISE_THR = 0.05               # median |laplacian| below this = TIN fill / artefact
N_SHOW = 12
PREFIX = {("DTEEC_002888_2025_002176_2025_A01", "flat"): "P", ("DTEEC_006712_2020_005855_2020_A01", "flat"): "Q",
          ("DTEEC_002888_2025_002176_2025_A01", "strong"): "R", ("DTEEC_006712_2020_005855_2020_A01", "strong"): "S"}


def hillshade(z: np.ndarray, res: float) -> np.ndarray:
    ls = LightSource(azdeg=300, altdeg=30)
    zz = np.where(np.isfinite(z), z, np.nanmedian(z))
    return ls.hillshade(zz, vert_exag=2.0, dx=res, dy=res)


def clay_boundaries(dcrs: CRS):
    lcrs = CRS.from_wkt(pyogrio.read_info(GDB, layer="Clay_unit")["crs"])
    _, _, g, _ = pyogrio.raw.read(GDB, layer="Clay_unit")
    gs = [shapely.force_2d(x) for x in shapely.from_wkb(g)]
    gs = [shapely.make_valid(shapely.transform(x, lambda xy: np.column_stack(rtransform(lcrs, dcrs, xy[:, 0], xy[:, 1])))) for x in gs]
    return shapely.union_all([shapely.boundary(x) for x in gs])


def draw_lines(ax, lines, box, color="#D55E00"):
    clip = shapely.intersection(lines, box)
    if clip.is_empty:
        return
    parts = clip.geoms if hasattr(clip, "geoms") else [clip]
    for p in parts:
        if p.is_empty or not hasattr(p, "coords"):
            continue
        xy = np.asarray(p.coords)
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.2)


def main() -> None:
    t0 = datetime.now(timezone.utc)
    offsets = json.loads(FRAME.read_text())["hirise"]
    d = pd.read_csv(OUT / "fringe_metrics.csv")
    d["noise_medlap_m"] = np.nan
    summaries = {}
    for dtm, path in HIRISE.items():
        o = offsets[dtm]["measured"]
        with rasterio.open(path) as ds:
            z1 = ds.read(1).astype(np.float32)
            nd = ds.nodata
            if nd is not None:
                z1[z1 == nd] = np.nan
            z1[(z1 < -1e30) | (z1 > 1e30)] = np.nan
            tr1 = Affine.translation(o["dx_m"], o["dy_m"]) * ds.transform
            dcrs = ds.crs
        res1 = abs(tr1.a)
        lap = np.abs(ndi.laplace(np.nan_to_num(z1, nan=0.0)))
        lap[~np.isfinite(z1)] = np.nan
        inv = ~tr1
        sel = d.dtm == dtm
        idx = d.index[sel]
        half = int(NOISE_BOX_M / res1)
        for i in idx:
            c, r = inv * (d.at[i, "x_m"], d.at[i, "y_m"])
            r, c = int(r), int(c)
            w = lap[max(0, r - half):r + half, max(0, c - half):c + half]
            if np.isfinite(w).sum() > 100:
                d.at[i, "noise_medlap_m"] = float(np.nanmedian(w))
        g = d[sel]
        clean = g[g.noise_medlap_m >= NOISE_THR]
        flat_all = g[(g.step_m.abs() <= 0.5)].dropna(subset=["bump_clay_max_m"])
        flat_clean = clean[(clean.step_m.abs() <= 0.5)].dropna(subset=["bump_clay_max_m"])
        summaries[dtm] = {
            "n": len(g), "n_clean": len(clean), "frac_noise": float((g.noise_medlap_m < NOISE_THR).mean()),
            "bump_clay_med_all": float(g.bump_clay_max_m.median()), "bump_clay_med_clean": float(clean.bump_clay_max_m.median()),
            "bump_clay_frac_ge1_clean": float((clean.bump_clay_max_m >= 1).mean()),
            "flatedge_n_all": len(flat_all), "flatedge_n_clean": len(flat_clean),
            "flatedge_bump_med_all": float(flat_all.bump_clay_max_m.median()),
            "flatedge_bump_med_clean": float(flat_clean.bump_clay_max_m.median()),
            "flatedge_frac_ge1_clean": float((flat_clean.bump_clay_max_m >= 1).mean()),
        }
        print(dtm, summaries[dtm])

        lines = clay_boundaries(dcrs)
        s = np.load(OUT / f"{dtm}_profiles.npz")
        for kind in ("flat", "strong"):
            gg = clean.dropna(subset=["bump_clay_max_m", "far_rms_clay_m"])
            if kind == "flat":
                gg = gg[(gg.step_m.abs() <= 0.5) & gg.bump_clay_max_m.between(1.0, 4.0)]
                gg = gg.assign(rank=gg.bump_clay_max_m / gg.far_rms_clay_m)
            else:
                gg = gg[gg.bump_clay_max_m.between(1.0, 8.0)]
                gg = gg.assign(rank=gg.bump_clay_max_m)
            gg = gg.sort_values("rank", ascending=False)
            keep = []
            for _, r in gg.iterrows():
                if all(np.hypot(r.x_m - k.x_m, r.y_m - k.y_m) > 150 for k in keep):
                    keep.append(r)
                if len(keep) >= N_SHOW:
                    break
            if not keep:
                continue
            pre = PREFIX[(dtm, kind)]
            fig, axes = plt.subplots(len(keep), 3, figsize=(13.5, 2.9 * len(keep)), squeeze=False)
            with rasterio.open(path) as ds:
                for ri, (r, (axc, axm, axp)) in enumerate(zip(keep, axes), 1):
                    lab = f"{pre}{ri}"
                    box = shapely.box(r.x_m - BOX_M, r.y_m - BOX_M, r.x_m + BOX_M, r.y_m + BOX_M)
                    ext = (r.x_m - BOX_M, r.x_m + BOX_M, r.y_m - BOX_M, r.y_m + BOX_M)
                    # CTX crop on a 3 m grid in the DTM CRS
                    n3 = int(2 * BOX_M / 3.0)
                    tr_box = Affine(3.0, 0, r.x_m - BOX_M, 0, -3.0, r.y_m + BOX_M)
                    ctx, _ = ctx_on_grid(dcrs, tr_box, (n3, n3))
                    axc.imshow(ctx, cmap="gray", extent=ext, vmin=np.nanpercentile(ctx, 2), vmax=np.nanpercentile(ctx, 98))
                    draw_lines(axc, lines, box)
                    axc.plot(r.x_m, r.y_m, "o", mfc="none", mec="#0072B2", ms=9, mew=1.5)
                    axc.set_ylabel(lab, fontsize=12, rotation=0, labelpad=18, va="center", fontweight="bold")
                    axc.set_xticks([]); axc.set_yticks([])
                    if ri == 1:
                        axc.set_title("CTX 6 m + mapped clay contact", fontsize=8)
                    # hillshade crop
                    invt = ~tr1
                    c0, r0 = invt * (r.x_m - BOX_M, r.y_m + BOX_M); c1, r1 = invt * (r.x_m + BOX_M, r.y_m - BOX_M)
                    win = rasterio.windows.Window(int(c0), int(r0), int(c1 - c0), int(r1 - r0))
                    zz = ds.read(1, window=win, boundless=True, fill_value=np.nan).astype(np.float32)
                    zz[(zz < -1e30) | (zz == ds.nodata)] = np.nan
                    axm.imshow(hillshade(zz, res1), cmap="gray", extent=ext)
                    draw_lines(axm, lines, box)
                    axm.plot(r.x_m, r.y_m, "o", mfc="none", mec="#0072B2", ms=9, mew=1.5)
                    axm.set_xticks([]); axm.set_yticks([])
                    if ri == 1:
                        axm.set_title("HiRISE 1 m hillshade + contact", fontsize=8)
                    axm.set_xlabel(f"bump {r.bump_clay_max_m:.1f} m at x={r.bump_clay_x_m:.0f} m; step {r.step_m:+.1f} m; noise {r.noise_medlap_m:.2f} m", fontsize=7)
                    # profile
                    prof = s["profs"][int(r.idx)]
                    rel = prof - np.nanmedian(prof[np.abs(X) <= 2])
                    axp.plot(X, rel, color="#0072B2", lw=1.5)
                    axp.axvline(0, color="0.4", lw=0.8); axp.axvline(r.bump_clay_x_m, color="#D55E00", lw=0.8, ls="--")
                    axp.set_xlabel("m from contact (clay < 0)", fontsize=8); axp.grid(True, color="0.93")
                    ylim = 5 if kind == "flat" else max(5, 1.5 * r.bump_clay_max_m)
                    axp.set_ylim(-ylim, ylim); axp.tick_params(labelsize=7)
            title_kind = "flat olivine edge (|step| <= 0.5 m), ranked bump/rms" if kind == "flat" else "strongest bumps 1-8 m (noise-filtered)"
            fig.suptitle(f"{dtm} - {title_kind}; orange = mapped clay contact (hand-drawn)", fontsize=10)
            fig.tight_layout()
            fig.savefig(OUT / f"fig_gallery_v2_{kind}_{dtm}.png", dpi=140)
            plt.close(fig)
            print(dtm, kind, "->", len(keep), "panels", pre + "1..")

    d.to_csv(OUT / "fringe_metrics_v2.csv", index=False)
    (OUT / "summary_v2.json").write_text(json.dumps({"noise_thr_medlap_m": NOISE_THR, "per_dtm": summaries,
        "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
