"""Do the carbonate patches coincide with the contact fringes?

1. Projects every Plebani carbonate patch centroid (stage 6, shared MTR3
   CRS) into the two HiRISE DTM frames; keeps patches inside DTM coverage.
2. For each such patch: nearest Stage 7 fringe-profile site and its bump.
3. Overlap statistic (002888, clean sites): clay-side bump at sites within
   150 m of a carbonate patch vs. all other sites.
4. v3-style gallery centred on the carbonate patches inside DTM coverage
   (labels C1.. for 002888, D1.. for 006712-with-caution): CTX + contact +
   patch marker | hillshade | slope (fill masked) | nearest-site profile.

Outputs: outputs/revision/contact_3d/stage6b_carb_fringe_overlap/

Usage: python code/contact_carbonate_fringe_overlap.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.transform import Affine
from rasterio.warp import transform as rtransform
from scipy import stats
from scipy.spatial import cKDTree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from contact_contact_fringe import HIRISE, X  # noqa: E402
from contact_coreg_frame import ctx_on_grid  # noqa: E402
from contact_fringe_gallery_v2 import NOISE_THR, clay_boundaries, draw_lines, hillshade  # noqa: E402
from contact_fringe_gallery_v3 import fill_mask, slope_deg  # noqa: E402

OUT = ROOT / "outputs" / "revision" / "contact_3d" / "stage6b_carb_fringe_overlap"
OUT.mkdir(parents=True, exist_ok=True)
FRAME = ROOT / "outputs" / "revision" / "contact_3d" / "frame" / "coreg_offsets.json"
PATCHES = ROOT / "outputs" / "revision" / "contact_3d" / "stage6_carbonate" / "carbonate_patches.csv"
METRICS = ROOT / "outputs" / "revision" / "contact_3d" / "stage7_fringe" / "fringe_metrics_v2.csv"
NEAR_M = 150.0
BOX_M = 120.0
N_SHOW = 12
LABEL = {"DTEEC_002888_2025_002176_2025_A01": "C", "DTEEC_006712_2020_005855_2020_A01": "D"}


def main() -> None:
    t0 = datetime.now(timezone.utc)
    offsets = json.loads(FRAME.read_text())["hirise"]
    patches = pd.read_csv(PATCHES)
    metrics = pd.read_csv(METRICS)
    with rasterio.open(sorted((ROOT / "outputs" / "_v8_mtr3_with_brown" / "FRT00003E12" / "geotiffs").glob("*.tif"))[0]) as ds:
        mtr3_crs = ds.crs
    rows, stats_out = [], {}
    for dtm, path in HIRISE.items():
        o = offsets[dtm]["measured"]
        with rasterio.open(path) as ds:
            tr1 = Affine.translation(o["dx_m"], o["dy_m"]) * ds.transform
            dcrs = ds.crs
            zfull = None  # lazy
            xs, ys = rtransform(mtr3_crs, dcrs, patches.x_m.to_numpy(), patches.y_m.to_numpy())
            inv = ~tr1
            cols, rws = inv * (np.asarray(xs), np.asarray(ys))
            inside = (rws >= 0) & (rws < ds.height) & (cols >= 0) & (cols < ds.width)
            # check finite z at the point (windowed reads for the subset)
            fin = np.zeros(len(patches), bool)
            for i in np.nonzero(inside)[0]:
                v = ds.read(1, window=rasterio.windows.Window(int(cols[i]), int(rws[i]), 1, 1), boundless=True, fill_value=np.nan)
                fin[i] = np.isfinite(v[0, 0]) and v[0, 0] > -1e30 and (ds.nodata is None or v[0, 0] != ds.nodata)
        sub = patches[fin].copy()
        sub["x_dtm"] = np.asarray(xs)[fin]
        sub["y_dtm"] = np.asarray(ys)[fin]
        sub["dtm"] = dtm
        # nearest fringe site
        g = metrics[(metrics.dtm == dtm) & (metrics.noise_medlap_m >= NOISE_THR)].dropna(subset=["bump_clay_max_m"]).reset_index()
        tree = cKDTree(np.column_stack([g.x_m, g.y_m]))
        dnn, inn = tree.query(np.column_stack([sub.x_dtm, sub.y_dtm]))
        sub["site_dist_m"] = dnn
        sub["site_bump_m"] = g.bump_clay_max_m.to_numpy()[inn]
        sub["site_idx"] = g["idx"].to_numpy()[inn]
        sub["site_x"] = g.x_m.to_numpy()[inn]
        sub["site_y"] = g.y_m.to_numpy()[inn]
        rows.append(sub)
        # overlap statistic: bump at sites near carbonate vs the rest
        if len(sub):
            ptree = cKDTree(np.column_stack([sub.x_dtm, sub.y_dtm]))
            dsite, _ = ptree.query(np.column_stack([g.x_m, g.y_m]))
            near = dsite <= NEAR_M
            a, b = g.bump_clay_max_m[near], g.bump_clay_max_m[~near]
            if near.sum() >= 30:
                u = stats.mannwhitneyu(a, b)
                stats_out[dtm] = {"n_patches_in_dtm": int(len(sub)), "n_sites_near_carb": int(near.sum()),
                                  "bump_near_carb_med": float(a.median()), "bump_far_med": float(b.median()),
                                  "cliffs_delta": float(2 * u.statistic / (len(a) * len(b)) - 1),
                                  "mwu_p_descriptive": float(u.pvalue),
                                  "note": "006712 caution: pervasive TIN fill" if "006712" in dtm else "clean DTM"}
                print(dtm[:22], stats_out[dtm])
            else:
                stats_out[dtm] = {"n_patches_in_dtm": int(len(sub)), "n_sites_near_carb": int(near.sum()), "note": "too few near sites"}
                print(dtm[:22], stats_out[dtm])

    allp = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    allp.to_csv(OUT / "carbonate_patches_in_dtms.csv", index=False)
    (OUT / "overlap_stats.json").write_text(json.dumps(stats_out, indent=2))

    # gallery per DTM
    profs_cache = {}
    for dtm, path in HIRISE.items():
        sub = allp[allp.dtm == dtm].sort_values("n_px", ascending=False)
        keep = []
        for _, r in sub.iterrows():
            if all(np.hypot(r.x_dtm - k.x_dtm, r.y_dtm - k.y_dtm) > 150 for k in keep):
                keep.append(r)
            if len(keep) >= N_SHOW:
                break
        if not keep:
            continue
        o = offsets[dtm]["measured"]
        pre = LABEL[dtm]
        with rasterio.open(path) as ds:
            tr1 = Affine.translation(o["dx_m"], o["dy_m"]) * ds.transform
            dcrs = ds.crs
            res1 = abs(tr1.a)
            lines = clay_boundaries(dcrs)
            s7 = profs_cache.setdefault(dtm, np.load(ROOT / "outputs" / "revision" / "contact_3d" / "stage7_fringe" / f"{dtm}_profiles.npz"))
            fig, axes = plt.subplots(len(keep), 4, figsize=(17.5, 2.9 * len(keep)), squeeze=False)
            for ri, (r, axrow) in enumerate(zip(keep, axes), 1):
                lab = f"{pre}{ri}"
                axc, axm, axs_, axp = axrow
                box = shapely.box(r.x_dtm - BOX_M, r.y_dtm - BOX_M, r.x_dtm + BOX_M, r.y_dtm + BOX_M)
                ext = (r.x_dtm - BOX_M, r.x_dtm + BOX_M, r.y_dtm - BOX_M, r.y_dtm + BOX_M)
                n3 = int(2 * BOX_M / 3.0)
                tr_box = Affine(3.0, 0, r.x_dtm - BOX_M, 0, -3.0, r.y_dtm + BOX_M)
                ctx, _ = ctx_on_grid(dcrs, tr_box, (n3, n3))
                axc.imshow(ctx, cmap="gray", extent=ext, vmin=np.nanpercentile(ctx, 2), vmax=np.nanpercentile(ctx, 98))
                draw_lines(axc, lines, box)
                ms = max(6, min(20, 3 * np.sqrt(r.n_px)))
                axc.plot(r.x_dtm, r.y_dtm, "o", mfc="none", mec="#1b9e77", ms=ms, mew=2)
                axc.set_ylabel(lab, fontsize=13, rotation=0, labelpad=20, va="center", fontweight="bold")
                inv = ~tr1
                c0, r0 = inv * (r.x_dtm - BOX_M, r.y_dtm + BOX_M); c1, r1 = inv * (r.x_dtm + BOX_M, r.y_dtm - BOX_M)
                win = rasterio.windows.Window(int(c0), int(r0), int(c1 - c0), int(r1 - r0))
                zz = ds.read(1, window=win, boundless=True, fill_value=np.nan).astype(np.float32)
                zz[(zz < -1e30) | (zz == ds.nodata)] = np.nan
                axm.imshow(hillshade(zz, res1), cmap="gray", extent=ext)
                draw_lines(axm, lines, box)
                axm.plot(r.x_dtm, r.y_dtm, "o", mfc="none", mec="#1b9e77", ms=ms, mew=2)
                axm.set_xlabel(f"{r.cube} patch {r.patch}: {r.n_px}px, MgCO3max {r.mgco3_max:.2f}; site {r.site_dist_m:.0f} m away, bump {r.site_bump_m:.1f} m", fontsize=7)
                sl = slope_deg(zz, res1); sl[fill_mask(zz, res1)] = np.nan
                axs_.imshow(sl, cmap="magma", extent=ext, vmin=0, vmax=20)
                draw_lines(axs_, lines, box, color="#00c800")
                axs_.plot(r.x_dtm, r.y_dtm, "o", mfc="none", mec="w", ms=ms, mew=2)
                if r.site_dist_m <= 60:
                    prof = s7["profs"][int(r.site_idx)]
                    rel = prof - np.nanmedian(prof[np.abs(X) <= 2])
                    axp.plot(X, rel, color="#0072B2", lw=1.5)
                    axp.axvline(0, color="0.4", lw=0.8)
                    axp.set_xlabel("m from contact (clay < 0)", fontsize=8)
                else:
                    axp.text(0.5, 0.5, f"nearest fringe site {r.site_dist_m:.0f} m away", ha="center", va="center", fontsize=9, transform=axp.transAxes)
                axp.grid(True, color="0.93"); axp.tick_params(labelsize=7)
                for ax in (axc, axm, axs_):
                    ax.set_xticks([]); ax.set_yticks([])
            axes[0, 0].set_title("CTX 6 m + contact + patch", fontsize=9)
            axes[0, 1].set_title("HiRISE hillshade + contact", fontsize=9)
            axes[0, 2].set_title("slope 0-20 deg (fill masked)", fontsize=9)
            axes[0, 3].set_title("nearest fringe-site profile", fontsize=9)
            note = " (CAUTION: pervasive TIN fill in this DTM)" if "006712" in dtm else ""
            fig.suptitle(f"Carbonate patches inside {dtm}{note}; green circle = patch centroid (size ~ patch px)", fontsize=10)
            fig.tight_layout(); fig.savefig(OUT / f"fig_gallery_carb_{dtm}.png", dpi=150); plt.close(fig)
            print(dtm, "gallery:", len(keep), "panels", pre + "1..")
    (OUT / "run_log.json").write_text(json.dumps({"near_m": NEAR_M, "noise_thr": NOISE_THR,
        "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
