"""Stacked HiRISE-DTM profiles across the mapped clay/olivine contact:
is there a raised "fringe" (pushed-rug fold) along the contact?

For each of the two Horseshoe HiRISE DTMs (1 m, measured Stage 0 shifts):
- rasterise the geologic-map units at 3 m on the DTM grid;
- take contact cells (olivine side, 4-adjacent to clay), subsampled to
  ~10 m spacing; local strike from PCA of nearby contact cells; profiles
  signed so +x = olivine side, -x = clay side;
- sample the 1 m DTM along the contact normal, x = -100..+100 m, median
  over a +-6 m along-strike swath;
- per profile and per side, detrend with a line fitted to the far field
  (40-100 m) and measure the maximum residual ("bump") within 2-30 m of
  the contact;
- stack: median profile (+IQR) per DTM and per spatial cluster (KMeans
  k=5, seed 42); report bump-amplitude statistics; save a gallery of the
  strongest fringe examples (hillshade + profile).

Outputs: outputs/revision/contact_3d/stage7_fringe/

Usage: python code/contact_contact_fringe.py
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
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.warp import transform as rtransform
from scipy.ndimage import binary_dilation, map_coordinates
from sklearn.cluster import KMeans

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from config import GEOMAP_SOURCE as GDB, HIRISE_PDS_DIR  # noqa: E402
FRAME = ROOT / "outputs" / "revision" / "contact_3d" / "frame" / "coreg_offsets.json"
HIRISE = {"DTEEC_006712_2020_005855_2020_A01": HIRISE_PDS_DIR / "DTEEC_006712_2020_005855_2020_A01.IMG",
          "DTEEC_002888_2025_002176_2025_A01": HIRISE_PDS_DIR / "DTEEC_002888_2025_002176_2025_A01.IMG"}
OUT = ROOT / "outputs" / "revision" / "contact_3d" / "stage7_fringe"
OUT.mkdir(parents=True, exist_ok=True)
X = np.arange(-100.0, 100.0 + 1.0, 1.0)           # m, negative = clay side
SWATH = np.array([-6.0, -3.0, 0.0, 3.0, 6.0])     # along-strike offsets
FARBAND = (40.0, 100.0)                            # detrend band per side
BUMPBAND = (2.0, 30.0)                             # fringe search band per side
SPACING_CELLS = 3                                  # ~10 m along-contact subsampling at 3 m cells
SEED = 42


def load_labels_on(z_shape, tr3: Affine, dcrs: CRS) -> np.ndarray:
    def load(layer):
        lcrs = CRS.from_wkt(pyogrio.read_info(GDB, layer=layer)["crs"])
        _, _, g, _ = pyogrio.raw.read(GDB, layer=layer)
        gs = [shapely.force_2d(x) for x in shapely.from_wkb(g)]
        return [shapely.make_valid(shapely.transform(x, lambda xy: np.column_stack(rtransform(lcrs, dcrs, xy[:, 0], xy[:, 1])))) for x in gs]
    units = {"clay": load("Clay_unit"), "cap": load("Mafic_cap"), "dune": load("olivine_dunes"),
             "graben": load("graben_materials"), "outline": load("Olivine_unit")}
    codes = {"outline": 1, "graben": 2, "clay": 3, "cap": 4, "dune": 5}
    shapes = [(g, codes[k]) for k in ("outline", "graben", "clay", "cap", "dune") for g in units[k]]
    return rasterize(shapes, out_shape=z_shape, transform=tr3, fill=0, dtype="uint8", all_touched=False)


def hillshade(z: np.ndarray, res: float) -> np.ndarray:
    ls = LightSource(azdeg=300, altdeg=30)
    zz = np.where(np.isfinite(z), z, np.nanmedian(z))
    return ls.hillshade(zz, vert_exag=2.0, dx=res, dy=res)


def main() -> None:
    t0 = datetime.now(timezone.utc)
    offsets = json.loads(FRAME.read_text())["hirise"]
    all_rows, stacks = [], {}
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
        f = max(1, int(round(3.0 / res1)))
        z3_shape = (z1.shape[0] // f, z1.shape[1] // f)
        tr3 = tr1 * Affine.scale(f)
        lab = load_labels_on(z3_shape, tr3, dcrs)
        st = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
        contact = (lab == 1) & binary_dilation(lab == 3, st)
        cr, cc = np.nonzero(contact)
        print(dtm, "contact cells:", cr.size)
        # world coords of contact cells
        cx, cy = tr3 * (cc + 0.5, cr + 0.5)
        pts = np.column_stack([cx, cy])
        # subsample
        order = np.lexsort((cc, cr))
        keep = order[::SPACING_CELLS]
        inv1 = ~tr1
        inv3 = ~tr3
        profs, meta = [], []
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        for i in keep:
            p = pts[i]
            nb = tree.query_ball_point(p, 15.0)
            if len(nb) < 3:
                continue
            local = pts[nb] - pts[nb].mean(axis=0)
            w, v = np.linalg.eigh(np.cov(local.T))
            strike = v[:, -1]                       # along-contact direction
            normal = np.array([-strike[1], strike[0]])
            # orient normal toward olivine: check label 12 m either way
            for sgn in (1.0, -1.0):
                q = p + sgn * 12.0 * normal
                c3, r3 = inv3 * (q[0], q[1])
                r3i, c3i = int(r3), int(c3)
                if 0 <= r3i < z3_shape[0] and 0 <= c3i < z3_shape[1] and lab[r3i, c3i] == 1:
                    normal_o = sgn * normal
                    break
            else:
                continue
            # opposite side must be clay
            q = p - 12.0 * normal_o
            c3, r3 = inv3 * (q[0], q[1])
            r3i, c3i = int(r3), int(c3)
            if not (0 <= r3i < z3_shape[0] and 0 <= c3i < z3_shape[1] and lab[r3i, c3i] == 3):
                continue
            zz = np.full((SWATH.size, X.size), np.nan, np.float32)
            for k, a in enumerate(SWATH):
                xs = p[0] + a * strike[0] + X * normal_o[0]
                ys = p[1] + a * strike[1] + X * normal_o[1]
                colf, rowf = inv1 * (xs, ys)
                zz[k] = map_coordinates(z1, [rowf, colf], order=1, mode="constant", cval=np.nan)
            prof = np.nanmedian(zz, axis=0)
            if np.isfinite(prof).mean() < 0.9:
                continue
            profs.append(prof)
            meta.append({"x_m": p[0], "y_m": p[1], "strike": float(np.degrees(np.arctan2(strike[1], strike[0])) % 180)})
        profs = np.array(profs)
        print(dtm, "profiles:", profs.shape)
        if profs.size == 0:
            continue
        # per-profile detrend per side; bump metrics
        rows = []
        resid_all = np.full_like(profs, np.nan)
        for j, prof in enumerate(profs):
            rec = dict(meta[j], dtm=dtm, idx=j)
            okp = np.isfinite(prof)
            for side, sname in ((-1, "clay"), (1, "oliv")):
                sel_far = okp & (np.sign(X) == side) & (np.abs(X) >= FARBAND[0]) & (np.abs(X) <= FARBAND[1])
                sel_all = okp & (np.sign(X) == side)
                if sel_far.sum() < 30:
                    continue
                co = np.polyfit(X[sel_far], prof[sel_far], 1)
                resid = prof - np.polyval(co, X)
                resid_all[j, sel_all] = resid[sel_all]
                band = sel_all & (np.abs(X) >= BUMPBAND[0]) & (np.abs(X) <= BUMPBAND[1])
                if band.sum() > 10:
                    rec[f"bump_{sname}_max_m"] = float(np.nanmax(resid[band]))
                    rec[f"bump_{sname}_x_m"] = float(X[band][np.nanargmax(resid[band])])
                    rec[f"far_rms_{sname}_m"] = float(np.nanstd(resid[sel_far]))
            rec["step_m"] = float(np.nanmedian(prof[(X >= 5) & (X <= 25)]) - np.nanmedian(prof[(X <= -5) & (X >= -25)]))
            rows.append(rec)
        df = pd.DataFrame(rows)
        all_rows.append(df)
        # spatial clusters for per-area stacks
        km = KMeans(n_clusters=5, random_state=SEED, n_init=10).fit(df[["x_m", "y_m"]])
        df["cluster"] = km.labels_
        stacks[dtm] = {"X": X, "profs": profs, "resid": resid_all, "df": df}
        np.savez_compressed(OUT / f"{dtm}_profiles.npz", X=X, profs=profs, resid=resid_all,
                            x_m=df.x_m.to_numpy(), y_m=df.y_m.to_numpy(), cluster=df.cluster.to_numpy())

    dfa = pd.concat(all_rows, ignore_index=True)
    dfa.to_csv(OUT / "fringe_metrics.csv", index=False)
    summary = {}
    for dtm, s in stacks.items():
        d = s["df"]
        summary[dtm] = {
            "n_profiles": len(d),
            "step_med_m": float(d.step_m.median()),
            "bump_clay_med_m": float(d.bump_clay_max_m.median()),
            "bump_clay_frac_ge_0.5m": float((d.bump_clay_max_m >= 0.5).mean()),
            "bump_clay_frac_ge_1m": float((d.bump_clay_max_m >= 1.0).mean()),
            "bump_clay_frac_ge_2x_rms": float((d.bump_clay_max_m >= 2 * d.far_rms_clay_m).mean()),
            "far_rms_clay_med_m": float(d.far_rms_clay_m.median()),
            "bump_oliv_med_m": float(d.bump_oliv_max_m.median()),
            "bump_clay_x_med_m": float(d.bump_clay_x_m.median()),
        }
        print(dtm, summary[dtm])
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # stacked figures
    fig, axes = plt.subplots(1, len(stacks), figsize=(7 * len(stacks), 4.6), squeeze=False)
    for ax, (dtm, s) in zip(axes[0], stacks.items()):
        profs = s["profs"] - np.nanmedian(s["profs"][:, (X >= -2) & (X <= 2)], axis=1, keepdims=True)
        med = np.nanmedian(profs, axis=0); q1 = np.nanpercentile(profs, 25, axis=0); q3 = np.nanpercentile(profs, 75, axis=0)
        ax.fill_between(X, q1, q3, color="#0072B2", alpha=0.2, lw=0)
        ax.plot(X, med, color="#0072B2", lw=2, label="median (z=0 at contact)")
        rmed = np.nanmedian(s["resid"], axis=0)
        ax.plot(X, rmed, color="#D55E00", lw=1.5, label="median far-field-detrended residual")
        ax.axvline(0, color="0.4", lw=0.8); ax.axhline(0, color="0.85", lw=0.8)
        ax.set_xlabel("distance from contact (m); clay side < 0 < olivine side")
        ax.set_ylabel("relative elevation (m)")
        ax.set_title(f"{dtm[:20]}  (n={len(s['df'])})", fontsize=10)
        ax.legend(fontsize=8, frameon=False); ax.grid(True, color="0.93"); ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(-100, 100)
    fig.suptitle("Stacked HiRISE 1 m profiles across the mapped clay/olivine contact", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig_stacked_profiles.png", dpi=200); fig.savefig(OUT / "fig_stacked_profiles.pdf"); plt.close(fig)

    # per-cluster stacks
    for dtm, s in stacks.items():
        d = s["df"]
        fig, axes = plt.subplots(1, 5, figsize=(20, 3.8), sharey=True)
        for k, ax in enumerate(axes):
            sel = (d.cluster == k).to_numpy()
            pr = s["resid"][d.idx.to_numpy()[sel]]
            if sel.sum() < 10:
                ax.set_visible(False); continue
            ax.fill_between(X, np.nanpercentile(pr, 25, axis=0), np.nanpercentile(pr, 75, axis=0), color="#0072B2", alpha=0.2, lw=0)
            ax.plot(X, np.nanmedian(pr, axis=0), color="#0072B2", lw=1.8)
            ax.axvline(0, color="0.4", lw=0.8); ax.axhline(0, color="0.85", lw=0.8)
            ax.set_title(f"area {k} (n={sel.sum()})", fontsize=9); ax.grid(True, color="0.93")
            ax.set_xlabel("m from contact"); ax.set_ylim(-4, 6)
        axes[0].set_ylabel("detrended residual (m)")
        fig.suptitle(f"{dtm}: detrended contact-normal residual by contact area (KMeans k=5)", fontsize=10)
        fig.tight_layout(); fig.savefig(OUT / f"fig_cluster_stacks_{dtm}.png", dpi=170); plt.close(fig)

    # gallery of strongest clay-side fringes
    for dtm, s in stacks.items():
        d = s["df"].dropna(subset=["bump_clay_max_m"])
        d = d[d.bump_clay_max_m >= 2 * d.far_rms_clay_m].sort_values("bump_clay_max_m", ascending=False).head(12)
        if d.empty:
            continue
        with rasterio.open(HIRISE[dtm]) as ds:
            tr1 = Affine.translation(offsets[dtm]["measured"]["dx_m"], offsets[dtm]["measured"]["dy_m"]) * ds.transform
            fig, axes = plt.subplots(len(d), 2, figsize=(11, 2.9 * len(d)), squeeze=False)
            for (ri, (_, r)), (axm, axp) in zip(enumerate(d.iterrows()), axes):
                inv = ~tr1
                cx, cy = r.x_m, r.y_m
                c0, r0 = inv * (cx - 120, cy + 120); c1, r1 = inv * (cx + 120, cy - 120)
                win = rasterio.windows.Window(int(c0), int(r0), int(c1 - c0), int(r1 - r0))
                zz = ds.read(1, window=win, boundless=True, fill_value=np.nan).astype(np.float32)
                zz[(zz < -1e30) | (zz == ds.nodata)] = np.nan
                axm.imshow(hillshade(zz, abs(tr1.a)), cmap="gray")
                axm.plot(zz.shape[1] / 2, zz.shape[0] / 2, "o", color="#D55E00", ms=5)
                axm.set_xticks([]); axm.set_yticks([])
                axm.set_title(f"({r.x_m:.0f}, {r.y_m:.0f})  bump {r.bump_clay_max_m:.1f} m at x={r.bump_clay_x_m:.0f} m", fontsize=8)
                prof = s["profs"][int(r.idx)]
                axp.plot(X, prof - np.nanmedian(prof[(X >= -2) & (X <= 2)]), color="#0072B2", lw=1.5)
                axp.axvline(0, color="0.4", lw=0.8); axp.axvline(r.bump_clay_x_m, color="#D55E00", lw=0.8, ls="--")
                axp.set_xlabel("m from contact (clay < 0)"); axp.grid(True, color="0.93")
            fig.suptitle(f"{dtm}: strongest clay-side fringe candidates (hillshade 300/30, 2x)", fontsize=10)
            fig.tight_layout(); fig.savefig(OUT / f"fig_fringe_gallery_{dtm}.png", dpi=140); plt.close(fig)

    (OUT / "run_log.json").write_text(json.dumps({"x_range_m": [float(X[0]), float(X[-1])], "swath_m": SWATH.tolist(),
        "far_band_m": FARBAND, "bump_band_m": BUMPBAND, "spacing_cells": SPACING_CELLS, "seed": SEED,
        "shifts": {k: offsets[k]["measured"] for k in HIRISE},
        "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
