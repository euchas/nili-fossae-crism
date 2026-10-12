"""Block-level distance test for the olivine-unit 2.3 um phase classes.

Classes (Plebani chain, per sensor pixel): A carbonate-mode ('2.3 only',
centre < 2.305 um), B serpentine/talc-mode ('2.3 only', >= 2.305 um),
C both bands (carbonate cores at contacts). Universe: map-pure olivine
pixels > 90 m from the cap, with usable 1.4 and 2.3 fits. For each class:
fraction of the universe per distance bin from the mapped clay unit, with
95% block-bootstrap CIs (504 m blocks, 1000 draws, seed 42), and the
near (36-150 m) minus far (> 400 m) contrast plus a 90-150 m near-band
robustness version. Pixel-level p-values are not used. Deterministic.

Usage:  python code/contact_mode_block_test.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.warp import transform as rtransform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import CUBE_IDS  # noqa: E402
from contact_distance_to_contact import PALETTE, load_cube, masks_from_labels, md5  # noqa: E402
from contact_band_center_map import DIST_EDGES, DIST_LABELS, unit_masks  # noqa: E402
from contact_carbonate_patches import CUBES, GEOMAP, FRAME, LONLAT, load_ddr_lonlat  # noqa: E402
from figstyle import label_panels, safe_savefig  # noqa: E402

BC = ROOT / "outputs" / "revision" / "band_center_map"
OH, MF23 = BC / "oh_filter", BC / "matched_filter" / "2300"
OUT = BC / "combinations"
SPLIT_UM = 2.305
BLOCK_PX, NBOOT = 28, 1000
RNG = np.random.default_rng(42)
NEAR, NEAR90, FAR_MIN = (36.1, 150.0), (90.0, 150.0), 400.0
CLASSES = {1: ("carbonate-mode", "#ff7f0e"), 2: ("serpentine/talc-mode", "#8c564b"), 3: ("both bands", "#2ca02c")}


def project(cube, c):
    lon, lat = load_ddr_lonlat(CUBES[cube]["ddr"])
    lon, lat = lon.ravel(), lat.ravel()
    fz = np.load(GEOMAP / f"{cube}_fractions.npz")
    tr = Affine.from_gdal(*fz["transform"])
    with rasterio.open(sorted((ROOT / "outputs" / "_v8_mtr3_with_brown" / cube / "geotiffs").glob("*.tif"))[0]) as ds:
        crs = ds.crs
    o = json.loads(FRAME.read_text())["crism"][cube]
    okp = np.isfinite(lon) & np.isfinite(lat)
    xs = np.full(lon.shape, np.nan); ys = np.full(lon.shape, np.nan)
    X, Y = rtransform(LONLAT, crs, lon[okp].astype(float), lat[okp].astype(float))
    xs[okp] = np.asarray(X) + o["dx_m"]; ys[okp] = np.asarray(Y) + o["dy_m"]
    colf, rowf = ~tr * (xs, ys)
    return xs, np.floor(rowf), np.floor(colf)


def block_boot_frac(ind: np.ndarray, bid: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray | None = None) -> dict:
    """Block bootstrap of mean(ind[mask_a]) or mean(ind[mask_a]) - mean(ind[mask_b])."""
    blocks = np.unique(bid[mask_a] if mask_b is None else np.concatenate([bid[mask_a], bid[mask_b]]))
    per_a = {b: ind[mask_a & (bid == b)] for b in blocks}
    per_b = {b: ind[mask_b & (bid == b)] for b in blocks} if mask_b is not None else None
    stat = float(ind[mask_a].mean() - (ind[mask_b].mean() if mask_b is not None else 0.0))
    if blocks.size < 5:
        return {"stat": stat, "ci_lo": np.nan, "ci_hi": np.nan, "n_blocks": int(blocks.size)}
    boots = []
    for _ in range(NBOOT):
        pick = RNG.choice(blocks, blocks.size, replace=True)
        a = np.concatenate([per_a[b] for b in pick])
        if mask_b is None:
            if a.size > 20:
                boots.append(a.mean())
        else:
            bb = np.concatenate([per_b[b] for b in pick])
            if a.size > 20 and bb.size > 20:
                boots.append(a.mean() - bb.mean())
    boots = np.array(boots)
    return {"stat": stat, "ci_lo": float(np.percentile(boots, 2.5)), "ci_hi": float(np.percentile(boots, 97.5)),
            "n_blocks": int(blocks.size)}


def main() -> None:
    t0 = datetime.now(timezone.utc)
    bin_rows, con_rows = [], []
    for cube in CUBE_IDS:
        a = np.load(OH / f"{cube}_ohfit_sensor.npz"); b = np.load(MF23 / f"{cube}_fit_sensor.npz")
        det14, snr14 = a["det"].astype(bool), a["snr"].astype(float)
        det23, cen23, snr23 = b["det"].astype(bool), b["centre"].astype(float), b["snr"].astype(float)
        c = load_cube(cube); m = masks_from_labels(c["labels"]); units, d, fr = unit_masks(cube, m)
        xs, gr, gc = project(cube, c)
        H, W = c["labels"].shape
        inside = np.isfinite(xs) & (gr >= 0) & (gr < H) & (gc >= 0) & (gc < W)
        gri = np.where(inside, gr, 0).astype(int); gci = np.where(inside, gc, 0).astype(int)
        pure = np.where(inside, units["olivine map-pure"][gri, gci], False)
        dclay = np.where(inside, d["clay"][gri, gci], np.nan)
        dcap = np.where(inside, d["cap"][gri, gci], np.nan)
        mvalid = np.where(inside, m["valid"][gri, gci], False)
        universe = inside & mvalid & pure & (dcap > 90) & np.isfinite(snr14) & np.isfinite(snr23)
        bid = (gri // BLOCK_PX) * 100000 + (gci // BLOCK_PX)
        cls = np.zeros(det23.size, np.int8)
        cls[det23 & ~det14 & (cen23 < SPLIT_UM)] = 1
        cls[det23 & ~det14 & (cen23 >= SPLIT_UM)] = 2
        cls[det23 & det14] = 3
        for k, (lab, _) in CLASSES.items():
            ind = (cls == k).astype(float)
            for (lo, hi), blab in zip(zip(DIST_EDGES[:-1], DIST_EDGES[1:]), DIST_LABELS):
                sel = universe & (dclay > lo) & (dclay <= hi)
                r = block_boot_frac(ind, bid, sel) if sel.sum() > 50 else {"stat": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n_blocks": 0}
                bin_rows.append({"cube": cube, "class": lab, "dist_bin_m": blab, "n": int(sel.sum()),
                                 "frac": r["stat"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "n_blocks": r["n_blocks"]})
            far = universe & (dclay > FAR_MIN)
            for nlab, nb in (("36-150", NEAR), ("90-150", NEAR90)):
                near = universe & (dclay > nb[0]) & (dclay <= nb[1])
                if near.sum() > 50 and far.sum() > 50:
                    r = block_boot_frac(ind, bid, near, far)
                    con_rows.append({"cube": cube, "class": lab, "near_band": nlab, "n_near": int(near.sum()),
                                     "n_far": int(far.sum()), "frac_near": float(ind[near].mean()),
                                     "frac_far": float(ind[far].mean()), "diff": r["stat"],
                                     "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "n_blocks": r["n_blocks"],
                                     "excludes_zero": bool(np.isfinite(r["ci_lo"]) and (r["ci_lo"] > 0 or r["ci_hi"] < 0))})
        print(cube, "done")
    bdf = pd.DataFrame(bin_rows); bdf.to_csv(OUT / "mode_block_bins.csv", index=False)
    cdf = pd.DataFrame(con_rows); cdf.to_csv(OUT / "mode_block_contrast.csv", index=False)
    print(cdf.round(4).to_string())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    xs_ = np.arange(len(DIST_LABELS))
    for ax, cube in zip(axes.ravel(), CUBE_IDS):
        for k, (lab, col) in CLASSES.items():
            sub = bdf[(bdf.cube == cube) & (bdf["class"] == lab)]
            y = sub.frac.to_numpy() * 100
            lo = (y - sub.ci_lo.to_numpy() * 100); hi = (sub.ci_hi.to_numpy() * 100 - y)
            ax.errorbar(xs_ + (k - 2) * 0.12, y, yerr=[np.nan_to_num(lo), np.nan_to_num(hi)], fmt="-o",
                        color=col, lw=1.8, ms=5, capsize=3, label=lab)
        ax.set_title(cube, fontsize=10, color=PALETTE[cube])
        ax.set_xticks(xs_); ax.set_xticklabels(DIST_LABELS)
        ax.set_ylabel("% of map-pure olivine pixels"); ax.grid(True, color="0.94")
        ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=8, frameon=False)
    for ax in axes[1]:
        ax.set_xlabel("distance from mapped clay (m)")
    fig.suptitle("Olivine-unit 2.3 um phase classes vs distance from the basement contact\n"
                 "(block-bootstrap 95% CIs, 504 m blocks; 36-90 m bin is mixing/registration-limited)", fontsize=10.5)
    fig.tight_layout()
    label_panels(fig)
    safe_savefig(fig, OUT / "fig_mode_block_test")
    plt.close(fig)
    log = {"script": Path(__file__).name, "script_md5": md5(Path(__file__)),
           "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
           "split_um": SPLIT_UM, "block_px": BLOCK_PX, "nboot": NBOOT, "seed": 42,
           "near": NEAR, "near90": NEAR90, "far_min": FAR_MIN}
    (OUT / "run_log_mode_block_test.json").write_text(json.dumps(log, indent=2, default=str))


if __name__ == "__main__":
    main()
