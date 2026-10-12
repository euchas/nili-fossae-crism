"""Joint 1.4 um / 2.3 um detection combinations (Plebani chain).

Combines the per-pixel matched-filter results of the 1.4 um OH run
(oh_filter) and the 2.3 um metal-OH run (matched_filter/2300) - same sensor
pixel order - into four classes: neither, 1.4-only, 2.3-only, both.
Products:
  - per-unit class counts/fractions (CSV);
  - class maps per cube with unit outlines;
  - joint histogram of fitted centres (1.4 um vs 2.3 um) for 'both' pixels,
    4 nm bins, per unit and cube, with reference mineral loci;
  - 1D histograms of 2.3 um centre for '2.3-only' vs 'both' pixels
    (Mg-carbonate has a 2.30 um band but no 1.4 um OH band; a hydrated
    phyllosilicate carries both).
Caveat carried everywhere: the 1.4 um band is intrinsically weaker and its
detection limit is higher, so '2.3-only' is not proof of an anhydrous phase.
Deterministic.

Usage:  python code/contact_band_combinations.py
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
from scipy.spatial import cKDTree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import CUBE_IDS  # noqa: E402
from contact_distance_to_contact import PALETTE, load_cube, masks_from_labels, md5  # noqa: E402
from contact_band_center_map import unit_masks  # noqa: E402
from contact_carbonate_patches import CUBES, GEOMAP, FRAME, LONLAT, load_ddr_lonlat  # noqa: E402
from figstyle import draw_units, label_panels, safe_savefig, unit_legend  # noqa: E402

BC = ROOT / "outputs" / "revision" / "band_center_map"
OH = BC / "oh_filter"
MF23 = BC / "matched_filter" / "2300"
OUT = BC / "combinations"
OUT.mkdir(parents=True, exist_ok=True)

BIN_UM = 0.004
R14 = (1.370, 1.430)
R23 = (2.280, 2.350)
FILL_TOL_M = 30.0
CLASS_COLS = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}       # 1.4-only blue, 2.3-only orange, both green
CLASS_LAB = {0: "neither", 1: "1.4 um only", 2: "2.3 um only", 3: "both"}
LOCI = {  # (1.4 um, 2.3 um) reference positions, approximate literature values
    "Fe/Mg smectite": (1.415, 2.300), "nontronite": (1.428, 2.290), "saponite": (1.392, 2.312),
    "serpentine": (1.386, 2.322), "talc": (1.392, 2.313), "Mg-chlorite": (1.410, 2.334),
}


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
    return xs, ys, np.floor(rowf), np.floor(colf), tr


def main() -> None:
    t0 = datetime.now(timezone.utc)
    rows, joint, one_d = [], {}, {}
    for cube in CUBE_IDS:
        a = np.load(OH / f"{cube}_ohfit_sensor.npz")
        b = np.load(MF23 / f"{cube}_fit_sensor.npz")
        det14, cen14 = a["det"].astype(bool), a["centre"].astype(float)
        det23, cen23 = b["det"].astype(bool), b["centre"].astype(float)
        assert det14.size == det23.size
        cls = (det14.astype(int) * 1) + (det23.astype(int) * 2)          # 0..3

        c = load_cube(cube)
        m = masks_from_labels(c["labels"])
        units, d, fr = unit_masks(cube, m)
        xs, ys, gr, gc, tr = project(cube, c)
        H, W = c["labels"].shape
        inside = np.isfinite(xs) & (gr >= 0) & (gr < H) & (gc >= 0) & (gc < W)
        gri = np.where(inside, gr, 0).astype(int); gci = np.where(inside, gc, 0).astype(int)
        samp = lambda arr: np.where(inside, arr[gri, gci], np.nan)
        f_clay, f_cap = samp(fr["clay"]), samp(fr["cap"])
        mvalid = np.where(inside, m["valid"][gri, gci], False)
        usable = inside & mvalid & np.isfinite(cen14) & np.isfinite(cen23)
        unit_defs = {"olivine map-pure": usable & np.where(inside, units["olivine map-pure"][gri, gci], False),
                     "clay map-pure": usable & (f_clay >= 0.9),
                     "cap map-pure": usable & (f_cap >= 0.9)}
        for uname, um in unit_defs.items():
            n = int(um.sum())
            row = {"cube": cube, "unit": uname, "n": n}
            for k, lab in CLASS_LAB.items():
                row[f"n_{lab}"] = int((um & (cls == k)).sum())
                row[f"frac_{lab}"] = float((um & (cls == k)).sum() / n) if n else np.nan
            d23 = um & det23
            row["frac_of_2.3_dets_with_1.4"] = float((d23 & det14).sum() / d23.sum()) if d23.any() else np.nan
            rows.append(row)
            joint[(cube, uname)] = (cen14[um & (cls == 3)], cen23[um & (cls == 3)])
            one_d[(cube, uname)] = {"2.3 only": cen23[um & (cls == 2)], "both": cen23[um & (cls == 3)]}

        # class map
        tree = cKDTree(np.column_stack([xs[usable], ys[usable]]))
        cc, rr = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        gx, gy = tr * (cc.ravel(), rr.ravel())
        dist, idx = tree.query(np.column_stack([gx, gy]), distance_upper_bound=FILL_TOL_M)
        hit = np.isfinite(dist)
        cmap_full = np.full(H * W, np.nan)
        cmap_full[hit] = cls[usable][idx[hit]]
        cmap_full = cmap_full.reshape(H, W)
        cmap_full[~m["valid"]] = np.nan
        np.save(OUT / f"{cube}_combo_class.npy", cmap_full)

        fig, ax = plt.subplots(figsize=(9, 8.5))
        ax.imshow(np.where(m["valid"], 0.93, np.nan), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        lay = np.where(cmap_full > 0, cmap_full, np.nan)
        ax.imshow(lay, cmap=ListedColormap([CLASS_COLS[1], CLASS_COLS[2], CLASS_COLS[3]]),
                  vmin=1, vmax=3, interpolation="nearest")
        draw_units(ax, fr)
        ax.set_xticks([]); ax.set_yticks([])
        handles = [Patch(color=CLASS_COLS[k], label=CLASS_LAB[k]) for k in (1, 2, 3)]
        leg1 = ax.legend(handles=handles, loc="lower right", fontsize=9, frameon=True, framealpha=0.85,
                         title="detections")
        ax.add_artist(leg1)
        unit_legend(ax, loc="lower left")
        nu = {k: int((cls[usable] == k).sum()) for k in (1, 2, 3)}
        ax.set_title(f"{cube}: joint 1.4 / 2.3 um matched-filter detections  "
                     f"(1.4 only n={nu[1]}, 2.3 only n={nu[2]}, both n={nu[3]})", fontsize=10)
        fig.tight_layout()
        label_panels(fig)
        safe_savefig(fig, OUT / f"fig_combo_map_{cube}", bbox_inches="tight")
        plt.close(fig)
        print(cube, nu)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "combo_counts.csv", index=False)
    print(df.round(4).to_string())

    # joint histogram, 'both' pixels: rows = unit, cols = cube
    xe = np.arange(R14[0], R14[1] + 1e-9, BIN_UM); ye = np.arange(R23[0], R23[1] + 1e-9, BIN_UM)
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5), sharex=True, sharey=True)
    for i, uname in enumerate(("clay map-pure", "olivine map-pure")):
        for j, cube in enumerate(CUBE_IDS):
            ax = axes[i, j]
            x, y = joint[(cube, uname)]
            if x.size:
                h, _, _ = np.histogram2d(x, y, bins=[xe, ye])
                im = ax.pcolormesh(xe, ye, h.T, cmap="magma", shading="flat")
            for name, (lx, ly) in LOCI.items():
                ax.plot(lx, ly, "o", mfc="none", mec="cyan", ms=7, mew=1.2)
                ax.annotate(name, (lx, ly), xytext=(4, 4), textcoords="offset points", fontsize=6.5, color="cyan")
            ax.set_title(f"{cube} - {uname.replace(' map-pure', '')} (both, n={x.size})", fontsize=9,
                         color=PALETTE[cube])
            ax.grid(True, color="0.85", lw=0.4)
    for ax in axes[1]:
        ax.set_xlabel("fitted 1.4 um centre (um)")
    for ax in axes[:, 0]:
        ax.set_ylabel("fitted 2.3 um centre (um)")
    fig.suptitle(f"Joint band-centre combinations for pixels detected at BOTH 1.4 and 2.3 um "
                 f"({BIN_UM * 1000:.0f} nm bins; cyan = literature loci, approximate)", fontsize=11)
    fig.tight_layout()
    label_panels(fig)
    safe_savefig(fig, OUT / "fig_combo_hist2d")
    plt.close(fig)

    # 1D: 2.3 um centre, '2.3 only' vs 'both', clay and olivine, all cubes
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharex=True)
    for i, uname in enumerate(("clay map-pure", "olivine map-pure")):
        for j, cube in enumerate(CUBE_IDS):
            ax = axes[i, j]
            dd = one_d[(cube, uname)]
            for lab, col in (("2.3 only", CLASS_COLS[2]), ("both", CLASS_COLS[3])):
                v = dd[lab]
                if v.size >= 20:
                    ax.hist(v, bins=ye, color=col, alpha=0.55, label=f"{lab} (n={v.size})")
            for xv, lab in ((2.300, "Mg-carb./smect."), (2.312, "sap./talc"), (2.322, "serp."), (2.334, "chl.")):
                ax.axvline(xv, color="0.75", lw=0.7)
                ax.annotate(lab, (xv, ax.get_ylim()[1]), fontsize=6.5, ha="center", color="0.4",
                            annotation_clip=False)
            ax.set_title(f"{cube} - {uname.replace(' map-pure', '')}", fontsize=9, color=PALETTE[cube])
            ax.legend(fontsize=7.5, frameon=False)
            ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("fitted 2.3 um centre (um)")
    for ax in axes[:, 0]:
        ax.set_ylabel("pixels")
    fig.suptitle("2.3 um centre of pixels WITHOUT a 1.4 um detection (orange) vs WITH one (green)\n"
                 "Mg-carbonate: 2.30 um and no 1.4 um band; hydrated phyllosilicates carry both "
                 "(1.4 um is the weaker band, so '2.3 only' is not proof of an anhydrous phase)", fontsize=10.5)
    fig.tight_layout()
    label_panels(fig)
    safe_savefig(fig, OUT / "fig_combo_hist1d")
    plt.close(fig)

    log = {"script": Path(__file__).name, "script_md5": md5(Path(__file__)),
           "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
           "bin_um": BIN_UM, "ranges": {"1.4": R14, "2.3": R23}, "fill_tol_m": FILL_TOL_M,
           "loci_note": "reference loci approximate; absolute centres carry the ~13 nm inter-chain "
                        "offset caveat - use for relative placement only",
           "inputs": {c: {"oh": str(OH / f"{c}_ohfit_sensor.npz"), "mf23": str(MF23 / f"{c}_fit_sensor.npz")}
                      for c in CUBE_IDS}, "deterministic": True}
    (OUT / "run_log.json").write_text(json.dumps(log, indent=2, default=str))
    print("done ->", OUT)


if __name__ == "__main__":
    main()
