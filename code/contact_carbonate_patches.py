"""Where do the Plebani-classifier carbonate patches sit relative to the
mapped clay contact?

Inputs (read-only):
- Carbonate maps from the Z: archive pass (carb-plebani-1.0, crism_ml HBM
  ensemble, carbonate classes 13 Ca/Fe CO3 and 30 MgCO3, threshold 0.5,
  >= 5 px regions): config.PLEBANI_CARB_DIR/
  <stem>_carb.npz, TRR3 sensor space (480 x 640). Available for
  FRT00003E12, FRT0000871C, FRT0000B438 (A4FC not in that archive).
- Sensor-space DDRs (DE166L_DDR1, lat band 4 / lon band 5) from the Nili
  Planum upload, same 480 x 640 grid.
- Geologic-map unit fractions on the shifted CRISM grid
  (outputs/revision/contact_3d/geomap/frame_ctx) and the Stage 0 offsets
  (the DDR coordinates carry the same pointing error as the MTR3 product,
  so the same per-cube shift is applied).

Outputs (outputs/revision/contact_3d/stage6_carbonate/): per-patch table,
distance CDF comparison vs the olivine-unit background, per-cube map
figures, run log.

Usage: python code/contact_carbonate_patches.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import transform as rtransform
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from contact_distance_to_contact import PIX_M, V8_PARAMS, load_cube, masks_from_labels  # noqa: E402

from config import CRISM_RAW_DIR, PLEBANI_CARB_DIR as ZMAPS  # noqa: E402
RAW = CRISM_RAW_DIR
GEOMAP = ROOT / "outputs" / "revision" / "contact_3d" / "geomap" / "frame_ctx"
FRAME = ROOT / "outputs" / "revision" / "contact_3d" / "frame" / "coreg_offsets.json"
OUT = ROOT / "outputs" / "revision" / "contact_3d" / "stage6_carbonate"
OUT.mkdir(parents=True, exist_ok=True)
LONLAT = "+proj=longlat +R=3393833.26 +no_defs"

CUBES = {
    "FRT00003E12": {"stem": "frt00003e12_07_if166l_trr3",
                    "ddr": RAW / "FRT00003E12_07_IF166L_TRR3" / "FRT00003E12_07_DE166L_DDR1" / "frt00003e12_07_de166l_ddr1.img"},
    "FRT0000871C": {"stem": "frt0000871c_07_if166l_trr3",
                    "ddr": RAW / "frt0000871c_07_if166l_trr3" / "frt0000871c_07_de166l_ddr1" / "frt0000871c_07_de166l_ddr1.img"},
    "FRT0000B438": {"stem": "frt0000b438_07_if166l_trr3",
                    "ddr": RAW / "frt0000b438_07_if166l_trr3" / "frt0000b438_07_de166l_ddr1" / "frt0000b438_07_de166l_ddr1.img"},
    "FRT0000A4FC": {"stem": "frt0000a4fc_07_if166l_trr3",
                    "ddr": RAW / "frt0000a4fc_07_if166l_trr3" / "frt0000a4fc_07_de166l_ddr1" / "frt0000a4fc_07_de166l_ddr1.img"},
}


def load_carb(stem: str) -> dict:
    d = np.load(ZMAPS / f"{stem}_carb.npz")
    H, W = d["im_shape"]
    return {"region": np.unpackbits(d["region_mask"]).reshape(H, W).astype(bool),
            "valid": np.unpackbits(d["valid"]).reshape(H, W).astype(bool),
            "mgco3": d["mgco3_u16"].astype(np.float32) / 65535.0,
            "cafeco3": d["cafeco3_u16"].astype(np.float32) / 65535.0,
            "meta": json.loads(str(d["meta"]))}


def load_ddr_lonlat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lbl = path.with_suffix(".lbl").read_text()
    import re
    lines = int(re.search(r"LINES\s*=\s*(\d+)", lbl).group(1))
    samples = int(re.search(r"LINE_SAMPLES\s*=\s*(\d+)", lbl).group(1))
    bands = int(re.search(r"BANDS\s*=\s*(\d+)", lbl).group(1))
    arr = np.fromfile(path, dtype="<f4").reshape(bands, lines, samples)
    lat, lon = arr[3].copy(), arr[4].copy()
    for a in (lat, lon):
        a[a >= 65535.0] = np.nan
    return lon, lat


def main() -> None:
    t0 = datetime.now(timezone.utc)
    offsets = json.loads(FRAME.read_text())["crism"]
    patch_rows, summary = [], {}
    fig_cdf, ax_cdf = plt.subplots(figsize=(6.5, 4.2))
    palette = {"FRT00003E12": "#0072B2", "FRT0000871C": "#E69F00", "FRT0000B438": "#CC79A7", "FRT0000A4FC": "#009E73"}
    for cube, cfg in CUBES.items():
        carb = load_carb(cfg["stem"])
        lon, lat = load_ddr_lonlat(cfg["ddr"])
        assert lon.shape == carb["region"].shape
        c = load_cube(cube)
        m = masks_from_labels(c["labels"])
        fz = np.load(GEOMAP / f"{cube}_fractions.npz")
        fr = {k: fz[k] for k in ("clay", "cap", "dune", "graben", "oliv", "outline")}
        tr = Affine.from_gdal(*fz["transform"])
        with rasterio.open(sorted((ROOT / "outputs" / "_v8_mtr3_with_brown" / cube / "geotiffs").glob("*.tif"))[0]) as ds:
            crs = ds.crs
        o = offsets[cube]
        # project all sensor pixels once (needed for patch context); shift = same correction as the MTR3 product
        ok = np.isfinite(lon) & np.isfinite(lat)
        xs = np.full(lon.shape, np.nan); ys = np.full(lon.shape, np.nan)
        X, Y = rtransform(LONLAT, crs, lon[ok].astype(float), lat[ok].astype(float))
        xs[ok] = np.asarray(X) + o["dx_m"]
        ys[ok] = np.asarray(Y) + o["dy_m"]
        inv = ~tr
        col, row = inv * (xs, ys)
        col = np.floor(col).astype(float); row = np.floor(row).astype(float)
        H, W = c["labels"].shape
        inside = ok & (row >= 0) & (row < H) & (col >= 0) & (col < W)

        d_clay = distance_transform_edt(~(fr["clay"] >= 0.5), sampling=(PIX_M, PIX_M))
        mapped = fr["outline"] >= 0.99
        # carbonate pixels on the grid
        sel = carb["region"] & inside
        gr = row[sel].astype(int); gc = col[sel].astype(int)
        dists = d_clay[gr, gc]
        on_map = mapped[gr, gc]
        # null: map-pure olivine pixels
        pure = m["valid"] & mapped & (fr["oliv"] >= 0.9) & (fr["clay"] == 0) & ~m["clay"]
        null_d = d_clay[pure]
        # per-patch table
        lab_p, n_p = ndi.label(carb["region"], structure=np.ones((3, 3), bool))
        for pid in range(1, n_p + 1):
            pm = lab_p == pid
            if not (pm & inside).any():
                continue
            prg = row[pm & inside].astype(int); pcg = col[pm & inside].astype(int)
            r0, c0 = int(np.median(prg)), int(np.median(pcg))
            patch_rows.append({"cube": cube, "patch": pid, "n_px": int(pm.sum()),
                               "mgco3_max": float(carb["mgco3"][pm].max()), "cafeco3_max": float(carb["cafeco3"][pm].max()),
                               "x_m": float(tr.c + tr.a * (c0 + 0.5)), "y_m": float(tr.f + tr.e * (r0 + 0.5)),
                               "dist_to_clay_contact_m": float(np.median(d_clay[prg, pcg])),
                               "f_clay": float(fr["clay"][r0, c0]), "f_oliv": float(fr["oliv"][r0, c0]),
                               "f_cap": float(fr["cap"][r0, c0]), "on_map": bool(mapped[r0, c0]),
                               "gmm_class": int(c["labels"][r0, c0]),
                               "BD2500_2": float(c["params"][r0, c0, V8_PARAMS.index("BD2500_2")]),
                               "D2300": float(c["params"][r0, c0, V8_PARAMS.index("D2300")])})
        onm = on_map & np.isfinite(dists)
        summary[cube] = {
            "n_carb_px": int(sel.sum()), "n_patches": int(n_p), "n_carb_px_on_map": int(onm.sum()),
            "carb_dist_med_m": float(np.median(dists[onm])) if onm.any() else np.nan,
            "carb_frac_within_150m": float((dists[onm] <= 150).mean()) if onm.any() else np.nan,
            "carb_frac_within_400m": float((dists[onm] <= 400).mean()) if onm.any() else np.nan,
            "null_dist_med_m": float(np.median(null_d)),
            "null_frac_within_150m": float((null_d <= 150).mean()),
            "null_frac_within_400m": float((null_d <= 400).mean()),
        }
        print(cube, summary[cube])
        # CDF
        if onm.sum() > 20:
            xs_ = np.sort(dists[onm]); ax_cdf.plot(xs_, np.linspace(0, 1, xs_.size), color=palette[cube], lw=2,
                                                   label=f"{cube} carbonate (n={onm.sum()})")
            xn = np.sort(null_d); ax_cdf.plot(xn, np.linspace(0, 1, xn.size), color=palette[cube], lw=1, ls="--", alpha=0.7)
        # map figure
        cmap = ListedColormap(["#f5f5f5", "#c9c9c9", "#3a3a3a", "#e8d5a6", "#8d8d8d"])
        img = np.zeros(c["labels"].shape, np.int8)
        img[mapped] = 3
        img[fr["clay"] >= 0.5] = 1
        img[fr["cap"] >= 0.5] = 2
        img[~m["valid"]] = 0
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
        ax.scatter(gc[on_map], gr[on_map], s=4, color="#1b9e77", label="carbonate (Plebani regions)")
        ax.scatter(gc[~on_map], gr[~on_map], s=4, color="#d95f02", label="carbonate off mapped area")
        ax.contour(d_clay, levels=[150, 400], colors=["#d55e00", "#0072b2"], linewidths=0.5)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
        ax.set_title(f"{cube}: Plebani carbonate regions on the geologic map (contours 150/400 m from clay)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout(); fig.savefig(OUT / f"fig_carbonate_map_{cube}.png", dpi=180); plt.close(fig)

    ax_cdf.set_xlim(0, 1500); ax_cdf.set_xlabel("distance from mapped clay contact (m)")
    ax_cdf.set_ylabel("cumulative fraction"); ax_cdf.grid(True, color="0.92")
    ax_cdf.spines[["top", "right"]].set_visible(False)
    ax_cdf.legend(fontsize=8, frameon=False, title="solid: carbonate px; dashed: olivine-unit background")
    fig_cdf.tight_layout(); fig_cdf.savefig(OUT / "fig_carbonate_distance_cdf.png", dpi=200)
    fig_cdf.savefig(OUT / "fig_carbonate_distance_cdf.pdf"); plt.close(fig_cdf)

    pd.DataFrame(patch_rows).sort_values(["cube", "dist_to_clay_contact_m"]).to_csv(OUT / "carbonate_patches.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "run_log.json").write_text(json.dumps({"maps": str(ZMAPS), "version_expected": "carb-plebani-1.0",
        "shift_applied": {k: {kk: offsets[k][kk] for kk in ("dx_m", "dy_m")} for k in CUBES},
        "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
