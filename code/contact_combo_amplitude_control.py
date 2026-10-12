"""Amplitude-matched control for the 1.4/2.3 um combination result.

'2.3 only' pixels could be (a) an anhydrous 2.3 um phase (Mg-carbonate) or
(b) a diluted hydrated phyllosilicate whose weaker 1.4 um band fell below
threshold. (b) predicts that, at the SAME 2.3 um amplitude, olivine-unit and
clay-unit pixels should show the same 1.4 um co-detection rate and the same
median 1.4 um amplitude SNR; (a) predicts the olivine unit stays far below
the clay unit at every amplitude. Bins 2.3-detected pixels by fitted 2.3 um
amplitude and compares the units. Deterministic.

Usage:  python code/contact_combo_amplitude_control.py
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
from contact_band_center_map import unit_masks  # noqa: E402
from contact_carbonate_patches import CUBES, GEOMAP, FRAME, LONLAT, load_ddr_lonlat  # noqa: E402
from figstyle import label_panels  # noqa: E402

BC = ROOT / "outputs" / "revision" / "band_center_map"
OH = BC / "oh_filter"
MF23 = BC / "matched_filter" / "2300"
OUT = BC / "combinations"
AMP_EDGES = np.array([0.006, 0.009, 0.012, 0.016, 0.022, 0.030, 0.045])
AMP_LABELS = [f"{a * 100:.1f}-{b * 100:.1f}%" for a, b in zip(AMP_EDGES[:-1], AMP_EDGES[1:])]
MIN_N = 100


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


def main() -> None:
    t0 = datetime.now(timezone.utc)
    rows = []
    for cube in CUBE_IDS:
        a = np.load(OH / f"{cube}_ohfit_sensor.npz")
        b = np.load(MF23 / f"{cube}_fit_sensor.npz")
        det14, snr14 = a["det"].astype(bool), a["snr"].astype(float)
        det23, amp23 = b["det"].astype(bool), b["amp"].astype(float)
        c = load_cube(cube)
        m = masks_from_labels(c["labels"])
        units, d, fr = unit_masks(cube, m)
        xs, gr, gc = project(cube, c)
        H, W = c["labels"].shape
        inside = np.isfinite(xs) & (gr >= 0) & (gr < H) & (gc >= 0) & (gc < W)
        gri = np.where(inside, gr, 0).astype(int); gci = np.where(inside, gc, 0).astype(int)
        f_clay = np.where(inside, fr["clay"][gri, gci], np.nan)
        mvalid = np.where(inside, m["valid"][gri, gci], False)
        usable = inside & mvalid & det23 & np.isfinite(snr14)
        unit_defs = {"olivine map-pure": usable & np.where(inside, units["olivine map-pure"][gri, gci], False),
                     "clay map-pure": usable & (f_clay >= 0.9)}
        for uname, um in unit_defs.items():
            for lo, hi, lab in zip(AMP_EDGES[:-1], AMP_EDGES[1:], AMP_LABELS):
                sel = um & (amp23 >= lo) & (amp23 < hi)
                n = int(sel.sum())
                rows.append({"cube": cube, "unit": uname, "amp23_bin": lab, "amp_lo": lo, "n": n,
                             "codet_frac_1.4": float(det14[sel].mean()) if n >= MIN_N else np.nan,
                             "snr14_median": float(np.median(snr14[sel])) if n >= MIN_N else np.nan,
                             "snr14_q75": float(np.percentile(snr14[sel], 75)) if n >= MIN_N else np.nan})
        print(cube, "done")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "combo_amplitude_control.csv", index=False)
    print(df.round(4).to_string())

    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharex=True)
    xs_ = np.arange(len(AMP_LABELS))
    for j, cube in enumerate(CUBE_IDS):
        for uname, col, mk in (("clay map-pure", "#1f77b4", "o"), ("olivine map-pure", "#d62728", "s")):
            sub = df[(df.cube == cube) & (df.unit == uname)]
            axes[0, j].plot(xs_, sub["codet_frac_1.4"], f"-{mk}", color=col, lw=1.8, ms=5,
                            label=uname.replace(" map-pure", ""))
            axes[1, j].plot(xs_, sub.snr14_median, f"-{mk}", color=col, lw=1.8, ms=5,
                            label=uname.replace(" map-pure", ""))
            for x, nv in zip(xs_, sub.n):
                axes[0, j].annotate(f"{int(nv)}", (x, -0.03), fontsize=6, ha="center", color=col,
                                    xytext=(0, -8 if uname.startswith("oliv") else 0), textcoords="offset points",
                                    annotation_clip=False)
        axes[0, j].set_title(cube, fontsize=10, color=PALETTE[cube])
        axes[0, j].set_ylim(0, 1); axes[0, j].set_ylabel("fraction with 1.4 um detection")
        axes[1, j].set_ylabel("median 1.4 um amplitude SNR")
        axes[1, j].set_xticks(xs_); axes[1, j].set_xticklabels(AMP_LABELS, rotation=45, ha="right", fontsize=8)
        axes[1, j].set_xlabel("fitted 2.3 um amplitude bin")
        for ax in axes[:, j]:
            ax.grid(True, color="0.94"); ax.spines[["top", "right"]].set_visible(False)
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Amplitude-matched control: among 2.3 um-detected pixels, how often is a 1.4 um band also "
                 "detected?\nIf the olivine-unit 2.3 um phase were diluted clay, red should follow blue at "
                 "matched amplitude (small numbers = n per bin)", fontsize=10.5)
    fig.tight_layout()
    label_panels(fig)
    fig.savefig(OUT / "fig_combo_amplitude_control.png", dpi=200)
    fig.savefig(OUT / "fig_combo_amplitude_control.pdf")
    plt.close(fig)
    log = {"script": Path(__file__).name, "script_md5": md5(Path(__file__)),
           "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
           "amp_edges": AMP_EDGES.tolist(), "min_n": MIN_N, "deterministic": True}
    (OUT / "run_log_amplitude_control.json").write_text(json.dumps(log, indent=2, default=str))
    print("done ->", OUT)


if __name__ == "__main__":
    main()
