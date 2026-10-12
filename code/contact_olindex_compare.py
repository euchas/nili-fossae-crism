"""OLINDEX2 vs OLINDEX3 on the same MTR3 cube, same code path, for the site-1 window.

Answers 'why does OLINDEX3 look different from the paper's OLINDEX2': both
are computed here from the FRT0000A4FC MTR3 I/F cube with identical
7-channel nanmedian spectral kernels and no spatial filtering, then shown
side by side in the paper-figure window with black = 0. Saves both
parameter arrays (18 m grid) for contact_paper_fig_site1.py. Deterministic.

OLINDEX2 (Salvatore et al. 2010): band depths at 1.08, 1.21, 1.33, 1.47 um
  (weights 0.1/0.1/0.4/0.4) against a 1.75-2.40 um continuum.
OLINDEX3 (Viviano-Beck et al. 2014): band depths at 1.21, 1.25, 1.263,
  1.276, 1.33 um (0.1/0.1/0.2/0.2/0.4) against a 1.75-1.862 um continuum.

Usage:  python code/contact_olindex_compare.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from stage_v7c_mtr3_pipeline import load_mtr3  # noqa: E402
from vb_parameters import _kernel, _rb, calc_olindex3  # noqa: E402
from contact_site_panels import SITES, working_crs, lonlat_to_xy, read_on_grid, scalebar  # noqa: E402
from contact_height_above_contact import md5  # noqa: E402
from figstyle import safe_savefig  # noqa: E402

CUBE = "FRT0000A4FC"
HDR = ROOT / "cubes" / f"{CUBE.lower()}_07_if166j_mtr3" / f"{CUBE.lower()}_07_if166j_mtr3.hdr"
OUT = ROOT / "outputs" / "revision" / "sites"
FRAME_JSON = ROOT / "outputs" / "revision" / "contact_3d" / "frame" / "coreg_offsets.json"


def calc_olindex2(data: np.ndarray, wav: np.ndarray) -> np.ndarray:
    R1080 = _kernel(data, wav, 1.080, 7); R1210 = _kernel(data, wav, 1.210, 7)
    R1330 = _kernel(data, wav, 1.330, 7); R1470 = _kernel(data, wav, 1.470, 7)
    R1750 = _kernel(data, wav, 1.750, 7); R2400 = _kernel(data, wav, 2.400, 7)
    slope = np.where(R2400 != R1750, (R2400 - R1750) / (2.400 - 1.750), 0.0)

    def rc(w):
        return R1750 + slope * (w - 1.750)

    return 0.1 * _rb(R1080, rc(1.080)) + 0.1 * _rb(R1210, rc(1.210)) + 0.4 * _rb(R1330, rc(1.330)) + 0.4 * _rb(R1470, rc(1.470))


def main() -> None:
    t0 = datetime.now(timezone.utc)
    cube, wav, meta = load_mtr3(HDR)
    valid = np.isfinite(cube[np.abs(wav - 1.33).argmin()])
    o2 = calc_olindex2(cube, wav).astype(np.float32); o3 = calc_olindex3(cube, wav).astype(np.float32)
    o2[~valid] = np.nan; o3[~valid] = np.nan
    del cube
    np.save(OUT / f"{CUBE}_OLINDEX2_mtr3.npy", o2); np.save(OUT / f"{CUBE}_OLINDEX3_mtr3.npy", o3)
    # cross-check against the v8 stack
    v8log = json.loads((ROOT / "outputs" / "_v8_mtr3_with_brown" / CUBE / "logs" / f"{CUBE}_v8_log.json").read_text())
    stack = np.load(ROOT / "outputs" / "_v8_mtr3_with_brown" / CUBE / "arrays" / f"{CUBE}_v8_param_stack.npy")
    v8 = stack[..., v8log["v8_params"].index("OLINDEX3")]
    ok = np.isfinite(o3) & np.isfinite(v8)
    print("OLINDEX3 fresh vs v8 stack: max |diff| =", float(np.nanmax(np.abs(o3[ok] - v8[ok]))), "n", int(ok.sum()))

    # window on the paper-figure grid
    crs = working_crs(); W = SITES["site1_wide"]
    xs, ys = lonlat_to_xy(W["lon"], W["lat"], crs)
    xc, yc = xs + W["center_offset_m"][0], ys + W["center_offset_m"][1]
    bw, bh = W["box_wh_m"]; g = 2.0
    nx, ny = int(bw / g), int(bh / g); tr = Affine(g, 0, xc - bw / 2, 0, -g, yc + bh / 2)
    gt = sorted((ROOT / "outputs" / "_v8_mtr3_with_brown" / CUBE / "geotiffs").glob("*.tif"))[0]
    with rasterio.open(gt) as ds:
        m_tr, m_crs = ds.transform, ds.crs
    off = json.loads(FRAME_JSON.read_text())["crism"][CUBE]
    m_tr_eff = Affine.translation(off["dx_m"], off["dy_m"]) * m_tr
    wins = {}
    for name, arr in (("OLINDEX2", o2), ("OLINDEX3", o3)):
        tmp = OUT / f"_tmp_{name}.tif"
        with rasterio.open(tmp, "w", driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1, dtype="float32",
                           crs=m_crs, transform=m_tr_eff, nodata=np.nan) as dst:
            dst.write(arr, 1)
        wins[name] = read_on_grid(str(tmp), m_tr_eff, m_crs, None, crs, tr, (ny, nx), Resampling.nearest)
        tmp.unlink(missing_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
    for ax, name in zip(axes[:2], ("OLINDEX2", "OLINDEX3")):
        a = wins[name]; vmax = float(np.nanpercentile(a, 99.5))
        im = ax.imshow(a, cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label=name)
        ax.set_title(f"{name}, {CUBE} MTR3, black = 0 (window 0-{vmax:.2f})\n"
                     f"{'1.08/1.21/1.33/1.47 um vs 1.75-2.40 continuum' if name == 'OLINDEX2' else '1.21-1.33 um vs 1.75-1.86 continuum'}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); scalebar(ax, nx, ny, g, length_m=150)
    ax = axes[2]
    a, b = wins["OLINDEX2"], wins["OLINDEX3"]; ok = np.isfinite(a) & np.isfinite(b)
    ax.scatter(a[ok][::7], b[ok][::7], s=3, alpha=0.3, color="k")
    r = np.corrcoef(a[ok], b[ok])[0, 1]
    ax.set_xlabel("OLINDEX2"); ax.set_ylabel("OLINDEX3"); ax.set_title(f"pixelwise in the window, r = {r:.2f}", fontsize=10)
    ax.grid(True, color="0.9")
    fig.suptitle("Same cube, same 7-channel spectral kernels, no spatial filtering: the two indices differ by formula only", fontsize=11)
    fig.tight_layout()
    safe_savefig(fig, OUT / f"fig_olindex2_vs_olindex3_{CUBE}", bbox_inches="tight")
    log = {"script": Path(__file__).name, "script_md5": md5(Path(__file__)), "cube": CUBE, "hdr": str(HDR.relative_to(ROOT)),
           "kernel_channels": 7, "spatial_filter": None, "corr_in_window": float(r),
           "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat()}
    (OUT / "run_log_olindex_compare.json").write_text(json.dumps(log, indent=2))
    print("r =", round(float(r), 3))


if __name__ == "__main__":
    main()
