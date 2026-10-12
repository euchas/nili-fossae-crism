"""Why OLINDEX2 and OLINDEX3 differ: unit-median spectra from the site-1 window
with each index's band positions, weights and continuum anchors drawn on.

Median MTR3 I/F spectra of map-pure olivine-unit and clay-unit pixels within
the paper-figure window (FRT0000A4FC, CTX-frame unit fractions), plotted over
0.9-2.6 um with (top) the OLINDEX2 sampling (1.08/1.21/1.33/1.47 um, weights
0.1/0.1/0.4/0.4, continuum 1.75-2.40 um) and (bottom) the OLINDEX3 sampling
(1.21/1.25/1.263/1.276/1.33 um, weights 0.1/0.1/0.2/0.2/0.4, continuum
1.75-1.862 um), plus the per-unit index values and their per-pixel scatter.
Deterministic.

Usage:  python code/contact_olindex_anatomy.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from stage_v7c_mtr3_pipeline import load_mtr3  # noqa: E402
from contact_height_above_contact import md5  # noqa: E402
from figstyle import safe_savefig  # noqa: E402

CUBE = "FRT0000A4FC"
HDR = ROOT / "cubes" / f"{CUBE.lower()}_07_if166j_mtr3" / f"{CUBE.lower()}_07_if166j_mtr3.hdr"
FR = ROOT / "outputs" / "revision" / "contact_3d" / "geomap" / "frame_ctx" / f"{CUBE}_fractions.npz"
OUT = ROOT / "outputs" / "revision" / "sites"
ROWS, COLS = (380, 424), (166, 210)          # site-1 window in the MTR3 array (CRISM native frame)
IDX = {
    "OLINDEX2": {"bands": [(1.080, 0.1), (1.210, 0.1), (1.330, 0.4), (1.470, 0.4)], "cont": (1.750, 2.400), "color": "#1f77b4"},
    "OLINDEX3": {"bands": [(1.210, 0.1), (1.250, 0.1), (1.263, 0.2), (1.276, 0.2), (1.330, 0.4)], "cont": (1.750, 1.862), "color": "#d62728"},
}


def kernel_med(spec, wav, target, width=7):
    i = int(np.abs(wav - target).argmin()); hw = width // 2
    return np.nanmedian(spec[max(0, i - hw):i + hw + 1])


def index_value(spec, wav, d):
    c0, c1 = d["cont"]
    r0, r1 = kernel_med(spec, wav, c0), kernel_med(spec, wav, c1)
    slope = (r1 - r0) / (c1 - c0)
    val = 0.0
    for w, wt in d["bands"]:
        rc = r0 + slope * (w - c0); val += wt * (rc - kernel_med(spec, wav, w)) / rc
    return val


def main() -> None:
    t0 = datetime.now(timezone.utc)
    cube, wav, meta = load_mtr3(HDR)
    fz = np.load(FR); f_ol, f_cl = fz["oliv"], fz["clay"]
    rs, cs = slice(*ROWS), slice(*COLS)
    sub = cube[:, rs, cs]; ol = f_ol[rs, cs] >= 0.9; cl = f_cl[rs, cs] >= 0.9
    good = np.isfinite(sub[np.abs(wav - 1.33).argmin()])
    ol &= good; cl &= good
    del cube
    spec = {"olivine unit": np.nanmedian(sub[:, ol], axis=1), "clay unit": np.nanmedian(sub[:, cl], axis=1)}
    n = {"olivine unit": int(ol.sum()), "clay unit": int(cl.sum())}
    per_pixel = {k: {u: np.array([index_value(sub[:, r, c], wav, d) for r, c in zip(*np.nonzero(m))])
                     for u, m in (("olivine unit", ol), ("clay unit", cl))} for k, d in IDX.items()}

    sel = (wav >= 0.9) & (wav <= 2.6)
    fig, axes = plt.subplots(2, 1, figsize=(11, 10), sharex=True)
    for ax, (name, d) in zip(axes, IDX.items()):
        for u, col in (("olivine unit", "#d62728"), ("clay unit", "#1f77b4")):
            s = spec[u]
            ax.plot(wav[sel], s[sel], color=col, lw=1.6, label=f"{u} median (n = {n[u]} pixels)")
            c0, c1 = d["cont"]
            r0, r1 = kernel_med(s, wav, c0), kernel_med(s, wav, c1)
            ax.plot([c0, c1], [r0, r1], color=col, lw=1.0, ls="--", alpha=0.9)
            # extrapolated continuum back to the sampled bands
            slope = (r1 - r0) / (c1 - c0)
            wmin = min(w for w, _ in d["bands"])
            ax.plot([wmin, c0], [r0 + slope * (wmin - c0), r0], color=col, lw=0.8, ls=":", alpha=0.9)
            for w, wt in d["bands"]:
                rc = r0 + slope * (w - c0); rb = kernel_med(s, wav, w)
                ax.plot([w, w], [rb, rc], color=col, lw=2.5 * wt + 0.5, alpha=0.8)
                ax.plot(w, rb, "o", color=col, ms=4)
            v = per_pixel[name][u]
            ax.text(0.99, 0.05 if u == "clay unit" else 0.12, f"{name} {u}: median {np.nanmedian(v):.3f}, pixel scatter (MAD) {np.nanmedian(np.abs(v - np.nanmedian(v))):.3f}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=col)
        for w, wt in d["bands"]:
            ax.axvline(w, color="0.85", lw=0.6, zorder=0)
        ax.axvspan(d["cont"][0], d["cont"][1], color="0.93", zorder=0)
        ax.set_title(f"{name}: band depths at {', '.join(f'{w:.3g}' for w, _ in d['bands'])} um "
                     f"(weights {'/'.join(f'{wt:g}' for _, wt in d['bands'])}); continuum {d['cont'][0]:.3g}-{d['cont'][1]:.3g} um (grey band)\n"
                     f"vertical bars = weighted depths below the extrapolated continuum (dotted)", fontsize=10)
        ax.set_ylabel("I/F (MTR3)"); ax.grid(True, color="0.95"); ax.legend(loc="upper left", fontsize=9, frameon=False)
    axes[1].set_xlabel("wavelength (um)")
    fig.suptitle(f"Anatomy of the two olivine indices on the site-1 window spectra ({CUBE} MTR3, map-pure unit pixels)", fontsize=12)
    fig.tight_layout()
    safe_savefig(fig, OUT / f"fig_olindex_anatomy_{CUBE}", bbox_inches="tight")
    stats = {k: {u: {"median": float(np.nanmedian(v)), "mad": float(np.nanmedian(np.abs(v - np.nanmedian(v)))), "n": int(v.size)}
                 for u, v in per_pixel[k].items()} for k in IDX}
    for k in IDX:
        o, c = stats[k]["olivine unit"], stats[k]["clay unit"]
        stats[k]["contrast_olivine_minus_clay"] = o["median"] - c["median"]
        stats[k]["contrast_over_scatter"] = (o["median"] - c["median"]) / max(0.5 * (o["mad"] + c["mad"]), 1e-9)
        print(f"{k}: olivine {o['median']:.3f} (MAD {o['mad']:.3f}), clay {c['median']:.3f} (MAD {c['mad']:.3f}), "
              f"contrast {stats[k]['contrast_olivine_minus_clay']:+.3f} = {stats[k]['contrast_over_scatter']:.1f} x scatter")
    log = {"script": Path(__file__).name, "script_md5": md5(Path(__file__)), "cube": CUBE, "window_rows": ROWS, "window_cols": COLS,
           "stats": stats, "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat()}
    (OUT / "run_log_olindex_anatomy.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
