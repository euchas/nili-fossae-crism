"""1.4 um band centres on Plebani bland-ratioed spectra (all four cubes).

The unratioed MTR3 I/F is dominated in this window by volcano-scan CO2
residuals (~1.441 / ~1.477 um, present in every unit). The crism_ml
bland-pixel COLUMN ratio divides out atmosphere shared within a column,
so this chain can search the full 1.360-1.490 um window - including the
prehnite 1.48 um diagnostic - legitimately. Same fit and projection
machinery as contact_band_center_plebani.py. Depth tiers 0.008/0.003
(ratioed-chain bands are shallower; logged). Deterministic.

Usage:  python code/contact_band_center_1400_plebani.py
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
from scipy.stats import gaussian_kde

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import CUBE_IDS  # noqa: E402
from contact_distance_to_contact import PALETTE, load_cube, masks_from_labels, md5  # noqa: E402
from contact_band_center_map import DIST_EDGES, DIST_LABELS, unit_masks  # noqa: E402
from contact_carbonate_patches import CUBES, GEOMAP, FRAME, LONLAT, load_ddr_lonlat  # noqa: E402
from contact_band_center_plebani import ZRAT, rasterize_median  # noqa: E402
from figstyle import draw_units, label_panels  # noqa: E402

OUT = ROOT / "outputs" / "revision" / "band_center_map" / "band1400" / "plebani"
OUT.mkdir(parents=True, exist_ok=True)

LEFT_ANCHOR = (1.316, 1.342)
RIGHT_ANCHOR = (1.505, 1.535)
SEARCH = (1.360, 1.490)
STRONG_DEPTH = 0.008
WEAK_DEPTH = 0.003
PREHNITE_MIN = 1.460
CMAP_RANGE = (1.375, 1.485)
PHASE_TICKS = [(1.385, "1.385 serp./talc"), (1.410, "1.410 smectite H2O"),
               (1.430, "1.430 nontronite"), (1.480, "1.480 prehnite")]
KDE_MAX_N = 50_000
RNG = np.random.default_rng(42)


def fit_rows(wav: np.ndarray, spec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    la = (wav >= LEFT_ANCHOR[0]) & (wav <= LEFT_ANCHOR[1])
    ra = (wav >= RIGHT_ANCHOR[0]) & (wav <= RIGHT_ANCHOR[1])
    rl, rr = np.nanmean(spec[:, la], axis=1), np.nanmean(spec[:, ra], axis=1)
    wl, wr = wav[la].mean(), wav[ra].mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        cr = spec / (rl[:, None] + (rr - rl)[:, None] * (wav[None, :] - wl) / (wr - wl))
    sw = np.where((wav >= SEARCH[0]) & (wav <= SEARCH[1]))[0]
    sub = cr[:, sw]
    ok = np.isfinite(sub).all(axis=1) & (np.abs(sub) > 1e-8).any(axis=1)
    center = np.full(spec.shape[0], np.nan, np.float32)
    depth = np.full(spec.shape[0], np.nan, np.float32)
    edge = np.zeros(spec.shape[0], bool)
    if not ok.any():
        return center, depth, edge
    s = sub[ok]
    i = np.argmin(s, axis=1)
    edge[ok] = (i == 0) | (i == s.shape[1] - 1)
    depth[ok] = (1.0 - s[np.arange(s.shape[0]), i]).astype(np.float32)
    ic = np.clip(i, 1, s.shape[1] - 2)
    y0 = s[np.arange(len(ic)), ic - 1]; y1 = s[np.arange(len(ic)), ic]; y2 = s[np.arange(len(ic)), ic + 1]
    x0, x1 = wav[sw[ic - 1]], wav[sw[ic]]
    den = y0 - 2 * y1 + y2
    with np.errstate(divide="ignore", invalid="ignore"):
        shift = 0.5 * (y0 - y2) / den * (x1 - x0)
    shift = np.where(np.isfinite(shift) & (np.abs(shift) <= (x1 - x0)), shift, 0.0)
    center[ok] = (x1 + shift).astype(np.float32)
    return center, depth, edge


def stats_row(v: np.ndarray) -> dict:
    if v.size == 0:
        return {"center_med_um": np.nan, "center_q25": np.nan, "center_q75": np.nan}
    return {"center_med_um": float(np.median(v)), "center_q25": float(np.percentile(v, 25)),
            "center_q75": float(np.percentile(v, 75))}


def main() -> None:
    t0 = datetime.now(timezone.utc)
    offsets = json.loads(FRAME.read_text())["crism"]
    sum_rows, prof_rows, inputs = [], [], {}
    grids = {}
    for cube in CUBE_IDS:
        stem = CUBES[cube]["stem"]
        z = np.load(ZRAT / f"{stem}_ratioed_bland.npz")
        if2 = z["if2"].astype(np.float32)
        wav = z["bands"].astype(np.float64)
        wav = wav / 1000.0 if wav.max() > 100 else wav
        inputs[cube] = {"ratioed_npz": str(ZRAT / f"{stem}_ratioed_bland.npz")}
        cen_s, dep_s, edg_s = fit_rows(wav, if2)
        del if2, z

        c = load_cube(cube)
        m = masks_from_labels(c["labels"])
        units, d, fr = unit_masks(cube, m)
        lon, lat = load_ddr_lonlat(CUBES[cube]["ddr"])
        lon, lat = lon.ravel(), lat.ravel()
        fz = np.load(GEOMAP / f"{cube}_fractions.npz")
        tr = Affine.from_gdal(*fz["transform"])
        with rasterio.open(sorted((ROOT / "outputs" / "_v8_mtr3_with_brown" / cube / "geotiffs").glob("*.tif"))[0]) as ds:
            crs = ds.crs
        o = offsets[cube]
        okp = np.isfinite(lon) & np.isfinite(lat)
        xs = np.full(lon.shape, np.nan); ys = np.full(lon.shape, np.nan)
        X, Y = rtransform(LONLAT, crs, lon[okp].astype(float), lat[okp].astype(float))
        xs[okp] = np.asarray(X) + o["dx_m"]; ys[okp] = np.asarray(Y) + o["dy_m"]
        colf, rowf = ~tr * (xs, ys)
        gc = np.floor(colf); gr = np.floor(rowf)
        H, W = c["labels"].shape
        inside = okp & (gr >= 0) & (gr < H) & (gc >= 0) & (gc < W)
        gri = np.where(inside, gr, 0).astype(int); gci = np.where(inside, gc, 0).astype(int)
        samp = lambda a: np.where(inside, a[gri, gci], np.nan)
        f_oliv, f_clay, f_cap = samp(fr["oliv"]), samp(fr["clay"]), samp(fr["cap"])
        pure = np.where(inside, units["olivine map-pure"][gri, gci], False)
        dclay, dcap = samp(d["clay"]), samp(d["cap"])
        mvalid = np.where(inside, m["valid"][gri, gci], False)

        good = np.isfinite(cen_s) & inside & mvalid
        strong = good & (dep_s >= STRONG_DEPTH) & ~edg_s
        weak = good & (dep_s >= WEAK_DEPTH) & (dep_s < STRONG_DEPTH)
        for tier, tm in (("strong", strong), ("weak", weak)):
            r = rasterize_median(cen_s[tm], gr[tm].astype(int), gc[tm].astype(int), (H, W))
            np.save(OUT / f"{cube}_bc140P_center_{tier}.npy", r)
        grids[cube] = {"fr": fr, "valid": m["valid"], "units": units, "wav": wav}

        pct = {u: {f"p{p}": float(np.nanpercentile(dep_s[um], p)) for p in (50, 90)}
               for u, um in (("olivine", good & (f_oliv >= 0.9)), ("clay", good & (f_clay >= 0.9)),
                             ("cap", good & (f_cap >= 0.9)))}
        prtail = int((strong & (f_oliv >= 0.5) & (cen_s >= PREHNITE_MIN)).sum())
        print(cube, f"good={int(good.sum())} strong={int(strong.sum())} weak={int(weak.sum())} "
              f"prehnite_tail_oliv_strong={prtail}", "depth_pct:", json.dumps(pct))

        unit_defs = {"olivine map-pure": pure & good, "clay map-pure": good & (f_clay >= 0.9),
                     "cap map-pure": good & (f_cap >= 0.9)}
        for uname, um in unit_defs.items():
            for tier, tm in (("strong", strong), ("weak", weak)):
                v = cen_s[um & tm]
                sum_rows.append({"cube": cube, "unit": uname, "tier": tier, "n": int(v.size),
                                 "n_ge_1460": int((v >= PREHNITE_MIN).sum()), **stats_row(v)})
        sel = pure & strong & (dcap > 90)
        for (lo, hi), lab in zip(zip(DIST_EDGES[:-1], DIST_EDGES[1:]), DIST_LABELS):
            bm = sel & (dclay > lo) & (dclay <= hi)
            v = cen_s[bm]
            prof_rows.append({"cube": cube, "dist_bin_m": lab, "n": int(v.size), **stats_row(v)})
        grids[cube]["hist"] = {u: cen_s[um & strong] for u, um in unit_defs.items()}

    sdf = pd.DataFrame(sum_rows); sdf.to_csv(OUT / "summary.csv", index=False)
    pdf_ = pd.DataFrame(prof_rows); pdf_.to_csv(OUT / "profile.csv", index=False)
    print(sdf.round(4).to_string()); print(pdf_.round(4).to_string())

    for cube in CUBE_IDS:
        g = grids[cube]
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharex=True, sharey=True)
        for ax, tier in zip(axes, ("strong", "weak")):
            r = np.load(OUT / f"{cube}_bc140P_center_{tier}.npy")
            bg = np.where(g["valid"], 0.92, np.nan)
            ax.imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            im = ax.imshow(r, cmap="viridis", vmin=CMAP_RANGE[0], vmax=CMAP_RANGE[1],
                           interpolation="nearest")
            draw_units(ax, g["fr"], units=("clay", "cap"))
            ax.set_title(f"{cube} - {tier} tier", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
        cbar.set_ticks([t for t, _ in PHASE_TICKS])
        cbar.set_ticklabels([lab for _, lab in PHASE_TICKS]); cbar.ax.tick_params(labelsize=8)
        fig.suptitle(f"{cube}: 1.4 um band centre, PLEBANI BLAND-RATIOED chain "
                     f"(strong >= {STRONG_DEPTH}, weak {WEAK_DEPTH}-{STRONG_DEPTH})", fontsize=11)
        label_panels(fig)
        fig.savefig(OUT / f"fig_map_{cube}.png", dpi=200, bbox_inches="tight")
        fig.savefig(OUT / f"fig_map_{cube}.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
    xs = np.arange(len(DIST_LABELS))
    for ax, cube in zip(axes.ravel(), CUBE_IDS):
        sub = pdf_[pdf_.cube == cube]
        ax.plot(xs, sub.center_med_um, "-o", color=PALETTE[cube], lw=2.0, ms=5,
                label=f"strong (n={int(sub.n.sum())})")
        ax.fill_between(xs, sub.center_q25.to_numpy(), sub.center_q75.to_numpy(),
                        color=PALETTE[cube], alpha=0.15, linewidth=0)
        for x, (nv, mv) in enumerate(zip(sub.n, sub.center_med_um)):
            if np.isfinite(mv):
                ax.annotate(f"n={int(nv)}", (x, mv), textcoords="offset points",
                            xytext=(0, 8), fontsize=7, ha="center", color="0.25")
        for yv, _ in PHASE_TICKS:
            ax.axhline(yv, color="0.92", lw=0.8, zorder=0)
        ax.set_title(cube, fontsize=10, color=PALETTE[cube])
        ax.set_xticks(xs); ax.set_xticklabels(DIST_LABELS)
        ax.grid(True, color="0.94"); ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
    for ax in axes[1]:
        ax.set_xlabel("distance from mapped clay contact (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("1.4 um band centre (um)")
    fig.suptitle("Map-pure olivine, strong tier: 1.4 um centre vs distance (Plebani chain)", fontsize=11)
    fig.tight_layout()
    label_panels(fig)
    fig.savefig(OUT / "fig_center_vs_distance.png", dpi=200)
    fig.savefig(OUT / "fig_center_vs_distance.pdf")
    plt.close(fig)

    wav0 = grids[CUBE_IDS[0]]["wav"]
    chan = wav0[(wav0 >= SEARCH[0]) & (wav0 <= SEARCH[1])]
    ch_edges = np.concatenate([[chan[0] - np.diff(chan).mean() / 2],
                               0.5 * (chan[1:] + chan[:-1]),
                               [chan[-1] + np.diff(chan).mean() / 2]])
    grid_x = np.linspace(SEARCH[0], SEARCH[1], 400)
    style = {"clay map-pure": "#7f7f7f", "cap map-pure": "#2b2b2b", "olivine map-pure": "#d55e00"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for ax, cube in zip(axes.ravel(), CUBE_IDS):
        for uname, col in style.items():
            v = grids[cube]["hist"][uname]
            v = v[np.isfinite(v)]
            if v.size < 200:
                ax.plot([], [], color=col, label=f"{uname} (n={v.size} < 200)")
                continue
            h, _ = np.histogram(v, bins=ch_edges, density=True)
            ax.stairs(h, ch_edges, color=col, alpha=0.30, fill=True)
            vs = v if v.size <= KDE_MAX_N else RNG.choice(v, KDE_MAX_N, replace=False)
            kde = gaussian_kde(vs, bw_method=0.0025 / vs.std())
            ax.plot(grid_x, kde(grid_x), color=col, lw=1.8, label=f"{uname} (n={v.size})")
        for xv, _ in PHASE_TICKS:
            ax.axvline(xv, color="0.9", lw=0.7, zorder=0)
        ax.set_title(cube, fontsize=10, color=PALETTE[cube])
        ax.grid(True, color="0.96"); ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, frameon=False)
    for ax in axes[1]:
        ax.set_xlabel("1.4 um band centre (um)")
    for ax in axes[:, 0]:
        ax.set_ylabel("density")
    fig.suptitle(f"Plebani chain, strong tier (depth >= {STRONG_DEPTH}): 1.4 um centre by map unit",
                 fontsize=11)
    fig.tight_layout()
    label_panels(fig)
    fig.savefig(OUT / "fig_histograms.png", dpi=200)
    fig.savefig(OUT / "fig_histograms.pdf")
    plt.close(fig)

    log = {"script": Path(__file__).name, "script_md5": md5(Path(__file__)),
           "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
           "left_anchor": LEFT_ANCHOR, "right_anchor": RIGHT_ANCHOR, "search_um": SEARCH,
           "strong_depth": STRONG_DEPTH, "weak_depth": WEAK_DEPTH, "prehnite_min_um": PREHNITE_MIN,
           "tier_note": "0.008/0.003: ratioed-chain bands are shallower (unit-shared component divided out)",
           "chain": "TRR3 sensor space, crism_ml bland-pixel column ratio (cached if2), no despike",
           "deterministic": True, "inputs": inputs}
    (OUT / "run_log_band1400_plebani.json").write_text(json.dumps(log, indent=2, default=str))
    print("done ->", OUT)


if __name__ == "__main__":
    main()
