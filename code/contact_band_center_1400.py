"""Per-pixel 1.4 um band-centre maps for all four cubes (OH-overtone region).

Mirror of contact_band_center_map.py for the 1.36-1.49 um band-minimum
region: serpentine/talc (sharp ~1.385-1.39 um), smectite H2O/OH
(1.41-1.43 um), nontronite (~1.43 um), prehnite (1.48 um - the
temperature-diagnostic target). Same continuum+quadratic fit and edge
flagging; search window includes 1.48 um. Depth tiers 0.010/0.004
(the 1.4 um band is ~2-3x shallower than 2.3 um; choice logged).
MTR3 chain, map-based contact on the CTX-corrected frame. Deterministic.

Caveat carried to all outputs: the 1.4 um region can hold volcano-scan /
atmospheric residuals; interpret unit CONTRASTS, not absolute centres.

Usage:  python code/contact_band_center_1400.py
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import CUBE_IDS  # noqa: E402
from contact_distance_to_contact import PALETTE, load_cube, masks_from_labels, md5  # noqa: E402
from contact_band_center_map import DIST_EDGES, DIST_LABELS, unit_masks  # noqa: E402
from figstyle import draw_units, label_panels  # noqa: E402

OUT = ROOT / "outputs" / "revision" / "band_center_map" / "band1400"
OUT.mkdir(parents=True, exist_ok=True)

WIN_UM = (1.28, 1.56)
LEFT_ANCHOR = (1.316, 1.342)
RIGHT_ANCHOR = (1.505, 1.535)
# Search restricted to the mineral OH region: the volcano-scan CO2-triplet
# residuals at ~1.441 and ~1.477 um dominate the unratioed MTR3 I/F in every
# unit (first run, 2026-09-04) and would swamp any mineral minimum beyond
# ~1.43 um. Prehnite (1.48 um) is therefore NOT testable in this chain.
SEARCH = (1.360, 1.425)
STRONG_DEPTH = 0.010
WEAK_DEPTH = 0.004
PREHNITE_MIN = 1.418           # long-tail counter (nontronite-range and beyond)
CMAP_RANGE = (1.375, 1.432)
PHASE_TICKS = [(1.385, "1.385 serp./talc"), (1.410, "1.410 smectite H2O"),
               (1.425, "1.425 (window edge)")]
KDE_MAX_N = 50_000
RNG = np.random.default_rng(42)


def read_window(cube: str) -> tuple[np.ndarray, np.ndarray]:
    low = cube.lower()
    hdr = ROOT / "cubes" / f"{low}_07_if166j_mtr3" / f"{low}_07_if166j_mtr3.hdr"
    txt = hdr.read_text()
    get = lambda k: int(re.search(rf"^{k}\s*=\s*(\d+)", txt, re.M | re.I).group(1))
    ns, nl, nb = get("samples"), get("lines"), get("bands")
    wav = np.array([float(x) for x in re.search(r"wavelength\s*=\s*\{([^}]*)\}", txt, re.S | re.I).group(1).split(",")])
    wav_um = wav / 1000.0 if wav.max() > 100 else wav
    sel = np.where((wav_um >= WIN_UM[0]) & (wav_um <= WIN_UM[1]))[0]
    mm = np.memmap(hdr.with_suffix(".img"), dtype="<f4", mode="r", shape=(nb, nl, ns))
    block = np.array(mm[sel.min():sel.max() + 1]).astype(np.float32)
    block[block >= 65535.0] = np.nan
    return wav_um[sel], block


def band_center_grid(wav: np.ndarray, block: np.ndarray, valid: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    H, W = valid.shape
    center = np.full((H, W), np.nan, np.float32)
    depth = np.full((H, W), np.nan, np.float32)
    edge = np.zeros((H, W), bool)
    rr, cc = np.nonzero(valid)
    spec = block[:, valid].T
    la = (wav >= LEFT_ANCHOR[0]) & (wav <= LEFT_ANCHOR[1])
    ra = (wav >= RIGHT_ANCHOR[0]) & (wav <= RIGHT_ANCHOR[1])
    rl, rgt = np.nanmean(spec[:, la], axis=1), np.nanmean(spec[:, ra], axis=1)
    wl, wr = wav[la].mean(), wav[ra].mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        cr = spec / (rl[:, None] + (rgt - rl)[:, None] * (wav[None, :] - wl) / (wr - wl))
    sw = np.where((wav >= SEARCH[0]) & (wav <= SEARCH[1]))[0]
    sub = cr[:, sw]
    ok = np.isfinite(sub).all(axis=1) & np.isfinite(rl) & np.isfinite(rgt)
    sub_ok = sub[ok]
    i = np.argmin(sub_ok, axis=1)
    on_edge = (i == 0) | (i == sub_ok.shape[1] - 1)
    dep = 1.0 - sub_ok[np.arange(sub_ok.shape[0]), i]
    ic = np.clip(i, 1, sub_ok.shape[1] - 2)
    y0 = sub_ok[np.arange(len(ic)), ic - 1]
    y1 = sub_ok[np.arange(len(ic)), ic]
    y2 = sub_ok[np.arange(len(ic)), ic + 1]
    x0, x1 = wav[sw[ic - 1]], wav[sw[ic]]
    den = y0 - 2 * y1 + y2
    with np.errstate(divide="ignore", invalid="ignore"):
        shift = 0.5 * (y0 - y2) / den * (x1 - x0)
    shift = np.where(np.isfinite(shift) & (np.abs(shift) <= (x1 - x0)), shift, 0.0)
    center[rr[ok], cc[ok]] = (x1 + shift).astype(np.float32)
    depth[rr[ok], cc[ok]] = dep.astype(np.float32)
    edge[rr[ok], cc[ok]] = on_edge
    return center, depth, edge


def stats_row(v: np.ndarray) -> dict:
    if v.size == 0:
        return {"center_med_um": np.nan, "center_q25": np.nan, "center_q75": np.nan}
    return {"center_med_um": float(np.median(v)), "center_q25": float(np.percentile(v, 25)),
            "center_q75": float(np.percentile(v, 75))}


def main() -> None:
    t0 = datetime.now(timezone.utc)
    sum_rows, prof_rows, inputs = [], [], {}
    grids = {}
    for cube in CUBE_IDS:
        c = load_cube(cube)
        m = masks_from_labels(c["labels"])
        units, d, fr = unit_masks(cube, m)
        wav, block = read_window(cube)
        inputs[cube] = {"labels_md5": md5(c["files"]["labels"])}
        center, depth, edgef = band_center_grid(wav, block, m["valid"])
        np.save(OUT / f"{cube}_bc140_center.npy", center)
        np.save(OUT / f"{cube}_bc140_depth.npy", depth)
        np.save(OUT / f"{cube}_bc140_edgeflag.npy", edgef)
        strong = np.isfinite(center) & (depth >= STRONG_DEPTH) & ~edgef
        weak = np.isfinite(center) & (depth >= WEAK_DEPTH) & (depth < STRONG_DEPTH)
        grids[cube] = {"center": center, "strong": strong, "weak": weak, "fr": fr,
                       "units": units, "d": d, "wav": wav}
        pct = {u: {f"p{p}": float(np.nanpercentile(depth[um], p)) for p in (50, 75, 90, 95)}
               for u, um in units.items() if um.any()}
        prtail = int((strong & (center >= PREHNITE_MIN)).sum())
        print(cube, f"strong={int(strong.sum())} weak={int(weak.sum())} "
              f"prehnite_tail_strong={prtail}", "depth_pct:", json.dumps(pct))
        for uname, um in units.items():
            n = int(um.sum())
            for tier, tm in (("strong", strong), ("weak", weak)):
                v = center[um & tm]
                sum_rows.append({"cube": cube, "unit": uname, "tier": tier, "n_unit": n,
                                 "n_tier": int(v.size),
                                 "frac_tier": float(v.size / n) if n else np.nan,
                                 "n_ge_1460": int((v >= PREHNITE_MIN).sum()),
                                 **stats_row(v)})
        sel = units["olivine map-pure"] & (d["cap"] > 90)
        dist = d["clay"]
        for tier, tm in (("strong", strong), ("weak", weak)):
            for (lo, hi), lab in zip(zip(DIST_EDGES[:-1], DIST_EDGES[1:]), DIST_LABELS):
                bm = sel & tm & (dist > lo) & (dist <= hi)
                v = center[bm]
                prof_rows.append({"cube": cube, "tier": tier, "dist_bin_m": lab,
                                  "n": int(v.size), **stats_row(v)})
        del block

    sdf = pd.DataFrame(sum_rows)
    sdf.to_csv(OUT / "band1400_summary.csv", index=False)
    pdf_ = pd.DataFrame(prof_rows)
    pdf_.to_csv(OUT / "band1400_profile.csv", index=False)
    print(sdf.round(4).to_string())
    print(pdf_.round(4).to_string())

    # Figure 1 per cube: strong/weak maps
    for cube in CUBE_IDS:
        g = grids[cube]
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharex=True, sharey=True)
        for ax, tier, title in zip(axes, ("strong", "weak"),
                                   (f"strong band (depth >= {STRONG_DEPTH})",
                                    f"weak band ({WEAK_DEPTH}-{STRONG_DEPTH}, noise-dominated)")):
            bg = np.where(np.isfinite(g["center"]), 0.92, np.nan)
            ax.imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            im = ax.imshow(np.where(g[tier], g["center"], np.nan), cmap="viridis",
                           vmin=CMAP_RANGE[0], vmax=CMAP_RANGE[1], interpolation="nearest")
            draw_units(ax, g["fr"], units=("clay", "cap"))
            ax.set_title(f"{cube} - {title}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
        cbar.set_ticks([t for t, _ in PHASE_TICKS])
        cbar.set_ticklabels([lab for _, lab in PHASE_TICKS])
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("1.4 um band centre (um)", fontsize=9)
        fig.suptitle(f"{cube}: 1.4 um band-centre map (blue = phyllosilicate basement, "
                     "dashed = mafic cap)", fontsize=11)
        label_panels(fig)
        fig.savefig(OUT / f"fig_map_{cube}.png", dpi=200, bbox_inches="tight")
        fig.savefig(OUT / f"fig_map_{cube}.pdf", bbox_inches="tight")
        plt.close(fig)

    # Figure 2: centre vs distance
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
    xs = np.arange(len(DIST_LABELS))
    for ax, cube in zip(axes.ravel(), CUBE_IDS):
        for tier, col, lw in (("weak", "0.75", 1.2), ("strong", PALETTE[cube], 2.0)):
            sub = pdf_[(pdf_.cube == cube) & (pdf_.tier == tier)]
            med = sub.center_med_um.to_numpy()
            ax.plot(xs, med, "-o", color=col, lw=lw, ms=5,
                    label=f"{tier} (n={int(sub.n.sum())})" + (" - noise-dominated" if tier == "weak" else ""))
            ax.fill_between(xs, sub.center_q25.to_numpy(), sub.center_q75.to_numpy(),
                            color=col, alpha=0.15, linewidth=0)
            if tier == "strong":
                for x, (nv, mv) in enumerate(zip(sub.n, med)):
                    if np.isfinite(mv):
                        ax.annotate(f"n={int(nv)}", (x, mv), textcoords="offset points",
                                    xytext=(0, 8), fontsize=7, ha="center", color="0.25")
        for yv, _ in PHASE_TICKS[:3]:
            ax.axhline(yv, color="0.9", lw=0.8, zorder=0)
        ax.set_title(cube, fontsize=10, color=PALETTE[cube])
        ax.set_xticks(xs); ax.set_xticklabels(DIST_LABELS)
        ax.grid(True, color="0.94"); ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
    for ax in axes[1]:
        ax.set_xlabel("distance from mapped clay contact (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("1.4 um band centre (um)")
    fig.suptitle("Map-pure olivine unit: 1.4 um band centre vs distance from the clay contact",
                 fontsize=11)
    fig.tight_layout()
    label_panels(fig)
    fig.savefig(OUT / "fig_center_vs_distance.png", dpi=200)
    fig.savefig(OUT / "fig_center_vs_distance.pdf")
    plt.close(fig)

    # Figure 3: channel-aligned histograms + KDE, strong tier
    wav0 = grids[CUBE_IDS[0]]["wav"]
    chan = wav0[(wav0 >= SEARCH[0]) & (wav0 <= SEARCH[1])]
    ch_edges = np.concatenate([[chan[0] - np.diff(chan).mean() / 2],
                               0.5 * (chan[1:] + chan[:-1]),
                               [chan[-1] + np.diff(chan).mean() / 2]])
    grid_x = np.linspace(SEARCH[0], SEARCH[1], 400)
    style = {"clay map-pure": "#7f7f7f", "cap map-pure": "#2b2b2b", "olivine map-pure": "#d55e00"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for ax, cube in zip(axes.ravel(), CUBE_IDS):
        g = grids[cube]
        for uname, col in style.items():
            v = g["center"][g["units"][uname] & g["strong"]]
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
    fig.suptitle(f"Strong tier (depth >= {STRONG_DEPTH}): 1.4 um band-centre distributions by map unit",
                 fontsize=11)
    fig.tight_layout()
    label_panels(fig)
    fig.savefig(OUT / "fig_histograms.png", dpi=200)
    fig.savefig(OUT / "fig_histograms.pdf")
    plt.close(fig)

    log = {"script": Path(__file__).name, "script_md5": md5(Path(__file__)),
           "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
           "window_um": WIN_UM, "left_anchor": LEFT_ANCHOR, "right_anchor": RIGHT_ANCHOR,
           "search_um": SEARCH, "strong_depth": STRONG_DEPTH, "weak_depth": WEAK_DEPTH,
           "tier_note": "tiers 0.010/0.004 (vs 0.015/0.005 at 2.3 um): 1.4 um band is 2-3x shallower",
           "prehnite_min_um": PREHNITE_MIN, "contact_source": "geologic map, CTX-corrected frame",
           "caveat": "1.4 um region may carry volcano-scan/atmospheric residuals; interpret unit contrasts",
           "deterministic": True, "inputs": inputs}
    (OUT / "run_log_band1400.json").write_text(json.dumps(log, indent=2, default=str))
    print("done ->", OUT)


if __name__ == "__main__":
    main()
