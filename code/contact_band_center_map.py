"""Tier 0 per-pixel 2.3 um band-centre MAP for all four cubes.

Extends contact_band_center_2300.py (which saved only population
histograms): computes the continuum-removed 2.3 um band centre for every
valid pixel, saves the full-grid centre/depth/edge-flag arrays, and splits
the display into a strong-band tier (depth >= 0.015, single-pixel centres
good to a few nm) and a weak-band tier (0.005-0.015, noise-dominated,
context only). Distances use the geologic-map contact on the CTX-corrected
frame (fractions npz), unlike the old script's GMM-class contact on the
unshifted frame. Search window widened to 2.355 um to reduce clipping of
chlorite-range minima; argmin at a search-window edge is flagged, and
edge-flagged pixels are excluded from the strong tier.

See PLAN_revision_2300_zonation.md (Tier 0). Deterministic - no RNG.

Usage:  python code/contact_band_center_map.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import CUBE_IDS  # noqa: E402
from contact_distance_to_contact import PALETTE, PIX_M, load_cube, masks_from_labels, md5  # noqa: E402
from contact_band_center_2300 import LEFT_ANCHOR, RIGHT_ANCHOR, read_window  # noqa: E402
from figstyle import draw_units, label_panels  # noqa: E402

GEOMAP = ROOT / "outputs" / "revision" / "contact_3d" / "geomap" / "frame_ctx"
OUT = ROOT / "outputs" / "revision" / "band_center_map"
OUT.mkdir(parents=True, exist_ok=True)

SEARCH = (2.265, 2.355)   # widened from 2.350 in contact_band_center_2300.py
STRONG_DEPTH = 0.015      # tier where single-pixel centres are trustworthy
WEAK_DEPTH = 0.005        # below STRONG: displayed for context, noise-dominated
EDGE_EXCL_M = 90.0
OTHER_CONTACT_EXCL_M = 90.0
DIST_EDGES = np.array([36.1, 90, 150, 250, 400, np.inf])
DIST_LABELS = ["36-90", "90-150", "150-250", "250-400", ">400"]
CMAP_RANGE = (2.285, 2.345)
PHASE_TICKS = [(2.287, "2.287 nontronite"), (2.300, "2.300 Fe/Mg smect."),
               (2.312, "2.312 sap./talc"), (2.320, "2.320 serpentine"),
               (2.330, "2.330 Mg-chlorite"), (2.345, "2.345")]


def band_center_grid(wav: np.ndarray, block: np.ndarray, valid: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full-grid continuum-removed 2.3 um minimum, depth, and edge flag."""
    H, W = valid.shape
    center = np.full((H, W), np.nan, np.float32)
    depth = np.full((H, W), np.nan, np.float32)
    edge = np.zeros((H, W), bool)
    rr, cc = np.nonzero(valid)
    spec = block[:, valid].T                                # (n, nw)
    la = (wav >= LEFT_ANCHOR[0]) & (wav <= LEFT_ANCHOR[1])
    ra = (wav >= RIGHT_ANCHOR[0]) & (wav <= RIGHT_ANCHOR[1])
    rl, rgt = np.nanmean(spec[:, la], axis=1), np.nanmean(spec[:, ra], axis=1)
    wl, wr = wav[la].mean(), wav[ra].mean()
    cont = rl[:, None] + (rgt - rl)[:, None] * (wav[None, :] - wl) / (wr - wl)
    with np.errstate(divide="ignore", invalid="ignore"):
        cr = spec / cont
    sw = np.where((wav >= SEARCH[0]) & (wav <= SEARCH[1]))[0]
    sub = cr[:, sw]
    finite = np.isfinite(sub)
    ok = finite.all(axis=1) & np.isfinite(rl) & np.isfinite(rgt)
    sub_ok = sub[ok]
    i = np.argmin(sub_ok, axis=1)
    on_edge = (i == 0) | (i == sub_ok.shape[1] - 1)
    dep = 1.0 - sub_ok[np.arange(sub_ok.shape[0]), i]
    ic = np.clip(i, 1, sub_ok.shape[1] - 2)
    y0 = sub_ok[np.arange(len(ic)), ic - 1]
    y1 = sub_ok[np.arange(len(ic)), ic]
    y2 = sub_ok[np.arange(len(ic)), ic + 1]
    x0, x1 = wav[sw[ic - 1]], wav[sw[ic]]
    denom = y0 - 2 * y1 + y2
    with np.errstate(divide="ignore", invalid="ignore"):
        shift = 0.5 * (y0 - y2) / denom * (x1 - x0)
    shift = np.where(np.isfinite(shift) & (np.abs(shift) <= (x1 - x0)), shift, 0.0)
    center[rr[ok], cc[ok]] = (x1 + shift).astype(np.float32)
    depth[rr[ok], cc[ok]] = dep.astype(np.float32)
    edge[rr[ok], cc[ok]] = on_edge
    return center, depth, edge


def unit_masks(cube: str, m: dict) -> tuple[dict, dict, np.ndarray]:
    """Map-based unit masks and distances on the CTX-corrected frame."""
    fz = np.load(GEOMAP / f"{cube}_fractions.npz")
    fr = {k: fz[k] for k in ("clay", "cap", "oliv", "outline")}
    mapped = fr["outline"] >= 0.99
    d = {"clay": distance_transform_edt(~(fr["clay"] >= 0.5), sampling=(PIX_M, PIX_M)),
         "cap": distance_transform_edt(~(fr["cap"] >= 0.5), sampling=(PIX_M, PIX_M)),
         "edge": distance_transform_edt(m["valid"] & mapped, sampling=(PIX_M, PIX_M))}
    units = {
        "olivine map-pure": (m["valid"] & mapped & (fr["oliv"] >= 0.9) & (fr["clay"] == 0)
                             & ~m["clay"] & (d["edge"] > EDGE_EXCL_M)),
        "clay map-pure": m["valid"] & mapped & (fr["clay"] >= 0.9),
        "cap map-pure": m["valid"] & mapped & (fr["cap"] >= 0.9),
    }
    return units, d, fr


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
        wav, block, img = read_window(cube)
        inputs[cube] = {"mtr3_img": str(img.relative_to(ROOT)), "mtr3_size_bytes": img.stat().st_size,
                        "labels_md5": md5(c["files"]["labels"]),
                        "fractions_npz": str((GEOMAP / f"{cube}_fractions.npz").relative_to(ROOT)),
                        "fractions_md5": md5(GEOMAP / f"{cube}_fractions.npz")}
        center, depth, edgef = band_center_grid(wav, block, m["valid"])
        np.save(OUT / f"{cube}_bc230_center.npy", center)
        np.save(OUT / f"{cube}_bc230_depth.npy", depth)
        np.save(OUT / f"{cube}_bc230_edgeflag.npy", edgef)
        strong = np.isfinite(center) & (depth >= STRONG_DEPTH) & ~edgef
        weak = np.isfinite(center) & (depth >= WEAK_DEPTH) & (depth < STRONG_DEPTH)
        grids[cube] = {"center": center, "strong": strong, "weak": weak, "fr": fr,
                       "units": units, "d": d}
        for uname, um in units.items():
            n = int(um.sum())
            for tier, tm in (("strong", strong), ("weak", weak)):
                v = center[um & tm]
                sum_rows.append({"cube": cube, "unit": uname, "tier": tier, "n_unit": n,
                                 "n_tier": int(v.size), "frac_tier": float(v.size / n) if n else np.nan,
                                 **stats_row(v)})
        sel = units["olivine map-pure"] & (d["cap"] > OTHER_CONTACT_EXCL_M)
        dist = d["clay"]
        for tier, tm in (("strong", strong), ("weak", weak)):
            for (lo, hi), lab in zip(zip(DIST_EDGES[:-1], DIST_EDGES[1:]), DIST_LABELS):
                bm = sel & tm & (dist > lo) & (dist <= hi)
                v = center[bm]
                prof_rows.append({"cube": cube, "tier": tier, "dist_bin_m": lab, "n": int(v.size),
                                  **stats_row(v)})
        print(cube, f"valid={int(m['valid'].sum())} strong={int(strong.sum())} weak={int(weak.sum())}",
              f"strong_in_pure_oliv={int((units['olivine map-pure'] & strong).sum())}")

    sdf = pd.DataFrame(sum_rows)
    sdf.to_csv(OUT / "band_center_map_summary.csv", index=False)
    pdf_ = pd.DataFrame(prof_rows)
    pdf_.to_csv(OUT / "band_center_map_profile.csv", index=False)
    print(sdf.round(4).to_string())
    print(pdf_.round(4).to_string())

    # Figure 1 per cube: strong- and weak-tier centre maps with mapped contacts
    for cube in CUBE_IDS:
        g = grids[cube]
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharex=True, sharey=True)
        for ax, tier, title in zip(axes, ("strong", "weak"),
                                   (f"strong band (depth >= {STRONG_DEPTH})",
                                    f"weak band ({WEAK_DEPTH}-{STRONG_DEPTH}, noise-dominated)")):
            bg = np.where(np.isfinite(g["center"]), 0.92, np.nan)
            ax.imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            shown = np.where(g[tier], g["center"], np.nan)
            im = ax.imshow(shown, cmap="viridis", vmin=CMAP_RANGE[0], vmax=CMAP_RANGE[1],
                           interpolation="nearest")
            draw_units(ax, g["fr"], units=("clay", "cap"))
            ax.set_title(f"{cube} - {title}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
        cbar.set_ticks([t for t, _ in PHASE_TICKS])
        cbar.set_ticklabels([lab for _, lab in PHASE_TICKS])
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("2.3 um band centre (um)", fontsize=9)
        fig.suptitle(f"{cube}: 2.3 um band-centre map (blue = phyllosilicate basement, "
                     "dashed = mafic cap)", fontsize=11)
        label_panels(fig)
        fig.savefig(OUT / f"fig_map_{cube}.png", dpi=200, bbox_inches="tight")
        fig.savefig(OUT / f"fig_map_{cube}.pdf", bbox_inches="tight")
        plt.close(fig)

    # Figure 2: strong-tier centre vs distance from the clay contact, pure olivine
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
    xs = np.arange(len(DIST_LABELS))
    for ax, cube in zip(axes.ravel(), CUBE_IDS):
        for tier, col, lw in (("weak", "0.75", 1.2), ("strong", PALETTE[cube], 2.0)):
            sub = pdf_[(pdf_.cube == cube) & (pdf_.tier == tier)]
            med = sub.center_med_um.to_numpy()
            q25, q75 = sub.center_q25.to_numpy(), sub.center_q75.to_numpy()
            lbl = f"{tier} (n={int(sub.n.sum())})" + (" - noise-dominated" if tier == "weak" else "")
            ax.plot(xs, med, "-o", color=col, lw=lw, ms=5, label=lbl)
            ax.fill_between(xs, q25, q75, color=col, alpha=0.15, linewidth=0)
            if tier == "strong":
                for x, (nv, mv) in enumerate(zip(sub.n, med)):
                    if np.isfinite(mv):
                        ax.annotate(f"n={int(nv)}", (x, mv), textcoords="offset points",
                                    xytext=(0, 8), fontsize=7, ha="center", color="0.25")
        for yv in (2.300, 2.320, 2.330):
            ax.axhline(yv, color="0.85", lw=0.8, zorder=0)
        ax.set_title(cube, fontsize=10, color=PALETTE[cube])
        ax.set_xticks(xs); ax.set_xticklabels(DIST_LABELS)
        ax.grid(True, color="0.94"); ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
    for ax in axes[1]:
        ax.set_xlabel("distance from mapped clay contact (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("2.3 um band centre (um)")
    fig.suptitle("Map-pure olivine unit: 2.3 um band centre vs distance from the clay contact\n"
                 "(median with IQR band; weak tier shown for context only)", fontsize=11)
    fig.tight_layout()
    label_panels(fig)
    fig.savefig(OUT / "fig_center_vs_distance.png", dpi=200)
    fig.savefig(OUT / "fig_center_vs_distance.pdf")
    plt.close(fig)

    # Figure 3: strong-tier histograms per unit
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    style = {"clay map-pure": ("#7f7f7f", "-"), "cap map-pure": ("#2b2b2b", ":"),
             "olivine map-pure": ("#d55e00", "-")}
    edges = np.arange(SEARCH[0], SEARCH[1] + 1e-9, 0.0033)
    for ax, cube in zip(axes.ravel(), CUBE_IDS):
        g = grids[cube]
        for uname, (col, ls) in style.items():
            v = g["center"][g["units"][uname] & g["strong"]]
            if v.size < 20:
                ax.plot([], [], color=col, ls=ls, label=f"{uname} (n={v.size} < 20)")
                continue
            h, _ = np.histogram(v, bins=edges, density=True)
            ax.plot(0.5 * (edges[1:] + edges[:-1]), h, color=col, ls=ls, lw=1.8,
                    label=f"{uname} (n={v.size})")
        for xv in (2.300, 2.320, 2.330):
            ax.axvline(xv, color="0.85", lw=0.8, zorder=0)
        ax.set_title(cube, fontsize=10, color=PALETTE[cube])
        ax.grid(True, color="0.94"); ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, frameon=False)
    for ax in axes[1]:
        ax.set_xlabel("2.3 um band centre (um)")
    for ax in axes[:, 0]:
        ax.set_ylabel("density")
    fig.suptitle(f"Strong-band tier (depth >= {STRONG_DEPTH}): band-centre distributions by map unit",
                 fontsize=11)
    fig.tight_layout()
    label_panels(fig)
    fig.savefig(OUT / "fig_histograms_strong.png", dpi=200)
    fig.savefig(OUT / "fig_histograms_strong.pdf")
    plt.close(fig)

    log = {"script": Path(__file__).name, "script_md5": md5(Path(__file__)),
           "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
           "left_anchor": LEFT_ANCHOR, "right_anchor": RIGHT_ANCHOR, "search_um": SEARCH,
           "search_note": "widened from 2.350 (contact_band_center_2300.py) to 2.355; edge-flagged "
                          "argmin excluded from strong tier",
           "strong_depth": STRONG_DEPTH, "weak_depth": WEAK_DEPTH,
           "edge_excl_m": EDGE_EXCL_M, "other_contact_excl_m": OTHER_CONTACT_EXCL_M,
           "dist_edges_m": list(DIST_EDGES[:-1]) + ["inf"], "contact_source": "geologic map, CTX-corrected frame",
           "deterministic": True, "inputs": inputs}
    (OUT / "run_log_band_center_map.json").write_text(json.dumps(log, indent=2, default=str))
    print("done ->", OUT)


if __name__ == "__main__":
    main()
