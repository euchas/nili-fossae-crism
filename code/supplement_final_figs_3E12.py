"""Publication supplement figures from the band-centre campaign, FRT00003E12 only.

FigS_carbonate: stage-6 Plebani MgCO3 carbonate patches (blue circles) and
  serpentinized-olivine patches (red circles; connected components of the
  olivine-unit 2.3-only >= 2.305 um class, >= SERP_MIN_PX pixels) over the
  geologic map drawn geologic-map style (black outlines, line-texture fills
  per unit, no colours). The distance statistics go to the caption only.
FigS_bandcentre: two rows, one column: (a) 1.4 um OH matched-filter detections
  coloured by band centre, (b) 2.3 um metal-OH detections coloured by band
  centre. Units as black line styles (olivine solid, clay dashed,
  cap dotted). All layers on the MTR3 18 m grid in the CTX-corrected frame,
  north up. A4 full-width layout, 300 dpi + PDF. Deterministic.

Usage:  python code/supplement_final_figs_3E12.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from contact_distance_to_contact import load_cube, masks_from_labels, md5, PIX_M  # noqa: E402
from contact_band_center_map import unit_masks  # noqa: E402
from contact_band_center_modes import PATCHES, patch_rowcol  # noqa: E402
from contact_carbonate_patches import CUBES, FRAME, LONLAT, load_carb, load_ddr_lonlat  # noqa: E402
import json as _json
import rasterio
from rasterio.transform import Affine
from rasterio.warp import transform as rtransform
from scipy.ndimage import binary_dilation

CUBE = "FRT00003E12"
BC = ROOT / "outputs" / "revision" / "band_center_map"
OUT = ROOT / "manuscript_v10" / "figures" / "final"
OUT.mkdir(parents=True, exist_ok=True)
FIG_W_IN = 7.3
FS_LETTER, FS_LABEL, FS_TICK = 15, 10, 9
STROKE = [pe.withStroke(linewidth=2.5, foreground="black")]
PATCH_COL = "#1f77b4"          # MgCO3 patches (blue)
SERP_COL = "#d62728"           # serpentinized-olivine patches (red)
SERP_MIN_PX = 5                # components smaller than this are not drawn as circles
PHASE_COL = {1: ("Fe/Mg smectite (1.4 + 2.3 µm)", "#2ca02c"),
             2: ("serpentinized olivine (2.3 µm only)", "#8c564b"),
             3: ("Mg-carbonate (2.3 + 2.5 µm; crism_ml, pixels enlarged)", "#9400D3")}
UNIT_LS = {"oliv": ("solid", "olivine unit"), "clay": ("dashed", "phyllosilicate basement"), "cap": ("dotted", "mafic cap")}
UNIT_HATCH = {"clay": ("////", "phyllosilicate basement"), "cap": ("....", "mafic cap")}
plt.rcParams["hatch.linewidth"] = 0.5


def crop_box(valid: np.ndarray, pad: int = 6) -> tuple[slice, slice]:
    r, c = np.nonzero(valid)
    return slice(max(r.min() - pad, 0), r.max() + pad), slice(max(c.min() - pad, 0), c.max() + pad)


def scalebar(ax, nx, ny, length_m=1000.0, color="white"):
    px = length_m / PIX_M
    x0, y0 = nx * 0.05, ny * 0.95
    ax.plot([x0, x0 + px], [y0, y0], color=color, lw=3.5, solid_capstyle="butt",
            path_effects=[pe.withStroke(linewidth=5.5, foreground="black")] if color == "white" else None)
    ax.text(x0 + px / 2, y0 - ny * 0.012, f"{length_m / 1000:g} km", color=color, ha="center", va="bottom", fontsize=FS_LABEL,
            fontweight="bold", path_effects=STROKE if color == "white" else None)


def north(ax, nx, ny, color="white"):
    ax.annotate("N", xy=(nx * 0.95, ny * 0.05), xytext=(nx * 0.95, ny * 0.14), color=color, ha="center", va="top", fontsize=FS_LABEL,
                fontweight="bold", path_effects=STROKE if color == "white" else None,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6))


def unit_lines(ax, fr, lw=0.9, casing=True):
    mapped = fr["outline"] >= 0.5
    for u, (ls, _) in UNIT_LS.items():
        m = np.where(mapped, (fr[u] >= 0.5).astype(float), np.nan)      # NaN outside the mapped area: no edge lines
        if casing:
            ax.contour(m, levels=[0.5], colors="white", linewidths=lw + 1.2, linestyles=ls, alpha=0.9)
        ax.contour(m, levels=[0.5], colors="black", linewidths=lw, linestyles=ls)


def letter(ax, let, color="white"):
    ax.text(0.02, 0.98, let, transform=ax.transAxes, ha="left", va="top", fontsize=FS_LETTER, fontweight="bold", color=color,
            path_effects=STROKE if color == "white" else None)


def main() -> None:
    t0 = datetime.now(timezone.utc)
    c = load_cube(CUBE); m = masks_from_labels(c["labels"]); units, d, fr = unit_masks(CUBE, m)
    valid = m["valid"]; rs, cs = crop_box(valid)
    V = valid[rs, cs]; FR = {k: v[rs, cs] for k, v in fr.items()}
    ny, nx = V.shape
    log = {"cube": CUBE, "crop_rows": [rs.start, rs.stop], "crop_cols": [cs.start, cs.stop]}

    # ================= FigS_carbonate =================
    patches = pd.read_csv(PATCHES)
    pdfc = patches[(patches.cube == CUBE) & patches.on_map].reset_index(drop=True)
    pr, pc = patch_rowcol(CUBE, pdfc, c["labels"].shape)
    npx = pdfc.n_px.to_numpy()[:len(pr)]
    bg = d["clay"][units["olivine map-pure"] & (d["cap"] > 90)]
    ks = ks_2samp(pdfc.dist_to_clay_contact_m.to_numpy(), bg)
    med_p, med_b = float(pdfc.dist_to_clay_contact_m.median()), float(np.median(bg))
    log["carbonate"] = {"n_patches": int(len(pdfc)), "patch_median_m": med_p, "background_median_m": med_b,
                        "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue), "n_background_px": int(bg.size)}

    fig = plt.figure(figsize=(FIG_W_IN * 0.62, FIG_W_IN * 0.62 * ny / nx + 0.75))
    gs = fig.add_gridspec(1, 1, left=0.01, right=0.99, top=0.99, bottom=0.14)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(np.where(V, 1.0, 0.86), cmap="gray", vmin=0, vmax=1, interpolation="nearest")   # CRISM footprint white, outside grey
    for u, (h, _) in UNIT_HATCH.items():
        ax.contourf(np.where(FR["outline"] >= 0.5, (FR[u] >= 0.5).astype(float), np.nan), levels=[0.5, 1.5], colors="none", hatches=[h])
    unit_lines(ax, FR, lw=0.8, casing=False)
    ax.scatter(pc - cs.start, pr - rs.start, s=8 + 5 * np.sqrt(npx), facecolors="none", edgecolors=PATCH_COL, linewidths=1.1, zorder=6)
    # serpentinized-olivine patches: connected components of the olivine-unit 2.3-only >= 2.305 um class
    from scipy.ndimage import label as cc_label, center_of_mass
    mode_full = np.load(BC / "combinations" / f"{CUBE}_mode_class.npy")
    labs, ncomp = cc_label(mode_full == 2, structure=np.ones((3, 3)))
    sizes = np.bincount(labs.ravel())[1:]
    keep = np.nonzero(sizes >= SERP_MIN_PX)[0] + 1
    cents = center_of_mass(mode_full == 2, labs, keep)
    sr = np.array([c_[0] for c_ in cents]) - rs.start; sc_ = np.array([c_[1] for c_ in cents]) - cs.start
    ax.scatter(sc_, sr, s=8 + 5 * np.sqrt(sizes[keep - 1]), facecolors="none", edgecolors=SERP_COL, linewidths=1.0, zorder=5)
    log["carbonate"]["serpentinized_olivine_patches"] = {"n_components_total": int(ncomp), "n_drawn": int(keep.size), "min_px": SERP_MIN_PX,
                                                        "pixels_drawn": int(sizes[keep - 1].sum()), "pixels_total": int((mode_full == 2).sum())}
    ax.set_xlim(0, nx); ax.set_ylim(ny, 0); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
    scalebar(ax, nx, ny, color="black"); north(ax, nx, ny, color="black")
    handles = [Patch(facecolor="white", edgecolor="black", label="olivine unit"),
               Patch(facecolor="white", edgecolor="black", hatch="////", label="phyllosilicate basement"),
               Patch(facecolor="white", edgecolor="black", hatch="....", label="mafic cap"),
               Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor=PATCH_COL, markersize=7, lw=0, label="MgCO$_3$ (2.3 + 2.5 µm; size ~ area)"),
               Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor=SERP_COL, markersize=7, lw=0, label="serpentinized olivine (2.3 µm only)"),
               Patch(facecolor="#dbdbdb", edgecolor="none", label="outside CRISM footprint")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.01), fontsize=7.5, frameon=False, ncol=2, handlelength=1.6, columnspacing=1.2, labelspacing=0.3)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"FigS_carbonate_FINAL.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    sp = log["carbonate"]["serpentinized_olivine_patches"]
    cap1 = (f"Fig. S-carbonate. Mg-carbonate detections (blue; crism_ml classifier, paired 2.3 + 2.5 µm bands, bland-ratioed TRR3 chain; {len(pdfc)} patches) "
            f"and serpentinized-olivine patches (red; olivine-unit pixels with a 2.3 µm band only, centred at or above 2.305 µm, no 1.9 µm water band in the "
            f"stacked spectrum; {sp['n_drawn']} connected patches of at least {SERP_MIN_PX} pixels, covering {sp['pixels_drawn']} of {sp['pixels_total']} such pixels), "
            f"circle size scaled by patch area, in CRISM {CUBE}, over the geologic map (olivine unit unpatterned, phyllosilicate basement hatched, mafic cap stippled); grey = outside the CRISM footprint; "
            f"north up, 18 m pixels. Patch median distance to the mapped clay contact {med_p:.0f} m vs {med_b:.0f} m for the map-pure olivine-unit background "
            f"(two-sample KS D = {ks.statistic:.2f}, p = {ks.pvalue:.1e}; numbers for the text, not shown).")
    (OUT / "FigS_carbonate_FINAL_caption.txt").write_text(cap1, encoding="utf-8")

    # ================= FigS_bandcentre =================
    oh_c = np.load(BC / "oh_filter" / f"{CUBE}_oh_centre_det.npy")[rs, cs]
    c23 = np.load(BC / "matched_filter" / "2300" / f"{CUBE}_centre_det.npy")[rs, cs]
    combo = np.load(BC / "combinations" / f"{CUBE}_combo_class.npy")[rs, cs]
    mode = np.load(BC / "combinations" / f"{CUBE}_mode_class.npy")[rs, cs]
    # crism_ml Mg-carbonate pixels (2.3 + 2.5 um classifier), sensor grid -> map grid as in stage 6
    carb = load_carb(CUBES[CUBE]["stem"]); lon, lat = load_ddr_lonlat(CUBES[CUBE]["ddr"])
    fz = np.load(ROOT / "outputs" / "revision" / "contact_3d" / "geomap" / "frame_ctx" / f"{CUBE}_fractions.npz")
    trm = Affine.from_gdal(*fz["transform"])
    with rasterio.open(sorted((ROOT / "outputs" / "_v8_mtr3_with_brown" / CUBE / "geotiffs").glob("*.tif"))[0]) as ds:
        crs = ds.crs
    o = _json.loads(FRAME.read_text())["crism"][CUBE]
    sel = carb["region"] & np.isfinite(lon) & np.isfinite(lat)
    X, Y = rtransform(LONLAT, crs, lon[sel].astype(float), lat[sel].astype(float))
    colf, rowf = ~trm * (np.asarray(X) + o["dx_m"], np.asarray(Y) + o["dy_m"])
    carb_map = np.zeros(c["labels"].shape, bool)
    gr, gc = np.floor(rowf).astype(int), np.floor(colf).astype(int)
    okc = (gr >= 0) & (gr < carb_map.shape[0]) & (gc >= 0) & (gc < carb_map.shape[1])
    carb_map[gr[okc], gc[okc]] = True
    carb_map = carb_map[rs, cs]
    phase = np.zeros(combo.shape, np.int8)
    phase[combo == 3] = 1                                    # Fe/Mg smectite: 1.4 + 2.3 um
    phase[mode == 2] = 2                                     # serpentinized olivine: olivine-unit 2.3-only >= 2.305 um
    phase[binary_dilation(carb_map, np.ones((3, 3), bool))] = 3   # Mg-carbonate, dilated 1 px for visibility
    log["bandcentre"] = {"n_oh_det": int(np.isfinite(oh_c).sum()), "n_23_det": int(np.isfinite(c23).sum()),
                         "phase_counts": {"smectite": int((combo == 3).sum()), "serpentinized_olivine": int((mode == 2).sum()),
                                          "mgco3_pixels": int(carb_map.sum()), "mgco3_pixels_dilated": int((phase == 3).sum())}}
    W1 = 3.6                                                   # single-column width
    fig, axes = plt.subplots(2, 1, figsize=(W1, W1 * ny / nx * 2 * 1.14), squeeze=False)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.07, top=0.995, hspace=0.20)
    bgimg = np.where(V, 0.90, 0.75)
    specs = [("a", "centre", oh_c, "viridis", (1.37, 1.43), "1.4 µm OH band centre (µm)"),
             ("b", "centre", c23, "viridis", (2.28, 2.35), "2.3 µm band centre (µm)")]
    for ax, (let, kind, arr, cmap, vr, cblabel) in zip(axes.ravel(), specs):
        ax.imshow(bgimg, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        if kind == "centre":
            im = ax.imshow(np.where(np.isfinite(arr), arr, np.nan), cmap=cmap, vmin=vr[0], vmax=vr[1], interpolation="nearest")
            cax = ax.inset_axes([0.50, -0.075, 0.48, 0.035])
            cb = fig.colorbar(im, cax=cax, orientation="horizontal")
            cb.ax.tick_params(labelsize=FS_TICK - 1, colors="black", length=2, pad=1)
            cax.set_title(cblabel, fontsize=FS_TICK - 1, pad=2)
        else:
            lay = np.where(arr > 0, arr.astype(float), np.nan)
            ax.imshow(lay, cmap=ListedColormap([cmap[1][1], cmap[2][1], cmap[3][1]]), vmin=1, vmax=3, interpolation="nearest")
            ax.legend(handles=[Patch(facecolor=cmap[k][1], edgecolor="none", label=cmap[k][0]) for k in (1, 2, 3)],
                      loc="upper left", bbox_to_anchor=(1.04, 0.98), fontsize=9, frameon=False, ncol=1, handlelength=1.6, labelspacing=0.6, title="phases", title_fontsize=10)
        unit_lines(ax, FR, lw=0.7, casing=True)
        ax.set_xlim(0, nx); ax.set_ylim(ny, 0); ax.set_xticks([]); ax.set_yticks([])
        scalebar(ax, nx, ny, color="black"); north(ax, nx, ny, color="black"); letter(ax, let, color="black")
    axes[0, 0].legend(handles=[Line2D([0], [0], color="black", ls=ls, lw=1.2, label=lab) for ls, lab in UNIT_LS.values()],
                      loc="upper left", bbox_to_anchor=(0.0, -0.01), fontsize=6.5, frameon=False, ncol=1, handlelength=2.0, labelspacing=0.2)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"FigS_bandcentre_FINAL.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    bc = log["bandcentre"]
    cap2 = (f"Fig. S-bandcentre. CRISM {CUBE}, bland-ratioed TRR3 chain (crism_ml), 18 m pixels, north up; geologic units as black lines "
            f"(solid = olivine unit, dashed = phyllosilicate basement, dotted = mafic cap). (a) Pixels with a 1.4 µm OH band detected by the "
            f"matched filter (empirical null threshold), coloured by band centre (n = {bc['n_oh_det']}). (b) Same for the 2.3 µm metal-OH band "
            f"(n = {bc['n_23_det']}). The basement smectite pairs 1.40-1.41 µm with 2.30 µm; the olivine unit carries 2.31-2.32 µm centres without a "
            f"1.4 µm OH band. Absolute band centres carry a ~13 nm inter-chain offset (single processing chain).")
    (OUT / "FigS_bandcentre_FINAL_caption.txt").write_text(cap2, encoding="utf-8")
    log.update({"script": Path(__file__).name, "script_md5": md5(Path(__file__)), "started_utc": t0.isoformat(),
                "finished_utc": datetime.now(timezone.utc).isoformat(), "captions": [cap1, cap2]})
    (OUT / "FigS_FINAL_run_log.json").write_text(json.dumps(log, indent=2, default=str))
    print(json.dumps({k: log[k] for k in ("carbonate", "bandcentre")}, indent=1))


if __name__ == "__main__":
    main()
