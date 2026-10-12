"""Clay-province stacked spectra, 2.33 um tail stack, and cafeco3 check.

Follow-up to contact_band_center_modes.py:

1. CLAY PROVINCES: connected components of the mapped clay unit are
   classified short/long by their median strong-tier 2.3 um band centre
   (threshold 2.311 um). For each class, the median I/F spectrum of
   map-pure clay pixels (f_clay >= 0.95, f_oliv = 0) is stacked over
   1.30-2.62 um, ratioed by the same scene cap denominator, and secondary
   bands (1.39, 1.9, 2.31, 2.35 septum, 2.39, 2.5 um) are measured on the
   stacks. Discriminates real basement heterogeneity (talc/serpentine-
   leaning vs smectite province) from spillover/depth artefacts.
2. TAIL STACK: olivine-unit pixels with centre >= 2.325 um stacked the
   same way vs an in-unit control (centre 2.312-2.322): Fe/Ca carbonate
   predicts a 2.53 um partner band; Fe-chlorite predicts 2.35 without it.
3. CAFECO3: Spearman correlation of patch-local band centre with the
   Plebani cafeco3_max / mgco3_max channels across the patch catalogue.

Stack-level readouts only (no new per-pixel parameter maps). Descriptive
stacks without CIs - exploratory; formal claims need the Tier 1 block
machinery. Deterministic. Requires Tier 0 + modes outputs.

Usage:  python code/contact_clay_provinces.py
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import label as cc_label
from scipy.stats import spearmanr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import CUBE_IDS  # noqa: E402
from contact_distance_to_contact import PALETTE, load_cube, masks_from_labels, md5  # noqa: E402
from contact_band_center_map import OUT as T0_OUT, STRONG_DEPTH, unit_masks  # noqa: E402
from contact_band_center_modes import MODE_MAP_THRESH, LONG_TAIL, PATCHES, patch_rowcol  # noqa: E402
from figstyle import draw_units, label_panels  # noqa: E402

OUT = T0_OUT / "provinces"
OUT.mkdir(parents=True, exist_ok=True)

WIN_UM = (1.30, 2.62)
MIN_COMPONENT_STRONG = 200        # strong-tier px needed to classify a clay component
CLAY_PURE_F = 0.95
TAIL_MIN_N = 300
CONTROL_RANGE = (2.312, 2.322)
FEATURES_UM = (1.39, 1.91, 2.31, 2.355, 2.39, 2.50)


def read_full_window(cube: str) -> tuple[np.ndarray, np.ndarray]:
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


def stack(block: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.nanmedian(block[:, mask], axis=1) if mask.any() else np.full(block.shape[0], np.nan)


def _cont(wav, spec, lo_win, hi_win):
    la = (wav >= lo_win[0]) & (wav <= lo_win[1])
    ra = (wav >= hi_win[0]) & (wav <= hi_win[1])
    rl, rr = np.nanmedian(spec[la]), np.nanmedian(spec[ra])
    wl, wr = wav[la].mean(), wav[ra].mean()
    return rl + (rr - rl) * (wav - wl) / (wr - wl)


def _min_fit(wav, cr, lo, hi):
    sw = np.where((wav >= lo) & (wav <= hi))[0]
    i = int(np.nanargmin(cr[sw]))
    depth = float(1.0 - cr[sw][i])
    j = int(np.clip(i, 1, len(sw) - 2))
    y0, y1, y2 = cr[sw][j - 1], cr[sw][j], cr[sw][j + 1]
    den = y0 - 2 * y1 + y2
    shift = 0.5 * (y0 - y2) / den * (wav[sw[1]] - wav[sw[0]]) if np.isfinite(den) and den != 0 else 0.0
    shift = shift if abs(shift) <= (wav[sw[1]] - wav[sw[0]]) else 0.0
    return float(wav[sw[j]] + shift), depth


def band_metrics(wav: np.ndarray, spec: np.ndarray) -> dict:
    """Secondary-band readouts on one stacked (ratioed) spectrum."""
    out = {}
    c = spec / _cont(wav, spec, (1.316, 1.342), (1.443, 1.469))
    out["bc1390_um"], out["bd1390"] = _min_fit(wav, c, 1.350, 1.440)
    c = spec / _cont(wav, spec, (1.800, 1.845), (2.060, 2.105))
    sel = (wav >= 1.900) & (wav <= 1.945)
    out["bd1900"] = float(1.0 - np.nanmedian(c[sel]))
    c = spec / _cont(wav, spec, (2.200, 2.225), (2.365, 2.395))
    out["bc2300_um"], out["bd2300"] = _min_fit(wav, c, 2.265, 2.355)
    c = spec / _cont(wav, spec, (2.146, 2.166), (2.416, 2.436))
    out["s2355"] = float(1.0 - np.nanmedian(c[(wav >= 2.344) & (wav <= 2.357)]))
    out["bd2390"] = float(1.0 - np.nanmedian(c[(wav >= 2.383) & (wav <= 2.396)]))
    c = spec / _cont(wav, spec, (2.430, 2.456), (2.560, 2.586))
    out["bc2500_um"], out["bd2500"] = _min_fit(wav, c, 2.460, 2.555)
    return out


def main() -> None:
    t0 = datetime.now(timezone.utc)
    patches_all = pd.read_csv(PATCHES)
    prov_rows, tail_rows, corr_rows, inputs = [], [], [], {}
    for cube in CUBE_IDS:
        c = load_cube(cube)
        m = masks_from_labels(c["labels"])
        units, d, fr = unit_masks(cube, m)
        cen = np.load(T0_OUT / f"{cube}_bc230_center.npy")
        dep = np.load(T0_OUT / f"{cube}_bc230_depth.npy")
        edg = np.load(T0_OUT / f"{cube}_bc230_edgeflag.npy")
        strong = np.isfinite(cen) & (dep >= STRONG_DEPTH) & ~edg
        wav, block = read_full_window(cube)
        inputs[cube] = {"center_md5": md5(T0_OUT / f"{cube}_bc230_center.npy")}

        cap_den = stack(block, m["valid"] & (fr["cap"] >= 0.9))

        # 1. clay provinces from connected components
        comp, ncomp = cc_label(fr["clay"] >= 0.5)
        prov = np.zeros(cen.shape, np.int8)          # 0 none, 1 short, 2 long
        comp_rows = []
        for k in range(1, ncomp + 1):
            cm = (comp == k) & strong
            n = int(cm.sum())
            if n < MIN_COMPONENT_STRONG:
                continue
            med = float(np.median(cen[cm]))
            cls = 1 if med < MODE_MAP_THRESH else 2
            prov[comp == k] = cls
            comp_rows.append({"cube": cube, "component": k, "n_strong": n, "center_med_um": med,
                              "class": "short" if cls == 1 else "long"})
        stacks = {}
        for cls, name in ((1, "short"), (2, "long")):
            pm = (prov == cls) & m["valid"] & (fr["clay"] >= CLAY_PURE_F) & (fr["oliv"] == 0)
            sp = stack(block, pm) / cap_den
            stacks[name] = (sp, int(pm.sum()))
            if pm.any():
                row = {"cube": cube, "province": name, "n_px": int(pm.sum()),
                       "n_components": sum(1 for r in comp_rows if r["class"] == name),
                       "depth_med_strong": float(np.median(dep[(prov == cls) & strong]))}
                row.update(band_metrics(wav, sp))
                prov_rows.append(row)

        # 2. tail stack vs in-unit control
        oliv = m["valid"] & (fr["oliv"] >= 0.5) & strong
        tail = oliv & (cen >= LONG_TAIL)
        ctrl = oliv & (cen >= CONTROL_RANGE[0]) & (cen <= CONTROL_RANGE[1])
        tail_stacks = {}
        if tail.sum() >= TAIL_MIN_N:
            for name, mask in (("tail", tail), ("control", ctrl)):
                sp = stack(block, mask) / cap_den
                tail_stacks[name] = (sp, int(mask.sum()))
                row = {"cube": cube, "set": name, "n_px": int(mask.sum()),
                       "depth_med": float(np.median(dep[mask]))}
                row.update(band_metrics(wav, sp))
                tail_rows.append(row)

        # 3. cafeco3 / mgco3 correlation with patch-local centre
        pdfc = patches_all[(patches_all.cube == cube) & patches_all.on_map].reset_index(drop=True)
        if len(pdfc):
            pr, pc = patch_rowcol(cube, pdfc, cen.shape)
            loc = []
            for r0, c0 in zip(pr, pc):
                w = cen[max(r0 - 1, 0):r0 + 2, max(c0 - 1, 0):c0 + 2]
                s = strong[max(r0 - 1, 0):r0 + 2, max(c0 - 1, 0):c0 + 2]
                loc.append(float(np.median(w[s])) if s.any() else np.nan)
            loc = np.array(loc)
            ok = np.isfinite(loc)
            sub = pdfc.iloc[:len(loc)]
            for chan in ("cafeco3_max", "mgco3_max"):
                rho, p = (spearmanr(loc[ok], sub[chan].to_numpy()[ok]) if ok.sum() >= 8 else (np.nan, np.nan))
                corr_rows.append({"cube": cube, "channel": chan, "n_patches": int(ok.sum()),
                                  "spearman_rho": float(rho), "p_naive": float(p)})

        # figures per cube
        fig = plt.figure(figsize=(15, 6.2))
        ax0 = fig.add_subplot(1, 3, 1)
        bg = np.where(np.isfinite(cen), 0.92, np.nan)
        ax0.imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        pm = np.where(prov > 0, prov.astype(float), np.nan)
        ax0.imshow(pm, cmap=matplotlib.colors.ListedColormap(["#0072B2", "#D55E00"]),
                   vmin=1, vmax=2, interpolation="nearest")
        draw_units(ax0, fr, units=("oliv",))
        ax0.set_title("clay components: blue = short, orange = long", fontsize=9)
        ax0.set_xticks([]); ax0.set_yticks([])
        ax1 = fig.add_subplot(1, 3, 2)
        off = 0.0
        for name, colr in (("short", "#0072B2"), ("long", "#D55E00")):
            sp, n = stacks[name]
            if np.isfinite(sp).any():
                ax1.plot(wav, sp / np.nanmedian(sp) + off, color=colr, lw=1.2,
                         label=f"{name} province (n={n})")
                off += 0.06
        for f in FEATURES_UM:
            ax1.axvline(f, color="0.85", lw=0.7, zorder=0)
        ax1.set_xlabel("wavelength (um)"); ax1.set_ylabel("ratioed I/F (normalized + offset)")
        ax1.legend(fontsize=8, frameon=False); ax1.set_title("cap-ratioed stacks", fontsize=9)
        ax1.grid(True, color="0.95"); ax1.spines[["top", "right"]].set_visible(False)
        ax2 = fig.add_subplot(1, 3, 3)
        if all(np.isfinite(stacks[k][0]).any() for k in ("short", "long")):
            rr = stacks["long"][0] / stacks["short"][0]
            ax2.plot(wav, rr / np.nanmedian(rr), color="k", lw=1.2)
        for f in FEATURES_UM:
            ax2.axvline(f, color="0.85", lw=0.7, zorder=0)
            ax2.annotate(f"{f}", (f, 1.0), xytext=(0, -12), textcoords="offset points",
                         fontsize=7, ha="center", color="0.4", annotation_clip=False)
        ax2.set_xlabel("wavelength (um)"); ax2.set_title("long / short ratio", fontsize=9)
        ax2.grid(True, color="0.95"); ax2.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"{cube}: clay provinces (f_clay >= {CLAY_PURE_F}, f_oliv = 0; "
                     f"components classified at {MODE_MAP_THRESH} um)", fontsize=11)
        fig.tight_layout()
        label_panels(fig)
        fig.savefig(OUT / f"fig_provinces_{cube}.png", dpi=200, bbox_inches="tight")
        fig.savefig(OUT / f"fig_provinces_{cube}.pdf", bbox_inches="tight")
        plt.close(fig)

        if tail_stacks:
            fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.8))
            for name, colr in (("tail", "#CC79A7"), ("control", "0.35")):
                sp, n = tail_stacks[name]
                a0.plot(wav, sp / np.nanmedian(sp), lw=1.2, color=colr, label=f"{name} (n={n})")
            for f in (2.31, 2.33, 2.355, 2.39, 2.50, 2.53):
                a0.axvline(f, color="0.88", lw=0.7, zorder=0)
            a0.set_xlim(2.1, 2.62); a0.legend(fontsize=8, frameon=False)
            a0.set_xlabel("wavelength (um)"); a0.set_ylabel("ratioed I/F (normalized)")
            a0.set_title("tail (>= 2.325 um) vs in-unit control", fontsize=9)
            rr = tail_stacks["tail"][0] / tail_stacks["control"][0]
            a1.plot(wav, rr / np.nanmedian(rr), color="k", lw=1.2)
            for f in (2.33, 2.53):
                a1.axvline(f, color="#CC79A7", lw=0.8, ls="--")
            a1.set_xlim(2.1, 2.62); a1.set_xlabel("wavelength (um)")
            a1.set_title("tail / control (Fe-Ca carbonate: dips at 2.33 AND 2.53)", fontsize=9)
            for ax in (a0, a1):
                ax.grid(True, color="0.95"); ax.spines[["top", "right"]].set_visible(False)
            fig.suptitle(f"{cube}: long-tail attribution stack", fontsize=11)
            fig.tight_layout()
            label_panels(fig)
            fig.savefig(OUT / f"fig_tail_{cube}.png", dpi=200, bbox_inches="tight")
            fig.savefig(OUT / f"fig_tail_{cube}.pdf", bbox_inches="tight")
            plt.close(fig)

        del block
        print(cube, "done: provinces",
              {r["province"]: r["n_px"] for r in prov_rows if r["cube"] == cube},
              "tail_n", int(tail.sum()))

    pd.DataFrame(prov_rows).to_csv(OUT / "clay_province_metrics.csv", index=False)
    pd.DataFrame(tail_rows).to_csv(OUT / "tail_stack_metrics.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(OUT / "cafeco3_correlation.csv", index=False)
    print(pd.DataFrame(prov_rows).round(4).to_string())
    print(pd.DataFrame(tail_rows).round(4).to_string())
    print(pd.DataFrame(corr_rows).round(4).to_string())

    log = {"script": Path(__file__).name, "script_md5": md5(Path(__file__)),
           "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
           "window_um": WIN_UM, "min_component_strong": MIN_COMPONENT_STRONG,
           "clay_pure_f": CLAY_PURE_F, "province_threshold_um": MODE_MAP_THRESH,
           "tail_um": LONG_TAIL, "control_range_um": CONTROL_RANGE, "tail_min_n": TAIL_MIN_N,
           "denominator": "median I/F of valid & f_cap>=0.9 pixels (same for all stacks in a cube)",
           "note": "stack-level readouts, no per-pixel parameter maps; descriptive, no CIs",
           "patches_csv_md5": md5(PATCHES), "inputs": inputs}
    (OUT / "run_log_provinces.json").write_text(json.dumps(log, indent=2, default=str))
    print("done ->", OUT)


if __name__ == "__main__":
    main()
