"""2.3 um band-centre and 1.9/2.3 band-ratio test for the contact halo.

Follows contact_distance_to_contact.py. For each cube, compares the
position of the 2.3 um absorption minimum and the BD1900R2 / D2300 ratio
between (a) interior Fe/Mg-phyllosilicate pixels, (b) olivine-bearing pixels
in distance bands from the clay contact, and (c) interior mafic-cap pixels.
Sub-resolution smectite mixing predicts near-contact olivine to inherit the
smectite band centre (~2.29-2.30 um) and the smectite 1.9/2.3 ratio; a
serpentine/talc-like phase predicts a longer band centre (>=2.31 um) and a
weak 1.9 um band. Reads only the 2.16-2.42 um bands from the MTR3 cubes.

Usage:  python code/contact_band_center_2300.py
"""
from __future__ import annotations

import hashlib
import json
import re
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
from contact_distance_to_contact import (  # noqa: E402
    EDGE_EXCL_M, OTHER_CONTACT_EXCL_M, PALETTE, PIX_M, V8_PARAMS, distances, load_cube, masks_from_labels, md5,
)

OUT = ROOT / "outputs" / "revision" / "distance_to_contact"
WIN_UM = (2.16, 2.42)          # bands read from disk
LEFT_ANCHOR = (2.200, 2.225)   # continuum anchors (mean reflectance in each range)
RIGHT_ANCHOR = (2.365, 2.395)
SEARCH = (2.265, 2.350)        # band-minimum search window
MIN_DEPTH = 0.005              # continuum-removed depth required to accept a band centre
BANDS = {                      # olivine populations by distance to clay (m)
    "olivine 36-90 m": (36.1, 90.0),
    "olivine 90-150 m": (90.0, 150.0),
    "olivine 150-400 m": (150.0, 400.0),
    "olivine >400 m": (400.0, np.inf),
}
INTERIOR_M = 54.0              # 'pure' clay / cap pixels: >54 m from any other class


def read_window(cube: str) -> tuple[np.ndarray, np.ndarray, Path]:
    low = cube.lower()
    hdr = ROOT / "cubes" / f"{low}_07_if166j_mtr3" / f"{low}_07_if166j_mtr3.hdr"
    txt = hdr.read_text()
    get = lambda k: int(re.search(rf"^{k}\s*=\s*(\d+)", txt, re.M | re.I).group(1))
    ns, nl, nb = get("samples"), get("lines"), get("bands")
    assert re.search(r"interleave\s*=\s*bsq", txt, re.I), "expected BSQ"
    wav = np.array([float(x) for x in re.search(r"wavelength\s*=\s*\{([^}]*)\}", txt, re.S | re.I).group(1).split(",")])
    wav_um = wav / 1000.0 if wav.max() > 100 else wav
    sel = np.where((wav_um >= WIN_UM[0]) & (wav_um <= WIN_UM[1]))[0]
    mm = np.memmap(hdr.with_suffix(".img"), dtype="<f4", mode="r", shape=(nb, nl, ns))
    block = np.array(mm[sel.min():sel.max() + 1]).astype(np.float32)   # (nw, nl, ns)
    block[block >= 65535.0] = np.nan
    return wav_um[sel], block, hdr.with_suffix(".img")


def band_center(wav: np.ndarray, block: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Continuum-removed 2.3 um minimum (quadratic-refined) and depth for masked pixels."""
    spec = block[:, mask].T                                 # (n, nw)
    la = (wav >= LEFT_ANCHOR[0]) & (wav <= LEFT_ANCHOR[1])
    ra = (wav >= RIGHT_ANCHOR[0]) & (wav <= RIGHT_ANCHOR[1])
    rl, rr = np.nanmean(spec[:, la], axis=1), np.nanmean(spec[:, ra], axis=1)
    wl, wr = wav[la].mean(), wav[ra].mean()
    cont = rl[:, None] + (rr - rl)[:, None] * (wav[None, :] - wl) / (wr - wl)
    cr = spec / cont
    sw = np.where((wav >= SEARCH[0]) & (wav <= SEARCH[1]))[0]
    sub = cr[:, sw]
    i = np.nanargmin(sub, axis=1)
    depth = 1.0 - sub[np.arange(sub.shape[0]), i]
    # quadratic through the minimum and its two neighbours (in the search window)
    i = np.clip(i, 1, sub.shape[1] - 2)
    y0, y1, y2 = sub[np.arange(len(i)), i - 1], sub[np.arange(len(i)), i], sub[np.arange(len(i)), i + 1]
    x0, x1, x2 = wav[sw[i - 1]], wav[sw[i]], wav[sw[i + 1]]
    denom = (y0 - 2 * y1 + y2)
    with np.errstate(divide="ignore", invalid="ignore"):
        shift = 0.5 * (y0 - y2) / denom * (x1 - x0)
    shift = np.where(np.isfinite(shift) & (np.abs(shift) <= (x1 - x0)), shift, 0.0)
    return x1 + shift, depth


def main() -> None:
    t0 = datetime.now(timezone.utc)
    rows, hist_data, inputs, excess_samples = [], {}, {}, {}
    for cube in CUBE_IDS:
        excess_samples[cube] = {}
        c = load_cube(cube)
        m = masks_from_labels(c["labels"])
        d = distances(m)
        wav, block, img = read_window(cube)
        inputs[cube] = {"mtr3_img": str(img.relative_to(ROOT)), "mtr3_size_bytes": img.stat().st_size,
                        "labels_md5": md5(c["files"]["labels"]), "params_md5": md5(c["files"]["params"])}
        d_nonclay = distance_transform_edt(m["clay"], sampling=(PIX_M, PIX_M))
        d_noncap = distance_transform_edt(m["cap"], sampling=(PIX_M, PIX_M))
        edge_ok = d["edge"] > EDGE_EXCL_M
        pops = {
            "phyllosilicate interior": m["clay"] & (d_nonclay > INTERIOR_M) & edge_ok,
            "mafic cap interior": m["cap"] & (d_noncap > INTERIOR_M) & edge_ok,
        }
        for name, (lo, hi) in BANDS.items():
            pops[name] = (m["oliv"] & edge_ok & (d["cap"] > OTHER_CONTACT_EXCL_M) & (d["clay"] > lo) & (d["clay"] <= hi))
        ip = {p: V8_PARAMS.index(p) for p in ("BD1900R2", "D2300", "BD2500_2")}
        hist_data[cube] = {}
        for name, mask in pops.items():
            n = int(mask.sum())
            if n < 50:
                continue
            bc, depth = band_center(wav, block, mask)
            ok = np.isfinite(bc) & (depth >= MIN_DEPTH)
            b19 = c["params"][..., ip["BD1900R2"]][mask]
            d23 = c["params"][..., ip["D2300"]][mask]
            ratio = b19 / d23
            good = np.isfinite(ratio) & (d23 > 0.005)
            row = {"cube": cube, "population": name, "n": n, "n_band_ok": int(ok.sum()), "frac_band_ok": float(ok.mean()),
                   "center_med_um": float(np.median(bc[ok])) if ok.any() else np.nan,
                   "center_q25": float(np.percentile(bc[ok], 25)) if ok.any() else np.nan,
                   "center_q75": float(np.percentile(bc[ok], 75)) if ok.any() else np.nan,
                   "frac_center_ge_2p315": float((bc[ok] >= 2.315).mean()) if ok.any() else np.nan,
                   "depth_med": float(np.median(depth[ok])) if ok.any() else np.nan,
                   "BD1900R2_med": float(np.nanmedian(b19)), "D2300_med": float(np.nanmedian(d23)),
                   "ratio_1900_2300_med": float(np.median(ratio[good])) if good.any() else np.nan,
                   "n_ratio": int(good.sum())}
            rows.append(row)
            hist_data[cube][name] = bc[ok]
            excess_samples[cube][name] = (b19, d23)
        print(cube, "done")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "band_center_2300.csv", index=False)
    print(df.round(4).to_string())

    # cap-baseline-corrected excess band depths and their ratio (bootstrap CI on the ratio of medians)
    rng = np.random.default_rng(42)
    ex_rows = []
    for cube in CUBE_IDS:
        cap = df[(df.cube == cube) & (df.population == "mafic cap interior")].iloc[0]
        b19_cap, d23_cap = cap.BD1900R2_med, cap.D2300_med
        for name in [p for p in df[df.cube == cube].population if p != "mafic cap interior"]:
            b19, d23 = excess_samples[cube][name]
            ex19, ex23 = np.nanmedian(b19) - b19_cap, np.nanmedian(d23) - d23_cap
            boots = []
            for _ in range(500):
                idx = rng.integers(0, b19.size, b19.size)
                e23 = np.nanmedian(d23[idx]) - d23_cap
                boots.append((np.nanmedian(b19[idx]) - b19_cap) / e23 if e23 > 0 else np.nan)
            boots = np.array(boots)
            ex_rows.append({"cube": cube, "population": name, "n": b19.size,
                            "excess_BD1900R2": float(ex19), "excess_D2300": float(ex23),
                            "excess_ratio_1900_2300": float(ex19 / ex23) if ex23 > 0 else np.nan,
                            "ratio_ci_lo": float(np.nanpercentile(boots, 2.5)), "ratio_ci_hi": float(np.nanpercentile(boots, 97.5))})
    exdf = pd.DataFrame(ex_rows)
    exdf.to_csv(OUT / "excess_ratio_1900_2300.csv", index=False)
    print("\nCap-baseline-corrected excess band depths (median - cap interior median):")
    print(exdf.round(4).to_string())

    # figure: band-centre distributions per population, one panel per cube
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    pop_style = {"phyllosilicate interior": ("#7f7f7f", "-"), "mafic cap interior": ("#2b2b2b", ":"),
                 "olivine 36-90 m": ("#d55e00", "-"), "olivine 90-150 m": ("#e69f00", "-"),
                 "olivine 150-400 m": ("#0072b2", "-"), "olivine >400 m": ("#009e73", "-")}
    edges = np.arange(2.265, 2.351, 0.0033)
    for ax, cube in zip(axes.ravel(), CUBE_IDS):
        for name, (col, ls) in pop_style.items():
            v = hist_data[cube].get(name)
            if v is None or v.size < 50:
                continue
            h, _ = np.histogram(v, bins=edges, density=True)
            ax.plot(0.5 * (edges[1:] + edges[:-1]), h, color=col, ls=ls, lw=1.8, label=f"{name} (n={v.size})")
        ax.axvline(2.300, color="0.6", lw=0.8, ls="--"); ax.axvline(2.320, color="0.6", lw=0.8, ls="--")
        ax.set_title(cube, fontsize=10, color=PALETTE[cube])
        ax.grid(True, color="0.92"); ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, frameon=False)
    for ax in axes[1]:
        ax.set_xlabel("2.3 um band centre (um), continuum 2.21-2.38 um")
    for ax in axes[:, 0]:
        ax.set_ylabel("density")
    fig.suptitle("2.3 um band-centre distributions: clay vs. olivine-bearing pixels by distance from the clay contact", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_band_center_2300.png", dpi=200)
    fig.savefig(OUT / "fig_band_center_2300.pdf")
    plt.close(fig)

    log = {"script": Path(__file__).name, "script_md5": md5(Path(__file__)), "started_utc": t0.isoformat(),
           "finished_utc": datetime.now(timezone.utc).isoformat(), "window_um": WIN_UM, "left_anchor": LEFT_ANCHOR,
           "right_anchor": RIGHT_ANCHOR, "search": SEARCH, "min_depth": MIN_DEPTH, "bands_m": BANDS,
           "interior_m": INTERIOR_M, "inputs": inputs}
    (OUT / "run_log_band_center.json").write_text(json.dumps(log, indent=2, default=str))


if __name__ == "__main__":
    main()
