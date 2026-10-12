"""Unit-internal anomaly ratio spectra (Jezero-style).

For each cube and each mapped unit (olivine, clay): stack the strong-tier
pixels in the top and bottom deciles of 2.3 um band centre (plus the fixed
>= 2.325 um tail in the olivine unit) and ratio each stack against the
median spectrum of the ENTIRE unit. Unit-internal ratios cancel
unit-typical absorptions, illumination and shared atmospheric/denominator
artefacts, isolating what makes the anomalous pixels different from their
own unit. Complements contact_clay_provinces.py (cap-ratioed stacks).

Stack-level readouts, descriptive (no CIs). Deterministic.

Usage:  python code/contact_anomaly_ratios.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import CUBE_IDS  # noqa: E402
from contact_distance_to_contact import load_cube, masks_from_labels, md5  # noqa: E402
from contact_band_center_map import OUT as T0_OUT, STRONG_DEPTH, unit_masks  # noqa: E402
from contact_band_center_modes import LONG_TAIL  # noqa: E402
from contact_clay_provinces import band_metrics, read_full_window, stack  # noqa: E402
from figstyle import label_panels  # noqa: E402

OUT = T0_OUT / "anomaly_ratios"
OUT.mkdir(parents=True, exist_ok=True)

DECILES = (0.10, 0.90)
TAIL_MIN_N = 300
UNITS = (("olivine", "oliv"), ("clay", "clay"))
FEATURES_UM = (1.39, 1.91, 2.31, 2.33, 2.355, 2.39, 2.50, 2.53)
ZOOM_UM = (2.27, 2.40)
ZOOM_TICKS = (2.30, 2.31, 2.32, 2.33, 2.34, 2.35)


def main() -> None:
    t0 = datetime.now(timezone.utc)
    rows, inputs = [], {}
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

        fig, axes = plt.subplots(2, len(UNITS), figsize=(13.5, 9.2))
        for j, (uname, fkey) in enumerate(UNITS):
            ax_full, ax_zoom = axes[0, j], axes[1, j]
            unit = m["valid"] & (fr[fkey] >= 0.5)
            sel = unit & strong
            if sel.sum() < 1000:
                ax_full.set_title(f"{uname}: n={int(sel.sum())} < 1000, skipped", fontsize=9)
                continue
            qlo, qhi = np.quantile(cen[sel], DECILES)
            den = stack(block, unit)
            sets = {f"low decile (<= {qlo:.4f})": sel & (cen <= qlo),
                    f"high decile (>= {qhi:.4f})": sel & (cen >= qhi)}
            if uname == "olivine" and (sel & (cen >= LONG_TAIL)).sum() >= TAIL_MIN_N:
                sets[f"tail (>= {LONG_TAIL})"] = sel & (cen >= LONG_TAIL)
            colors = {list(sets)[0]: "#0072B2", list(sets)[1]: "#D55E00"}
            if len(sets) == 3:
                colors[list(sets)[2]] = "#CC79A7"
            curves = {}
            for name, mask in sets.items():
                r = stack(block, mask) / den
                row = {"cube": cube, "unit": uname, "set": name, "n_px": int(mask.sum()),
                       "n_unit_den": int(unit.sum()), "depth_med": float(np.median(dep[mask]))}
                row.update(band_metrics(wav, r))
                rows.append(row)
                curves[name] = r / np.nanmedian(r)
            for name, y in curves.items():
                ls = "--" if name.startswith("tail") else "-"
                ax_full.plot(wav, y, lw=1.7, color=colors[name], ls=ls,
                             label=f"{name}, n={int(sets[name].sum())}")
                ax_zoom.plot(wav, y, lw=1.9, color=colors[name], ls=ls)
            for f in FEATURES_UM:
                ax_full.axvline(f, color="0.92", lw=0.7, zorder=0)
            ax_full.set_title(f"{uname} unit (f >= 0.5): anomaly / unit-median", fontsize=10)
            ax_full.grid(True, color="0.96"); ax_full.spines[["top", "right"]].set_visible(False)
            ax_full.legend(fontsize=8, frameon=False)
            # zoom on the 2.3 um region with labelled ticks
            ax_zoom.set_xlim(ZOOM_UM)
            ax_zoom.axvline(2.30, color="0.35", lw=1.0, zorder=0)
            ax_zoom.set_xticks(ZOOM_TICKS)
            ax_zoom.set_xticklabels([f"{t:.2f}" for t in ZOOM_TICKS], fontsize=9)
            for t in ZOOM_TICKS[1:]:
                ax_zoom.axvline(t, color="0.90", lw=0.7, zorder=0)
            zs = (wav >= ZOOM_UM[0]) & (wav <= ZOOM_UM[1])
            ylo = min(np.nanmin(y[zs]) for y in curves.values())
            yhi = max(np.nanmax(y[zs]) for y in curves.values())
            pad = 0.15 * (yhi - ylo)
            ax_zoom.set_ylim(ylo - pad, yhi + pad)
            ax_zoom.set_xlabel("wavelength (um)")
            ax_zoom.set_title("zoom 2.3 um region", fontsize=9)
            ax_zoom.grid(False); ax_zoom.spines[["top", "right"]].set_visible(False)
        for ax in axes[:, 0]:
            ax.set_ylabel("ratioed I/F (normalized)")
        fig.suptitle(f"{cube}: unit-internal anomaly ratios (strong tier; deciles of 2.3 um centre)",
                     fontsize=11)
        fig.tight_layout()
        label_panels(fig)
        fig.savefig(OUT / f"fig_anomaly_{cube}.png", dpi=200, bbox_inches="tight")
        fig.savefig(OUT / f"fig_anomaly_{cube}.pdf", bbox_inches="tight")
        plt.close(fig)
        del block
        print(cube, "done")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "anomaly_ratio_metrics.csv", index=False)
    print(df.round(4).to_string())

    log = {"script": Path(__file__).name, "script_md5": md5(Path(__file__)),
           "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
           "deciles": DECILES, "tail_um": LONG_TAIL, "tail_min_n": TAIL_MIN_N,
           "denominator": "median I/F of ALL valid unit pixels (f >= 0.5), per unit per cube",
           "note": "unit-internal ratios; stack-level readouts; descriptive, no CIs",
           "inputs": inputs}
    (OUT / "run_log_anomaly_ratios.json").write_text(json.dumps(log, indent=2, default=str))
    print("done ->", OUT)


if __name__ == "__main__":
    main()
