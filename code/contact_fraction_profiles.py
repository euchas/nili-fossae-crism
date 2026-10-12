"""Stage 1b of PLAN_revision_contact_3D.md: distance-to-contact profiles
controlled by the geologic map.

Uses the map-derived unit fractions (contact_geomap_fractions.py) to
(1) define the clay contact from the map instead of the spectral class map,
(2) restrict the olivine population to pixels the map says are >= 90 %
olivine unit (and, strictly, contain no mapped clay at all), and
(3) regress D2300 on the sub-pixel clay fraction and test whether the
distance dependence survives. Also tabulates GMM-class vs map-unit
agreement, which doubles as a check of the Stage 0 frame shift.

Usage: python code/contact_fraction_profiles.py --frame ctx|pds
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import distance_transform_edt

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import CUBE_IDS  # noqa: E402
from contact_distance_to_contact import (  # noqa: E402
    DIST_EDGES, DIST_LABELS, EDGE_EXCL_M, MIX_BINS, PALETTE, PIX_M, V8_PARAMS, load_cube, masks_from_labels, profile,
)

GEOMAP = ROOT / "outputs" / "revision" / "contact_3d" / "geomap"
OUTBASE = ROOT / "outputs" / "revision" / "contact_3d"
MAP_CONTACT_FRAC = 0.5      # a pixel is "clay" (or "cap") for the distance transform if >= this fraction
OLIV_MIN = 0.9              # olivine population: map olivine fraction >= this
PURE = 0.8                  # agreement table: pixels with one unit >= this


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def agreement(labels: np.ndarray, fr: dict) -> dict:
    """GMM class counts for map-pure pixels; expected: clay->c4/c5, cap->c1, oliv/dune->c0/c2/c3."""
    out = {}
    exp = {"clay": (4, 5), "cap": (1,), "oliv": (0, 2, 3), "dune": (0, 2, 3)}
    for unit, classes in exp.items():
        sel = (fr[unit] >= PURE) & (labels >= 0)
        n = int(sel.sum())
        counts = {f"c{k}": int((labels[sel] == k).sum()) for k in range(6)}
        out[unit] = {"n": n, "counts": counts, "agree_frac": float(np.isin(labels[sel], classes).mean()) if n else np.nan}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--frame", choices=["pds", "ctx"], default="ctx")
    a = ap.parse_args()
    out = OUTBASE / f"stage1_{a.frame}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now(timezone.utc)
    prof_rows, stat_rows, agree_all, reg_rows = [], [], {}, []
    for cube in CUBE_IDS:
        c = load_cube(cube)
        m = masks_from_labels(c["labels"])
        fz = np.load(GEOMAP / f"frame_{a.frame}" / f"{cube}_fractions.npz")
        fr = {k: fz[k] for k in ("clay", "cap", "dune", "graben", "oliv", "outline")}
        mapped = fr["outline"] >= 0.99
        # map-based contacts and distances
        mm = {"clay": (fr["clay"] >= MAP_CONTACT_FRAC), "cap": (fr["cap"] >= MAP_CONTACT_FRAC), "carb": m["carb"], "valid": m["valid"] & mapped}
        d = {"clay": distance_transform_edt(~mm["clay"], sampling=(PIX_M, PIX_M)),
             "cap": distance_transform_edt(~mm["cap"], sampling=(PIX_M, PIX_M)),
             "edge": distance_transform_edt(mm["valid"], sampling=(PIX_M, PIX_M))}
        agree_all[cube] = agreement(c["labels"], fr)
        variants = {
            "gmm_olivine": m["oliv"] & mapped,                                   # spectral olivine, map-based distance
            "map_oliv_ge0.9": m["valid"] & mapped & (fr["oliv"] >= OLIV_MIN) & ~m["clay"],
            "map_clay_eq0": m["valid"] & mapped & (fr["oliv"] >= OLIV_MIN) & (fr["clay"] == 0) & ~m["clay"],
            "map_clay_eq0_gmm_oliv": m["oliv"] & mapped & (fr["oliv"] >= OLIV_MIN) & (fr["clay"] == 0),
        }
        for variant, sel in variants.items():
            for contact in ("clay", "cap"):
                rows, st = profile(cube, contact, sel, d, mm, c["params"], c["elev"], variant)
                prof_rows += rows; stat_rows.append(st)
        # regression of D2300 on clay fraction among olivine-unit pixels near the contact
        i23, i19, i25 = V8_PARAMS.index("D2300"), V8_PARAMS.index("BD1900R2"), V8_PARAMS.index("BD2500_2")
        sel = m["valid"] & mapped & (fr["cap"] < 0.1) & (fr["dune"] < 0.1) & (fr["graben"] < 0.1) & (d["edge"] > EDGE_EXCL_M) & (d["clay"] <= 700) & ~m["clay"]
        x = fr["clay"][sel]; y = c["params"][..., i23][sel]; dist = d["clay"][sel]
        ok = np.isfinite(y)
        slope, intercept, r, p, se = stats.linregress(x[ok], y[ok])
        resid = y[ok] - (intercept + slope * x[ok])
        idx = np.digitize(dist[ok], DIST_EDGES) - 1
        for b, lab in enumerate(DIST_LABELS):
            s = idx == b
            if s.sum() >= 100:
                reg_rows.append({"cube": cube, "bin": lab, "bin_idx": b, "n": int(s.sum()), "f_clay_med": float(np.median(x[ok][s])),
                                 "D2300_med": float(np.median(y[ok][s])), "D2300_resid_med": float(np.median(resid[s])),
                                 "slope": float(slope), "intercept": float(intercept), "r": float(r)})
        print(cube, "agreement:", {u: round(v["agree_frac"], 3) for u, v in agree_all[cube].items()},
              "| D2300 ~ f_clay: slope", round(float(slope), 4), "r", round(float(r), 3),
              "| n per variant:", {k: int(v.sum()) for k, v in variants.items()})

    prof = pd.DataFrame(prof_rows); prof.to_csv(out / "distance_profiles.csv", index=False)
    statdf = pd.DataFrame(stat_rows); statdf.to_csv(out / "distance_stats.csv", index=False)
    reg = pd.DataFrame(reg_rows); reg.to_csv(out / "d2300_vs_clayfraction_residuals.csv", index=False)
    (out / "gmm_vs_map_agreement.json").write_text(json.dumps(agree_all, indent=2))

    # figure: profiles vs map-based clay distance for the strict population, with the unrestricted spectral population for comparison
    panels = [("carb_frac", "fraction in GMM altered-olivine class (c0)"), ("D2300_med", "D2300 median"),
              ("BD1900R2_med", "BD1900R2 median"), ("BD2500_2_med", "BD2500_2 median"), ("centroid_med", "1 um centroid median (um)")]
    fig, axes = plt.subplots(len(panels), 2, figsize=(10, 13), sharex=True)
    for col, variant in enumerate(["gmm_olivine", "map_clay_eq0"]):
        for row, (key, ylabel) in enumerate(panels):
            ax = axes[row, col]
            ax.axvspan(-0.5, MIX_BINS - 0.5, color="0.9", zorder=0)
            for cube in CUBE_IDS:
                s = prof[(prof.variant == variant) & (prof.contact == "clay") & (prof.cube == cube)].sort_values("bin_idx")
                s = s[s.n >= 100]
                ax.plot(s.bin_idx, s[key], "-o", ms=4, lw=1.6, color=PALETTE[cube], label=cube)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if row == 0:
                ax.set_title({"gmm_olivine": "spectral olivine pixels, map-based contact",
                              "map_clay_eq0": "map: >=90% olivine unit, 0% mapped clay"}[variant], fontsize=10)
            ax.grid(True, color="0.92"); ax.spines[["top", "right"]].set_visible(False)
        axes[-1, col].set_xticks(range(len(DIST_LABELS))); axes[-1, col].set_xticklabels(DIST_LABELS, rotation=45, ha="right", fontsize=8)
        axes[-1, col].set_xlabel("distance from mapped Fe/Mg-phyllosilicate contact (m)")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Olivine pixels vs. distance from the mapped clay contact (frame: {a.frame})", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "fig_profiles_map_contact.png", dpi=200); fig.savefig(out / "fig_profiles_map_contact.pdf"); plt.close(fig)

    # figure: residual D2300 (after removing the clay-fraction regression) vs distance
    fig, ax = plt.subplots(figsize=(7, 4))
    for cube in CUBE_IDS:
        s = reg[reg.cube == cube].sort_values("bin_idx")
        ax.plot(s.bin_idx, s.D2300_resid_med, "-o", ms=4, color=PALETTE[cube], label=f"{cube} (slope {s.slope.iloc[0]:.3f}, r {s.r.iloc[0]:.2f})")
    ax.axhline(0, color="0.5", lw=0.8); ax.axvspan(-0.5, MIX_BINS - 0.5, color="0.9", zorder=0)
    ax.set_xticks(range(len(DIST_LABELS))); ax.set_xticklabels(DIST_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("median D2300 residual after f_clay regression"); ax.set_xlabel("distance from mapped clay contact (m)")
    ax.legend(fontsize=7, frameon=False); ax.grid(True, color="0.92"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "fig_d2300_residual_vs_distance.png", dpi=200); plt.close(fig)

    log = {"script_md5": md5(Path(__file__)), "frame": a.frame, "map_contact_frac": MAP_CONTACT_FRAC, "oliv_min": OLIV_MIN, "pure": PURE,
           "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
           "inputs": {cube: str((GEOMAP / f"frame_{a.frame}" / f"{cube}_fractions.npz").relative_to(ROOT)) for cube in CUBE_IDS}}
    (out / "run_log.json").write_text(json.dumps(log, indent=2, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()
