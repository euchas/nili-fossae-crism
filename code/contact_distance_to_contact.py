"""Distance-to-contact test for the olivine unit at Horseshoe.

Tests whether carbonate / hydration indicators in olivine-bearing pixels
vary with distance from the nearest Fe/Mg-phyllosilicate pixel, using the
mafic-cap contact as a control and MOLA elevation (CRISM DDR) as a second
axis. Runs on the map-projected MTR3 grid (18 m/px). See
PLAN_revision_contact.md for the rationale.

Usage:  python code/contact_distance_to_contact.py
"""
from __future__ import annotations

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
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import CUBE_IDS  # noqa: E402

OUT = ROOT / "outputs" / "revision" / "distance_to_contact"
OUT.mkdir(parents=True, exist_ok=True)

PIX_M = 18.0
V8_PARAMS = ("centroid", "asymmetry", "OLINDEX3", "BD1400", "BD1900R2", "D2300", "BD2500_2")
METRICS = ("BD2500_2", "D2300", "BD1900R2", "OLINDEX3", "centroid", "asymmetry")
CLASS = {"carb": (0,), "cap": (1,), "oliv": (0, 2, 3), "clay": (4, 5)}
DIST_EDGES = np.array([0, 18.1, 36.1, 54.1, 90, 150, 250, 400, 700, 1200, np.inf])
DIST_LABELS = ["18", "36", "54", "55-90", "90-150", "150-250", "250-400", "400-700", "700-1200", ">1200"]
MIX_BINS = 2                      # first two bins (<=36 m) flagged as sub-pixel mixing
EDGE_EXCL_M = 90.0                # drop pixels within this distance of nodata / scene edge
OTHER_CONTACT_EXCL_M = 90.0       # profile vs contact A uses pixels > this from contact B
NEAR_RANGE = (36.1, 150.0)        # "near" band for the near/far ratio
FAR_MIN = 400.0
ELEV_BIN_M = 25.0
OLINDEX_PCTL = 60.0               # GMM-free olivine definition for robustness

PALETTE = {  # Okabe-Ito, fixed order per cube
    "FRT00003E12": "#0072B2",
    "FRT0000871C": "#E69F00",
    "FRT0000A4FC": "#009E73",
    "FRT0000B438": "#CC79A7",
}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cube(cube: str) -> dict:
    lab_dir = ROOT / "outputs" / "_v9_1_per_cube_scaler" / "v9_1b_7param" / cube / "arrays"
    files = {
        "labels": lab_dir / f"{cube}_v9_1b_gmm_labels.npy",
        "novel": lab_dir / f"{cube}_v9_1b_novel_mask.npy",
        "params": ROOT / "outputs" / "_v8_mtr3_with_brown" / cube / "arrays" / f"{cube}_v8_param_stack.npy",
    }
    low = cube.lower()
    ddr_hdr = ROOT / "cubes" / f"{low}_07_if166j_mtr3" / f"{low}_07_de166l_mtr1.hdr"
    ddr_img = ddr_hdr.with_suffix(".img")
    files["ddr"] = ddr_img

    labels = np.load(files["labels"]).astype(np.int16)
    novel = np.load(files["novel"]).astype(bool)
    params = np.load(files["params"]).astype(np.float32)
    H, W = labels.shape
    assert params.shape == (H, W, len(V8_PARAMS)), params.shape

    hdr = ddr_hdr.read_text()
    nb = int([l for l in hdr.splitlines() if l.strip().startswith("bands")][0].split("=")[1])
    ddr = np.fromfile(ddr_img, dtype="<f4").reshape(nb, H, W)
    elev = ddr[9].astype(np.float32)
    elev[elev >= 65535.0] = np.nan
    return {"labels": labels, "novel": novel, "params": params, "elev": elev, "files": files}


def masks_from_labels(labels: np.ndarray) -> dict:
    valid = labels >= 0
    m = {k: np.isin(labels, v) & valid for k, v in CLASS.items()}
    m["valid"] = valid
    return m


def distances(m: dict) -> dict:
    d = {}
    d["clay"] = distance_transform_edt(~m["clay"], sampling=(PIX_M, PIX_M))
    d["cap"] = distance_transform_edt(~m["cap"], sampling=(PIX_M, PIX_M))
    d["edge"] = distance_transform_edt(m["valid"], sampling=(PIX_M, PIX_M))
    return d


def profile(cube: str, contact: str, sel: np.ndarray, d: dict, m: dict, params: np.ndarray,
            elev: np.ndarray, variant: str) -> tuple[list[dict], dict]:
    other = "cap" if contact == "clay" else "clay"
    keep = sel & (d["edge"] > EDGE_EXCL_M) & (d[other] > OTHER_CONTACT_EXCL_M) & ~m[contact] & ~m[other]
    dist = d[contact][keep]
    carb = m["carb"][keep]
    vals = {p: params[..., i][keep] for i, p in enumerate(V8_PARAMS)}
    el = elev[keep]
    idx = np.digitize(dist, DIST_EDGES) - 1
    rows = []
    for b, lab in enumerate(DIST_LABELS):
        s = idx == b
        n = int(s.sum())
        row = {"cube": cube, "variant": variant, "contact": contact, "bin": lab, "bin_idx": b,
               "mixing_flag": b < MIX_BINS, "n": n,
               "carb_frac": float(carb[s].mean()) if n else np.nan,
               "elev_med": float(np.nanmedian(el[s])) if n else np.nan}
        if n:
            lo, hi = wilson(carb[s].sum(), n)
            row["carb_frac_lo"], row["carb_frac_hi"] = lo, hi
        for p in METRICS:
            v = vals[p][s]
            v = v[np.isfinite(v)]
            row[f"{p}_med"] = float(np.median(v)) if v.size else np.nan
            row[f"{p}_q25"] = float(np.percentile(v, 25)) if v.size else np.nan
            row[f"{p}_q75"] = float(np.percentile(v, 75)) if v.size else np.nan
        rows.append(row)

    # statistics beyond the mixing zone
    infer = (dist > DIST_EDGES[MIX_BINS]) & (dist <= 1200.0)
    st = {"cube": cube, "variant": variant, "contact": contact, "n_infer": int(infer.sum())}
    for p in ("BD2500_2", "D2300", "BD1900R2", "centroid"):
        v = vals[p][infer]
        ok = np.isfinite(v)
        if ok.sum() > 10:
            rho, pv = stats.spearmanr(dist[infer][ok], v[ok])
            st[f"spearman_{p}"] = float(rho)
            st[f"spearman_{p}_p"] = float(pv)
    near = (dist > NEAR_RANGE[0]) & (dist <= NEAR_RANGE[1])
    far = dist > FAR_MIN
    st["n_near"], st["n_far"] = int(near.sum()), int(far.sum())
    if near.sum() and far.sum():
        fn, ff = carb[near].mean(), carb[far].mean()
        st["carb_frac_near"], st["carb_frac_far"] = float(fn), float(ff)
        st["carb_near_far_ratio"] = float(fn / ff) if ff > 0 else np.inf
        st["carb_near_ci"] = wilson(carb[near].sum(), near.sum())
        st["carb_far_ci"] = wilson(carb[far].sum(), far.sum())
        for p in ("BD2500_2", "D2300", "BD1900R2"):
            a, b_ = vals[p][near], vals[p][far]
            a, b_ = a[np.isfinite(a)], b_[np.isfinite(b_)]
            st[f"{p}_near_med"], st[f"{p}_far_med"] = float(np.median(a)), float(np.median(b_))
            u = stats.mannwhitneyu(a, b_, alternative="two-sided")
            st[f"{p}_mwu_p"] = float(u.pvalue)
            # rank-biserial effect size
            st[f"{p}_rank_biserial"] = float(1 - 2 * u.statistic / (a.size * b_.size))
    return rows, st


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(c - h), float(c + h))


def elevation_profile(cube: str, sel: np.ndarray, m: dict, d: dict, params: np.ndarray,
                      elev: np.ndarray, variant: str) -> list[dict]:
    keep = sel & (d["edge"] > EDGE_EXCL_M) & np.isfinite(elev)
    el = elev[keep]
    carb = m["carb"][keep]
    bd = params[..., V8_PARAMS.index("BD2500_2")][keep]
    dclay = d["clay"][keep]
    edges = np.arange(np.floor(el.min() / ELEV_BIN_M) * ELEV_BIN_M, el.max() + ELEV_BIN_M, ELEV_BIN_M)
    idx = np.digitize(el, edges) - 1
    rows = []
    for b in range(len(edges) - 1):
        s = idx == b
        n = int(s.sum())
        if n < 50:
            continue
        rows.append({"cube": cube, "variant": variant, "elev_lo": float(edges[b]), "elev_hi": float(edges[b + 1]),
                     "n": n, "carb_frac": float(carb[s].mean()),
                     "BD2500_2_med": float(np.nanmedian(bd[s])),
                     "dist_clay_med": float(np.median(dclay[s]))})
    return rows


def plot_profiles(df: pd.DataFrame, variant: str, fname: str) -> None:
    sub = df[df.variant == variant]
    fig, axes = plt.subplots(5, 2, figsize=(10, 13), sharex=True)
    panels = [("carb_frac", "Fraction of olivine-bearing pixels\nin olivine-carbonate class (c0)"),
              ("BD2500_2_med", "BD2500_2 median\n(2.5 um carbonate band; higher = stronger)"),
              ("D2300_med", "D2300 median\n(2.3 um Mg-OH / carbonate)"),
              ("BD1900R2_med", "BD1900R2 median\n(1.9 um H2O)"),
              ("centroid_med", "1 um band centroid median (um)")]
    x = np.arange(len(DIST_LABELS))
    for col, (contact, title) in enumerate([("clay", "Distance to Fe/Mg-phyllosilicate contact"),
                                            ("cap", "Control: distance to mafic-cap contact")]):
        for row, (key, ylabel) in enumerate(panels):
            ax = axes[row, col]
            ax.axvspan(-0.5, MIX_BINS - 0.5, color="0.9", zorder=0)
            for cube in CUBE_IDS:
                s = sub[(sub.contact == contact) & (sub.cube == cube)].sort_values("bin_idx")
                s = s[s.n >= 100]
                ax.plot(s.bin_idx, s[key], "-o", color=PALETTE[cube], ms=5, lw=1.8, label=cube)
                if key == "carb_frac" and "carb_frac_lo" in s:
                    ax.fill_between(s.bin_idx, s.carb_frac_lo, s.carb_frac_hi, color=PALETTE[cube], alpha=0.15, lw=0)
            if row == 0:
                ax.set_title(title, fontsize=11)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True, color="0.92", lw=0.8)
            ax.spines[["top", "right"]].set_visible(False)
        axes[-1, col].set_xticks(x)
        axes[-1, col].set_xticklabels(DIST_LABELS, rotation=45, ha="right", fontsize=8)
        axes[-1, col].set_xlabel("Distance from contact (m; 18 m/pixel)")
    axes[0, 0].text(MIX_BINS - 0.6, axes[0, 0].get_ylim()[1], "sub-pixel\nmixing", ha="right", va="top", fontsize=7, color="0.4")
    axes[0, 0].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle(f"Olivine-bearing pixels vs. distance from contact ({variant})", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / f"{fname}.png", dpi=200)
    fig.savefig(OUT / f"{fname}.pdf")
    plt.close(fig)


def plot_maps(data: dict, fname: str) -> None:
    cmap = ListedColormap(["#f2f2f2", "#7f7f7f", "#2b2b2b", "#e6c78a", "#1b9e77"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    for ax, cube in zip(axes.ravel(), CUBE_IDS):
        m = data[cube]["masks"]
        d = data[cube]["dist"]
        img = np.zeros(m["valid"].shape, dtype=np.int8)
        img[m["clay"]] = 1
        img[m["cap"]] = 2
        img[m["oliv"]] = 3
        img[m["carb"]] = 4
        ax.imshow(img, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
        ax.contour(np.where(m["valid"], d["clay"], np.nan), levels=[150, 400], colors=["#d55e00", "#0072b2"], linewidths=0.6)
        ax.set_title(cube, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#7f7f7f", "#2b2b2b", "#e6c78a", "#1b9e77"]]
    fig.legend(handles + [plt.Line2D([0], [0], color="#d55e00"), plt.Line2D([0], [0], color="#0072b2")],
               ["Fe/Mg-phyllosilicate (c4+c5)", "mafic cap (c1)", "olivine (c2+c3)", "olivine-carbonate (c0)",
                "150 m from clay", "400 m from clay"],
               loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("v9.1-B GMM classes on the MTR3 grid with distance-to-clay contours", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(OUT / f"{fname}.png", dpi=200)
    fig.savefig(OUT / f"{fname}.pdf")
    plt.close(fig)


def plot_elevation(df: pd.DataFrame, variant: str, fname: str) -> None:
    sub = df[df.variant == variant]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for cube in CUBE_IDS:
        s = sub[sub.cube == cube].sort_values("elev_lo")
        xm = (s.elev_lo + s.elev_hi) / 2
        axes[0].plot(xm, s.carb_frac, "-o", ms=3, color=PALETTE[cube], label=cube)
        axes[1].plot(xm, s.BD2500_2_med, "-o", ms=3, color=PALETTE[cube])
        axes[2].plot(xm, s.dist_clay_med, "-o", ms=3, color=PALETTE[cube])
    axes[0].set_ylabel("olivine-carbonate (c0) fraction")
    axes[1].set_ylabel("BD2500_2 median")
    axes[2].set_ylabel("median distance to clay (m)")
    for ax in axes:
        ax.set_xlabel("MOLA elevation (m)")
        ax.grid(True, color="0.92")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Olivine-bearing pixels vs. elevation ({variant}); {ELEV_BIN_M:.0f} m bins, n>=50", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"{fname}.png", dpi=200)
    fig.savefig(OUT / f"{fname}.pdf")
    plt.close(fig)


def main() -> None:
    t0 = datetime.now(timezone.utc)
    data, prof_rows, stat_rows, elev_rows = {}, [], [], []
    for cube in CUBE_IDS:
        c = load_cube(cube)
        m = masks_from_labels(c["labels"])
        d = distances(m)
        data[cube] = {"masks": m, "dist": d}
        olx = c["params"][..., V8_PARAMS.index("OLINDEX3")]
        thr = np.nanpercentile(olx[m["valid"]], OLINDEX_PCTL)
        variants = {
            "all_valid": m["oliv"],
            "excl_novel": m["oliv"] & ~c["novel"],
            f"olindex3_gt_p{OLINDEX_PCTL:.0f}": m["valid"] & (olx > thr) & ~m["clay"] & ~m["cap"],
        }
        for variant, sel in variants.items():
            for contact in ("clay", "cap"):
                rows, st = profile(cube, contact, sel, d, m, c["params"], c["elev"], variant)
                prof_rows += rows
                stat_rows.append(st)
            elev_rows += elevation_profile(cube, sel, m, d, c["params"], c["elev"], variant)
        data[cube]["files"] = c["files"]
        data[cube]["olindex3_thr"] = float(thr)
        data[cube]["class_counts"] = {k: int(v.sum()) for k, v in m.items()}
        print(cube, "done; class px:", data[cube]["class_counts"], "OLINDEX3 thr", round(float(thr), 4))

    prof = pd.DataFrame(prof_rows)
    prof.to_csv(OUT / "distance_profiles.csv", index=False)
    statdf = pd.DataFrame(stat_rows)
    statdf.to_csv(OUT / "distance_stats.csv", index=False)
    elevdf = pd.DataFrame(elev_rows)
    elevdf.to_csv(OUT / "elevation_profiles.csv", index=False)

    for variant in prof.variant.unique():
        plot_profiles(prof, variant, f"fig_distance_profiles_{variant}")
        plot_elevation(elevdf, variant, f"fig_elevation_{variant}")
    plot_maps(data, "fig_contact_maps")

    log = {
        "script": Path(__file__).name, "script_md5": md5(Path(__file__)),
        "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
        "pixel_m": PIX_M, "dist_edges_m": DIST_EDGES.tolist(), "mixing_bins": MIX_BINS,
        "edge_exclusion_m": EDGE_EXCL_M, "other_contact_exclusion_m": OTHER_CONTACT_EXCL_M,
        "near_range_m": NEAR_RANGE, "far_min_m": FAR_MIN, "elev_bin_m": ELEV_BIN_M,
        "class_map": CLASS, "olindex3_percentile": OLINDEX_PCTL,
        "inputs": {cube: {k: {"path": str(p.relative_to(ROOT)), "md5": md5(p)} for k, p in data[cube]["files"].items()}
                   for cube in CUBE_IDS},
        "class_counts": {cube: data[cube]["class_counts"] for cube in CUBE_IDS},
        "olindex3_thresholds": {cube: data[cube]["olindex3_thr"] for cube in CUBE_IDS},
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": __import__("scipy").__version__,
                     "matplotlib": matplotlib.__version__, "python": sys.version.split()[0]},
    }
    (OUT / "run_log.json").write_text(json.dumps(log, indent=2, default=str))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
