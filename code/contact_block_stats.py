"""Post-verification statistics for the contact-3D analysis.

1. Block-bootstrap confidence intervals (504 m blocks, 1000 draws, seed 42)
   and correctly-signed Cliff's delta (positive = near-contact higher) for
   the near (36-150 m) vs far (> 400 m) comparison of D2300, BD1900R2 and
   BD2500_2, map-pure olivine population, clay and cap contacts, all cubes.
2. Overlap split: the same clay-contact statistic for B438 inside/outside
   the 3E12 footprint and vice versa.
3. Contact-step test: elevation difference between the clay side and the
   olivine side of the mapped contact, paired by nearest contact cell, on
   the CTX DEM (18 m grid) and on the two HiRISE DTMs (3 m grid, measured
   shifts applied), in bands 6-24 m and 30-60 m from the contact line.

Usage: python code/contact_block_stats.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
import rasterio
import shapely
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.warp import reproject, transform as rtransform
from scipy import stats
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import CUBE_IDS  # noqa: E402
from contact_distance_to_contact import PIX_M, V8_PARAMS, load_cube, masks_from_labels  # noqa: E402
from contact_height_above_contact import CTX_DEM, HIRISE, warp_dem  # noqa: E402

GEOMAP = ROOT / "outputs" / "revision" / "contact_3d" / "geomap" / "frame_ctx"
FRAME = ROOT / "outputs" / "revision" / "contact_3d" / "frame" / "coreg_offsets.json"
from config import GEOMAP_SOURCE as GDB  # noqa: E402
OUT = ROOT / "outputs" / "revision" / "contact_3d" / "stage5_stats"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(42)
BLOCK_PX, NBOOT = 28, 1000
NEAR, FAR_MIN = (36.1, 150.0), 400.0
METRICS = ("D2300", "BD1900R2", "BD2500_2")


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Positive = a stochastically larger than b."""
    u1 = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2.0 * u1 / (a.size * b.size) - 1.0)


def block_ids(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    return (rows // BLOCK_PX) * 100000 + (cols // BLOCK_PX)


def block_boot_diff(vn: np.ndarray, bn: np.ndarray, vf: np.ndarray, bf: np.ndarray) -> dict:
    """Bootstrap blocks; statistic = median(near) - median(far)."""
    blocks = np.unique(np.concatenate([bn, bf]))
    stat = float(np.median(vn) - np.median(vf))
    if blocks.size < 5:
        return {"diff": stat, "ci_lo": np.nan, "ci_hi": np.nan, "n_blocks": int(blocks.size)}
    per_n = {b: vn[bn == b] for b in np.unique(bn)}
    per_f = {b: vf[bf == b] for b in np.unique(bf)}
    boots = []
    for _ in range(NBOOT):
        pick = RNG.choice(blocks, blocks.size, replace=True)
        n = np.concatenate([per_n[b] for b in pick if b in per_n]) if any(b in per_n for b in pick) else np.array([])
        f = np.concatenate([per_f[b] for b in pick if b in per_f]) if any(b in per_f for b in pick) else np.array([])
        if n.size > 20 and f.size > 20:
            boots.append(np.median(n) - np.median(f))
    boots = np.array(boots)
    # within-block paired Wilcoxon on blocks holding both populations
    both = [b for b in blocks if b in per_n and b in per_f and per_n[b].size >= 10 and per_f[b].size >= 10]
    paired = np.array([np.median(per_n[b]) - np.median(per_f[b]) for b in both])
    wil_p = float(stats.wilcoxon(paired).pvalue) if paired.size >= 8 else np.nan
    return {"diff": stat, "ci_lo": float(np.percentile(boots, 2.5)), "ci_hi": float(np.percentile(boots, 97.5)),
            "n_blocks": int(blocks.size), "n_paired_blocks": len(both),
            "paired_median_diff": float(np.median(paired)) if paired.size else np.nan, "paired_wilcoxon_p": wil_p}


def cube_setup(cube: str, c: dict):
    m = masks_from_labels(c["labels"])
    fz = np.load(GEOMAP / f"{cube}_fractions.npz")
    fr = {k: fz[k] for k in ("clay", "cap", "dune", "graben", "oliv", "outline")}
    tr = Affine.from_gdal(*fz["transform"])
    mapped = fr["outline"] >= 0.99
    d = {"clay": distance_transform_edt(~(fr["clay"] >= 0.5), sampling=(PIX_M, PIX_M)),
         "cap": distance_transform_edt(~(fr["cap"] >= 0.5), sampling=(PIX_M, PIX_M)),
         "edge": distance_transform_edt(m["valid"] & mapped, sampling=(PIX_M, PIX_M))}
    pure = m["valid"] & mapped & (fr["oliv"] >= 0.9) & (fr["clay"] == 0) & ~m["clay"] & (d["edge"] > 90)
    return m, fr, tr, d, pure


def near_far(cube: str, c: dict, d: dict, pure: np.ndarray, contact: str, other: str, restrict: np.ndarray | None = None) -> list[dict]:
    sel = pure & (d[other] > 90)
    if restrict is not None:
        sel = sel & restrict
    rows_, cols_ = np.nonzero(sel)
    dist = d[contact][sel]
    bid = block_ids(rows_, cols_)
    out = []
    for p in METRICS:
        v = c["params"][..., V8_PARAMS.index(p)][sel]
        ok = np.isfinite(v)
        near = ok & (dist > NEAR[0]) & (dist <= NEAR[1]); far = ok & (dist > FAR_MIN)
        if near.sum() < 100 or far.sum() < 100:
            continue
        r = block_boot_diff(v[near], bid[near], v[far], bid[far])
        r.update({"cube": cube, "contact": contact, "param": p, "n_near": int(near.sum()), "n_far": int(far.sum()),
                  "near_med": float(np.median(v[near])), "far_med": float(np.median(v[far])),
                  "cliffs_delta_near_gt_far": cliffs_delta(v[near], v[far])})
        out.append(r)
    return out


def other_cube_footprint(this_tr: Affine, this_shape, other_cube: str, others: dict) -> np.ndarray:
    """Both grids share one CRS: map pixel centres of this grid into the other grid's valid mask."""
    om, otr, oshape = others[other_cube]
    H, W = this_shape
    cc, rr = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    xs, ys = this_tr * (cc, rr)
    oc, orow = ~otr * (xs, ys)
    oc = np.floor(oc).astype(int); orow = np.floor(orow).astype(int)
    inside = (orow >= 0) & (orow < oshape[0]) & (oc >= 0) & (oc < oshape[1])
    out = np.zeros(this_shape, bool)
    out[inside] = om[orow[inside], oc[inside]]
    return out


def contact_step_grid(z: np.ndarray, clay: np.ndarray, oliv: np.ndarray, res: float, bands) -> list[dict]:
    """Pair clay-side and olivine-side elevations by nearest contact cell."""
    st = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    from scipy.ndimage import binary_dilation
    contact = oliv & binary_dilation(clay, st)      # single-sided line so both sides share pairing keys
    if contact.sum() < 10:
        return []
    dcont, (ir, ic) = distance_transform_edt(~contact, sampling=(res, res), return_indices=True)
    key = ir.astype(np.int64) * z.shape[1] + ic
    out = []
    for lo, hi in bands:
        res_rows = {}
        for side, mask in (("clay", clay), ("oliv", oliv)):
            s = mask & (dcont >= lo) & (dcont <= hi) & np.isfinite(z)
            df = pd.DataFrame({"k": key[s], "z": z[s]})
            res_rows[side] = df.groupby("k")["z"].median()
        j = pd.concat([res_rows["clay"].rename("zc"), res_rows["oliv"].rename("zo")], axis=1).dropna()
        if len(j) < 30:
            continue
        dz = (j.zc - j.zo).to_numpy()
        out.append({"band_m": f"{lo:.0f}-{hi:.0f}", "n_pairs": len(j), "dz_clay_minus_oliv_med": float(np.median(dz)),
                    "dz_q25": float(np.percentile(dz, 25)), "dz_q75": float(np.percentile(dz, 75)),
                    "frac_oliv_higher_2m": float((dz < -2).mean()), "frac_clay_higher_2m": float((dz > 2).mean())})
    return out


def hirise_contact_step(dtm: str, path: Path, offsets: dict) -> list[dict]:
    mres = offsets["hirise"][dtm]["measured"]
    with rasterio.open(path) as ds:
        dcrs, tr0 = ds.crs, ds.transform
        f = max(1, int(round(3.0 / abs(tr0.a))))
        z = ds.read(1, out_shape=(ds.height // f, ds.width // f), resampling=Resampling.average).astype(np.float32)
        nd = ds.nodata
        if nd is not None:
            z[z == nd] = np.nan
        z[(z < -1e30) | (z > 1e30)] = np.nan
        tr3 = Affine.translation(mres["dx_m"], mres["dy_m"]) * tr0 * Affine.scale(f)
    def load(layer):
        lcrs = CRS.from_wkt(pyogrio.read_info(GDB, layer=layer)["crs"])   # per-layer CRS: Olivine_unit differs from CaSSIS_units
        _, _, g, _ = pyogrio.raw.read(GDB, layer=layer)
        gs = [shapely.force_2d(x) for x in shapely.from_wkb(g)]
        return [shapely.make_valid(shapely.transform(x, lambda xy: np.column_stack(rtransform(lcrs, dcrs, xy[:, 0], xy[:, 1])))) for x in gs]
    units = {"clay": load("Clay_unit"), "cap": load("Mafic_cap"), "dune": load("olivine_dunes"),
             "graben": load("graben_materials"), "outline": load("Olivine_unit")}
    codes = {"outline": 1, "graben": 2, "clay": 3, "cap": 4, "dune": 5}
    shapes = [(g, codes[k]) for k in ("outline", "graben", "clay", "cap", "dune") for g in units[k]]
    lab = rasterize(shapes, out_shape=z.shape, transform=tr3, fill=0, dtype="uint8", all_touched=False)
    rows = contact_step_grid(z, lab == 3, lab == 1, abs(tr3.a), [(6, 24), (30, 60)])
    for r in rows:
        r.update({"grid": dtm, "res_m": abs(tr3.a)})
    return rows


def main() -> None:
    t0 = datetime.now(timezone.utc)
    offsets = json.loads(FRAME.read_text())
    cubes = {cube: load_cube(cube) for cube in CUBE_IDS}
    setups = {cube: cube_setup(cube, cubes[cube]) for cube in CUBE_IDS}
    footprints = {cube: (setups[cube][0]["valid"], setups[cube][2], cubes[cube]["labels"].shape) for cube in CUBE_IDS}

    rows = []
    for cube in CUBE_IDS:
        m, fr, tr, d, pure = setups[cube]
        rows += near_far(cube, cubes[cube], d, pure, "clay", "cap")
        rows += near_far(cube, cubes[cube], d, pure, "cap", "clay")
    # overlap split (footprint masks are in the same CRS; reproject valid masks between grids)
    for this, other in (("FRT0000B438", "FRT00003E12"), ("FRT00003E12", "FRT0000B438")):
        m, fr, tr, d, pure = setups[this]
        fp = other_cube_footprint(tr, cubes[this]["labels"].shape, other, footprints)
        for tag, restrict in (("inside_" + other, fp), ("outside_" + other, ~fp)):
            for r in near_far(this, cubes[this], d, pure, "clay", "cap", restrict=restrict):
                r["subset"] = tag; rows.append(r)
    bb = pd.DataFrame(rows)
    bb.to_csv(OUT / "block_bootstrap_near_far.csv", index=False)
    print(bb[[c for c in ("cube", "contact", "param", "subset", "near_med", "far_med", "diff", "ci_lo", "ci_hi",
                          "cliffs_delta_near_gt_far", "n_blocks", "n_paired_blocks", "paired_wilcoxon_p") if c in bb]].round(5).to_string())

    steps = []
    for cube in CUBE_IDS:
        m, fr, tr, d, pure = setups[cube]
        with rasterio.open(sorted((ROOT / "outputs" / "_v8_mtr3_with_brown" / cube / "geotiffs").glob("*.tif"))[0]) as ds:
            crs = ds.crs
        z = warp_dem(CTX_DEM, (0.0, 0.0), crs, tr, cubes[cube]["labels"].shape, Resampling.bilinear)
        others = (fr["cap"] + fr["dune"] + fr["graben"]) >= 0.1
        for r in contact_step_grid(z, (fr["clay"] >= 0.5) & ~others, (fr["oliv"] >= 0.5) & ~others, PIX_M, [(18, 36), (54, 108)]):
            r.update({"grid": f"{cube}_ctx18m", "res_m": PIX_M}); steps.append(r)
    for dtm, path in HIRISE.items():
        steps += hirise_contact_step(dtm, path, offsets)
    sdf = pd.DataFrame(steps)
    sdf.to_csv(OUT / "contact_step.csv", index=False)
    print(sdf.round(3).to_string())
    (OUT / "run_log.json").write_text(json.dumps({"block_px": BLOCK_PX, "nboot": NBOOT, "near": NEAR, "far_min": FAR_MIN,
        "seed": 42, "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
