"""Stage 1a of PLAN_revision_contact_3D.md: sub-pixel unit fractions.

Rasterises the geologic map units (Nili fossae mapping.gdb, CaSSIS_units)
at 3 m in the MTR3 CRS and block-averages them to the 18 m CRISM grid of
each cube, giving per-pixel area fractions f_clay, f_cap, f_dune, f_graben,
f_outline and f_oliv = f_outline - (others). Optionally applies the Stage 0
CRISM-vs-CTX shift so that the fractions describe what each CRISM pixel
actually covers in the CTX/map frame.

Usage: python code/contact_geomap_fractions.py [--frame pds|ctx]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyogrio
import rasterio
import shapely
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.warp import transform as rtransform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import CUBE_IDS  # noqa: E402

from config import GEOMAP_SOURCE as GDB  # noqa: E402
OUT = ROOT / "outputs" / "revision" / "contact_3d" / "geomap"
FRAME_JSON = ROOT / "outputs" / "revision" / "contact_3d" / "frame" / "coreg_offsets.json"
SUB = 6                      # 18 m / 6 = 3 m rasterisation
UNITS = {"clay": "Clay_unit", "cap": "Mafic_cap", "dune": "olivine_dunes", "graben": "graben_materials", "outline": "Olivine_unit"}
MIN_SHIFT_PX = 0.5


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_units(dst_crs: CRS) -> dict[str, list]:
    out = {}
    for key, layer in UNITS.items():
        info = pyogrio.read_info(GDB, layer=layer)
        src = CRS.from_wkt(info["crs"])
        _, _, geoms, _ = pyogrio.raw.read(GDB, layer=layer)
        gs = [shapely.force_2d(g) for g in shapely.from_wkb(geoms)]
        gs = [shapely.transform(g, lambda xy: np.column_stack(rtransform(src, dst_crs, xy[:, 0], xy[:, 1]))) for g in gs]
        gs = [shapely.make_valid(g) for g in gs]
        out[key] = gs
    return out


def fractions_for_grid(units: dict, tr: Affine, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Single label raster at SUB x resolution painted in stacking order (later wins), then fractions."""
    H, W = shape
    sub_tr = tr * Affine.scale(1.0 / SUB)
    codes = {"outline": 1, "graben": 2, "clay": 3, "cap": 4, "dune": 5}      # paint order = stratigraphic priority
    shapes = [(g, codes[k]) for k in ("outline", "graben", "clay", "cap", "dune") for g in units[k]]
    lab = rasterize(shapes, out_shape=(H * SUB, W * SUB), transform=sub_tr, fill=0, dtype="uint8", all_touched=False)
    lab4 = lab.reshape(H, SUB, W, SUB)
    fr = {k: (lab4 == c).mean(axis=(1, 3)).astype(np.float32) for k, c in codes.items()}
    fr["oliv"] = fr.pop("outline")                       # outline remainder = olivine unit
    fr["outline"] = (lab4 > 0).mean(axis=(1, 3)).astype(np.float32)
    fr["overlap_excess"] = np.zeros((H, W), np.float32)   # resolved by stacking; kept for file compatibility
    return fr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", choices=["pds", "ctx"], default="ctx", help="ctx: apply Stage 0 CRISM shift (if confident and > 0.5 px)")
    a = ap.parse_args()
    t0 = datetime.now(timezone.utc)
    out = OUT / f"frame_{a.frame}"
    out.mkdir(parents=True, exist_ok=True)
    offsets = json.loads(FRAME_JSON.read_text())["crism"] if (a.frame == "ctx" and FRAME_JSON.exists()) else {}
    log = {"script_md5": md5(Path(__file__)), "gdb": GDB.name, "sub_px": SUB, "frame": a.frame,
           "started_utc": t0.isoformat(), "cubes": {}}
    units_cache = {}
    for cube in CUBE_IDS:
        gt = sorted(glob.glob(str(ROOT / "outputs" / "_v8_mtr3_with_brown" / cube / "geotiffs" / "*.tif")))[0]
        with rasterio.open(gt) as ds:
            crs, tr, shape = ds.crs, ds.transform, ds.shape
        key = crs.to_wkt()
        if key not in units_cache:
            units_cache[key] = load_units(crs)
        shift = {"dx_m": 0.0, "dy_m": 0.0, "applied": False}
        if cube in offsets and offsets[cube].get("ok"):
            o = offsets[cube]
            px = np.hypot(o["drow"], o["dcol"])
            if px >= MIN_SHIFT_PX and o["conf_median"] >= 0.3:
                shift = {"dx_m": o["dx_m"], "dy_m": o["dy_m"], "applied": True}
        tr_eff = Affine.translation(shift["dx_m"], shift["dy_m"]) * tr
        fr = fractions_for_grid(units_cache[key], tr_eff, shape)
        np.savez_compressed(out / f"{cube}_fractions.npz", **fr, transform=np.array(tr_eff.to_gdal()), shift_dx_m=shift["dx_m"], shift_dy_m=shift["dy_m"])
        bands = ["clay", "cap", "dune", "graben", "outline", "oliv"]
        with rasterio.open(out / f"{cube}_unit_fractions.tif", "w", driver="GTiff", width=shape[1], height=shape[0], count=len(bands),
                           dtype="float32", crs=crs, transform=tr_eff, nodata=-1.0, compress="deflate") as dst:
            for i, b in enumerate(bands, 1):
                dst.write(fr[b], i); dst.set_band_description(i, f"f_{b}")
        stats = {b: {"mean": float(fr[b].mean()), "frac_ge_0.9": float((fr[b] >= 0.9).mean())} for b in bands}
        stats["overlap_excess_max"] = float(fr["overlap_excess"].max())
        log["cubes"][cube] = {"grid_from": str(Path(gt).relative_to(ROOT)), "shape": shape, "shift": shift, "stats": stats}
        print(cube, "shift", shift, {b: round(stats[b]["mean"], 3) for b in bands}, "overlap_excess_max", round(stats["overlap_excess_max"], 3))
    log["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (out / "run_log.json").write_text(json.dumps(log, indent=2, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()
