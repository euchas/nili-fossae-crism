"""Convert pansharpened CaSSIS-HiRISE GeoTIFFs (float32 I/F) to scaled uint16 for Zenodo.

Encoding: DN = round(I/F / SCALE), I/F = DN * SCALE, nodata = 65535.
Output is tiled, ZSTD-compressed GeoTIFF with internal overviews; band
names, georeferencing and scale factor are preserved in the file metadata.
Re-runnable: scenes whose output and log already exist are skipped.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

from release_config import PANSHARP_DIR, PANSHARP_SCENES, STAGING_DIR

SCALE = 2e-5
NODATA_OUT = 65535
MAX_DN = 65534
ROWS_PER_CHUNK = 512
OVERVIEW_LEVELS = (2, 4, 8, 16, 32, 64)
OUT_DIR = STAGING_DIR / "F_pansharpened_CaSSIS_HiRISE"


def sha256(path: Path, bufsize: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def output_name(src_name: str) -> str:
    return Path(src_name).stem + "_u16.tif"


def convert(src_path: Path, dst_path: Path) -> dict:
    stats = {"n_valid": 0, "n_nodata": 0, "n_clipped_low": 0, "n_clipped_high": 0,
             "min": np.inf, "max": -np.inf, "max_abs_quant_err": 0.0}
    with rasterio.open(src_path) as src:
        nd_in = src.nodata
        profile = src.profile.copy()
        profile.update(dtype="uint16", nodata=NODATA_OUT, driver="GTiff", tiled=True,
                       blockxsize=512, blockysize=512, compress="zstd", zstd_level=9,
                       predictor=2, bigtiff="IF_SAFER", interleave="pixel")
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.descriptions = src.descriptions
            dst.scales = [SCALE] * src.count
            dst.offsets = [0.0] * src.count
            dst.update_tags(
                SOURCE_FILE=src_path.name,
                ENCODING=f"I/F = DN * {SCALE}; nodata DN = {NODATA_OUT}",
                PRODUCT="Pansharpened CaSSIS-HiRISE (Valantinas et al., EPSL)",
            )
            for row0 in range(0, src.height, ROWS_PER_CHUNK):
                win = Window(0, row0, src.width, min(ROWS_PER_CHUNK, src.height - row0))
                a = src.read(window=win)
                bad = ~np.isfinite(a)
                if nd_in is not None:
                    bad |= a == nd_in
                good = ~bad
                v = a[good]
                if v.size:
                    stats["min"] = min(stats["min"], float(v.min()))
                    stats["max"] = max(stats["max"], float(v.max()))
                    stats["n_clipped_low"] += int((v < 0).sum())
                    stats["n_clipped_high"] += int((v > MAX_DN * SCALE).sum())
                dn = np.full(a.shape, NODATA_OUT, dtype=np.uint16)
                q = np.clip(np.rint(v / SCALE), 0, MAX_DN)
                dn[good] = q.astype(np.uint16)
                if v.size:
                    err = np.abs(q * SCALE - np.clip(v, 0, MAX_DN * SCALE))
                    stats["max_abs_quant_err"] = max(stats["max_abs_quant_err"], float(err.max()))
                stats["n_valid"] += int(good.sum())
                stats["n_nodata"] += int(bad.sum())
                dst.write(dn, window=win)
        info = {"width": src.width, "height": src.height, "count": src.count,
                "res_m": list(src.res), "crs": src.crs.to_wkt() if src.crs else None,
                "bands": list(src.descriptions), "source_dtype": src.dtypes[0],
                "source_nodata": nd_in}
    with rasterio.Env(COMPRESS_OVERVIEW="ZSTD", ZSTD_LEVEL_OVERVIEW=9, PREDICTOR_OVERVIEW=2,
                      BIGTIFF_OVERVIEW="IF_SAFER"):
        with rasterio.open(dst_path, "r+") as dst:
            dst.build_overviews(list(OVERVIEW_LEVELS), Resampling.average)
            dst.update_tags(ns="rio_overview", resampling="average")
    return {**info, **stats}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = OUT_DIR / "_logs"
    log_dir.mkdir(exist_ok=True)
    for name in PANSHARP_SCENES:
        src = PANSHARP_DIR / name
        dst = OUT_DIR / output_name(name)
        log_path = log_dir / (dst.stem + ".json")
        if dst.exists() and log_path.exists():
            print(f"skip {name}", flush=True)
            continue
        t0 = time.time()
        print(f"convert {name} ...", flush=True)
        tmp = dst.with_suffix(".tmp.tif")
        result = convert(src, tmp)
        tmp.replace(dst)
        record = {
            "source": str(src), "source_bytes": src.stat().st_size,
            "source_mtime": src.stat().st_mtime, "output": dst.name,
            "output_bytes": dst.stat().st_size, "output_sha256": sha256(dst),
            "scale": SCALE, "nodata": NODATA_OUT, "overviews": list(OVERVIEW_LEVELS),
            "seconds": round(time.time() - t0, 1),
            "python": sys.version, "platform": platform.platform(),
            "rasterio": rasterio.__version__, "gdal": rasterio.__gdal_version__,
            "numpy": np.__version__, **result,
        }
        log_path.write_text(json.dumps(record, indent=2))
        print(f"  done {dst.stat().st_size / 1e9:.2f} GB "
              f"({100 * dst.stat().st_size / src.stat().st_size:.0f}% of source), "
              f"range [{result['min']:.4f}, {result['max']:.4f}], "
              f"clipped {result['n_clipped_low']}/{result['n_clipped_high']}, "
              f"{record['seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
