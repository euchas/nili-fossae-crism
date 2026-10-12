"""Losslessly recompress HiRISE DEMs and orthoimages (A. Tullo) for Zenodo.

Values are copied bit-for-bit (float32); only the container changes:
tiled, ZSTD + floating-point predictor, internal overviews. A full-array
equality check is run on every output. Re-runnable.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

from release_config import HIRISE_DEM_DIR, STAGING_DIR

FILES = (
    "PSP_006712_2020_PSP_005855_2020-DEM_aligned.tif",
    "PSP_006712_2020_PSP_005855_2020-DRG.tif",
    "ESP_027691_2025_ESP_026992_2025-DEM_geoid-aligned.tif",
    "ESP_027691_2025_ESP_026992_2025-DRG.tif",
)
ROWS_PER_CHUNK = 1024
OVERVIEW_LEVELS = (2, 4, 8, 16, 32)
OUT_DIR = STAGING_DIR / "G_HiRISE_DEMs_Tullo"


def sha256(path: Path, bufsize: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def chunks(ds):
    for row0 in range(0, ds.height, ROWS_PER_CHUNK):
        yield Window(0, row0, ds.width, min(ROWS_PER_CHUNK, ds.height - row0))


def recompress(src_path: Path, dst_path: Path) -> None:
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(driver="GTiff", tiled=True, blockxsize=512, blockysize=512,
                       compress="zstd", zstd_level=9, predictor=3, bigtiff="IF_SAFER")
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.descriptions = src.descriptions
            dst.update_tags(**src.tags(), SOURCE_FILE=src_path.name)
            for win in chunks(src):
                dst.write(src.read(window=win), window=win)
    with rasterio.Env(COMPRESS_OVERVIEW="ZSTD", PREDICTOR_OVERVIEW=3, BIGTIFF_OVERVIEW="IF_SAFER"):
        with rasterio.open(dst_path, "r+") as dst:
            dst.build_overviews(list(OVERVIEW_LEVELS), Resampling.average)


def identical(a_path: Path, b_path: Path) -> bool:
    with rasterio.open(a_path) as a, rasterio.open(b_path) as b:
        if (a.crs, a.transform, a.shape, a.nodata) != (b.crs, b.transform, b.shape, b.nodata):
            return False
        return all(np.array_equal(a.read(window=w), b.read(window=w), equal_nan=True) for w in chunks(a))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = OUT_DIR / "_logs"
    log_dir.mkdir(exist_ok=True)
    for name in FILES:
        src, dst = HIRISE_DEM_DIR / name, OUT_DIR / name
        log_path = log_dir / (dst.stem + ".json")
        if dst.exists() and log_path.exists():
            print(f"skip {name}", flush=True)
            continue
        t0 = time.time()
        tmp = dst.with_suffix(".tmp.tif")
        recompress(src, tmp)
        if not identical(src, tmp):
            raise RuntimeError(f"lossless check failed for {name}")
        tmp.replace(dst)
        rec = {"source": str(src), "source_bytes": src.stat().st_size, "output": dst.name,
               "output_bytes": dst.stat().st_size, "output_sha256": sha256(dst),
               "lossless_verified": True, "overviews": list(OVERVIEW_LEVELS),
               "seconds": round(time.time() - t0, 1), "rasterio": rasterio.__version__,
               "gdal": rasterio.__gdal_version__}
        log_path.write_text(json.dumps(rec, indent=2))
        print(f"{name}: {dst.stat().st_size / 1e9:.2f} GB "
              f"({100 * dst.stat().st_size / src.stat().st_size:.0f}%), verified, {rec['seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
