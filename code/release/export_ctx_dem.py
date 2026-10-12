"""Export the cleaned and interpolated CTX DEMs (P03_002176 / P05_002888) to GeoTIFF for Zenodo.

The interpolated DEM lives as a raster inside `Isidis Project.gdb` and is the
CTX DEM used by the contact-metamorphism scripts (config.CTX_DEM). The cleaned
GeoTIFF has a LOCAL_CS and an unaligned transform; ArcGIS positions it with
its .tfwx world file, so that world file (pixel-centre convention) and the
interpolated DEM's Mars CRS are written into the exported GeoTIFF, and the
.tfwx is shipped alongside. Values are copied bit-for-bit and verified.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

from release_config import PROJECT_ROOT, STAGING_DIR

GIS = PROJECT_ROOT / "Nili Planum" / "CaSSIS-CRISM-Nili_Fossae"
STEM = "P03_002176_2024_XI_22N283W__P05_002888_2025_XI_22N283W"
SOURCES = {
    f"{STEM}_dtm_interpolate.tif":
        f"OpenFileGDB:{GIS / 'Isidis Project' / 'Isidis Project.gdb'}:{STEM}_dtm_interpolate",
    f"{STEM}_dtm_cleaned.tif": str(GIS / "DTMs" / "CTX" / f"{STEM}_dtm_cleaned.tif"),
}
OUT_DIR = STAGING_DIR / "H_CTX_DEM"


def worldfile_affine(path: Path) -> Affine:
    """Affine (pixel-corner) from a 6-line world file (pixel-centre convention)."""
    a, d, b, e, c, f = (float(v) for v in path.read_text().split())
    return Affine(a, b, c, d, e, f) * Affine.translation(-0.5, -0.5)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = []
    with rasterio.open(SOURCES[f"{STEM}_dtm_interpolate.tif"]) as ref:
        mars_crs = ref.crs
    for name, src_path in SOURCES.items():
        dst = OUT_DIR / name
        tfwx = Path(src_path).with_suffix(".tfwx") if not src_path.startswith("OpenFileGDB") else None
        with rasterio.open(src_path) as src:
            data = src.read()
            profile = src.profile.copy()
            if tfwx is not None and tfwx.exists():
                profile.update(transform=worldfile_affine(tfwx), crs=mars_crs)
                shutil.copy2(tfwx, OUT_DIR / tfwx.name)
            profile.update(driver="GTiff", tiled=True, blockxsize=256, blockysize=256,
                           compress="zstd", predictor=3)
            with rasterio.open(dst, "w", **profile) as o:
                o.write(data)
        with rasterio.open(dst) as chk:
            assert np.array_equal(chk.read(), data, equal_nan=True)
        log.append({"file": name, "source": src_path, "shape": list(data.shape[1:]),
                    "georef": f"world file {tfwx.name} + CRS of interpolated DEM" if tfwx and tfwx.exists() else "as source",
                    "nodata": profile["nodata"], "lossless_verified": True,
                    "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
                    "bytes": dst.stat().st_size})
        print(f"{name}: {dst.stat().st_size / 1e6:.1f} MB, verified")
    (OUT_DIR / "_logs").mkdir(exist_ok=True)
    (OUT_DIR / "_logs" / "export_ctx_dem.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
