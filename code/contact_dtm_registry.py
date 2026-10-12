"""Registry of every usable HiRISE DEM at Horseshoe, with georeferencing
overrides, so all fringe/scan/overlap scripts run off one list.

DEMs (2026-08-31):
- ADI_006712: Adriano's aligned ASP DEM of PSP_006712/005855 (0.8 m,
  GeoTIFF with LOCAL_CS and a .tfwx ArcGIS override -> CRS assigned from
  the ESP product, transform from the world file). PRIMARY for the south.
- ADI_027691: Adriano's geoid-aligned ASP DEM of ESP_027691/026992
  (0.8 m, proper Mars-2015-sphere CRS). New third area.
- PDS_006712 / PDS_002888: the PDS DTEEC products (1 m) with the Stage 0
  measured shifts. PDS_006712 carries pervasive TIN fill in its clay unit
  (facet caution); kept for cross-checks only.

The `shift` field is an additional (dx, dy) translation applied on top of
the base transform; for the PDS DEMs it is the Stage 0 measured value, for
Adriano's products it starts at (0, 0) and is updated after the DRG-vs-CTX
check (contact_new_dem_coreg.py writes new_dem_offsets.json).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parents[1]
from config import HIRISE_DEM_DIR as NEW, HIRISE_PDS_DIR as NPD  # noqa: E402
FRAME = ROOT / "outputs" / "revision" / "contact_3d" / "frame" / "coreg_offsets.json"
NEW_OFFSETS = ROOT / "outputs" / "revision" / "contact_3d" / "frame" / "new_dem_offsets.json"

MARS2015_EQC = CRS.from_wkt(
    'PROJCS["Mars_2015_Sphere_Equirectangular_clon0",GEOGCS["GCS_Mars_2015_Sphere",'
    'DATUM["D_Mars_2015_Sphere",SPHEROID["Mars_2015_Sphere",3396190,0]],'
    'PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Equirectangular"],PARAMETER["standard_parallel_1",0],'
    'PARAMETER["central_meridian",0],PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],UNIT["metre",1]]')


def _worldfile_affine(path: Path) -> Affine:
    a, d, b, e, c, f = [float(x) for x in path.read_text().split()[:6]]
    return Affine(a, b, c - a / 2 - b / 2, d, e, f - d / 2 - e / 2)


def _stage0_shift(dtm_key: str) -> tuple[float, float]:
    off = json.loads(FRAME.read_text())["hirise"][dtm_key]["measured"]
    return (off["dx_m"], off["dy_m"])


def _new_shift(name: str) -> tuple[float, float]:
    if NEW_OFFSETS.exists():
        d = json.loads(NEW_OFFSETS.read_text())
        if name in d and d[name].get("ok"):
            return (d[name]["dx_m"], d[name]["dy_m"])
    return (0.0, 0.0)


def registry() -> dict:
    return {
        "ADI_006712": {
            "dem": NEW / "PSP_006712_2020_PSP_005855_2020-DEM_aligned.tif",
            "ortho": NEW / "PSP_006712_2020_PSP_005855_2020-DRG.tif",
            "crs": MARS2015_EQC,
            "dem_transform": _worldfile_affine(NEW / "PSP_006712_2020_PSP_005855_2020-DEM_aligned.tfwx"),
            "ortho_transform": _worldfile_affine(NEW / "PSP_006712_2020_PSP_005855_2020-DRG.tfwx"),
            "shift": _new_shift("ADI_006712"),
            "quality": "primary (ASP, holes not facets)",
        },
        "ADI_027691": {
            "dem": NEW / "ESP_027691_2025_ESP_026992_2025-DEM_geoid-aligned.tif",
            "ortho": NEW / "ESP_027691_2025_ESP_026992_2025-DRG.tif",
            "crs": None,               # use file CRS (Mars 2015 sphere)
            "dem_transform": None,     # use file transform
            "ortho_transform": None,
            "shift": _new_shift("ADI_027691"),
            "quality": "primary (ASP, geoid-aligned)",
        },
        "PDS_002888": {
            "dem": NPD / "DTEEC_002888_2025_002176_2025_A01.IMG",
            "ortho": NPD / "PSP_002176_2025_RED_A_01_ORTHO.JP2",
            "crs": None, "dem_transform": None, "ortho_transform": None,
            "shift": _stage0_shift("DTEEC_002888_2025_002176_2025_A01"),
            "quality": "clean PDS DTEEC",
        },
        "PDS_006712": {
            "dem": NPD / "DTEEC_006712_2020_005855_2020_A01.IMG",
            "ortho": NPD / "PSP_006712_2020_RED_A_01_ORTHO.JP2",
            "crs": None, "dem_transform": None, "ortho_transform": None,
            "shift": _stage0_shift("DTEEC_006712_2020_005855_2020_A01"),
            "quality": "CAUTION pervasive TIN fill in clay unit",
        },
    }


def open_meta(entry: dict, which: str = "dem") -> tuple[Path, Affine, CRS, float | None]:
    """Effective (path, transform incl. shift, crs, nodata) for 'dem' or 'ortho'."""
    path = entry[which]
    with rasterio.open(path) as ds:
        tr = entry.get(f"{which}_transform") or ds.transform
        crs = entry.get("crs") or ds.crs
        nod = ds.nodata
    tr = Affine.translation(*entry["shift"]) * tr
    return path, tr, crs, nod


def read_dem_full(entry: dict) -> tuple[np.ndarray, Affine, CRS]:
    path, tr, crs, nod = open_meta(entry, "dem")
    with rasterio.open(path) as ds:
        z = ds.read(1).astype(np.float32)
    if nod is not None:
        z[z == nod] = np.nan
    z[(z < -1e30) | (z > 1e30)] = np.nan
    return z, tr, crs
