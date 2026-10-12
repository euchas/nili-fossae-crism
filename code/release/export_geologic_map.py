"""Export the hand-drawn Horseshoe geologic map to an open GeoPackage for Zenodo.

All File Geodatabase layers and the stand-alone mapping shapefiles go into
one GeoPackage (one layer each, CRS and attributes preserved, Z kept, M
dropped). The original File GDB and shapefiles are also copied unchanged
for ArcGIS users. Writes a JSON log with layer names, geometry types,
feature counts and CRS.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pyogrio
from pyogrio.raw import read, write

from release_config import PROJECT_ROOT, STAGING_DIR

MAP_DIR = PROJECT_ROOT / "Nili Planum" / "CaSSIS-CRISM-Nili_Fossae" / "CRISM site Creta Vulcanica"
GDB = MAP_DIR / "Nili fossae mapping.gdb"
SHAPEFILES = (
    "dike map", "amorphous matrix", "best ridge sites", "bright exposures",
    "quasi circular features", "rupture_locations", "Profile 1", "Profile 2", "Profile 3",
)
SHP_SIDECARS = (".shp", ".shx", ".dbf", ".prj", ".cpg", ".shp.xml")
OUT_DIR = STAGING_DIR / "A_geologic_map"
GPKG = OUT_DIR / "Horseshoe_geologic_map.gpkg"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def layer_name(name: str) -> str:
    return name.replace(" ", "_")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if GPKG.exists():
        GPKG.unlink()
    log = {"gpkg": GPKG.name, "layers": []}
    sources = [(GDB, lyr) for lyr, _ in pyogrio.list_layers(GDB)]
    sources += [(MAP_DIR / f"{s}.shp", None) for s in SHAPEFILES]
    for path, lyr in sources:
        meta, _, geom, fields = read(path, layer=lyr) if lyr else read(path)
        out_name = layer_name(lyr or path.stem)
        if meta["geometry_type"] is None or geom is None or len(geom) == 0:
            continue
        write(GPKG, geom, fields, meta["fields"], layer=out_name, driver="GPKG",
              crs=meta["crs"], geometry_type=meta["geometry_type"], encoding="UTF-8",
              append=GPKG.exists())
        log["layers"].append({
            "layer": out_name, "source": path.name + (f"::{lyr}" if lyr else ""),
            "n_features": len(geom), "geometry_type": meta["geometry_type"],
            "columns": list(meta["fields"]), "crs": meta["crs"],
        })
        print(f"{out_name:30s} {len(geom):5d}  {meta['geometry_type']}")
    log["gpkg_sha256"] = sha256(GPKG)

    gdb_out = OUT_DIR / GDB.name
    if gdb_out.exists():
        shutil.rmtree(gdb_out)
    shutil.copytree(GDB, gdb_out, ignore=shutil.ignore_patterns("*.lock"))
    log["original_gdb"] = gdb_out.name
    shp_dir = OUT_DIR / "shapefiles"
    shp_dir.mkdir(exist_ok=True)
    for s in SHAPEFILES:
        for ext in SHP_SIDECARS:
            f = MAP_DIR / f"{s}{ext}"
            if f.exists():
                shutil.copy2(f, shp_dir / f.name)

    (OUT_DIR / "_logs").mkdir(exist_ok=True)
    (OUT_DIR / "_logs" / "export_geologic_map.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
