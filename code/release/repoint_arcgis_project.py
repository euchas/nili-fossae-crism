"""Copy the paper ArcGIS Pro project into the Zenodo package and repoint its layers.

Run with ArcGIS Pro's Python (arcpy), e.g.
  "C:\\Program Files\\ArcGIS\\Pro\\bin\\Python\\envs\\arcgispro-py3\\python.exe" repoint_arcgis_project.py

Layers whose data are in the package (geologic map, CRISM parameter maps,
Brown maps, pansharpened scenes, DEMs) are repointed to the staged copies so
the project resolves them by relative path once all sections are unzipped
side by side. All other layers keep their original source; every layer and
its status is listed in layer_sources.csv.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import arcpy

from release_config import PAPER_DIR, STAGING_DIR

SRC_APRX = PAPER_DIR / "Isidis project only Nili Fossae.aprx"
STAGING = STAGING_DIR
OUT_DIR = STAGING / "B_ArcGIS_project"
OUT_APRX = OUT_DIR / "Nili_Fossae_paper_figures.aprx"
SECTIONS = ("A_geologic_map", "C_Brown2020_olivine_1um_analysis", "D_CRISM_band_parameters",
            "F_pansharpened_CaSSIS_HiRISE", "G_HiRISE_DEMs_Tullo", "H_CTX_DEM")
GDB_NAME = "Nili fossae mapping.gdb"


def staged_index() -> dict[str, Path]:
    """Map lower-case file name (and pansharpened original name) to its staged path."""
    idx: dict[str, Path] = {}
    for sec in SECTIONS:
        for f in (STAGING / sec).rglob("*"):
            if not f.is_file() or "_logs" in f.parts or ".gdb" in str(f.parent):
                continue
            idx.setdefault(f.name.lower(), f)
            if f.name.endswith("_u16.tif"):
                idx.setdefault(f.name[:-len("_u16.tif")].lower() + ".tif", f)
    return idx


def new_connection(cp: dict, idx: dict[str, Path]) -> dict | None:
    """Return updated connection properties if the layer's data are in the package."""
    info = cp.get("connection_info", {})
    db = info.get("database", "")
    ds = cp.get("dataset", "")
    if db.lower().endswith(GDB_NAME.lower()):
        new = json.loads(json.dumps(cp))
        new["connection_info"]["database"] = str(STAGING / "A_geologic_map" / GDB_NAME)
        return new
    hit = idx.get(ds.lower()) or idx.get((ds + ".shp").lower())
    if hit is None or cp.get("workspace_factory") not in ("Raster", "Shape File"):
        return None
    new = json.loads(json.dumps(cp))
    new["connection_info"]["database"] = str(hit.parent)
    new["dataset"] = hit.name
    return new


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_DIR / "_tmp_copy.aprx"
    shutil.copy2(SRC_APRX, tmp)
    aprx = arcpy.mp.ArcGISProject(str(tmp))
    idx = staged_index()
    rows = []
    for m in aprx.listMaps():
        for lyr in m.listLayers():
            if lyr.isGroupLayer or not lyr.supports("DATASOURCE"):
                continue
            src = lyr.dataSource
            status = "external"
            try:
                cp = lyr.connectionProperties
                new = new_connection(cp, idx) if cp else None
                if new:
                    lyr.updateConnectionProperties(cp, new, validate=False)
                    status = "repointed" if not lyr.isBroken else "repointed_but_broken"
            except Exception as e:
                status = f"error: {e}"
            rows.append({"map": m.name, "layer": lyr.longName, "status": status,
                         "original_source": src,
                         "new_source": lyr.dataSource if status.startswith("repointed") else ""})
    aprx.saveACopy(str(OUT_APRX))
    del aprx
    with open(OUT_DIR / "layer_sources.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    counts: dict[str, int] = {}
    for r in rows:
        k = r["status"].split(":")[0]
        counts[k] = counts.get(k, 0) + 1
    print(json.dumps(counts, indent=1))
    try:
        tmp.unlink()
    except PermissionError:
        print(f"note: delete {tmp} manually (still locked by arcpy)")


if __name__ == "__main__":
    main()
