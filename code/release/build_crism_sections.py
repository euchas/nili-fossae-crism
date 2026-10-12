"""Stage the CRISM-derived Zenodo sections (C: Brown 2020 olivine, D: band parameters, E: contact metamorphism).

Files are copied unchanged except the map-projected VB parameter stack,
which is written from its .npy array to a multi-band GeoTIFF using the
georeferencing of the matching v7c GeoTIFF. Each section gets a
MANIFEST.sha256 and a JSON log listing every staged file and its source.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import rasterio

from release_config import PAPER_DIR, PROJECT_ROOT, STAGING_DIR

CUBES = ("FRT00003E12", "FRT0000871C", "FRT0000A4FC", "FRT0000B438")
OUTPUTS = PROJECT_ROOT / "outputs"
BD_DIR = (PROJECT_ROOT / "Nili Planum" / "CaSSIS-CRISM-Nili_Fossae"
          / "CRISM site Creta Vulcanica" / "BD analysis")
BD_OBSERVATIONS = ("frt00003e12", "frt0000871c", "frt0000a4fc", "frt0000b438", "hrl00011336")
BROWN_CSV_DIR = PAPER_DIR / "olivine 1um centroid analysis (Brown et al. 2020)"
V7_PARAMS = ("OLINDEX3", "BD1400", "BD1900R2", "D2300", "BD2500_2")
REVISION_EXCLUDE_SUFFIXES = (".pkl",)
# Archive-project outputs redistributed in section E: the folders holding the
# *_ratioed_bland.npz and *_carb.npz files (see config.py for the same variables).
PLEBANI_RATIO_DIR = Path(os.environ.get("NILI_PLEBANI_RATIO_DIR", PROJECT_ROOT / "plebani" / "ratioed"))
PLEBANI_CARB_DIR = Path(os.environ.get("NILI_PLEBANI_CARB_DIR", PROJECT_ROOT / "plebani" / "carbonate_maps"))

SECTION_C = STAGING_DIR / "C_Brown2020_olivine_1um_analysis"
SECTION_D = STAGING_DIR / "D_CRISM_band_parameters"
SECTION_E = STAGING_DIR / "E_contact_metamorphism"


def sha256(path: Path, bufsize: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


class Section:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.entries: list[dict] = []
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)

    def copy(self, src: Path, rel: str) -> None:
        dst = self.root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        self.entries.append({"file": rel, "source": str(src)})

    def add_generated(self, rel: str, source: str) -> Path:
        dst = self.root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        self.entries.append({"file": rel, "source": source})
        return dst

    def finish(self) -> None:
        lines = []
        for e in self.entries:
            e["sha256"] = sha256(self.root / e["file"])
            e["bytes"] = (self.root / e["file"]).stat().st_size
            lines.append(f"{e['sha256']}  {e['file']}")
        (self.root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
        (self.root / "_logs").mkdir(exist_ok=True)
        (self.root / "_logs" / "build_log.json").write_text(json.dumps(self.entries, indent=2))
        total = sum(e["bytes"] for e in self.entries)
        print(f"{self.root.name}: {len(self.entries)} files, {total / 1e6:.0f} MB")


def build_c() -> None:
    s = Section(SECTION_C)
    for cube in CUBES:
        ter3 = PROJECT_ROOT / "cubes" / f"{cube}_07_IF166J_TER3"
        for f in sorted(ter3.glob("*_map_v5.npy")):
            s.copy(f, f"TER3_sensor_geometry_v5/{cube}/{f.name}")
        brown = OUTPUTS / "_v7_mtr3_map_projected" / cube / "brown"
        for f in sorted(brown.glob("*_v6.tif")):
            s.copy(f, f"MTR3_map_projected_v6/{cube}/{f.name}")
    for f in sorted(BROWN_CSV_DIR.glob("*.csv")):
        s.copy(f, f"ROI_and_class_tables/{f.name}")
    s.finish()


def export_vb_stack(s: Section, cube: str) -> None:
    v7 = OUTPUTS / "_v7_mtr3_map_projected" / cube
    stack = np.load(v7 / "arrays" / f"{cube}_v7c_VB_stack.npy")
    valid = np.load(v7 / "arrays" / f"{cube}_v7c_pixel_valid_mask.npy")
    ref = v7 / "geotiffs" / f"{cube}_v7c_PC1_score.tif"
    with rasterio.open(ref) as r:
        profile = r.profile.copy()
    profile.update(count=stack.shape[2], dtype="float32", nodata=np.nan, compress="zstd",
                   predictor=3, tiled=True, blockxsize=256, blockysize=256)
    data = np.moveaxis(stack.astype("float32"), 2, 0)
    data[:, ~valid] = np.nan
    dst = s.add_generated(f"python_VB2014_stack_MTR3/{cube}_VB_parameters_MTR3.tif",
                          f"{v7 / 'arrays' / (cube + '_v7c_VB_stack.npy')} + georef from {ref.name}")
    with rasterio.open(dst, "w", **profile) as o:
        o.write(data)
        o.descriptions = V7_PARAMS
        o.update_tags(PARAMETERS=",".join(V7_PARAMS),
                      DEFINITION="Viviano-Beck et al. (2014); code/vb_parameters.py",
                      SOURCE_CUBE=f"{cube.lower()}_07_if166j_mtr3")


def build_d() -> None:
    s = Section(SECTION_D)
    for obs in BD_OBSERVATIONS:
        for f in sorted((BD_DIR / obs).glob("*.tif")):
            s.copy(f, f"ArcGIS_band_parameter_maps/{obs}/{f.name}")
            aux = f.with_name(f.name + ".aux.xml")
            if aux.exists():
                s.copy(aux, f"ArcGIS_band_parameter_maps/{obs}/{aux.name}")
    for cube in CUBES:
        export_vb_stack(s, cube)
        for f in sorted((OUTPUTS / "_v8_mtr3_with_brown" / cube / "geotiffs").glob("*.tif")):
            s.copy(f, f"PCA_GMM_MTR3/{cube}/{f.name}")
    for f in sorted((OUTPUTS / "feature_stacks").glob("*")):
        if f.suffix in (".npz", ".parquet"):
            s.copy(f, f"feature_stacks_TER3/{f.name}")
    s.finish()


def build_e() -> None:
    s = Section(SECTION_E)
    src_root = OUTPUTS / "revision"
    for f in sorted(src_root.rglob("*")):
        if f.is_file() and f.suffix not in REVISION_EXCLUDE_SUFFIXES:
            s.copy(f, f"outputs/{f.relative_to(src_root).as_posix()}")
    for cube in CUBES:
        stem = f"{cube.lower()}_07_if166l_trr3"
        s.copy(PLEBANI_RATIO_DIR / f"{stem}_ratioed_bland.npz", f"plebani_inputs/ratioed/{stem}_ratioed_bland.npz")
        s.copy(PLEBANI_CARB_DIR / f"{stem}_carb.npz", f"plebani_inputs/carbonate_maps/{stem}_carb.npz")
    s.finish()


if __name__ == "__main__":
    build_c()
    build_d()
    build_e()
