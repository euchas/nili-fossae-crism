"""Project path and configuration constants.

Centralises file layout so analysis scripts don't have to hardcode paths.
Replaces the Dropbox-specific CUBE_DIR / OUT_DIR constants in the original
PCA_CRISM_analysis_v2.py.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(var: str, default: Path) -> Path:
    """Return Path from environment variable `var`, else `default`."""
    return Path(os.environ[var]) if os.environ.get(var) else default


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CUBES_DIR = _env_path("NILI_CUBES_DIR", PROJECT_ROOT / "cubes")
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
STYLE_PATH = PROJECT_ROOT / "code" / "my_style.mplstyle.txt"

def _first_existing(*candidates: Path) -> Path:
    """Return the first existing path, else the last candidate."""
    return next((p for p in candidates if p.exists()), candidates[-1])


# Unpacked Zenodo record (doi:10.5281/zenodo.23163697) and public PDS/PSA
# downloads; see DATA.md. The original working-copy locations are tried first.
DATA_DIR = _env_path("NILI_DATA_DIR", PROJECT_ROOT / "data")
PDS_DIR = _env_path("NILI_PDS_DIR", DATA_DIR / "pds_downloads")
_GIS = PROJECT_ROOT / "Nili Planum" / "CaSSIS-CRISM-Nili_Fossae"
_MAPPING = _GIS / "CRISM site Creta Vulcanica"

PANSHARP_DIR = _env_path("NILI_PANSHARP_DIR", DATA_DIR / "F_pansharpened_CaSSIS_HiRISE")
HIRISE_DEM_DIR = _first_existing(PROJECT_ROOT / "HiRISE DEMs", DATA_DIR / "G_HiRISE_DEMs_Tullo")
HIRISE_PDS_DIR = _first_existing(_GIS / "DTMs", PDS_DIR)
CRISM_RAW_DIR = _first_existing(_MAPPING / "raw data", PDS_DIR)
CASSIS_RAW_DIR = _first_existing(_GIS / "CaSSIS cubes", PDS_DIR)
BROWN_ROI_DIR = _env_path("NILI_BROWN_ROI_DIR",
                          DATA_DIR / "C_Brown2020_olivine_1um_analysis" / "ROI_and_class_tables")
GEOMAP_SOURCE = _first_existing(_MAPPING / "Nili fossae mapping.gdb",
                                DATA_DIR / "A_geologic_map" / "Horseshoe_geologic_map.gpkg")
GEOMAP_SHP_DIR = _first_existing(_MAPPING / "dike map.shp",
                                 DATA_DIR / "A_geologic_map" / "shapefiles" / "dike map.shp").parent
_CTX_GDB = _GIS / "Isidis Project" / "Isidis Project.gdb"
_CTX_NAME = "P03_002176_2024_XI_22N283W__P05_002888_2025_XI_22N283W_dtm_interpolate"
CTX_DEM = (f"OpenFileGDB:{_CTX_GDB}:{_CTX_NAME}" if _CTX_GDB.exists()
           else str(DATA_DIR / "H_CTX_DEM" / f"{_CTX_NAME}.tif"))
PLEBANI_RATIO_DIR = _env_path("NILI_PLEBANI_RATIO_DIR",
                              DATA_DIR / "E_contact_metamorphism" / "plebani_inputs" / "ratioed")
PLEBANI_CARB_DIR = _env_path("NILI_PLEBANI_CARB_DIR",
                             DATA_DIR / "E_contact_metamorphism" / "plebani_inputs" / "carbonate_maps")


def pansharp_file(name: str) -> Path:
    """Locate a pansharpened scene: original float32 file, else the Zenodo uint16 version."""
    stem = Path(name).stem
    return _first_existing(PANSHARP_DIR / name, PANSHARP_DIR / f"{stem}_u16.tif")


def read_reflectance(ds, *args, **kwargs):
    """ds.read() returning float32 I/F with nodata as NaN for both float32 and scaled uint16 files."""
    import numpy as np
    if ds.dtypes[0] == "uint16" and "fill_value" in kwargs:
        kwargs["fill_value"] = ds.nodata
    a = ds.read(*args, **kwargs).astype(np.float32)
    if ds.nodata is not None:
        a[a == np.float32(ds.nodata)] = np.nan
    if ds.dtypes[0] == "uint16":
        a *= np.float32(ds.scales[0])
    return a

RANDOM_SEED = 42

CUBE_IDS = [
    "FRT00003E12",
    "FRT0000871C",
    "FRT0000A4FC",
    "FRT0000B438",
]

BROWN_VERSION = "v5"

OUTPUT_SUBFOLDERS = (
    "feature_stacks",
    "embeddings",
    "cluster_labels",
    "figures",
    "logs",
    "reports",
)

DEFAULT_PARAM_ORDER = (
    "centroid",
    "asymmetry",
    "amplitude",
    "hwhm",
    "OLINDEX3",
    "D2300",
    "BD1900R2",
    "BD2500_2",
)

STAGE_B_PARAM_ORDER = (
    # Brown v5 — 1 µm asymmetric-Gaussian fit outputs
    "centroid",
    "asymmetry",
    "amplitude",
    "hwhm",
    # VB 2014 — VNIR Fe-mineral + olivine / pyroxene
    "BD530_2",
    "BD920_2",
    "BDI1000VIS",
    "OLINDEX3",
    "LCPINDEX2",
    # VB 2014 — OH / H2O / Fe-Mg-OH
    "BD1400",
    "BD1900R2",
    "D2300",
    "BD2290",
    "BD2355",
    # VB 2014 — carbonate
    "BD2500_2",
)

STAGE_B_BROWN_PARAMS = ("centroid", "asymmetry", "amplitude", "hwhm")
STAGE_B_VB_REQUIRED = (
    "BD530_2", "BD920_2", "BDI1000VIS", "OLINDEX3", "LCPINDEX2",
    "BD1400", "BD1900R2", "D2300", "BD2500_2",
)
STAGE_B_VB_EXPLORATORY = ("BD2290", "BD2355")
STAGE_B_REDUCED_4 = ("centroid", "OLINDEX3", "D2300", "BD2500_2")


def cube_dir(cube_id: str) -> Path:
    """Return the folder containing <cube_id>_07_IF166J_TER3 products."""
    return CUBES_DIR / f"{cube_id}_07_IF166J_TER3"


def out_dir(subfolder: str) -> Path:
    """Return an outputs/<subfolder> path, creating it if it does not exist."""
    if subfolder not in OUTPUT_SUBFOLDERS:
        raise ValueError(
            f"Unknown output subfolder {subfolder!r}; expected one of {OUTPUT_SUBFOLDERS}"
        )
    p = OUTPUTS_DIR / subfolder
    p.mkdir(parents=True, exist_ok=True)
    return p
