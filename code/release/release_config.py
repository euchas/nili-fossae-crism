"""Source and staging locations used to build the Zenodo record (see DATA.md).

Set NILI_PAPER_DIR (working copy with the pansharpened scenes and HiRISE DEMs) and
NILI_STAGING_DIR (where the sections are staged) to rebuild the package elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = Path(os.environ.get("NILI_PAPER_DIR", PROJECT_ROOT / "paper"))
PANSHARP_DIR = PAPER_DIR / "pansharpening"
HIRISE_DEM_DIR = PAPER_DIR / "HiRISE DEMs"
STAGING_DIR = Path(os.environ.get("NILI_STAGING_DIR", PROJECT_ROOT / "release_staging"))

ZENODO_RECORD_ID = 23163697
ZENODO_DOI = "10.5281/zenodo.23163697"

PANSHARP_SCENES = (
    "GSA1_MY37_027047_023_1.tif",
    "GSA2_MY37_027047_023_1.tif",
    "GSA3_MY37_027047_023_1.tif",
    "GSA4_MY37_027047_023_1.tif",
    "GSA5_Mosaic027025-007.tif",
    "GSA6_MosaicWest.tif",
    "GSA7corr_MY37_025124_157_0.tif",
    "GSA8_Mosaic027025.tif",
    "GSA9_Mosaic027025.tif",
    "GSA10_Mosaic027025.tif",
    "GSA11corr_MY37_025124_157_0.tif",
    "GSA12_MY37_025124_157_0-013.tif",
    "GSA13_CasMosaic025013-001.tif",
    "GSA14_Mosaic025013-002.tif",
    "GSA16_MY37_025554_021_0.tif",
    "GSA17corr_Mosaic025013.tif",
    "GSA18e_MY35_013481_158_0.tif",
    "GSA18n_MY37_024813_159_0-010.tif",
    "GSA18s_MY37_024813_159_0_NRB-009.tif",
    "GSA19_MY37_024813_159_0.tif",
    "GSA20_MY37_024813_159_0.tif",
    "GSA21_MY37_024813_159_0.tif",
    "GSA22_MY37_027358_024_0.tif",
    "GSA23_MY37_027358_024_0.tif",
    "GSA24_MY37_027358_024_0.tif",
    "GSA_MY36_014433_022_0_nodatacorr.tif",
)

PANSHARP_SUPERSEDED = (
    "GSA7_MY37_025124_157_0.tif",
    "GSA11_MY37_025124_157_0-008.tif",
    "GSA17_Mosaic025013-001.tif",
    "GSA_MY36_014433_022_0-001.tif",
)
