"""Write section READMEs and manifests, then zip every staged section for Zenodo upload.

Zips use ZIP_STORED (the rasters are already compressed). Section F is split
into several zips of at most ~6 GB so each upload can be retried on its own.
Output: STAGING_DIR/_upload/ with the zips, README.txt and MANIFEST.sha256.
Usage: python package_zenodo.py [SECTION ...]   (no argument = all sections)
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from release_config import STAGING_DIR, ZENODO_DOI

UPLOAD = STAGING_DIR / "_upload"
MAX_ZIP_BYTES = 2_000_000_000
PAPER = ("Valantinas, A., Mustard, J. F., Tullo, A., Bramble, M. S., & Head, J. W. (2026). "
         "An exposed magmatic feeder system for the regional olivine unit in Nili Fossae, Mars. "
         "Earth and Planetary Science Letters, 696, 120390. https://doi.org/10.1016/j.epsl.2026.120390")
CODE = "https://github.com/euchas/nili-fossae-crism"
CRS_NOTE = "Coordinates: Mars equirectangular projections as stored in each file (see the file's CRS)."

SECTIONS = {
    "A_geologic_map": f"""Hand-drawn geologic map of the Horseshoe site, western Nili Fossae (A. Valantinas).

Horseshoe_geologic_map.gpkg  20 layers (open GeoPackage):
  Clay_unit, Mafic_cap, Olivine_unit, olivine_dunes, graben_materials  - unit polygons
  Clay_unit_original, Mafic_cap_original  - earlier versions of the clay and cap polygons, kept for reference
  Raised_ridges, Ruptured_ridges, Flow_fronts  - line features
  Intrusions  - mapped intrusion locations (points)
  dike_map, amorphous_matrix  - intrusion 1 at site 1 (paper Figure 7)
  best_ridge_sites, bright_exposures, quasi_circular_features, rupture_locations  - point annotations
  Profile_1..3  - topographic profile lines
Nili fossae mapping.gdb  the original ArcGIS File Geodatabase (same features; M values preserved).
shapefiles/  the stand-alone shapefiles as drawn.
Geometries in the GeoPackage were verified identical to the File GDB.
{CRS_NOTE}""",
    "B_ArcGIS_project": """ArcGIS Pro project used to make the paper's map figures.

Nili_Fossae_paper_figures.aprx  17 maps/scenes and 17 layouts (paper and supplement figures).
layer_sources.csv  every layer, its original data source, and whether it was repointed.

Unzip all sections of this record into one folder so that B_ArcGIS_project sits next to
A_..., C_..., D_..., F_..., G_..., H_.... 676 layers then resolve to files in this record
by relative path. The remaining 1221 layers point to public CaSSIS, HiRISE, CTX and CRISM
products (or to working files not archived) and open as broken links; use the original
source in layer_sources.csv and ArcGIS "Repair data source" after downloading them.
The pansharpened layers now read the uint16 files (section F); stretches based on
statistics are recomputed by ArcGIS, so colours can differ slightly from the paper figures.""",
    "C_Brown2020_olivine_1um_analysis": f"""1 um olivine band fits after Brown et al. (2020), for CRISM FRT00003E12, FRT0000871C,
FRT0000A4FC, FRT0000B438.

TER3_sensor_geometry_v5/<cube>/  centroid, asymmetry, amplitude, hwhm, mask and residual maps
  (.npy, CRISM sensor geometry, same grid as the TER3 cube). Validated against Brown et al. (2020)
  for FRT00003E12.
MTR3_map_projected_v6/<cube>/  the same fit on the map-projected MTR3 cubes (GeoTIFF) and RGB composites.
ROI_and_class_tables/  per-ROI pixel tables, fit parameters, PCA class definitions and spectra (CSV).
Code: {CODE} (code/1um_centroid_analysis_CRISM_*_v1.py).""",
    "D_CRISM_band_parameters": f"""CRISM spectral parameter maps (Viviano-Beck et al., 2014 definitions).

ArcGIS_band_parameter_maps/<observation>/  BD1400, BD1900r, D2300, HCPINDEX(2), LCPINDEX(2),
  OLINDEX2, OLINDEX3 and TAN/TRU composites for FRT00003E12, FRT0000871C, FRT0000A4FC,
  FRT0000B438 and HRL00011336 (map-projected GeoTIFF; .aux.xml holds statistics).
python_VB2014_stack_MTR3/  OLINDEX3, BD1400, BD1900R2, D2300, BD2500_2 computed with
  code/vb_parameters.py on the MTR3 cubes (5-band GeoTIFF, NaN = invalid).
PCA_GMM_MTR3/<cube>/  PCA scores and 6-class GMM labels on the MTR3 grid (v8 run).
feature_stacks_TER3/  8- and 15-parameter stacks in TER3 geometry (.npz and long-format .parquet).
Code: {CODE}.
{CRS_NOTE}""",
    "E_contact_metamorphism": f"""Outputs of the contact-metamorphism and band-centre tests (paper revision and supplement).

outputs/  all outputs of the contact_*.py scripts (written to outputs/revision/ when the code is run): band-centre maps and modes,
  matched-filter detections, distance-to-contact and height-above-contact rasters, unit
  fractions on the CRISM grid, carbonate patches, contact-fringe profiles, figures, and a
  REPORT.md / JSON log per analysis naming its inputs.
plebani_inputs/ratioed/  bland-column-ratioed TRR3 cubes for the four FRTs (Plebani-chain inputs, .npz).
plebani_inputs/carbonate_maps/  carbonate classifier maps for the four FRTs (.npz).
Code: {CODE} (code/contact_*.py, code/supplement_fig_*.py; run order in code/REPRODUCE.md).""",
    "F_pansharpened_CaSSIS_HiRISE": """CaSSIS colour images pansharpened with HiRISE (Gram-Schmidt), 26 scenes, 0.25-0.29 m/pixel.

Bands: NIR, RED, PAN, BLU (GSA18s: NIR, RED, BLU).
Encoding: uint16, I/F = DN * 2e-5 (scale factor stored in the GeoTIFF), nodata DN = 65535.
  Maximum quantisation error 1e-5 I/F. Converted from the original float32 files by
  code/release/convert_pansharpened.py; per-scene logs in _logs/.
Tiled, ZSTD-compressed, internal overviews (average, 2-64). Reduced-resolution reads use these
  overviews, so decimated renderings can differ slightly from those made from the original files.
Negative I/F values (pansharpening artefacts) were clipped to 0 in five scenes:
  GSA13 0.077 %, GSA16 0.092 %, GSA4 0.0004 %, GSA21 26 values, GSA11corr 2 values.
Where an uncorrected and a corrected ("corr") version existed, only the corrected one is included.
Source CaSSIS and HiRISE observation IDs are listed in DATA.md of the code repository.
Split across several zip files; unzip all into the same folder.""",
    "G_HiRISE_DEMs_Tullo": """HiRISE stereo DEMs and orthoimages produced by A. Tullo.

PSP_006712_2020 / PSP_005855_2020: DEM (aligned) and DRG orthoimage. These files carry a local CRS;
  their georeferencing is given by the accompanying .tfwx world files (see
  code/contact_dtm_registry.py).
ESP_027691_2025 / ESP_026992_2025: DEM (geoid-aligned) and DRG orthoimage.
Values are bit-identical to the originals; only the compression changed (lossless ZSTD,
internal overviews). Logs in _logs/.""",
    "H_CTX_DEM": """CTX stereo DEM for the pair P03_002176_2024_XI_22N283W / P05_002888_2025_XI_22N283W,
derived from the UCL CTX DTM product for this pair, with artefacts removed and gaps
interpolated in Python by A. Valantinas.

*_dtm_interpolate.tif  the CTX DEM used by the contact-metamorphism scripts (config.CTX_DEM);
  exported from the ArcGIS geodatabase where it was stored.
*_dtm_cleaned.tif  the cleaned DEM before the final interpolation step. Its original file had a
  local coordinate system and was positioned in ArcGIS by the accompanying .tfwx world file;
  that world file and the Mars CRS have been written into this GeoTIFF (the .tfwx is included).
  Values agree with the interpolated DEM except in filled areas (median difference 0 m; 8 % of
  pixels differ by more than 1 m).
Values are bit-identical to the working copies; lossless ZSTD.
Please also cite the UCL CTX DTM source for this pair.""",
}


def sha256(path: Path, bufsize: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def write_readme_and_manifest(name: str, text: str) -> list[Path]:
    root = STAGING_DIR / name
    header = f"{name}\n{'=' * len(name)}\n\nPart of the data record doi:{ZENODO_DOI} for:\n{PAPER}\n\n"
    (root / "README.txt").write_text(header + text + "\n\nLicense: CC-BY-4.0\n", encoding="utf-8")
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256")
    lines = [f"{sha256(p)}  {p.relative_to(root).as_posix()}" for p in files]
    (root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return files + [root / "MANIFEST.sha256"]


def zip_section(name: str, files: list[Path]) -> list[Path]:
    root = STAGING_DIR / name
    always = [root / "README.txt", root / "MANIFEST.sha256"] + [p for p in files if "_logs" in p.parts]
    data = [p for p in files if p not in always]
    parts: list[list[Path]] = [[]]
    size = 0
    for p in sorted(data, key=lambda q: q.relative_to(root).as_posix()):
        if parts[-1] and size + p.stat().st_size > MAX_ZIP_BYTES:
            parts.append([])
            size = 0
        parts[-1].append(p)
        size += p.stat().st_size
    out = []
    for i, part in enumerate(parts, 1):
        zname = f"{name}.zip" if len(parts) == 1 else f"{name}_part{i}of{len(parts)}.zip"
        zpath = UPLOAD / zname
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
            for p in (always if i == 1 else [root / "README.txt"]) + part:
                z.write(p, (Path(name) / p.relative_to(root)).as_posix())
        out.append(zpath)
        print(f"  {zname}: {zpath.stat().st_size / 1e9:.2f} GB")
    return out


def main(only: list[str] | None = None) -> None:
    """Package all sections, or only those named in `only` (other zips are kept as they are)."""
    UPLOAD.mkdir(exist_ok=True)
    manifest_path = UPLOAD / "MANIFEST.sha256"
    cache = {}
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            digest, name = line.split("  ", 1)
            cache[name] = digest
    for name, text in SECTIONS.items():
        if only and name not in only:
            continue
        print(name)
        for old in UPLOAD.glob(f"{name}*.zip"):
            old.unlink()
            cache.pop(old.name, None)
        files = write_readme_and_manifest(name, text)
        zip_section(name, files)
    zips = sorted(UPLOAD.glob("*.zip"))
    top = UPLOAD / "README.txt"
    top.write_text(
        f"Data for: {PAPER}\n\nDOI: {ZENODO_DOI}\nCode: {CODE}\nLicense: CC-BY-4.0\n\n"
        "Each zip holds one section (F is split into parts). Unzip all into one folder.\n"
        "Every section has its own README.txt and MANIFEST.sha256 (sha256sum -c MANIFEST.sha256).\n\n"
        + "\n".join(f"{n}: {t.splitlines()[0]}" for n, t in SECTIONS.items()) + "\n"
        "\nRaw CRISM, HiRISE, CTX and CaSSIS products are not redistributed; their IDs are in DATA.md\n"
        "of the code repository.\n", encoding="utf-8")
    cache.pop("README.txt", None)
    lines = [f"{cache.get(z.name) or sha256(z)}  {z.name}" for z in zips] + [f"{sha256(top)}  {top.name}"]
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (UPLOAD / "upload_list.json").write_text(json.dumps(
        [{"file": z.name, "bytes": z.stat().st_size} for z in zips + [top, manifest_path]], indent=2))
    total = sum(z.stat().st_size for z in zips)
    print(f"{len(zips)} zips, {total / 1e9:.1f} GB")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)
