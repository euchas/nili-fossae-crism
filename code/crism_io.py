"""I/O helpers for CRISM TER3 joined cubes and associated Brown v5 maps.

Extracted verbatim (no behaviour change) from PCA_CRISM_analysis_v2.py so
analysis modules can reuse them without pulling in the matplotlib
interactive class.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import numpy as np

from config import BROWN_VERSION, cube_dir


def parse_envi_header(hdr_file: str | Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    with open(hdr_file, "r") as f:
        content = f.read()
    lines = content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "=" in line and "{" not in line:
            key, value = line.split("=", 1)
            metadata[key.strip()] = value.strip()
        elif "=" in line and "{" in line:
            key = line.split("=")[0].strip()
            val_str = line.split("=")[1].strip()
            while "}" not in val_str and i < len(lines) - 1:
                i += 1
                val_str += " " + lines[i].strip()
            val_str = val_str.replace("{", "").replace("}", "").strip()
            metadata[key] = [v.strip() for v in val_str.split(",") if v.strip()]
        i += 1
    return metadata


def read_ter3_cube(img_file: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a joined TER3 cube as (bands, lines, samples) float32 with NaN masks.

    Mirrors the reader in PCA_CRISM_analysis_v2.py. NaN-fills ignore-value
    pixels, detector saturation, and BBL-flagged bands.
    """
    img_file = str(img_file)
    hdr = img_file.replace(".img", ".hdr").replace(".IMG", ".hdr")
    if not Path(hdr).exists():
        hdr = img_file + ".hdr"
    meta = parse_envi_header(hdr)

    s = int(meta.get("samples", 0))
    l = int(meta.get("lines", 0))
    b = int(meta.get("bands", 0))
    dt = int(meta.get("data type", 4))
    order = int(meta.get("byte order", 0))
    inter = meta.get("interleave", "bsq").lower()

    if meta.get("wavelength"):
        wav = np.array([float(w) for w in meta["wavelength"]])
        if wav.max() > 100:
            wav = wav / 1000.0
    else:
        raise ValueError(f"No wavelengths in header {hdr}")

    bbl = (
        np.array([int(float(x)) for x in meta["bbl"]]) if meta.get("bbl") else None
    )
    ign = float(meta.get("data ignore value", 65535))

    dtype_map = {1: np.uint8, 2: np.int16, 4: np.float32, 12: np.uint16}
    dtype = dtype_map.get(dt, np.float32)
    data = np.fromfile(img_file, dtype=dtype, count=l * b * s)
    if order == 1:
        data = data.byteswap()

    if inter == "bil":
        data = data.reshape((l, b, s)).transpose((1, 0, 2))
    elif inter == "bip":
        data = data.reshape((l, s, b)).transpose((2, 0, 1))
    elif inter == "bsq":
        data = data.reshape((b, l, s))
    else:
        raise ValueError(f"Unknown interleave {inter!r} in {hdr}")

    data = data.astype(np.float32)
    data[data == ign] = np.nan
    data[data >= 65534] = np.nan
    if bbl is not None:
        data[np.where(bbl == 0)[0], :, :] = np.nan

    return data, wav


def find_cube_and_brown_maps(
    cube_dir_path: str | Path, version: str = BROWN_VERSION
) -> tuple[str, dict[str, np.ndarray]]:
    """Locate the TER3 .img and load the Brown v5 .npy parameter maps.

    Unlike the original helper, this version also loads ``residual`` so we
    can diagnose fit quality without re-locating files.
    """
    cube_dir_path = Path(cube_dir_path)
    patterns = ["*_if*ter3*.img", "*_IF*TER3*.img", "*_if*TER3*.img"]
    img_file = None
    for pat in patterns:
        matches = sorted(glob.glob(str(cube_dir_path / pat)))
        if matches:
            img_file = matches[0]
            break
    if img_file is None:
        raise FileNotFoundError(f"No TER3 cube in {cube_dir_path}")

    base = Path(img_file).stem
    brown: dict[str, np.ndarray] = {}
    required = ("centroid", "asymmetry", "amplitude", "hwhm", "mask", "residual")
    for param in required:
        path = cube_dir_path / f"{base}_{param}_map_{version}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing Brown map: {path}")
        brown[param] = np.load(path)
    return img_file, brown


def load_cube_by_id(cube_id: str) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], str]:
    """High-level convenience: given a cube ID, return (cube, wav, brown, img_path)."""
    cdir = cube_dir(cube_id)
    img_file, brown = find_cube_and_brown_maps(cdir)
    cube, wav = read_ter3_cube(img_file)
    return cube, wav, brown, img_file
