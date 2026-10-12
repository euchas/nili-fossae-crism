"""
CRISM TER3 + Brown Parameter ROI Explorer (v1.0)
-------------------------------------------------
Combines:
  - TER3 param-RGB + IR false-color viewer (from ter3_spectra_extractor v4.2)
  - Brown et al. (2020) 1 um olivine band parameter maps (from brown2020_batch_ter3)
  - Brown parameter exploration dashboard (from brown_explore_dashboard)

Workflow:
  1. Loads a CRISM TER3 cube and its coregistered Brown parameter .npy files
     (centroid_v5, asymmetry_v5, hwhm_v5, amplitude_v5).
  2. User picks an RGB parameter composite (Viviano-Beck 2014 registry).
  3. Map figure shows 4 panels with shared zoom/pan + dynamic restretch:
        Panel 1 (lasso-enabled) : param RGB composite
        Panel 2                 : IR false color
        Panel 3                 : Brown centroid map
        Panel 4                 : Brown asymmetry map
  4. Lasso an ROI on panel 1. ALL pixels inside the lasso are pulled from
     the Brown maps and scatter-plotted across the Brown exploration figure:
        Panel A : Centroid vs Asymmetry  (Brown 2020 Fig 5 analog)
        Panel B : Centroid vs HWHM       (grain size vs composition)
        Panel C : Amplitude vs Asymmetry
        Panel D : Centroid histogram per ROI
  5. Each ROI gets its own color (tab10). ROIs accumulate as scatter clouds.

Keybindings:
  d   delete last ROI
  r   reset all ROIs
  f   force dynamic restretch of the map panels on current view
  s   save ROI pixel values to CSV (per-pixel rows)
  q   quit

Author: AV Valantinas (merged from TER3 v4.2 + Brown exploration tools)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import CUBES_DIR, STYLE_PATH
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from matplotlib.lines import Line2D
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# USER CONFIGURATION
# =============================================================================
FILEPATH = str(CUBES_DIR / "FRT0000A4FC_07_IF166J_TER3" / "frt0000a4fc_07_if166j_ter3.img")

# Brown map version suffix (v5 are the latest files)
BROWN_VERSION = "v5"

# Display transform (applied to every map panel AND to incoming lasso coords)
FLIP_VERTICAL = True
FLIP_HORIZONTAL = True

# Percentile limits for dynamic per-view stretching of RGB panels
STRETCH_PMIN = 2
STRETCH_PMAX = 98

# Colormap ranges for the Brown parameter map panels
CENTROID_VMIN, CENTROID_VMAX, CENTROID_CMAP = 1.05, 1.35, 'RdYlBu_r'
ASYMMETRY_VMIN, ASYMMETRY_VMAX, ASYMMETRY_CMAP = 0.8, 6.0, 'magma'

# Brown 2020 interpretive reference lines
BROWN_THRESHOLD_CENTROID = 1.05
POPULATION_ZERO_CENTROID = 1.20

# Optional mplstyle
style_path = str(STYLE_PATH)
if os.path.exists(style_path):
    plt.style.use(style_path)


# =============================================================================
# DISPLAY TRANSFORM (mirrors the TER3 v4.2 convention)
# =============================================================================

def apply_display_transform(image):
    """Flip a 2D or [H, W, C] array to DISPLAY orientation."""
    result = image.copy()
    if FLIP_VERTICAL:
        result = np.flipud(result)
    if FLIP_HORIZONTAL:
        result = np.fliplr(result)
    return result


def display_to_data_coords(x, y, shape):
    """Convert click/lasso (x, y) in DISPLAY coords to DATA (line, sample)."""
    lines, samples = shape[0], shape[1]
    x_data, y_data = x, y
    if FLIP_HORIZONTAL:
        x_data = samples - 1 - x
    if FLIP_VERTICAL:
        y_data = lines - 1 - y
    return x_data, y_data


# =============================================================================
# CUBE READER (from TER3 v4.2)
# =============================================================================

def parse_envi_header(hdr_file):
    metadata = {}
    with open(hdr_file, 'r') as f:
        content = f.read()
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if '=' in line and '{' not in line:
            key, value = line.split('=', 1)
            metadata[key.strip()] = value.strip()
        elif '=' in line and '{' in line:
            key = line.split('=')[0].strip()
            val_str = line.split('=')[1].strip()
            while '}' not in val_str and i < len(lines) - 1:
                i += 1
                val_str += ' ' + lines[i].strip()
            val_str = val_str.replace('{', '').replace('}', '').strip()
            metadata[key] = [v.strip() for v in val_str.split(',') if v.strip()]
        i += 1
    return metadata


def read_crism_ter3_cube(img_file):
    hdr = img_file.replace('.img', '.hdr').replace('.IMG', '.hdr')
    if not os.path.exists(hdr):
        hdr = img_file + '.hdr'
    if not os.path.exists(hdr):
        raise ValueError(f"No header for {img_file}")

    meta = parse_envi_header(hdr)
    s = int(meta.get('samples', 0))
    l = int(meta.get('lines', 0))
    b = int(meta.get('bands', 0))
    dt = int(meta.get('data type', 4))
    order = int(meta.get('byte order', 0))
    inter = meta.get('interleave', 'bsq').lower()

    print(f"Reading {os.path.basename(img_file)} ({s}x{l}x{b})")

    if meta.get('wavelength'):
        wav = np.array([float(w) for w in meta.get('wavelength', [])])
        units = str(meta.get('wavelength units', 'Nanometers')).lower()
        if 'nano' in units or wav.max() > 100:
            wav = wav / 1000.0
    else:
        wav = np.linspace(0.362, 3.937, b)

    bbl = np.array([int(x) for x in meta.get('bbl', [])]) if meta.get('bbl') else None
    ign = float(meta.get('data ignore value', 65535))

    dtype = {1: np.uint8, 2: np.int16, 4: np.float32, 12: np.uint16}.get(dt, np.float32)
    data = np.fromfile(img_file, dtype=dtype, count=l * b * s)
    if order == 1:
        data = data.byteswap()

    if inter == 'bil':
        data = data.reshape((l, b, s)).transpose((1, 0, 2))
    elif inter == 'bip':
        data = data.reshape((l, s, b)).transpose((2, 0, 1))
    elif inter == 'bsq':
        data = data.reshape((b, l, s))

    data = data.astype(np.float32)
    data[data == ign] = np.nan
    data[data >= 65534] = np.nan
    if bbl is not None:
        data[np.where(bbl == 0)[0], :, :] = np.nan

    return data, wav


# =============================================================================
# BROWN .npy LOADER
# =============================================================================

def load_brown_maps(img_file, version=BROWN_VERSION):
    """
    Load Brown parameter maps coregistered to the TER3 cube.

    Assumes naming pattern:
        <cube_dir>/<base_name>_{param}_map_{version}.npy
    where base_name is the .img filename stem.

    Returns dict with keys: centroid, asymmetry, amplitude, hwhm (optional: mask, residual).
    """
    cube_dir = os.path.dirname(img_file)
    base_name = os.path.splitext(os.path.basename(img_file))[0]

    params = ['centroid', 'asymmetry', 'amplitude', 'hwhm']
    optional_params = ['mask', 'residual']

    maps = {}
    print(f"\nLoading Brown parameter maps (version={version}):")
    for p in params:
        path = os.path.join(cube_dir, f"{base_name}_{p}_map_{version}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing Brown map: {path}")
        arr = np.load(path)
        maps[p] = arr
        valid = np.sum(np.isfinite(arr))
        print(f"  {p:10s}: shape {arr.shape}, {valid:,} valid pixels "
              f"({100 * valid / arr.size:.1f}%)")

    for p in optional_params:
        path = os.path.join(cube_dir, f"{base_name}_{p}_map_{version}.npy")
        if os.path.exists(path):
            maps[p] = np.load(path)
            print(f"  {p:10s}: shape {maps[p].shape} (optional, loaded)")

    return maps, base_name


# =============================================================================
# SPECTRAL PARAMETERS (Viviano-Beck et al. 2014) - trimmed to what we need here
# =============================================================================

def get_kernel(data, wav, target, width):
    idx = np.abs(wav - target).argmin()
    if width <= 1:
        return data[idx, :, :]
    hw = width // 2
    return np.nanmedian(data[max(0, idx - hw):min(len(wav), idx + hw + 1), :, :], axis=0)


def _ratio(num, den):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(den != 0, 1.0 - (num / den), 0.0)
    return np.nan_to_num(result, nan=0.0)


def _rb(R, RC):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(RC != 0, (RC - R) / RC, 0.0)
    return np.nan_to_num(result, nan=0.0)


# Mafic
def calc_olindex3(data, wav):
    R1210 = get_kernel(data, wav, 1.210, 7); R1250 = get_kernel(data, wav, 1.250, 7)
    R1263 = get_kernel(data, wav, 1.263, 7); R1276 = get_kernel(data, wav, 1.276, 7)
    R1330 = get_kernel(data, wav, 1.330, 7); R1750 = get_kernel(data, wav, 1.750, 7)
    R1862 = get_kernel(data, wav, 1.862, 7)
    slope = np.where((R1862 != R1750), (R1862 - R1750) / (1.862 - 1.750), 0.0)
    def rc(w): return R1750 + slope * (w - 1.750)
    return (0.1 * _rb(R1210, rc(1.210)) + 0.1 * _rb(R1250, rc(1.250)) +
            0.2 * _rb(R1263, rc(1.263)) + 0.2 * _rb(R1276, rc(1.276)) +
            0.4 * _rb(R1330, rc(1.330)))


def calc_lcpindex2(data, wav):
    R1560 = get_kernel(data, wav, 1.560, 7); R1690 = get_kernel(data, wav, 1.690, 7)
    R1750 = get_kernel(data, wav, 1.750, 7); R1810 = get_kernel(data, wav, 1.810, 7)
    R1870 = get_kernel(data, wav, 1.870, 7); R2450 = get_kernel(data, wav, 2.450, 7)
    slope = np.where((R2450 != R1560), (R2450 - R1560) / (2.450 - 1.560), 0.0)
    def rc(w): return R1560 + slope * (w - 1.560)
    return (0.20 * _rb(R1690, rc(1.690)) + 0.20 * _rb(R1750, rc(1.750)) +
            0.30 * _rb(R1810, rc(1.810)) + 0.30 * _rb(R1870, rc(1.870)))


def calc_hcpindex2(data, wav):
    R1690 = get_kernel(data, wav, 1.690, 7); R2120 = get_kernel(data, wav, 2.120, 5)
    R2140 = get_kernel(data, wav, 2.140, 7); R2230 = get_kernel(data, wav, 2.230, 7)
    R2250 = get_kernel(data, wav, 2.250, 7); R2430 = get_kernel(data, wav, 2.430, 7)
    R2460 = get_kernel(data, wav, 2.460, 7); R2530 = get_kernel(data, wav, 2.530, 7)
    slope = np.where((R2530 != R1690), (R2530 - R1690) / (2.530 - 1.690), 0.0)
    def rc(w): return R1690 + slope * (w - 1.690)
    return (0.10 * _rb(R2120, rc(2.120)) + 0.10 * _rb(R2140, rc(2.140)) +
            0.15 * _rb(R2230, rc(2.230)) + 0.30 * _rb(R2250, rc(2.250)) +
            0.20 * _rb(R2430, rc(2.430)) + 0.15 * _rb(R2460, rc(2.460)))


# Hydration
def calc_bd1900_2(data, wav):
    R1850 = get_kernel(data, wav, 1.850, 5); R1930 = get_kernel(data, wav, 1.930, 5)
    R1985 = get_kernel(data, wav, 1.985, 5); R2067 = get_kernel(data, wav, 2.067, 5)
    b = (1.930 - 1.850) / (2.067 - 1.850); a = 1.0 - b
    bd1 = _ratio(R1930, a * R1850 + b * R2067)
    b2 = (1.985 - 1.850) / (2.067 - 1.850); a2 = 1.0 - b2
    bd2 = _ratio(R1985, a2 * R1850 + b2 * R2067)
    return 0.5 * bd1 + 0.5 * bd2


def calc_bd1900r2(data, wav):
    num_w = [1.908, 1.914, 1.921, 1.928, 1.934, 1.941]
    den_w = [1.862, 1.869, 1.875, 2.112, 2.120, 2.126]
    R1850 = get_kernel(data, wav, 1.850, 5); R2060 = get_kernel(data, wav, 2.060, 5)
    slope = np.where((R2060 != R1850), (R2060 - R1850) / (2.060 - 1.850), 0.0)
    def rc(w): return R1850 + slope * (w - 1.850)
    with np.errstate(divide='ignore', invalid='ignore'):
        num_sum = sum(np.where(rc(w) != 0, get_kernel(data, wav, w, 1) / rc(w), 0.0) for w in num_w)
        den_sum = sum(np.where(rc(w) != 0, get_kernel(data, wav, w, 1) / rc(w), 0.0) for w in den_w)
        result = np.where(den_sum != 0, 1.0 - num_sum / den_sum, 0.0)
    return np.nan_to_num(result, nan=0.0)


def calc_bd2100_2(data, wav):
    R2132 = get_kernel(data, wav, 2.132, 5); R1930 = get_kernel(data, wav, 1.930, 3)
    R2250 = get_kernel(data, wav, 2.250, 3)
    b = (2.132 - 1.930) / (2.250 - 1.930); a = 1.0 - b
    return _ratio(R2132, a * R1930 + b * R2250)


# Al-OH
def calc_d2200(data, wav):
    R1815 = get_kernel(data, wav, 1.815, 7); R2165 = get_kernel(data, wav, 2.165, 5)
    R2210 = get_kernel(data, wav, 2.210, 7); R2230 = get_kernel(data, wav, 2.230, 7)
    R2430 = get_kernel(data, wav, 2.430, 7)
    slope = np.where((R2430 != R1815), (R2430 - R1815) / (2.430 - 1.815), 0.0)
    def rc(w): return R1815 + slope * (w - 1.815)
    RC2165 = rc(2.165); RC2210 = rc(2.210); RC2230 = rc(2.230)
    with np.errstate(divide='ignore', invalid='ignore'):
        r2165 = np.where(RC2165 != 0, R2165 / RC2165, 0.0)
        r2210 = np.where(RC2210 != 0, R2210 / RC2210, 0.0)
        r2230 = np.where(RC2230 != 0, R2230 / RC2230, 0.0)
        avg = (r2210 + r2230) / 2.0
        result = np.where(r2165 != 0, 1.0 - avg / r2165, 0.0)
    return np.nan_to_num(result, nan=0.0)


def calc_bd2210_2(data, wav):
    R2165 = get_kernel(data, wav, 2.165, 5); R2210 = get_kernel(data, wav, 2.210, 5)
    R2250 = get_kernel(data, wav, 2.250, 5)
    b = (2.210 - 2.165) / (2.250 - 2.165); a = 1.0 - b
    return _ratio(R2210, a * R2165 + b * R2250)


def calc_bd2165(data, wav):
    R2120 = get_kernel(data, wav, 2.120, 5); R2165 = get_kernel(data, wav, 2.165, 3)
    R2230 = get_kernel(data, wav, 2.230, 3)
    b = (2.165 - 2.120) / (2.230 - 2.120); a = 1.0 - b
    return _ratio(R2165, a * R2120 + b * R2230)


def calc_bd2190(data, wav):
    R2120 = get_kernel(data, wav, 2.120, 5); R2185 = get_kernel(data, wav, 2.185, 3)
    R2250 = get_kernel(data, wav, 2.250, 3)
    b = (2.185 - 2.120) / (2.250 - 2.120); a = 1.0 - b
    return _ratio(R2185, a * R2120 + b * R2250)


# Fe/Mg-OH / phyllo
def calc_bd2290(data, wav):
    R2250 = get_kernel(data, wav, 2.250, 5); R2290 = get_kernel(data, wav, 2.290, 5)
    R2350 = get_kernel(data, wav, 2.350, 5)
    b = (2.290 - 2.250) / (2.350 - 2.250); a = 1.0 - b
    return _ratio(R2290, a * R2250 + b * R2350)


def calc_d2300(data, wav):
    R1815 = get_kernel(data, wav, 1.815, 5); R2120 = get_kernel(data, wav, 2.120, 5)
    R2170 = get_kernel(data, wav, 2.170, 5); R2210 = get_kernel(data, wav, 2.210, 5)
    R2290 = get_kernel(data, wav, 2.290, 3); R2320 = get_kernel(data, wav, 2.320, 3)
    R2330 = get_kernel(data, wav, 2.330, 3); R2530 = get_kernel(data, wav, 2.530, 5)
    slope = np.where((R2530 != R1815), (R2530 - R1815) / (2.530 - 1.815), 0.0)
    def rc(w): return R1815 + slope * (w - 1.815)
    with np.errstate(divide='ignore', invalid='ignore'):
        num = (np.where(rc(2.290) != 0, R2290 / rc(2.290), 0.0) +
               np.where(rc(2.320) != 0, R2320 / rc(2.320), 0.0) +
               np.where(rc(2.330) != 0, R2330 / rc(2.330), 0.0))
        den = (np.where(rc(2.120) != 0, R2120 / rc(2.120), 0.0) +
               np.where(rc(2.170) != 0, R2170 / rc(2.170), 0.0) +
               np.where(rc(2.210) != 0, R2210 / rc(2.210), 0.0))
        result = np.where(den != 0, 1.0 - num / den, 0.0)
    return np.nan_to_num(result, nan=0.0)


def calc_bd2355(data, wav):
    R2300 = get_kernel(data, wav, 2.300, 5); R2355 = get_kernel(data, wav, 2.355, 5)
    R2450 = get_kernel(data, wav, 2.450, 5)
    b = (2.355 - 2.300) / (2.450 - 2.300); a = 1.0 - b
    return _ratio(R2355, a * R2300 + b * R2450)


def calc_min2250(data, wav):
    R2165 = get_kernel(data, wav, 2.165, 5); R2210 = get_kernel(data, wav, 2.210, 3)
    R2265 = get_kernel(data, wav, 2.265, 3); R2350 = get_kernel(data, wav, 2.350, 5)
    b = (2.210 - 2.165) / (2.350 - 2.165); a = 1.0 - b
    den = a * R2165 + b * R2350
    bd1 = _ratio(R2210, den); bd2 = _ratio(R2265, den)
    return np.minimum(bd1, bd2)


def calc_bd2250(data, wav):
    R2120 = get_kernel(data, wav, 2.120, 5); R2245 = get_kernel(data, wav, 2.245, 7)
    R2340 = get_kernel(data, wav, 2.340, 3)
    b = (2.245 - 2.120) / (2.340 - 2.120); a = 1.0 - b
    return _ratio(R2245, a * R2120 + b * R2340)


# Sulfate / carbonate
def calc_sindex2(data, wav):
    R2120 = get_kernel(data, wav, 2.120, 5); R2290 = get_kernel(data, wav, 2.290, 7)
    R2400 = get_kernel(data, wav, 2.400, 3)
    b = (2.290 - 2.120) / (2.400 - 2.120); a = 1.0 - b
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(R2290 != 0, 1.0 - (a * R2120 + b * R2400) / R2290, 0.0)
    return np.nan_to_num(result, nan=0.0)


def calc_bd2500_2(data, wav):
    R2364 = get_kernel(data, wav, 2.364, 5); R2480 = get_kernel(data, wav, 2.480, 5)
    R2570 = get_kernel(data, wav, 2.570, 5)
    b = (2.480 - 2.364) / (2.570 - 2.364); a = 1.0 - b
    return _ratio(R2480, a * R2364 + b * R2570)


def calc_min2295_2480(data, wav):
    R2165 = get_kernel(data, wav, 2.165, 5); R2295 = get_kernel(data, wav, 2.295, 5)
    R2364 = get_kernel(data, wav, 2.364, 5); R2480 = get_kernel(data, wav, 2.480, 5)
    R2570 = get_kernel(data, wav, 2.570, 5)
    b1 = (2.295 - 2.165) / (2.364 - 2.165); a1 = 1.0 - b1
    bd1 = _ratio(R2295, a1 * R2165 + b1 * R2364)
    b2 = (2.480 - 2.364) / (2.570 - 2.364); a2 = 1.0 - b2
    bd2 = _ratio(R2480, a2 * R2364 + b2 * R2570)
    return np.minimum(bd1, bd2)


def calc_min2345_2537(data, wav):
    R2250 = get_kernel(data, wav, 2.250, 5); R2345 = get_kernel(data, wav, 2.345, 5)
    R2430 = get_kernel(data, wav, 2.430, 5); R2537 = get_kernel(data, wav, 2.537, 5)
    R2602 = get_kernel(data, wav, 2.602, 5)
    b1 = (2.345 - 2.250) / (2.430 - 2.250); a1 = 1.0 - b1
    bd1 = _ratio(R2345, a1 * R2250 + b1 * R2430)
    b2 = (2.537 - 2.430) / (2.602 - 2.430); a2 = 1.0 - b2
    bd2 = _ratio(R2537, a2 * R2430 + b2 * R2602)
    return np.minimum(bd1, bd2)


def calc_cindex2(data, wav):
    R3450 = get_kernel(data, wav, 3.450, 9); R3610 = get_kernel(data, wav, 3.610, 11)
    R3875 = get_kernel(data, wav, 3.875, 7)
    b = (3.610 - 3.450) / (3.875 - 3.450); a = 1.0 - b
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(R3610 != 0, 1.0 - (a * R3450 + b * R3875) / R3610, 0.0)
    return np.nan_to_num(result, nan=0.0)


PARAM_REGISTRY = {
    'OLINDEX3':      (calc_olindex3,      'Olivine index (1 um broad)'),
    'LCPINDEX2':     (calc_lcpindex2,     'LCP index (1.81 um broad)'),
    'HCPINDEX2':     (calc_hcpindex2,     'HCP index (2.12 um broad)'),
    'BD1900_2':      (calc_bd1900_2,      '1.9 um H2O'),
    'BD1900r2':      (calc_bd1900r2,      '1.9 um H2O ratio v2'),
    'BD2100_2':      (calc_bd2100_2,      '2.1 um H2O (monohydr. sulfates)'),
    'D2200':         (calc_d2200,         '2.2 um dropoff (Al-OH)'),
    'BD2210_2':      (calc_bd2210_2,      '2.21 um Al-OH'),
    'BD2165':        (calc_bd2165,        '2.165 um Al-OH (Pyrophyllite)'),
    'BD2190':        (calc_bd2190,        '2.19 um Al-OH (Beidellite)'),
    'BD2290':        (calc_bd2290,        '2.29 um Mg,Fe-OH / CO2 ice'),
    'D2300':         (calc_d2300,         '2.3 um dropoff (Fe/Mg-OH)'),
    'BD2355':        (calc_bd2355,        '2.35 um (Chlorite, Prehnite)'),
    'MIN2250':       (calc_min2250,       '2.25 um doublet (Opal)'),
    'BD2250':        (calc_bd2250,        '2.25 um broad Al/Si-OH (Opal)'),
    'SINDEX2':       (calc_sindex2,       'Sulfate index (2.29 um convexity)'),
    'BD2500_2':      (calc_bd2500_2,      '2.5 um Mg carbonate'),
    'MIN2295_2480':  (calc_min2295_2480,  'Mg carbonate doublet'),
    'MIN2345_2537':  (calc_min2345_2537,  'Ca/Fe carbonate doublet'),
    'CINDEX2':       (calc_cindex2,       'Carbonate 3.6 um convexity'),
}

PRESET_COMPOSITES = {
    'PFM':  ('BD2355',       'D2300',        'BD2290',   'Phyllosilicates with Fe and Mg'),
    'PHY':  ('D2300',        'D2200',        'BD1900r2', 'Phyllosilicates'),
    'MAF':  ('OLINDEX3',     'LCPINDEX2',    'HCPINDEX2','Mafic mineralogy'),
    'HYD':  ('SINDEX2',      'BD2100_2',     'BD1900_2', 'Hydrated mineralogy'),
    'PAL':  ('BD2210_2',     'BD2190',       'BD2165',   'Phyllosilicates with Al'),
    'HYS':  ('MIN2250',      'BD2250',       'BD1900r2', 'Hydrated silica'),
    'CR2':  ('MIN2295_2480', 'MIN2345_2537', 'CINDEX2',  'Carbonates v2'),
    'CAR':  ('D2300',        'BD2500_2',     'BD1900_2', 'Carbonates'),
}


def select_rgb_composite():
    param_names = list(PARAM_REGISTRY.keys())
    print("\n" + "=" * 70)
    print("  SPECTRAL PARAMETER RGB COMPOSITE SELECTION")
    print("=" * 70)
    print("\n  PRESET COMPOSITES:")
    print("  " + "-" * 50)
    for key, (r, g, b, desc) in PRESET_COMPOSITES.items():
        print(f"  {key:5s}  R={r}, G={g}, B={b}  ({desc})")
    print(f"\n  INDIVIDUAL PARAMETERS:")
    print("  " + "-" * 50)
    for i, name in enumerate(param_names):
        print(f"  {i+1:3d}) {name:14s} {PARAM_REGISTRY[name][1]}")
    print("\n  Enter preset (e.g. PFM, MAF) or three numbers (e.g. 1,5,10)")
    print("=" * 70)

    while True:
        choice = input("\n  Choice: ").strip().upper()
        if choice in PRESET_COMPOSITES:
            r, g, b, desc = PRESET_COMPOSITES[choice]
            print(f"  -> {choice}: R={r}, G={g}, B={b}")
            return r, g, b, f"{choice} ({desc})"
        try:
            parts = choice.replace(',', ' ').split()
            if len(parts) == 3:
                ri, gi, bi = [int(p) - 1 for p in parts]
                if all(0 <= x < len(param_names) for x in [ri, gi, bi]):
                    rn, gn, bn = param_names[ri], param_names[gi], param_names[bi]
                    print(f"  -> Custom: R={rn}, G={gn}, B={bn}")
                    return rn, gn, bn, f"Custom (R={rn}, G={gn}, B={bn})"
        except ValueError:
            pass
        print("  Invalid. Try again.")


def compute_param_rgb_raw(data, wav, r_name, g_name, b_name):
    cache = {}
    def get_param(name):
        if name not in cache:
            func, desc = PARAM_REGISTRY[name]
            print(f"  Computing {name}: {desc}...")
            cache[name] = func(data, wav)
        return cache[name]
    r, g, b = get_param(r_name), get_param(g_name), get_param(b_name)
    return np.dstack([r, g, b]).astype(np.float32)


def create_browse_ir_raw(data, wav, rw=2.529, gw=1.506, bw=1.080):
    r = data[np.abs(wav - rw).argmin()]
    g = data[np.abs(wav - gw).argmin()]
    b = data[np.abs(wav - bw).argmin()]
    return np.dstack([r, g, b]).astype(np.float32)


def dynamic_stretch(data, p_min=STRETCH_PMIN, p_max=STRETCH_PMAX):
    res = np.zeros_like(data, dtype=np.float32)
    for i in range(data.shape[2]):
        c = data[:, :, i]
        valid = c[np.isfinite(c)]
        if len(valid) > 0:
            vmin, vmax = np.percentile(valid, p_min), np.percentile(valid, p_max)
            if vmax > vmin:
                res[:, :, i] = np.clip((c - vmin) / (vmax - vmin), 0, 1)
    return res


# =============================================================================
# DYNAMIC STRETCH MANAGER (RGB panels only)
# =============================================================================

class DynamicStretchManager:
    def __init__(self, fig, panels, p_min=STRETCH_PMIN, p_max=STRETCH_PMAX):
        self.fig = fig
        self.panels = panels
        self.p_min = p_min
        self.p_max = p_max
        self._pending = False

        for p in self.panels:
            p['raw_rgb_display'] = apply_display_transform(p['raw_rgb'])

        self.restretch_all(force_full=True)

        for p in self.panels:
            p['ax'].callbacks.connect('xlim_changed', self._on_lim_changed)
            p['ax'].callbacks.connect('ylim_changed', self._on_lim_changed)

    def _on_lim_changed(self, ax):
        if self._pending:
            return
        self._pending = True
        try:
            self.fig.canvas.mpl_connect('draw_event', self._on_draw_once)
        except Exception:
            self._pending = False
            self.restretch_all()

    def _on_draw_once(self, event):
        if not self._pending:
            return
        self._pending = False
        self.restretch_all()
        self.fig.canvas.draw_idle()

    def restretch_all(self, force_full=False):
        for p in self.panels:
            self._restretch_panel(p, force_full=force_full)

    def _restretch_panel(self, p, force_full=False):
        img_disp = p['raw_rgb_display']
        ax = p['ax']
        H, W, _ = img_disp.shape

        if force_full:
            x0, x1, y0, y1 = 0, W, 0, H
        else:
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            x0 = max(0, int(np.floor(min(xlim))))
            x1 = min(W, int(np.ceil(max(xlim))))
            y0 = max(0, int(np.floor(min(ylim))))
            y1 = min(H, int(np.ceil(max(ylim))))
            if x1 - x0 < 2 or y1 - y0 < 2:
                x0, x1, y0, y1 = 0, W, 0, H

        sub = img_disp[y0:y1, x0:x1, :]
        stretched = np.zeros_like(img_disp, dtype=np.float32)
        for ci in range(3):
            full_c = img_disp[:, :, ci]
            sub_c = sub[:, :, ci]
            valid = sub_c[np.isfinite(sub_c)]
            if len(valid) < 10:
                valid = full_c[np.isfinite(full_c)]
            if len(valid) > 0:
                vmin, vmax = np.percentile(valid, self.p_min), np.percentile(valid, self.p_max)
                if vmax > vmin:
                    stretched[:, :, ci] = np.clip((full_c - vmin) / (vmax - vmin), 0, 1)
        p['im'].set_data(stretched)


# =============================================================================
# APP
# =============================================================================

class BrownROIExplorer:
    def __init__(self):
        self.data = None
        self.wav = None
        self.brown = {}     # 'centroid', 'asymmetry', 'amplitude', 'hwhm', ...
        self.base_name = None

        self.rois = []       # list of dicts: roi, pixels (display coords), data coords, param arrays, arts, color, label
        self.stretch_mgr = None

        self.load()
        self.init_plots()

    # -------------------------------------------------------------------------

    def load(self):
        # Cube
        self.data, self.wav = read_crism_ter3_cube(FILEPATH)
        print(f"  Wavelength range: {self.wav.min():.3f} - {self.wav.max():.3f} um")
        print(f"  Cube shape (bands, lines, samples): {self.data.shape}")

        # Brown maps
        self.brown, self.base_name = load_brown_maps(FILEPATH, BROWN_VERSION)

        # Sanity check shape
        expected = (self.data.shape[1], self.data.shape[2])
        for k in ['centroid', 'asymmetry', 'amplitude', 'hwhm']:
            if self.brown[k].shape != expected:
                raise ValueError(
                    f"Brown {k} shape {self.brown[k].shape} != cube "
                    f"(lines, samples) {expected}")
        print(f"  All Brown maps coregistered to cube grid {expected}. OK.")

        # RGB composite
        r_name, g_name, b_name, label = select_rgb_composite()
        self.composite_label = label
        self.rgb_r_name = r_name
        self.rgb_g_name = g_name
        self.rgb_b_name = b_name
        print(f"\nComputing parameter composite: {label}")
        self.rgb_param_raw = compute_param_rgb_raw(
            self.data, self.wav, r_name, g_name, b_name)
        self.rgb_ir_raw = create_browse_ir_raw(self.data, self.wav)

    # -------------------------------------------------------------------------
    # FIGURE SETUP
    # -------------------------------------------------------------------------

    def init_plots(self):
        # -- Map figure: 2x2 panels, shared zoom/pan ------------------------
        self.fig_map, axs = plt.subplots(2, 2, figsize=(14, 11))
        self.axs_map = axs.flatten()

        param_title = (f'Params ({self.rgb_r_name}, {self.rgb_g_name}, '
                       f'{self.rgb_b_name})')

        # Display-transformed Brown maps for the 2 scalar panels
        centroid_disp = apply_display_transform(self.brown['centroid'])
        asymm_disp = apply_display_transform(self.brown['asymmetry'])

        # Initial stretched RGBs (replaced by DynamicStretchManager below)
        disp_param = apply_display_transform(self.rgb_param_raw)
        disp_ir = apply_display_transform(self.rgb_ir_raw)
        init_param = dynamic_stretch(disp_param)
        init_ir = dynamic_stretch(disp_ir)

        im_param = self.axs_map[0].imshow(init_param, origin='upper',
                                          interpolation='nearest')
        im_ir = self.axs_map[1].imshow(init_ir, origin='upper',
                                       interpolation='nearest')
        im_cent = self.axs_map[2].imshow(
            centroid_disp, origin='upper', interpolation='nearest',
            cmap=CENTROID_CMAP, vmin=CENTROID_VMIN, vmax=CENTROID_VMAX)
        im_asym = self.axs_map[3].imshow(
            asymm_disp, origin='upper', interpolation='nearest',
            cmap=ASYMMETRY_CMAP, vmin=ASYMMETRY_VMIN, vmax=ASYMMETRY_VMAX)

        self.axs_map[0].set_title(param_title, fontsize=10)
        self.axs_map[1].set_title('IR False Color', fontsize=10)
        self.axs_map[2].set_title('Brown centroid (um)', fontsize=10)
        self.axs_map[3].set_title('Brown asymmetry', fontsize=10)

        # Colorbars for the scalar map panels
        cb1 = self.fig_map.colorbar(im_cent, ax=self.axs_map[2],
                                    shrink=0.75, pad=0.02)
        cb1.set_label('1 um centroid (um)', fontsize=9)
        cb2 = self.fig_map.colorbar(im_asym, ax=self.axs_map[3],
                                    shrink=0.75, pad=0.02)
        cb2.set_label('Asymmetry', fontsize=9)

        # Share axes for joint zoom/pan
        for ax in self.axs_map[1:]:
            ax.sharex(self.axs_map[0])
            ax.sharey(self.axs_map[0])

        # Dynamic stretch ONLY on the two RGB panels
        self.stretch_mgr = DynamicStretchManager(
            self.fig_map,
            [{'ax': self.axs_map[0], 'raw_rgb': self.rgb_param_raw, 'im': im_param},
             {'ax': self.axs_map[1], 'raw_rgb': self.rgb_ir_raw,    'im': im_ir}],
            p_min=STRETCH_PMIN, p_max=STRETCH_PMAX)

        self.fig_map.suptitle(
            f'{self.base_name}  |  Brown ROI Explorer  |  Lasso on panel 1',
            fontsize=11)

        # Status text
        self.status_txt = self.fig_map.text(
            0.5, 0.01, "", ha='center', fontweight='bold')

        # Lasso on the param RGB panel
        self.lasso = LassoSelector(self.axs_map[0], self.on_lasso, button=[1])

        # -- Brown exploration figure: 2x2 panels ---------------------------
        self.fig_brown, axs_b = plt.subplots(2, 2, figsize=(14, 11))
        self.ax_ca, self.ax_ch, self.ax_aa, self.ax_hist = axs_b.flatten()
        self.setup_brown_axes()

        # Keys
        for f in [self.fig_map, self.fig_brown]:
            f.canvas.mpl_connect('key_press_event', self.on_key)

        self.update_status()
        plt.show()

    def setup_brown_axes(self):
        # Centroid vs Asymmetry
        self.ax_ca.set_xlabel('1 um band centroid (um)')
        self.ax_ca.set_ylabel('Asymmetry')
        self.ax_ca.set_title('Centroid vs Asymmetry (Brown 2020 Fig. 5 analog)',
                             fontsize=10)
        self.ax_ca.axvline(BROWN_THRESHOLD_CENTROID, color='gray', ls=':',
                           lw=0.8, alpha=0.6)
        self.ax_ca.axvline(POPULATION_ZERO_CENTROID, color='darkred', ls='--',
                           lw=0.8, alpha=0.5)
        self.ax_ca.grid(True, alpha=0.3)

        # Centroid vs HWHM
        self.ax_ch.set_xlabel('1 um band centroid (um)')
        self.ax_ch.set_ylabel('HWHM (um)')
        self.ax_ch.set_title('Centroid vs HWHM (grain size sensitivity)',
                             fontsize=10)
        self.ax_ch.axvline(BROWN_THRESHOLD_CENTROID, color='gray', ls=':',
                           lw=0.8, alpha=0.6)
        self.ax_ch.grid(True, alpha=0.3)

        # Amplitude vs Asymmetry
        self.ax_aa.set_xlabel('Amplitude (band depth)')
        self.ax_aa.set_ylabel('Asymmetry')
        self.ax_aa.set_title('Amplitude vs Asymmetry', fontsize=10)
        self.ax_aa.grid(True, alpha=0.3)

        # Centroid histogram
        self.ax_hist.set_xlabel('1 um band centroid (um)')
        self.ax_hist.set_ylabel('Pixel count')
        self.ax_hist.set_title('Centroid distribution per ROI', fontsize=10)
        self.ax_hist.axvline(BROWN_THRESHOLD_CENTROID, color='gray', ls=':',
                             lw=0.8, alpha=0.6)
        self.ax_hist.axvline(POPULATION_ZERO_CENTROID, color='darkred',
                             ls='--', lw=0.8, alpha=0.5)
        self.ax_hist.grid(True, alpha=0.3)

        self.fig_brown.suptitle('Brown parameters per ROI (one marker per pixel)',
                                fontsize=11)

    def update_status(self):
        n = len(self.rois)
        msg = (f"Lasso on panel 1 to add ROI ({n} so far)  |  "
               f"d=delete last, r=reset, f=restretch, s=save CSV, q=quit")
        self.status_txt.set_text(msg)
        self.status_txt.set_bbox(dict(facecolor='lightblue', alpha=0.8))
        self.fig_map.canvas.draw_idle()

    # -------------------------------------------------------------------------
    # LASSO / ROI
    # -------------------------------------------------------------------------

    def get_pixels_in_lasso(self, verts):
        """Return list of (x_display, y_display) pixel centers inside lasso.

        These are coordinates in the *displayed* image's pixel grid."""
        path = Path(verts)
        h, w = self.data.shape[1], self.data.shape[2]
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        points = np.column_stack([xs.ravel(), ys.ravel()])
        mask = path.contains_points(points)
        return points[mask]  # Nx2 array of (x_display, y_display)

    def display_to_data_array(self, disp_pts):
        """Vectorized display-to-data conversion for Nx2 array."""
        h, w = self.data.shape[1], self.data.shape[2]
        x_data = disp_pts[:, 0].copy()
        y_data = disp_pts[:, 1].copy()
        if FLIP_HORIZONTAL:
            x_data = w - 1 - x_data
        if FLIP_VERTICAL:
            y_data = h - 1 - y_data
        # x -> sample, y -> line
        samples = x_data.astype(int)
        lines = y_data.astype(int)
        # Bounds (should already be safe, but clamp anyway)
        samples = np.clip(samples, 0, w - 1)
        lines = np.clip(lines, 0, h - 1)
        return lines, samples

    def on_lasso(self, verts):
        disp_pts = self.get_pixels_in_lasso(verts)
        if len(disp_pts) == 0:
            print("No pixels in lasso.")
            return

        lines, samples = self.display_to_data_array(disp_pts)

        # Pull Brown params for every pixel in this ROI
        centroid = self.brown['centroid'][lines, samples]
        asymm = self.brown['asymmetry'][lines, samples]
        amp = self.brown['amplitude'][lines, samples]
        hwhm = self.brown['hwhm'][lines, samples]

        # Keep only pixels where all four params are finite
        good = (np.isfinite(centroid) & np.isfinite(asymm) &
                np.isfinite(amp) & np.isfinite(hwhm))
        n_total = len(centroid)
        n_good = int(good.sum())

        if n_good == 0:
            print(f"  ROI has {n_total} pixels but 0 with valid Brown fits. "
                  "Skipping.")
            return

        roi_id = len(self.rois) + 1
        label = f"R{roi_id}"
        col = plt.cm.tab10((roi_id - 1) % 10)

        # Plot pixels on each Brown panel
        c = centroid[good]; a = asymm[good]; h = hwhm[good]; am = amp[good]

        s_ca = self.ax_ca.scatter(c, a, s=18, color=col, alpha=0.6,
                                  edgecolors='black', linewidths=0.3,
                                  label=f'{label} (n={n_good})')
        s_ch = self.ax_ch.scatter(c, h, s=18, color=col, alpha=0.6,
                                  edgecolors='black', linewidths=0.3,
                                  label=f'{label} (n={n_good})')
        s_aa = self.ax_aa.scatter(am, a, s=18, color=col, alpha=0.6,
                                  edgecolors='black', linewidths=0.3,
                                  label=f'{label} (n={n_good})')

        # Histogram (accumulates: re-draw whole thing each time)
        hist_art = None
        if n_good >= 3:
            _, _, patches = self.ax_hist.hist(
                c, bins=20, color=col, alpha=0.55,
                label=f'{label} (n={n_good})',
                edgecolor='black', linewidth=0.3)
            hist_art = patches
        else:
            # Too few for a hist: just rug-style scatter at the bottom
            y_rug = -0.5
            s_rug = self.ax_hist.scatter(c, np.full_like(c, y_rug),
                                         marker='|', color=col, s=150,
                                         linewidth=2, alpha=0.9,
                                         label=f'{label} (n={n_good})')
            hist_art = s_rug

        # ROI marker on each map panel (small dots at the display pixel centers)
        map_arts = []
        xs_disp = disp_pts[:, 0]
        ys_disp = disp_pts[:, 1]
        for ax in self.axs_map:
            m = ax.plot(xs_disp, ys_disp, 's', color=col, markersize=3,
                        alpha=0.35, markeredgecolor='white',
                        markeredgewidth=0.2, zorder=5)[0]
            map_arts.append(m)

        # ROI centroid label on panel 1
        cx, cy = float(np.mean(xs_disp)), float(np.mean(ys_disp))
        txt_art = self.axs_map[0].text(
            cx, cy, label,
            fontsize=10, color='white', fontweight='bold', ha='center', va='center',
            bbox=dict(facecolor=col, alpha=0.85, edgecolor='black', pad=1),
            zorder=6)

        # Update legends
        for ax in [self.ax_ca, self.ax_ch, self.ax_aa, self.ax_hist]:
            ax.legend(fontsize=8, loc='best')

        # Autoscale histogram y-axis (includes rug if any)
        self.ax_hist.relim()
        self.ax_hist.autoscale_view()

        self.fig_map.canvas.draw_idle()
        self.fig_brown.canvas.draw_idle()

        # Summary stats for this ROI
        print(f"\n  ROI {label} added: {n_good}/{n_total} pixels with valid Brown fits")
        print(f"    centroid  : mean={np.nanmean(c):.4f}  "
              f"median={np.nanmedian(c):.4f}  std={np.nanstd(c):.4f}  "
              f"range [{np.nanmin(c):.4f}, {np.nanmax(c):.4f}]")
        print(f"    asymmetry : mean={np.nanmean(a):.3f}  "
              f"median={np.nanmedian(a):.3f}  std={np.nanstd(a):.3f}")
        print(f"    hwhm      : mean={np.nanmean(h):.4f}  "
              f"median={np.nanmedian(h):.4f}  std={np.nanstd(h):.4f}")
        print(f"    amplitude : mean={np.nanmean(am):.4f}  "
              f"median={np.nanmedian(am):.4f}  std={np.nanstd(am):.4f}")

        self.rois.append({
            'id': roi_id,
            'label': label,
            'color': col,
            'n_total': n_total,
            'n_good': n_good,
            'lines': lines[good],
            'samples': samples[good],
            'disp_x': xs_disp,
            'disp_y': ys_disp,
            'centroid': c,
            'asymmetry': a,
            'hwhm': h,
            'amplitude': am,
            'arts': {
                'scatter_ca': s_ca,
                'scatter_ch': s_ch,
                'scatter_aa': s_aa,
                'hist': hist_art,
                'map_markers': map_arts,
                'label_txt': txt_art,
            },
        })
        self.update_status()

    # -------------------------------------------------------------------------
    # KEYS
    # -------------------------------------------------------------------------

    def on_key(self, event):
        if event.key == 'q':
            plt.close('all')
        elif event.key == 'd':
            self.delete_last()
        elif event.key == 'r':
            self.reset()
        elif event.key == 'f':
            if self.stretch_mgr is not None:
                self.stretch_mgr.restretch_all()
                self.fig_map.canvas.draw_idle()
                print("Forced restretch on current view.")
        elif event.key == 's':
            self.save_csv()

    def delete_last(self):
        if not self.rois:
            return
        roi = self.rois.pop()
        arts = roi['arts']

        # Remove scatters
        for key in ['scatter_ca', 'scatter_ch', 'scatter_aa']:
            try:
                arts[key].remove()
            except Exception:
                pass

        # Remove histogram (either BarContainer patches or a rug scatter)
        h = arts['hist']
        try:
            for p in h:
                p.remove()
        except TypeError:
            try:
                h.remove()
            except Exception:
                pass
        except Exception:
            pass

        # Map markers + label
        for m in arts['map_markers']:
            try:
                m.remove()
            except Exception:
                pass
        try:
            arts['label_txt'].remove()
        except Exception:
            pass

        # Refresh legends
        for ax in [self.ax_ca, self.ax_ch, self.ax_aa, self.ax_hist]:
            if ax.get_legend() is not None:
                ax.legend(fontsize=8, loc='best')

        self.ax_hist.relim()
        self.ax_hist.autoscale_view()

        self.fig_map.canvas.draw_idle()
        self.fig_brown.canvas.draw_idle()
        print(f"Deleted ROI {roi['label']}")
        self.update_status()

    def reset(self):
        while self.rois:
            self.delete_last()
        print("All ROIs cleared.")
        self.update_status()

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------

    def save_csv(self):
        if not self.rois:
            print("No ROIs to save.")
            return

        try:
            out_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            out_dir = os.getcwd()

        fpath = os.path.join(out_dir,
                             f"{self.base_name}_brown_roi_pixels.csv")

        # Per-pixel rows
        rows = []
        for roi in self.rois:
            for i in range(roi['n_good']):
                rows.append({
                    'cube': self.base_name,
                    'roi_id': roi['id'],
                    'roi_label': roi['label'],
                    'line': int(roi['lines'][i]),
                    'sample': int(roi['samples'][i]),
                    'centroid': float(roi['centroid'][i]),
                    'asymmetry': float(roi['asymmetry'][i]),
                    'hwhm': float(roi['hwhm'][i]),
                    'amplitude': float(roi['amplitude'][i]),
                })

        df_new = pd.DataFrame(rows)

        if os.path.exists(fpath):
            try:
                df_old = pd.read_csv(fpath)
                # Bump ROI ids so they don't collide
                if 'roi_id' in df_old.columns and len(df_old) > 0:
                    offset = int(df_old['roi_id'].max())
                    df_new['roi_id'] = df_new['roi_id'] + offset
                    df_new['roi_label'] = df_new['roi_id'].apply(lambda x: f"R{x}")
                df_out = pd.concat([df_old, df_new], ignore_index=True)
            except Exception as e:
                print(f"  Could not merge with existing file ({e}). "
                      "Writing new with timestamp.")
                ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                fpath = fpath.replace('.csv', f'_{ts}.csv')
                df_out = df_new
        else:
            df_out = df_new

        df_out.to_csv(fpath, index=False)
        print(f"  Saved {len(df_new)} pixels across {len(self.rois)} ROIs to:")
        print(f"    {fpath}")

        # Also save an ROI-mean summary alongside
        summary_path = fpath.replace('.csv', '_summary.csv')
        sum_rows = []
        for roi in self.rois:
            sum_rows.append({
                'cube': self.base_name,
                'roi_id': roi['id'],
                'roi_label': roi['label'],
                'n_pixels': roi['n_good'],
                'centroid_mean':  float(np.nanmean(roi['centroid'])),
                'centroid_median':float(np.nanmedian(roi['centroid'])),
                'centroid_std':   float(np.nanstd(roi['centroid'])),
                'asymmetry_mean': float(np.nanmean(roi['asymmetry'])),
                'asymmetry_median':float(np.nanmedian(roi['asymmetry'])),
                'asymmetry_std':  float(np.nanstd(roi['asymmetry'])),
                'hwhm_mean':      float(np.nanmean(roi['hwhm'])),
                'hwhm_median':    float(np.nanmedian(roi['hwhm'])),
                'hwhm_std':       float(np.nanstd(roi['hwhm'])),
                'amplitude_mean': float(np.nanmean(roi['amplitude'])),
                'amplitude_median':float(np.nanmedian(roi['amplitude'])),
                'amplitude_std':  float(np.nanstd(roi['amplitude'])),
            })
        df_sum = pd.DataFrame(sum_rows)
        if os.path.exists(summary_path):
            try:
                df_old_sum = pd.read_csv(summary_path)
                df_sum = pd.concat([df_old_sum, df_sum], ignore_index=True)
            except Exception:
                pass
        df_sum.to_csv(summary_path, index=False)
        print(f"  Saved ROI summary to:")
        print(f"    {summary_path}")


# =============================================================================

if __name__ == "__main__":
    app = BrownROIExplorer()