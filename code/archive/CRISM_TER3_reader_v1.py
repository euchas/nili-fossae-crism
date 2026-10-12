"""
CRISM TER3 (Joined) Interactive Spectra Extractor (v4.2-TER3)
-------------------------------------------------------------
Same as v4.1-TER3, but with DYNAMIC STRETCHING tied to the map view:
every time you zoom or pan any of the map panels, the RGB composites
are re-stretched using only the pixels currently visible in the view.
This greatly improves contrast when zooming into small regions.

TER3 NOTES:
  - TER3 files are joined VNIR+IR cubes (MRO:SENSOR_ID = "J").
  - Wavelengths are stored in NANOMETERS in the ENVI header - converted to um.
  - Both 'S' and 'L' detectors are already merged into a single cube, so:
      * No VNIR/IR stitching is performed.
      * No S->L pixel shift is applied.
      * A single raw and a single ratioed spectrum are plotted per ROI.

Features retained from v4.1:
  - Lasso ROI selection (sample first, then dust).
  - Full Viviano-Beck (2014) spectral parameter registry + RGB presets.
  - Band fitting: press 'b' then click two shoulders on ratioed plot.
    Fits a 4th-order polynomial to the continuum-removed absorption
    and reports band center, depth, FWHM for all plotted spectra.
  - CSV saving with appending.

New in v4.2:
  - Dynamic per-view restretching on zoom/pan for all map panels.
  - Press 'f' on a map figure to force a restretch of all panels.

Author: AV Valantinas (Modified for TER3)
"""

import numpy as np
import matplotlib.pyplot as plt
from config import CUBES_DIR, STYLE_PATH
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import warnings
import os
import pandas as pd

warnings.filterwarnings('ignore')

# =============================================================================
# USER CONFIGURATION
# =============================================================================
FILEPATH = str(CUBES_DIR / "FRT00003E12_07_IF166J_TER3" / "frt00003e12_07_if166j_ter3.img")

FLIP_VERTICAL = True
FLIP_HORIZONTAL = True

# Percentile limits used for every stretch (full-scene AND dynamic per-view)
STRETCH_PMIN = 2
STRETCH_PMAX = 98

style_path = str(STYLE_PATH)
if os.path.exists(style_path): plt.style.use(style_path)

def apply_display_transform(image):
    result = image.copy()
    if FLIP_VERTICAL: result = np.flipud(result)
    if FLIP_HORIZONTAL: result = np.fliplr(result)
    return result

def display_to_data_coords(x, y, shape):
    lines, samples = shape[0], shape[1]
    x_data, y_data = x, y
    if FLIP_HORIZONTAL: x_data = samples - 1 - x
    if FLIP_VERTICAL: y_data = lines - 1 - y
    return x_data, y_data

# =============================================================================
# DATA LOADING
# =============================================================================

def parse_envi_header(hdr_file):
    metadata = {}
    with open(hdr_file, 'r') as f: content = f.read()
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

def parse_pds_label(lbl_file):
    metadata = {}
    with open(lbl_file, 'r') as f: content = f.read()
    for line in content.split('\n'):
        line = line.strip()
        if '=' in line and not line.startswith('/*'):
            parts = line.split('=', 1)
            if len(parts) == 2:
                key, val = parts[0].strip(), parts[1].strip().strip('"\'')
                if key == 'LINES': metadata['lines'] = val
                elif key == 'LINE_SAMPLES': metadata['samples'] = val
                elif key == 'BANDS': metadata['bands'] = val
                elif key == 'BAND_STORAGE_TYPE':
                    metadata['interleave'] = 'bil' if 'LINE' in val else 'bip' if 'PIXEL' in val else 'bsq'
                elif key == 'SAMPLE_BITS': metadata['data type'] = '4' if val == '32' else '2'
                elif key == 'SAMPLE_TYPE': metadata['byte order'] = '0' if 'PC_REAL' in val else '1'
                elif key == 'MRO:SENSOR_ID': metadata['sensor_id'] = val
    return metadata

def read_crism_ter3_cube(img_file):
    """Read a joined CRISM TER3 cube. Wavelengths returned in um."""
    hdr = img_file.replace('.img', '.hdr').replace('.IMG', '.hdr')
    if not os.path.exists(hdr): hdr = img_file + '.hdr'
    lbl = img_file.replace('.img', '.lbl').replace('.IMG', '.lbl')
    if not os.path.exists(lbl): lbl = img_file.replace('.img', '.LBL').replace('.IMG', '.LBL')

    if os.path.exists(hdr):
        meta, use_envi = parse_envi_header(hdr), True
    elif os.path.exists(lbl):
        meta, use_envi = parse_pds_label(lbl), False
    else:
        raise ValueError(f"No header/label for {img_file}")

    s, l, b = int(meta.get('samples', 0)), int(meta.get('lines', 0)), int(meta.get('bands', 0))
    dt, order = int(meta.get('data type', 4)), int(meta.get('byte order', 0))
    inter = meta.get('interleave', 'bsq').lower()

    print(f"Reading {img_file} ({s}x{l}x{b})")

    if use_envi and meta.get('wavelength'):
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
    if order == 1: data = data.byteswap()

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

def dynamic_stretch(data, p_min=STRETCH_PMIN, p_max=STRETCH_PMAX):
    """Per-channel percentile stretch. `data` must be [H, W, 3] float."""
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
# SPECTRAL PARAMETERS - Viviano-Beck et al. (2014), vectorized
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

# --- Mafic indices ---

def calc_olindex3(data, wav):
    R1210 = get_kernel(data, wav, 1.210, 7)
    R1250 = get_kernel(data, wav, 1.250, 7)
    R1263 = get_kernel(data, wav, 1.263, 7)
    R1276 = get_kernel(data, wav, 1.276, 7)
    R1330 = get_kernel(data, wav, 1.330, 7)
    R1750 = get_kernel(data, wav, 1.750, 7)
    R1862 = get_kernel(data, wav, 1.862, 7)

    slope = np.where((R1862 != R1750), (R1862 - R1750) / (1.862 - 1.750), 0.0)
    def rc(w): return R1750 + slope * (w - 1.750)

    return (0.1 * _rb(R1210, rc(1.210)) + 0.1 * _rb(R1250, rc(1.250)) +
            0.2 * _rb(R1263, rc(1.263)) + 0.2 * _rb(R1276, rc(1.276)) +
            0.4 * _rb(R1330, rc(1.330)))

def calc_olindex2(data, wav):
    R1080 = get_kernel(data, wav, 1.080, 7)
    R1210 = get_kernel(data, wav, 1.210, 7)
    R1330 = get_kernel(data, wav, 1.330, 7)
    R1470 = get_kernel(data, wav, 1.470, 7)
    R1750 = get_kernel(data, wav, 1.750, 7)
    R2400 = get_kernel(data, wav, 2.400, 7)

    slope = np.where((R2400 != R1750), (R2400 - R1750) / (2.400 - 1.750), 0.0)
    def rc(w): return R1750 + slope * (w - 1.750)

    return (0.1 * _rb(R1080, rc(1.080)) + 0.1 * _rb(R1210, rc(1.210)) +
            0.4 * _rb(R1330, rc(1.330)) + 0.4 * _rb(R1470, rc(1.470)))

def calc_lcpindex2(data, wav):
    R1560 = get_kernel(data, wav, 1.560, 7)
    R1690 = get_kernel(data, wav, 1.690, 7)
    R1750 = get_kernel(data, wav, 1.750, 7)
    R1810 = get_kernel(data, wav, 1.810, 7)
    R1870 = get_kernel(data, wav, 1.870, 7)
    R2450 = get_kernel(data, wav, 2.450, 7)

    slope = np.where((R2450 != R1560), (R2450 - R1560) / (2.450 - 1.560), 0.0)
    def rc(w): return R1560 + slope * (w - 1.560)

    return (0.20 * _rb(R1690, rc(1.690)) + 0.20 * _rb(R1750, rc(1.750)) +
            0.30 * _rb(R1810, rc(1.810)) + 0.30 * _rb(R1870, rc(1.870)))

def calc_hcpindex2(data, wav):
    R1690 = get_kernel(data, wav, 1.690, 7)
    R2120 = get_kernel(data, wav, 2.120, 5)
    R2140 = get_kernel(data, wav, 2.140, 7)
    R2230 = get_kernel(data, wav, 2.230, 7)
    R2250 = get_kernel(data, wav, 2.250, 7)
    R2430 = get_kernel(data, wav, 2.430, 7)
    R2460 = get_kernel(data, wav, 2.460, 7)
    R2530 = get_kernel(data, wav, 2.530, 7)

    slope = np.where((R2530 != R1690), (R2530 - R1690) / (2.530 - 1.690), 0.0)
    def rc(w): return R1690 + slope * (w - 1.690)

    return (0.10 * _rb(R2120, rc(2.120)) + 0.10 * _rb(R2140, rc(2.140)) +
            0.15 * _rb(R2230, rc(2.230)) + 0.30 * _rb(R2250, rc(2.250)) +
            0.20 * _rb(R2430, rc(2.430)) + 0.15 * _rb(R2460, rc(2.460)))

def calc_lcpindex(data, wav):
    R1050 = get_kernel(data, wav, 1.050, 5)
    R1330 = get_kernel(data, wav, 1.330, 5)
    R1815 = get_kernel(data, wav, 1.815, 5)

    d1 = R1330 + R1050
    d2 = R1330 + R1815
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = np.where(d1 != 0, (R1330 - R1050) / d1, 0.0)
        t2 = np.where(d2 != 0, (R1330 - R1815) / d2, 0.0)
    return np.nan_to_num(t1 * t2, nan=0.0)

def calc_hcpindex(data, wav):
    R1050 = get_kernel(data, wav, 1.050, 5)
    R1470 = get_kernel(data, wav, 1.470, 5)
    R2067 = get_kernel(data, wav, 2.067, 5)

    d1 = R1470 + R1050
    d2 = R1470 + R2067
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = np.where(d1 != 0, (R1470 - R1050) / d1, 0.0)
        t2 = np.where(d2 != 0, (R1470 - R2067) / d2, 0.0)
    return np.nan_to_num(t1 * t2, nan=0.0)

# --- Hydration / H2O indices ---

def calc_bd1400(data, wav):
    R1395 = get_kernel(data, wav, 1.395, 3)
    R1330 = get_kernel(data, wav, 1.330, 5)
    R1467 = get_kernel(data, wav, 1.467, 5)
    b_coeff = (1.395 - 1.330) / (1.467 - 1.330)
    a_coeff = 1.0 - b_coeff
    return _ratio(R1395, a_coeff * R1330 + b_coeff * R1467)

def calc_bd1750_2(data, wav):
    R1750 = get_kernel(data, wav, 1.750, 3)
    R1690 = get_kernel(data, wav, 1.690, 5)
    R1815 = get_kernel(data, wav, 1.815, 5)
    b_coeff = (1.750 - 1.690) / (1.815 - 1.690)
    a_coeff = 1.0 - b_coeff
    return _ratio(R1750, a_coeff * R1690 + b_coeff * R1815)

def calc_bd1900r(data, wav):
    num_w = [1.908, 1.914, 1.921, 1.928, 1.934, 1.941]
    den_w = [1.862, 1.869, 1.875, 2.112, 2.120, 2.126]
    num = sum(get_kernel(data, wav, w, 1) for w in num_w)
    den = sum(get_kernel(data, wav, w, 1) for w in den_w)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(den != 0, 1.0 - num / den, 0.0)
    return np.nan_to_num(result, nan=0.0)

def calc_bd1900r2(data, wav):
    num_w = [1.908, 1.914, 1.921, 1.928, 1.934, 1.941]
    den_w = [1.862, 1.869, 1.875, 2.112, 2.120, 2.126]
    R1850 = get_kernel(data, wav, 1.850, 5)
    R2060 = get_kernel(data, wav, 2.060, 5)
    slope = np.where((R2060 != R1850), (R2060 - R1850) / (2.060 - 1.850), 0.0)
    def rc(w): return R1850 + slope * (w - 1.850)
    with np.errstate(divide='ignore', invalid='ignore'):
        num_sum = sum(np.where(rc(w) != 0, get_kernel(data, wav, w, 1) / rc(w), 0.0) for w in num_w)
        den_sum = sum(np.where(rc(w) != 0, get_kernel(data, wav, w, 1) / rc(w), 0.0) for w in den_w)
        result = np.where(den_sum != 0, 1.0 - num_sum / den_sum, 0.0)
    return np.nan_to_num(result, nan=0.0)

def calc_bd1900_2(data, wav):
    R1850 = get_kernel(data, wav, 1.850, 5)
    R1930 = get_kernel(data, wav, 1.930, 5)
    R1985 = get_kernel(data, wav, 1.985, 5)
    R2067 = get_kernel(data, wav, 2.067, 5)
    b_coeff = (1.930 - 1.850) / (2.067 - 1.850)
    a_coeff = 1.0 - b_coeff
    den = a_coeff * R1850 + b_coeff * R2067
    bd1 = _ratio(R1930, den)
    b2 = (1.985 - 1.850) / (2.067 - 1.850)
    a2 = 1.0 - b2
    den2 = a2 * R1850 + b2 * R2067
    bd2 = _ratio(R1985, den2)
    return 0.5 * bd1 + 0.5 * bd2

def calc_bd2100_2(data, wav):
    R2132 = get_kernel(data, wav, 2.132, 5)
    R1930 = get_kernel(data, wav, 1.930, 3)
    R2250 = get_kernel(data, wav, 2.250, 3)
    b_coeff = (2.132 - 1.930) / (2.250 - 1.930)
    a_coeff = 1.0 - b_coeff
    return _ratio(R2132, a_coeff * R1930 + b_coeff * R2250)

# --- Al-OH / Si-OH indices ---

def calc_d2200(data, wav):
    R1815 = get_kernel(data, wav, 1.815, 7)
    R2165 = get_kernel(data, wav, 2.165, 5)
    R2210 = get_kernel(data, wav, 2.210, 7)
    R2230 = get_kernel(data, wav, 2.230, 7)
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
    R2165 = get_kernel(data, wav, 2.165, 5)
    R2210 = get_kernel(data, wav, 2.210, 5)
    R2250 = get_kernel(data, wav, 2.250, 5)
    b_coeff = (2.210 - 2.165) / (2.250 - 2.165)
    a_coeff = 1.0 - b_coeff
    return _ratio(R2210, a_coeff * R2165 + b_coeff * R2250)

def calc_bd2165(data, wav):
    R2120 = get_kernel(data, wav, 2.120, 5)
    R2165 = get_kernel(data, wav, 2.165, 3)
    R2230 = get_kernel(data, wav, 2.230, 3)
    b_coeff = (2.165 - 2.120) / (2.230 - 2.120)
    a_coeff = 1.0 - b_coeff
    return _ratio(R2165, a_coeff * R2120 + b_coeff * R2230)

def calc_bd2190(data, wav):
    R2120 = get_kernel(data, wav, 2.120, 5)
    R2185 = get_kernel(data, wav, 2.185, 3)
    R2250 = get_kernel(data, wav, 2.250, 3)
    b_coeff = (2.185 - 2.120) / (2.250 - 2.120)
    a_coeff = 1.0 - b_coeff
    return _ratio(R2185, a_coeff * R2120 + b_coeff * R2250)

def calc_min2200(data, wav):
    R2120 = get_kernel(data, wav, 2.120, 5)
    R2165 = get_kernel(data, wav, 2.165, 3)
    R2210 = get_kernel(data, wav, 2.210, 3)
    R2350 = get_kernel(data, wav, 2.350, 5)
    b_coeff = (2.165 - 2.120) / (2.350 - 2.120)
    a_coeff = 1.0 - b_coeff
    den = a_coeff * R2120 + b_coeff * R2350
    bd1 = _ratio(R2165, den)
    bd2 = _ratio(R2210, den)
    return np.minimum(bd1, bd2)

def calc_min2250(data, wav):
    R2165 = get_kernel(data, wav, 2.165, 5)
    R2210 = get_kernel(data, wav, 2.210, 3)
    R2265 = get_kernel(data, wav, 2.265, 3)
    R2350 = get_kernel(data, wav, 2.350, 5)
    b_coeff = (2.210 - 2.165) / (2.350 - 2.165)
    a_coeff = 1.0 - b_coeff
    den = a_coeff * R2165 + b_coeff * R2350
    bd1 = _ratio(R2210, den)
    bd2 = _ratio(R2265, den)
    return np.minimum(bd1, bd2)

def calc_bd2250(data, wav):
    R2120 = get_kernel(data, wav, 2.120, 5)
    R2245 = get_kernel(data, wav, 2.245, 7)
    R2340 = get_kernel(data, wav, 2.340, 3)
    b_coeff = (2.245 - 2.120) / (2.340 - 2.120)
    a_coeff = 1.0 - b_coeff
    return _ratio(R2245, a_coeff * R2120 + b_coeff * R2340)

# --- Fe/Mg-OH and phyllosilicate indices ---

def calc_bd2290(data, wav):
    R2250 = get_kernel(data, wav, 2.250, 5)
    R2290 = get_kernel(data, wav, 2.290, 5)
    R2350 = get_kernel(data, wav, 2.350, 5)
    b_coeff = (2.290 - 2.250) / (2.350 - 2.250)
    a_coeff = 1.0 - b_coeff
    return _ratio(R2290, a_coeff * R2250 + b_coeff * R2350)

def calc_d2300(data, wav):
    R1815 = get_kernel(data, wav, 1.815, 5)
    R2120 = get_kernel(data, wav, 2.120, 5)
    R2170 = get_kernel(data, wav, 2.170, 5)
    R2210 = get_kernel(data, wav, 2.210, 5)
    R2290 = get_kernel(data, wav, 2.290, 3)
    R2320 = get_kernel(data, wav, 2.320, 3)
    R2330 = get_kernel(data, wav, 2.330, 3)
    R2530 = get_kernel(data, wav, 2.530, 5)
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
    R2300 = get_kernel(data, wav, 2.300, 5)
    R2355 = get_kernel(data, wav, 2.355, 5)
    R2450 = get_kernel(data, wav, 2.450, 5)
    b_coeff = (2.355 - 2.300) / (2.450 - 2.300)
    a_coeff = 1.0 - b_coeff
    return _ratio(R2355, a_coeff * R2300 + b_coeff * R2450)

def calc_bd2230(data, wav):
    R2210 = get_kernel(data, wav, 2.210, 3)
    R2235 = get_kernel(data, wav, 2.235, 3)
    R2252 = get_kernel(data, wav, 2.252, 3)
    b_coeff = (2.235 - 2.210) / (2.252 - 2.210)
    a_coeff = 1.0 - b_coeff
    return _ratio(R2235, a_coeff * R2210 + b_coeff * R2252)

# --- Sulfate indices ---

def calc_sindex2(data, wav):
    R2120 = get_kernel(data, wav, 2.120, 5)
    R2290 = get_kernel(data, wav, 2.290, 7)
    R2400 = get_kernel(data, wav, 2.400, 3)
    b_coeff = (2.290 - 2.120) / (2.400 - 2.120)
    a_coeff = 1.0 - b_coeff
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(R2290 != 0, 1.0 - (a_coeff * R2120 + b_coeff * R2400) / R2290, 0.0)
    return np.nan_to_num(result, nan=0.0)

# --- Carbonate indices ---

def calc_bd2500_2(data, wav):
    R2364 = get_kernel(data, wav, 2.364, 5)
    R2480 = get_kernel(data, wav, 2.480, 5)
    R2570 = get_kernel(data, wav, 2.570, 5)
    b_coeff = (2.480 - 2.364) / (2.570 - 2.364)
    a_coeff = 1.0 - b_coeff
    return _ratio(R2480, a_coeff * R2364 + b_coeff * R2570)

def calc_min2295_2480(data, wav):
    R2165 = get_kernel(data, wav, 2.165, 5)
    R2295 = get_kernel(data, wav, 2.295, 5)
    R2364 = get_kernel(data, wav, 2.364, 5)
    R2480 = get_kernel(data, wav, 2.480, 5)
    R2570 = get_kernel(data, wav, 2.570, 5)
    b1 = (2.295 - 2.165) / (2.364 - 2.165)
    a1 = 1.0 - b1
    bd1 = _ratio(R2295, a1 * R2165 + b1 * R2364)
    b2 = (2.480 - 2.364) / (2.570 - 2.364)
    a2 = 1.0 - b2
    bd2 = _ratio(R2480, a2 * R2364 + b2 * R2570)
    return np.minimum(bd1, bd2)

def calc_min2345_2537(data, wav):
    R2250 = get_kernel(data, wav, 2.250, 5)
    R2345 = get_kernel(data, wav, 2.345, 5)
    R2430 = get_kernel(data, wav, 2.430, 5)
    R2537 = get_kernel(data, wav, 2.537, 5)
    R2602 = get_kernel(data, wav, 2.602, 5)
    b1 = (2.345 - 2.250) / (2.430 - 2.250)
    a1 = 1.0 - b1
    bd1 = _ratio(R2345, a1 * R2250 + b1 * R2430)
    b2 = (2.537 - 2.430) / (2.602 - 2.430)
    a2 = 1.0 - b2
    bd2 = _ratio(R2537, a2 * R2430 + b2 * R2602)
    return np.minimum(bd1, bd2)

def calc_cindex2(data, wav):
    R3450 = get_kernel(data, wav, 3.450, 9)
    R3610 = get_kernel(data, wav, 3.610, 11)
    R3875 = get_kernel(data, wav, 3.875, 7)
    b_coeff = (3.610 - 3.450) / (3.875 - 3.450)
    a_coeff = 1.0 - b_coeff
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(R3610 != 0, 1.0 - (a_coeff * R3450 + b_coeff * R3875) / R3610, 0.0)
    return np.nan_to_num(result, nan=0.0)

# --- Slope / albedo ---

def calc_islope1(data, wav):
    R1815 = get_kernel(data, wav, 1.815, 5)
    R2530 = get_kernel(data, wav, 2.530, 5)
    W1815 = wav[np.abs(wav - 1.815).argmin()]
    W2530 = wav[np.abs(wav - 2.530).argmin()]
    return np.nan_to_num((R1815 - R2530) / (W2530 - W1815), nan=0.0)

def calc_r770(data, wav):  return get_kernel(data, wav, 0.770, 5)
def calc_r1330(data, wav): return get_kernel(data, wav, 1.330, 11)
def calc_r2529(data, wav): return get_kernel(data, wav, 2.529, 5)

# --- Misc ---

def calc_bd2265(data, wav):
    R2210 = get_kernel(data, wav, 2.210, 5)
    R2265 = get_kernel(data, wav, 2.265, 3)
    R2340 = get_kernel(data, wav, 2.340, 5)
    b_coeff = (2.265 - 2.210) / (2.340 - 2.210)
    a_coeff = 1.0 - b_coeff
    return _ratio(R2265, a_coeff * R2210 + b_coeff * R2340)

def calc_bd3000(data, wav):
    R2210 = get_kernel(data, wav, 2.210, 5)
    R2530 = get_kernel(data, wav, 2.530, 5)
    R3000 = get_kernel(data, wav, 3.000, 5)
    with np.errstate(divide='ignore', invalid='ignore'):
        den = R2530 * np.where(R2210 != 0, R2530 / R2210, 0.0)
        result = np.where(den != 0, 1.0 - R3000 / den, 0.0)
    return np.nan_to_num(result, nan=0.0)

def calc_doub2200h(data, wav):
    R2172 = get_kernel(data, wav, 2.172, 5)
    R2205 = get_kernel(data, wav, 2.205, 3)
    R2258 = get_kernel(data, wav, 2.258, 3)
    R2311 = get_kernel(data, wav, 2.311, 5)
    den = R2172 + R2311
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(den != 0, 1.0 - (R2205 + R2258) / den, 0.0)
    return np.nan_to_num(result, nan=0.0)


# =============================================================================
# PARAMETER REGISTRY & RGB COMPOSITE SELECTION
# =============================================================================

PARAM_REGISTRY = {
    'OLINDEX3':      (calc_olindex3,      'Olivine index (1 um broad)'),
    'OLINDEX2':      (calc_olindex2,      'Olivine index v2 (retired)'),
    'LCPINDEX2':     (calc_lcpindex2,     'LCP index (1.81 um broad)'),
    'LCPINDEX':      (calc_lcpindex,      'LCP index (retired)'),
    'HCPINDEX2':     (calc_hcpindex2,     'HCP index (2.12 um broad)'),
    'HCPINDEX':      (calc_hcpindex,      'HCP index (retired)'),
    'BD1400':        (calc_bd1400,        '1.4 um H2O/OH'),
    'BD1750_2':      (calc_bd1750_2,      '1.7 um H2O (Gypsum, Alunite)'),
    'BD1900_2':      (calc_bd1900_2,      '1.9 um H2O'),
    'BD1900r':       (calc_bd1900r,       '1.9 um H2O ratio (retired)'),
    'BD1900r2':      (calc_bd1900r2,      '1.9 um H2O ratio v2'),
    'BD2100_2':      (calc_bd2100_2,      '2.1 um H2O (monohydr. sulfates)'),
    'BD2165':        (calc_bd2165,        '2.165 um Al-OH (Pyrophyllite)'),
    'BD2190':        (calc_bd2190,        '2.19 um Al-OH (Beidellite)'),
    'D2200':         (calc_d2200,         '2.2 um dropoff (Al-OH)'),
    'BD2210_2':      (calc_bd2210_2,      '2.21 um Al-OH'),
    'DOUB2200H':     (calc_doub2200h,     '2.2 um Si-OH doublet (retired)'),
    'MIN2200':       (calc_min2200,       '2.2 um doublet (Kaolinite)'),
    'BD2230':        (calc_bd2230,        '2.23 um (Hydrox. ferric sulfate)'),
    'BD2250':        (calc_bd2250,        '2.25 um broad Al/Si-OH (Opal)'),
    'MIN2250':       (calc_min2250,       '2.25 um doublet (Opal)'),
    'BD2265':        (calc_bd2265,        '2.265 um (Jarosite, Gibbsite)'),
    'BD2290':        (calc_bd2290,        '2.29 um Mg,Fe-OH / CO2 ice'),
    'D2300':         (calc_d2300,         '2.3 um dropoff (Fe/Mg-OH)'),
    'BD2355':        (calc_bd2355,        '2.35 um (Chlorite, Prehnite)'),
    'SINDEX2':       (calc_sindex2,       'Sulfate index (2.29 um convexity)'),
    'BD2500_2':      (calc_bd2500_2,      '2.5 um Mg carbonate'),
    'MIN2295_2480':  (calc_min2295_2480,  'Mg carbonate doublet'),
    'MIN2345_2537':  (calc_min2345_2537,  'Ca/Fe carbonate doublet'),
    'CINDEX2':       (calc_cindex2,       'Carbonate 3.6 um convexity'),
    'BD3000':        (calc_bd3000,        '3 um bound H2O'),
    'ISLOPE1':       (calc_islope1,       'Spectral slope 1'),
    'R770':          (calc_r770,          '0.77 um reflectance'),
    'R1330':         (calc_r1330,         'IR albedo (1.33 um)'),
    'R2529':         (calc_r2529,         '2.53 um reflectance'),
}

PRESET_COMPOSITES = {
    'PFM':  ('BD2355',  'D2300',      'BD2290',     'Phyllosilicates with Fe and Mg'),
    'PHY':  ('D2300',   'D2200',      'BD1900r2',   'Phyllosilicates'),
    'MAF':  ('OLINDEX3','LCPINDEX2',  'HCPINDEX2',  'Mafic mineralogy'),
    'HYD':  ('SINDEX2', 'BD2100_2',   'BD1900_2',   'Hydrated mineralogy'),
    'PAL':  ('BD2210_2','BD2190',     'BD2165',     'Phyllosilicates with Al'),
    'HYS':  ('MIN2250', 'BD2250',     'BD1900r2',   'Hydrated silica'),
    'CR2':  ('MIN2295_2480','MIN2345_2537','CINDEX2','Carbonates v2'),
    'CAR':  ('D2300',   'BD2500_2',   'BD1900_2',   'Carbonates'),
}


def select_rgb_composite():
    param_names = list(PARAM_REGISTRY.keys())
    print("\n" + "=" * 70)
    print("  SPECTRAL PARAMETER RGB COMPOSITE SELECTION")
    print("=" * 70)
    print("\n  PRESET COMPOSITES:")
    print("  " + "-" * 50)
    preset_keys = list(PRESET_COMPOSITES.keys())
    for i, key in enumerate(preset_keys):
        r, g, b, desc = PRESET_COMPOSITES[key]
        print(f"  P{i+1}) {key:5s}  R={r}, G={g}, B={b}")
        print(f"          {desc}")
    print(f"\n  INDIVIDUAL PARAMETERS (for custom RGB):")
    print("  " + "-" * 50)
    for i, name in enumerate(param_names):
        _, desc = PARAM_REGISTRY[name]
        print(f"  {i+1:3d}) {name:18s} {desc}")
    print("\n" + "=" * 70)
    print("  Enter a preset code (e.g. PFM, PHY, MAF, ...)")
    print("  OR enter three numbers for custom R,G,B (e.g. 1,5,10)")
    print("=" * 70)

    while True:
        choice = input("\n  Your choice: ").strip().upper()
        if choice in PRESET_COMPOSITES:
            r, g, b, desc = PRESET_COMPOSITES[choice]
            print(f"\n  -> {choice}: R={r}, G={g}, B={b}")
            print(f"     {desc}")
            return r, g, b, f"{choice} ({desc})"
        if choice.startswith('P') and choice[1:].isdigit():
            idx = int(choice[1:]) - 1
            if 0 <= idx < len(preset_keys):
                key = preset_keys[idx]
                r, g, b, desc = PRESET_COMPOSITES[key]
                print(f"\n  -> {key}: R={r}, G={g}, B={b}")
                return r, g, b, f"{key} ({desc})"
        try:
            parts = choice.replace(',', ' ').split()
            if len(parts) == 3:
                ri, gi, bi = int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2]) - 1
                if all(0 <= x < len(param_names) for x in [ri, gi, bi]):
                    rn, gn, bn = param_names[ri], param_names[gi], param_names[bi]
                    print(f"\n  -> Custom: R={rn}, G={gn}, B={bn}")
                    return rn, gn, bn, f"Custom (R={rn}, G={gn}, B={bn})"
        except ValueError:
            pass

        print("  Let's pick one parameter per channel.")
        channels = {}
        for ch_name in ['RED', 'GREEN', 'BLUE']:
            while True:
                val = input(f"    {ch_name} channel (1-{len(param_names)}): ").strip()
                try:
                    idx = int(val) - 1
                    if 0 <= idx < len(param_names):
                        channels[ch_name] = param_names[idx]
                        print(f"      -> {param_names[idx]}")
                        break
                except ValueError:
                    val_up = val.upper()
                    if val_up in PARAM_REGISTRY:
                        channels[ch_name] = val_up
                        print(f"      -> {val_up}")
                        break
                print(f"    Invalid. Enter 1-{len(param_names)} or parameter name.")
        rn, gn, bn = channels['RED'], channels['GREEN'], channels['BLUE']
        return rn, gn, bn, f"Custom (R={rn}, G={gn}, B={bn})"


def compute_param_rgb_raw(data, wav, r_name, g_name, b_name):
    """Return the UNSTRETCHED stacked RGB array [H, W, 3] and the individual param maps."""
    param_cache = {}
    def get_param(name):
        if name not in param_cache:
            func, desc = PARAM_REGISTRY[name]
            print(f"  Computing {name}: {desc}...")
            param_cache[name] = func(data, wav)
        return param_cache[name]
    r_map = get_param(r_name)
    g_map = get_param(g_name)
    b_map = get_param(b_name)
    raw = np.dstack([r_map, g_map, b_map]).astype(np.float32)
    maps = {r_name: r_map, g_name: g_map, b_name: b_map}
    return raw, maps


# =============================================================================
# BROWSE PRODUCTS (RAW versions, stretching handled by DynamicImage)
# =============================================================================

def create_browse_ir_raw(data, wav, rw=2.529, gw=1.506, bw=1.080):
    r = data[np.abs(wav - rw).argmin()]
    g = data[np.abs(wav - gw).argmin()]
    b = data[np.abs(wav - bw).argmin()]
    return np.dstack([r, g, b]).astype(np.float32)

def create_browse_vnir_raw(data, wav, rw=0.600, gw=0.530, bw=0.440):
    if wav.min() > 0.50 or wav.max() < 0.65:
        return None
    r = data[np.abs(wav - rw).argmin()]
    g = data[np.abs(wav - gw).argmin()]
    b = data[np.abs(wav - bw).argmin()]
    return np.dstack([r, g, b]).astype(np.float32)

# =============================================================================
# RATIOING
# =============================================================================

def apply_ratio(samp, dust, wav):
    with np.errstate(divide='ignore', invalid='ignore'):
        rat = samp / dust
    rat[~np.isfinite(rat)] = np.nan
    rat[(wav >= 0.99) & (wav <= 1.02)] = np.nan
    rat[(wav >= 2.65) & (wav <= 2.75)] = np.nan
    return rat


# =============================================================================
# DYNAMIC STRETCH MANAGER
# =============================================================================

class DynamicStretchManager:
    """
    Holds the raw (unstretched) RGB arrays for each map axis and restretches
    them whenever the view limits change. The stretch uses only pixels
    currently visible in the axis.

    Works in DISPLAY coordinate space (after apply_display_transform), which is
    what the user interacts with in the plot.
    """

    def __init__(self, fig, panels, p_min=STRETCH_PMIN, p_max=STRETCH_PMAX):
        """
        panels : list of dicts, each with:
            'ax'       : matplotlib Axes
            'raw_rgb'  : [H, W, 3] float array in DATA coords (unstretched)
            'im'       : AxesImage returned by imshow
        """
        self.fig = fig
        self.panels = panels
        self.p_min = p_min
        self.p_max = p_max
        self._pending = False

        # Prepare display-space raw arrays (apply flip ONCE, store)
        for p in self.panels:
            p['raw_rgb_display'] = apply_display_transform(p['raw_rgb'])

        # Initial stretch (full scene)
        self.restretch_all(force_full=True)

        # Hook into view-change events on each axis
        for p in self.panels:
            ax = p['ax']
            ax.callbacks.connect('xlim_changed', self._on_lim_changed)
            ax.callbacks.connect('ylim_changed', self._on_lim_changed)

    def _on_lim_changed(self, ax):
        # Debounce: matplotlib emits xlim_changed AND ylim_changed for a
        # single zoom; coalesce into a single draw using idle callback.
        if self._pending:
            return
        self._pending = True
        try:
            self.fig.canvas.mpl_connect('draw_event', self._on_draw_once)
        except Exception:
            self._pending = False
            self.restretch_all()

    def _on_draw_once(self, event):
        # Only run once per draw cycle
        if not self._pending:
            return
        self._pending = False
        self.restretch_all()
        # Trigger a redraw so the new image data is shown
        self.fig.canvas.draw_idle()

    def restretch_all(self, force_full=False):
        for p in self.panels:
            self._restretch_panel(p, force_full=force_full)

    def _restretch_panel(self, p, force_full=False):
        img_disp = p['raw_rgb_display']     # [H, W, 3] in display coords
        ax = p['ax']
        H, W, _ = img_disp.shape

        if force_full:
            x0, x1 = 0, W
            y0, y1 = 0, H
        else:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            # ylim is often inverted (top is smaller y); normalize
            x0 = int(np.floor(min(xlim)))
            x1 = int(np.ceil(max(xlim)))
            y0 = int(np.floor(min(ylim)))
            y1 = int(np.ceil(max(ylim)))
            # Clamp to array bounds
            x0 = max(0, x0); x1 = min(W, x1)
            y0 = max(0, y0); y1 = min(H, y1)
            if x1 - x0 < 2 or y1 - y0 < 2:
                # Too small - fall back to full scene
                x0, x1, y0, y1 = 0, W, 0, H

        sub = img_disp[y0:y1, x0:x1, :]

        # Compute per-channel percentiles on the visible subregion only
        stretched = np.zeros_like(img_disp, dtype=np.float32)
        for ci in range(3):
            full_c = img_disp[:, :, ci]
            sub_c = sub[:, :, ci]
            valid = sub_c[np.isfinite(sub_c)]
            if len(valid) < 10:
                # Not enough pixels in view; fall back to global
                valid = full_c[np.isfinite(full_c)]
            if len(valid) > 0:
                vmin, vmax = np.percentile(valid, self.p_min), np.percentile(valid, self.p_max)
                if vmax > vmin:
                    stretched[:, :, ci] = np.clip((full_c - vmin) / (vmax - vmin), 0, 1)

        p['im'].set_data(stretched)


# =============================================================================
# APP
# =============================================================================

class InteractiveTool:
    def __init__(self):
        self.data, self.wav = None, None
        self.pending_sample = None
        self.samples = []
        self.waiting_for_dust = False

        self.lasso_arts = []

        self.bandfit_mode = False
        self.bandfit_clicks = []
        self.bandfit_click_arts = []
        self._persistent_shoulder_arts = []
        self.bandfit_cid = None

        self.stretch_mgr = None

        self.load()
        self.init_plots()

    def load(self):
        print("Loading TER3 joined cube...")
        self.data, self.wav = read_crism_ter3_cube(FILEPATH)
        print(f"  Wavelength range: {self.wav.min():.3f} - {self.wav.max():.3f} um")
        print(f"  Cube shape (bands, lines, samples): {self.data.shape}")

        r_name, g_name, b_name, composite_label = select_rgb_composite()
        self.composite_label = composite_label
        self.rgb_r_name, self.rgb_g_name, self.rgb_b_name = r_name, g_name, b_name

        print(f"\nComputing parameter composite: {composite_label}")
        self.rgb_param_raw, self.maps = compute_param_rgb_raw(
            self.data, self.wav, r_name, g_name, b_name)

        self.rgb_ir_raw = create_browse_ir_raw(self.data, self.wav)
        self.rgb_vnir_raw = create_browse_vnir_raw(self.data, self.wav)

    def init_plots(self):
        n_panels = 3 if self.rgb_vnir_raw is not None else 2
        self.fig_map, self.axs_map = plt.subplots(1, n_panels, figsize=(16, 6))
        self.axs_map = np.array(self.axs_map).flatten()

        param_title = f'Params ({self.rgb_r_name}, {self.rgb_g_name}, {self.rgb_b_name})'
        titles = [param_title, 'IR False Color']
        raws = [self.rgb_param_raw, self.rgb_ir_raw]
        if self.rgb_vnir_raw is not None:
            titles.append('VNIR True Color')
            raws.append(self.rgb_vnir_raw)

        panels = []
        for ax, raw, tit in zip(self.axs_map, raws, titles):
            # Initial placeholder: pre-stretched on full scene (will be overwritten
            # immediately by DynamicStretchManager in force_full mode, but this
            # sets up the AxesImage with the right extent and shape).
            disp_raw = apply_display_transform(raw)
            init_img = dynamic_stretch(disp_raw)
            im = ax.imshow(init_img, origin='upper', interpolation='nearest')
            ax.set_title(tit)
            panels.append({'ax': ax, 'raw_rgb': raw, 'im': im, 'title': tit})

        # Link all axes for joint pan/zoom
        for ax in self.axs_map[1:]:
            ax.sharex(self.axs_map[0])
            ax.sharey(self.axs_map[0])

        # Dynamic stretch manager
        self.stretch_mgr = DynamicStretchManager(
            self.fig_map, panels, p_min=STRETCH_PMIN, p_max=STRETCH_PMAX)

        # Spectra figure
        self.fig_spec, (self.ax_raw, self.ax_rat, self.ax_fit) = plt.subplots(1, 3, figsize=(20, 5))
        self.setup_spec_axes()

        self.lasso = LassoSelector(self.axs_map[0], self.on_lasso, button=[1])
        self.txt = self.fig_map.text(
            0.5, 0.02, "",
            ha='center', fontweight='bold')

        for f in [self.fig_map, self.fig_spec]:
            f.canvas.mpl_connect('key_press_event', self.on_key)

        self.update_state()
        plt.show()

    def setup_spec_axes(self):
        self.ax_raw.set_title("Raw I/F")
        self.ax_rat.set_title("Ratioed")
        for ax in [self.ax_raw, self.ax_rat]:
            ax.grid(True, alpha=0.3)
            ax.set_xlim(max(0.35, self.wav.min()), min(2.65, self.wav.max()))
            ax.set_xlabel("Wavelength (um)")
        self.ax_raw.set_ylabel("I/F")
        self.ax_rat.set_ylabel("Ratio")
        self.ax_rat.axhline(1, c='gray', ls='--', alpha=0.5)

        self.ax_fit.set_title("Band Fit (press 'b' to activate)")
        self.ax_fit.grid(True, alpha=0.3)
        self.ax_fit.set_xlabel("Wavelength (um)")
        self.ax_fit.set_ylabel("Continuum-removed")
        self.ax_fit.text(0.5, 0.5, "Press 'b' then click\ntwo shoulders on\nratioed plot",
                         ha='center', va='center', transform=self.ax_fit.transAxes,
                         fontsize=11, color='gray', alpha=0.6)

    def update_state(self):
        if not self.waiting_for_dust:
            msg = "STEP 1: DRAW LASSO around SAMPLE pixels   |   keys: b=bandfit, d=del, r=reset, f=restretch, q=quit"
            self.txt.set_text(msg)
            self.txt.set_bbox(dict(facecolor='lightblue', alpha=1))
        else:
            msg = "STEP 2: DRAW LASSO around DUST pixels   |   keys: b=bandfit, d=del, r=reset, f=restretch, q=quit"
            self.txt.set_text(msg)
            self.txt.set_bbox(dict(facecolor='orange', alpha=1))
        self.fig_map.canvas.draw_idle()

    def get_pixels_in_lasso(self, verts):
        path = Path(verts)
        h, w = self.data.shape[1], self.data.shape[2]
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        points = np.column_stack([xs.ravel(), ys.ravel()])
        mask = path.contains_points(points)
        inside = points[mask]
        return [(int(p[0]), int(p[1])) for p in inside]

    def get_spectrum_from_pixels(self, pixels):
        h, w = self.data.shape[1], self.data.shape[2]
        specs = []
        for px, py in pixels:
            xd, yd = display_to_data_coords(px, py, (h, w))
            xd, yd = int(round(xd)), int(round(yd))
            if 0 <= xd < w and 0 <= yd < h:
                specs.append(self.data[:, yd, xd])
        return np.nanmean(specs, axis=0) if specs else None

    def draw_roi_pixels(self, pixels, color, is_dust):
        arts = []
        xs = [p[0] for p in pixels]
        ys = [p[1] for p in pixels]
        for ax in self.axs_map:
            if ax.images:
                m = ax.plot(xs, ys, 's', color=color, markersize=4, alpha=0.6,
                            markeredgecolor='white', markeredgewidth=0.3)[0]
                arts.append(m)
        return arts

    def on_lasso(self, verts):
        pixels = self.get_pixels_in_lasso(verts)
        if not pixels:
            print("No pixels in lasso.")
            return

        spec = self.get_spectrum_from_pixels(pixels)
        if spec is None:
            print("No valid data in selection.")
            return

        if not self.waiting_for_dust:
            lbl = f"S{len(self.samples)+1}"
            col = plt.cm.tab10(len(self.samples) % 10)
            roi_arts = self.draw_roi_pixels(pixels, col, False)
            self.pending_sample = {
                'spec': spec,
                'pixels': pixels,
                'arts': roi_arts
            }
            self.waiting_for_dust = True
            print(f"Sample {lbl} selected ({len(pixels)} px). Now draw Dust lasso.")
        else:
            dust_arts = self.draw_roi_pixels(pixels, 'orange', True)
            self.process_pair(spec, dust_arts)

        self.update_state()
        self.fig_map.canvas.draw_idle()

    def process_pair(self, dust_spec, dust_arts):
        samp = self.pending_sample
        sample_spec = samp['spec']
        lbl = f"S{len(self.samples)+1}"
        col = plt.cm.tab10(len(self.samples) % 10)

        ratio = apply_ratio(sample_spec, dust_spec, self.wav)

        l1, = self.ax_raw.plot(self.wav, sample_spec, c=col, label=lbl)
        l2, = self.ax_rat.plot(self.wav, ratio, c=col, label=lbl)
        arts = [l1, l2]

        arts.extend(samp['arts'])
        arts.extend(dust_arts)
        self.ax_raw.legend()
        self.ax_rat.legend()

        self._autoscale_ratio_axis()

        self.fig_spec.canvas.draw_idle()
        self.fig_map.canvas.draw_idle()

        mask = np.isfinite(ratio)
        wav_valid = self.wav[mask]
        rat_valid = ratio[mask]
        sort_idx = np.argsort(wav_valid)

        self.samples.append({
            'lbl': lbl,
            'raw': sample_spec,
            'dust': dust_spec,
            'ratio': ratio,
            'arts': arts,
            'col': col,
            'wav_stitched': wav_valid[sort_idx],
            'rat_stitched': rat_valid[sort_idx],
        })

        print(f"Processed {lbl} (Pair complete)")

        self.pending_sample = None
        self.waiting_for_dust = False

    def _autoscale_ratio_axis(self):
        all_y = []
        for line in self.ax_rat.get_lines():
            yd = np.array(line.get_ydata(), dtype=float)
            valid = yd[np.isfinite(yd)]
            if len(valid) > 0:
                all_y.append(valid)
        if all_y:
            all_y = np.concatenate(all_y)
            ymin, ymax = np.nanpercentile(all_y, 1), np.nanpercentile(all_y, 99)
            pad = (ymax - ymin) * 0.1
            if pad == 0: pad = 0.05
            self.ax_rat.set_ylim(ymin - pad, ymax + pad)

    def save(self):
        if not self.samples: return
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except:
            script_dir = os.getcwd()
        fpath = os.path.join(script_dir, "ter3_ratioed_spectra.csv")

        valid_mask = np.any(np.isfinite(self.data), axis=(1, 2))
        wav_out = self.wav[valid_mask]

        start_id = 0
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath)
                s_cols = [c for c in df.columns if c.startswith('S') and c[1:].isdigit()]
                if s_cols: start_id = max(int(c[1:]) for c in s_cols)
            except:
                pass

        new_data = {}
        if not os.path.exists(fpath):
            new_data['wavelength_um'] = wav_out

        for i, s in enumerate(self.samples):
            key = f"S{start_id + i + 1}"
            new_data[key] = s['ratio'][valid_mask]

        df_new = pd.DataFrame(new_data)

        if os.path.exists(fpath):
            df_old = pd.read_csv(fpath)
            if len(df_old) == len(df_new):
                pd.concat([df_old, df_new], axis=1).to_csv(fpath, index=False)
            else:
                print("Wavelength mismatch. Saving new file.")
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                df_new.insert(0, 'wavelength_um', wav_out)
                df_new.to_csv(fpath.replace('.csv', f'_{timestamp}.csv'), index=False)
        else:
            df_new.to_csv(fpath, index=False)
        print(f"Saved {len(self.samples)} spectra to {fpath}")

    def delete_last(self):
        if not self.samples: return
        s = self.samples.pop()
        for a in s['arts']:
            try: a.remove()
            except: pass
        self._autoscale_ratio_axis()
        self.fig_map.canvas.draw_idle()
        self.fig_spec.canvas.draw_idle()
        print(f"Deleted {s['lbl']}")

    def reset(self):
        while self.samples: self.delete_last()
        if self.pending_sample:
            for a in self.pending_sample['arts']:
                try: a.remove()
                except: pass
        self.pending_sample = None
        self.waiting_for_dust = False
        self.update_state()
        print("Reset.")

    def on_key(self, event):
        if event.key == 'q':
            plt.close('all')
        elif event.key == 'd':
            self.delete_last()
        elif event.key == 'r':
            self.reset()
        elif event.key == 'b':
            self.toggle_bandfit_mode()
        elif event.key == 'f':
            # Manual forced restretch of the current view
            if self.stretch_mgr is not None:
                self.stretch_mgr.restretch_all()
                self.fig_map.canvas.draw_idle()
                print("Forced restretch on current view.")

    # =========================================================================
    # BAND FITTING
    # =========================================================================

    def toggle_bandfit_mode(self):
        if not self.samples:
            print("No spectra to fit. Extract spectra first.")
            return

        if self.bandfit_mode:
            self.bandfit_mode = False
            if self.bandfit_cid is not None:
                self.fig_spec.canvas.mpl_disconnect(self.bandfit_cid)
                self.bandfit_cid = None
            self.bandfit_clicks = []
            self._clear_bandfit_click_arts()
            self._clear_persistent_shoulders()
            self.ax_rat.set_title("Ratioed")
            self.fig_spec.canvas.draw_idle()
            print("Band fit mode OFF.")
        else:
            self.bandfit_mode = True
            self.bandfit_clicks = []
            self._clear_bandfit_click_arts()
            self.bandfit_cid = self.fig_spec.canvas.mpl_connect(
                'button_press_event', self._on_bandfit_click)
            self.ax_rat.set_title("Ratioed - CLICK TWO SHOULDERS")
            self.fig_spec.canvas.draw_idle()
            print("Band fit mode ON. Click two shoulder points on the ratioed plot.")

    def _clear_bandfit_click_arts(self):
        for a in self.bandfit_click_arts:
            try: a.remove()
            except: pass
        self.bandfit_click_arts = []

    def _clear_persistent_shoulders(self):
        for a in self._persistent_shoulder_arts:
            try: a.remove()
            except: pass
        self._persistent_shoulder_arts = []

    def _on_bandfit_click(self, event):
        if event.inaxes != self.ax_rat:
            return
        if event.button != 1:
            return

        wav_click = event.xdata
        self.bandfit_clicks.append(wav_click)

        vline = self.ax_rat.axvline(wav_click, color='red', ls='--', lw=1.5, alpha=0.8)
        self.bandfit_click_arts.append(vline)
        self.fig_spec.canvas.draw_idle()

        print(f"  Shoulder {len(self.bandfit_clicks)}: {wav_click:.4f} um")

        if len(self.bandfit_clicks) >= 2:
            w1, w2 = sorted(self.bandfit_clicks[:2])
            print(f"  Fitting band between {w1:.4f} - {w2:.4f} um ...")
            self._perform_bandfit(w1, w2)

            self._clear_bandfit_click_arts()
            self._clear_persistent_shoulders()
            vl1 = self.ax_rat.axvline(w1, color='red', ls='--', lw=1.5, alpha=0.8)
            vl2 = self.ax_rat.axvline(w2, color='red', ls='--', lw=1.5, alpha=0.8)
            self._persistent_shoulder_arts = [vl1, vl2]

            self.bandfit_clicks = []
            self.ax_rat.set_title("Ratioed - CLICK TWO SHOULDERS (or 'b' to exit)")
            self.fig_spec.canvas.draw_idle()

    def _perform_bandfit(self, w_left, w_right):
        self.ax_fit.cla()
        self.ax_fit.set_title(f"Band Fit: {w_left:.3f} - {w_right:.3f} um")
        self.ax_fit.set_xlabel("Wavelength (um)")
        self.ax_fit.set_ylabel("Continuum-removed ratio")
        self.ax_fit.grid(True, alpha=0.3)
        self.ax_fit.axhline(1, c='gray', ls='--', alpha=0.3)

        results = []
        y_offset = 0.0

        for s in self.samples:
            wav = s['wav_stitched']
            rat = s['rat_stitched']
            lbl = s['lbl']
            col = s['col']

            if wav is None or rat is None:
                continue

            mask = (wav >= w_left) & (wav <= w_right) & np.isfinite(rat)
            if np.sum(mask) < 6:
                print(f"  {lbl}: Not enough valid points in range. Skipping.")
                continue

            w_sub = wav[mask]
            r_sub = rat[mask]

            n_edge = min(3, len(w_sub) // 4)
            if n_edge < 1: n_edge = 1
            val_left = np.nanmedian(r_sub[:n_edge])
            val_right = np.nanmedian(r_sub[-n_edge:])
            w_left_actual = np.nanmedian(w_sub[:n_edge])
            w_right_actual = np.nanmedian(w_sub[-n_edge:])

            if w_right_actual == w_left_actual:
                continue

            slope = (val_right - val_left) / (w_right_actual - w_left_actual)
            continuum = val_left + slope * (w_sub - w_left_actual)

            with np.errstate(divide='ignore', invalid='ignore'):
                cr = np.where(continuum != 0, r_sub / continuum, np.nan)

            valid = np.isfinite(cr)
            if np.sum(valid) < 6:
                print(f"  {lbl}: Continuum removal failed. Skipping.")
                continue

            w_valid = w_sub[valid]
            cr_valid = cr[valid]

            self.ax_fit.plot(w_valid, cr_valid, 'o', color=col, markersize=3, alpha=0.5)

            try:
                poly_order = min(4, len(w_valid) - 1)
                coeffs = np.polyfit(w_valid, cr_valid, poly_order)
                poly = np.poly1d(coeffs)

                w_fine = np.linspace(w_valid.min(), w_valid.max(), 500)
                cr_fine = poly(w_fine)

                self.ax_fit.plot(w_fine, cr_fine, '-', color=col, lw=2, label=lbl)

                dpoly = poly.deriv()
                roots = np.roots(dpoly)
                real_roots = roots[np.isreal(roots)].real
                in_range = real_roots[(real_roots >= w_valid.min()) &
                                      (real_roots <= w_valid.max())]

                if len(in_range) > 0:
                    root_vals = poly(in_range)
                    band_center = in_range[np.argmin(root_vals)]
                    band_cr_val = poly(band_center)
                else:
                    min_idx = np.argmin(cr_fine)
                    band_center = w_fine[min_idx]
                    band_cr_val = cr_fine[min_idx]

                band_depth = 1.0 - band_cr_val

                half_level = 1.0 - band_depth / 2.0
                fwhm = self._compute_fwhm(w_fine, cr_fine, half_level, band_center)

                self.ax_fit.axvline(band_center, color=col, ls=':', lw=1, alpha=0.7)
                self.ax_fit.plot(band_center, band_cr_val, 'v', color=col,
                               markersize=10, markeredgecolor='black', markeredgewidth=0.5)

                if fwhm is not None:
                    self.ax_fit.hlines(half_level, band_center - fwhm/2,
                                      band_center + fwhm/2, color=col, ls='-', lw=1.5, alpha=0.5)

                fwhm_str = f"{fwhm*1000:.1f} nm" if fwhm is not None else "N/A"
                txt = f"{lbl}: center={band_center*1000:.1f} nm, depth={band_depth:.4f}, FWHM={fwhm_str}"
                self.ax_fit.text(0.02, 0.97 - y_offset, txt, transform=self.ax_fit.transAxes,
                                fontsize=8, color=col, fontweight='bold', va='top',
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                y_offset += 0.06

                print(f"  {txt}")
                results.append({'lbl': lbl, 'band_center': band_center,
                               'band_depth': band_depth, 'fwhm': fwhm})

            except Exception as e:
                print(f"  {lbl}: Fit failed - {e}")
                continue

        pad_w = (w_right - w_left) * 0.05
        self.ax_fit.set_xlim(w_left - pad_w, w_right + pad_w)

        self.ax_fit.legend(fontsize=8)
        self.fig_spec.tight_layout()
        self.fig_spec.canvas.draw_idle()

        if len(results) > 1:
            print("\n  --- Band Center Comparison ---")
            for r in results:
                fwhm_str = f"{r['fwhm']*1000:.1f} nm" if r['fwhm'] is not None else "N/A"
                print(f"    {r['lbl']}: {r['band_center']*1000:.1f} nm  "
                      f"(depth={r['band_depth']:.4f}, FWHM={fwhm_str})")
            centers = [r['band_center'] for r in results]
            spread = (max(centers) - min(centers)) * 1000
            print(f"    Spread: {spread:.1f} nm")

    def _compute_fwhm(self, w, cr, half_level, center):
        try:
            left_mask = w < center
            right_mask = w > center

            w_left = None
            if np.any(left_mask):
                wl = w[left_mask]
                cl = cr[left_mask]
                crossings = np.where(np.diff(np.sign(cl - half_level)))[0]
                if len(crossings) > 0:
                    idx = crossings[-1]
                    w_left = wl[idx] + (half_level - cl[idx]) / (cl[idx+1] - cl[idx]) * (wl[idx+1] - wl[idx])

            w_right = None
            if np.any(right_mask):
                wr = w[right_mask]
                crr = cr[right_mask]
                crossings = np.where(np.diff(np.sign(crr - half_level)))[0]
                if len(crossings) > 0:
                    idx = crossings[0]
                    w_right = wr[idx] + (half_level - crr[idx]) / (crr[idx+1] - crr[idx]) * (wr[idx+1] - wr[idx])

            if w_left is not None and w_right is not None:
                return w_right - w_left
        except:
            pass
        return None


if __name__ == "__main__":
    tool = InteractiveTool()
    if tool.samples:
        if input("Save? (y/n) ").lower() == 'y': tool.save()