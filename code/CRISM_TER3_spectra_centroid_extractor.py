"""
CRISM TER3 Spectra + Brown Parameter Analyzer (v1.0)
-----------------------------------------------------
Combined tool: extract ratioed spectra from TER3 cubes AND sample
Brown et al. (2020) asymmetric Gaussian fit parameters per ROI.

WORKFLOW:
  1. Set WORKING_DIR to the TER3 cube directory.
  2. Script auto-discovers the *if*ter3*.img cube and matching *_v5.npy
     Brown parameter maps in that directory.
  3. Three map panels: RGB composite | OLINDEX3 | Centroid map
  4. Lasso SAMPLE ROI → lasso DUST ROI → ratioed spectrum plotted.
  5. Brown parameters (centroid, asymmetry, amplitude, HWHM) extracted
     for each SAMPLE ROI from pre-computed .npy maps.
  6. On quit (or 's'): saves CSV with ratioed spectra, Brown param
     stats, and ROI pixel coordinates for reproducibility.

OUTPUT (per cube, saved in WORKING_DIR):
  - <cube_id>_analysis.csv        Ratioed spectra (wavelength × ROIs)
  - <cube_id>_brown_params.csv    Brown parameter stats per ROI
  - <cube_id>_roi_pixels.csv      Pixel coordinates per ROI (for re-plotting)

Keys:
  d — delete last ROI
  r — reset all ROIs
  s — save and continue
  f — force restretch
  q — quit (prompts to save)

Author: AV Valantinas
"""

import numpy as np
import matplotlib.pyplot as plt
from config import CUBES_DIR, STYLE_PATH
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import warnings
import os
import glob
import pandas as pd

warnings.filterwarnings('ignore')

# =============================================================================
# USER CONFIGURATION
# =============================================================================

WORKING_DIR = str(CUBES_DIR / "FRT0000A4FC_07_IF166J_TER3")

# Brown parameter map version (must match batch output)
BROWN_VERSION = "v5"

# Display flips for TER3 (unprojected, sensor space)
FLIP_VERTICAL = True
FLIP_HORIZONTAL = True

# Stretch percentiles
STRETCH_PMIN = 2
STRETCH_PMAX = 98

# Default RGB composite preset
DEFAULT_COMPOSITE = 'ask'  # Options: PFM, PHY, MAF, HYD, CAR, CR2, or 'ask'

style_path = str(STYLE_PATH)
if os.path.exists(style_path):
    plt.style.use(style_path)

ROI_COLORS = plt.cm.tab10.colors


# =============================================================================
# DATA LOADING
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


def read_ter3_cube(img_file):
    """Read a CRISM TER3 cube. Returns (data[bands, lines, samples], wav_um)."""
    hdr = img_file.replace('.img', '.hdr').replace('.IMG', '.hdr')
    if not os.path.exists(hdr):
        hdr = img_file + '.hdr'
    if not os.path.exists(hdr):
        base = os.path.splitext(img_file)[0]
        for ext in ['.hdr', '.HDR']:
            if os.path.exists(base + ext):
                hdr = base + ext
                break
        else:
            raise FileNotFoundError(f"No header for {img_file}")

    meta = parse_envi_header(hdr)
    s = int(meta.get('samples', 0))
    l = int(meta.get('lines', 0))
    b = int(meta.get('bands', 0))
    dt = int(meta.get('data type', 4))
    order = int(meta.get('byte order', 0))
    inter = meta.get('interleave', 'bsq').lower()

    print(f"  Reading {os.path.basename(img_file)} "
          f"({s}×{l}×{b}, {inter})")

    if meta.get('wavelength'):
        wav = np.array([float(w) for w in meta['wavelength']])
        if wav[0] > 100:
            wav = wav / 1000.0
    else:
        raise ValueError("No wavelengths in header")

    bbl = (np.array([int(float(x)) for x in meta['bbl']])
           if meta.get('bbl') else None)
    ign = float(meta.get('data ignore value', 65535))

    dtype = {1: np.uint8, 2: np.int16, 4: np.float32,
             12: np.uint16}.get(dt, np.float32)
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

    print(f"  Shape: {data.shape}, "
          f"wav: {wav[wav < 5].min():.3f}–{wav[wav < 5].max():.3f} µm")
    return data, wav


def find_cube_and_maps(working_dir):
    """Auto-discover TER3 I/F cube and Brown .npy maps in working_dir."""
    # Find I/F cube
    patterns = ['*_if*ter3*.img', '*_IF*TER3*.img', '*_if*TER3*.img']
    img_file = None
    for pat in patterns:
        matches = glob.glob(os.path.join(working_dir, pat))
        if matches:
            img_file = matches[0]
            break
    if img_file is None:
        raise FileNotFoundError(
            f"No TER3 I/F cube (*_if*ter3*.img) in {working_dir}")

    base = os.path.splitext(os.path.basename(img_file))[0]

    # Extract cube ID (e.g. frt00003e12)
    cube_id = base.split('_')[0]

    # Find Brown parameter maps
    brown_maps = {}
    for param in ['centroid', 'asymmetry', 'amplitude', 'hwhm', 'mask']:
        npy_path = os.path.join(working_dir,
                                f'{base}_{param}_map_{BROWN_VERSION}.npy')
        if os.path.exists(npy_path):
            brown_maps[param] = np.load(npy_path)
            print(f"  Loaded Brown {param}: {brown_maps[param].shape}")
        else:
            print(f"  WARNING: Brown {param} map not found: "
                  f"{os.path.basename(npy_path)}")

    return img_file, cube_id, base, brown_maps


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def apply_display_transform(image):
    result = image.copy()
    if FLIP_VERTICAL:
        result = np.flipud(result)
    if FLIP_HORIZONTAL:
        result = np.fliplr(result)
    return result


def display_to_data_coords(x, y, shape):
    """Convert display pixel coords to data (line, sample) coords."""
    lines, samples = shape[0], shape[1]
    x_data, y_data = x, y
    if FLIP_HORIZONTAL:
        x_data = samples - 1 - x
    if FLIP_VERTICAL:
        y_data = lines - 1 - y
    return int(round(x_data)), int(round(y_data))


def dynamic_stretch(data, p_min=STRETCH_PMIN, p_max=STRETCH_PMAX):
    res = np.zeros_like(data, dtype=np.float32)
    for i in range(data.shape[2]):
        c = data[:, :, i]
        valid = c[np.isfinite(c)]
        if len(valid) > 0:
            vmin = np.percentile(valid, p_min)
            vmax = np.percentile(valid, p_max)
            if vmax > vmin:
                res[:, :, i] = np.clip((c - vmin) / (vmax - vmin), 0, 1)
    return res


def stretch_single(data, p_min=STRETCH_PMIN, p_max=STRETCH_PMAX):
    """Stretch a single 2D map for display."""
    valid = data[np.isfinite(data)]
    if len(valid) == 0:
        return np.zeros_like(data)
    vmin = np.percentile(valid, p_min)
    vmax = np.percentile(valid, p_max)
    if vmax <= vmin:
        return np.zeros_like(data)
    return np.clip((data - vmin) / (vmax - vmin), 0, 1)


# =============================================================================
# SPECTRAL PARAMETERS (only what we need for the panels)
# =============================================================================

def get_kernel(data, wav, target, width):
    idx = np.abs(wav - target).argmin()
    if width <= 1:
        return data[idx, :, :]
    hw = width // 2
    return np.nanmedian(
        data[max(0, idx - hw):min(len(wav), idx + hw + 1), :, :], axis=0)


def _rb(R, RC):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(RC != 0, (RC - R) / RC, 0.0)
    return np.nan_to_num(result, nan=0.0)


def _ratio(num, den):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(den != 0, 1.0 - (num / den), 0.0)
    return np.nan_to_num(result, nan=0.0)


def calc_olindex3(data, wav):
    R1210 = get_kernel(data, wav, 1.210, 7)
    R1250 = get_kernel(data, wav, 1.250, 7)
    R1263 = get_kernel(data, wav, 1.263, 7)
    R1276 = get_kernel(data, wav, 1.276, 7)
    R1330 = get_kernel(data, wav, 1.330, 7)
    R1750 = get_kernel(data, wav, 1.750, 7)
    R1862 = get_kernel(data, wav, 1.862, 7)
    slope = np.where((R1862 != R1750),
                     (R1862 - R1750) / (1.862 - 1.750), 0.0)
    def rc(w): return R1750 + slope * (w - 1.750)
    return (0.1 * _rb(R1210, rc(1.210)) + 0.1 * _rb(R1250, rc(1.250)) +
            0.2 * _rb(R1263, rc(1.263)) + 0.2 * _rb(R1276, rc(1.276)) +
            0.4 * _rb(R1330, rc(1.330)))


def calc_lcpindex2(data, wav):
    R1560 = get_kernel(data, wav, 1.560, 7)
    R1690 = get_kernel(data, wav, 1.690, 7)
    R1750 = get_kernel(data, wav, 1.750, 7)
    R1810 = get_kernel(data, wav, 1.810, 7)
    R1870 = get_kernel(data, wav, 1.870, 7)
    R2450 = get_kernel(data, wav, 2.450, 7)
    slope = np.where((R2450 != R1560),
                     (R2450 - R1560) / (2.450 - 1.560), 0.0)
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
    slope = np.where((R2530 != R1690),
                     (R2530 - R1690) / (2.530 - 1.690), 0.0)
    def rc(w): return R1690 + slope * (w - 1.690)
    return (0.10 * _rb(R2120, rc(2.120)) + 0.10 * _rb(R2140, rc(2.140)) +
            0.15 * _rb(R2230, rc(2.230)) + 0.30 * _rb(R2250, rc(2.250)) +
            0.20 * _rb(R2430, rc(2.430)) + 0.15 * _rb(R2460, rc(2.460)))


def calc_d2300(data, wav):
    R1815 = get_kernel(data, wav, 1.815, 5)
    R2120 = get_kernel(data, wav, 2.120, 5)
    R2170 = get_kernel(data, wav, 2.170, 5)
    R2210 = get_kernel(data, wav, 2.210, 5)
    R2290 = get_kernel(data, wav, 2.290, 3)
    R2320 = get_kernel(data, wav, 2.320, 3)
    R2330 = get_kernel(data, wav, 2.330, 3)
    R2530 = get_kernel(data, wav, 2.530, 5)
    slope = np.where((R2530 != R1815),
                     (R2530 - R1815) / (2.530 - 1.815), 0.0)
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


def calc_bd2500_2(data, wav):
    R2364 = get_kernel(data, wav, 2.364, 5)
    R2480 = get_kernel(data, wav, 2.480, 5)
    R2570 = get_kernel(data, wav, 2.570, 5)
    b_coeff = (2.480 - 2.364) / (2.570 - 2.364)
    a_coeff = 1.0 - b_coeff
    return _ratio(R2480, a_coeff * R2364 + b_coeff * R2570)


def calc_bd2355(data, wav):
    R2300 = get_kernel(data, wav, 2.300, 5)
    R2355 = get_kernel(data, wav, 2.355, 5)
    R2450 = get_kernel(data, wav, 2.450, 5)
    b_coeff = (2.355 - 2.300) / (2.450 - 2.300)
    a_coeff = 1.0 - b_coeff
    return _ratio(R2355, a_coeff * R2300 + b_coeff * R2450)


def calc_bd2290(data, wav):
    R2250 = get_kernel(data, wav, 2.250, 5)
    R2290 = get_kernel(data, wav, 2.290, 5)
    R2350 = get_kernel(data, wav, 2.350, 5)
    b_coeff = (2.290 - 2.250) / (2.350 - 2.250)
    a_coeff = 1.0 - b_coeff
    return _ratio(R2290, a_coeff * R2250 + b_coeff * R2350)


def calc_d2200(data, wav):
    R1815 = get_kernel(data, wav, 1.815, 7)
    R2165 = get_kernel(data, wav, 2.165, 5)
    R2210 = get_kernel(data, wav, 2.210, 7)
    R2230 = get_kernel(data, wav, 2.230, 7)
    R2430 = get_kernel(data, wav, 2.430, 7)
    slope = np.where((R2430 != R1815),
                     (R2430 - R1815) / (2.430 - 1.815), 0.0)
    def rc(w): return R1815 + slope * (w - 1.815)
    with np.errstate(divide='ignore', invalid='ignore'):
        r2165 = np.where(rc(2.165) != 0, R2165 / rc(2.165), 0.0)
        r2210 = np.where(rc(2.210) != 0, R2210 / rc(2.210), 0.0)
        r2230 = np.where(rc(2.230) != 0, R2230 / rc(2.230), 0.0)
        avg = (r2210 + r2230) / 2.0
        result = np.where(r2165 != 0, 1.0 - avg / r2165, 0.0)
    return np.nan_to_num(result, nan=0.0)


def calc_bd1900r2(data, wav):
    num_w = [1.908, 1.914, 1.921, 1.928, 1.934, 1.941]
    den_w = [1.862, 1.869, 1.875, 2.112, 2.120, 2.126]
    R1850 = get_kernel(data, wav, 1.850, 5)
    R2060 = get_kernel(data, wav, 2.060, 5)
    slope = np.where((R2060 != R1850),
                     (R2060 - R1850) / (2.060 - 1.850), 0.0)
    def rc(w): return R1850 + slope * (w - 1.850)
    with np.errstate(divide='ignore', invalid='ignore'):
        num_sum = sum(np.where(rc(w) != 0,
                               get_kernel(data, wav, w, 1) / rc(w), 0.0)
                      for w in num_w)
        den_sum = sum(np.where(rc(w) != 0,
                               get_kernel(data, wav, w, 1) / rc(w), 0.0)
                      for w in den_w)
        result = np.where(den_sum != 0, 1.0 - num_sum / den_sum, 0.0)
    return np.nan_to_num(result, nan=0.0)


# Composite presets
PRESET_COMPOSITES = {
    'PFM': ('BD2355',  'D2300',     'BD2290',    'Phyllosilicates Fe/Mg'),
    'PHY': ('D2300',   'D2200',     'BD1900r2',  'Phyllosilicates'),
    'MAF': ('OLINDEX3','LCPINDEX2', 'HCPINDEX2', 'Mafic mineralogy'),
    'CAR': ('D2300',   'BD2500_2',  'BD1900r2',  'Carbonates'),
}

PARAM_FUNCS = {
    'OLINDEX3':  calc_olindex3,
    'LCPINDEX2': calc_lcpindex2,
    'HCPINDEX2': calc_hcpindex2,
    'D2300':     calc_d2300,
    'D2200':     calc_d2200,
    'BD2500_2':  calc_bd2500_2,
    'BD2355':    calc_bd2355,
    'BD2290':    calc_bd2290,
    'BD1900r2':  calc_bd1900r2,
}


def compute_rgb(data, wav, preset):
    """Compute RGB composite from preset name."""
    r_name, g_name, b_name, desc = PRESET_COMPOSITES[preset]
    print(f"  Computing {preset}: R={r_name}, G={g_name}, B={b_name}")
    r = PARAM_FUNCS[r_name](data, wav)
    g = PARAM_FUNCS[g_name](data, wav)
    b = PARAM_FUNCS[b_name](data, wav)
    return np.dstack([r, g, b]).astype(np.float32), desc


# =============================================================================
# RATIOING
# =============================================================================

def apply_ratio(samp, dust, wav):
    with np.errstate(divide='ignore', invalid='ignore'):
        rat = samp / dust
    rat[~np.isfinite(rat)] = np.nan
    # Mask detector boundary and CO2 regions
    rat[(wav >= 0.99) & (wav <= 1.02)] = np.nan
    rat[(wav >= 2.65) & (wav <= 2.75)] = np.nan
    return rat


# =============================================================================
# INTERACTIVE TOOL
# =============================================================================

class TER3Analyzer:
    def __init__(self):
        # Discover files
        print(f"\nWorking directory: {WORKING_DIR}")
        self.img_file, self.cube_id, self.base_name, self.brown = \
            find_cube_and_maps(WORKING_DIR)

        # Load cube
        print(f"\nLoading cube...")
        self.data, self.wav = read_ter3_cube(self.img_file)
        self.n_bands, self.n_lines, self.n_samples = self.data.shape

        # Compute spectral parameter maps
        print(f"\nComputing spectral parameter maps...")

        if DEFAULT_COMPOSITE == 'ask':
            print("\nAvailable presets:", ', '.join(PRESET_COMPOSITES.keys()))
            choice = input("  Composite choice: ").strip().upper()
            if choice not in PRESET_COMPOSITES:
                choice = 'MAF'
        else:
            choice = DEFAULT_COMPOSITE

        self.rgb_raw, self.rgb_desc = compute_rgb(
            self.data, self.wav, choice)
        self.rgb_label = f"{choice} ({self.rgb_desc})"

        print(f"  Computing IR false-color browse...")
        ir_r = self.data[np.abs(self.wav - 2.529).argmin()]
        ir_g = self.data[np.abs(self.wav - 1.506).argmin()]
        ir_b = self.data[np.abs(self.wav - 1.080).argmin()]
        self.rgb_ir_raw = np.dstack([ir_r, ir_g, ir_b]).astype(np.float32)

        # State
        self.mode = 'sample'
        self.pending_sample = None
        self.rois = []       # completed ROIs
        self.roi_counter = 0

        # Build UI
        self._build_ui()

    def _build_ui(self):
        # --- Map figure: 3 panels ---
        self.fig_map, self.axs = plt.subplots(1, 3, figsize=(18, 6))

        # Panel 1: RGB composite
        rgb_disp = apply_display_transform(self.rgb_raw)
        rgb_stretched = dynamic_stretch(rgb_disp)
        self.axs[0].imshow(rgb_stretched, origin='upper',
                           interpolation='nearest')
        self.axs[0].set_title(self.rgb_label, fontsize=9)

        # Panel 2: IR false color
        ir_disp = apply_display_transform(self.rgb_ir_raw)
        ir_stretched = dynamic_stretch(ir_disp)
        self.axs[1].imshow(ir_stretched, origin='upper',
                           interpolation='nearest')
        self.axs[1].set_title('IR False Color (R=2.53 G=1.51 B=1.08 µm)',
                              fontsize=9)

        # Panel 3: Centroid map (if available)
        if 'centroid' in self.brown:
            cent_disp = apply_display_transform(self.brown['centroid'])
            cent_show = cent_disp
            self.axs[2].imshow(cent_show, cmap='RdYlBu_r', origin='upper',
                               interpolation='nearest',
                               vmin=1.15, vmax=1.45)
            self.axs[2].set_title('Brown centroid (µm)', fontsize=9)
        else:
            self.axs[2].text(0.5, 0.5, 'No centroid map\nRun brown2020_batch_ter3.py first',
                             ha='center', va='center',
                             transform=self.axs[2].transAxes, fontsize=11)
            self.axs[2].set_title('Brown centroid (not found)', fontsize=9)

        # Link axes
        for ax in self.axs[1:]:
            ax.sharex(self.axs[0])
            ax.sharey(self.axs[0])

        self.fig_map.suptitle(
            f"{self.cube_id} — TER3 Analyzer\n[Mode: SAMPLE]",
            fontsize=10, y=0.98)

        # Lasso on first panel
        self.lasso = LassoSelector(self.axs[0], self._on_lasso, button=[1])

        # --- Spectra figure: raw + ratioed ---
        self.fig_spec, (self.ax_raw, self.ax_rat) = plt.subplots(
            1, 2, figsize=(14, 5))
        self._setup_spec_axes()

        # Key bindings
        for fig in [self.fig_map, self.fig_spec]:
            fig.canvas.mpl_connect('key_press_event', self._on_key)

        self.fig_map.tight_layout()
        self.fig_spec.tight_layout()

        print("\n" + "=" * 60)
        print("  WORKFLOW:")
        print("  1. Lasso a SAMPLE region on the RGB panel")
        print("  2. Lasso a DUST region")
        print("  3. Ratioed spectrum + Brown params appear")
        print("  4. Repeat for more ROIs")
        print("  Keys: d=delete last, r=reset, s=save, q=quit")
        print("=" * 60)

    def _setup_spec_axes(self):
        """Configure spectra axes — matches TER3 v4.2 style."""
        self.ax_raw.set_title("Raw I/F")
        self.ax_rat.set_title("Ratioed")
        wmin = max(0.35, self.wav[self.wav < 5].min())
        wmax = min(2.65, self.wav[self.wav < 5].max())
        for ax in [self.ax_raw, self.ax_rat]:
            ax.grid(True, alpha=0.3)
            ax.set_xlim(wmin, wmax)
            ax.set_xlabel("Wavelength (µm)")
        self.ax_raw.set_ylabel("I/F")
        self.ax_rat.set_ylabel("Ratio")
        self.ax_rat.axhline(1, c='gray', ls='--', alpha=0.5)

    def _get_data_pixels(self, display_pixels):
        """Convert display pixel list [(x,y),...] to data coords [(sample, line),...]."""
        data_pixels = []
        for (dx, dy) in display_pixels:
            sx, ly = display_to_data_coords(dx, dy,
                                            (self.n_lines, self.n_samples))
            if 0 <= sx < self.n_samples and 0 <= ly < self.n_lines:
                data_pixels.append((ly, sx))  # (line, sample)
        return data_pixels

    def _extract_spectrum(self, data_pixels):
        """Mean spectrum from data-coord pixel list [(line, sample),...]."""
        specs = []
        for (line, samp) in data_pixels:
            spec = self.data[:, line, samp]
            if np.any(np.isfinite(spec)):
                specs.append(spec)
        if len(specs) == 0:
            return None
        return np.nanmean(np.array(specs), axis=0)

    def _extract_brown_params(self, data_pixels):
        """Extract Brown parameter values for data-coord pixels.
        Returns dict with per-pixel arrays and summary stats."""
        result = {}
        for param in ['centroid', 'asymmetry', 'amplitude', 'hwhm']:
            if param not in self.brown:
                continue
            vals = []
            for (line, samp) in data_pixels:
                v = self.brown[param][line, samp]
                if np.isfinite(v):
                    vals.append(v)
            vals = np.array(vals)
            n_valid = len(vals)
            n_total = len(data_pixels)
            if n_valid > 0:
                result[param] = {
                    'values': vals,
                    'mean': np.mean(vals),
                    'median': np.median(vals),
                    'std': np.std(vals),
                    'min': np.min(vals),
                    'max': np.max(vals),
                    'n_valid': n_valid,
                    'n_total': n_total,
                }
            else:
                result[param] = {
                    'values': vals,
                    'mean': np.nan, 'median': np.nan,
                    'std': np.nan, 'min': np.nan, 'max': np.nan,
                    'n_valid': 0, 'n_total': n_total,
                }
        # Also check how many pass Brown threshold
        if 'mask' in self.brown:
            n_pass = sum(1 for (l, s) in data_pixels
                         if self.brown['mask'][l, s])
            result['n_threshold_pass'] = n_pass
        return result

    def _on_lasso(self, verts):
        path = Path(verts)
        h, w = self.n_lines, self.n_samples
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        points = np.column_stack([xs.ravel(), ys.ravel()])
        mask = path.contains_points(points)
        display_pixels = [(int(p[0]), int(p[1])) for p in points[mask]]

        if len(display_pixels) == 0:
            print("  Empty selection.")
            return

        data_pixels = self._get_data_pixels(display_pixels)
        spec = self._extract_spectrum(data_pixels)
        if spec is None:
            print("  No valid spectra.")
            return

        if self.mode == 'sample':
            color = ROI_COLORS[self.roi_counter % len(ROI_COLORS)]
            # Draw on all panels
            arts = []
            dxs = [p[0] for p in display_pixels]
            dys = [p[1] for p in display_pixels]
            for ax in self.axs:
                m = ax.plot(dxs, dys, 's', color=color, markersize=3,
                            alpha=0.5, markeredgecolor='white',
                            markeredgewidth=0.2)[0]
                arts.append(m)

            # Extract Brown params for sample ROI
            brown_stats = self._extract_brown_params(data_pixels)

            self.pending_sample = {
                'spec': spec,
                'data_pixels': data_pixels,
                'display_pixels': display_pixels,
                'arts': arts,
                'color': color,
                'brown': brown_stats,
            }

            self.mode = 'dust'
            self.fig_map.suptitle(
                f"{self.cube_id} — TER3 Analyzer\n[Mode: DUST]",
                fontsize=10, y=0.98)
            self.fig_map.canvas.draw_idle()
            print(f"  Sample selected ({len(data_pixels)} px). "
                  f"Now lasso DUST region.")

            # Print Brown param preview
            for param in ['centroid', 'asymmetry', 'amplitude', 'hwhm']:
                if param in brown_stats:
                    s = brown_stats[param]
                    print(f"    Brown {param}: "
                          f"mean={s['mean']:.3f} ± {s['std']:.3f} "
                          f"[{s['min']:.3f}, {s['max']:.3f}] "
                          f"({s['n_valid']}/{s['n_total']} valid)")

        elif self.mode == 'dust':
            # Draw dust ROI
            dxs = [p[0] for p in display_pixels]
            dys = [p[1] for p in display_pixels]
            dust_arts = []
            for ax in self.axs:
                m = ax.plot(dxs, dys, 's', color='orange', markersize=3,
                            alpha=0.4, markeredgecolor='white',
                            markeredgewidth=0.2)[0]
                dust_arts.append(m)

            # Compute ratio: SAMPLE / DUST
            sample_spec = self.pending_sample['spec']
            ratio = apply_ratio(sample_spec, spec, self.wav)

            print(f"    Ratio = sample / dust "
                  f"(sample mean I/F={np.nanmean(sample_spec):.4f}, "
                  f"dust mean I/F={np.nanmean(spec):.4f})")

            # Plot
            color = self.pending_sample['color']
            lbl = f"ROI {self.roi_counter + 1}"

            l1, = self.ax_raw.plot(self.wav, sample_spec, color=color,
                                   lw=1, label=lbl)
            l2, = self.ax_rat.plot(self.wav, ratio, color=color,
                                   lw=1, label=lbl)

            # Also plot dust on raw (first time only, or update)
            self.ax_raw.plot(self.wav, spec, color='orange', lw=0.8,
                             alpha=0.5, ls='--', label='_dust')

            self.ax_raw.legend(fontsize=7)
            self.ax_rat.legend(fontsize=7)
            self._autoscale_ratio()

            # Store completed ROI
            all_arts = self.pending_sample['arts'] + dust_arts + [l1, l2]

            self.rois.append({
                'label': lbl,
                'sample_spec': sample_spec,
                'dust_spec': spec,
                'ratio': ratio,
                'sample_data_pixels': self.pending_sample['data_pixels'],
                'dust_data_pixels': data_pixels,
                'color': color,
                'arts': all_arts,
                'brown': self.pending_sample['brown'],
            })
            self.roi_counter += 1

            print(f"  {lbl} complete ({len(self.pending_sample['data_pixels'])} "
                  f"sample px, {len(data_pixels)} dust px)")

            self.pending_sample = None
            self.mode = 'sample'
            self.fig_map.suptitle(
                f"{self.cube_id} — TER3 Analyzer\n"
                f"[Mode: SAMPLE — {len(self.rois)} ROIs]",
                fontsize=10, y=0.98)
            self.fig_map.canvas.draw_idle()
            self.fig_spec.canvas.draw_idle()

    def _autoscale_ratio(self):
        all_y = []
        xlim = self.ax_rat.get_xlim()
        for line in self.ax_rat.get_lines():
            xd = np.array(line.get_xdata(), dtype=float)
            yd = np.array(line.get_ydata(), dtype=float)
            in_range = (xd >= xlim[0]) & (xd <= xlim[1])
            valid = yd[in_range & np.isfinite(yd)]
            if len(valid) > 0:
                all_y.append(valid)
        if all_y:
            all_y = np.concatenate(all_y)
            ymin = np.nanpercentile(all_y, 1)
            ymax = np.nanpercentile(all_y, 99)
            pad = (ymax - ymin) * 0.1
            if pad == 0:
                pad = 0.05
            self.ax_rat.set_ylim(ymin - pad, ymax + pad)

    def _on_key(self, event):
        if event.key == 'q':
            plt.close('all')
        elif event.key == 's':
            self.save_all()
        elif event.key == 'd':
            self._delete_last()
        elif event.key == 'r':
            self._reset()
        elif event.key == 'f':
            # Could add dynamic restretch here if needed
            print("  (Restretch not implemented for static panels)")

    def _delete_last(self):
        if not self.rois:
            print("  No ROIs to delete.")
            return
        roi = self.rois.pop()
        self.roi_counter = max(0, self.roi_counter - 1)
        for a in roi['arts']:
            try:
                a.remove()
            except:
                pass
        self._autoscale_ratio()
        self.fig_map.canvas.draw_idle()
        self.fig_spec.canvas.draw_idle()
        print(f"  Deleted {roi['label']}")

    def _reset(self):
        while self.rois:
            self._delete_last()
        if self.pending_sample:
            for a in self.pending_sample['arts']:
                try:
                    a.remove()
                except:
                    pass
        self.pending_sample = None
        self.mode = 'sample'
        self.roi_counter = 0
        self.ax_raw.clear()
        self.ax_rat.clear()
        self._setup_spec_axes()
        self.fig_map.suptitle(
            f"{self.cube_id} — TER3 Analyzer\n[Mode: SAMPLE]",
            fontsize=10, y=0.98)
        self.fig_map.canvas.draw_idle()
        self.fig_spec.canvas.draw_idle()
        print("  Reset all ROIs.")

    # =========================================================================
    # SAVING
    # =========================================================================

    def save_all(self):
        if not self.rois:
            print("  No ROIs to save.")
            return

        try:
            out_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            out_dir = os.getcwd()
        prefix = self.cube_id

        # --- 1. Ratioed spectra CSV ---
        spec_path = os.path.join(out_dir, f"{prefix}_analysis.csv")
        spec_data = {'wavelength_um': self.wav}
        for roi in self.rois:
            spec_data[f"{roi['label']}_ratio"] = roi['ratio']
            spec_data[f"{roi['label']}_raw"] = roi['sample_spec']
        # Add one dust column (last ROI's dust, or first — they may differ)
        for i, roi in enumerate(self.rois):
            spec_data[f"dust_{i+1}"] = roi['dust_spec']
        df_spec = pd.DataFrame(spec_data)
        df_spec.to_csv(spec_path, index=False)
        print(f"  Saved spectra: {spec_path}")

        # --- 2. Brown parameter stats CSV ---
        brown_path = os.path.join(out_dir, f"{prefix}_brown_params.csv")
        rows = []
        for roi in self.rois:
            row = {
                'roi': roi['label'],
                'n_sample_pixels': len(roi['sample_data_pixels']),
                'n_dust_pixels': len(roi['dust_data_pixels']),
            }
            for param in ['centroid', 'asymmetry', 'amplitude', 'hwhm']:
                if param in roi['brown']:
                    s = roi['brown'][param]
                    row[f'{param}_mean'] = s['mean']
                    row[f'{param}_median'] = s['median']
                    row[f'{param}_std'] = s['std']
                    row[f'{param}_min'] = s['min']
                    row[f'{param}_max'] = s['max']
                    row[f'{param}_n_valid'] = s['n_valid']
            if 'n_threshold_pass' in roi['brown']:
                row['n_brown_threshold_pass'] = roi['brown']['n_threshold_pass']
            rows.append(row)
        df_brown = pd.DataFrame(rows)
        df_brown.to_csv(brown_path, index=False)
        print(f"  Saved Brown params: {brown_path}")

        # --- 3. ROI pixel coordinates CSV ---
        pix_path = os.path.join(out_dir, f"{prefix}_roi_pixels.csv")
        pix_rows = []
        for roi in self.rois:
            for (line, samp) in roi['sample_data_pixels']:
                pix_rows.append({
                    'roi': roi['label'],
                    'type': 'sample',
                    'line': line,
                    'sample': samp,
                })
            for (line, samp) in roi['dust_data_pixels']:
                pix_rows.append({
                    'roi': roi['label'],
                    'type': 'dust',
                    'line': line,
                    'sample': samp,
                })
        df_pix = pd.DataFrame(pix_rows)
        df_pix.to_csv(pix_path, index=False)
        print(f"  Saved ROI pixels: {pix_path} "
              f"({len(pix_rows)} pixel entries)")

        print(f"\n  All outputs saved to: {out_dir}")

    def run(self):
        plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    analyzer = TER3Analyzer()
    analyzer.run()
    # plt.show() returns after all windows are closed
    if analyzer.rois:
        resp = input("\nSave results? (y/n): ").strip().lower()
        if resp == 'y':
            analyzer.save_all()
    else:
        print("\nNo ROIs extracted. Nothing to save.")