"""
brown2020_batch_ter3.py

Batch processing of the olivine 1 µm band complex in CRISM TER3 data,
following Brown, Viviano & Goudge (2020), JGR Planets, 125, e2019JE006011.

Searches TER3 directory (one level deep) for joined CRISM TER3 cubes
(*ter3*.img), fits the asymmetric Gaussian, and outputs:
  - .npy arrays (centroid, amplitude, asymmetry, hwhm, residual, mask)
  - PNG overview plots (thresholded + all fits)

TER3 NOTES:
  - TER3 files are joined VNIR+IR cubes (MRO:SENSOR_ID = "J").
  - Wavelengths stored in NANOMETERS in the ENVI header — converted to µm.
  - TER3 is UNPROJECTED (sensor-space) — no GeoTIFF output.
  - Output .npy arrays share the same pixel grid as the TER3 cube,
    so pixel indices from the TER3 spectra extractor map directly.

Algorithm: scipy.optimize.minimize (Nelder-Mead simplex)
  — Follows Brown et al. (2020) exactly: Nelder & Mead (1965) simplex
    with fitting in wavenumber / apparent-absorbance space.

Author: AV Valantinas
"""

import numpy as np
from scipy.optimize import minimize
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import traceback
import sys
import os
import glob
import gc
from config import CUBES_DIR
import time


# ============================================================================
# USER CONFIGURATION
# ============================================================================

ROOT_DIRECTORY = str(CUBES_DIR)

# Fitting window (µm)
SH_LEFT = 0.75
SH_RIGHT = 1.75

# Savitzky-Golay smoothing before fitting
SMOOTH = False
SMOOTH_WINDOW = 5
SMOOTH_POLY = 2

# Maximum fraction of bad bands allowed within the fitting window
MAX_BAD_FRAC = 0.15

# Bad pixel sentinel value
NAN_VALUE = 65535.0

# Brown et al. (2020) Equation 2 — threshold parameters
THRESH_AMP_MIN = 0.1
THRESH_AMP_MAX = 1.0
THRESH_CENTROID_MIN = 1.15
THRESH_CENTROID_MAX = 1.5
THRESH_HWHM_MAX = 0.3
THRESH_ASYM_MIN = 0.8
THRESH_ASYM_MAX = 9.0

# Plot range
PLOT_VMIN = 1.15
PLOT_VMAX = 1.45

# Version tag
VERSION = "v5"

# Overwrite existing results?
OVERWRITE = True


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class FitResult:
    centroid: float       # λ₀ [µm]
    amplitude: float      # α [0–1]
    hwhm: float           # σ [µm]
    asymmetry: float      # χ [dimensionless]
    rms_residual: float   # RMS of fit residual
    success: bool
    cost: float


# ============================================================================
# ENVI header parser (standalone, no spectral library dependency)
# ============================================================================

def parse_envi_header(hdr_file):
    """Parse ENVI header file into a metadata dict."""
    metadata = {}
    with open(hdr_file, 'r') as f:
        content = f.read()
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if '=' in line and '{' not in line:
            key, value = line.split('=', 1)
            metadata[key.strip().lower()] = value.strip()
        elif '=' in line and '{' in line:
            key = line.split('=')[0].strip().lower()
            val_str = line.split('=')[1].strip()
            while '}' not in val_str and i < len(lines) - 1:
                i += 1
                val_str += ' ' + lines[i].strip()
            val_str = val_str.replace('{', '').replace('}', '').strip()
            metadata[key] = [v.strip() for v in val_str.split(',') if v.strip()]
        i += 1
    return metadata


def read_ter3_cube(img_file):
    """
    Read a CRISM TER3 cube from .img file.
    Returns: (wavelengths_um, cube[lines, samples, bands], bbl)
    """
    hdr = img_file.replace('.img', '.hdr').replace('.IMG', '.hdr')
    if not os.path.exists(hdr):
        hdr = img_file + '.hdr'
    if not os.path.exists(hdr):
        # Try case-insensitive
        base = os.path.splitext(img_file)[0]
        for ext in ['.hdr', '.HDR']:
            candidate = base + ext
            if os.path.exists(candidate):
                hdr = candidate
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

    img_size = os.path.getsize(img_file)
    print(f"  Reading {os.path.basename(img_file)}")
    print(f"    {s} samples × {l} lines × {b} bands, {inter}, "
          f"{img_size / 1024 / 1024:.1f} MB")

    # Wavelengths
    if meta.get('wavelength'):
        wav = np.array([float(w) for w in meta['wavelength']])
        # TER3 wavelengths are in nanometers — convert to µm
        if wav[0] > 100:
            print(f"    Wavelengths in nm — converting to µm")
            wav = wav / 1000.0
        if len(wav) != b:
            raise ValueError(f"Wavelength count ({len(wav)}) != bands ({b})")
    else:
        raise ValueError("No wavelengths in TER3 header")

    print(f"    Wavelengths: {wav[0]:.3f} – {wav[-1]:.3f} µm ({b} bands)")

    # Bad band list
    bbl = None
    if meta.get('bbl'):
        bbl = np.array([int(float(x)) for x in meta['bbl']])
        n_bad_bbl = np.sum(bbl == 0)
        print(f"    BBL: {n_bad_bbl} bad bands flagged")

    # Data ignore value
    ign = float(meta.get('data ignore value', 65535))

    # Read binary data
    dtype = {1: np.uint8, 2: np.int16, 4: np.float32,
             12: np.uint16}.get(dt, np.float32)

    expected_size = l * b * s * np.dtype(dtype).itemsize
    if img_size < expected_size:
        raise ValueError(
            f"IMG file truncated: expected {expected_size:,} bytes, "
            f"got {img_size:,}")

    data = np.fromfile(img_file, dtype=dtype, count=l * b * s)
    if order == 1:
        data = data.byteswap()

    # Reshape according to interleave
    if inter == 'bil':
        data = data.reshape((l, b, s))              # [lines, bands, samples]
    elif inter == 'bip':
        data = data.reshape((l, s, b))               # [lines, samples, bands]
    elif inter == 'bsq':
        data = data.reshape((b, l, s))                # [bands, lines, samples]
        data = data.transpose((1, 2, 0))              # → [lines, samples, bands]

    # For BIL: transpose to [lines, samples, bands]
    if inter == 'bil':
        data = data.transpose((0, 2, 1))              # → [lines, samples, bands]

    data = data.astype(np.float32)

    # Mask bad values
    data[data == ign] = np.nan
    data[data >= 65534] = np.nan
    if bbl is not None:
        data[:, :, np.where(bbl == 0)[0]] = np.nan

    n_total = data.size
    n_valid = np.sum(np.isfinite(data) & (data < NAN_VALUE - 1))
    print(f"    Loaded: shape {data.shape}, "
          f"{data.nbytes / 1024 / 1024:.0f} MB, "
          f"{n_valid:,}/{n_total:,} valid values")

    return wav, data, bbl


# ============================================================================
# Asymmetric Gaussian — Brown et al. (2020) Equation 1
# ============================================================================

def asym_gaussian(x, centroid, amplitude, hwhm, asymmetry):
    out = np.empty_like(x)
    left = x <= centroid
    right = ~left
    out[left] = amplitude * np.exp(-((x[left] - centroid) / hwhm)**2)
    sigma_r = asymmetry * hwhm
    out[right] = amplitude * np.exp(-((x[right] - centroid) / sigma_r)**2)
    return out


# ============================================================================
# Continuum removal
# ============================================================================

def remove_continuum(wl, ref, sh_left=SH_LEFT, sh_right=SH_RIGHT):
    i_l = np.argmin(np.abs(wl - sh_left))
    i_r = np.argmin(np.abs(wl - sh_right))
    wl_sub = wl[i_l:i_r + 1]
    ref_sub = ref[i_l:i_r + 1].copy()

    bad = (~np.isfinite(ref_sub)) | (ref_sub >= NAN_VALUE - 1) | (ref_sub <= 0)
    n_good = np.sum(~bad)
    if n_good < 10:
        return wl_sub, np.ones_like(wl_sub), np.zeros_like(wl_sub)
    if np.any(bad):
        ref_sub[bad] = np.interp(wl_sub[bad], wl_sub[~bad], ref_sub[~bad])

    frac = (wl_sub - wl_sub[0]) / (wl_sub[-1] - wl_sub[0])
    continuum = ref_sub[0] + (ref_sub[-1] - ref_sub[0]) * frac

    with np.errstate(divide='ignore', invalid='ignore'):
        cr = np.where(continuum > 0, ref_sub / continuum, 1.0)

    return wl_sub, cr, 1.0 - cr


# ============================================================================
# Single-spectrum fit — Nelder-Mead in energy space
# ============================================================================

def fit_olivine_band(wl, ref, sh_left=SH_LEFT, sh_right=SH_RIGHT,
                     smooth=SMOOTH, smooth_window=SMOOTH_WINDOW,
                     smooth_poly=SMOOTH_POLY):
    if smooth and len(ref) >= smooth_window:
        ref_s = savgol_filter(ref, smooth_window, smooth_poly)
    else:
        ref_s = ref.copy()

    wl_cr, cr, bd = remove_continuum(wl, ref_s, sh_left, sh_right)

    if len(wl_cr) < 10:
        return (FitResult(np.nan, np.nan, np.nan, np.nan, np.nan, False, np.nan),
                wl_cr, bd)

    # Apparent absorbance in wavenumber space
    cr_clipped = np.clip(cr, 1e-6, 1.0)
    absorbance = -np.log10(cr_clipped)
    wn = 10000.0 / wl_cr

    if np.max(absorbance) <= 0.002:
        c0_wl = wl_cr[np.argmax(bd)]
        return (FitResult(c0_wl, np.max(bd), 0.1, 1.0, np.nan, False, np.nan),
                wl_cr, bd)

    idx_max = np.argmax(absorbance)
    c0_wn = wn[idx_max]
    a0 = absorbance[idx_max]
    hwhm0_wn = 1500.0

    def cost(p):
        centroid_wn, amplitude, hwhm_wn, asymmetry = p
        if amplitude <= 0 or hwhm_wn <= 0 or asymmetry <= 0:
            return 1e10
        model = asym_gaussian(wn, centroid_wn, amplitude, hwhm_wn, asymmetry)
        return np.sum((absorbance - model)**2)

    p0 = [c0_wn, a0, hwhm0_wn, 1.5]

    try:
        sol = minimize(cost, p0, method='Nelder-Mead',
                       options={'maxiter': 5000, 'maxfev': 10000,
                                'xatol': 1e-4, 'fatol': 1e-4,
                                'adaptive': True})
        c_wn, a_abs, s_wn, chi = sol.x

        c_um = 10000.0 / c_wn if c_wn > 0 else np.nan
        s_um = (c_um**2 * s_wn / 10000.0) if np.isfinite(c_um) else np.nan
        a_bd = 1.0 - 10.0**(-a_abs) if a_abs > 0 else 0.0

        model = asym_gaussian(wn, c_wn, a_abs, s_wn, chi)
        rms = np.sqrt(np.mean((absorbance - model)**2))

        return (FitResult(c_um, a_bd, s_um, chi, rms, sol.success, sol.fun),
                wl_cr, bd)
    except Exception:
        return (FitResult(np.nan, np.nan, np.nan, np.nan, np.nan, False, np.nan),
                wl_cr, bd)


# ============================================================================
# Brown Eq. 2 threshold
# ============================================================================

def passes_brown_threshold(r):
    if np.isnan(r.centroid):
        return False
    return (THRESH_AMP_MIN < r.amplitude < THRESH_AMP_MAX and
            THRESH_CENTROID_MIN < r.centroid < THRESH_CENTROID_MAX and
            r.hwhm < THRESH_HWHM_MAX and
            r.hwhm > 0 and
            THRESH_ASYM_MIN < r.asymmetry < THRESH_ASYM_MAX)


# ============================================================================
# Pixel-by-pixel mapping
# ============================================================================

def map_centroid(wl, cube, sh_left=SH_LEFT, sh_right=SH_RIGHT,
                 smooth=SMOOTH, nan_value=NAN_VALUE,
                 max_bad_frac=MAX_BAD_FRAC):
    """
    cube shape: [lines, samples, bands]
    """
    nrows, ncols, nbands = cube.shape
    centroid_map  = np.full((nrows, ncols), np.nan)
    asymmetry_map = np.full((nrows, ncols), np.nan)
    amplitude_map = np.full((nrows, ncols), np.nan)
    hwhm_map      = np.full((nrows, ncols), np.nan)
    residual_map  = np.full((nrows, ncols), np.nan)
    mask = np.zeros((nrows, ncols), dtype=bool)

    fit_mask = (wl >= sh_left) & (wl <= sh_right)
    n_fit_bands = np.sum(fit_mask)
    max_bad = int(n_fit_bands * max_bad_frac)

    total = nrows * ncols
    n_pass = 0
    n_attempted = 0
    n_skipped_bad = 0

    t0 = time.time()

    for i in range(nrows):
        for j in range(ncols):
            spec = cube[i, j, :]

            spec_fit = spec[fit_mask]
            n_bad = np.sum((spec_fit >= nan_value - 1) | (~np.isfinite(spec_fit)))

            if n_bad > max_bad:
                n_skipped_bad += 1
                continue

            valid = spec_fit[np.isfinite(spec_fit) & (spec_fit < nan_value - 1)]
            if len(valid) < 10 or np.max(valid) <= 0:
                n_skipped_bad += 1
                continue

            n_attempted += 1
            result, _, _ = fit_olivine_band(wl, spec, sh_left, sh_right, smooth)

            if not np.isnan(result.centroid):
                if result.asymmetry <= 0 or result.asymmetry > 20:
                    continue
                if result.centroid < 0.8 or result.centroid > 2.0:
                    continue
                centroid_map[i, j]  = result.centroid
                asymmetry_map[i, j] = result.asymmetry
                amplitude_map[i, j] = result.amplitude
                hwhm_map[i, j]      = result.hwhm
                residual_map[i, j]  = result.rms_residual
                if passes_brown_threshold(result):
                    mask[i, j] = True
                    n_pass += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (nrows - i - 1) / rate if rate > 0 else 0
            pct = 100 * (i + 1) / nrows
            print(f"    Row {i+1}/{nrows} ({pct:.0f}%) — "
                  f"{n_pass} pass, {n_attempted} fit, "
                  f"{n_skipped_bad} skipped — ETA {eta:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s. "
          f"{n_pass}/{total} pixels pass threshold ({100*n_pass/total:.1f}%). "
          f"{n_attempted} attempted, {n_skipped_bad} skipped.")

    return {'centroid': centroid_map, 'asymmetry': asymmetry_map,
            'amplitude': amplitude_map, 'hwhm': hwhm_map,
            'residual': residual_map, 'mask': mask}


# ============================================================================
# Save outputs — .npy arrays (same pixel grid as TER3 cube)
# ============================================================================

def save_npy_maps(maps, base_name, out_dir):
    """Save all parameter maps as .npy files for direct pixel-index access."""
    params = ['centroid', 'asymmetry', 'amplitude', 'hwhm', 'residual', 'mask']
    n_saved = 0
    for param in params:
        npy_path = os.path.join(out_dir,
                                f'{base_name}_{param}_map_{VERSION}.npy')
        np.save(npy_path, maps[param])
        n_saved += 1
    print(f"  Saved {n_saved} .npy maps ({', '.join(params)})")


# ============================================================================
# Plot: parameter maps
# ============================================================================

def plot_param_map(maps, param_name, title="",
                   vmin=None, vmax=None, cmap='RdYlBu_r',
                   figsize=(10, 8), save_path=None, label=""):
    """Two-panel plot: thresholded (left) + all fits (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))

    data_thresh = maps[param_name].copy()
    data_thresh[~maps['mask']] = np.nan
    n_thresh = np.sum(maps['mask'])

    data_all = maps[param_name].copy()
    n_valid = np.sum(np.isfinite(data_all))

    # Panel 1: Thresholded
    ax = axes[0]
    im = ax.imshow(data_thresh, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest', origin='upper')
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label(label, fontsize=11)
    ax.set_title(f'{title}\n(Brown Eq.2 threshold, n={n_thresh})',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Line')

    # Panel 2: All fits
    ax2 = axes[1]
    im2 = ax2.imshow(data_all, cmap=cmap, vmin=vmin, vmax=vmax,
                     interpolation='nearest', origin='upper')
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.75, pad=0.02)
    cbar2.set_label(label, fontsize=11)
    ax2.set_title(f'{title}\n(all fits, n={n_valid})',
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Line')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  Saved plot: {os.path.basename(save_path)}")
    plt.close(fig)


def save_all_plots(maps, base_name, out_dir):
    """Generate PNG overview plots for all four parameters."""

    # Centroid
    plot_param_map(maps, 'centroid', title=base_name,
                   vmin=PLOT_VMIN, vmax=PLOT_VMAX, cmap='RdYlBu_r',
                   label='1 µm band centroid (µm)',
                   save_path=os.path.join(
                       out_dir, f'{base_name}_centroid_map_{VERSION}.png'))

    # Asymmetry
    plot_param_map(maps, 'asymmetry', title=base_name,
                   vmin=0.8, vmax=9.0, cmap='magma',
                   label='Asymmetry χ',
                   save_path=os.path.join(
                       out_dir, f'{base_name}_asymmetry_map_{VERSION}.png'))

    # Amplitude
    plot_param_map(maps, 'amplitude', title=base_name,
                   vmin=0.0, vmax=0.6, cmap='inferno',
                   label='Amplitude (band depth)',
                   save_path=os.path.join(
                       out_dir, f'{base_name}_amplitude_map_{VERSION}.png'))

    # HWHM
    plot_param_map(maps, 'hwhm', title=base_name,
                   vmin=0.0, vmax=0.3, cmap='viridis',
                   label='HWHM σ (µm)',
                   save_path=os.path.join(
                       out_dir, f'{base_name}_hwhm_map_{VERSION}.png'))


# ============================================================================
# Find TER3 cubes — one level of subdirectories only
# ============================================================================

def find_ter3_cubes(root_dir):
    """
    Find TER3 .img files in immediate subdirectories of root_dir.
    Pattern: <root>/<subdir>/*ter3*.img (case-insensitive match)
    """
    cubes = []
    if not os.path.isdir(root_dir):
        print(f"ERROR: Root directory does not exist: {root_dir}")
        return cubes

    for entry in sorted(os.listdir(root_dir)):
        subdir = os.path.join(root_dir, entry)
        if not os.path.isdir(subdir):
            continue
        for fname in sorted(os.listdir(subdir)):
            if 'if' in fname.lower() and 'ter3' in fname.lower() and fname.lower().endswith('.img'):
                cubes.append(os.path.join(subdir, fname))

    return cubes


# ============================================================================
# Main Batch Processor
# ============================================================================

def batch_process_directory(root_dir):
    img_files = find_ter3_cubes(root_dir)

    if not img_files:
        print(f"No TER3 .img files found in subdirectories of: {root_dir}")
        return

    print(f"\nFound {len(img_files)} TER3 cubes to process:")
    for f in img_files:
        print(f"  {os.path.basename(f)}")

    print(f"\nMethod: Brown et al. (2020) — Nelder-Mead simplex")
    print(f"Post-hoc threshold (Eq. 2): amp=[{THRESH_AMP_MIN}, {THRESH_AMP_MAX}], "
          f"centroid=[{THRESH_CENTROID_MIN}, {THRESH_CENTROID_MAX}], "
          f"hwhm<{THRESH_HWHM_MAX}, asym=[{THRESH_ASYM_MIN}, {THRESH_ASYM_MAX}]")
    print(f"Fit window: {SH_LEFT} – {SH_RIGHT} µm")
    print(f"Output: .npy arrays (same pixel grid as TER3 cube)")
    print(f"Version: {VERSION}")

    for i, img_path in enumerate(img_files, 1):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        out_dir = os.path.dirname(img_path)

        print(f"\n{'=' * 70}")
        print(f"  Cube {i}/{len(img_files)}: {base_name}")
        print(f"  Directory: {out_dir}")
        print(f"{'=' * 70}")

        # Check if already processed
        npy_check = os.path.join(out_dir,
                                 f'{base_name}_centroid_map_{VERSION}.npy')
        if not OVERWRITE and os.path.exists(npy_check):
            print(f"  SKIP (already processed): {base_name}")
            continue

        cube = None
        maps = None

        try:
            # 1. Load TER3 cube
            wl, cube, bbl = read_ter3_cube(img_path)

            # 2. Fit olivine band
            print(f"\n  Fitting olivine 1 µm band ({SH_LEFT}–{SH_RIGHT} µm)...")
            print(f"  Optimizer: Nelder-Mead simplex (Brown et al. 2020)")
            t_start = time.time()
            maps = map_centroid(wl, cube)
            t_fit = time.time() - t_start
            print(f"  Total fit time: {t_fit:.0f}s ({t_fit / 60:.1f} min)")

            # Free cube memory
            del cube
            cube = None
            gc.collect()

            # 3. Save .npy arrays
            save_npy_maps(maps, base_name, out_dir)

            # 4. Save PNGs (all four parameters)
            save_all_plots(maps, base_name, out_dir)

            # 5. Print summary statistics
            for param in ['centroid', 'asymmetry', 'amplitude', 'hwhm']:
                valid = maps[param][np.isfinite(maps[param])]
                thresh = maps[param].copy()
                thresh[~maps['mask']] = np.nan
                thresh_valid = thresh[np.isfinite(thresh)]
                if len(valid) > 0:
                    print(f"  {param:12s}: all fits "
                          f"mean={np.mean(valid):.3f} "
                          f"std={np.std(valid):.3f} "
                          f"[{np.min(valid):.3f}, {np.max(valid):.3f}] "
                          f"(n={len(valid)})")
                    if len(thresh_valid) > 0:
                        print(f"  {' ':12s}  threshold "
                              f"mean={np.mean(thresh_valid):.3f} "
                              f"std={np.std(thresh_valid):.3f} "
                              f"[{np.min(thresh_valid):.3f}, "
                              f"{np.max(thresh_valid):.3f}] "
                              f"(n={len(thresh_valid)})")

        except Exception as e:
            print(f"\n  {'!' * 60}")
            print(f"  ERROR processing {base_name}:")
            print(f"  {type(e).__name__}: {e}")
            print(f"  {'!' * 60}")
            traceback.print_exc()

        finally:
            plt.close('all')
            if cube is not None:
                del cube
            if maps is not None:
                del maps
            gc.collect()

    print(f"\n{'=' * 70}")
    print("Batch processing complete.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ROOT_DIRECTORY = sys.argv[1]

    batch_process_directory(ROOT_DIRECTORY)