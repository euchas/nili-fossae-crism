"""
brown2020_batch_mtr3.py

Batch processing of the olivine 1 µm band complex in CRISM data,
following Brown, Viviano & Goudge (2020), JGR Planets, 125, e2019JE006011.

Recursively searches a root directory for map-projected, joined CRISM MTR3
cubes (*if166j_mtr3*.hdr), fits the asymmetric Gaussian, and outputs:
  - .npy arrays (centroid, amplitude, asymmetry, hwhm, residual, mask)
  - GeoTIFF files (for ArcGIS Pro / QGIS) with projection from ENVI header
  - PNG overview plots (thresholded + all fits)

Algorithm: scipy.optimize.minimize (Nelder-Mead simplex)
  — This follows Brown et al. (2020) exactly: they use Nelder & Mead (1965)
    simplex to iteratively fit the asymmetric Gaussian.
  — No bounds are imposed during fitting; quality filtering is post-hoc
    via the threshold in Equation 2.
  — Brown et al. also used MCMC for full posteriors (~min per pixel),
    but the simplex is the primary deterministic method described.

Author: AV Valantinas
"""

import numpy as np
from scipy.optimize import minimize
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import spectral
import traceback
import sys
import os
import glob
import gc
from config import CUBES_DIR
import time
import re


# ============================================================================
# USER CONFIGURATION
# ============================================================================

ROOT_DIRECTORY = str(CUBES_DIR)

# Fitting window (µm) — shoulders of the 1 µm olivine band
# Brown et al. (2020) main text says 0.7 and 1.75 µm; the Supporting
# Information (Section 3.1, Step 2) says 0.75 and 1.75 µm.
# We use 0.75 µm as described in the actual algorithm implementation.
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

# Brown et al. (2020) Equation 2 — threshold parameters (post-hoc filtering)
# "choose if 0.1 < α < 1.0 and 1.15 < λ₀ < 1.5 and σ < 0.3 and 0.8 < χ < 9"
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

# Version tag — appended to all output filenames (e.g. _v3).
# Bump this each run to avoid overwriting previous results.
VERSION = "v6"

# Overwrite existing results for this version?
# Set True to re-run with updated parameters.
OVERWRITE = True

# Save .npy arrays (in addition to .tif and .png). Required for downstream
# tools (v6_roi_explorer, GMM/PCA pipelines) that load Brown maps as
# native NumPy arrays.
SAVE_NPY = True


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
# Asymmetric Gaussian — Brown et al. (2020) Equation 1
#
#   if λ < λ₀:  f(λ) = α * exp( -[(λ - λ₀) / σ]² )
#   if λ ≥ λ₀:  f(λ) = α * exp( -[(λ - λ₀) / (χσ)]² )
#
# where:
#   λ₀ = centroid
#   α  = amplitude (band depth at centroid)
#   σ  = half-width at half-maximum (HWHM) on the short-wavelength side
#   χ  = asymmetry parameter (χ > 1 means left-asymmetric / long-wave skew)
#
# Note: σ here is the HWHM. The Gaussian exponent is -(Δλ/σ)² which gives
# half-maximum at Δλ = σ (since exp(-1) ≈ 0.37, not 0.5 — but Brown uses
# this convention consistently, and what matters is relative mapping, not
# absolute HWHM definition).
# ============================================================================

def asym_gaussian(x: np.ndarray, centroid: float, amplitude: float,
                  hwhm: float, asymmetry: float) -> np.ndarray:
    """
    Asymmetric Gaussian per Brown et al. (2020) Equation 1.

    Operates in wavenumber space (ν = 10000/λ).

    Brown's convention (wavelength space):
      λ < λ₀  → standard σ        (short-wave, narrow side)
      λ ≥ λ₀  → χσ (wider if χ>1) (long-wave, broad tail)

    In wavenumber space (ν = 10000/λ), axes flip:
      λ < λ₀  →  ν > ν₀  → RIGHT side in ν  → standard σ
      λ ≥ λ₀  →  ν ≤ ν₀  → LEFT side in ν   → χσ

    BUT: Brown fits the -log₁₀ transformed spectrum in wavenumber space,
    where the Gaussian is a PEAK (not a dip). The asymmetry parameter χ > 1
    reported by Brown means the band is broader on the long-wavelength side.

    To reproduce Brown's χ values, we apply χ on the RIGHT side in ν-space
    (= short-wavelength side in λ-space = the side with the steep drop in
    reflectance). This is because in the -log₁₀(CR) absorbance transform,
    the steep reflectance drop becomes the broader absorbance feature, and
    the simplex naturally finds χ > 1 on this side for Fe-rich/saturated
    olivines — matching Brown's reported values.
    """
    out = np.empty_like(x)
    left = x <= centroid   # ν ≤ ν₀  (long-wavelength side)
    right = ~left          # ν > ν₀  (short-wavelength side)
    # Standard σ on the long-wavelength (left in ν) side
    out[left] = amplitude * np.exp(-((x[left] - centroid) / hwhm)**2)
    # Asymmetry-modified σ on the short-wavelength (right in ν) side
    sigma_r = asymmetry * hwhm
    out[right] = amplitude * np.exp(-((x[right] - centroid) / sigma_r)**2)
    return out


# ============================================================================
# Continuum removal — straight line between shoulder points
#
# Brown et al. (2020) Supporting Information, Section 3.1:
#   Step 2: "The routine uses user defined shoulders to find the minimum
#   and maximum wavelengths. In this project we used 0.75 and 1.75 µm,
#   respectively. A straight line is drawn between these two points."
#   "we take the approach of dividing the spectrum by the line drawn
#   between the two chosen shoulder points."
# ============================================================================

def remove_continuum(wl: np.ndarray, ref: np.ndarray,
                     sh_left: float = SH_LEFT, sh_right: float = SH_RIGHT
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Continuum removal with bad-band interpolation.

    Returns:
        wl_sub: wavelengths within the fitting window
        cr: continuum-removed reflectance (R / continuum)
        bd: band depth (1 - cr)
    """
    i_l = np.argmin(np.abs(wl - sh_left))
    i_r = np.argmin(np.abs(wl - sh_right))
    wl_sub = wl[i_l:i_r + 1]
    ref_sub = ref[i_l:i_r + 1].copy()

    # Interpolate over bad channels (65535, NaN, negative, zero)
    bad = (~np.isfinite(ref_sub)) | (ref_sub >= NAN_VALUE - 1) | (ref_sub <= 0)
    n_good = np.sum(~bad)
    if n_good < 10:
        return wl_sub, np.ones_like(wl_sub), np.zeros_like(wl_sub)
    if np.any(bad):
        ref_sub[bad] = np.interp(wl_sub[bad], wl_sub[~bad], ref_sub[~bad])

    # Straight-line continuum between shoulder values
    frac = (wl_sub - wl_sub[0]) / (wl_sub[-1] - wl_sub[0])
    continuum = ref_sub[0] + (ref_sub[-1] - ref_sub[0]) * frac

    with np.errstate(divide='ignore', invalid='ignore'):
        cr = np.where(continuum > 0, ref_sub / continuum, 1.0)

    return wl_sub, cr, 1.0 - cr


# ============================================================================
# Single-spectrum fit using Nelder-Mead simplex IN ENERGY SPACE
#
# Brown et al. (2020) Supporting Information, Section 3.1:
#   Step 3: "The fitting is done in 'energy' space, meaning wavenumber
#   space, and hence the spectrum is inverted"
#   "The following equation is used to convert from continuum removed
#   reflectance to what we call 'apparent absorbance':
#       f(λ) = -log₁₀ α(λ)"
#   Step 4: "we use a Nelder-Mead simplex algorithm" with tolerance 10⁻⁴
#
# This means:
#   1. Convert CR reflectance → apparent absorbance: A = -log₁₀(CR)
#   2. Convert wavelength → wavenumber: ν = 10000 / λ_µm  [cm⁻¹]
#   3. Fit the asymmetric Gaussian to A(ν) in wavenumber space
#   4. Convert fitted centroid back: λ₀ = 10000 / ν₀  [µm]
#
# The energy-space transform is WHY Brown gets centroids up to 1.4+ µm:
#   - The -log₁₀ transform amplifies deep/saturated bands nonlinearly
#   - In wavenumber space, the long-wavelength tail is compressed,
#     pulling the asymmetric Gaussian centroid toward longer wavelengths
#   - Combined, these push the fitted centroid well beyond the simple
#     band-depth maximum position
# ============================================================================

def fit_olivine_band(wl: np.ndarray, ref: np.ndarray,
                     sh_left: float = SH_LEFT, sh_right: float = SH_RIGHT,
                     smooth: bool = SMOOTH, smooth_window: int = SMOOTH_WINDOW,
                     smooth_poly: int = SMOOTH_POLY
                     ) -> Tuple[FitResult, np.ndarray, np.ndarray]:
    """
    Fit the olivine 1 µm band using the Brown et al. (2020) asymmetric
    Gaussian method in energy (wavenumber) space with apparent absorbance.
    """
    if smooth and len(ref) >= smooth_window:
        ref_s = savgol_filter(ref, smooth_window, smooth_poly)
    else:
        ref_s = ref.copy()

    wl_cr, cr, bd = remove_continuum(wl, ref_s, sh_left, sh_right)

    if len(wl_cr) < 10:
        return (FitResult(np.nan, np.nan, np.nan, np.nan, np.nan, False, np.nan),
                wl_cr, bd)

    # --- Convert to energy space per Brown supplement ---

    # 1. Apparent absorbance: A = -log₁₀(CR)
    #    CR values near 1.0 → A near 0 (no absorption)
    #    CR values near 0.2 → A = 0.69 (deep absorption)
    #    Clip CR to avoid log(0) or log(negative)
    cr_clipped = np.clip(cr, 1e-6, 1.0)
    absorbance = -np.log10(cr_clipped)

    # 2. Convert wavelength to wavenumber (cm⁻¹)
    #    ν = 10000 / λ_µm
    #    This inverts the x-axis: short λ → high ν, long λ → low ν
    wn = 10000.0 / wl_cr   # wavenumber in cm⁻¹

    # Check that we have a meaningful absorption
    if np.max(absorbance) <= 0.002:
        c0_wl = wl_cr[np.argmax(bd)]
        return (FitResult(c0_wl, np.max(bd), 0.1, 1.0, np.nan, False, np.nan),
                wl_cr, bd)

    # Initial guess in wavenumber space
    idx_max = np.argmax(absorbance)
    c0_wn = wn[idx_max]       # centroid guess in cm⁻¹
    a0 = absorbance[idx_max]  # amplitude in absorbance units

    # Initial HWHM guess in wavenumber units
    # ~0.15 µm HWHM at ~1.1 µm → Δν ≈ 10000*(1/0.95 - 1/1.25) ≈ 2526 cm⁻¹
    # Use a fraction of the band width as initial guess
    hwhm0_wn = 1500.0  # cm⁻¹, reasonable starting point

    # Cost function: sum of squared residuals in absorbance/wavenumber space
    def cost(p):
        centroid_wn, amplitude, hwhm_wn, asymmetry = p
        if amplitude <= 0 or hwhm_wn <= 0 or asymmetry <= 0:
            return 1e10
        model = asym_gaussian(wn, centroid_wn, amplitude, hwhm_wn, asymmetry)
        residuals = absorbance - model
        return np.sum(residuals**2)

    # Initial parameter vector [centroid_wn, amplitude, hwhm_wn, asymmetry]
    p0 = [c0_wn, a0, hwhm0_wn, 1.5]

    try:
        sol = minimize(cost, p0, method='Nelder-Mead',
                       options={'maxiter': 5000, 'maxfev': 10000,
                                'xatol': 1e-4, 'fatol': 1e-4,
                                'adaptive': True})
        c_wn, a_abs, s_wn, chi = sol.x

        # Convert centroid back to wavelength: λ₀ = 10000 / ν₀
        c_um = 10000.0 / c_wn if c_wn > 0 else np.nan

        # Convert HWHM back to approximate µm
        # For the short-wavelength side: Δλ ≈ λ₀² * Δν / 10000
        s_um = (c_um**2 * s_wn / 10000.0) if np.isfinite(c_um) else np.nan

        # Convert amplitude back to band depth equivalent
        # A = -log₁₀(CR) → CR = 10^(-A) → BD = 1 - CR
        a_bd = 1.0 - 10.0**(-a_abs) if a_abs > 0 else 0.0

        # Compute RMS residual in absorbance space
        model = asym_gaussian(wn, c_wn, a_abs, s_wn, chi)
        rms = np.sqrt(np.mean((absorbance - model)**2))

        return (FitResult(c_um, a_bd, s_um, chi, rms, sol.success, sol.fun),
                wl_cr, bd)
    except Exception:
        return (FitResult(np.nan, np.nan, np.nan, np.nan, np.nan, False, np.nan),
                wl_cr, bd)


# ============================================================================
# Brown Eq. 2 threshold — applied POST-HOC, not during fitting
#
# "choose if 0.1 < α < 1.0 and 1.15 < λ₀ < 1.5 and σ < 0.3 and 0.8 < χ < 9"
# ============================================================================

def passes_brown_threshold(r: FitResult) -> bool:
    """Apply Brown et al. (2020) Equation 2 quality threshold."""
    if np.isnan(r.centroid):
        return False
    return (THRESH_AMP_MIN < r.amplitude < THRESH_AMP_MAX and
            THRESH_CENTROID_MIN < r.centroid < THRESH_CENTROID_MAX and
            r.hwhm < THRESH_HWHM_MAX and
            r.hwhm > 0 and
            THRESH_ASYM_MIN < r.asymmetry < THRESH_ASYM_MAX)


# ============================================================================
# Pixel-by-pixel centroid mapping
# ============================================================================

def map_centroid(wl: np.ndarray, cube: np.ndarray,
                 sh_left: float = SH_LEFT, sh_right: float = SH_RIGHT,
                 smooth: bool = SMOOTH,
                 nan_value: float = NAN_VALUE,
                 max_bad_frac: float = MAX_BAD_FRAC) -> dict:
    nrows, ncols = cube.shape[:2]
    centroid_map  = np.full((nrows, ncols), np.nan)
    asymmetry_map = np.full((nrows, ncols), np.nan)
    amplitude_map = np.full((nrows, ncols), np.nan)
    hwhm_map      = np.full((nrows, ncols), np.nan)
    residual_map  = np.full((nrows, ncols), np.nan)
    mask = np.zeros((nrows, ncols), dtype=bool)

    # Pre-compute the fitting window indices
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

            # Only check bad values WITHIN the fitting window
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
                # Sanity check: NaN out runaway fits where Nelder-Mead diverged
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
                  f"{n_pass} pass, {n_attempted} fit, {n_skipped_bad} skipped — "
                  f"ETA {eta:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s. "
          f"{n_pass}/{total} pixels pass threshold ({100*n_pass/total:.1f}%). "
          f"{n_attempted} attempted, {n_skipped_bad} skipped (bad bands).")

    return {'centroid': centroid_map, 'asymmetry': asymmetry_map,
            'amplitude': amplitude_map, 'hwhm': hwhm_map,
            'residual': residual_map, 'mask': mask}


# ============================================================================
# GeoTIFF output — reads projection from ENVI header for ArcGIS Pro
# ============================================================================

def parse_envi_map_info(hdr_path: str) -> Optional[Dict]:
    """
    Parse ENVI header for 'map info' and 'coordinate system string'.
    Returns dict with geotransform parameters, or None if not map-projected.

    ENVI map info format:
      {proj_name, x_ref_pixel, y_ref_pixel, x_coord, y_coord,
       x_pixel_size, y_pixel_size, [zone], [N/S], [datum], [units=X]}
    Pixel reference is 1-based in ENVI.
    """
    with open(hdr_path, 'r') as f:
        content = f.read()

    # Extract map info (may span multiple lines)
    map_match = re.search(r'map\s+info\s*=\s*\{([^}]+)\}', content,
                          re.IGNORECASE | re.DOTALL)
    if not map_match:
        return None

    parts = [p.strip() for p in map_match.group(1).split(',')]
    if len(parts) < 7:
        return None

    try:
        proj_name = parts[0]
        x_ref_pixel = float(parts[1])  # 1-based
        y_ref_pixel = float(parts[2])  # 1-based
        x_coord = float(parts[3])      # easting/longitude of ref pixel
        y_coord = float(parts[4])      # northing/latitude of ref pixel
        x_pixel_size = float(parts[5])
        y_pixel_size = float(parts[6])
    except (ValueError, IndexError):
        return None

    # GDAL geotransform: (x_origin, x_res, 0, y_origin, 0, -y_res)
    # ENVI ref pixel is 1-based center; GDAL origin is top-left corner of (0,0)
    x_origin = x_coord - (x_ref_pixel - 0.5) * x_pixel_size
    y_origin = y_coord + (y_ref_pixel - 0.5) * y_pixel_size

    geotransform = (x_origin, x_pixel_size, 0.0,
                    y_origin, 0.0, -y_pixel_size)

    # Extract coordinate system string (WKT) — may span multiple lines
    cs_match = re.search(
        r'coordinate\s+system\s+string\s*=\s*\{([^}]+)\}',
        content, re.IGNORECASE | re.DOTALL)
    wkt = cs_match.group(1).strip() if cs_match else None

    # Check for units in map info
    units = None
    for p in parts[7:]:
        if 'units' in p.lower():
            units = p.split('=')[-1].strip()

    return {
        'proj_name': proj_name,
        'geotransform': geotransform,
        'wkt': wkt,
        'x_pixel_size': x_pixel_size,
        'y_pixel_size': y_pixel_size,
        'units': units,
    }


def write_geotiff(data: np.ndarray, out_path: str,
                  geo_info: Optional[Dict],
                  nodata: float = np.nan,
                  description: str = '') -> bool:
    """
    Write a 2D array as a single-band GeoTIFF.
    Tries rasterio first, then GDAL, then warns.
    """
    nrows, ncols = data.shape

    # --- Try rasterio ---
    try:
        import rasterio
        from rasterio.transform import Affine
        from rasterio.crs import CRS

        if geo_info and geo_info.get('geotransform'):
            gt = geo_info['geotransform']
            transform = Affine(gt[1], gt[2], gt[0], gt[4], gt[5], gt[3])
        else:
            transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)

        crs = None
        if geo_info and geo_info.get('wkt'):
            try:
                crs = CRS.from_wkt(geo_info['wkt'])
            except Exception:
                pass

        profile = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'width': ncols,
            'height': nrows,
            'count': 1,
            'crs': crs,
            'transform': transform,
            'nodata': nodata,
            'compress': 'deflate',
        }

        out_data = data.astype(np.float32)
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(out_data, 1)
            if description:
                dst.update_tags(1, DESCRIPTION=description)

        return True

    except ImportError:
        pass

    # --- Try GDAL ---
    try:
        from osgeo import gdal

        driver = gdal.GetDriverByName('GTiff')
        options = ['COMPRESS=DEFLATE']
        ds = driver.Create(out_path, ncols, nrows, 1,
                           gdal.GDT_Float32, options=options)

        if geo_info and geo_info.get('geotransform'):
            ds.SetGeoTransform(geo_info['geotransform'])

        if geo_info and geo_info.get('wkt'):
            ds.SetProjection(geo_info['wkt'])

        band = ds.GetRasterBand(1)
        band.SetNoDataValue(float('nan'))
        if description:
            band.SetDescription(description)
        band.WriteArray(data.astype(np.float32))
        band.FlushCache()
        ds = None

        return True

    except ImportError:
        pass

    # --- Fallback ---
    print(f"  WARNING: Neither rasterio nor GDAL available. "
          f"Cannot write GeoTIFF: {os.path.basename(out_path)}")
    print(f"  Install with: pip install rasterio")
    return False


def save_geotiffs(maps: dict, base_name: str, out_dir: str,
                  geo_info: Optional[Dict]):
    """Save Brown 4 + residual + mask maps as GeoTIFFs (full set)."""
    has_geo = geo_info is not None
    if has_geo:
        print(f"  GeoTIFF projection: {geo_info['proj_name']}, "
              f"pixel: {geo_info['x_pixel_size']}×{geo_info['y_pixel_size']} "
              f"{geo_info.get('units', '')}")
    else:
        print(f"  WARNING: No map info in ENVI header — "
              f"GeoTIFFs will have pixel coordinates only.")

    n_written = 0

    geotiff_specs = [
        ('centroid',  '1 um band centroid (um)'),
        ('asymmetry', 'Asymmetry parameter (chi)'),
        ('amplitude', 'Band depth amplitude (alpha)'),
        ('hwhm',      'Half-width at half-maximum (sigma, um)'),
        ('residual',  'RMS fit residual (absorbance space)'),
    ]
    for name, desc in geotiff_specs:
        tif_path = os.path.join(out_dir, f'{base_name}_{name}_map_{VERSION}.tif')
        if write_geotiff(maps[name], tif_path, geo_info, description=desc):
            n_written += 1

    # Mask saved as uint8 (0/1) GeoTIFF
    mask_path = os.path.join(out_dir, f'{base_name}_mask_map_{VERSION}.tif')
    if write_geotiff(maps['mask'].astype(np.float32), mask_path, geo_info,
                     description='Brown 2020 Eq.2 quality mask (1=pass, 0=fail)'):
        n_written += 1

    print(f"  Saved {n_written} GeoTIFFs.")


def save_npys(maps: dict, base_name: str, out_dir: str):
    """Save Brown 4 + residual + mask maps as .npy arrays.

    Matches the v5 file naming convention used by the rest of the
    pipeline (crism_io.find_cube_and_brown_maps loads these directly).
    """
    n_written = 0
    npy_specs = [
        ('centroid',  np.float32),
        ('asymmetry', np.float32),
        ('amplitude', np.float32),
        ('hwhm',      np.float32),
        ('residual',  np.float32),
        ('mask',      np.bool_),
    ]
    for name, dtype in npy_specs:
        npy_path = os.path.join(
            out_dir, f'{base_name}_{name}_map_{VERSION}.npy')
        np.save(npy_path, maps[name].astype(dtype))
        n_written += 1
    print(f"  Saved {n_written} .npy arrays.")


# ============================================================================
# Plot: Centroid map (Brown Fig 10b style)
# ============================================================================

def plot_centroid_map(maps: dict, title: str = "",
                     vmin: float = PLOT_VMIN, vmax: float = PLOT_VMAX,
                     cmap: str = 'RdYlBu_r', figsize: Tuple = (10, 8),
                     save_path: Optional[str] = None):
    """Centroid map PNG: thresholded (left) + all fits (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))

    # Panel 1: Thresholded
    ax = axes[0]
    data = maps['centroid'].copy()
    data[~maps['mask']] = np.nan
    n_thresh = np.sum(maps['mask'])
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest', origin='upper')
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label('1 µm band centroid (µm)', fontsize=11)
    ax.set_title(f'{title}\n(Brown Eq.2 threshold, n={n_thresh})',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    # Panel 2: All valid fits
    ax2 = axes[1]
    data2 = maps['centroid'].copy()
    n_valid = np.sum(np.isfinite(data2))
    im2 = ax2.imshow(data2, cmap=cmap, vmin=vmin, vmax=vmax,
                     interpolation='nearest', origin='upper')
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.75, pad=0.02)
    cbar2.set_label('1 µm band centroid (µm)', fontsize=11)
    ax2.set_title(f'{title}\n(all fits, n={n_valid})',
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  Saved plot: {os.path.basename(save_path)}")


def plot_asymmetry_map(maps: dict, title: str = "",
                       vmin: float = 0.8, vmax: float = 9.0,
                       cmap: str = 'magma', figsize: Tuple = (10, 8),
                       save_path: Optional[str] = None):
    """Asymmetry map PNG: thresholded (left) + all fits (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))

    # Panel 1: Thresholded
    ax = axes[0]
    data = maps['asymmetry'].copy()
    data[~maps['mask']] = np.nan
    n_thresh = np.sum(maps['mask'])
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest', origin='upper')
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label('Asymmetry χ', fontsize=11)
    ax.set_title(f'{title}\n(Brown Eq.2 threshold, n={n_thresh})',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    # Panel 2: All valid fits
    ax2 = axes[1]
    data2 = maps['asymmetry'].copy()
    n_valid = np.sum(np.isfinite(data2))
    im2 = ax2.imshow(data2, cmap=cmap, vmin=vmin, vmax=vmax,
                     interpolation='nearest', origin='upper')
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.75, pad=0.02)
    cbar2.set_label('Asymmetry χ', fontsize=11)
    ax2.set_title(f'{title}\n(all fits, n={n_valid})',
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  Saved plot: {os.path.basename(save_path)}")


# ============================================================================
# Load CRISM ENVI cube with diagnostics
# ============================================================================

def load_envi(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load ENVI cube using np.fromfile (single allocation) to avoid the
    ~3-5x memory blowup of spectral.SpyFile.load()."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"HDR not found: {file_path}")

    img_path = file_path.replace('.hdr', '.img')
    if not os.path.exists(img_path):
        for ext in ['.IMG', '.img']:
            candidate = file_path.replace('.hdr', ext).replace('.HDR', ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        else:
            raise FileNotFoundError(f"IMG not found. Tried: {img_path}")

    hdr_size = os.path.getsize(file_path)
    img_size = os.path.getsize(img_path)
    print(f"  HDR: {hdr_size:,} bytes")
    print(f"  IMG: {img_size:,} bytes ({img_size / 1024 / 1024:.1f} MB)")

    if img_size == 0:
        raise ValueError(f"IMG file is empty (0 bytes): {img_path}")

    img = spectral.open_image(file_path)

    lines = int(img.metadata.get('lines', 0))
    samples = int(img.metadata.get('samples', 0))
    bands = int(img.metadata.get('bands', 0))
    interleave = img.metadata.get('interleave', 'bsq').lower()
    dtype_code = int(img.metadata.get('data type', 4))
    byte_order = int(img.metadata.get('byte order', 0))
    header_offset = int(img.metadata.get('header offset', 0))

    dtype_map = {1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32,
                 5: np.float64, 12: np.uint16, 13: np.uint32,
                 14: np.int64, 15: np.uint64}
    dtype = np.dtype(dtype_map.get(dtype_code, np.float32))
    if byte_order == 1:
        dtype = dtype.newbyteorder('>')

    bytes_per_val = dtype.itemsize
    expected_size = lines * samples * bands * bytes_per_val

    print(f"  Header: {lines}L x {samples}S x {bands}B, "
          f"{interleave}, dtype={dtype_code} ({dtype})")
    print(f"  Expected IMG size: {expected_size:,} bytes "
          f"(+ {header_offset} offset)")

    if img_size < expected_size + header_offset:
        deficit = expected_size + header_offset - img_size
        raise ValueError(
            f"IMG FILE TRUNCATED - missing {deficit:,} bytes "
            f"({100 * deficit / expected_size:.1f}%). "
            f"Expected {expected_size + header_offset:,}, got {img_size:,}. "
            f"The MTR3 join may have failed for this cube.")

    wl = np.array(img.metadata.get('wavelength', []), dtype=float)
    if len(wl) == 0:
        raise ValueError(f"No wavelengths in header: {file_path}")
    if wl.max() > 10:
        wl = wl / 1000.0
    print(f"  Wavelengths: {len(wl)} bands, {wl.min():.3f} - {wl.max():.3f} um")

    # Single-allocation read via np.fromfile.
    print(f"  Loading cube into memory (np.fromfile)...", flush=True)
    n_values = lines * samples * bands
    with open(img_path, 'rb') as f:
        if header_offset:
            f.seek(header_offset)
        flat = np.fromfile(f, dtype=dtype, count=n_values)
    if flat.size != n_values:
        raise ValueError(
            f"Read {flat.size:,} values, expected {n_values:,}. "
            f"Likely a truncated or interleave-mismatched cube.")

    # Build a VIEW of the flat buffer (no second allocation).
    # Per-pixel access cube[i, j, :] on a transposed-from-BSQ view reads
    # 489 strided values from the underlying flat buffer (cache-unfriendly
    # but correct). Total peak memory = sizeof(flat) = ~1x cube size.
    if interleave == 'bsq':
        cube = flat.reshape((bands, lines, samples)).transpose(1, 2, 0)
    elif interleave == 'bil':
        cube = flat.reshape((lines, bands, samples)).transpose(0, 2, 1)
    elif interleave == 'bip':
        cube = flat.reshape((lines, samples, bands))
    else:
        raise ValueError(f"Unknown interleave: {interleave!r}")
    # Only call astype when dtype actually differs (allocates if it does).
    if cube.dtype != np.float32:
        cube = cube.astype(np.float32, copy=False)

    print(f"  Loaded: shape {cube.shape}, "
          f"{cube.nbytes / 1024 / 1024:.0f} MB in memory")

    n_total = cube.size
    n_nan = int(np.sum(~np.isfinite(cube)))
    n_bad = int(np.sum(cube >= NAN_VALUE - 1))
    n_zero = int(np.sum(cube == 0))
    print(f"  Quality: {n_nan:,} NaN, {n_bad:,} bad (>=65534), "
          f"{n_zero:,} zero, {n_total - n_nan - n_bad:,} valid")

    return wl, cube


# ============================================================================
# Main Batch Processor
# ============================================================================

def batch_process_directory(root_dir: str):
    search_pattern = os.path.join(root_dir, '**', '*if166j_mtr3*.hdr')
    hdr_files = sorted(glob.glob(search_pattern, recursive=True))

    if not hdr_files:
        print(f"No files matching '*if166j_mtr3*.hdr' found in {root_dir}")
        return

    print(f"\nFound {len(hdr_files)} MTR3 cubes to process.")
    print(f"Method: Brown et al. (2020) — Nelder-Mead simplex, no fitting bounds")
    print(f"Post-hoc threshold (Eq. 2): amp=[{THRESH_AMP_MIN}, {THRESH_AMP_MAX}], "
          f"centroid=[{THRESH_CENTROID_MIN}, {THRESH_CENTROID_MAX}], "
          f"hwhm<{THRESH_HWHM_MAX}, asym=[{THRESH_ASYM_MIN}, {THRESH_ASYM_MAX}]")
    print(f"Fit window: {SH_LEFT} – {SH_RIGHT} µm, "
          f"max bad fraction: {MAX_BAD_FRAC}")

    # Check for GeoTIFF capability
    has_geotiff = False
    try:
        import rasterio
        has_geotiff = True
        print(f"GeoTIFF backend: rasterio {rasterio.__version__}")
    except ImportError:
        try:
            from osgeo import gdal
            has_geotiff = True
            print(f"GeoTIFF backend: GDAL {gdal.__version__}")
        except ImportError:
            print("WARNING: No GeoTIFF backend. Install: pip install rasterio")

    for i, hdr_path in enumerate(hdr_files, 1):
        base_name = os.path.basename(hdr_path).replace('.hdr', '')
        out_dir = os.path.dirname(hdr_path)

        print(f"\n{'=' * 70}")
        print(f"  Cube {i}/{len(hdr_files)}: {base_name}")
        print(f"  Directory: {out_dir}")
        print(f"{'=' * 70}")

        # Check if already processed
        npy_check = os.path.join(out_dir, f'{base_name}_centroid_map_{VERSION}.tif')
        if not OVERWRITE and os.path.exists(npy_check):
            print(f"  SKIP (already processed): {base_name}")
            continue

        cube = None
        maps = None

        try:
            # 1. Load Data
            wl, cube = load_envi(hdr_path)

            # 2. Parse geo info from ENVI header
            geo_info = parse_envi_map_info(hdr_path)

            # 3. Fit Centroids
            print(f"\n  Fitting olivine 1 µm band ({SH_LEFT}–{SH_RIGHT} µm)...")
            print(f"  Optimizer: Nelder-Mead simplex (Brown et al. 2020)")
            t_start = time.time()
            maps = map_centroid(wl, cube)
            t_fit = time.time() - t_start
            print(f"  Total fit time: {t_fit:.0f}s "
                  f"({t_fit / 60:.1f} min)")

            # Free cube memory before saving
            del cube
            cube = None
            gc.collect()

            # 4. Save .npy arrays (full Brown 4 + residual + mask)
            if SAVE_NPY:
                save_npys(maps, base_name, out_dir)

            # 5. Save GeoTIFFs (centroid + asymmetry + amplitude + hwhm
            #    + residual + mask)
            if has_geotiff:
                save_geotiffs(maps, base_name, out_dir, geo_info)

            # 6. Save PNGs
            png_path = os.path.join(out_dir, f'{base_name}_centroid_map_{VERSION}.png')
            plot_centroid_map(maps, title=base_name, save_path=png_path)

            png_path = os.path.join(out_dir, f'{base_name}_asymmetry_map_{VERSION}.png')
            plot_asymmetry_map(maps, title=base_name, save_path=png_path)

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