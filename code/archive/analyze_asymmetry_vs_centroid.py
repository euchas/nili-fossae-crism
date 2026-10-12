"""
plot_centroid_vs_asymmetry.py

Plots centroid vs asymmetry scatter (Brown et al. 2020, Figure 8b/9f style)
from the GeoTIFF output maps.

Usage:
    python plot_centroid_vs_asymmetry.py
    python plot_centroid_vs_asymmetry.py <centroid.tif> <asymmetry.tif>
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
import os
from config import CUBES_DIR


# ============================================================================
# DEFAULT PATHS — edit these or pass as arguments
# ============================================================================

DEFAULT_CENTROID = str(CUBES_DIR / "frt00003e12_07_if166j_mtr3" / "frt00003e12_07_if166j_mtr3_centroid_map_v5.tif")
DEFAULT_ASYMMETRY = str(CUBES_DIR / "frt00003e12_07_if166j_mtr3" / "frt00003e12_07_if166j_mtr3_asymmetry_map_v5.tif")


def load_tif(path: str) -> np.ndarray:
    """Load a single-band GeoTIFF as a 2D array."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            return src.read(1)
    except ImportError:
        from osgeo import gdal
        ds = gdal.Open(path)
        data = ds.GetRasterBand(1).ReadAsArray()
        ds = None
        return data


def plot_centroid_vs_asymmetry(centroid_path: str, asymmetry_path: str):
    # Load maps
    centroid = load_tif(centroid_path)
    asymmetry = load_tif(asymmetry_path)

    # Extract observation name from filename
    base = os.path.basename(centroid_path)
    # Try to get the CRISM ID (e.g. "3E12" from "frt00003e12_...")
    obs_name = base.split('_')[0].upper()
    if 'frt' in base.lower() or 'hrl' in base.lower():
        # Extract short ID like "3E12"
        parts = base.lower().replace('frt', '').replace('hrl', '')
        short_id = parts.split('_')[0].lstrip('0').upper()
        obs_name = short_id

    # Mask to valid pixels (both maps must be finite)
    valid = np.isfinite(centroid) & np.isfinite(asymmetry)
    cx = centroid[valid].ravel()
    ay = asymmetry[valid].ravel()

    print(f"Valid pixels: {len(cx):,}")
    print(f"Centroid range:  {cx.min():.3f} – {cx.max():.3f} µm")
    print(f"Asymmetry range: {ay.min():.3f} – {ay.max():.3f}")

    # Report outliers above the plot range
    plot_ymax = 7.0
    outliers = ay > plot_ymax
    n_out = np.sum(outliers)
    print(f"\nPixels with χ > {plot_ymax} (above plot range): {n_out:,}")
    if n_out > 0:
        # Get their (row, col) locations
        mask_outlier = np.isfinite(asymmetry) & (asymmetry > plot_ymax)
        rows, cols = np.where(mask_outlier)
        print(f"  χ values of outliers: min={ay[outliers].min():.3f}, "
              f"max={ay[outliers].max():.3f}, median={np.median(ay[outliers]):.3f}")
        print(f"  Centroid of outliers: min={cx[outliers].min():.3f}, "
              f"max={cx[outliers].max():.3f}, median={np.median(cx[outliers]):.3f}")
        print(f"  Locations (row, col) — first 20:")
        for r, c in zip(rows[:20], cols[:20]):
            print(f"    ({r}, {c}): χ={asymmetry[r,c]:.3f}, "
                  f"centroid={centroid[r,c]:.3f} µm")

    # ---- Plot: scatter (Brown Fig 8b/9f style) ----
    fig, ax = plt.subplots(figsize=(10, 8))

    # Simple scatter — small dots, semi-transparent
    ax.scatter(cx, ay, s=0.3, c='black', alpha=0.15, edgecolors='none',
               rasterized=True)

    # Threshold line at 1.15 µm
    ax.axvline(1.15, color='grey', ls='-', lw=0.8, alpha=0.5)

    # Axis labels matching Brown
    ax.set_xlabel('1 µm band Centroid (µm)', fontsize=14)
    ax.set_ylabel('Asymmetry χ', fontsize=14)

    # Axis limits matching Brown Fig 9f
    ax.set_xlim(0.9, 1.55)
    ax.set_ylim(0, 7.0)

    # Grid
    ax.grid(True, alpha=0.3, lw=0.5)

    # Observation label
    ax.text(0.95, 0.05, obs_name, transform=ax.transAxes,
            fontsize=16, fontweight='bold', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='grey', alpha=0.8))

    plt.tight_layout()

    # Save next to the centroid file
    out_dir = os.path.dirname(centroid_path)
    out_png = os.path.join(out_dir,
                           os.path.basename(centroid_path)
                           .replace('centroid_map', 'centroid_vs_asymmetry')
                           .replace('.tif', '.png'))
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    print(f"Saved: {out_png}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        plot_centroid_vs_asymmetry(sys.argv[1], sys.argv[2])
    else:
        plot_centroid_vs_asymmetry(DEFAULT_CENTROID, DEFAULT_ASYMMETRY)