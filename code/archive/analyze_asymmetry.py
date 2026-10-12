"""
analyze_asymmetry.py

Quick diagnostic: load an asymmetry GeoTIFF and show the distribution.
Usage:
    python analyze_asymmetry.py path/to/asymmetry_map_v4.tif
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def analyze(tif_path: str):
    # Try rasterio first, then GDAL
    data = None
    try:
        import rasterio
        with rasterio.open(tif_path) as src:
            data = src.read(1)
    except ImportError:
        from osgeo import gdal
        ds = gdal.Open(tif_path)
        data = ds.GetRasterBand(1).ReadAsArray()
        ds = None

    # Get valid (non-NaN) pixels
    valid = data[np.isfinite(data)]
    n_total = data.size
    n_valid = len(valid)
    n_nan = n_total - n_valid

    print(f"File: {os.path.basename(tif_path)}")
    print(f"Shape: {data.shape}")
    print(f"Total pixels: {n_total:,}")
    print(f"Valid pixels: {n_valid:,} ({100*n_valid/n_total:.1f}%)")
    print(f"NaN pixels:   {n_nan:,}")
    print()

    if n_valid == 0:
        print("No valid pixels found.")
        return

    # Basic stats
    print("--- Distribution ---")
    print(f"Min:    {np.min(valid):.4f}")
    print(f"Max:    {np.max(valid):.4f}")
    print(f"Mean:   {np.mean(valid):.4f}")
    print(f"Median: {np.median(valid):.4f}")
    print(f"Std:    {np.std(valid):.4f}")
    print()

    # Percentiles
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("--- Percentiles ---")
    for p in pcts:
        print(f"  {p:3d}%:  {np.percentile(valid, p):.4f}")
    print()

    # Bin counts for key ranges (Brown threshold: 0.8 < χ < 9.0)
    bins = [
        ("χ < 0",       valid < 0),
        ("0 ≤ χ < 0.5", (valid >= 0) & (valid < 0.5)),
        ("0.5 ≤ χ < 0.8", (valid >= 0.5) & (valid < 0.8)),
        ("0.8 ≤ χ < 1.0", (valid >= 0.8) & (valid < 1.0)),
        ("1.0 ≤ χ < 1.5", (valid >= 1.0) & (valid < 1.5)),
        ("1.5 ≤ χ < 2.0", (valid >= 1.5) & (valid < 2.0)),
        ("2.0 ≤ χ < 3.0", (valid >= 2.0) & (valid < 3.0)),
        ("3.0 ≤ χ < 5.0", (valid >= 3.0) & (valid < 5.0)),
        ("5.0 ≤ χ < 9.0", (valid >= 5.0) & (valid < 9.0)),
        ("χ ≥ 9.0",     valid >= 9.0),
    ]

    print("--- Bin counts ---")
    for label, mask in bins:
        n = np.sum(mask)
        pct = 100 * n / n_valid
        print(f"  {label:18s}: {n:8,}  ({pct:5.1f}%)")
    print()

    # Brown threshold pass rate
    brown_pass = (valid > 0.8) & (valid < 9.0)
    print(f"Brown Eq.2 asymmetry pass (0.8 < χ < 9): "
          f"{np.sum(brown_pass):,} ({100*np.sum(brown_pass)/n_valid:.1f}%)")
    print()

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Asymmetry diagnostic: {os.path.basename(tif_path)}", 
                 fontsize=13, fontweight='bold')

    # 1. Full histogram
    ax = axes[0, 0]
    ax.hist(valid, bins=200, color='steelblue', edgecolor='none', alpha=0.8)
    ax.set_xlabel('Asymmetry χ')
    ax.set_ylabel('Pixel count')
    ax.set_title('Full distribution')
    ax.axvline(0.8, color='red', ls='--', lw=1, label='Brown min (0.8)')
    ax.axvline(9.0, color='red', ls='--', lw=1, label='Brown max (9.0)')
    ax.legend(fontsize=8)

    # 2. Zoomed histogram (0–10 range)
    ax = axes[0, 1]
    clipped = valid[(valid > 0) & (valid < 10)]
    if len(clipped) > 0:
        ax.hist(clipped, bins=200, color='steelblue', edgecolor='none', alpha=0.8)
    ax.set_xlabel('Asymmetry χ')
    ax.set_ylabel('Pixel count')
    ax.set_title('Zoomed: 0 < χ < 10')
    ax.axvline(0.8, color='red', ls='--', lw=1)
    ax.axvline(9.0, color='red', ls='--', lw=1)

    # 3. Log-scale histogram
    ax = axes[1, 0]
    ax.hist(valid, bins=200, color='steelblue', edgecolor='none', alpha=0.8)
    ax.set_yscale('log')
    ax.set_xlabel('Asymmetry χ')
    ax.set_ylabel('Pixel count (log)')
    ax.set_title('Full distribution (log scale)')
    ax.axvline(0.8, color='red', ls='--', lw=1)
    ax.axvline(9.0, color='red', ls='--', lw=1)

    # 4. Spatial map
    ax = axes[1, 1]
    im = ax.imshow(data, cmap='magma', vmin=0.8, vmax=9.0,
                   interpolation='nearest', origin='upper')
    plt.colorbar(im, ax=ax, shrink=0.75, label='Asymmetry χ')
    ax.set_title('Spatial map (0.8–9.0 range)')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    plt.tight_layout()

    out_png = tif_path.replace('.tif', '_diagnostic.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_png}")
    plt.show()


if __name__ == "__main__":
    from config import CUBES_DIR
    DEFAULT_PATH = str(CUBES_DIR / "frt00003e12_07_if166j_mtr3" / "frt00003e12_07_if166j_mtr3_asymmetry_map_v4.tif")

    if len(sys.argv) >= 2:
        analyze(sys.argv[1])
    else:
        analyze(DEFAULT_PATH)