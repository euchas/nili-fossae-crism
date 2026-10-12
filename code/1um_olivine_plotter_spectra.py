"""
Ratioed Spectra Comparison — Intrusions vs Ol-Carb vs Olivine
--------------------------------------------------------------
Three-panel plot showing ratioed spectra per class, one ROI per line.
Each panel shows all ROIs of one class, overlaid, with mean ± 1σ envelope.

Reads *_analysis_*.csv files containing ratioed spectra columns.

Author: AV Valantinas
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import BROWN_ROI_DIR, STYLE_PATH

# =============================================================================
# CONFIG
# =============================================================================

WORKING_DIR = str(BROWN_ROI_DIR)

style_path = str(STYLE_PATH)
if os.path.exists(style_path):
    plt.style.use(style_path)

# Classes to plot (one panel each)
CLASSES = ['intrusions', 'ol-carb', 'olivine', 'ridge']

# Matching colors from the dashboard
CLASS_COLORS = {
    'olivine':    '#1b9e77',
    'ol-carb':    '#d95f02',
    'intrusions': '#e7298a',
    'ridge':      '#7570b3',
}

# Cube line styles
CUBE_LINESTYLES = {
    'frt00003e12': '-',
    'frt0000a4fc': '--',
    'frt0000b438': ':',
}

# Wavelength range to plot (S + L detectors, excluding noisy >2.6 µm)
WMIN, WMAX = 0.4, 2.6

# Reference vertical lines (diagnostic wavelengths)
REF_LINES = {
    '1 µm band (olivine/pyx)': 1.05,
    'LCP/HCP center': 1.90,
    'phyllosilicate 2.3 µm': 2.30,
    'carbonate 2.50 µm': 2.50,
}


# =============================================================================
# DATA LOADING
# =============================================================================

def parse_filename(path):
    base = os.path.basename(path).replace('.csv', '')
    parts = base.split('_analysis_')
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]


def load_class_spectra(working_dir, target_class):
    """Load all ratioed spectra for a given class across all cubes.

    Returns: list of dicts with keys 'cube', 'roi', 'wavelength', 'ratio'.
    """
    pattern = os.path.join(working_dir, f'*_analysis_{target_class}.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  No files for class '{target_class}'")
        return []

    all_spectra = []
    for f in files:
        cube, cls = parse_filename(f)
        df = pd.read_csv(f)
        wav = df['wavelength_um'].values

        # Find all ROI ratio columns
        ratio_cols = [c for c in df.columns if c.endswith('_ratio')]
        for col in ratio_cols:
            roi_name = col.replace('_ratio', '')
            all_spectra.append({
                'cube': cube,
                'class': cls,
                'roi': roi_name,
                'wavelength': wav,
                'ratio': df[col].values,
                'file': os.path.basename(f),
            })
        print(f"  {os.path.basename(f):50s} | {len(ratio_cols)} ROIs")

    return all_spectra


# =============================================================================
# PLOTTING
# =============================================================================

def plot_class_panel(ax, spectra, cls, show_mean=True, offset_per_cube=False):
    """Plot all ratioed spectra for one class on one panel."""
    color = CLASS_COLORS.get(cls, '#333')

    # Collect all ratios for mean/std computation (common wavelength grid)
    if not spectra:
        ax.text(0.5, 0.5, f'No data for {cls}',
                ha='center', va='center', transform=ax.transAxes)
        return

    # All spectra share the same wavelength grid (same instrument + TER3 pipeline)
    wav = spectra[0]['wavelength']
    ratio_stack = np.array([s['ratio'] for s in spectra])  # [n_roi, n_bands]

    # Individual ROI lines — thin, lower alpha
    for s in spectra:
        ls = CUBE_LINESTYLES.get(s['cube'], '-')
        ax.plot(s['wavelength'], s['ratio'],
                color=color, linewidth=0.7, alpha=0.45,
                linestyle=ls, zorder=2)

    # Mean ± 1σ
    if show_mean:
        with np.errstate(invalid='ignore'):
            mean = np.nanmean(ratio_stack, axis=0)
            std = np.nanstd(ratio_stack, axis=0)
        ax.plot(wav, mean, color=color, linewidth=2.2,
                label=f'Mean (n={len(spectra)})', zorder=4)
        ax.fill_between(wav, mean - std, mean + std,
                        color=color, alpha=0.18, zorder=1,
                        label='±1σ')

    # Reference vertical lines
    for label, w in REF_LINES.items():
        if WMIN <= w <= WMAX:
            ax.axvline(w, color='gray', linestyle=':',
                       linewidth=0.6, alpha=0.4, zorder=0)

    # Horizontal at ratio=1
    ax.axhline(1.0, color='black', linestyle='-',
               linewidth=0.5, alpha=0.4, zorder=0)

    ax.set_xlim(WMIN, WMAX)
    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Ratio (sample / dust)')
    ax.set_title(f'{cls.upper()}  ({len(spectra)} ROIs, '
                 f'{len(set(s["cube"] for s in spectra))} cubes)',
                 fontsize=10, color=color, fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc='best')


def annotate_bands(ax_top):
    """Annotate the top panel with band labels for reference."""
    band_labels = [
        (1.05, '1 µm\n(ol/pyx)', 'top'),
        (1.90, 'H₂O', 'top'),
        (2.30, 'Fe/Mg-OH\nCO₃', 'top'),
        (2.50, 'CO₃', 'top'),
    ]
    ymin, ymax = ax_top.get_ylim()
    y_text = ymax - 0.02 * (ymax - ymin)
    for w, label, va in band_labels:
        if WMIN <= w <= WMAX:
            ax_top.text(w, y_text, label, fontsize=7, color='gray',
                        ha='center', va=va, alpha=0.8,
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor='white', edgecolor='none',
                                  alpha=0.6))


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"\nLoading spectra from: {WORKING_DIR}\n")

    all_class_spectra = {}
    for cls in CLASSES:
        print(f"Class: {cls}")
        specs = load_class_spectra(WORKING_DIR, cls)
        all_class_spectra[cls] = specs
        print(f"  → {len(specs)} total ROIs\n")

    # Build figure: 1 row, 4 columns, shared x and y
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5),
                             sharex=True, sharey=True)

    for ax, cls in zip(axes, CLASSES):
        plot_class_panel(ax, all_class_spectra[cls], cls)

    # Fixed y-range
    axes[0].set_ylim(0.8, 1.6)

    # Annotate bands on leftmost panel only
    annotate_bands(axes[0])

    # Only leftmost panel gets y-label
    for ax in axes[1:]:
        ax.set_ylabel('')

    fig.suptitle('Ratioed spectra by class — line per ROI, thick line = class mean',
                 fontsize=12, y=0.99)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    out_path = os.path.join(WORKING_DIR, 'spectra_3class_comparison.png')
    try:
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"  Saved: {out_path}")
    except Exception as e:
        print(f"  Could not save: {e}")

    plt.show()


if __name__ == "__main__":
    main()