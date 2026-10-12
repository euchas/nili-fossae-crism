"""
Brown Parameter Exploration Dashboard
--------------------------------------
First-pass diagnostic plots across all cubes and classes.

Reads all *_brown_params_*.csv files in WORKING_DIR, one point per ROI.
Four panels:
  1. Centroid vs Asymmetry  (Brown 2020 Fig 5 analog)
  2. Centroid vs HWHM       (grain size vs composition)
  3. Amplitude vs Asymmetry (band strength vs shape)
  4. Centroid KDE per class (marginal distributions)

Symbol shape = cube ID    |    Color = class (lithology)

Author: AV Valantinas
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import BROWN_ROI_DIR, STYLE_PATH
from matplotlib.lines import Line2D

# =============================================================================
# USER CONFIGURATION
# =============================================================================

WORKING_DIR = str(BROWN_ROI_DIR)
# If your CSVs are spread across subdirectories, uncomment:
# RECURSIVE = True
RECURSIVE = True

style_path = str(STYLE_PATH)
if os.path.exists(style_path):
    plt.style.use(style_path)

# Class color scheme (lithology-meaningful)
CLASS_COLORS = {
    'olivine':    '#1b9e77',  # green
    'ol-carb':    '#d95f02',  # orange
    'ridge':      '#7570b3',  # purple
    'intrusions': '#e7298a',  # magenta
    'clays':      '#66a61e',  # olive green
    'nontronite': '#e6ab02',  # yellow-ochre
}

# Cube symbol scheme
CUBE_MARKERS = {
    'frt00003e12': 'o',
    'frt0000a4fc': 's',
    'frt0000b438': '^',
}

# Brown 2020 interpretive regions
POPULATION_ZERO_CENTROID = 1.20   # Brown's redshifted olivine threshold
BROWN_THRESHOLD_CENTROID = 1.05   # Brown's olivine mask lower bound


# =============================================================================
# DATA LOADING
# =============================================================================

def parse_filename(path):
    """Extract cube_id and class from filename like
    'frt00003e12_brown_params_olivine.csv'."""
    base = os.path.basename(path).replace('.csv', '')
    parts = base.split('_brown_params_')
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]


def load_all_brown(working_dir, recursive=True):
    """Find and concatenate all *_brown_params_*.csv files."""
    pattern = '**/*_brown_params_*.csv' if recursive else '*_brown_params_*.csv'
    files = sorted(glob.glob(os.path.join(working_dir, pattern),
                             recursive=recursive))

    if not files:
        raise FileNotFoundError(
            f"No *_brown_params_*.csv files in {working_dir}")

    rows = []
    for f in files:
        cube, cls = parse_filename(f)
        if cube is None:
            print(f"  Skipping (unparseable): {os.path.basename(f)}")
            continue
        df = pd.read_csv(f)
        df['cube'] = cube
        df['class'] = cls
        df['source_file'] = os.path.basename(f)
        rows.append(df)
        print(f"  {os.path.basename(f):50s} | {len(df):3d} ROIs")

    full = pd.concat(rows, ignore_index=True)
    print(f"\n  Total: {len(full)} ROIs across "
          f"{full['cube'].nunique()} cubes, "
          f"{full['class'].nunique()} classes")
    return full


# =============================================================================
# PLOTTING
# =============================================================================

def get_style(cube, cls):
    """Return (color, marker) for a given cube/class combination."""
    color = CLASS_COLORS.get(cls, '#333333')
    marker = CUBE_MARKERS.get(cube, 'D')
    return color, marker


def scatter_with_errorbars(ax, df, xcol, ycol,
                           xstdcol=None, ystdcol=None,
                           point_size=140, alpha=0.85):
    """Scatter ROI means with optional error bars from std columns."""
    for _, row in df.iterrows():
        color, marker = get_style(row['cube'], row['class'])
        x, y = row[xcol], row[ycol]
        if np.isnan(x) or np.isnan(y):
            continue

        # Error bars if std columns provided
        if xstdcol and ystdcol and \
           np.isfinite(row[xstdcol]) and np.isfinite(row[ystdcol]):
            ax.errorbar(x, y,
                        xerr=row[xstdcol], yerr=row[ystdcol],
                        fmt='none', ecolor=color, elinewidth=0.8,
                        alpha=0.35, capsize=0, zorder=1)

        ax.scatter(x, y, s=point_size, c=color, marker=marker,
                   edgecolors='black', linewidths=0.6,
                   alpha=alpha, zorder=3)


def annotate_brown_regions(ax, xaxis='centroid', yaxis='asymmetry'):
    """Shade Brown 2020 interpretive regions."""
    if xaxis == 'centroid':
        # Brown mask threshold (vertical dashed line)
        ax.axvline(BROWN_THRESHOLD_CENTROID, color='gray',
                   linestyle=':', linewidth=0.8, alpha=0.6, zorder=0)
        ax.text(BROWN_THRESHOLD_CENTROID + 0.002, ax.get_ylim()[1],
                'Brown threshold', rotation=90, fontsize=7,
                color='gray', va='top', ha='left', alpha=0.7)

        # Population zero shaded region
        xlim = ax.get_xlim()
        if xlim[1] > POPULATION_ZERO_CENTROID:
            ylim = ax.get_ylim()
            ax.axvspan(POPULATION_ZERO_CENTROID, xlim[1],
                       alpha=0.06, color='red', zorder=0)
            ax.text(POPULATION_ZERO_CENTROID + 0.005, ylim[0] + 0.02*(ylim[1]-ylim[0]),
                    'Population zero\n(redshifted)', fontsize=7,
                    color='darkred', alpha=0.7, ha='left', va='bottom')


def build_legend(ax, df):
    """Custom legend: class colors + cube markers, separate groups."""
    classes_present = sorted(df['class'].unique())
    cubes_present = sorted(df['cube'].unique())

    class_handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=CLASS_COLORS.get(c, '#333'),
               markeredgecolor='black', markeredgewidth=0.5,
               markersize=10, label=c)
        for c in classes_present
    ]
    cube_handles = [
        Line2D([0], [0], marker=CUBE_MARKERS.get(c, 'D'),
               color='w', markerfacecolor='lightgray',
               markeredgecolor='black', markeredgewidth=0.5,
               markersize=10, label=c.upper())
        for c in cubes_present
    ]

    leg1 = ax.legend(handles=class_handles, title='Class',
                     loc='upper left', bbox_to_anchor=(1.02, 1.0),
                     fontsize=8, title_fontsize=9, frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=cube_handles, title='Cube',
              loc='upper left', bbox_to_anchor=(1.02, 0.55),
              fontsize=8, title_fontsize=9, frameon=True)


# =============================================================================
# PANELS
# =============================================================================

def panel_centroid_asymmetry(ax, df):
    scatter_with_errorbars(ax, df,
                           'centroid_mean', 'asymmetry_mean',
                           'centroid_std', 'asymmetry_std')
    ax.set_xlabel('1 µm band centroid (µm)')
    ax.set_ylabel('Asymmetry')
    ax.set_title('Centroid vs Asymmetry — Brown 2020 Fig. 5 analog',
                 fontsize=10)
    ax.grid(True, alpha=0.3)
    annotate_brown_regions(ax, 'centroid', 'asymmetry')


def panel_centroid_hwhm(ax, df):
    scatter_with_errorbars(ax, df,
                           'centroid_mean', 'hwhm_mean',
                           'centroid_std', 'hwhm_std')
    ax.set_xlabel('1 µm band centroid (µm)')
    ax.set_ylabel('HWHM (µm)')
    ax.set_title('Centroid vs HWHM — grain size sensitivity',
                 fontsize=10)
    ax.grid(True, alpha=0.3)


def panel_amplitude_asymmetry(ax, df):
    scatter_with_errorbars(ax, df,
                           'amplitude_mean', 'asymmetry_mean',
                           'amplitude_std', 'asymmetry_std')
    ax.set_xlabel('Amplitude (band depth)')
    ax.set_ylabel('Asymmetry')
    ax.set_title('Amplitude vs Asymmetry — saturated bands',
                 fontsize=10)
    ax.grid(True, alpha=0.3)


def panel_centroid_kde(ax, df):
    """KDE of ROI centroid means per class."""
    try:
        from scipy.stats import gaussian_kde
        has_scipy = True
    except ImportError:
        has_scipy = False

    classes_present = sorted(df['class'].unique())
    x_grid = np.linspace(df['centroid_mean'].min() - 0.02,
                         df['centroid_mean'].max() + 0.02, 300)

    for cls in classes_present:
        sub = df[df['class'] == cls]
        vals = sub['centroid_mean'].dropna().values
        if len(vals) == 0:
            continue
        color = CLASS_COLORS.get(cls, '#333')

        # KDE if enough points, else just rug + hist
        if has_scipy and len(vals) >= 3:
            kde = gaussian_kde(vals, bw_method=0.3)
            ax.plot(x_grid, kde(x_grid), color=color, linewidth=2,
                    label=f'{cls} (n={len(vals)})')
            ax.fill_between(x_grid, 0, kde(x_grid), color=color, alpha=0.15)
        else:
            ax.hist(vals, bins=8, color=color, alpha=0.4, density=True,
                    label=f'{cls} (n={len(vals)})')

        # Rug plot at bottom
        ax.scatter(vals, np.full_like(vals, -0.2), marker='|',
                   color=color, s=80, linewidth=1.5, alpha=0.8)

    ax.axvline(BROWN_THRESHOLD_CENTROID, color='gray',
               linestyle=':', linewidth=0.8, alpha=0.6)
    ax.axvline(POPULATION_ZERO_CENTROID, color='darkred',
               linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('1 µm band centroid (µm)')
    ax.set_ylabel('Density')
    ax.set_title('Centroid distribution per class', fontsize=10)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def print_summary(df):
    """Per-class, per-cube summary stats."""
    print("\n" + "=" * 88)
    print("  SUMMARY: ROI-mean Brown parameters, median (IQR) per class × cube")
    print("=" * 88)

    for cls in sorted(df['class'].unique()):
        sub = df[df['class'] == cls]
        print(f"\n  Class: {cls.upper()}")
        for cube in sorted(sub['cube'].unique()):
            cube_sub = sub[sub['cube'] == cube]
            n = len(cube_sub)
            c_med = cube_sub['centroid_mean'].median()
            c_iqr = (cube_sub['centroid_mean'].quantile(0.75) -
                     cube_sub['centroid_mean'].quantile(0.25))
            a_med = cube_sub['asymmetry_mean'].median()
            a_iqr = (cube_sub['asymmetry_mean'].quantile(0.75) -
                     cube_sub['asymmetry_mean'].quantile(0.25))
            h_med = cube_sub['hwhm_mean'].median()
            print(f"    {cube.upper()}  n={n:2d}  "
                  f"centroid={c_med:.3f} (IQR {c_iqr:.3f})  "
                  f"asym={a_med:.3f} (IQR {a_iqr:.3f})  "
                  f"hwhm={h_med:.3f}")
    print("=" * 88 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"\nScanning: {WORKING_DIR}\n")
    df = load_all_brown(WORKING_DIR, recursive=RECURSIVE)

    print_summary(df)

    # Build dashboard
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    panel_centroid_asymmetry(axes[0, 0], df)
    panel_centroid_hwhm(axes[0, 1], df)
    panel_amplitude_asymmetry(axes[1, 0], df)
    panel_centroid_kde(axes[1, 1], df)

    # Legend on the first panel
    build_legend(axes[0, 0], df)

    fig.suptitle(
        f'Brown Parameter Exploration — '
        f'{df["cube"].nunique()} cubes, {df["class"].nunique()} classes, '
        f'{len(df)} ROIs (one point per ROI)',
        fontsize=12, y=0.995)

    plt.tight_layout(rect=[0, 0, 0.88, 0.97])

    # Save alongside the CSVs
    out_path = os.path.join(WORKING_DIR, 'brown_explore_dashboard.png')
    try:
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"  Saved: {out_path}")
    except Exception as e:
        print(f"  Could not save to {out_path}: {e}")

    plt.show()


if __name__ == "__main__":
    main()