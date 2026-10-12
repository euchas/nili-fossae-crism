"""
PC-Cluster Spectral Extraction
-------------------------------
Reads the per-pixel PCA output + Brown .npy maps + TER3 cube.
Lets the user draw rectangles on the PC1-PC2 density plot to define classes.
For each class, pulls raw I/F spectra from the cube, computes mean ± std.
Then lets the user pick a class as ratio denominator and plots the
ratioed spectra for all other classes.

WORKFLOW:
  1. Reload TER3 cube + Brown .npy maps.
  2. Recompute PCA (same as pixel_pca.py) to get per-pixel PC1, PC2.
  3. PC1-PC2 density plot appears with spatial preview panel.
     - Draw rectangle on density plot → defines a class.
     - Press '1' through '9' to label the rectangle before drawing the next.
     - 'd' deletes last class, 'r' resets, Enter finishes selection.
  4. Plot raw mean I/F per class.
  5. Prompt for denominator class (by number).
  6. Plot ratioed spectra (all_classes / denominator_class).
  7. Save results to CSVs and PNGs.

Keys during rectangle selection:
  Draw rectangle with mouse on the density panel (top left)
  d  : delete last class
  r  : reset all classes
  s  : save current classes and proceed
  Enter (in terminal) : finalize

Author: AV Valantinas
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import BROWN_ROI_DIR, CUBES_DIR, STYLE_PATH
from matplotlib.widgets import RectangleSelector
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

warnings.filterwarnings('ignore')

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except ImportError:
    raise ImportError("Install scikit-learn:  pip install scikit-learn")


# =============================================================================
# USER CONFIGURATION (same as pixel_pca.py)
# =============================================================================

CUBE_DIR = str(CUBES_DIR / "FRT00003E12_07_IF166J_TER3")
BROWN_VERSION = "v5"
ROI_CSV_DIR = str(BROWN_ROI_DIR)
CUBE_ID = "frt00003e12"

FLIP_VERTICAL = True
FLIP_HORIZONTAL = True

OUT_DIR = ROI_CSV_DIR  # where plots + CSVs go

style_path = str(STYLE_PATH)
if os.path.exists(style_path):
    plt.style.use(style_path)

# Colors for classes (extendable)
CLASS_COLOR_CYCLE = [
    '#1b9e77', '#d95f02', '#7570b3', '#e7298a',
    '#66a61e', '#e6ab02', '#a6761d', '#666666', '#1f78b4',
]


# =============================================================================
# CUBE LOADING + PARAMETER CALCULATIONS
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
    hdr = img_file.replace('.img', '.hdr').replace('.IMG', '.hdr')
    if not os.path.exists(hdr):
        hdr = img_file + '.hdr'
    meta = parse_envi_header(hdr)
    s = int(meta.get('samples', 0))
    l = int(meta.get('lines', 0))
    b = int(meta.get('bands', 0))
    dt = int(meta.get('data type', 4))
    order = int(meta.get('byte order', 0))
    inter = meta.get('interleave', 'bsq').lower()

    if meta.get('wavelength'):
        wav = np.array([float(w) for w in meta['wavelength']])
        if wav.max() > 100:
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

    return data, wav


def get_kernel(data, wav, target, width):
    idx = np.abs(wav - target).argmin()
    if width <= 1:
        return data[idx, :, :]
    hw = width // 2
    return np.nanmedian(
        data[max(0, idx - hw):min(len(wav), idx + hw + 1), :, :], axis=0)


def _ratio(num, den):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(den != 0, 1.0 - (num / den), 0.0)
    return np.nan_to_num(result, nan=np.nan)


def _rb(R, RC):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(RC != 0, (RC - R) / RC, 0.0)
    return np.nan_to_num(result, nan=np.nan)


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
    return np.nan_to_num(result, nan=np.nan)


def calc_bd1900r2(data, wav):
    num_w = [1.908, 1.914, 1.921, 1.928, 1.934, 1.941]
    den_w = [1.862, 1.869, 1.875, 2.112, 2.120, 2.126]
    R1850 = get_kernel(data, wav, 1.850, 5); R2060 = get_kernel(data, wav, 2.060, 5)
    slope = np.where((R2060 != R1850), (R2060 - R1850) / (2.060 - 1.850), 0.0)
    def rc(w): return R1850 + slope * (w - 1.850)
    with np.errstate(divide='ignore', invalid='ignore'):
        num_sum = sum(np.where(rc(w) != 0, get_kernel(data, wav, w, 1) / rc(w), 0.0)
                      for w in num_w)
        den_sum = sum(np.where(rc(w) != 0, get_kernel(data, wav, w, 1) / rc(w), 0.0)
                      for w in den_w)
        result = np.where(den_sum != 0, 1.0 - num_sum / den_sum, 0.0)
    return np.nan_to_num(result, nan=np.nan)


def calc_bd2500_2(data, wav):
    R2364 = get_kernel(data, wav, 2.364, 5)
    R2480 = get_kernel(data, wav, 2.480, 5)
    R2570 = get_kernel(data, wav, 2.570, 5)
    b = (2.480 - 2.364) / (2.570 - 2.364); a = 1.0 - b
    return _ratio(R2480, a * R2364 + b * R2570)


# =============================================================================
# LOADING
# =============================================================================

def find_cube_and_brown_maps(cube_dir, version):
    patterns = ['*_if*ter3*.img', '*_IF*TER3*.img', '*_if*TER3*.img']
    img_file = None
    for pat in patterns:
        matches = glob.glob(os.path.join(cube_dir, pat))
        if matches:
            img_file = matches[0]; break
    if img_file is None:
        raise FileNotFoundError(f"No TER3 cube in {cube_dir}")

    base = os.path.splitext(os.path.basename(img_file))[0]
    brown = {}
    for param in ['centroid', 'asymmetry', 'amplitude', 'hwhm', 'mask']:
        path = os.path.join(cube_dir, f'{base}_{param}_map_{version}.npy')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing Brown map: {path}")
        brown[param] = np.load(path)
    return img_file, brown


PARAM_ORDER = ['centroid', 'asymmetry', 'amplitude', 'hwhm',
               'OLINDEX3', 'D2300', 'BD1900r2', 'BD2500_2']


def build_pca_pixels(cube_dir, version):
    """Returns (df_pca, shape, cube_data, wav).

    df_pca: DataFrame with line, sample, all 8 params, and PC1..PC8.
    cube_data kept for raw spectrum extraction.
    """
    img_file, brown = find_cube_and_brown_maps(cube_dir, version)

    print("  Reading cube...")
    data, wav = read_ter3_cube(img_file)
    print(f"  Cube shape: {data.shape}")

    print("  Computing VB parameters...")
    olindex3 = calc_olindex3(data, wav)
    d2300    = calc_d2300(data, wav)
    bd1900r2 = calc_bd1900r2(data, wav)
    bd2500_2 = calc_bd2500_2(data, wav)

    finite_vb = (np.isfinite(olindex3) & np.isfinite(d2300) &
                 np.isfinite(bd1900r2) & np.isfinite(bd2500_2))
    finite_brown = (np.isfinite(brown['centroid']) & np.isfinite(brown['asymmetry']) &
                    np.isfinite(brown['amplitude']) & np.isfinite(brown['hwhm']))
    valid = finite_vb & finite_brown
    yy, xx = np.where(valid)
    n_lines, n_samples = brown['centroid'].shape

    df = pd.DataFrame({
        'line': yy.astype(np.int32), 'sample': xx.astype(np.int32),
        'centroid':   brown['centroid'][yy, xx],
        'asymmetry':  brown['asymmetry'][yy, xx],
        'amplitude':  brown['amplitude'][yy, xx],
        'hwhm':       brown['hwhm'][yy, xx],
        'OLINDEX3':   olindex3[yy, xx],
        'D2300':      d2300[yy, xx],
        'BD1900r2':   bd1900r2[yy, xx],
        'BD2500_2':   bd2500_2[yy, xx],
    })

    # PCA
    X = df[PARAM_ORDER].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=len(PARAM_ORDER))
    Xp = pca.fit_transform(Xs)
    for i in range(Xp.shape[1]):
        df[f'PC{i+1}'] = Xp[:, i]

    print(f"  PCA: {len(df):,} pixels. "
          f"PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
          f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%, "
          f"PC3={pca.explained_variance_ratio_[2]*100:.1f}%")

    return df, (n_lines, n_samples), data, wav


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def apply_display_transform(image):
    result = image.copy()
    if FLIP_VERTICAL:  result = np.flipud(result)
    if FLIP_HORIZONTAL: result = np.fliplr(result)
    return result


def data_to_display_coords(line, sample, shape):
    """Convert data (line, sample) → display coords for plotting overlays."""
    n_lines, n_samples = shape
    dx, dy = sample, line
    if FLIP_HORIZONTAL: dx = n_samples - 1 - dx
    if FLIP_VERTICAL:   dy = n_lines - 1 - dy
    return dx, dy


# =============================================================================
# INTERACTIVE RECTANGLE SELECTION
# =============================================================================

class PCRectangleSelector:
    """Lets user draw rectangles on the PC1-PC2 density plot.
    Classes auto-named class_1..class_N. Rename later by editing CSV or
    by the class name text box below the plot."""

    def __init__(self, df_pca, shape, wav):
        self.df = df_pca
        self.shape = shape
        self.wav = wav
        self.classes = []   # list of dicts

        self._build_figure()

    def _build_figure(self):
        self.fig, (self.ax_pc, self.ax_sp) = plt.subplots(
            1, 2, figsize=(15, 7), constrained_layout=True)

        # Density plot
        self.hb = self.ax_pc.hexbin(
            self.df['PC1'], self.df['PC2'], gridsize=80,
            cmap='viridis', mincnt=1, norm=LogNorm())
        plt.colorbar(self.hb, ax=self.ax_pc, shrink=0.75, label='pixels (log)')
        self.ax_pc.set_xlabel('PC1')
        self.ax_pc.set_ylabel('PC2')
        self.ax_pc.axhline(0, color='gray', ls=':', lw=0.5, alpha=0.5)
        self.ax_pc.axvline(0, color='gray', ls=':', lw=0.5, alpha=0.5)
        # Lock axis limits so the RectangleSelector box can't size to full axes
        self._pc1_lim = self.ax_pc.get_xlim()
        self._pc2_lim = self.ax_pc.get_ylim()

        # Spatial preview background
        xs_all, ys_all = self._data_xy_to_display(self.df['line'].values,
                                                   self.df['sample'].values)
        self.ax_sp.scatter(xs_all, ys_all, s=0.3, color='lightgray', alpha=0.3)
        self.ax_sp.set_aspect('equal')
        self.ax_sp.invert_yaxis()
        self.ax_sp.set_xticks([]); self.ax_sp.set_yticks([])

        # Create the RectangleSelector (interactive=True allows drag to resize
        # before release, and is more robust than interactive=False)
        self._make_rect_selector()

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._refresh_status()

    def _make_rect_selector(self):
        self.rect_selector = RectangleSelector(
            self.ax_pc, self._on_rectangle,
            useblit=True, button=[1],
            minspanx=0.2, minspany=0.2,  # require meaningful drag
            spancoords='data', interactive=True,
            props=dict(facecolor='none', edgecolor='red',
                       alpha=0.8, linewidth=2))

    def _data_xy_to_display(self, lines, samples):
        n_lines, n_samples = self.shape
        xs = samples.copy().astype(np.int64)
        ys = lines.copy().astype(np.int64)
        if FLIP_HORIZONTAL:
            xs = n_samples - 1 - xs
        if FLIP_VERTICAL:
            ys = n_lines - 1 - ys
        return xs, ys

    def _on_rectangle(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if None in (x1, y1, x2, y2):
            return
        pc1_min, pc1_max = sorted([x1, x2])
        pc2_min, pc2_max = sorted([y1, y2])

        # Defensive check: reject box that spans >80% of either axis
        # (likely accidental full-axis rubberband)
        span1 = pc1_max - pc1_min
        span2 = pc2_max - pc2_min
        full1 = self._pc1_lim[1] - self._pc1_lim[0]
        full2 = self._pc2_lim[1] - self._pc2_lim[0]
        if span1 / full1 > 0.8 and span2 / full2 > 0.8:
            print("  Rectangle spans the whole plot — ignored. "
                  "Draw a tighter box.")
            return

        mask = ((self.df['PC1'] >= pc1_min) & (self.df['PC1'] <= pc1_max) &
                (self.df['PC2'] >= pc2_min) & (self.df['PC2'] <= pc2_max))
        n_pixels = int(mask.sum())

        if n_pixels < 5:
            print(f"  Only {n_pixels} pixels in box — skipped.")
            return

        # Auto-name: class_1, class_2, ...
        name = f"class_{len(self.classes) + 1}"
        color = CLASS_COLOR_CYCLE[len(self.classes) % len(CLASS_COLOR_CYCLE)]

        # Draw rectangle patch + label on PC plot
        rect_patch = Rectangle((pc1_min, pc2_min),
                               pc1_max - pc1_min, pc2_max - pc2_min,
                               facecolor='none', edgecolor=color,
                               linewidth=2, alpha=0.9, zorder=5)
        self.ax_pc.add_patch(rect_patch)
        text_art = self.ax_pc.text(
            (pc1_min + pc1_max) / 2, pc2_max + 0.05, name,
            color=color, fontweight='bold', fontsize=10,
            ha='center', va='bottom',
            bbox=dict(facecolor='white', alpha=0.85,
                      edgecolor=color, pad=2))

        # Overlay pixels on spatial panel
        sub = self.df[mask]
        xs, ys = self._data_xy_to_display(sub['line'].values, sub['sample'].values)
        sp_art = self.ax_sp.scatter(xs, ys, s=3, color=color,
                                     alpha=0.65,
                                     label=f'{name} (n={n_pixels})')
        self.ax_sp.legend(fontsize=8, loc='upper right',
                          markerscale=3, framealpha=0.9)

        self.classes.append({
            'name': name, 'color': color,
            'pc1_min': pc1_min, 'pc1_max': pc1_max,
            'pc2_min': pc2_min, 'pc2_max': pc2_max,
            'n_pixels': n_pixels,
            'pixel_idx': np.where(mask.values)[0],
            'rect_artist': rect_patch,
            'text_artist': text_art,
            'sp_artist': sp_art,
        })

        print(f"  Added {name}: PC1 ∈ [{pc1_min:+.2f}, {pc1_max:+.2f}], "
              f"PC2 ∈ [{pc2_min:+.2f}, {pc2_max:+.2f}], n={n_pixels}")
        self._refresh_status()
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == 'd':
            self._delete_last()
        elif event.key == 'r':
            self._reset()
        elif event.key in ('s', 'enter', 'q'):
            plt.close(self.fig)

    def _delete_last(self):
        if not self.classes:
            print("  No classes to delete.")
            return
        c = self.classes.pop()
        for key in ('rect_artist', 'text_artist', 'sp_artist'):
            try: c[key].remove()
            except Exception: pass

        # Rebuild spatial legend
        leg = self.ax_sp.get_legend()
        if leg is not None:
            leg.remove()
        remaining_handles = []
        for cc in self.classes:
            remaining_handles.append(cc['sp_artist'])
        if remaining_handles:
            self.ax_sp.legend(handles=remaining_handles, fontsize=8,
                              loc='upper right', markerscale=3, framealpha=0.9)

        print(f"  Deleted {c['name']}")
        self._refresh_status()
        self.fig.canvas.draw_idle()

    def _reset(self):
        # Remove all class artists
        for c in self.classes:
            for key in ('rect_artist', 'text_artist', 'sp_artist'):
                try: c[key].remove()
                except Exception: pass
        self.classes = []

        # Clear legend
        leg = self.ax_sp.get_legend()
        if leg is not None:
            leg.remove()

        # Disconnect + recreate the RectangleSelector — the old one gets
        # confused after reset when its internal state was mid-drag.
        try:
            self.rect_selector.set_active(False)
            self.rect_selector.disconnect_events()
        except Exception:
            pass
        self._make_rect_selector()

        print("  Reset — draw new rectangles.")
        self._refresh_status()
        self.fig.canvas.draw_idle()

    def _refresh_status(self):
        n = len(self.classes)
        self.ax_pc.set_title(
            f'Draw rectangles → auto-named class_1..N  |  '
            f'{n} classes  |  d=del last, r=reset, s/q/Enter=finish',
            fontsize=9)
        self.ax_sp.set_title('Spatial view of defined classes', fontsize=9)
        self.fig.suptitle(
            f'{CUBE_ID.upper()} — PC-cluster class definition',
            fontsize=11)

    def run(self):
        print("\n" + "=" * 70)
        print("  Click-drag on the LEFT panel to draw a rectangle for each class.")
        print("  Classes auto-named class_1, class_2, ... (rename in CSV later).")
        print("  Keys:  d=delete last   r=reset all   s/q/Enter=finish")
        print("=" * 70)
        plt.show()
        return self.classes


# =============================================================================
# SPECTRUM EXTRACTION + RATIOING
# =============================================================================

def extract_class_spectra(classes, df, cube_data):
    """For each class, pull mean I/F spectrum from cube.

    cube_data shape: (bands, lines, samples)
    Returns classes (augmented in-place) with 'mean_spec', 'std_spec', 'n_valid_px'.
    """
    n_bands = cube_data.shape[0]
    for c in classes:
        idx = c['pixel_idx']
        lines = df.iloc[idx]['line'].values
        samples = df.iloc[idx]['sample'].values

        # Extract all pixel spectra
        # cube_data is (bands, lines, samples), so index (:, lines, samples)
        specs = cube_data[:, lines, samples]  # shape (bands, n_pixels)

        # Count how many are finite per band
        finite_mask = np.isfinite(specs)
        n_finite_per_band = finite_mask.sum(axis=1)

        with np.errstate(invalid='ignore'):
            mean_spec = np.nanmean(specs, axis=1)
            std_spec = np.nanstd(specs, axis=1)

        c['mean_spec'] = mean_spec
        c['std_spec'] = std_spec
        c['n_finite_per_band'] = n_finite_per_band
        c['n_valid_px'] = len(idx)
        print(f"  {c['name']:20s}  n={len(idx):>6d} px   "
              f"mean I/F @ 1.33 µm = {mean_spec[np.argmin(np.abs(cube_data.shape[0] - 200))]:.4f}"
              if False else
              f"  {c['name']:20s}  n={len(idx):>6d} px")
    return classes


def apply_ratio(samp, dust, wav):
    with np.errstate(divide='ignore', invalid='ignore'):
        rat = samp / dust
    rat[~np.isfinite(rat)] = np.nan
    # Mask CO2 region + detector boundary
    rat[(wav >= 0.99) & (wav <= 1.02)] = np.nan
    rat[(wav >= 2.65) & (wav <= 2.75)] = np.nan
    return rat


# =============================================================================
# PLOTTING
# =============================================================================

def select_denominator_interactive(classes, wav):
    """Plot raw mean spectra and let user click on a class name in a
    side panel to choose it as denominator. Returns class name, or None.

    Click any colored button in the right panel to pick a denominator
    (or 'SKIP' to not ratio).
    """
    fig, (ax_sp, ax_btn) = plt.subplots(
        1, 2, figsize=(15, 6),
        gridspec_kw={'width_ratios': [3, 1]}, constrained_layout=True)

    wmin, wmax = 0.4, 2.6
    for c in classes:
        mean = c['mean_spec']
        std = c['std_spec']
        mask_plot = (wav >= wmin) & (wav <= wmax)
        ax_sp.plot(wav[mask_plot], mean[mask_plot], color=c['color'],
                   linewidth=1.8, label=f"{c['name']} (n={c['n_valid_px']})")
        ax_sp.fill_between(wav[mask_plot],
                           mean[mask_plot] - std[mask_plot],
                           mean[mask_plot] + std[mask_plot],
                           color=c['color'], alpha=0.15)
    ax_sp.set_xlabel('Wavelength (µm)')
    ax_sp.set_ylabel('Mean I/F')
    ax_sp.set_title('Mean I/F per class (±1σ)  — pick denominator on right',
                    fontsize=10)
    ax_sp.grid(True, alpha=0.3)
    ax_sp.legend(fontsize=8, loc='best')
    ax_sp.set_xlim(wmin, wmax)

    # Button panel on the right
    ax_btn.set_xlim(0, 1)
    ax_btn.set_ylim(0, 1)
    ax_btn.axis('off')
    ax_btn.set_title('Click denominator', fontsize=10)

    button_items = []
    n = len(classes) + 1  # +1 for SKIP button
    btn_h = 0.8 / n
    btn_pad = 0.02

    for i, c in enumerate(classes):
        y = 0.95 - (i + 1) * (btn_h + btn_pad)
        rect = Rectangle((0.05, y), 0.9, btn_h,
                         facecolor=c['color'], alpha=0.85,
                         edgecolor='black', linewidth=1)
        ax_btn.add_patch(rect)
        ax_btn.text(0.5, y + btn_h / 2,
                    f"{c['name']}\n(n={c['n_valid_px']})",
                    ha='center', va='center', fontsize=9,
                    fontweight='bold', color='white')
        button_items.append({'name': c['name'], 'x': 0.05, 'y': y,
                             'w': 0.9, 'h': btn_h, 'rect': rect})

    # SKIP button
    y_skip = 0.95 - (len(classes) + 1) * (btn_h + btn_pad)
    skip_rect = Rectangle((0.05, y_skip), 0.9, btn_h,
                          facecolor='lightgray',
                          edgecolor='black', linewidth=1)
    ax_btn.add_patch(skip_rect)
    ax_btn.text(0.5, y_skip + btn_h / 2, 'SKIP\n(no ratio)',
                ha='center', va='center', fontsize=9,
                fontweight='bold', color='black')
    button_items.append({'name': None, 'x': 0.05, 'y': y_skip,
                         'w': 0.9, 'h': btn_h, 'rect': skip_rect})

    selected = {'name': None, 'clicked': False}

    def on_click(event):
        if event.inaxes != ax_btn:
            return
        for b in button_items:
            if (b['x'] <= event.xdata <= b['x'] + b['w'] and
                    b['y'] <= event.ydata <= b['y'] + b['h']):
                selected['name'] = b['name']
                selected['clicked'] = True
                label = b['name'] if b['name'] else 'SKIP'
                print(f"  → Denominator selected: {label}")
                plt.close(fig)
                return

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.suptitle('Choose denominator: click a colored button on the right '
                 '(or SKIP).  Close window to cancel.', fontsize=10)
    plt.show()

    return selected['name'] if selected['clicked'] else None


def plot_ratioed_spectra(classes, wav, denom_name, out_path=None):
    denom = next((c for c in classes if c['name'] == denom_name), None)
    if denom is None:
        raise ValueError(f"Denominator class '{denom_name}' not found")

    fig, ax = plt.subplots(1, 1, figsize=(11, 6), constrained_layout=True)
    wmin, wmax = 0.4, 2.6
    mask_plot = (wav >= wmin) & (wav <= wmax)

    for c in classes:
        if c['name'] == denom_name:
            continue
        ratio = apply_ratio(c['mean_spec'], denom['mean_spec'], wav)
        ax.plot(wav[mask_plot], ratio[mask_plot], color=c['color'],
                linewidth=1.8,
                label=f"{c['name']} / {denom_name}  (n={c['n_valid_px']})")

    ax.axhline(1.0, color='black', ls='--', lw=0.6, alpha=0.5)

    # Reference wavelengths
    for w, lbl in [(1.05, '1 µm'), (1.90, '1.9 µm H₂O'),
                   (2.30, '2.3 µm Fe/Mg-OH'), (2.50, '2.5 µm CO₃')]:
        if wmin <= w <= wmax:
            ax.axvline(w, color='gray', ls=':', lw=0.5, alpha=0.35)
            ax.text(w, ax.get_ylim()[1] if False else 1.05, lbl,
                    rotation=90, fontsize=7, color='gray',
                    va='bottom', ha='right', alpha=0.7)

    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Ratioed I/F')
    ax.set_title(f'Class mean spectra ratioed by "{denom_name}" (PC-defined classes)',
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    ax.set_xlim(wmin, wmax)

    if out_path:
        fig.savefig(out_path, dpi=180, bbox_inches='tight')
        print(f"  Saved: {out_path}")
    return fig


def save_spectra_csv(classes, wav, out_path, denom_name=None):
    """Save a single CSV with wavelength + each class mean I/F (and ratioed if denom provided)."""
    data = {'wavelength_um': wav}
    for c in classes:
        data[f"{c['name']}_mean_IF"] = c['mean_spec']
        data[f"{c['name']}_std_IF"] = c['std_spec']
    if denom_name:
        denom = next((c for c in classes if c['name'] == denom_name), None)
        if denom is not None:
            for c in classes:
                if c['name'] == denom_name: continue
                ratio = apply_ratio(c['mean_spec'], denom['mean_spec'], wav)
                data[f"{c['name']}_over_{denom_name}"] = ratio
    df = pd.DataFrame(data)
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")


def save_classes_def_csv(classes, out_path):
    """Save the class definitions (PC bounds, pixel count) for reproducibility."""
    rows = []
    for c in classes:
        rows.append({
            'name': c['name'],
            'pc1_min': c['pc1_min'], 'pc1_max': c['pc1_max'],
            'pc2_min': c['pc2_min'], 'pc2_max': c['pc2_max'],
            'n_pixels': c['n_pixels'],
            'color': c['color'],
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"\n{'='*70}\n  PC-cluster spectral extraction for {CUBE_ID}\n{'='*70}\n")

    print("Loading cube + Brown maps + running PCA...")
    df, shape, cube_data, wav = build_pca_pixels(CUBE_DIR, BROWN_VERSION)

    # Interactive class definition
    selector = PCRectangleSelector(df, shape, wav)
    classes = selector.run()

    if not classes:
        print("No classes defined. Exiting.")
        return

    print(f"\n  Defined {len(classes)} classes:")
    for c in classes:
        print(f"    {c['name']:20s}  "
              f"PC1=[{c['pc1_min']:+.2f}, {c['pc1_max']:+.2f}]  "
              f"PC2=[{c['pc2_min']:+.2f}, {c['pc2_max']:+.2f}]  "
              f"n={c['n_pixels']}")

    print("\nExtracting mean I/F spectra per class...")
    classes = extract_class_spectra(classes, df, cube_data)

    # Save standalone raw-means figure
    raw_plot_path = os.path.join(OUT_DIR, f'{CUBE_ID}_pc_classes_raw_mean.png')
    fig_raw, ax_raw = plt.subplots(1, 1, figsize=(11, 6), constrained_layout=True)
    wmin, wmax = 0.4, 2.6
    for c in classes:
        mean = c['mean_spec']; std = c['std_spec']
        mask_plot = (wav >= wmin) & (wav <= wmax)
        ax_raw.plot(wav[mask_plot], mean[mask_plot], color=c['color'],
                    linewidth=1.8, label=f"{c['name']} (n={c['n_valid_px']})")
        ax_raw.fill_between(wav[mask_plot],
                            mean[mask_plot] - std[mask_plot],
                            mean[mask_plot] + std[mask_plot],
                            color=c['color'], alpha=0.15)
    ax_raw.set_xlabel('Wavelength (µm)'); ax_raw.set_ylabel('Mean I/F')
    ax_raw.set_title('Mean I/F per PC-defined class (raw, ±1σ)', fontsize=11)
    ax_raw.grid(True, alpha=0.3); ax_raw.legend(fontsize=9, loc='best')
    ax_raw.set_xlim(wmin, wmax)
    fig_raw.savefig(raw_plot_path, dpi=180, bbox_inches='tight')
    print(f"  Saved: {raw_plot_path}")
    plt.close(fig_raw)

    # Interactive denominator selection (click buttons, no terminal input)
    print("\n  Opening denominator selection window — click a class to ratio by.")
    denom_name = select_denominator_interactive(classes, wav)

    if denom_name is not None:
        print(f"\n  Using '{denom_name}' as denominator.")
        ratio_plot_path = os.path.join(
            OUT_DIR, f'{CUBE_ID}_pc_classes_ratioed_over_{denom_name}.png')
        fig_rat = plot_ratioed_spectra(classes, wav, denom_name,
                                       out_path=ratio_plot_path)
    else:
        print("\n  No denominator selected — skipping ratioing.")

    # Save CSVs
    spec_csv = os.path.join(OUT_DIR, f'{CUBE_ID}_pc_classes_spectra.csv')
    save_spectra_csv(classes, wav, spec_csv, denom_name=denom_name)

    defs_csv = os.path.join(OUT_DIR, f'{CUBE_ID}_pc_classes_definitions.csv')
    save_classes_def_csv(classes, defs_csv)

    plt.show()


if __name__ == "__main__":
    main()