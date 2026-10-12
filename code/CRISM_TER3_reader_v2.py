"""
CRISM TER3 Multi-Panel Interactive Viewer (3x3, dynamic stretch)
-----------------------------------------------------------------
Nine parameter/browse panels, linked pan/zoom, per-view dynamic stretching.
No lasso, no ratioing, no band-fit — pure visualization tool for exploring
a single TER3 cube spatially.

Panels (left→right, top→bottom):
  1. D2300           — 2.3 µm dropoff (Fe/Mg-OH)                → phyllosilicates, carbonate
  2. BD2500_2        — 2.5 µm Mg-carbonate                      → carbonate
  3. BD1900r2        — 1.9 µm H2O band                          → hydration
  4. OLINDEX3        — 1 µm broad olivine                       → olivine
  5. LCPINDEX2       — 1.81 µm low-Ca pyroxene                  → LCP
  6. BD1400          — 1.4 µm H2O/OH                            → hydroxylation
  7. IR False Color  — R=2.529 G=1.506 B=1.080 µm
  8. VNIR True Color — R=0.600 G=0.530 B=0.440 µm
  9. BD920_2         — 0.92 µm mafic (Fe2+ crystal field)       → coarse mafic / Fe-olivine

Dynamic stretching: zoom or pan any panel → all panels restretch per-view.
Press 'f' to force a restretch.

Author: AV Valantinas
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from config import CUBES_DIR, STYLE_PATH

warnings.filterwarnings('ignore')

# =============================================================================
# USER CONFIGURATION
# =============================================================================

FILEPATH = str(CUBES_DIR / "FRT00003E12_07_IF166J_TER3" / "frt00003e12_07_if166j_ter3.img")

FLIP_VERTICAL = True
FLIP_HORIZONTAL = True

STRETCH_PMIN = 2
STRETCH_PMAX = 98

style_path = str(STYLE_PATH)
if os.path.exists(style_path):
    plt.style.use(style_path)


# =============================================================================
# DISPLAY TRANSFORMS
# =============================================================================

def apply_display_transform(image):
    result = image.copy()
    if FLIP_VERTICAL:
        result = np.flipud(result)
    if FLIP_HORIZONTAL:
        result = np.fliplr(result)
    return result


# =============================================================================
# DATA LOADING  (unchanged from v4.2)
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
        raise FileNotFoundError(f"No header for {img_file}")

    meta = parse_envi_header(hdr)
    s = int(meta.get('samples', 0))
    l = int(meta.get('lines', 0))
    b = int(meta.get('bands', 0))
    dt = int(meta.get('data type', 4))
    order = int(meta.get('byte order', 0))
    inter = meta.get('interleave', 'bsq').lower()

    print(f"Reading {os.path.basename(img_file)} ({s}x{l}x{b}, {inter})")

    if meta.get('wavelength'):
        wav = np.array([float(w) for w in meta['wavelength']])
        units = str(meta.get('wavelength units', 'Nanometers')).lower()
        if 'nano' in units or wav.max() > 100:
            wav = wav / 1000.0
    else:
        wav = np.linspace(0.362, 3.937, b)

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


# =============================================================================
# SPECTRAL PARAMETER HELPERS
# =============================================================================

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
    return np.nan_to_num(result, nan=0.0)


def _rb(R, RC):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(RC != 0, (RC - R) / RC, 0.0)
    return np.nan_to_num(result, nan=0.0)


# =============================================================================
# PARAMETER CALCULATIONS
# =============================================================================

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
    b = (2.480 - 2.364) / (2.570 - 2.364)
    a = 1.0 - b
    return _ratio(R2480, a * R2364 + b * R2570)


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


def calc_bd1400(data, wav):
    R1395 = get_kernel(data, wav, 1.395, 3)
    R1330 = get_kernel(data, wav, 1.330, 5)
    R1467 = get_kernel(data, wav, 1.467, 5)
    b = (1.395 - 1.330) / (1.467 - 1.330)
    a = 1.0 - b
    return _ratio(R1395, a * R1330 + b * R1467)


def calc_bd920_2(data, wav):
    """Band depth at 920 nm — Fe2+ crystal field absorption (Viviano-Beck 2014).
    Continuum: linear between 0.800 and 0.984 µm. Band depth at 0.920 µm."""
    R800 = get_kernel(data, wav, 0.800, 5)
    R920 = get_kernel(data, wav, 0.920, 5)
    R984 = get_kernel(data, wav, 0.984, 5)
    b = (0.920 - 0.800) / (0.984 - 0.800)
    a = 1.0 - b
    return _ratio(R920, a * R800 + b * R984)


# =============================================================================
# BROWSE PRODUCTS
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
# DYNAMIC STRETCH MANAGER (handles both single-channel and RGB panels)
# =============================================================================

class DynamicStretchManager:
    """Per-view dynamic stretching for mixed single-band and RGB panels.

    Each panel dict has:
        'ax'       : Axes
        'raw'      : np.ndarray, shape (H,W) for single-band or (H,W,3) for RGB
        'im'       : AxesImage
        'is_rgb'   : bool
        'cmap'     : str  (single-band only)
    """

    def __init__(self, fig, panels, p_min=STRETCH_PMIN, p_max=STRETCH_PMAX):
        self.fig = fig
        self.panels = panels
        self.p_min = p_min
        self.p_max = p_max
        self._pending = False

        # Pre-apply display transform once
        for p in self.panels:
            p['raw_display'] = apply_display_transform(p['raw'])

        # Initial full-scene stretch
        self.restretch_all(force_full=True)

        # Listen for view changes on each axis
        for p in self.panels:
            ax = p['ax']
            ax.callbacks.connect('xlim_changed', self._on_lim_changed)
            ax.callbacks.connect('ylim_changed', self._on_lim_changed)

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

    def _get_view_bounds(self, p, force_full):
        img = p['raw_display']
        if img.ndim == 3:
            H, W, _ = img.shape
        else:
            H, W = img.shape
        ax = p['ax']
        if force_full:
            return 0, W, 0, H
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x0 = max(0, int(np.floor(min(xlim))))
        x1 = min(W, int(np.ceil(max(xlim))))
        y0 = max(0, int(np.floor(min(ylim))))
        y1 = min(H, int(np.ceil(max(ylim))))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return 0, W, 0, H
        return x0, x1, y0, y1

    def _restretch_panel(self, p, force_full=False):
        img = p['raw_display']
        x0, x1, y0, y1 = self._get_view_bounds(p, force_full)

        if p['is_rgb']:
            sub = img[y0:y1, x0:x1, :]
            stretched = np.zeros_like(img, dtype=np.float32)
            for ci in range(3):
                full_c = img[:, :, ci]
                sub_c = sub[:, :, ci]
                valid = sub_c[np.isfinite(sub_c)]
                if len(valid) < 10:
                    valid = full_c[np.isfinite(full_c)]
                if len(valid) > 0:
                    vmin = np.percentile(valid, self.p_min)
                    vmax = np.percentile(valid, self.p_max)
                    if vmax > vmin:
                        stretched[:, :, ci] = np.clip(
                            (full_c - vmin) / (vmax - vmin), 0, 1)
            p['im'].set_data(stretched)
        else:
            sub = img[y0:y1, x0:x1]
            valid = sub[np.isfinite(sub)]
            if len(valid) < 10:
                valid = img[np.isfinite(img)]
            if len(valid) > 0:
                vmin = np.percentile(valid, self.p_min)
                vmax = np.percentile(valid, self.p_max)
                if vmax > vmin:
                    p['im'].set_clim(vmin, vmax)

    def restretch_all(self, force_full=False):
        for p in self.panels:
            self._restretch_panel(p, force_full=force_full)


# =============================================================================
# MAIN VIEWER
# =============================================================================

class MultiPanelViewer:

    # (label, function-or-None, is_rgb, cmap)
    # All parameter panels use the same grayscale cmap for visual consistency.
    PANEL_SPECS = [
        ('D2300',           calc_d2300,      False, 'gray'),
        ('BD2500_2',        calc_bd2500_2,   False, 'gray'),
        ('BD1900r2',        calc_bd1900r2,   False, 'gray'),
        ('OLINDEX3',        calc_olindex3,   False, 'gray'),
        ('LCPINDEX2',       calc_lcpindex2,  False, 'gray'),
        ('BD1400',          calc_bd1400,     False, 'gray'),
        ('IR False Color',  None,            True,  None),
        ('VNIR True Color', None,            True,  None),
        ('BD920_2',         calc_bd920_2,    False, 'gray'),
    ]

    def __init__(self):
        print(f"\nLoading cube...")
        self.data, self.wav = read_crism_ter3_cube(FILEPATH)
        print(f"  Wavelength range: {self.wav.min():.3f}–{self.wav.max():.3f} µm")
        print(f"  Cube shape: {self.data.shape}\n")

        # Compute all parameter maps + browse products
        print("Computing parameter maps...")
        self.raws = []
        for label, func, is_rgb, cmap in self.PANEL_SPECS:
            if is_rgb:
                if label == 'IR False Color':
                    raw = create_browse_ir_raw(self.data, self.wav)
                elif label == 'VNIR True Color':
                    raw = create_browse_vnir_raw(self.data, self.wav)
                else:
                    raw = None
                if raw is None:
                    print(f"  {label}: NOT AVAILABLE (VNIR out of range?)")
                else:
                    print(f"  {label}: computed")
            else:
                print(f"  {label}: computing...")
                raw = func(self.data, self.wav)
            self.raws.append(raw)

        self._build_figure()

    def _build_figure(self):
        self.fig, axes = plt.subplots(3, 3, figsize=(14, 13),
                                      constrained_layout=True)
        self.axes = axes.flatten()

        panels = []
        for i, ((label, func, is_rgb, cmap), raw, ax) in enumerate(
                zip(self.PANEL_SPECS, self.raws, self.axes)):

            if raw is None:
                ax.text(0.5, 0.5, f'{label}\n(unavailable)',
                        ha='center', va='center',
                        transform=ax.transAxes, fontsize=10, color='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            disp = apply_display_transform(raw)
            if is_rgb:
                im = ax.imshow(np.zeros_like(disp), origin='upper',
                               interpolation='nearest', aspect='equal')
            else:
                im = ax.imshow(disp, origin='upper',
                               interpolation='nearest', cmap=cmap,
                               aspect='equal')
                try:
                    cb = self.fig.colorbar(im, ax=ax, shrink=0.7,
                                           pad=0.02, fraction=0.046)
                    cb.ax.tick_params(labelsize=7)
                except Exception:
                    pass

            ax.set_title(label, fontsize=10, pad=6)
            ax.set_xticks([])
            ax.set_yticks([])

            panels.append({
                'ax': ax,
                'raw': raw,
                'im': im,
                'is_rgb': is_rgb,
                'cmap': cmap,
                'label': label,
            })

        if len(panels) >= 2:
            first_ax = panels[0]['ax']
            for p in panels[1:]:
                p['ax'].sharex(first_ax)
                p['ax'].sharey(first_ax)

        self.stretch_mgr = DynamicStretchManager(self.fig, panels)

        self.fig.suptitle(
            f'{os.path.basename(FILEPATH)} — 3×3 Spectral Parameter Viewer   '
            '(zoom/pan any panel → all restretch;  f = force restretch,  q = quit)',
            fontsize=10)

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        plt.show()

    def _on_key(self, event):
        if event.key == 'q':
            plt.close('all')
        elif event.key == 'f':
            if self.stretch_mgr is not None:
                self.stretch_mgr.restretch_all()
                self.fig.canvas.draw_idle()
                print("  Forced restretch on current view.")


if __name__ == "__main__":
    MultiPanelViewer()