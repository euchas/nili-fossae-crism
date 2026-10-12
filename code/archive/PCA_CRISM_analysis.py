"""
Per-Pixel PCA: Brown + Viviano-Beck parameters for FRT00003E12
---------------------------------------------------------------
Extracts Brown parameters (from pre-computed .npy maps) and VB parameters
(computed fresh from the cube) per-pixel, standardizes, runs PCA, and
visualizes the scene's natural structure in parameter space.

Parameters used (8 total):
  Brown:  centroid, asymmetry, amplitude, HWHM
  VB:     OLINDEX3, D2300, BD1900r2, BD2500_2

Outputs:
  1. Scree plot (explained variance per PC)
  2. Loadings heatmap (how each parameter weights into each PC)
  3. PC1 vs PC2 hexbin density (scene-wide distribution)
  4. PC1 vs PC2 colored by each individual parameter (facet grid)
  5. ROI pixels overlaid on PC1 vs PC2 by class label
  6. Spatial maps of PC1 and PC2 across the scene

Author: AV Valantinas
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import BROWN_ROI_DIR, CUBES_DIR, STYLE_PATH
from matplotlib.colors import LogNorm

warnings.filterwarnings('ignore')

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except ImportError:
    raise ImportError("Install scikit-learn:  pip install scikit-learn")


# =============================================================================
# USER CONFIGURATION
# =============================================================================

CUBE_DIR = str(CUBES_DIR / "FRT00003E12_07_IF166J_TER3")

# Brown .npy map version
BROWN_VERSION = "v5"

# ROI pixel CSVs (same folder as the analysis CSVs)
ROI_CSV_DIR = str(BROWN_ROI_DIR)

CUBE_ID = "frt00003e12"   # used for .npy map discovery and ROI filtering

FLIP_VERTICAL = True
FLIP_HORIZONTAL = True

# Plot style
style_path = str(STYLE_PATH)
if os.path.exists(style_path):
    plt.style.use(style_path)

CLASS_COLORS = {
    'olivine':    '#1b9e77',
    'ol-carb':    '#d95f02',
    'ridge':      '#7570b3',
    'intrusions': '#e7298a',
    'clays':      '#66a61e',
    'nontronite': '#e6ab02',
}


# =============================================================================
# CUBE LOADING + VB PARAMETER CALCULATIONS  (reduced set)
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
    return np.nan_to_num(result, nan=np.nan)


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
    return np.nan_to_num(result, nan=np.nan)


def calc_bd2500_2(data, wav):
    R2364 = get_kernel(data, wav, 2.364, 5)
    R2480 = get_kernel(data, wav, 2.480, 5)
    R2570 = get_kernel(data, wav, 2.570, 5)
    b = (2.480 - 2.364) / (2.570 - 2.364)
    a = 1.0 - b
    return _ratio(R2480, a * R2364 + b * R2570)


# =============================================================================
# DATA ASSEMBLY
# =============================================================================

def find_cube_and_brown_maps(cube_dir, version):
    """Return (img_path, {param: ndarray})."""
    patterns = ['*_if*ter3*.img', '*_IF*TER3*.img', '*_if*TER3*.img']
    img_file = None
    for pat in patterns:
        matches = glob.glob(os.path.join(cube_dir, pat))
        if matches:
            img_file = matches[0]
            break
    if img_file is None:
        raise FileNotFoundError(f"No TER3 cube in {cube_dir}")

    base = os.path.splitext(os.path.basename(img_file))[0]
    brown = {}
    for param in ['centroid', 'asymmetry', 'amplitude', 'hwhm', 'mask']:
        path = os.path.join(cube_dir, f'{base}_{param}_map_{version}.npy')
        if os.path.exists(path):
            brown[param] = np.load(path)
            print(f"  Loaded Brown {param}: {brown[param].shape}")
        else:
            raise FileNotFoundError(f"Missing Brown map: {path}")

    return img_file, brown


def build_parameter_stack(cube_dir, cube_id, version):
    """Build a per-pixel DataFrame of all 8 parameters.

    Returns:
        df : DataFrame with columns
             ['line', 'sample', 'centroid', 'asymmetry', 'amplitude',
              'hwhm', 'OLINDEX3', 'D2300', 'BD1900r2', 'BD2500_2']
             one row per valid pixel.
        shape : (n_lines, n_samples)
    """
    img_file, brown = find_cube_and_brown_maps(cube_dir, version)

    print(f"\n  Reading cube to compute VB parameters...")
    data, wav = read_ter3_cube(img_file)
    print(f"  Cube shape: {data.shape}")

    print("  Computing OLINDEX3 ..."); olindex3  = calc_olindex3(data, wav)
    print("  Computing D2300 ..."); d2300     = calc_d2300(data, wav)
    print("  Computing BD1900r2 ..."); bd1900r2  = calc_bd1900r2(data, wav)
    print("  Computing BD2500_2 ..."); bd2500_2  = calc_bd2500_2(data, wav)

    free = data  # keep reference; GC after param extraction
    del free

    n_lines, n_samples = brown['centroid'].shape

    # Option A: use all pixels with finite Brown parameters + finite VB.
    # Do NOT filter by brown['mask'] (Eq. 2) — that would exclude LCP /
    # pyroxene-bearing pixels, which we want to characterize alongside olivine.
    finite_vb = (np.isfinite(olindex3) & np.isfinite(d2300) &
                 np.isfinite(bd1900r2) & np.isfinite(bd2500_2))
    finite_brown = (np.isfinite(brown['centroid']) & np.isfinite(brown['asymmetry']) &
                    np.isfinite(brown['amplitude']) & np.isfinite(brown['hwhm']))
    valid = finite_vb & finite_brown

    n_valid = int(valid.sum())
    n_finite_brown = int(finite_brown.sum())
    n_brown_pass = int(brown['mask'].astype(bool).sum())
    n_total = brown['centroid'].size

    print(f"\n  Cube total pixels:         {n_total:>8,}")
    print(f"  Finite Brown fit:          {n_finite_brown:>8,}  ({100*n_finite_brown/n_total:5.1f}%)")
    print(f"  Pass Brown Eq.2 (for ref): {n_brown_pass:>8,}  ({100*n_brown_pass/n_total:5.1f}%)")
    print(f"  Finite Brown + finite VB:  {n_valid:>8,}  ({100*n_valid/n_total:5.1f}%)  <- used for PCA")

    yy, xx = np.where(valid)

    df = pd.DataFrame({
        'line':       yy.astype(np.int32),
        'sample':     xx.astype(np.int32),
        'centroid':   brown['centroid'][yy, xx],
        'asymmetry':  brown['asymmetry'][yy, xx],
        'amplitude':  brown['amplitude'][yy, xx],
        'hwhm':       brown['hwhm'][yy, xx],
        'OLINDEX3':   olindex3[yy, xx],
        'D2300':      d2300[yy, xx],
        'BD1900r2':   bd1900r2[yy, xx],
        'BD2500_2':   bd2500_2[yy, xx],
    })

    return df, (n_lines, n_samples)


def load_roi_labels(roi_csv_dir, cube_id):
    """Load ROI pixel coordinates + class labels for a given cube."""
    pattern = os.path.join(roi_csv_dir, f'{cube_id}_roi_pixels_*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  No ROI pixel CSVs for {cube_id} in {roi_csv_dir}")
        return pd.DataFrame(columns=['line', 'sample', 'class', 'roi', 'type'])

    rows = []
    for f in files:
        base = os.path.basename(f).replace('.csv', '')
        cls = base.split('_roi_pixels_')[-1]
        df = pd.read_csv(f)
        # Only keep 'sample' type (sample ROI pixels, not dust)
        df = df[df['type'] == 'sample'].copy()
        df['class'] = cls
        rows.append(df)
        print(f"  {os.path.basename(f):45s} | {len(df)} sample pixels")

    labels = pd.concat(rows, ignore_index=True)
    return labels


# =============================================================================
# PCA + PLOTTING
# =============================================================================

PARAM_ORDER = ['centroid', 'asymmetry', 'amplitude', 'hwhm',
               'OLINDEX3', 'D2300', 'BD1900r2', 'BD2500_2']


def run_pca(df, params=PARAM_ORDER):
    X = df[params].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=len(params))
    Xp = pca.fit_transform(Xs)
    return Xp, pca, scaler


def attach_pca(df, Xp):
    out = df.copy()
    for i in range(Xp.shape[1]):
        out[f'PC{i+1}'] = Xp[:, i]
    return out


def panel_scree(ax, pca):
    ev = pca.explained_variance_ratio_ * 100
    cum = np.cumsum(ev)
    xs = np.arange(1, len(ev) + 1)
    ax.bar(xs, ev, color='steelblue', alpha=0.7, label='Individual')
    ax.plot(xs, cum, 'o-', color='darkred', label='Cumulative')
    ax.axhline(80, color='gray', ls=':', alpha=0.6)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_title('Scree plot')
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    for i, (v, c) in enumerate(zip(ev, cum)):
        ax.text(i + 1, v + 1.5, f'{v:.0f}%', ha='center', fontsize=7)


def panel_loadings(ax, pca, params):
    loadings = pca.components_.T  # [n_params, n_pcs]
    n_pcs_show = min(4, loadings.shape[1])
    im = ax.imshow(loadings[:, :n_pcs_show], aspect='auto',
                   cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(n_pcs_show))
    ax.set_xticklabels([f'PC{i+1}\n({pca.explained_variance_ratio_[i]*100:.0f}%)'
                        for i in range(n_pcs_show)], fontsize=9)
    ax.set_yticks(range(len(params)))
    ax.set_yticklabels(params, fontsize=9)
    ax.set_title('Loadings')
    # Annotate
    for i in range(len(params)):
        for j in range(n_pcs_show):
            v = loadings[i, j]
            ax.text(j, i, f'{v:+.2f}', ha='center', va='center',
                    fontsize=8, color='white' if abs(v) > 0.5 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)


def panel_density(ax, df_pca):
    hb = ax.hexbin(df_pca['PC1'], df_pca['PC2'], gridsize=80,
                   cmap='viridis', mincnt=1, norm=LogNorm())
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'Scene-wide density (n = {len(df_pca):,})')
    plt.colorbar(hb, ax=ax, shrink=0.7, pad=0.02, label='pixels (log)')
    ax.axhline(0, color='gray', ls=':', lw=0.5, alpha=0.5)
    ax.axvline(0, color='gray', ls=':', lw=0.5, alpha=0.5)


def panel_pc_by_param(ax, df_pca, param, subsample=50000):
    # Subsample for plotting (speed)
    if len(df_pca) > subsample:
        sub = df_pca.sample(subsample, random_state=42)
    else:
        sub = df_pca
    sc = ax.scatter(sub['PC1'], sub['PC2'], c=sub[param], cmap='viridis',
                    s=1, alpha=0.4, rasterized=True)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(param, fontsize=10)
    plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)


def panel_roi_overlay(ax, df_pca, roi_df):
    # Background: full scene density
    ax.hexbin(df_pca['PC1'], df_pca['PC2'], gridsize=80,
              cmap='Greys', mincnt=1, norm=LogNorm(), alpha=0.6)

    if len(roi_df) == 0:
        ax.text(0.5, 0.5, 'No ROI labels', ha='center', va='center',
                transform=ax.transAxes)
        return

    # Build (line, sample) → PC lookup
    key = df_pca['line'].astype(np.int64) * 100000 + df_pca['sample'].astype(np.int64)
    pc_map = dict(zip(key.values, zip(df_pca['PC1'].values, df_pca['PC2'].values)))

    for cls in sorted(roi_df['class'].unique()):
        sub = roi_df[roi_df['class'] == cls]
        pcs = [pc_map.get(int(r['line']) * 100000 + int(r['sample']))
               for _, r in sub.iterrows()]
        pcs = [p for p in pcs if p is not None]
        if not pcs:
            continue
        pc1_vals = [p[0] for p in pcs]
        pc2_vals = [p[1] for p in pcs]
        ax.scatter(pc1_vals, pc2_vals, s=14, alpha=0.85,
                   color=CLASS_COLORS.get(cls, '#000'),
                   edgecolors='black', linewidths=0.3,
                   label=f'{cls} (n={len(pcs)})')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('ROI pixels overlaid on scene density')
    ax.legend(fontsize=8, markerscale=2, loc='best')


def apply_display_transform(image):
    result = image.copy()
    if FLIP_VERTICAL:
        result = np.flipud(result)
    if FLIP_HORIZONTAL:
        result = np.fliplr(result)
    return result


def panel_spatial_pc(ax, df_pca, shape, pc_name='PC1'):
    img = np.full(shape, np.nan, dtype=np.float32)
    img[df_pca['line'].values, df_pca['sample'].values] = df_pca[pc_name].values
    disp = apply_display_transform(img)
    valid = disp[np.isfinite(disp)]
    if len(valid) > 0:
        vmin, vmax = np.percentile(valid, 2), np.percentile(valid, 98)
    else:
        vmin, vmax = 0, 1
    im = ax.imshow(disp, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                   origin='upper', interpolation='nearest')
    ax.set_title(f'{pc_name} spatial')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"\n{'='*70}\n  Per-pixel PCA for {CUBE_ID}\n{'='*70}\n")

    print("Building parameter stack...")
    df, shape = build_parameter_stack(CUBE_DIR, CUBE_ID, BROWN_VERSION)

    print("\nLoading ROI labels...")
    roi_df = load_roi_labels(ROI_CSV_DIR, CUBE_ID)

    print(f"\nRunning PCA on {len(df):,} pixels × {len(PARAM_ORDER)} parameters...")
    Xp, pca, scaler = run_pca(df, PARAM_ORDER)
    df_pca = attach_pca(df, Xp)

    print("\nExplained variance:")
    for i, v in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {v*100:5.2f}%  (cumulative {np.sum(pca.explained_variance_ratio_[:i+1])*100:5.2f}%)")

    # ======================= FIGURE 1: PCA overview =======================
    fig1 = plt.figure(figsize=(16, 11), constrained_layout=True)
    gs = fig1.add_gridspec(2, 3)

    ax_scree = fig1.add_subplot(gs[0, 0])
    panel_scree(ax_scree, pca)

    ax_load = fig1.add_subplot(gs[0, 1])
    panel_loadings(ax_load, pca, PARAM_ORDER)

    ax_dens = fig1.add_subplot(gs[0, 2])
    panel_density(ax_dens, df_pca)

    ax_roi = fig1.add_subplot(gs[1, 0])
    panel_roi_overlay(ax_roi, df_pca, roi_df)

    ax_pc1 = fig1.add_subplot(gs[1, 1])
    panel_spatial_pc(ax_pc1, df_pca, shape, 'PC1')

    ax_pc2 = fig1.add_subplot(gs[1, 2])
    panel_spatial_pc(ax_pc2, df_pca, shape, 'PC2')

    fig1.suptitle(f'{CUBE_ID.upper()} — per-pixel PCA '
                  f'({len(df):,} valid Brown-pass pixels)',
                  fontsize=12)

    # ======================= FIGURE 2: PC1-PC2 colored by each parameter ===
    fig2, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True,
                              sharex=True, sharey=True)
    for ax, param in zip(axes.flatten(), PARAM_ORDER):
        panel_pc_by_param(ax, df_pca, param)
    fig2.suptitle(f'{CUBE_ID.upper()} — PC1 vs PC2 colored by each parameter',
                  fontsize=12)

    # ======================= SAVE =========================================
    out_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
    # Save to the ROI CSV directory (alongside analysis products)
    out_dir = ROI_CSV_DIR if os.path.isdir(ROI_CSV_DIR) else out_dir
    try:
        p1 = os.path.join(out_dir, f'{CUBE_ID}_pca_overview.png')
        p2 = os.path.join(out_dir, f'{CUBE_ID}_pca_by_param.png')
        fig1.savefig(p1, dpi=180, bbox_inches='tight')
        fig2.savefig(p2, dpi=180, bbox_inches='tight')
        print(f"\n  Saved: {p1}")
        print(f"  Saved: {p2}")
    except Exception as e:
        print(f"  Save warning: {e}")

    # ======================= CSV EXPORT ====================================
    # Save PC-augmented dataframe (subsampled for size — full one would be huge)
    sub_n = min(200000, len(df_pca))
    csv_path = os.path.join(out_dir, f'{CUBE_ID}_pca_pixels.csv')
    try:
        df_pca.sample(sub_n, random_state=42).to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path} ({sub_n:,} rows subsampled)")
    except Exception as e:
        print(f"  CSV save warning: {e}")

    plt.show()


if __name__ == "__main__":
    main()