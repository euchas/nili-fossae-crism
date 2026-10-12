"""Stage v4 - VB-only PCA workflow + Brown band-fit RGB workflow.

Implements the full PLAN.md v4 spec. Drops Brown 4 from the PCA basis,
drops BD2290 / BD2355 (redundant with D2300), adds SH600_2 + HCPINDEX2.
Brown 4 are processed independently as RGB composites + grayscale maps.

Outputs land under outputs/_v4_vb_pca_brown_rgb/<cube>/{arrays,figures,
tables,logs}/ plus top-level figures/ and reports/.

Run: python stage_v4_vb_pca_and_brown_rgb.py [cube_id ...]
     (no args = all four cubes)
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, ListedColormap
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from config import CUBE_IDS, RANDOM_SEED, OUTPUTS_DIR
from crism_io import find_cube_and_brown_maps, load_cube_by_id
from display import apply_display_transform
from vb_parameters import V4_VB_PARAM_ORDER, compute_v4_vb_stack

# v2 styling - default matplotlib, white background
plt.rcdefaults()

V4_ROOT = OUTPUTS_DIR / "_v4_vb_pca_brown_rgb"

# Cube colours per spec section 1
CUBE_COLOURS = {
    "FRT00003E12": "tab:blue",
    "FRT0000871C": "tab:orange",
    "FRT0000A4FC": "tab:green",
    "FRT0000B438": "tab:red",
}
CUBE_MARKERS = {
    "FRT00003E12": "o",
    "FRT0000871C": "s",
    "FRT0000A4FC": "^",
    "FRT0000B438": "D",
}

# Discrete 6-class palette for GMM scene maps (consistent across all cubes
# after Hungarian alignment).
GMM_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c",
               "#d62728", "#9467bd", "#8c564b"]

# Mafic-cap z-sum parameters per Stage C convention.
MAFIC_CAP_PARAMS = ("BD1900R2", "D2300", "BD2500_2", "OLINDEX3")

# Brown 4 stretch ranges - v4.1 data-driven matched stretches
# (see DECISIONS.md "v4.1" entry). v4 used Brown 2020 eq. 2 validity bounds
# which left 40-60% of the colormap unused on real data. v4.1 retightens
# to four-cube p2/p98 (with amplitude lower bound held at the eq.2 floor
# of 0.10) so the matched-stretch property survives but the colormap is
# fully used.
BROWN_STRETCH = {
    "centroid": (1.15, 1.40),
    "amplitude": (0.10, 0.37),
    "hwhm": (0.18, 0.30),
    "asymmetry": (1.25, 5.00),
}
BROWN_STRETCH_NOTE = (
    "Stretches are data-driven, matched across cubes (4-cube pooled "
    "p2/p98; amplitude lower bound at the Brown 2020 eq. 2 floor of 0.10). "
    "Identical color = identical parameter value across cubes. Brown 2020 "
    "eq. 2 + residual<=p95 applied as a separate transparency mask."
)
ASYMMETRY_SAT_NOTE = (
    "Asymmetry > 5.0 (band-saturation regime per Brown 2020) clips to "
    "pure blue."
)

K_FIXED = 6
N_PCS_USED = 5


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def cube_root(cube_id: str) -> Path:
    p = V4_ROOT / cube_id
    (p / "arrays").mkdir(parents=True, exist_ok=True)
    (p / "figures").mkdir(parents=True, exist_ok=True)
    (p / "tables").mkdir(parents=True, exist_ok=True)
    (p / "logs").mkdir(parents=True, exist_ok=True)
    return p


def percentile_norm(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip+normalise to [0,1] using fixed range. NaN passes through."""
    out = np.full_like(arr, np.nan, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return out
    if hi - lo < 1e-9:
        out[finite] = 0.5
        return out
    out[finite] = np.clip((arr[finite] - lo) / (hi - lo), 0, 1)
    return out


# =============================================================================
# Step 1: VB-only feature stack per cube (full-range mask)
# =============================================================================
def build_vb_stack(cube_id: str, log: dict) -> dict:
    print(f"  [build] loading TER3 cube...")
    data, wav, brown, img_path = load_cube_by_id(cube_id)
    print(f"  [build] cube shape {data.shape}, computing 11 VB params...")
    t0 = time.time()
    vb = compute_v4_vb_stack(data, wav)
    print(f"  [build] VB stack done in {time.time()-t0:.1f}s")
    shape = next(iter(vb.values())).shape

    # Pixel mask: all 11 VB params finite. No BDI > 0 filter (per spec).
    finite = np.ones(shape, dtype=bool)
    for name in V4_VB_PARAM_ORDER:
        nan_frac = float(np.isnan(vb[name]).sum() / vb[name].size)
        log["param_nan_frac"][name] = round(nan_frac, 4)
        finite &= np.isfinite(vb[name])
    n_total = int(np.prod(shape))
    n_valid = int(finite.sum())
    print(f"  [build] mask counts: total={n_total}  valid={n_valid} ({100*n_valid/n_total:.1f}%)")

    # SH600_2 sign sanity check
    sh = vb["SH600_2"][finite]
    sh_med = float(np.nanmedian(sh))
    log["sh600_2_median"] = sh_med
    log["sh600_2_warning"] = ""
    if sh_med < -0.05:
        log["sh600_2_warning"] = (
            f"SH600_2 median is strongly negative ({sh_med:.4f}); "
            "this may indicate a sign/anchor mismatch -- inspect"
        )
        print("  ** WARN: " + log["sh600_2_warning"])

    # Save raw stack as (rows, cols, 11) for v5
    stack = np.stack([vb[p] for p in V4_VB_PARAM_ORDER], axis=-1).astype(np.float32)
    np.save(cube_root(cube_id) / "arrays" / f"{cube_id}_VB_stack_v4.npy", stack)
    np.save(cube_root(cube_id) / "arrays" / f"{cube_id}_pixel_valid_mask_v4.npy", finite)

    # Per-parameter raw stats
    raw_stats = {}
    for name in V4_VB_PARAM_ORDER:
        a = vb[name][finite]
        raw_stats[name] = {
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "median": float(np.median(a)),
            "p16": float(np.percentile(a, 16)),
            "p84": float(np.percentile(a, 84)),
        }
    (cube_root(cube_id) / "arrays" / f"{cube_id}_VB_param_names_v4.json").write_text(
        json.dumps({
            "param_order": list(V4_VB_PARAM_ORDER),
            "raw_stats": raw_stats,
        }, indent=2),
        encoding="utf-8",
    )
    log["n_pixels_total"] = n_total
    log["n_pixels_valid"] = n_valid
    log["valid_fraction"] = round(n_valid / n_total, 4)
    log["raw_stats"] = raw_stats
    return {"vb": vb, "valid": finite, "shape": shape, "wav": wav,
            "ter3_data": data, "img_path": img_path, "brown": brown}


# =============================================================================
# Step 2: PCA + GMM k=6 (whitened), save raw + whitened scores
# =============================================================================
def run_pca_gmm(cube_id: str, art: dict, log: dict) -> dict:
    valid = art["valid"]
    shape = art["shape"]
    X = np.column_stack([art["vb"][p][valid].astype(np.float64)
                         for p in V4_VB_PARAM_ORDER])
    print(f"  [pca]  X shape: {X.shape}")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=len(V4_VB_PARAM_ORDER), random_state=RANDOM_SEED)
    Xp = pca.fit_transform(Xs)  # raw scores
    cum = np.cumsum(pca.explained_variance_ratio_)
    n_pcs_for_90 = int(np.searchsorted(cum, 0.90) + 1)
    n_pcs_table = max(N_PCS_USED, n_pcs_for_90)
    print(f"  [pca]  PC1-5 cum={cum[4]*100:.1f}%  ; PC1-{n_pcs_for_90} reaches 90%")

    # Save raw + whitened scores back-projected to (nrows, ncols, 5)
    raw_scene = np.full((shape[0], shape[1], N_PCS_USED), np.nan, dtype=np.float32)
    whit_scene = np.full((shape[0], shape[1], N_PCS_USED), np.nan, dtype=np.float32)
    col_std = Xp[:, :N_PCS_USED].std(axis=0, ddof=1)
    Xp_whit = Xp[:, :N_PCS_USED] / col_std
    raw_scene[valid] = Xp[:, :N_PCS_USED].astype(np.float32)
    whit_scene[valid] = Xp_whit.astype(np.float32)
    np.save(cube_root(cube_id) / "arrays" / f"{cube_id}_PC_scores_raw_v4.npy", raw_scene)
    np.save(cube_root(cube_id) / "arrays" / f"{cube_id}_PC_scores_whitened_v4.npy", whit_scene)

    # GMM k=6 on whitened
    gmm = GaussianMixture(
        n_components=K_FIXED, covariance_type="full",
        n_init=5, random_state=RANDOM_SEED, reg_covar=1e-5,
    )
    gmm.fit(Xp_whit)
    raw_labels = gmm.predict(Xp_whit).astype(np.int16)
    bic = float(gmm.bic(Xp_whit))
    aic = float(gmm.aic(Xp_whit))
    sizes = [int((raw_labels == c).sum()) for c in range(K_FIXED)]
    print(f"  [gmm]  sizes={sizes}  BIC={bic:.0f}  AIC={aic:.0f}")

    # Save unaligned labels back-projected to (nrows, ncols)
    raw_label_scene = np.full(shape, -1, dtype=np.int16)
    raw_label_scene[valid] = raw_labels
    np.save(cube_root(cube_id) / "arrays" / f"{cube_id}_GMM_k6_labels_unaligned_v4.npy",
            raw_label_scene)

    log["pca"] = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": cum.tolist(),
        "cum_at_pc5": float(cum[4]),
        "n_pcs_for_90pct": n_pcs_for_90,
        "n_pcs_table": n_pcs_table,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "components": pca.components_.tolist(),
    }
    log["gmm"] = {"sizes": sizes, "bic": bic, "aic": aic}
    return {
        "scaler": scaler, "pca": pca,
        "X": X, "Xs": Xs, "Xp": Xp, "Xp_whit": Xp_whit,
        "raw_labels": raw_labels, "n_pcs_table": n_pcs_table,
    }


# =============================================================================
# Step 3: Per-cube PCA / GMM figures (loadings, scree, pair matrix, GMM scene)
# =============================================================================
def fig_loadings(cube_id: str, pca, n_pcs: int, evr) -> Path:
    comps = pca.components_[:n_pcs]
    fig, ax = plt.subplots(figsize=(11, 0.7 * n_pcs + 1.2),
                           constrained_layout=True)
    vmax = float(np.max(np.abs(comps)))
    im = ax.imshow(comps, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(V4_VB_PARAM_ORDER)))
    ax.set_xticklabels(V4_VB_PARAM_ORDER, rotation=45, ha="right")
    ax.set_yticks(range(n_pcs))
    ax.set_yticklabels([f"PC{i+1} ({evr[i]*100:.1f}%)" for i in range(n_pcs)])
    for i in range(n_pcs):
        for j in range(len(V4_VB_PARAM_ORDER)):
            ax.text(j, i, f"{comps[i, j]:+.3f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if abs(comps[i, j]) > 0.5 * vmax else "black")
    plt.colorbar(im, ax=ax, label="loading")
    ax.set_title(f"{cube_id}  v4 PCA loadings (PC1-{n_pcs})")
    p = cube_root(cube_id) / "figures" / f"{cube_id}_PCA_loadings_v4.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # CSV companion
    csv_p = cube_root(cube_id) / "tables" / f"{cube_id}_PCA_loadings_v4.csv"
    with open(csv_p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["PC", "explained_var_pct"] + list(V4_VB_PARAM_ORDER))
        for i in range(n_pcs):
            w.writerow([f"PC{i+1}", f"{evr[i]*100:.3f}"]
                       + [f"{x:.4f}" for x in comps[i]])
    return p


def fig_scree(cube_id: str, pca, n_show: int = 10) -> Path:
    evr = pca.explained_variance_ratio_
    n = min(n_show, len(evr))
    cum = np.cumsum(evr[:n])
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.bar(range(1, n + 1), evr[:n] * 100, color="tab:blue", alpha=0.8,
           label="per-PC variance")
    ax.plot(range(1, n + 1), cum * 100, marker="o", color="tab:red",
            label="cumulative")
    ax.axhline(90, color="gray", ls="--", lw=0.7, label="90% threshold")
    ax.set_xlabel("PC index")
    ax.set_ylabel("explained variance (%)")
    ax.set_title(f"{cube_id}  v4 PCA scree (PC1-{n})")
    ax.set_xticks(range(1, n + 1))
    ax.legend(loc="center right")
    ax.grid(alpha=0.3)
    p = cube_root(cube_id) / "figures" / f"{cube_id}_PCA_scree_v4.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_pair_matrix(cube_id: str, Xp, evr) -> Path:
    n = N_PCS_USED
    fig, axes = plt.subplots(n, n, figsize=(15, 13), constrained_layout=True)
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.hist(Xp[:, i], bins=80, color="tab:blue", alpha=0.85)
                ax.set_title(f"PC{i+1}  {evr[i]*100:.1f}%", fontsize=9)
            else:
                ax.hexbin(Xp[:, j], Xp[:, i], gridsize=60, cmap="viridis",
                          mincnt=1, norm=LogNorm())
                ax.set_title(f"PC{j+1} ({evr[j]*100:.1f}%) vs PC{i+1} ({evr[i]*100:.1f}%)",
                             fontsize=8)
            if i == n - 1:
                ax.set_xlabel(f"PC{j+1}", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"PC{i+1}", fontsize=8)
    fig.suptitle(f"{cube_id}  v4 PCA pair matrix (PC1-{n}, hexbin log)",
                 fontsize=12)
    p = cube_root(cube_id) / "figures" / f"{cube_id}_PCA_pairmatrix_v4.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_gmm_scene(cube_id: str, label_scene: np.ndarray, sizes, bic) -> Path:
    disp = apply_display_transform(label_scene.astype(np.int16))
    show = np.where(disp >= 0, disp, np.nan)
    cmap = ListedColormap(GMM_PALETTE[:K_FIXED])
    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    ax.imshow(show, cmap=cmap, vmin=0, vmax=K_FIXED - 1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(
        f"{cube_id}  GMM k=6 on whitened PC1-5 (PC/std)\n"
        f"sizes: {sizes}   |   BIC: {bic:.0f}",
        fontsize=10,
    )
    p = cube_root(cube_id) / "figures" / f"{cube_id}_GMM_k6_scene_v4.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# =============================================================================
# Step 4: Brown filter + RGB + grayscale + scatter (per cube)
# =============================================================================
def brown_eq2_filter(brown: dict, residual_p95: float | None = None) -> tuple[np.ndarray, dict]:
    """Brown 2020 eq. 2: 0.1 < amplitude < 1.0 AND 1.15 < centroid < 1.5
    AND HWHM < 0.3 AND 0.8 < asymmetry < 9. Plus residual <= p95."""
    centroid = brown["centroid"].astype(np.float32)
    amplitude = brown["amplitude"].astype(np.float32)
    hwhm = brown["hwhm"].astype(np.float32)
    asym = brown["asymmetry"].astype(np.float32)
    residual = brown["residual"].astype(np.float32)

    finite = (np.isfinite(centroid) & np.isfinite(amplitude)
              & np.isfinite(hwhm) & np.isfinite(asym))
    eq2 = (
        (amplitude > 0.10) & (amplitude < 1.00)
        & (centroid > 1.15) & (centroid < 1.50)
        & (hwhm < 0.30)
        & (asym > 0.80) & (asym < 9.00)
    )
    valid = finite & eq2
    if residual_p95 is None:
        if valid.any():
            residual_p95 = float(np.nanpercentile(residual[valid], 95))
        else:
            residual_p95 = float("inf")
    valid = valid & (residual <= residual_p95)
    info = {
        "n_finite_brown": int(finite.sum()),
        "n_passing_eq2": int(eq2.sum()),
        "n_after_residual": int(valid.sum()),
        "residual_p95_threshold": residual_p95,
    }
    return valid, info


def fig_brown_grayscale(cube_id: str, brown: dict, valid: np.ndarray) -> list[Path]:
    paths = []
    for name in ("centroid", "amplitude", "hwhm", "asymmetry"):
        arr = brown[name].astype(np.float32)
        masked = np.where(valid, arr, np.nan)
        disp = apply_display_transform(masked)
        lo, hi = BROWN_STRETCH[name]
        fig, ax = plt.subplots(figsize=(8, 7.0), constrained_layout=True)
        im = ax.imshow(disp, cmap="viridis", vmin=lo, vmax=hi,
                       interpolation="nearest")
        unit = " (um)" if name in ("centroid", "hwhm") else ""
        plt.colorbar(im, ax=ax, label=f"{name}{unit}", shrink=0.85)
        ax.set_title(f"{cube_id}  Brown {name}  stretch: {lo:.2f}-{hi:.2f}{unit}",
                     fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        # Footnote: data-driven matched stretch + asymmetry saturation note
        footnote = BROWN_STRETCH_NOTE
        if name == "asymmetry":
            footnote = footnote + " " + ASYMMETRY_SAT_NOTE
        fig.text(0.5, 0.005, footnote, ha="center", va="bottom",
                 fontsize=7, color="#444444", wrap=True)
        suffix = {"centroid": "centroid", "amplitude": "amplitude",
                  "hwhm": "HWHM", "asymmetry": "asymmetry"}[name]
        p = cube_root(cube_id) / "figures" / f"{cube_id}_Brown_{suffix}_v4.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths


def make_brown_rgb(brown: dict, valid: np.ndarray, channels: list[str]):
    """Build an HxWx4 RGBA stack from three Brown channels with per-channel
    fixed stretches per BROWN_STRETCH. Invalid pixels -> alpha=0."""
    h, w = brown["centroid"].shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for k, name in enumerate(channels):
        lo, hi = BROWN_STRETCH[name]
        rgba[..., k] = percentile_norm(np.where(valid, brown[name].astype(np.float32),
                                                np.nan), lo, hi)
    rgba[..., 3] = valid.astype(np.float32)
    nanmask = np.isnan(rgba[..., 0]) | np.isnan(rgba[..., 1]) | np.isnan(rgba[..., 2])
    for k in range(3):
        rgba[..., k] = np.where(nanmask, 0.0, rgba[..., k])
    rgba[nanmask, 3] = 0.0
    return rgba


def fig_brown_rgb(cube_id: str, brown: dict, valid: np.ndarray,
                  channels: list[str], suffix: str) -> Path:
    rgba = make_brown_rgb(brown, valid, channels)
    disp = apply_display_transform(rgba)
    fig, ax = plt.subplots(figsize=(8.5, 7.0), constrained_layout=True)
    ax.imshow(disp, interpolation="nearest")
    label_map = {"centroid": "centroid", "amplitude": "amplitude",
                 "hwhm": "HWHM", "asymmetry": "asymmetry"}
    ranges = " | ".join(
        f"{label_map[c]}: {BROWN_STRETCH[c][0]:.2f}-{BROWN_STRETCH[c][1]:.2f}"
        for c in channels
    )
    ax.set_title(
        f"{cube_id}  Brown RGB = {label_map[channels[0]]} / "
        f"{label_map[channels[1]]} / {label_map[channels[2]]}\n{ranges}",
        fontsize=10,
    )
    ax.set_xticks([]); ax.set_yticks([])
    footnote = BROWN_STRETCH_NOTE
    if "asymmetry" in channels:
        footnote = footnote + " " + ASYMMETRY_SAT_NOTE
    fig.text(0.5, 0.005, footnote, ha="center", va="bottom",
             fontsize=7, color="#444444", wrap=True)
    p = cube_root(cube_id) / "figures" / f"{cube_id}_Brown_RGB_{suffix}_v4.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_brown_centroid_asymmetry(cube_id: str, brown: dict, valid: np.ndarray) -> Path:
    centroid = brown["centroid"].astype(np.float32)[valid]
    asym = brown["asymmetry"].astype(np.float32)[valid]
    fig, ax = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
    hb = ax.hexbin(centroid, asym, gridsize=80, cmap="viridis",
                   mincnt=1, norm=LogNorm(),
                   extent=(1.05, 1.50, 0.8, 9.0))
    plt.colorbar(hb, ax=ax, label="pixel density (log)")
    # Fo-group region annotations per Brown 2020 Figure 16
    ax.axvspan(1.05, 1.20, alpha=0.05, color="green")  # high Fo (>66)
    ax.axvspan(1.20, 1.30, alpha=0.05, color="orange")  # med Fo (40-66)
    ax.axvspan(1.30, 1.50, alpha=0.05, color="red")  # low Fo (<40)
    ax.text(1.125, 8.5, "high Fo\n(>66)", ha="center", va="top", fontsize=8,
            color="green", alpha=0.9)
    ax.text(1.25, 8.5, "med Fo\n(40-66)", ha="center", va="top", fontsize=8,
            color="darkorange", alpha=0.9)
    ax.text(1.40, 8.5, "low Fo\n(<40)", ha="center", va="top", fontsize=8,
            color="red", alpha=0.9)
    ax.set_xlabel("centroid (um)")
    ax.set_ylabel("asymmetry")
    ax.set_xlim(1.05, 1.50)
    ax.set_ylim(0.8, 9.0)
    ax.set_title(f"{cube_id}  Brown centroid vs asymmetry", fontsize=11)
    ax.grid(alpha=0.2)
    p = cube_root(cube_id) / "figures" / f"{cube_id}_Brown_centroid_asymmetry_v4.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# =============================================================================
# Step 5: Hungarian alignment across cubes (separate function, runs after all
# cubes have raw labels written)
# =============================================================================
def hungarian_align_all(per_cube: dict) -> dict:
    """Align each cube's k=6 GMM clusters to FRT00003E12 in 11-param VB-only
    standardized space. Reorder so cluster 0 = most-mafic-cap on 3E12."""
    REF = "FRT00003E12"
    ref = per_cube[REF]

    # Standardize each cube's VB stack independently and compute per-cluster
    # centroids in that cube's standardized space.
    def cluster_centroids_std(art, raw_labels):
        valid = art["valid"]
        X = np.column_stack([art["vb"][p][valid] for p in V4_VB_PARAM_ORDER]).astype(np.float64)
        Xs = StandardScaler().fit_transform(X)
        ctr = np.full((K_FIXED, X.shape[1]), np.nan)
        for c in range(K_FIXED):
            m = raw_labels == c
            if m.any():
                ctr[c] = Xs[m].mean(axis=0)
        return ctr

    # Mafic-cap order on the reference
    ref_ctr = cluster_centroids_std(ref["art"], ref["raw_labels"])
    mafic_idx = [V4_VB_PARAM_ORDER.index(p) for p in MAFIC_CAP_PARAMS]
    lcp_idx = V4_VB_PARAM_ORDER.index("LCPINDEX2")
    score = ref_ctr[:, mafic_idx].sum(axis=1)
    # Tie-break on LCPINDEX2 (prefer cluster with HIGHER LCPINDEX2 if tied)
    sort_order = np.lexsort((-ref_ctr[:, lcp_idx], score))  # primary: score asc; tie-break: lcp desc
    ref_relabel = np.argsort(sort_order)
    ref_aligned = np.array([ref_relabel[c] for c in ref["raw_labels"]], dtype=np.int16)
    ref_ctr_aligned = ref_ctr[sort_order]  # already in aligned order

    # Save reference aligned label scene
    ref_scene = np.full(ref["art"]["shape"], -1, dtype=np.int16)
    ref_scene[ref["art"]["valid"]] = ref_aligned
    np.save(cube_root(REF) / "arrays" / f"{REF}_GMM_k6_labels_aligned_v4.npy",
            ref_scene)

    alignment_log = {
        REF: {
            "permutation": ref_relabel.tolist(),
            "distances_to_ref": [0.0] * K_FIXED,
        }
    }
    aligned_labels_by_cube = {REF: ref_aligned}

    for cube_id in CUBE_IDS:
        if cube_id == REF:
            continue
        art = per_cube[cube_id]["art"]
        raw_labels = per_cube[cube_id]["raw_labels"]
        ctr = cluster_centroids_std(art, raw_labels)
        D = cdist(ctr, ref_ctr_aligned)
        row_ind, col_ind = linear_sum_assignment(D)
        # row_ind[i]=i in order; col_ind[i] = aligned ref label that
        # maps to other cluster i. So relabel[i] = col_ind[i].
        relabel = col_ind.copy()
        aligned = np.array([relabel[c] for c in raw_labels], dtype=np.int16)
        scene = np.full(art["shape"], -1, dtype=np.int16)
        scene[art["valid"]] = aligned
        np.save(cube_root(cube_id) / "arrays" / f"{cube_id}_GMM_k6_labels_aligned_v4.npy",
                scene)
        alignment_log[cube_id] = {
            "permutation": relabel.tolist(),
            "distances_to_ref": D[row_ind, col_ind].tolist(),
            "full_distance_matrix": D.tolist(),
        }
        aligned_labels_by_cube[cube_id] = aligned

    out_path = V4_ROOT / "reports" / "03_v4_alignment.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(alignment_log, indent=2), encoding="utf-8")
    return aligned_labels_by_cube


# =============================================================================
# Step 6: Cross-cube comparison plots
# =============================================================================
def fig_centroid_per_param(per_cube: dict, aligned_by_cube: dict,
                           brown_valid_by_cube: dict) -> Path:
    """Section 5.2: 15 panels (11 VB + 4 Brown) of aligned-cluster medians."""
    vb_params = list(V4_VB_PARAM_ORDER)
    brown_params = ["centroid", "asymmetry", "amplitude", "hwhm"]
    all_params = vb_params + brown_params  # 15 panels
    nrows, ncols = 3, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 12), constrained_layout=True)

    for ax, param in zip(axes.ravel(), all_params):
        is_brown = param in brown_params
        for cube_id in CUBE_IDS:
            art = per_cube[cube_id]["art"]
            valid = art["valid"]
            aligned = aligned_by_cube[cube_id]
            if is_brown:
                arr_full = art["brown"][param].astype(np.float32)
                brown_v = brown_valid_by_cube[cube_id]
                row_brown_valid = brown_v[valid]  # at PCA-valid positions
                ys = []
                ns = []
                for c in range(K_FIXED):
                    cluster_mask = aligned == c
                    inter = cluster_mask & row_brown_valid
                    n = int(inter.sum())
                    ns.append(n)
                    if n == 0:
                        ys.append(np.nan)
                    else:
                        ys.append(float(np.nanmedian(arr_full[valid][inter])))
                # Hollow markers for clusters with < 100 Brown-valid pixels
                facecolors = [CUBE_COLOURS[cube_id] if n >= 100 else "none" for n in ns]
                for c, (yv, fc, n) in enumerate(zip(ys, facecolors, ns)):
                    if not np.isnan(yv):
                        ax.scatter(c, yv, marker=CUBE_MARKERS[cube_id],
                                   facecolors=fc,
                                   edgecolors=CUBE_COLOURS[cube_id], s=55,
                                   linewidths=1.4, zorder=3)
                # Connecting line through non-NaN points
                xs_line = [c for c, y in enumerate(ys) if not np.isnan(y)]
                ys_line = [y for y in ys if not np.isnan(y)]
                ax.plot(xs_line, ys_line, color=CUBE_COLOURS[cube_id], lw=1.0,
                        alpha=0.7, label=cube_id)
            else:
                arr = art["vb"][param][valid]
                ys = []
                for c in range(K_FIXED):
                    cluster_mask = aligned == c
                    if cluster_mask.any():
                        ys.append(float(np.nanmedian(arr[cluster_mask])))
                    else:
                        ys.append(np.nan)
                ax.plot(range(K_FIXED), ys, marker=CUBE_MARKERS[cube_id],
                        color=CUBE_COLOURS[cube_id], lw=1.4, label=cube_id)
        ax.set_title(param + ("  (Brown)" if is_brown else ""), fontsize=10)
        ax.set_xticks(range(K_FIXED))
        ax.set_xlabel("aligned cluster")
        ax.grid(alpha=0.3)

    # Single legend in top-left first panel
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle("v4 aligned k=6 cluster medians per parameter "
                 "(11 VB + 4 Brown; Brown panels: hollow marker if <100 Brown-valid pixels)",
                 fontsize=12)
    p = V4_ROOT / "figures" / "comparison_aligned_k6_centroid_per_param_v4.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_median_subtracted(per_cube: dict, brown_valid_by_cube: dict) -> Path:
    """Section 5.3: 15-panel median-subtracted distributions per parameter."""
    vb_params = list(V4_VB_PARAM_ORDER)
    brown_params = ["centroid", "asymmetry", "amplitude", "hwhm"]
    all_params = vb_params + brown_params
    nrows, ncols = 3, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 12), constrained_layout=True)

    for ax, param in zip(axes.ravel(), all_params):
        is_brown = param in brown_params
        # Pool all 4-cube subtracted distributions to compute global x-range
        pooled_sub = []
        per_cube_sub = {}
        for cube_id in CUBE_IDS:
            art = per_cube[cube_id]["art"]
            if is_brown:
                arr = art["brown"][param].astype(np.float32)
                bv = brown_valid_by_cube[cube_id]
                a = arr[bv]
            else:
                a = art["vb"][param][art["valid"]].astype(np.float32)
            a = a[np.isfinite(a)]
            if a.size < 50:
                per_cube_sub[cube_id] = (None, np.nan)
                continue
            med = float(np.median(a))
            sub = a - med
            per_cube_sub[cube_id] = (sub, med)
            pooled_sub.append(sub)
        if not pooled_sub:
            ax.set_visible(False)
            continue
        pooled = np.concatenate(pooled_sub)
        lo, hi = float(np.percentile(pooled, 0.1)), float(np.percentile(pooled, 99.9))
        for cube_id in CUBE_IDS:
            sub, med = per_cube_sub[cube_id]
            if sub is None:
                continue
            ax.hist(sub, bins=120, range=(lo, hi), histtype="step",
                    density=True, color=CUBE_COLOURS[cube_id], lw=1.3,
                    label=f"{cube_id} (med={med:+.3g})")
        ax.set_title(param + ("  (Brown)" if is_brown else ""), fontsize=10)
        ax.set_xlabel(f"{param} - per-cube median")
        ax.grid(alpha=0.3)

    axes[0, 0].legend(loc="best", fontsize=7)
    fig.suptitle(
        "v4 four-cube median-subtracted distributions (density, 0.1-99.9% range)",
        fontsize=12,
    )
    p = V4_ROOT / "figures" / "comparison_all_cubes_median_subtracted_v4.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_loadings_4cube(per_cube: dict) -> Path:
    """Section 5.4: 2x2 loadings heatmaps with shared RdBu_r scale, fixed PC1-5."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 10), constrained_layout=True)
    all_loadings = []
    cum5_per_cube = {}
    for cube_id in CUBE_IDS:
        comps = per_cube[cube_id]["pca_data"].components_[:N_PCS_USED]
        all_loadings.append(comps)
        cum5_per_cube[cube_id] = float(np.sum(per_cube[cube_id]["pca_data"].explained_variance_ratio_[:N_PCS_USED]))
    vmax = max(np.max(np.abs(c)) for c in all_loadings)
    for ax, cube_id, comps in zip(axes.ravel(), CUBE_IDS, all_loadings):
        evr = per_cube[cube_id]["pca_data"].explained_variance_ratio_
        im = ax.imshow(comps, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(V4_VB_PARAM_ORDER)))
        ax.set_xticklabels(V4_VB_PARAM_ORDER, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(N_PCS_USED))
        ax.set_yticklabels([f"PC{i+1} ({evr[i]*100:.1f}%)" for i in range(N_PCS_USED)],
                           fontsize=8)
        for i in range(N_PCS_USED):
            for j in range(len(V4_VB_PARAM_ORDER)):
                ax.text(j, i, f"{comps[i, j]:+.2f}", ha="center", va="center",
                        fontsize=6,
                        color="white" if abs(comps[i, j]) > 0.5 * vmax else "black")
        ax.set_title(f"{cube_id}  cum5={cum5_per_cube[cube_id]*100:.1f}%",
                     fontsize=10)
    fig.suptitle("v4 four-cube PCA loadings (PC1-5, shared scale)", fontsize=13)
    fig.colorbar(im, ax=axes, shrink=0.6, label="loading")
    p = V4_ROOT / "figures" / "comparison_PCA_loadings_4cube_v4.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_gmm_4cube(per_cube: dict, aligned_by_cube: dict) -> Path:
    """Section 5.5: 2x2 grid of GMM scenes using aligned cluster colors."""
    cmap = ListedColormap(GMM_PALETTE[:K_FIXED])
    fig, axes = plt.subplots(2, 2, figsize=(18, 13), constrained_layout=True)
    for ax, cube_id in zip(axes.ravel(), CUBE_IDS):
        art = per_cube[cube_id]["art"]
        aligned = aligned_by_cube[cube_id]
        scene = np.full(art["shape"], -1, dtype=np.int16)
        scene[art["valid"]] = aligned
        disp = apply_display_transform(scene)
        show = np.where(disp >= 0, disp, np.nan)
        ax.imshow(show, cmap=cmap, vmin=0, vmax=K_FIXED - 1, interpolation="nearest")
        ax.set_title(cube_id, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("v4 four-cube GMM k=6 scene maps (aligned colors)", fontsize=13)
    p = V4_ROOT / "figures" / "comparison_GMM_k6_4cube_v4.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_brown_rgb_4cube(per_cube: dict, brown_valid_by_cube: dict,
                         channels: list[str], suffix: str, label: str) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(18, 13.5), constrained_layout=True)
    for ax, cube_id in zip(axes.ravel(), CUBE_IDS):
        rgba = make_brown_rgb(per_cube[cube_id]["art"]["brown"],
                              brown_valid_by_cube[cube_id], channels)
        disp = apply_display_transform(rgba)
        ax.imshow(disp, interpolation="nearest")
        ax.set_title(cube_id, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    label_map = {"centroid": "centroid", "amplitude": "amplitude",
                 "hwhm": "HWHM", "asymmetry": "asymmetry"}
    ranges = " | ".join(
        f"{label_map[c]}: {BROWN_STRETCH[c][0]:.2f}-{BROWN_STRETCH[c][1]:.2f}"
        for c in channels
    )
    fig.suptitle(f"v4 four-cube Brown RGB ({label})  [{ranges}]", fontsize=12)
    footnote = BROWN_STRETCH_NOTE
    if "asymmetry" in channels:
        footnote = footnote + " " + ASYMMETRY_SAT_NOTE
    fig.text(0.5, 0.003, footnote, ha="center", va="bottom",
             fontsize=8, color="#444444", wrap=True)
    p = V4_ROOT / "figures" / f"comparison_Brown_RGB_{suffix}_4cube_v4.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


# =============================================================================
# Per-cube driver
# =============================================================================
def process_cube(cube_id: str) -> dict:
    t0 = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"\n=== {cube_id} v4 ===")
    log = {
        "cube_id": cube_id, "stage": "v4_vb_pca_brown_rgb",
        "run_timestamp_utc": timestamp,
        "random_seed": RANDOM_SEED,
        "vb_param_order": list(V4_VB_PARAM_ORDER),
        "param_nan_frac": {},
    }
    art = build_vb_stack(cube_id, log)
    pca_pack = run_pca_gmm(cube_id, art, log)

    # Per-cube figures
    n_pcs_table = log["pca"]["n_pcs_table"]
    fig_loadings(cube_id, pca_pack["pca"], n_pcs_table,
                 pca_pack["pca"].explained_variance_ratio_)
    fig_scree(cube_id, pca_pack["pca"], n_show=10)
    fig_pair_matrix(cube_id, pca_pack["Xp"], pca_pack["pca"].explained_variance_ratio_)

    # Brown filter + per-cube Brown figures
    brown_valid, binfo = brown_eq2_filter(art["brown"])
    log["brown_filter"] = binfo
    np.save(cube_root(cube_id) / "arrays" / f"{cube_id}_brown_valid_mask_v4.npy",
            brown_valid)
    fig_brown_grayscale(cube_id, art["brown"], brown_valid)
    fig_brown_rgb(cube_id, art["brown"], brown_valid,
                  ["centroid", "amplitude", "hwhm"], "centroid_amp_hwhm")
    fig_brown_rgb(cube_id, art["brown"], brown_valid,
                  ["centroid", "amplitude", "asymmetry"], "centroid_amp_asym")
    fig_brown_centroid_asymmetry(cube_id, art["brown"], brown_valid)

    log["wall_clock_seconds"] = round(time.time() - t0, 2)
    log["source_code_md5"] = {
        "stage_v4_vb_pca_and_brown_rgb.py": md5_of(Path(__file__)),
        "vb_parameters.py": md5_of(Path(__file__).with_name("vb_parameters.py")),
        "config.py": md5_of(Path(__file__).with_name("config.py")),
        "crism_io.py": md5_of(Path(__file__).with_name("crism_io.py")),
        "display.py": md5_of(Path(__file__).with_name("display.py")),
    }
    log_path = cube_root(cube_id) / "logs" / f"{cube_id}_params_v4.json"
    log_path.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")

    # Run log txt
    txt = cube_root(cube_id) / "logs" / f"{cube_id}_run_log_v4.txt"
    lines = [
        f"{cube_id} v4 run log",
        f"timestamp: {timestamp}",
        f"random seed: {RANDOM_SEED}",
        "",
        f"VB-only mask: {log['n_pixels_valid']}/{log['n_pixels_total']} pixels "
        f"({log['valid_fraction']*100:.2f}%)",
        f"BDI1000VIS NaN frac: {log['param_nan_frac']['BDI1000VIS']:.4f}",
        f"SH600_2 median: {log['sh600_2_median']:+.4f}  "
        f"warning: {log['sh600_2_warning'] or 'none'}",
        "",
        f"PCA cumulative variance at PC5: {log['pca']['cum_at_pc5']*100:.2f}%",
        f"PCs needed to reach >=90%: {log['pca']['n_pcs_for_90pct']}",
        f"PCs in loadings table: {log['pca']['n_pcs_table']}",
        "",
        f"GMM k=6 (whitened) BIC: {log['gmm']['bic']:.0f}  AIC: {log['gmm']['aic']:.0f}",
        f"GMM raw cluster sizes: {log['gmm']['sizes']}",
        "",
        f"Brown finite: {binfo['n_finite_brown']}",
        f"Brown passing eq.2: {binfo['n_passing_eq2']}",
        f"Brown after residual <= p95 ({binfo['residual_p95_threshold']:.4f}): "
        f"{binfo['n_after_residual']}",
    ]
    txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"  done in {log['wall_clock_seconds']}s -> {log_path.name}")
    return {
        "art": art, "pca_pack": pca_pack,
        "raw_labels": pca_pack["raw_labels"],
        "pca_data": pca_pack["pca"],
        "brown_valid": brown_valid,
        "log": log,
    }


def regenerate_brown_only(cube_ids: list[str]) -> dict:
    """v4.1 helper: regenerate Brown figures (per-cube + cross-cube) using
    cached brown_valid_mask_v4.npy from arrays/. Skips TER3 cube load,
    VB compute, PCA, GMM. Useful for restretch passes."""
    import json as _json
    per_cube = {}
    for cube_id in cube_ids:
        print(f"  [brown-only] {cube_id}: loading Brown maps...")
        _, brown = find_cube_and_brown_maps(
            __import__("config").cube_dir(cube_id))
        bv_path = (cube_root(cube_id) / "arrays"
                   / f"{cube_id}_brown_valid_mask_v4.npy")
        if not bv_path.exists():
            raise SystemExit(
                f"Missing {bv_path} - run the full pipeline at least once "
                "before --brown-only")
        brown_valid = np.load(bv_path)
        # Per-cube Brown figures
        fig_brown_grayscale(cube_id, brown, brown_valid)
        fig_brown_rgb(cube_id, brown, brown_valid,
                      ["centroid", "amplitude", "hwhm"], "centroid_amp_hwhm")
        fig_brown_rgb(cube_id, brown, brown_valid,
                      ["centroid", "amplitude", "asymmetry"], "centroid_amp_asym")
        # Centroid-asymmetry scatter X-axis stays 1.05-1.50 per spec
        fig_brown_centroid_asymmetry(cube_id, brown, brown_valid)
        per_cube[cube_id] = {
            "art": {"brown": brown}, "brown_valid": brown_valid,
        }
    # Cross-cube panels
    if set(cube_ids) == set(CUBE_IDS):
        brown_valid_by_cube = {c: per_cube[c]["brown_valid"] for c in CUBE_IDS}
        fig_brown_rgb_4cube(per_cube, brown_valid_by_cube,
                            ["centroid", "amplitude", "hwhm"],
                            "centroid_amp_hwhm",
                            "composition / abundance / texture")
        fig_brown_rgb_4cube(per_cube, brown_valid_by_cube,
                            ["centroid", "amplitude", "asymmetry"],
                            "centroid_amp_asym",
                            "composition / abundance / asymmetry")
    return per_cube


def main(argv: list[str]) -> None:
    args = list(argv[1:])
    brown_only = False
    if "--brown-only" in args:
        brown_only = True
        args.remove("--brown-only")
    targets = args if args else list(CUBE_IDS)
    if any(t not in CUBE_IDS for t in targets):
        raise SystemExit(f"unknown cube id(s); known: {CUBE_IDS}")
    if brown_only:
        print(f"=== v4.1 Brown-only restretch on {targets} ===")
        regenerate_brown_only(targets)
        print("Brown-only regeneration complete.")
        return
    if any(t == cid for t in targets for cid in CUBE_IDS) and len(targets) < len(CUBE_IDS):
        print(f"NOTE: cross-cube comparison plots require all four cubes; "
              f"running per-cube only for {targets}")

    per_cube = {}
    for cube_id in targets:
        per_cube[cube_id] = process_cube(cube_id)

    # If we ran all four, do the cross-cube products
    if set(targets) == set(CUBE_IDS):
        print("\n=== cross-cube alignment + comparison plots ===")
        aligned_by_cube = hungarian_align_all(per_cube)
        brown_valid_by_cube = {c: per_cube[c]["brown_valid"] for c in CUBE_IDS}
        # GMM scenes per cube using aligned labels
        for cube_id in CUBE_IDS:
            art = per_cube[cube_id]["art"]
            scene = np.full(art["shape"], -1, dtype=np.int16)
            scene[art["valid"]] = aligned_by_cube[cube_id]
            sizes = [int((aligned_by_cube[cube_id] == c).sum())
                     for c in range(K_FIXED)]
            bic = per_cube[cube_id]["log"]["gmm"]["bic"]
            fig_gmm_scene(cube_id, scene, sizes, bic)
        fig_centroid_per_param(per_cube, aligned_by_cube, brown_valid_by_cube)
        fig_median_subtracted(per_cube, brown_valid_by_cube)
        fig_loadings_4cube(per_cube)
        fig_gmm_4cube(per_cube, aligned_by_cube)
        fig_brown_rgb_4cube(per_cube, brown_valid_by_cube,
                             ["centroid", "amplitude", "hwhm"], "centroid_amp_hwhm",
                             "composition / abundance / texture")
        fig_brown_rgb_4cube(per_cube, brown_valid_by_cube,
                             ["centroid", "amplitude", "asymmetry"], "centroid_amp_asym",
                             "composition / abundance / asymmetry")
        print("cross-cube plots done")


if __name__ == "__main__":
    main(sys.argv)
