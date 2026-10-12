"""Stage v9 - 3E12-seeded GMM for cross-cube comparison.

Replaces the per-cube fit + Hungarian alignment of v7-C / v8 with a
single GMM fit on FRT00003E12 and applied unchanged to the other
three cubes. Pixels that don't match any 3E12-defined class are
flagged as 'novel' and rendered neutral gray rather than forced into
the nearest cluster.

Two parallel runs:
  v9-A (control): 5 params - OLINDEX3, BD1400, BD1900R2, D2300, BD2500_2
  v9-B (primary): 7 params - v9-A + Brown centroid + asymmetry

Output: outputs/_v9_3e12_seeded/{reference, v9a_5param, v9b_7param, reports}/

See PLAN_v9.md for the full spec and rationale.
"""

from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from config import CUBE_IDS, RANDOM_SEED, OUTPUTS_DIR
from stage_v7c_mtr3_pipeline import (
    load_mtr3, write_float_geotiff,
    MTR3_CUBE_DIRS, CUBES_DIR, K_FIXED, REFERENCE,
)
from stage_v8_mtr3_with_brown import load_brown_v6
from vb_parameters import (
    calc_olindex3, calc_bd1400, calc_bd1900r2, calc_d2300, calc_bd2500_2,
)

# =============================================================================
# Run configurations
# =============================================================================
RUN_CONFIGS = {
    "v9a_5param": {
        "tag": "v9a",
        "params": ("OLINDEX3", "BD1400", "BD1900R2", "D2300", "BD2500_2"),
        "d": 5,
        "chi2_crit": float(chi2.ppf(0.95, df=5)),  # 11.0705
    },
    "v9b_7param": {
        "tag": "v9b",
        "params": ("centroid", "asymmetry",
                   "OLINDEX3", "BD1400", "BD1900R2", "D2300", "BD2500_2"),
        "d": 7,
        "chi2_crit": float(chi2.ppf(0.95, df=7)),  # 14.0671
    },
}

VB_FUNCS = {
    "OLINDEX3": calc_olindex3,
    "BD1400":   calc_bd1400,
    "BD1900R2": calc_bd1900r2,
    "D2300":    calc_d2300,
    "BD2500_2": calc_bd2500_2,
}
BROWN_PARAMS = ("centroid", "asymmetry")
MAFIC_CAP_PARAMS = ("BD1900R2", "D2300", "BD2500_2", "OLINDEX3")

MAX_POSTERIOR_THRESHOLD = 0.6
N_INIT = 10
REG_COVAR = 1e-5

SNAPSHOT = OUTPUTS_DIR / "_v9_3e12_seeded"
NOVEL_LABEL = 254       # encoded into composite GeoTIFF for novel pixels
INVALID_LABEL = 255     # encoded into composite GeoTIFF for invalid pixels

# Same 6-class palette as v7-C / v8; semantic meaning fixed by mafic-cap
# z-sum sort on 3E12 (cluster 0 = lowest alteration index, cluster 5 = highest).
GMM_PALETTE_RGB = [
    (31, 119, 180),    # 0 - tab:blue       (least altered / mafic cap)
    (255, 127, 14),    # 1 - tab:orange
    (44, 160, 44),     # 2 - tab:green
    (214, 39, 40),     # 3 - tab:red
    (148, 103, 189),   # 4 - tab:purple
    (140, 86, 75),     # 5 - tab:brown      (most altered / olivine-carbonate)
]
NOVEL_RGB = (160, 160, 160)  # neutral gray
SUBCLUSTER_PALETTE_RGB = [
    (252, 141, 98),    # warm orange
    (141, 160, 203),   # cool blue
    (231, 138, 195),   # pink
]


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# Param stack construction (shared by reference fit and non-ref predict)
# =============================================================================
def build_param_stack(cube_id: str, param_names: tuple[str, ...]) -> dict:
    """Compute / load the parameter stack for one cube.

    Returns dict with keys: params (name -> 2D array), valid_mask (HxW),
    H, W, geo, timestamp. Cube buffer is freed before return.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"  [{cube_id}] loading MTR3 cube...")
    dir_name = MTR3_CUBE_DIRS[cube_id]
    hdr_path = CUBES_DIR / dir_name / f"{dir_name}.hdr"
    cube_bls, wav, meta = load_mtr3(hdr_path)
    H, W = meta["lines"], meta["samples"]

    needs_vb = any(p in VB_FUNCS for p in param_names)
    needs_brown = any(p in BROWN_PARAMS for p in param_names)

    params: dict[str, np.ndarray] = {}
    if needs_vb:
        print(f"  [{cube_id}] computing VB params...")
        for p in param_names:
            if p in VB_FUNCS:
                t_p = time.time()
                params[p] = VB_FUNCS[p](cube_bls, wav).astype(np.float32)
                print(f"    {p}: {time.time()-t_p:.1f}s   "
                      f"nan_frac={np.isnan(params[p]).mean():.3f}")
    del cube_bls
    gc.collect()

    if needs_brown:
        print(f"  [{cube_id}] loading Brown v6 maps...")
        brown = load_brown_v6(cube_id)
        for p in BROWN_PARAMS:
            if p in param_names:
                params[p] = brown[p]

    valid = np.ones((H, W), dtype=bool)
    for p in param_names:
        if p not in params:
            raise RuntimeError(f"missing param {p} for {cube_id}")
        valid &= np.isfinite(params[p])
    n_valid = int(valid.sum())
    n_total = H * W
    print(f"  [{cube_id}] valid pixels: {n_valid}/{n_total} "
          f"({100*n_valid/n_total:.2f}%)")

    return {
        "cube_id": cube_id, "params": params, "valid_mask": valid,
        "H": H, "W": W, "geo": meta["geo"],
        "n_valid": n_valid, "n_total": n_total, "timestamp": timestamp,
    }


# =============================================================================
# Cluster reordering (mafic-cap-sorted on 3E12)
# =============================================================================
def mafic_sort_permutation(Xs: np.ndarray, labels: np.ndarray,
                           param_names: tuple[str, ...],
                           k: int = K_FIXED) -> np.ndarray:
    """Return sort_order such that cluster sort_order[0] has the lowest
    mafic-cap z-sum (= least altered). Use `gmm.X = gmm.X[sort_order]`
    to reorder GMM params, and `np.argsort(sort_order)[old_label]` to
    relabel pixels.
    """
    mafic_idx = [param_names.index(p) for p in MAFIC_CAP_PARAMS
                 if p in param_names]
    centroids = np.full((k, Xs.shape[1]), np.nan)
    for c in range(k):
        m = labels == c
        if m.any():
            centroids[c] = Xs[m].mean(axis=0)
    score = centroids[:, mafic_idx].sum(axis=1)
    return np.argsort(score)


def reorder_gmm(gmm: GaussianMixture, sort_order: np.ndarray) -> None:
    """In-place reorder of GMM components by sort_order."""
    gmm.means_ = gmm.means_[sort_order]
    gmm.covariances_ = gmm.covariances_[sort_order]
    gmm.weights_ = gmm.weights_[sort_order]
    gmm.precisions_ = gmm.precisions_[sort_order]
    gmm.precisions_cholesky_ = gmm.precisions_cholesky_[sort_order]


# =============================================================================
# Reference cube (3E12) - fit and save
# =============================================================================
def fit_reference(run_key: str) -> dict:
    """Fit scaler + PCA(whiten=True) + GMM(k=6) on FRT00003E12 valid pixels.

    Saves the three transformations to outputs/_v9_3e12_seeded/reference/
    and returns the in-memory objects for downstream use.
    """
    cfg = RUN_CONFIGS[run_key]
    tag = cfg["tag"]
    params_list = cfg["params"]
    d = cfg["d"]

    print(f"\n=== fit reference (3E12) - {run_key} ({d} params) ===")
    stack = build_param_stack(REFERENCE, params_list)
    valid = stack["valid_mask"]

    X = np.column_stack([stack["params"][p][valid] for p in params_list]
                        ).astype(np.float64)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    print(f"  X shape = {X.shape}, standardized.")

    pca = PCA(n_components=d, whiten=True, random_state=RANDOM_SEED)
    Xp_whit = pca.fit_transform(Xs)
    print(f"  PCA(whiten=True) explained variance: "
          f"{[f'{v*100:.1f}%' for v in pca.explained_variance_ratio_]}")
    cum = np.cumsum(pca.explained_variance_ratio_)
    print(f"  cumulative: {[f'{v*100:.1f}%' for v in cum]}")

    gmm = GaussianMixture(
        n_components=K_FIXED, covariance_type="full",
        n_init=N_INIT, random_state=RANDOM_SEED, reg_covar=REG_COVAR,
    )
    gmm.fit(Xp_whit)
    raw_labels = gmm.predict(Xp_whit).astype(np.int16)
    sizes_pre = [int((raw_labels == c).sum()) for c in range(K_FIXED)]
    print(f"  GMM raw (pre-sort) sizes={sizes_pre}  BIC={gmm.bic(Xp_whit):.0f}")

    sort_order = mafic_sort_permutation(Xs, raw_labels, params_list)
    print(f"  mafic-cap sort_order = {sort_order.tolist()}")
    reorder_gmm(gmm, sort_order)
    new_id_for_old = np.argsort(sort_order)
    sorted_labels = new_id_for_old[raw_labels].astype(np.int16)
    sizes_post = [int((sorted_labels == c).sum()) for c in range(K_FIXED)]
    print(f"  GMM after mafic-cap reorder: sizes={sizes_post}")

    ref_dir = SNAPSHOT / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, ref_dir / f"3e12_{tag}_scaler.joblib")
    joblib.dump(pca,    ref_dir / f"3e12_{tag}_pca.joblib")
    joblib.dump(gmm,    ref_dir / f"3e12_{tag}_gmm.joblib")
    print(f"  saved scaler / pca / gmm to {ref_dir}")

    return {
        "run_key": run_key, "scaler": scaler, "pca": pca, "gmm": gmm,
        "params_list": params_list, "d": d,
        "Xs": Xs, "Xp_whit": Xp_whit,
        "labels": sorted_labels, "stack": stack,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "sizes_pre_sort": sizes_pre, "sizes_post_sort": sizes_post,
        "sort_order": sort_order.tolist(),
    }


# =============================================================================
# Non-reference cubes - hard predict
# =============================================================================
def per_pixel_mahalanobis2(X_pc_whit: np.ndarray, labels: np.ndarray,
                            gmm: GaussianMixture) -> np.ndarray:
    """Mahalanobis^2 of each pixel to its assigned cluster mean, in the
    whitened PC space the GMM was fit on. Uses precisions_cholesky_:
        m^2 = || (x - mu_k)^T L_k ||^2   where Sigma_k^-1 = L_k L_k^T.
    """
    n, d = X_pc_whit.shape
    m2 = np.empty(n, dtype=np.float64)
    for k in range(K_FIXED):
        mask = labels == k
        if not mask.any():
            continue
        diff = X_pc_whit[mask] - gmm.means_[k]
        proj = diff @ gmm.precisions_cholesky_[k]
        m2[mask] = (proj * proj).sum(axis=1)
    return m2


def warm_start_drift(X_pc_whit: np.ndarray, ref_gmm: GaussianMixture,
                     d: int) -> dict:
    """Warm-start re-fit using ref_gmm parameters as init. Reports
    per-class ||mu_warm - mu_ref|| / sigma_within_ref where
    sigma_within_ref^2 = trace(Sigma_k_ref) / d.
    """
    gmm_warm = GaussianMixture(
        n_components=K_FIXED, covariance_type="full",
        n_init=1, random_state=RANDOM_SEED, reg_covar=REG_COVAR,
        means_init=ref_gmm.means_,
        precisions_init=ref_gmm.precisions_,
        weights_init=ref_gmm.weights_,
    )
    gmm_warm.fit(X_pc_whit)
    drift = []
    for k in range(K_FIXED):
        sigma_within = float(np.sqrt(np.trace(ref_gmm.covariances_[k]) / d))
        delta = float(np.linalg.norm(gmm_warm.means_[k] - ref_gmm.means_[k]))
        drift.append({
            "class": k,
            "delta_mu": delta,
            "sigma_within_ref": sigma_within,
            "drift_sigma": delta / sigma_within if sigma_within > 0 else None,
        })
    return {
        "warm_converged": bool(gmm_warm.converged_),
        "warm_n_iter": int(gmm_warm.n_iter_),
        "per_class": drift,
    }


def subcluster_novel(X_pc_whit: np.ndarray, novel_mask: np.ndarray,
                     k: int = 2) -> tuple[np.ndarray | None, dict]:
    """Fit a small GMM (default k=2) on novel pixels alone.
    Returns (per-pixel sub-label among novel pixels OR None, info dict).
    Sub-labels: int16 array length=novel_mask.sum(); -1 for n/a is
    handled by the caller assembling back into the scene.
    """
    info = {"n_novel": int(novel_mask.sum()), "k_subclusters": k}
    if info["n_novel"] < max(50, 10 * k):
        info["status"] = "skipped_too_few_novel_pixels"
        return None, info
    Xn = X_pc_whit[novel_mask]
    g = GaussianMixture(
        n_components=k, covariance_type="full",
        n_init=5, random_state=RANDOM_SEED, reg_covar=REG_COVAR,
    )
    g.fit(Xn)
    sub_labels = g.predict(Xn).astype(np.int16)
    info["status"] = "fit"
    info["sub_sizes"] = [int((sub_labels == c).sum()) for c in range(k)]
    info["sub_bic"] = float(g.bic(Xn))
    info["sub_converged"] = bool(g.converged_)
    return sub_labels, info


def predict_cube(cube_id: str, ref: dict, is_reference: bool) -> dict:
    """Hard-predict (or replay) for one cube using the reference GMM."""
    cfg = RUN_CONFIGS[ref["run_key"]]
    params_list = cfg["params"]
    d = cfg["d"]
    chi2_crit = cfg["chi2_crit"]
    print(f"\n--- predict {cube_id} - {ref['run_key']} (d={d}) ---")

    if is_reference:
        # Reuse the data already computed during fit_reference
        stack = ref["stack"]
        Xp_whit = ref["Xp_whit"]
        labels = ref["labels"]
        Xs = ref["Xs"]
        valid = stack["valid_mask"]
        # predict_proba on the same data
        proba = ref["gmm"].predict_proba(Xp_whit)
        max_proba = proba.max(axis=1)
        m2 = per_pixel_mahalanobis2(Xp_whit, labels, ref["gmm"])
    else:
        stack = build_param_stack(cube_id, params_list)
        valid = stack["valid_mask"]
        X = np.column_stack([stack["params"][p][valid] for p in params_list]
                            ).astype(np.float64)
        Xs = ref["scaler"].transform(X)
        Xp_whit = ref["pca"].transform(Xs)
        proba = ref["gmm"].predict_proba(Xp_whit)
        labels = proba.argmax(axis=1).astype(np.int16)
        max_proba = proba.max(axis=1)
        m2 = per_pixel_mahalanobis2(Xp_whit, labels, ref["gmm"])

    novel = (m2 > chi2_crit) | (max_proba < MAX_POSTERIOR_THRESHOLD)
    n_novel = int(novel.sum())
    n_valid = int(valid.sum())
    print(f"  novel: {n_novel}/{n_valid} "
          f"({100*n_novel/max(1, n_valid):.2f}%) "
          f"chi2_crit={chi2_crit:.3f}, p_thr={MAX_POSTERIOR_THRESHOLD}")

    drift = None
    if not is_reference:
        print(f"  warm-start drift sanity pass...")
        drift = warm_start_drift(Xp_whit, ref["gmm"], d)
        for entry in drift["per_class"]:
            flag = " *FLAG*" if (entry["drift_sigma"] is not None
                                 and entry["drift_sigma"] > 0.5) else ""
            print(f"    class {entry['class']}: drift = "
                  f"{entry['drift_sigma']:.3f} sigma{flag}")

    sub_labels_flat, sub_info = subcluster_novel(Xp_whit, novel, k=2)

    return {
        "cube_id": cube_id, "stack": stack,
        "Xs": Xs, "Xp_whit": Xp_whit,
        "labels": labels, "max_proba": max_proba.astype(np.float32),
        "m2": m2.astype(np.float32), "novel_mask": novel,
        "sub_labels_flat": sub_labels_flat, "sub_info": sub_info,
        "drift": drift, "n_novel": n_novel, "n_valid": n_valid,
    }


# =============================================================================
# GeoTIFF writers - composite map (gray novel + transparent invalid)
# =============================================================================
def _make_transform(geo: dict | None) -> Affine:
    if geo and geo.get("geotransform"):
        gt = geo["geotransform"]
        return Affine(gt[1], gt[2], gt[0], gt[4], gt[5], gt[3])
    return Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)


def _make_crs(geo: dict | None):
    if geo and geo.get("wkt"):
        try:
            return CRS.from_wkt(geo["wkt"])
        except Exception:
            return None
    return None


def write_composite_geotiff(scene: np.ndarray, path: Path,
                             geo: dict | None, description: str) -> None:
    """uint8 categorical map: 0..5 = GMM classes (palette), 254 = novel
    (gray), 255 = invalid (transparent).
    """
    h, w = scene.shape
    profile = {
        "driver": "GTiff", "dtype": "uint8",
        "width": w, "height": h, "count": 1,
        "crs": _make_crs(geo), "transform": _make_transform(geo),
        "nodata": INVALID_LABEL, "compress": "deflate",
    }
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(scene.astype(np.uint8), 1)
        ct = {i: (*GMM_PALETTE_RGB[i], 255) for i in range(K_FIXED)}
        ct[NOVEL_LABEL] = (*NOVEL_RGB, 255)
        ct[INVALID_LABEL] = (0, 0, 0, 0)
        dst.write_colormap(1, ct)
        if description:
            dst.set_band_description(1, description)


def write_subcluster_geotiff(scene: np.ndarray, path: Path,
                              geo: dict | None, description: str,
                              k_sub: int) -> None:
    """uint8 categorical map of novel sub-clusters: 0..k_sub-1 = sub
    classes, 255 = not-novel/invalid.
    """
    h, w = scene.shape
    profile = {
        "driver": "GTiff", "dtype": "uint8",
        "width": w, "height": h, "count": 1,
        "crs": _make_crs(geo), "transform": _make_transform(geo),
        "nodata": INVALID_LABEL, "compress": "deflate",
    }
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(scene.astype(np.uint8), 1)
        ct = {i: (*SUBCLUSTER_PALETTE_RGB[i], 255)
              for i in range(min(k_sub, len(SUBCLUSTER_PALETTE_RGB)))}
        ct[INVALID_LABEL] = (0, 0, 0, 0)
        dst.write_colormap(1, ct)
        if description:
            dst.set_band_description(1, description)


# =============================================================================
# Output assembly per cube
# =============================================================================
def cube_run_dir(run_key: str, cube_id: str) -> Path:
    p = SNAPSHOT / run_key / cube_id
    (p / "arrays").mkdir(parents=True, exist_ok=True)
    (p / "geotiffs").mkdir(parents=True, exist_ok=True)
    (p / "logs").mkdir(parents=True, exist_ok=True)
    return p


def emit_cube_outputs(run_key: str, cube: dict, ref: dict) -> dict:
    """Save .npy + GeoTIFFs + log for one cube; return a small summary dict."""
    cfg = RUN_CONFIGS[run_key]
    tag = cfg["tag"]
    d = cfg["d"]
    chi2_crit = cfg["chi2_crit"]
    cube_id = cube["cube_id"]
    stack = cube["stack"]
    H, W = stack["H"], stack["W"]
    geo = stack["geo"]
    valid = stack["valid_mask"]

    cdir = cube_run_dir(run_key, cube_id)
    arr_dir = cdir / "arrays"
    gtiff_dir = cdir / "geotiffs"

    # Param stack as (H, W, n_params)
    pstack = np.dstack([stack["params"][p] for p in cfg["params"]])
    np.save(arr_dir / f"{cube_id}_{tag}_param_stack.npy", pstack)
    np.save(arr_dir / f"{cube_id}_{tag}_valid_mask.npy", valid)

    # PC scores back-projected (H, W, d)
    pc_scenes = np.full((d, H, W), np.nan, dtype=np.float32)
    for k in range(d):
        sc = np.full((H, W), np.nan, dtype=np.float32)
        sc[valid] = cube["Xp_whit"][:, k].astype(np.float32)
        pc_scenes[k] = sc
    np.save(arr_dir / f"{cube_id}_{tag}_pc_scores.npy", pc_scenes)

    # GMM labels back-projected
    label_scene = np.full((H, W), -1, dtype=np.int16)
    label_scene[valid] = cube["labels"]
    np.save(arr_dir / f"{cube_id}_{tag}_gmm_labels.npy", label_scene)

    max_p_scene = np.full((H, W), np.nan, dtype=np.float32)
    max_p_scene[valid] = cube["max_proba"]
    np.save(arr_dir / f"{cube_id}_{tag}_max_posterior.npy", max_p_scene)

    m2_scene = np.full((H, W), np.nan, dtype=np.float32)
    m2_scene[valid] = cube["m2"]
    np.save(arr_dir / f"{cube_id}_{tag}_mahalanobis2.npy", m2_scene)

    novel_scene = np.zeros((H, W), dtype=bool)
    novel_scene[valid] = cube["novel_mask"]
    np.save(arr_dir / f"{cube_id}_{tag}_novel_mask.npy", novel_scene)

    # PCs (d float GeoTIFFs)
    for k in range(d):
        write_float_geotiff(
            pc_scenes[k],
            gtiff_dir / f"{cube_id}_{tag}_PC{k+1}_whitened.tif",
            geo, description=f"v9 {tag} PC{k+1} whitened (3E12-fitted)")

    # Composite map (labels with novel as gray)
    composite = np.full((H, W), INVALID_LABEL, dtype=np.uint8)
    valid_assigned = valid & (~novel_scene)
    composite[valid_assigned] = label_scene[valid_assigned].astype(np.uint8)
    composite[valid & novel_scene] = NOVEL_LABEL
    write_composite_geotiff(
        composite,
        gtiff_dir / f"{cube_id}_{tag}_composite_map.tif",
        geo,
        description=f"v9 {tag} GMM k=6 composite (3E12-seeded; "
                    "0..5=class, 254=novel/gray, 255=invalid)")

    # Plain labels (for users who don't want the gray overlay)
    label_only = np.where(label_scene >= 0,
                          label_scene.astype(np.uint8),
                          INVALID_LABEL)
    write_composite_geotiff(
        label_only,
        gtiff_dir / f"{cube_id}_{tag}_gmm_labels.tif",
        geo,
        description=f"v9 {tag} GMM k=6 labels only (3E12-seeded)")

    # Diagnostic float layers
    write_float_geotiff(
        max_p_scene, gtiff_dir / f"{cube_id}_{tag}_max_posterior.tif",
        geo, description=f"v9 {tag} max posterior probability per pixel")
    write_float_geotiff(
        m2_scene, gtiff_dir / f"{cube_id}_{tag}_mahalanobis2.tif",
        geo, description=f"v9 {tag} Mahalanobis^2 to assigned cluster")

    # Novel mask as 0/1 categorical
    novel_uint8 = np.where(novel_scene & valid, 1, 0).astype(np.uint8)
    novel_uint8[~valid] = INVALID_LABEL
    profile = {
        "driver": "GTiff", "dtype": "uint8",
        "width": W, "height": H, "count": 1,
        "crs": _make_crs(geo), "transform": _make_transform(geo),
        "nodata": INVALID_LABEL, "compress": "deflate",
    }
    with rasterio.open(str(gtiff_dir / f"{cube_id}_{tag}_novel_mask.tif"),
                       "w", **profile) as dst:
        dst.write(novel_uint8, 1)
        dst.write_colormap(1, {
            0: (60, 60, 60, 255),
            1: (*NOVEL_RGB, 255),
            INVALID_LABEL: (0, 0, 0, 0),
        })
        dst.set_band_description(
            1, f"v9 {tag} novel pixels (m^2>{chi2_crit:.3f} OR "
               f"p_max<{MAX_POSTERIOR_THRESHOLD})")

    # Sub-cluster of novel pixels (only when fit succeeded)
    sub_info = cube["sub_info"]
    if cube["sub_labels_flat"] is not None:
        sub_scene_flat_to_full = np.full((H, W), INVALID_LABEL, dtype=np.uint8)
        # build a (H,W) array of novel positions, fill by sub-label
        # cube["novel_mask"] is the 1-D mask over valid pixels;
        # cube["sub_labels_flat"] is the same length as novel.sum()
        novel_2d_in_valid = cube["novel_mask"]
        # need indexer that maps valid -> 2D position
        valid_idx = np.flatnonzero(valid.ravel())
        novel_full_idx = valid_idx[novel_2d_in_valid]
        flat_sub = sub_scene_flat_to_full.ravel()
        flat_sub[novel_full_idx] = cube["sub_labels_flat"].astype(np.uint8)
        sub_scene = flat_sub.reshape(H, W)
        np.save(arr_dir / f"{cube_id}_{tag}_novel_subclusters.npy", sub_scene)
        write_subcluster_geotiff(
            sub_scene,
            gtiff_dir / f"{cube_id}_{tag}_novel_subclusters.tif",
            geo,
            description=f"v9 {tag} novel-pixel sub-clusters "
                        f"(k={sub_info['k_subclusters']}; non-novel = transparent)",
            k_sub=sub_info["k_subclusters"],
        )

    # Per-cube log
    log = {
        "stage": "v9_3e12_seeded",
        "run_key": run_key, "tag": tag, "cube_id": cube_id,
        "params": list(cfg["params"]), "d": d,
        "is_reference": cube_id == REFERENCE,
        "shape": [H, W],
        "n_valid": int(valid.sum()),
        "n_total": int(H * W),
        "n_novel": cube["n_novel"],
        "novel_fraction": cube["n_novel"] / max(1, cube["n_valid"]),
        "thresholds": {
            "chi2_crit_0.95": chi2_crit,
            "max_posterior_threshold": MAX_POSTERIOR_THRESHOLD,
        },
        "cluster_sizes": [int((cube["labels"] == c).sum())
                          for c in range(K_FIXED)],
        "subcluster_info": cube["sub_info"],
        "drift_check": cube["drift"],
        "timestamp_utc": stack["timestamp"],
        "geo_proj_name": geo["proj_name"] if geo else None,
        "ref_random_seed": RANDOM_SEED,
    }
    log_path = cdir / "logs" / f"{tag}_{cube_id}_{stack['timestamp']}.json"
    log_path.write_text(json.dumps(log, indent=2, default=str),
                        encoding="utf-8")
    print(f"  wrote {tag} {cube_id}: {d} PC + composite + label-only + "
          f"max_p + m2 + novel{' + subcluster' if cube['sub_labels_flat'] is not None else ''} "
          "GeoTIFFs + log")
    return {
        "cube_id": cube_id,
        "n_valid": int(valid.sum()),
        "n_novel": cube["n_novel"],
        "novel_fraction": cube["n_novel"] / max(1, cube["n_valid"]),
        "cluster_sizes": log["cluster_sizes"],
        "drift": cube["drift"], "subcluster_info": cube["sub_info"],
    }


# =============================================================================
# Top-level run
# =============================================================================
def run(run_key: str) -> dict:
    cfg = RUN_CONFIGS[run_key]
    print(f"\n############# {run_key.upper()} ({cfg['d']} params) #############")
    t0 = time.time()
    ref = fit_reference(run_key)

    # Always include the reference cube as the first prediction so it
    # participates in palette decisions, novelty stats, and outputs.
    summaries = {}
    for cube_id in CUBE_IDS:
        is_ref = cube_id == REFERENCE
        cube = predict_cube(cube_id, ref, is_reference=is_ref)
        summary = emit_cube_outputs(run_key, cube, ref)
        summaries[cube_id] = summary
        # Free large arrays for the non-ref cubes; the reference data is
        # kept alive by `ref`
        if not is_ref:
            del cube
            gc.collect()
    wall = round(time.time() - t0, 2)
    print(f"\n[{run_key}] wall = {wall}s")
    return {
        "run_key": run_key, "cfg": {
            "params": list(cfg["params"]),
            "d": cfg["d"],
            "chi2_crit": cfg["chi2_crit"],
            "max_posterior_threshold": MAX_POSTERIOR_THRESHOLD,
            "n_init": N_INIT, "reg_covar": REG_COVAR,
            "random_seed": RANDOM_SEED,
        },
        "explained_variance_ratio": ref["explained_variance_ratio"],
        "sort_order_3e12": ref["sort_order"],
        "sizes_post_sort_3e12": ref["sizes_post_sort"],
        "per_cube": summaries,
        "wall_seconds": wall,
    }


def write_aggregate_reports(results: dict[str, dict]) -> None:
    """results: {run_key -> top-level run summary}"""
    rep_dir = SNAPSHOT / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    drift_check = {}
    novelty_stats = {}
    for run_key, R in results.items():
        drift_check[run_key] = {}
        novelty_stats[run_key] = {
            "thresholds": {
                "chi2_crit_0.95": R["cfg"]["chi2_crit"],
                "max_posterior_threshold": MAX_POSTERIOR_THRESHOLD,
            },
            "explained_variance_ratio": R["explained_variance_ratio"],
            "sort_order_3e12": R["sort_order_3e12"],
            "sizes_post_sort_3e12": R["sizes_post_sort_3e12"],
            "per_cube": {},
        }
        for cube_id, s in R["per_cube"].items():
            novelty_stats[run_key]["per_cube"][cube_id] = {
                "n_valid": s["n_valid"],
                "n_novel": s["n_novel"],
                "novel_fraction": s["novel_fraction"],
                "cluster_sizes": s["cluster_sizes"],
                "subcluster_info": s["subcluster_info"],
            }
            if s["drift"] is not None:
                drift_check[run_key][cube_id] = s["drift"]

    (rep_dir / "v9_drift_check.json").write_text(
        json.dumps(drift_check, indent=2, default=str), encoding="utf-8")
    (rep_dir / "v9_novelty_stats.json").write_text(
        json.dumps(novelty_stats, indent=2, default=str), encoding="utf-8")

    palette = {
        "encoding": {
            "0": list(GMM_PALETTE_RGB[0]),
            "1": list(GMM_PALETTE_RGB[1]),
            "2": list(GMM_PALETTE_RGB[2]),
            "3": list(GMM_PALETTE_RGB[3]),
            "4": list(GMM_PALETTE_RGB[4]),
            "5": list(GMM_PALETTE_RGB[5]),
            "novel_254": list(NOVEL_RGB),
            "invalid_255": [0, 0, 0, 0],
        },
        "ordering": ("cluster IDs are sorted ascending by mafic-cap z-sum on "
                     "FRT00003E12. cluster 0 has the lowest "
                     "BD1900R2+D2300+BD2500_2+OLINDEX3 mean (least altered). "
                     "cluster 5 has the highest (most altered / olivine-"
                     "carbonate-rich)."),
        "subcluster_palette": [list(c) for c in SUBCLUSTER_PALETTE_RGB],
        "novel_threshold": {
            "chi2_crit_v9a_d5": RUN_CONFIGS["v9a_5param"]["chi2_crit"],
            "chi2_crit_v9b_d7": RUN_CONFIGS["v9b_7param"]["chi2_crit"],
            "max_posterior_threshold": MAX_POSTERIOR_THRESHOLD,
        },
    }
    (rep_dir / "v9_palette.json").write_text(
        json.dumps(palette, indent=2), encoding="utf-8")
    print(f"\nwrote {rep_dir}/v9_drift_check.json, v9_novelty_stats.json, "
          "v9_palette.json")


def main(argv: list[str]) -> None:
    t0 = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    (SNAPSHOT / "reports").mkdir(parents=True, exist_ok=True)

    targets = list(RUN_CONFIGS) if len(argv) <= 1 else argv[1:]
    unknown = [t for t in targets if t not in RUN_CONFIGS]
    if unknown:
        raise SystemExit(f"unknown run keys: {unknown}; "
                          f"known: {list(RUN_CONFIGS)}")

    results = {}
    for run_key in targets:
        results[run_key] = run(run_key)

    if set(targets) == set(RUN_CONFIGS):
        write_aggregate_reports(results)

    top_log = {
        "stage": "v9_3e12_seeded",
        "timestamp_utc": timestamp,
        "random_seed": RANDOM_SEED,
        "n_init": N_INIT,
        "reg_covar": REG_COVAR,
        "max_posterior_threshold": MAX_POSTERIOR_THRESHOLD,
        "reference_cube": REFERENCE,
        "runs": {rk: {"params": list(RUN_CONFIGS[rk]["params"]),
                      "d": RUN_CONFIGS[rk]["d"],
                      "chi2_crit": RUN_CONFIGS[rk]["chi2_crit"],
                      "wall_seconds": results[rk]["wall_seconds"]}
                 for rk in results},
        "wall_clock_seconds": round(time.time() - t0, 2),
        "source_code_md5": {
            "stage_v9_3e12_seeded.py": md5_of(Path(__file__)),
            "stage_v8_mtr3_with_brown.py": md5_of(
                Path(__file__).with_name("stage_v8_mtr3_with_brown.py")),
            "stage_v7c_mtr3_pipeline.py": md5_of(
                Path(__file__).with_name("stage_v7c_mtr3_pipeline.py")),
            "vb_parameters.py": md5_of(
                Path(__file__).with_name("vb_parameters.py")),
        },
    }
    (SNAPSHOT / "reports" / f"v9_run_log_{timestamp}.json").write_text(
        json.dumps(top_log, indent=2), encoding="utf-8")
    print(f"\nv9 complete in {round(time.time()-t0, 2)}s "
          f"(runs: {list(results)})")


if __name__ == "__main__":
    main(sys.argv)
