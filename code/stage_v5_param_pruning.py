"""Stage v5 - VB parameter pruning + full-scene Brown RGB.

Two stacks per cube:
  v5-A (7 params): drops SH600_2, HCPINDEX2, BDI1000VIS, LCPINDEX2.
  v5-B (5 params): also drops BD530_2, BD920_2 -> "pure alteration"
                   suite (OLINDEX3 + BD1400 + BD1900R2 + D2300 + BD2500_2).

Inputs: cached v4 arrays (no TER3 reload, no VB recompute, no Brown
recompute). All inputs live under outputs/_v4_vb_pca_brown_rgb/<cube>/arrays/.

Outputs: outputs/_v5_param_pruning/<cube>/{arrays,figures,tables}/
plus top-level _v5_param_pruning/{figures,reports}/.

Run:
  python stage_v5_param_pruning.py        # all four cubes + cross-cube
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

from config import CUBE_IDS, RANDOM_SEED, OUTPUTS_DIR, cube_dir
from crism_io import find_cube_and_brown_maps
from display import apply_display_transform
from vb_parameters import V4_VB_PARAM_ORDER

# v4-matching styling
plt.rcdefaults()

# v5-A and v5-B parameter bases
V5A_PARAMS = ("BD530_2", "BD920_2", "OLINDEX3",
              "BD1400", "BD1900R2", "D2300", "BD2500_2")
V5B_PARAMS = ("OLINDEX3", "BD1400", "BD1900R2", "D2300", "BD2500_2")

V5_ROOT = OUTPUTS_DIR / "_v5_param_pruning"
V4_ROOT = OUTPUTS_DIR / "_v4_vb_pca_brown_rgb"

# Cube colours per v2 convention
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

# Discrete 6-class palette for GMM scene maps
GMM_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c",
               "#d62728", "#9467bd", "#8c564b"]

# Mafic-cap z-sum parameters per Stage C convention
MAFIC_CAP_PARAMS = ("BD1900R2", "D2300", "BD2500_2", "OLINDEX3")

# Brown stretches (v4.1 values for direct comparability)
BROWN_STRETCH_V4P1 = {
    "centroid": (1.15, 1.40),
    "amplitude": (0.10, 0.37),
    "hwhm": (0.18, 0.30),
    "asymmetry": (1.25, 5.00),
}
BROWN_STRETCH_NOTE_V4P1 = (
    "v4.1 stretch (Brown-eq.2-valid pixels p2/p98). Centered for direct "
    "comparison to the eq.2-masked Brown RGBs in _v4_vb_pca_brown_rgb/. "
    "Full-scene pixels outside this range clip."
)
BROWN_STRETCH_NOTE_V5 = (
    "v5 stretch (full-scene 4-cube pooled p2/p98). Centered on the "
    "actual full-scene data range (no eq.2 mask). Identical color = "
    "identical parameter value across cubes."
)
ASYMMETRY_SAT_NOTE = (
    "Asymmetry saturating to pure blue at the upper bound corresponds "
    "to the band-saturation regime per Brown 2020."
)

K_FIXED = 6
N_PCS_V5A = 5
N_PCS_V5B = 5  # rank-limited; PC4-5 carry near-trivial variance


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def cube_root(cube_id: str) -> Path:
    p = V5_ROOT / cube_id
    (p / "arrays").mkdir(parents=True, exist_ok=True)
    (p / "figures").mkdir(parents=True, exist_ok=True)
    (p / "tables").mkdir(parents=True, exist_ok=True)
    return p


def percentile_norm(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
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
# Step 1: Load cached v4 VB stack and build v5 sub-stacks
# =============================================================================
def load_v4_vb_arrays(cube_id: str) -> dict:
    arr_root = V4_ROOT / cube_id / "arrays"
    vb_stack = np.load(arr_root / f"{cube_id}_VB_stack_v4.npy")  # (H, W, 11)
    info = json.loads((arr_root / f"{cube_id}_VB_param_names_v4.json").read_text())
    assert info["param_order"] == list(V4_VB_PARAM_ORDER), \
        f"v4 param order mismatch in {cube_id}"
    return {"vb_stack": vb_stack, "info": info}


def build_v5_substack(vb_stack_v4: np.ndarray, target_params: tuple) -> tuple:
    """Reindex the v4 11-param stack down to the v5 target list."""
    cols = [V4_VB_PARAM_ORDER.index(p) for p in target_params]
    sub = vb_stack_v4[..., cols]
    H, W, _ = sub.shape
    flat = sub.reshape(-1, sub.shape[-1])
    finite = np.isfinite(flat).all(axis=1)
    return sub, finite.reshape(H, W)


# =============================================================================
# Step 2: PCA + GMM per stack per cube
# =============================================================================
def run_pca_gmm(stack: np.ndarray, valid: np.ndarray, n_pcs: int,
                k: int = K_FIXED) -> dict:
    H, W, P = stack.shape
    X = stack[valid].astype(np.float64)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=P, random_state=RANDOM_SEED)
    Xp = pca.fit_transform(Xs)
    col_std = Xp[:, :n_pcs].std(axis=0, ddof=1)
    Xp_whit = Xp[:, :n_pcs] / col_std
    gmm = GaussianMixture(
        n_components=k, covariance_type="full", n_init=5,
        random_state=RANDOM_SEED, reg_covar=1e-5,
    )
    gmm.fit(Xp_whit)
    labels = gmm.predict(Xp_whit).astype(np.int16)
    proba = gmm.predict_proba(Xp_whit).astype(np.float32)
    return {
        "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
        "components": pca.components_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "Xp_raw": Xp[:, :n_pcs].astype(np.float32),
        "Xp_whit": Xp_whit.astype(np.float32),
        "labels": labels, "proba": proba,
        "bic": float(gmm.bic(Xp_whit)),
        "aic": float(gmm.aic(Xp_whit)),
        "Xs_full": Xs,  # for cluster-centroid alignment in standardized space
    }


# =============================================================================
# Step 3: Hungarian alignment to FRT00003E12 reference, mafic-cap ordered
# =============================================================================
def cluster_centroids_std(Xs: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    ctr = np.full((k, Xs.shape[1]), np.nan)
    for c in range(k):
        m = labels == c
        if m.any():
            ctr[c] = Xs[m].mean(axis=0)
    return ctr


def align_stack(per_cube: dict, params: tuple, key: str) -> dict:
    REF = "FRT00003E12"
    ref = per_cube[REF][key]
    ref_ctr = cluster_centroids_std(ref["Xs_full"], ref["labels"], K_FIXED)

    # Mafic-cap order on reference (lowest z-sum first)
    mafic_idx = [params.index(p) for p in MAFIC_CAP_PARAMS if p in params]
    score = ref_ctr[:, mafic_idx].sum(axis=1)
    if "LCPINDEX2" in params:
        lcp_idx = params.index("LCPINDEX2")
        sort_order = np.lexsort((-ref_ctr[:, lcp_idx], score))
    else:
        sort_order = np.argsort(score)
    ref_relabel = np.argsort(sort_order)
    ref_aligned = np.array([ref_relabel[c] for c in ref["labels"]],
                           dtype=np.int16)
    ref_ctr_aligned = ref_ctr[sort_order]

    aligned_by_cube = {REF: ref_aligned}
    log = {REF: {"permutation": ref_relabel.tolist(),
                 "distances_to_ref": [0.0] * K_FIXED}}
    for cube_id in CUBE_IDS:
        if cube_id == REF:
            continue
        c_data = per_cube[cube_id][key]
        ctr = cluster_centroids_std(c_data["Xs_full"], c_data["labels"], K_FIXED)
        D = cdist(ctr, ref_ctr_aligned)
        row_ind, col_ind = linear_sum_assignment(D)
        relabel = col_ind.copy()
        aligned = np.array([relabel[c] for c in c_data["labels"]],
                           dtype=np.int16)
        aligned_by_cube[cube_id] = aligned
        log[cube_id] = {
            "permutation": relabel.tolist(),
            "distances_to_ref": D[row_ind, col_ind].tolist(),
            "full_distance_matrix": D.tolist(),
        }
    return {"aligned_by_cube": aligned_by_cube, "log": log}


# =============================================================================
# Step 4: Per-cube saves
# =============================================================================
def save_per_cube_arrays(cube_id: str, vab: dict, vbb: dict,
                         valid_a: np.ndarray, valid_b: np.ndarray,
                         shape: tuple) -> None:
    arr = cube_root(cube_id) / "arrays"
    np.save(arr / f"{cube_id}_v5a_PC_scores.npy", vab["Xp_raw"])
    np.save(arr / f"{cube_id}_v5b_PC_scores.npy", vbb["Xp_raw"])
    np.save(arr / f"{cube_id}_v5a_PC_scores_whitened.npy", vab["Xp_whit"])
    np.save(arr / f"{cube_id}_v5b_PC_scores_whitened.npy", vbb["Xp_whit"])

    # Labels back-projected to (H, W); -1 outside valid mask
    for name, run, valid in (("v5a", vab, valid_a), ("v5b", vbb, valid_b)):
        scene = np.full(shape, -1, dtype=np.int16)
        scene[valid] = run["labels"]
        np.save(arr / f"{cube_id}_{name}_GMM_k6_labels.npy", scene)
        np.save(arr / f"{cube_id}_{name}_GMM_k6_proba.npy", run["proba"])

    counts = {
        "shape": list(shape),
        "v5a_finite_pixels": int(valid_a.sum()),
        "v5b_finite_pixels": int(valid_b.sum()),
        "v5a_BIC": vab["bic"], "v5a_AIC": vab["aic"],
        "v5b_BIC": vbb["bic"], "v5b_AIC": vbb["aic"],
        "v5a_cluster_sizes": [int((vab["labels"] == c).sum()) for c in range(K_FIXED)],
        "v5b_cluster_sizes": [int((vbb["labels"] == c).sum()) for c in range(K_FIXED)],
        "v5a_explained_variance_ratio": vab["explained_variance_ratio"].tolist(),
        "v5b_explained_variance_ratio": vbb["explained_variance_ratio"].tolist(),
    }
    (arr / f"{cube_id}_v5_pixel_counts.json").write_text(
        json.dumps(counts, indent=2), encoding="utf-8")


def save_per_cube_loadings(cube_id: str, run: dict, params: tuple,
                           tag: str) -> Path:
    csv_p = cube_root(cube_id) / "tables" / f"{cube_id}_{tag}_loadings.csv"
    comps = run["components"][:5]
    evr = run["explained_variance_ratio"]
    with open(csv_p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["PC", "explained_var_pct"] + list(params))
        for i in range(min(5, comps.shape[0])):
            w.writerow([f"PC{i+1}", f"{evr[i]*100:.3f}"] +
                       [f"{x:.4f}" for x in comps[i]])
    return csv_p


# =============================================================================
# Step 5: Per-cube GMM scene map (per-cube color order, raw labels)
# =============================================================================
def fig_gmm_scene_per_cube(cube_id: str, label_scene: np.ndarray,
                            counts: dict, tag: str) -> Path:
    disp = apply_display_transform(label_scene)
    show = np.where(disp >= 0, disp, np.nan)
    cmap = ListedColormap(GMM_PALETTE[:K_FIXED])
    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    ax.imshow(show, cmap=cmap, vmin=0, vmax=K_FIXED - 1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    sizes = counts[f"{tag}_cluster_sizes"]
    bic = counts[f"{tag}_BIC"]
    ax.set_title(
        f"{cube_id}  GMM k=6 on whitened {tag.upper()} PC1-5 "
        f"(per-cube color order)\n"
        f"sizes: {sizes}   |   BIC: {bic:.0f}", fontsize=10,
    )
    p = cube_root(cube_id) / "figures" / f"{cube_id}_GMM_k6_{tag}_map.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# =============================================================================
# Step 6: Full-scene Brown RGB regeneration (no eq.2 mask)
# =============================================================================
def make_brown_rgb(brown: dict, valid: np.ndarray, channels: list[str],
                    stretch: dict) -> np.ndarray:
    h, w = brown["centroid"].shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for k, name in enumerate(channels):
        lo, hi = stretch[name]
        rgba[..., k] = percentile_norm(
            np.where(valid, brown[name].astype(np.float32), np.nan), lo, hi)
    rgba[..., 3] = valid.astype(np.float32)
    nanmask = (np.isnan(rgba[..., 0]) | np.isnan(rgba[..., 1])
               | np.isnan(rgba[..., 2]))
    for k in range(3):
        rgba[..., k] = np.where(nanmask, 0.0, rgba[..., k])
    rgba[nanmask, 3] = 0.0
    return rgba


def fig_brown_fullscene_rgb(cube_id: str, brown: dict, full_valid: np.ndarray,
                             stretch: dict, suffix: str, note: str) -> Path:
    channels = ["centroid", "amplitude", "asymmetry"]
    rgba = make_brown_rgb(brown, full_valid, channels, stretch)
    disp = apply_display_transform(rgba)
    fig, ax = plt.subplots(figsize=(8.5, 7.0), constrained_layout=True)
    ax.imshow(disp, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    label_map = {"centroid": "centroid", "amplitude": "amplitude",
                 "asymmetry": "asymmetry"}
    ranges = " | ".join(
        f"{label_map[c]}: {stretch[c][0]:.2f}-{stretch[c][1]:.2f}"
        for c in channels
    )
    ax.set_title(
        f"{cube_id}  Brown RGB (full scene, no eq.2 mask) = "
        f"centroid / amplitude / asymmetry\n{ranges}",
        fontsize=10,
    )
    fig.text(0.5, 0.005, note + " " + ASYMMETRY_SAT_NOTE,
             ha="center", va="bottom", fontsize=7, color="#444444",
             wrap=True)
    p = (cube_root(cube_id) / "figures"
         / f"{cube_id}_Brown_RGB_centroid_amp_asym_v5_fullscene_{suffix}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_brown_fullscene_rgb_4cube(per_cube: dict, full_valid_by_cube: dict,
                                   stretch: dict, suffix: str, note: str) -> Path:
    channels = ["centroid", "amplitude", "asymmetry"]
    fig, axes = plt.subplots(2, 2, figsize=(18, 13.5), constrained_layout=True)
    for ax, cube_id in zip(axes.ravel(), CUBE_IDS):
        rgba = make_brown_rgb(per_cube[cube_id]["brown"],
                              full_valid_by_cube[cube_id], channels, stretch)
        disp = apply_display_transform(rgba)
        ax.imshow(disp, interpolation="nearest")
        ax.set_title(cube_id, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    label_map = {"centroid": "centroid", "amplitude": "amplitude",
                 "asymmetry": "asymmetry"}
    ranges = " | ".join(
        f"{label_map[c]}: {stretch[c][0]:.2f}-{stretch[c][1]:.2f}"
        for c in channels
    )
    fig.suptitle(
        f"v5 four-cube Brown RGB (full scene, no eq.2 mask)  [{ranges}]",
        fontsize=12)
    fig.text(0.5, 0.003, note + " " + ASYMMETRY_SAT_NOTE,
             ha="center", va="bottom", fontsize=8, color="#444444",
             wrap=True)
    p = (V5_ROOT / "figures"
         / f"comparison_Brown_RGB_centroid_amp_asym_4cube_v5_fullscene_{suffix}.png")
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


def compute_v5_brown_stretch(full_browns: dict) -> dict:
    """Compute 4-cube pooled p2/p98 on full-scene Brown-finite pixels for the
    three RGB channels used (centroid, amplitude, asymmetry)."""
    pooled = {n: [] for n in ("centroid", "amplitude", "asymmetry")}
    for cube_id, b in full_browns.items():
        for name in pooled:
            arr = b["brown"][name].astype(np.float32)
            v = b["full_valid"]
            arr = arr[v]
            arr = arr[np.isfinite(arr)]
            pooled[name].append(arr)
    out = {}
    for name, arrs in pooled.items():
        all_arr = np.concatenate(arrs)
        lo, hi = float(np.percentile(all_arr, 2)), float(np.percentile(all_arr, 98))
        out[name] = (lo, hi)
    # HWHM not used in asymmetry RGB but include for symmetry/clarity
    return out


# =============================================================================
# Step 7: Cross-cube comparison plots
# =============================================================================
def fig_loadings_4cube_2stack(per_cube: dict) -> Path:
    """4 cubes x 2 stacks loadings heatmap, PC1-3 only (rank-limited on v5-B)."""
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), constrained_layout=True)
    vmaxs = []
    for cube_id in CUBE_IDS:
        for tag in ("v5a", "v5b"):
            comps = per_cube[cube_id][tag]["components"][:3]
            vmaxs.append(np.max(np.abs(comps)))
    vmax = max(vmaxs)
    for row, cube_id in enumerate(CUBE_IDS):
        for col, (tag, params) in enumerate(
                (("v5a", V5A_PARAMS), ("v5b", V5B_PARAMS))):
            ax = axes[row, col]
            comps = per_cube[cube_id][tag]["components"][:3]
            evr = per_cube[cube_id][tag]["explained_variance_ratio"]
            im = ax.imshow(comps, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           aspect="auto")
            ax.set_xticks(range(len(params)))
            ax.set_xticklabels(params, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(3))
            ax.set_yticklabels(
                [f"PC{i+1} ({evr[i]*100:.1f}%)" for i in range(3)],
                fontsize=8)
            for i in range(3):
                for j in range(len(params)):
                    ax.text(j, i, f"{comps[i, j]:+.2f}", ha="center",
                            va="center", fontsize=7,
                            color="white" if abs(comps[i, j]) > 0.5 * vmax else "black")
            ax.set_title(f"{cube_id} {tag.upper()} ({len(params)}-param)",
                         fontsize=10)
    fig.suptitle("v5-A (7-param) vs v5-B (5-param) PC1-PC3 loadings per cube",
                 fontsize=13)
    fig.colorbar(im, ax=axes, shrink=0.5, label="loading")
    p = V5_ROOT / "figures" / "comparison_v5a_v5b_loadings.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_aligned_4cube(per_cube: dict, aligned_by_cube: dict, tag: str,
                       valid_key: str) -> Path:
    """2x2 aligned GMM scene panels."""
    cmap = ListedColormap(GMM_PALETTE[:K_FIXED])
    fig, axes = plt.subplots(2, 2, figsize=(18, 13), constrained_layout=True)
    for ax, cube_id in zip(axes.ravel(), CUBE_IDS):
        valid = per_cube[cube_id][valid_key]
        aligned = aligned_by_cube[cube_id]
        scene = np.full(valid.shape, -1, dtype=np.int16)
        scene[valid] = aligned
        disp = apply_display_transform(scene)
        show = np.where(disp >= 0, disp, np.nan)
        ax.imshow(show, cmap=cmap, vmin=0, vmax=K_FIXED - 1,
                  interpolation="nearest")
        ax.set_title(cube_id, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"v5 four-cube aligned GMM k=6 scene maps ({tag.upper()})",
                 fontsize=13)
    p = V5_ROOT / "figures" / f"comparison_GMM_k6_{tag}_4cube_aligned.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_v5a_v5b_confusion(per_cube: dict, aligned_a: dict,
                           aligned_b: dict) -> Path:
    """Per-cube v5-A vs v5-B confusion matrices on the aligned labels."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 6), constrained_layout=True)
    for ax, cube_id in zip(axes, CUBE_IDS):
        # Restrict to pixels valid in BOTH stacks
        v_a = per_cube[cube_id]["valid_a"]
        v_b = per_cube[cube_id]["valid_b"]
        both = v_a & v_b
        # Get labels only for "both" pixels
        labels_a_full = np.full(v_a.shape, -1, dtype=np.int16)
        labels_a_full[v_a] = aligned_a[cube_id]
        labels_b_full = np.full(v_b.shape, -1, dtype=np.int16)
        labels_b_full[v_b] = aligned_b[cube_id]
        la = labels_a_full[both]; lb = labels_b_full[both]
        M = np.zeros((K_FIXED, K_FIXED), dtype=np.int64)
        for i in range(K_FIXED):
            for j in range(K_FIXED):
                M[i, j] = int(((la == i) & (lb == j)).sum())
        n = M.sum()
        Mn = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
        diag = float(np.trace(M)) / max(1, n)
        im = ax.imshow(Mn, cmap="magma", vmin=0, vmax=1)
        ax.set_xticks(range(K_FIXED)); ax.set_yticks(range(K_FIXED))
        ax.set_xlabel("v5-B aligned cluster")
        ax.set_ylabel("v5-A aligned cluster")
        ax.set_title(f"{cube_id}\ndiag agreement: {100*diag:.1f}%  "
                     f"n={n}", fontsize=10)
        for i in range(K_FIXED):
            for j in range(K_FIXED):
                ax.text(j, i, f"{Mn[i, j]:.2f}", ha="center", va="center",
                        color="white" if Mn[i, j] > 0.5 else "black",
                        fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle("v5-A vs v5-B Hungarian-aligned cluster confusion per cube",
                 fontsize=13)
    p = V5_ROOT / "figures" / "comparison_v5a_v5b_GMM_aligned.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


# =============================================================================
# Main per-cube driver
# =============================================================================
def process_cube(cube_id: str) -> dict:
    t0 = time.time()
    print(f"\n=== {cube_id} v5 ===")
    v4_arr = load_v4_vb_arrays(cube_id)
    vb_stack_v4 = v4_arr["vb_stack"]  # (H, W, 11)
    H, W, _ = vb_stack_v4.shape

    # v5-A: 7 params
    sub_a, valid_a = build_v5_substack(vb_stack_v4, V5A_PARAMS)
    print(f"  [v5a] valid pixels: {int(valid_a.sum())}/{valid_a.size}")
    run_a = run_pca_gmm(sub_a, valid_a, N_PCS_V5A)
    save_per_cube_loadings(cube_id, run_a, V5A_PARAMS, "v5a")
    print(f"  [v5a] PC1-5 cum={np.sum(run_a['explained_variance_ratio'][:5])*100:.1f}%  "
          f"BIC={run_a['bic']:.0f}")

    # v5-B: 5 params
    sub_b, valid_b = build_v5_substack(vb_stack_v4, V5B_PARAMS)
    print(f"  [v5b] valid pixels: {int(valid_b.sum())}/{valid_b.size}")
    run_b = run_pca_gmm(sub_b, valid_b, N_PCS_V5B)
    save_per_cube_loadings(cube_id, run_b, V5B_PARAMS, "v5b")
    print(f"  [v5b] PC1-5 cum={np.sum(run_b['explained_variance_ratio'][:5])*100:.1f}%  "
          f"BIC={run_b['bic']:.0f}")

    # Save arrays + counts
    save_per_cube_arrays(cube_id, run_a, run_b, valid_a, valid_b, (H, W))

    # Per-cube GMM scene maps (raw labels, per-cube color order)
    counts_path = (cube_root(cube_id) / "arrays"
                   / f"{cube_id}_v5_pixel_counts.json")
    counts = json.loads(counts_path.read_text())
    label_scene_a = np.full((H, W), -1, dtype=np.int16)
    label_scene_a[valid_a] = run_a["labels"]
    fig_gmm_scene_per_cube(cube_id, label_scene_a, counts, "v5a")
    label_scene_b = np.full((H, W), -1, dtype=np.int16)
    label_scene_b[valid_b] = run_b["labels"]
    fig_gmm_scene_per_cube(cube_id, label_scene_b, counts, "v5b")

    # Load Brown maps for this cube (independent of TER3)
    print(f"  [brown] loading Brown maps (no TER3)...")
    _, brown = find_cube_and_brown_maps(cube_dir(cube_id))
    full_valid = (np.isfinite(brown["centroid"])
                  & np.isfinite(brown["amplitude"])
                  & np.isfinite(brown["hwhm"])
                  & np.isfinite(brown["asymmetry"]))
    n_full = int(full_valid.sum())
    print(f"  [brown] full-scene Brown-finite pixels: {n_full}")

    print(f"  done in {time.time()-t0:.1f}s")
    return {
        "shape": (H, W),
        "v5a": run_a, "v5b": run_b,
        "valid_a": valid_a, "valid_b": valid_b,
        "brown": brown, "full_valid": full_valid,
    }


def main(argv: list[str]) -> None:
    t0 = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    targets = list(CUBE_IDS) if len(argv) <= 1 else argv[1:]
    if any(t not in CUBE_IDS for t in targets):
        raise SystemExit(f"unknown cube; known: {CUBE_IDS}")
    V5_ROOT.mkdir(parents=True, exist_ok=True)
    (V5_ROOT / "figures").mkdir(parents=True, exist_ok=True)
    (V5_ROOT / "reports").mkdir(parents=True, exist_ok=True)

    per_cube = {}
    for cube_id in targets:
        per_cube[cube_id] = process_cube(cube_id)

    if set(targets) != set(CUBE_IDS):
        print(f"NOTE: cross-cube products require all four cubes; skipping.")
        return

    # ---- Hungarian alignment per stack ----
    print("\n=== cross-cube alignment ===")
    aln_a = align_stack(per_cube, V5A_PARAMS, "v5a")
    aln_b = align_stack(per_cube, V5B_PARAMS, "v5b")
    # Save aligned label scenes per cube
    for cube_id in CUBE_IDS:
        for name, aln, valid_key in (("v5a", aln_a, "valid_a"),
                                      ("v5b", aln_b, "valid_b")):
            valid = per_cube[cube_id][valid_key]
            scene = np.full(valid.shape, -1, dtype=np.int16)
            scene[valid] = aln["aligned_by_cube"][cube_id]
            np.save(cube_root(cube_id) / "arrays"
                    / f"{cube_id}_{name}_GMM_k6_labels_aligned.npy", scene)
    (V5_ROOT / "reports" / "03_v5_alignment.json").write_text(
        json.dumps({"v5a": aln_a["log"], "v5b": aln_b["log"]}, indent=2),
        encoding="utf-8")

    # ---- Cross-cube comparison figures ----
    print("=== cross-cube figures ===")
    fig_loadings_4cube_2stack(per_cube)
    fig_aligned_4cube(per_cube, aln_a["aligned_by_cube"], "v5a", "valid_a")
    fig_aligned_4cube(per_cube, aln_b["aligned_by_cube"], "v5b", "valid_b")
    fig_v5a_v5b_confusion(per_cube, aln_a["aligned_by_cube"],
                           aln_b["aligned_by_cube"])

    # ---- Brown RGB regeneration (full scene, both stretches) ----
    print("=== full-scene Brown RGB ===")
    full_browns = {c: {"brown": per_cube[c]["brown"],
                       "full_valid": per_cube[c]["full_valid"]}
                   for c in CUBE_IDS}
    v5_stretch = compute_v5_brown_stretch(full_browns)
    # Carry HWHM from v4.1 since it's not in the RGB but might be referenced
    v5_stretch.setdefault("hwhm", BROWN_STRETCH_V4P1["hwhm"])
    print(f"  v5 4-cube pooled p2/p98 stretches: {v5_stretch}")

    for cube_id in CUBE_IDS:
        b = per_cube[cube_id]["brown"]
        v = per_cube[cube_id]["full_valid"]
        fig_brown_fullscene_rgb(cube_id, b, v, BROWN_STRETCH_V4P1,
                                 "v4p1stretch", BROWN_STRETCH_NOTE_V4P1)
        fig_brown_fullscene_rgb(cube_id, b, v, v5_stretch,
                                 "v5stretch", BROWN_STRETCH_NOTE_V5)

    full_valid_by_cube = {c: per_cube[c]["full_valid"] for c in CUBE_IDS}
    fig_brown_fullscene_rgb_4cube(per_cube, full_valid_by_cube,
                                   BROWN_STRETCH_V4P1, "v4p1stretch",
                                   BROWN_STRETCH_NOTE_V4P1)
    fig_brown_fullscene_rgb_4cube(per_cube, full_valid_by_cube,
                                   v5_stretch, "v5stretch",
                                   BROWN_STRETCH_NOTE_V5)

    # ---- Run log ----
    log = {
        "stage": "v5_param_pruning",
        "run_timestamp_utc": timestamp,
        "random_seed": RANDOM_SEED,
        "v5a_params": list(V5A_PARAMS),
        "v5b_params": list(V5B_PARAMS),
        "v5_brown_stretch": v5_stretch,
        "wall_clock_seconds": round(time.time() - t0, 2),
        "source_code_md5": {
            "stage_v5_param_pruning.py": md5_of(Path(__file__)),
            "vb_parameters.py": md5_of(Path(__file__).with_name("vb_parameters.py")),
            "config.py": md5_of(Path(__file__).with_name("config.py")),
            "crism_io.py": md5_of(Path(__file__).with_name("crism_io.py")),
            "display.py": md5_of(Path(__file__).with_name("display.py")),
        },
    }
    (V5_ROOT / "reports" / f"v5_run_log_{timestamp}.json").write_text(
        json.dumps(log, indent=2), encoding="utf-8")
    print(f"\nv5 complete in {log['wall_clock_seconds']:.1f}s")


if __name__ == "__main__":
    main(sys.argv)
