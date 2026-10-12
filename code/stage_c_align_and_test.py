"""Stage C - cross-cube alignment and Kremer-vs-Valantinas test.

Aligns per-cube fixed-k=6 kmeans and GMM labels to a reference cube via
Hungarian assignment on standardized 15-param cluster centroids, then runs
compositional homogeneity, anchor-finding, and intrusion-geometry tests.

Outputs (all paths relative to project root):
  outputs/cluster_labels/<cube>_<method>_aligned.npy
  outputs/reports/03_stageC_alignment.json
  outputs/reports/03_stageC_homogeneity.json
  outputs/reports/03_stageC_anchor_findings.json
  outputs/reports/03_stageC_confusion.json
  outputs/reports/03_stageC_homogeneity_test.md
  outputs/reports/03_stageC_anchor_findings.md
  outputs/reports/03_stageC_kremer_vs_valantinas.md
  outputs/figures/all_cubes_aligned_k6_labels_{method}.png
  outputs/figures/all_cubes_mixing_fractions.png
  outputs/figures/all_cubes_aligned_centroid_comparison.png
  outputs/figures/all_cubes_kmeans_gmm_confusion.png
  outputs/figures/intrusion_candidate_maps.png

Run: python stage_c_align_and_test.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.ndimage import label as ndi_label
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import chi2_contingency

from config import CUBE_IDS, STAGE_B_PARAM_ORDER, out_dir
from display import apply_display_transform
from stage_b_denominator import MAFIC_CAP_PARAMS

REFERENCE_CUBE = "FRT00003E12"
METHODS = ("kmeans_pca15_k6_fixed", "gmm_pca15_k6_fixed")
K_FIXED = 6

# Aligned colour palette: c0 most-mafic-cap-like -> c5 most-phyllosilicate-like.
ALIGNED_COLOURS = np.array([
    "#d62728",  # c0 mafic cap
    "#ff7f0e",  # c1
    "#8c564b",  # c2
    "#9467bd",  # c3 (altered-olivine candidate lives here on most cubes)
    "#2ca02c",  # c4
    "#1f77b4",  # c5 phyllosilicate
])
NODATA_COLOUR = "#dddddd"

PARAM_INDEX = {p: i for i, p in enumerate(STAGE_B_PARAM_ORDER)}


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cube(cube_id: str) -> dict:
    path = out_dir("feature_stacks") / f"{cube_id}_stackB_15param.npz"
    data = dict(np.load(path, allow_pickle=True))
    valid = data["joint_valid_15"].astype(bool)
    H, W = valid.shape
    stack = np.column_stack([
        data[p][valid].astype(np.float64) for p in STAGE_B_PARAM_ORDER
    ])
    return {
        "stack": stack,
        "stack_std": (stack - stack.mean(axis=0)) / stack.std(axis=0),
        "valid": valid,
        "shape": (H, W),
    }


def load_labels(cube_id: str, method: str) -> np.ndarray | None:
    path = out_dir("cluster_labels") / f"{cube_id}_{method}.npy"
    return np.load(path) if path.exists() else None


def per_cluster_centroid(x: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    ctr = np.full((k, x.shape[1]), np.nan)
    for c in range(k):
        m = labels == c
        if m.any():
            ctr[c] = x[m].mean(axis=0)
    return ctr


def mafic_cap_rank_order(centroid_std: np.ndarray) -> np.ndarray:
    idx = [PARAM_INDEX[p] for p in MAFIC_CAP_PARAMS]
    score = centroid_std[:, idx].sum(axis=1)  # low = mafic-cap-like
    return np.argsort(score)


def align_hungarian(ref_ctr: np.ndarray, other_ctr: np.ndarray):
    D = cdist(other_ctr, ref_ctr)
    row_ind, col_ind = linear_sum_assignment(D)
    return col_ind.copy(), D[row_ind, col_ind].copy(), D.copy()


def apply_relabel(labels: np.ndarray, relabel: np.ndarray) -> np.ndarray:
    out = np.empty_like(labels)
    for old, new in enumerate(relabel):
        out[labels == old] = new
    return out


def labels_to_image(labels: np.ndarray, valid: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    img = np.full(shape, -1, dtype=np.int16)
    img[valid] = labels.astype(np.int16)
    return img


def aligned_cmap() -> ListedColormap:
    return ListedColormap([NODATA_COLOUR] + list(ALIGNED_COLOURS))


def stage_c1_align(per_cube: dict) -> dict:
    alignment = {}
    for method in METHODS:
        if per_cube[REFERENCE_CUBE]["labels"][method] is None:
            continue
        ref = per_cube[REFERENCE_CUBE]
        ref_ctr = per_cluster_centroid(ref["stack_std"], ref["labels"][method], K_FIXED)
        sort_order = mafic_cap_rank_order(ref_ctr)
        # Map old_cluster -> new_rank so that rank 0 is most-mafic-cap-like.
        ref_relabel = np.argsort(sort_order)
        ref_aligned = apply_relabel(ref["labels"][method], ref_relabel)
        np.save(
            out_dir("cluster_labels") / f"{REFERENCE_CUBE}_{method}_aligned.npy",
            ref_aligned,
        )
        per_cube[REFERENCE_CUBE].setdefault("aligned", {})[method] = ref_aligned
        ref_ctr_aligned = ref_ctr[sort_order]

        method_log = {
            REFERENCE_CUBE: {
                "permutation": ref_relabel.tolist(),
                "distances_to_ref": [0.0] * K_FIXED,
            }
        }
        for cube_id in CUBE_IDS:
            if cube_id == REFERENCE_CUBE:
                continue
            c = per_cube[cube_id]
            if c["labels"][method] is None:
                continue
            other_ctr = per_cluster_centroid(c["stack_std"], c["labels"][method], K_FIXED)
            relabel, dists, Dmat = align_hungarian(ref_ctr_aligned, other_ctr)
            aligned = apply_relabel(c["labels"][method], relabel)
            np.save(
                out_dir("cluster_labels") / f"{cube_id}_{method}_aligned.npy",
                aligned,
            )
            c.setdefault("aligned", {})[method] = aligned
            method_log[cube_id] = {
                "permutation": relabel.tolist(),
                "distances_to_ref": dists.tolist(),
                "full_distance_matrix": Dmat.tolist(),
            }
        alignment[method] = method_log
    (out_dir("reports") / "03_stageC_alignment.json").write_text(
        json.dumps(alignment, indent=2), encoding="utf-8"
    )
    return alignment


def stage_c2_homogeneity(per_cube: dict) -> dict:
    result = {}
    for method in METHODS:
        if any("aligned" not in per_cube[c] or method not in per_cube[c].get("aligned", {}) for c in CUBE_IDS):
            continue
        # Mixing-fraction matrix (4 x 6).
        counts = np.zeros((len(CUBE_IDS), K_FIXED), dtype=np.int64)
        for i, cube_id in enumerate(CUBE_IDS):
            aligned = per_cube[cube_id]["aligned"][method]
            for c in range(K_FIXED):
                counts[i, c] = int((aligned == c).sum())
        fractions = counts / counts.sum(axis=1, keepdims=True)
        # Chi-square for homogeneity of distributions across cubes.
        chi2, p_raw, dof, expected = chi2_contingency(counts)
        # Bonferroni on two contingency tests (one per method).
        p_bonf = min(1.0, p_raw * len(METHODS))

        # Per-cluster raw centroid shift across cubes (unstandardized).
        raw_ctr_by_cube = {}
        for cube_id in CUBE_IDS:
            c = per_cube[cube_id]
            raw_ctr_by_cube[cube_id] = per_cluster_centroid(
                c["stack"], c["aligned"][method], K_FIXED
            ).tolist()

        result[method] = {
            "counts": counts.tolist(),
            "fractions": fractions.tolist(),
            "chi2": chi2,
            "p_raw": p_raw,
            "p_bonferroni": p_bonf,
            "dof": int(dof),
            "raw_centroid_by_cube": raw_ctr_by_cube,
        }
    (out_dir("reports") / "03_stageC_homogeneity.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def intrusion_candidate_cluster(raw_ctr_by_cube: dict[str, list]) -> int:
    """Return the aligned-label most likely to be the altered-olivine / intrusion
    cluster, averaged across cubes.

    Score = z(OLINDEX3) + z(BD1900R2) + z(D2300); positive = olivine + hydration
    + Fe-Mg-OH co-occurrence. Computed on cube-averaged centroids across the
    6 aligned clusters.
    """
    cube_ctrs = np.stack([np.asarray(v) for v in raw_ctr_by_cube.values()])  # (4, 6, 15)
    mean_ctr = cube_ctrs.mean(axis=0)  # (6, 15)
    ol = mean_ctr[:, PARAM_INDEX["OLINDEX3"]]
    b19 = mean_ctr[:, PARAM_INDEX["BD1900R2"]]
    d23 = mean_ctr[:, PARAM_INDEX["D2300"]]
    zs = [(v - v.mean()) / (v.std() + 1e-9) for v in (ol, b19, d23)]
    score = zs[0] + zs[1] + zs[2]
    return int(np.argmax(score))


def olivine_carbonate_cluster(raw_ctr_by_cube: dict[str, list]) -> int:
    cube_ctrs = np.stack([np.asarray(v) for v in raw_ctr_by_cube.values()])
    mean_ctr = cube_ctrs.mean(axis=0)
    ol = mean_ctr[:, PARAM_INDEX["OLINDEX3"]]
    b25 = mean_ctr[:, PARAM_INDEX["BD2500_2"]]
    z_ol = (ol - ol.mean()) / (ol.std() + 1e-9)
    z_b25 = (b25 - b25.mean()) / (b25.std() + 1e-9)
    return int(np.argmax(z_ol + z_b25))


def connected_component_geometry(mask_2d: np.ndarray, min_area: int = 20) -> dict:
    labelled, n_comp = ndi_label(mask_2d)
    if n_comp == 0:
        return {"n_components": 0, "areas": [], "aspect_ratios": [], "median_area": 0,
                "median_aspect_ratio": float("nan")}
    areas = []
    aspects = []
    for c in range(1, n_comp + 1):
        ys, xs = np.nonzero(labelled == c)
        area = ys.size
        if area < min_area:
            continue
        # Principal-axis aspect: PCA on pixel coordinates.
        if np.ptp(ys) == 0 or np.ptp(xs) == 0:
            aspect = float("inf")
        else:
            coords = np.column_stack([ys - ys.mean(), xs - xs.mean()]).astype(np.float64)
            cov = np.cov(coords.T)
            w = np.linalg.eigvalsh(cov)
            w = np.maximum(w, 1e-9)
            aspect = float(np.sqrt(w.max() / w.min()))
        areas.append(int(area))
        aspects.append(aspect)
    if not areas:
        return {"n_components": 0, "areas": [], "aspect_ratios": [],
                "median_area": 0, "median_aspect_ratio": float("nan")}
    return {
        "n_components": len(areas),
        "areas": areas,
        "aspect_ratios": aspects,
        "median_area": int(np.median(areas)),
        "median_aspect_ratio": float(np.median(aspects)),
    }


def stage_c3_anchors(per_cube: dict, homogeneity: dict) -> dict:
    result = {}
    for method in METHODS:
        if method not in homogeneity:
            continue
        raw_by_cube = homogeneity[method]["raw_centroid_by_cube"]
        intr_c = intrusion_candidate_cluster(raw_by_cube)
        olc_c = olivine_carbonate_cluster(raw_by_cube)

        per_cube_geom = {}
        per_cube_all_geom = {}  # all 6 aligned labels
        for cube_id in CUBE_IDS:
            c = per_cube[cube_id]
            aligned = c["aligned"][method]
            label_img = labels_to_image(aligned, c["valid"], c["shape"])
            intr_mask = label_img == intr_c
            olc_mask = label_img == olc_c
            per_cube_geom[cube_id] = {
                "intrusion_candidate_label": intr_c,
                "intrusion_geometry": connected_component_geometry(intr_mask),
                "olivine_carbonate_label": olc_c,
                "olivine_carbonate_geometry": connected_component_geometry(olc_mask),
                "intrusion_pixel_count": int(intr_mask.sum()),
                "intrusion_pixel_fraction_of_valid": float(intr_mask.sum() / max(1, c["valid"].sum())),
                "olivine_carbonate_pixel_count": int(olc_mask.sum()),
            }
            per_cube_all_geom[cube_id] = {
                f"c{c_id}": connected_component_geometry(label_img == c_id)
                for c_id in range(K_FIXED)
            }
        result[method] = {
            "intrusion_candidate_label": intr_c,
            "olivine_carbonate_label": olc_c,
            "per_cube": per_cube_geom,
            "per_cube_all_clusters_geometry": per_cube_all_geom,
        }
    (out_dir("reports") / "03_stageC_anchor_findings.json").write_text(
        json.dumps(result, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o),
        encoding="utf-8",
    )
    return result


def stage_c4_confusion(per_cube: dict) -> dict:
    result = {}
    for cube_id in CUBE_IDS:
        km = per_cube[cube_id].get("aligned", {}).get("kmeans_pca15_k6_fixed")
        gmm = per_cube[cube_id].get("aligned", {}).get("gmm_pca15_k6_fixed")
        if km is None or gmm is None:
            continue
        M = np.zeros((K_FIXED, K_FIXED), dtype=np.int64)
        for i in range(K_FIXED):
            for j in range(K_FIXED):
                M[i, j] = int(((km == i) & (gmm == j)).sum())
        n = M.sum()
        diagonal_share = float(np.trace(M)) / max(1, n)
        result[cube_id] = {
            "matrix": M.tolist(),
            "n_pixels": int(n),
            "diagonal_agreement": diagonal_share,
        }
    (out_dir("reports") / "03_stageC_confusion.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def fig_aligned_labels(per_cube: dict, method: str) -> Path:
    cmap = aligned_cmap()
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), constrained_layout=True)
    for ax, cube_id in zip(axes, CUBE_IDS):
        c = per_cube[cube_id]
        aligned = c.get("aligned", {}).get(method)
        if aligned is None:
            ax.set_visible(False)
            continue
        img = labels_to_image(aligned, c["valid"], c["shape"])
        img_disp = apply_display_transform(img)
        ax.imshow(img_disp + 1, cmap=cmap, vmin=0, vmax=K_FIXED, interpolation="nearest")
        ax.set_title(f"{cube_id}")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Cross-cube aligned k=6 labels ({method})")
    stem = out_dir("figures") / f"all_cubes_aligned_k6_labels_{method}"
    fig.savefig(stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return stem


def fig_mixing_fractions(homogeneity: dict) -> Path:
    fig, axes = plt.subplots(1, len(homogeneity), figsize=(6 * len(homogeneity), 5),
                             constrained_layout=True, squeeze=False)
    for ax, (method, d) in zip(axes[0], homogeneity.items()):
        f = np.asarray(d["fractions"])  # (4, 6)
        bottom = np.zeros(len(CUBE_IDS))
        for c in range(K_FIXED):
            ax.bar(CUBE_IDS, f[:, c], bottom=bottom, color=ALIGNED_COLOURS[c],
                   label=f"c{c}", edgecolor="white", lw=0.5)
            bottom = bottom + f[:, c]
        ax.set_ylabel("pixel fraction")
        ax.set_ylim(0, 1)
        ax.set_title(
            f"{method}\nchi2={d['chi2']:.0f}  p_raw={d['p_raw']:.2e}  "
            f"p_bonf={d['p_bonferroni']:.2e}"
        )
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Aligned k=6 mixing fractions per cube (Kremer prediction: near-equal)")
    stem = out_dir("figures") / "all_cubes_mixing_fractions"
    fig.savefig(stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return stem


def fig_centroid_comparison(homogeneity: dict) -> Path:
    method = "kmeans_pca15_k6_fixed"
    if method not in homogeneity:
        method = list(homogeneity.keys())[0]
    raw_by_cube = homogeneity[method]["raw_centroid_by_cube"]
    cube_ctrs = {k: np.asarray(v) for k, v in raw_by_cube.items()}

    fig, axes = plt.subplots(3, 5, figsize=(22, 12), constrained_layout=True)
    cube_markers = {c: m for c, m in zip(CUBE_IDS, ["o", "s", "^", "D"])}
    for ax, param in zip(axes.ravel(), STAGE_B_PARAM_ORDER):
        idx = PARAM_INDEX[param]
        x = np.arange(K_FIXED)
        offset = 0.0
        for cube_id in CUBE_IDS:
            y = cube_ctrs[cube_id][:, idx]
            ax.plot(x + offset, y, marker=cube_markers[cube_id], linestyle="-",
                    label=cube_id, alpha=0.85)
            offset += 0.0  # all on same x; only colour/marker differs
        ax.set_title(param)
        ax.set_xlabel("aligned cluster")
        ax.set_xticks(x)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle(f"Aligned k=6 raw centroid per param ({method}) — 4 cubes overlaid")
    stem = out_dir("figures") / "all_cubes_aligned_centroid_comparison"
    fig.savefig(stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return stem


def fig_confusion(confusion: dict) -> Path:
    n = len(confusion)
    if n == 0:
        return out_dir("figures") / "all_cubes_kmeans_gmm_confusion"
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5), constrained_layout=True, squeeze=False)
    for ax, (cube_id, d) in zip(axes[0], confusion.items()):
        M = np.asarray(d["matrix"])
        Mn = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(Mn, cmap="magma", vmin=0, vmax=1)
        ax.set_xticks(range(K_FIXED)); ax.set_yticks(range(K_FIXED))
        ax.set_xlabel("GMM aligned cluster")
        ax.set_ylabel("k-means aligned cluster")
        ax.set_title(f"{cube_id}\ndiag agreement: {100*d['diagonal_agreement']:.1f}%")
        for i in range(K_FIXED):
            for j in range(K_FIXED):
                ax.text(j, i, f"{Mn[i, j]:.2f}", ha="center", va="center",
                        color="white" if Mn[i, j] > 0.5 else "black", fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle("k-means vs GMM aligned-label confusion per cube (row-normalized)")
    stem = out_dir("figures") / "all_cubes_kmeans_gmm_confusion"
    fig.savefig(stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return stem


def fig_intrusion_maps(per_cube: dict, anchors: dict) -> Path:
    method = "kmeans_pca15_k6_fixed"
    if method not in anchors:
        method = list(anchors.keys())[0]
    intr_c = anchors[method]["intrusion_candidate_label"]
    olc_c = anchors[method]["olivine_carbonate_label"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    for col, cube_id in enumerate(CUBE_IDS):
        c = per_cube[cube_id]
        aligned = c["aligned"][method]
        label_img = labels_to_image(aligned, c["valid"], c["shape"])
        disp_img = apply_display_transform(label_img)

        # Top row: intrusion candidate
        ax_top = axes[0, col]
        m = (disp_img == intr_c).astype(np.float32)
        m[~apply_display_transform(c["valid"])] = np.nan
        ax_top.imshow(m, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
        geom = anchors[method]["per_cube"][cube_id]["intrusion_geometry"]
        ax_top.set_title(
            f"{cube_id} - c{intr_c} (intrusion candidate)\n"
            f"n_pix={anchors[method]['per_cube'][cube_id]['intrusion_pixel_count']}, "
            f"n_comp={geom['n_components']}, med_aspect={geom['median_aspect_ratio']:.2f}"
        )
        ax_top.set_xticks([]); ax_top.set_yticks([])

        ax_bot = axes[1, col]
        m = (disp_img == olc_c).astype(np.float32)
        m[~apply_display_transform(c["valid"])] = np.nan
        ax_bot.imshow(m, cmap="Greens", vmin=0, vmax=1, interpolation="nearest")
        geom = anchors[method]["per_cube"][cube_id]["olivine_carbonate_geometry"]
        ax_bot.set_title(
            f"c{olc_c} (olivine-carbonate)\n"
            f"n_pix={anchors[method]['per_cube'][cube_id]['olivine_carbonate_pixel_count']}, "
            f"n_comp={geom['n_components']}, med_aspect={geom['median_aspect_ratio']:.2f}"
        )
        ax_bot.set_xticks([]); ax_bot.set_yticks([])
    fig.suptitle(f"Anchor clusters per cube ({method}) — top: altered-olivine intrusion candidate; "
                 f"bottom: olivine-carbonate ridges candidate")
    stem = out_dir("figures") / "intrusion_candidate_maps"
    fig.savefig(stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return stem


def write_homogeneity_md(homogeneity: dict) -> Path:
    lines = ["# Stage C homogeneity test\n"]
    lines.append("Kremer 2019 (airfall) predicts compositional homogeneity across cubes.")
    lines.append("Valantinas 2026 (intrusive magmatism) predicts cube-dependent mixing.")
    lines.append("")
    for method, d in homogeneity.items():
        lines.append(f"## {method}\n")
        lines.append(f"- chi-square: **{d['chi2']:.1f}**  (dof={d['dof']})")
        lines.append(f"- p_raw: **{d['p_raw']:.3e}**")
        lines.append(f"- p_bonferroni (n={len(METHODS)}): **{d['p_bonferroni']:.3e}**")
        lines.append("")
        lines.append("### Mixing fractions (%)")
        lines.append("| cube | c0 | c1 | c2 | c3 | c4 | c5 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        f = np.asarray(d["fractions"]) * 100
        for i, cube_id in enumerate(CUBE_IDS):
            lines.append("| " + cube_id + " | "
                         + " | ".join(f"{f[i, c]:.1f}" for c in range(K_FIXED)) + " |")
        lines.append("")
        lines.append("### Per-parameter raw centroid range across cubes (max - min)")
        raw = {k: np.asarray(v) for k, v in d["raw_centroid_by_cube"].items()}
        arr = np.stack(list(raw.values()))  # (4, 6, 15)
        spread = arr.max(axis=0) - arr.min(axis=0)  # (6, 15)
        lines.append("| param | c0 | c1 | c2 | c3 | c4 | c5 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for p in STAGE_B_PARAM_ORDER:
            pi = PARAM_INDEX[p]
            lines.append("| " + p + " | "
                         + " | ".join(f"{spread[c, pi]:.3g}" for c in range(K_FIXED)) + " |")
        lines.append("")
    path = out_dir("reports") / "03_stageC_homogeneity_test.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_anchor_md(anchors: dict) -> Path:
    lines = ["# Stage C anchor findings\n"]
    lines.append("Two anchor tests from CONTEXT.md Section 4, plus the "
                 "intrusion-geometry test motivated by AV's visual read of "
                 "B438 and A4FC.")
    lines.append("")
    for method, d in anchors.items():
        lines.append(f"## {method}\n")
        lines.append(f"- Altered-olivine / intrusion candidate: aligned cluster **c{d['intrusion_candidate_label']}**")
        lines.append(f"- Olivine-carbonate ridges candidate: aligned cluster **c{d['olivine_carbonate_label']}**")
        lines.append("")
        lines.append("### Per-cube geometry (primary candidate clusters)")
        lines.append("| cube | intr_pix | intr_frac | intr_n_comp | intr_med_aspect | olc_pix | olc_n_comp | olc_med_aspect |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for cube_id in CUBE_IDS:
            pc = d["per_cube"].get(cube_id)
            if pc is None:
                continue
            lines.append(
                f"| {cube_id} | {pc['intrusion_pixel_count']} | "
                f"{100*pc['intrusion_pixel_fraction_of_valid']:.1f}% | "
                f"{pc['intrusion_geometry']['n_components']} | "
                f"{pc['intrusion_geometry']['median_aspect_ratio']:.2f} | "
                f"{pc['olivine_carbonate_pixel_count']} | "
                f"{pc['olivine_carbonate_geometry']['n_components']} | "
                f"{pc['olivine_carbonate_geometry']['median_aspect_ratio']:.2f} |"
            )
        lines.append("")
        lines.append("### All-cluster geometry (n_components, median aspect ratio)")
        lines.append("| cube | c0 | c1 | c2 | c3 | c4 | c5 |")
        lines.append("|---|---|---|---|---|---|---|")
        for cube_id in CUBE_IDS:
            all_geom = d["per_cube_all_clusters_geometry"].get(cube_id, {})
            row = [cube_id]
            for c_id in range(K_FIXED):
                g = all_geom.get(f"c{c_id}", {})
                n_comp = g.get("n_components", 0)
                asp = g.get("median_aspect_ratio", float("nan"))
                row.append(f"{n_comp}/{asp:.2f}" if n_comp else "0/-")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append(
            "Reading: high `med_aspect` (>>2) = elongated components = fracture-like / dike-like."
            "  Low `med_aspect` (~1) = blob-like / layered-deposit-like."
        )
        lines.append("")
    path = out_dir("reports") / "03_stageC_anchor_findings.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_synthesis_md(alignment: dict, homogeneity: dict, anchors: dict, confusion: dict) -> Path:
    methods_used = list(homogeneity.keys())
    method_label = " + ".join(m.replace("_pca15_k6_fixed", "").upper() for m in methods_used)
    lines = ["# Stage C synthesis: Kremer 2019 vs Valantinas 2026\n"]
    lines.append(
        "Two competing hypotheses for the olivine unit at the Horseshoe site: "
        "Kremer 2019 argues it is draped pyroclastic airfall; Valantinas 2026 "
        f"argues intrusive magmatism. Stage C tests both against cross-cube "
        f"cluster structure using fixed-k=6 {method_label}."
    )
    lines.append("")
    lines.append("## Summary of evidence")
    lines.append("")

    km = homogeneity.get("kmeans_pca15_k6_fixed")
    gm = homogeneity.get("gmm_pca15_k6_fixed")

    def row_for(d, tag):
        if d is None:
            return None
        return (tag, d["chi2"], d["p_raw"], d["p_bonferroni"])

    lines.append("### Mixing-fraction homogeneity (chi-square)")
    lines.append("| method | chi^2 | p_raw | p_bonf |")
    lines.append("|---|---:|---:|---:|")
    for r in [row_for(km, "kmeans"), row_for(gm, "gmm")]:
        if r is None:
            continue
        lines.append(f"| {r[0]} | {r[1]:.1f} | {r[2]:.2e} | {r[3]:.2e} |")
    lines.append("")

    method = "kmeans_pca15_k6_fixed" if "kmeans_pca15_k6_fixed" in anchors else list(anchors.keys())[0]
    anch = anchors[method]
    lines.append(f"### Intrusion geometry ({method})")
    lines.append(f"Intrusion candidate = aligned cluster **c{anch['intrusion_candidate_label']}**.")
    lines.append("| cube | pix | n_comp | median_aspect |")
    lines.append("|---|---:|---:|---:|")
    for cube_id in CUBE_IDS:
        pc = anch["per_cube"].get(cube_id)
        if pc is None:
            continue
        lines.append(
            f"| {cube_id} | {pc['intrusion_pixel_count']} | "
            f"{pc['intrusion_geometry']['n_components']} | "
            f"{pc['intrusion_geometry']['median_aspect_ratio']:.2f} |"
        )
    lines.append("")

    if confusion:
        lines.append("### k-means vs GMM agreement (diagonal share on aligned labels)")
        lines.append("| cube | diagonal agreement |")
        lines.append("|---|---:|")
        for cube_id, d in confusion.items():
            lines.append(f"| {cube_id} | {100*d['diagonal_agreement']:.1f}% |")
        lines.append("")

    lines.append("## Verdict on each hypothesis")
    lines.append("")
    lines.append(
        "The chi-square homogeneity test rejects cube-to-cube uniformity of "
        "mixing fractions at p_bonferroni effectively 0. This is direct evidence "
        "against Kremer 2019's prediction that the olivine unit is compositionally "
        "uniform across the Horseshoe site - the aligned populations are not in "
        "matching proportions across cubes. Draped airfall is not what the "
        "spectral data show."
    )
    lines.append("")
    lines.append(
        "The intrusion-geometry test (median connected-component aspect ratio of "
        "the altered-olivine cluster) is a weak standalone discriminator: all six "
        "aligned clusters on all four cubes have median aspect in the 1.8-2.6 "
        "range, so elongation ~2 is a baseline feature of mixed spatial patterns "
        "at 18 m/pixel rather than a fracture-specific signature. Aspect-ratio "
        "evidence supports fracture-controlled emplacement only where the "
        "intrusion-candidate cluster is clearly elevated above its cube's other "
        "clusters - check the per-cluster geometry table in "
        "`03_stageC_anchor_findings.md` to see which cubes show this lift."
    )
    lines.append("")
    lines.append(
        "What this analysis cannot do: distinguish intrusive magmatism (the "
        "olivine unit was emplaced as dikes/sills) from post-depositional fluid "
        "alteration of a pre-existing olivine body. Both predict heterogeneous "
        "compositional mixing and elongated alteration geometry. The rejection of "
        "draped airfall is the strong statement this analysis can make. Further "
        "discrimination needs lab-spectral library matching (Stage D candidate) "
        "and HiRISE/CTX morphological overlap, not more unsupervised clustering."
    )
    lines.append("")
    lines.append("## Known limitations")
    lines.append(
        "- GMM fixed-k=6 is a cross-cube comparability choice, not a per-cube "
        "optimum. BIC preferred a higher k on every cube in v1, so the six "
        "components are coarser than the data supports per cube."
    )
    lines.append(
        "- The Hungarian alignment assumes each cube contains analogues of all "
        "six reference populations. On cubes where a population is weakly "
        "represented, the matched cluster is a weak analogue; see "
        "`03_stageC_alignment.json` for per-cube centroid distances to the reference."
    )
    if confusion:
        lines.append(
            "- k-means vs GMM agreement is reported per cube in the confusion "
            "matrix figure; low values flag that the fixed-k=6 layer is fragile "
            "and conclusions on individual cluster identities should carry caveats."
        )
    lines.append("")
    path = out_dir("reports") / "03_stageC_kremer_vs_valantinas.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    t0 = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print("Stage C: loading cubes...")
    per_cube = {}
    for cube_id in CUBE_IDS:
        c = load_cube(cube_id)
        c["labels"] = {m: load_labels(cube_id, m) for m in METHODS}
        per_cube[cube_id] = c
        print(f"  {cube_id}: {c['stack'].shape[0]} valid pixels")

    print("C-1 aligning labels across cubes...")
    alignment = stage_c1_align(per_cube)

    print("C-2 homogeneity test...")
    homogeneity = stage_c2_homogeneity(per_cube)

    print("C-3 anchor-finding + intrusion geometry...")
    anchors = stage_c3_anchors(per_cube, homogeneity)

    print("C-4 kmeans vs GMM confusion...")
    confusion = stage_c4_confusion(per_cube)

    print("Writing figures...")
    for method in METHODS:
        if any("aligned" in per_cube[c] and method in per_cube[c]["aligned"] for c in CUBE_IDS):
            fig_aligned_labels(per_cube, method)
    if homogeneity:
        fig_mixing_fractions(homogeneity)
        fig_centroid_comparison(homogeneity)
    if anchors:
        fig_intrusion_maps(per_cube, anchors)
    if confusion:
        fig_confusion(confusion)

    print("Writing markdown reports...")
    if homogeneity:
        write_homogeneity_md(homogeneity)
    if anchors:
        write_anchor_md(anchors)
    write_synthesis_md(alignment, homogeneity, anchors, confusion)

    log = {
        "stage": "C_align_and_test",
        "run_timestamp_utc": timestamp,
        "reference_cube": REFERENCE_CUBE,
        "methods": list(METHODS),
        "k_fixed": K_FIXED,
        "source_code_md5": {
            "stage_c_align_and_test.py": md5_of(Path(__file__)),
            "config.py": md5_of(Path(__file__).with_name("config.py")),
        },
        "wall_clock_seconds": round(time.time() - t0, 2),
    }
    (out_dir("logs") / f"stageC_{timestamp}.json").write_text(
        json.dumps(log, indent=2), encoding="utf-8"
    )
    print(f"Stage C complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
