"""Stage B-8 — cross-cube diagnostics.

Produces:
  - all_cubes_median_subtracted.png:  for each of the 15 params, overlay
    per-cube histograms after subtracting the per-cube median. Tests whether
    cube-to-cube shape is preserved once the bulk offset is removed (i.e.
    is the centroid heterogeneity a bulk shift, or is the distribution
    shape itself different?).
  - all_cubes_centroid_ks_wasserstein.png:  4x4 matrices of pairwise KS
    statistic and Wasserstein distance for the centroid distribution.
  - all_cubes_pca15_loadings_compare.png:  side-by-side PC1 and PC2
    loadings heatmaps, four cubes on one figure.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from config import CUBE_IDS, STAGE_B_PARAM_ORDER, out_dir

CUBE_COLOURS = {
    "FRT00003E12": "#1f77b4",
    "FRT0000871C": "#d62728",
    "FRT0000A4FC": "#2ca02c",
    "FRT0000B438": "#9467bd",
}


def load_valid(cube_id: str) -> dict[str, np.ndarray]:
    data = dict(np.load(
        out_dir("feature_stacks") / f"{cube_id}_stackB_15param.npz",
        allow_pickle=True,
    ))
    valid = data["joint_valid_15"].astype(bool)
    return {name: data[name][valid].astype(np.float32) for name in STAGE_B_PARAM_ORDER}


def median_subtracted_overlay() -> Path:
    per_cube = {c: load_valid(c) for c in CUBE_IDS}
    fig, axes = plt.subplots(3, 5, figsize=(22, 12), constrained_layout=True)
    for ax, name in zip(axes.ravel(), STAGE_B_PARAM_ORDER):
        for cube_id in CUBE_IDS:
            arr = per_cube[cube_id][name]
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            med = float(np.median(arr))
            sub = arr - med
            lo, hi = np.quantile(sub, [0.001, 0.999])
            ax.hist(
                sub, bins=120, range=(lo, hi), histtype="step",
                density=True, color=CUBE_COLOURS[cube_id], lw=1.3,
                label=f"{cube_id} (med={med:+.3g})",
            )
        ax.set_title(name)
        ax.set_xlabel(f"{name} - median")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Four-cube median-subtracted distributions (density, 0.1-99.9% range)"
    )
    stem = out_dir("figures") / "all_cubes_median_subtracted"
    fig.savefig(stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return stem


def pairwise_centroid_stats() -> Path:
    per_cube = {c: load_valid(c)["centroid"] for c in CUBE_IDS}
    n = len(CUBE_IDS)
    ks_stat = np.zeros((n, n))
    ws_dist = np.zeros((n, n))
    for i, ci in enumerate(CUBE_IDS):
        ai = per_cube[ci]
        ai = ai[np.isfinite(ai)]
        for j, cj in enumerate(CUBE_IDS):
            if i == j:
                ks_stat[i, j] = 0.0
                ws_dist[i, j] = 0.0
                continue
            aj = per_cube[cj]
            aj = aj[np.isfinite(aj)]
            ks_stat[i, j] = ks_2samp(ai, aj, method="asymp").statistic
            ws_dist[i, j] = wasserstein_distance(ai, aj)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, mat, title, fmt in [
        (axes[0], ks_stat, "Kolmogorov-Smirnov statistic", "{:.3f}"),
        (axes[1], ws_dist, "Wasserstein distance (um)", "{:.4f}"),
    ]:
        im = ax.imshow(mat, cmap="viridis")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(CUBE_IDS, rotation=45, ha="right")
        ax.set_yticklabels(CUBE_IDS)
        for i in range(n):
            for j in range(n):
                ax.text(
                    j, i, fmt.format(mat[i, j]),
                    ha="center", va="center", fontsize=9,
                    color="white" if mat[i, j] > 0.5 * mat.max() else "black",
                )
        plt.colorbar(im, ax=ax, shrink=0.85)
        ax.set_title(title)
    fig.suptitle("Centroid-distribution pairwise distances (four cubes)")
    stem = out_dir("figures") / "all_cubes_centroid_ks_wasserstein"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Tabulate for the report.
    table_path = out_dir("reports") / "02_stageB_centroid_pairwise.csv"
    with open(table_path, "w") as f:
        f.write("metric,a,b,value\n")
        for i, ci in enumerate(CUBE_IDS):
            for j, cj in enumerate(CUBE_IDS):
                if j <= i:
                    continue
                f.write(f"ks,{ci},{cj},{ks_stat[i,j]:.6f}\n")
                f.write(f"wasserstein,{ci},{cj},{ws_dist[i,j]:.6f}\n")
    print(f"  wrote {table_path}")
    return stem


def pca_loadings_compare() -> Path:
    # Load 15-param PCA for each cube and plot loadings side-by-side.
    fig, axes = plt.subplots(2, 4, figsize=(22, 8), constrained_layout=True)
    for col, cube_id in enumerate(CUBE_IDS):
        pca_path = out_dir("embeddings") / f"{cube_id}_pca_15param.npz"
        data = np.load(pca_path, allow_pickle=True)
        comps = data["components"][:2]
        names = data["param_names"]
        vmax = np.max(np.abs(comps))
        for row in range(2):
            ax = axes[row, col]
            loadings_row = comps[row]
            ax.bar(range(len(names)), loadings_row,
                   color=["#d62728" if v < 0 else "#1f77b4" for v in loadings_row])
            ax.axhline(0, color="black", lw=0.5)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
            ax.set_ylim(-vmax * 1.1, vmax * 1.1)
            ax.set_ylabel(f"PC{row+1} loading")
            evr = data["explained_variance_ratio"][row]
            ax.set_title(f"{cube_id} PC{row+1} ({evr*100:.1f}%)")
    fig.suptitle("15-param PCA loadings — four cubes")
    stem = out_dir("figures") / "all_cubes_pca15_loadings_compare"
    fig.savefig(stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return stem


def main() -> None:
    t0 = time.time()
    print("median-subtracted overlay...")
    median_subtracted_overlay()
    print("pairwise centroid stats...")
    pairwise_centroid_stats()
    print("pca loadings compare...")
    pca_loadings_compare()
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
