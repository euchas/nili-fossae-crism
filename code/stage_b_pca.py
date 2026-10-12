"""Stage B-1 — PCA on three feature stacks per cube.

For each cube:
  1. Load the Stage B 15-param feature stack (joint_valid_15 pixels).
  2. Run StandardScaler-then-PCA on three input stacks:
       - 4-param reduced (centroid, OLINDEX3, D2300, BD2500_2)
       - 8-param Stage A baseline (Brown 4 + 4 VB used in Stage A)
       - 15-param Stage B (the canonical clustering input)
  3. Save components, loadings, transformed coords, explained variance.
  4. Plot per-cube: variance bars, loadings heatmap, PC1-PC2 hexbin
     density, spatial PC1 / PC2 maps (display-flipped).

Outputs:
  outputs/embeddings/<cube>_pca_4reduced.npz
  outputs/embeddings/<cube>_pca_8param.npz
  outputs/embeddings/<cube>_pca_15param.npz
  outputs/figures/<cube>_pca15_variance.png
  outputs/figures/<cube>_pca15_loadings.png
  outputs/figures/<cube>_pca15_density.png
  outputs/figures/<cube>_pc1_pc2_maps.png
  outputs/logs/<cube>_stageB_pca_<TIMESTAMP>.json
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
from matplotlib.colors import LogNorm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import (
    CUBE_IDS,
    DEFAULT_PARAM_ORDER,
    RANDOM_SEED,
    STAGE_B_PARAM_ORDER,
    STAGE_B_REDUCED_4,
    out_dir,
)
from display import apply_display_transform


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def fit_pca(
    stack: dict[str, np.ndarray],
    valid_mask: np.ndarray,
    param_names: tuple[str, ...],
) -> dict:
    """Fit standardised PCA on the given parameter subset and pixel mask."""
    X = np.column_stack([stack[name][valid_mask] for name in param_names])
    finite = np.isfinite(X).all(axis=1)
    X = X[finite]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    n_components = X.shape[1]
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    Xp = pca.fit_transform(Xs)
    yy, xx = np.where(valid_mask)
    yy = yy[finite]
    xx = xx[finite]
    return {
        "param_names": param_names,
        "n_pixels": int(X.shape[0]),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "components": pca.components_,            # (n_components, n_features)
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "transformed": Xp.astype(np.float32),     # (n_pixels, n_components)
        "line": yy.astype(np.int32),
        "sample": xx.astype(np.int32),
    }


def save_pca(cube_id: str, label: str, result: dict) -> Path:
    path = out_dir("embeddings") / f"{cube_id}_pca_{label}.npz"
    np.savez_compressed(
        path,
        param_names=np.array(result["param_names"]),
        n_pixels=result["n_pixels"],
        scaler_mean=result["scaler_mean"],
        scaler_scale=result["scaler_scale"],
        components=result["components"],
        explained_variance_ratio=result["explained_variance_ratio"],
        transformed=result["transformed"],
        line=result["line"],
        sample=result["sample"],
    )
    return path


def plot_variance(result15: dict, result8: dict, result4: dict, cube_id: str) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for label, res, colour in [
        ("15-param", result15, "#1f77b4"),
        ("8-param", result8, "#d62728"),
        ("4-param reduced", result4, "#2ca02c"),
    ]:
        evr = res["explained_variance_ratio"]
        cumulative = np.cumsum(evr)
        x = np.arange(1, evr.size + 1)
        ax.bar(
            x + (-0.25 if label == "15-param" else 0.25 if label == "4-param reduced" else 0.0),
            evr,
            width=0.25,
            color=colour,
            alpha=0.75,
            label=f"{label} (cum at {evr.size}: {cumulative[-1]*100:.1f}%)",
        )
    ax.set_xlabel("PC index")
    ax.set_ylabel("explained variance ratio")
    ax.set_title(f"{cube_id} - PCA variance per stack")
    ax.legend(fontsize=9, loc="upper right")
    stem = out_dir("figures") / f"{cube_id}_pca15_variance"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return stem


def plot_loadings(result15: dict, cube_id: str, n_pcs: int = 5) -> Path:
    comps = result15["components"][:n_pcs]
    names = result15["param_names"]
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    vmax = np.max(np.abs(comps))
    im = ax.imshow(
        comps,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(n_pcs))
    ax.set_yticklabels([f"PC{i+1} ({result15['explained_variance_ratio'][i]*100:.1f}%)"
                        for i in range(n_pcs)])
    for i in range(n_pcs):
        for j in range(len(names)):
            ax.text(j, i, f"{comps[i, j]:+.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(comps[i, j]) > 0.5 else "black")
    plt.colorbar(im, ax=ax, label="loading")
    ax.set_title(f"{cube_id} - 15-param PCA loadings (top {n_pcs} PCs)")
    stem = out_dir("figures") / f"{cube_id}_pca15_loadings"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return stem


def plot_density(result15: dict, cube_id: str) -> Path:
    Xp = result15["transformed"]
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    hb = ax.hexbin(Xp[:, 0], Xp[:, 1], gridsize=80, cmap="viridis",
                   mincnt=1, norm=LogNorm())
    plt.colorbar(hb, ax=ax, label="pixels (log)")
    ax.axhline(0, color="gray", ls=":", lw=0.5, alpha=0.5)
    ax.axvline(0, color="gray", ls=":", lw=0.5, alpha=0.5)
    ax.set_xlabel(
        f"PC1 ({result15['explained_variance_ratio'][0]*100:.1f}%)"
    )
    ax.set_ylabel(
        f"PC2 ({result15['explained_variance_ratio'][1]*100:.1f}%)"
    )
    ax.set_title(f"{cube_id} - 15-param PCA density (PC1 vs PC2)")
    stem = out_dir("figures") / f"{cube_id}_pca15_density"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return stem


def plot_spatial_pcs(result15: dict, cube_id: str, shape: tuple[int, int]) -> Path:
    pc1 = np.full(shape, np.nan, dtype=np.float32)
    pc2 = np.full(shape, np.nan, dtype=np.float32)
    yy = result15["line"]
    xx = result15["sample"]
    pc1[yy, xx] = result15["transformed"][:, 0]
    pc2[yy, xx] = result15["transformed"][:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for ax, arr, title in [
        (axes[0], pc1, f"PC1 ({result15['explained_variance_ratio'][0]*100:.1f}%)"),
        (axes[1], pc2, f"PC2 ({result15['explained_variance_ratio'][1]*100:.1f}%)"),
    ]:
        finite = arr[np.isfinite(arr)]
        vmin, vmax = np.quantile(finite, [0.02, 0.98]) if finite.size else (-1, 1)
        im = ax.imshow(
            apply_display_transform(arr),
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            origin="upper",
            aspect="equal",
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle(f"{cube_id} - 15-param PCA spatial maps (display-flipped)")
    stem = out_dir("figures") / f"{cube_id}_pc1_pc2_maps"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return stem


def process_cube(cube_id: str) -> dict:
    t0 = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"\n=== {cube_id} ===")

    stack_path = out_dir("feature_stacks") / f"{cube_id}_stackB_15param.npz"
    if not stack_path.exists():
        raise SystemExit(f"Missing {stack_path}; run stage_b_build_stack.py first.")
    stack = dict(np.load(stack_path, allow_pickle=True))
    joint = stack["joint_valid_15"].astype(bool)
    shape = joint.shape

    print(f"  joint_valid_15: {joint.sum()} pixels")

    result15 = fit_pca(stack, joint, STAGE_B_PARAM_ORDER)
    result8 = fit_pca(stack, joint, DEFAULT_PARAM_ORDER)
    result4 = fit_pca(stack, joint, STAGE_B_REDUCED_4)

    npz15 = save_pca(cube_id, "15param", result15)
    npz8 = save_pca(cube_id, "8param", result8)
    npz4 = save_pca(cube_id, "4reduced", result4)
    print(f"  saved {npz15.name}, {npz8.name}, {npz4.name}")

    evr15 = result15["explained_variance_ratio"]
    print(
        f"  15-param PCA: PC1={evr15[0]*100:.1f}% PC2={evr15[1]*100:.1f}% "
        f"PC3={evr15[2]*100:.1f}% cum5={np.sum(evr15[:5])*100:.1f}%"
    )

    plot_variance(result15, result8, result4, cube_id)
    plot_loadings(result15, cube_id)
    plot_density(result15, cube_id)
    plot_spatial_pcs(result15, cube_id, shape)
    print("  saved figures")

    log: dict = {
        "cube_id": cube_id,
        "stage": "B-1_pca",
        "run_timestamp_utc": timestamp,
        "random_seed": RANDOM_SEED,
        "n_pixels_used": result15["n_pixels"],
        "explained_variance_ratio_15param": result15["explained_variance_ratio"].tolist(),
        "explained_variance_ratio_8param": result8["explained_variance_ratio"].tolist(),
        "explained_variance_ratio_4reduced": result4["explained_variance_ratio"].tolist(),
        "param_order_15": list(result15["param_names"]),
        "scaler_mean_15": result15["scaler_mean"].tolist(),
        "scaler_scale_15": result15["scaler_scale"].tolist(),
        "output_npz_15": str(npz15),
        "output_npz_8": str(npz8),
        "output_npz_4": str(npz4),
        "source_code_md5": {
            "stage_b_pca.py": md5_of(Path(__file__)),
            "config.py": md5_of(Path(__file__).with_name("config.py")),
        },
        "wall_clock_seconds": round(time.time() - t0, 2),
    }
    log_path = out_dir("logs") / f"{cube_id}_stageB_pca_{timestamp}.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  saved log {log_path}")
    return log


def main(argv: list[str]) -> None:
    targets = argv[1:] if len(argv) > 1 else list(CUBE_IDS)
    logs = []
    for cube_id in targets:
        logs.append(process_cube(cube_id))
    print("\n=== summary ===")
    print(f"{'cube_id':15s} {'PC1%':>6s} {'PC2%':>6s} {'PC3%':>6s} {'cum5%':>6s}")
    for log in logs:
        evr = log["explained_variance_ratio_15param"]
        print(f"{log['cube_id']:15s} {evr[0]*100:6.1f} {evr[1]*100:6.1f} {evr[2]*100:6.1f} {sum(evr[:5])*100:6.1f}")


if __name__ == "__main__":
    main(sys.argv)
