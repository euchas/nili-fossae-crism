"""Stage A — sanity plots: parameter histograms, spatial maps, correlations,
four-cube overlays, and per-cube residual distributions.

All plots render from the saved feature-stack .npz files so this script can
rerun without re-loading the cubes.

Display conventions
-------------------
Spatial images are flipped both vertically and horizontally via
`display.apply_display_transform` to match the original
PCA_CRISM_analysis_v2.py convention (detector orientation is upside-down
relative to standard Mars-map view). On-disk arrays are never flipped.

Masking conventions
-------------------
- Histograms, correlation matrices, and the feature-stack columns used in
  clustering use ``joint_valid`` — the AND of brown_mask, finite Brown
  parameters, and finite VB parameters. Every pixel used for PCA / UMAP
  / HDBSCAN must have every feature defined.
- Parameter spatial maps use per-parameter masking:
  * Brown 4 (centroid, asymmetry, amplitude, hwhm) honour
    ``brown_mask & finite(param)`` — these parameters are *outputs* of
    the Brown v5 fit and only exist where the fit succeeded.
  * VB 4 (OLINDEX3, D2300, BD1900R2, BD2500_2) honour only
    ``finite(param)`` — none of them depends on the 1 µm fit succeeding,
    so Brown-masking them in display throws away genuine signal about
    hydration, Fe/Mg-OH, or carbonate.
- Residual maps honour ``brown_mask`` (the v5 fit residual only exists
  where the fit was attempted).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CUBE_IDS, DEFAULT_PARAM_ORDER, out_dir
from display import apply_display_transform

CUBE_COLOURS = {
    "FRT00003E12": "#1f77b4",
    "FRT0000871C": "#d62728",
    "FRT0000A4FC": "#2ca02c",
    "FRT0000B438": "#9467bd",
}

BROWN_PARAMS = ("centroid", "asymmetry", "amplitude", "hwhm")
VB_PARAMS = ("OLINDEX3", "D2300", "BD1900R2", "BD2500_2")


def load_stack(cube_id: str) -> dict[str, np.ndarray]:
    npz_path = out_dir("feature_stacks") / f"{cube_id}_stackA_8param.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing feature stack: {npz_path}")
    return dict(np.load(npz_path, allow_pickle=True))


def save_both(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _display_mask_for(name: str, stack: dict[str, np.ndarray]) -> np.ndarray:
    """Return the boolean mask used for spatial display of parameter ``name``."""
    arr = stack[name].astype(float)
    finite = np.isfinite(arr)
    if name in BROWN_PARAMS:
        return stack["brown_mask"].astype(bool) & finite
    if name in VB_PARAMS:
        return finite
    raise KeyError(f"Unknown parameter name {name!r}")


def per_cube_histograms(cube_id: str) -> None:
    stack = load_stack(cube_id)
    joint = stack["joint_valid"].astype(bool)
    params = list(DEFAULT_PARAM_ORDER)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for ax, name in zip(axes.ravel(), params):
        arr = stack[name][joint]
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            ax.set_title(f"{name} (no valid pixels)")
            continue
        q05, q50, q95 = np.quantile(arr, [0.05, 0.5, 0.95])
        ax.hist(arr, bins=120, color="#3366aa", edgecolor="none")
        for q, ls in [(q05, ":"), (q50, "-"), (q95, ":")]:
            ax.axvline(q, color="k", ls=ls, lw=0.8, alpha=0.7)
        ax.set_title(f"{name}  n={arr.size:,}  med={q50:.3g}")
        ax.set_xlabel(name)
        ax.set_ylabel("pixels")
    fig.suptitle(
        f"{cube_id} — Stage A parameter histograms (joint_valid for clustering)"
    )
    stem = out_dir("figures") / f"{cube_id}_stackA_histograms"
    save_both(fig, stem)
    print(f"  wrote {stem}.png")


def per_cube_param_maps(cube_id: str) -> None:
    stack = load_stack(cube_id)
    params = list(DEFAULT_PARAM_ORDER)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    for ax, name in zip(axes.ravel(), params):
        arr = stack[name].astype(float).copy()
        mask = _display_mask_for(name, stack)
        arr[~mask] = np.nan
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            ax.set_title(f"{name} (empty)")
            continue
        vmin, vmax = np.quantile(finite, [0.02, 0.98])
        disp = apply_display_transform(arr)
        im = ax.imshow(
            disp,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            origin="upper",
            interpolation="nearest",
            aspect="equal",
        )
        plt.colorbar(im, ax=ax, shrink=0.75, label=name)
        mask_note = "brown+finite" if name in BROWN_PARAMS else "finite only"
        ax.set_title(f"{name}  ({mask_note})")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"{cube_id} — spatial parameter maps (2–98% stretch, flipped to map view)"
    )
    stem = out_dir("figures") / f"{cube_id}_stackA_param_maps"
    save_both(fig, stem)
    print(f"  wrote {stem}.png")


def per_cube_correlation(cube_id: str) -> None:
    stack = load_stack(cube_id)
    valid = stack["joint_valid"].astype(bool)
    cols = list(DEFAULT_PARAM_ORDER)
    data = {name: stack[name][valid] for name in cols}
    df = pd.DataFrame(data)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) > 50_000:
        df = df.sample(50_000, random_state=42)
    corr = df.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(
                j,
                i,
                f"{corr.values[i, j]:+.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(corr.values[i, j]) > 0.6 else "black",
            )
    plt.colorbar(im, ax=ax, shrink=0.8, label="Spearman rho")
    ax.set_title(f"{cube_id} — parameter correlation matrix (Spearman)")
    stem = out_dir("figures") / f"{cube_id}_stackA_correlation"
    save_both(fig, stem)
    print(f"  wrote {stem}.png")


def per_cube_residual(cube_id: str) -> None:
    stack = load_stack(cube_id)
    residual = stack["residual"].astype(float)
    brown_mask = stack["brown_mask"].astype(bool)
    joint = stack["joint_valid"].astype(bool)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    r_all = residual[brown_mask]
    r_all = r_all[np.isfinite(r_all)]
    r_joint = residual[joint]
    r_joint = r_joint[np.isfinite(r_joint)]
    if r_all.size:
        axes[0].hist(r_all, bins=120, color="gray", alpha=0.55, label="brown_mask")
    if r_joint.size:
        axes[0].hist(r_joint, bins=120, color="#3366aa", alpha=0.8, label="joint_valid")
    axes[0].set_xlabel("Brown v5 fit residual")
    axes[0].set_ylabel("pixels")
    axes[0].set_title(f"{cube_id} — residual histogram")
    axes[0].legend(fontsize=8)

    res_img = residual.copy()
    res_img[~brown_mask] = np.nan
    finite = res_img[np.isfinite(res_img)]
    if finite.size:
        vmin, vmax = np.quantile(finite, [0.02, 0.98])
    else:
        vmin, vmax = 0, 1
    disp = apply_display_transform(res_img)
    im = axes[1].imshow(
        disp,
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        aspect="equal",
    )
    axes[1].set_title("spatial residual map (flipped to map view)")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    plt.colorbar(im, ax=axes[1], shrink=0.75, label="residual")
    stem = out_dir("figures") / f"{cube_id}_stackA_residual"
    save_both(fig, stem)
    print(f"  wrote {stem}.png")


def four_cube_overlay() -> None:
    params = list(DEFAULT_PARAM_ORDER)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    stats_rows: list[dict] = []
    for cube_id in CUBE_IDS:
        stack = load_stack(cube_id)
        valid = stack["joint_valid"].astype(bool)
        colour = CUBE_COLOURS[cube_id]
        for ax, name in zip(axes.ravel(), params):
            arr = stack[name][valid]
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            lo, hi = np.quantile(arr, [0.001, 0.999])
            ax.hist(
                arr,
                bins=120,
                range=(lo, hi),
                histtype="step",
                density=True,
                color=colour,
                linewidth=1.3,
                label=cube_id,
            )
            ax.set_title(name)
            ax.set_xlabel(name)
        cent_valid = stack["centroid"][valid]
        cent_valid = cent_valid[np.isfinite(cent_valid)]
        q05, q50, q95 = np.quantile(cent_valid, [0.05, 0.5, 0.95])
        stats_rows.append(
            {
                "cube_id": cube_id,
                "centroid_p05": q05,
                "centroid_p50": q50,
                "centroid_p95": q95,
            }
        )
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Four-cube parameter distributions (density, 0.1-99.9% range)")
    stem = out_dir("figures") / "all_cubes_stackA_distributions"
    save_both(fig, stem)
    print(f"  wrote {stem}.png")

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for cube_id in CUBE_IDS:
        stack = load_stack(cube_id)
        valid = stack["joint_valid"].astype(bool)
        arr = stack["centroid"][valid]
        arr = arr[np.isfinite(arr)]
        lo, hi = np.quantile(arr, [0.001, 0.999])
        ax.hist(
            arr,
            bins=120,
            range=(lo, hi),
            histtype="step",
            density=True,
            color=CUBE_COLOURS[cube_id],
            linewidth=1.4,
            label=cube_id,
        )
    ax.set_xlabel("Brown v5 centroid (um)")
    ax.set_ylabel("pixel density")
    ax.set_title(
        "1 um olivine band centroid - four-cube overlay\n"
        "Kremer 2019 airfall predicts compositional homogeneity."
    )
    ax.legend(fontsize=9)
    stem = out_dir("figures") / "all_cubes_centroid_hist"
    save_both(fig, stem)
    print(f"  wrote {stem}.png")

    pd.DataFrame(stats_rows).to_csv(
        out_dir("reports") / "01_stageA_centroid_summary.csv", index=False
    )


def main(argv: list[str]) -> None:
    targets = argv[1:] if len(argv) > 1 else list(CUBE_IDS)
    for cube_id in targets:
        print(f"\n--- {cube_id} ---")
        per_cube_histograms(cube_id)
        per_cube_param_maps(cube_id)
        per_cube_correlation(cube_id)
        per_cube_residual(cube_id)
    if sorted(targets) == sorted(list(CUBE_IDS)):
        print("\n--- four-cube overlay ---")
        four_cube_overlay()


if __name__ == "__main__":
    main(sys.argv)
