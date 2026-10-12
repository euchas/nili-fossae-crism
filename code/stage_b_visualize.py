"""Stage B-7 — cluster visualisation: label maps, mean/ratioed spectra,
parameter-space positions, GMM transition maps.

For each cube and each clustering method available:
  - load the per-pixel labels (in joint_valid_15 row order);
  - make a flipped spatial cluster map;
  - extract raw mean I/F spectra per cluster directly from the TER3 cube;
  - auto-select the mafic-cap denominator (Stage B-7 selector) and produce
    a ratioed spectra plot - display-only, never used upstream;
  - plot cluster centroids in four 2-D parameter projections;
  - for GMM runs: a posterior-of-second-best "transition" map.

Methods handled:
  - kmeans_pca15_k<k>_adaptive, _fixed
  - gmm_pca15_k<k>_adaptive,    _fixed
  - hdbscan_heuristic,          hdbscan_validity

Each cube's cube is loaded once and held in memory.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import CUBE_IDS, STAGE_B_PARAM_ORDER, out_dir
from crism_io import load_cube_by_id
from display import apply_display_transform
from stage_b_denominator import (
    MAFIC_CAP_PARAMS,
    load_overrides,
    resolve_denominator,
    select_mafic_cluster,
)

# Discrete cluster palette (Tableau-style, 16 colours).
CLUSTER_COLOURS = np.array([
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
    "#98df8a", "#ff9896", "#c5b0d5", "#c49c94",
])
NOISE_COLOUR = "#dddddd"


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_label_files(cube_id: str) -> list[tuple[str, Path]]:
    """Return a list of (method_id, path) pairs for all cluster labels found."""
    cdir = out_dir("cluster_labels")
    patterns = [
        (r"^" + cube_id + r"_kmeans_pca15_k(\d+)_(adaptive|fixed)\.npy$", "kmeans"),
        (r"^" + cube_id + r"_gmm_pca15_k(\d+)_(adaptive|fixed)\.npy$", "gmm"),
        (r"^" + cube_id + r"_hdbscan_(heuristic|validity)\.npy$", "hdbscan"),
    ]
    found: list[tuple[str, Path]] = []
    for f in sorted(cdir.iterdir()):
        if not f.name.endswith(".npy"):
            continue
        if "_posterior" in f.name:
            continue
        for pat, prefix in patterns:
            m = re.match(pat, f.name)
            if not m:
                continue
            if prefix == "kmeans":
                k, regime = m.groups()
                method_id = f"kmeans_k{k}_{regime}"
            elif prefix == "gmm":
                k, regime = m.groups()
                method_id = f"gmm_k{k}_{regime}"
            else:
                protocol = m.group(1)
                method_id = f"hdbscan_{protocol}"
            found.append((method_id, f))
            break
    return found


def spatial_label_map(
    labels: np.ndarray,
    yy: np.ndarray,
    xx: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Build an (H, W, 3) RGB image encoding cluster labels, -1 as noise."""
    img = np.full(shape + (3,), 1.0, dtype=np.float32)  # white background
    for lbl in np.unique(labels):
        if lbl == -1:
            rgb = np.array(
                [int(NOISE_COLOUR[1:3], 16) / 255,
                 int(NOISE_COLOUR[3:5], 16) / 255,
                 int(NOISE_COLOUR[5:7], 16) / 255],
                dtype=np.float32,
            )
        else:
            hexcol = CLUSTER_COLOURS[int(lbl) % len(CLUSTER_COLOURS)]
            rgb = np.array(
                [int(hexcol[1:3], 16) / 255,
                 int(hexcol[3:5], 16) / 255,
                 int(hexcol[5:7], 16) / 255],
                dtype=np.float32,
            )
        sel = labels == lbl
        img[yy[sel], xx[sel]] = rgb
    return img


def plot_label_map(
    cube_id: str,
    method_id: str,
    labels: np.ndarray,
    yy: np.ndarray,
    xx: np.ndarray,
    shape: tuple[int, int],
    cluster_sizes: dict[int, int],
) -> Path:
    img = spatial_label_map(labels, yy, xx, shape)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.imshow(apply_display_transform(img), origin="upper", aspect="equal",
              interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    # Build a compact legend.
    handles = []
    for lbl, n in sorted(cluster_sizes.items()):
        if lbl == -1:
            colour = NOISE_COLOUR
            label = f"noise  ({n:,})"
        else:
            colour = CLUSTER_COLOURS[int(lbl) % len(CLUSTER_COLOURS)]
            label = f"c{lbl}  ({n:,})"
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=colour, ec="black", lw=0.5, label=label))
    ax.legend(
        handles=handles, fontsize=8, loc="upper right", framealpha=0.9,
        ncol=1 if len(handles) <= 8 else 2,
    )
    ax.set_title(f"{cube_id} - {method_id} (display-flipped)")
    stem = out_dir("figures") / f"{cube_id}_{method_id}_labels"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return stem


def plot_mean_spectra(
    cube_id: str,
    method_id: str,
    wav: np.ndarray,
    mean_specs: dict[int, np.ndarray],
    std_specs: dict[int, np.ndarray],
    cluster_sizes: dict[int, int],
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    wmin, wmax = 0.4, 2.6
    mask_plot = (wav >= wmin) & (wav <= wmax)
    for lbl, mean in sorted(mean_specs.items()):
        if lbl == -1:
            continue
        colour = CLUSTER_COLOURS[int(lbl) % len(CLUSTER_COLOURS)]
        ax.plot(wav[mask_plot], mean[mask_plot], color=colour, lw=1.4,
                label=f"c{lbl} (n={cluster_sizes[lbl]:,})")
        std = std_specs[lbl]
        ax.fill_between(
            wav[mask_plot],
            mean[mask_plot] - std[mask_plot],
            mean[mask_plot] + std[mask_plot],
            color=colour, alpha=0.12,
        )
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Mean I/F +/- 1 sigma")
    ax.set_xlim(wmin, wmax)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.set_title(f"{cube_id} - {method_id} mean I/F (raw, display-only)")
    stem = out_dir("figures") / f"{cube_id}_{method_id}_mean_spectra"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return stem


def plot_ratioed_spectra(
    cube_id: str,
    method_id: str,
    wav: np.ndarray,
    mean_specs: dict[int, np.ndarray],
    denom_cluster: int,
    source: str,
    cluster_sizes: dict[int, int],
) -> Path | None:
    if denom_cluster not in mean_specs:
        return None
    denom = mean_specs[denom_cluster]
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    wmin, wmax = 0.4, 2.6
    mask_plot = (wav >= wmin) & (wav <= wmax)
    for lbl, mean in sorted(mean_specs.items()):
        if lbl == denom_cluster or lbl == -1:
            continue
        colour = CLUSTER_COLOURS[int(lbl) % len(CLUSTER_COLOURS)]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = mean / denom
        ratio[~np.isfinite(ratio)] = np.nan
        # Mask CO2 + detector boundary regions for display legibility.
        ratio[(wav >= 0.99) & (wav <= 1.02)] = np.nan
        ratio[(wav >= 2.65) & (wav <= 2.75)] = np.nan
        ax.plot(wav[mask_plot], ratio[mask_plot], color=colour, lw=1.4,
                label=f"c{lbl} / c{denom_cluster}  (n={cluster_sizes[lbl]:,})")
    for w, lab in [(1.05, "1 um"), (1.9, "1.9 um H2O"),
                   (2.3, "2.3 um Fe/Mg-OH"), (2.5, "2.5 um CO3")]:
        if wmin <= w <= wmax:
            ax.axvline(w, color="gray", ls=":", lw=0.5, alpha=0.4)
    ax.axhline(1.0, color="black", ls="--", lw=0.6, alpha=0.5)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel(f"Ratio to cluster {denom_cluster} (mafic-cap denom, {source})")
    ax.set_xlim(wmin, wmax)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.set_title(f"{cube_id} - {method_id} ratioed spectra (display-only)")
    stem = out_dir("figures") / f"{cube_id}_{method_id}_ratioed_spectra"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return stem


def plot_param_positions(
    cube_id: str,
    method_id: str,
    per_cluster_medians: dict[int, dict[str, float]],
    cluster_sizes: dict[int, int],
) -> Path:
    proj_pairs = [
        ("centroid", "OLINDEX3"),
        ("D2300", "BD2500_2"),
        ("BD1400", "BD2290"),
        ("LCPINDEX2", "BDI1000VIS"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, (xp, yp) in zip(axes.ravel(), proj_pairs):
        for lbl, meds in per_cluster_medians.items():
            colour = (
                NOISE_COLOUR if lbl == -1
                else CLUSTER_COLOURS[int(lbl) % len(CLUSTER_COLOURS)]
            )
            ax.scatter(
                meds[xp], meds[yp], s=max(30, min(400, cluster_sizes[lbl] / 200)),
                color=colour, edgecolor="black", linewidth=0.7,
                label=(f"c{lbl}" if lbl != -1 else "noise"),
            )
            ax.annotate(
                str(lbl) if lbl != -1 else "N",
                (meds[xp], meds[yp]),
                fontsize=8, ha="center", va="center",
            )
        ax.set_xlabel(xp); ax.set_ylabel(yp)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle(
        f"{cube_id} - {method_id} cluster medians in 2-D parameter projections"
    )
    stem = out_dir("figures") / f"{cube_id}_{method_id}_param_positions"
    fig.savefig(stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return stem


def plot_gmm_transition(
    cube_id: str,
    method_id: str,
    posterior: np.ndarray,
    yy: np.ndarray,
    xx: np.ndarray,
    shape: tuple[int, int],
) -> Path:
    # Compute entropy-like "transitionness" as 1 - max posterior.
    top = np.partition(posterior, -1, axis=1)[:, -1]
    transition = 1.0 - top
    img = np.full(shape, np.nan, dtype=np.float32)
    img[yy, xx] = transition
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    finite = img[np.isfinite(img)]
    vmin = 0.0
    vmax = float(np.quantile(finite, 0.98)) if finite.size else 1.0
    im = ax.imshow(
        apply_display_transform(img),
        cmap="inferno",
        vmin=vmin, vmax=vmax,
        origin="upper", aspect="equal", interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, shrink=0.75, label="1 - max posterior (transition)")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{cube_id} - {method_id} transition map (display-flipped)")
    stem = out_dir("figures") / f"{cube_id}_{method_id}_transition"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return stem


def process_cube(cube_id: str, cube_data: np.ndarray, wav: np.ndarray) -> dict:
    t0 = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"\n=== {cube_id} ===")
    stack = dict(np.load(
        out_dir("feature_stacks") / f"{cube_id}_stackB_15param.npz",
        allow_pickle=True,
    ))
    joint = stack["joint_valid_15"].astype(bool)
    shape = joint.shape
    yy, xx = np.where(joint)
    n_pix = int(yy.size)

    # Pre-extract per-pixel parameter values for per-cluster medians.
    param_values = {
        name: stack[name][yy, xx].astype(np.float32) for name in STAGE_B_PARAM_ORDER
    }

    # Pre-extract all raw pixel spectra (bands, n_pix).
    print(f"  extracting pixel spectra for {n_pix} pixels...")
    specs_all = cube_data[:, yy, xx]  # (bands, n_pix)
    print(f"  specs_all shape: {specs_all.shape}")

    methods = discover_label_files(cube_id)
    if not methods:
        print("  no label files found")
        return {"cube_id": cube_id, "methods": {}}

    overrides = load_overrides()
    summary: dict = {"cube_id": cube_id, "methods": {}}

    for method_id, label_path in methods:
        labels = np.load(label_path)
        if labels.size != n_pix:
            print(f"  SKIP {method_id}: label length {labels.size} != n_pix {n_pix}")
            continue
        uniq = np.unique(labels)
        cluster_sizes = {int(u): int((labels == u).sum()) for u in uniq}
        print(f"  [{method_id}] clusters={sorted(cluster_sizes.items())}")

        # Per-cluster median parameter values.
        per_cluster_medians: dict[int, dict[str, float]] = {}
        for lbl in uniq:
            sel = labels == lbl
            per_cluster_medians[int(lbl)] = {
                name: float(np.nanmedian(param_values[name][sel]))
                for name in STAGE_B_PARAM_ORDER
            }

        # Per-cluster raw mean spectra.
        mean_specs: dict[int, np.ndarray] = {}
        std_specs: dict[int, np.ndarray] = {}
        for lbl in uniq:
            sel = labels == lbl
            with np.errstate(invalid="ignore"):
                mean_specs[int(lbl)] = np.nanmean(specs_all[:, sel], axis=1)
                std_specs[int(lbl)] = np.nanstd(specs_all[:, sel], axis=1)

        plot_label_map(cube_id, method_id, labels, yy, xx, shape, cluster_sizes)
        plot_mean_spectra(cube_id, method_id, wav, mean_specs, std_specs, cluster_sizes)

        # Denominator selection (skip noise cluster for mafic-cap scoring).
        mafic_inputs = {
            lbl: {p: per_cluster_medians[lbl][p] for p in MAFIC_CAP_PARAMS}
            for lbl in per_cluster_medians
            if lbl != -1
        }
        lcp_inputs = {
            lbl: per_cluster_medians[lbl]["LCPINDEX2"]
            for lbl in per_cluster_medians
            if lbl != -1
        }
        auto_pick = select_mafic_cluster(mafic_inputs, lcp_inputs)
        denom_cluster, source = resolve_denominator(
            cube_id, method_id, auto_pick, overrides
        )

        plot_ratioed_spectra(
            cube_id, method_id, wav, mean_specs, denom_cluster, source, cluster_sizes
        )
        plot_param_positions(cube_id, method_id, per_cluster_medians, cluster_sizes)

        # GMM transition map (if posterior exists).
        post_path = label_path.with_name(label_path.stem + "_posterior.npy")
        if post_path.exists():
            posterior = np.load(post_path)
            plot_gmm_transition(cube_id, method_id, posterior, yy, xx, shape)

        summary["methods"][method_id] = {
            "n_clusters": int(len([u for u in uniq if u != -1])),
            "cluster_sizes": cluster_sizes,
            "auto_mafic_pick": auto_pick,
            "denominator_used": denom_cluster,
            "denominator_source": source,
            "per_cluster_medians": {
                int(k): {p: v for p, v in meds.items()}
                for k, meds in per_cluster_medians.items()
            },
        }

    log = {
        "cube_id": cube_id,
        "stage": "B-7_visualise",
        "run_timestamp_utc": timestamp,
        "n_pixels_joint_valid": n_pix,
        "methods_summary": {
            mid: {
                "n_clusters": s["n_clusters"],
                "auto_mafic_pick": s["auto_mafic_pick"],
                "denominator_used": s["denominator_used"],
                "denominator_source": s["denominator_source"],
            }
            for mid, s in summary["methods"].items()
        },
        "source_code_md5": {
            "stage_b_visualize.py": md5_of(Path(__file__)),
            "stage_b_denominator.py": md5_of(Path(__file__).with_name("stage_b_denominator.py")),
            "config.py": md5_of(Path(__file__).with_name("config.py")),
        },
        "wall_clock_seconds": round(time.time() - t0, 2),
    }
    log_path = out_dir("logs") / f"{cube_id}_stageB_visualize_{timestamp}.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  saved log {log_path}  ({log['wall_clock_seconds']:.0f}s)")
    return summary


def main(argv: list[str]) -> None:
    targets = argv[1:] if len(argv) > 1 else list(CUBE_IDS)
    all_summaries = {}
    for cube_id in targets:
        print(f"\n--- loading cube {cube_id} ---")
        cube, wav, _, _ = load_cube_by_id(cube_id)
        summary = process_cube(cube_id, cube, wav)
        all_summaries[cube_id] = summary
    summary_path = out_dir("reports") / "02_stageB_method_summary.json"
    summary_path.write_text(json.dumps(all_summaries, indent=2))
    print(f"\nwrote cross-cube method summary {summary_path}")


if __name__ == "__main__":
    main(sys.argv)
