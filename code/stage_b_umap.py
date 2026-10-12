"""Stage B-2 / B-3 — UMAP hyperparameter sweep + selection + full-cube transform.

For each cube:
  1. Load the 15-param stack, StandardScaler the joint-valid pixels, draw a
     30k-point stratified random subsample (full set if fewer than 30k).
  2. Fit 12 UMAP configs (n_neighbors x min_dist) on the subsample.
  3. Save each subsample embedding; render a 4x3 mosaic of the density plots.
  4. Pick a config by two protocols:
       - heuristic (written rule: prefer non-collapsed and multi-modal
         embeddings, large n_neighbors, small min_dist);
       - validity_index (HDBSCAN validity_index at the cube's scaled
         min_cluster_size on each subsample embedding).
  5. For the chosen config(s), refit UMAP on the full joint-valid pixel set
     and save the full-cube embedding.

Outputs:
  outputs/embeddings/<cube>_umap_subsample_nN_mD.npz  (12 per cube)
  outputs/embeddings/<cube>_umap_selected_heuristic.npz
  outputs/embeddings/<cube>_umap_selected_validity.npz  (only if differs)
  outputs/figures/<cube>_umap_sweep.png
  outputs/figures/<cube>_umap_selected_density.png
  outputs/figures/<cube>_umap_selected_coloredby_pc1.png
  outputs/logs/<cube>_stageB_umap_<TIMESTAMP>.json
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib.colors import LogNorm
from sklearn.preprocessing import StandardScaler

from config import (
    CUBE_IDS,
    RANDOM_SEED,
    STAGE_B_PARAM_ORDER,
    out_dir,
)

SUBSAMPLE_SIZE = 30_000
UMAP_N_NEIGHBORS = (15, 30, 50, 100)
UMAP_MIN_DIST = (0.0, 0.1, 0.3)

# Per-cube HDBSCAN min_cluster_size (scaled to ~0.06% of joint_valid, with A4FC
# pinned at 15 per the PLAN).
HDBSCAN_MIN_CLUSTER_SIZE = {
    "FRT00003E12": 150,
    "FRT0000871C": 100,
    "FRT0000A4FC": 15,
    "FRT0000B438": 75,
}


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_stack(cube_id: str) -> dict[str, np.ndarray]:
    path = out_dir("feature_stacks") / f"{cube_id}_stackB_15param.npz"
    return dict(np.load(path, allow_pickle=True))


def build_feature_matrix(
    stack: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Return (X_scaled, line_idx, sample_idx, scaler) for joint_valid_15 pixels."""
    valid = stack["joint_valid_15"].astype(bool)
    yy, xx = np.where(valid)
    X = np.column_stack(
        [stack[name][yy, xx].astype(np.float64) for name in STAGE_B_PARAM_ORDER]
    )
    finite = np.isfinite(X).all(axis=1)
    X = X[finite]
    yy = yy[finite]
    xx = xx[finite]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, yy.astype(np.int32), xx.astype(np.int32), scaler


def sweep_subsample(
    cube_id: str,
    Xs_sub: np.ndarray,
) -> dict[str, dict]:
    """Fit each UMAP config on the subsample; return {config_key: {model, embedding}}."""
    results: dict[str, dict] = {}
    for nn in UMAP_N_NEIGHBORS:
        for md in UMAP_MIN_DIST:
            key = f"n{nn}_m{md:.1f}"
            t0 = time.time()
            reducer = umap.UMAP(
                n_neighbors=nn,
                min_dist=md,
                n_components=2,
                metric="euclidean",
                random_state=RANDOM_SEED,
                verbose=False,
            )
            emb = reducer.fit_transform(Xs_sub)
            dt = time.time() - t0
            print(f"    {cube_id} UMAP {key}  n={len(Xs_sub):6d}  {dt:5.1f}s")
            results[key] = {
                "model": reducer,
                "embedding": emb.astype(np.float32),
                "n_neighbors": nn,
                "min_dist": md,
                "fit_seconds": round(dt, 2),
            }
            npz_path = (
                out_dir("embeddings")
                / f"{cube_id}_umap_subsample_{key}.npz"
            )
            np.savez_compressed(
                npz_path,
                embedding=emb.astype(np.float32),
                n_neighbors=nn,
                min_dist=md,
                seed=RANDOM_SEED,
            )
    return results


def plot_sweep_mosaic(cube_id: str, sweep: dict[str, dict]) -> Path:
    fig, axes = plt.subplots(
        len(UMAP_N_NEIGHBORS),
        len(UMAP_MIN_DIST),
        figsize=(4 * len(UMAP_MIN_DIST), 3.6 * len(UMAP_N_NEIGHBORS)),
        constrained_layout=True,
    )
    for i, nn in enumerate(UMAP_N_NEIGHBORS):
        for j, md in enumerate(UMAP_MIN_DIST):
            key = f"n{nn}_m{md:.1f}"
            emb = sweep[key]["embedding"]
            ax = axes[i, j]
            ax.hexbin(emb[:, 0], emb[:, 1], gridsize=60, cmap="viridis",
                      mincnt=1, norm=LogNorm())
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"n_neighbors={nn}, min_dist={md}", fontsize=9)
    fig.suptitle(
        f"{cube_id} - UMAP sweep (30k subsample, euclidean, seed 42)"
    )
    stem = out_dir("figures") / f"{cube_id}_umap_sweep"
    fig.savefig(stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return stem


def heuristic_pick(sweep: dict[str, dict]) -> str:
    """Written heuristic: prefer larger n_neighbors and small min_dist, rejecting
    configs where the embedding has collapsed to a single blob."""
    best = None
    best_score = -np.inf
    for key, payload in sweep.items():
        nn = payload["n_neighbors"]
        md = payload["min_dist"]
        emb = payload["embedding"]
        # Reject if sharply collapsed (std on either axis < 0.5).
        if np.std(emb[:, 0]) < 0.5 or np.std(emb[:, 1]) < 0.5:
            continue
        score = nn - 10 * md  # bigger n, smaller min_dist
        if score > best_score:
            best_score = score
            best = key
    return best or "n50_m0.0"


def validity_pick(
    sweep: dict[str, dict], min_cluster_size: int
) -> tuple[str, dict[str, float]]:
    """Quantitative pick: for each config, cluster with HDBSCAN at the cube's
    scaled min_cluster_size on the subsample embedding, compute DBCV via
    hdbscan.validity.validity_index on the embedding coordinates, pick max."""
    scores: dict[str, float] = {}
    for key, payload in sweep.items():
        emb = payload["embedding"].astype(np.float64)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom",
            allow_single_cluster=False,
        )
        labels = clusterer.fit_predict(emb)
        n_clusters = int(labels.max()) + 1 if labels.size else 0
        if n_clusters < 2:
            scores[key] = -np.inf
            continue
        try:
            v = hdbscan.validity.validity_index(emb, labels, metric="euclidean")
        except Exception:
            v = -np.inf
        scores[key] = float(v)
    best = max(scores, key=scores.get)
    return best, scores


def fit_full_embedding(
    cube_id: str,
    Xs_full: np.ndarray,
    yy: np.ndarray,
    xx: np.ndarray,
    key: str,
    protocol: str,
) -> Path:
    nn, md = key_to_params(key)
    t0 = time.time()
    reducer = umap.UMAP(
        n_neighbors=nn,
        min_dist=md,
        n_components=2,
        metric="euclidean",
        random_state=RANDOM_SEED,
        verbose=False,
    )
    emb = reducer.fit_transform(Xs_full)
    dt = time.time() - t0
    print(f"    {cube_id} UMAP full-fit [{protocol}] {key}  n={len(Xs_full)}  {dt:.1f}s")
    path = out_dir("embeddings") / f"{cube_id}_umap_selected_{protocol}.npz"
    np.savez_compressed(
        path,
        embedding=emb.astype(np.float32),
        line=yy,
        sample=xx,
        n_neighbors=nn,
        min_dist=md,
        seed=RANDOM_SEED,
        protocol=protocol,
        fit_seconds=round(dt, 2),
    )
    return path


def key_to_params(key: str) -> tuple[int, float]:
    n_part, m_part = key.split("_")
    return int(n_part[1:]), float(m_part[1:])


def plot_selected_density(cube_id: str, protocol: str, emb: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.hexbin(emb[:, 0], emb[:, 1], gridsize=80, cmap="viridis",
              mincnt=1, norm=LogNorm())
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(f"{cube_id} - UMAP selected ({protocol})")
    stem = out_dir("figures") / f"{cube_id}_umap_selected_density_{protocol}"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return stem


def plot_selected_by_pc1(
    cube_id: str, protocol: str, emb: np.ndarray, pc1: np.ndarray
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    # Subsample if too many points for scatter.
    n = len(emb)
    if n > 60_000:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(n, 60_000, replace=False)
        emb = emb[idx]
        pc1 = pc1[idx]
    vmin, vmax = np.quantile(pc1, [0.02, 0.98])
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=pc1, s=1, cmap="viridis",
                    vmin=vmin, vmax=vmax, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="PC1 (15-param PCA)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(f"{cube_id} - UMAP selected, coloured by PC1 ({protocol})")
    stem = out_dir("figures") / f"{cube_id}_umap_selected_coloredby_pc1_{protocol}"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return stem


def process_cube(cube_id: str) -> dict:
    t0 = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"\n=== {cube_id} ===")

    stack = load_stack(cube_id)
    Xs_full, yy, xx, _ = build_feature_matrix(stack)
    n_full = len(Xs_full)
    print(f"  full joint-valid (15-param finite): {n_full} pixels")

    rng = np.random.default_rng(RANDOM_SEED)
    if n_full > SUBSAMPLE_SIZE:
        sub_idx = rng.choice(n_full, SUBSAMPLE_SIZE, replace=False)
    else:
        sub_idx = np.arange(n_full)
    Xs_sub = Xs_full[sub_idx]
    print(f"  subsample: {len(Xs_sub)} pixels")

    sweep = sweep_subsample(cube_id, Xs_sub)
    plot_sweep_mosaic(cube_id, sweep)

    pick_heur = heuristic_pick(sweep)
    min_cluster_size = HDBSCAN_MIN_CLUSTER_SIZE[cube_id]
    pick_val, val_scores = validity_pick(sweep, min_cluster_size)
    agree = pick_heur == pick_val
    print(f"  heuristic pick: {pick_heur}")
    print(f"  validity_index pick: {pick_val}  (scores: {val_scores})")
    print(f"  agree: {agree}")

    # Full-cube fit for the chosen config(s).
    pca15_path = out_dir("embeddings") / f"{cube_id}_pca_15param.npz"
    pca15 = np.load(pca15_path, allow_pickle=True)
    pc1_full = pca15["transformed"][:, 0].astype(np.float32)
    # The PCA was fit on the SAME joint-valid set in the same order as here,
    # so pc1_full aligns with Xs_full row order. Guard by checking lengths.
    if len(pc1_full) != n_full:
        raise RuntimeError(
            f"PCA vs UMAP row mismatch: pca={len(pc1_full)} umap={n_full}"
        )

    full_paths: dict[str, Path] = {}
    emb_cache: dict[str, np.ndarray] = {}
    full_paths["heuristic"] = fit_full_embedding(
        cube_id, Xs_full, yy, xx, pick_heur, "heuristic"
    )
    emb_cache["heuristic"] = np.load(full_paths["heuristic"])["embedding"]
    plot_selected_density(cube_id, "heuristic", emb_cache["heuristic"])
    plot_selected_by_pc1(cube_id, "heuristic", emb_cache["heuristic"], pc1_full)

    if not agree:
        full_paths["validity"] = fit_full_embedding(
            cube_id, Xs_full, yy, xx, pick_val, "validity"
        )
        emb_cache["validity"] = np.load(full_paths["validity"])["embedding"]
        plot_selected_density(cube_id, "validity", emb_cache["validity"])
        plot_selected_by_pc1(cube_id, "validity", emb_cache["validity"], pc1_full)

    log = {
        "cube_id": cube_id,
        "stage": "B-2_3_umap",
        "run_timestamp_utc": timestamp,
        "random_seed": RANDOM_SEED,
        "n_pixels_full": n_full,
        "n_pixels_subsample": int(len(Xs_sub)),
        "heuristic_pick": pick_heur,
        "validity_pick": pick_val,
        "validity_scores": val_scores,
        "picks_agree": bool(agree),
        "hdbscan_min_cluster_size_used_for_validity": min_cluster_size,
        "full_paths": {k: str(v) for k, v in full_paths.items()},
        "source_code_md5": {
            "stage_b_umap.py": md5_of(Path(__file__)),
            "config.py": md5_of(Path(__file__).with_name("config.py")),
        },
        "wall_clock_seconds": round(time.time() - t0, 2),
    }
    log_path = out_dir("logs") / f"{cube_id}_stageB_umap_{timestamp}.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  saved log {log_path}  ({log['wall_clock_seconds']:.0f}s)")
    return log


def main(argv: list[str]) -> None:
    targets = argv[1:] if len(argv) > 1 else list(CUBE_IDS)
    logs = []
    for cube_id in targets:
        logs.append(process_cube(cube_id))
    print("\n=== summary ===")
    for log in logs:
        flag = "agree" if log["picks_agree"] else "DISAGREE"
        print(
            f"  {log['cube_id']:15s} heuristic={log['heuristic_pick']:12s}  "
            f"validity={log['validity_pick']:12s}  {flag}  "
            f"({log['wall_clock_seconds']:.0f}s)"
        )


if __name__ == "__main__":
    main(sys.argv)
