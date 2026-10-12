"""Stage B-4 / B-6 — k-means and GMM clustering on 15-param PCA.

Both methods operate on the first 5 PCs of the 15-param PCA. Two regimes per
method:
  - adaptive: k in {2..10}, choose per-cube by silhouette (k-means) or BIC (GMM)
  - fixed k = 6 for cross-cube comparability (per PLAN)

Outputs:
  outputs/cluster_labels/<cube>_kmeans_pca15_k<k>_<regime>.npy
  outputs/cluster_labels/<cube>_gmm_pca15_k<k>_<regime>.npy
  outputs/cluster_labels/<cube>_gmm_pca15_k<k>_<regime>_posterior.npy
  outputs/figures/<cube>_kmeans_silhouette_sweep.png
  outputs/figures/<cube>_gmm_bic_sweep.png
  outputs/logs/<cube>_stageB_cluster_pca_<TIMESTAMP>.json
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
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.mixture import GaussianMixture

from config import CUBE_IDS, RANDOM_SEED, out_dir

N_PCS_USED = 5
K_SWEEP = range(2, 11)
FIXED_K = 6
SILHOUETTE_SUBSAMPLE = 50_000


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pca15(cube_id: str) -> dict:
    path = out_dir("embeddings") / f"{cube_id}_pca_15param.npz"
    data = dict(np.load(path, allow_pickle=True))
    return data


def kmeans_sweep(X: np.ndarray) -> tuple[dict, int]:
    rng = np.random.default_rng(RANDOM_SEED)
    if len(X) > SILHOUETTE_SUBSAMPLE:
        idx = rng.choice(len(X), SILHOUETTE_SUBSAMPLE, replace=False)
        Xs = X[idx]
    else:
        Xs = X
    scores = {"k": [], "inertia": [], "silhouette": [], "calinski_harabasz": []}
    for k in K_SWEEP:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_SEED)
        labels = km.fit_predict(Xs)
        sil = silhouette_score(Xs, labels, sample_size=min(10_000, len(Xs)),
                               random_state=RANDOM_SEED)
        ch = calinski_harabasz_score(Xs, labels)
        scores["k"].append(k)
        scores["inertia"].append(float(km.inertia_))
        scores["silhouette"].append(float(sil))
        scores["calinski_harabasz"].append(float(ch))
    best_k = int(scores["k"][int(np.argmax(scores["silhouette"]))])
    return scores, best_k


def kmeans_fit_full(X: np.ndarray, k: int) -> np.ndarray:
    km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_SEED)
    return km.fit_predict(X).astype(np.int16)


def gmm_sweep(X: np.ndarray) -> tuple[dict, int]:
    scores = {"k": [], "bic": [], "aic": []}
    for k in K_SWEEP:
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                n_init=5,
                random_state=RANDOM_SEED,
                reg_covar=1e-5,
            )
            gmm.fit(X)
            scores["k"].append(k)
            scores["bic"].append(float(gmm.bic(X)))
            scores["aic"].append(float(gmm.aic(X)))
        except Exception as exc:  # pragma: no cover
            print(f"    GMM k={k} failed: {exc}")
            scores["k"].append(k)
            scores["bic"].append(np.inf)
            scores["aic"].append(np.inf)
    best_k = int(scores["k"][int(np.argmin(scores["bic"]))])
    return scores, best_k


def gmm_fit_full(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        n_init=5,
        random_state=RANDOM_SEED,
        reg_covar=1e-5,
    )
    gmm.fit(X)
    labels = gmm.predict(X).astype(np.int16)
    posterior = gmm.predict_proba(X).astype(np.float32)
    return labels, posterior


def plot_kmeans_sweep(cube_id: str, scores: dict, best_k: int) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    axes[0].plot(scores["k"], scores["inertia"], marker="o")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("inertia (elbow)"); axes[0].grid(alpha=0.3)
    axes[1].plot(scores["k"], scores["silhouette"], marker="o", color="#2ca02c")
    axes[1].axvline(best_k, color="red", ls="--", lw=0.8, label=f"best silhouette k={best_k}")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("silhouette"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=9)
    axes[2].plot(scores["k"], scores["calinski_harabasz"], marker="o", color="#d62728")
    axes[2].set_xlabel("k"); axes[2].set_ylabel("Calinski-Harabasz"); axes[2].grid(alpha=0.3)
    fig.suptitle(f"{cube_id} - k-means sweep on 15-param PCA (first {N_PCS_USED} PCs)")
    stem = out_dir("figures") / f"{cube_id}_kmeans_silhouette_sweep"
    fig.savefig(stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return stem


def plot_gmm_sweep(cube_id: str, scores: dict, best_k: int) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(scores["k"], scores["bic"], marker="o", label="BIC")
    ax.plot(scores["k"], scores["aic"], marker="s", linestyle="--", color="gray", label="AIC")
    ax.axvline(best_k, color="red", ls="--", lw=0.8, label=f"min BIC k={best_k}")
    ax.set_xlabel("k"); ax.set_ylabel("information criterion")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.set_title(f"{cube_id} - GMM sweep on 15-param PCA (first {N_PCS_USED} PCs)")
    stem = out_dir("figures") / f"{cube_id}_gmm_bic_sweep"
    fig.savefig(stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return stem


def process_cube(cube_id: str) -> dict:
    t0 = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"\n=== {cube_id} ===")

    pca = load_pca15(cube_id)
    X = pca["transformed"][:, :N_PCS_USED].astype(np.float32)
    print(f"  X shape: {X.shape}")

    print("  k-means sweep...")
    km_scores, km_best_k = kmeans_sweep(X)
    print(f"  k-means best-by-silhouette k={km_best_k}")
    plot_kmeans_sweep(cube_id, km_scores, km_best_k)

    print("  GMM sweep...")
    gmm_scores, gmm_best_k = gmm_sweep(X)
    print(f"  GMM best-by-BIC k={gmm_best_k}")
    plot_gmm_sweep(cube_id, gmm_scores, gmm_best_k)

    # k-means adaptive + fixed
    km_adapt_labels = kmeans_fit_full(X, km_best_k)
    km_fixed_labels = kmeans_fit_full(X, FIXED_K)
    np.save(
        out_dir("cluster_labels")
        / f"{cube_id}_kmeans_pca15_k{km_best_k}_adaptive.npy",
        km_adapt_labels,
    )
    np.save(
        out_dir("cluster_labels")
        / f"{cube_id}_kmeans_pca15_k{FIXED_K}_fixed.npy",
        km_fixed_labels,
    )
    print(f"  k-means labels: adaptive(k={km_best_k}) + fixed(k={FIXED_K})")

    # GMM adaptive + fixed
    gmm_adapt_labels, gmm_adapt_post = gmm_fit_full(X, gmm_best_k)
    gmm_fixed_labels, gmm_fixed_post = gmm_fit_full(X, FIXED_K)
    np.save(
        out_dir("cluster_labels")
        / f"{cube_id}_gmm_pca15_k{gmm_best_k}_adaptive.npy",
        gmm_adapt_labels,
    )
    np.save(
        out_dir("cluster_labels")
        / f"{cube_id}_gmm_pca15_k{gmm_best_k}_adaptive_posterior.npy",
        gmm_adapt_post,
    )
    np.save(
        out_dir("cluster_labels")
        / f"{cube_id}_gmm_pca15_k{FIXED_K}_fixed.npy",
        gmm_fixed_labels,
    )
    np.save(
        out_dir("cluster_labels")
        / f"{cube_id}_gmm_pca15_k{FIXED_K}_fixed_posterior.npy",
        gmm_fixed_post,
    )
    print(f"  GMM labels: adaptive(k={gmm_best_k}) + fixed(k={FIXED_K})")

    log = {
        "cube_id": cube_id,
        "stage": "B-4_6_kmeans_gmm",
        "run_timestamp_utc": timestamp,
        "random_seed": RANDOM_SEED,
        "n_pcs_used": N_PCS_USED,
        "fixed_k": FIXED_K,
        "k_sweep": list(K_SWEEP),
        "kmeans_scores": km_scores,
        "kmeans_best_k": km_best_k,
        "gmm_scores": gmm_scores,
        "gmm_best_k": gmm_best_k,
        "source_code_md5": {
            "stage_b_cluster_pca.py": md5_of(Path(__file__)),
            "config.py": md5_of(Path(__file__).with_name("config.py")),
        },
        "wall_clock_seconds": round(time.time() - t0, 2),
    }
    log_path = out_dir("logs") / f"{cube_id}_stageB_cluster_pca_{timestamp}.json"
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
        print(
            f"  {log['cube_id']:15s}  km_best_k={log['kmeans_best_k']}  "
            f"gmm_best_k={log['gmm_best_k']}  ({log['wall_clock_seconds']:.0f}s)"
        )


if __name__ == "__main__":
    main(sys.argv)
