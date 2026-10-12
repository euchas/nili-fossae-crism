"""Stage B GMM k=6 only - focused rerun for the full-range analysis.

Loads the rebuilt 15-param PCA scores, fits a single GMM with k=6 fixed
on the first 5 PCs, saves labels + posteriors. No k-sweep, no k-means.

Outputs:
  outputs/cluster_labels/<cube>_gmm_pca15_k6_fixed.npy
  outputs/cluster_labels/<cube>_gmm_pca15_k6_fixed_posterior.npy
  outputs/logs/<cube>_stageB_gmm_k6_fullrange_<TIMESTAMP>.json
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture

from config import CUBE_IDS, RANDOM_SEED, out_dir

N_PCS_USED = 5
FIXED_K = 6


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def process(cube_id: str) -> dict:
    t0 = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"\n=== {cube_id} ===")
    pca = dict(np.load(out_dir("embeddings") / f"{cube_id}_pca_15param.npz", allow_pickle=True))
    X = pca["transformed"][:, :N_PCS_USED].astype(np.float32)
    print(f"  X shape: {X.shape}")

    gmm = GaussianMixture(
        n_components=FIXED_K,
        covariance_type="full",
        n_init=5,
        random_state=RANDOM_SEED,
        reg_covar=1e-5,
    )
    gmm.fit(X)
    labels = gmm.predict(X).astype(np.int16)
    post = gmm.predict_proba(X).astype(np.float32)

    np.save(out_dir("cluster_labels") / f"{cube_id}_gmm_pca15_k{FIXED_K}_fixed.npy", labels)
    np.save(out_dir("cluster_labels") / f"{cube_id}_gmm_pca15_k{FIXED_K}_fixed_posterior.npy", post)

    sizes = [int((labels == c).sum()) for c in range(FIXED_K)]
    print(f"  cluster sizes: {sizes}  (sum={sum(sizes)})  bic={gmm.bic(X):.0f}")

    log = {
        "cube_id": cube_id,
        "stage": "B-6_gmm_k6_fullrange",
        "run_timestamp_utc": timestamp,
        "random_seed": RANDOM_SEED,
        "n_pcs_used": N_PCS_USED,
        "fixed_k": FIXED_K,
        "n_pixels": int(len(labels)),
        "cluster_sizes": sizes,
        "bic": float(gmm.bic(X)),
        "aic": float(gmm.aic(X)),
        "source_code_md5": {
            "stage_b_gmm_k6_only.py": md5_of(Path(__file__)),
            "config.py": md5_of(Path(__file__).with_name("config.py")),
        },
        "wall_clock_seconds": round(time.time() - t0, 2),
    }
    log_path = out_dir("logs") / f"{cube_id}_stageB_gmm_k6_fullrange_{timestamp}.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"  saved log {log_path.name}  ({log['wall_clock_seconds']:.0f}s)")
    return log


def main(argv: list[str]) -> None:
    targets = argv[1:] if len(argv) > 1 else list(CUBE_IDS)
    for cube_id in targets:
        process(cube_id)


if __name__ == "__main__":
    main(sys.argv)
