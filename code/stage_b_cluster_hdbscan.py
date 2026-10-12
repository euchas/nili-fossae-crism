"""Stage B-5 — HDBSCAN on selected UMAP embedding.

Uses the selected-config UMAP embedding (heuristic pick; also validity pick
if it disagreed) and clusters with HDBSCAN at the cube's scaled
min_cluster_size.

Outputs:
  outputs/cluster_labels/<cube>_hdbscan_<protocol>.npy
  outputs/logs/<cube>_stageB_hdbscan_<TIMESTAMP>.json
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import hdbscan
import numpy as np

from config import CUBE_IDS, RANDOM_SEED, out_dir

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


def cluster(cube_id: str, protocol: str, mcs: int) -> dict | None:
    emb_path = out_dir("embeddings") / f"{cube_id}_umap_selected_{protocol}.npz"
    if not emb_path.exists():
        return None
    emb_data = np.load(emb_path, allow_pickle=True)
    emb = emb_data["embedding"].astype(np.float64)
    t0 = time.time()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=mcs,
        metric="euclidean",
        cluster_selection_method="eom",
        allow_single_cluster=False,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(emb).astype(np.int16)
    dt = time.time() - t0
    n_clusters = int(labels.max()) + 1 if labels.size else 0
    n_noise = int((labels == -1).sum())
    print(
        f"  {cube_id} [{protocol}] mcs={mcs}  clusters={n_clusters}  "
        f"noise={n_noise} ({100*n_noise/len(labels):.1f}%)  {dt:.1f}s"
    )
    out = out_dir("cluster_labels") / f"{cube_id}_hdbscan_{protocol}.npy"
    np.save(out, labels)
    return {
        "protocol": protocol,
        "min_cluster_size": mcs,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "n_pixels": int(len(labels)),
        "noise_fraction": round(n_noise / max(1, len(labels)), 4),
        "labels_path": str(out),
        "seconds": round(dt, 2),
    }


def process_cube(cube_id: str) -> dict:
    t0 = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"\n=== {cube_id} ===")
    mcs = HDBSCAN_MIN_CLUSTER_SIZE[cube_id]
    results = {}
    for protocol in ("heuristic", "validity"):
        res = cluster(cube_id, protocol, mcs)
        if res is not None:
            results[protocol] = res
    log = {
        "cube_id": cube_id,
        "stage": "B-5_hdbscan",
        "run_timestamp_utc": timestamp,
        "random_seed": RANDOM_SEED,
        "min_cluster_size": mcs,
        "results": results,
        "source_code_md5": {
            "stage_b_cluster_hdbscan.py": md5_of(Path(__file__)),
            "config.py": md5_of(Path(__file__).with_name("config.py")),
        },
        "wall_clock_seconds": round(time.time() - t0, 2),
    }
    log_path = out_dir("logs") / f"{cube_id}_stageB_hdbscan_{timestamp}.json"
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
        for prot, res in log["results"].items():
            print(
                f"  {log['cube_id']:15s}  {prot:10s}  k={res['n_clusters']}  "
                f"noise={100*res['noise_fraction']:.1f}%"
            )


if __name__ == "__main__":
    main(sys.argv)
