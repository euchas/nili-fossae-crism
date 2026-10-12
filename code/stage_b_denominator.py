"""Mafic-cap denominator auto-selector for Stage B ratioed-spectra plots.

AV's rule: the denominator cluster for ratio-spectra display is the one
that is lowest in hydration AND low in olivine index AND low in carbonate -
the mafic cap unit. We score each cluster by z-summing four parameters
(BD1900R2, D2300, BD2500_2, OLINDEX3) negatively and pick the cluster with
the most-negative z-sum. Ties are broken by preferring clusters with
higher LCPINDEX2 (some pyroxene signature).

This helper also supports a JSON override so AV can replace the auto-pick
for any (cube, method) pair without rerunning clustering.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import out_dir

MAFIC_CAP_PARAMS = ("BD1900R2", "D2300", "BD2500_2", "OLINDEX3")


def select_mafic_cluster(
    cluster_medians: dict[int, dict[str, float]],
    lcpindex2_medians: dict[int, float] | None = None,
) -> int:
    """Given per-cluster median values for each of MAFIC_CAP_PARAMS, return the
    cluster id with the lowest summed z-score across those four parameters."""
    cluster_ids = sorted(cluster_medians.keys())
    if not cluster_ids:
        raise ValueError("No clusters to select from")
    vals = np.array(
        [[cluster_medians[c][p] for p in MAFIC_CAP_PARAMS] for c in cluster_ids],
        dtype=np.float64,
    )  # shape (n_clusters, 4)
    mean = vals.mean(axis=0)
    std = np.where(vals.std(axis=0) > 1e-9, vals.std(axis=0), 1.0)
    z = (vals - mean) / std
    score = z.sum(axis=1)  # lower = more "mafic-cap-like"
    order = np.argsort(score)
    best = cluster_ids[int(order[0])]

    # tie-break on LCPINDEX2 if the top two are within 0.2 z-units
    if lcpindex2_medians is not None and len(order) > 1:
        runner_up = cluster_ids[int(order[1])]
        if abs(score[order[0]] - score[order[1]]) < 0.2:
            if lcpindex2_medians.get(runner_up, -np.inf) > lcpindex2_medians.get(best, -np.inf):
                best = runner_up
    return int(best)


def load_overrides() -> dict:
    path = out_dir("reports") / "denominator_overrides.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def resolve_denominator(
    cube_id: str,
    method: str,
    auto_pick: int,
    overrides: dict | None = None,
) -> tuple[int, str]:
    """Return (cluster_id, source) where source is 'override' or 'auto'."""
    ov = overrides if overrides is not None else load_overrides()
    key = f"{cube_id}/{method}"
    if key in ov:
        return int(ov[key]), "override"
    return int(auto_pick), "auto"
