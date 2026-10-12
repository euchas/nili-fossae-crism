"""Stage v9.1 - per-cube StandardScaler, 3E12-fit PCA + GMM.

Delta from v9:
  1. Other cubes fit a fresh StandardScaler on their own valid pixels
     (do NOT reuse 3E12's). PCA + GMM stay locked to 3E12.
  2. No mafic-cap z-sum reorder. Cluster IDs come out in sklearn's
     natural EM-converged order; cluster means in original VB units
     are dumped to a JSON for AV to label by inspection.

Argmax cluster-size pileup is reported per cube; flag if any cluster
holds > 60% of valid pixels (residual covariance mismatch).

See PLAN_v9_1.md for the full spec.
"""

from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import rasterio
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from config import CUBE_IDS, RANDOM_SEED, OUTPUTS_DIR
from stage_v7c_mtr3_pipeline import write_float_geotiff, K_FIXED, REFERENCE
from stage_v9_3e12_seeded import (
    RUN_CONFIGS as V9_RUN_CONFIGS,
    MAX_POSTERIOR_THRESHOLD, N_INIT, REG_COVAR,
    NOVEL_LABEL, INVALID_LABEL,
    GMM_PALETTE_RGB, NOVEL_RGB, SUBCLUSTER_PALETTE_RGB,
    build_param_stack, per_pixel_mahalanobis2, warm_start_drift,
    subcluster_novel,
    write_composite_geotiff, write_subcluster_geotiff,
    _make_transform, _make_crs,
)

# v9.1 keeps the v9 param lists exactly; only the snapshot dir + tags change.
RUN_CONFIGS = {
    "v9_1a_5param": {
        "tag": "v9_1a",
        "params": V9_RUN_CONFIGS["v9a_5param"]["params"],
        "d": V9_RUN_CONFIGS["v9a_5param"]["d"],
        "chi2_crit": V9_RUN_CONFIGS["v9a_5param"]["chi2_crit"],
    },
    "v9_1b_7param": {
        "tag": "v9_1b",
        "params": V9_RUN_CONFIGS["v9b_7param"]["params"],
        "d": V9_RUN_CONFIGS["v9b_7param"]["d"],
        "chi2_crit": V9_RUN_CONFIGS["v9b_7param"]["chi2_crit"],
    },
}

PILEUP_FLAG_FRACTION = 0.60  # WARN if any single cluster holds > this
PILEUP_NOTE_FRACTION = 0.40  # soft note if > this (per the project specification)

SNAPSHOT = OUTPUTS_DIR / "_v9_1_per_cube_scaler"


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def cluster_means_original_units(scaler: StandardScaler, pca: PCA,
                                  gmm: GaussianMixture) -> np.ndarray:
    """Cluster means inverted through PCA(whiten=True) and StandardScaler
    back to original VB / Brown units. Returns shape (k, n_features).
    """
    standardized = pca.inverse_transform(gmm.means_)
    original = scaler.inverse_transform(standardized)
    return original


def fit_reference(run_key: str) -> dict:
    """Fit scaler + PCA(whiten=True) + GMM(k=6) on FRT00003E12 only.

    No mafic-cap reordering. Cluster means in original units are dumped
    to log/output for AV's manual labeling.
    """
    cfg = RUN_CONFIGS[run_key]
    tag = cfg["tag"]
    params_list = cfg["params"]
    d = cfg["d"]

    print(f"\n=== fit reference (3E12) - {run_key} ({d} params) ===")
    stack = build_param_stack(REFERENCE, params_list)
    valid = stack["valid_mask"]
    X = np.column_stack([stack["params"][p][valid] for p in params_list]
                        ).astype(np.float64)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    print(f"  X shape = {X.shape}, standardized.")
    print(f"  3E12 scaler.mean_ = {[f'{m:.4f}' for m in scaler.mean_]}")
    print(f"  3E12 scaler.scale_ = {[f'{s:.4f}' for s in scaler.scale_]}")

    pca = PCA(n_components=d, whiten=True, random_state=RANDOM_SEED)
    Xp_whit = pca.fit_transform(Xs)
    print(f"  PCA(whiten=True) explained var: "
          f"{[f'{v*100:.1f}%' for v in pca.explained_variance_ratio_]}")
    cum = np.cumsum(pca.explained_variance_ratio_)
    print(f"  cumulative: {[f'{v*100:.1f}%' for v in cum]}")

    gmm = GaussianMixture(
        n_components=K_FIXED, covariance_type="full",
        n_init=N_INIT, random_state=RANDOM_SEED, reg_covar=REG_COVAR,
    )
    gmm.fit(Xp_whit)
    labels = gmm.predict(Xp_whit).astype(np.int16)
    sizes = [int((labels == c).sum()) for c in range(K_FIXED)]
    print(f"  GMM (no reorder) sizes={sizes}  BIC={gmm.bic(Xp_whit):.0f}")

    means_orig = cluster_means_original_units(scaler, pca, gmm)
    print(f"\n  Cluster means in original VB units (un-standardized):")
    header = "  cluster" + "".join(f"{p:>11}" for p in params_list)
    print(header)
    for c in range(K_FIXED):
        row = f"  {c:>7d}" + "".join(f"{means_orig[c, j]:>+11.4f}"
                                     for j in range(len(params_list)))
        size_str = f"  (n={sizes[c]:>7d})"
        print(row + size_str)
    print()

    ref_dir = SNAPSHOT / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, ref_dir / f"3e12_{tag}_scaler.joblib")
    joblib.dump(pca,    ref_dir / f"3e12_{tag}_pca.joblib")
    joblib.dump(gmm,    ref_dir / f"3e12_{tag}_gmm.joblib")
    print(f"  saved scaler / pca / gmm to {ref_dir}")

    return {
        "run_key": run_key, "scaler": scaler, "pca": pca, "gmm": gmm,
        "params_list": list(params_list), "d": d,
        "Xs": Xs, "Xp_whit": Xp_whit, "labels": labels,
        "stack": stack,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "sizes": sizes,
        "cluster_means_original_units": means_orig.tolist(),
    }


def predict_cube(cube_id: str, ref: dict, is_reference: bool) -> dict:
    """v9.1 prediction. Non-ref cubes fit a fresh StandardScaler before
    projecting through the saved 3E12 PCA + GMM."""
    cfg = RUN_CONFIGS[ref["run_key"]]
    params_list = cfg["params"]
    d = cfg["d"]
    chi2_crit = cfg["chi2_crit"]
    print(f"\n--- predict {cube_id} - {ref['run_key']} (d={d}) ---")

    if is_reference:
        stack = ref["stack"]
        Xs = ref["Xs"]
        Xp_whit = ref["Xp_whit"]
        valid = stack["valid_mask"]
        scaler_local = ref["scaler"]  # alias; not refit
        proba = ref["gmm"].predict_proba(Xp_whit)
        labels = proba.argmax(axis=1).astype(np.int16)
        max_proba = proba.max(axis=1)
        m2 = per_pixel_mahalanobis2(Xp_whit, labels, ref["gmm"])
    else:
        stack = build_param_stack(cube_id, params_list)
        valid = stack["valid_mask"]
        X = np.column_stack([stack["params"][p][valid] for p in params_list]
                            ).astype(np.float64)
        # *** v9.1 change 1: per-cube StandardScaler ***
        scaler_local = StandardScaler()
        Xs = scaler_local.fit_transform(X)
        print(f"  per-cube scaler.mean_ = "
              f"{[f'{m:.4f}' for m in scaler_local.mean_]}")
        print(f"  per-cube scaler.scale_ = "
              f"{[f'{s:.4f}' for s in scaler_local.scale_]}")

        Xp_whit = ref["pca"].transform(Xs)  # 3E12-fit PCA, no refit
        proba = ref["gmm"].predict_proba(Xp_whit)  # 3E12-fit GMM
        labels = proba.argmax(axis=1).astype(np.int16)
        max_proba = proba.max(axis=1)
        m2 = per_pixel_mahalanobis2(Xp_whit, labels, ref["gmm"])

    # Argmax pile-up check (BEFORE novelty filter)
    sizes = [int((labels == c).sum()) for c in range(K_FIXED)]
    n_total = len(labels)
    max_idx = int(np.argmax(sizes))
    max_frac = sizes[max_idx] / max(1, n_total)
    pileup_status = "ok"
    if max_frac > PILEUP_FLAG_FRACTION:
        pileup_status = "FLAG"
        print(f"  *** PILE-UP FLAG *** cluster {max_idx} holds "
              f"{100*max_frac:.1f}% of {cube_id} pixels (>"
              f"{100*PILEUP_FLAG_FRACTION:.0f}%); residual covariance "
              "mismatch likely")
    elif max_frac > PILEUP_NOTE_FRACTION:
        pileup_status = "note"
        print(f"  pile-up note: cluster {max_idx} holds "
              f"{100*max_frac:.1f}% of {cube_id} (>"
              f"{100*PILEUP_NOTE_FRACTION:.0f}%); modest concentration")
    print(f"  cluster sizes: {sizes}")

    novel = (m2 > chi2_crit) | (max_proba < MAX_POSTERIOR_THRESHOLD)
    n_novel = int(novel.sum())
    print(f"  novel: {n_novel}/{n_total} "
          f"({100*n_novel/max(1, n_total):.2f}%) "
          f"chi2_crit={chi2_crit:.3f}, p_thr={MAX_POSTERIOR_THRESHOLD}")

    drift = None
    if not is_reference:
        print(f"  warm-start drift sanity pass...")
        drift = warm_start_drift(Xp_whit, ref["gmm"], d)
        for entry in drift["per_class"]:
            flag = " *FLAG*" if (entry["drift_sigma"] is not None
                                 and entry["drift_sigma"] > 0.5) else ""
            print(f"    class {entry['class']}: drift = "
                  f"{entry['drift_sigma']:.3f} sigma{flag}")

    sub_labels_flat, sub_info = subcluster_novel(Xp_whit, novel, k=2)

    return {
        "cube_id": cube_id, "stack": stack,
        "Xs": Xs, "Xp_whit": Xp_whit,
        "labels": labels, "max_proba": max_proba.astype(np.float32),
        "m2": m2.astype(np.float32), "novel_mask": novel,
        "sub_labels_flat": sub_labels_flat, "sub_info": sub_info,
        "drift": drift,
        "n_novel": n_novel, "n_valid": n_total,
        "scaler_local": scaler_local,
        "cluster_sizes": sizes,
        "argmax_pileup": {
            "max_cluster": max_idx, "max_fraction": float(max_frac),
            "status": pileup_status,
            "flag_threshold": PILEUP_FLAG_FRACTION,
            "note_threshold": PILEUP_NOTE_FRACTION,
        },
    }


def cube_run_dir(run_key: str, cube_id: str) -> Path:
    p = SNAPSHOT / run_key / cube_id
    (p / "arrays").mkdir(parents=True, exist_ok=True)
    (p / "geotiffs").mkdir(parents=True, exist_ok=True)
    (p / "logs").mkdir(parents=True, exist_ok=True)
    return p


def emit_cube_outputs(run_key: str, cube: dict, ref: dict) -> dict:
    cfg = RUN_CONFIGS[run_key]
    tag = cfg["tag"]
    d = cfg["d"]
    chi2_crit = cfg["chi2_crit"]
    cube_id = cube["cube_id"]
    stack = cube["stack"]
    H, W = stack["H"], stack["W"]
    geo = stack["geo"]
    valid = stack["valid_mask"]

    cdir = cube_run_dir(run_key, cube_id)
    arr_dir = cdir / "arrays"
    gtiff_dir = cdir / "geotiffs"

    pstack = np.dstack([stack["params"][p] for p in cfg["params"]])
    np.save(arr_dir / f"{cube_id}_{tag}_param_stack.npy", pstack)
    np.save(arr_dir / f"{cube_id}_{tag}_valid_mask.npy", valid)

    # Per-cube scaler also saved alongside arrays for reproducibility.
    if cube_id != REFERENCE:
        joblib.dump(cube["scaler_local"],
                    arr_dir / f"{cube_id}_{tag}_per_cube_scaler.joblib")

    pc_scenes = np.full((d, H, W), np.nan, dtype=np.float32)
    for k in range(d):
        sc = np.full((H, W), np.nan, dtype=np.float32)
        sc[valid] = cube["Xp_whit"][:, k].astype(np.float32)
        pc_scenes[k] = sc
    np.save(arr_dir / f"{cube_id}_{tag}_pc_scores.npy", pc_scenes)

    label_scene = np.full((H, W), -1, dtype=np.int16)
    label_scene[valid] = cube["labels"]
    np.save(arr_dir / f"{cube_id}_{tag}_gmm_labels.npy", label_scene)

    max_p_scene = np.full((H, W), np.nan, dtype=np.float32)
    max_p_scene[valid] = cube["max_proba"]
    np.save(arr_dir / f"{cube_id}_{tag}_max_posterior.npy", max_p_scene)

    m2_scene = np.full((H, W), np.nan, dtype=np.float32)
    m2_scene[valid] = cube["m2"]
    np.save(arr_dir / f"{cube_id}_{tag}_mahalanobis2.npy", m2_scene)

    novel_scene = np.zeros((H, W), dtype=bool)
    novel_scene[valid] = cube["novel_mask"]
    np.save(arr_dir / f"{cube_id}_{tag}_novel_mask.npy", novel_scene)

    for k in range(d):
        write_float_geotiff(
            pc_scenes[k],
            gtiff_dir / f"{cube_id}_{tag}_PC{k+1}_whitened.tif",
            geo, description=f"v9.1 {tag} PC{k+1} whitened (3E12 PCA, "
                              "per-cube scaler)")

    composite = np.full((H, W), INVALID_LABEL, dtype=np.uint8)
    valid_assigned = valid & (~novel_scene)
    composite[valid_assigned] = label_scene[valid_assigned].astype(np.uint8)
    composite[valid & novel_scene] = NOVEL_LABEL
    write_composite_geotiff(
        composite,
        gtiff_dir / f"{cube_id}_{tag}_composite_map.tif",
        geo,
        description=f"v9.1 {tag} GMM k=6 composite "
                    "(per-cube scaler, 3E12 PCA+GMM; "
                    "0..5=class, 254=novel/gray, 255=invalid)")

    label_only = np.where(label_scene >= 0,
                          label_scene.astype(np.uint8),
                          INVALID_LABEL)
    write_composite_geotiff(
        label_only,
        gtiff_dir / f"{cube_id}_{tag}_gmm_labels.tif",
        geo,
        description=f"v9.1 {tag} GMM k=6 labels only (sklearn EM order)")

    write_float_geotiff(
        max_p_scene, gtiff_dir / f"{cube_id}_{tag}_max_posterior.tif",
        geo, description=f"v9.1 {tag} max posterior probability per pixel")
    write_float_geotiff(
        m2_scene, gtiff_dir / f"{cube_id}_{tag}_mahalanobis2.tif",
        geo, description=f"v9.1 {tag} Mahalanobis^2 to assigned cluster")

    novel_uint8 = np.where(novel_scene & valid, 1, 0).astype(np.uint8)
    novel_uint8[~valid] = INVALID_LABEL
    profile = {
        "driver": "GTiff", "dtype": "uint8",
        "width": W, "height": H, "count": 1,
        "crs": _make_crs(geo), "transform": _make_transform(geo),
        "nodata": INVALID_LABEL, "compress": "deflate",
    }
    with rasterio.open(str(gtiff_dir / f"{cube_id}_{tag}_novel_mask.tif"),
                       "w", **profile) as dst:
        dst.write(novel_uint8, 1)
        dst.write_colormap(1, {
            0: (60, 60, 60, 255),
            1: (*NOVEL_RGB, 255),
            INVALID_LABEL: (0, 0, 0, 0),
        })
        dst.set_band_description(
            1, f"v9.1 {tag} novel pixels (m^2>{chi2_crit:.3f} OR "
               f"p_max<{MAX_POSTERIOR_THRESHOLD})")

    sub_info = cube["sub_info"]
    if cube["sub_labels_flat"] is not None:
        sub_scene_flat = np.full((H, W), INVALID_LABEL, dtype=np.uint8)
        valid_idx = np.flatnonzero(valid.ravel())
        novel_full_idx = valid_idx[cube["novel_mask"]]
        flat_sub = sub_scene_flat.ravel()
        flat_sub[novel_full_idx] = cube["sub_labels_flat"].astype(np.uint8)
        sub_scene = flat_sub.reshape(H, W)
        np.save(arr_dir / f"{cube_id}_{tag}_novel_subclusters.npy", sub_scene)
        write_subcluster_geotiff(
            sub_scene,
            gtiff_dir / f"{cube_id}_{tag}_novel_subclusters.tif",
            geo,
            description=f"v9.1 {tag} novel-pixel sub-clusters "
                        f"(k={sub_info['k_subclusters']})",
            k_sub=sub_info["k_subclusters"],
        )

    log = {
        "stage": "v9_1_per_cube_scaler",
        "run_key": run_key, "tag": tag, "cube_id": cube_id,
        "params": list(cfg["params"]), "d": d,
        "is_reference": cube_id == REFERENCE,
        "shape": [H, W],
        "n_valid": int(valid.sum()),
        "n_total": int(H * W),
        "n_novel": cube["n_novel"],
        "novel_fraction": cube["n_novel"] / max(1, cube["n_valid"]),
        "thresholds": {
            "chi2_crit_0.95": chi2_crit,
            "max_posterior_threshold": MAX_POSTERIOR_THRESHOLD,
            "pileup_flag_fraction": PILEUP_FLAG_FRACTION,
            "pileup_note_fraction": PILEUP_NOTE_FRACTION,
        },
        "cluster_sizes": cube["cluster_sizes"],
        "argmax_pileup": cube["argmax_pileup"],
        "subcluster_info": cube["sub_info"],
        "drift_check": cube["drift"],
        "per_cube_scaler_mean": (cube["scaler_local"].mean_.tolist()
                                  if cube_id != REFERENCE else None),
        "per_cube_scaler_scale": (cube["scaler_local"].scale_.tolist()
                                   if cube_id != REFERENCE else None),
        "timestamp_utc": stack["timestamp"],
        "geo_proj_name": geo["proj_name"] if geo else None,
        "ref_random_seed": RANDOM_SEED,
    }
    log_path = cdir / "logs" / f"{tag}_{cube_id}_{stack['timestamp']}.json"
    log_path.write_text(json.dumps(log, indent=2, default=str),
                        encoding="utf-8")
    return {
        "cube_id": cube_id,
        "n_valid": int(valid.sum()),
        "n_novel": cube["n_novel"],
        "novel_fraction": cube["n_novel"] / max(1, cube["n_valid"]),
        "cluster_sizes": cube["cluster_sizes"],
        "argmax_pileup": cube["argmax_pileup"],
        "drift": cube["drift"], "subcluster_info": cube["sub_info"],
    }


def run(run_key: str) -> dict:
    cfg = RUN_CONFIGS[run_key]
    print(f"\n############# {run_key.upper()} ({cfg['d']} params) #############")
    t0 = time.time()
    ref = fit_reference(run_key)

    summaries = {}
    for cube_id in CUBE_IDS:
        is_ref = cube_id == REFERENCE
        cube = predict_cube(cube_id, ref, is_reference=is_ref)
        summary = emit_cube_outputs(run_key, cube, ref)
        summaries[cube_id] = summary
        if not is_ref:
            del cube
            gc.collect()
    wall = round(time.time() - t0, 2)
    print(f"\n[{run_key}] wall = {wall}s")
    return {
        "run_key": run_key,
        "cfg": {
            "params": list(cfg["params"]),
            "d": cfg["d"],
            "chi2_crit": cfg["chi2_crit"],
            "max_posterior_threshold": MAX_POSTERIOR_THRESHOLD,
            "n_init": N_INIT, "reg_covar": REG_COVAR,
            "random_seed": RANDOM_SEED,
            "pileup_flag_fraction": PILEUP_FLAG_FRACTION,
        },
        "explained_variance_ratio": ref["explained_variance_ratio"],
        "sizes_3e12": ref["sizes"],
        "cluster_means_original_units_3e12": ref["cluster_means_original_units"],
        "params_list": ref["params_list"],
        "per_cube": summaries,
        "wall_seconds": wall,
    }


def write_aggregate_reports(results: dict[str, dict]) -> None:
    rep_dir = SNAPSHOT / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    drift_check = {}
    novelty_stats = {}
    cluster_means = {}
    palette = {
        "encoding": {
            **{str(i): list(GMM_PALETTE_RGB[i]) for i in range(K_FIXED)},
            "novel_254": list(NOVEL_RGB),
            "invalid_255": [0, 0, 0, 0],
        },
        "ordering": ("Cluster IDs are in sklearn EM-converged order "
                     "(deterministic on seed 42). They do NOT carry "
                     "automatic mineralogical meaning. AV assigns labels "
                     "by inspecting cluster_means_original_units_3e12 in "
                     "v9_1_cluster_means_3e12.json and writes them into "
                     "the labels{} block of this file."),
        "labels": {str(i): None for i in range(K_FIXED)},
        "subcluster_palette": [list(c) for c in SUBCLUSTER_PALETTE_RGB],
        "novel_threshold": {
            "chi2_crit_v9_1a_d5": RUN_CONFIGS["v9_1a_5param"]["chi2_crit"],
            "chi2_crit_v9_1b_d7": RUN_CONFIGS["v9_1b_7param"]["chi2_crit"],
            "max_posterior_threshold": MAX_POSTERIOR_THRESHOLD,
        },
    }
    for run_key, R in results.items():
        drift_check[run_key] = {}
        novelty_stats[run_key] = {
            "thresholds": {
                "chi2_crit_0.95": R["cfg"]["chi2_crit"],
                "max_posterior_threshold": MAX_POSTERIOR_THRESHOLD,
                "pileup_flag_fraction": PILEUP_FLAG_FRACTION,
            },
            "explained_variance_ratio": R["explained_variance_ratio"],
            "sizes_3e12": R["sizes_3e12"],
            "per_cube": {},
        }
        cluster_means[run_key] = {
            "params": R["params_list"],
            "cluster_means_original_units_3e12": R["cluster_means_original_units_3e12"],
            "sizes_3e12": R["sizes_3e12"],
        }
        for cube_id, s in R["per_cube"].items():
            novelty_stats[run_key]["per_cube"][cube_id] = {
                "n_valid": s["n_valid"],
                "n_novel": s["n_novel"],
                "novel_fraction": s["novel_fraction"],
                "cluster_sizes": s["cluster_sizes"],
                "argmax_pileup": s["argmax_pileup"],
                "subcluster_info": s["subcluster_info"],
            }
            if s["drift"] is not None:
                drift_check[run_key][cube_id] = s["drift"]

    (rep_dir / "v9_1_drift_check.json").write_text(
        json.dumps(drift_check, indent=2, default=str), encoding="utf-8")
    (rep_dir / "v9_1_novelty_stats.json").write_text(
        json.dumps(novelty_stats, indent=2, default=str), encoding="utf-8")
    (rep_dir / "v9_1_cluster_means_3e12.json").write_text(
        json.dumps(cluster_means, indent=2, default=str), encoding="utf-8")
    (rep_dir / "v9_1_palette.json").write_text(
        json.dumps(palette, indent=2), encoding="utf-8")
    print(f"\nwrote {rep_dir}/v9_1_drift_check.json, "
          "v9_1_novelty_stats.json, v9_1_cluster_means_3e12.json, "
          "v9_1_palette.json")


def main(argv: list[str]) -> None:
    t0 = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    (SNAPSHOT / "reports").mkdir(parents=True, exist_ok=True)

    targets = list(RUN_CONFIGS) if len(argv) <= 1 else argv[1:]
    unknown = [t for t in targets if t not in RUN_CONFIGS]
    if unknown:
        raise SystemExit(f"unknown run keys: {unknown}; "
                          f"known: {list(RUN_CONFIGS)}")

    results = {}
    for run_key in targets:
        results[run_key] = run(run_key)

    if set(targets) == set(RUN_CONFIGS):
        write_aggregate_reports(results)

    top_log = {
        "stage": "v9_1_per_cube_scaler",
        "timestamp_utc": timestamp,
        "random_seed": RANDOM_SEED,
        "n_init": N_INIT, "reg_covar": REG_COVAR,
        "max_posterior_threshold": MAX_POSTERIOR_THRESHOLD,
        "pileup_flag_fraction": PILEUP_FLAG_FRACTION,
        "reference_cube": REFERENCE,
        "runs": {rk: {
            "params": list(RUN_CONFIGS[rk]["params"]),
            "d": RUN_CONFIGS[rk]["d"],
            "chi2_crit": RUN_CONFIGS[rk]["chi2_crit"],
            "wall_seconds": results[rk]["wall_seconds"],
        } for rk in results},
        "wall_clock_seconds": round(time.time() - t0, 2),
        "source_code_md5": {
            "stage_v9_1_per_cube_scaler.py": md5_of(Path(__file__)),
            "stage_v9_3e12_seeded.py": md5_of(
                Path(__file__).with_name("stage_v9_3e12_seeded.py")),
            "stage_v8_mtr3_with_brown.py": md5_of(
                Path(__file__).with_name("stage_v8_mtr3_with_brown.py")),
            "stage_v7c_mtr3_pipeline.py": md5_of(
                Path(__file__).with_name("stage_v7c_mtr3_pipeline.py")),
            "vb_parameters.py": md5_of(
                Path(__file__).with_name("vb_parameters.py")),
        },
    }
    (SNAPSHOT / "reports" / f"v9_1_run_log_{timestamp}.json").write_text(
        json.dumps(top_log, indent=2), encoding="utf-8")
    print(f"\nv9.1 complete in {round(time.time()-t0, 2)}s "
          f"(runs: {list(results)})")


if __name__ == "__main__":
    main(sys.argv)
