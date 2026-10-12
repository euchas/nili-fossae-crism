"""Stage v8 - v5-B + Brown centroid + asymmetry, on MTR3 cubes.

Re-runs the v7-C pipeline with a SEVEN-parameter stack:
  centroid, asymmetry  (from v6 Brown fit, full range, no eq.2 clip)
  OLINDEX3, BD1400, BD1900R2, D2300, BD2500_2  (v5-B alteration suite)

AV's hypothesis: adding the band-fit composition (centroid -> Fo
number) and texture (asymmetry -> grain size / saturation) axes
back into the clustering input space will sharpen the mineralogical
class boundaries vs the v5-B / v7-C "alteration only" basis.

Pipeline matches v7-C otherwise:
  per-cube standardize -> PCA -> whitened GMM k=6 ->
  Hungarian align to FRT00003E12 reference -> GeoTIFFs + .npy.

Outputs: outputs/_v8_mtr3_with_brown/<cube>/{arrays,geotiffs,logs}/
"""

from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from config import CUBE_IDS, RANDOM_SEED, OUTPUTS_DIR
from stage_v7c_mtr3_pipeline import (
    load_mtr3, write_float_geotiff, write_categorical_geotiff,
    cluster_centroids_std, GMM_PALETTE, MTR3_CUBE_DIRS, CUBES_DIR,
    K_FIXED, MAFIC_CAP_PARAMS, REFERENCE,
)
from vb_parameters import (
    calc_olindex3, calc_bd1400, calc_bd1900r2, calc_d2300, calc_bd2500_2,
)

# v8: 7-param stack = 2 Brown (centroid, asymmetry) + 5 VB.
# Brown loaded from v6 .npy; VB computed from MTR3 cube.
V8_PARAMS = ("centroid", "asymmetry",
             "OLINDEX3", "BD1400", "BD1900R2", "D2300", "BD2500_2")
N_PCS = 7  # full rank, no DR; same role as in v7-C (preconditioner)

VB_FUNCS = {
    "OLINDEX3": calc_olindex3,
    "BD1400":   calc_bd1400,
    "BD1900R2": calc_bd1900r2,
    "D2300":    calc_d2300,
    "BD2500_2": calc_bd2500_2,
}

SNAPSHOT = OUTPUTS_DIR / "_v8_mtr3_with_brown"


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def cube_root(cube_id: str) -> Path:
    p = SNAPSHOT / cube_id
    (p / "arrays").mkdir(parents=True, exist_ok=True)
    (p / "geotiffs").mkdir(parents=True, exist_ok=True)
    (p / "logs").mkdir(parents=True, exist_ok=True)
    return p


def load_brown_v6(cube_id: str) -> dict:
    dir_name = MTR3_CUBE_DIRS[cube_id]
    base = dir_name
    cube_dir = CUBES_DIR / dir_name
    out = {}
    for name in ("centroid", "asymmetry"):
        path = cube_dir / f"{base}_{name}_map_v6.npy"
        if not path.exists():
            raise SystemExit(f"missing v6 Brown map: {path}")
        out[name] = np.load(path).astype(np.float32)
    return out


def process_cube(cube_id: str) -> dict:
    t0 = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"\n=== {cube_id} v8 ===")
    dir_name = MTR3_CUBE_DIRS[cube_id]
    hdr_path = CUBES_DIR / dir_name / f"{dir_name}.hdr"

    print("  loading MTR3 cube...")
    cube_bls, wav, meta = load_mtr3(hdr_path)
    H, W = meta["lines"], meta["samples"]

    # Compute 5 VB params from cube
    print("  computing 5 VB params...")
    vb = {}
    for name in ("OLINDEX3", "BD1400", "BD1900R2", "D2300", "BD2500_2"):
        t_p = time.time()
        vb[name] = VB_FUNCS[name](cube_bls, wav).astype(np.float32)
        print(f"    {name}: {time.time()-t_p:.1f}s   "
              f"nan_frac={np.isnan(vb[name]).mean():.3f}")
    del cube_bls
    gc.collect()

    # Load Brown centroid + asymmetry (FULL RANGE - no eq.2 clip)
    print("  loading Brown centroid + asymmetry from v6...")
    brown = load_brown_v6(cube_id)
    for name, arr in brown.items():
        nan_frac = np.isnan(arr).mean()
        rng = (np.nanmin(arr), np.nanmax(arr))
        print(f"    {name}: nan_frac={nan_frac:.3f}   range={rng[0]:.3f} - {rng[1]:.3f}")

    # Combine into 7-param dict
    all_params = {**brown, **vb}
    assert all(p in all_params for p in V8_PARAMS), \
        f"Missing v8 params: {[p for p in V8_PARAMS if p not in all_params]}"

    # Pixel-valid: all 7 params finite
    finite = np.ones((H, W), dtype=bool)
    for name in V8_PARAMS:
        finite &= np.isfinite(all_params[name])
    n_valid = int(finite.sum())
    n_total = H * W
    print(f"  valid pixels: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")

    # Stack + standardize
    X = np.column_stack([all_params[p][finite] for p in V8_PARAMS]).astype(np.float64)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    print(f"  X shape = {X.shape}, standardized.")

    # PCA decorrelation + whitening (7 -> 7, full rank)
    pca = PCA(n_components=N_PCS, random_state=RANDOM_SEED)
    Xp = pca.fit_transform(Xs)
    col_std = Xp.std(axis=0, ddof=1)
    Xp_whit = Xp / col_std
    print(f"  PCA explained variance: "
          f"{[f'{v*100:.1f}%' for v in pca.explained_variance_ratio_]}")
    cum = np.cumsum(pca.explained_variance_ratio_)
    print(f"  cumulative: "
          f"{[f'{v*100:.1f}%' for v in cum]}")

    # GMM k=6
    gmm = GaussianMixture(
        n_components=K_FIXED, covariance_type="full",
        n_init=5, random_state=RANDOM_SEED, reg_covar=1e-5,
    )
    gmm.fit(Xp_whit)
    raw_labels = gmm.predict(Xp_whit).astype(np.int16)
    bic = float(gmm.bic(Xp_whit))
    sizes = [int((raw_labels == c).sum()) for c in range(K_FIXED)]
    print(f"  GMM raw: sizes={sizes}  BIC={bic:.0f}")

    # Save .npy
    arr_dir = cube_root(cube_id) / "arrays"
    np.save(arr_dir / f"{cube_id}_v8_param_stack.npy",
            np.dstack([all_params[p] for p in V8_PARAMS]))
    np.save(arr_dir / f"{cube_id}_v8_pixel_valid_mask.npy", finite)
    pc_scenes = np.full((N_PCS, H, W), np.nan, dtype=np.float32)
    pc_whit_scenes = np.full((N_PCS, H, W), np.nan, dtype=np.float32)
    for k in range(N_PCS):
        sc = np.full((H, W), np.nan, dtype=np.float32)
        sc[finite] = Xp[:, k].astype(np.float32)
        pc_scenes[k] = sc
        sc2 = np.full((H, W), np.nan, dtype=np.float32)
        sc2[finite] = Xp_whit[:, k].astype(np.float32)
        pc_whit_scenes[k] = sc2
    np.save(arr_dir / f"{cube_id}_v8_PC_scores.npy", pc_scenes)
    np.save(arr_dir / f"{cube_id}_v8_PC_scores_whitened.npy", pc_whit_scenes)
    raw_label_scene = np.full((H, W), -1, dtype=np.int16)
    raw_label_scene[finite] = raw_labels
    np.save(arr_dir / f"{cube_id}_v8_GMM_k6_labels_raw.npy", raw_label_scene)

    return {
        "cube_id": cube_id, "shape": (H, W), "n_valid": n_valid,
        "all_params": all_params, "finite": finite,
        "Xs": Xs, "raw_labels": raw_labels, "Xp": Xp,
        "pca": pca, "scaler": scaler,
        "bic": bic, "sizes": sizes, "wall": round(time.time() - t0, 2),
        "geo": meta["geo"], "timestamp": timestamp,
        "raw_label_scene": raw_label_scene,
        "pc_scenes": pc_scenes, "pc_whit_scenes": pc_whit_scenes,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }


def hungarian_align(per_cube: dict) -> dict:
    """Align v8 labels across cubes to FRT00003E12 mafic-cap ordering.
    Mafic-cap z-sum uses the same 4 alteration params as v7-C
    (BD1900R2, D2300, BD2500_2, OLINDEX3) - all of which exist in V8_PARAMS."""
    ref = per_cube[REFERENCE]
    ref_ctr = cluster_centroids_std(ref["Xs"], ref["raw_labels"])
    mafic_idx = [V8_PARAMS.index(p) for p in MAFIC_CAP_PARAMS if p in V8_PARAMS]
    score = ref_ctr[:, mafic_idx].sum(axis=1)
    sort_order = np.argsort(score)
    ref_relabel = np.argsort(sort_order)
    ref_aligned = np.array([ref_relabel[c] for c in ref["raw_labels"]],
                           dtype=np.int16)
    ref_ctr_aligned = ref_ctr[sort_order]

    aligned_by_cube = {REFERENCE: ref_aligned}
    log = {REFERENCE: {"permutation": ref_relabel.tolist(),
                       "distances_to_ref": [0.0] * K_FIXED}}
    for cube_id, c in per_cube.items():
        if cube_id == REFERENCE:
            continue
        ctr = cluster_centroids_std(c["Xs"], c["raw_labels"])
        D = cdist(ctr, ref_ctr_aligned)
        row_ind, col_ind = linear_sum_assignment(D)
        relabel = col_ind.copy()
        aligned = np.array([relabel[c2] for c2 in c["raw_labels"]],
                           dtype=np.int16)
        aligned_by_cube[cube_id] = aligned
        log[cube_id] = {
            "permutation": relabel.tolist(),
            "distances_to_ref": D[row_ind, col_ind].tolist(),
            "full_distance_matrix": D.tolist(),
        }
    return {"aligned_by_cube": aligned_by_cube, "log": log}


def main(argv: list[str]) -> None:
    t0 = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    targets = list(CUBE_IDS) if len(argv) <= 1 else argv[1:]
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    (SNAPSHOT / "reports").mkdir(parents=True, exist_ok=True)

    per_cube: dict = {}
    for cube_id in targets:
        per_cube[cube_id] = process_cube(cube_id)

    if set(targets) == set(CUBE_IDS):
        print("\n=== cross-cube Hungarian alignment ===")
        aln = hungarian_align(per_cube)
        for cube_id, c in per_cube.items():
            aligned_scene = np.full(c["shape"], -1, dtype=np.int16)
            aligned_scene[c["finite"]] = aln["aligned_by_cube"][cube_id]
            np.save(cube_root(cube_id) / "arrays"
                    / f"{cube_id}_v8_GMM_k6_labels_aligned.npy",
                    aligned_scene)
            c["aligned_scene"] = aligned_scene
        (SNAPSHOT / "reports" / "03_v8_alignment.json").write_text(
            json.dumps(aln["log"], indent=2), encoding="utf-8")

    # GeoTIFFs (per cube)
    print("\n=== GeoTIFF outputs ===")
    for cube_id, c in per_cube.items():
        gdir = cube_root(cube_id) / "geotiffs"
        for k in range(N_PCS):
            write_float_geotiff(
                c["pc_scenes"][k],
                gdir / f"{cube_id}_v8_PC{k+1}_score.tif",
                c["geo"], description=f"v8 PC{k+1} score (raw)")
            write_float_geotiff(
                c["pc_whit_scenes"][k],
                gdir / f"{cube_id}_v8_PC{k+1}_whitened.tif",
                c["geo"], description=f"v8 PC{k+1} whitened score")
        write_categorical_geotiff(
            c["raw_label_scene"], gdir / f"{cube_id}_v8_GMM_k6_raw.tif",
            c["geo"], description="v8 GMM k=6 raw labels (per-cube)")
        if "aligned_scene" in c:
            write_categorical_geotiff(
                c["aligned_scene"],
                gdir / f"{cube_id}_v8_GMM_k6_aligned.tif",
                c["geo"],
                description="v8 GMM k=6 Hungarian-aligned to FRT00003E12")
        log_path = cube_root(cube_id) / "logs" / f"{cube_id}_v8_log.json"
        log_path.write_text(json.dumps({
            "cube_id": cube_id,
            "stage": "v8_mtr3_with_brown",
            "timestamp_utc": c["timestamp"],
            "v8_params": list(V8_PARAMS),
            "shape": list(c["shape"]),
            "n_valid": int(c["n_valid"]),
            "gmm_sizes_raw": c["sizes"],
            "gmm_bic": c["bic"],
            "explained_variance_ratio": c["explained_variance_ratio"],
            "wall_clock_seconds": c["wall"],
            "geo_proj_name": c["geo"]["proj_name"] if c["geo"] else None,
        }, indent=2, default=str), encoding="utf-8")
        print(f"  {cube_id}: wrote {N_PCS*2 + 2} GeoTIFFs + log")

    (SNAPSHOT / "reports" / f"v8_run_log_{timestamp}.json").write_text(
        json.dumps({
            "stage": "v8_mtr3_with_brown",
            "timestamp_utc": timestamp,
            "random_seed": RANDOM_SEED,
            "v8_params": list(V8_PARAMS),
            "k_fixed": K_FIXED,
            "n_pcs": N_PCS,
            "reference_cube": REFERENCE,
            "wall_clock_seconds": round(time.time() - t0, 2),
            "source_code_md5": {
                "stage_v8_mtr3_with_brown.py": md5_of(Path(__file__)),
                "stage_v7c_mtr3_pipeline.py":
                    md5_of(Path(__file__).with_name("stage_v7c_mtr3_pipeline.py")),
                "vb_parameters.py":
                    md5_of(Path(__file__).with_name("vb_parameters.py")),
            },
        }, indent=2), encoding="utf-8")
    print(f"\nv8 complete in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main(sys.argv)
