"""Stage v7-C - Port of v5-B GMM clustering to map-projected MTR3 cubes.

For each of the four MTR3 I/F cubes:
  1. Load with the memory-efficient np.fromfile pattern from
     1um_centroid_analysis_CRISM_MTR3_v1.py (BSQ -> (bands, lines, samples)
     contiguous view, single allocation ~1.2 GB peak).
  2. Compute the v5-B 5-parameter VB stack (OLINDEX3, BD1400, BD1900R2,
     D2300, BD2500_2) using the existing vb_parameters helpers.
  3. Build a finite-only mask, z-score per cube, decorrelate + whiten,
     fit GMM(k=6, full covariance, n_init=5, seed 42).
  4. Hungarian-align to FRT00003E12 reference, mafic-cap-ordered.
  5. Save .npy arrays (VB stack, PC scores, GMM raw + aligned labels) +
     GeoTIFFs (5 PC scores float32 + GMM raw + aligned uint8 categorical).

No display flip per AV's v7-C decision: MTR3 cubes are already map-
projected, north-up by IAU convention.

Outputs: outputs/_v7_mtr3_map_projected/<cube>/{arrays,geotiffs,logs}/
"""

from __future__ import annotations

import gc
import hashlib
import json
import re
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
import spectral

from config import CUBE_IDS, CUBES_DIR, RANDOM_SEED, OUTPUTS_DIR
from vb_parameters import (
    calc_olindex3, calc_bd1400, calc_bd1900r2, calc_d2300, calc_bd2500_2,
)

V5B_PARAMS = ("OLINDEX3", "BD1400", "BD1900R2", "D2300", "BD2500_2")
VB_FUNCS = {
    "OLINDEX3": calc_olindex3,
    "BD1400":   calc_bd1400,
    "BD1900R2": calc_bd1900r2,
    "D2300":    calc_d2300,
    "BD2500_2": calc_bd2500_2,
}
MAFIC_CAP_PARAMS = ("BD1900R2", "D2300", "BD2500_2", "OLINDEX3")
LCP_TIE = "OLINDEX3"  # tie-break (LCPINDEX2 not in v5-B)
K_FIXED = 6
N_PCS = 5
NAN_VALUE = 65535.0
REFERENCE = "FRT00003E12"
GMM_NODATA_VAL = 255

SNAPSHOT = OUTPUTS_DIR / "_v7_mtr3_map_projected"

MTR3_CUBE_DIRS = {
    "FRT00003E12": "frt00003e12_07_if166j_mtr3",
    "FRT0000871C": "frt0000871c_07_if166j_mtr3",
    "FRT0000A4FC": "frt0000a4fc_07_if166j_mtr3",
    "FRT0000B438": "frt0000b438_07_if166j_mtr3",
}

GMM_PALETTE = [(31, 119, 180), (255, 127, 14), (44, 160, 44),
               (214, 39, 40), (148, 103, 189), (140, 86, 75)]


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# MTR3 load (single allocation, BSQ -> (bands, lines, samples) contiguous)
# =============================================================================
def load_mtr3(hdr_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    img = spectral.open_image(str(hdr_path))
    lines = int(img.metadata["lines"])
    samples = int(img.metadata["samples"])
    bands = int(img.metadata["bands"])
    interleave = img.metadata.get("interleave", "bsq").lower()
    dtype_code = int(img.metadata.get("data type", 4))
    byte_order = int(img.metadata.get("byte order", 0))
    header_offset = int(img.metadata.get("header offset", 0))

    dtype_map = {1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32,
                 5: np.float64, 12: np.uint16, 13: np.uint32}
    dtype = np.dtype(dtype_map.get(dtype_code, np.float32))
    if byte_order == 1:
        dtype = dtype.newbyteorder(">")

    img_path = str(hdr_path).replace(".hdr", ".img")
    n_values = lines * samples * bands
    with open(img_path, "rb") as f:
        if header_offset:
            f.seek(header_offset)
        flat = np.fromfile(f, dtype=dtype, count=n_values)
    if flat.size != n_values:
        raise ValueError(f"Read {flat.size:,} values, expected {n_values:,}")

    # Reshape (bands, lines, samples) for BSQ -- contiguous, no transpose.
    if interleave == "bsq":
        cube_bls = flat.reshape((bands, lines, samples))
    elif interleave == "bil":
        cube_bls = flat.reshape((lines, bands, samples)).transpose(1, 0, 2)
    elif interleave == "bip":
        cube_bls = flat.reshape((lines, samples, bands)).transpose(2, 0, 1)
    else:
        raise ValueError(f"unknown interleave: {interleave}")

    if cube_bls.dtype != np.float32:
        cube_bls = cube_bls.astype(np.float32, copy=False)

    # Replace bad-value sentinels with NaN
    cube_bls = np.where(cube_bls >= NAN_VALUE - 1, np.nan, cube_bls)

    wav = np.array(img.metadata["wavelength"], dtype=float)
    if wav.max() > 10:
        wav = wav / 1000.0

    geo = parse_envi_geo(hdr_path)
    return cube_bls, wav.astype(np.float32), {
        "lines": lines, "samples": samples, "bands": bands,
        "interleave": interleave, "dtype_code": dtype_code,
        "geo": geo,
    }


def parse_envi_geo(hdr_path: Path) -> dict | None:
    content = hdr_path.read_text()
    m = re.search(r"map\s+info\s*=\s*\{([^}]+)\}",
                  content, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    parts = [p.strip() for p in m.group(1).split(",")]
    if len(parts) < 7:
        return None
    try:
        proj_name = parts[0]
        x_ref = float(parts[1]); y_ref = float(parts[2])
        x_coord = float(parts[3]); y_coord = float(parts[4])
        x_pix = float(parts[5]); y_pix = float(parts[6])
    except (ValueError, IndexError):
        return None
    x_origin = x_coord - (x_ref - 0.5) * x_pix
    y_origin = y_coord + (y_ref - 0.5) * y_pix
    geotransform = (x_origin, x_pix, 0.0, y_origin, 0.0, -y_pix)
    cs = re.search(r"coordinate\s+system\s+string\s*=\s*\{([^}]+)\}",
                   content, re.IGNORECASE | re.DOTALL)
    wkt = cs.group(1).strip() if cs else None
    return {"proj_name": proj_name, "geotransform": geotransform,
            "wkt": wkt, "x_pixel_size": x_pix, "y_pixel_size": y_pix}


# =============================================================================
# GeoTIFF writers
# =============================================================================
def _make_transform(geo: dict | None) -> Affine:
    if geo and geo.get("geotransform"):
        gt = geo["geotransform"]
        return Affine(gt[1], gt[2], gt[0], gt[4], gt[5], gt[3])
    return Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)


def _make_crs(geo: dict | None):
    if geo and geo.get("wkt"):
        try:
            return CRS.from_wkt(geo["wkt"])
        except Exception:
            return None
    return None


def write_float_geotiff(arr: np.ndarray, path: Path, geo: dict | None,
                        description: str = "") -> None:
    h, w = arr.shape
    profile = {
        "driver": "GTiff", "dtype": "float32",
        "width": w, "height": h, "count": 1,
        "crs": _make_crs(geo), "transform": _make_transform(geo),
        "nodata": float("nan"), "compress": "deflate",
    }
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)
        if description:
            dst.set_band_description(1, description)


def write_categorical_geotiff(labels: np.ndarray, path: Path,
                              geo: dict | None, description: str = "") -> None:
    """Single-band uint8 with NoData=255, RGB color table for the 6 GMM
    classes. Labels of -1 (no data) are mapped to 255."""
    h, w = labels.shape
    arr = labels.astype(np.int16)
    out = np.where(arr < 0, GMM_NODATA_VAL, arr).astype(np.uint8)
    profile = {
        "driver": "GTiff", "dtype": "uint8",
        "width": w, "height": h, "count": 1,
        "crs": _make_crs(geo), "transform": _make_transform(geo),
        "nodata": GMM_NODATA_VAL, "compress": "deflate",
    }
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(out, 1)
        # 6-class color table + NoData = transparent
        ct = {i: (*GMM_PALETTE[i], 255) for i in range(K_FIXED)}
        ct[GMM_NODATA_VAL] = (0, 0, 0, 0)
        dst.write_colormap(1, ct)
        if description:
            dst.set_band_description(1, description)


# =============================================================================
# v5-B pipeline (per cube)
# =============================================================================
def cube_root(cube_id: str) -> Path:
    p = SNAPSHOT / cube_id
    (p / "arrays").mkdir(parents=True, exist_ok=True)
    (p / "geotiffs").mkdir(parents=True, exist_ok=True)
    (p / "logs").mkdir(parents=True, exist_ok=True)
    return p


def cluster_centroids_std(Xs: np.ndarray, labels: np.ndarray,
                          k: int = K_FIXED) -> np.ndarray:
    ctr = np.full((k, Xs.shape[1]), np.nan)
    for c in range(k):
        m = labels == c
        if m.any():
            ctr[c] = Xs[m].mean(axis=0)
    return ctr


def process_cube(cube_id: str) -> dict:
    t0 = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"\n=== {cube_id} v7-C ===")
    dir_name = MTR3_CUBE_DIRS[cube_id]
    hdr_path = CUBES_DIR / dir_name / f"{dir_name}.hdr"
    base = dir_name

    print(f"  loading MTR3 cube...")
    cube_bls, wav, meta = load_mtr3(hdr_path)
    H, W = meta["lines"], meta["samples"]
    print(f"  shape (bands, lines, samples) = {cube_bls.shape}, wav range {wav[0]:.3f} - {wav[-1]:.3f} um")

    # Compute v5-B 5-param stack
    print(f"  computing 5 VB params...")
    vb = {}
    for name in V5B_PARAMS:
        t_p = time.time()
        vb[name] = VB_FUNCS[name](cube_bls, wav).astype(np.float32)
        print(f"    {name}: {time.time()-t_p:.1f}s   "
              f"shape={vb[name].shape}   nan_frac={np.isnan(vb[name]).mean():.3f}")

    # Free the cube buffer (we only need the param maps now)
    del cube_bls
    gc.collect()

    # Build finite-only mask + standardized matrix
    finite = np.ones((H, W), dtype=bool)
    for name in V5B_PARAMS:
        finite &= np.isfinite(vb[name])
    n_valid = int(finite.sum())
    n_total = H * W
    print(f"  valid pixels: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")

    X = np.column_stack([vb[p][finite] for p in V5B_PARAMS]).astype(np.float64)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    print(f"  X shape = {X.shape}, standardized.")

    # PCA decorrelation + whitening (5 -> 5, full rank, preconditioner only)
    pca = PCA(n_components=N_PCS, random_state=RANDOM_SEED)
    Xp = pca.fit_transform(Xs)
    col_std = Xp.std(axis=0, ddof=1)
    Xp_whit = Xp / col_std
    print(f"  PCA components: {pca.explained_variance_ratio_}")

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

    # Save .npy arrays
    arr_dir = cube_root(cube_id) / "arrays"
    np.save(arr_dir / f"{cube_id}_v7c_VB_stack.npy",
            np.dstack([vb[p] for p in V5B_PARAMS]))
    np.save(arr_dir / f"{cube_id}_v7c_pixel_valid_mask.npy", finite)
    # PC scores and GMM labels back-projected to (H, W)
    pc_scenes = np.full((N_PCS, H, W), np.nan, dtype=np.float32)
    for k in range(N_PCS):
        scene = np.full((H, W), np.nan, dtype=np.float32)
        scene[finite] = Xp[:, k].astype(np.float32)
        pc_scenes[k] = scene
    np.save(arr_dir / f"{cube_id}_v7c_PC_scores.npy", pc_scenes)
    pc_whit_scenes = np.full((N_PCS, H, W), np.nan, dtype=np.float32)
    for k in range(N_PCS):
        scene = np.full((H, W), np.nan, dtype=np.float32)
        scene[finite] = Xp_whit[:, k].astype(np.float32)
        pc_whit_scenes[k] = scene
    np.save(arr_dir / f"{cube_id}_v7c_PC_scores_whitened.npy", pc_whit_scenes)
    raw_label_scene = np.full((H, W), -1, dtype=np.int16)
    raw_label_scene[finite] = raw_labels
    np.save(arr_dir / f"{cube_id}_v7c_GMM_k6_labels_raw.npy", raw_label_scene)

    return {
        "cube_id": cube_id, "shape": (H, W), "n_valid": n_valid,
        "vb": vb, "finite": finite,
        "Xs": Xs, "raw_labels": raw_labels,
        "Xp": Xp, "pca": pca, "scaler": scaler,
        "bic": bic, "sizes": sizes, "wall": round(time.time() - t0, 2),
        "geo": meta["geo"], "timestamp": timestamp,
        "raw_label_scene": raw_label_scene,
        "pc_scenes": pc_scenes, "pc_whit_scenes": pc_whit_scenes,
    }


# =============================================================================
# Cross-cube alignment
# =============================================================================
def hungarian_align(per_cube: dict) -> dict:
    """Align v5-B labels across cubes to FRT00003E12 mafic-cap ordering."""
    ref = per_cube[REFERENCE]
    ref_ctr = cluster_centroids_std(ref["Xs"], ref["raw_labels"])
    mafic_idx = [V5B_PARAMS.index(p) for p in MAFIC_CAP_PARAMS if p in V5B_PARAMS]
    score = ref_ctr[:, mafic_idx].sum(axis=1)
    tie_idx = V5B_PARAMS.index(LCP_TIE)
    sort_order = np.lexsort((-ref_ctr[:, tie_idx], score))
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


# =============================================================================
# Main
# =============================================================================
def main(argv: list[str]) -> None:
    t0 = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    targets = list(CUBE_IDS) if len(argv) <= 1 else argv[1:]
    if any(t not in CUBE_IDS for t in targets):
        raise SystemExit(f"unknown cube id; known: {CUBE_IDS}")
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    (SNAPSHOT / "reports").mkdir(parents=True, exist_ok=True)

    per_cube: dict = {}
    for cube_id in targets:
        per_cube[cube_id] = process_cube(cube_id)

    # Cross-cube alignment
    if set(targets) == set(CUBE_IDS):
        print("\n=== cross-cube Hungarian alignment ===")
        aln = hungarian_align(per_cube)
        for cube_id, c in per_cube.items():
            aligned_scene = np.full(c["shape"], -1, dtype=np.int16)
            aligned_scene[c["finite"]] = aln["aligned_by_cube"][cube_id]
            np.save(cube_root(cube_id) / "arrays"
                    / f"{cube_id}_v7c_GMM_k6_labels_aligned.npy",
                    aligned_scene)
            c["aligned_scene"] = aligned_scene
        (SNAPSHOT / "reports" / "03_v7c_alignment.json").write_text(
            json.dumps(aln["log"], indent=2), encoding="utf-8")

    # GeoTIFFs (per cube)
    print("\n=== GeoTIFF outputs ===")
    for cube_id, c in per_cube.items():
        gdir = cube_root(cube_id) / "geotiffs"
        # PC scores (5 single-band float32)
        for k in range(N_PCS):
            write_float_geotiff(
                c["pc_scenes"][k],
                gdir / f"{cube_id}_v7c_PC{k+1}_score.tif",
                c["geo"], description=f"v5-B PC{k+1} score (raw)")
            write_float_geotiff(
                c["pc_whit_scenes"][k],
                gdir / f"{cube_id}_v7c_PC{k+1}_whitened.tif",
                c["geo"], description=f"v5-B PC{k+1} whitened score")
        # GMM raw + aligned (uint8 categorical w/ palette)
        write_categorical_geotiff(
            c["raw_label_scene"], gdir / f"{cube_id}_v7c_GMM_k6_raw.tif",
            c["geo"], description="v5-B GMM k=6 raw labels (per-cube)")
        if "aligned_scene" in c:
            write_categorical_geotiff(
                c["aligned_scene"],
                gdir / f"{cube_id}_v7c_GMM_k6_aligned.tif",
                c["geo"],
                description="v5-B GMM k=6 Hungarian-aligned to FRT00003E12")
        # 5 VB param GeoTIFFs (skip per AV's decision -- only PCs + GMM)
        # (Per spec: "5 VB params" was not explicitly requested as GeoTIFFs;
        # only PC scores + GMM. .npy stack is preserved in arrays/.)
        # Per-cube run log
        log_path = cube_root(cube_id) / "logs" / f"{cube_id}_v7c_log.json"
        log_path.write_text(json.dumps({
            "cube_id": cube_id,
            "stage": "v7c_mtr3_pipeline",
            "timestamp_utc": c["timestamp"],
            "v5b_params": list(V5B_PARAMS),
            "shape": list(c["shape"]),
            "n_valid": int(c["n_valid"]),
            "gmm_sizes_raw": c["sizes"],
            "gmm_bic": c["bic"],
            "wall_clock_seconds": c["wall"],
            "geo_proj_name": c["geo"]["proj_name"] if c["geo"] else None,
            "geo_pixel_size": (c["geo"]["x_pixel_size"]
                               if c["geo"] else None),
        }, indent=2, default=str), encoding="utf-8")
        print(f"  {cube_id}: wrote {N_PCS*2 + 2} GeoTIFFs + log")

    # Top-level run log
    (SNAPSHOT / "reports" / f"v7c_run_log_{timestamp}.json").write_text(
        json.dumps({
            "stage": "v7c_mtr3_pipeline",
            "timestamp_utc": timestamp,
            "random_seed": RANDOM_SEED,
            "v5b_params": list(V5B_PARAMS),
            "k_fixed": K_FIXED,
            "n_pcs": N_PCS,
            "reference_cube": REFERENCE,
            "wall_clock_seconds": round(time.time() - t0, 2),
            "source_code_md5": {
                "stage_v7c_mtr3_pipeline.py": md5_of(Path(__file__)),
                "vb_parameters.py":
                    md5_of(Path(__file__).with_name("vb_parameters.py")),
            },
        }, indent=2), encoding="utf-8")
    print(f"\nv7-C complete in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main(sys.argv)
