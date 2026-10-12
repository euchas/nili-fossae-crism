"""Stage A — build per-cube 8-parameter feature stacks + run logs.

For each of the four cubes:
  1. Load the TER3 joined cube + wavelengths.
  2. Load the six Brown v5 .npy maps.
  3. Compute OLINDEX3, D2300, BD1900R2, BD2500_2 on the cube.
  4. Build an 8-parameter stack (Brown 4 + VB 4), a Brown-valid mask, and a
     joint-valid (Brown AND VB finite) mask.
  5. Write:
       outputs/feature_stacks/<CUBE_ID>_stackA_8param.npz
       outputs/feature_stacks/<CUBE_ID>_stackA_8param_long.parquet
       outputs/logs/<CUBE_ID>_stackA_<YYYYmmddTHHMMSS>.json

Usage:
    python stage_a_build_feature_stack.py              # process all 4 cubes
    python stage_a_build_feature_stack.py FRT00003E12  # single cube
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import BROWN_VERSION, CUBE_IDS, DEFAULT_PARAM_ORDER, RANDOM_SEED, out_dir
from crism_io import load_cube_by_id
from vb_parameters import compute_default_vb_stack


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def nan_fraction(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 1.0
    return float(np.isnan(arr).sum()) / arr.size


def process_cube(cube_id: str) -> dict:
    t0 = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"\n=== {cube_id} ===")
    cube, wav, brown, img_path = load_cube_by_id(cube_id)
    print(
        f"  cube shape={cube.shape}  wav=[{wav[0]:.3f}, {wav[-1]:.3f}] µm  bands={len(wav)}"
    )

    for k, v in brown.items():
        print(
            f"  brown[{k:9s}] shape={v.shape} dtype={v.dtype}  nan_frac={nan_fraction(v):.3f}"
        )

    print("  computing VB parameters...")
    vb = compute_default_vb_stack(cube, wav)
    for k, v in vb.items():
        print(f"  vb[{k:8s}] nan_frac={nan_fraction(v):.3f}")

    params = {
        "centroid": brown["centroid"].astype(np.float32),
        "asymmetry": brown["asymmetry"].astype(np.float32),
        "amplitude": brown["amplitude"].astype(np.float32),
        "hwhm": brown["hwhm"].astype(np.float32),
        "OLINDEX3": vb["OLINDEX3"].astype(np.float32),
        "D2300": vb["D2300"].astype(np.float32),
        "BD1900R2": vb["BD1900R2"].astype(np.float32),
        "BD2500_2": vb["BD2500_2"].astype(np.float32),
    }

    shape = params["centroid"].shape
    for name, arr in params.items():
        if arr.shape != shape:
            raise RuntimeError(
                f"Shape mismatch: {name} has {arr.shape}, expected {shape}"
            )

    brown_mask = brown["mask"].astype(bool)
    finite_brown = (
        np.isfinite(params["centroid"])
        & np.isfinite(params["asymmetry"])
        & np.isfinite(params["amplitude"])
        & np.isfinite(params["hwhm"])
    )
    finite_vb = (
        np.isfinite(params["OLINDEX3"])
        & np.isfinite(params["D2300"])
        & np.isfinite(params["BD1900R2"])
        & np.isfinite(params["BD2500_2"])
    )
    joint_valid = brown_mask & finite_brown & finite_vb
    residual = brown["residual"].astype(np.float32)

    n_total = int(np.prod(shape))
    n_brown_mask = int(brown_mask.sum())
    n_finite_brown = int(finite_brown.sum())
    n_finite_vb = int(finite_vb.sum())
    n_joint = int(joint_valid.sum())
    print(
        f"  valid pixels: brown_mask={n_brown_mask}  finite_brown={n_finite_brown}  "
        f"finite_vb={n_finite_vb}  joint={n_joint} ({100.0*n_joint/n_total:.1f}%)"
    )

    # Save .npz
    npz_path = out_dir("feature_stacks") / f"{cube_id}_stackA_8param.npz"
    np.savez_compressed(
        npz_path,
        centroid=params["centroid"],
        asymmetry=params["asymmetry"],
        amplitude=params["amplitude"],
        hwhm=params["hwhm"],
        OLINDEX3=params["OLINDEX3"],
        D2300=params["D2300"],
        BD1900R2=params["BD1900R2"],
        BD2500_2=params["BD2500_2"],
        brown_mask=brown_mask,
        joint_valid=joint_valid,
        residual=residual,
        wavelengths_um=wav.astype(np.float32),
        param_order=np.array(list(DEFAULT_PARAM_ORDER)),
    )
    print(f"  saved {npz_path}")

    # Save tidy parquet of valid pixels.
    yy, xx = np.where(joint_valid)
    df_valid = pd.DataFrame(
        {
            "line": yy.astype(np.int32),
            "sample": xx.astype(np.int32),
            **{name: arr[yy, xx] for name, arr in params.items()},
            "residual": residual[yy, xx],
        }
    )
    parquet_path = (
        out_dir("feature_stacks") / f"{cube_id}_stackA_8param_long.parquet"
    )
    df_valid.to_parquet(parquet_path, index=False, compression="snappy")
    print(f"  saved {parquet_path} ({len(df_valid)} rows)")

    # Log
    log: dict = {
        "cube_id": cube_id,
        "run_timestamp_utc": timestamp,
        "random_seed": RANDOM_SEED,
        "brown_version": BROWN_VERSION,
        "img_path": str(img_path),
        "cube_shape_bands_lines_samples": list(cube.shape),
        "spatial_shape_lines_samples": list(shape),
        "n_bands": int(cube.shape[0]),
        "wavelength_min_um": float(wav[0]),
        "wavelength_max_um": float(wav[-1]),
        "param_order": list(DEFAULT_PARAM_ORDER),
        "vb_kernel_note": (
            "OLINDEX3 uses width=7 at all seven anchor channels per the existing "
            "PCA_CRISM_analysis_v2 convention. D2300 uses width=5 for 1.815/2.120/"
            "2.170/2.210/2.530 and width=3 for 2.290/2.320/2.330. BD1900R2 uses "
            "width=5 for the continuum anchors (1.850, 2.060) and width=1 for the "
            "six numerator/denominator probe channels. BD2500_2 uses width=5 for "
            "all three (2.364, 2.480, 2.570)."
        ),
        "n_pixels_total": n_total,
        "n_pixels_brown_mask": n_brown_mask,
        "n_pixels_finite_brown_all4": n_finite_brown,
        "n_pixels_finite_vb_all4": n_finite_vb,
        "n_pixels_joint_valid": n_joint,
        "joint_valid_fraction": round(n_joint / n_total, 4),
        "nan_fraction_by_param": {
            name: round(nan_fraction(arr), 4) for name, arr in params.items()
        },
        "nan_fraction_brown_residual": round(nan_fraction(residual), 4),
        "output_npz": str(npz_path),
        "output_parquet": str(parquet_path),
        "source_code_md5": {
            "stage_a_build_feature_stack.py": md5_of(Path(__file__)),
            "vb_parameters.py": md5_of(Path(__file__).with_name("vb_parameters.py")),
            "crism_io.py": md5_of(Path(__file__).with_name("crism_io.py")),
            "config.py": md5_of(Path(__file__).with_name("config.py")),
        },
        "wall_clock_seconds": round(time.time() - t0, 2),
    }
    log_path = out_dir("logs") / f"{cube_id}_stackA_{timestamp}.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  saved log {log_path}")
    return log


def main(argv: list[str]) -> None:
    targets = argv[1:] if len(argv) > 1 else list(CUBE_IDS)
    unknown = [t for t in targets if t not in CUBE_IDS]
    if unknown:
        raise SystemExit(f"Unknown cube id(s): {unknown}; known: {CUBE_IDS}")
    for cube_id in targets:
        process_cube(cube_id)


if __name__ == "__main__":
    main(sys.argv)
