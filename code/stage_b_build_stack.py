"""Stage B-0 — build the 15-parameter feature stack for each cube.

For each cube:
  1. Load the TER3 joined cube + wavelengths.
  2. Load the Brown v5 maps (same as Stage A).
  3. Compute the 11 VB parameters defined in the Stage B plan.
  4. Build masks:
       - finite across all 4 Brown params                    -> brown_valid
         (when STRICT_BROWN_MASK is True the analysis is further
          restricted to brown["mask"], which is the Brown 2020
          centroid >= 1.15 um + residual-quality cut)
       - finite across all 11 VB params                      -> vb_valid
       - brown_valid & vb_valid                              -> joint_valid_15
  5. Save:
       outputs/feature_stacks/<CUBE_ID>_stackB_15param.npz
       outputs/feature_stacks/<CUBE_ID>_stackB_15param_long.parquet
       outputs/logs/<CUBE_ID>_stageB_build_<TIMESTAMP>.json

BDI1000VIS is the only nontrivial parameter. The log records per-cube
BDI1000VIS NaN fraction vs the baseline VB-finite fraction; if the gap
is > 5 percentage points, the script prints a loud warning and still
writes the outputs (so the fit is auditable) but the downstream runs
should inspect before using.

STRICT_BROWN_MASK: set to True to reproduce the v1 Brown 2020-style
analysis (centroid >= 1.15 um). Set to False for the full-range
analysis that includes mafic-cap and phyllosilicate populations.
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

from config import (
    BROWN_VERSION,
    CUBE_IDS,
    RANDOM_SEED,
    STAGE_B_BROWN_PARAMS,
    STAGE_B_PARAM_ORDER,
    STAGE_B_VB_EXPLORATORY,
    STAGE_B_VB_REQUIRED,
    out_dir,
)
from crism_io import load_cube_by_id
from vb_parameters import compute_extended_vb_stack

STRICT_BROWN_MASK = False


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
    print(f"  cube={cube.shape}  wav=[{wav[0]:.3f},{wav[-1]:.3f}] um bands={len(wav)}")

    print("  computing 11 VB parameters...")
    t_vb0 = time.time()
    vb = compute_extended_vb_stack(cube, wav)
    print(f"  VB stack done in {time.time() - t_vb0:.1f}s")
    for name, arr in vb.items():
        print(f"    vb[{name:11s}] nan_frac={nan_fraction(arr):.4f}")

    params: dict[str, np.ndarray] = {
        "centroid": brown["centroid"].astype(np.float32),
        "asymmetry": brown["asymmetry"].astype(np.float32),
        "amplitude": brown["amplitude"].astype(np.float32),
        "hwhm": brown["hwhm"].astype(np.float32),
    }
    for name, arr in vb.items():
        params[name] = arr.astype(np.float32)

    for name in STAGE_B_PARAM_ORDER:
        if name not in params:
            raise RuntimeError(f"Missing Stage B parameter: {name}")

    shape = params["centroid"].shape
    for name, arr in params.items():
        if arr.shape != shape:
            raise RuntimeError(
                f"Shape mismatch for {name}: {arr.shape} vs {shape}"
            )

    brown_mask_raw = brown["mask"].astype(bool)
    finite_brown = np.ones(shape, dtype=bool)
    for p in STAGE_B_BROWN_PARAMS:
        finite_brown &= np.isfinite(params[p])
    if STRICT_BROWN_MASK:
        brown_valid = brown_mask_raw & finite_brown
    else:
        brown_valid = finite_brown

    vb_valid = np.ones(shape, dtype=bool)
    vb_names = tuple(STAGE_B_VB_REQUIRED) + tuple(STAGE_B_VB_EXPLORATORY)
    for p in vb_names:
        vb_valid &= np.isfinite(params[p])

    joint_valid_15 = brown_valid & vb_valid
    residual = brown["residual"].astype(np.float32)

    n_total = int(np.prod(shape))
    mode = "STRICT (centroid>=1.15)" if STRICT_BROWN_MASK else "FULL RANGE (no centroid clip)"
    print(
        f"  mode={mode}\n"
        f"  mask counts: brown_raw={brown_mask_raw.sum()}  "
        f"finite_brown={finite_brown.sum()}  "
        f"brown_valid={brown_valid.sum()}  "
        f"vb_valid={vb_valid.sum()}  "
        f"joint_valid_15={joint_valid_15.sum()} "
        f"({100.0 * joint_valid_15.sum() / n_total:.1f}%)"
    )

    # Compare BDI1000VIS NaN fraction to the baseline VB-finite count;
    # VB-without-BDI is the most informative reference because it excludes
    # the one heavy computation.
    vb_without_bdi = np.ones(shape, dtype=bool)
    for p in vb_names:
        if p == "BDI1000VIS":
            continue
        vb_without_bdi &= np.isfinite(params[p])
    bdi_finite = np.isfinite(params["BDI1000VIS"])
    baseline = vb_without_bdi.sum()
    bdi_and_baseline = (bdi_finite & vb_without_bdi).sum()
    bdi_loss = 1.0 - (bdi_and_baseline / max(1, baseline))
    print(
        f"  BDI1000VIS loss vs VB-without-BDI baseline: {100.0*bdi_loss:.2f}% "
        f"({baseline - bdi_and_baseline} pixels lost)"
    )
    bdi_flag = bdi_loss > 0.05
    if bdi_flag:
        print("  *** WARNING: BDI1000VIS NaN gap exceeds 5%; inspect before clustering ***")

    npz_path = out_dir("feature_stacks") / f"{cube_id}_stackB_15param.npz"
    np.savez_compressed(
        npz_path,
        brown_mask=brown_mask_raw,
        brown_valid=brown_valid,
        vb_valid=vb_valid,
        joint_valid_15=joint_valid_15,
        residual=residual,
        wavelengths_um=wav.astype(np.float32),
        param_order=np.array(list(STAGE_B_PARAM_ORDER)),
        **params,
    )
    print(f"  saved {npz_path}")

    yy, xx = np.where(joint_valid_15)
    tidy = {
        "line": yy.astype(np.int32),
        "sample": xx.astype(np.int32),
        **{name: params[name][yy, xx] for name in STAGE_B_PARAM_ORDER},
        "residual": residual[yy, xx],
    }
    parquet_path = (
        out_dir("feature_stacks") / f"{cube_id}_stackB_15param_long.parquet"
    )
    pd.DataFrame(tidy).to_parquet(parquet_path, index=False, compression="snappy")
    print(f"  saved {parquet_path} ({len(yy)} rows)")

    log: dict = {
        "cube_id": cube_id,
        "stage": "B-0_build_stack",
        "run_timestamp_utc": timestamp,
        "random_seed": RANDOM_SEED,
        "brown_version": BROWN_VERSION,
        "strict_brown_mask": STRICT_BROWN_MASK,
        "img_path": str(img_path),
        "cube_shape_bands_lines_samples": list(cube.shape),
        "spatial_shape_lines_samples": list(shape),
        "wavelength_min_um": float(wav[0]),
        "wavelength_max_um": float(wav[-1]),
        "param_order_15": list(STAGE_B_PARAM_ORDER),
        "brown_params": list(STAGE_B_BROWN_PARAMS),
        "vb_required": list(STAGE_B_VB_REQUIRED),
        "vb_exploratory": list(STAGE_B_VB_EXPLORATORY),
        "n_pixels_total": n_total,
        "n_pixels_brown_mask_raw": int(brown_mask_raw.sum()),
        "n_pixels_finite_brown": int(finite_brown.sum()),
        "n_pixels_brown_valid": int(brown_valid.sum()),
        "n_pixels_vb_valid": int(vb_valid.sum()),
        "n_pixels_joint_valid_15": int(joint_valid_15.sum()),
        "joint_valid_15_fraction": round(joint_valid_15.sum() / n_total, 4),
        "nan_fraction_by_param": {
            name: round(nan_fraction(params[name]), 4)
            for name in STAGE_B_PARAM_ORDER
        },
        "bdi1000vis_loss_vs_baseline": round(float(bdi_loss), 4),
        "bdi1000vis_flag_over_5pct": bool(bdi_flag),
        "output_npz": str(npz_path),
        "output_parquet": str(parquet_path),
        "source_code_md5": {
            "stage_b_build_stack.py": md5_of(Path(__file__)),
            "vb_parameters.py": md5_of(Path(__file__).with_name("vb_parameters.py")),
            "crism_io.py": md5_of(Path(__file__).with_name("crism_io.py")),
            "config.py": md5_of(Path(__file__).with_name("config.py")),
        },
        "wall_clock_seconds": round(time.time() - t0, 2),
    }
    log_path = out_dir("logs") / f"{cube_id}_stageB_build_{timestamp}.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  saved log {log_path}")
    return log


def main(argv: list[str]) -> None:
    targets = argv[1:] if len(argv) > 1 else list(CUBE_IDS)
    unknown = [t for t in targets if t not in CUBE_IDS]
    if unknown:
        raise SystemExit(f"Unknown cube id(s): {unknown}; known: {CUBE_IDS}")
    logs = []
    for cube_id in targets:
        logs.append(process_cube(cube_id))

    # Concise cross-cube summary so the next scripts can sanity-check.
    print("\n=== summary ===")
    print(f"{'cube_id':15s}  {'n_joint15':>10s}  {'joint%':>7s}  {'BDI_loss%':>10s}  BDI_flag")
    for log in logs:
        n_joint = log["n_pixels_joint_valid_15"]
        frac = log["joint_valid_15_fraction"] * 100
        bdi_loss = log["bdi1000vis_loss_vs_baseline"] * 100
        flag = "**" if log["bdi1000vis_flag_over_5pct"] else ""
        print(f"{log['cube_id']:15s}  {n_joint:10d}  {frac:7.1f}  {bdi_loss:10.2f}  {flag}")


if __name__ == "__main__":
    main(sys.argv)
