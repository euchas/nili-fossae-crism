"""Robustness check: clay-contact hydration gradient with a 90-150 m near band.

Verifies the response-letter claim that the block-level near/far contrast
does not depend on the registration-limited 36-90 m bins: recomputes the
block-bootstrap near-minus-far difference for D2300 and BD1900R2 in the
two conclusive cubes with near = (90, 150] m instead of (36.1, 150] m.

Usage:  python code/contact_robustness_90m.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from contact_distance_to_contact import V8_PARAMS, load_cube, md5  # noqa: E402
from contact_block_stats import block_boot_diff, block_ids, cube_setup  # noqa: E402

OUT = ROOT / "manuscript_v10" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
CUBES = ("FRT00003E12", "FRT0000B438")
NEAR90 = (90.0, 150.0)
FAR_MIN = 400.0
METRICS = ("D2300", "BD1900R2")


def main() -> None:
    t0 = datetime.now(timezone.utc)
    rows = []
    for cube in CUBES:
        c = load_cube(cube)
        m, fr, tr, d, pure = cube_setup(cube, c)
        sel = pure & (d["cap"] > 90)
        rows_, cols_ = np.nonzero(sel)
        dist = d["clay"][sel]
        bid = block_ids(rows_, cols_)
        for p in METRICS:
            v = c["params"][..., V8_PARAMS.index(p)][sel]
            ok = np.isfinite(v)
            near = ok & (dist > NEAR90[0]) & (dist <= NEAR90[1])
            far = ok & (dist > FAR_MIN)
            r = block_boot_diff(v[near], bid[near], v[far], bid[far])
            r.update({"cube": cube, "param": p, "near_band_m": f"{NEAR90[0]}-{NEAR90[1]}",
                      "n_near": int(near.sum()), "n_far": int(far.sum()),
                      "near_med": float(np.median(v[near])), "far_med": float(np.median(v[far]))})
            rows.append(r)
        print(cube, "done")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "robustness_near90.csv", index=False)
    print(df.round(5).to_string())
    log = {"script": Path(__file__).name, "script_md5": md5(Path(__file__)),
           "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
           "near_band_m": NEAR90, "far_min_m": FAR_MIN, "seed_note": "block bootstrap RNG seed 42 (module)"}
    (OUT / "run_log_robustness_near90.json").write_text(json.dumps(log, indent=2, default=str))


if __name__ == "__main__":
    main()
