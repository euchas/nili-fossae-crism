"""Test: is the 1.4 um 'depth' the olivine 1 um band wing, or hydration?

Correlates the MTR3 restricted-window 1.4 um band depth with OLINDEX3
(olivine 1 um band strength) and with the hydration parameters D2300 /
BD1900R2, within map-pure olivine and map-pure clay pixels per cube.
Wing hypothesis: depth tracks OLINDEX3 in the olivine unit; hydration
hypothesis: depth tracks D2300/BD1900R2. Descriptive Spearman rho
(pixel-level, autocorrelated - no p-values quoted). Deterministic.

Usage:  python code/contact_band1400_wing_check.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import CUBE_IDS  # noqa: E402
from contact_distance_to_contact import V8_PARAMS, load_cube, masks_from_labels, md5  # noqa: E402
from contact_band_center_map import unit_masks  # noqa: E402

B14 = ROOT / "outputs" / "revision" / "band_center_map" / "band1400"
PARAMS = ("OLINDEX3", "D2300", "BD1900R2", "centroid")


def main() -> None:
    t0 = datetime.now(timezone.utc)
    rows = []
    for cube in CUBE_IDS:
        c = load_cube(cube)
        m = masks_from_labels(c["labels"])
        units, d, fr = unit_masks(cube, m)
        dep = np.load(B14 / f"{cube}_bc140_depth.npy")
        for uname in ("olivine map-pure", "clay map-pure"):
            um = units[uname] & np.isfinite(dep)
            dv = dep[um]
            row = {"cube": cube, "unit": uname, "n": int(dv.size)}
            for p in PARAMS:
                pv = c["params"][..., V8_PARAMS.index(p)][um]
                ok = np.isfinite(pv)
                row[f"rho_{p}"] = float(spearmanr(dv[ok], pv[ok]).statistic) if ok.sum() > 100 else np.nan
            rows.append(row)
        print(cube, "done")
    df = pd.DataFrame(rows)
    df.to_csv(B14 / "wing_check.csv", index=False)
    print(df.round(3).to_string())
    log = {"script": Path(__file__).name, "script_md5": md5(Path(__file__)),
           "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
           "note": "descriptive pixel-level Spearman; no p-values (autocorrelation)"}
    (B14 / "run_log_wing_check.json").write_text(json.dumps(log, indent=2, default=str))


if __name__ == "__main__":
    main()
