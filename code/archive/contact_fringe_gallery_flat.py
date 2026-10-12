"""Gallery of clay-side fringe candidates at FLAT-EDGE contacts.

The first gallery ranked by raw bump amplitude and surfaced large scarps.
This one selects the population that actually tests the "pushed rug"
hypothesis: contacts where the olivine edge itself has little relief
(|step| <= 0.5 m) yet the clay side still rolls up, ranked by
bump / far-field rms, bump amplitude capped at 1-4 m.

Usage: python code/contact_fringe_gallery_flat.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from contact_contact_fringe import HIRISE, X  # noqa: E402

OUT = ROOT / "outputs" / "revision" / "contact_3d" / "stage7_fringe"
FRAME = ROOT / "outputs" / "revision" / "contact_3d" / "frame" / "coreg_offsets.json"
N_SHOW = 10


def hillshade(z: np.ndarray, res: float) -> np.ndarray:
    ls = LightSource(azdeg=300, altdeg=30)
    zz = np.where(np.isfinite(z), z, np.nanmedian(z))
    return ls.hillshade(zz, vert_exag=2.0, dx=res, dy=res)


def main() -> None:
    offsets = json.loads(FRAME.read_text())["hirise"]
    d = pd.read_csv(OUT / "fringe_metrics.csv")
    for dtm, path in HIRISE.items():
        s = np.load(OUT / f"{dtm}_profiles.npz")
        g = d[(d.dtm == dtm)].dropna(subset=["bump_clay_max_m", "far_rms_clay_m", "step_m"])
        g = g[(g.step_m.abs() <= 0.5) & g.bump_clay_max_m.between(1.0, 4.0) & (g.far_rms_clay_m > 0)]
        g = g.assign(snr=g.bump_clay_max_m / g.far_rms_clay_m).sort_values("snr", ascending=False)
        # keep spatially distinct sites (>= 150 m apart)
        keep = []
        for _, r in g.iterrows():
            if all(np.hypot(r.x_m - d2.x_m, r.y_m - d2.y_m) > 150 for d2 in keep):
                keep.append(r)
            if len(keep) >= N_SHOW:
                break
        if not keep:
            print(dtm, "no flat-edge candidates"); continue
        o = offsets[dtm]["measured"]
        with rasterio.open(path) as ds:
            tr1 = Affine.translation(o["dx_m"], o["dy_m"]) * ds.transform
            fig, axes = plt.subplots(len(keep), 2, figsize=(11, 2.9 * len(keep)), squeeze=False)
            for r, (axm, axp) in zip(keep, axes):
                inv = ~tr1
                c0, r0 = inv * (r.x_m - 120, r.y_m + 120); c1, r1 = inv * (r.x_m + 120, r.y_m - 120)
                win = rasterio.windows.Window(int(c0), int(r0), int(c1 - c0), int(r1 - r0))
                zz = ds.read(1, window=win, boundless=True, fill_value=np.nan).astype(np.float32)
                zz[(zz < -1e30) | (zz == ds.nodata)] = np.nan
                axm.imshow(hillshade(zz, abs(tr1.a)), cmap="gray")
                axm.plot(zz.shape[1] / 2, zz.shape[0] / 2, "o", mfc="none", mec="#D55E00", ms=10, mew=1.5)
                axm.set_xticks([]); axm.set_yticks([])
                axm.set_title(f"({r.x_m:.0f}, {r.y_m:.0f})  bump {r.bump_clay_max_m:.1f} m at x={r.bump_clay_x_m:.0f} m; step {r.step_m:+.1f} m; snr {r.snr:.1f}", fontsize=8)
                prof = s["profs"][int(r.idx)]
                rel = prof - np.nanmedian(prof[(np.abs(X) <= 2)])
                axp.plot(X, rel, color="#0072B2", lw=1.5)
                axp.axvline(0, color="0.4", lw=0.8); axp.axvline(r.bump_clay_x_m, color="#D55E00", lw=0.8, ls="--")
                axp.set_xlabel("m from contact (clay < 0)"); axp.grid(True, color="0.93")
                axp.set_ylim(-4, 4)
            fig.suptitle(f"{dtm}: clay-side fringes at FLAT olivine edges (|step| <= 0.5 m), ranked by bump/rms", fontsize=10)
            fig.tight_layout(); fig.savefig(OUT / f"fig_fringe_flatedge_{dtm}.png", dpi=140); plt.close(fig)
            print(dtm, "flat-edge gallery:", len(keep), "sites")


if __name__ == "__main__":
    main()
