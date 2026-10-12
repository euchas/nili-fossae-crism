"""Frame check for Adriano's two DEMs: DRG orthos (decimated to 6 m) vs the
Robbins controlled CTX mosaic, same method as Stage 0. Writes
outputs/revision/contact_3d/frame/new_dem_offsets.json consumed by
contact_dtm_registry (shift applied only if > 0.5 px and confident).

Usage: python code/contact_new_dem_coreg.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from contact_coreg_frame import ctx_on_grid, qc_figure, register  # noqa: E402
from contact_dtm_registry import registry  # noqa: E402

OUTD = ROOT / "outputs" / "revision" / "contact_3d" / "frame"


def main() -> None:
    t0 = datetime.now(timezone.utc)
    out = {}
    for name in ("ADI_006712", "ADI_027691"):
        e = registry()[name]
        e = dict(e); e["shift"] = (0.0, 0.0)          # measure against the raw (worldfile/file) georef
        from contact_dtm_registry import open_meta
        path, tr0, crs, nod = open_meta(e, "ortho")
        with rasterio.open(path) as ds:
            f = max(1, int(round(6.0 / abs(tr0.a))))
            img = ds.read(1, out_shape=(ds.height // f, ds.width // f), resampling=Resampling.average).astype(np.float32)
        img[(img == nod) | (img <= 0) | ~np.isfinite(img)] = np.nan
        tr = tr0 * Affine.scale(f)
        valid = np.isfinite(img)
        ctx, ctxv = ctx_on_grid(crs, tr, img.shape)
        fit, tiles, rh, mh = register(ctx, ctxv, img, valid, abs(tr.a), tile=256, stride=128,
                                      label=f"{name} DRG (6 m) vs CTX", coarse_max_dim=2048)
        qc_figure(rh, mh, tiles, fit, fit["label"], OUTD / f"{name}_qc.png")
        rec = {"ok": bool(fit.get("ok")), "label": fit.get("label")}
        if fit.get("ok"):
            px = float(np.hypot(fit["drow"], fit["dcol"]))
            rec.update({"dx_m": fit["dx_m"], "dy_m": fit["dy_m"], "dx_mad_m": fit["dx_mad_m"], "dy_mad_m": fit["dy_mad_m"],
                        "n_good": fit["n_good"], "conf_median": fit["conf_median"], "offset_px": px,
                        "applied": bool(px >= 0.5 and fit["conf_median"] >= 0.3)})
            if not rec["applied"]:
                rec["dx_m"] = rec["dy_m"] = 0.0
        out[name] = rec
        print(name, rec)
    out["note"] = "shifts are corrections applied on top of the worldfile/file georef; measured vs Robbins CTX"
    out["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (OUTD / "new_dem_offsets.json").write_text(json.dumps(out, indent=2))
    print("wrote", OUTD / "new_dem_offsets.json")


if __name__ == "__main__":
    main()
