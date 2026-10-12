"""Georegistration check: geologic-map contours over CRISM false-colour RGB.

For each cube, builds a FAL-style composite from the MTR3 I/F cube
(R 2.529, G 1.506, B 1.080 um; median of 3 channels each, 2-98 percent
stretch) and overlays the geologic-map unit contours from the
CTX-corrected fractions rasters (the same contours used in
contact_band_center_map.py). If AV's hand-mapped contacts follow the
terrain boundaries visible in the composite, the coregistration chain
(CaSSIS pansharp -> Robbins CTX -> measured CRISM shift) is validated at
the +-1-2 px residual level. Full scene plus a zoom on the clay contact.

See PLAN_revision_2300_zonation.md (Tier 0 follow-up). Deterministic.

Usage:  python code/contact_map_overlay_check.py
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import center_of_mass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from config import CUBE_IDS  # noqa: E402
from contact_distance_to_contact import md5  # noqa: E402
from figstyle import draw_units, label_panels  # noqa: E402

GEOMAP = ROOT / "outputs" / "revision" / "contact_3d" / "geomap" / "frame_ctx"
COREG = ROOT / "outputs" / "revision" / "contact_3d" / "frame" / "coreg_offsets.json"
OUT = ROOT / "outputs" / "revision" / "band_center_map" / "overlay_check"
OUT.mkdir(parents=True, exist_ok=True)

RGB_UM = (2.5295, 1.5066, 1.0800)   # FAL composite: R, G, B
STRETCH_PCT = (2.0, 98.0)
ZOOM_MAX_PX = 420


def read_rgb(cube: str) -> tuple[np.ndarray, Path]:
    low = cube.lower()
    hdr = ROOT / "cubes" / f"{low}_07_if166j_mtr3" / f"{low}_07_if166j_mtr3.hdr"
    txt = hdr.read_text()
    get = lambda k: int(re.search(rf"^{k}\s*=\s*(\d+)", txt, re.M | re.I).group(1))
    ns, nl, nb = get("samples"), get("lines"), get("bands")
    wav = np.array([float(x) for x in re.search(r"wavelength\s*=\s*\{([^}]*)\}", txt, re.S | re.I).group(1).split(",")])
    wav_um = wav / 1000.0 if wav.max() > 100 else wav
    mm = np.memmap(hdr.with_suffix(".img"), dtype="<f4", mode="r", shape=(nb, nl, ns))
    chans = []
    for target in RGB_UM:
        i = int(np.argmin(np.abs(wav_um - target)))
        sl = np.array(mm[max(i - 1, 0):i + 2]).astype(np.float32)
        sl[sl >= 65535.0] = np.nan
        chans.append(np.nanmedian(sl, axis=0))
    rgb = np.stack(chans, axis=-1)                      # (nl, ns, 3)
    lo = np.nanpercentile(rgb, STRETCH_PCT[0], axis=(0, 1))
    hi = np.nanpercentile(rgb, STRETCH_PCT[1], axis=(0, 1))
    rgb = np.clip((rgb - lo) / (hi - lo), 0, 1)
    rgb[~np.isfinite(rgb)] = 0.0
    return rgb, hdr.with_suffix(".img")


def draw_contours(ax, fr: dict) -> None:
    draw_units(ax, fr, units=("clay", "cap"))
    ax.contour((fr["outline"] >= 0.99).astype(float), levels=[0.5], colors="#ffd700",
               linewidths=0.9, linestyles="dotted")


def zoom_box(fr: dict) -> tuple[slice, slice] | None:
    sel = (fr["clay"] >= 0.5) & (fr["outline"] >= 0.99)
    if not sel.any():
        return None
    r, c = center_of_mass(sel)
    rr, cc = np.nonzero(sel)
    side = min(max(int(np.ptp(rr)), int(np.ptp(cc))) + 60, ZOOM_MAX_PX)
    H, W = sel.shape
    r0 = int(np.clip(r - side / 2, 0, H - side)); c0 = int(np.clip(c - side / 2, 0, W - side))
    return slice(r0, r0 + side), slice(c0, c0 + side)


def main() -> None:
    t0 = datetime.now(timezone.utc)
    shifts = json.loads(COREG.read_text()) if COREG.exists() else {}
    inputs = {}
    for cube in CUBE_IDS:
        rgb, img = read_rgb(cube)
        fz = np.load(GEOMAP / f"{cube}_fractions.npz")
        fr = {k: fz[k] for k in ("clay", "cap", "oliv", "outline")}
        sh = {k: float(fz[k]) for k in ("shift_dx_m", "shift_dy_m") if k in fz.files}
        inputs[cube] = {"mtr3_img": str(img.relative_to(ROOT)), "mtr3_size_bytes": img.stat().st_size,
                       "fractions_md5": md5(GEOMAP / f"{cube}_fractions.npz"), "applied_shift_m": sh}
        zb = zoom_box(fr)
        ncols = 2 if zb else 1
        fig, axes = plt.subplots(1, ncols, figsize=(7.2 * ncols, 7.2))
        axes = np.atleast_1d(axes)
        axes[0].imshow(rgb, interpolation="nearest")
        draw_contours(axes[0], fr)
        axes[0].set_title(f"{cube} full scene", fontsize=10)
        if zb:
            rs, cs = zb
            axes[1].imshow(rgb[rs, cs], interpolation="nearest")
            frz = {k: v[rs, cs] for k, v in fr.items()}
            draw_contours(axes[1], frz)
            axes[1].set_title(f"zoom on clay contact ({rs.stop - rs.start} px "
                              f"= {(rs.stop - rs.start) * 18 / 1000:.1f} km)", fontsize=10)
            from matplotlib.patches import Rectangle
            axes[0].add_patch(Rectangle((cs.start, rs.start), cs.stop - cs.start, rs.stop - rs.start,
                                        fill=False, edgecolor="#ff4444", lw=1.2))
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        shift_txt = (f"applied coreg shift: dx={sh.get('shift_dx_m', float('nan')):+.0f} m, "
                     f"dy={sh.get('shift_dy_m', float('nan')):+.0f} m" if sh else
                     "applied coreg shift: see coreg_offsets.json")
        fig.suptitle(f"{cube}: geologic-map contours on CRISM FAL composite "
                     f"(R 2.53 / G 1.51 / B 1.08 um)\n"
                     f"blue = phyllosilicate basement, dashed = mafic cap, gold dotted = mapped-area outline; "
                     f"{shift_txt}", fontsize=10)
        fig.tight_layout()
        label_panels(fig)
        fig.savefig(OUT / f"fig_overlay_{cube}.png", dpi=200, bbox_inches="tight")
        fig.savefig(OUT / f"fig_overlay_{cube}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(cube, "done", sh)

    log = {"script": Path(__file__).name, "script_md5": md5(Path(__file__)),
           "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
           "rgb_um": RGB_UM, "stretch_pct": STRETCH_PCT, "coreg_offsets": shifts,
           "deterministic": True, "inputs": inputs}
    (OUT / "run_log_overlay_check.json").write_text(json.dumps(log, indent=2, default=str))
    print("done ->", OUT)


if __name__ == "__main__":
    main()
