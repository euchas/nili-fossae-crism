"""Stage v3 - Section 7: cross-cube lobe spectrum overlays.

Reads the same PCA + lobe definitions as the per-cube driver, extracts
mafic-bulk and smectite-equivalent lobe mean spectra from each cube's
TER3 file, and overlays them on a single figure each.

File: cross_cube_lobe_comparison_v3.png
Plus separate cross_cube_mafic_v3.png / cross_cube_smectite_v3.png
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import CUBE_IDS, RANDOM_SEED, out_dir
from crism_io import load_cube_by_id
from stage_v3_common import (
    LOBE_DEFS, MAFIC_LOBE, SMECTITE_LOBE,
    annotate_spectrum, figpath, lobe_mask_pca, load_cube_artifacts, md5_of,
    v3_dark_style,
)

CUBE_COLOURS = {
    "FRT00003E12": "#1f77b4",
    "FRT0000871C": "#d62728",
    "FRT0000A4FC": "#2ca02c",
    "FRT0000B438": "#9467bd",
}


def extract_mean_spectrum(cube_art: dict, cube_data: np.ndarray, lobe_label: str):
    yy = cube_art["pca"]["line"]
    xx = cube_art["pca"]["sample"]
    lobe = LOBE_DEFS[cube_art["cube_id"]][lobe_label]
    mask = lobe_mask_pca(cube_art, lobe)
    n = int(mask.sum())
    if n < 5:
        return None, n, lobe["label"]
    specs = cube_data[:, yy[mask], xx[mask]]
    return np.nanmean(specs, axis=1), n, lobe["label"]


def main() -> None:
    t0 = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    bundles = {}
    for cube_id in CUBE_IDS:
        print(f"loading {cube_id}...")
        cube_art = load_cube_artifacts(cube_id)
        cube_data, _, _, _ = load_cube_by_id(cube_id)
        wav = cube_art["wav"]
        mafic_spec, mafic_n, mafic_label = extract_mean_spectrum(
            cube_art, cube_data, MAFIC_LOBE[cube_id])
        smec_spec_v, smec_n_v, smec_label_v = extract_mean_spectrum(
            cube_art, cube_data, SMECTITE_LOBE[cube_id])
        bundles[cube_id] = {
            "wav": wav,
            "mafic_spec": mafic_spec, "mafic_n": mafic_n, "mafic_label": mafic_label,
            "smectite_spec": smec_spec_v, "smectite_n": smec_n_v, "smectite_label": smec_label_v,
        }
        del cube_data  # free memory

    # Combined comparison figure
    with v3_dark_style():
        fig, axes = plt.subplots(2, 1, figsize=(11, 10), constrained_layout=True,
                                 sharex=True)
        for cube_id, b in bundles.items():
            colour = CUBE_COLOURS[cube_id]
            if b["mafic_spec"] is not None:
                axes[0].plot(b["wav"], b["mafic_spec"], lw=1.4, color=colour,
                             label=f"{cube_id} (n={b['mafic_n']}) - {b['mafic_label']}")
            if b["smectite_spec"] is not None:
                axes[1].plot(b["wav"], b["smectite_spec"], lw=1.4, color=colour,
                             label=f"{cube_id} (n={b['smectite_n']}) - {b['smectite_label']}")
        axes[0].set_title("Mafic-bulk lobe (Lobe A) - mean I/F per cube", fontsize=11)
        axes[1].set_title("Smectite-equivalent lobe - mean I/F per cube", fontsize=11)
        for ax in axes:
            ax.set_xlim(0.4, 2.6)
            annotate_spectrum(ax)
            ax.set_ylabel("I/F (TER3, atmospherically corrected)")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8, loc="upper right")
        axes[1].set_xlabel(r"wavelength ($\mu$m)")
        fig.suptitle("Cross-cube lobe-mean spectrum comparison", fontsize=13)
        fig.savefig(figpath("cross_cube_lobe_comparison_v3"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Separate figures (mafic only, smectite only) - useful for slides
    for tag, ylabel in (("mafic", "mafic-bulk lobe (Lobe A)"),
                        ("smectite", "smectite-equivalent lobe")):
        with v3_dark_style():
            fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
            for cube_id, b in bundles.items():
                spec = b[f"{tag}_spec"]
                n = b[f"{tag}_n"]
                lbl = b[f"{tag}_label"]
                if spec is None:
                    continue
                ax.plot(b["wav"], spec, lw=1.4, color=CUBE_COLOURS[cube_id],
                        label=f"{cube_id} (n={n}) - {lbl}")
            ax.set_xlim(0.4, 2.6)
            annotate_spectrum(ax)
            ax.set_xlabel(r"wavelength ($\mu$m)")
            ax.set_ylabel(f"I/F ({ylabel})")
            ax.set_title(f"Cross-cube {ylabel} comparison", fontsize=11)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(alpha=0.2)
            fig.savefig(figpath(f"cross_cube_{tag}_v3"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

    log = {
        "stage": "v3_cross_cube",
        "run_timestamp_utc": timestamp,
        "random_seed": RANDOM_SEED,
        "cubes": list(CUBE_IDS),
        "mafic_lobe_per_cube": dict(MAFIC_LOBE),
        "smectite_lobe_per_cube": dict(SMECTITE_LOBE),
        "wall_clock_seconds": round(time.time() - t0, 2),
        "source_code_md5": {
            "stage_v3_cross_cube.py": md5_of(Path(__file__)),
            "stage_v3_common.py": md5_of(Path(__file__).with_name("stage_v3_common.py")),
            "config.py": md5_of(Path(__file__).with_name("config.py")),
        },
    }
    log_path = out_dir("logs") / f"v3_cross_cube_{timestamp}.json"
    log_path.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
    print(f"done in {log['wall_clock_seconds']}s -> {log_path.name}")


if __name__ == "__main__":
    main()
