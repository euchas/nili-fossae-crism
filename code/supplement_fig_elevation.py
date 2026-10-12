"""Supplement figure: contact-margin elevation profiles (stage-7 fringe).

Panels: (a) stacked median detrended profile with IQR for both HiRISE DTMs
(clean profiles), with the 002888 flat-olivine-edge subset overlaid - the
control where talus/caprock-shoulder explanations fail; (b) three example
flat-edge profiles from the clean DTM (deterministic pick: bump closest to
the flat-edge median). Data: stage7_fringe profile arrays + metrics_v2.
Deterministic.

Usage:  python code/supplement_fig_elevation.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from contact_distance_to_contact import md5  # noqa: E402

S7 = ROOT / "outputs" / "revision" / "contact_3d" / "stage7_fringe"
OUT = ROOT / "manuscript_v10" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

DTMS = {"DTEEC_002888_2025_002176_2025_A01": ("#0072B2", "HiRISE DTM 002888 (clean)"),
        "DTEEC_006712_2020_005855_2020_A01": ("#999999", "HiRISE DTM 006712")}
CLEAN_DTM = "DTEEC_002888_2025_002176_2025_A01"
NOISE_THR = 0.05
FLAT_STEP_M = 0.5
XLIM = (-60, 60)


def main() -> None:
    t0 = datetime.now(timezone.utc)
    met = pd.read_csv(S7 / "fringe_metrics_v2.csv")
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    picks = None
    for dtm, (col, lab) in DTMS.items():
        z = np.load(S7 / f"{dtm}_profiles.npz")
        X, resid = z["X"], z["resid"]
        md = met[met.dtm == dtm].set_index("idx")
        clean = md[md.noise_medlap_m >= NOISE_THR].index.to_numpy()  # low |laplacian| = TIN fill
        r = resid[clean]
        med = np.nanmedian(r, axis=0)
        q25, q75 = np.nanpercentile(r, 25, axis=0), np.nanpercentile(r, 75, axis=0)
        ax_a.plot(X, med, color=col, lw=2.0, label=f"{lab}, n={len(clean)}")
        ax_a.fill_between(X, q25, q75, color=col, alpha=0.15, linewidth=0)
        if dtm == CLEAN_DTM:
            flat = md[(md.noise_medlap_m >= NOISE_THR) & (md.step_m.abs() <= FLAT_STEP_M)]
            rf = resid[flat.index.to_numpy()]
            ax_a.plot(X, np.nanmedian(rf, axis=0), color=col, lw=1.8, ls="--",
                      label=f"002888 flat olivine edges, n={len(flat)}")
            fe_med = float(flat.bump_clay_max_m.median())
            picks = flat.iloc[(flat.bump_clay_max_m - fe_med).abs().argsort()[:3]]
            pick_resid = resid[picks.index.to_numpy()]
    ax_a.axvline(0, color="0.4", lw=0.9)
    ax_a.annotate("clay basement", (XLIM[0] + 3, ax_a.get_ylim()[1] * 0.0 + 1.9), fontsize=9, color="0.35")
    ax_a.annotate("olivine unit", (18, 1.9), fontsize=9, color="0.35")
    ax_a.set_xlim(XLIM)
    ax_a.set_xlabel("distance from mapped contact (m); clay side negative", fontsize=9)
    ax_a.set_ylabel("detrended elevation residual (m)", fontsize=9)
    ax_a.set_title("(a) stacked contact-normal profiles (median, IQR)", fontsize=10)
    ax_a.legend(fontsize=8, frameon=False)
    ax_a.grid(True, color="0.95"); ax_a.spines[["top", "right"]].set_visible(False)

    off = 0.0
    for (idx, row), r in zip(picks.iterrows(), pick_resid):
        ax_b.plot(np.load(S7 / f"{CLEAN_DTM}_profiles.npz")["X"], r + off, lw=1.4, color="#0072B2")
        ax_b.annotate(f"bump {row.bump_clay_max_m:.1f} m", (XLIM[1] - 2, off + 0.4),
                      fontsize=8, ha="right", color="0.35")
        off += 4.0
    ax_b.axvline(0, color="0.4", lw=0.9)
    ax_b.set_xlim(XLIM)
    ax_b.set_xlabel("distance from mapped contact (m); clay side negative", fontsize=9)
    ax_b.set_ylabel("residual (m, offset for clarity)", fontsize=9)
    ax_b.set_title("(b) example flat-edge profiles, DTM 002888", fontsize=10)
    ax_b.grid(True, color="0.95"); ax_b.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Raised clay-side margins at the olivine-basement contact (1 m HiRISE DTMs)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_S_elevation.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig_S_elevation.pdf", bbox_inches="tight")
    plt.close(fig)

    log = {"script": Path(__file__).name, "script_md5": md5(Path(__file__)),
           "started_utc": t0.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
           "noise_thr_medlap_m": NOISE_THR, "flat_step_m": FLAT_STEP_M,
           "example_picks_idx": [int(i) for i in picks.index]}
    (OUT / "run_log_fig_S_elevation.json").write_text(json.dumps(log, indent=2, default=str))
    print("done ->", OUT, "picks:", list(picks.index))


if __name__ == "__main__":
    main()
