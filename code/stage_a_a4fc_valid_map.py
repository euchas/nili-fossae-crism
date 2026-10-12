"""Supplemental Stage A figure: where are FRT0000A4FC's valid pixels?

AV's review of the 01_stageA_validation.md report asked for a spatial
map of the A4FC joint_valid mask so he can tell whether the 8.3 %
surviving pixels form a coherent blob at the edge of an olivine unit
(useful) or a scattered pattern (suspicious).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from config import out_dir
from display import apply_display_transform

CUBE_ID = "FRT0000A4FC"


def main() -> None:
    npz = out_dir("feature_stacks") / f"{CUBE_ID}_stackA_8param.npz"
    if not npz.exists():
        raise SystemExit(f"Missing {npz}; run stage_a_build_feature_stack.py first.")
    data = np.load(npz, allow_pickle=True)
    brown_mask = data["brown_mask"].astype(bool)
    joint = data["joint_valid"].astype(bool)
    olindex3 = data["OLINDEX3"].astype(float)

    n_total = brown_mask.size
    n_brown = int(brown_mask.sum())
    n_joint = int(joint.sum())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    axes[0].imshow(
        apply_display_transform(brown_mask),
        cmap="gray_r", origin="upper", interpolation="nearest", aspect="equal",
    )
    axes[0].set_title(
        f"Brown v5 mask ({n_brown:,} px, {100*n_brown/n_total:.1f}% of frame)"
    )
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].imshow(
        apply_display_transform(joint),
        cmap="gray_r", origin="upper", interpolation="nearest", aspect="equal",
    )
    axes[1].set_title(
        f"joint_valid ({n_joint:,} px, {100*n_joint/n_total:.1f}% of frame)"
    )
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # OLINDEX3 stretched 2-98 % of joint_valid pixels, with basement shaded grey
    oli = olindex3.copy()
    oli[~joint] = np.nan
    finite = oli[np.isfinite(oli)]
    if finite.size:
        vmin, vmax = np.quantile(finite, [0.02, 0.98])
    else:
        vmin, vmax = 0, 0.3
    bg = np.where(brown_mask & ~joint, 0.5, np.nan)
    axes[2].imshow(
        apply_display_transform(bg),
        cmap="Greys", vmin=0, vmax=1, origin="upper",
        aspect="equal", interpolation="nearest",
    )
    im = axes[2].imshow(
        apply_display_transform(oli),
        cmap="viridis", vmin=vmin, vmax=vmax, origin="upper",
        aspect="equal", interpolation="nearest",
    )
    plt.colorbar(im, ax=axes[2], shrink=0.75, label="OLINDEX3")
    axes[2].set_title("OLINDEX3 over joint_valid (grey = brown-mask only)")
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    fig.suptitle(
        f"{CUBE_ID} — valid-pixel footprint diagnostic (Stage A supplement)"
    )
    stem = out_dir("figures") / f"{CUBE_ID}_stackA_validmap"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {stem}.png")

    # Summary stats for the report append.
    yy, xx = np.where(joint)
    if yy.size:
        y_span = (int(yy.min()), int(yy.max()))
        x_span = (int(xx.min()), int(xx.max()))
        bbox_area = (y_span[1] - y_span[0] + 1) * (x_span[1] - x_span[0] + 1)
        fill = n_joint / bbox_area if bbox_area else 0.0
        print(f"joint_valid bounding box: lines {y_span}, samples {x_span}  "
              f"-> bbox_area={bbox_area}, fill={100*fill:.1f}%")


if __name__ == "__main__":
    main()
