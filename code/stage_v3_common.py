"""Stage v3 - shared helpers, styling, lobe definitions, PC interpretations.

Used by stage_v3_per_cube.py and stage_v3_cross_cube.py.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from config import STAGE_B_PARAM_ORDER, out_dir
from display import apply_display_transform


# Per-cube PC3-5 interpretations from AV's v3 spec.
# PC1 = olivine, PC2 = Fe/Mg smectite -- universal across all four cubes.
PC_INTERPRETATIONS: dict[str, dict[int, str]] = {
    "FRT00003E12": {
        1: "olivine (universal)",
        2: "Fe/Mg smectite (universal)",
        3: "ferric-rich vs carbonate-rich axis",
        4: "ferric + carbonate co-association",
        5: "crystalline vs nanophase ferric",
    },
    "FRT0000871C": {
        1: "olivine (universal)",
        2: "Fe/Mg smectite (universal)",
        3: "carbonate / ferric mixed",
        4: "redox + carbonate",
        5: "crystalline ferric",
    },
    "FRT0000A4FC": {
        1: "olivine (universal)",
        2: "Fe/Mg smectite (universal)",
        3: "Fe3+/Fe2+ redox",
        4: "olivine band-width axis (NOT carbonate)",
        5: "Mg-carbonate",
    },
    "FRT0000B438": {
        1: "olivine (universal)",
        2: "Fe/Mg smectite (universal)",
        3: "Fe3+/Fe2+ redox",
        4: "Mg-carbonate (clean)",
        5: "crystalline ferric",
    },
}

# Targeted PC subsets for Section 4c.
TARGETED_GMM: dict[str, dict] = {
    "FRT00003E12": {"pcs": [3, 4, 5], "k": 5,
                     "rationale": "ferric-carbonate coupling generates intermediate classes"},
    "FRT0000871C": {"pcs": [3, 4, 5], "k": 4,
                    "rationale": "bulk + carbonate + ferric + redox"},
    "FRT0000A4FC": {"pcs": [3, 5], "k": 4,
                    "rationale": "PC4 is band-width, not compositional - skip it"},
    "FRT0000B438": {"pcs": [3, 4, 5], "k": 4,
                    "rationale": "bulk + redox + carbonate + ferric"},
}

# Starting lobe definitions from AV's v3 spec.
# Each entry: {lobe_name: {"pcs": (i, j), "x": (lo, hi)|None, "y": (lo, hi)|None,
#                          "x_op": ">"|"<"|None, "y_op": ">"|"<"|None, "label": str}}
# When x/y is a tuple, both bounds apply; when x_op/y_op is "<" or ">",
# only one-sided constraint applies. None means no constraint on that axis.
# A lobe with pcs=(i, j) is defined in PC{i} (x) vs PC{j} (y) space.
LOBE_DEFS: dict[str, dict[str, dict]] = {
    "FRT0000B438": {
        "A": {"pcs": (1, 2), "x": (3.0, 7.0), "y": (-2.0, 2.0), "label": "olivine bulk"},
        "B": {"pcs": (1, 2), "x": (-4.0, -1.0), "y": (-2.0, 2.0), "label": "altered/smectite bulk"},
        "C": {"pcs": (1, 2), "x": (-1.0, 2.0), "y": (6.0, None), "label": "high-smectite tail"},
        "D": {"pcs": (1, 2), "x": (4.0, None), "y": (None, -3.0), "label": "redox/crystalline Fe tail"},
    },
    "FRT0000A4FC": {
        "A": {"pcs": (1, 2), "x": (-1.5, 1.0), "y": (-2.5, 0.0), "label": "mafic bulk"},
        "B": {"pcs": (1, 2), "x": (-2.0, 1.0), "y": (7.0, None), "label": "smectite tail"},
        "C": {"pcs": (1, 2), "x": (8.0, None), "y": (-2.0, 1.5), "label": "right-arm tip (olivine endmember)"},
        # D defined on PC3 vs PC5 (PC4 is band-width on A4FC). Use top quartile of PC5
        # (PC5 = Mg-carbonate on A4FC) AND positive PC3 (redox-altered).
        "D": {"pcs": (3, 5), "x": (0.5, None), "y": (1.5, None), "label": "carbonate candidate (PC3+ AND PC5+)"},
    },
    "FRT0000871C": {
        "A": {"pcs": (1, 2), "x": (-1.5, 1.5), "y": (-2.5, 0.0), "label": "mafic bulk"},
        "B": {"pcs": (1, 2), "x": (-3.0, 1.0), "y": (8.0, None), "label": "smectite spine"},
        "C": {"pcs": (1, 2), "x": (8.0, None), "y": (-2.0, 2.0), "label": "right-arm tip"},
        # D on PC3 vs PC5 -- PC3 = carbonate/ferric mixed, PC5 = crystalline ferric.
        # Carbonate candidate = high PC3, low PC5 (carbonate w/o crystalline ferric).
        "D": {"pcs": (3, 5), "x": (1.5, None), "y": (None, -0.5), "label": "carbonate candidate (PC3+ AND PC5-)"},
    },
    "FRT00003E12": {
        "A":  {"pcs": (1, 2), "x": (1.5, 4.0), "y": (-1.0, 1.0), "label": "mafic #1"},
        "Aprime": {"pcs": (1, 2), "x": (5.0, 7.0), "y": (-1.0, 2.0), "label": "mafic #2"},
        "B":  {"pcs": (1, 2), "x": (-3.5, -1.0), "y": (-3.0, -1.0), "label": "altered #1"},
        "C":  {"pcs": (1, 2), "x": (-4.5, -2.0), "y": (2.0, 4.0), "label": "altered #2"},
        "D":  {"pcs": (1, 2), "x": (-1.0, 2.0), "y": (8.0, None), "label": "smectite tail"},
        # E on PC3 vs PC4 -- PC3 = ferric-vs-carbonate, PC4 = ferric+carbonate co-assoc.
        # Co-association candidate = both PC3 and PC4 elevated.
        "E":  {"pcs": (3, 4), "x": (1.5, None), "y": (1.5, None), "label": "ferric-carbonate co-association candidate"},
    },
}

# Mafic-bulk lobe label per cube (used as ratio denominator in Section 6).
MAFIC_LOBE = {
    "FRT00003E12": "A",
    "FRT0000871C": "A",
    "FRT0000A4FC": "A",
    "FRT0000B438": "A",
}

# Smectite lobe label per cube (cross-cube comparison).
SMECTITE_LOBE = {
    "FRT00003E12": "D",
    "FRT0000871C": "B",
    "FRT0000A4FC": "B",
    "FRT0000B438": "C",
}

# Spectral feature wavelengths (um) to annotate.
ANNOTATION_BANDS = {
    1.00: "1.0 (olivine/pyx)",
    1.40: "1.4 (OH/H2O)",
    1.90: "1.9 (H2O)",
    2.30: "2.3 (Fe-Mg-OH)",
    2.50: "2.5 (carbonate)",
}
CO2_RESIDUAL = (1.95, 2.05)  # shaded as "residual CO2" caveat


PARAM_INDEX = {p: i for i, p in enumerate(STAGE_B_PARAM_ORDER)}


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


@contextmanager
def v3_dark_style():
    """Apply AV's v3 styling: white axes on dark background."""
    with plt.style.context("dark_background"):
        with mpl.rc_context({
            "axes.edgecolor": "white",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "axes.titlecolor": "white",
            "figure.facecolor": "#0a0a0a",
            "axes.facecolor": "#101010",
            "savefig.facecolor": "#0a0a0a",
            "image.cmap": "viridis",
        }):
            yield


def load_cube_artifacts(cube_id: str) -> dict:
    pca = dict(np.load(out_dir("embeddings") / f"{cube_id}_pca_15param.npz", allow_pickle=True))
    fs = dict(np.load(out_dir("feature_stacks") / f"{cube_id}_stackB_15param.npz", allow_pickle=True))
    gmm_labels = np.load(out_dir("cluster_labels") / f"{cube_id}_gmm_pca15_k6_fixed.npy")
    return {
        "cube_id": cube_id,
        "pca": pca,                    # 'transformed', 'components', 'explained_variance_ratio', 'line', 'sample', 'param_names'
        "feature_stack": fs,           # joint_valid_15, wavelengths_um, all 15 params, residual
        "gmm_labels": gmm_labels,      # (N,) int16
        "valid": fs["joint_valid_15"].astype(bool),
        "shape": fs["joint_valid_15"].shape,
        "wav": fs["wavelengths_um"].astype(np.float32),
    }


def backproject(scores_1d: np.ndarray, valid: np.ndarray, fill=np.nan) -> np.ndarray:
    """Place a length-N vector back into the (H, W) frame using the valid mask."""
    img = np.full(valid.shape, fill, dtype=np.float32)
    img[valid] = scores_1d.astype(np.float32)
    return img


def percentile_stretch(arr: np.ndarray, lo: float = 2.0, hi: float = 98.0):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return -1.0, 1.0
    return float(np.percentile(finite, lo)), float(np.percentile(finite, hi))


def percentile_norm(arr: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Stretch to [0, 1] using lo/hi percentiles. NaN stays NaN."""
    out = np.full_like(arr, np.nan, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return out
    a, b = np.percentile(arr[finite], [lo, hi])
    if b - a < 1e-9:
        out[finite] = 0.5
        return out
    out[finite] = np.clip((arr[finite] - a) / (b - a), 0, 1)
    return out


def annotate_spectrum(ax, fontsize: int = 7) -> None:
    """Add vertical lines at standard absorption wavelengths + shade CO2 residual."""
    ax.axvspan(CO2_RESIDUAL[0], CO2_RESIDUAL[1], color="#666666", alpha=0.18,
               zorder=0, label="residual CO2")
    for wl, lbl in ANNOTATION_BANDS.items():
        ax.axvline(wl, color="#aa8888", lw=0.6, ls=":", alpha=0.7, zorder=0)
        ax.annotate(
            lbl, xy=(wl, 1.0), xycoords=("data", "axes fraction"),
            xytext=(0, -8), textcoords="offset points",
            ha="center", va="top", fontsize=fontsize, color="#cccccc",
            rotation=90,
        )


def lobe_mask_pca(cube_art: dict, lobe_def: dict) -> np.ndarray:
    """Return a boolean mask of shape (N_pca,) for pixels inside the lobe.

    A lobe is defined by axis bounds in PC{i} vs PC{j} space.
    """
    pcs = lobe_def["pcs"]
    transformed = cube_art["pca"]["transformed"]
    x = transformed[:, pcs[0] - 1]
    y = transformed[:, pcs[1] - 1]
    mask = np.ones_like(x, dtype=bool)
    for axis, val in (("x", x), ("y", y)):
        bounds = lobe_def.get(axis)
        if bounds is None:
            continue
        lo, hi = bounds
        if lo is not None:
            mask &= val >= lo
        if hi is not None:
            mask &= val <= hi
    return mask


def lobe_color(lobe_name: str) -> str:
    palette = {
        "A": "#1f77b4", "Aprime": "#aec7e8",
        "B": "#d62728", "C": "#2ca02c",
        "D": "#ff7f0e", "E": "#9467bd",
    }
    return palette.get(lobe_name, "#888888")


def figpath(name: str, suffix: str = ".png") -> Path:
    return out_dir("figures") / f"{name}{suffix}"


def csvpath(name: str) -> Path:
    return out_dir("reports") / f"{name}.csv"
