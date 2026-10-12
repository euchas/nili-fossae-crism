"""Display helpers for Stage A+ figures.

Cubes are stored on disk in detector orientation (increasing line index
goes away from MRO along-track, increasing sample goes across-track in
detector coordinates). Rendering that directly leaves spatial figures
upside-down / mirrored relative to standard Mars-map convention.

The convention adopted by the original PCA_CRISM_analysis_v2.py is to
flip both axes for display. This module centralises those flips so every
plotting script applies them consistently. **Data on disk is never
flipped — only what reaches the matplotlib call.**
"""

from __future__ import annotations

import numpy as np

FLIP_VERTICAL = True
FLIP_HORIZONTAL = True


def apply_display_transform(image: np.ndarray) -> np.ndarray:
    """Return a view of *image* with the configured flips applied.

    Accepts 2-D arrays (parameter maps, masks) and 3-D arrays of shape
    (lines, samples, channels) (RGBA composites). The first two axes are
    treated as the spatial axes and flipped independently.
    """
    result = image
    if FLIP_VERTICAL:
        result = np.flipud(result)
    if FLIP_HORIZONTAL:
        result = np.fliplr(result)
    return result


def data_to_display_xy(
    lines: np.ndarray, samples: np.ndarray, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Convert (line, sample) data coordinates to display (x, y) coordinates.

    Mirrors the helper in the original PCA script and preserves the
    invariant that scatter plots align with the flipped parameter maps.
    """
    n_lines, n_samples = shape
    xs = np.asarray(samples).copy()
    ys = np.asarray(lines).copy()
    if FLIP_HORIZONTAL:
        xs = n_samples - 1 - xs
    if FLIP_VERTICAL:
        ys = n_lines - 1 - ys
    return xs, ys
