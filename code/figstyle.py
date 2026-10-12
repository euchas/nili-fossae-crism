"""Shared figure style: geologic-unit contours for all map figures.

Olivine unit red, phyllosilicate (clay) basement blue, mafic cap dark grey
dashed; each drawn over a thin white casing so the lines stay legible on
both the light background and dark colormap areas. AV's request,
2026-09-05. Import and call draw_units(ax, fr) wherever unit contours
are drawn; fr is the fractions dict from contact_band_center_map.unit_masks.
"""
from __future__ import annotations

import numpy as np

UNIT_STYLE = {
    "oliv": {"color": "#d62728", "ls": "solid", "label": "olivine unit"},
    "clay": {"color": "#1f77b4", "ls": "solid", "label": "phyllosilicate basement"},
    "cap": {"color": "#222222", "ls": "dashed", "label": "mafic cap"},
}
LW = 1.7
CASING_LW = 3.2


def draw_units(ax, fr: dict, units=("oliv", "clay", "cap"), lw: float = LW,
               casing: bool = True, thresh: float = 0.5) -> None:
    for u in units:
        if u not in fr:
            continue
        m = (np.asarray(fr[u]) >= thresh).astype(float)
        st = UNIT_STYLE[u]
        if casing:
            ax.contour(m, levels=[0.5], colors="white", linewidths=lw + (CASING_LW - LW),
                       linestyles=st["ls"], alpha=0.9)
        ax.contour(m, levels=[0.5], colors=st["color"], linewidths=lw, linestyles=st["ls"])


def label_panels(fig, fontsize: int = 11, loc=(0.012, 0.985), skip_single: bool = True) -> list:
    """Stamp (a), (b), ... on each plotting axes of a figure, in reading order.

    Colorbar and twin axes are skipped; single-panel figures are left
    unlabelled unless skip_single is False. Returns the labelled axes.
    """
    import string
    axes = []
    seen_bboxes = []
    for ax in fig.axes:
        if ax.get_label() == "<colorbar>":
            continue
        bb = tuple(np.round(ax.get_position().bounds, 4))
        if bb in seen_bboxes:            # twin axes share the same box
            continue
        seen_bboxes.append(bb)
        axes.append(ax)
    if skip_single and len(axes) <= 1:
        return []
    axes.sort(key=lambda a: (-round(a.get_position().y1, 3), a.get_position().x0))
    for i, ax in enumerate(axes):
        ax.text(loc[0], loc[1], f"({string.ascii_lowercase[i]})", transform=ax.transAxes,
                ha="left", va="top", fontsize=fontsize, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85), zorder=20)
    return axes


def safe_savefig(fig, path, dpi: int = 200, pdf_dpi: int | None = None, **kw) -> list:
    """Save <path>.png and .pdf; if a file is locked (open in a viewer), write *_new instead.
    pdf_dpi sets the resolution of raster (imshow) parts inside the PDF; None = matplotlib default (100).
    Returns the list of targets that had to be redirected."""
    from pathlib import Path
    path = Path(path)
    redirected = []
    for ext in (".png", ".pdf"):
        target = path.with_suffix(ext)
        try:
            fig.savefig(target, dpi=dpi if ext == ".png" else pdf_dpi, **kw)
        except OSError:
            for suffix in ("_new", "_new2", "_new3", "_new4"):      # locked viewers may hold several copies open
                alt = path.with_name(path.stem + suffix).with_suffix(ext)
                try:
                    fig.savefig(alt, dpi=dpi if ext == ".png" else pdf_dpi, **kw)
                    break
                except OSError:
                    continue
            redirected.append(str(target))
    if redirected:
        print("LOCKED, written as *_new:", redirected)
    return redirected


def unit_legend(ax, units=("oliv", "clay", "cap"), loc="lower left", fontsize=8):
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=UNIT_STYLE[u]["color"], lw=LW,
                      ls=UNIT_STYLE[u]["ls"], label=UNIT_STYLE[u]["label"]) for u in units]
    ax.legend(handles=handles, loc=loc, fontsize=fontsize, frameon=True, framealpha=0.85)
