"""Tile-based phase-correlation registration, adapted from the author's CaSSIS
coregistration utility (coreg_cassis.py, M3-derived method): high-pass prefilter, masked coarse
lock, per-tile subpixel phase correlation with an edge-NCC confidence gate,
and a confidence-weighted median fit.

Sign convention (verified by `self_test`): with
`phase_cross_correlation(reference, moving)` returning (drow, dcol), the
moving image content must be shifted by (+drow, +dcol) pixels to align
with the reference. For a north-up raster with pixel size `res`, the
moving raster's georeferencing is corrected by
    UL_X += dcol * res,   UL_Y -= drow * res.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.registration import phase_cross_correlation


def highpass(img: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian high-pass with normalised convolution so invalid pixels do not bleed."""
    img = np.where(valid, img, 0.0).astype(np.float64)
    w = valid.astype(np.float64)
    num = ndi.gaussian_filter(img, sigma)
    den = ndi.gaussian_filter(w, sigma)
    low = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0.0)
    hp = np.where(valid, img - low, 0.0)
    s = hp[valid].std()
    return hp / s if s > 0 else hp


def _shift(img: np.ndarray, drow: float, dcol: float) -> np.ndarray:
    return ndi.shift(img, (drow, dcol), order=1, mode="constant", cval=0.0)


def _edge(img: np.ndarray) -> np.ndarray:
    return np.hypot(ndi.sobel(img, 0), ndi.sobel(img, 1))


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def edge_ncc_confidence(ref_hp: np.ndarray, mov_hp: np.ndarray, drow: float, dcol: float) -> tuple[float, float]:
    """Edge-magnitude NCC after applying the shift vs. before (sign-invariant)."""
    er, em = _edge(ref_hp), _edge(mov_hp)
    return _ncc(er, _edge(_shift(mov_hp, drow, dcol))), _ncc(er, em)


def _block_decimate(img: np.ndarray, valid: np.ndarray, f: int) -> tuple[np.ndarray, np.ndarray]:
    """f x f block average over valid pixels; a block is valid if >= half its pixels are."""
    H, W = (img.shape[0] // f) * f, (img.shape[1] // f) * f
    a = np.where(valid[:H, :W], img[:H, :W], 0.0).reshape(H // f, f, W // f, f).sum(axis=(1, 3))
    n = valid[:H, :W].reshape(H // f, f, W // f, f).sum(axis=(1, 3))
    return a / np.maximum(n, 1), n >= (f * f) / 2


def coarse_lock(ref_hp: np.ndarray, mov_hp: np.ndarray, ref_valid: np.ndarray, mov_valid: np.ndarray,
                max_dim: int | None = None) -> tuple[float, float]:
    """Masked full-frame lock. The masked phase correlation allocates ~a dozen complex
    arrays of the padded frame size; `max_dim` block-decimates both frames first so the
    FFTs stay small (the per-tile fit restores precision downstream). The returned shift
    is in full-resolution pixels either way."""
    f = 1
    if max_dim is not None:
        f = max(1, -(-max(*ref_hp.shape, *mov_hp.shape) // max_dim))
    if f > 1:
        ref_hp, ref_valid = _block_decimate(ref_hp, ref_valid, f)
        mov_hp, mov_valid = _block_decimate(mov_hp, mov_valid, f)
    shift = phase_cross_correlation(ref_hp, mov_hp, reference_mask=ref_valid, moving_mask=mov_valid, overlap_ratio=0.3)
    shift = shift[0] if isinstance(shift, tuple) else shift
    return float(shift[0]) * f, float(shift[1]) * f


def tile_offsets(ref_hp: np.ndarray, mov_hp: np.ndarray, ref_valid: np.ndarray, mov_valid: np.ndarray,
                 coarse: tuple[float, float], tile: int, stride: int, upsample: int = 10,
                 min_valid_frac: float = 0.6, texture_min: float = 0.05, max_dev_px: float = 20.0) -> list[dict]:
    """Per-tile subpixel offsets around the coarse lock. Returns dicts with row, col, drow, dcol, conf."""
    H, W = ref_hp.shape
    drow0, dcol0 = coarse
    mov_pre = _shift(mov_hp, drow0, dcol0)          # apply coarse lock to the moving image
    movv_pre = _shift(mov_valid.astype(float), drow0, dcol0) > 0.5
    out = []
    for r0 in range(0, H - tile + 1, stride):
        for c0 in range(0, W - tile + 1, stride):
            sl = (slice(r0, r0 + tile), slice(c0, c0 + tile))
            v = ref_valid[sl] & movv_pre[sl]
            if v.mean() < min_valid_frac:
                continue
            a, b = ref_hp[sl] * v, mov_pre[sl] * v
            if np.abs(a).mean() < texture_min or np.abs(b).mean() < texture_min:
                continue
            res = phase_cross_correlation(a, b, upsample_factor=upsample)
            sh = res[0] if isinstance(res, tuple) else res
            drow, dcol = float(sh[0]), float(sh[1])
            if np.hypot(drow, dcol) > max_dev_px:
                continue
            conf, conf0 = edge_ncc_confidence(a, b, drow, dcol)
            out.append({"row": r0 + tile // 2, "col": c0 + tile // 2, "drow": drow0 + drow, "dcol": dcol0 + dcol,
                        "conf": conf, "conf_unshifted": conf0})
    return out


def fit_offsets(tiles: list[dict], conf_gate: float = 0.30) -> dict:
    good = [t for t in tiles if t["conf"] >= conf_gate]
    if len(good) < 3:
        return {"n_tiles": len(tiles), "n_good": len(good), "ok": False}
    w = np.array([t["conf"] for t in good]); dr = np.array([t["drow"] for t in good]); dc = np.array([t["dcol"] for t in good])
    def wmed(v):
        o = np.argsort(v); cw = np.cumsum(w[o]); return float(v[o][np.searchsorted(cw, cw[-1] / 2)])
    drow, dcol = wmed(dr), wmed(dc)
    mad_r = float(np.median(np.abs(dr - drow))); mad_c = float(np.median(np.abs(dc - dcol)))
    return {"n_tiles": len(tiles), "n_good": len(good), "ok": True, "drow": drow, "dcol": dcol,
            "mad_row": mad_r, "mad_col": mad_c, "conf_median": float(np.median(w))}


def self_test() -> None:
    """Synthetic check of the sign convention: moving = reference shifted by (+3, -5) px content."""
    rng = np.random.default_rng(42)
    ref = ndi.gaussian_filter(rng.normal(size=(256, 256)), 2.0)
    mov = _shift(ref, 3.0, -5.0)                    # content moved down 3, left 5
    valid = np.ones_like(ref, bool)
    rh, mh = highpass(ref, valid, 8.0), highpass(mov, valid, 8.0)
    drow, dcol = coarse_lock(rh, mh, valid, valid)
    # to realign, mov must be shifted by (-3, +5)
    assert abs(drow + 3.0) < 0.6 and abs(dcol - 5.0) < 0.6, (drow, dcol)
    drow2, dcol2 = coarse_lock(rh, mh, valid, valid, max_dim=64)   # 4x decimated path
    assert abs(drow2 + 3.0) < 4.0 and abs(dcol2 - 5.0) < 4.0, (drow2, dcol2)
    tiles = tile_offsets(rh, mh, valid, valid, (drow, dcol), 128, 64)
    fit = fit_offsets(tiles)
    assert fit["ok"] and abs(fit["drow"] + 3.0) < 0.3 and abs(fit["dcol"] - 5.0) < 0.3, fit


if __name__ == "__main__":
    self_test()
    print("self_test passed")
