"""Viviano-Beck 2014 spectral parameters used in Stage A.

The four implementations (OLINDEX3, D2300, BD1900R2, BD2500_2) are copied
verbatim from PCA_CRISM_analysis_v2.py — this is a refactor for reuse, not
a reformulation. Kernel widths match the existing script, which in turn
follows the Viviano-Beck 2014 Table 2 convention for the 1 µm and
continuum anchor channels.

Stage D will extend this module with BD2290, BD1400, SINDEX2, LCPINDEX2,
and MIN2295_2480 when the 12-parameter sensitivity analysis is set up.
"""

from __future__ import annotations

import numpy as np


def _kernel(data: np.ndarray, wav: np.ndarray, target: float, width: int) -> np.ndarray:
    idx = int(np.abs(wav - target).argmin())
    if width <= 1:
        return data[idx, :, :]
    hw = width // 2
    lo = max(0, idx - hw)
    hi = min(len(wav), idx + hw + 1)
    return np.nanmedian(data[lo:hi, :, :], axis=0)


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(den != 0, 1.0 - (num / den), 0.0)
    return np.nan_to_num(result, nan=np.nan)


def _rb(R: np.ndarray, RC: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(RC != 0, (RC - R) / RC, 0.0)
    return np.nan_to_num(result, nan=np.nan)


def calc_olindex3(data: np.ndarray, wav: np.ndarray) -> np.ndarray:
    R1210 = _kernel(data, wav, 1.210, 7)
    R1250 = _kernel(data, wav, 1.250, 7)
    R1263 = _kernel(data, wav, 1.263, 7)
    R1276 = _kernel(data, wav, 1.276, 7)
    R1330 = _kernel(data, wav, 1.330, 7)
    R1750 = _kernel(data, wav, 1.750, 7)
    R1862 = _kernel(data, wav, 1.862, 7)
    slope = np.where(R1862 != R1750, (R1862 - R1750) / (1.862 - 1.750), 0.0)

    def rc(w: float) -> np.ndarray:
        return R1750 + slope * (w - 1.750)

    return (
        0.1 * _rb(R1210, rc(1.210))
        + 0.1 * _rb(R1250, rc(1.250))
        + 0.2 * _rb(R1263, rc(1.263))
        + 0.2 * _rb(R1276, rc(1.276))
        + 0.4 * _rb(R1330, rc(1.330))
    )


def calc_d2300(data: np.ndarray, wav: np.ndarray) -> np.ndarray:
    R1815 = _kernel(data, wav, 1.815, 5)
    R2120 = _kernel(data, wav, 2.120, 5)
    R2170 = _kernel(data, wav, 2.170, 5)
    R2210 = _kernel(data, wav, 2.210, 5)
    R2290 = _kernel(data, wav, 2.290, 3)
    R2320 = _kernel(data, wav, 2.320, 3)
    R2330 = _kernel(data, wav, 2.330, 3)
    R2530 = _kernel(data, wav, 2.530, 5)
    slope = np.where(R2530 != R1815, (R2530 - R1815) / (2.530 - 1.815), 0.0)

    def rc(w: float) -> np.ndarray:
        return R1815 + slope * (w - 1.815)

    with np.errstate(divide="ignore", invalid="ignore"):
        num = (
            np.where(rc(2.290) != 0, R2290 / rc(2.290), 0.0)
            + np.where(rc(2.320) != 0, R2320 / rc(2.320), 0.0)
            + np.where(rc(2.330) != 0, R2330 / rc(2.330), 0.0)
        )
        den = (
            np.where(rc(2.120) != 0, R2120 / rc(2.120), 0.0)
            + np.where(rc(2.170) != 0, R2170 / rc(2.170), 0.0)
            + np.where(rc(2.210) != 0, R2210 / rc(2.210), 0.0)
        )
        result = np.where(den != 0, 1.0 - num / den, 0.0)
    return np.nan_to_num(result, nan=np.nan)


def calc_bd1900r2(data: np.ndarray, wav: np.ndarray) -> np.ndarray:
    num_w = [1.908, 1.914, 1.921, 1.928, 1.934, 1.941]
    den_w = [1.862, 1.869, 1.875, 2.112, 2.120, 2.126]
    R1850 = _kernel(data, wav, 1.850, 5)
    R2060 = _kernel(data, wav, 2.060, 5)
    slope = np.where(R2060 != R1850, (R2060 - R1850) / (2.060 - 1.850), 0.0)

    def rc(w: float) -> np.ndarray:
        return R1850 + slope * (w - 1.850)

    with np.errstate(divide="ignore", invalid="ignore"):
        num_sum = sum(
            np.where(rc(w) != 0, _kernel(data, wav, w, 1) / rc(w), 0.0) for w in num_w
        )
        den_sum = sum(
            np.where(rc(w) != 0, _kernel(data, wav, w, 1) / rc(w), 0.0) for w in den_w
        )
        result = np.where(den_sum != 0, 1.0 - num_sum / den_sum, 0.0)
    return np.nan_to_num(result, nan=np.nan)


def calc_bd2500_2(data: np.ndarray, wav: np.ndarray) -> np.ndarray:
    R2364 = _kernel(data, wav, 2.364, 5)
    R2480 = _kernel(data, wav, 2.480, 5)
    R2570 = _kernel(data, wav, 2.570, 5)
    b = (2.480 - 2.364) / (2.570 - 2.364)
    a = 1.0 - b
    return _ratio(R2480, a * R2364 + b * R2570)


# =============================================================================
# Stage B additions (Viviano-Beck 2014 Table 2)
# =============================================================================


def _band_depth_3pt(
    data: np.ndarray,
    wav: np.ndarray,
    w_left: float,
    w_centre: float,
    w_right: float,
    kernel_left: int,
    kernel_centre: int,
    kernel_right: int,
) -> np.ndarray:
    """Generic three-point band depth `1 - R_centre / (a * R_left + b * R_right)`
    with linear continuum interpolation between *w_left* and *w_right*."""
    Rl = _kernel(data, wav, w_left, kernel_left)
    Rc = _kernel(data, wav, w_centre, kernel_centre)
    Rr = _kernel(data, wav, w_right, kernel_right)
    b = (w_centre - w_left) / (w_right - w_left)
    a = 1.0 - b
    return _ratio(Rc, a * Rl + b * Rr)


def _band_depth_multi_pt(
    data: np.ndarray,
    wav: np.ndarray,
    centres: list[float],
    anchors: tuple[float, float],
    centre_kernel: int = 5,
    anchor_kernel: int = 5,
    weights: list[float] | None = None,
) -> np.ndarray:
    """Generic multi-channel weighted-mean band depth with linear continuum
    interpolated between *anchors* (left, right). Used by HCPINDEX2.

    For each centre wavelength, computes (RC_centre - R_centre) / RC_centre
    where RC is the linear-interpolated continuum at that wavelength. Returns
    the (optionally weighted) mean across all centres.
    """
    w_left, w_right = anchors
    R_left = _kernel(data, wav, w_left, anchor_kernel)
    R_right = _kernel(data, wav, w_right, anchor_kernel)
    slope = np.where(R_right != R_left,
                     (R_right - R_left) / (w_right - w_left), 0.0)
    if weights is None:
        weights = [1.0 / len(centres)] * len(centres)
    if len(weights) != len(centres):
        raise ValueError("len(weights) must match len(centres)")
    total = np.zeros_like(R_left)
    for w_i, ww in zip(centres, weights):
        Ri = _kernel(data, wav, w_i, centre_kernel)
        Rci = R_left + slope * (w_i - w_left)
        total = total + ww * _rb(Ri, Rci)
    return total


def calc_sh600_2(data: np.ndarray, wav: np.ndarray) -> np.ndarray:
    """600 nm shoulder strength (VB 2014 Table 1).

    SH600_2 = 1 - (a*R533 + b*R716) / R600  -- INVERTED from band depth.
    Numerator is the linear-continuum reflectance at 600 nm; denominator is
    the observed reflectance at 600 nm. Positive = reflectance peak at
    600 nm relative to the 533-716 continuum line (continuum below data),
    expected for ferric-bearing pixels. Kernel widths 5/5/5 per VB 2014.
    """
    R533 = _kernel(data, wav, 0.533, 5)
    R600 = _kernel(data, wav, 0.600, 5)
    R716 = _kernel(data, wav, 0.716, 5)
    b = (0.600 - 0.533) / (0.716 - 0.533)
    a = 1.0 - b
    rc = a * R533 + b * R716
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(R600 != 0, 1.0 - rc / R600, 0.0)
    return np.nan_to_num(result, nan=np.nan)


def calc_hcpindex2(data: np.ndarray, wav: np.ndarray) -> np.ndarray:
    """High-Ca pyroxene index (VB 2014 Table 1).

    Equal-weighted mean of band depths at {2.120, 2.140, 2.230, 2.250,
    2.430, 2.460} um with continuum anchored at R1690 / R2530. Kernel
    widths 5 channels per VB 2014.
    """
    return _band_depth_multi_pt(
        data, wav,
        centres=[2.120, 2.140, 2.230, 2.250, 2.430, 2.460],
        anchors=(1.690, 2.530),
        centre_kernel=5,
        anchor_kernel=5,
        weights=None,  # equal weights per VB 2014
    )


def calc_bd530_2(data: np.ndarray, wav: np.ndarray) -> np.ndarray:
    """0.53 µm band depth — fine-grained crystalline hematite proxy (FM2)."""
    return _band_depth_3pt(data, wav, 0.440, 0.530, 0.614, 5, 5, 5)


def calc_bd920_2(data: np.ndarray, wav: np.ndarray) -> np.ndarray:
    """0.92 µm band depth — crystalline ferric / low-Ca pyroxene."""
    return _band_depth_3pt(data, wav, 0.807, 0.920, 0.984, 5, 5, 5)


def calc_bd1400(data: np.ndarray, wav: np.ndarray) -> np.ndarray:
    """1.4 µm OH / H2O band depth — hydrated / hydroxylated minerals."""
    return _band_depth_3pt(data, wav, 1.330, 1.395, 1.467, 5, 5, 5)


def calc_bd2290(data: np.ndarray, wav: np.ndarray) -> np.ndarray:
    """2.29 µm Mg/Fe-OH narrow band depth (split chlorite vs talc per Viviano 2013)."""
    return _band_depth_3pt(data, wav, 2.250, 2.290, 2.350, 5, 5, 5)


def calc_bd2355(data: np.ndarray, wav: np.ndarray) -> np.ndarray:
    """2.35 µm chlorite / prehnite band depth."""
    return _band_depth_3pt(data, wav, 2.300, 2.355, 2.450, 5, 5, 5)


def calc_bd2330(data: np.ndarray, wav: np.ndarray) -> np.ndarray:
    """2.33 µm long-wavelength-shifted Mg/Fe-OH band depth.

    BD2330 = 1 - R2330 / (a*R2250 + b*R2400),  a=0.467, b=0.533
    (linear-interpolation continuum, VB 2014 convention).
    5-channel kernel on each anchor wavelength.
    """
    R2330 = _kernel(data, wav, 2.330, 5)
    R2250 = _kernel(data, wav, 2.250, 5)
    R2400 = _kernel(data, wav, 2.400, 5)
    cont = 0.467 * R2250 + 0.533 * R2400
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(cont != 0, 1.0 - (R2330 / cont), np.nan)
    return out.astype(np.float32)


def calc_bd1900_2(data: np.ndarray, wav: np.ndarray) -> np.ndarray:
    """1.9 µm H2O bend-stretch combination band (VB 2014 BD1900_2 form).

    BD1900_2 = 1 - 0.5*(R1930 + R1985) / (a*R1850 + b*R2067)
    where the continuum is linear-interpolated to the midpoint of the
    target band (~1957.5 nm), giving a~=0.504, b~=0.495. Kernel width
    5 on every anchor.
    """
    R1930 = _kernel(data, wav, 1.930, 5)
    R1985 = _kernel(data, wav, 1.985, 5)
    R1850 = _kernel(data, wav, 1.850, 5)
    R2067 = _kernel(data, wav, 2.067, 5)
    a = (2.067 - 1.9575) / (2.067 - 1.850)
    b = (1.9575 - 1.850) / (2.067 - 1.850)
    cont = a * R1850 + b * R2067
    target = 0.5 * (R1930 + R1985)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(cont != 0, 1.0 - (target / cont), np.nan)
    return out.astype(np.float32)


def calc_r_chlor(data: np.ndarray, wav: np.ndarray,
                  bd_floor: float = 0.005) -> np.ndarray:
    """R_chlor = BD(2.328) / BD(2.302) - long-wavelength shift diagnostic.

    Each BD uses the standard Viviano-Beck formulation
        BD(c) = 1 - R(c) / (a*R(2.250) + b*R(2.400))
    with 5-channel nanmedian kernels and linear-interpolation weights for
    the continuum anchors:
        a = (2.400 - c) / (2.400 - 2.250)
        b = (c - 2.250) / (2.400 - 2.250)

    Pixels where BD(2.302) < `bd_floor` are masked to NaN to avoid
    amplifying noise in the denominator. the project specification: bd_floor = 0.005.
    """
    R_lo = _kernel(data, wav, 2.250, 5)
    R_hi = _kernel(data, wav, 2.400, 5)
    R_2328 = _kernel(data, wav, 2.328, 5)
    R_2302 = _kernel(data, wav, 2.302, 5)

    span = (2.400 - 2.250)
    a_2328 = (2.400 - 2.328) / span
    b_2328 = (2.328 - 2.250) / span
    a_2302 = (2.400 - 2.302) / span
    b_2302 = (2.302 - 2.250) / span

    cont_2328 = a_2328 * R_lo + b_2328 * R_hi
    cont_2302 = a_2302 * R_lo + b_2302 * R_hi

    with np.errstate(divide="ignore", invalid="ignore"):
        bd_2328 = np.where(cont_2328 != 0,
                            1.0 - R_2328 / cont_2328, np.nan)
        bd_2302 = np.where(cont_2302 != 0,
                            1.0 - R_2302 / cont_2302, np.nan)
        ratio = np.where(np.abs(bd_2302) > 1e-9, bd_2328 / bd_2302, np.nan)
    out = np.where(bd_2302 >= bd_floor, ratio, np.nan)
    return out.astype(np.float32)


def calc_lcpindex2(data: np.ndarray, wav: np.ndarray) -> np.ndarray:
    """Low-Ca pyroxene index (VB 2014 #15).

    LCPINDEX2 = RB1690 * 0.20 + RB1750 * 0.20 + RB1810 * 0.30 + RB1870 * 0.30
    where RB#### = (RC#### - R####) / RC####
    and the continuum slope RC is anchored at R1560 and R2450.
    """
    R1560 = _kernel(data, wav, 1.560, 7)
    R1690 = _kernel(data, wav, 1.690, 7)
    R1750 = _kernel(data, wav, 1.750, 7)
    R1810 = _kernel(data, wav, 1.810, 7)
    R1870 = _kernel(data, wav, 1.870, 7)
    R2450 = _kernel(data, wav, 2.450, 7)
    slope = np.where(R2450 != R1560, (R2450 - R1560) / (2.450 - 1.560), 0.0)

    def rc(w: float) -> np.ndarray:
        return R1560 + slope * (w - 1.560)

    return (
        0.20 * _rb(R1690, rc(1.690))
        + 0.20 * _rb(R1750, rc(1.750))
        + 0.30 * _rb(R1810, rc(1.810))
        + 0.30 * _rb(R1870, rc(1.870))
    )


def calc_bdi1000vis(
    data: np.ndarray,
    wav: np.ndarray,
    fit_wmin: float = 0.44,
    fit_wmax: float = 1.00,
    band_wmin: float = 0.833,
    band_wmax: float = 1.023,
) -> np.ndarray:
    """VNIR 1 µm integrated band depth (VB 2014 #10).

    Per VB 2014 / Pelkey et al. 2007:
      1. Fit a 5th-order polynomial to all-valid VNIR reflectances per pixel.
      2. RPEAK1 is the wavelength of the polynomial maximum (modeled
         continuum reflectance).
      3. Normalise observed reflectances at all VNIR channels in
         band_wmin..band_wmax by the modeled value at RPEAK1.
      4. BDI = trapezoidal integral of (1 - normalised) over wavelength.

    Pixels with any NaN reflectance in the fit window are returned as NaN.

    Implementation notes:
      * Uses a centred wavelength basis to keep the Vandermonde matrix
        well-conditioned.
      * Solves all valid pixels simultaneously via np.linalg.lstsq.
      * RPEAK1 is found by argmax over a dense (200-point) wavelength
        grid evaluated from the fitted polynomial.
    """
    n_bands, n_lines, n_samples = data.shape
    fit_mask = (wav >= fit_wmin) & (wav <= fit_wmax)
    band_mask = (wav >= band_wmin) & (wav <= band_wmax)
    if fit_mask.sum() < 8 or band_mask.sum() < 3:
        raise ValueError(
            f"Insufficient VNIR coverage for BDI1000VIS: fit_n={fit_mask.sum()}, "
            f"band_n={band_mask.sum()}"
        )

    w_fit_all = wav[fit_mask].astype(np.float64)
    w_band_all = wav[band_mask].astype(np.float64)

    R_fit_all = data[fit_mask, :, :].reshape(w_fit_all.size, -1).astype(np.float64)
    R_band_all = data[band_mask, :, :].reshape(w_band_all.size, -1).astype(np.float64)

    # CRISM BBL flags some bands as invalid for every pixel. Those bands are
    # NaN across the entire frame and would force the per-pixel "all finite"
    # test to reject every pixel. Drop them from the fit window.
    fit_band_keep = np.isfinite(R_fit_all).any(axis=1)
    band_band_keep = np.isfinite(R_band_all).any(axis=1)
    if fit_band_keep.sum() < 8 or band_band_keep.sum() < 3:
        raise ValueError(
            "Too few BBL-valid bands in the VNIR range after dropping all-NaN "
            f"bands: fit_valid={fit_band_keep.sum()}, band_valid={band_band_keep.sum()}"
        )

    w_fit = w_fit_all[fit_band_keep]
    R_fit = R_fit_all[fit_band_keep]
    w_band = w_band_all[band_band_keep]
    R_band = R_band_all[band_band_keep]
    w_fit_centre = float(w_fit.mean())

    # Per-pixel finite check only on the remaining (BBL-valid) bands.
    fit_valid = np.isfinite(R_fit).all(axis=0)
    band_valid = np.isfinite(R_band).all(axis=0)
    pix_valid = fit_valid & band_valid

    bdi = np.full(R_fit.shape[1], np.nan, dtype=np.float64)
    if not pix_valid.any():
        return bdi.reshape(n_lines, n_samples).astype(np.float32)

    # Centred Vandermonde matrix (orders 0..5).
    V = np.vander(w_fit - w_fit_centre, N=6, increasing=True)
    coeffs, *_ = np.linalg.lstsq(V, R_fit[:, pix_valid], rcond=None)

    # Dense evaluation for RPEAK1 (argmax of fitted polynomial).
    w_dense = np.linspace(w_fit.min(), w_fit.max(), 200)
    V_dense = np.vander(w_dense - w_fit_centre, N=6, increasing=True)
    fit_dense = V_dense @ coeffs   # shape (200, n_valid_px)
    peak_idx = np.argmax(fit_dense, axis=0)
    n_valid = peak_idx.size
    r_peak1 = fit_dense[peak_idx, np.arange(n_valid)]
    # Guard against a degenerate fit (zero or negative continuum).
    safe = r_peak1 > 1e-6

    normalised = R_band[:, pix_valid] / r_peak1[np.newaxis, :]
    band_depth = 1.0 - normalised
    integrated = np.trapezoid(band_depth, w_band, axis=0)
    integrated = np.where(safe, integrated, np.nan)

    bdi[pix_valid] = integrated
    return bdi.reshape(n_lines, n_samples).astype(np.float32)


def compute_default_vb_stack(
    data: np.ndarray, wav: np.ndarray
) -> dict[str, np.ndarray]:
    """Return a dict of the four VB parameters used in Stage A."""
    return {
        "OLINDEX3": calc_olindex3(data, wav),
        "D2300": calc_d2300(data, wav),
        "BD1900R2": calc_bd1900r2(data, wav),
        "BD2500_2": calc_bd2500_2(data, wav),
    }


def compute_extended_vb_stack(
    data: np.ndarray, wav: np.ndarray
) -> dict[str, np.ndarray]:
    """Return the 11 VB parameters that ride the Stage B 15-D feature stack."""
    return {
        "BD530_2": calc_bd530_2(data, wav),
        "BD920_2": calc_bd920_2(data, wav),
        "BDI1000VIS": calc_bdi1000vis(data, wav),
        "OLINDEX3": calc_olindex3(data, wav),
        "LCPINDEX2": calc_lcpindex2(data, wav),
        "BD1400": calc_bd1400(data, wav),
        "BD1900R2": calc_bd1900r2(data, wav),
        "D2300": calc_d2300(data, wav),
        "BD2290": calc_bd2290(data, wav),
        "BD2355": calc_bd2355(data, wav),
        "BD2500_2": calc_bd2500_2(data, wav),
    }


# v4 VB-only basis: drops BD2290, BD2355; adds SH600_2, HCPINDEX2.
V4_VB_PARAM_ORDER = (
    "BD530_2",
    "SH600_2",
    "BD920_2",
    "BDI1000VIS",
    "OLINDEX3",
    "LCPINDEX2",
    "HCPINDEX2",
    "BD1400",
    "BD1900R2",
    "D2300",
    "BD2500_2",
)


def compute_v4_vb_stack(
    data: np.ndarray, wav: np.ndarray
) -> dict[str, np.ndarray]:
    """Return the 11 VB parameters used in the v4 PCA basis.

    Drops the redundant 2.3 um neighbours (BD2290, BD2355 - all three loaded
    identically with D2300 in v2). Adds SH600_2 (independent ferric edge)
    and HCPINDEX2 (paired with LCPINDEX2 for full pyroxene story).

    Brown 4 are NOT included here - v4 separates the Brown workflow from
    the VB PCA workflow.
    """
    return {
        "BD530_2": calc_bd530_2(data, wav),
        "SH600_2": calc_sh600_2(data, wav),
        "BD920_2": calc_bd920_2(data, wav),
        "BDI1000VIS": calc_bdi1000vis(data, wav),
        "OLINDEX3": calc_olindex3(data, wav),
        "LCPINDEX2": calc_lcpindex2(data, wav),
        "HCPINDEX2": calc_hcpindex2(data, wav),
        "BD1400": calc_bd1400(data, wav),
        "BD1900R2": calc_bd1900r2(data, wav),
        "D2300": calc_d2300(data, wav),
        "BD2500_2": calc_bd2500_2(data, wav),
    }
