"""
Fast Frost speckle filters using Numba. 

frost_filter_fast does minimal numerical checking. It will break (silently) if there are *any* NaNs in the input.

frost_filter_nanny is NaN-aware, but this comes with a performance penalty.

Based on findpeaks.filters [0] / PyRadar [1], and the original paper [2].

Mostly generated with ChatGPT o3 (17-Jun-2025).


[0]: https://github.com/erdogant/findpeaks/blob/master/findpeaks/filters/frost.py

[1]: https://github.com/PyRadar/pyradar

[2]: Frost, Victor S., Josephine Abbott Stiles, K. S. Shanmugan, and Julian C. Holtzman. 'A Model for Radar Images and Its Application to Adaptive Digital Filtering of Multiplicative Noise'. IEEE Transactions on Pattern Analysis and Machine Intelligence PAMI-4, no. 2 (March 1982): 15766. https://doi.org/10.1109/TPAMI.1982.4767223.

Licence: LGPL-3.0.

"""

import numpy as np
from scipy.ndimage import uniform_filter
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def _frost_core(img, factor_A, r):
    """Inner loop: compute weighted average for every pixel."""
    n_rows, n_cols = img.shape
    out = np.empty_like(img)

    for i in prange(n_rows):               # parallel loop
        for j in range(n_cols):
            # window bounds (clamped to image)
            x0 = max(i - r, 0);  x1 = min(i + r + 1, n_rows)
            y0 = max(j - r, 0);  y1 = min(j + r + 1, n_cols)

            centre = img[i, j]
            A      = factor_A[i, j]

            w_sum = 0.0
            v_sum = 0.0
            for x in range(x0, x1):
                for y in range(y0, y1):
                    diff = abs(img[x, y] - centre)
                    w = np.exp(-A * diff)
                    w_sum += w
                    v_sum += w * img[x, y]

            out[i, j] = v_sum / w_sum if w_sum > 0.0 else 0.0
    return out


def frost_filter_fast(img, *, damping_factor: float = 2.0,
                 win_size: int = 3, dtype=np.float32):
    """
    Frost speckle filter accelerated with Numba.

    Parameters
    ----------
    img : 2-D ndarray
    damping_factor : float, default 2.0
    win_size : odd int ≥ 3, default 3
    dtype : numpy dtype, default float32

    Returns
    -------
    ndarray  (same shape as *img*)
    """
    if img.ndim != 2:
        raise ValueError("Input must be a 2-D array.")
    if win_size < 3 or win_size % 2 == 0:
        raise ValueError("win_size must be odd and ≥ 3.")

    img = np.asarray(img, dtype=dtype, order='C')  # contiguous for Numba
    r   = win_size // 2

    # Local mean and variance via fast box filter
    mean     = uniform_filter(img, size=win_size, mode='reflect')
    mean_sq  = uniform_filter(img * img, size=win_size, mode='reflect')
    var      = np.clip(mean_sq - mean * mean, 0, None)
    std      = np.sqrt(var)

    # Frost parameters
    sigma0   = std / np.maximum(mean, 1e-12) / np.maximum(mean, 1e-12)
    factor_A = damping_factor * sigma0

    return _frost_core(img, factor_A, r).astype(img.dtype, copy=False)


@njit(parallel=True, fastmath=True)
def _frost_core_nanny(img, factor_A, win_size):
    """Numba kernel: weight computation + pixel update (ignores NaNs)."""
    n_rows, n_cols = img.shape
    r = win_size // 2
    out = np.empty_like(img)

    for i in prange(n_rows):
        for j in range(n_cols):
            centre = img[i, j]
            if np.isnan(centre):              # centre missing → output NaN
                out[i, j] = np.nan
                continue

            x0 = max(i - r, 0)
            x1 = min(i + r + 1, n_rows)
            y0 = max(j - r, 0)
            y1 = min(j + r + 1, n_cols)

            A = factor_A[i, j]
            w_sum = 0.0
            v_sum = 0.0

            for x in range(x0, x1):
                for y in range(y0, y1):
                    pix = img[x, y]
                    if np.isnan(pix):          # skip NaNs in the window
                        continue
                    diff = abs(pix - centre)
                    w = np.exp(-A * diff)
                    w_sum += w
                    v_sum += w * pix

            out[i, j] = v_sum / w_sum if w_sum > 0 else np.nan
    return out


def frost_filter_nanny(img, *, damping_factor: float = 2.0, win_size: int = 3,
                      dtype=np.float32):
    """
    Frost speckle filter accelerated with Numba (NaN-aware).

    Parameters
    ----------
    img : 2-D ndarray
        SAR intensity image (NaNs allowed).
    damping_factor : float, default 2.0
        Frost damping constant.
    win_size : int, default 3 (must be odd ≥ 3)
        Square window size.
    dtype : numpy dtype, default np.float32
        Internal working precision.

    Returns
    -------
    filtered : ndarray
        Frost-filtered image (same shape as *img*).
    """
    if img.ndim != 2:
        raise ValueError("Input must be a 2-D array.")
    if win_size < 3 or win_size % 2 == 0:
        raise ValueError("win_size must be odd and ≥ 3.")

    img = np.asarray(img, dtype=dtype, order='C')          # C-contiguous

    # ---- local mean & variance, ignoring NaNs ---- #
    nan_mask = np.isfinite(img).astype(dtype)              # 1 where valid, 0 where NaN
    img_filled = np.where(nan_mask, img, 0)

    # uniform_filter returns an *average* → multiply by win_size**2 to get sums
    size = win_size
    win_area = float(size * size)

    sum_valid = uniform_filter(img_filled, size=size, mode='reflect') * win_area
    sum_sq_valid = uniform_filter(img_filled * img_filled, size=size, mode='reflect') * win_area
    count_valid = uniform_filter(nan_mask, size=size, mode='reflect') * win_area

    # avoid division by zero where window has no valid pixels
    mean = np.where(count_valid > 0, sum_valid / count_valid, np.nan)
    var = np.where(count_valid > 0, sum_sq_valid / count_valid - mean * mean, np.nan)
    std = np.sqrt(np.maximum(var, 0))

    cv = std / mean                                     # coefficient of variation
    sigma_zero = cv / mean                              # std / mean²
    factor_A = damping_factor * sigma_zero
    factor_A = np.where(np.isfinite(factor_A), factor_A, 0)

    # ---- JIT kernel ---- #
    filtered = _frost_core_nanny(img, factor_A.astype(dtype), win_size)

    return filtered

