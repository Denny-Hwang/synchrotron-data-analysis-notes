"""Ring artifact — dead-stripe detection + interpolation (Vo 2018, algorithm 4-6).

Dead or unresponsive detector columns produce *unrecoverable* stripes:
no filter can restore information that was never measured. Vo, Atwood &
Drakopoulos (2018) therefore treat them differently from filterable
stripes — **detect** the defective columns robustly, then **replace**
them by interpolating from their healthy neighbours.

Detection here follows the spirit of Sarepy's ``detect_stripe``: build
a per-column response curve (the column means), divide out its
median-filtered baseline, and flag columns whose normalised response
deviates from 1 by more than ``snr`` robust standard deviations (MAD).
Flagged columns are then rebuilt row-by-row with linear interpolation
across the nearest good columns.

Reference:
    Vo, N. T., Atwood, R. C., Drakopoulos, M. (2018). *Superior
    techniques for eliminating ring artifacts in X-ray micro-tomography.*
    Optics Express 26(22), 28396–28412.
    https://doi.org/10.1364/OE.26.028396
"""

from __future__ import annotations

import numpy as np


def remove_stripe_interpolation(
    sinogram: np.ndarray,
    snr: float = 3.0,
    size: int = 31,
) -> np.ndarray:
    """Detect defective detector columns and rebuild them by interpolation.

    Args:
        sinogram: 2-D array, rows = projection angles.
        snr: Detection sensitivity — a column is flagged when its
            baseline-normalised response deviates by more than ``snr``
            robust standard deviations. Lower = more aggressive
            (flags more columns).
        size: Window (detector pixels) of the median filter that forms
            the stripe-free baseline. Must comfortably exceed the widest
            stripe; too large starts flattening real trends.

    Returns:
        Corrected float32 sinogram, same shape.
    """
    from scipy.ndimage import median_filter

    if sinogram.ndim != 2:
        raise ValueError(f"Expected 2-D sinogram, got shape {sinogram.shape}")
    if snr <= 0:
        raise ValueError(f"snr must be > 0, got {snr}")
    if size < 3:
        raise ValueError(f"size must be >= 3, got {size}")

    arr = sinogram.astype(np.float32, copy=False)
    ncol = arr.shape[1]

    # Column response vs its stripe-free baseline.
    response = arr.mean(axis=0)
    baseline = median_filter(response, size=int(size), mode="nearest")
    eps = 1e-6 * max(1.0, float(np.abs(baseline).max()))
    ratio = response / np.where(np.abs(baseline) < eps, eps, baseline)

    dev = np.abs(ratio - 1.0)
    mad = float(np.median(np.abs(dev - np.median(dev))))
    robust_std = 1.4826 * mad if mad > 0 else float(dev.std())
    if robust_std <= 0:
        return arr.copy()
    bad = dev > float(snr) * robust_std

    if not bad.any() or bad.all():
        # Nothing detected (or everything flagged — detection failed):
        # return the input unchanged rather than fabricating data.
        return arr.copy()

    good_idx = np.flatnonzero(~bad)
    bad_idx = np.flatnonzero(bad)
    cols = np.arange(ncol)

    corrected = arr.copy()
    # Rebuild every defective column row-by-row from the healthy ones.
    for row in range(arr.shape[0]):
        corrected[row, bad_idx] = np.interp(cols[bad_idx], good_idx, arr[row, good_idx])
    return corrected.astype(np.float32)
