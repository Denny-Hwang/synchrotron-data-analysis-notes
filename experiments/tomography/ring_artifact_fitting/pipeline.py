"""Ring artifact — fitting-based stripe removal (Vo 2018, algorithm 1).

Vo, Atwood & Drakopoulos (2018) classify sinogram stripes and give a
dedicated remover per class. The **fitting-based** method targets
*full* stripes sitting on a smooth background: fit a low-order
polynomial to every detector column along the projection-angle axis,
smooth the fitted surface across the detector axis, and rescale the
sinogram by the ratio of the two — columns whose response deviates from
their smoothed neighbourhood (the stripes) are pulled back onto it.

Adapted from Sarepy's ``remove_stripe_based_fitting`` (Apache-2.0),
the reference implementation that also ships in Algotom and TomoPy —
the current production standard at tomography beamlines.

Reference:
    Vo, N. T., Atwood, R. C., Drakopoulos, M. (2018). *Superior
    techniques for eliminating ring artifacts in X-ray micro-tomography.*
    Optics Express 26(22), 28396–28412.
    https://doi.org/10.1364/OE.26.028396
"""

from __future__ import annotations

import numpy as np


def remove_stripe_fitting(
    sinogram: np.ndarray,
    order: int = 2,
    sigma: float = 5.0,
) -> np.ndarray:
    """Fitting-based stripe removal on a (angles, detector) sinogram.

    Args:
        sinogram: 2-D array, rows = projection angles.
        order: Polynomial order fitted to each column along the angle
            axis. 1–2 captures the smooth intensity trend; higher
            orders start absorbing real structure.
        sigma: Gaussian smoothing width (in detector pixels) applied to
            the fitted surface across the detector axis. Larger =
            stronger stripe suppression, more risk of bleeding into
            real features.

    Returns:
        Corrected float32 sinogram, same shape.
    """
    from scipy.ndimage import gaussian_filter

    if sinogram.ndim != 2:
        raise ValueError(f"Expected 2-D sinogram, got shape {sinogram.shape}")
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")

    arr = sinogram.astype(np.float32, copy=False)
    nrow = arr.shape[0]

    # Fit each detector column with a polynomial along the angle axis.
    x = np.arange(nrow, dtype=np.float32)
    coeffs = np.polynomial.polynomial.polyfit(x, arr, int(order))
    sinofit = np.polynomial.polynomial.polyval(x, coeffs).T.astype(np.float32)

    # Smooth the fitted surface across the detector axis only — this is
    # what the stripe-free response *should* look like.
    sinofit_smooth = gaussian_filter(sinofit, sigma=(0.0, float(sigma)))

    # Rescale so the correction is mean-preserving, then pull each
    # column's fitted response onto the smoothed one.
    num1 = float(np.mean(sinofit))
    num2 = float(np.mean(sinofit_smooth))
    if abs(num2) > 1e-12:
        sinofit_smooth = sinofit_smooth * (num1 / num2)

    eps = 1e-6 * max(1.0, float(np.abs(sinofit).max()))
    corrected = arr / np.where(np.abs(sinofit) < eps, eps, sinofit) * sinofit_smooth
    return corrected.astype(np.float32)
