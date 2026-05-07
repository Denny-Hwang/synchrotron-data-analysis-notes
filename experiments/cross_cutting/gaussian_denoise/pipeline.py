"""Gaussian / median denoising — the baseline every paper compares to.

Before reaching for U-Net or noise2void, the practitioner usually
runs a classical low-pass filter and decides whether the trade-off
(noise reduction vs. resolution loss) is acceptable. This recipe
exposes the two most common baselines on the same UI:

* **Gaussian blur** — convolution with a 2-D Gaussian kernel of
  standard deviation ``sigma``. Linear, separable, fast.
* **Median filter** — 2-D moving median with footprint ``size``.
  Non-linear, edge-preserving on impulse noise.

The pedagogical value is direct: users see *why* learned methods
exist (the baselines blur edges) and what they have to beat.
"""

from __future__ import annotations

import numpy as np


def denoise_classical(
    image: np.ndarray,
    method: str = "gaussian",
    sigma: float = 1.5,
    size: int = 3,
) -> np.ndarray:
    """Apply a classical denoising baseline to a 2-D image.

    Args:
        image: 2-D float / int array.
        method: ``"gaussian"`` or ``"median"``.
        sigma: Gaussian kernel σ (only used when ``method='gaussian'``).
        size: Median filter footprint side length, in pixels (only used
            when ``method='median'``). Should be odd; even values are
            bumped up by one inside scipy.
    """
    arr = image.astype(np.float32, copy=False)
    if method == "median":
        from scipy.ndimage import median_filter

        return median_filter(arr, size=int(max(1, size))).astype(np.float32)
    # Default: gaussian
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(arr, sigma=float(max(0.0, sigma))).astype(np.float32)
