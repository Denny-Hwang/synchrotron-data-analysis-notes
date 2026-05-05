"""Sorting-based stripe / ring artifact removal — pipeline.

Implements the canonical sorting-based filter from:

    Vo, N. T., Atwood, R. C., & Drakopoulos, M. (2018).
    Superior techniques for eliminating ring artifacts in X-ray micro-tomography.
    Optics Express, 26(22), 28396-28412.
    https://doi.org/10.1364/OE.26.028396

Algorithm: For each detector column (axis 1), sort across angles (axis 0),
apply a 1-D median filter along the sorted axis, then revert ordering.
This smooths the persistent column-wise response (the stripe) while
preserving sample structure that is uncorrelated across angles.

Pure function — no Streamlit, no I/O. The recipe.yaml in this folder
declares parameters, samples, and metrics.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


def remove_stripe_based_sorting(sino: np.ndarray, size: int = 21) -> np.ndarray:
    """Remove stripe / ring artifacts from a sinogram via sorting + median filter.

    Args:
        sino: 2-D sinogram, shape ``(n_angles, n_detectors)``. Any numeric
            dtype; output preserves the input dtype.
        size: Median filter window length along the sorted axis. Must be
            odd and at least 3. Larger values smooth more aggressively
            but blur fine angular features.

    Returns:
        Sinogram with stripes attenuated, same shape and dtype as input.

    Raises:
        ValueError: If ``sino`` is not 2-D, or ``size`` is even or < 3.
    """
    if sino.ndim != 2:
        raise ValueError(f"Expected 2-D sinogram, got shape {sino.shape}")
    if size < 3 or size % 2 == 0:
        raise ValueError(f"size must be odd and >= 3, got {size}")

    orig_dtype = sino.dtype
    arr = sino.astype(np.float32, copy=False)

    sortidx = np.argsort(arr, axis=0)
    invidx = np.argsort(sortidx, axis=0)
    sorted_arr = np.take_along_axis(arr, sortidx, axis=0)
    smoothed = median_filter(sorted_arr, size=(size, 1))
    out = np.take_along_axis(smoothed, invidx, axis=0)

    return out.astype(orig_dtype, copy=False)
