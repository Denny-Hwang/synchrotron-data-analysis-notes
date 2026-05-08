"""Vo 2018 sorting filter — neutron-CT variant.

Algorithmically identical to ``experiments.tomography.ring_artifact`` but
sized for the smaller, lower-SNR Sarepy neutron sinogram (459 × 503,
uint16). Default filter window is shorter (15 vs 21) because there are
fewer projection angles, so a long median can over-smooth.

Reference:
    Vo, N. T., Atwood, R. C., Drakopoulos, M. (2018) — Optics Express
    26(22), 28396. https://doi.org/10.1364/OE.26.028396

The neutron data were originally captured at NIST in collaboration
with the Sarepy authors and ship under Apache-2.0.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


def remove_stripe_neutron(sino: np.ndarray, size: int = 15) -> np.ndarray:
    """Sort-filter-unsort stripe removal for neutron tomography sinograms.

    Args:
        sino: 2-D sinogram, shape ``(n_angles, n_detectors)``.
        size: Median filter window length along the sorted axis. Must be
            odd and >= 3. Smaller default (15) than the X-ray recipe's 21
            because the neutron sinogram has many fewer angles (459 vs
            1801) so over-smoothing kicks in earlier.

    Returns:
        Sinogram with stripes attenuated; same shape and dtype as input.
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
