"""Bilateral filter — Tomasi & Manduchi (1998).

Edge-preserving smoothing that averages each pixel with neighbours
weighted by **both** spatial proximity and intensity similarity. A
pixel on a sharp edge is not averaged with its neighbour across the
edge because their intensities differ; a pixel in a flat region
averages everything around it. Faster than NLM and TV, slower than
Gaussian.

Reference:
    Tomasi, C., Manduchi, R. (1998). *Bilateral filtering for gray
    and color images.* ICCV 1998, 839-846.
    https://doi.org/10.1109/ICCV.1998.710815
"""

from __future__ import annotations

import numpy as np


def denoise_bilateral(
    image: np.ndarray,
    sigma_spatial: float = 1.5,
    sigma_color_factor: float = 1.0,
    win_size: int = 7,
) -> np.ndarray:
    """Bilateral filter. Wraps ``skimage.restoration.denoise_bilateral``.

    Args:
        image: 2-D float array.
        sigma_spatial: Gaussian width for spatial weighting (pixels).
        sigma_color_factor: Multiplier on the auto-estimated noise σ
            for the intensity-similarity weighting. >1 = blur across
            more edges; <1 = sharper edge preservation.
        win_size: Kernel window size (odd). Larger = more candidates
            considered = stronger denoising, more cost.

    Returns:
        Denoised float32 array, same shape as input.
    """
    from skimage.restoration import denoise_bilateral, estimate_sigma

    if image.ndim != 2:
        raise ValueError(f"Expected 2-D image, got shape {image.shape}")
    if win_size < 3 or win_size % 2 == 0:
        raise ValueError(f"win_size must be odd and >= 3, got {win_size}")
    if sigma_spatial <= 0:
        raise ValueError(f"sigma_spatial must be > 0, got {sigma_spatial}")

    arr = image.astype(np.float32, copy=False)
    auto_sigma = float(estimate_sigma(arr, average_sigmas=True))
    sigma_color = max(auto_sigma * float(sigma_color_factor), 1e-6)

    out = denoise_bilateral(
        arr,
        win_size=int(win_size),
        sigma_color=sigma_color,
        sigma_spatial=float(sigma_spatial),
        channel_axis=None,
    )
    return np.asarray(out, dtype=np.float32)
