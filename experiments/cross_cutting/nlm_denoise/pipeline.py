"""Non-Local Means denoising — Buades, Coll & Morel (2005).

Replaces every pixel with a weighted average of *similar patches*
elsewhere in the image. Repeating sample features (grain edges, fibre
cross-sections) get strong noise suppression because their many
copies average out, while unique features keep most of their
amplitude. Heavier than Gaussian/TV but excellent on textured
synchrotron data.

Reference:
    Buades, A., Coll, B., Morel, J.-M. (2005). *A non-local algorithm
    for image denoising.* CVPR 2005, 60-65.
    https://doi.org/10.1109/CVPR.2005.38
"""

from __future__ import annotations

import numpy as np


def denoise_nlm(
    image: np.ndarray,
    h_factor: float = 1.0,
    patch_size: int = 5,
    patch_distance: int = 6,
    fast_mode: bool = True,
) -> np.ndarray:
    """Non-Local Means denoising. Wraps ``skimage.restoration.denoise_nl_means``.

    Args:
        image: 2-D float / int array.
        h_factor: Multiplier on the auto-estimated noise σ. Skimage uses
            ``h ≈ 0.6 σ`` by default; ``h_factor`` lets the user push
            it more aggressive (>1) or more conservative (<1).
        patch_size: Side length of the comparison patch (odd). 5–7 is
            typical; smaller catches finer detail, larger averages more.
        patch_distance: Search-window radius. Larger = more candidate
            patches considered = stronger denoising, but quadratic cost.
        fast_mode: Skimage's vectorised implementation; ~20× faster
            with negligible quality difference.

    Returns:
        Denoised float32 array, same shape as input.
    """
    from skimage.restoration import denoise_nl_means, estimate_sigma

    if image.ndim != 2:
        raise ValueError(f"Expected 2-D image, got shape {image.shape}")
    if patch_size < 3 or patch_size % 2 == 0:
        raise ValueError(f"patch_size must be odd and >= 3, got {patch_size}")

    arr = image.astype(np.float32, copy=False)
    sigma = float(estimate_sigma(arr, average_sigmas=True))
    h = max(0.6 * sigma * float(h_factor), 1e-6)

    out = denoise_nl_means(
        arr,
        patch_size=int(patch_size),
        patch_distance=int(patch_distance),
        h=h,
        fast_mode=bool(fast_mode),
        sigma=sigma,
    )
    return np.asarray(out, dtype=np.float32)
