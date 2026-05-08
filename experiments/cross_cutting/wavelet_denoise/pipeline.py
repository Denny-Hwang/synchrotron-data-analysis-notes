"""Wavelet shrinkage denoising — generic baseline.

Same algorithm as ``experiments.tomography.low_dose_denoise`` but
exposed as a *cross-cutting* recipe so users can compare it side-by-side
against TV / NLM / bilateral on the same image. This is the textbook
``denoise everything via wavelet thresholding`` baseline.

Reference:
    Donoho, D. L., Johnstone, J. M. (1994). *Ideal spatial adaptation
    by wavelet shrinkage.* Biometrika 81(3), 425-455.
"""

from __future__ import annotations

import numpy as np


def denoise_wavelet_generic(
    image: np.ndarray,
    method: str = "BayesShrink",
    mode: str = "soft",
    wavelet: str = "sym8",
    levels: int = 4,
) -> np.ndarray:
    """Wavelet shrinkage denoising. Wraps ``skimage.restoration.denoise_wavelet``.

    Args:
        image: 2-D float / int array.
        method: ``BayesShrink`` (level-adaptive) or ``VisuShrink``
            (universal threshold).
        mode: ``soft`` or ``hard`` thresholding.
        wavelet: PyWavelets family name.
        levels: Decomposition depth.

    Returns:
        Denoised float32 array, same shape as input.
    """
    from skimage.restoration import denoise_wavelet

    if image.ndim != 2:
        raise ValueError(f"Expected 2-D image, got shape {image.shape}")
    if levels < 1:
        raise ValueError(f"levels must be >= 1, got {levels}")

    arr = image.astype(np.float32, copy=False)
    out = denoise_wavelet(
        arr,
        wavelet=str(wavelet),
        mode=str(mode),
        wavelet_levels=int(levels),
        method=str(method),
        rescale_sigma=True,
    )
    return np.asarray(out, dtype=np.float32)
