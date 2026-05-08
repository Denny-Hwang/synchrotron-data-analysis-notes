"""Low-dose / photon-counting denoising — wavelet shrinkage baseline.

Liu et al. (2020) *TomoGAN* showed that learned denoisers can recover
detail from sub-12-photon-per-pixel synchrotron tomography. We cannot
ship the trained GAN weights inside the lab (CC BY-NC), but we can
ship the **classical wavelet-shrinkage baseline** the paper compares
against.

The pipeline routes to `skimage.restoration.denoise_wavelet` with
choice of soft / hard / Bayesian shrinkage modes (Donoho & Johnstone
1994, Chang et al. 2000). Sigma is auto-estimated when the parameter
is at its default (`auto`) or fixed manually.

Reference:
    Liu, Z., Bicer, T., Kettimuthu, R., Gursoy, D., De Carlo, F.,
    Foster, I. (2020). *TomoGAN: low-dose synchrotron x-ray tomography
    with generative adversarial networks.* JOSA A 37(3), 422–434.
    https://doi.org/10.1364/JOSAA.375595

    Donoho, D. L., Johnstone, J. M. (1994). *Ideal spatial adaptation
    by wavelet shrinkage.* Biometrika 81(3), 425–455.
"""

from __future__ import annotations

import numpy as np


def denoise_wavelet_shrinkage(
    image: np.ndarray,
    method: str = "BayesShrink",
    mode: str = "soft",
    wavelet: str = "db5",
    levels: int = 4,
    sigma_factor: float = 1.0,
) -> np.ndarray:
    """Wavelet-shrinkage denoising with auto sigma estimation.

    Args:
        image: 2-D float / int array (transmission sinogram, projection,
            or reconstruction slice).
        method: ``BayesShrink`` (level-adaptive, recommended for
            unknown noise) or ``VisuShrink`` (universal threshold,
            stronger smoothing).
        mode: ``soft`` (Donoho & Johnstone — recommended) or ``hard``
            thresholding. Hard preserves sharp edges but is unstable.
        wavelet: PyWavelets family name. ``db5`` matches the wavelet-FFT
            ring-removal recipe; try ``sym8`` for slightly less
            ringing, ``haar`` for faster but blockier.
        levels: Wavelet decomposition depth. Higher = wider scales of
            noise targeted, but eats real low-frequency structure.
        sigma_factor: Multiplier on the estimated noise σ. <1 keeps
            more noise (preserves texture); >1 smooths more (kills
            noise but blurs detail).

    Returns:
        Denoised 2-D array, dtype float32, same shape as input.
    """
    from skimage.restoration import denoise_wavelet, estimate_sigma

    if image.ndim != 2:
        raise ValueError(f"Expected 2-D image, got shape {image.shape}")
    if levels < 1:
        raise ValueError(f"levels must be >= 1, got {levels}")
    if sigma_factor <= 0:
        raise ValueError(f"sigma_factor must be > 0, got {sigma_factor}")

    arr = image.astype(np.float32, copy=False)

    # Per skimage docs, sigma is None ⇒ auto-estimated. We multiply by
    # ``sigma_factor`` so the user can be more aggressive / conservative
    # than the auto estimate.
    auto_sigma = float(estimate_sigma(arr, average_sigmas=True))
    target_sigma = auto_sigma * float(sigma_factor)

    out = denoise_wavelet(
        arr,
        sigma=target_sigma,
        wavelet=str(wavelet),
        mode=str(mode),
        wavelet_levels=int(levels),
        method=str(method),
        rescale_sigma=False,
    )
    return np.asarray(out, dtype=np.float32)
