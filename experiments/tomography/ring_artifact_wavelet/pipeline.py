"""Combined wavelet-Fourier stripe / ring artifact removal.

Implementation of Münch et al. (2009) "Stripe and ring artifact removal
with combined wavelet-Fourier filtering". Optics Express 17(10):
8567-8591. https://doi.org/10.1364/OE.17.008567

Algorithm (sketch):

1. Decompose the sinogram with a discrete wavelet transform (DWT) up
   to ``level`` levels. Stripes accumulate in the **vertical-detail**
   coefficient at each level because they are horizontally localised
   (one or a few detector columns) and vertically extended (across
   many projection angles).
2. For each level's vertical-detail band, take the 1-D FFT along the
   column axis. The persistent column response — the stripe — appears
   as a low-frequency component in this 1-D spectrum.
3. Multiply by a Gaussian damping
   ``1 - exp(-k^2 / (2 sigma^2))`` that zeroes the DC component and
   leaves higher frequencies untouched.
4. Inverse FFT, then inverse DWT.

Pure function — no Streamlit, no I/O.
"""

from __future__ import annotations

import numpy as np
from numpy.fft import fft, fftshift, ifft, ifftshift


def remove_stripe_wavelet_fft(
    sino: np.ndarray,
    level: int = 4,
    sigma: float = 2.0,
    wname: str = "db5",
) -> np.ndarray:
    """Munch et al. (2009) combined wavelet-Fourier stripe removal.

    Args:
        sino: 2-D sinogram, shape ``(n_angles, n_detectors)``.
        level: Number of wavelet decomposition levels. Higher = wider
            spatial scales of stripe targeted; typical 3-6.
        sigma: Width of the Gaussian damping applied to the FFT of the
            vertical-detail coefficient. Smaller values target
            persistent (low-frequency) stripes more aggressively;
            typical 1-4.
        wname: PyWavelets wavelet family name. ``db5`` (Daubechies 5)
            is the recommended default in the original paper. ``haar``
            is faster but coarser.

    Returns:
        Sinogram with stripes attenuated, same shape and dtype as input.

    Raises:
        ValueError: If ``sino`` is not 2-D or ``level`` is non-positive.
    """
    import pywt  # lazy import — heavy native library

    if sino.ndim != 2:
        raise ValueError(f"Expected 2-D sinogram, got shape {sino.shape}")
    if level < 1:
        raise ValueError(f"level must be >= 1, got {level}")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")

    orig_dtype = sino.dtype
    arr = sino.astype(np.float32, copy=False)

    # Forward DWT — wavedec2 handles odd-size padding bookkeeping for us.
    coeffs = pywt.wavedec2(arr, wavelet=wname, level=level)
    cA = coeffs[0]
    details = coeffs[1:]  # list of (cH, cV, cD) tuples, one per level

    # Process each level's vertical-detail coefficient.
    new_details: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for cH, cV, cD in details:
        nrows = cV.shape[0]
        # FFT along axis 0 (the column / detector axis after rotation).
        fV = fftshift(fft(cV, axis=0), axes=0)
        k = np.arange(nrows) - nrows // 2
        damping = 1.0 - np.exp(-(k * k) / (2.0 * sigma * sigma))
        fV *= damping[:, None]
        cV_new = np.real(ifft(ifftshift(fV, axes=0), axis=0))
        new_details.append((cH, cV_new, cD))

    # Inverse DWT.
    new_coeffs: list = [cA, *new_details]
    out = pywt.waverec2(new_coeffs, wavelet=wname)

    # Trim back to original shape (waverec2 may pad by 1 for odd sizes).
    out = out[: arr.shape[0], : arr.shape[1]]
    return out.astype(orig_dtype, copy=False)
