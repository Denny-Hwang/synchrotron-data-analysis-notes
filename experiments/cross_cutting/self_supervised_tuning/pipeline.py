"""Self-supervised denoiser calibration — Noise2Self J-invariance (Batson 2019).

The modern self-supervised denoising literature rests on one insight:
with pixel-independent noise, a denoiser can be scored **without any
clean reference** by hiding pixels from it and asking it to predict
them — the noise in the held-out pixels cannot be predicted, so the
self-supervised loss differs from the true (clean-target) loss only by
a constant. Noise2Self (Batson & Royer, ICML 2019) formalised this as
**J-invariance**; Noise2Void, Noise2Inverse (Hendriksen 2020) and the
2025 synchrotron bone-μCT application (Obata et al., J. Synchrotron
Rad. 32) are all members of the same family.

This recipe applies the principle in its CPU-friendly form, using
scikit-image's reference implementation
(``skimage.restoration.calibrate_denoiser``): sweep a classical
denoiser's strength, score each setting with the J-invariant
self-supervised loss, and denoise the full image at the winning
strength. No training, no weights, no ground truth — yet the Lab's
Impact card (which *does* hold a clean reference the algorithm never
saw) can verify the choice lands near the oracle.

References:
    Batson, J., Royer, L. (2019). *Noise2Self: Blind Denoising by
    Self-Supervision.* ICML. arXiv:1901.11365
    Hendriksen, A. A., Pelt, D. M., Batenburg, K. J. (2020).
    *Noise2Inverse: Self-Supervised Deep Convolutional Denoising for
    Tomography.* IEEE Trans. Comput. Imaging 6, 1320–1335.
    Obata, Y., Parkinson, D. Y., Pelt, D. M., Acevedo, C. (2025).
    *Enhancing synchrotron radiation micro-CT images using deep
    learning: an application of Noise2Inverse on bone imaging.*
    J. Synchrotron Rad. 32, 690–699.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _parameter_grid(method: str, n_candidates: int, max_strength: float) -> dict:
    strengths = np.linspace(
        float(max_strength) / n_candidates, float(max_strength), int(n_candidates)
    )
    if method == "tv":
        return {"weight": list(strengths)}
    if method == "gaussian":
        return {"sigma": list(strengths)}
    if method == "wavelet":
        return {"sigma": list(strengths)}
    raise ValueError(f"Unknown method {method!r}; expected tv | gaussian | wavelet")


def _denoise_function(method: str):
    if method == "tv":
        from skimage.restoration import denoise_tv_chambolle

        return lambda img, weight: denoise_tv_chambolle(img, weight=weight, channel_axis=None)
    if method == "gaussian":
        from scipy.ndimage import gaussian_filter

        return lambda img, sigma: gaussian_filter(img, sigma=sigma)
    if method == "wavelet":
        from skimage.restoration import denoise_wavelet

        return lambda img, sigma: denoise_wavelet(
            img, sigma=sigma, mode="soft", rescale_sigma=True, channel_axis=None
        )
    raise ValueError(f"Unknown method {method!r}")


def denoise_self_calibrated(
    image: np.ndarray,
    method: str = "tv",
    n_candidates: int = 8,
    max_strength: float = 0.4,
) -> np.ndarray:
    """Denoise at the strength chosen by the J-invariant self-supervised loss.

    Args:
        image: 2-D array (sinogram or image).
        method: ``"tv"`` (strength = TV weight), ``"gaussian"``
            (strength = sigma), or ``"wavelet"`` (strength = shrinkage
            sigma, BayesShrink-style soft thresholding).
        n_candidates: How many strengths to score, spaced linearly in
            ``(0, max_strength]``. More = finer optimum, slower.
        max_strength: Upper end of the sweep. Useful ranges:
            TV 0.05–0.5 · Gaussian 0.5–3 · wavelet 0.05–0.5 (in
            normalised units).

    Returns:
        The full image denoised at the self-selected strength, float32.
    """
    from skimage.restoration import calibrate_denoiser

    if image.ndim != 2:
        raise ValueError(f"Expected 2-D image, got shape {image.shape}")
    if n_candidates < 2:
        raise ValueError(f"n_candidates must be >= 2, got {n_candidates}")
    if max_strength <= 0:
        raise ValueError(f"max_strength must be > 0, got {max_strength}")

    # Work in normalised units so one strength scale serves all samples.
    arr = image.astype(np.float32, copy=False)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return arr.copy()
    norm = (arr - lo) / (hi - lo)

    fn = _denoise_function(method)
    grid = _parameter_grid(method, int(n_candidates), float(max_strength))

    # J-invariant calibration (Noise2Self): scores each parameter set on
    # masked pixels the denoiser could not see. ``extra_output`` hands us
    # the tested parameter sets + losses so we can apply the *plain*
    # denoiser at the winner (the returned calibrated function is the
    # J-invariant variant, which is slower and slightly blurrier).
    _, (params_tested, losses) = calibrate_denoiser(
        norm,
        fn,
        denoise_parameters=grid,
        extra_output=True,
    )
    best = params_tested[int(np.argmin(losses))]
    logger.info("self_supervised_tuning: method=%s picked %s", method, best)

    out = np.asarray(fn(norm, **best), dtype=np.float32)
    return (out * (hi - lo) + lo).astype(np.float32)
