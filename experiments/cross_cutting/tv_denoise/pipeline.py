"""Total-Variation denoising — Chambolle (2004).

The standard edge-preserving denoiser. TV minimisation suppresses
high-frequency speckle while preserving the sharp boundaries that a
Gaussian blur would smear. In synchrotron pipelines TV is the go-to
choice when a learned model is unavailable and edges (sample-air
boundary, grain boundaries) must be kept.

Reference:
    Chambolle, A. (2004). *An algorithm for total variation minimization
    and applications.* Journal of Mathematical Imaging and Vision,
    20(1-2), 89–97. https://doi.org/10.1023/B:JMIV.0000011321.19549.88
"""

from __future__ import annotations

import numpy as np


def denoise_tv(
    image: np.ndarray,
    weight: float = 0.1,
    max_iter: int = 200,
) -> np.ndarray:
    """TV-Chambolle denoising. Wraps ``skimage.restoration.denoise_tv_chambolle``.

    Args:
        image: 2-D array (any numeric dtype).
        weight: TV regularisation weight. Larger = smoother / more edges
            preserved at the cost of contrast loss; smaller = closer to
            the original. Useful range 0.05-0.5.
        max_iter: Iteration cap for the optimisation. 200 is plenty
            for typical sinograms.

    Returns:
        Denoised float32 array of the same shape as ``image``.
    """
    from skimage.restoration import denoise_tv_chambolle

    if image.ndim != 2:
        raise ValueError(f"Expected 2-D image, got shape {image.shape}")
    if weight <= 0:
        raise ValueError(f"weight must be > 0, got {weight}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")

    arr = image.astype(np.float32, copy=False)
    out = denoise_tv_chambolle(
        arr,
        weight=float(weight),
        max_num_iter=int(max_iter),
        channel_axis=None,
    )
    return np.asarray(out, dtype=np.float32)
