"""Dead / hot pixel inpainting — biharmonic prior (Bertalmio 2000).

Detector dead/hot pixels and dead columns are common: scintillator
defects, electronics failures, sensor manufacturing variation. This
recipe detects and inpaints them in one pass:

1. Build a fault mask: pixels at zero (dead) or above an outlier
   threshold (hot).
2. Solve the biharmonic equation under the mask to fill the holes
   (``skimage.restoration.inpaint_biharmonic``).

Reference:
    Bertalmio, M., Sapiro, G., Caselles, V., Ballester, C. (2000).
    *Image inpainting.* SIGGRAPH 2000, 417-424.
    https://doi.org/10.1145/344779.344972
"""

from __future__ import annotations

import numpy as np


def _build_fault_mask(arr: np.ndarray, hot_factor: float, dead_threshold: float) -> np.ndarray:
    """Detect dead (≈0) and hot (>= mean + hot_factor * sigma) pixels."""
    mean = float(arr.mean())
    sigma = float(arr.std())
    dead = arr <= dead_threshold
    hot = arr >= mean + hot_factor * sigma
    return dead | hot


def inpaint_dead_pixels(
    image: np.ndarray,
    hot_factor: float = 6.0,
    dead_threshold: float = 0.0,
) -> np.ndarray:
    """Detect dead/hot pixels and inpaint them with a biharmonic prior.

    Args:
        image: 2-D array.
        hot_factor: A pixel is hot if its value is at least
            ``mean + hot_factor * sigma``. 6.0 catches obvious outliers
            without flagging real bright structure.
        dead_threshold: A pixel is dead if its value is at or below
            this threshold. Default ``0.0`` catches the all-zero
            failures.

    Returns:
        Inpainted float32 array, same shape as input.
    """
    from skimage.restoration import inpaint_biharmonic

    if image.ndim != 2:
        raise ValueError(f"Expected 2-D image, got shape {image.shape}")

    arr = image.astype(np.float32, copy=False)
    mask = _build_fault_mask(arr, float(hot_factor), float(dead_threshold))

    if not mask.any():
        return arr.copy()

    out = inpaint_biharmonic(arr, mask, channel_axis=None)
    return np.asarray(out, dtype=np.float32)
