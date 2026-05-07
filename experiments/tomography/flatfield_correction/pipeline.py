"""Flat-field correction — the textbook tomography preprocessing step.

Every X-ray detector has its own pixel-by-pixel response: gain
inhomogeneity, dust on the scintillator, vignetting. Flat-field
correction normalises each projection by an open-beam reference
(``flat``) and subtracts the dark-current floor (``dark``):

    corrected = (raw − dark) / (flat − dark)

This is the ``I0`` normalisation step every reconstruction pipeline
runs first. The bundled sample is a single projection from the
``flatfield_correction`` dataset; the dramatic improvement makes it
the most pedagogically useful recipe in the lab — users see *why*
flat-field correction matters.

Reference:
    Münch, B. et al. *Optics Express* 17, 8567-8591 (2009) — discusses
    flat-field artifacts (the zero-noise reconstruction baseline).
"""

from __future__ import annotations

import numpy as np


def correct_flatfield(
    image: np.ndarray,
    flat: np.ndarray | None = None,
    dark: np.ndarray | None = None,
    *,
    epsilon: float = 1.0,
    clip_low: float = 0.0,
    clip_high: float = 2.0,
) -> np.ndarray:
    """Apply standard flat / dark correction to a projection.

    Args:
        image: Raw projection (any dtype, treated as float).
        flat: Open-beam (no sample) flat field of the same shape. When
            ``None`` (UI default), the function computes a smooth-row
            substitute from ``image`` so the slider remains useful even
            when the user picks a sample without a paired flat.
        dark: Dark-current frame of the same shape. When ``None``, the
            minimum value of ``image`` is used.
        epsilon: Stabiliser added to the denominator to avoid divide-by-zero.
        clip_low, clip_high: Clip the corrected image to this range —
            cosmic rays + dust spots can produce wild outliers that
            blow up the dynamic range otherwise.
    """
    img = image.astype(np.float32, copy=False)
    if flat is None:
        # Pseudo-flat = column-wise mean of input. Useful for showcasing
        # the algorithm on samples without a paired flat reference.
        flat = np.broadcast_to(np.mean(img, axis=0, keepdims=True), img.shape).copy()
    else:
        flat = flat.astype(np.float32, copy=False)
    if dark is None:
        dark_val = float(img.min())
        dark = np.full_like(img, dark_val)
    else:
        dark = dark.astype(np.float32, copy=False)

    numer = img - dark
    denom = (flat - dark) + epsilon
    corrected = numer / denom
    if clip_low < clip_high:
        corrected = np.clip(corrected, clip_low, clip_high)
    return corrected.astype(np.float32)
