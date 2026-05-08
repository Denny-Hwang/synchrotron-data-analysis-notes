"""Beam hardening correction — polynomial linearisation.

Polychromatic X-ray sources cause the *cupping artefact* in tomography:
soft photons are preferentially absorbed by high-attenuation rays,
making the sinogram non-linear in path length. The classical
correction is a polynomial mapping (Krumm et al. 2008) calibrated
once per material.

This recipe demonstrates the linearisation in a parameter-tunable form:
the user picks the polynomial coefficients and watches the |Δ| panel
fade as the calibration converges on the inverse of the simulated
hardening transform (`norm + 0.35·norm² − 0.25·norm³`).

Reference:
    Krumm, M., Kasperl, S., Franz, M. (2008). *Reducing non-linear
    artifacts of multi-material objects in industrial 3D computed
    tomography.* NDT & E International, 41(4), 242-251.
"""

from __future__ import annotations

import numpy as np


def correct_beam_hardening(
    image: np.ndarray,
    quad_coef: float = -0.30,
    cubic_coef: float = 0.20,
) -> np.ndarray:
    """Apply polynomial linearisation: f(x) = x + quad·x² + cubic·x³.

    Args:
        image: 2-D float / int array (sinogram or projection).
        quad_coef: Coefficient of the squared term. Negative values
            push the cupping back; positive deepen it.
        cubic_coef: Coefficient of the cubed term. Combined with
            ``quad_coef``, models the inverse of the bundled
            hardening simulation `norm + 0.35·norm² − 0.25·norm³`
            when set near (-0.30, +0.20).

    Returns:
        Linearised float32 array, same shape as input.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2-D image, got shape {image.shape}")

    arr = image.astype(np.float32, copy=False)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-12:
        return arr.copy()

    norm = (arr - lo) / (hi - lo)
    corrected = norm + float(quad_coef) * norm * norm + float(cubic_coef) * norm * norm * norm
    out = corrected * (hi - lo) + lo
    return out.astype(np.float32, copy=False)
