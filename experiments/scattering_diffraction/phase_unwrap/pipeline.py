"""2-D phase unwrapping — Herraez et al. (2002) quality-guided.

The single workhorse for phase-contrast imaging, coherent diffraction
imaging (CDI) exit-wave reconstruction, and InSAR. ``skimage`` ships
the canonical reliability-following implementation; this recipe wraps
it so users can compare the unwrapped output against a synthetic
ground truth phase.

References:
    Itoh, K. (1982). Analysis of the phase unwrapping algorithm.
    Applied Optics 21(14), 2470.

    Herráez, M. A., Burton, D. R., Lalor, M. J., Gdeisat, M. A. (2002).
    Fast two-dimensional phase-unwrapping algorithm based on sorting
    by reliability following a noncontinuous path.
    Applied Optics 41(35), 7437–7444.
    https://doi.org/10.1364/AO.41.007437
"""

from __future__ import annotations

import numpy as np


def unwrap_phase_2d(
    wrapped: np.ndarray,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Unwrap a 2-D wrapped-phase array via skimage's reliability-guided algorithm.

    Args:
        wrapped: 2-D float array of wrapped phase values in (-π, π].
        seed: Forwarded to ``skimage.restoration.unwrap_phase`` to make
            tie-breaking deterministic. Exposed so users can confirm
            the algorithm is reproducible. The skimage default would
            otherwise pull from ``numpy.random`` global state.

    Returns:
        Unwrapped phase, dtype float32, same shape as input.
    """
    from skimage.restoration import unwrap_phase

    if wrapped.ndim != 2:
        raise ValueError(f"Expected 2-D phase array, got shape {wrapped.shape}")

    arr = wrapped.astype(np.float32, copy=False)
    out = unwrap_phase(arr, rng=int(seed))
    return np.asarray(out, dtype=np.float32)
