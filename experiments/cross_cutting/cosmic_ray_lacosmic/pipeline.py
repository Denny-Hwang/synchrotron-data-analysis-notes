"""L.A.Cosmic — Laplacian-edge cosmic ray detection pipeline.

Wraps :func:`astroscrappy.detect_cosmics` with the kwargs the recipe.yaml
exposes to the UI. Returns the cleaned image only; the boolean mask is
discarded so the pipeline contract stays ``ndarray → ndarray``.

Why this matters for synchrotron data: the same Laplacian-edge mechanism
identifies **zingers** in X-ray tomography (gamma rays leaking through
shielding), **outlier spectra** in spectroscopy, and **dead/hot pixels**
in XRF maps. The astronomical CCD frame in ``gmos.fits`` is included as
a clean, license-permissive proxy for these cases.

Reference:
    van Dokkum, P. G. (2001). Cosmic-ray rejection by Laplacian edge
    detection. *PASP, 113*, 1420-1427. https://doi.org/10.1086/323894
"""

from __future__ import annotations

import numpy as np


def detect_cosmics_lacosmic(
    image: np.ndarray,
    sigclip: float = 4.5,
    objlim: float = 5.0,
    niter: int = 4,
    readnoise: float = 6.5,
) -> np.ndarray:
    """Detect and clean cosmic rays via the L.A.Cosmic algorithm.

    Args:
        image: 2-D image array. Any numeric dtype; converted to float32
            internally. The function expects values to be in *counts*
            (sky background should NOT be subtracted), so noise can be
            estimated correctly.
        sigclip: Detection threshold in units of sigma. Lower values flag
            more pixels (more aggressive). Typical: 4.0-5.0.
        objlim: Limit on the contrast between the Laplacian and the
            fine-structure image. Higher values protect real point sources
            from being flagged as cosmic rays. Typical: 4.0-6.0.
        niter: Number of detection-and-cleaning iterations. More iterations
            catch more cosmic rays but slow the pipeline.
        readnoise: Detector read noise in counts. Affects the per-pixel
            noise model.

    Returns:
        Cleaned 2-D image, same shape as input, dtype float32.
    """
    from astroscrappy import detect_cosmics  # lazy import — heavy C ext.

    if image.ndim != 2:
        raise ValueError(f"Expected 2-D image, got shape {image.shape}")
    if niter < 1:
        raise ValueError(f"niter must be >= 1, got {niter}")

    arr = np.ascontiguousarray(image, dtype=np.float32)
    _crmask, cleaned = detect_cosmics(
        arr,
        sigclip=float(sigclip),
        objlim=float(objlim),
        niter=int(niter),
        readnoise=float(readnoise),
        cleantype="meanmask",
        verbose=False,
    )
    return np.asarray(cleaned, dtype=np.float32)
