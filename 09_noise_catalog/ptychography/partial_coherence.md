# Partial Coherence Effects

## Classification

| Attribute | Value |
|-----------|-------|
| **Modality** | Ptychography |
| **Noise Type** | Instrumental |
| **Severity** | Major |
| **Frequency** | Common |
| **Detection Difficulty** | Hard |

## Visual Examples

```
Fully coherent illumination          Partially coherent illumination

  ┌──────────────────────┐           ┌──────────────────────┐
  │  Sharp fringes       │           │  Washed-out fringes  │
  │  ╱╲╱╲╱╲╱╲╱╲╱╲╱╲    │           │  ∿∿∿∿∿∿∿∿∿∿∿∿∿     │
  │  Clear phase edges   │    →      │  Blurred boundaries  │
  │  ──┐    ┌──          │           │  ──╲    ╱──          │
  │    │    │            │           │     ╲  ╱             │
  └──────────────────────┘           └──────────────────────┘

  FRC reaches detector edge           FRC drops at mid-frequency
```

> **External references:**
> - [Thibault & Menzel — mixed-state ptychography](https://doi.org/10.1038/nature11806)
> - [APS Upgrade coherence improvements](https://www.aps.anl.gov/APS-Upgrade)

## Description

Partial coherence effects arise when the illuminating X-ray beam deviates from the ideal assumption of a fully coherent plane wave or point source. Reduced spatial and temporal coherence washes out high-frequency interference features in diffraction patterns, causing contrast loss and resolution degradation in the reconstructed phase and amplitude images. These effects are particularly insidious because they produce smooth, plausible-looking reconstructions that simply lack fine detail, making the artifact harder to detect than gross distortions.

## Root Cause

Spatial coherence is limited by the finite source size: third-generation synchrotron sources have an emittance that produces a transverse coherence length often comparable to or smaller than the beam-defining optic aperture. Temporal (longitudinal) coherence is set by the monochromator bandwidth — a Si(111) monochromator provides ΔE/E ~ 1.4×10⁻⁴, yielding a longitudinal coherence length of ~1 μm. Mechanical vibrations of optical elements (monochromator, mirrors, zone plate) further reduce effective coherence by smearing the source position during exposure. When the coherence length is smaller than the illuminated region, the measured diffraction pattern becomes an incoherent sum of multiple coherent modes, violating the single-mode assumption used in standard ptychographic algorithms (ePIE, DM).

## Quick Diagnosis

```python
import numpy as np

# Load a diffraction pattern from the center of the scan
# dp = diffraction_data[len(diffraction_data)//2]
# Check fringe visibility at high spatial frequencies
radial_profile = np.mean(dp.reshape(-1)[np.argsort(
    np.hypot(*np.mgrid[-dp.shape[0]//2:dp.shape[0]//2,
                        -dp.shape[1]//2:dp.shape[1]//2]).reshape(2, -1)).ravel()
)].reshape(-1, 4), axis=1)  # crude radial binning
visibility = (radial_profile.max() - radial_profile.min()) / (radial_profile.max() + 1e-10)
print(f"Fringe visibility: {visibility:.3f} (< 0.3 suggests partial coherence)")
```

## Detection Methods

### Visual Indicators

- Diffraction patterns show reduced contrast in high-angle speckle compared to simulation.
- Reconstructed object appears smooth with suppressed fine features; edges look soft.
- Fourier ring correlation (FRC) between half-dataset reconstructions drops at spatial frequencies well below the detector-limited cutoff.
- Reconstructed probe appears broader than expected from the focusing optic geometry.
- The reconstruction error metric converges to a higher plateau than expected for the measured photon counts.

### Automated Detection

```python
import numpy as np
from scipy.ndimage import uniform_filter


def detect_partial_coherence(diffraction_data, expected_speckle_size=3,
                              visibility_threshold=0.15):
    """
    Detect partial coherence by analyzing speckle visibility in
    diffraction patterns.

    Parameters
    ----------
    diffraction_data : np.ndarray
        3D array (num_positions, ny, nx) of diffraction patterns.
    expected_speckle_size : int
        Expected speckle grain size in pixels (set by beam-defining aperture).
    visibility_threshold : float
        Minimum acceptable speckle visibility.

    Returns
    -------
    dict with keys:
        'mean_visibility' : float — average speckle visibility across patterns
        'visibility_per_pattern' : np.ndarray — per-pattern visibility
        'coherence_ratio' : float — estimated ratio of coherent modes
        'has_coherence_issue' : bool
    """
    num_patterns = diffraction_data.shape[0]
    ny, nx = diffraction_data.shape[1], diffraction_data.shape[2]
    visibilities = np.zeros(num_patterns)

    # Sample a subset for efficiency
    sample_idx = np.linspace(0, num_patterns - 1, min(50, num_patterns), dtype=int)

    for i, idx in enumerate(sample_idx):
        dp = diffraction_data[idx].astype(float)
        dp_sqrt = np.sqrt(dp + 1)  # Poisson-safe transform

        # Local mean and variance via uniform filter
        kernel = expected_speckle_size * 2 + 1
        local_mean = uniform_filter(dp_sqrt, size=kernel)
        local_var = uniform_filter(dp_sqrt**2, size=kernel) - local_mean**2

        # Speckle contrast = std / mean in annular region (exclude center and edges)
        cy, cx = ny // 2, nx // 2
        yy, xx = np.ogrid[:ny, :nx]
        r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        mask = (r > ny * 0.15) & (r < ny * 0.45)

        if np.sum(mask) > 100 and np.mean(local_mean[mask]) > 0:
            contrast = np.sqrt(np.mean(local_var[mask])) / np.mean(local_mean[mask])
            visibilities[i] = contrast
        else:
            visibilities[i] = np.nan

    valid = visibilities[~np.isnan(visibilities)]
    mean_vis = float(np.mean(valid)) if len(valid) > 0 else 0.0

    # For M incoherent modes, speckle contrast ~ 1/sqrt(M)
    # Fully coherent: contrast ~ 1.0; M modes: contrast ~ 1/sqrt(M)
    estimated_modes = max(1, int(round(1.0 / (mean_vis**2 + 1e-10))))

    return {
        "mean_visibility": mean_vis,
        "visibility_per_pattern": visibilities,
        "coherence_ratio": 1.0 / estimated_modes,
        "estimated_modes": estimated_modes,
        "has_coherence_issue": mean_vis < visibility_threshold,
    }
```

## Solutions and Mitigation

### Prevention (Before Data Collection)

- Use a coherence-preserving beamline optic layout; minimize aperture size to select the coherent fraction of the beam.
- Choose a higher-resolution monochromator (e.g., Si(220) or channel-cut) when temporal coherence is limiting.
- Reduce vibration of monochromator crystals and upstream optics with improved cooling and mechanical isolation.
- Upgrade to a fourth-generation source (MBA lattice) such as APS-U, ESRF-EBS, or MAX IV, which provide orders of magnitude higher coherent flux.
- Use shorter exposure times to reduce effective source smearing from vibrations.

### Correction — Traditional Methods

Mixed-state ptychography decomposes the illumination into a set of mutually incoherent probe modes, each reconstructed simultaneously. This recovers the object with correct contrast even under partial coherence.

```python
import numpy as np


def mixed_state_ePIE(diffraction_data, positions, num_modes=5,
                     num_iterations=100, probe_init=None, object_init=None):
    """
    Simplified mixed-state ePIE reconstruction handling partial coherence
    via multiple probe modes (Thibault & Menzel 2013).

    Parameters
    ----------
    diffraction_data : np.ndarray
        Measured patterns, shape (N, py, px).
    positions : np.ndarray
        Scan positions, shape (N, 2), in pixel units.
    num_modes : int
        Number of incoherent probe modes to reconstruct.
    num_iterations : int
        Reconstruction iterations.
    probe_init : np.ndarray or None
        Initial probe guess, shape (py, px). Modes are initialized from this.
    object_init : np.ndarray or None
        Initial object guess.

    Returns
    -------
    object_recon : np.ndarray — reconstructed complex object
    probe_modes : list of np.ndarray — reconstructed probe modes
    """
    N = len(diffraction_data)
    py, px = diffraction_data.shape[1], diffraction_data.shape[2]

    # Initialize probe modes with decreasing amplitude
    if probe_init is None:
        probe_init = np.ones((py, px), dtype=complex)
    probe_modes = []
    for m in range(num_modes):
        # Each higher mode starts with lower amplitude + random phase
        amplitude_scale = 1.0 / (m + 1)
        mode = probe_init * amplitude_scale
        if m > 0:
            mode *= np.exp(2j * np.pi * np.random.rand(py, px))
        probe_modes.append(mode.copy())

    # Initialize object
    obj_shape = (
        int(np.max(positions[:, 0])) + py + 10,
        int(np.max(positions[:, 1])) + px + 10,
    )
    if object_init is None:
        obj = np.ones(obj_shape, dtype=complex)
    else:
        obj = object_init.copy()

    alpha = 1.0  # ePIE step size

    for iteration in range(num_iterations):
        order = np.random.permutation(N)
        total_error = 0.0

        for idx in order:
            row = int(round(positions[idx, 0]))
            col = int(round(positions[idx, 1]))
            obj_patch = obj[row:row + py, col:col + px]

            # Forward propagate all modes
            exit_waves = [pm * obj_patch for pm in probe_modes]
            exit_fts = [np.fft.fft2(ew) for ew in exit_waves]

            # Incoherent sum of mode intensities
            intensity_sum = sum(np.abs(ef)**2 for ef in exit_fts)
            measured_amp = np.sqrt(diffraction_data[idx].astype(float))

            # Compute scaling factor per pixel
            calc_amp = np.sqrt(intensity_sum + 1e-15)
            total_error += np.sum((calc_amp - measured_amp)**2)

            # Apply Fourier constraint: scale each mode proportionally
            for m in range(num_modes):
                corrected_ft = exit_fts[m] * measured_amp / (calc_amp + 1e-15)
                exit_corrected = np.fft.ifft2(corrected_ft)
                diff = exit_corrected - exit_waves[m]

                # Update object
                obj[row:row + py, col:col + px] += (
                    alpha * np.conj(probe_modes[m])
                    / (np.max(np.abs(probe_modes[m]))**2 + 1e-8) * diff
                )

                # Update probe mode
                probe_modes[m] += (
                    alpha * np.conj(obj_patch)
                    / (np.max(np.abs(obj_patch))**2 + 1e-8) * diff
                )

        if iteration % 10 == 0:
            print(f"Iteration {iteration}: error = {total_error:.4e}")

    return obj, probe_modes
```

### Correction — AI/ML Methods

Neural network approaches can learn to deconvolve partial coherence effects from reconstructed images. Autoencoders trained on paired data (partially coherent reconstruction vs. fully coherent ground truth) can sharpen features and restore lost contrast. Additionally, physics-informed neural networks can parameterize the coherence function and jointly optimize it with the object during reconstruction, potentially reducing the number of required probe modes.

## Impact If Uncorrected

Partial coherence reduces the effective resolution and phase contrast of ptychographic reconstructions, often by a factor of 2-10x relative to the diffraction limit. Fine features such as grain boundaries, nanoscale porosity, or thin-film interfaces may be completely invisible in the reconstruction. Quantitative phase measurements become unreliable because the phase transfer function is attenuated at high spatial frequencies. If the standard single-mode algorithm is used despite significant partial coherence, the reconstruction may converge to a smooth, biased solution that appears plausible but is quantitatively incorrect — a particularly dangerous failure mode for materials science and biological imaging applications.

## Related Resources

- [Position Error](position_error.md) — position errors compound with coherence issues, making both harder to diagnose
- [Stitching Artifact](stitching_artifact.md) — coherence variations across the scan area affect tile-to-tile consistency
- [Ptychography overview](../../01_modalities/ptychography.md) — fundamentals of coherent diffraction imaging
- [Thibault & Menzel 2013](https://doi.org/10.1038/nature11806) — original mixed-state ptychography paper
- [PtyPy — python ptychography framework](https://github.com/ptycho/ptypy) — includes mixed-state reconstruction

## Key Takeaway

Partial coherence is an unavoidable reality at most synchrotron beamlines and silently degrades reconstruction quality. Always use mixed-state ptychography with at least 3-5 probe modes as a default — the computational overhead is modest, and it prevents the insidious loss of resolution and contrast that occurs when partial coherence is ignored.
