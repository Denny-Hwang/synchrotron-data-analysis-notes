# Position Error

## Classification

| Attribute | Value |
|-----------|-------|
| **Modality** | Ptychography |
| **Noise Type** | Systematic |
| **Severity** | Critical |
| **Frequency** | Common |
| **Detection Difficulty** | Hard |

## Visual Examples

```
Expected (correct positions)      Observed (position errors)

  +---+---+---+---+               +---+--+----+---+
  | . | . | . | . |               | . |. |  . | . |
  +---+---+---+---+        →     +--+---+--+----+
  | . | . | . | . |               |  .| .  |. |  .|
  +---+---+---+---+               +---+--+---+----+

  Sharp reconstruction             Blurred / ghosted features
  with clear edges                 with doubled edges
```

> **External references:**
> - [Odstrcil et al. — Ptychographic position correction](https://doi.org/10.1038/s41467-023-41496-z)
> - [PtychoShelves — modular ptychography framework](https://github.com/PtychoShelves)

## Description

Position errors occur when the recorded scan positions of the probe deviate from the true physical positions on the sample. This mismatch causes the iterative reconstruction algorithm to incorrectly overlap diffraction patterns, leading to blurred features, ghosting, and failure to converge. Position errors are one of the most common and damaging artifacts in scanning ptychography, particularly at high resolution where sub-pixel accuracy is required.

## Root Cause

The primary sources of position error are interferometer drift, piezo stage hysteresis and creep, encoder quantization errors, and thermal expansion of the sample stage. Vibrations from the facility (floor, pumps, cryocoolers) introduce high-frequency jitter that smears the effective probe position during each exposure. At nanometer-scale resolution, even a few nanometers of untracked motion can exceed the overlap constraint required for unique phase retrieval. The problem is exacerbated during long scans where cumulative drift can reach tens of nanometers.

## Quick Diagnosis

```python
import numpy as np

# Load recorded scan positions (N x 2 array: [y, x] in meters)
# positions = np.load('scan_positions.npy')
# Check for irregular spacing that indicates stage errors
dy = np.diff(positions[:, 0])
dx = np.diff(positions[:, 1])
step_sizes = np.sqrt(dy**2 + dx**2)
irregularity = np.std(step_sizes) / np.mean(step_sizes)
print(f"Step size CV: {irregularity:.4f} (>0.05 suggests position issues)")
print(f"Max deviation from mean step: {np.max(np.abs(step_sizes - np.mean(step_sizes))):.2e} m")
```

## Detection Methods

### Visual Indicators

- Blurred or smeared features in the reconstructed object, especially at high spatial frequencies.
- Doubled or ghosted edges where sharp interfaces should appear.
- The reconstruction error metric (chi-squared or SSE) fails to decrease below a plateau after many iterations.
- Probe reconstruction shows distorted or asymmetric features inconsistent with the expected focusing optic.
- Fourier ring correlation (FRC) plateaus well below the expected resolution.

### Automated Detection

```python
import numpy as np
from scipy.signal import correlate2d


def detect_position_errors(diffraction_data, positions, probe_diameter,
                           convergence_threshold=1e-3, max_iter=200):
    """
    Detect position errors by monitoring reconstruction convergence
    and analyzing position regularity.

    Parameters
    ----------
    diffraction_data : np.ndarray
        3D array (num_positions, ny, nx) of measured diffraction patterns.
    positions : np.ndarray
        2D array (num_positions, 2) of recorded probe positions in meters.
    probe_diameter : float
        Expected probe diameter in meters, used to compute overlap.
    convergence_threshold : float
        Minimum relative error decrease per iteration to consider converged.
    max_iter : int
        Number of iterations used in the reconstruction attempt.

    Returns
    -------
    dict with keys:
        'position_regularity' : float — coefficient of variation of step sizes
        'overlap_ratio' : float — mean overlap between adjacent positions
        'has_position_errors' : bool
        'diagnostics' : str
    """
    num_pos = positions.shape[0]

    # 1. Analyze position regularity
    step_vectors = np.diff(positions, axis=0)
    step_sizes = np.linalg.norm(step_vectors, axis=1)
    step_cv = np.std(step_sizes) / (np.mean(step_sizes) + 1e-15)

    # 2. Compute overlap ratio
    mean_step = np.mean(step_sizes)
    overlap_ratio = max(0, 1.0 - mean_step / probe_diameter)

    # 3. Check for drift (linear trend in positions)
    t = np.arange(num_pos)
    drift_y = np.polyfit(t, positions[:, 0], 1)[0]  # slope in y
    drift_x = np.polyfit(t, positions[:, 1], 1)[0]  # slope in x
    drift_rate = np.sqrt(drift_y**2 + drift_x**2)

    # 4. Check for high-frequency jitter via power spectral density
    pos_residual_y = positions[:, 0] - np.polyval(np.polyfit(t, positions[:, 0], 2), t)
    jitter_rms = np.sqrt(np.mean(pos_residual_y**2))

    # Decision logic
    has_errors = (step_cv > 0.05) or (jitter_rms > probe_diameter * 0.02)

    diagnostics = (
        f"Step size CV: {step_cv:.4f} (threshold: 0.05)\n"
        f"Overlap ratio: {overlap_ratio:.2f} (minimum ~0.6 recommended)\n"
        f"Drift rate: {drift_rate:.2e} m/point\n"
        f"Jitter RMS: {jitter_rms:.2e} m (threshold: {probe_diameter * 0.02:.2e})\n"
    )

    return {
        "position_regularity": step_cv,
        "overlap_ratio": overlap_ratio,
        "has_position_errors": has_errors,
        "diagnostics": diagnostics,
    }
```

## Solutions and Mitigation

### Prevention (Before Data Collection)

- Use laser interferometry for position feedback rather than relying solely on stage encoders.
- Minimize vibration sources: turn off unnecessary pumps, use vibration-isolated optical tables.
- Characterize and compensate for piezo hysteresis and creep before scanning.
- Use shorter scan times to reduce cumulative drift, or interleave reference scans.
- Ensure adequate overlap (>60%) between adjacent scan positions to provide the redundancy needed for position refinement.

### Correction — Traditional Methods

Joint position refinement during iterative reconstruction corrects recorded positions using the diffraction data itself. This is implemented in most modern ptychography engines.

```python
import numpy as np

def position_correction_annealing(
    diffraction_stack, positions, probe, object_est,
    num_iterations=50, step_size=1e-9, anneal_start=10
):
    """
    Simple gradient-based position correction during ePIE reconstruction.
    Refines positions by computing the gradient of the error metric
    with respect to probe position shifts.

    Parameters
    ----------
    diffraction_stack : np.ndarray
        Measured diffraction patterns, shape (N, py, px).
    positions : np.ndarray
        Initial positions, shape (N, 2), in pixel units.
    probe : np.ndarray
        Current probe estimate, shape (py, px), complex.
    object_est : np.ndarray
        Current object estimate, complex.
    num_iterations : int
        Number of refinement iterations.
    step_size : float
        Position update step size in pixels.
    anneal_start : int
        Iteration at which to begin position refinement.

    Returns
    -------
    refined_positions : np.ndarray
        Corrected positions, shape (N, 2).
    """
    refined_positions = positions.copy().astype(float)
    py, px = probe.shape
    num_pos = len(positions)

    for iteration in range(num_iterations):
        order = np.random.permutation(num_pos)
        for idx in order:
            row, col = int(round(refined_positions[idx, 0])), int(round(refined_positions[idx, 1]))
            obj_patch = object_est[row:row+py, col:col+px]
            exit_wave = probe * obj_patch

            # Fourier constraint
            exit_ft = np.fft.fft2(exit_wave)
            measured_amp = np.sqrt(diffraction_stack[idx])
            corrected_ft = measured_amp * np.exp(1j * np.angle(exit_ft))
            exit_corrected = np.fft.ifft2(corrected_ft)

            # ePIE object and probe update
            diff = exit_corrected - exit_wave
            object_est[row:row+py, col:col+px] += (
                np.conj(probe) / (np.max(np.abs(probe))**2 + 1e-8) * diff
            )

            # Position correction via cross-correlation gradient (after anneal_start)
            if iteration >= anneal_start:
                grad_y = np.real(np.sum(
                    np.conj(diff) * np.roll(exit_wave, 1, axis=0) -
                    np.conj(diff) * np.roll(exit_wave, -1, axis=0)
                ))
                grad_x = np.real(np.sum(
                    np.conj(diff) * np.roll(exit_wave, 1, axis=1) -
                    np.conj(diff) * np.roll(exit_wave, -1, axis=1)
                ))
                refined_positions[idx, 0] -= step_size * np.sign(grad_y)
                refined_positions[idx, 1] -= step_size * np.sign(grad_x)

    return refined_positions
```

### Correction — AI/ML Methods

Deep learning approaches can directly predict position corrections from diffraction data. **edgePtychoNN** (Babu et al. 2023) uses a convolutional neural network trained on simulated ptychographic datasets with known position errors to predict sub-pixel position offsets. The network learns the mapping from local diffraction pattern features to the required position correction, providing a fast initialization for subsequent iterative refinement.

```python
import numpy as np
# Conceptual workflow for DL-based position correction
# (requires trained model from edgePtychoNN or similar framework)

def dl_position_correction(diffraction_stack, initial_positions, model):
    """
    Use a pre-trained neural network to predict position corrections.

    Parameters
    ----------
    diffraction_stack : np.ndarray
        Measured patterns, shape (N, py, px).
    initial_positions : np.ndarray
        Recorded positions, shape (N, 2).
    model : trained neural network
        Predicts (dy, dx) corrections from diffraction features.

    Returns
    -------
    corrected_positions : np.ndarray
    """
    corrections = np.zeros_like(initial_positions)
    for i in range(len(diffraction_stack)):
        # Preprocess: log-scale, normalize
        pattern = np.log1p(diffraction_stack[i])
        pattern = (pattern - pattern.mean()) / (pattern.std() + 1e-8)
        # Predict correction (dy, dx) in pixel units
        dy, dx = model.predict(pattern[np.newaxis, ..., np.newaxis]).flatten()
        corrections[i] = [dy, dx]

    corrected_positions = initial_positions + corrections
    return corrected_positions
```

## Impact If Uncorrected

Position errors directly limit the achievable resolution in ptychographic reconstruction. Even a few nanometers of systematic error can reduce effective resolution by a factor of 2-5x compared to the theoretical limit. The reconstruction may fail to converge entirely, producing meaningless phase maps. Quantitative phase measurements become unreliable, corrupting downstream analysis such as electron density mapping, strain measurement, or chemical speciation. In fly-scan ptychography, uncorrected position errors create streaking artifacts that can mimic or obscure real sample features.

## Related Resources

- [Partial Coherence Effects](partial_coherence.md) — coherence issues compound position error effects
- [Stitching Artifact](stitching_artifact.md) — position errors at tile boundaries exacerbate stitching issues
- [Ptychography overview](../../02_xray_modalities/ptychography/README.md) — fundamentals of ptychographic reconstruction
- [edgePtychoNN — DL position correction (Babu et al. 2023)](https://doi.org/10.1038/s41467-023-41496-z)
- [PtychoShelves framework](https://github.com/PtychoShelves) — modular reconstruction with position refinement

## Key Takeaway

Position errors are the single most common cause of degraded resolution in ptychography. Always enable joint position refinement during reconstruction and, when possible, validate positions against independent interferometer readings — correcting positions post-hoc using the diffraction data itself is both effective and essential for achieving diffraction-limited resolution.
