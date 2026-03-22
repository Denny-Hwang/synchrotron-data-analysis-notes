# Motion Artifact

## Classification

| Attribute | Value |
|-----------|-------|
| **Modality** | Tomography |
| **Noise Type** | Systematic |
| **Severity** | Critical |
| **Frequency** | Occasional |
| **Detection Difficulty** | Moderate |

## Visual Examples

```
  ASCII fallback — motion artifact in reconstructed slice:

  Clean reconstruction        With motion artifact
  ┌─────────────────┐        ┌─────────────────┐
  │                 │        │                 │
  │    ┌───────┐   │        │    ┌───────┐   │
  │    │       │   │        │   ┌┤  ___  ├┐  │
  │    │  ○  ○ │   │        │   │○ ○ ○  ○│   │
  │    │       │   │        │    │  __   │   │
  │    │  ───  │   │        │    │ ─── ──│   │
  │    │       │   │        │    │    __ │   │
  │    └───────┘   │        │    └──────┘│   │
  │                 │        │         ───┘   │
  │                 │        │                 │
  └─────────────────┘        └─────────────────┘

  Blurring, ghosting, and double edges indicate sample motion.
```

## Description

Motion artifacts appear as blurring, ghosting, or double edges in reconstructed CT slices, caused by the sample moving during the tomographic scan. The artifact severity depends on the magnitude and nature of the motion — translational drift creates directional blurring, sudden jumps produce ghost copies, and continuous deformation leads to inconsistent boundaries. Features may appear smeared along the direction of motion, and edges that should be sharp become diffuse or duplicated.

## Root Cause

Tomographic reconstruction assumes the sample remains perfectly stationary throughout the scan. Any deviation from this assumption — including thermal expansion/contraction, mechanical vibration of the rotation stage, gravitational sagging of soft samples, sample deformation under environmental loading (heating, compression), or drift of the rotation axis — introduces inconsistencies between projections taken at different times. Each projection records a slightly different object configuration, and the reconstruction algorithm, assuming a static object, produces a blurred or ghosted compromise of the different states.

## Quick Diagnosis

```python
import numpy as np

# Detect sample motion by comparing first and last (0° and 360°) projections
# If identical angle is acquired at start and end, they should match
proj_first = projections[0].astype(float)
proj_last = projections[-1].astype(float)
nrmse = np.sqrt(np.mean((proj_first - proj_last)**2)) / (np.max(proj_first) - np.min(proj_first))
print(f"NRMSE between 0° and 360° projections: {nrmse:.4f}")
print(f"Likely motion: {nrmse > 0.02}")
```

## Detection Methods

### Visual Indicators

- Blurred or doubled edges in the reconstructed slice that should be sharp.
- Ghost features — faint duplicate copies of structures offset from the original.
- Directional smearing visible when scrolling through sequential slices.
- Comparison of 0° and 180° projections (flipped) shows misalignment.

### Automated Detection

```python
import numpy as np
from scipy import ndimage


def detect_motion_artifacts(projections, angular_step_deg=None):
    """
    Detect sample motion during a tomographic scan by analyzing
    projection-to-projection consistency.

    Parameters
    ----------
    projections : np.ndarray
        3D array of shape (num_projections, height, width).
    angular_step_deg : float or None
        Angular step between projections in degrees.

    Returns
    -------
    dict with keys:
        'drift_curve' : np.ndarray — estimated drift per projection
        'max_drift_pixels' : float
        'has_motion' : bool
        'motion_type' : str — 'none', 'gradual_drift', 'sudden_jump', 'vibration'
    """
    num_proj = projections.shape[0]

    # Track center of mass shift across projections
    com_x = np.zeros(num_proj)
    com_y = np.zeros(num_proj)

    for i in range(num_proj):
        proj = projections[i].astype(np.float64)
        # Use negative log if transmission data
        total = np.sum(proj)
        if total > 0:
            yy, xx = np.mgrid[:proj.shape[0], :proj.shape[1]]
            com_x[i] = np.sum(xx * proj) / total
            com_y[i] = np.sum(yy * proj) / total

    # Remove linear trend (rotation-induced COM shift is expected)
    from numpy.polynomial import polynomial as P
    t = np.arange(num_proj)
    coeff_x = P.polyfit(t, com_x, deg=2)
    coeff_y = P.polyfit(t, com_y, deg=2)
    trend_x = P.polyval(t, coeff_x)
    trend_y = P.polyval(t, coeff_y)
    residual_x = com_x - trend_x
    residual_y = com_y - trend_y

    drift = np.sqrt(residual_x**2 + residual_y**2)
    max_drift = np.max(drift)

    # Classify motion type
    diff_drift = np.diff(drift)
    if max_drift < 0.5:
        motion_type = "none"
    elif np.max(np.abs(diff_drift)) > 3 * np.std(diff_drift):
        motion_type = "sudden_jump"
    elif np.std(diff_drift) > 0.3 * np.mean(np.abs(diff_drift)):
        motion_type = "vibration"
    else:
        motion_type = "gradual_drift"

    return {
        "drift_curve": drift,
        "max_drift_pixels": float(max_drift),
        "has_motion": max_drift > 0.5,
        "motion_type": motion_type,
    }
```

## Solutions and Mitigation

### Prevention (Before Data Collection)

- Use faster scan protocols (fly-scan, continuous rotation) to minimize time for drift.
- Ensure stable sample mounting with appropriate adhesives or clamping.
- Allow thermal equilibration before scanning.
- Use vibration-isolated rotation stages.
- Acquire reference projections (e.g., 0° repeated periodically) to monitor drift in real time.

### Correction — Traditional Methods

Post-acquisition motion correction involves re-registering projections to a common reference frame before reconstruction.

```python
import numpy as np
from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation


def correct_motion_projections(projections, reference_idx=0):
    """
    Correct translational motion by aligning all projections
    to a reference using phase cross-correlation.
    """
    corrected = projections.copy().astype(np.float64)
    reference = projections[reference_idx].astype(np.float64)

    shifts_detected = np.zeros((projections.shape[0], 2))

    for i in range(projections.shape[0]):
        if i == reference_idx:
            continue

        current = projections[i].astype(np.float64)

        # Compute sub-pixel shift relative to reference
        detected_shift, error, _ = phase_cross_correlation(
            reference, current, upsample_factor=10
        )

        shifts_detected[i] = detected_shift

        # Apply correction
        corrected[i] = shift(current, -detected_shift, order=3)

    return corrected, shifts_detected


# For 180° scans, also compare proj[i] with flipped proj[i+N/2]
# to detect rotation-axis-perpendicular motion
```

### Correction — AI/ML Methods

Implicit neural representations (INRs) have emerged as a powerful tool for motion-compensated tomographic reconstruction. By representing the reconstructed volume as a continuous function parameterized by a neural network (e.g., coordinate-based MLP or hash-grid encoding), INRs can jointly optimize the volume and per-projection motion parameters. The network learns to explain the observed projections while estimating and compensating for the motion that occurred during acquisition. This approach handles both rigid and non-rigid motion and does not require explicit motion measurements or fiducial markers.

## Impact If Uncorrected

Motion artifacts cause loss of spatial resolution proportional to the motion amplitude, effectively blurring the reconstruction beyond the detector pixel size. Double edges lead to incorrect dimensional measurements and wall-thickness estimates. Segmentation algorithms produce inaccurate boundaries, and quantitative density measurements are corrupted by the averaging of different object configurations. In dynamic or in-situ experiments, uncorrected motion can be indistinguishable from real structural changes.

## Related Resources

- [INR dynamic reconstruction](../../03_ai_ml_methods/reconstruction/inr_dynamic.md) — implicit neural representations for motion-compensated CT
- Related artifact: [Rotation Center Error](rotation_center_error.md) — can mimic or compound motion artifacts
- Related artifact: [Sparse-Angle Artifact](sparse_angle_artifact.md) — fast scans to reduce motion may under-sample angles

## Key Takeaway

Motion artifacts arise whenever the sample moves during acquisition, violating the static-object assumption of standard reconstruction. Prevention through fast scans and stable mounting is best; when motion is unavoidable, projection re-registration or implicit neural representation-based joint reconstruction-and-alignment can recover image quality.
