# Beam Intensity Drop

## Classification

| Attribute | Value |
|-----------|-------|
| **Modality** | Tomography |
| **Noise Type** | Instrumental |
| **Severity** | Major |
| **Frequency** | Occasional |
| **Detection Difficulty** | Moderate |

## Visual Examples

```
  ASCII fallback — beam intensity drop in sinogram and reconstruction:

  Sinogram view                    I0 monitor trace (synthetic)
  ┌────────────────────────┐      ┌────────────────────────┐
  │████████████████████████│      │────────────────────────│
  │████████████████████████│      │────────────────────────│
  │████████████████████████│      │────────────────────────│
  │░░░░░░░░░░░░░░░░░░░░░░░│ ←    │         ╲              │ ← beam drop
  │░░░░░░░░░░░░░░░░░░░░░░░│      │          ╲_____        │
  │████████████████████████│      │                ╱───────│ ← recovery
  │████████████████████████│      │────────────────────────│
  │████████████████████████│      │────────────────────────│
  └────────────────────────┘      └────────────────────────┘

  Horizontal bright/dark band       Sudden I0 drop followed by
  in sinogram at drop event         recovery (beam dump/top-up)
```

## Description

Beam intensity drops appear as horizontal bright or dark bands in the sinogram, corresponding to a group of projections acquired during abnormal beam conditions. In the reconstruction, these manifest as ring-like artifacts at specific radii (from the band position in the sinogram) or as abrupt intensity jumps between adjacent slices. The affected projections may show globally reduced or increased intensity, altered contrast, or complete signal loss depending on the severity of the beam event.

## Root Cause

Synchrotron beam intensity can drop suddenly due to several operational events: beam dumps (complete loss of stored beam), top-up injection failures (scheduled injection does not fully restore beam current), shutter malfunctions (beam blocked or partially blocked during acquisition), beam position feedback oscillations, or insertion device (undulator/wiggler) changes. Storage ring instabilities, vacuum events, or radio-frequency system trips can also cause sudden current loss. Even with top-up mode, small intensity fluctuations occur during injection windows. The I0 monitor (incident beam intensity measurement) records these events but standard flat-field correction only accounts for the average beam state.

## Quick Diagnosis

```python
import numpy as np

# Monitor per-projection total intensity as proxy for I0
proj_intensity = np.mean(projections, axis=(1, 2))
median_intensity = np.median(proj_intensity)
# Flag projections with significant intensity deviation
drop_mask = proj_intensity < 0.9 * median_intensity
print(f"Projections with >10% intensity drop: {np.sum(drop_mask)}")
print(f"Drop indices: {np.where(drop_mask)[0]}")
```

## Detection Methods

### Visual Indicators

- Horizontal bands (bright or dark) spanning the full width of the sinogram.
- Abrupt intensity change visible when scrolling through projections sequentially.
- If I0 monitor data is available, a clear dip or spike in the beam current trace.
- In the reconstruction, ring-like bands at specific heights or intensity discontinuities.

### Automated Detection

```python
import numpy as np


def detect_beam_intensity_drops(projections, i0_monitor=None,
                                 drop_threshold=0.9, spike_threshold=1.1):
    """
    Detect beam intensity drop events from projection data or I0 monitor.

    Parameters
    ----------
    projections : np.ndarray
        3D projection stack (num_proj, height, width).
    i0_monitor : np.ndarray or None
        1D array of I0 monitor readings per projection.
    drop_threshold : float
        Fraction of median below which a projection is flagged as a drop.
    spike_threshold : float
        Fraction of median above which a projection is flagged as a spike.

    Returns
    -------
    dict with keys:
        'intensity_curve' : np.ndarray — per-projection intensity
        'drop_indices' : list of int — projections with drops
        'spike_indices' : list of int — projections with spikes
        'has_beam_issues' : bool
        'affected_fraction' : float
    """
    if i0_monitor is not None:
        intensity = i0_monitor.astype(np.float64)
    else:
        # Use mean projection intensity as proxy
        intensity = np.mean(projections.astype(np.float64), axis=(1, 2))

    median_i = np.median(intensity)

    # Detect drops
    drop_indices = np.where(intensity < drop_threshold * median_i)[0].tolist()

    # Detect spikes (e.g., injection overshoot)
    spike_indices = np.where(intensity > spike_threshold * median_i)[0].tolist()

    affected = len(drop_indices) + len(spike_indices)
    total = len(intensity)

    return {
        "intensity_curve": intensity,
        "drop_indices": drop_indices,
        "spike_indices": spike_indices,
        "has_beam_issues": affected > 0,
        "affected_fraction": affected / total,
    }
```

## Solutions and Mitigation

### Prevention (Before Data Collection)

- Schedule scans during stable beam periods, avoiding injection windows if top-up is not continuous.
- Use a beam current monitor (I0 ion chamber) recorded synchronously with projections.
- Configure the scan control system to pause acquisition during beam dumps and resume after recovery.
- Use top-up storage ring mode to maintain near-constant beam current.
- Set up automated beam current thresholds that trigger scan suspension.

### Correction — Traditional Methods

Per-projection I0 normalization removes intensity fluctuations. Severely affected projections can be removed and the missing angles handled by interpolation or iterative reconstruction.

```python
import numpy as np


def correct_beam_intensity(projections, i0_monitor=None, remove_bad=True,
                            drop_threshold=0.8):
    """
    Correct beam intensity variations using I0 monitor data or
    estimated per-projection intensity.

    Parameters
    ----------
    projections : np.ndarray
        3D projection stack (num_proj, height, width).
    i0_monitor : np.ndarray or None
        I0 readings per projection. If None, estimated from data.
    remove_bad : bool
        If True, remove projections with severe drops.
    drop_threshold : float
        Fraction of median I0 below which a projection is removed.

    Returns
    -------
    corrected : np.ndarray — intensity-corrected projections
    valid_mask : np.ndarray — boolean mask of retained projections
    """
    if i0_monitor is not None:
        i0 = i0_monitor.astype(np.float64)
    else:
        # Estimate I0 from projection margins (regions outside sample)
        margin = 20  # pixels at detector edges
        left_margin = np.mean(projections[:, :, :margin].astype(np.float64),
                              axis=(1, 2))
        right_margin = np.mean(projections[:, :, -margin:].astype(np.float64),
                               axis=(1, 2))
        i0 = 0.5 * (left_margin + right_margin)

    median_i0 = np.median(i0)

    # Normalize each projection by its I0 value
    corrected = np.zeros_like(projections, dtype=np.float64)
    for i in range(projections.shape[0]):
        if i0[i] > 0:
            corrected[i] = projections[i].astype(np.float64) * (median_i0 / i0[i])
        else:
            corrected[i] = projections[i].astype(np.float64)

    # Identify and optionally remove severely affected projections
    valid_mask = i0 > drop_threshold * median_i0

    if remove_bad and not np.all(valid_mask):
        bad_indices = np.where(~valid_mask)[0]
        print(f"Removing {len(bad_indices)} severely affected projections")

        # Interpolate removed projections from neighbors
        for idx in bad_indices:
            left = idx - 1
            right = idx + 1
            while left >= 0 and not valid_mask[left]:
                left -= 1
            while right < len(valid_mask) and not valid_mask[right]:
                right += 1

            if left >= 0 and right < len(valid_mask):
                weight = (idx - left) / (right - left)
                corrected[idx] = ((1 - weight) * corrected[left] +
                                  weight * corrected[right])
                valid_mask[idx] = True
            elif left >= 0:
                corrected[idx] = corrected[left]
                valid_mask[idx] = True
            elif right < len(valid_mask):
                corrected[idx] = corrected[right]
                valid_mask[idx] = True

    return corrected, valid_mask
```

### Correction — AI/ML Methods

No established AI/ML methods specifically for beam intensity drop correction. The artifact is well-handled by I0 normalization and projection interpolation. Deep learning sinogram inpainting methods developed for other artifacts (e.g., metal artifact reduction) could in principle be applied to fill in severely corrupted sinogram rows, but this is rarely necessary given the effectiveness of traditional I0 correction.

## Impact If Uncorrected

Uncorrected beam intensity drops introduce false density variations in the reconstruction. Projections acquired during a beam drop have reduced SNR and altered contrast, producing ring-like bands or intensity jumps in the reconstructed volume. Quantitative attenuation values are corrupted, making material identification and density measurements unreliable. In severe cases (complete beam loss), entire angular ranges are missing data, creating streak artifacts similar to limited-angle tomography.

## Related Resources

- [Tomography EDA notebook](../../06_data_structures/eda/tomo_eda.md) — beam intensity monitoring and projection quality checks
- Related artifact: [Flat-Field Issues](flatfield_issues.md) — beam drift that flat-fielding partially addresses
- Related artifact: [Ring Artifact](ring_artifact.md) — intensity bands can masquerade as ring sources in the sinogram

## Real-World Before/After Examples

The following published sources provide real experimental before/after comparisons:

| Source | Type | Figure | Description | License |
|--------|------|--------|-------------|---------|
| APS operation modes documentation | Facility docs | -- | Beam fill patterns and I0 normalization procedures showing intensity drop events and correction | -- |
| [Diamond Light Source I13 documentation](https://www.diamond.ac.uk/Instruments/Imaging-and-Microscopy/I13/) | Facility docs | -- | I13 imaging beamline documentation on I0 normalization and beam monitoring best practices | -- |
| [ESRF/EDF data format documentation](https://www.esrf.fr/home/UsersAndScience/Experiments/Imaging.html) | Facility docs | -- | ESRF imaging documentation showing I0 monitoring and data format best practices for intensity normalization | -- |

**Key references with published before/after comparisons:**
- **Diamond Light Source I13**: Documentation on I0 normalization procedures for tomography imaging experiments.
- **ESRF Imaging Group**: EDF data format documentation showing I0 monitoring best practices for beam intensity correction.

> **Recommended reference**: Consult your facility's machine status and operation modes documentation for beam fill patterns and I0 normalization best practices.

## Key Takeaway

Always record the I0 beam monitor synchronously with projections and apply per-projection intensity normalization as a standard preprocessing step. Check the I0 trace before reconstruction — projections acquired during beam dumps or injection failures should be flagged and either corrected or removed to prevent reconstruction artifacts.
