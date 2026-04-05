# Specimen Drift & Mechanical Vibration

## Classification

| Attribute | Value |
|-----------|-------|
| **Modality** | SEM / TEM / Synchrotron Microscopy |
| **Noise Type** | Systematic |
| **Severity** | Major |
| **Frequency** | Common |
| **Detection Difficulty** | Moderate |
| **Origin Domain** | Electron Microscopy (SEM/TEM) |

## Visual Examples

![Before and after — drift and vibration](../images/drift_vibration_before_after.png)

> **Image source:** Synthetic image with progressive horizontal drift (~20 pixels total). Left: distorted image from specimen drift during scan. Right: after cross-correlation line-by-line alignment. MIT license.

## Description

Specimen drift and mechanical vibration cause spatial blurring and distortion in microscopy images. Drift is a slow, continuous movement of the sample (thermal expansion, piezo creep, mechanical relaxation), while vibration is rapid oscillatory motion (building vibrations, acoustic noise, pump vibrations). Both degrade spatial resolution — drift causes directional elongation of features, vibration causes isotropic blurring.

**Synchrotron relevance:** Critical for nanoprobe XRF, ptychography, and nano-tomography where position accuracy must be better than the beam size (often < 100 nm).

## Root Cause

- **Thermal drift:** Temperature changes cause differential expansion of sample/stage (~nm/s)
- **Piezo creep:** Piezoelectric scanners exhibit logarithmic creep after repositioning
- **Mechanical vibration:** Building vibrations (1-100 Hz), vacuum pumps, water cooling lines
- **Acoustic coupling:** Sound waves from equipment or environment vibrate sample
- **Electrostatic forces:** Charging can cause sample movement in EM

## Quick Diagnosis

```python
import numpy as np

def measure_drift_from_repeated_scans(image1, image2, pixel_size_nm=1.0):
    """Measure drift between two sequential scans via cross-correlation."""
    from scipy.signal import fftconvolve
    # Cross-correlation
    corr = fftconvolve(image1, image2[::-1, ::-1], mode='same')
    peak = np.unravel_index(corr.argmax(), corr.shape)
    center = np.array(corr.shape) // 2
    drift_pixels = np.array(peak) - center
    drift_nm = drift_pixels * pixel_size_nm
    print(f"Drift: ({drift_nm[0]:.1f}, {drift_nm[1]:.1f}) nm")
    print(f"Drift rate: {np.linalg.norm(drift_nm):.1f} nm/frame")
    return drift_nm
```

## Detection Methods

### Visual Indicators

- **Drift:** Features elongated in one direction; slow scans show more distortion than fast scans
- **Vibration:** Isotropic blurring; loss of high-frequency detail; "jittery" edges
- Sequential frames show systematic shift
- FFT shows anisotropic high-frequency rolloff (drift) or isotropic rolloff (vibration)

### Automated Detection

```python
import numpy as np

def drift_from_fft_anisotropy(image):
    """Detect drift direction from FFT anisotropy."""
    F = np.abs(np.fft.fftshift(np.fft.fft2(image)))
    ny, nx = F.shape
    cy, cx = ny // 2, nx // 2
    # Compute power in radial sectors
    Y, X = np.ogrid[-cy:ny-cy, -cx:nx-cx]
    angles = np.arctan2(Y, X)
    radii = np.sqrt(X**2 + Y**2)
    high_freq = (radii > min(cy, cx) * 0.3) & (radii < min(cy, cx) * 0.8)
    # Angular power distribution
    n_sectors = 36
    sector_power = []
    for i in range(n_sectors):
        a0 = -np.pi + 2 * np.pi * i / n_sectors
        a1 = a0 + 2 * np.pi / n_sectors
        mask = high_freq & (angles >= a0) & (angles < a1)
        sector_power.append(F[mask].mean() if mask.any() else 0)
    anisotropy = np.max(sector_power) / (np.mean(sector_power) + 1e-10)
    drift_direction = np.argmax(sector_power) * 360 / n_sectors - 180
    return anisotropy, drift_direction
```

## Correction Methods

### Traditional Approaches

1. **Cross-correlation alignment:** Register sequential frames/scanlines to correct drift
2. **MotionCor2 / RELION:** Movie-frame alignment for cryo-EM
3. **Scan-line registration:** Correct each scan line independently (for SEM)
4. **Vibration isolation:** Passive (air table) or active (feedback-controlled) isolation
5. **Temperature stabilization:** Thermal enclosure around microscope

```python
def correct_scanline_drift(image, reference_line=0):
    """Correct horizontal drift in SEM image via cross-correlation per line."""
    corrected = image.copy()
    ref = image[reference_line, :]
    for i in range(image.shape[0]):
        corr = np.correlate(ref, image[i, :], mode='full')
        shift = corr.argmax() - len(ref) + 1
        corrected[i, :] = np.roll(image[i, :], -shift)
    return corrected
```

### AI/ML Approaches

- **MotionCor2:** GPU-accelerated frame alignment (Zheng et al., 2017)
- **Deep drift correction:** CNN predicting drift trajectory from image features
- **Neural implicit representations:** Model continuous drift during scan

## Key References

- **Zheng et al. (2017)** — "MotionCor2: anisotropic correction of beam-induced motion for improved cryo-EM"
- **Jones et al. (2015)** — "Smart Align: drift correction for scanning probe microscopy"
- **Ophus et al. (2016)** — "Correcting nonlinear drift distortion of scanning probe and scanning transmission electron microscopies"

## Relevance to Synchrotron Data

| Scenario | Relevance |
|----------|-----------|
| Nanoprobe XRF/XAS | Sample drift during raster scan distorts elemental maps |
| Ptychography | Position errors from drift (see position_error.md) |
| Nano-tomography | Drift between projections causes alignment errors |
| In-situ experiments | Thermal/mechanical changes during reaction cause drift |
| Long EXAFS scans | Monochromator drift (energy calibration drift) |

## Real-World Before/After Examples

The following published sources provide real experimental before/after comparisons:

| Source | Type | Figure | Description | License |
|--------|------|--------|-------------|---------|
| [Zheng et al. 2017 — MotionCor2](https://doi.org/10.1038/nmeth.4193) | Paper | Fig 1 | Before/after drift correction in cryo-EM movie frames — anisotropic beam-induced motion correction | -- |

> **Recommended reference**: [Zheng et al. 2017 — MotionCor2 (Nature Methods)](https://doi.org/10.1038/nmeth.4193)

## Related Resources

- [Position error](../ptychography/position_error.md) — Drift causes position errors in ptychography
- [Motion artifact](../tomography/motion_artifact.md) — Analogous motion effects in tomography
- [Energy calibration drift](../spectroscopy/energy_calibration_drift.md) — Drift in energy domain
