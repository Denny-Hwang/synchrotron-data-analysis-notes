# Detector Common Issues

## Classification

| Attribute | Value |
|-----------|-------|
| **Modality** | Cross-cutting |
| **Noise Type** | Instrumental |
| **Severity** | Major |
| **Frequency** | Common |
| **Detection Difficulty** | Moderate |

## Visual Examples

```
Non-uniform response          Afterglow / Ghosting         Pixel cross-talk

┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│ ▓▓▓▓████████▒▒▒▒│         │                  │         │                  │
│ ▓▓▓▓████████▒▒▒▒│         │  Frame N:  ●     │         │   True:  ●       │
│ ▓▓▓▓████████▒▒▒▒│         │                  │         │                  │
│ ▓▓▓▓████████▒▒▒▒│         │  Frame N+1: ◌   │         │   Measured: ●●   │
│  dim  bright dim │         │  (ghost of ●)   │         │   (bleeds to     │
│                  │         │                  │         │    neighbors)    │
└──────────────────┘         └──────────────────┘         └──────────────────┘

  Flat field shows              Previous exposure             Sharp feature
  gain variation                bleeds into next              appears broadened
```

> **External references:**
> - [Detector calibration at synchrotrons — Gruner et al.](https://doi.org/10.1063/1.1480445)
> - [Dectris EIGER documentation](https://www.dectris.com/detectors/x-ray-detectors/eiger2/)

## Description

Detector common issues encompass a range of artifacts arising from the physics of X-ray detection, including non-uniform pixel response (gain variation), afterglow and ghosting from previous exposures, pixel cross-talk (charge sharing between adjacent pixels), and time-dependent degradation of scintillator materials. These artifacts are present to some degree in every detector and affect all synchrotron imaging modalities. While individually each effect may be small, their cumulative impact can significantly degrade quantitative accuracy if not properly calibrated and corrected.

## Root Cause

**Non-uniform response** arises from manufacturing variations in pixel sensitivity, scintillator thickness non-uniformity, and optical coupling variations in indirect detectors. Each pixel has a slightly different gain and offset, creating a fixed spatial pattern in the response.

**Afterglow/ghosting** occurs in scintillator-based detectors when luminescent decay extends beyond the frame readout time. CsI:Tl and GOS scintillators exhibit decay components spanning milliseconds to seconds, causing signal from intense features to persist into subsequent frames. Direct-detection photon-counting detectors (e.g., Dectris EIGER, Pilatus) are largely immune to afterglow.

**Pixel cross-talk** results from charge sharing in hybrid pixel detectors (photon-generated charge cloud spreads across pixel boundaries) and optical spreading in scintillator-coupled detectors. The effect is strongest for photon energies where the charge cloud size approaches the pixel pitch.

**Scintillator degradation** causes the conversion efficiency to decrease with cumulative radiation dose, creating position-dependent sensitivity changes that evolve over months. High-flux regions degrade faster, introducing spatially varying gain drift.

## Quick Diagnosis

```python
import numpy as np

# Load flat-field images (multiple frames of uniform illumination)
# flats = np.array([...])  # shape: (num_frames, ny, nx)
mean_flat = np.mean(flats, axis=0)
# Non-uniformity: coefficient of variation of the flat field
cv = np.std(mean_flat) / np.mean(mean_flat)
print(f"Flat-field CV: {cv:.4f} (>0.05 indicates significant non-uniformity)")
# Afterglow check: ratio of dark frame after bright exposure to bright frame
# dark_after_bright = ...
afterglow_fraction = np.mean(dark_after_bright) / np.mean(mean_flat)
print(f"Afterglow fraction: {afterglow_fraction:.4e} (>1e-3 is problematic)")
```

## Detection Methods

### Visual Indicators

- Flat-field image shows structured spatial patterns (stripes, gradients, patches) rather than uniform intensity.
- Ghost images of intense features visible in frames acquired shortly after high-exposure events.
- Point sources appear broader than expected or develop asymmetric halos (cross-talk).
- Progressive darkening of high-flux regions over weeks/months of beamline operation (scintillator burn).
- Systematic bright or dark pixel clusters that persist across all images (dead/hot pixel modules).

### Automated Detection

```python
import numpy as np
from scipy.ndimage import median_filter, uniform_filter


def detect_detector_issues(flat_frames, dark_frames, bright_then_dark=None,
                            nonuniformity_threshold=0.10,
                            afterglow_threshold=1e-3,
                            bad_pixel_sigma=5.0):
    """
    Comprehensive detector characterization from calibration frames.

    Parameters
    ----------
    flat_frames : np.ndarray
        Stack of flat-field images, shape (num_flats, ny, nx).
    dark_frames : np.ndarray
        Stack of dark images (beam off), shape (num_darks, ny, nx).
    bright_then_dark : np.ndarray or None
        Dark frames acquired immediately after a bright exposure,
        shape (num_frames, ny, nx). Used for afterglow detection.
    nonuniformity_threshold : float
        Maximum acceptable flat-field coefficient of variation.
    afterglow_threshold : float
        Maximum acceptable afterglow fraction.
    bad_pixel_sigma : float
        Sigma threshold for identifying bad pixels from flat field.

    Returns
    -------
    dict with keys:
        'gain_map' : np.ndarray — per-pixel relative gain
        'bad_pixel_mask' : np.ndarray — boolean mask of defective pixels
        'nonuniformity_cv' : float
        'afterglow_fraction' : float or None
        'dark_current_map' : np.ndarray
        'issues_detected' : list of str
    """
    issues = []

    # Dark current characterization
    mean_dark = np.mean(dark_frames.astype(np.float64), axis=0)
    dark_std = np.std(dark_frames.astype(np.float64), axis=0)

    # Flat field analysis
    mean_flat = np.mean(flat_frames.astype(np.float64), axis=0)
    dark_corrected_flat = mean_flat - mean_dark

    # Gain map (normalized response)
    global_mean = np.mean(dark_corrected_flat)
    gain_map = dark_corrected_flat / (global_mean + 1e-10)

    # Non-uniformity
    cv = float(np.std(gain_map[gain_map > 0.1]) / np.mean(gain_map[gain_map > 0.1]))
    if cv > nonuniformity_threshold:
        issues.append(
            f"High non-uniformity: CV={cv:.4f} (threshold: {nonuniformity_threshold})"
        )

    # Bad pixel detection using median absolute deviation
    local_median = median_filter(gain_map, size=5)
    deviation = np.abs(gain_map - local_median)
    mad = np.median(deviation[deviation > 0])
    mad = mad if mad > 0 else 1e-10
    bad_pixel_mask = (deviation / (1.4826 * mad)) > bad_pixel_sigma

    # Also flag dead pixels (gain ~ 0) and hot pixels (gain >> 1)
    bad_pixel_mask |= gain_map < 0.1
    bad_pixel_mask |= gain_map > 3.0

    num_bad = int(np.sum(bad_pixel_mask))
    total_pixels = bad_pixel_mask.size
    bad_fraction = num_bad / total_pixels
    if bad_fraction > 0.001:
        issues.append(
            f"Bad pixels: {num_bad} ({100*bad_fraction:.3f}% of detector)"
        )

    # Dark current analysis
    mean_dark_level = float(np.mean(mean_dark))
    hot_dark_pixels = np.sum(mean_dark > mean_dark_level + 5 * np.std(mean_dark))
    if hot_dark_pixels > 100:
        issues.append(f"Hot dark pixels: {hot_dark_pixels}")

    # Afterglow detection
    afterglow_frac = None
    if bright_then_dark is not None:
        # Compare dark-after-bright with normal dark
        mean_after_bright = np.mean(bright_then_dark.astype(np.float64), axis=0)
        afterglow_signal = mean_after_bright - mean_dark
        afterglow_frac = float(
            np.mean(afterglow_signal) / (global_mean + 1e-10)
        )
        if afterglow_frac > afterglow_threshold:
            issues.append(
                f"Afterglow detected: {afterglow_frac:.4e} "
                f"(threshold: {afterglow_threshold})"
            )

    return {
        "gain_map": gain_map,
        "bad_pixel_mask": bad_pixel_mask,
        "nonuniformity_cv": cv,
        "afterglow_fraction": afterglow_frac,
        "dark_current_map": mean_dark,
        "num_bad_pixels": num_bad,
        "issues_detected": issues,
    }
```

## Solutions and Mitigation

### Prevention (Before Data Collection)

- Perform flat-field and dark-current calibration at the start of each experiment and periodically during long sessions.
- Replace or refurbish scintillators showing visible burn marks or significant efficiency loss.
- For afterglow-sensitive experiments (fast tomography, XPCS), use photon-counting detectors or allow sufficient dead time between frames.
- Maintain a bad-pixel map for each detector and update it regularly.
- Operate within the detector's linear response range; avoid saturation which accelerates scintillator damage.

### Correction — Traditional Methods

Standard three-step correction: dark subtraction, flat-field normalization, and bad-pixel interpolation.

```python
import numpy as np
from scipy.ndimage import median_filter


def apply_detector_corrections(raw_data, dark_map, flat_map, bad_pixel_mask,
                                afterglow_correction=False,
                                afterglow_decay=0.005):
    """
    Apply standard detector corrections to raw image data.

    Parameters
    ----------
    raw_data : np.ndarray
        Raw detector images, shape (num_frames, ny, nx) or (ny, nx).
    dark_map : np.ndarray
        Mean dark image, shape (ny, nx).
    flat_map : np.ndarray
        Mean flat-field image (dark-subtracted), shape (ny, nx).
    bad_pixel_mask : np.ndarray
        Boolean mask, True for bad pixels, shape (ny, nx).
    afterglow_correction : bool
        Whether to apply frame-to-frame afterglow subtraction.
    afterglow_decay : float
        Afterglow decay fraction per frame (detector-specific).

    Returns
    -------
    corrected : np.ndarray — corrected image data, same shape as input
    """
    is_single = raw_data.ndim == 2
    if is_single:
        raw_data = raw_data[np.newaxis]

    num_frames = raw_data.shape[0]
    corrected = np.empty_like(raw_data, dtype=np.float32)

    # Precompute gain map (inverse of normalized flat field)
    flat_norm = flat_map / (np.mean(flat_map) + 1e-10)
    # Avoid division by zero for dead pixels
    gain_correction = np.where(flat_norm > 0.1, 1.0 / flat_norm, 0.0)

    previous_frame = None

    for i in range(num_frames):
        frame = raw_data[i].astype(np.float32)

        # 1. Dark subtraction
        frame -= dark_map

        # 2. Afterglow correction (subtract decayed previous frame)
        if afterglow_correction and previous_frame is not None:
            frame -= afterglow_decay * previous_frame
        previous_frame = frame.copy()

        # 3. Flat-field normalization
        frame *= gain_correction

        # 4. Bad pixel interpolation via local median
        if np.any(bad_pixel_mask):
            # Use median filter to estimate values at bad pixel locations
            median_img = median_filter(frame, size=3)
            frame[bad_pixel_mask] = median_img[bad_pixel_mask]

        corrected[i] = frame

    if is_single:
        corrected = corrected[0]

    return corrected


def create_bad_pixel_mask(flat_frames, dark_frames, sigma=5.0,
                           min_gain=0.1, max_gain=3.0):
    """
    Generate a bad pixel mask from calibration frames.

    Parameters
    ----------
    flat_frames : np.ndarray
        Stack of flat-field images.
    dark_frames : np.ndarray
        Stack of dark images.
    sigma : float
        Outlier detection threshold.
    min_gain : float
        Minimum acceptable relative gain.
    max_gain : float
        Maximum acceptable relative gain.

    Returns
    -------
    mask : np.ndarray — boolean, True = bad pixel
    """
    mean_flat = np.mean(flat_frames.astype(np.float64), axis=0)
    mean_dark = np.mean(dark_frames.astype(np.float64), axis=0)

    corrected = mean_flat - mean_dark
    normalized = corrected / (np.mean(corrected) + 1e-10)

    # Statistical outlier detection
    local_med = median_filter(normalized, size=5)
    deviation = np.abs(normalized - local_med)
    mad = np.median(deviation[deviation > 0])
    outliers = deviation > sigma * 1.4826 * mad

    # Range-based detection
    too_low = normalized < min_gain
    too_high = normalized > max_gain

    # High dark current pixels
    dark_outliers = mean_dark > np.median(mean_dark) + sigma * np.std(mean_dark)

    mask = outliers | too_low | too_high | dark_outliers

    print(f"Bad pixel mask: {np.sum(mask)} pixels "
          f"({100 * np.sum(mask) / mask.size:.3f}%)")

    return mask
```

### Correction — AI/ML Methods

Deep learning-based detector correction can learn complex, non-linear response functions that traditional flat-field methods cannot capture. A convolutional neural network trained on paired data (raw detector output vs. ground truth from averaging many exposures) can correct for position-dependent non-linearity, charge-sharing effects, and spatially varying point-spread functions simultaneously. This is especially valuable for photon-counting detectors where charge sharing creates energy-dependent spatial distortions at pixel boundaries.

## Impact If Uncorrected

Non-uniform detector response directly corrupts quantitative intensity measurements across all modalities. In tomography, uncorrected gain variations create ring artifacts. In ptychography, intensity errors propagate into phase errors. In XRF, non-uniform efficiency biases elemental quantification across the field of view. Afterglow creates temporal blurring that reduces effective frame rate in fast experiments and introduces correlated noise in time-resolved studies. Pixel cross-talk degrades spatial resolution below the physical pixel pitch, limiting the achievable resolution of the imaging system. Scintillator degradation causes slow, undetected drift in sensitivity that can bias longitudinal studies comparing data collected months apart.

## Related Resources

- [Ring Artifact](../tomography/ring_artifact.md) — ring artifacts are a direct consequence of detector non-uniformity
- [Dead/Hot Pixel](../xrf_microscopy/dead_hot_pixel.md) — specific case of extreme pixel response deviation
- [Photon Counting Noise](../xrf_microscopy/photon_counting_noise.md) — quantum detection efficiency affects counting statistics
- [Tomography EDA notebook](../../06_data_structures/eda/tomo_eda.md) — flat-field quality inspection
- [Dectris detector documentation](https://www.dectris.com/detectors/x-ray-detectors/eiger2/)
- [Gruner et al. — Detector calibration](https://doi.org/10.1063/1.1480445)

## Key Takeaway

Detector artifacts are universal and affect every measurement — regular calibration with fresh flat-field and dark images is the single most important quality assurance step at any synchrotron beamline, and a well-maintained bad-pixel mask combined with proper flat-field correction eliminates the majority of detector-related systematic errors.
