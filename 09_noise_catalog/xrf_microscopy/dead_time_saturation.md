# Dead-Time Saturation

## Classification

| Attribute | Value |
|-----------|-------|
| **Modality** | XRF Microscopy |
| **Noise Type** | Instrumental |
| **Severity** | Critical |
| **Frequency** | Common |
| **Detection Difficulty** | Moderate |

## Visual Examples

```
 Detector response curve:

 Output                         Ideal (linear)
 Count Rate                   ╱
 (OCR)                      ╱
   │                      ╱
   │                    ╱
   │                  ╱  · · · · · Paralyzable (rolls over!)
   │                ╱ ·
   │              ╱·          ── Non-paralyzable (saturates)
   │            ╱·       ─────────────────────
   │          ╱·    ─────
   │        ╱· ────    ·
   │      ╱·──         ·
   │    ╱··─            ·
   │  ╱·─                ·
   │╱·─                    ·
   └──────────────────────────── Input Count Rate (ICR)

 Fe elemental map (uncorrected)        Fe map (dead-time corrected)
 ┌──────────────────────────┐          ┌──────────────────────────┐
 │ ░░░▒▒▓▓▓▓▓▓▓▓▓▒▒░░░░░░░ │          │ ░░░▒▒▓▓████████▓▒░░░░░░ │
 │ ░░▒▒▓▓▓▓▓▓▓▓▓▓▓▒▒░░░░░░ │          │ ░░▒▒▓▓██████████▓▒░░░░░ │
 │ ░░▒▓▓▓▓▓▓▓▓▓▓▓▓▓▒░░░░░░ │          │ ░░▒▓▓████████████▓▒░░░░ │
 │ ░░▒▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░░░░░ │          │ ░░▒▓▓██████████████▒░░░ │
 │ ░░░▒▒▓▓▓▓▓▓▓▓▒▒▒░░░░░░░ │          │ ░░░▒▒▓▓████████▓▓▒░░░░░ │
 └──────────────────────────┘          └──────────────────────────┘
  Concentration plateau in center       True concentration gradient restored
  (detector saturated there)            after ICR/OCR correction
```

## Description

Dead-time saturation occurs when the detector's incoming photon rate exceeds its processing capacity, causing the recorded output count rate (OCR) to fall below the true input count rate (ICR). The result is a non-linear response where high-concentration regions appear to have a lower concentration than they actually do, producing a "ceiling" effect in elemental maps. In severe cases with paralyzable detector systems, the output can actually decrease as the input increases, inverting the concentration contrast entirely.

## Root Cause

Every photon event detected by an energy-dispersive detector requires a finite processing time (the "dead time," typically 0.1–10 microseconds per event depending on shaping time). During this processing interval, the detector is unable to register additional photons, and those arrivals are lost. At low count rates, the fraction of lost photons is negligible. As the count rate increases toward 10^5–10^6 counts per second, the cumulative dead time becomes a significant fraction of the total measurement time. For a non-paralyzable system, the OCR asymptotically approaches a maximum (1/tau). For a paralyzable system, each lost photon extends the dead window, causing the OCR to peak and then decrease at very high input rates.

## Quick Diagnosis

```python
import numpy as np

# ICR, OCR: 2D maps of input and output count rates per pixel
dead_time_fraction = 1.0 - (ocr_map / icr_map)
max_dt = np.max(dead_time_fraction)
mean_dt = np.mean(dead_time_fraction)
print(f"Mean dead-time fraction: {mean_dt:.1%}")
print(f"Max dead-time fraction:  {max_dt:.1%}")
print(f"Correction needed: {'YES — significant' if max_dt > 0.10 else 'Minimal'}")
```

## Detection Methods

### Visual Indicators

- Elemental maps show a concentration "ceiling" — bright regions flatten out instead of showing the expected dynamic range.
- The ICR map (total incoming counts) shows much higher contrast than the elemental maps derived from OCR data.
- Histograms of elemental intensity are truncated or compressed at the high end.
- For paralyzable systems, the brightest regions in the ICR map may appear as dips in the elemental maps.

### Automated Detection

```python
import numpy as np
from scipy import stats


def detect_dead_time_saturation(icr_map, ocr_map, element_map=None,
                                 warning_threshold=0.10,
                                 critical_threshold=0.30):
    """
    Assess dead-time saturation severity from ICR and OCR maps.

    Parameters
    ----------
    icr_map : np.ndarray
        2D map of input (incoming) count rate per pixel.
    ocr_map : np.ndarray
        2D map of output (recorded) count rate per pixel.
    element_map : np.ndarray or None
        Optional elemental map to check for saturation plateau.
    warning_threshold : float
        Dead-time fraction above which to issue a warning.
    critical_threshold : float
        Dead-time fraction above which data is severely compromised.

    Returns
    -------
    dict with diagnostic information
    """
    valid = icr_map > 0
    dt_fraction = np.zeros_like(icr_map, dtype=float)
    dt_fraction[valid] = 1.0 - (ocr_map[valid] / icr_map[valid])

    max_dt = float(np.max(dt_fraction[valid]))
    mean_dt = float(np.mean(dt_fraction[valid]))
    fraction_warning = float(np.sum(dt_fraction > warning_threshold) / np.sum(valid))
    fraction_critical = float(np.sum(dt_fraction > critical_threshold) / np.sum(valid))

    # Estimate dead time constant tau from ICR-OCR relationship
    # For non-paralyzable: OCR = ICR / (1 + ICR * tau)
    # Rearranging: 1/OCR = 1/ICR + tau
    icr_flat = icr_map[valid].ravel().astype(float)
    ocr_flat = ocr_map[valid].ravel().astype(float)

    with np.errstate(divide='ignore'):
        inv_ocr = 1.0 / ocr_flat
        inv_icr = 1.0 / icr_flat
        finite = np.isfinite(inv_ocr) & np.isfinite(inv_icr)

    if np.sum(finite) > 10:
        slope, intercept, _, _, _ = stats.linregress(inv_icr[finite], inv_ocr[finite])
        tau_estimated = intercept  # dead time in same units as count rate^-1
    else:
        tau_estimated = None

    # Check for paralyzable rollover
    paralyzable = False
    if element_map is not None:
        high_icr = icr_map > np.percentile(icr_map[valid], 95)
        if np.sum(high_icr) > 5:
            high_icr_elem = np.mean(element_map[high_icr])
            mid_icr = (icr_map > np.percentile(icr_map[valid], 70)) & \
                      (icr_map < np.percentile(icr_map[valid], 85))
            mid_icr_elem = np.mean(element_map[mid_icr])
            if high_icr_elem < mid_icr_elem * 0.9:
                paralyzable = True

    severity = 'none'
    if max_dt > critical_threshold:
        severity = 'critical'
    elif max_dt > warning_threshold:
        severity = 'warning'

    return {
        'max_dead_time_fraction': max_dt,
        'mean_dead_time_fraction': mean_dt,
        'fraction_pixels_warning': fraction_warning,
        'fraction_pixels_critical': fraction_critical,
        'estimated_tau': tau_estimated,
        'paralyzable_rollover_detected': paralyzable,
        'severity': severity,
    }
```

## Solutions and Mitigation

### Prevention (Before Data Collection)

- Monitor the ICR/OCR ratio in real time during scan setup; keep dead-time fraction below 10%.
- Reduce incident beam flux using attenuators or detuning the monochromator when dead time exceeds threshold.
- Use faster detector electronics with shorter shaping times (at the cost of slightly worse energy resolution).
- Move the detector farther from the sample to reduce solid angle if count rates are too high.
- Use a multi-element detector to distribute the count rate across multiple channels.

### Correction — Traditional Methods

```python
import numpy as np


def correct_dead_time_nonparalyzable(element_map, icr_map, ocr_map):
    """
    Apply dead-time correction to an XRF elemental map using the
    ICR/OCR ratio (non-paralyzable detector model).

    The correction factor is simply ICR/OCR at each pixel: the
    elemental map (derived from the processed spectrum, which is
    proportional to OCR) must be scaled up by the fraction of
    photons that were lost.

    Parameters
    ----------
    element_map : np.ndarray
        2D elemental map (counts or concentration) from spectral fitting.
    icr_map : np.ndarray
        2D map of input count rate per pixel.
    ocr_map : np.ndarray
        2D map of output count rate per pixel.

    Returns
    -------
    np.ndarray — dead-time corrected elemental map
    """
    correction = np.ones_like(element_map, dtype=float)
    valid = ocr_map > 0
    correction[valid] = icr_map[valid].astype(float) / ocr_map[valid].astype(float)

    # Safety cap: reject correction factors > 10x (data too saturated to recover)
    correction = np.clip(correction, 1.0, 10.0)

    corrected = element_map.astype(float) * correction
    return corrected


def correct_dead_time_paralyzable(element_map, icr_map, ocr_map, tau,
                                   max_iterations=50, tolerance=1e-6):
    """
    Apply dead-time correction for a paralyzable detector system
    using iterative Newton-Raphson to invert the paralyzable model.

    Model: OCR = ICR_true * exp(-ICR_true * tau)

    Parameters
    ----------
    element_map : np.ndarray
        2D elemental map from spectral fitting.
    icr_map : np.ndarray
        2D input count rate map (may itself be underestimated
        if the detector's ICR measurement is also affected).
    ocr_map : np.ndarray
        2D output count rate map.
    tau : float
        Dead time constant in seconds (e.g., 1e-6 for 1 microsecond).
    max_iterations : int
        Maximum Newton-Raphson iterations.
    tolerance : float
        Convergence tolerance for relative change.

    Returns
    -------
    np.ndarray — dead-time corrected elemental map
    """
    # Solve for true ICR using Newton-Raphson on: f(n) = n*exp(-n*tau) - OCR = 0
    ocr = ocr_map.astype(float).ravel()
    n_true = ocr.copy()  # initial guess

    for _ in range(max_iterations):
        exp_term = np.exp(-n_true * tau)
        f = n_true * exp_term - ocr
        f_prime = exp_term * (1.0 - n_true * tau)
        f_prime = np.where(np.abs(f_prime) < 1e-30, 1e-30, f_prime)

        delta = f / f_prime
        n_true -= delta

        n_true = np.maximum(n_true, ocr)  # true rate >= observed rate

        if np.max(np.abs(delta) / (n_true + 1e-30)) < tolerance:
            break

    correction = np.ones_like(ocr)
    valid = ocr > 0
    correction[valid] = n_true[valid] / ocr[valid]
    correction = np.clip(correction, 1.0, 20.0)

    corrected = element_map.astype(float) * correction.reshape(element_map.shape)
    return corrected
```

### Correction — AI/ML Methods

For detectors with complex dead-time behavior that does not follow simple paralyzable or non-paralyzable models, a neural network can be trained to learn the true ICR-to-OCR mapping from calibration data collected with known standards at varying count rates. This approach captures any detector-specific nonlinearities, pile-up effects, and channel-dependent dead times that analytical models cannot fully describe.

## Impact If Uncorrected

Dead-time saturation compresses the dynamic range of elemental maps, making high-concentration regions appear similar to medium-concentration regions. Quantitative analysis becomes unreliable — concentration calibration curves derived from standards at one count rate do not apply at different count rates. In paralyzable systems, the concentration inversion effect can cause the most element-rich regions to appear depleted, completely misleading the analyst. Elemental ratios also become distorted because different elements contribute different amounts to the total count rate and thus experience different dead-time fractions.

## Related Resources

- [MAPS Workflow Analysis](../../05_tools_and_code/maps_software/workflow_analysis.md) — dead-time correction in MAPS processing pipeline
- Related artifact: [I0 Normalization](i0_normalization.md) — should be applied after dead-time correction
- Related artifact: [Dead/Hot Pixel](dead_hot_pixel.md) — saturated detector channels can mimic hot pixels

## Key Takeaway

Always check the ICR/OCR ratio before trusting quantitative XRF maps. If the dead-time fraction exceeds 10% anywhere in the scan, apply the ICR/OCR correction to all elemental maps. For critical quantitative work, keep dead time below 5% by adjusting beam intensity or detector geometry during data collection.
