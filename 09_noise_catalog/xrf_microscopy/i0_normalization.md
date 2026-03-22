# I0 Normalization Issues

## Classification

| Attribute | Value |
|-----------|-------|
| **Modality** | XRF Microscopy |
| **Noise Type** | Systematic |
| **Severity** | Major |
| **Frequency** | Common |
| **Detection Difficulty** | Easy |

## Visual Examples

![Before and after — synthetic XRF map with I0 beam current drops](../images/i0_drop_before_after.png)

> **Image source:** Synthetic — simulated XRF map with I0 beam current drops causing horizontal stripes, corrected via I0 normalization.

```
 I0 monitor signal during scan           Unnormalized Fe map
 (beam current over time)
   │ ╲                                    ┌──────────────────────────┐
   │  ╲                                   │ ▓▓▓▓▓▓██████▓▓▓▒▒▒░░░░ │
   │   ╲─────                             │ ▓▓▓▓▓██████▓▓▓▒▒░░░░░░ │
   │        ╲                             │ ▓▓▓▓██████▓▓▒▒▒░░░░░░░ │ ← gradient
   │         ╲─────────                   │ ▓▓▓██████▓▓▒▒░░░░░░░░░ │   follows I0
   │                  ──                  │ ▓▓██████▓▓▒▒░░░░░░░░░░ │   not sample
   └──────────────────────►               └──────────────────────────┘
   Row 0                Row N

 I0-normalized Fe map (corrected)
 ┌──────────────────────────┐
 │ ░░░░▒▒▓▓████▓▓▒▒░░░░░░░ │
 │ ░░░▒▒▓▓████████▓▒▒░░░░░ │
 │ ░░░▒▒▓▓████████▓▒▒░░░░░ │  ← true sample structure
 │ ░░░▒▒▓▓████████▓▒▒░░░░░ │     revealed after I0
 │ ░░░░▒▒▓▓████▓▓▒▒░░░░░░░ │     normalization
 └──────────────────────────┘
```

## Description

I0 normalization issues arise when the incident X-ray beam intensity varies during a raster scan but this variation is not corrected in the elemental maps. The fluorescence signal at each pixel is proportional to both the elemental concentration and the incident flux, so any spatial or temporal variation in I0 imprints directly onto the elemental maps as a systematic intensity gradient or pattern. This is one of the most common and easily correctable artifacts in XRF microscopy, yet it is frequently overlooked, leading to apparent compositional trends that are purely instrumental.

## Root Cause

The incident beam intensity at synchrotron sources varies for several reasons: storage ring current decays between top-up injections (typically 1-2% variation), monochromator thermal drift changes the flux over minutes to hours, and beam position instabilities shift the beam relative to focusing optics. An I0 ion chamber or diode placed upstream of the sample records these fluctuations. If the raw fluorescence counts are not divided by the corresponding I0 value at each pixel, the resulting elemental maps contain a multiplicative artifact proportional to the I0 variation pattern. Fly-scan (continuous motion) measurements are particularly susceptible because I0 can vary significantly across a long scan line.

## Quick Diagnosis

```python
import numpy as np

# i0_map: 2D map of I0 values (same shape as elemental maps)
i0_variation = (np.max(i0_map) - np.min(i0_map)) / np.mean(i0_map)
row_means = np.mean(i0_map, axis=1)
row_trend = (row_means[-1] - row_means[0]) / np.mean(row_means)
print(f"I0 total variation: {i0_variation:.1%}")
print(f"I0 row trend (first→last): {row_trend:+.1%}")
print(f"Normalization needed: {'YES' if i0_variation > 0.02 else 'Probably not'}")
```

## Detection Methods

### Visual Indicators

- Elemental maps show a smooth gradient that does not correspond to any sample feature.
- The gradient direction correlates with the scan direction (row-by-row or column-by-column).
- All elemental maps show the same gradient pattern (since all scale with I0).
- Plotting row or column averages reveals a systematic trend matching the I0 time series.

### Automated Detection

```python
import numpy as np
from scipy import stats


def detect_i0_normalization_issue(element_map, i0_map,
                                   correlation_threshold=0.5):
    """
    Detect whether an elemental map contains an I0-correlated artifact.

    Parameters
    ----------
    element_map : np.ndarray
        2D elemental fluorescence map (raw, unnormalized).
    i0_map : np.ndarray
        2D map of I0 monitor values at each pixel.
    correlation_threshold : float
        Minimum correlation to flag as I0-affected.

    Returns
    -------
    dict with diagnostic information
    """
    # Row-averaged comparison
    elem_row_avg = np.mean(element_map.astype(float), axis=1)
    i0_row_avg = np.mean(i0_map.astype(float), axis=1)

    corr_row, p_row = stats.pearsonr(elem_row_avg, i0_row_avg)

    # Column-averaged comparison
    elem_col_avg = np.mean(element_map.astype(float), axis=0)
    i0_col_avg = np.mean(i0_map.astype(float), axis=0)

    corr_col, p_col = stats.pearsonr(elem_col_avg, i0_col_avg)

    # I0 variation statistics
    i0_flat = i0_map.astype(float)
    i0_cv = np.std(i0_flat) / np.mean(i0_flat)
    i0_range = (np.max(i0_flat) - np.min(i0_flat)) / np.mean(i0_flat)

    # Check if element map variation tracks I0 pixel-by-pixel
    pixel_corr, pixel_p = stats.pearsonr(
        element_map.ravel().astype(float),
        i0_map.ravel().astype(float)
    )

    needs_normalization = (
        (abs(corr_row) > correlation_threshold) or
        (abs(corr_col) > correlation_threshold)
    ) and i0_range > 0.02

    return {
        'row_correlation': float(corr_row),
        'col_correlation': float(corr_col),
        'pixel_correlation': float(pixel_corr),
        'i0_cv': float(i0_cv),
        'i0_range_fraction': float(i0_range),
        'needs_normalization': needs_normalization,
    }
```

## Solutions and Mitigation

### Prevention (Before Data Collection)

- Ensure the I0 ion chamber or photodiode is properly positioned upstream and recording at each pixel.
- Verify I0 signal quality before starting the scan: stable, not saturated, adequate count rate.
- Use top-up mode at the synchrotron to minimize ring current decay.
- For long scans, check monochromator stability and beam position feedback systems.

### Correction — Traditional Methods

```python
import numpy as np


def normalize_by_i0(element_map, i0_map, i0_reference=None):
    """
    Normalize an XRF elemental map by the incident beam intensity (I0).

    Parameters
    ----------
    element_map : np.ndarray
        2D raw fluorescence intensity map.
    i0_map : np.ndarray
        2D map of I0 monitor values (same shape as element_map).
    i0_reference : float or None
        Reference I0 value to scale the result. If None, uses the
        mean I0 so that the normalized map has similar magnitude
        to the raw map.

    Returns
    -------
    np.ndarray — I0-normalized elemental map
    """
    if i0_reference is None:
        i0_reference = np.mean(i0_map[i0_map > 0])

    i0_safe = i0_map.astype(float).copy()
    i0_safe[i0_safe <= 0] = np.nan  # avoid division by zero

    normalized = element_map.astype(float) / i0_safe * i0_reference

    # Replace NaN (from zero I0) with zero
    normalized = np.nan_to_num(normalized, nan=0.0)
    return normalized


def normalize_all_elements(element_maps_dict, i0_map, i0_reference=None):
    """
    Apply I0 normalization to a dictionary of elemental maps.

    Parameters
    ----------
    element_maps_dict : dict
        {element_name: 2D_map} dictionary of raw elemental maps.
    i0_map : np.ndarray
        2D I0 monitor map.
    i0_reference : float or None
        Reference I0 for scaling.

    Returns
    -------
    dict — {element_name: normalized_2D_map}
    """
    if i0_reference is None:
        i0_reference = np.mean(i0_map[i0_map > 0])

    normalized = {}
    for elem, emap in element_maps_dict.items():
        normalized[elem] = normalize_by_i0(emap, i0_map, i0_reference)
    return normalized


def diagnose_i0_quality(i0_map, max_dropout_fraction=0.01,
                         max_spike_sigma=5.0):
    """
    Check I0 map quality before using it for normalization.
    Bad I0 values (dropouts, spikes) will corrupt the normalized maps.

    Parameters
    ----------
    i0_map : np.ndarray
        2D I0 monitor map.
    max_dropout_fraction : float
        Maximum allowed fraction of zero/near-zero I0 pixels.
    max_spike_sigma : float
        Threshold for detecting I0 spikes (in MAD units).

    Returns
    -------
    dict with I0 quality metrics
    """
    i0 = i0_map.astype(float).ravel()
    i0_positive = i0[i0 > 0]

    if len(i0_positive) == 0:
        return {'quality': 'FAIL', 'reason': 'No positive I0 values'}

    median_i0 = np.median(i0_positive)
    mad_i0 = np.median(np.abs(i0_positive - median_i0))

    # Dropout detection
    dropout_threshold = median_i0 * 0.1
    n_dropouts = np.sum(i0 < dropout_threshold)
    dropout_fraction = n_dropouts / len(i0)

    # Spike detection
    if mad_i0 > 0:
        z_scores = np.abs(i0 - median_i0) / (1.4826 * mad_i0)
        n_spikes = np.sum(z_scores > max_spike_sigma)
    else:
        n_spikes = 0

    quality = 'GOOD'
    issues = []
    if dropout_fraction > max_dropout_fraction:
        quality = 'WARNING'
        issues.append(f'{n_dropouts} I0 dropouts ({dropout_fraction:.2%})')
    if n_spikes > 0:
        quality = 'WARNING'
        issues.append(f'{n_spikes} I0 spikes detected')

    return {
        'quality': quality,
        'median_i0': float(median_i0),
        'i0_cv': float(np.std(i0_positive) / median_i0),
        'n_dropouts': int(n_dropouts),
        'dropout_fraction': float(dropout_fraction),
        'n_spikes': int(n_spikes),
        'issues': issues,
    }
```

### Correction — AI/ML Methods

I0 normalization is straightforward division and does not typically require ML approaches. However, in cases where the I0 monitor itself is noisy or has dropouts, a learned smooth model of the I0 temporal evolution (e.g., Gaussian process regression on the I0 time series) can provide a more stable normalization signal than the raw I0 values.

## Impact If Uncorrected

Without I0 normalization, all elemental maps contain a multiplicative artifact that mirrors the beam intensity pattern. This creates false compositional gradients across the sample, corrupts quantitative concentration estimates, and can lead to incorrect conclusions about spatial relationships between elements. Since all elements are equally affected, elemental ratios are preserved, but absolute concentrations and any comparison between different scan regions become unreliable. Maps collected at different times or on different beamlines cannot be compared without proper normalization.

## Related Resources

- [XRF EDA notebook](../../06_data_structures/eda/xrf_eda.md) — I0 quality checks and normalization workflows
- Related artifact: [Dead-Time Saturation](dead_time_saturation.md) — dead-time correction should precede I0 normalization
- Related artifact: [Scan Stripe](scan_stripe.md) — I0 fluctuations within a scan line cause stripes

## Key Takeaway

I0 normalization is the single most important correction step in XRF data processing. Always divide every elemental map by the corresponding I0 map before any quantitative analysis. Check I0 quality first — dropouts and spikes in the I0 monitor will propagate as artifacts into all normalized maps.
