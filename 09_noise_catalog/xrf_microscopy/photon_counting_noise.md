# Photon Counting Noise

## Classification

| Attribute | Value |
|-----------|-------|
| **Modality** | XRF Microscopy |
| **Noise Type** | Statistical |
| **Severity** | Major |
| **Frequency** | Always |
| **Detection Difficulty** | Easy |

## Visual Examples

```
 Noisy elemental map (Fe, 1 ms dwell)       Cleaner map (Fe, 50 ms dwell)
 ┌─────────────────────────────┐            ┌─────────────────────────────┐
 │ ·  ▒  ·  ░  ·  ▒  ·  ░  · │            │ ░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░ │
 │ ░  ·  ▒  ·  ░  ·  ▒  ·  ░ │            │ ░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░░░░░░ │
 │ ·  ░  ·  ▒  ·  ░  ·  ▒  · │            │ ░▒▒▓▓████████▓▓▓▒▒░░░░░░░ │
 │ ▒  ·  ░  ·  ▒  ·  ░  ·  ▒ │            │ ░▒▓▓██████████▓▓▒▒░░░░░░░ │
 │ ·  ▒  ·  ░  ·  ▒  ·  ░  · │            │ ░░▒▒▓▓▓▓▓▓▓▓▓▓▓▒▒░░░░░░░ │
 │ ░  ·  ▒  ·  ░  ·  ▒  ·  ░ │            │ ░░░▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░ │
 └─────────────────────────────┘            └─────────────────────────────┘
 Mean ~4 counts/pixel → SNR ~2              Mean ~2500 counts/pixel → SNR ~50
```

## Description

Photon counting noise (also called Poisson noise or shot noise) produces grainy, speckled elemental maps where each pixel's measured intensity fluctuates around the true value. The noise magnitude follows Poisson statistics: the standard deviation equals the square root of the number of detected photons, so the signal-to-noise ratio scales as sqrt(N). This is the fundamental limiting noise source in all XRF microscopy measurements, especially pronounced for trace elements, short dwell times, or low-flux beamlines.

## Root Cause

Each pixel in an XRF map records the number of fluorescence photons detected during the dwell time at that position. The photon arrival process is inherently stochastic and follows Poisson statistics. When the total number of detected photons N per pixel is small (e.g., N < 100), the relative uncertainty 1/sqrt(N) exceeds 10%, producing visibly noisy maps. Low dwell times, dilute elements, low beam flux, and thick matrices that absorb fluorescence all reduce N and amplify the noise.

## Quick Diagnosis

```python
import numpy as np

# Load elemental map (2D array, units = photon counts)
# fe_map = ...
mean_counts = np.mean(fe_map[fe_map > 0])
expected_snr = np.sqrt(mean_counts)
print(f"Mean counts/pixel: {mean_counts:.1f}")
print(f"Expected Poisson SNR: {expected_snr:.1f}")
print(f"Relative noise: {100/expected_snr:.1f}%")
```

## Detection Methods

### Visual Indicators

- Elemental map appears grainy or speckled, especially in low-concentration regions.
- Noise is spatially uniform (not correlated with scan direction or detector geometry).
- Zooming in reveals random pixel-to-pixel fluctuations with no spatial structure.
- Histogram of pixel values shows a broad, asymmetric distribution (Poisson-like) rather than tight peaks.

### Automated Detection

```python
import numpy as np
from scipy import ndimage


def diagnose_photon_noise(element_map, min_counts_threshold=100):
    """
    Assess the severity of photon counting noise in an XRF elemental map.

    Parameters
    ----------
    element_map : np.ndarray
        2D array of photon counts per pixel.
    min_counts_threshold : int
        Minimum counts/pixel for acceptable SNR (~10%).

    Returns
    -------
    dict with keys:
        'mean_counts' : float — average counts per pixel
        'median_counts' : float — median counts per pixel
        'expected_snr' : float — theoretical Poisson SNR
        'observed_snr' : float — measured SNR from local variance
        'fraction_low_counts' : float — fraction of pixels below threshold
        'noise_limited' : bool — True if map is severely noise-limited
    """
    valid = element_map[element_map > 0]
    if len(valid) == 0:
        return {'noise_limited': True, 'mean_counts': 0.0}

    mean_counts = np.mean(valid)
    median_counts = np.median(valid)
    expected_snr = np.sqrt(mean_counts)

    # Estimate local variance using Laplacian (high-frequency content)
    laplacian = ndimage.laplace(element_map.astype(float))
    local_noise = np.std(laplacian) / np.sqrt(6)  # normalization for Laplacian kernel
    observed_snr = mean_counts / local_noise if local_noise > 0 else np.inf

    fraction_low = np.sum(element_map < min_counts_threshold) / element_map.size

    return {
        'mean_counts': float(mean_counts),
        'median_counts': float(median_counts),
        'expected_snr': float(expected_snr),
        'observed_snr': float(observed_snr),
        'fraction_low_counts': float(fraction_low),
        'noise_limited': mean_counts < min_counts_threshold,
    }
```

## Solutions and Mitigation

### Prevention (Before Data Collection)

- Increase per-pixel dwell time: doubling the dwell doubles counts and improves SNR by ~1.4x.
- Use the highest available beam flux (undulator harmonics, KB mirror focusing).
- For trace elements, consider running multiple fast scans and summing rather than one slow scan (reduces risk of drift).
- Position the fluorescence detector as close to the sample as geometry allows to maximize solid angle.

### Correction — Traditional Methods

```python
import numpy as np
from scipy import ndimage


def denoise_xrf_map_traditional(element_map, method='gaussian', **kwargs):
    """
    Reduce photon counting noise in an XRF elemental map using
    traditional filtering approaches.

    Parameters
    ----------
    element_map : np.ndarray
        2D array of photon counts.
    method : str
        'gaussian' — Gaussian blur (default sigma=1.0)
        'median'   — median filter (default size=3)
        'binning'  — spatial binning (default factor=2)
        'wiener'   — Wiener filter (default noise_var=None, auto-estimated)

    Returns
    -------
    np.ndarray — denoised map
    """
    if method == 'gaussian':
        sigma = kwargs.get('sigma', 1.0)
        return ndimage.gaussian_filter(element_map.astype(float), sigma=sigma)

    elif method == 'median':
        size = kwargs.get('size', 3)
        return ndimage.median_filter(element_map.astype(float), size=size)

    elif method == 'binning':
        factor = kwargs.get('factor', 2)
        ny, nx = element_map.shape
        ny_new = ny // factor
        nx_new = nx // factor
        cropped = element_map[:ny_new * factor, :nx_new * factor]
        binned = cropped.reshape(ny_new, factor, nx_new, factor).sum(axis=(1, 3))
        return binned

    elif method == 'wiener':
        from scipy.signal import wiener
        noise_var = kwargs.get('noise_var', None)
        if noise_var is None:
            # Estimate noise variance from Poisson assumption
            noise_var = np.mean(element_map[element_map > 0])
        return wiener(element_map.astype(float), noise=noise_var)

    else:
        raise ValueError(f"Unknown method: {method}")


def sum_repeated_scans(scan_list):
    """
    Sum multiple repeat scans to improve photon statistics.
    SNR improves by sqrt(N_scans).

    Parameters
    ----------
    scan_list : list of np.ndarray
        List of 2D elemental maps from repeated scans.

    Returns
    -------
    np.ndarray — summed map
    """
    stacked = np.stack(scan_list, axis=0)
    return np.sum(stacked, axis=0)
```

### Correction — AI/ML Methods

Deep learning denoising has shown remarkable performance for XRF maps with very low photon counts. A convolutional neural network (e.g., a U-Net or residual network) trained on pairs of noisy and high-count reference maps learns to predict the underlying signal. The Noise2Noise approach is particularly useful when clean reference data is unavailable — the network trains on pairs of independent noisy measurements of the same sample. These methods can recover spatial features that are completely obscured by noise in individual maps, effectively achieving the SNR of much longer acquisitions.

## Impact If Uncorrected

Photon counting noise directly limits the minimum detectable concentration and the spatial resolution of elemental maps. In noisy maps, trace elements become undetectable, boundaries between compositional domains blur into noise, and quantitative concentration estimates carry large uncertainties. Downstream analysis such as elemental correlation plots, principal component analysis, and segmentation all degrade when the input maps are dominated by Poisson noise.

## Related Resources

- [XRF EDA notebook](../../06_data_structures/eda/xrf_eda.md) — quality metrics and noise assessment for XRF maps
- Related artifact: [Dead/Hot Pixel](dead_hot_pixel.md) — extreme single-pixel anomalies distinct from statistical noise
- Related artifact: [Peak Overlap](peak_overlap.md) — spectral interference can reduce effective counts for a given element

## Key Takeaway

Photon counting noise is the fundamental floor of XRF microscopy data quality. Always check mean counts per pixel before analysis — if counts are below ~100, consider longer dwell times, scan summation, or spatial binning before attempting quantitative interpretation.
