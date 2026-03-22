# Zinger (Gamma-ray Spike)

## Classification

| Attribute | Value |
|-----------|-------|
| **Modality** | Tomography |
| **Noise Type** | Instrumental |
| **Severity** | Major |
| **Frequency** | Occasional |
| **Detection Difficulty** | Easy |

## Visual Examples

![Before and after](../images/zinger_before_after.png)

## Description

Zingers appear as isolated, extremely bright spots (single pixels or small clusters) in individual projection images. They are transient events that typically affect only one or a few projections at random spatial locations. In the sinogram, they manifest as isolated bright dots rather than continuous vertical stripes, distinguishing them from ring artifacts.

## Root Cause

Zingers are caused by high-energy cosmic rays or gamma-ray photons striking the detector directly, depositing large amounts of energy in a single pixel or small pixel cluster during a single exposure. They can also originate from radioactive decay events in the scintillator material itself, electronic noise bursts in the readout circuitry, or secondary radiation from the beamline shielding. Their occurrence is stochastic and independent of the sample or beam characteristics.

## Quick Diagnosis

```python
import numpy as np

# Compare consecutive projections to find transient bright spots
diff = np.abs(projections[i + 1].astype(float) - projections[i].astype(float))
# Zingers produce isolated extreme values in the difference image
threshold = np.median(diff) + 10 * np.median(np.abs(diff - np.median(diff)))
zinger_mask = diff > threshold
print(f"Zinger pixels in projection {i}: {np.sum(zinger_mask)}")
```

## Detection Methods

### Visual Indicators

- Extremely bright single-pixel or few-pixel spots in individual projections.
- Spots appear randomly and do not persist across consecutive projections.
- In the sinogram, zingers appear as isolated bright dots (not vertical stripes).
- Affected pixels have values far exceeding the expected dynamic range of the sample.

### Automated Detection

```python
import numpy as np
from scipy import ndimage


def detect_zingers(projections, sigma_threshold=10.0, kernel_size=3):
    """
    Detect zinger events across a projection stack by comparing each
    projection to a local median baseline.

    Parameters
    ----------
    projections : np.ndarray
        3D array of shape (num_projections, height, width).
    sigma_threshold : float
        Number of MAD-scaled deviations above local median to flag.
    kernel_size : int
        Size of the median filter kernel for local baseline.

    Returns
    -------
    dict with keys:
        'zinger_mask' : np.ndarray — boolean mask same shape as input
        'zinger_count_per_proj' : np.ndarray — count per projection
        'total_zingers' : int
    """
    zinger_mask = np.zeros(projections.shape, dtype=bool)
    counts = np.zeros(projections.shape[0], dtype=int)

    for i in range(projections.shape[0]):
        proj = projections[i].astype(np.float64)

        # Local median baseline
        baseline = ndimage.median_filter(proj, size=kernel_size)
        residual = proj - baseline

        # Robust scale estimate using MAD
        mad = np.median(np.abs(residual))
        mad = mad if mad > 0 else 1e-10
        threshold = sigma_threshold * 1.4826 * mad

        mask_i = residual > threshold
        zinger_mask[i] = mask_i
        counts[i] = np.sum(mask_i)

    return {
        "zinger_mask": zinger_mask,
        "zinger_count_per_proj": counts,
        "total_zingers": int(np.sum(counts)),
    }
```

## Solutions and Mitigation

### Prevention (Before Data Collection)

- Use longer individual exposures to improve signal-to-zinger ratio (zingers affect a fixed number of frames regardless of exposure time).
- Acquire multiple frames per projection angle and combine with a median or trimmed-mean to reject outliers.
- Shield the detector from ambient radiation where possible.

### Correction — Traditional Methods

Median filtering or outlier replacement on the projection stack is highly effective because zingers are spatially and temporally isolated.

```python
import tomopy
import numpy as np
from scipy import ndimage


# Method 1: TomoPy built-in outlier removal
# Replaces pixels that deviate by more than `dif` from the median of neighbors
proj_clean = tomopy.remove_outlier(proj, dif=500, size=3)

# Method 2: Temporal median filter across adjacent projections
def remove_zingers_temporal(projections, window=3, threshold=5.0):
    """Replace zinger pixels using temporal median of neighboring projections."""
    cleaned = projections.copy().astype(np.float64)
    half_w = window // 2

    for i in range(projections.shape[0]):
        lo = max(0, i - half_w)
        hi = min(projections.shape[0], i + half_w + 1)
        neighborhood = projections[lo:hi].astype(np.float64)
        local_median = np.median(neighborhood, axis=0)
        residual = cleaned[i] - local_median
        mad = np.median(np.abs(residual))
        mad = mad if mad > 0 else 1e-10
        mask = residual > threshold * 1.4826 * mad
        cleaned[i][mask] = local_median[mask]

    return cleaned


proj_clean = remove_zingers_temporal(proj, window=5, threshold=8.0)
```

### Correction — AI/ML Methods

No established AI/ML methods for this artifact type. Zingers are effectively handled by simple statistical outlier detection, making deep learning approaches unnecessary. However, general-purpose denoising networks (e.g., Noise2Noise) may incidentally suppress zingers as part of broader noise removal.

## Impact If Uncorrected

Uncorrected zingers propagate through the reconstruction as streak-like artifacts radiating from the zinger location in the sinogram. A single bright zinger pixel can corrupt an entire row in the reconstructed slice. Multiple zingers across different projections create a pattern of short, randomly oriented streaks that degrade image quality and interfere with segmentation, especially in low-contrast regions.

## Related Resources

- [Tomography EDA notebook](../../06_data_structures/eda/tomo_eda.md) — projection quality inspection
- Related artifact: [Ring Artifact](ring_artifact.md) — persistent pixel defects (vs. transient zingers)
- Related artifact: [Low-Dose Poisson Noise](low_dose_noise.md) — statistical noise that also benefits from outlier filtering

## Key Takeaway

Zingers are transient bright-pixel events caused by cosmic rays or detector noise bursts. They are easy to detect by comparing consecutive projections and straightforward to remove with median-based outlier replacement — always apply zinger removal before reconstruction.
