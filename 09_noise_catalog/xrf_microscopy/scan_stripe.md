# Scan Stripe

## Classification

| Attribute | Value |
|-----------|-------|
| **Modality** | XRF Microscopy |
| **Noise Type** | Systematic |
| **Severity** | Major |
| **Frequency** | Occasional |
| **Detection Difficulty** | Easy |

## Visual Examples

```
 Striped Fe map (raw)                 After destriping correction
 ┌────────────────────────────┐      ┌────────────────────────────┐
 │ ░░░░▒▒▓▓████▓▓▒▒░░░░░░░░░ │ ←dim │ ░░░░▒▒▓▓████▓▓▒▒░░░░░░░░░ │
 │▒▒▒▒▓▓▓████████▓▓▒▒░░░░░░░ │ ←hot │ ░░░░▒▒▓▓████▓▓▒▒░░░░░░░░░ │
 │ ░░░░▒▒▓▓████▓▓▒▒░░░░░░░░░ │ ←dim │ ░░░░▒▒▓▓████▓▓▒▒░░░░░░░░░ │
 │▒▒▒▒▓▓▓████████▓▓▒▒░░░░░░░ │ ←hot │ ░░░░▒▒▓▓████▓▓▒▒░░░░░░░░░ │
 │ ░░░░▒▒▓▓████▓▓▒▒░░░░░░░░░ │ ←dim │ ░░░░▒▒▓▓████▓▓▒▒░░░░░░░░░ │
 │▒▒▒▒▓▓▓████████▓▓▒▒░░░░░░░ │ ←hot │ ░░░░▒▒▓▓████▓▓▒▒░░░░░░░░░ │
 └────────────────────────────┘      └────────────────────────────┘
 Alternating bright/dim rows          Uniform background after
 from bidirectional raster scan       row normalization

 Row mean profile:                    Frequency spectrum of row means:
   │ ╱╲  ╱╲  ╱╲  ╱╲                    │
   │╱  ╲╱  ╲╱  ╲╱  ╲                   │    ╱╲
   │                  ╲                 │   ╱  ╲     ← spike at
   └──────────────────────►             │  ╱    ╲      stripe frequency
   Row index                            └──────────────►
                                        Spatial frequency
```

## Description

Scan stripes are horizontal or vertical intensity bands visible in XRF elemental maps that align with the raster scan direction. They appear as alternating bright and dim rows (or columns), giving the map a "venetian blind" appearance. The stripes are systematic artifacts unrelated to the sample's true elemental distribution and affect all elements equally. They are most visible in flat, featureless regions of the sample and can mask subtle compositional variations.

## Root Cause

Several mechanisms produce scan stripes. In bidirectional (serpentine) raster scans, slight misalignment between the forward and reverse scan directions creates a one-pixel offset that manifests as alternating bright/dim row pairs. Beam position drift during the scan causes the illuminated spot to shift gradually, so that different rows sample slightly different positions or experience different flux. I0 fluctuations that are faster than the per-pixel dwell time but slower than a full scan line create row-to-row intensity variations. Stage encoder errors or stick-slip motion in the scanning stage cause uneven pixel spacing that appears as intensity modulation. Top-up injection events at the synchrotron storage ring can create sudden I0 jumps mid-scan.

## Quick Diagnosis

```python
import numpy as np

# element_map: 2D XRF elemental map
row_means = np.mean(element_map.astype(float), axis=1)
row_variation = np.std(row_means) / np.mean(row_means)
# Check for alternating pattern (high frequency in row means)
diff = np.diff(row_means)
sign_changes = np.sum(np.diff(np.sign(diff)) != 0)
alternating_fraction = sign_changes / len(diff)
print(f"Row-mean CV: {row_variation:.3f}")
print(f"Alternating pattern score: {alternating_fraction:.2f} (>0.6 = stripes)")
```

## Detection Methods

### Visual Indicators

- Regular horizontal or vertical banding across the entire map, independent of sample features.
- Stripes appear in all elemental maps at the same positions.
- Zooming out makes the stripe pattern more visible against the smooth sample structure.
- Row (or column) averages oscillate with a regular period, often period-2 for bidirectional scan artifacts.

### Automated Detection

```python
import numpy as np
from scipy import fft


def detect_scan_stripes(element_map, scan_direction='horizontal',
                         significance_threshold=3.0):
    """
    Detect scan-stripe artifacts by analyzing the frequency content
    of row (or column) averages.

    Parameters
    ----------
    element_map : np.ndarray
        2D elemental map.
    scan_direction : str
        'horizontal' (stripes along rows) or 'vertical' (along columns).
    significance_threshold : float
        Number of standard deviations above the mean spectral power
        to flag a stripe frequency.

    Returns
    -------
    dict with diagnostic information
    """
    img = element_map.astype(float)

    if scan_direction == 'horizontal':
        line_means = np.mean(img, axis=1)
    else:
        line_means = np.mean(img, axis=0)

    # Remove linear trend
    x = np.arange(len(line_means))
    coeffs = np.polyfit(x, line_means, 1)
    detrended = line_means - np.polyval(coeffs, x)

    # FFT of row/column means
    spectrum = np.abs(fft.rfft(detrended))
    freqs = fft.rfftfreq(len(detrended))

    # Skip DC component
    spectrum_no_dc = spectrum[1:]
    freqs_no_dc = freqs[1:]

    # Find significant peaks
    mean_power = np.mean(spectrum_no_dc)
    std_power = np.std(spectrum_no_dc)
    threshold = mean_power + significance_threshold * std_power

    peak_mask = spectrum_no_dc > threshold
    peak_freqs = freqs_no_dc[peak_mask]
    peak_powers = spectrum_no_dc[peak_mask]

    # Row-mean coefficient of variation
    cv = np.std(line_means) / np.mean(line_means)

    # Check for period-2 (bidirectional artifact)
    nyquist_idx = len(spectrum_no_dc) - 1
    period2_power = spectrum_no_dc[nyquist_idx] if len(spectrum_no_dc) > 0 else 0
    has_period2 = period2_power > threshold

    stripes_detected = len(peak_freqs) > 0 and cv > 0.01

    return {
        'stripes_detected': stripes_detected,
        'line_mean_cv': float(cv),
        'n_stripe_frequencies': int(np.sum(peak_mask)),
        'dominant_frequency': float(peak_freqs[np.argmax(peak_powers)]) if len(peak_freqs) > 0 else None,
        'has_bidirectional_artifact': has_period2,
        'peak_frequencies': peak_freqs.tolist(),
        'peak_powers': peak_powers.tolist(),
    }
```

## Solutions and Mitigation

### Prevention (Before Data Collection)

- Use unidirectional raster scanning (all rows scanned in the same direction) to avoid bidirectional offset artifacts.
- Ensure I0 normalization is applied (eliminates most flux-related stripes).
- Calibrate the stage encoder and verify uniform pixel spacing before the scan.
- Avoid starting scans immediately after top-up injection; wait for beam stabilization.
- Use fly-scan with continuous encoder triggering rather than step-scan for more uniform pixel timing.

### Correction — Traditional Methods

```python
import numpy as np
from scipy import ndimage, fft


def correct_row_normalization(element_map, scan_direction='horizontal'):
    """
    Remove scan stripes by normalizing each row (or column) to have
    the same mean intensity.

    Parameters
    ----------
    element_map : np.ndarray
        2D elemental map with stripe artifacts.
    scan_direction : str
        'horizontal' (normalize rows) or 'vertical' (normalize columns).

    Returns
    -------
    np.ndarray — destriped map
    """
    img = element_map.astype(float).copy()

    if scan_direction == 'horizontal':
        row_means = np.mean(img, axis=1, keepdims=True)
        global_mean = np.mean(img)
        row_means[row_means == 0] = 1.0
        img = img / row_means * global_mean
    else:
        col_means = np.mean(img, axis=0, keepdims=True)
        global_mean = np.mean(img)
        col_means[col_means == 0] = 1.0
        img = img / col_means * global_mean

    return img


def correct_bidirectional_offset(element_map, shift_pixels=1):
    """
    Fix the pixel offset between forward and reverse scan lines
    in a bidirectional (serpentine) raster scan.

    Parameters
    ----------
    element_map : np.ndarray
        2D elemental map from bidirectional scan.
    shift_pixels : int
        Number of pixels to shift odd rows. Use positive or negative
        values; optimize by minimizing row-to-row discontinuity.

    Returns
    -------
    np.ndarray — corrected map with aligned rows
    """
    corrected = element_map.astype(float).copy()

    # Shift every other row (the reverse-scanned lines)
    for i in range(1, corrected.shape[0], 2):
        corrected[i, :] = np.roll(corrected[i, :], shift_pixels)

    return corrected


def destripe_fourier(element_map, scan_direction='horizontal',
                      stripe_width=3):
    """
    Remove scan stripes using Fourier filtering. Suppress narrow
    frequency bands in the FFT that correspond to stripe periodicity.

    Parameters
    ----------
    element_map : np.ndarray
        2D elemental map with stripe artifacts.
    scan_direction : str
        'horizontal' or 'vertical'.
    stripe_width : int
        Width (in frequency bins) of the notch filter applied
        to suppress stripe frequencies.

    Returns
    -------
    np.ndarray — destriped map
    """
    img = element_map.astype(float)

    # 2D FFT
    fft_img = np.fft.fft2(img)
    fft_shifted = np.fft.fftshift(fft_img)

    ny, nx = img.shape
    cy, cx = ny // 2, nx // 2

    if scan_direction == 'horizontal':
        # Stripes = sharp features along vertical frequency axis
        # Zero out the vertical line in Fourier space (except DC)
        mask = np.ones_like(fft_shifted)
        mask[:, cx - stripe_width:cx + stripe_width + 1] = 0
        # Preserve the DC row
        mask[cy, :] = 1
    else:
        mask = np.ones_like(fft_shifted)
        mask[cy - stripe_width:cy + stripe_width + 1, :] = 0
        mask[:, cx] = 1

    fft_filtered = fft_shifted * mask
    fft_unshifted = np.fft.ifftshift(fft_filtered)
    corrected = np.real(np.fft.ifft2(fft_unshifted))

    return corrected


def optimize_bidirectional_shift(element_map, max_shift=5):
    """
    Automatically determine the optimal pixel shift to correct
    bidirectional scan offset by minimizing row-to-row discontinuity.

    Parameters
    ----------
    element_map : np.ndarray
        2D elemental map from bidirectional scan.
    max_shift : int
        Maximum shift to test in each direction.

    Returns
    -------
    int — optimal shift in pixels
    """
    best_shift = 0
    best_score = np.inf

    for shift in range(-max_shift, max_shift + 1):
        test = element_map.astype(float).copy()
        for i in range(1, test.shape[0], 2):
            test[i, :] = np.roll(test[i, :], shift)

        # Score: sum of absolute differences between adjacent rows
        row_diff = np.sum(np.abs(np.diff(test, axis=0)))
        if row_diff < best_score:
            best_score = row_diff
            best_shift = shift

    return best_shift
```

### Correction — AI/ML Methods

Deep learning destriping methods trained on paired striped/clean image data can remove complex stripe patterns that do not follow simple row-normalization models. Wavelet-domain destriping using learned thresholds has shown strong performance for remote sensing data with similar stripe artifacts and can be adapted for XRF maps. These methods preserve real sample features that happen to be aligned with the scan direction, which simple row normalization may incorrectly suppress.

## Impact If Uncorrected

Scan stripes corrupt visual interpretation by obscuring subtle compositional boundaries and creating artificial banding that can be misinterpreted as layered sample structure. Quantitative analysis is affected because pixel intensities in different rows are not on the same scale. Frequency-domain analyses (e.g., spatial correlation functions, power spectra) are dominated by the stripe signal rather than true sample structure. Downstream processing steps such as segmentation and clustering may group pixels by stripe position rather than by composition.

## Related Resources

- Related artifact: [I0 Normalization](i0_normalization.md) — proper I0 correction eliminates most flux-related stripes
- Related artifact: [Photon Counting Noise](photon_counting_noise.md) — noisy maps make stripe detection harder
- Related artifact: [Probe Blurring](probe_blurring.md) — destriping should precede deconvolution

## Real-World Before/After Examples

The following published sources provide real experimental before/after comparisons:

| Source | Type | Figure/Location | Description | License |
|--------|------|-----------------|-------------|---------|
| [Mak et al. 2014](https://doi.org/10.1016/j.sab.2014.06.012) | Paper | Multiple | Non-negative matrix analysis for effective feature extraction in X-ray spectromicroscopy — scan stripe correction in XRF data | -- |

**Key references with published before/after comparisons:**
- **Mak et al. (2014)**: Non-negative matrix analysis showing scan stripe correction in XRF microscopy data. DOI: 10.1016/j.sab.2014.06.012

## Key Takeaway

Scan stripes are a common raster-scan artifact with multiple possible causes (bidirectional offset, I0 drift, stage errors). Always inspect row and column averages for systematic oscillations, and apply row normalization or Fourier destriping before quantitative analysis. Using unidirectional scanning and proper I0 normalization prevents most stripe artifacts at the source.
