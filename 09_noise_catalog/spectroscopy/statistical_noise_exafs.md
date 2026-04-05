# Statistical Noise in EXAFS

## Classification

| Attribute | Value |
|-----------|-------|
| **Modality** | Spectroscopy |
| **Noise Type** | Statistical |
| **Severity** | Major |
| **Frequency** | Always |
| **Detection Difficulty** | Easy |

## Visual Examples

```
chi(k) with good statistics vs. noisy data:

  chi(k)
   0.8 |
   0.4 |  /\      /\        Clean (many scans merged)
   0.0 |_/  \  __/  \  __/\____
  -0.4 |     \/      \/
  -0.8 |
       +----+----+----+----+-----> k (Å⁻¹)
       2    4    6    8   10   12

  chi(k)
   0.8 |  *   *
   0.4 | /\* *  /\  *      Noisy (single scan)
   0.0 |/ *\* _/ *\ * _/\*_*__
  -0.4 |  *  \/ *   \/*  *
  -0.8 | *        *    *
       +----+----+----+----+-----> k (Å⁻¹)
       2    4    6    8   10   12

  |FT(chi)|                        |FT(chi)|
   ^                                ^
   |   /\                           |   /\
   |  /  \    /\                    |  / .\  . /\.
   | /    \  /  \                   | / .  \./. . \.  .
   |/      \/    \___               |/ .    . .    .\__..
   +--+--+--+--+--+-> R(Å)         +--+--+--+--+--+-> R(Å)
   0  1  2  3  4  5                 0  1  2  3  4  5
   Clean FT peaks                   Noisy: reduced peaks, raised baseline
```

## Description

Statistical noise in EXAFS arises from the finite number of photons collected at each energy point. The EXAFS signal chi(k) oscillates as a function of photoelectron wavenumber k, and its amplitude decays rapidly at high k (roughly as 1/k^2 due to the Debye-Waller factor and mean free path effects). Because the signal-to-noise ratio degrades sharply with increasing k, the high-k region of chi(k) is most affected by photon counting statistics. In Fourier-transformed spectra, this noise raises the baseline and reduces the apparent amplitude of coordination shell peaks, potentially masking weak scattering paths and degrading structural fits.

## Root Cause

The fundamental limitation is photon counting statistics. In transmission mode, the measured absorbance mu(E) = ln(I0/I1) depends on the ratio of incident (I0) and transmitted (I1) intensities. At energies well above the absorption edge, I1 approaches I0 (thin samples) or becomes very small (thick samples), both of which degrade the signal-to-noise ratio. In fluorescence mode, the fluorescence count rate is limited by detector solid angle, concentration, and dead-time. Since the EXAFS oscillation amplitude decreases with k while the noise floor remains roughly constant, the SNR drops precipitously in the high-k region. Short counting times, dilute samples, and detector inefficiency all exacerbate the problem.

## Quick Diagnosis

```python
import numpy as np

# Given chi(k) data arrays
# k: wavenumber array (Å⁻¹)
# chi: EXAFS chi(k) array

# Estimate noise from high-k region where signal is weakest
high_k_mask = k > 10.0  # typically noise-dominated region
if np.sum(high_k_mask) > 10:
    noise_estimate = np.std(chi[high_k_mask])
    signal_estimate = np.max(np.abs(chi[k < 8.0]))
    snr = signal_estimate / noise_estimate
    print(f"Estimated noise (k > 10): {noise_estimate:.6f}")
    print(f"Peak signal (k < 8):      {signal_estimate:.6f}")
    print(f"Estimated SNR:            {snr:.1f}")
    if snr < 5:
        print("WARNING: Poor SNR — consider merging more scans")
```

## Detection Methods

### Visual Indicators

- High-k region of chi(k) shows erratic oscillations that do not follow a smooth decaying envelope.
- k^2-weighted or k^3-weighted chi(k) amplifies the high-k noise, making it clearly visible as large irregular spikes.
- Fourier transform magnitude |chi(R)| shows an elevated baseline (noise floor) that can be comparable to real scattering peaks.
- Comparing successive scans reveals non-reproducible fluctuations at high k while low-k features remain consistent.

### Automated Detection

```python
import numpy as np
from scipy.signal import savgol_filter


def assess_exafs_noise(k, chi, k_min_signal=2.0, k_max_signal=8.0,
                       k_min_noise=10.0, k_max_noise=None,
                       k_weight=2):
    """
    Assess statistical noise level in EXAFS chi(k) data.

    Parameters
    ----------
    k : np.ndarray
        Photoelectron wavenumber array in Å⁻¹.
    chi : np.ndarray
        EXAFS chi(k) signal (dimensionless).
    k_min_signal : float
        Lower bound of k range for signal amplitude estimation.
    k_max_signal : float
        Upper bound of k range for signal amplitude estimation.
    k_min_noise : float
        Lower bound of k range for noise estimation.
    k_max_noise : float or None
        Upper bound of k range for noise estimation (None = use all).
    k_weight : int
        k-weighting exponent for analysis (commonly 2 or 3).

    Returns
    -------
    dict with keys:
        'noise_level' : float — RMS noise in high-k region (k-weighted)
        'signal_amplitude' : float — peak amplitude in signal region (k-weighted)
        'snr' : float — signal-to-noise ratio
        'noise_quality' : str — 'excellent', 'good', 'fair', or 'poor'
        'recommended_kmax' : float — suggested k_max for Fourier transform
    """
    if k_max_noise is None:
        k_max_noise = k[-1]

    chi_kw = chi * k ** k_weight

    # Signal region
    sig_mask = (k >= k_min_signal) & (k <= k_max_signal)
    signal_amplitude = np.max(np.abs(chi_kw[sig_mask])) if np.any(sig_mask) else 0.0

    # Noise region: fit smooth curve and measure residuals
    noise_mask = (k >= k_min_noise) & (k <= k_max_noise)
    if np.sum(noise_mask) < 10:
        return {
            'noise_level': np.nan,
            'signal_amplitude': signal_amplitude,
            'snr': np.nan,
            'noise_quality': 'insufficient data in noise region',
            'recommended_kmax': k_max_signal,
        }

    chi_noise = chi_kw[noise_mask]
    # Use Savitzky-Golay filter to estimate smooth trend
    window = min(11, len(chi_noise) // 2 * 2 + 1)
    if window >= 5:
        smooth = savgol_filter(chi_noise, window, polyorder=3)
        residuals = chi_noise - smooth
    else:
        residuals = chi_noise - np.mean(chi_noise)

    noise_level = np.sqrt(np.mean(residuals ** 2))
    snr = signal_amplitude / noise_level if noise_level > 0 else np.inf

    # Quality classification
    if snr > 50:
        quality = 'excellent'
    elif snr > 20:
        quality = 'good'
    elif snr > 5:
        quality = 'fair'
    else:
        quality = 'poor'

    # Recommend k_max: find where local noise exceeds half the local signal
    recommended_kmax = k_max_noise
    dk = k[1] - k[0]
    window_pts = max(int(1.0 / dk), 5)  # ~1 Å⁻¹ sliding window
    for i in range(len(k) - window_pts):
        local_rms = np.std(chi_kw[i:i + window_pts])
        local_amp = np.max(np.abs(chi_kw[i:i + window_pts]))
        if local_amp > 0 and local_rms / local_amp > 0.5 and k[i] > 6.0:
            recommended_kmax = k[i]
            break

    return {
        'noise_level': noise_level,
        'signal_amplitude': signal_amplitude,
        'snr': snr,
        'noise_quality': quality,
        'recommended_kmax': recommended_kmax,
    }


def compare_scan_reproducibility(k, chi_scans, k_weight=2):
    """
    Compare multiple EXAFS scans to assess reproducibility and identify
    noise-dominated k-regions.

    Parameters
    ----------
    k : np.ndarray
        Common k-grid (Å⁻¹).
    chi_scans : np.ndarray
        Array of shape (n_scans, n_k) with chi(k) from each scan.
    k_weight : int
        k-weighting exponent.

    Returns
    -------
    dict with keys:
        'mean_chi' : np.ndarray — mean chi(k) across scans
        'std_chi' : np.ndarray — standard deviation at each k point
        'snr_vs_k' : np.ndarray — point-wise SNR estimate
        'recommended_kmax' : float — k value where SNR drops below 2
    """
    chi_kw = chi_scans * k[np.newaxis, :] ** k_weight
    mean_chi = np.mean(chi_kw, axis=0)
    std_chi = np.std(chi_kw, axis=0)

    # Point-wise SNR (avoid division by zero)
    snr_vs_k = np.where(std_chi > 0, np.abs(mean_chi) / std_chi, np.inf)

    # Smooth the SNR curve to find recommended kmax
    from scipy.ndimage import uniform_filter1d
    snr_smooth = uniform_filter1d(snr_vs_k, size=10)
    below_threshold = np.where((snr_smooth < 2.0) & (k > 4.0))[0]
    recommended_kmax = k[below_threshold[0]] if len(below_threshold) > 0 else k[-1]

    return {
        'mean_chi': mean_chi / k ** k_weight,
        'std_chi': std_chi,
        'snr_vs_k': snr_vs_k,
        'recommended_kmax': recommended_kmax,
    }
```

## Solutions and Mitigation

### Prevention (Before Data Collection)

- Plan sufficient counting time per scan and number of repeat scans based on required k-range and sample concentration.
- Use variable counting time grids: short dwell in XANES region, progressively longer dwell at high k (e.g., k^2-weighted time grid).
- Optimize sample thickness for transmission (mu*x ~ 1 above edge) or concentration for fluorescence.
- Choose the detector mode (transmission vs. fluorescence) appropriate for the sample concentration.
- Maximize photon flux by optimizing beamline optics (mirror focusing, slit sizes).

### Correction — Traditional Methods

```python
import numpy as np
from scipy.signal import savgol_filter


def merge_exafs_scans(k, chi_scans, weights=None, sigma_clip=3.0):
    """
    Merge multiple EXAFS scans with optional sigma-clipping to reject
    outlier points before averaging.

    Parameters
    ----------
    k : np.ndarray
        Common k-grid (Å⁻¹).
    chi_scans : np.ndarray
        Shape (n_scans, n_k) — chi(k) from each scan on the same grid.
    weights : np.ndarray or None
        Shape (n_scans,) — per-scan weights (e.g., from edge step or I0).
        If None, equal weights are used.
    sigma_clip : float
        Number of standard deviations for outlier rejection at each k point.

    Returns
    -------
    dict with keys:
        'k' : np.ndarray — wavenumber grid
        'chi_merged' : np.ndarray — merged chi(k)
        'chi_std' : np.ndarray — standard deviation at each k
        'n_used' : np.ndarray — number of scans used at each k (after clipping)
    """
    n_scans, n_k = chi_scans.shape
    if weights is None:
        weights = np.ones(n_scans)
    weights = weights / np.sum(weights)

    chi_merged = np.zeros(n_k)
    chi_std = np.zeros(n_k)
    n_used = np.zeros(n_k, dtype=int)

    for j in range(n_k):
        values = chi_scans[:, j]
        med = np.median(values)
        mad = np.median(np.abs(values - med))
        robust_std = 1.4826 * mad if mad > 0 else np.std(values)

        # Sigma clipping
        mask = np.abs(values - med) <= sigma_clip * robust_std
        if np.sum(mask) < 2:
            mask = np.ones(n_scans, dtype=bool)  # fallback: use all

        w = weights[mask] / np.sum(weights[mask])
        chi_merged[j] = np.sum(w * values[mask])
        chi_std[j] = np.sqrt(np.sum(w * (values[mask] - chi_merged[j]) ** 2))
        n_used[j] = np.sum(mask)

    return {
        'k': k,
        'chi_merged': chi_merged,
        'chi_std': chi_std,
        'n_used': n_used,
    }


def optimal_krange_selection(k, chi, k_weight=3, snr_threshold=2.0):
    """
    Determine the optimal k-range for Fourier transform based on local SNR.

    Uses the local variability of k^n*chi(k) to estimate where noise
    overwhelms signal.

    Parameters
    ----------
    k : np.ndarray
        Wavenumber array (Å⁻¹).
    chi : np.ndarray
        EXAFS chi(k) signal.
    k_weight : int
        k-weighting exponent.
    snr_threshold : float
        Minimum acceptable local SNR.

    Returns
    -------
    tuple of (k_min, k_max) — recommended Fourier transform window
    """
    chi_kw = chi * k ** k_weight

    # Sliding window analysis
    dk = k[1] - k[0]
    win = max(int(2.0 / dk), 11)  # ~2 Å⁻¹ window

    k_max_rec = k[-1]
    for i in range(win, len(k) - win):
        segment = chi_kw[i - win // 2:i + win // 2]
        smooth = savgol_filter(segment, min(win, len(segment)), 3)
        residual_rms = np.std(segment - smooth)
        signal_amp = np.max(np.abs(smooth))
        if signal_amp > 0 and signal_amp / residual_rms < snr_threshold:
            k_max_rec = k[i]
            break

    # k_min is usually 2-3 Å⁻¹ to avoid XANES region
    k_min_rec = max(2.0, k[0])

    return k_min_rec, k_max_rec
```

### Correction — AI/ML Methods

Denoising autoencoders and convolutional neural networks have been applied to EXAFS noise reduction. A 1D U-Net or variational autoencoder is trained on pairs of noisy (single-scan) and clean (many-scan merged) chi(k) data. The network learns the characteristic oscillation patterns of EXAFS and can separate them from random noise. Wavelet-based neural networks that operate in a multi-resolution representation of chi(k) have shown particular promise for preserving structural information at high k while suppressing noise. Transfer learning from simulated FEFF-generated EXAFS paths can augment limited experimental training data.

## Impact If Uncorrected

Statistical noise in EXAFS directly degrades the reliability of structural parameters extracted from fits. Coordination numbers become unreliable due to reduced apparent amplitudes, bond lengths gain uncertainty from phase noise, and Debye-Waller factors absorb the noise as artificially increased disorder. In severe cases, weak scattering paths (e.g., multiple scattering or distant shells) become undetectable. Fourier transform peaks broaden and diminish, leading to incorrect structural models. Systematic underestimation of coordination numbers by 10-30% is typical when fitting noisy data with an excessive k-range.

## Related Resources

- [Spectroscopy EDA notebook](../../06_data_structures/eda/spectroscopy_eda.md) — data quality checks for XAS spectra
- Related artifact: [Outlier Spectra](outlier_spectra.md) — individual bad scans increase noise when merged
- Related artifact: [Energy Calibration Drift](energy_calibration_drift.md) — misaligned scans broaden merged features

## Real-World Before/After Examples

The following published sources provide real experimental before/after comparisons:

| Source | Type | Figure | Description | License |
|--------|------|--------|-------------|---------|
| [Kelly et al. 2018](https://doi.org/10.1107/S1600577518006021) | Paper | Figs 2--4 | Improving the quality of XAFS data — real spectra before/after noise reduction techniques | -- |
| [Sources of Noise in EXAFS Experiments (IIT training)](https://gbxafs.iit.edu/training/Noise.pdf) | Training slides | Multiple | Comprehensive overview of noise sources in EXAFS with experimental examples | -- |

> **Recommended reference**: [Kelly et al. 2018 — Improving the quality of XAFS data (J. Synchrotron Rad.)](https://doi.org/10.1107/S1600577518006021)

## Key Takeaway

Statistical noise is the most fundamental limitation of EXAFS data quality. Always merge multiple scans, use k-dependent counting times during acquisition, and choose the Fourier transform k-range based on the actual SNR of your data rather than a fixed convention. A well-chosen k_max that excludes noise-dominated data yields more reliable structural parameters than extending to high k with poor statistics.
