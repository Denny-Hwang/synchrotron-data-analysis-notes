# Cross-Domain Noise Estimation & Characterization Methods

## Classification

| Attribute | Value |
|-----------|-------|
| **Modality** | Cross-cutting (All modalities) |
| **Noise Type** | Methodology Reference |
| **Severity** | N/A (reference document) |
| **Frequency** | N/A |
| **Detection Difficulty** | N/A |
| **Origin Domain** | Medical Imaging, Astronomy, Electron Microscopy |

## Description

This document catalogs noise estimation and characterization methodologies from multiple imaging domains that are applicable to synchrotron data. These cross-domain methods provide standardized frameworks for quantifying noise, comparing data quality, and validating denoising algorithms.

## Noise Characterization Methods

### 1. Noise Power Spectrum (NPS) / Wiener Spectrum

**Origin:** Medical imaging (CT quality assurance) and astronomy
**Purpose:** Frequency-dependent noise characterization

```python
import numpy as np

def compute_nps(image, roi_size=128, n_rois=20):
    """Compute Noise Power Spectrum from uniform region ROIs."""
    ny, nx = image.shape
    nps_sum = np.zeros((roi_size, roi_size))
    for _ in range(n_rois):
        y0 = np.random.randint(0, ny - roi_size)
        x0 = np.random.randint(0, nx - roi_size)
        roi = image[y0:y0+roi_size, x0:x0+roi_size]
        # Remove mean (DC component)
        roi = roi - roi.mean()
        # 2D DFT
        F = np.fft.fftshift(np.fft.fft2(roi))
        nps_sum += np.abs(F)**2
    nps = nps_sum / n_rois * (1.0 / roi_size**2)
    return nps

def radial_nps(nps_2d, pixel_size=1.0):
    """Compute 1D radial NPS from 2D NPS."""
    ny, nx = nps_2d.shape
    cy, cx = ny // 2, nx // 2
    Y, X = np.ogrid[-cy:ny-cy, -cx:nx-cx]
    r = np.sqrt(X**2 + Y**2).astype(int)
    r_max = min(cy, cx)
    radial = np.array([nps_2d[r == ri].mean() for ri in range(1, r_max)])
    freq = np.arange(1, r_max) / (2 * r_max * pixel_size)
    return freq, radial
```

**Synchrotron applications:**
- CT image quality assessment
- Detector characterization
- Comparing reconstruction algorithms
- Denoising validation

### 2. Detective Quantum Efficiency (DQE)

**Origin:** Medical imaging and detector physics
**Purpose:** Quantify detector's signal transfer efficiency vs frequency

```python
import numpy as np

def compute_dqe(mtf, nps, mean_signal, incident_quanta):
    """Compute DQE from MTF and NPS."""
    # DQE(f) = MTF(f)² × mean_signal² / (NPS(f) × incident_quanta)
    dqe = mtf**2 * mean_signal**2 / (nps * incident_quanta + 1e-10)
    return np.clip(dqe, 0, 1)
```

**Key metric:** DQE(0) = output SNR² / input SNR² — overall efficiency

### 3. Mean-Variance Analysis (Poisson-Gaussian Noise Model)

**Origin:** Astronomy (CCD characterization) and fluorescence microscopy
**Purpose:** Separate Poisson (signal-dependent) and Gaussian (readout) noise components

```python
import numpy as np

def mean_variance_analysis(image_pairs):
    """Estimate gain and readnoise from mean-variance relationship.

    Args:
        image_pairs: list of (flat1, flat2) image pairs at different exposure levels
    """
    means, variances = [], []
    for flat1, flat2 in image_pairs:
        mean_val = (flat1.mean() + flat2.mean()) / 2
        # Variance of difference / 2 = temporal noise variance
        diff = flat1.astype(float) - flat2.astype(float)
        var_val = diff.var() / 2
        means.append(mean_val)
        variances.append(var_val)
    means, variances = np.array(means), np.array(variances)
    # Linear fit: Var = (1/gain) * Mean + readnoise²
    slope, intercept = np.polyfit(means, variances, 1)
    gain = 1.0 / slope  # e⁻/ADU
    readnoise = np.sqrt(abs(intercept))  # ADU
    print(f"Gain: {gain:.2f} e⁻/ADU")
    print(f"Read noise: {readnoise:.2f} ADU ({readnoise*gain:.2f} e⁻)")
    return gain, readnoise
```

### 4. MAD Estimator (Robust Noise Estimation from Single Image)

**Origin:** Signal processing / wavelet analysis
**Purpose:** Estimate noise standard deviation without clean reference

```python
import numpy as np

def mad_noise_estimate(image):
    """Estimate noise sigma using Median Absolute Deviation of wavelet coefficients."""
    from scipy.ndimage import convolve
    # Horizontal detail coefficients (Haar wavelet approximation)
    kernel = np.array([[1, -1]])
    detail = convolve(image.astype(float), kernel)
    # MAD estimator (robust to signal content)
    mad = np.median(np.abs(detail - np.median(detail)))
    sigma = mad / 0.6745  # MAD to sigma conversion for Gaussian
    print(f"Estimated noise sigma: {sigma:.3f}")
    return sigma
```

### 5. Noise2Void / Noise2Self Framework

**Origin:** Computer vision / deep learning
**Purpose:** Self-supervised noise estimation without clean references

```python
# Conceptual framework — actual implementation requires deep learning library
def noise2void_concept(noisy_image):
    """
    Noise2Void (Krull et al., 2019):
    - Train CNN to predict center pixel from surrounding context
    - Blind-spot network: exclude center pixel from receptive field
    - Network learns signal (correlated) but cannot learn noise (independent)
    - Applicable when: noise is pixel-wise independent, no clean data available

    Applicable to synchrotron:
    - Low-dose tomography denoising
    - XRF map denoising
    - Any detector with pixel-independent noise
    """
    pass
```

## Standard Benchmarks & Datasets

### Natural Image Denoising

| Dataset | Description | Use |
|---------|-------------|-----|
| BSD68 | 68 images from Berkeley Segmentation Dataset | Standard PSNR/SSIM benchmark |
| Set12 | 12 classic test images | Quick evaluation |
| CBSD68 | Color version of BSD68 | Color denoising |
| McMaster | 18 high-quality images | Color denoising |

### Domain-Specific

| Dataset | Domain | Description |
|---------|--------|-------------|
| LDCT Grand Challenge (AAPM) | Medical CT | Paired normal/low-dose CT |
| FMD | Fluorescence microscopy | Confocal, two-photon, wide-field |
| Planaria / Tribolium | Fluorescence microscopy | Paired noisy/clean biology images |
| EMPIAR | Cryo-EM | Single-particle benchmark data |
| TomoBank | Synchrotron | APS tomography datasets |
| SciCat | Synchrotron | ESS/PSI data catalog |

### Noise Level Conventions

| Domain | Noise Parameterization |
|--------|----------------------|
| Natural image denoising | Additive Gaussian σ = 15, 25, 50 |
| Medical CT | Dose level (mAs), noise index |
| Astronomy | Gain (e⁻/ADU) + read noise (e⁻) |
| Electron microscopy | Dose (e⁻/Å²) |
| Synchrotron XRF | Dwell time (ms), total counts |
| Synchrotron CT | Exposure time, I₀ counts |

## Quality Metrics

```python
import numpy as np

def compute_psnr(clean, denoised, data_range=None):
    """Peak Signal-to-Noise Ratio."""
    mse = np.mean((clean - denoised)**2)
    if data_range is None:
        data_range = clean.max() - clean.min()
    return 10 * np.log10(data_range**2 / mse)

def compute_ssim(clean, denoised, win_size=7):
    """Structural Similarity Index (simplified)."""
    from scipy.ndimage import uniform_filter
    C1 = (0.01 * (clean.max() - clean.min()))**2
    C2 = (0.03 * (clean.max() - clean.min()))**2
    mu1 = uniform_filter(clean, win_size)
    mu2 = uniform_filter(denoised, win_size)
    sigma1_sq = uniform_filter(clean**2, win_size) - mu1**2
    sigma2_sq = uniform_filter(denoised**2, win_size) - mu2**2
    sigma12 = uniform_filter(clean * denoised, win_size) - mu1 * mu2
    ssim_map = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def compute_cnr(image, roi_signal, roi_background):
    """Contrast-to-Noise Ratio — standard in medical imaging."""
    signal_mean = image[roi_signal].mean()
    bg_mean = image[roi_background].mean()
    bg_std = image[roi_background].std()
    cnr = abs(signal_mean - bg_mean) / (bg_std + 1e-10)
    return cnr
```

## Key References

- **van Dokkum (2001)** — L.A.Cosmic (cosmic ray / outlier detection)
- **Krull et al. (2019)** — "Noise2Void: Learning Denoising from Single Noisy Images"
- **Batson & Royer (2019)** — "Noise2Self: Blind Denoising by Self-Supervision"
- **Zhang et al. (2017)** — "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN (DnCNN)"
- **Lehtinen et al. (2018)** — "Noise2Noise: Learning Image Restoration without Clean Data"
- **Siewerdsen et al. (2002)** — "Framework for NPS of multi-dimensional imaging systems"
- **Cunningham & Shaw (1999)** — "Signal-to-noise optimization of medical imaging systems" (DQE framework)
- **Foi et al. (2008)** — "Practical Poissonian-Gaussian noise modeling for photography and microscopy"

## Related Resources

- [Low-dose noise](../tomography/low_dose_noise.md) — Applies NPS and DQE concepts
- [Photon counting noise](../xrf_microscopy/photon_counting_noise.md) — Poisson-Gaussian model
- [DL hallucination](dl_hallucination.md) — Denoising artifacts from ML methods
- [Detector common issues](detector_common_issues.md) — Detector characterization
