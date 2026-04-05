# Probe Blurring

## Classification

| Attribute | Value |
|-----------|-------|
| **Modality** | XRF Microscopy |
| **Noise Type** | Instrumental |
| **Severity** | Minor |
| **Frequency** | Always |
| **Detection Difficulty** | Hard |

## Visual Examples

```
 True sample structure            Measured XRF map              Deconvolved map
 (sub-beam resolution)            (convolved with probe)        (Richardson-Lucy)
 ┌────────────────────┐           ┌────────────────────┐        ┌────────────────────┐
 │                    │           │                    │        │                    │
 │    ██              │           │   ░▒▓▓▒░           │        │    ▓█▓             │
 │    ██              │           │   ▒▓██▓▒           │        │    ▓██             │
 │                    │           │   ░▒▓▓▒░           │        │                    │
 │          ████████  │           │       ░▒▓████▓▒░   │        │         ▒▓█████▓   │
 │          ████████  │           │       ░▒▓████▓▒░   │        │         ▒▓█████▓   │
 │                    │           │                    │        │                    │
 │ ██████████████████ │           │ ░▒▓████████████▓▒░ │        │ ▒▓██████████████▓  │
 │                    │           │ ░▒▓▒░░░░░░░░░▒▓▒░ │        │                    │
 └────────────────────┘           └────────────────────┘        └────────────────────┘
  Sharp boundaries                 Blurred edges, features       Partially recovered
  2 nm features                    appear ~200 nm wider          sharpness

 Probe point-spread function (PSF):
              ╱╲
            ╱    ╲
          ╱   ██   ╲        Gaussian-like profile
        ╱   ██████   ╲      FWHM = beam spot size
      ╱  ████████████  ╲    (typically 50 nm – 10 μm)
 ───╱████████████████████╲───
```

## Description

Probe blurring is the spatial resolution limit imposed by the finite size of the focused X-ray beam. Every pixel in an XRF map records fluorescence from a region defined by the beam footprint, not a mathematical point. The measured map is the true elemental distribution convolved with the beam's point-spread function (PSF), causing features smaller than the beam to appear enlarged and smeared, sharp boundaries to become gradual gradients, and closely spaced features to merge together. This is an intrinsic property of every scanning microscopy measurement and sets the fundamental spatial resolution.

## Root Cause

The X-ray beam at a synchrotron microprobe is focused by optical elements (Kirkpatrick-Baez mirrors, zone plates, or compound refractive lenses) to a spot of finite size, typically 50 nm to 10 um FWHM depending on the beamline and optics. The beam profile is usually approximately Gaussian but may have extended tails from imperfect optics, vibrations, or diffraction effects. At each scan position, the measured fluorescence is the integral of the elemental distribution weighted by the beam intensity profile. Mathematically, the measured map M(x,y) = PSF(x,y) * C(x,y), where * denotes 2D convolution and C(x,y) is the true concentration. Scanning with a step size smaller than the beam FWHM oversamples the blurred image but does not recover the lost resolution.

## Quick Diagnosis

```python
import numpy as np
from scipy import ndimage

# Find sharpest edge in the map and measure its width
# element_map: 2D XRF map
gradient = np.hypot(*np.gradient(element_map.astype(float)))
edge_profile = element_map[np.unravel_index(np.argmax(gradient), gradient.shape)[0], :]
fwhm_pixels = np.sum(edge_profile > 0.5 * np.max(edge_profile))
print(f"Sharpest edge FWHM: ~{fwhm_pixels} pixels")
print(f"If step size = 0.5 μm → beam FWHM ≈ {fwhm_pixels * 0.5:.1f} μm")
```

## Detection Methods

### Visual Indicators

- Features known to have sharp boundaries (e.g., cell walls, mineral grain boundaries) appear as gradual gradients spanning several pixels.
- Small features (inclusions, nanoparticles) appear as diffuse blobs larger than expected from other microscopy techniques (SEM, optical).
- The map looks "soft" or "out of focus" compared to simultaneously acquired optical or SEM images.
- Line profiles across known sharp edges show a Gaussian-like transition rather than a step function.

### Automated Detection

```python
import numpy as np
from scipy.optimize import curve_fit
from scipy import ndimage


def measure_psf_from_edge(element_map, pixel_size_um,
                           edge_direction='horizontal'):
    """
    Estimate the beam PSF width from a sharp edge in the elemental map
    by fitting an error function (integrated Gaussian) to edge profiles.

    Parameters
    ----------
    element_map : np.ndarray
        2D elemental map containing at least one sharp compositional boundary.
    pixel_size_um : float
        Pixel size in micrometers.
    edge_direction : str
        'horizontal' (edge runs left-right, profile along columns)
        or 'vertical' (edge runs top-bottom, profile along rows).

    Returns
    -------
    dict with keys:
        'psf_fwhm_um' : float — estimated beam FWHM in micrometers
        'psf_sigma_um' : float — Gaussian sigma in micrometers
        'edge_profiles' : list — extracted edge profiles
        'fit_quality' : float — R-squared of the error function fit
    """
    from scipy.special import erf

    img = element_map.astype(float)

    # Find strong edges using gradient magnitude
    if edge_direction == 'horizontal':
        gradient = np.abs(np.gradient(img, axis=0))
        # Average columns to get robust edge profile
        edge_strength = np.mean(gradient, axis=1)
        edge_row = np.argmax(edge_strength)
        # Extract profiles perpendicular to edge (along columns)
        margin = min(20, img.shape[0] // 4)
        r_start = max(0, edge_row - margin)
        r_end = min(img.shape[0], edge_row + margin)
        profiles = [img[r_start:r_end, c] for c in range(img.shape[1])]
    else:
        gradient = np.abs(np.gradient(img, axis=1))
        edge_strength = np.mean(gradient, axis=0)
        edge_col = np.argmax(edge_strength)
        margin = min(20, img.shape[1] // 4)
        c_start = max(0, edge_col - margin)
        c_end = min(img.shape[1], edge_col + margin)
        profiles = [img[r, c_start:c_end] for r in range(img.shape[0])]

    # Average multiple profiles for noise reduction
    avg_profile = np.mean(profiles, axis=0)
    x = np.arange(len(avg_profile)) * pixel_size_um

    # Fit error function: erf is the integral of Gaussian
    def erf_model(x, amplitude, center, sigma, offset):
        return offset + amplitude * 0.5 * (1 + erf((x - center) / (sigma * np.sqrt(2))))

    try:
        p0 = [np.ptp(avg_profile), x[len(x)//2],
              2 * pixel_size_um, np.min(avg_profile)]
        popt, pcov = curve_fit(erf_model, x, avg_profile, p0=p0)
        sigma_um = abs(popt[2])
        fwhm_um = sigma_um * 2.3548

        # R-squared
        residuals = avg_profile - erf_model(x, *popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((avg_profile - np.mean(avg_profile)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    except RuntimeError:
        sigma_um = float('nan')
        fwhm_um = float('nan')
        r_squared = 0.0

    return {
        'psf_fwhm_um': float(fwhm_um),
        'psf_sigma_um': float(sigma_um),
        'edge_profiles': profiles,
        'fit_quality': float(r_squared),
    }
```

## Solutions and Mitigation

### Prevention (Before Data Collection)

- Use the smallest available beam size (nano-probe beamlines offer 30–50 nm resolution).
- Ensure focusing optics are properly aligned and aberration-free before scanning.
- Match the scan step size to the beam size (oversampling by 2-3x enables deconvolution).
- Minimize vibrations (table, cryojet, sample stage) that effectively enlarge the beam.

### Correction — Traditional Methods

```python
import numpy as np
from scipy.signal import fftconvolve


def richardson_lucy_deconvolution(image, psf, n_iterations=50,
                                    clip_negative=True):
    """
    Apply Richardson-Lucy deconvolution to sharpen an XRF elemental map.

    Parameters
    ----------
    image : np.ndarray
        2D blurred elemental map (non-negative counts).
    psf : np.ndarray
        2D point-spread function of the beam. Must be normalized
        to sum to 1.0 and same size or smaller than image.
    n_iterations : int
        Number of RL iterations. More iterations = sharper result
        but also amplifies noise. Typical range: 10–100.
    clip_negative : bool
        Clip negative values to zero after each iteration.

    Returns
    -------
    np.ndarray — deconvolved elemental map
    """
    # Normalize PSF
    psf = psf / np.sum(psf)
    psf_mirror = psf[::-1, ::-1]

    # Initialize with the observed image
    estimate = image.astype(float).copy()
    estimate[estimate <= 0] = 1e-10  # avoid division by zero

    for i in range(n_iterations):
        # Forward model: what would the detector see given current estimate?
        blurred_estimate = fftconvolve(estimate, psf, mode='same')
        blurred_estimate[blurred_estimate <= 0] = 1e-10

        # Ratio of observed to predicted
        ratio = image.astype(float) / blurred_estimate

        # Correction factor
        correction = fftconvolve(ratio, psf_mirror, mode='same')

        # Update estimate
        estimate *= correction

        if clip_negative:
            estimate = np.maximum(estimate, 0)

    return estimate


def make_gaussian_psf(fwhm_pixels, size=None):
    """
    Create a 2D Gaussian PSF kernel for deconvolution.

    Parameters
    ----------
    fwhm_pixels : float
        Beam FWHM in pixel units.
    size : int or None
        Kernel size (must be odd). If None, auto-set to 4*FWHM+1.

    Returns
    -------
    np.ndarray — normalized 2D Gaussian PSF
    """
    sigma = fwhm_pixels / 2.3548

    if size is None:
        size = int(4 * fwhm_pixels) + 1
        if size % 2 == 0:
            size += 1

    center = size // 2
    y, x = np.mgrid[:size, :size]
    psf = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
    psf /= np.sum(psf)
    return psf
```

### Correction — AI/ML Methods

Deep learning super-resolution methods trained on paired low-resolution/high-resolution XRF data can recover sub-beam features that Richardson-Lucy deconvolution cannot. Networks such as deep residual architectures learn the mapping from blurred, noisy XRF maps to sharp, denoised versions using training data from high-resolution nano-probe measurements as ground truth. These approaches simultaneously denoise and deblur, avoiding the noise amplification inherent in classical deconvolution. Transfer learning from natural-image super-resolution networks has shown promising results even with limited synchrotron training data.

## Impact If Uncorrected

Probe blurring limits the effective spatial resolution of XRF maps to the beam size, regardless of scan step size. Sub-beam features are smeared out and cannot be resolved, sharp compositional boundaries appear as gradual gradients, and quantitative analysis of small features (e.g., nanoparticle inclusions, sub-cellular organelles) yields diluted concentrations because the beam averages over the feature and its surroundings. Without deconvolution, the spatial resolution claimed from the scan step size overstates the true information content of the data.

## Related Resources

- [Deep Residual XRF Denoising](../../03_ai_ml_methods/denoising/deep_residual_xrf.md) — neural network approaches to simultaneous denoising and deblurring
- Related artifact: [Photon Counting Noise](photon_counting_noise.md) — deconvolution amplifies noise; denoise first
- Related artifact: [Scan Stripe](scan_stripe.md) — destriping should precede deconvolution

## Real-World Before/After Examples

The following published sources provide real experimental before/after comparisons:

| Source | Type | Figure | Description | License |
|--------|------|--------|-------------|---------|
| [Wu et al. 2023](https://doi.org/10.1038/s41524-023-00995-9) | Paper | Fig 3 | Resolution-Enhanced XRF Microscopy via Deep Residual Networks — real before/after super-resolution on synchrotron XRF data | CC BY 4.0 |

> **Recommended reference**: [Wu et al. 2023 — Resolution-Enhanced XRF Microscopy (npj Computational Materials)](https://doi.org/10.1038/s41524-023-00995-9)

## Key Takeaway

Probe blurring is an unavoidable consequence of the finite beam size and sets the true spatial resolution of XRF microscopy. For quantitative analysis of small features, either use a beamline with a smaller probe or apply Richardson-Lucy deconvolution with a measured PSF — but be aware that deconvolution amplifies noise and requires careful regularization.
