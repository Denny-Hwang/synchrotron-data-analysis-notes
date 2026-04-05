# Peak Overlap (Spectral Interference)

## Classification

| Attribute | Value |
|-----------|-------|
| **Modality** | XRF Microscopy |
| **Noise Type** | Systematic |
| **Severity** | Major |
| **Frequency** | Common |
| **Detection Difficulty** | Moderate |

## Visual Examples

```
 Fe Kα map (6.40 keV)         Co Kα map (6.93 keV)         Scatter plot
 ┌─────────────────────┐      ┌─────────────────────┐      Fe vs Co (raw)
 │ ░░░░░▓▓▓▓▓▓░░░░░░░░ │      │ ░░░░░▒▒▒▒▒▒░░░░░░░░ │        ╱
 │ ░░░▓▓██████▓▓░░░░░░ │      │ ░░░▒▒▓▓▓▓▓▓▒▒░░░░░░ │       ╱  ← spurious
 │ ░░▓▓████████▓▓░░░░░ │      │ ░░▒▒▓▓▓▓▓▓▓▓▒▒░░░░░ │      ╱   correlation
 │ ░░░▓▓██████▓▓░░░░░░ │      │ ░░░▒▒▓▓▓▓▓▓▒▒░░░░░░ │     ╱    (Fe Kβ leaks
 │ ░░░░░▓▓▓▓▓▓░░░░░░░░ │      │ ░░░░░▒▒▒▒▒▒░░░░░░░░ │    ╱     into Co ROI)
 └─────────────────────┘      └─────────────────────┘   ──────────
 Real Fe distribution          Ghost of Fe in Co map     Fe counts →

 After spectral deconvolution:
 Co Kα map (corrected)
 ┌─────────────────────┐
 │ ░░░░░░░░░░░░░░░░░░░ │  ← Co was never there;
 │ ░░░░░░░░░░░░░░░░░░░ │     the signal was entirely
 │ ░░░░░░░░░░░░░░░░░░░ │     Fe Kβ (7.06 keV) leaking
 │ ░░░░░░░░░░░░░░░░░░░ │     into the Co Kα window
 │ ░░░░░░░░░░░░░░░░░░░ │
 └─────────────────────┘
```

## Description

Peak overlap (spectral interference) occurs when the X-ray fluorescence emission lines of two or more elements are close enough in energy that they cannot be fully resolved by the detector's energy resolution. The result is that intensity from one element's emission line "leaks" into the region-of-interest (ROI) window assigned to another element, creating phantom signals or inflated concentrations. This is one of the most insidious artifacts in XRF microscopy because it produces elemental maps that appear physically plausible — two elements apparently co-localized — when in reality only one is present.

## Root Cause

Energy-dispersive silicon drift detectors (SDDs) used at synchrotron XRF beamlines typically have an energy resolution of 120–150 eV FWHM. Many XRF emission lines are separated by less than this resolution. Common problematic overlaps include Fe Kbeta (7.06 keV) with Co Kalpha (6.93 keV), S Kalpha (2.31 keV) with Mo Lalpha (2.29 keV), Ti Kbeta (4.93 keV) with V Kalpha (4.95 keV), and As Kalpha (10.54 keV) with Pb Lalpha (10.55 keV). Simple ROI-based integration cannot separate these overlapping peaks, and any intensity in the overlap region is attributed to whichever element's window it falls in.

## Quick Diagnosis

```python
import numpy as np

# Check for suspicious correlation between elements
# fe_map, co_map: 2D elemental maps from ROI integration
correlation = np.corrcoef(fe_map.ravel(), co_map.ravel())[0, 1]
ratio = np.mean(co_map[fe_map > np.percentile(fe_map, 90)]) / np.mean(fe_map[fe_map > np.percentile(fe_map, 90)])
print(f"Fe-Co correlation: {correlation:.3f}")
print(f"Co/Fe ratio in Fe-rich regions: {ratio:.4f}")
print(f"Expected Fe Kβ/Kα ratio: ~0.13")
print(f"Overlap likely: {correlation > 0.9 and abs(ratio - 0.13) < 0.05}")
```

## Detection Methods

### Visual Indicators

- Two elements that should be geochemically independent show nearly identical spatial distributions.
- The "ghost" element map looks like a faint, blurred copy of the dominant element.
- The intensity ratio between the two maps is roughly constant across the entire image (characteristic of a fixed spectral crosstalk fraction).
- Inspecting a single-pixel spectrum from a region of interest shows overlapping peaks that the ROI windows cannot separate.

### Automated Detection

```python
import numpy as np
from scipy import stats


def detect_peak_overlap(map_a, map_b, element_a='Fe', element_b='Co',
                        known_ratios=None):
    """
    Detect potential peak overlap between two elemental maps by
    checking for suspiciously high correlation and constant intensity ratio.

    Parameters
    ----------
    map_a, map_b : np.ndarray
        2D elemental maps (e.g., from ROI integration).
    element_a, element_b : str
        Element labels for reporting.
    known_ratios : dict or None
        Known Kbeta/Kalpha branching ratios for common overlaps.
        Default includes Fe-Co and Ti-V.

    Returns
    -------
    dict with keys:
        'correlation' : float — Pearson correlation between maps
        'mean_ratio' : float — mean(map_b) / mean(map_a) in high-signal region
        'ratio_std' : float — standard deviation of pixel-wise ratio
        'overlap_likely' : bool — True if overlap is probable
        'estimated_crosstalk' : float — estimated fraction of map_b from map_a
    """
    if known_ratios is None:
        # Kbeta/Kalpha branching ratios for common overlaps
        known_ratios = {
            ('Fe', 'Co'): 0.13,   # Fe Kβ / Fe Kα
            ('Ti', 'V'): 0.12,    # Ti Kβ / Ti Kα
            ('S', 'Mo'): None,    # S Kα / Mo Lα — no fixed ratio
            ('As', 'Pb'): None,   # As Kα / Pb Lα — no fixed ratio
        }

    # Use high-signal pixels to avoid noise-dominated regions
    threshold_a = np.percentile(map_a[map_a > 0], 75)
    mask = map_a > threshold_a

    if np.sum(mask) < 10:
        return {'overlap_likely': False, 'correlation': 0.0}

    a_vals = map_a[mask].astype(float)
    b_vals = map_b[mask].astype(float)

    correlation = np.corrcoef(a_vals, b_vals)[0, 1]

    # Pixel-wise ratio in high-signal region
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = b_vals / a_vals
        ratio = ratio[np.isfinite(ratio)]

    mean_ratio = np.mean(ratio)
    ratio_std = np.std(ratio)

    # Check against known branching ratios
    pair = (element_a, element_b)
    expected_ratio = known_ratios.get(pair)
    if expected_ratio is not None:
        ratio_match = abs(mean_ratio - expected_ratio) < 0.05
    else:
        ratio_match = False

    overlap_likely = (correlation > 0.85) and (ratio_std / mean_ratio < 0.3)

    return {
        'correlation': float(correlation),
        'mean_ratio': float(mean_ratio),
        'ratio_std': float(ratio_std),
        'overlap_likely': overlap_likely,
        'ratio_matches_branching': ratio_match,
        'estimated_crosstalk': float(mean_ratio) if overlap_likely else 0.0,
    }
```

## Solutions and Mitigation

### Prevention (Before Data Collection)

- Use the highest energy resolution detector available (e.g., newer SDDs with 125 eV FWHM at Mn Kalpha).
- When studying specific overlapping pairs (e.g., Fe+Co), consider wavelength-dispersive spectroscopy (WDS) for those elements if available.
- Choose excitation energy carefully: tuning below an interfering element's absorption edge eliminates its fluorescence entirely.
- Collect full spectra at each pixel rather than hardware ROIs, enabling offline spectral fitting.

### Correction — Traditional Methods

```python
import numpy as np
from scipy.optimize import curve_fit


def subtract_crosstalk(map_dominant, map_ghost, crosstalk_fraction):
    """
    Remove spectral crosstalk from a ghost element map by subtracting
    the known fraction contributed by the dominant element.

    Parameters
    ----------
    map_dominant : np.ndarray
        2D map of the dominant (interfering) element.
    map_ghost : np.ndarray
        2D map of the element contaminated by overlap.
    crosstalk_fraction : float
        Fraction of dominant element signal leaking into ghost ROI
        (e.g., 0.13 for Fe Kβ into Co Kα window).

    Returns
    -------
    np.ndarray — corrected map for the ghost element
    """
    corrected = map_ghost.astype(float) - crosstalk_fraction * map_dominant.astype(float)
    corrected = np.maximum(corrected, 0)  # no negative counts
    return corrected


def fit_spectrum_multipeak(energy, spectrum, peak_energies, peak_labels,
                           fwhm=0.15):
    """
    Fit a multi-Gaussian model to a single-pixel XRF spectrum to
    deconvolve overlapping peaks.

    Parameters
    ----------
    energy : np.ndarray
        1D energy axis in keV.
    spectrum : np.ndarray
        1D measured spectrum (counts).
    peak_energies : list of float
        Known emission line energies in keV.
    peak_labels : list of str
        Labels for each peak.
    fwhm : float
        Detector energy resolution in keV.

    Returns
    -------
    dict — {label: fitted_area} for each peak
    """
    sigma = fwhm / 2.3548  # FWHM to Gaussian sigma

    def multi_gaussian(x, *amplitudes):
        result = np.zeros_like(x, dtype=float)
        for amp, center in zip(amplitudes, peak_energies):
            result += amp * np.exp(-0.5 * ((x - center) / sigma) ** 2)
        return result

    n_peaks = len(peak_energies)
    p0 = [np.max(spectrum) / n_peaks] * n_peaks
    bounds = ([0] * n_peaks, [np.inf] * n_peaks)

    try:
        popt, pcov = curve_fit(multi_gaussian, energy, spectrum,
                               p0=p0, bounds=bounds)
        perr = np.sqrt(np.diag(pcov))
    except RuntimeError:
        popt = np.zeros(n_peaks)
        perr = np.full(n_peaks, np.inf)

    # Area under each Gaussian = amplitude * sigma * sqrt(2*pi)
    areas = popt * sigma * np.sqrt(2 * np.pi)

    results = {}
    for label, area, err in zip(peak_labels, areas, perr):
        results[label] = {
            'area': float(area),
            'amplitude': float(popt[peak_labels.index(label)]),
            'uncertainty': float(err * sigma * np.sqrt(2 * np.pi)),
        }
    return results


def deconvolve_map_pixelwise(spectral_cube, energy_axis, peak_energies,
                              peak_labels, fwhm=0.15):
    """
    Apply per-pixel spectral fitting to a full XRF spectral image cube.

    Parameters
    ----------
    spectral_cube : np.ndarray
        3D array of shape (ny, nx, n_channels).
    energy_axis : np.ndarray
        1D energy axis in keV.
    peak_energies : list of float
        Emission line energies to fit.
    peak_labels : list of str
        Element labels for each peak.
    fwhm : float
        Detector resolution in keV.

    Returns
    -------
    dict of np.ndarray — {label: 2D_map} for each fitted element
    """
    ny, nx, _ = spectral_cube.shape
    element_maps = {label: np.zeros((ny, nx)) for label in peak_labels}

    for iy in range(ny):
        for ix in range(nx):
            spectrum = spectral_cube[iy, ix, :]
            if np.sum(spectrum) < 10:
                continue
            result = fit_spectrum_multipeak(
                energy_axis, spectrum, peak_energies, peak_labels, fwhm
            )
            for label in peak_labels:
                element_maps[label][iy, ix] = result[label]['area']

    return element_maps
```

### Correction — AI/ML Methods

Deep learning approaches to spectral deconvolution train a neural network on simulated XRF spectra with known ground-truth peak areas, teaching the model to separate overlapping peaks even at low count rates where traditional fitting fails. These models can process entire spectral image cubes orders of magnitude faster than pixel-by-pixel curve fitting. Software packages like PyXRF (Brookhaven National Lab) incorporate optimized fitting routines, and MAPS (Argonne National Lab) uses non-negative matrix factorization combined with spectral standards for robust deconvolution.

## Impact If Uncorrected

Peak overlap leads to false identification of elements that are not present in the sample, incorrect spatial distributions, and inflated concentration estimates. In environmental science, misidentifying Fe Kbeta leakage as cobalt contamination could trigger unnecessary remediation. In cultural heritage studies, apparent titanium from Ti Kbeta overlap with vanadium could misattribute pigment composition. Elemental correlation analysis becomes meaningless when two maps are correlated only through spectral crosstalk rather than true co-localization.

## Related Resources

- [XRF Analysis Pipeline](../../02_xray_modalities/xrf_microscopy/analysis_pipeline.md) — spectral fitting and deconvolution workflows
- Related artifact: [Photon Counting Noise](photon_counting_noise.md) — low counts make peak overlap harder to resolve
- Related artifact: [Dead-Time Saturation](dead_time_saturation.md) — distorted spectra at high count rates exacerbate overlap

## Real-World Before/After Examples

The following published sources provide real experimental before/after comparisons:

| Source | Type | Figure/Location | Description | License |
|--------|------|-----------------|-------------|---------|
| [Sole et al. 2007](https://doi.org/10.1016/j.sab.2007.09.002) | Paper | Multiple | PyMca: A Multiplatform Code for the Analysis of Energy-Dispersive X-Ray Fluorescence Spectra — spectral deconvolution examples showing peak separation | -- |
| [PyMca (GitHub)](https://github.com/vasole/pymca) | Repository | Examples | PyMca X-ray fluorescence analysis toolkit with spectral deconvolution demonstrations | MIT |

**Key references with published before/after comparisons:**
- **Sole et al. (2007)**: PyMca spectral deconvolution examples showing overlapping peaks separated via fitting. DOI: 10.1016/j.sab.2007.09.002

## Key Takeaway

Never trust ROI-based elemental maps for elements with known spectral overlaps. Always perform proper spectral deconvolution using multi-peak Gaussian fitting or dedicated software (PyXRF, MAPS) before reporting co-localization or quantitative concentrations for overlapping element pairs.
