# Self-Absorption (XRF)

## Classification

| Attribute | Value |
|-----------|-------|
| **Modality** | XRF Microscopy |
| **Noise Type** | Systematic |
| **Severity** | Major |
| **Frequency** | Common |
| **Detection Difficulty** | Hard |

## Visual Examples

```
 Cross-section view: X-ray beam enters sample, fluorescence exits

       Incident X-ray beam (E₀)
              │
              ▼
 ┌────────────────────────────────┐  ← Sample surface
 │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │  Thin region: little self-absorption
 │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │  XRF signal ≈ true concentration
 ├────────────────────────────────┤
 │  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ │  Medium: moderate attenuation
 │  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ │  XRF signal < true concentration
 │  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ │
 ├────────────────────────────────┤
 │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │  Thick/dense: severe attenuation
 │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │  Fluorescence from deep layers
 │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │  re-absorbed before escaping
 │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │  ← XRF signal << true concentration
 └────────────────────────────────┘
              │    ╲
              │     ╲  Fluorescence photon path
              │      ╲ (partially absorbed)
              ▼       ╲
         Transmitted    → Detector
         beam             (sees attenuated signal)

 Apparent vs True concentration:

 True conc.  ████████████████████  (uniform)
 Apparent    ████████████░░░░░░░░  (decreases with depth/density)
```

## Description

Self-absorption occurs when fluorescence X-rays emitted by atoms within the sample are partially re-absorbed by the sample matrix before they can reach the detector. This causes the measured fluorescence intensity to underestimate the true elemental concentration, with the effect being most severe for low-energy emission lines (which are more easily absorbed), thick or dense samples, and elements embedded deep within a high-Z matrix. Self-absorption is particularly insidious because it produces maps that look qualitatively correct but are quantitatively wrong — concentrations in thick regions are systematically suppressed.

## Root Cause

The fluorescence photon emitted at depth d within the sample must travel through the remaining sample material to reach the detector. Along this path, the photon has a probability of being absorbed that depends on the mass attenuation coefficient of the matrix at the fluorescence energy, the density and thickness of material traversed, and the exit angle. The measured intensity follows the Beer-Lambert law: I_measured = I_true * exp(-mu * rho * t), where mu is the mass attenuation coefficient, rho is the density, and t is the effective path length. Low-energy fluorescence lines (e.g., S Kalpha at 2.31 keV, Ca Kalpha at 3.69 keV) are attenuated much more strongly than high-energy lines (e.g., Fe Kalpha at 6.40 keV). The incident beam is also attenuated as it penetrates the sample, compounding the effect.

## Quick Diagnosis

```python
import numpy as np

# Compare thin vs thick regions of the same element
# ca_map: 2D elemental map for Ca; thickness_map: estimated sample thickness
thin_mask = thickness_map < np.percentile(thickness_map, 25)
thick_mask = thickness_map > np.percentile(thickness_map, 75)
ratio = np.mean(ca_map[thick_mask]) / np.mean(ca_map[thin_mask])
print(f"Thick/Thin Ca ratio: {ratio:.3f}")
print(f"If ratio << 1.0 and Ca is expected uniform → self-absorption likely")
```

## Detection Methods

### Visual Indicators

- Elemental maps for low-energy lines (S, P, Si, Ca) appear dimmer or washed out in thick or dense sample regions.
- High-energy lines (Fe, Cu, Zn) show expected contrast in the same regions.
- Compton scatter map (which tracks total sample mass) shows an inverse pattern: bright where elemental maps are suppressed.
- Elemental ratios (e.g., Ca/Fe) vary spatially in a pattern that correlates with sample thickness rather than true composition.

### Automated Detection

```python
import numpy as np
from scipy import stats


def detect_self_absorption(element_map, scatter_map, element_energy_keV,
                            element_name=''):
    """
    Detect potential self-absorption by correlating an elemental map
    with the Compton scatter map (proxy for sample mass/thickness).

    Parameters
    ----------
    element_map : np.ndarray
        2D elemental fluorescence map.
    scatter_map : np.ndarray
        2D Compton scatter map (proxy for mass thickness).
    element_energy_keV : float
        Fluorescence energy of the element in keV.
    element_name : str
        Label for reporting.

    Returns
    -------
    dict with keys:
        'correlation' : float — Pearson correlation (negative = absorption)
        'p_value' : float — statistical significance
        'self_absorption_likely' : bool
        'estimated_severity' : str — 'none', 'mild', 'moderate', 'severe'
    """
    valid = (element_map > 0) & (scatter_map > 0)
    if np.sum(valid) < 20:
        return {'self_absorption_likely': False}

    e_vals = element_map[valid].ravel()
    s_vals = scatter_map[valid].ravel()

    # Normalize
    e_norm = e_vals / np.mean(e_vals)
    s_norm = s_vals / np.mean(s_vals)

    # Ratio should be constant if no self-absorption
    ratio = e_norm / s_norm
    cv = np.std(ratio) / np.mean(ratio)

    corr, p_val = stats.pearsonr(e_vals, s_vals)

    # For low-energy lines, anti-correlation with scatter = self-absorption
    # For high-energy lines, positive correlation is expected (more material = more signal)
    if element_energy_keV < 4.0:
        # Low energy: expect positive correlation IF no self-absorption
        # Negative or weak correlation suggests self-absorption dominates
        self_abs = corr < 0.3
    else:
        # Higher energy: self-absorption is less common
        # Negative correlation is a stronger indicator
        self_abs = corr < -0.1

    if not self_abs:
        severity = 'none'
    elif cv < 0.2:
        severity = 'mild'
    elif cv < 0.5:
        severity = 'moderate'
    else:
        severity = 'severe'

    return {
        'element': element_name,
        'energy_keV': element_energy_keV,
        'correlation': float(corr),
        'p_value': float(p_val),
        'ratio_cv': float(cv),
        'self_absorption_likely': self_abs,
        'estimated_severity': severity,
    }
```

## Solutions and Mitigation

### Prevention (Before Data Collection)

- Prepare thin sections (< 10 um for soft X-ray elements, < 100 um for hard X-ray) to minimize the absorption path length.
- Use confocal XRF geometry with a polycapillary optic on the detector to restrict the detected volume to a thin slice.
- Orient the sample at 45 degrees to both beam and detector to minimize the total path length.
- For thick samples, consider measuring from both sides and comparing for consistency.

### Correction — Traditional Methods

```python
import numpy as np


def correct_self_absorption_fundamental_params(
    element_map, thickness_map, density, mu_in, mu_out,
    theta_in=45.0, theta_out=45.0
):
    """
    Apply fundamental-parameters self-absorption correction to an
    XRF elemental map using known sample thickness and composition.

    Parameters
    ----------
    element_map : np.ndarray
        2D measured fluorescence intensity map.
    thickness_map : np.ndarray
        2D map of local sample thickness in cm.
    density : float
        Sample density in g/cm^3.
    mu_in : float
        Mass attenuation coefficient at incident energy (cm^2/g).
    mu_out : float
        Mass attenuation coefficient at fluorescence energy (cm^2/g).
    theta_in : float
        Incident beam angle from surface normal (degrees).
    theta_out : float
        Detector angle from surface normal (degrees).

    Returns
    -------
    np.ndarray — corrected elemental map
    """
    theta_in_rad = np.radians(theta_in)
    theta_out_rad = np.radians(theta_out)

    # Effective attenuation coefficient
    mu_eff = (mu_in / np.cos(theta_in_rad)) + (mu_out / np.cos(theta_out_rad))

    # Absorption factor at each pixel
    # For a layer of thickness t: correction = mu_eff * rho * t / (1 - exp(-mu_eff * rho * t))
    rho_t = density * thickness_map
    exponent = mu_eff * rho_t

    # Avoid numerical issues for very thin or very thick regions
    correction = np.ones_like(element_map, dtype=float)
    thin_mask = exponent < 0.01
    thick_mask = exponent > 10.0
    mid_mask = ~thin_mask & ~thick_mask

    # Thin limit: correction ≈ 1 (no correction needed)
    correction[thin_mask] = 1.0

    # Thick limit: correction ≈ mu_eff * rho * t
    correction[thick_mask] = exponent[thick_mask]

    # General case
    correction[mid_mask] = exponent[mid_mask] / (
        1.0 - np.exp(-exponent[mid_mask])
    )

    corrected = element_map.astype(float) * correction
    return corrected


def estimate_thickness_from_scatter(scatter_map, calibration_factor):
    """
    Estimate local sample thickness from the Compton scatter map.

    Parameters
    ----------
    scatter_map : np.ndarray
        2D Compton scatter intensity map.
    calibration_factor : float
        Conversion factor from scatter intensity to thickness (cm/count),
        determined from a reference standard of known thickness.

    Returns
    -------
    np.ndarray — estimated thickness map in cm
    """
    return scatter_map.astype(float) * calibration_factor
```

### Correction — AI/ML Methods

Machine learning approaches can learn the complex nonlinear relationship between measured fluorescence intensities, scatter signals, and true concentrations without requiring explicit knowledge of sample composition and geometry. A neural network trained on paired data from thin-section references (where self-absorption is negligible) and thick-section measurements can predict correction factors for arbitrary sample geometries. This is particularly valuable for heterogeneous samples where the fundamental-parameters approach breaks down due to spatially varying composition and density.

## Impact If Uncorrected

Self-absorption leads to systematic underestimation of elemental concentrations in thick or dense sample regions, creating apparent compositional gradients that do not exist. Quantitative analysis becomes unreliable, particularly for light elements (Z < 20) whose low-energy fluorescence lines are most susceptible. Elemental ratios involving elements with widely different fluorescence energies (e.g., S/Fe) become thickness-dependent rather than composition-dependent. In geological or biological samples with variable thickness, self-absorption can invert the true concentration pattern — the thickest, most element-rich region appears depleted.

## Related Resources

- Related artifact: [I0 Normalization Issues](i0_normalization.md) — another systematic intensity correction
- Related artifact: [Peak Overlap](peak_overlap.md) — self-absorption correction requires accurate spectral decomposition first
- Related artifact: [Dead-Time Saturation](dead_time_saturation.md) — must be corrected before self-absorption analysis

## Key Takeaway

Self-absorption is a hidden quantitative bias that preferentially affects low-energy fluorescence lines and thick samples. Always assess its severity by comparing elemental maps with the Compton scatter map, and apply fundamental-parameters corrections or use thin-section preparation to ensure reliable quantification.
