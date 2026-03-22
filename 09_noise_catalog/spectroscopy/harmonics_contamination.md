# Harmonics Contamination

## Classification

| Attribute | Value |
|-----------|-------|
| **Modality** | Spectroscopy |
| **Noise Type** | Instrumental |
| **Severity** | Major |
| **Frequency** | Occasional |
| **Detection Difficulty** | Hard |

## Visual Examples

```
Effect of harmonic contamination on XANES spectrum:

  mu(E)                              mu(E)
   ^                                  ^
   |      /|                          |      /|
   |     / |                          |     / |
   |    /  |                          |    /  \___
   |   /   |   Sharp white line       |   /         Damped white line
   |  /    \___   (clean)             |  /           (harmonics present)
   | /         \___                   | /     \____
   |/              \___               |/           \___
   +----+----+----+-----> E          +----+----+----+-----> E

Edge step comparison:

  ln(I0/I1)                          ln(I0/I1)
   ^                                  ^
   |         ___________              |         ________
   |        /                         |       _/
   |       /    Edge step = 1.0       |      /   Edge step < 1.0
   |      /     (no harmonics)        |     /    (suppressed by harmonics)
   |     /                            |    /
   |____/                             |___/      Contaminated step is
   +----+----+----+-----> E          +----+----+----+-----> E
                                               reduced and gradual

Pre-edge region distortion:

  mu(E) pre-edge detail:
   ^
   |                Clean: flat pre-edge
   |  _______________/
   |                     <- pre-edge should be linear
   |  _ _ _ _ _ _./      Harmonics: curved pre-edge with
   |            /           apparent slope change
   +----+----+----+-> E
   -100  -50    E0
```

## Description

Harmonics contamination occurs when higher-order Bragg reflections (n=2, n=3, etc.) from the monochromator crystal pass through to the sample along with the desired fundamental energy. For a Si(111) monochromator set to deliver energy E, the second harmonic at 2E and third harmonic at 3E can also satisfy the Bragg condition and contaminate the beam. These higher-energy photons are absorbed differently by the sample and the ion chambers, distorting the measured absorption spectrum. The effect is most pronounced below and near the absorption edge where the fundamental photons are weakly absorbed but the harmonic photons (at 2x or 3x the energy) may be above a different absorption edge. The result is a systematic amplitude reduction in XANES features, a suppressed edge step, and a distorted pre-edge baseline.

## Root Cause

The Bragg equation 2d*sin(theta) = n*lambda allows multiple orders n to diffract simultaneously. While the structure factor of diamond-cubic Si crystals forbids the n=2 reflection for Si(111) (making the first problematic harmonic n=3 at 3E), Si(311) and Si(220) crystals do allow n=2. The harmonic content depends on: the source spectrum (bending magnet vs. undulator), which determines the relative flux at E vs. 2E and 3E; the crystal reflection used; the beam energy range; and any harmonic rejection optics in the beampath. At low energies (below ~5 keV), the harmonic fraction can be substantial because the source flux at 2E or 3E may be comparable to or even exceed the flux at E. Ion chamber gas fills also matter: a gas that absorbs E efficiently but transmits 2E poorly will distort the I0 measurement if harmonics are present.

## Quick Diagnosis

```python
import numpy as np

# Quick test: check if the pre-edge region has unexpected curvature
# energy: energy array (eV), mu: absorption array
# Find pre-edge region (50-150 eV below E0)
e0_approx = energy[np.argmax(np.gradient(mu, energy))]
pre_mask = (energy > e0_approx - 150) & (energy < e0_approx - 20)

if np.sum(pre_mask) > 10:
    e_pre = energy[pre_mask]
    mu_pre = mu[pre_mask]
    # Fit linear vs quadratic and compare
    p1 = np.polyfit(e_pre, mu_pre, 1)
    p2 = np.polyfit(e_pre, mu_pre, 2)
    res_linear = np.std(mu_pre - np.polyval(p1, e_pre))
    res_quad = np.std(mu_pre - np.polyval(p2, e_pre))
    curvature = np.abs(p2[0])
    print(f"Pre-edge curvature coefficient: {curvature:.2e}")
    print(f"Linear residual: {res_linear:.4e}, Quadratic residual: {res_quad:.4e}")
    if res_linear > 2 * res_quad and curvature > 1e-6:
        print("WARNING: Significant pre-edge curvature — possible harmonics")
```

## Detection Methods

### Visual Indicators

- The pre-edge region exhibits non-linear (curved) behavior instead of the expected linear or mildly sloping baseline.
- The white-line peak intensity is systematically lower than expected for the compound, or lower than literature values for the same material.
- The absorption edge step (delta_mu) is smaller than the theoretical value calculated from sample composition and thickness.
- Comparing transmission and fluorescence spectra of the same sample reveals amplitude differences in XANES features (fluorescence is less affected by harmonics).
- The EXAFS amplitude (coordination number) is systematically suppressed relative to known standards.

### Automated Detection

```python
import numpy as np
from scipy.optimize import curve_fit


def detect_harmonics_contamination(energy, mu, e0=None,
                                    pre_range=(-150, -20),
                                    post_range=(50, 200)):
    """
    Detect harmonics contamination by analyzing pre-edge curvature
    and edge step suppression.

    Parameters
    ----------
    energy : np.ndarray
        Energy array in eV.
    mu : np.ndarray
        Absorption coefficient (transmission: ln(I0/I1)).
    e0 : float or None
        Edge energy. If None, estimated from derivative maximum.
    pre_range : tuple
        (lower, upper) bounds relative to E0 for pre-edge region (eV).
    post_range : tuple
        (lower, upper) bounds relative to E0 for post-edge region (eV).

    Returns
    -------
    dict with keys:
        'pre_edge_curvature' : float — quadratic coefficient (eV^-2)
        'curvature_significance' : float — ratio of linear/quadratic residual
        'edge_step' : float — measured edge step
        'harmonics_fraction_est' : float — estimated fraction of harmonic
            content (0 to 1, approximate)
        'is_contaminated' : bool
        'severity' : str — 'none', 'mild', 'moderate', or 'severe'
    """
    if e0 is None:
        dmu = np.gradient(mu, energy)
        e0 = energy[np.argmax(dmu)]

    # Pre-edge analysis
    pre_mask = (energy >= e0 + pre_range[0]) & (energy <= e0 + pre_range[1])
    post_mask = (energy >= e0 + post_range[0]) & (energy <= e0 + post_range[1])

    results = {}

    if np.sum(pre_mask) > 5:
        e_pre = energy[pre_mask] - e0  # center for numerical stability
        mu_pre = mu[pre_mask]

        p1 = np.polyfit(e_pre, mu_pre, 1)
        p2 = np.polyfit(e_pre, mu_pre, 2)
        res_lin = np.std(mu_pre - np.polyval(p1, e_pre))
        res_quad = np.std(mu_pre - np.polyval(p2, e_pre))

        curvature = np.abs(p2[0])
        if res_quad > 0:
            curvature_sig = res_lin / res_quad
        else:
            curvature_sig = 1.0

        results['pre_edge_curvature'] = curvature
        results['curvature_significance'] = curvature_sig
    else:
        results['pre_edge_curvature'] = np.nan
        results['curvature_significance'] = 1.0

    # Edge step
    if np.sum(pre_mask) > 0 and np.sum(post_mask) > 0:
        pre_level = np.mean(mu[pre_mask])
        post_level = np.mean(mu[post_mask])
        edge_step = post_level - pre_level
        results['edge_step'] = edge_step
    else:
        results['edge_step'] = np.nan

    # Estimate harmonic fraction from edge step suppression
    # A contaminated spectrum has: mu_meas = (1-f)*mu_fund + f*mu_harm
    # The harmonic has no edge at E0, so the edge step is reduced by (1-f)
    # Without knowing the true edge step, we use the curvature as a proxy
    curvature_sig = results.get('curvature_significance', 1.0)
    if curvature_sig > 1.5:
        # Rough estimate: higher curvature ratio suggests more contamination
        f_est = min(0.5, (curvature_sig - 1.0) * 0.05)
    else:
        f_est = 0.0

    results['harmonics_fraction_est'] = f_est

    # Classify severity
    if f_est < 0.01:
        results['severity'] = 'none'
        results['is_contaminated'] = False
    elif f_est < 0.05:
        results['severity'] = 'mild'
        results['is_contaminated'] = True
    elif f_est < 0.15:
        results['severity'] = 'moderate'
        results['is_contaminated'] = True
    else:
        results['severity'] = 'severe'
        results['is_contaminated'] = True

    return results


def glitch_scan_harmonic_test(energy, mu, glitch_energy_threshold=3.0):
    """
    Identify sharp glitches in mu(E) that may arise from harmonic Bragg
    reflections (monochromator glitches where a harmonic coincides with
    a Bragg reflection of a parasitic crystal plane).

    Parameters
    ----------
    energy : np.ndarray
        Energy array (eV).
    mu : np.ndarray
        Absorption spectrum.
    glitch_energy_threshold : float
        Sigma threshold for identifying glitch points.

    Returns
    -------
    dict with keys:
        'glitch_indices' : np.ndarray — indices of glitch points
        'glitch_energies' : np.ndarray — energies of glitches
        'n_glitches' : int
    """
    # Compute second derivative and look for spikes
    d2mu = np.gradient(np.gradient(mu, energy), energy)
    med = np.median(d2mu)
    mad = np.median(np.abs(d2mu - med))
    robust_std = 1.4826 * mad

    glitch_mask = np.abs(d2mu - med) > glitch_energy_threshold * robust_std
    glitch_indices = np.where(glitch_mask)[0]

    return {
        'glitch_indices': glitch_indices,
        'glitch_energies': energy[glitch_indices],
        'n_glitches': len(glitch_indices),
    }
```

## Solutions and Mitigation

### Prevention (Before Data Collection)

- Detune the second crystal of the double-crystal monochromator by a few percent (typically 30-50% of peak rocking curve intensity). Harmonics have a narrower rocking curve and are rejected more efficiently than the fundamental.
- Install a harmonic rejection mirror at a grazing angle that reflects the fundamental energy but not the harmonics (mirror cutoff energy between E and 2E).
- Choose the appropriate Si reflection: Si(111) has no second-order harmonic (n=2 is forbidden), so the first contaminating harmonic is at 3E. Si(311) and Si(220) allow n=2.
- For low-energy experiments, consider using a mirror pair for harmonic rejection upstream of the monochromator.
- Choose ion chamber gas fills that are appropriate for the fundamental energy and do not have significant absorption changes at the harmonic energies.

### Correction — Traditional Methods

```python
import numpy as np
from scipy.optimize import minimize_scalar


def correct_harmonics_transmission(energy, i0, i1, harmonic_fraction,
                                    mu_harmonic_ratio=0.1):
    """
    Correct transmission XAS data for a known harmonic contamination fraction.

    The measured intensities contain a harmonic component:
        I0_meas = (1-f)*I0_fund + f*I0_harm
        I1_meas = (1-f)*I1_fund + f*I1_harm

    where f is the harmonic fraction, and the harmonic component passes
    through the sample with a different (usually much lower) absorption.

    Parameters
    ----------
    energy : np.ndarray
        Energy array (eV).
    i0 : np.ndarray
        Incident beam intensity (measured).
    i1 : np.ndarray
        Transmitted beam intensity (measured).
    harmonic_fraction : float
        Fraction of beam intensity from harmonics (0 to 1).
    mu_harmonic_ratio : float
        Ratio of harmonic absorption to fundamental absorption.
        Typically << 1 since harmonic energy is well above the edge.

    Returns
    -------
    dict with keys:
        'energy' : np.ndarray
        'mu_corrected' : np.ndarray — corrected ln(I0/I1)
        'mu_uncorrected' : np.ndarray — original ln(I0/I1)
    """
    f = harmonic_fraction

    # Original (contaminated) mu
    mu_meas = np.log(i0 / i1)

    # Model: I0_meas = (1-f)*I0_f + f*I0_h
    # I1_meas = (1-f)*I0_f*exp(-mu_f*x) + f*I0_h*exp(-mu_h*x)
    # where mu_h*x ≈ mu_harmonic_ratio * mu_f*x
    # Simplify: let a = exp(-mu_f*x), then:
    # I1/I0 = [(1-f)*a + f*a^r] / 1  (normalized I0=1)
    # mu_meas = -ln[(1-f)*a + f*a^r]
    # We solve for a (which gives mu_corrected = -ln(a)):

    mu_corrected = np.zeros_like(mu_meas)
    r = mu_harmonic_ratio

    for i in range(len(energy)):
        ratio = np.exp(-mu_meas[i])  # I1/I0 measured

        # Solve: (1-f)*a + f*a^r = ratio for a
        def objective(log_a):
            a = np.exp(log_a)
            predicted = (1 - f) * a + f * a**r
            return (predicted - ratio)**2

        result = minimize_scalar(objective, bounds=(-10, 0), method='bounded')
        mu_corrected[i] = -result.x  # mu = -ln(a)

    return {
        'energy': energy,
        'mu_corrected': mu_corrected,
        'mu_uncorrected': mu_meas,
    }


def estimate_harmonic_fraction_from_step(energy, mu, e0, theoretical_step,
                                          pre_range=(-150, -20),
                                          post_range=(50, 200)):
    """
    Estimate the harmonic contamination fraction by comparing the
    measured edge step to the theoretical value.

    Parameters
    ----------
    energy : np.ndarray
        Energy array (eV).
    mu : np.ndarray
        Absorption spectrum.
    e0 : float
        Edge energy (eV).
    theoretical_step : float
        Expected edge step from sample composition/thickness calculation.
    pre_range, post_range : tuple
        Energy ranges relative to E0 for baseline estimates.

    Returns
    -------
    float — estimated harmonic fraction
    """
    pre_mask = (energy >= e0 + pre_range[0]) & (energy <= e0 + pre_range[1])
    post_mask = (energy >= e0 + post_range[0]) & (energy <= e0 + post_range[1])

    measured_step = np.mean(mu[post_mask]) - np.mean(mu[pre_mask])

    # Edge step is suppressed by factor (1 - f) where f is harmonic fraction
    if theoretical_step > 0:
        f_est = max(0.0, 1.0 - measured_step / theoretical_step)
    else:
        f_est = 0.0

    return min(f_est, 1.0)
```

### Correction — AI/ML Methods

Machine learning approaches to harmonics correction train models on pairs of contaminated and clean spectra generated from theoretical calculations. A forward model simulates the effect of varying harmonic fractions on known reference spectra. A neural network (typically a 1D CNN or fully connected network) is then trained to invert this model — given a contaminated spectrum, it predicts the clean spectrum and the harmonic fraction simultaneously. This is especially useful when the harmonic fraction varies with energy due to beamline optics, making a single-parameter correction insufficient. Bayesian neural networks can additionally provide uncertainty estimates on the corrected spectrum.

## Impact If Uncorrected

Harmonic contamination systematically distorts the measured absorption spectrum in ways that propagate to all downstream analyses. XANES features are damped, reducing the diagnostic power of fingerprinting and linear combination fitting. White-line intensities used for oxidation state determination are suppressed, potentially leading to incorrect redox assignments. In EXAFS, the amplitude reduction factor S0^2 absorbs the harmonic effect, leading to underestimated coordination numbers. The edge step is reduced, causing errors in concentration determination. Pre-edge features used for site symmetry analysis (e.g., 1s-3d transitions in transition metals) are distorted relative to the main edge, corrupting peak area ratios. The effect is insidious because the spectrum still looks plausible — just slightly "softer" than expected.

## Related Resources

- Related artifact: [Self-Absorption (XAS)](self_absorption_xas.md) — also causes amplitude damping but with a different mechanism
- Related artifact: [Energy Calibration Drift](energy_calibration_drift.md) — both affect the energy scale accuracy
- Related artifact: [Statistical Noise in EXAFS](statistical_noise_exafs.md) — noise assessment should account for harmonics

## Key Takeaway

Harmonic contamination is a subtle but pervasive artifact that damps all spectral features uniformly. Always detune the monochromator or use harmonic rejection mirrors, and verify the pre-edge linearity and edge step magnitude against theoretical expectations. If harmonics are suspected, compare transmission and fluorescence measurements — the fluorescence channel is inherently less sensitive to harmonic contamination because the detector is energy-discriminating.
