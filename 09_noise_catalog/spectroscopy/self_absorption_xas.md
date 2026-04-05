# Self-Absorption (XAS)

## Classification

| Attribute | Value |
|-----------|-------|
| **Modality** | Spectroscopy |
| **Noise Type** | Systematic |
| **Severity** | Major |
| **Frequency** | Common |
| **Detection Difficulty** | Moderate |

## Visual Examples

```
Fluorescence vs. transmission for concentrated sample:

  mu(E)                               mu(E)
   ^                                   ^
   |      /|                           |      /|
   |     / |                           |     /  \
   |    /  |   Transmission            |    /    \__    Fluorescence
   |   /   |   (true spectrum)         |   /         \_ (self-absorbed)
   |  /    \___                        |  /
   | /         \___                    | /           White line
   |/              \___                |/            flattened
   +----+----+----+-----> E           +----+----+----+-----> E

EXAFS amplitude comparison:

  k^2*chi(k)                          k^2*chi(k)
   ^                                   ^
   | /\    /\      /\                  |  /\    /\     /\
   |/  \  /  \  __/  \                 | / .\  / .\  _/ .\
   |    \/    \/      \/\              |/.   \/   .\/    .\/\
   |                      \            |                      \
   +----+----+----+-----> k           +----+----+----+-----> k
   True amplitude (dilute)             Reduced amplitude (concentrated)

Concentration dependence:

  White line     |
  peak height    |  x dilute
                 |  x  10% wt
   ^             |    x  30% wt
   |  x          |      x  50% wt
   |    x        |         x  pure compound
   |      x      |
   |        x x  |   Peak height saturates with increasing
   |              |   concentration (non-linear response)
   +---+---+---+-+-> Concentration
```

## Description

Self-absorption (also called over-absorption) is a systematic distortion of XAS spectra measured in fluorescence mode on concentrated or thick samples. When the absorbing element is present in high concentration, the incident X-rays that excite fluorescence are attenuated within the sample, and the emitted fluorescence photons are further re-absorbed before escaping. This dual attenuation preferentially suppresses the signal at energies where the absorption coefficient is highest — namely, at the white line and near-edge features. The result is a "flattened" XANES spectrum with a reduced white-line peak and suppressed EXAFS amplitudes. The effect is not random noise but a deterministic, concentration-dependent distortion that mimics a change in the electronic or structural properties of the sample.

## Root Cause

In fluorescence detection, the measured signal I_f is proportional to the fluorescence yield times the absorption coefficient mu_f(E) of the target element. However, both the incident beam (at energy E) and the fluorescence photons (at energy E_f < E) are attenuated as they travel through the sample. For a thick, concentrated sample the effective fluorescence signal saturates: increasing mu(E) at the white line does not produce a proportional increase in I_f because the additional absorption occurs deeper in the sample where both the incident beam and the fluorescence escape probability are diminished. Mathematically, I_f is proportional to mu_f(E) / [mu_total(E) + mu_total(E_f) * sin(theta_in)/sin(theta_out)], where the denominator grows with mu_f(E) and thus suppresses the peak. The effect scales with: the concentration of the absorbing element, the total sample thickness, the incident and exit angles, and the ratio of the target element's absorption to the matrix absorption.

## Quick Diagnosis

```python
import numpy as np

# Quick check: compare normalized white-line heights
# If fluorescence white-line is significantly lower than transmission
# or literature value, self-absorption is likely.

# For a single fluorescence spectrum:
# energy, mu_fluo: fluorescence-derived mu(E) (normalized)
e0_idx = np.argmax(np.gradient(mu_fluo, energy))
post_edge = np.mean(mu_fluo[e0_idx + 100:e0_idx + 200])

# Normalize
mu_norm = mu_fluo / post_edge
white_line_height = np.max(mu_norm[e0_idx - 10:e0_idx + 50])

print(f"Normalized white-line peak height: {white_line_height:.2f}")
print(f"(Expected for dilute standard: typically > 1.5 for L-edges)")
if white_line_height < 1.2:
    print("WARNING: White line appears flattened — possible self-absorption")
```

## Detection Methods

### Visual Indicators

- The white-line peak in fluorescence is visibly lower and broader compared to a dilute reference or transmission measurement of the same compound.
- Normalized XANES spectra from samples of different concentrations do not overlay — higher concentration samples have lower white-line peaks.
- EXAFS amplitudes (coordination numbers from fits) are systematically lower than expected crystallographic values.
- The spectrum shape looks "compressed" toward the step normalization baseline.
- Pre-edge features may appear relatively enhanced compared to the main edge because they are less affected.

### Automated Detection

```python
import numpy as np
from scipy.interpolate import interp1d


def detect_self_absorption(energy, mu_fluo, mu_ref=None,
                            e0=None, amplitude_threshold=0.8):
    """
    Detect self-absorption by comparing a fluorescence spectrum to a
    reference (transmission or dilute standard).

    Parameters
    ----------
    energy : np.ndarray
        Energy array (eV).
    mu_fluo : np.ndarray
        Fluorescence-derived mu(E), normalized to edge step = 1.
    mu_ref : np.ndarray or None
        Reference spectrum (transmission or dilute standard) on the
        same energy grid and normalization. If None, uses heuristic
        tests only.
    e0 : float or None
        Edge energy. If None, estimated from derivative.
    amplitude_threshold : float
        Ratio of fluo/ref white-line below which self-absorption
        is flagged.

    Returns
    -------
    dict with keys:
        'white_line_ratio' : float — fluo/ref peak ratio (1.0 = no effect)
        'is_self_absorbed' : bool
        'severity' : str — 'none', 'mild', 'moderate', or 'severe'
        'amplitude_damping' : float — estimated overall amplitude factor
    """
    if e0 is None:
        dmu = np.gradient(mu_fluo, energy)
        e0 = energy[np.argmax(dmu)]

    # Define edge region for white-line comparison
    edge_mask = (energy >= e0 - 5) & (energy <= e0 + 40)
    wl_fluo = np.max(mu_fluo[edge_mask])

    if mu_ref is not None:
        wl_ref = np.max(mu_ref[edge_mask])
        wl_ratio = wl_fluo / wl_ref if wl_ref > 0 else 1.0

        # Compare EXAFS amplitudes in post-edge oscillation region
        post_mask = (energy >= e0 + 50) & (energy <= e0 + 300)
        if np.sum(post_mask) > 20:
            osc_fluo = np.std(mu_fluo[post_mask] - np.mean(mu_fluo[post_mask]))
            osc_ref = np.std(mu_ref[post_mask] - np.mean(mu_ref[post_mask]))
            amp_ratio = osc_fluo / osc_ref if osc_ref > 0 else 1.0
        else:
            amp_ratio = wl_ratio
    else:
        # Heuristic: check if white line is unusually low for normalized data
        wl_ratio = wl_fluo  # For normalized data, white line > 1.0 expected
        amp_ratio = wl_ratio

    # Classify
    if wl_ratio > amplitude_threshold:
        severity = 'none'
        is_absorbed = False
    elif wl_ratio > 0.6:
        severity = 'mild'
        is_absorbed = True
    elif wl_ratio > 0.4:
        severity = 'moderate'
        is_absorbed = True
    else:
        severity = 'severe'
        is_absorbed = True

    return {
        'white_line_ratio': wl_ratio,
        'is_self_absorbed': is_absorbed,
        'severity': severity,
        'amplitude_damping': amp_ratio,
    }


def concentration_series_test(energy_list, mu_list, concentrations):
    """
    Test for self-absorption using a series of spectra at different
    concentrations. In the absence of self-absorption, normalized
    spectra should be identical regardless of concentration.

    Parameters
    ----------
    energy_list : list of np.ndarray
        Energy arrays for each concentration.
    mu_list : list of np.ndarray
        Normalized mu(E) arrays for each concentration.
    concentrations : list of float
        Concentration values (e.g., weight percent).

    Returns
    -------
    dict with keys:
        'white_line_heights' : list of float
        'damping_vs_concentration' : np.ndarray — shape (n,) of WL heights
        'has_self_absorption' : bool
        'critical_concentration' : float or None — concentration above which
            self-absorption exceeds 10% damping
    """
    wl_heights = []
    for energy, mu in zip(energy_list, mu_list):
        dmu = np.gradient(mu, energy)
        e0 = energy[np.argmax(dmu)]
        edge_mask = (energy >= e0 - 5) & (energy <= e0 + 40)
        wl_heights.append(np.max(mu[edge_mask]))

    wl_heights = np.array(wl_heights)
    concentrations = np.array(concentrations)

    # Sort by concentration
    sort_idx = np.argsort(concentrations)
    conc_sorted = concentrations[sort_idx]
    wl_sorted = wl_heights[sort_idx]

    # Check for monotonic decrease with concentration
    # Normalize to the lowest concentration (most dilute)
    wl_norm = wl_sorted / wl_sorted[0] if wl_sorted[0] > 0 else wl_sorted

    # Find critical concentration (where damping > 10%)
    below_90 = np.where(wl_norm < 0.9)[0]
    crit_conc = conc_sorted[below_90[0]] if len(below_90) > 0 else None

    # Self-absorption present if highest concentration shows >5% reduction
    has_sa = (wl_norm[-1] < 0.95) if len(wl_norm) > 1 else False

    return {
        'white_line_heights': wl_heights.tolist(),
        'damping_vs_concentration': wl_norm.tolist(),
        'has_self_absorption': has_sa,
        'critical_concentration': crit_conc,
    }
```

## Solutions and Mitigation

### Prevention (Before Data Collection)

- For concentrated samples, use transmission mode instead of fluorescence whenever possible. Optimize sample thickness so that the total absorption mu*x is approximately 1-2 above the edge.
- Dilute concentrated samples with an inert matrix (e.g., BN, cellulose, polyethylene) to reduce the absorber concentration below the self-absorption threshold — typically below 5-10 atomic percent for K-edges of 3d transition metals.
- Use grazing-exit geometry where the fluorescence takeoff angle is small, reducing the effective sample thickness seen by the fluorescence photons.
- For thin films or surface studies, consider total electron yield (TEY) detection which samples only the near-surface region and is less susceptible.

### Correction — Traditional Methods

```python
import numpy as np
from scipy.optimize import minimize


def booth_bridges_correction(energy, mu_fluo, mu_total_below,
                              alpha, element_fraction,
                              fluorescence_energy):
    """
    Apply the Booth & Bridges self-absorption correction to fluorescence
    XAS data.

    Based on: C.H. Booth and F. Bridges, Physica Scripta T115, 202 (2005).

    The correction inverts the self-absorption formula:
        mu_fluo_meas(E) ∝ mu_a(E) / [mu_total(E) + mu_total(E_f) * alpha]

    where alpha = sin(theta_in) / sin(theta_out) is the geometric factor.

    Parameters
    ----------
    energy : np.ndarray
        Energy array (eV).
    mu_fluo : np.ndarray
        Measured fluorescence mu(E) (normalized to edge step ~ 1).
    mu_total_below : float
        Total absorption coefficient of the sample matrix at energies
        just below the edge (cm^-1 or consistent units).
    alpha : float
        Geometric factor sin(theta_in)/sin(theta_out).
    element_fraction : float
        Atomic fraction of the absorbing element.
    fluorescence_energy : float
        Energy of the fluorescence line (eV), e.g., Fe Ka = 6404 eV.

    Returns
    -------
    dict with keys:
        'energy' : np.ndarray
        'mu_corrected' : np.ndarray — self-absorption corrected spectrum
        'correction_factor' : np.ndarray — multiplicative correction applied
    """
    # Estimate mu_total(E_f) from tabulated values or approximate
    # as mu_total just below the edge (the fluorescence energy is always
    # below the edge, so the absorbing element contributes only background)
    mu_f = mu_total_below  # approximation for mu at fluorescence energy

    # The measured fluorescence is proportional to:
    # chi_f(E) = mu_a(E) / [mu_bg(E) + mu_a(E) + alpha * mu_f]
    # where mu_bg is the background (matrix) absorption at energy E
    # and mu_a(E) is the absorption of the target element.

    # Normalize: let mu_fluo be on scale where edge step ~ 1
    # Then mu_a(E) = element_fraction * mu_fluo(E) * scale_factor
    # We work in normalized units:

    # Self-absorption correction factor per point:
    # mu_true(E) = mu_meas(E) / [1 - mu_meas(E) * element_fraction /
    #              (mu_total_below * (1 + alpha))]

    denominator_scale = mu_total_below * (1 + alpha)

    mu_corrected = np.zeros_like(mu_fluo)
    for i in range(len(energy)):
        sa_factor = element_fraction * mu_fluo[i] / denominator_scale
        if sa_factor < 0.99:  # avoid divergence
            mu_corrected[i] = mu_fluo[i] / (1 - sa_factor)
        else:
            mu_corrected[i] = mu_fluo[i]  # cannot correct saturated data

    # Renormalize to edge step = 1
    pre_mask = energy < (energy[0] + 0.1 * (energy[-1] - energy[0]))
    post_mask = energy > (energy[0] + 0.5 * (energy[-1] - energy[0]))
    if np.sum(pre_mask) > 0 and np.sum(post_mask) > 0:
        pre_level = np.mean(mu_corrected[pre_mask])
        post_level = np.mean(mu_corrected[post_mask])
        step = post_level - pre_level
        if step > 0:
            mu_corrected = (mu_corrected - pre_level) / step

    correction_factor = np.where(mu_fluo != 0,
                                  mu_corrected / mu_fluo, 1.0)

    return {
        'energy': energy,
        'mu_corrected': mu_corrected,
        'correction_factor': correction_factor,
    }


def iterative_self_absorption_correction(energy, mu_fluo, mu_ref,
                                          max_iter=50, tol=1e-4):
    """
    Iterative self-absorption correction by fitting the fluorescence
    spectrum to match a reference (transmission or dilute standard).

    Finds the self-absorption parameter that best maps mu_fluo onto mu_ref.

    Parameters
    ----------
    energy : np.ndarray
        Common energy grid (eV).
    mu_fluo : np.ndarray
        Fluorescence spectrum (normalized).
    mu_ref : np.ndarray
        Reference transmission/dilute spectrum (normalized).
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance on the self-absorption parameter.

    Returns
    -------
    dict with keys:
        'mu_corrected' : np.ndarray
        'sa_parameter' : float — fitted self-absorption parameter
        'residual' : float — final RMS difference from reference
        'converged' : bool
    """
    # Model: mu_meas = mu_true / (1 + beta * mu_true)
    # Invert: mu_true = mu_meas / (1 - beta * mu_meas)
    # Find beta that minimizes ||mu_true - mu_ref||^2

    def apply_correction(mu_meas, beta):
        denom = 1 - beta * mu_meas
        denom = np.maximum(denom, 0.01)  # prevent divergence
        return mu_meas / denom

    def objective(beta):
        mu_corr = apply_correction(mu_fluo, beta)
        # Renormalize
        pre = np.mean(mu_corr[:20])
        post = np.mean(mu_corr[-50:])
        step = post - pre
        if step > 0:
            mu_norm = (mu_corr - pre) / step
        else:
            mu_norm = mu_corr
        return np.mean((mu_norm - mu_ref)**2)

    result = minimize(objective, x0=0.1, bounds=[(0.0, 0.95)],
                      method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': tol})

    beta_opt = result.x[0]
    mu_corrected = apply_correction(mu_fluo, beta_opt)

    # Final normalization
    pre = np.mean(mu_corrected[:20])
    post = np.mean(mu_corrected[-50:])
    step = post - pre
    if step > 0:
        mu_corrected = (mu_corrected - pre) / step

    residual = np.sqrt(np.mean((mu_corrected - mu_ref)**2))

    return {
        'mu_corrected': mu_corrected,
        'sa_parameter': beta_opt,
        'residual': residual,
        'converged': result.success,
    }
```

### Correction — AI/ML Methods

Deep learning methods for self-absorption correction train a neural network to map self-absorbed spectra to their corrected counterparts. Training data is generated by applying the forward self-absorption model at various concentrations and geometries to a library of clean transmission spectra. A 1D convolutional encoder-decoder learns the nonlinear inverse mapping. The advantage over analytical correction is that the network can handle complex sample geometries (particles, inhomogeneous films) where the simple slab model breaks down. Physics-informed neural networks that embed the self-absorption equation as a constraint in the loss function improve generalization to unseen compositions.

## Impact If Uncorrected

Self-absorption produces systematic errors that are often mistaken for real chemical or structural effects. White-line peak heights — used for oxidation state fingerprinting — are suppressed, potentially causing a reduced metal to be misidentified as a more oxidized form. EXAFS coordination numbers are underestimated because the amplitude is damped; errors of 20-50% in N (coordination number) are common for concentrated samples. Bond distances from EXAFS fitting are generally less affected, but Debye-Waller factors compensate for the amplitude error, yielding artificially large disorder parameters. Linear combination fitting of XANES can produce incorrect phase fractions because the self-absorbed spectrum no longer matches any standard in the reference library.

## Related Resources

- [Spectroscopy EDA notebook](../../06_data_structures/eda/spectroscopy_eda.md) — fluorescence vs. transmission comparison checks
- Related artifact: [Harmonics Contamination](harmonics_contamination.md) — also causes amplitude damping but via a different mechanism
- Related artifact: [Statistical Noise in EXAFS](statistical_noise_exafs.md) — noise effects compound with self-absorption

## Real-World Before/After Examples

The following published sources provide real experimental before/after comparisons:

| Source | Type | Figure/Location | Description | License |
|--------|------|-----------------|-------------|---------|
| [Trevorah et al. 2019](https://doi.org/10.1107/S160057751901302X) | Paper | Figs 2--3 | Correcting for self-absorption in fluorescence XAS — distorted vs corrected XAS spectra | -- |

**Key references with published before/after comparisons:**
- **Trevorah et al. (2019)**: Figs 2-3 show distorted (self-absorbed) vs corrected XAS spectra from real fluorescence measurements. DOI: 10.1107/S160057751901302X

## Key Takeaway

Self-absorption is the most common source of systematic error in fluorescence XAS of concentrated samples. Always compare fluorescence and transmission spectra when both are available, dilute samples whenever practical, and apply the Booth & Bridges correction algorithm when dilution is not feasible. Never trust coordination numbers from uncorrected fluorescence data on samples with more than a few atomic percent of the target element.
