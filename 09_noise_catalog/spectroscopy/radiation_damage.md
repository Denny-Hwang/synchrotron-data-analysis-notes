# Radiation Damage

## Classification

| Attribute | Value |
|-----------|-------|
| **Modality** | Spectroscopy |
| **Noise Type** | Systematic |
| **Severity** | Critical |
| **Frequency** | Occasional |
| **Detection Difficulty** | Moderate |

## Visual Examples

```
Progressive spectral changes with successive scans:

  mu(E)
   ^
   |      /|  Scan 1 (fresh spot)
   |     / |
   |    /  |         /\  Scan 2
   |   /   \___     / .\__
   |  /         \  /  .    \    Scan 5
   | /           \/   .     \_____
   |/            .    .
   +----+----+----+----+-----> E(eV)
        E0

   Progressive: white line decreases, edge shifts,
   post-edge structure changes

Edge position shift with dose:

  E0 (eV)
   ^
   | x                      Reduction: E0 shifts to
   |   x                    lower energy with dose
   |     x
   |       x x
   |           x x x
   +----+----+----+-----> Scan number / Dose
   1    2    3    4    5

   E0 (eV)
   ^
   |               x x x   Oxidation: E0 shifts to
   |           x x          higher energy with dose
   |       x
   |   x
   | x
   +----+----+----+-----> Scan number / Dose
   1    2    3    4    5

Difference spectra revealing damage:

  mu(E) - mu_scan1(E)
   ^
   |          /\
   |         /  \          Positive: new features growing
   |  ______/    \______
   |        \    /         Negative: original features diminishing
   |         \  /
   |          \/
   +----+----+----+-----> E(eV)
   Difference between scan N and scan 1
   shows systematic spectral changes
```

## Description

Radiation damage occurs when the intense synchrotron X-ray beam induces chemical or structural changes in the sample during measurement. The most common manifestation is a progressive shift in the absorption edge position across successive scans, indicating a change in oxidation state (typically reduction of metal centers by photoelectrons). Other symptoms include changes in the white-line intensity, appearance or disappearance of pre-edge features, and alterations in the EXAFS oscillation pattern reflecting bond breaking or rearrangement. Radiation damage is especially problematic for biological samples, redox-sensitive materials (e.g., Mn, Ce, S compounds), organometallic complexes, and hydrated or organic-containing specimens. The damage is cumulative and irreversible.

## Root Cause

The X-ray beam deposits energy in the sample through photoelectric absorption and Compton scattering. This energy generates photoelectrons and secondary (Auger) electrons that can break chemical bonds, reduce or oxidize metal centers, generate free radicals, and drive radiolysis of water or organic solvents. Specific damage mechanisms include: (1) direct photoreduction of metal ions by photoelectrons (e.g., Mn(IV) to Mn(II), Fe(III) to Fe(II)); (2) radiolysis of water producing hydroxyl radicals and solvated electrons that attack the sample; (3) bond scission in organic ligands; (4) local heating at the beam spot causing thermal decomposition; (5) generation of color centers in crystalline matrices. The damage rate scales with the photon flux density (photons per second per unit area), the absorbed dose per scan, and the radiation sensitivity of the material. Focused undulator beams at third-generation sources deliver dose rates orders of magnitude higher than bending magnet beamlines, making damage much more likely.

## Quick Diagnosis

```python
import numpy as np

# Given: multiple scans of the same sample spot
# energy_scans: list of energy arrays
# mu_scans: list of mu(E) arrays

def find_e0(energy, mu):
    """Find E0 as energy of maximum derivative."""
    dmu = np.gradient(mu, energy)
    return energy[np.argmax(dmu)]

e0_values = [find_e0(e, m) for e, m in zip(energy_scans, mu_scans)]
e0_shift = np.array(e0_values) - e0_values[0]

print("E0 shift relative to scan 1:")
for i, shift in enumerate(e0_shift):
    marker = " <-- DAMAGE" if abs(shift) > 0.3 else ""
    print(f"  Scan {i+1}: {shift:+.2f} eV{marker}")

# Check for monotonic drift (hallmark of radiation damage)
diffs = np.diff(e0_values)
if np.all(diffs < -0.05):
    print("WARNING: Monotonic edge shift to lower energy — likely photoreduction")
elif np.all(diffs > 0.05):
    print("WARNING: Monotonic edge shift to higher energy — likely photo-oxidation")
```

## Detection Methods

### Visual Indicators

- Overlay of successive scans shows a progressive shift in the absorption edge (E0) in one direction.
- The white-line peak intensity changes monotonically with scan number — decreasing for most photoreduction cases.
- Pre-edge features may grow or shrink as the local symmetry changes with damage.
- Difference spectra (scan N minus scan 1) show systematic, growing deviations rather than random noise.
- EXAFS oscillations change phase or amplitude progressively, indicating bond length or coordination changes.
- If the beam is moved to a fresh spot, the first scan at the new position matches the original first scan, confirming beam-induced changes.

### Automated Detection

```python
import numpy as np
from scipy.stats import linregress
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d


def detect_radiation_damage(energy_scans, mu_scans, e0_shift_threshold=0.2,
                             spectral_change_threshold=0.02):
    """
    Detect radiation damage by analyzing progressive spectral changes
    across sequential scans of the same sample spot.

    Parameters
    ----------
    energy_scans : list of np.ndarray
        Energy arrays for each scan (eV).
    mu_scans : list of np.ndarray
        Absorption coefficient arrays for each scan.
    e0_shift_threshold : float
        Minimum cumulative E0 shift (eV) to flag damage.
    spectral_change_threshold : float
        Minimum RMS spectral change (normalized) to flag damage.

    Returns
    -------
    dict with keys:
        'e0_per_scan' : list of float
        'e0_shift_total' : float — total E0 drift (eV)
        'e0_drift_rate' : float — eV per scan (from linear fit)
        'spectral_rms_change' : list of float — RMS change per scan vs scan 1
        'is_damaged' : bool
        'damage_onset_scan' : int or None — first scan showing significant change
        'damage_type' : str — 'reduction', 'oxidation', 'structural', or 'none'
    """
    n_scans = len(energy_scans)

    # Find E0 for each scan
    e0_values = []
    for energy, mu in zip(energy_scans, mu_scans):
        dmu = np.gradient(mu, energy)
        dmu_smooth = uniform_filter1d(dmu, size=5)
        e0 = energy[np.argmax(dmu_smooth)]
        e0_values.append(e0)

    e0_values = np.array(e0_values)
    e0_shift_total = e0_values[-1] - e0_values[0]

    # Linear drift rate
    scan_nums = np.arange(n_scans)
    if n_scans > 2:
        slope, intercept, r_value, p_value, _ = linregress(scan_nums, e0_values)
        e0_drift_rate = slope
    else:
        e0_drift_rate = e0_shift_total / max(n_scans - 1, 1)
        r_value = 1.0

    # Interpolate all scans onto common grid for spectral comparison
    e_min = max(e.min() for e in energy_scans)
    e_max = min(e.max() for e in energy_scans)
    de = np.median([e[1] - e[0] for e in energy_scans])
    common_e = np.arange(e_min, e_max, de)

    mu_common = []
    for energy, mu in zip(energy_scans, mu_scans):
        f = interp1d(energy, mu, kind='linear', bounds_error=False,
                     fill_value=np.nan)
        mu_common.append(f(common_e))
    mu_common = np.array(mu_common)

    # Normalize each spectrum
    for i in range(n_scans):
        pre = np.nanmean(mu_common[i, :30])
        post = np.nanmean(mu_common[i, -50:])
        step = post - pre
        if step > 0:
            mu_common[i] = (mu_common[i] - pre) / step

    # RMS change relative to scan 1
    rms_changes = []
    for i in range(n_scans):
        diff = mu_common[i] - mu_common[0]
        rms = np.sqrt(np.nanmean(diff**2))
        rms_changes.append(rms)

    # Find damage onset
    damage_onset = None
    for i in range(1, n_scans):
        if (abs(e0_values[i] - e0_values[0]) > e0_shift_threshold or
                rms_changes[i] > spectral_change_threshold):
            damage_onset = i + 1  # 1-indexed scan number
            break

    # Classify damage type
    if abs(e0_shift_total) < e0_shift_threshold:
        if max(rms_changes) > spectral_change_threshold:
            damage_type = 'structural'
        else:
            damage_type = 'none'
    elif e0_shift_total < 0:
        damage_type = 'reduction'
    else:
        damage_type = 'oxidation'

    is_damaged = (damage_type != 'none')

    return {
        'e0_per_scan': e0_values.tolist(),
        'e0_shift_total': e0_shift_total,
        'e0_drift_rate': e0_drift_rate,
        'spectral_rms_change': rms_changes,
        'is_damaged': is_damaged,
        'damage_onset_scan': damage_onset,
        'damage_type': damage_type,
    }


def dose_estimation(flux, beam_size_h, beam_size_v, scan_time,
                     sample_density, mass_atten_coeff):
    """
    Estimate the absorbed radiation dose per scan.

    Parameters
    ----------
    flux : float
        Photon flux at sample (photons/s).
    beam_size_h : float
        Horizontal beam size at sample (cm).
    beam_size_v : float
        Vertical beam size at sample (cm).
    scan_time : float
        Duration of one scan (seconds).
    sample_density : float
        Sample density (g/cm^3).
    mass_atten_coeff : float
        Mass energy-absorption coefficient at beam energy (cm^2/g).

    Returns
    -------
    dict with keys:
        'dose_per_scan_Gy' : float — absorbed dose in Gray
        'dose_rate_Gy_per_s' : float
        'flux_density' : float — photons/s/cm^2
    """
    beam_area = beam_size_h * beam_size_v  # cm^2
    flux_density = flux / beam_area

    # Energy per photon (approximate, assuming ~10 keV)
    # For precise calculation, pass the photon energy
    E_photon_J = 10e3 * 1.602e-19  # 10 keV in Joules

    # Absorbed power per unit mass
    power_density = flux_density * E_photon_J * mass_atten_coeff  # W/g = Gy/s

    dose_rate = power_density  # Gy/s
    dose_per_scan = dose_rate * scan_time

    return {
        'dose_per_scan_Gy': dose_per_scan,
        'dose_rate_Gy_per_s': dose_rate,
        'flux_density': flux_density,
    }
```

## Solutions and Mitigation

### Prevention (Before Data Collection)

- Use cryogenic sample cooling (liquid nitrogen or liquid helium) to slow radical diffusion and reduce thermal damage. Cryo-cooling is essential for biological samples and redox-sensitive materials.
- Reduce the photon flux using attenuators (metal foils) upstream of the sample. Accept longer scan times for a proportional reduction in dose rate.
- Defocus the beam or use a larger beam spot to distribute the dose over a larger sample area.
- Use quick-scan (continuous) mode rather than step-scan to minimize the total exposure time. QEXAFS or energy-dispersive XAS can acquire a full spectrum in seconds.
- Collect multiple short scans at different sample positions rather than repeated scans at the same spot.
- For solution samples, use a flow cell to continuously refresh the sample in the beam path.

### Correction — Traditional Methods

```python
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


def extrapolate_to_zero_dose(energy_scans, mu_scans, scan_times=None):
    """
    Extrapolate spectra to zero dose by fitting the spectral changes
    as a function of cumulative dose (approximated by scan number or time)
    and extrapolating back to t=0.

    Parameters
    ----------
    energy_scans : list of np.ndarray
        Energy arrays for each scan.
    mu_scans : list of np.ndarray
        Absorption spectra for sequential scans of the same spot.
    scan_times : list of float or None
        Cumulative exposure times. If None, uses scan index as proxy.

    Returns
    -------
    dict with keys:
        'energy' : np.ndarray — common energy grid
        'mu_zero_dose' : np.ndarray — extrapolated zero-dose spectrum
        'mu_scans_common' : np.ndarray — all scans on common grid
        'damage_rate_per_point' : np.ndarray — rate of change at each energy
    """
    n_scans = len(energy_scans)
    if scan_times is None:
        scan_times = np.arange(n_scans, dtype=float)
    scan_times = np.array(scan_times, dtype=float)

    # Common energy grid
    e_min = max(e.min() for e in energy_scans)
    e_max = min(e.max() for e in energy_scans)
    de = np.median([e[1] - e[0] for e in energy_scans])
    common_e = np.arange(e_min, e_max, de)
    n_pts = len(common_e)

    mu_common = np.zeros((n_scans, n_pts))
    for i, (energy, mu) in enumerate(zip(energy_scans, mu_scans)):
        f = interp1d(energy, mu, kind='cubic', bounds_error=False,
                     fill_value=np.nan)
        mu_common[i] = f(common_e)

    # At each energy point, fit mu vs. dose and extrapolate to dose=0
    mu_zero_dose = np.zeros(n_pts)
    damage_rate = np.zeros(n_pts)

    for j in range(n_pts):
        y = mu_common[:, j]
        valid = ~np.isnan(y)
        if np.sum(valid) < 2:
            mu_zero_dose[j] = y[valid][0] if np.any(valid) else np.nan
            continue

        # Linear fit: mu(t) = mu(0) + rate * t
        coeffs = np.polyfit(scan_times[valid], y[valid], 1)
        mu_zero_dose[j] = coeffs[1]  # intercept = mu at t=0
        damage_rate[j] = coeffs[0]   # slope = damage rate

    return {
        'energy': common_e,
        'mu_zero_dose': mu_zero_dose,
        'mu_scans_common': mu_common,
        'damage_rate_per_point': damage_rate,
    }


def select_undamaged_scans(energy_scans, mu_scans, max_e0_shift=0.2,
                            max_spectral_change=0.01):
    """
    Select only scans collected before significant radiation damage
    occurred, for clean merging.

    Parameters
    ----------
    energy_scans : list of np.ndarray
        Energy arrays for sequential scans.
    mu_scans : list of np.ndarray
        Absorption spectra.
    max_e0_shift : float
        Maximum allowed E0 shift from scan 1 (eV).
    max_spectral_change : float
        Maximum allowed RMS spectral difference from scan 1.

    Returns
    -------
    dict with keys:
        'good_scan_indices' : list of int — indices of undamaged scans
        'n_good_scans' : int
        'n_rejected' : int
        'e0_shifts' : list of float
    """
    from scipy.ndimage import uniform_filter1d

    # Reference: scan 1
    e_ref = energy_scans[0]
    mu_ref = mu_scans[0]
    dmu_ref = np.gradient(mu_ref, e_ref)
    dmu_ref_smooth = uniform_filter1d(dmu_ref, size=5)
    e0_ref = e_ref[np.argmax(dmu_ref_smooth)]

    good_indices = [0]
    e0_shifts = [0.0]

    for i in range(1, len(energy_scans)):
        energy, mu = energy_scans[i], mu_scans[i]
        dmu = np.gradient(mu, energy)
        dmu_smooth = uniform_filter1d(dmu, size=5)
        e0_i = energy[np.argmax(dmu_smooth)]
        e0_shift = abs(e0_i - e0_ref)
        e0_shifts.append(e0_i - e0_ref)

        # Interpolate to common grid for spectral comparison
        f = interp1d(energy, mu, kind='linear', bounds_error=False,
                     fill_value=np.nan)
        mu_interp = f(e_ref)
        rms_diff = np.sqrt(np.nanmean((mu_interp - mu_ref)**2))

        if e0_shift <= max_e0_shift and rms_diff <= max_spectral_change:
            good_indices.append(i)

    return {
        'good_scan_indices': good_indices,
        'n_good_scans': len(good_indices),
        'n_rejected': len(energy_scans) - len(good_indices),
        'e0_shifts': e0_shifts,
    }
```

### Correction — AI/ML Methods

Machine learning approaches to radiation damage correction use generative models trained on damage progression series. An autoencoder or variational autoencoder learns the latent representation of "undamaged" spectra from a training set of first-scan data. Given a damaged spectrum, the model projects it into the learned undamaged manifold to produce a corrected output. More sophisticated approaches use recurrent neural networks (LSTMs) that model the temporal evolution of damage across a scan series and learn to invert the damage trajectory back to t=0. Physics-constrained networks that enforce sum rules and edge-step conservation provide additional robustness.

## Impact If Uncorrected

Radiation damage introduces systematic errors that mimic real chemical changes, making it one of the most dangerous artifacts in XAS. Photoreduction shifts the apparent oxidation state — for example, Mn(IV) in battery cathodes can appear partially reduced to Mn(II), leading to incorrect conclusions about charge state. In biological systems (metalloenzymes, metalloproteins), beam damage can destroy the active site structure that the experiment was designed to probe. EXAFS fitting of damaged spectra yields incorrect bond lengths and coordination numbers that reflect the damaged structure rather than the native one. Published results have been retracted when radiation damage was later identified as the cause of anomalous spectral features.

## Related Resources

- Related artifact: [Energy Calibration Drift](energy_calibration_drift.md) — energy drift and radiation damage both shift E0 but from different causes
- Related artifact: [Outlier Spectra](outlier_spectra.md) — early or late scans in a damage series may appear as outliers

## Real-World Before/After Examples

The following published sources provide real experimental before/after comparisons:

| Source | Type | Figure | Description | License |
|--------|------|--------|-------------|---------|
| [Holton 2009](https://doi.org/10.1016/j.jsb.2009.02.007) | Paper | Multiple | A beginner's guide to radiation damage — comprehensive overview with real experimental examples of progressive damage | -- |
| [Henderson 1990](https://doi.org/10.1098/rspb.1990.0057) | Paper | Multiple | Cryo-protection of protein crystals against radiation damage in electron and X-ray diffraction — foundational reference | -- |
| [Owen et al. 2006](https://doi.org/10.1073/pnas.0600973103) | Paper | Multiple | Experimental determination of the radiation dose limit for cryocooled protein crystals — dose-dependent damage examples | -- |

**Key references with published before/after comparisons:**
- **Holton (2009)**: Comprehensive overview with real experimental examples of progressive radiation damage in XAS. DOI: 10.1016/j.jsb.2009.02.007
- **Henderson (1990)**: Foundational reference on cryo-protection against radiation damage. DOI: 10.1098/rspb.1990.0057
- **Owen et al. (2006)**: Experimental determination of radiation dose limits with dose-dependent damage examples. DOI: 10.1073/pnas.0600973103

> **Recommended reference**: [Holton 2009 — A beginner's guide to radiation damage (J. Struct. Biol.)](https://doi.org/10.1016/j.jsb.2009.02.007)

## Key Takeaway

Radiation damage is a critical systematic artifact that can invalidate XAS results entirely. Always compare successive scans for progressive spectral changes before merging. Use cryogenic cooling, attenuators, and defocused beams as standard practice for radiation-sensitive samples. If damage is detected, either extrapolate to zero dose or use only the undamaged scans. A quick pre-experiment dose test (a few rapid scans at one spot) should be standard protocol for any new sample type.
