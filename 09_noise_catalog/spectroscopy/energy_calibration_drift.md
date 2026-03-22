# Energy Calibration Drift

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
Edge position shift between successive scans:

  mu(E)                              mu(E)
   ^                                  ^
   |        ___________               |      /--- Scan 1
   |       /                          |     /  /-- Scan 2
   |      /                           |    / /  /- Scan 3
   |     /                            |   //  /
   |    /                             |  /  /     Edge shifts right
   |___/                              |_/ /       with each scan
   +-----+-----+-----+-> E(eV)       +--+--+--+--+-> E(eV)
    7100  7110  7120  7130             7110 7115 7120 7125
    Stable calibration                 Drifting calibration

Merged spectrum broadening:

  mu(E)                     mu(E)
   ^                         ^
   |     /|                  |     /\
   |    / |                  |    /  |      White line peak
   |   /  |   Sharp peak     |   /   \     is broadened and
   |  /   |   (aligned)      |  /     \    reduced (misaligned)
   | /    |                  | /       \
   |/     \_____             |/         \_____
   +---+---+---+-> E        +---+---+---+---+-> E

  Derivative d(mu)/dE:

   ^          |              ^         |\ /|
   |         _|_             |        _| X |_
   |        | | |            |       | |/ \| |   Split or
   |       _| | |_           |      _|      |_   broadened
   |______/  | |  \____      |_____/  |    |  \__derivative peak
   +---+---+-+-+---+-> E    +---+---+-+--+-+---+-> E
            E0                      E0 range
   Aligned scans                Drifted scans
```

## Description

Energy calibration drift refers to a systematic shift in the monochromator energy scale over time during an XAS experiment. The nominal energy assigned to each data point gradually deviates from the true photon energy, causing the absorption edge position (E0) to appear at different energies in successive scans. When multiple scans are merged without correcting for this drift, absorption features — especially sharp near-edge structures like the white line — become broadened and reduced in amplitude. The drift is typically monotonic (increasing or decreasing) over a session but can also exhibit non-linear behavior correlated with beam conditions.

## Root Cause

The primary cause is thermal loading of the monochromator crystal. As the synchrotron beam heats the first crystal of the double-crystal monochromator (DCM), the crystal lattice expands, altering the d-spacing and shifting the Bragg condition to a slightly different angle-energy relationship. This effect is most pronounced at high flux beamlines and during the first minutes after beam delivery begins or after a top-up injection. Additional contributions come from: encoder drift in the monochromator angle readout; mechanical backlash or hysteresis in the goniometer; and thermal expansion of the monochromator mechanical components that changes the geometry between the two crystals. Cryogenically cooled monochromators (liquid nitrogen) largely mitigate thermal drift but do not eliminate it entirely, especially during fill pattern changes.

## Quick Diagnosis

```python
import numpy as np

# Given: list of energy arrays and mu arrays from multiple scans
# energy_scans: list of np.ndarray (energy in eV)
# mu_scans: list of np.ndarray (absorption coefficient)

def find_e0(energy, mu):
    """Find E0 as energy of maximum derivative."""
    dmu = np.gradient(mu, energy)
    return energy[np.argmax(dmu)]

e0_values = [find_e0(e, m) for e, m in zip(energy_scans, mu_scans)]
e0_drift = np.max(e0_values) - np.min(e0_values)
print(f"E0 values per scan: {[f'{e:.2f}' for e in e0_values]}")
print(f"Total E0 drift: {e0_drift:.2f} eV")
if e0_drift > 0.3:
    print("WARNING: Significant energy drift detected — align before merging")
```

## Detection Methods

### Visual Indicators

- Absorption edge position (inflection point) shifts systematically across sequential scans when overlaid.
- Pre-edge and white-line features appear at slightly different energies in individual scans.
- Merged spectra show broader, shorter peaks compared to any individual scan.
- The derivative spectrum d(mu)/dE of the merged data shows a split or broadened peak at E0 instead of a single sharp peak.
- Reference foil spectrum (if collected simultaneously) shows corresponding shifts in its edge position.

### Automated Detection

```python
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import correlate


def detect_energy_drift(energy_scans, mu_scans, reference_energy=None,
                        drift_threshold=0.3):
    """
    Detect energy calibration drift across multiple XAS scans.

    Parameters
    ----------
    energy_scans : list of np.ndarray
        Energy arrays for each scan (eV).
    mu_scans : list of np.ndarray
        Absorption coefficient arrays for each scan.
    reference_energy : float or None
        Known E0 for the element/edge. If None, uses the first scan as
        reference.
    drift_threshold : float
        Energy shift (eV) above which drift is flagged.

    Returns
    -------
    dict with keys:
        'e0_per_scan' : list of float — E0 for each scan
        'shifts' : list of float — shift relative to first scan (eV)
        'total_drift' : float — max minus min E0 (eV)
        'drift_rate' : float — linear drift rate (eV/scan)
        'has_significant_drift' : bool
        'drift_direction' : str — 'increasing', 'decreasing', or 'non-monotonic'
    """
    n_scans = len(energy_scans)

    # Find E0 for each scan using derivative method
    e0_values = []
    for energy, mu in zip(energy_scans, mu_scans):
        dmu = np.gradient(mu, energy)
        # Smooth derivative to reduce noise effects
        from scipy.ndimage import uniform_filter1d
        dmu_smooth = uniform_filter1d(dmu, size=5)
        e0 = energy[np.argmax(dmu_smooth)]
        e0_values.append(e0)

    e0_values = np.array(e0_values)
    shifts = e0_values - e0_values[0]
    total_drift = np.max(e0_values) - np.min(e0_values)

    # Linear drift rate via simple regression
    scan_indices = np.arange(n_scans)
    if n_scans > 1:
        drift_rate = np.polyfit(scan_indices, e0_values, 1)[0]
    else:
        drift_rate = 0.0

    # Check monotonicity
    diffs = np.diff(e0_values)
    if np.all(diffs >= -0.05):
        direction = 'increasing'
    elif np.all(diffs <= 0.05):
        direction = 'decreasing'
    else:
        direction = 'non-monotonic'

    return {
        'e0_per_scan': e0_values.tolist(),
        'shifts': shifts.tolist(),
        'total_drift': total_drift,
        'drift_rate': drift_rate,
        'has_significant_drift': total_drift > drift_threshold,
        'drift_direction': direction,
    }


def cross_correlation_shift(energy, mu_ref, mu_target, edge_region_width=50):
    """
    Measure energy shift between two spectra using cross-correlation
    of the derivative spectra in the edge region.

    Parameters
    ----------
    energy : np.ndarray
        Common energy grid (eV).
    mu_ref : np.ndarray
        Reference absorption spectrum.
    mu_target : np.ndarray
        Target spectrum to measure shift for.
    edge_region_width : float
        Width (eV) around E0 to use for correlation.

    Returns
    -------
    float — energy shift in eV (positive means target shifted to higher E)
    """
    dmu_ref = np.gradient(mu_ref, energy)
    dmu_target = np.gradient(mu_target, energy)

    # Find edge region
    e0_idx = np.argmax(dmu_ref)
    de = energy[1] - energy[0]
    half_width = int(edge_region_width / (2 * de))
    lo = max(0, e0_idx - half_width)
    hi = min(len(energy), e0_idx + half_width)

    seg_ref = dmu_ref[lo:hi]
    seg_target = dmu_target[lo:hi]

    # Normalize
    seg_ref = (seg_ref - np.mean(seg_ref)) / (np.std(seg_ref) + 1e-10)
    seg_target = (seg_target - np.mean(seg_target)) / (np.std(seg_target) + 1e-10)

    # Cross-correlate
    cc = correlate(seg_target, seg_ref, mode='full')
    lag_indices = np.arange(-len(seg_ref) + 1, len(seg_ref))
    best_lag = lag_indices[np.argmax(cc)]

    return best_lag * de
```

## Solutions and Mitigation

### Prevention (Before Data Collection)

- Always collect a simultaneous reference foil spectrum using a third ion chamber (I2) downstream of the sample. The foil provides an absolute energy reference for every scan.
- Allow the monochromator to thermally equilibrate after beam delivery begins — collect and discard 1-2 "warm-up" scans before starting the experiment.
- Use a cryogenically cooled monochromator (LN2-cooled Si crystal) to minimize thermal lattice expansion.
- Monitor the I0 signal for step changes that correlate with ring current top-up events.
- Use encoder feedback on the monochromator rather than stepper motor steps.

### Correction — Traditional Methods

```python
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d


def align_scans_to_reference(energy_scans, mu_scans, ref_e0=None,
                              method='derivative'):
    """
    Align multiple XAS scans by shifting each energy scale so that E0
    matches a common reference value.

    Parameters
    ----------
    energy_scans : list of np.ndarray
        Energy arrays for each scan (eV).
    mu_scans : list of np.ndarray
        Absorption coefficient arrays for each scan.
    ref_e0 : float or None
        Target E0 value. If None, uses the median E0 of all scans.
    method : str
        'derivative' — align on max of d(mu)/dE
        'half_edge' — align on energy at 50% of edge step

    Returns
    -------
    dict with keys:
        'energy_aligned' : list of np.ndarray — shifted energy arrays
        'mu_aligned' : list of np.ndarray — corresponding mu arrays
        'shifts_applied' : list of float — energy shift applied to each scan
        'common_energy' : np.ndarray — common interpolated grid
        'mu_on_common' : np.ndarray — shape (n_scans, n_pts) on common grid
    """
    n_scans = len(energy_scans)
    e0_values = []

    for energy, mu in zip(energy_scans, mu_scans):
        if method == 'derivative':
            dmu = np.gradient(mu, energy)
            dmu_smooth = uniform_filter1d(dmu, size=5)
            e0 = energy[np.argmax(dmu_smooth)]
        elif method == 'half_edge':
            # Estimate pre-edge and post-edge levels
            pre = np.mean(mu[:20])
            post = np.mean(mu[-20:])
            half = (pre + post) / 2
            idx = np.argmin(np.abs(mu - half))
            e0 = energy[idx]
        else:
            raise ValueError(f"Unknown method: {method}")
        e0_values.append(e0)

    if ref_e0 is None:
        ref_e0 = np.median(e0_values)

    shifts = [ref_e0 - e0 for e0 in e0_values]
    energy_aligned = [e + shift for e, shift in zip(energy_scans, shifts)]
    mu_aligned = list(mu_scans)

    # Interpolate onto common grid
    e_min = max(e.min() for e in energy_aligned)
    e_max = min(e.max() for e in energy_aligned)
    de = np.median([e[1] - e[0] for e in energy_aligned])
    common_energy = np.arange(e_min, e_max, de)

    mu_on_common = np.zeros((n_scans, len(common_energy)))
    for i, (e, m) in enumerate(zip(energy_aligned, mu_aligned)):
        f = interp1d(e, m, kind='cubic', bounds_error=False, fill_value=np.nan)
        mu_on_common[i, :] = f(common_energy)

    return {
        'energy_aligned': energy_aligned,
        'mu_aligned': mu_aligned,
        'shifts_applied': shifts,
        'common_energy': common_energy,
        'mu_on_common': mu_on_common,
    }


def align_using_reference_foil(energy_scans, mu_scans, ref_foil_scans,
                                known_e0, foil_energy_scans=None):
    """
    Align scans using a simultaneously measured reference foil.

    Parameters
    ----------
    energy_scans : list of np.ndarray
        Energy arrays for sample scans.
    mu_scans : list of np.ndarray
        Sample absorption spectra.
    ref_foil_scans : list of np.ndarray
        Reference foil mu(E) measured simultaneously (same energy grid).
    known_e0 : float
        Tabulated E0 for the reference foil (eV).
    foil_energy_scans : list of np.ndarray or None
        If None, assumes same energy grid as sample scans.

    Returns
    -------
    dict — same structure as align_scans_to_reference
    """
    if foil_energy_scans is None:
        foil_energy_scans = energy_scans

    # Determine shift from foil edge position
    shifts = []
    for e_foil, mu_foil in zip(foil_energy_scans, ref_foil_scans):
        dmu = np.gradient(mu_foil, e_foil)
        dmu_smooth = uniform_filter1d(dmu, size=5)
        foil_e0 = e_foil[np.argmax(dmu_smooth)]
        shifts.append(known_e0 - foil_e0)

    energy_aligned = [e + s for e, s in zip(energy_scans, shifts)]

    # Common grid
    e_min = max(e.min() for e in energy_aligned)
    e_max = min(e.max() for e in energy_aligned)
    de = np.median([e[1] - e[0] for e in energy_aligned])
    common_energy = np.arange(e_min, e_max, de)

    mu_on_common = np.zeros((len(mu_scans), len(common_energy)))
    for i, (e, m) in enumerate(zip(energy_aligned, mu_scans)):
        f = interp1d(e, m, kind='cubic', bounds_error=False, fill_value=np.nan)
        mu_on_common[i, :] = f(common_energy)

    return {
        'energy_aligned': energy_aligned,
        'mu_aligned': mu_scans,
        'shifts_applied': shifts,
        'common_energy': common_energy,
        'mu_on_common': mu_on_common,
    }
```

### Correction — AI/ML Methods

Neural network approaches for energy calibration use a learned spectral fingerprint model. A convolutional network is trained on a library of spectra at known energies to predict the absolute energy offset from spectral features alone. This is especially useful when no reference foil was collected: the network learns the relationship between spectral shape and edge position from training data on standards. Gaussian process regression has also been applied to model the drift trajectory as a smooth function of time or scan number, enabling interpolation-based correction for scans where no reference data is available.

## Impact If Uncorrected

Energy drift introduces systematic errors in every downstream analysis step. Merged spectra have artificially broadened features, reducing the apparent intensity of white-line peaks by 10-50% depending on drift magnitude. XANES fingerprinting and linear combination fitting will produce incorrect phase fractions because the broadened reference no longer matches the standards. In EXAFS, energy misalignment translates to errors in the k-scale origin, corrupting the extracted bond lengths — a 1 eV error at E0 translates to approximately 0.01-0.02 Angstrom error in nearest-neighbor distances. Edge position measurements used for oxidation state determination can be shifted by the full drift range, potentially leading to incorrect redox state assignments.

## Related Resources

- [Spectroscopy EDA notebook](../../06_data_structures/eda/spectroscopy_eda.md) — energy calibration checks and scan alignment
- Related artifact: [Statistical Noise in EXAFS](statistical_noise_exafs.md) — noise assessment should be done after alignment
- Related artifact: [Outlier Spectra](outlier_spectra.md) — energy-shifted scans may appear as outliers

## Key Takeaway

Always collect a simultaneous reference foil spectrum and align the E0 of every scan before merging. Even sub-eV drifts can significantly broaden near-edge features and corrupt EXAFS bond lengths. Alignment is a mandatory preprocessing step — never assume the monochromator energy scale is stable across a measurement session.
