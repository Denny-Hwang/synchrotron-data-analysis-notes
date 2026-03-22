# Outlier Spectra

## Classification

| Attribute | Value |
|-----------|-------|
| **Modality** | Spectroscopy |
| **Noise Type** | Statistical |
| **Severity** | Minor |
| **Frequency** | Occasional |
| **Detection Difficulty** | Moderate |

## Visual Examples

```
Overlay of 10 scans with one outlier:

  mu(E)
   ^
   |      /|||||  9 scans overlap
   |     /|||||
   |    / |||||         /\  <-- Outlier scan
   |   /  |||||\___    / |      (shifted, distorted)
   |  /   |||||     \ /  \___
   | /    |||||      X
   |/     |||||     / \
   +----+----+----+----+-----> E(eV)

Monochromator glitch in single scan:

  mu(E)
   ^
   |         _____|_____
   |        /     |     \      Sharp spike from
   |       /      |      \     mono glitch
   |      /       |
   |     /     Glitch
   |____/
   +----+----+----+----+-----> E(eV)

Beam dump during scan:

  mu(E)
   ^
   |         _____
   |        /     \
   |       /       \____
   |      /              |     Abrupt discontinuity
   |     /               |     from beam loss
   |____/                |____
   +----+----+----+----+-----> E(eV)

Chi-squared per scan for outlier identification:

  chi^2
   ^
   |                     x     <-- Outlier (scan 8)
   |
   |                  Threshold = median + 3*MAD
   | - - - - - - - - - - - - - - - - - - - -
   |  x  x  x x  x  x x    x x
   +--+--+--+--+--+--+--+--+--+---> Scan number
   1  2  3  4  5  6  7  8  9  10
```

## Description

Outlier spectra are individual scans within a multi-scan series that deviate significantly from the ensemble average due to transient instrumental or beam problems. Common causes include monochromator glitches (sharp spikes from parasitic Bragg reflections), beam dumps or injection transients (sudden intensity changes), sample motion or instability, and electronic noise bursts in the detector system. Unlike systematic artifacts such as radiation damage or energy drift, outlier spectra typically affect only one or a few scans and exhibit abrupt, non-physical features rather than gradual trends. Including outlier scans in a merge degrades the data quality by introducing artifacts into the averaged spectrum and inflating the noise level.

## Root Cause

Multiple transient phenomena can produce outlier scans: (1) Monochromator glitches occur at specific energies where a parasitic crystal plane satisfies the Bragg condition simultaneously with the primary reflection, causing a sharp intensity spike or dip. These are reproducible in energy but may affect only some scans if the beam position drifts. (2) Beam dumps or top-up injection events cause sudden changes in the ring current and thus the X-ray intensity, producing discontinuities in the I0 and I1 signals. (3) Sample instability — bubbling in liquid cells, particle movement in loose powders, or ice crystal growth in cryo-cooled samples — changes the effective absorption path length during a scan. (4) Electronic glitches in ion chambers, amplifiers, or digitizers produce random spikes. (5) Detector saturation or dead-time miscorrection in fluorescence detectors can distort individual scans if the count rate fluctuates.

## Quick Diagnosis

```python
import numpy as np

# Given: multiple scans on a common energy grid
# energy: common energy array
# mu_scans: np.ndarray of shape (n_scans, n_energy)

# Compute chi-squared per scan relative to the median spectrum
mu_median = np.median(mu_scans, axis=0)
mu_mad = np.median(np.abs(mu_scans - mu_median[np.newaxis, :]), axis=0)
mu_mad = np.maximum(mu_mad, 1e-10)  # avoid division by zero

chi2_per_scan = np.mean(((mu_scans - mu_median) / (1.4826 * mu_mad))**2,
                         axis=1)

threshold = np.median(chi2_per_scan) + 3 * 1.4826 * np.median(
    np.abs(chi2_per_scan - np.median(chi2_per_scan)))

for i, c2 in enumerate(chi2_per_scan):
    flag = " <-- OUTLIER" if c2 > threshold else ""
    print(f"  Scan {i+1}: chi^2 = {c2:.2f}{flag}")
```

## Detection Methods

### Visual Indicators

- One or more scans visually stand out when all scans are overlaid — shifted edge position, anomalous features, or discontinuities.
- Sharp spikes or dips at specific energies in a single scan that do not appear in other scans (monochromator glitches).
- Abrupt step-like discontinuities in mu(E) within a scan (beam dump or sample jump).
- The merged spectrum has features that are not present in the majority of individual scans.
- The standard deviation across scans at each energy point shows localized peaks at energies where an outlier scan deviates.

### Automated Detection

```python
import numpy as np
from scipy.interpolate import interp1d


def detect_outlier_scans(energy, mu_scans, method='chi_squared',
                          threshold_sigma=3.0):
    """
    Identify outlier scans in a multi-scan XAS dataset.

    Parameters
    ----------
    energy : np.ndarray
        Common energy grid (eV).
    mu_scans : np.ndarray
        Shape (n_scans, n_energy) — absorption spectra.
    method : str
        Detection method:
        - 'chi_squared' : per-scan chi-squared vs. median
        - 'pca' : PCA-based outlier detection
        - 'correlation' : pairwise correlation matrix analysis
    threshold_sigma : float
        Number of robust standard deviations for outlier threshold.

    Returns
    -------
    dict with keys:
        'outlier_indices' : list of int — indices of outlier scans
        'scores' : np.ndarray — anomaly score per scan
        'threshold' : float — score threshold used
        'n_outliers' : int
    """
    n_scans, n_energy = mu_scans.shape

    if method == 'chi_squared':
        mu_median = np.median(mu_scans, axis=0)
        mu_mad = np.median(np.abs(mu_scans - mu_median[np.newaxis, :]),
                           axis=0)
        mu_robust_std = 1.4826 * np.maximum(mu_mad, 1e-10)

        scores = np.mean(
            ((mu_scans - mu_median) / mu_robust_std)**2,
            axis=1
        )

    elif method == 'pca':
        # PCA-based: outliers have large residuals from low-rank model
        from numpy.linalg import svd

        # Center the data
        mu_mean = np.mean(mu_scans, axis=0)
        mu_centered = mu_scans - mu_mean[np.newaxis, :]

        U, S, Vt = svd(mu_centered, full_matrices=False)

        # Keep components explaining 95% of variance
        variance_explained = np.cumsum(S**2) / np.sum(S**2)
        n_components = np.searchsorted(variance_explained, 0.95) + 1
        n_components = max(1, min(n_components, n_scans - 1))

        # Reconstruct using top components
        mu_reconstructed = U[:, :n_components] @ np.diag(
            S[:n_components]) @ Vt[:n_components, :]
        residuals = mu_centered - mu_reconstructed

        # Score: RMS residual per scan
        scores = np.sqrt(np.mean(residuals**2, axis=1))

    elif method == 'correlation':
        # Correlation-based: outliers have low average correlation
        corr_matrix = np.corrcoef(mu_scans)
        # Average correlation of each scan with all others
        avg_corr = (np.sum(corr_matrix, axis=1) - 1) / (n_scans - 1)
        # Convert to anomaly score (lower correlation = higher score)
        scores = 1.0 - avg_corr

    else:
        raise ValueError(f"Unknown method: {method}")

    # Robust threshold
    med_score = np.median(scores)
    mad_score = np.median(np.abs(scores - med_score))
    threshold = med_score + threshold_sigma * 1.4826 * mad_score

    outlier_indices = np.where(scores > threshold)[0].tolist()

    return {
        'outlier_indices': outlier_indices,
        'scores': scores,
        'threshold': threshold,
        'n_outliers': len(outlier_indices),
    }


def detect_glitches_in_scan(energy, mu, sigma_threshold=5.0):
    """
    Detect monochromator glitches (sharp spikes) within a single scan.

    Parameters
    ----------
    energy : np.ndarray
        Energy array (eV).
    mu : np.ndarray
        Absorption spectrum.
    sigma_threshold : float
        Threshold in robust standard deviations for spike detection.

    Returns
    -------
    dict with keys:
        'glitch_indices' : np.ndarray — indices of glitch points
        'glitch_energies' : np.ndarray — energies of glitches
        'n_glitches' : int
        'glitch_amplitudes' : np.ndarray — deviation at each glitch
    """
    # Use second derivative to detect sharp features
    d2mu = np.gradient(np.gradient(mu, energy), energy)

    med = np.median(d2mu)
    mad = np.median(np.abs(d2mu - med))
    robust_std = 1.4826 * mad if mad > 0 else np.std(d2mu)

    deviation = np.abs(d2mu - med) / robust_std
    glitch_mask = deviation > sigma_threshold

    # Cluster adjacent glitch points (a single glitch may span 2-3 points)
    glitch_indices = np.where(glitch_mask)[0]
    if len(glitch_indices) > 0:
        # Merge adjacent indices (within 3 points)
        clusters = []
        current = [glitch_indices[0]]
        for idx in glitch_indices[1:]:
            if idx - current[-1] <= 3:
                current.append(idx)
            else:
                clusters.append(current)
                current = [idx]
        clusters.append(current)

        # Take the peak of each cluster
        peak_indices = [c[np.argmax(deviation[c])] for c in clusters]
        peak_indices = np.array(peak_indices)
    else:
        peak_indices = np.array([], dtype=int)

    return {
        'glitch_indices': peak_indices,
        'glitch_energies': energy[peak_indices] if len(peak_indices) > 0
                          else np.array([]),
        'n_glitches': len(peak_indices),
        'glitch_amplitudes': deviation[peak_indices] if len(peak_indices) > 0
                            else np.array([]),
    }


def detect_beam_dumps(energy, i0, threshold_fraction=0.5):
    """
    Detect beam dump or injection events from I0 (incident intensity)
    signal discontinuities.

    Parameters
    ----------
    energy : np.ndarray
        Energy array (eV).
    i0 : np.ndarray
        Incident beam intensity signal.
    threshold_fraction : float
        Fractional drop in I0 that constitutes a beam event.

    Returns
    -------
    dict with keys:
        'event_indices' : np.ndarray — indices of beam events
        'event_energies' : np.ndarray — energies of events
        'n_events' : int
        'i0_drops' : np.ndarray — fractional I0 change at each event
    """
    # Compute fractional change in I0
    di0 = np.diff(i0) / np.maximum(np.abs(i0[:-1]), 1e-10)

    event_mask = np.abs(di0) > threshold_fraction
    event_indices = np.where(event_mask)[0]

    return {
        'event_indices': event_indices,
        'event_energies': energy[event_indices] if len(event_indices) > 0
                         else np.array([]),
        'n_events': len(event_indices),
        'i0_drops': di0[event_indices] if len(event_indices) > 0
                   else np.array([]),
    }
```

## Solutions and Mitigation

### Prevention (Before Data Collection)

- Monitor the ring current and I0 signal in real time during data collection; pause acquisition during top-up injections if they cause beam instabilities.
- Characterize monochromator glitch energies beforehand using an empty-beam scan and record them for later flagging.
- Secure the sample firmly to prevent mechanical instability; use Kapton tape, capillary holders, or cryo-stream stabilization.
- Set up automated scan rejection at the beamline software level to flag scans where I0 drops below a threshold during acquisition.
- For liquid samples in capillaries, degas the solution to prevent bubble formation under the beam.

### Correction — Traditional Methods

```python
import numpy as np


def reject_outlier_scans(energy, mu_scans, method='chi_squared',
                          threshold_sigma=3.0):
    """
    Remove outlier scans and merge the remaining clean scans.

    Parameters
    ----------
    energy : np.ndarray
        Common energy grid (eV).
    mu_scans : np.ndarray
        Shape (n_scans, n_energy) — all scans.
    method : str
        Outlier detection method ('chi_squared', 'pca', 'correlation').
    threshold_sigma : float
        Outlier threshold in robust sigma.

    Returns
    -------
    dict with keys:
        'energy' : np.ndarray
        'mu_merged' : np.ndarray — merged clean spectrum
        'mu_std' : np.ndarray — standard deviation of clean scans
        'n_kept' : int — number of scans kept
        'rejected_indices' : list of int — indices of rejected scans
    """
    result = detect_outlier_scans(energy, mu_scans, method=method,
                                  threshold_sigma=threshold_sigma)
    outlier_set = set(result['outlier_indices'])
    good_indices = [i for i in range(mu_scans.shape[0])
                    if i not in outlier_set]

    if len(good_indices) == 0:
        # Fallback: keep all scans
        good_indices = list(range(mu_scans.shape[0]))

    mu_good = mu_scans[good_indices]
    mu_merged = np.mean(mu_good, axis=0)
    mu_std = np.std(mu_good, axis=0)

    return {
        'energy': energy,
        'mu_merged': mu_merged,
        'mu_std': mu_std,
        'n_kept': len(good_indices),
        'rejected_indices': list(outlier_set),
    }


def repair_glitches(energy, mu, glitch_indices, window=5):
    """
    Repair monochromator glitches by interpolating over affected points.

    Parameters
    ----------
    energy : np.ndarray
        Energy array (eV).
    mu : np.ndarray
        Absorption spectrum with glitches.
    glitch_indices : np.ndarray or list
        Indices of glitch points to repair.
    window : int
        Number of clean points on each side to use for interpolation.

    Returns
    -------
    np.ndarray — repaired spectrum
    """
    mu_fixed = mu.copy()

    for idx in glitch_indices:
        # Define interpolation region excluding the glitch
        lo = max(0, idx - window)
        hi = min(len(energy), idx + window + 1)
        mask = np.ones(hi - lo, dtype=bool)

        # Mark glitch point(s) and neighbors within 2 points
        for gi in glitch_indices:
            local_gi = gi - lo
            if 0 <= local_gi < len(mask):
                mask[local_gi] = False
                if local_gi > 0:
                    mask[local_gi - 1] = False
                if local_gi < len(mask) - 1:
                    mask[local_gi + 1] = False

        e_local = energy[lo:hi]
        mu_local = mu[lo:hi]

        if np.sum(mask) >= 2:
            coeffs = np.polyfit(e_local[mask], mu_local[mask], 2)
            mu_fixed[idx] = np.polyval(coeffs, energy[idx])

    return mu_fixed


def iterative_sigma_clip_merge(energy, mu_scans, sigma=3.0, max_iter=5):
    """
    Merge spectra with iterative per-point sigma clipping.

    At each energy point, reject values more than sigma*std from the
    mean, recompute, and iterate until convergence.

    Parameters
    ----------
    energy : np.ndarray
        Common energy grid (eV).
    mu_scans : np.ndarray
        Shape (n_scans, n_energy).
    sigma : float
        Clipping threshold in standard deviations.
    max_iter : int
        Maximum iterations for clipping convergence.

    Returns
    -------
    dict with keys:
        'energy' : np.ndarray
        'mu_merged' : np.ndarray
        'mu_std' : np.ndarray
        'n_used' : np.ndarray — number of scans used at each point
    """
    n_scans, n_energy = mu_scans.shape
    mu_merged = np.zeros(n_energy)
    mu_std = np.zeros(n_energy)
    n_used = np.zeros(n_energy, dtype=int)

    for j in range(n_energy):
        values = mu_scans[:, j].copy()
        mask = np.ones(n_scans, dtype=bool)

        for iteration in range(max_iter):
            mean_val = np.mean(values[mask])
            std_val = np.std(values[mask])
            if std_val == 0:
                break
            new_mask = np.abs(values - mean_val) <= sigma * std_val
            new_mask |= ~mask  # keep previously rejected as rejected
            new_mask = np.abs(values - mean_val) <= sigma * std_val
            if np.sum(new_mask) < 2:
                break  # don't reject too many
            if np.array_equal(mask, new_mask):
                break
            mask = new_mask

        mu_merged[j] = np.mean(values[mask])
        mu_std[j] = np.std(values[mask])
        n_used[j] = np.sum(mask)

    return {
        'energy': energy,
        'mu_merged': mu_merged,
        'mu_std': mu_std,
        'n_used': n_used,
    }
```

### Correction — AI/ML Methods

Autoencoder-based anomaly detection trains a neural network on the distribution of "normal" XAS scans. The autoencoder learns to reconstruct typical spectra with low error; outlier scans produce high reconstruction error and are automatically flagged. This approach is advantageous because it does not require explicit modeling of every possible failure mode — the network implicitly learns the manifold of valid spectra. For glitch repair, a masked autoencoder or inpainting network can reconstruct the corrupted energy region from the surrounding spectral context, similar to image inpainting but in 1D. Isolation forests and one-class SVMs operating on spectral features (PCA components, wavelet coefficients) provide interpretable anomaly scores.

## Impact If Uncorrected

Including outlier scans in a merge introduces non-physical features into the averaged spectrum. Monochromator glitches appear as spurious peaks or dips that can be mistaken for real spectral features (e.g., pre-edge transitions or EXAFS beat nodes). Beam dump artifacts cause discontinuities that corrupt the background subtraction and normalization. Even a single bad scan in a set of 10 increases the noise level by up to a factor of 3 at the affected energy points and can shift the apparent peak positions. In EXAFS, a glitch in one scan introduces a spurious oscillation component that produces ghost peaks in the Fourier transform at incorrect distances.

## Related Resources

- [Spectroscopy EDA notebook](../../06_data_structures/eda/spectroscopy_eda.md) — scan-by-scan quality inspection
- Related artifact: [Statistical Noise in EXAFS](statistical_noise_exafs.md) — outlier rejection improves merged SNR
- Related artifact: [Energy Calibration Drift](energy_calibration_drift.md) — drifted scans may be flagged as outliers
- Related artifact: [Radiation Damage](radiation_damage.md) — progressively damaged scans are systematic, not random outliers

## Key Takeaway

Always inspect individual scans visually and with automated metrics before merging. A single outlier scan can corrupt the merged spectrum far more than random noise. Use chi-squared per scan and PCA-based anomaly detection as standard quality control, and employ sigma-clipping during the merge to provide point-level robustness against isolated glitch points that survive scan-level rejection.
