# Spectroscopy Data Exploratory Data Analysis

## Overview

X-ray Absorption Spectroscopy (XAS) encompasses XANES (X-ray Absorption Near Edge
Structure) and EXAFS (Extended X-ray Absorption Fine Structure), both of which measure
the absorption coefficient as a function of incident photon energy across an elemental
absorption edge. EDA for spectroscopy data focuses on verifying edge alignment, assessing
noise levels, checking normalization quality, and identifying spectral outliers.

At APS BER beamlines, spectroscopy data may be collected as:
- **Point spectra** -- Single-location energy scans (1D: energy vs. absorption)
- **Spectral imaging** -- XANES stacks where a full image is collected at each energy
  point (3D: energy x rows x cols)
- **Quick-EXAFS** -- Rapid continuous-scan spectra for time-resolved studies

## EDA Checklist for Spectroscopy

### 1. Data Completeness

- [ ] All energy points are present (typically 200--1000 for XANES, 500--2000 for EXAFS)
- [ ] Energy array is monotonically increasing
- [ ] Energy range spans at least 50 eV below and 150 eV above the edge (XANES)
- [ ] For EXAFS, energy extends to at least k = 12 Angstrom^-1 above the edge
- [ ] Reference standard spectra are included

### 2. Edge Detection and Alignment

- [ ] Absorption edge is at the expected energy (within 1--2 eV)
- [ ] Edge jump is clearly visible above noise
- [ ] No energy calibration drift between scans
- [ ] Reference foil edge position is stable across measurements

### 3. Normalization Quality

- [ ] Pre-edge region is flat (no residual slope)
- [ ] Post-edge region reaches ~1.0 after normalization
- [ ] Edge jump magnitude is consistent with expected sample thickness/concentration
- [ ] No over-absorption (self-absorption) distortion

### 4. Noise Assessment

- [ ] Noise level in pre-edge and post-edge regions is quantified
- [ ] SNR is sufficient for intended analysis (linear combination fitting, PCA, etc.)
- [ ] No systematic oscillations in pre-edge (possible harmonics contamination)

## Edge Detection and Verification

```python
import numpy as np
import matplotlib.pyplot as plt

def find_edge_energy(energy, mu):
    """Find absorption edge energy as the maximum of the first derivative."""
    dmu = np.gradient(mu, energy)
    edge_idx = np.argmax(dmu)
    return energy[edge_idx], edge_idx

# Load spectroscopy data
# Assuming energy (1D) and mu (1D for point spectra, or 3D for imaging)
energy = np.load("energy_axis.npy")         # [nenergy] in eV
mu_spectra = np.load("mu_spectra.npy")       # [nspectra, nenergy]

# Check all spectra for edge position
edge_energies = []
for i in range(mu_spectra.shape[0]):
    e_edge, _ = find_edge_energy(energy, mu_spectra[i])
    edge_energies.append(e_edge)

edge_energies = np.array(edge_energies)
print(f"Edge energy: {np.mean(edge_energies):.2f} +/- {np.std(edge_energies):.2f} eV")
print(f"Range: {edge_energies.min():.2f} to {edge_energies.max():.2f} eV")

# Flag spectra with shifted edges
nominal_edge = np.median(edge_energies)
shifted = np.abs(edge_energies - nominal_edge) > 2.0  # >2 eV shift
if shifted.any():
    print(f"WARNING: {shifted.sum()} spectra have edge shifts > 2 eV")
```

## Normalization Check

Proper normalization is essential for quantitative analysis. The Athena/Larch standard
procedure fits pre-edge and post-edge lines:

```python
def normalize_xas(energy, mu, pre_range=(-150, -30), post_range=(50, 300),
                  e0=None):
    """Simple XAS normalization."""
    if e0 is None:
        e0, _ = find_edge_energy(energy, mu)

    # Relative energy
    e_rel = energy - e0

    # Pre-edge fit (linear)
    pre_mask = (e_rel >= pre_range[0]) & (e_rel <= pre_range[1])
    pre_coeffs = np.polyfit(energy[pre_mask], mu[pre_mask], 1)
    pre_line = np.polyval(pre_coeffs, energy)

    # Post-edge fit (linear)
    post_mask = (e_rel >= post_range[0]) & (e_rel <= post_range[1])
    post_coeffs = np.polyfit(energy[post_mask], mu[post_mask], 1)
    post_line = np.polyval(post_coeffs, energy)

    # Normalize
    edge_step = np.polyval(post_coeffs, e0) - np.polyval(pre_coeffs, e0)
    mu_norm = (mu - pre_line) / edge_step

    return mu_norm, e0, edge_step

# Normalize and visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Raw spectrum
axes[0].plot(energy, mu_spectra[0], "b-", lw=0.8)
axes[0].set_title("Raw Spectrum")
axes[0].set_xlabel("Energy (eV)")
axes[0].set_ylabel("mu(E)")

# Normalized spectrum
mu_norm, e0, step = normalize_xas(energy, mu_spectra[0])
axes[1].plot(energy, mu_norm, "b-", lw=0.8)
axes[1].axhline(0, color="gray", ls="--")
axes[1].axhline(1, color="gray", ls="--")
axes[1].set_title(f"Normalized (E0={e0:.1f} eV, step={step:.3f})")
axes[1].set_xlabel("Energy (eV)")

# First derivative
dmu = np.gradient(mu_norm, energy)
axes[2].plot(energy, dmu, "r-", lw=0.8)
axes[2].set_title("First Derivative")
axes[2].set_xlabel("Energy (eV)")

plt.tight_layout()
```

## Noise Level Assessment

```python
def estimate_noise(energy, mu_norm, region="pre_edge", e0=None):
    """Estimate noise level in a specified spectral region."""
    if e0 is None:
        e0 = energy[np.argmax(np.gradient(mu_norm, energy))]

    e_rel = energy - e0

    if region == "pre_edge":
        mask = (e_rel >= -100) & (e_rel <= -30)
    elif region == "post_edge":
        mask = (e_rel >= 100) & (e_rel <= 300)
    else:
        raise ValueError("region must be 'pre_edge' or 'post_edge'")

    segment = mu_norm[mask]
    # Fit and subtract a linear trend, then measure residual
    x = energy[mask]
    coeffs = np.polyfit(x, segment, 1)
    residual = segment - np.polyval(coeffs, x)

    return np.std(residual)

# Assess noise for all spectra
noise_levels = []
edge_steps = []
for i in range(mu_spectra.shape[0]):
    mu_n, e0, step = normalize_xas(energy, mu_spectra[i])
    noise = estimate_noise(energy, mu_n, "post_edge", e0)
    noise_levels.append(noise)
    edge_steps.append(step)

noise_levels = np.array(noise_levels)
edge_steps = np.array(edge_steps)
snr = edge_steps / noise_levels

print(f"Post-edge noise: {np.mean(noise_levels):.5f} +/- {np.std(noise_levels):.5f}")
print(f"Edge step:       {np.mean(edge_steps):.4f} +/- {np.std(edge_steps):.4f}")
print(f"SNR:             {np.mean(snr):.1f} +/- {np.std(snr):.1f}")
```

## Outlier Detection

Identify spectra that deviate significantly from the population:

```python
from scipy.spatial.distance import cdist

# Normalize all spectra
normed = np.zeros_like(mu_spectra)
for i in range(mu_spectra.shape[0]):
    normed[i], _, _ = normalize_xas(energy, mu_spectra[i])

# Compute mean spectrum and distance of each spectrum
mean_spectrum = np.mean(normed, axis=0)
distances = np.sqrt(np.sum((normed - mean_spectrum) ** 2, axis=1))

threshold = np.mean(distances) + 3 * np.std(distances)
outliers = np.where(distances > threshold)[0]

print(f"Outlier spectra (3-sigma): {outliers}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distance plot
axes[0].bar(range(len(distances)), distances, color="steelblue")
axes[0].axhline(threshold, color="red", ls="--", label="3-sigma threshold")
axes[0].set_xlabel("Spectrum Index")
axes[0].set_ylabel("L2 Distance from Mean")
axes[0].legend()

# Overlay outlier spectra
axes[1].plot(energy, mean_spectrum, "k-", lw=2, label="Mean")
for idx in outliers:
    axes[1].plot(energy, normed[idx], "r-", lw=0.8, alpha=0.6)
axes[1].set_xlabel("Energy (eV)")
axes[1].set_ylabel("Normalized mu")
axes[1].set_title(f"Outlier Spectra ({len(outliers)} found)")
axes[1].legend()

plt.tight_layout()
```

## Principal Component Analysis (PCA)

PCA identifies the number of distinct spectral components in a dataset and can
reveal systematic variations:

```python
from sklearn.decomposition import PCA

# PCA on normalized spectra
pca = PCA(n_components=min(10, mu_spectra.shape[0]))
scores = pca.fit_transform(normed)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Scree plot
axes[0].bar(range(1, 11), pca.explained_variance_ratio_[:10] * 100)
axes[0].set_xlabel("Component")
axes[0].set_ylabel("Variance Explained (%)")
axes[0].set_title("PCA Scree Plot")

# First 3 components
for i in range(3):
    axes[1].plot(energy, pca.components_[i], label=f"PC{i+1}")
axes[1].set_xlabel("Energy (eV)")
axes[1].set_title("Principal Components")
axes[1].legend()

# Score scatter (PC1 vs PC2)
axes[2].scatter(scores[:, 0], scores[:, 1], c="steelblue", s=20)
if len(outliers) > 0:
    axes[2].scatter(scores[outliers, 0], scores[outliers, 1],
                    c="red", s=40, label="Outliers")
axes[2].set_xlabel("PC1 Score")
axes[2].set_ylabel("PC2 Score")
axes[2].set_title("PCA Score Plot")
axes[2].legend()

plt.tight_layout()
```

Interpretation:
- **1--2 significant components**: Sample is relatively homogeneous
- **3--5 components**: Multiple chemical species present
- **Gradual variance decay**: Noise-dominated, may need more averaging
- **PC1 often captures edge shift** -- indicates oxidation state variation
- **PC2 often captures coordination changes** -- local structure differences

## Self-Absorption Check

For concentrated samples (fluorescence detection), self-absorption can distort spectra:

```python
# Compare transmission and fluorescence if both available
# Self-absorption suppresses XANES features and reduces edge jump

# Quick check: compare white-line height to reference
ref_spectrum = np.load("reference_standard.npy")  # Known good spectrum
ref_norm, _, _ = normalize_xas(energy, ref_spectrum)

sample_norm = normed[0]

wl_range = (energy > e0) & (energy < e0 + 15)
ref_wl_height = np.max(ref_norm[wl_range])
sample_wl_height = np.max(sample_norm[wl_range])
ratio = sample_wl_height / ref_wl_height

if ratio < 0.8:
    print(f"WARNING: White-line ratio = {ratio:.2f} -- possible self-absorption")
else:
    print(f"White-line ratio = {ratio:.2f} -- acceptable")
```

## Related Resources

- [Spectroscopy EDA notebook](notebooks/03_spectral_eda.ipynb)
- [Larch XAS analysis](https://xraypy.github.io/xraylarch/)
- [Athena/Artemis (Demeter)](https://bruceravel.github.io/demeter/)
- [Spectroscopy modality overview](../../02_xray_modalities/)
