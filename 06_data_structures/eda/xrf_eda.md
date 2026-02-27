# XRF Data Exploratory Data Analysis

## Overview

X-ray Fluorescence (XRF) microscopy datasets are multi-channel 2D maps where each pixel
contains a full energy-dispersive spectrum. EDA for XRF data focuses on verifying spectral
quality, assessing elemental map distributions, identifying detector artifacts, and
discovering spatial correlations between elements.

This guide assumes data in the MAPS HDF5 format as described in
[xrf_hdf5_schema.md](../hdf5_structure/xrf_hdf5_schema.md).

## EDA Checklist for XRF Data

### 1. File Integrity

- [ ] HDF5 file opens without errors
- [ ] `/MAPS/Spectra/mca_arr` shape matches expected (nrow, ncol, nchan)
- [ ] `/MAPS/XRF_Analyzed/Fitted/Counts_Per_Sec` contains fitted maps
- [ ] Channel names are populated and match expected elements
- [ ] Scan coordinates (x_axis, y_axis) span the intended scan area

### 2. Integrated Spectrum Inspection

- [ ] Sum spectrum shows expected fluorescence peaks
- [ ] Peak positions align with known emission energies
- [ ] Compton and elastic scatter peaks are present at correct energy
- [ ] No unexpected peaks suggesting contamination
- [ ] Background level is reasonable (not dominated by scatter)

### 3. Per-Element Map Quality

- [ ] Each elemental map has reasonable dynamic range
- [ ] No all-zero or all-NaN maps
- [ ] Spatial features are consistent with known sample morphology
- [ ] No stripe artifacts from scan motion errors

### 4. Noise and SNR Assessment

- [ ] SNR estimated for each element
- [ ] Low-concentration elements identified for potential exclusion
- [ ] Noise floor characterized from sample-free regions

## Channel Histograms

Intensity histograms reveal the distribution of elemental concentrations and help
identify outliers, saturation, and background levels.

```python
import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File("xrf_scan.h5", "r") as f:
    maps = f["MAPS/XRF_Analyzed/Fitted/Counts_Per_Sec"][:]
    names = [n.decode() for n in f["MAPS/XRF_Analyzed/Channel_Names"][:]]

fig, axes = plt.subplots(4, 5, figsize=(20, 12))
axes = axes.ravel()

for i, (ax, name) in enumerate(zip(axes, names)):
    data = maps[i].ravel()
    data = data[data > 0]  # Exclude zeros
    if len(data) > 0:
        ax.hist(data, bins=100, log=True, color="steelblue", edgecolor="none")
        ax.set_title(f"{name} (median={np.median(data):.1f})")
        ax.axvline(np.percentile(data, 99), color="red", ls="--", label="99th pct")
    ax.set_xlabel("Counts/sec")

plt.tight_layout()
plt.savefig("xrf_channel_histograms.png", dpi=150)
```

## Signal-to-Noise Ratio (SNR) Analysis

SNR per element helps prioritize which channels contain meaningful signal:

```python
def compute_snr(element_map, background_roi=None):
    """Estimate SNR for an elemental map.

    Args:
        element_map: 2D array of fitted counts
        background_roi: tuple (row_slice, col_slice) for noise estimation
    """
    if background_roi is not None:
        noise_region = element_map[background_roi[0], background_roi[1]]
    else:
        # Use lowest 10% of pixels as noise estimate
        threshold = np.percentile(element_map, 10)
        noise_region = element_map[element_map <= threshold]

    signal = np.mean(element_map)
    noise = np.std(noise_region) if len(noise_region) > 0 else 1e-10
    return signal / noise

# Compute SNR for all elements
bg_roi = (slice(0, 10), slice(0, 10))  # Top-left corner assumed background
snr_values = {}
for i, name in enumerate(names):
    snr_values[name] = compute_snr(maps[i], bg_roi)

# Sort and display
for name, snr in sorted(snr_values.items(), key=lambda x: -x[1]):
    quality = "HIGH" if snr > 10 else "MED" if snr > 3 else "LOW"
    print(f"  {name:4s}: SNR = {snr:8.1f}  [{quality}]")
```

## Correlation Matrix

Element correlation matrices reveal co-localization patterns that may indicate
mineral phases, biological structures, or contamination:

```python
import seaborn as sns

# Reshape maps to (nelem, npixels) for correlation
nelem, nrow, ncol = maps.shape
flat_maps = maps.reshape(nelem, -1)

# Compute Pearson correlation
corr_matrix = np.corrcoef(flat_maps)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, xticklabels=names, yticklabels=names,
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            annot=True, fmt=".2f", ax=ax)
ax.set_title("XRF Element Correlation Matrix")
plt.tight_layout()
plt.savefig("xrf_correlation_matrix.png", dpi=150)
```

Interpretation:
- **Strong positive correlation** (r > 0.7): Elements co-localized, possibly same phase
- **Strong negative correlation** (r < -0.3): Anti-correlated, different phases
- **Fe-Mn correlation**: Common in soil -- indicates iron/manganese oxide phases
- **Ca-P correlation**: Biological -- indicates calcium phosphate (bones, cells)

## RGB Composite Maps

False-color composites assign three elements to R, G, B channels for rapid spatial
pattern recognition:

```python
def make_rgb_composite(maps, names, r_elem, g_elem, b_elem,
                       percentile_clip=99):
    """Create an RGB composite from three elemental maps."""
    rgb = np.zeros((*maps[0].shape, 3))

    for ch, elem in enumerate([r_elem, g_elem, b_elem]):
        idx = names.index(elem)
        channel = maps[idx].astype(float)
        vmin = np.percentile(channel, 1)
        vmax = np.percentile(channel, percentile_clip)
        channel = np.clip((channel - vmin) / (vmax - vmin + 1e-10), 0, 1)
        rgb[:, :, ch] = channel

    return rgb

# Example composites
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

combos = [("Fe", "Ca", "Zn"), ("Cu", "Mn", "P"), ("S", "K", "Fe")]
for ax, (r, g, b) in zip(axes, combos):
    rgb = make_rgb_composite(maps, names, r, g, b)
    ax.imshow(rgb, origin="lower")
    ax.set_title(f"R={r}, G={g}, B={b}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("xrf_rgb_composites.png", dpi=150)
```

## Dead Pixel Detection

Dead or hot pixels appear as isolated anomalous values that can bias analysis:

```python
from scipy.ndimage import median_filter

def detect_dead_pixels(image, threshold=5.0):
    """Detect dead/hot pixels by comparing to local median."""
    median_img = median_filter(image, size=3)
    diff = np.abs(image - median_img)
    mad = np.median(diff)  # Median absolute deviation
    mask = diff > threshold * mad
    return mask

# Check all elemental maps
total_dead = np.zeros((nrow, ncol), dtype=bool)
for i, name in enumerate(names):
    dead = detect_dead_pixels(maps[i])
    n_dead = dead.sum()
    if n_dead > 0:
        print(f"  {name}: {n_dead} dead/hot pixels ({100*n_dead/(nrow*ncol):.2f}%)")
    total_dead |= dead

print(f"\nTotal unique dead pixel positions: {total_dead.sum()}")
```

## I0 Normalization Check

Verify that the incident flux (I0) normalization was applied correctly:

```python
with h5py.File("xrf_scan.h5", "r") as f:
    scaler_names = [n.decode() for n in f["MAPS/Scalers/Names"][:]]
    scalers = f["MAPS/Scalers/Values"][:]
    i0_idx = scaler_names.index("I0")
    i0_map = scalers[i0_idx]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].imshow(i0_map, cmap="viridis", origin="lower")
axes[0].set_title("I0 (Incident Flux) Map")
plt.colorbar(axes[0].images[0], ax=axes[0])

axes[1].hist(i0_map.ravel(), bins=100, color="steelblue")
axes[1].set_title("I0 Distribution")
axes[1].set_xlabel("I0 counts")

# Check for I0 drops (beam dumps, top-up events)
i0_mean = np.mean(i0_map)
low_i0 = i0_map < 0.5 * i0_mean
if low_i0.any():
    print(f"WARNING: {low_i0.sum()} pixels have I0 < 50% of mean")
    print("  This may indicate beam dump or shutter issues")

plt.tight_layout()
```

## Summary Statistics Table

Generate a comprehensive summary table for documentation:

```python
import pandas as pd

rows = []
for i, name in enumerate(names):
    m = maps[i]
    rows.append({
        "Element": name,
        "Min": f"{m.min():.2f}",
        "Max": f"{m.max():.2f}",
        "Mean": f"{m.mean():.2f}",
        "Median": f"{np.median(m):.2f}",
        "Std": f"{m.std():.2f}",
        "SNR": f"{snr_values[name]:.1f}",
        "Dead Px": detect_dead_pixels(m).sum(),
    })

df = pd.DataFrame(rows)
print(df.to_markdown(index=False))
```

## Related Resources

- [XRF HDF5 schema](../hdf5_structure/xrf_hdf5_schema.md)
- [XRF EDA notebook](notebooks/01_xrf_eda.ipynb)
- [XRF modality overview](../../02_xray_modalities/)
