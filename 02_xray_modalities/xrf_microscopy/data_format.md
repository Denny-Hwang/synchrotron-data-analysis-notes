# XRF Microscopy Data Formats

## Raw Data: Full Spectral Maps

### HDF5 Structure (MAPS Output)

XRF data at APS is typically processed through the **MAPS** software, producing HDF5
files with the following structure:

```
scan_xrf.h5
├── /MAPS/
│   ├── /Spectra/
│   │   ├── /mca_arr        [shape: (Ndet, Ny, Nx, Nchannels)]
│   │   │                    # Raw MCA spectra per pixel per detector
│   │   │                    # Nchannels typically 2048
│   │   │                    # Ndet = number of detector elements (1-4)
│   │   └── /Integrated_Spectra/
│   │       └── /Spectra     [shape: (Nchannels,)]
│   │                        # Sum spectrum over entire map
│   │
│   ├── /XRF_Analyzed/
│   │   ├── /Fitted/
│   │   │   ├── /Counts_Per_Sec  [shape: (Nelements, Ny, Nx)]
│   │   │   │                     # Fitted elemental maps
│   │   │   └── /Channel_Names   [shape: (Nelements,)]
│   │   │                         # Element names: ['P', 'S', 'K', 'Ca', 'Fe', 'Zn', ...]
│   │   ├── /NNLS/               # Non-negative least squares fitting
│   │   │   └── /Counts_Per_Sec  [shape: (Nelements, Ny, Nx)]
│   │   └── /ROI/                # Simple region-of-interest integration
│   │       └── /Counts_Per_Sec  [shape: (Nelements, Ny, Nx)]
│   │
│   ├── /Scalers/
│   │   ├── /Names              # ['SRcurrent', 'us_ic', 'ds_ic', ...]
│   │   └── /Values             [shape: (Nscalers, Ny, Nx)]
│   │                           # Normalization signals (ion chambers, current)
│   │
│   └── /Scan/
│       ├── /x_axis             [shape: (Nx,)]  # X positions (µm)
│       ├── /y_axis             [shape: (Ny,)]  # Y positions (µm)
│       ├── /scan_time_stamp
│       └── /extra_pvs/                         # EPICS process variables
│           ├── /Names
│           └── /Values
```

### Raw Detector Data (Pre-MAPS)

Some experiments also store raw detector data:

```
raw_data.h5
├── /entry/
│   ├── /instrument/
│   │   └── /detector/
│   │       ├── /data          [shape: (Nscanpoints, Nchannels)]
│   │       │                   # Raw pulse-height spectra
│   │       ├── /live_time     [shape: (Nscanpoints,)]
│   │       ├── /real_time     [shape: (Nscanpoints,)]
│   │       └── /dead_time     [shape: (Nscanpoints,)]
│   └── /data/
│       ├── /x_position        [shape: (Nscanpoints,)]
│       └── /y_position        [shape: (Nscanpoints,)]
```

## Metadata

### Essential Metadata Fields

| Parameter | Description | Example |
|-----------|-------------|---------|
| `incident_energy` | X-ray energy (keV) | 10.0 |
| `dwell_time` | Integration time per pixel (ms) | 100 |
| `step_size_x/y` | Scan step size (µm) | 0.5, 0.5 |
| `beam_size` | FWHM of focused beam (nm or µm) | 200 nm |
| `detector_type` | Energy-dispersive detector model | "Vortex ME-4" |
| `ring_current` | Storage ring current (mA) | 100 |
| `us_ic / ds_ic` | Upstream/downstream ion chamber | (normalization) |

## Python Data Loading

```python
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Load MAPS-processed XRF data
with h5py.File('scan_xrf.h5', 'r') as f:
    # Get element names
    channel_names = [name.decode() for name in
                     f['/MAPS/XRF_Analyzed/Fitted/Channel_Names'][:]]

    # Load fitted elemental maps
    elemental_maps = f['/MAPS/XRF_Analyzed/Fitted/Counts_Per_Sec'][:]
    # shape: (Nelements, Ny, Nx)

    # Load positions
    x_axis = f['/MAPS/Scan/x_axis'][:]
    y_axis = f['/MAPS/Scan/y_axis'][:]

    # Load normalization (upstream ion chamber)
    scaler_names = [n.decode() for n in f['/MAPS/Scalers/Names'][:]]
    scalers = f['/MAPS/Scalers/Values'][:]
    ic_idx = scaler_names.index('us_ic')
    normalization = scalers[ic_idx]

# Print available elements
for i, name in enumerate(channel_names):
    print(f"{name}: min={elemental_maps[i].min():.1f}, "
          f"max={elemental_maps[i].max():.1f}, "
          f"mean={elemental_maps[i].mean():.1f}")

# Normalize by ion chamber
normalized_maps = elemental_maps / (normalization[np.newaxis, :, :] + 1e-10)

# Plot selected elements
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
elements_to_plot = ['P', 'S', 'K', 'Ca', 'Fe', 'Zn']
for ax, elem in zip(axes.flat, elements_to_plot):
    if elem in channel_names:
        idx = channel_names.index(elem)
        im = ax.imshow(normalized_maps[idx], cmap='viridis',
                       extent=[x_axis[0], x_axis[-1], y_axis[-1], y_axis[0]])
        ax.set_title(elem)
        plt.colorbar(im, ax=ax)
plt.tight_layout()
```

## Spectral Data Structure

Each pixel contains a full energy-dispersive spectrum:

```
Channel number (0-2047)  →  Energy (keV): E = channel × gain + offset
                              typically: E = channel × 0.01 + 0.0 keV

Spectrum features:
├── Elastic scatter peak (Rayleigh) at incident energy
├── Compton scatter peak (below incident energy)
├── Escape peaks (detector artifacts)
├── Sum peaks (pile-up artifacts at high count rates)
└── Fluorescence peaks:
    ├── Kα, Kβ lines for each element
    └── Lα, Lβ lines for heavy elements (Z > 40)
```

## Preprocessing Steps

### 1. Dead Time Correction
```python
corrected = raw_counts / (1 - dead_time_fraction)
```
Accounts for detector saturation at high count rates.

### 2. Normalization
```python
normalized = counts / (ring_current × dwell_time × ion_chamber)
```
Corrects for beam intensity variations during the scan.

### 3. Spectral Fitting
- **ROI integration**: Simple sum of channels around expected peak position
- **Gaussian fitting**: Fit each peak with Gaussian + linear background
- **Full-spectrum fitting**: MAPS/PyXRF fit all peaks simultaneously with detector response
- **NNLS**: Non-negative least squares decomposition using reference spectra

### 4. Quantification
- Convert fluorescence intensity to concentration (µg/cm² or ppm)
- Requires: reference standards, fundamental parameters, or calibrated standards
- MAPS uses fundamental parameters approach with thin-film approximation
