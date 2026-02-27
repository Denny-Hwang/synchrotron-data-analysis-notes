# XRF HDF5 Schema (MAPS Output Format)

## Overview

X-ray Fluorescence (XRF) microscopy data at APS is processed primarily by **MAPS**
(Microscopy Analysis and Processing Software), which outputs results in a well-defined
HDF5 structure. Each HDF5 file contains fitted elemental maps, raw spectra, calibration
data, and scan metadata for a single raster scan.

A typical XRF dataset at eBERlight beamlines (2-ID-D, 2-ID-E) consists of:
- A 2D raster scan over the sample surface
- Full energy-dispersive spectra at each pixel
- Fitted elemental concentration maps
- Quantification standards and calibration curves

## MAPS HDF5 Group Hierarchy

```
/MAPS/
  Spectra/                      # Raw and integrated spectra
    mca_arr         [nrow, ncol, nchan]   float32   # Raw MCA spectra per pixel
    integrated_spectra [nchan]  float32              # Sum spectrum
    energy           [nchan]    float32              # Energy axis (keV)
    energy_calib     [3]        float32              # Quadratic calibration coefficients

  XRF_Analyzed/                 # Fitted elemental maps
    Fitted/
      Counts_Per_Sec  [nelem, nrow, ncol]  float32  # Fitted counts/sec per element
    Channel_Names     [nelem]   bytes                # Element symbols ("Fe", "Cu", ...)
    Channel_Units     [nelem]   bytes                # "ug/cm2" or "counts"

  XRF_ROI/                      # Region-of-interest integrated maps
    Counts_Per_Sec    [nroi, nrow, ncol]   float32
    ROI_Names         [nroi]    bytes
    ROI_Limits        [nroi, 2] float32              # Energy window [low, high] keV

  Scalers/                      # Detector and I0 scalers
    Names             [nscaler] bytes                # "I0", "I1", "SRcurrent", ...
    Values            [nscaler, nrow, ncol] float32

  Scan/                         # Scan parameters
    x_axis            [ncol]    float32              # X positions (microns)
    y_axis            [nrow]    float32              # Y positions (microns)
    scan_time_stamp   scalar    bytes                # ISO 8601 timestamp
    dwell_time        scalar    float32              # Per-pixel dwell time (seconds)
    beamline          scalar    bytes                # "2-ID-D"
    incident_energy   scalar    float32              # keV
    detector_distance scalar    float32              # mm

  Quantification/               # Standards and calibration
    Calibration_Curve [nelem, 2] float32             # Slope and intercept per element
    Standard_Name     scalar    bytes                # e.g., "AXO_RF_7"
    Standard_Filenames [nstd]   bytes                # Reference standard files
```

## Typical Dimensions

| Parameter | Microprobe (2-ID-D) | Bionanoprobe (9-ID) |
|-----------|---------------------|---------------------|
| Rows (nrow) | 100--500 | 200--2000 |
| Columns (ncol) | 100--500 | 200--2000 |
| Channels (nchan) | 2048 or 4096 | 4096 |
| Elements (nelem) | 10--25 | 10--30 |
| Pixel size | 0.5--5 um | 20--200 nm |
| File size | 0.2--2 GB | 1--20 GB |

## Python Loading Examples

### Opening and Inspecting Structure

```python
import h5py
import numpy as np

filepath = "sample_xrf_scan.h5"

with h5py.File(filepath, "r") as f:
    # Print full tree structure
    def print_tree(name, obj):
        print(name)
    f.visititems(print_tree)

    # Check top-level groups
    print("Groups:", list(f["MAPS"].keys()))
```

### Loading Elemental Maps

```python
with h5py.File(filepath, "r") as f:
    maps = f["MAPS/XRF_Analyzed/Fitted/Counts_Per_Sec"][:]     # [nelem, nrow, ncol]
    element_names = [n.decode() for n in f["MAPS/XRF_Analyzed/Channel_Names"][:]]

    # Extract a single element map
    fe_idx = element_names.index("Fe")
    fe_map = maps[fe_idx]    # [nrow, ncol]

    print(f"Iron map shape: {fe_map.shape}")
    print(f"Fe range: {fe_map.min():.2f} -- {fe_map.max():.2f} ug/cm2")
```

### Loading Raw Spectra

```python
with h5py.File(filepath, "r") as f:
    spectra = f["MAPS/Spectra/mca_arr"]         # [nrow, ncol, nchan] -- lazy
    energy = f["MAPS/Spectra/energy"][:]         # [nchan]
    summed = f["MAPS/Spectra/integrated_spectra"][:]  # [nchan]

    # Extract spectrum from a single pixel
    pixel_spectrum = spectra[50, 75, :]          # Row 50, Col 75

    # Load a subregion only (memory efficient)
    roi_spectra = spectra[40:60, 70:90, :]       # 20x20 pixel ROI
```

### Loading Scan Coordinates and Scalers

```python
with h5py.File(filepath, "r") as f:
    x = f["MAPS/Scan/x_axis"][:]                 # microns
    y = f["MAPS/Scan/y_axis"][:]
    dwell = f["MAPS/Scan/dwell_time"][()]         # scalar

    scaler_names = [n.decode() for n in f["MAPS/Scalers/Names"][:]]
    scaler_vals = f["MAPS/Scalers/Values"][:]     # [nscaler, nrow, ncol]

    i0_idx = scaler_names.index("I0")
    i0_map = scaler_vals[i0_idx]                  # Incident flux map
```

### Quick Visualization

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

elements_to_plot = ["Fe", "Cu", "Zn"]
for ax, elem in zip(axes, elements_to_plot):
    idx = element_names.index(elem)
    im = ax.imshow(maps[idx], cmap="inferno", origin="lower",
                   extent=[x[0], x[-1], y[0], y[-1]])
    ax.set_title(elem)
    ax.set_xlabel("X (um)")
    ax.set_ylabel("Y (um)")
    plt.colorbar(im, ax=ax, label="ug/cm2")

plt.tight_layout()
plt.savefig("xrf_elemental_maps.png", dpi=150)
```

## Multi-Detector Handling

Modern XRF beamlines use 4-element or 7-element Vortex ME4/ME7 detectors. In MAPS output:

- Individual detector spectra are stored in `/MAPS/Spectra/mca_arr_detN` (N = 0, 1, 2, ...)
- The summed array `/MAPS/Spectra/mca_arr` combines all detectors after dead-time correction
- Dead-time correction factors are in `/MAPS/Scalers/` as per-detector scaler channels

## Batch Processing Pattern

For scanning multiple HDF5 files in a dataset directory:

```python
from pathlib import Path

data_dir = Path("/data/eBERlight/2024-3/xrf_scans/")
h5_files = sorted(data_dir.glob("*.h5"))

all_fe_maps = []
for fpath in h5_files:
    with h5py.File(fpath, "r") as f:
        names = [n.decode() for n in f["MAPS/XRF_Analyzed/Channel_Names"][:]]
        fe_idx = names.index("Fe")
        fe_map = f["MAPS/XRF_Analyzed/Fitted/Counts_Per_Sec"][fe_idx]
        all_fe_maps.append(fe_map)

print(f"Loaded {len(all_fe_maps)} Fe maps")
```

## Related Resources

- [MAPS software documentation](https://www.aps.anl.gov/Microscopy/Software)
- [XRF EDA guide](../eda/xrf_eda.md)
- [XRF modality overview](../../02_xray_modalities/)
