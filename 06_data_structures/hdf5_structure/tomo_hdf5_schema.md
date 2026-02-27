# Tomography HDF5 Schema (Data Exchange Format)

## Overview

Tomography datasets at APS follow the **Data Exchange** (DXchange) format, an HDF5-based
convention developed at Argonne for synchrotron tomography. The format stores projection
images, flat-field and dark-field references, rotation angles, and experimental metadata
in a standardized hierarchy that is directly consumed by reconstruction tools such as
**TomoPy** and **TomocuPy**.

At APS BER beamlines (e.g., 2-BM, 7-BM, 32-ID), a single tomography scan captures
hundreds to thousands of projections as the sample rotates through 180 or 360 degrees.

## Data Exchange HDF5 Hierarchy

```
/exchange/
  data              [nproj, nrow, ncol]   uint16 / float32   # Projection images
  data_white        [nflat, nrow, ncol]   uint16 / float32   # Flat-field (open beam)
  data_dark         [ndark, nrow, ncol]   uint16 / float32   # Dark-field (beam off)
  theta             [nproj]               float32             # Rotation angles (radians)

/measurement/
  instrument/
    source/
      @name          = "APS"
      @energy        = 6.0                 # GeV ring energy
      @current       = 100.0               # mA ring current
    monochromator/
      @energy        = 20.0                # keV beam energy
      @energy_mode   = "white" | "mono"
    detector/
      @manufacturer  = "Teledyne FLIR"
      @model         = "Oryx 10GigE"
      @pixel_size_x  = 6.5e-6             # meters
      @pixel_size_y  = 6.5e-6
      @exposure_time = 0.01                # seconds
      @binning       = 1
      roi_x          [2]  int32            # [start, size] pixels
      roi_y          [2]  int32
    scintillator/
      @material      = "LuAG:Ce"
      @thickness     = 50e-6               # meters (50 um)
    objective/
      @magnification = 5.0

  sample/
    @name            = "rhizosphere_core_12"
    @description     = "Soil rhizosphere section, sieved < 2mm"
    experimenter/
      @name          = "J. Smith"
      @institution   = "ANL"
    environment/
      @temperature   = 22.0                # Celsius
      @humidity      = 45.0                # percent

/process/
  acquisition/
    @start_date      = "2025-03-15T08:30:00"
    @end_date        = "2025-03-15T09:15:00"
    rotation/
      @range         = 180.0               # degrees
      @num_angles    = 1800
      @speed         = 1.0                 # deg/sec (for fly scan)
    flat_field/
      @num_images    = 20
      @frequency     = "every 200 projections"
```

## Typical Dimensions

| Parameter | Standard (2-BM) | High-res (32-ID) | Micro-CT (7-BM) |
|-----------|-----------------|-------------------|------------------|
| Projections (nproj) | 900--1800 | 1500--3600 | 1200--2400 |
| Rows (nrow) | 2048 | 2048--4096 | 2048 |
| Columns (ncol) | 2448 | 2048--4096 | 2448 |
| Flat fields (nflat) | 10--40 | 20--50 | 10--20 |
| Dark fields (ndark) | 10--20 | 10--20 | 10--20 |
| Pixel size | 0.65--6.5 um | 0.03--1.0 um | 1.3--6.5 um |
| File size | 8--20 GB | 20--100 GB | 5--15 GB |

## Python Loading with dxchange

```python
import dxchange

# Load all arrays in one call
proj, flat, dark, theta = dxchange.read_aps_32id(
    "tomo_scan_001.h5", sino=(500, 600)    # Load sinogram range [500:600]
)
print(f"Projections: {proj.shape}")         # (nproj, 100, ncol)
print(f"Flat fields: {flat.shape}")         # (nflat, 100, ncol)
print(f"Dark fields: {dark.shape}")         # (ndark, 100, ncol)
print(f"Theta:       {theta.shape}")        # (nproj,)
```

### Direct h5py Access

```python
import h5py
import numpy as np

with h5py.File("tomo_scan_001.h5", "r") as f:
    # Lazy access -- no data loaded yet
    proj_dset = f["/exchange/data"]
    print(f"Shape: {proj_dset.shape}, dtype: {proj_dset.dtype}")

    # Load a single projection
    proj_0 = proj_dset[0, :, :]

    # Load a sinogram slice
    sino = proj_dset[:, 1024, :]             # All angles, row 1024

    # Load theta
    theta = f["/exchange/theta"][:]

    # Read metadata
    energy = f["/measurement/instrument/monochromator"].attrs["energy"]
    px_size = f["/measurement/instrument/detector"].attrs["pixel_size_x"]
```

## Preprocessing Steps

Before reconstruction, tomography data requires several preprocessing stages. Each step
is typically applied in this order:

### 1. Dark-Field Subtraction

Remove detector thermal noise by subtracting the mean dark-field image:

```python
import tomopy

dark_mean = np.mean(dark, axis=0)
proj_corrected = proj - dark_mean
flat_corrected = flat - dark_mean
```

### 2. Flat-Field Normalization

Normalize projections by the flat field to correct for beam non-uniformity:

```python
proj_norm = tomopy.normalize(proj, flat, dark)
```

### 3. Negative Logarithm

Convert transmission to absorption (Beer-Lambert law):

```python
proj_norm = tomopy.minus_log(proj_norm)
```

### 4. Ring Artifact Removal

Suppress ring artifacts caused by defective detector pixels:

```python
proj_clean = tomopy.remove_stripe_fw(proj_norm, level=7, wname="db5")
```

### 5. Phase Retrieval (Optional)

For propagation-based phase contrast data:

```python
proj_phase = tomopy.retrieve_phase(
    proj_clean, pixel_size=px_size, dist=50.0,
    energy=energy, alpha=1e-3, pad=True
)
```

### 6. Rotation Center Finding

Determine the center of rotation (critical for reconstruction quality):

```python
rot_center = tomopy.find_center_vo(proj_clean)
print(f"Rotation center: {rot_center:.2f} pixels")
```

### 7. Reconstruction

```python
recon = tomopy.recon(proj_clean, theta, center=rot_center, algorithm="gridrec")
recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)
```

## Time-Series Tomography

For 4D (3D + time) experiments, multiple scans are stored in a sequence:

```
experiment_dir/
  tomo_t000.h5      # t = 0 min
  tomo_t010.h5      # t = 10 min
  tomo_t020.h5      # t = 20 min
  ...
```

Each file follows the same Data Exchange schema. A master index file may provide:

```
/time_series/
  timestamps    [ntime]   float64        # Seconds from start
  filenames     [ntime]   bytes          # Relative file paths
  conditions    [ntime]   bytes          # Experimental notes per timepoint
```

## Related Resources

- [Data Exchange specification](https://dxchange.readthedocs.io/)
- [TomoPy documentation](https://tomopy.readthedocs.io/)
- [Tomography EDA guide](../eda/tomo_eda.md)
- [Tomography modality overview](../../02_xray_modalities/)
