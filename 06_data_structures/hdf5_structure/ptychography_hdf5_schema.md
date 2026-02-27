# Ptychography HDF5 / CXI Schema

## Overview

Ptychographic datasets at APS BER beamlines (e.g., 2-ID-D, 26-ID) are stored using
the **CXI (Coherent X-ray Imaging)** format, an HDF5-based convention developed by the
CXIDB community for coherent diffraction experiments. The format captures diffraction
patterns, scan positions, probe information, and detector geometry in a standardized
hierarchy compatible with reconstruction codes such as **PtychoNN**, **Tike**, and
**PyNX**.

Ptychography is a scanning coherent diffraction technique that recovers both amplitude
and phase of the sample's transmission function at resolutions well beyond the focusing
optic's limit, routinely achieving sub-10 nm spatial resolution.

## CXI HDF5 Hierarchy

```
/entry_1/
  @NX_class = "NXentry"

  instrument_1/
    @NX_class = "NXinstrument"

    source_1/
      energy          scalar     float64       # Photon energy (eV)
      @units = "eV"

    detector_1/
      @NX_class = "NXdetector"
      data            [npos, ny_det, nx_det]  float32  # Diffraction patterns
      distance        scalar     float64       # Sample-detector distance (m)
      x_pixel_size    scalar     float64       # Detector pixel size (m)
      y_pixel_size    scalar     float64       # Detector pixel size (m)
      corner_position [3]        float64       # Detector corner in lab frame (m)
      mask            [ny_det, nx_det]  uint8  # Bad pixel mask (0=good, 1=bad)
      saturation_value scalar    float64       # Detector saturation threshold

    detector_1/detectorSpecific/
      countrate_correction_applied  scalar  int32
      flatfield_applied             scalar  int32
      pixel_mask                    [ny_det, nx_det]  uint32

  sample_1/
    @NX_class = "NXsample"
    @name = "biofilm_section_A"
    geometry_1/
      translation     [npos, 3]   float64     # Scan positions (m), columns = [x, y, z]
      @units = "m"

  data_1/
    @NX_class = "NXdata"
    data            -> /entry_1/instrument_1/detector_1/data   # Soft link
    translation     -> /entry_1/sample_1/geometry_1/translation

  image_1/                                     # Reconstructed result (optional)
    data            [ny_obj, nx_obj]  complex64 # Complex object transmission
    @is_fft_shifted = 1
    mask            [ny_obj, nx_obj]  float32   # Support mask
    process_1/
      @algorithm     = "ePIE"
      @iterations    = 500
      @error_metric  [500]  float32            # Convergence history
```

## Scan Position Conventions

Ptychography requires overlapping illumination spots. Scan positions define the probe
center at each measurement point:

| Scan Pattern | Description | Typical Parameters |
|-------------|-------------|-------------------|
| Raster | Regular grid | step = 100--400 nm, overlap 60--75% |
| Fermat spiral | Non-periodic spiral | npos = 200--5000, max_radius |
| Concentric rings | Ring-based | nrings, points_per_ring |
| Random jittered | Grid + random offset | step + jitter_amplitude |

The Fermat spiral is preferred at APS because it avoids periodic artifacts in
reconstruction and provides nearly uniform coverage with minimal positions.

```python
# Fermat spiral scan positions
import numpy as np

def fermat_spiral(n_points, step_size=100e-9):
    """Generate Fermat spiral scan positions."""
    golden_angle = np.pi * (3 - np.sqrt(5))
    indices = np.arange(n_points)
    radii = step_size * np.sqrt(indices)
    angles = golden_angle * indices
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y, np.zeros(n_points)])  # [npos, 3]
```

## Typical Dimensions

| Parameter | Soft X-ray (2-ID-D) | Hard X-ray (26-ID) |
|-----------|---------------------|---------------------|
| Positions (npos) | 500--5000 | 200--2000 |
| Detector (ny, nx) | 256x256 | 512x512 or 256x256 |
| Photon energy | 500--2000 eV | 6000--12000 eV |
| Detector distance | 1--3 m | 2--5 m |
| Pixel size (detector) | 50--75 um | 50--75 um |
| Achieved resolution | 5--20 nm | 8--50 nm |
| File size | 0.5--5 GB | 1--10 GB |

## Python Loading Examples

### Opening and Inspecting a CXI File

```python
import h5py
import numpy as np

filepath = "ptycho_scan_001.cxi"

with h5py.File(filepath, "r") as f:
    # Print structure
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
        else:
            print(f"  {name}/")
    f.visititems(visitor)
```

### Loading Diffraction Patterns and Positions

```python
with h5py.File(filepath, "r") as f:
    # Diffraction patterns
    diff_dset = f["/entry_1/instrument_1/detector_1/data"]
    print(f"Diffraction patterns: {diff_dset.shape}")   # (npos, ny, nx)

    # Load a single pattern
    pattern_0 = diff_dset[0, :, :]

    # Scan positions in meters
    positions = f["/entry_1/sample_1/geometry_1/translation"][:]  # (npos, 3)
    pos_x = positions[:, 0] * 1e6   # Convert to microns
    pos_y = positions[:, 1] * 1e6

    # Detector parameters
    det = f["/entry_1/instrument_1/detector_1"]
    distance = det["distance"][()]           # meters
    px_size = det["x_pixel_size"][()]        # meters
    energy_eV = f["/entry_1/instrument_1/source_1/energy"][()]

    # Wavelength calculation
    h_c = 1.23984198e-6   # eV * m
    wavelength = h_c / energy_eV
    print(f"Wavelength: {wavelength*1e10:.4f} Angstrom")
```

### Loading and Applying the Detector Mask

```python
with h5py.File(filepath, "r") as f:
    patterns = f["/entry_1/instrument_1/detector_1/data"][:]
    mask = f["/entry_1/instrument_1/detector_1/mask"][:]

    # Apply mask: set bad pixels to zero
    patterns_masked = patterns * (1 - mask)[np.newaxis, :, :]
```

### Visualizing Scan Coverage

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Scan positions
axes[0].scatter(pos_x, pos_y, s=2, c=np.arange(len(pos_x)), cmap="viridis")
axes[0].set_xlabel("X (um)")
axes[0].set_ylabel("Y (um)")
axes[0].set_title("Scan Positions (Fermat Spiral)")
axes[0].set_aspect("equal")

# Log-scaled diffraction pattern
axes[1].imshow(np.log1p(patterns[len(patterns)//2]),
               cmap="gray", origin="lower")
axes[1].set_title("Central Diffraction Pattern (log scale)")

plt.tight_layout()
```

## Preprocessing Requirements

Before reconstruction, ptychography data needs:

1. **Background subtraction** -- Remove detector dark current
2. **Hot/dead pixel correction** -- Interpolate using the detector mask
3. **Cropping/binning** -- Reduce detector frames to the coherent speckle region
4. **Position correction** -- Refine scan positions using cross-correlation or
   joint optimization during reconstruction
5. **Probe initialization** -- Estimate initial probe from Fresnel propagation of the
   known zone plate or KB mirror aperture

## Reconstruction Output

Reconstruction produces a complex-valued object array where:
- **Amplitude** = absorption contrast (related to beta, imaginary part of refractive index)
- **Phase** = phase shift (related to delta, real part of refractive index)

```python
# After reconstruction
obj = recon_object   # complex64 array [ny_obj, nx_obj]

amplitude = np.abs(obj)
phase = np.angle(obj)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(amplitude, cmap="gray")
axes[0].set_title("Amplitude")
axes[1].imshow(phase, cmap="twilight")
axes[1].set_title("Phase")
```

## Related Resources

- [CXI format specification (CXIDB)](https://www.cxidb.org/cxi.html)
- [Tike reconstruction library](https://tike.readthedocs.io/)
- [PtychoNN -- neural network ptychography](https://github.com/mcherukara/PtychoNN)
- [Ptychography modality overview](../../02_xray_modalities/)
