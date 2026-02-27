# Tomography Data Formats

## Raw Data: Projections

### HDF5 (Data Exchange Format)

The **Data Exchange** format is the standard HDF5 schema for synchrotron tomography at APS:

```
scan.h5
├── /exchange/
│   ├── /data              [shape: (Nangles, Ny, Nx), dtype: uint16/float32]
│   │                      # Projection images
│   ├── /data_white        [shape: (Nflat, Ny, Nx), dtype: uint16/float32]
│   │                      # Flat field (open beam) images
│   ├── /data_dark         [shape: (Ndark, Ny, Nx), dtype: uint16/float32]
│   │                      # Dark field (no beam) images
│   └── /theta             [shape: (Nangles,), dtype: float32]
│                          # Rotation angles in degrees
│
├── /measurement/
│   ├── /instrument/
│   │   ├── /monochromator/
│   │   │   └── /energy    # X-ray energy (keV)
│   │   ├── /detector/
│   │   │   ├── /manufacturer
│   │   │   ├── /model
│   │   │   ├── /pixel_size   # µm
│   │   │   └── /exposure_time # seconds
│   │   └── /source/
│   │       ├── /beamline
│   │       └── /current   # ring current (mA)
│   └── /sample/
│       ├── /name
│       ├── /description
│       └── /experimenter
│
└── /process/
    └── /acquisition/
        ├── /rotation/
        │   ├── /range     # total rotation (°)
        │   └── /num_angles
        └── /magnification
```

### TIFF Stacks (Legacy)

Some beamlines still output individual TIFF files:

```
scan_001/
├── proj_0000.tif    # Projection at angle 0°
├── proj_0001.tif    # Projection at angle Δθ
├── ...
├── proj_1799.tif    # Projection at angle 179.9°
├── flat_0000.tif    # Flat field image 1
├── flat_0001.tif    # Flat field image 2
├── dark_0000.tif    # Dark field image 1
└── dark_0001.tif    # Dark field image 2
```

## Intermediate Data: Sinograms

After normalization, data can be rearranged into **sinograms** — one sinogram per
horizontal slice of the sample:

```
Sinogram shape: (Nangles, Nx)
    - Each row = one projection angle
    - Each column = one detector column (spatial position)
    - One sinogram per detector row (slice)
```

Sinograms are the natural input for reconstruction algorithms.

## Reconstructed Data: 3D Volumes

### Reconstructed Volume Structure

```
reconstructed.h5
├── /exchange/
│   └── /data     [shape: (Nz, Ny, Nx), dtype: float32]
│                  # 3D attenuation coefficient map
└── /metadata/
    ├── /pixel_size   # µm
    ├── /algorithm     # "FBP", "SIRT", "gridrec"
    └── /center        # rotation center (pixels)
```

### TIFF Stack Output

```
recon_slices/
├── recon_0000.tif    # Slice z=0
├── recon_0001.tif    # Slice z=1
├── ...
└── recon_2047.tif    # Slice z=2047
```

### Typical Volume Sizes

| Detector | Volume Dimensions | Size (16-bit) | Size (32-bit) |
|----------|------------------|---------------|---------------|
| 2048×2048 | 2048×2048×2048 | 16 GB | 32 GB |
| 4096×4096 | 4096×4096×4096 | 128 GB | 256 GB |

## Python Data Loading

```python
import h5py
import numpy as np

# Load Data Exchange HDF5
with h5py.File('scan.h5', 'r') as f:
    projections = f['/exchange/data'][:]        # (Nangles, Ny, Nx)
    flat_fields = f['/exchange/data_white'][:]  # (Nflat, Ny, Nx)
    dark_fields = f['/exchange/data_dark'][:]   # (Ndark, Ny, Nx)
    theta = f['/exchange/theta'][:]             # (Nangles,)

    print(f"Projections: {projections.shape}")
    print(f"Angles: {theta[0]:.1f}° to {theta[-1]:.1f}°")

# Normalize projections
dark = np.mean(dark_fields, axis=0)
flat = np.mean(flat_fields, axis=0)
normalized = (projections - dark) / (flat - dark + 1e-6)

# Negative log transform (Beer-Lambert)
sinograms = -np.log(np.clip(normalized, 1e-6, None))

# Load TIFF stack
import tifffile
projections = tifffile.imread('scan_001/proj_*.tif')  # shape: (Nangles, Ny, Nx)
```

## Preprocessing Steps

### 1. Dark/Flat Field Correction
```
corrected = (projection - dark_mean) / (flat_mean - dark_mean)
```
Removes detector background and normalizes for non-uniform beam intensity.

### 2. Negative Log Transform
```
sinogram = -log(corrected)
```
Converts transmission to line integral of attenuation coefficient.

### 3. Ring Artifact Removal
- **Fourier filtering**: High-pass filter in sinogram domain
- **TomoPy `remove_stripe_*`**: Multiple algorithms (Vo, Münch, etc.)
- Removes artifacts caused by defective/miscalibrated detector pixels

### 4. Rotation Center Determination
- Critical for reconstruction quality
- Methods: Vo's algorithm, phase correlation, manual optimization
- Error of even 1 pixel causes visible artifacts

### 5. Phase Retrieval (Optional)
- For phase-contrast data: Paganin single-distance phase retrieval
- Converts edge-enhancement contrast to area contrast
- Improves segmentation of low-contrast materials (biological samples)
