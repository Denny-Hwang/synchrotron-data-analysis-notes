# Ptychography Data Formats

## Raw Data: Diffraction Patterns

### CXI Format (Coherent X-ray Imaging)

The **CXI format** (based on HDF5) is the emerging standard for coherent imaging data:

```
scan_ptycho.cxi
├── /entry_1/
│   ├── /data_1/
│   │   └── /data            [shape: (Npositions, Ny_det, Nx_det), dtype: uint16/float32]
│   │                         # Diffraction patterns (one per scan position)
│   │                         # Typical: (2000, 256, 256) or (5000, 512, 512)
│   │
│   ├── /sample_1/
│   │   └── /geometry_1/
│   │       └── /translation  [shape: (Npositions, 3), dtype: float64]
│   │                         # Scan positions in meters (x, y, z)
│   │
│   ├── /instrument_1/
│   │   ├── /detector_1/
│   │   │   ├── /distance     # Sample-detector distance (m), e.g., 2.0
│   │   │   ├── /x_pixel_size # Detector pixel size (m), e.g., 75e-6
│   │   │   ├── /y_pixel_size
│   │   │   ├── /corner_position  [shape: (3,)]
│   │   │   │                     # Detector corner position in meters
│   │   │   └── /mask         [shape: (Ny_det, Nx_det), dtype: uint8]
│   │   │                     # Bad pixel mask (1 = valid, 0 = masked)
│   │   │
│   │   ├── /source_1/
│   │   │   ├── /energy       # Photon energy (eV), e.g., 8000
│   │   │   └── /wavelength   # Wavelength (m)
│   │   │
│   │   └── /illumination_1/
│   │       └── /probe        [shape: (Ny_probe, Nx_probe), dtype: complex64]
│   │                         # Initial probe guess (optional)
│   │
│   └── /image_1/
│       ├── /data_real        [shape: (Ny_obj, Nx_obj), dtype: float32]
│       │                     # Reconstructed object (real part)
│       ├── /data_imag        [shape: (Ny_obj, Nx_obj), dtype: float32]
│       │                     # Reconstructed object (imaginary part)
│       ├── /data_phase       [shape: (Ny_obj, Nx_obj), dtype: float32]
│       │                     # Phase image (radians)
│       └── /data_amplitude   [shape: (Ny_obj, Nx_obj), dtype: float32]
│                             # Amplitude image
```

### APS HDF5 Format

APS beamlines may use a simplified HDF5 format:

```
scan.h5
├── /exchange/
│   ├── /data              [shape: (Npositions, Ny_det, Nx_det)]
│   │                       # Diffraction patterns
│   ├── /positions_x       [shape: (Npositions,)]
│   │                       # X scan positions (µm)
│   ├── /positions_y       [shape: (Npositions,)]
│   │                       # Y scan positions (µm)
│   └── /theta             [shape: (Nangles,)]  # For ptycho-tomo
│
├── /instrument/
│   ├── /detector/
│   │   ├── /distance      # meters
│   │   └── /pixel_size    # meters
│   └── /source/
│       └── /energy        # keV
```

## Metadata

### Essential Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `energy` | Photon energy | 8–12 keV |
| `wavelength` | X-ray wavelength | 0.1–0.2 nm |
| `detector_distance` | Sample-detector distance | 1–5 m |
| `pixel_size` | Detector pixel size | 55–75 µm |
| `scan_positions` | (x, y) coordinates per frame | µm precision |
| `step_size` | Nominal step between positions | 50–500 nm |
| `exposure_time` | Per-position integration time | 0.5–100 ms |

### Derived Parameters

```
# Pixel size in reconstructed image:
dx_recon = wavelength × detector_distance / (N_pixels × pixel_size_detector)

# Example: λ=0.155 nm, z=2 m, N=256, pixel=75 µm
# dx_recon = 0.155e-9 × 2 / (256 × 75e-6) = 16.1 nm
```

## Python Data Loading

```python
import h5py
import numpy as np

# Load CXI format
with h5py.File('scan_ptycho.cxi', 'r') as f:
    # Diffraction patterns
    patterns = f['/entry_1/data_1/data'][:]
    print(f"Patterns: {patterns.shape}")  # (Npos, Ny, Nx)

    # Scan positions (meters → micrometers)
    positions = f['/entry_1/sample_1/geometry_1/translation'][:] * 1e6
    pos_x = positions[:, 0]
    pos_y = positions[:, 1]

    # Detector parameters
    distance = f['/entry_1/instrument_1/detector_1/distance'][()]
    pixel_size = f['/entry_1/instrument_1/detector_1/x_pixel_size'][()]
    energy_eV = f['/entry_1/instrument_1/source_1/energy'][()]

    # Wavelength
    wavelength = 12398.4 / energy_eV * 1e-10  # eV → meters

    # Pixel size in reconstruction
    N = patterns.shape[-1]
    dx_recon = wavelength * distance / (N * pixel_size)
    print(f"Reconstruction pixel size: {dx_recon*1e9:.1f} nm")

# Load reconstructed image
with h5py.File('reconstruction.cxi', 'r') as f:
    phase = f['/entry_1/image_1/data_phase'][:]
    amplitude = f['/entry_1/image_1/data_amplitude'][:]

    # Complex object
    obj = amplitude * np.exp(1j * phase)
```

## Reconstructed Data

### Output Format

```
reconstruction.h5
├── /object/
│   ├── /complex           [shape: (Ny, Nx), dtype: complex64]
│   ├── /phase             [shape: (Ny, Nx), dtype: float32]
│   ├── /amplitude         [shape: (Ny, Nx), dtype: float32]
│   └── /pixel_size        # meters
│
├── /probe/
│   ├── /complex           [shape: (Ny_probe, Nx_probe), dtype: complex64]
│   └── /modes             [shape: (Nmodes, Ny_probe, Nx_probe), dtype: complex64]
│                          # For mixed-state / partial coherence
│
└── /metrics/
    ├── /error_per_iteration  [shape: (Niterations,)]
    ├── /algorithm             # "ePIE", "DM", "LSQ-ML"
    └── /n_iterations          # total iterations
```

### Ptycho-Tomography 3D

For ptychographic tomography, reconstruct 2D phase images at each angle,
then apply tomographic reconstruction:

```
ptycho_tomo/
├── angle_000/
│   ├── phase.tif          # Phase image at 0°
│   └── amplitude.tif
├── angle_001/
│   ├── phase.tif          # Phase image at 1°
│   └── amplitude.tif
├── ...
└── reconstructed_3d.h5
    └── /volume [shape: (Nz, Ny, Nx), dtype: float32]
        # 3D electron density map
```

## Preprocessing Requirements

1. **Dark subtraction**: Remove detector dark current
2. **Hot pixel masking**: Identify and mask defective pixels
3. **Center of mass alignment**: Align diffraction patterns to common center
4. **Background subtraction**: Remove air scatter and parasitic scattering
5. **Saturation handling**: Mask or scale saturated pixels near beam stop
6. **Position refinement**: Correct scan positions (interferometer or image-based)
