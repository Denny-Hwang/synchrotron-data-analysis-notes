# Scattering Data Formats

## Raw Data: 2D Scattering Patterns

### HDF5 Format (EIGER/PILATUS Detectors)

```
saxs_data.h5
├── /entry/
│   ├── /instrument/
│   │   ├── /detector/
│   │   │   ├── /data              [shape: (Nframes, Ny, Nx), dtype: uint32]
│   │   │   │                       # 2D scattering patterns
│   │   │   ├── /x_pixel_size      # 172 µm (PILATUS) or 75 µm (EIGER)
│   │   │   ├── /y_pixel_size
│   │   │   ├── /distance          # Sample-detector distance (m)
│   │   │   ├── /beam_center_x     # pixels
│   │   │   ├── /beam_center_y
│   │   │   └── /mask              [shape: (Ny, Nx), dtype: uint8]
│   │   │                           # Bad pixel / beamstop mask
│   │   ├── /source/
│   │   │   ├── /energy            # keV
│   │   │   └── /wavelength        # Å
│   │   └── /sample/
│   │       ├── /name
│   │       ├── /temperature       # K or °C
│   │       └── /concentration     # mg/mL (for protein SAXS)
│   │
│   └── /data/
│       └── /data -> /entry/instrument/detector/data
```

### XPCS Time Series

```
xpcs_timeseries.h5
├── /exchange/
│   ├── /data                  [shape: (Nframes, Ny, Nx), dtype: uint16]
│   │                           # Time series of speckle patterns
│   ├── /timestamps            [shape: (Nframes,), dtype: float64]
│   │                           # Frame timestamps (seconds)
│   └── /frame_rate            # Hz
│
├── /instrument/
│   ├── /detector/
│   │   ├── /distance
│   │   ├── /pixel_size
│   │   └── /mask              [shape: (Ny, Nx)]
│   └── /source/
│       └── /energy
│
└── /metadata/
    ├── /temperature
    ├── /sample_name
    └── /q_range               # [q_min, q_max] in Å⁻¹
```

## Processed Data: 1D Profiles

### Azimuthally Integrated I(q)

The primary reduced data product is the 1D scattering profile obtained by
azimuthal integration of the 2D pattern:

```
# ASCII format (common exchange format)
# Sample: Protein_solution_1mgml
# Beamline: 12-ID-B, APS
# Energy: 12.0 keV
# Distance: 3.5 m
# q (Å⁻¹)    I(q) (cm⁻¹)    σ(I)
0.00500      125.678         2.345
0.00520      124.891         2.301
0.00540      123.567         2.278
...
0.50000      0.00123         0.00056
```

### HDF5 Processed Format

```
saxs_processed.h5
├── /data/
│   ├── /q                    [shape: (Nq,), dtype: float64]
│   │                          # Scattering vector (Å⁻¹)
│   ├── /I                    [shape: (Nsamples, Nq), dtype: float64]
│   │                          # Intensity profiles
│   ├── /sigma                [shape: (Nsamples, Nq), dtype: float64]
│   │                          # Standard deviation
│   └── /sample_names         [shape: (Nsamples,)]
│
├── /guinier/
│   ├── /Rg                   # Radius of gyration (Å)
│   ├── /I0                   # Forward scattering intensity
│   ├── /qRg_range            # [qmin×Rg, qmax×Rg] for fit
│   └── /guinier_plot         [shape: (Nq_guinier, 2)]
│                              # ln(I) vs q²
│
└── /processing/
    ├── /background_file
    ├── /normalization         # e.g., "concentration"
    └── /software              # e.g., "ATSAS", "pyFAI"
```

## XPCS Processed Data: Correlation Functions

```
xpcs_processed.h5
├── /exchange/
│   ├── /norm-0-stderr        [shape: (Nq, Ntau)]
│   │                          # g₂ standard error
│   ├── /norm-0-g2            [shape: (Nq, Ntau)]
│   │                          # g₂(q, τ) correlation functions
│   ├── /tau                  [shape: (Ntau,), dtype: float64]
│   │                          # Delay times (seconds)
│   ├── /qphi_bin_centers     [shape: (Nq,)]
│   │                          # q values for each correlation
│   └── /twotime/
│       └── /C_all            [shape: (Nq, Nframes, Nframes)]
│                              # Two-time correlation matrices
│
├── /fits/
│   ├── /tau_relaxation       [shape: (Nq,)]
│   │                          # Fitted relaxation times
│   ├── /stretching_exponent  [shape: (Nq,)]
│   │                          # β exponent (KWW fit)
│   └── /contrast             [shape: (Nq,)]
│                              # Speckle contrast
│
└── /metadata/
    ├── /temperature
    ├── /frame_rate
    └── /acquisition_time
```

## Python Data Loading and Processing

```python
import numpy as np
import h5py

# === SAXS: Load and azimuthally integrate ===
import pyFAI

# Setup integrator
ai = pyFAI.AzimuthalIntegrator(
    dist=3.5,              # sample-detector distance (m)
    poni1=0.085,           # beam center Y (m)
    poni2=0.085,           # beam center X (m)
    pixel1=172e-6,         # pixel size Y (m)
    pixel2=172e-6,         # pixel size X (m)
    wavelength=1.033e-10   # wavelength (m) for 12 keV
)

# Load 2D pattern
with h5py.File('saxs_data.h5', 'r') as f:
    pattern_2d = f['/entry/instrument/detector/data'][0]
    mask = f['/entry/instrument/detector/mask'][:]

# Azimuthal integration
q, I, sigma = ai.integrate1d(
    pattern_2d, 1000,       # number of bins
    mask=mask,
    unit='q_A^-1',          # q in Å⁻¹
    error_model='poisson'
)

# === XPCS: Load and compute g2 ===
with h5py.File('xpcs_processed.h5', 'r') as f:
    g2 = f['/exchange/norm-0-g2'][:]    # (Nq, Ntau)
    tau = f['/exchange/tau'][:]          # delay times
    q_values = f['/exchange/qphi_bin_centers'][:]

# Plot correlation function for a specific q
import matplotlib.pyplot as plt

q_idx = 5  # select q bin
plt.semilogx(tau, g2[q_idx], 'o-')
plt.xlabel('τ (s)')
plt.ylabel('g₂(q, τ)')
plt.title(f'q = {q_values[q_idx]:.4f} Å⁻¹')
```

## Preprocessing Steps

### SAXS/WAXS

1. **Dark subtraction**: Remove detector dark current
2. **Flat-field correction**: Normalize pixel response (usually applied by detector firmware)
3. **Solid angle correction**: Account for varying solid angle across detector
4. **Polarization correction**: Correct for X-ray beam polarization
5. **Masking**: Mask beamstop, bad pixels, inter-module gaps
6. **Azimuthal integration**: 2D → 1D profile I(q)
7. **Background subtraction**: Subtract solvent/empty capillary scattering
8. **Normalization**: By concentration, transmission, exposure time
9. **Absolute scaling**: Convert to cm⁻¹ using glassy carbon standard

### XPCS

1. **Pixel masking**: Remove bad pixels and beamstop shadow
2. **q-binning**: Assign pixels to q bins for multi-q analysis
3. **Multi-tau correlation**: Compute g₂(q,τ) using multi-tau algorithm
4. **Two-time correlation**: Compute C(q, t₁, t₂) for non-stationary dynamics
5. **Fitting**: Fit g₂(τ) with exponential or stretched exponential (KWW) model:
   ```
   g₂(τ) = 1 + β × exp(-2(τ/τ_r)^γ)
   where τ_r = relaxation time, γ = stretching exponent
   ```

## Key Software

| Software | Purpose | Language |
|----------|---------|---------|
| **pyFAI** | Azimuthal integration (SAXS/WAXS) | Python |
| **ATSAS** | BioSAXS analysis suite | C/Fortran |
| **SASView** | SAXS/SANS modeling and fitting | Python |
| **XPCS-Eigen** | XPCS correlation analysis | C++/Python |
| **Xana** | XPCS data analysis | Python |
| **scikit-beam** | General scattering analysis | Python |
