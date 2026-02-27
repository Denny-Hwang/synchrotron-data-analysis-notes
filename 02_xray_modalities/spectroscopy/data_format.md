# Spectroscopy Data Formats

## Raw Data: Energy-Absorption Spectra

### Standard XAS Data (ASCII)

Most XAS data is exchanged as simple columnar ASCII files:

```
# Fe K-edge XANES
# Sample: Soil sample A
# Beamline: 20-BM, APS
# Date: 2025-01-15
# Columns: Energy(eV)  I0  I1  IF  mu_transmission  mu_fluorescence
7100.0   234567   223456   1234   0.0487   0.00526
7100.5   234789   223567   1256   0.0489   0.00535
7101.0   234890   223678   1278   0.0490   0.00544
...
7200.0   235678   198765   8765   0.1721   0.03722
...
```

### Column Definitions

| Column | Description | Units |
|--------|-------------|-------|
| `Energy` | Incident X-ray energy | eV |
| `I0` | Incident beam intensity (ion chamber) | counts |
| `I1` | Transmitted beam intensity | counts |
| `IF` | Fluorescence intensity | counts |
| `mu_trans` | Absorption coefficient (transmission) | -ln(I1/I0) |
| `mu_fluor` | Absorption coefficient (fluorescence) | IF/I0 |

### HDF5 Format (for µ-XANES Imaging)

```
xanes_imaging.h5
├── /exchange/
│   ├── /data          [shape: (Nenergies, Ny, Nx), dtype: float32]
│   │                   # Absorption maps at each energy
│   ├── /energy        [shape: (Nenergies,), dtype: float64]
│   │                   # Energy values (eV)
│   └── /positions/
│       ├── /x         [shape: (Nx,)]
│       └── /y         [shape: (Ny,)]
│
├── /metadata/
│   ├── /element       # "Fe"
│   ├── /edge          # "K"
│   ├── /edge_energy   # 7112.0 eV
│   └── /beamline      # "20-BM"
│
└── /references/
    ├── /standard_1/
    │   ├── /name      # "Fe2O3 (hematite)"
    │   ├── /spectrum   [shape: (Nenergies,)]
    │   └── /oxidation_state  # "Fe(III)"
    └── /standard_2/
        ├── /name      # "FeO (wüstite)"
        └── ...
```

### Athena Project Files (.prj)

Athena (part of Demeter/Larch suite) uses its own project format:
- Stores multiple spectra with processing parameters
- Contains normalization, background subtraction settings
- Can export to standard ASCII, JSON

## Python Data Loading

```python
import numpy as np

# Load ASCII XAS data
def load_xas_ascii(filename):
    """Load standard XAS ASCII data file."""
    data = np.loadtxt(filename, comments='#')
    energy = data[:, 0]      # eV
    i0 = data[:, 1]
    i1 = data[:, 2]          # transmission
    i_fluor = data[:, 3]     # fluorescence

    # Calculate absorption coefficient
    mu_trans = -np.log(i1 / i0)
    mu_fluor = i_fluor / i0

    return energy, mu_trans, mu_fluor

# Load using larch (comprehensive XAS library)
import larch
from larch.io import read_ascii
from larch.xafs import pre_edge, autobk

group = read_ascii('fe_spectrum.dat')
pre_edge(group)         # normalize: sets group.norm, group.edge_step
autobk(group)           # background subtraction: sets group.chi, group.k

# Access processed data
energy = group.energy   # eV
norm = group.norm       # normalized µ(E)
k = group.k             # wavenumber (Å⁻¹)
chi_k = group.chi       # χ(k) EXAFS oscillations

# Load µ-XANES imaging
import h5py

with h5py.File('xanes_imaging.h5', 'r') as f:
    energy_stack = f['/exchange/data'][:]    # (Nenergies, Ny, Nx)
    energies = f['/exchange/energy'][:]       # (Nenergies,)
```

## Processing Steps

### 1. Energy Calibration
- Align edge energy to known standard value
- Correct for monochromator drift between scans
- Critical for accurate speciation

### 2. Normalization (Pre-edge / Post-edge)
```python
# Normalize XAS spectrum:
# 1. Fit pre-edge region (linear/polynomial) → extrapolate as baseline
# 2. Fit post-edge region (linear/polynomial) → extrapolate as post-edge line
# 3. Normalized µ(E) = [µ(E) - pre_edge(E)] / [post_edge(E₀) - pre_edge(E₀)]
```

### 3. Background Subtraction (for EXAFS)
```python
# AUTOBK algorithm (Newville et al.):
# - Fit smooth spline to post-edge µ(E) as µ₀(E)
# - χ(E) = [µ(E) - µ₀(E)] / Δµ₀
# - Convert E → k: k = √(2m(E-E₀)/ℏ²)
# - Weight: χ(k) × k^n (n = 1, 2, or 3)
```

### 4. Fourier Transform (EXAFS → Real Space)
```python
# FT of k-weighted χ(k) gives radial distribution function:
# χ̃(R) = FT[k^n × χ(k) × W(k)]
# Peaks correspond to coordination shells (shifted from true distance)
```

### 5. Linear Combination Fitting (LCF)
```python
# Fit unknown spectrum as weighted sum of reference standards:
# µ_sample(E) = Σᵢ fᵢ × µ_ref_i(E) + residual
# subject to: Σ fᵢ = 1, fᵢ ≥ 0
#
# Provides: fraction of each chemical species in the sample
```

## Key Software

| Software | Description | Language |
|----------|-------------|---------|
| **Larch** | Comprehensive XAS analysis (successor to IFEFFIT) | Python |
| **Athena** | GUI for XANES processing (Demeter package) | Perl/GUI |
| **Artemis** | GUI for EXAFS fitting | Perl/GUI |
| **FEFF** | Ab initio XAFS calculation | Fortran |
| **FDMNES** | XANES simulation from crystal structure | Fortran |
