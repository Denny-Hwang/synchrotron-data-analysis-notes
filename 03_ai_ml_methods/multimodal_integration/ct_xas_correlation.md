# CT + XAS Correlation: Structural and Chemical Speciation

## Overview

Correlating X-ray computed tomography (CT) with X-ray absorption spectroscopy (XAS)
combines 3D structural information with chemical speciation, revealing not only what
a sample looks like in 3D but also the chemical state of specific elements within it.

## Approaches

### 1. Sequential Measurement

```
Step 1: µCT scan → 3D volume (morphology, density)
Step 2: µ-XANES → 2D speciation maps (select slices or ROIs)
Step 3: Register CT and XANES data
Step 4: Correlate structure with chemistry
```

**Beamlines**: µCT at 2-BM-A → XANES at 20-BM (same sample, different instruments)

### 2. Spectroscopic Tomography (XANES-CT)

Collect tomographic scans at multiple energies spanning an absorption edge:

```
For each energy E_i (spanning edge, e.g., Fe K-edge 7100-7200 eV):
    Collect full tomographic scan (900+ projections)
    Reconstruct 3D volume at energy E_i

Stack: µ(x,y,z,E) → 4D dataset (spatial + spectral)
Extract XANES spectrum at each voxel → 3D speciation map
```

**Challenge**: Requires N_energies × full tomographic scans (extremely time-intensive)

### 3. Dual-Energy CT

Simplified version using only 2-3 energies:

```
Scan at E_below_edge → µ₁(x,y,z)
Scan at E_above_edge → µ₂(x,y,z)

Ratio map: R(x,y,z) = µ₂/µ₁
→ Highlights regions containing the target element
→ Ratio value indicates oxidation state (for well-separated edge energies)
```

## Integration Workflow

```python
import numpy as np
from skimage import measure

# Load CT volume and XANES speciation map
ct_volume = load_ct('ct_scan.h5')           # shape: (Nz, Ny, Nx)
xanes_map = load_xanes('xanes_scan.h5')     # shape: (Ny_xanes, Nx_xanes, N_energies)

# Register XANES to CT coordinate system
from skimage.transform import AffineTransform, warp
# Apply known transformation (from fiducial markers or manual alignment)

# Segment CT into structural phases
labels = segment_ct(ct_volume)  # e.g., pore=0, mineral=1, organic=2

# For each phase, extract XANES spectra
for phase_id in [1, 2]:
    mask = labels == phase_id
    # Get XANES spectra within this structural phase
    phase_spectra = extract_spectra_in_region(xanes_map, mask)
    # Average and fit to determine chemical speciation
    avg_spectrum = phase_spectra.mean(axis=0)
    fractions = linear_combination_fit(avg_spectrum, references)
    print(f"Phase {phase_id}: {fractions}")
```

## Scientific Applications

### Soil Science Example
```
µCT reveals:
  - Mineral grain locations and sizes
  - Pore network structure
  - Organic matter aggregates

XANES at Fe K-edge reveals:
  - Fe(II) vs Fe(III) distribution
  - Fe-organic complexes vs crystalline Fe minerals
  - Redox gradients within aggregates

Correlation:
  - Fe(II) preferentially at pore-mineral interfaces
  - Fe(III) in well-connected pore regions (oxidized)
  - Fe-organic complexes in organic-rich aggregates
```

## ML Opportunities

1. **Automated registration**: CNN-based image alignment between CT and XANES
2. **Super-resolution speciation**: Predict fine-scale speciation from coarse XANES + detailed CT
3. **Transfer learning**: Use CT morphology to predict likely chemical states
4. **Sparse XANES-CT**: Use ML to reconstruct full spectroscopic tomography from fewer energies
5. **Joint segmentation**: Segment CT and XANES simultaneously using multi-channel networks

## Challenges

1. **Registration accuracy**: Sub-pixel alignment required between modalities
2. **Time**: Full XANES-CT takes hours to days
3. **Dose**: Multiple scans increase cumulative radiation dose
4. **Resolution mismatch**: CT often higher resolution than XANES
5. **Self-absorption**: XANES fluorescence signal affected by sample density/thickness
