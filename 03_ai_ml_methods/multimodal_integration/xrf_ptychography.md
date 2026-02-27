# XRF + Ptychography: Simultaneous Structure and Elemental Mapping

## Overview

At beamline **2-ID-E** (and future 33-ID-C), XRF and ptychography data are collected
simultaneously: the same focused X-ray beam generates both fluorescence signal and
coherent diffraction patterns at each scan position.

## Experimental Setup

```
Focused coherent X-ray beam
    │
    ▼
Sample on scanning stage
    │
    ├──→ Fluorescence detector (90°)  → Elemental maps (XRF)
    │         Vortex SDD
    │
    └──→ Pixelated area detector       → Phase + amplitude (ptychography)
              (far field)
              Eiger 500K
```

Both signals from the same scan position → **naturally co-registered**.

## Data Structure

```
combined_scan.h5
├── /xrf/
│   ├── /spectra       [shape: (Npos, Nchannels)]   # Raw XRF spectra
│   ├── /elemental_maps/
│   │   ├── /Fe        [shape: (Ny, Nx)]
│   │   ├── /Zn        [shape: (Ny, Nx)]
│   │   └── ...
│   └── /positions     [shape: (Npos, 2)]
│
├── /ptychography/
│   ├── /diffraction   [shape: (Npos, Ndet_y, Ndet_x)]  # Diffraction patterns
│   └── /positions     [shape: (Npos, 2)]  # Same as XRF positions
│
└── /reconstructed/
    ├── /phase          [shape: (Ny_recon, Nx_recon)]
    ├── /amplitude      [shape: (Ny_recon, Nx_recon)]
    └── /pixel_size     # nm
```

## Integration Approaches

### 1. Overlay Visualization

Simplest approach: overlay XRF elemental maps on ptychographic phase image.

```python
import matplotlib.pyplot as plt
import numpy as np

# Ptychography provides structural context
# XRF provides elemental identity

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(phase_image, cmap='gray', alpha=1.0)
ax.imshow(fe_map_resampled, cmap='hot', alpha=0.5)
ax.set_title('Phase (structure) + Fe distribution (XRF)')
```

### 2. Correlated Analysis

```python
# Extract structural features at each XRF pixel
# Correlate elemental concentration with structural properties

for cell_region in segmented_cells:
    # From ptychography: phase contrast value, texture features
    phase_mean = phase_image[cell_region].mean()
    phase_std = phase_image[cell_region].std()

    # From XRF: elemental concentrations
    fe_conc = fe_map[cell_region].mean()
    zn_conc = zn_map[cell_region].mean()

    # Correlation: do Fe-rich regions have different density?
```

### 3. Joint Reconstruction (Advanced)

Reconstruct XRF and ptychographic images simultaneously, sharing the
object model:

```
Shared object model:
  - Thickness t(x,y) ← constrains both modalities
  - Composition c_i(x,y) ← element concentrations from XRF
  - Phase φ(x,y) = Σ_i c_i(x,y) × δ_i × t(x,y)  ← physics linking phase to composition

Joint optimization:
  min ||I_ptycho - |F{P·O}|²||² + ||I_xrf - σ_xrf(c_i, t)||²
```

## Benefits

1. **Natural co-registration**: Same scan → no alignment needed
2. **Complementary information**: Structure (ptycho) + composition (XRF)
3. **Resolution bridging**: Ptychography at 10 nm, XRF at 50-100 nm
4. **Quantitative**: Both provide quantitative measurements
5. **Self-consistent**: Joint reconstruction constrains both modalities

## Challenges

1. **Resolution mismatch**: Ptychography has finer resolution than XRF
2. **Different SNR**: XRF may have low counts for trace elements
3. **Computational**: Joint reconstruction is computationally expensive
4. **Self-absorption**: XRF signal affected by sample thickness and composition
5. **Limited algorithms**: Few tools support true joint analysis

## Future Directions

- Deep learning for joint XRF + ptychography analysis
- Transfer structural features from ptychography to improve XRF segmentation
- Multi-modal foundation models trained on paired datasets
- Real-time joint analysis during scanning
