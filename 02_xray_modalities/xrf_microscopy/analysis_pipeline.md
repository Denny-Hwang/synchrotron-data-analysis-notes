# XRF Microscopy Analysis Pipeline

## Overview

The XRF analysis pipeline transforms raw detector spectra into quantitative elemental
maps and ultimately into scientific insights about spatial element distributions.

## Pipeline Stages

```
Raw detector data (spectra per pixel)
    │
    ├─→ 1. Preprocessing (dead time, normalization)
    │
    ├─→ 2. Spectral fitting (MAPS, PyXRF)
    │       Output: elemental intensity maps
    │
    ├─→ 3. Quantification (fundamental parameters)
    │       Output: concentration maps (µg/cm²)
    │
    ├─→ 4. Spatial analysis
    │       ├── Segmentation (cells, grains, phases)
    │       ├── Correlation analysis (element co-localization)
    │       └── Statistical analysis (PCA, clustering)
    │
    └─→ 5. Interpretation
            ├── Element association identification
            ├── Chemical speciation inference
            └── Biological/geological process interpretation
```

## MAPS Software

**MAPS** (Microanalysis Toolkit) is the primary XRF analysis software at APS,
developed by the X-ray Science Division.

### Features
- Full-spectrum fitting using fundamental parameters
- Multiple fitting methods: ROI, Gaussian, SVD, NNLS
- Batch processing of large scan datasets
- Per-pixel and averaged spectrum analysis
- Absorption correction for self-absorption effects
- Export to HDF5, TIFF, ASCII

### Spectral Fitting Methods

#### ROI (Region of Interest)
```
intensity = Σ (counts[ch_low : ch_high])
```
- Simplest method: sum counts in energy window around expected peak
- Fast but susceptible to peak overlap and background variation
- Best for well-separated, strong peaks

#### Gaussian Peak Fitting
```
I(E) = Σ_elements [ A_i × G(E; E_i, σ) ] + background(E)

where G(E; E_i, σ) = (1/σ√2π) × exp(-(E-E_i)²/2σ²)
```
- Fits each peak with Gaussian profile
- Handles moderate peak overlap
- Provides peak area, position, and width

#### Full-Spectrum Fitting (NNLS)
```
Measured spectrum = Σ_i (c_i × reference_spectrum_i) + residual

Minimize ||measured - Σ c_i × ref_i||² subject to c_i ≥ 0
```
- Uses reference spectra (computed from fundamental parameters or measured standards)
- Best handling of peak overlaps, escape peaks, scatter
- Most accurate quantification

### MAPS Quantification

Fundamental parameters approach:
```
C_element = I_measured / (I_theoretical × T × Ω × ε)

where:
  I_theoretical = theoretical yield per unit concentration
  T = sample transmission correction
  Ω = detector solid angle
  ε = detector efficiency at fluorescence energy
```

## Spatial Analysis Methods

### Element Co-localization

Quantify spatial correlation between element pairs:

```python
# Pearson correlation between element maps
from scipy.stats import pearsonr

r_Fe_P, p_value = pearsonr(fe_map.flatten(), p_map.flatten())
# r > 0.5: co-localized, r ≈ 0: independent, r < -0.5: anti-correlated
```

### RGB Composite Maps

Assign three elements to R, G, B channels for visualization:
- **Biological**: R=Fe, G=Zn, B=P (common for cell studies)
- **Environmental**: R=Fe, G=Ca, B=S (for soil/mineral studies)
- Co-localization appears as color mixing (e.g., Fe+Zn = yellow)

### PCA on Multi-Element Maps

Principal Component Analysis reduces dimensionality of multi-element data:

```python
from sklearn.decomposition import PCA

# Stack all element maps: shape (Npixels, Nelements)
X = np.column_stack([maps[e].flatten() for e in elements])
X_standardized = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)

pca = PCA(n_components=5)
scores = pca.fit_transform(X_standardized)
loadings = pca.components_

# Reshape scores back to image dimensions
pc1_map = scores[:, 0].reshape(ny, nx)
```

PCA identifies:
- **PC1**: Usually total signal intensity (brightness)
- **PC2+**: Element association patterns (e.g., Fe-P vs Ca-S groupings)
- Loadings reveal which elements contribute to each component

### Clustering

Group pixels by elemental composition:

- **K-means**: Hard clustering, each pixel assigned to one cluster
- **Fuzzy k-means**: Soft clustering, membership probabilities per cluster (used in ROI-Finder)
- **GMM**: Gaussian Mixture Model, probabilistic cluster assignment
- **HDBSCAN**: Density-based, identifies clusters of arbitrary shape

## Advanced Analysis

### µ-XANES Imaging

Combined XRF + XANES: collect full XANES spectrum at each pixel
- Map chemical speciation spatially
- Requires scanning energy at each position (very slow)
- Alternative: select 2-3 key energies for ratio mapping

### Tomographic XRF

3D elemental mapping by:
1. Collect XRF maps at multiple rotation angles
2. Reconstruct 3D elemental distribution using filtered back projection
3. Challenge: self-absorption correction in 3D

### Correlative Imaging

Combine XRF with:
- **Ptychography**: Structure + composition (same beamline: 2-ID-E)
- **XANES**: Speciation + distribution (same beamline: 8-BM-B)
- **Optical microscopy**: Morphology context + ROI selection (ROI-Finder approach)

## Output Formats

| Output | Format | Content |
|--------|--------|---------|
| Elemental maps | HDF5, TIFF | 2D arrays per element |
| Spectra | HDF5, ASCII | Energy-intensity pairs |
| Quantified maps | HDF5, CSV | Concentration values (µg/cm²) |
| Correlation matrix | CSV | Element-element correlations |
| PCA results | HDF5 | Score maps + loading vectors |
