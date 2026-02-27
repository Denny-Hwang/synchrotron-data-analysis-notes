# AI/ML Methods for XRF Microscopy

## Overview

XRF microscopy generates rich multi-element spatial datasets that are well-suited for
machine learning analysis. Key ML applications include automated ROI selection,
cell/particle segmentation, unsupervised clustering of element associations, and
resolution enhancement.

## ML Problem Classification

| Problem | Type | Input | Output |
|---------|------|-------|--------|
| ROI selection | Recommendation | Multi-element maps | Priority ROI list |
| Cell segmentation | Instance segmentation | Elemental maps | Cell boundaries + labels |
| Clustering | Unsupervised | Multi-element pixel vectors | Cluster assignments |
| Spectral fitting | Regression | Raw spectrum | Element concentrations |
| Resolution enhancement | Image-to-image | Low-res map | High-res map |
| Anomaly detection | Classification | Pixel spectrum | Normal/anomalous |

## ROI-Finder

**ROI-Finder** is the flagship ML tool for XRF microscopy at APS, developed for
autonomous experiment steering during beam time.

### Workflow
```
Coarse XRF scan (large area, low resolution)
    │
    ├─→ Cell segmentation
    │       Otsu thresholding → morphological ops → connected components
    │
    ├─→ Feature extraction (per cell)
    │       PCA on multi-element vectors → reduced feature space
    │
    ├─→ Clustering
    │       Fuzzy k-means → soft membership assignment
    │
    └─→ ROI Recommendation
            Rank cells by diversity/interest → suggest for detailed scanning
```

### Key Innovation
- Unsupervised: no labeled training data needed
- Operates during beam time: guides experimentalist to most interesting regions
- Interactive: GUI for annotation and parameter adjustment

**Reference**: Chowdhury et al., J. Synchrotron Rad. 29 (2022), DOI: 10.1107/S1600577522008876

*See detailed analysis: [05_tools_and_code/roi_finder/](../../05_tools_and_code/roi_finder/)*

## Cell Segmentation Methods

### Binary Thresholding Pipeline (ROI-Finder)

```python
import cv2
import numpy as np
from skimage import morphology, measure

# 1. Select high-contrast element channel (e.g., Zn)
channel = elemental_maps['Zn']

# 2. Otsu thresholding
threshold = cv2.threshold(channel_uint8, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# 3. Morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cleaned = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)   # remove noise
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)     # fill holes

# 4. Connected component labeling
labels = measure.label(cleaned)

# 5. Filter by area/shape
regions = measure.regionprops(labels)
filtered = [r for r in regions if min_area < r.area < max_area]
```

### Limitations
- Fails for overlapping/touching cells (need watershed or instance segmentation)
- Sensitive to channel choice and SNR
- Cannot handle variable cell brightness

### Advanced Alternatives
- **Cellpose**: Pre-trained instance segmentation for cells (generalist model)
- **Mask R-CNN**: Instance segmentation with bounding boxes
- **StarDist**: Star-convex polygon detection for convex cells
- **U-Net**: Semantic segmentation (cell vs. background)

## Unsupervised Clustering

### GMM (Gaussian Mixture Model)

Ward et al. (2013) applied GMM to XRF data from malaria-infected red blood cells:

```python
from sklearn.mixture import GaussianMixture

# Prepare pixel-wise multi-element feature vectors
# shape: (Npixels, Nelements)
X = np.column_stack([maps[e].flatten() for e in selected_elements])

# Fit GMM
gmm = GaussianMixture(n_components=k, covariance_type='full')
gmm.fit(X)

# Soft cluster assignments
probabilities = gmm.predict_proba(X)  # shape: (Npixels, k)
labels = gmm.predict(X)               # shape: (Npixels,)
```

**Advantages**:
- Soft clustering (probabilistic membership)
- Models ellipsoidal clusters (captures element correlations)
- BIC/AIC for optimal k selection

### Fuzzy K-Means (ROI-Finder)

```python
import skfuzzy as fuzz

# Fuzzy c-means clustering
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X.T, c=k, m=2, error=0.005, maxiter=1000
)
# u: membership matrix, shape: (k, Npixels)
# cntr: cluster centers, shape: (k, Nelements)
```

**Advantages**:
- Soft membership (each pixel belongs partially to multiple clusters)
- Better handles transitional regions between cell types
- Fuzzy partition coefficient (FPC) for quality assessment

### Other Clustering Approaches
- **HDBSCAN**: Density-based, finds clusters of arbitrary shape, handles noise
- **Spectral clustering**: Graph-based, captures non-linear relationships
- **Self-Organizing Maps (SOM)**: Topological dimensionality reduction + clustering

## Feature Extraction

### PCA-Based (ROI-Finder Approach)
```
Multi-element maps (Ny, Nx, Nelements)
    → Reshape to (Npixels, Nelements)
    → Standardize
    → PCA → Top k components
    → Per-cell mean features (aggregated within segmented regions)
```

### Deep Learning Alternatives
- **Autoencoders**: Learn compressed representation of multi-element spectra
- **Contrastive learning**: Self-supervised features from augmented spectral data
- **VAE**: Variational autoencoders for generative modeling of spectral distributions

## Resolution Enhancement

### Deep Residual Networks for XRF

**Reference**: npj Computational Materials (2023), DOI: 10.1038/s41524-023-00995-9

Enhances XRF spatial resolution beyond the beam probe size:

```
Low-resolution XRF map → Deep Residual Network → Enhanced-resolution map
```

**Method**:
- Train on paired low-res (large step) / high-res (fine step) scans
- Residual learning: network predicts the difference (high-res - upsampled low-res)
- Effectively deconvolves the probe profile

**Results**: 2-4× effective resolution improvement demonstrated

## Current Limitations

1. **Segmentation**: Standard methods struggle with overlapping cells and variable contrast
2. **Feature engineering**: PCA may miss non-linear element associations
3. **Training data**: Few labeled XRF datasets exist for supervised ML
4. **3D XRF**: ML methods mostly limited to 2D; 3D tomographic XRF needs development
5. **Quantification uncertainty**: ML-based spectral fitting lacks rigorous uncertainty estimates

## Improvement Opportunities

1. **Instance segmentation**: Cellpose/StarDist adapted for XRF elemental maps
2. **Self-supervised pre-training**: Learn features from unlabeled XRF data
3. **Multimodal fusion**: Joint XRF + ptychography analysis
4. **Active learning**: ML suggests which cells to label for maximum model improvement
5. **Real-time streaming**: ML-guided scanning with sub-second decision making
6. **Physics-informed ML**: Incorporate XRF physics (self-absorption, matrix effects) into networks
