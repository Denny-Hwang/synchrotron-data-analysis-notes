# Tomography Segmentation

## Overview

Segmentation of reconstructed tomographic volumes identifies distinct phases, features,
or structures within the 3D data. This is essential for quantitative analysis such as
porosity measurement, grain size distribution, and morphological characterization.

## Common Segmentation Targets

| Target | Application | Challenges |
|--------|-------------|-----------|
| **Pores** | Soil porosity, rock permeability | Small pores near resolution limit |
| **Grains/particles** | Soil aggregate structure | Touching particles |
| **Mineral phases** | Geochemistry, petrology | Similar attenuation coefficients |
| **Biological tissue** | Root anatomy, biofilm | Low contrast, radiation damage |
| **Cracks/fractures** | Geomechanics | Thin features, partial volume |
| **Fluid/gas** | Multiphase flow | Dynamic, low contrast |

## Traditional Methods

### Global Thresholding (Otsu)

```python
from skimage.filters import threshold_otsu

# Simple binary segmentation
thresh = threshold_otsu(volume)
binary = volume > thresh
# binary: True = high-density phase, False = low-density/pore
```

Best for: High-contrast, two-phase systems (e.g., pore vs. solid)

### Multi-Otsu (Multi-Phase)

```python
from skimage.filters import threshold_multiotsu

# Segment into 3 phases (e.g., pore, organic, mineral)
thresholds = threshold_multiotsu(volume, classes=3)
regions = np.digitize(volume, bins=thresholds)
# regions: 0, 1, or 2 for each voxel
```

### Watershed Segmentation

For separating touching objects (grains, particles):

```python
from scipy import ndimage
from skimage import morphology, segmentation

# 1. Binary threshold
binary = volume > threshold_otsu(volume)

# 2. Distance transform
distance = ndimage.distance_transform_edt(binary)

# 3. Find local maxima (seeds)
local_max = morphology.local_maxima(distance)
markers = measure.label(local_max)

# 4. Watershed
labels = segmentation.watershed(-distance, markers, mask=binary)
```

### Random Walker

Graph-based semi-supervised segmentation:

```python
from skimage.segmentation import random_walker

# Provide seed labels for some voxels
labels = np.zeros_like(volume, dtype=int)
labels[volume < low_threshold] = 1    # pore
labels[volume > high_threshold] = 2   # solid

# Random walker fills in uncertain regions
segmented = random_walker(volume, labels, beta=100)
```

### Graph Cut (Max-Flow)

```python
import maxflow

# Define graph with data + smoothness terms
# Data term: likelihood of voxel belonging to each class
# Smoothness: penalty for neighboring voxels with different labels
# Optimal segmentation via max-flow/min-cut algorithm
```

## Deep Learning Methods

### 3D U-Net

Standard architecture for volumetric segmentation (see [unet_variants.md](unet_variants.md)).

**Patch-based strategy**:
```
Full volume (2048³) → Extract patches (128³) → Predict → Stitch
```

### V-Net

Variant of 3D U-Net with residual connections and Dice loss:
- Designed specifically for volumetric medical image segmentation
- Residual blocks improve gradient flow in deep networks
- Dice loss better handles class imbalance

### DLSIA (Deep Learning for Scientific Image Analysis)

Framework developed at LBNL for synchrotron data:

- **Mixed-Scale Dense (MSD) Networks**: Combine multi-scale features without pooling
- **Tunable U-Nets**: Parameterized architecture search
- **Scientific focus**: Designed for scientific (not medical) data characteristics

### Self-Supervised Approaches

Address the labeled data scarcity problem:

1. **Contrastive learning**: Learn features by comparing augmented views
2. **Masked autoencoder**: Predict masked regions of volume
3. **Pseudo-labeling**: Use traditional methods for initial labels, iteratively refine

## Challenges Specific to Synchrotron Tomography

### 1. Ring Artifacts

Residual ring artifacts after preprocessing can fool segmentation algorithms:
- Create false circular boundaries
- Affect threshold-based methods most severely
- DL methods can learn to ignore rings if present in training data

### 2. Partial Volume Effects

At material interfaces, voxels contain a mixture of phases:
```
True boundary:  Phase A | Phase B
Voxel values:   ... A  [A+B mixed]  B ...
```
- Creates intermediate gray values at boundaries
- Affects pore/grain boundary accuracy
- Mitigation: Sub-voxel segmentation, partial volume estimation

### 3. Beam Hardening (Polychromatic Sources)

Dense materials preferentially absorb low-energy X-rays:
- Cupping artifact: edges appear denser than center
- Creates false gradients in reconstructed volume
- Pre-correction needed before segmentation

### 4. Massive Volume Sizes

| Volume | Voxels | Size (32-bit) | GPU Fit? |
|--------|--------|---------------|----------|
| 512³ | 134M | 0.5 GB | Yes |
| 1024³ | 1.07B | 4 GB | Barely |
| 2048³ | 8.59B | 32 GB | No |
| 4096³ | 68.7B | 256 GB | No |

**Strategies**:
- Patch-based with overlap stitching
- Downsampled coarse pass → full-resolution refinement
- Distributed inference across multiple GPUs
- 2D slice-by-slice with 3D consistency post-processing

## Evaluation Metrics

| Metric | Formula | Best For |
|--------|---------|----------|
| **Dice coefficient** | 2|A∩B| / (|A|+|B|) | Overall overlap |
| **IoU (Jaccard)** | |A∩B| / |A∪B| | Per-class accuracy |
| **Surface distance** | Mean distance between boundaries | Boundary accuracy |
| **Volume fraction** | Volume_phase / Volume_total | Porosity measurement |
| **Euler number** | Topological connectivity | Pore network topology |

## Practical Workflow

```
Reconstructed 3D volume
    │
    ├─→ Preprocessing
    │       ├── Cropping (remove reconstruction edges)
    │       ├── Ring artifact suppression
    │       ├── Noise reduction (3D median or NLM filter)
    │       └── Intensity normalization
    │
    ├─→ Segmentation
    │       ├── Quick: Multi-Otsu thresholding
    │       ├── Standard: Watershed for individual particles
    │       └── Advanced: 3D U-Net (if training data available)
    │
    ├─→ Post-processing
    │       ├── Morphological cleanup (opening/closing)
    │       ├── Connected component filtering (remove small noise)
    │       ├── Hole filling
    │       └── Manual correction (if needed)
    │
    └─→ Quantitative analysis
            ├── Porosity (volume fraction of pore phase)
            ├── Pore size distribution (equivalent sphere diameter)
            ├── Connectivity (Euler number, percolation)
            ├── Grain size distribution
            └── Surface area (marching cubes)
```
