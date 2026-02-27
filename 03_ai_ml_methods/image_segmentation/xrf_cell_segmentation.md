# XRF Cell Segmentation

## Overview

Cell segmentation in XRF microscopy involves identifying and delineating individual
biological cells from multi-element fluorescence maps. This is a critical preprocessing
step for cell-level analysis, enabling per-cell elemental quantification and classification.

## ROI-Finder Segmentation Pipeline

The ROI-Finder tool (Chowdhury et al., 2022) implements a classical image processing
pipeline for cell segmentation in XRF data.

### Pipeline Steps

```
Multi-element XRF map
    │
    ├─→ 1. Channel selection (choose high-contrast element)
    │
    ├─→ 2. Preprocessing (normalization, smoothing)
    │
    ├─→ 3. Otsu thresholding (binary: cell vs background)
    │
    ├─→ 4. Morphological operations (clean up)
    │
    ├─→ 5. Connected component labeling (identify individual cells)
    │
    └─→ 6. Area/shape filtering (remove debris, merge fragments)
```

### Step 1: Channel Selection

Choose an elemental channel with good cell-vs-background contrast:

```python
import numpy as np

# Evaluate contrast for each element channel
def channel_contrast(channel):
    """Estimate cell-background contrast using coefficient of variation."""
    return np.std(channel) / (np.mean(channel) + 1e-10)

# Select best channel
contrasts = {elem: channel_contrast(maps[elem]) for elem in elements}
best_channel = max(contrasts, key=contrasts.get)
# Typically Zn, Fe, or P give best contrast for biological cells
```

**Common choices**:
- **Zn**: Strong in most biological cells, good contrast
- **Fe**: High in certain cell types (e.g., red blood cells)
- **P**: Ubiquitous in cells (nucleic acids), moderate contrast
- **Composite**: Max(normalized channels) or PCA first component

### Step 2: Preprocessing

```python
from skimage import filters, exposure

# Normalize to [0, 255] for OpenCV compatibility
channel_norm = exposure.rescale_intensity(channel, out_range=(0, 255)).astype(np.uint8)

# Optional: Gaussian smoothing to reduce noise
channel_smooth = filters.gaussian(channel_norm, sigma=1.0)
```

### Step 3: Otsu Thresholding

```python
import cv2

# Otsu's method: automatically determines optimal threshold
threshold_value, binary = cv2.threshold(
    channel_uint8, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
# binary: 255 where cell, 0 where background
```

**How Otsu works**:
- Assumes bimodal intensity histogram (cells + background)
- Finds threshold that minimizes intra-class variance
- Equivalent to maximizing inter-class variance
- Fully automatic (no parameter tuning needed)

**Limitation**: Fails when histogram is not bimodal (low contrast, many cell types)

### Step 4: Morphological Operations

```python
# Define structuring element
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Opening: remove small noise (erosion → dilation)
cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# Closing: fill small holes (dilation → erosion)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

# Optional: dilation to slightly enlarge cell boundaries
dilated = cv2.dilate(cleaned, kernel, iterations=1)
```

**Morphological operations explained**:
- **Erosion**: Shrinks foreground (removes thin protrusions, noise)
- **Dilation**: Grows foreground (fills small holes, connects fragments)
- **Opening** (erosion → dilation): Removes small bright noise
- **Closing** (dilation → erosion): Fills small dark holes

### Step 5: Connected Component Labeling

```python
from skimage import measure

# Label connected regions
labels = measure.label(cleaned)
# labels: 0 = background, 1 = cell_1, 2 = cell_2, ...

regions = measure.regionprops(labels)
print(f"Found {len(regions)} connected components")
```

### Step 6: Area/Shape Filtering

```python
# Filter by area (remove debris and oversized objects)
min_area = 50     # pixels (adjust based on expected cell size)
max_area = 5000   # pixels

filtered_labels = np.zeros_like(labels)
cell_id = 1
for region in regions:
    if min_area < region.area < max_area:
        # Optional: filter by shape (eccentricity, solidity)
        if region.eccentricity < 0.95 and region.solidity > 0.5:
            filtered_labels[labels == region.label] = cell_id
            cell_id += 1

print(f"Retained {cell_id - 1} cells after filtering")
```

## Complete Pipeline Code

```python
import numpy as np
import cv2
from skimage import measure, exposure, filters

def segment_cells_xrf(elemental_maps, channel='Zn',
                       min_area=50, max_area=5000,
                       morph_kernel_size=5, gaussian_sigma=1.0):
    """
    Segment cells from XRF elemental maps using ROI-Finder-style pipeline.

    Parameters
    ----------
    elemental_maps : dict
        {element_name: 2D numpy array}
    channel : str
        Element to use for segmentation
    min_area, max_area : int
        Cell area bounds in pixels
    morph_kernel_size : int
        Size of morphological structuring element
    gaussian_sigma : float
        Gaussian smoothing sigma

    Returns
    -------
    labels : 2D numpy array
        Labeled cell mask (0 = background)
    properties : list
        Region properties for each cell
    """
    # 1. Select and preprocess channel
    img = elemental_maps[channel].copy()
    img = filters.gaussian(img, sigma=gaussian_sigma)
    img_uint8 = exposure.rescale_intensity(img, out_range=(0, 255)).astype(np.uint8)

    # 2. Otsu threshold
    _, binary = cv2.threshold(img_uint8, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (morph_kernel_size, morph_kernel_size))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    # 4. Label and filter
    all_labels = measure.label(cleaned)
    regions = measure.regionprops(all_labels)

    filtered = np.zeros_like(all_labels)
    cell_id = 1
    kept_regions = []
    for r in regions:
        if min_area < r.area < max_area:
            filtered[all_labels == r.label] = cell_id
            kept_regions.append(r)
            cell_id += 1

    return filtered, kept_regions
```

## Limitations of the Classical Pipeline

### 1. Overlapping/Touching Cells
- Connected component labeling merges touching cells into one region
- **Mitigation**: Watershed segmentation on distance transform
```python
from scipy import ndimage

# Distance transform → watershed for splitting touching cells
dist = ndimage.distance_transform_edt(cleaned)
local_max = measure.label(dist > 0.5 * dist.max())
labels = measure.watershed(-dist, local_max, mask=cleaned)
```

### 2. Variable Cell Brightness
- Otsu assumes bimodal distribution — fails with multiple cell types
- **Mitigation**: Adaptive thresholding or multi-Otsu
```python
# Adaptive thresholding
adaptive = cv2.adaptiveThreshold(img_uint8, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 51, -5)
```

### 3. Low SNR Channels
- Elements at trace concentrations may have poor signal-to-noise
- Segmentation on low-SNR channels produces noisy boundaries
- **Mitigation**: Use high-SNR channel for segmentation, apply mask to all channels

### 4. Non-Cellular Samples
- Pipeline designed for isolated cells on flat support
- Fails for tissue sections, biofilms, soil particles
- Need different approaches for non-cellular targets

## Advanced Alternatives

| Method | Strengths | Limitations |
|--------|-----------|-------------|
| **Cellpose** | Pre-trained, handles diverse cell shapes | Needs fine-tuning for XRF data |
| **StarDist** | Fast, accurate for convex cells | Assumes star-convex shape |
| **Mask R-CNN** | Instance segmentation, bounding boxes | Needs more training data |
| **SAM** | Zero-shot, prompt-based | Not optimized for XRF contrast |
| **Watershed** | Splits touching cells | Over-segmentation risk |

## Evaluation Metrics

```python
from skimage.metrics import variation_of_information

# Jaccard Index (IoU) per cell
def cell_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / (union + 1e-10)

# Mean IoU across all cells
# Detection rate: fraction of GT cells with IoU > 0.5 match
# False positive rate: fraction of predicted cells without GT match
```
