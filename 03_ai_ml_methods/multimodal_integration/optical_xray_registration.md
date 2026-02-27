# Optical-X-ray Image Registration

## Overview

Registration between optical microscope images and X-ray data enables:
1. Using optical images to pre-select ROIs before X-ray measurements
2. Correlating visible-light features with elemental/structural X-ray data
3. Navigating to specific features identified optically during beam time

This is particularly relevant for ROI-Finder's planned roadmap of integrating
optical microscopy with XRF scanning.

## The Registration Problem

```
Optical microscope image          X-ray map (XRF, ptychography, etc.)
┌─────────────────────┐           ┌──────────────────────┐
│  High resolution     │           │  Lower resolution     │
│  Color (RGB)         │   Register │  Multi-channel        │
│  Large FOV          │ ═══════════│  Smaller FOV          │
│  Surface morphology │           │  Bulk composition      │
│  No beam time needed│           │  Requires beam time    │
└─────────────────────┘           └──────────────────────┘
```

### Challenges

| Challenge | Description |
|-----------|------------|
| **Different contrast** | Optical: reflectance/absorption. X-ray: fluorescence/phase |
| **Resolution mismatch** | Optical: 0.3–1 µm. XRF: 0.05–20 µm |
| **FOV mismatch** | Optical: mm–cm. X-ray: µm–mm |
| **Distortion** | Different optical paths, sample mounting changes |
| **Feature correspondence** | Same features may look very different |

## Registration Methods

### 1. Fiducial Marker-Based

```
Place known markers (e.g., gold grid lines) on sample
    │
    ├─→ Identify markers in optical image → (x₁,y₁), (x₂,y₂), ...
    │
    ├─→ Identify markers in X-ray map → (x₁',y₁'), (x₂',y₂'), ...
    │
    └─→ Compute affine transformation:
         [x']   [a b tx] [x]
         [y'] = [c d ty] [y]
         [1 ]   [0 0 1 ] [1]

         Solving: T = argmin Σᵢ ||(x'ᵢ,y'ᵢ) - T(xᵢ,yᵢ)||²
```

**Pros**: Reliable, well-defined correspondence points
**Cons**: Requires markers, may obscure sample regions

### 2. Feature-Based Registration

```python
import cv2
import numpy as np

# Detect features in both images
sift = cv2.SIFT_create()
kp1, desc1 = sift.detectAndCompute(optical_image, None)
kp2, desc2 = sift.detectAndCompute(xray_image, None)

# Match features
bf = cv2.BFMatcher()
matches = bf.knnMatch(desc1, desc2, k=2)

# Lowe's ratio test
good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

# Compute transformation
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Apply transformation
registered = cv2.warpPerspective(optical_image, M, xray_image.shape[::-1])
```

**Pros**: No markers needed, automatic
**Cons**: Requires common features visible in both modalities (often difficult)

### 3. Mutual Information-Based

For images with different contrast mechanisms:

```python
from skimage.registration import phase_cross_correlation
from scipy.optimize import minimize

def mutual_information(image1, image2):
    """Compute mutual information between two images."""
    hist_2d, _, _ = np.histogram2d(
        image1.ravel(), image2.ravel(), bins=50
    )
    # Normalize
    pxy = hist_2d / hist_2d.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    # MI = H(X) + H(Y) - H(X,Y)
    hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
    hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
    hxy = -np.sum(pxy[pxy > 0] * np.log(pxy[pxy > 0]))

    return hx + hy - hxy

# Optimize transformation to maximize mutual information
# (handles different contrast mechanisms)
```

### 4. Deep Learning Registration

```
Optical image ──→ CNN feature extractor ──→ Features A ──┐
                                                          ├─→ Spatial Transformer
X-ray image ───→ CNN feature extractor ──→ Features B ──┘     → Transformation
                                                                parameters
```

- **VoxelMorph**: Originally for medical image registration
- **DeepReg**: General-purpose deep learning registration framework
- **Self-supervised**: Train on augmented single-modality data, apply cross-modality

## ROI-Finder Integration Roadmap

### Current State
- ROI-Finder operates only on XRF data
- Optical images used manually for navigation

### Planned Enhancement
```
1. Pre-beam time: Optical microscopy of sample
   → Identify candidate cells/regions optically
   → Store coordinates relative to fiducial markers

2. At beamline: Quick coarse XRF scan
   → Register with optical image
   → Transfer optical ROI selections to XRF coordinate system

3. During beam time:
   → Scan pre-selected ROIs at high resolution
   → Optionally run ROI-Finder on XRF to refine selection
```

### Benefits
- Save beam time by pre-screening optically (free and fast)
- Better ROI selection by combining morphological (optical) and elemental (XRF) information
- Track same cells across multiple experiments (optical identification)

## Challenges and Future Directions

1. **Cross-modal learning**: Train models that understand correspondence between optical and X-ray features
2. **Multi-scale registration**: Handle large resolution differences (100× or more)
3. **Deformable registration**: Account for sample changes between optical and X-ray imaging
4. **Real-time**: Registration during beam time for immediate ROI transfer
5. **3D**: Register optical surface images with 3D X-ray tomography data
