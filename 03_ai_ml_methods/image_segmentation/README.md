# Image Segmentation for Synchrotron Data

## Overview

Image segmentation partitions synchrotron images into meaningful regions (e.g., cells,
pores, grains, phases). It is one of the most critical analysis steps, as all downstream
quantification depends on accurate segmentation.

## Segmentation Types

| Type | Output | Example Application |
|------|--------|-------------------|
| **Semantic** | Pixel-wise class labels | Phase identification in µCT (pore/grain/mineral) |
| **Instance** | Individual object boundaries | Cell detection in XRF maps |
| **Panoptic** | Semantic + instance combined | Complete tissue/sample analysis |

## Method Comparison

| Method | Type | Training Data | Speed | Quality | Best For |
|--------|------|--------------|-------|---------|----------|
| **Otsu thresholding** | Traditional | None | ★★★★★ | ★★ | High-contrast binary |
| **Watershed** | Traditional | None | ★★★★ | ★★★ | Touching objects |
| **Random walker** | Traditional | Seeds | ★★★ | ★★★★ | Low contrast |
| **U-Net** | DL semantic | 50-500 labeled | ★★★★ | ★★★★★ | General purpose |
| **nnU-Net** | DL semantic | 50-500 labeled | ★★★ | ★★★★★ | Auto-configured |
| **Cellpose** | DL instance | Pre-trained | ★★★★ | ★★★★★ | Cell-like objects |
| **StarDist** | DL instance | Pre-trained | ★★★★ | ★★★★ | Convex objects |
| **Mask R-CNN** | DL instance | 100+ labeled | ★★★ | ★★★★ | Complex shapes |

## Key Challenges for Synchrotron Data

1. **Volume size**: 2048³ voxels = 8 billion voxels; cannot fit in GPU memory
2. **Label scarcity**: Expert annotation of synchrotron data is extremely expensive
3. **Domain shift**: Models trained on one beamline/sample may not generalize
4. **Class imbalance**: Features of interest are often a tiny fraction of the volume
5. **3D vs 2D**: True 3D segmentation vs. slice-by-slice with consistency

## Directory Contents

| File | Content |
|------|---------|
| [unet_variants.md](unet_variants.md) | U-Net architecture family and adaptations |
| [xrf_cell_segmentation.md](xrf_cell_segmentation.md) | Cell segmentation pipeline for XRF data |
| [tomography_segmentation.md](tomography_segmentation.md) | Phase/feature segmentation in µCT |
