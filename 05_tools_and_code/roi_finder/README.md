# ROI-Finder

## Overview

**ROI-Finder** is an open-source Python tool for ML-guided region-of-interest (ROI)
selection in X-ray fluorescence (XRF) microscopy experiments at synchrotron beamlines.

| | |
|---|---|
| **Repository** | [https://github.com/arshadzahangirchowdhury/ROI-Finder](https://github.com/arshadzahangirchowdhury/ROI-Finder) |
| **Authors** | Arshad Zahangir Chowdhury et al. |
| **Paper** | J. Synchrotron Rad. 29 (2022), DOI: 10.1107/S1600577522008876 |
| **Language** | Python 3.x |
| **License** | Not explicitly stated (check repository) |
| **Last Active** | See GitHub for latest commits |

## Purpose

ROI-Finder addresses the key challenge in XRF microscopy: how to efficiently select
the most scientifically interesting cells or regions for detailed high-resolution
scanning when beam time is limited.

### Target Users
- Synchrotron XRF beamline scientists
- Environmental and biological researchers using XRF microscopy
- Anyone needing to prioritize regions in multi-element spatial data

## Core Capabilities

1. **Cell Segmentation**: Automatically identify individual cells in XRF elemental maps
2. **Feature Extraction**: Reduce multi-element data to informative feature space using PCA
3. **Clustering**: Group cells by elemental composition using fuzzy k-means
4. **Recommendation**: Rank and recommend cells for detailed scanning
5. **Annotation**: Interactive GUI for viewing, selecting, and annotating cells

## Installation

```bash
# Clone repository
git clone https://github.com/arshadzahangirchowdhury/ROI-Finder.git
cd ROI-Finder

# Create conda environment (recommended)
conda create -n roifinder python=3.8
conda activate roifinder

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

| Package | Purpose | Version |
|---------|---------|---------|
| numpy | Array operations | ≥1.20 |
| scipy | Scientific computing | ≥1.6 |
| scikit-learn | PCA, preprocessing | ≥0.24 |
| scikit-fuzzy | Fuzzy k-means clustering | ≥0.4 |
| scikit-image | Image processing, morphology | ≥0.18 |
| opencv-python | Thresholding, morphological ops | ≥4.5 |
| h5py | HDF5 data loading | ≥3.0 |
| matplotlib | Visualization | ≥3.3 |
| tkinter | GUI (standard library) | — |

## Related Files

| File | Content |
|------|---------|
| [reverse_engineering.md](reverse_engineering.md) | Code structure analysis, algorithm flow |
| [pros_cons.md](pros_cons.md) | Strengths, limitations, improvement opportunities |
| [reproduction_guide.md](reproduction_guide.md) | Step-by-step guide to reproduce results |
| [notebooks/](notebooks/) | Jupyter notebooks for hands-on exploration |
