# AI/ML Methods for Synchrotron Science

## Overview

This section provides a comprehensive taxonomy of AI/ML methods applied to synchrotron
X-ray data analysis, organized by task category. Each category covers traditional
approaches, deep learning methods, and their specific adaptations for synchrotron data.

## Method Taxonomy

```
AI/ML for Synchrotron Science
├── Image Segmentation
│   ├── Semantic (pixel classification)
│   ├── Instance (individual object detection)
│   └── Panoptic (semantic + instance)
│
├── Denoising
│   ├── Supervised (paired training data)
│   ├── Self-supervised (no clean targets)
│   └── GAN-based (adversarial training)
│
├── Reconstruction
│   ├── GPU-accelerated classical (TomocuPy)
│   ├── CNN post-processing (FBPConvNet)
│   ├── Learned iterative (unrolled optimization)
│   └── Implicit Neural Representations (INR)
│
├── Autonomous Experiment
│   ├── ROI selection (ROI-Finder)
│   ├── Bayesian optimization
│   └── Unsupervised fingerprinting (AI-NERD)
│
└── Multimodal Integration
    ├── Cross-modal registration
    ├── Joint analysis
    └── Fusion networks
```

## Method Comparison by Modality

| Method Category | Crystallography | Tomography | XRF | Spectroscopy | Ptychography | Scattering |
|----------------|----------------|------------|-----|-------------|-------------|-----------|
| **Segmentation** | Crystal detection | Phase ID, pore/grain | Cell segmentation | — | Feature extraction | — |
| **Denoising** | — | TomoGAN, N2N | Resolution enhance | Spectral smoothing | — | — |
| **Reconstruction** | Phase retrieval | TomocuPy, DL-recon | — | — | PtychoNet, AI@Edge | — |
| **Autonomous** | Crystal centering | Smart scanning | ROI-Finder | Active energy selection | Adaptive scanning | AI-NERD |
| **Multimodal** | AlphaFold + MR | CT + XAS | XRF + ptycho | XAS + XRF mapping | Ptycho + XRF | SAXS + WAXS |

## Cross-Cutting Themes

### 1. Training Data Challenge
- Labeled synchrotron data is scarce (expert annotation is expensive)
- Self-supervised and unsupervised methods are preferred
- Simulation-based pre-training is increasingly common
- Active learning minimizes annotation requirements

### 2. Scale Challenge
- Synchrotron data volumes are massive (TB per experiment)
- Models must handle 2K³ to 4K³ voxel volumes
- Patch-based approaches with stitching artifacts
- Distributed/parallel inference on HPC systems

### 3. Real-Time Requirement
- Post APS-U, data rates demand real-time analysis
- Streaming inference during acquisition
- Edge computing deployment (FPGA, embedded GPU)
- Latency budget: ms for experiment steering decisions

### 4. Physical Correctness
- ML outputs must be physically meaningful
- Physics-informed loss functions and architectures
- Uncertainty quantification for scientific confidence
- Hybrid approaches: ML + physics-based validation

## Directory Contents

| Subdirectory | Category | Key Methods |
|-------------|----------|-------------|
| [image_segmentation/](image_segmentation/) | Segmentation | U-Net variants, cell segmentation, tomographic phase ID |
| [denoising/](denoising/) | Denoising | TomoGAN, Noise2Noise, deep residual networks |
| [reconstruction/](reconstruction/) | Reconstruction | TomocuPy, PtychoNet, INR for dynamic imaging |
| [autonomous_experiment/](autonomous_experiment/) | Autonomous | ROI-Finder, Bayesian optimization, AI-NERD |
| [multimodal_integration/](multimodal_integration/) | Integration | XRF+ptycho, CT+XAS, optical+X-ray registration |
