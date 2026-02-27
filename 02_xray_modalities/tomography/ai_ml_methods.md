# AI/ML Methods for Tomography

## Overview

Tomography is one of the most active areas for AI/ML in synchrotron science. The
combination of large data volumes, well-defined mathematical models, and clear quality
metrics makes it ideal for deep learning approaches. Key application areas include
denoising, reconstruction acceleration, segmentation, and artifact removal.

## ML Problem Classification

| Problem | Type | Input | Output |
|---------|------|-------|--------|
| Denoising | Image-to-image | Noisy reconstruction | Clean reconstruction |
| Super-resolution | Image-to-image | Low-res volume | High-res volume |
| Segmentation | Pixel classification | Grayscale volume | Labeled volume |
| Artifact removal | Image-to-image | Artifact-containing | Clean image |
| Sparse reconstruction | Inverse problem | Few projections | Full volume |
| Anomaly detection | Classification | Slice/region | Normal/anomalous |

## Denoising Methods

### TomoGAN

**Architecture**: GAN-based denoising (Generator: U-Net, Discriminator: PatchGAN)

**Training**:
- Input: Low-dose (noisy) reconstructions
- Target: Full-dose (high-quality) reconstructions
- Loss: L1 + adversarial + perceptual (VGG feature matching)

**Results**:
- SSIM improvement: 0.85 → 0.95+ (low-dose → denoised)
- PSNR: +5-10 dB improvement
- Preserves fine structural details better than Gaussian/median filtering

**Strengths**: Perceptual quality, sharp edges, fine detail preservation
**Weaknesses**: Potential hallucination, mode collapse during training, needs paired data

*See detailed review: [03_ai_ml_methods/denoising/tomogan.md](../../03_ai_ml_methods/denoising/tomogan.md)*

### Noise2Noise

**Principle**: Self-supervised — train with two noisy observations of the same sample
- No clean target needed
- Requires two scans of same sample under same conditions

**Application at APS**: Collect two rapid scans → use as training pairs
**Limitation**: Assumes static sample between scans

*See: [03_ai_ml_methods/denoising/noise2noise.md](../../03_ai_ml_methods/denoising/noise2noise.md)*

## Reconstruction Acceleration

### TomocuPy (GPU-Accelerated FBP)

- CuPy-based GPU implementation of standard reconstruction algorithms
- 20-30× speedup over TomoPy (CPU) for equivalent quality
- Supports FBP, gridrec, and iterative methods on GPU
- Streaming pipeline for large datasets exceeding GPU memory

*See detailed analysis: [05_tools_and_code/tomocupy/](../../05_tools_and_code/tomocupy/)*

### Deep Learning Reconstruction

**FBPConvNet** approach:
1. Perform fast FBP from sparse angles (artifact-containing)
2. Apply trained CNN to remove artifacts and denoise
3. Result: high-quality reconstruction from fewer projections (4-10× dose reduction)

**Implicit Neural Representations (INR)**:
- Represent volume as continuous function f_θ(x,y,z)
- Optimize network parameters to match measured projections
- Natural handling of sparse, irregular angular sampling
- Can incorporate time dimension for 4D: f_θ(x,y,z,t)

*See: [03_ai_ml_methods/reconstruction/inr_dynamic.md](../../03_ai_ml_methods/reconstruction/inr_dynamic.md)*

## Segmentation

### Traditional Methods
- **Otsu thresholding**: Simple, fast, works for high-contrast data
- **Watershed**: For touching/overlapping objects
- **Random walker**: Graph-based, handles low contrast
- **Region growing**: Good for connected structures

### Deep Learning Segmentation

**3D U-Net**:
- Standard architecture for volumetric medical/scientific image segmentation
- Encoder-decoder with skip connections
- Input: 3D patch (64³ to 128³ voxels)
- Output: Voxel-wise class probabilities

**nnU-Net**:
- Self-configuring framework — automatically determines optimal U-Net configuration
- Handles 2D, 3D, and cascaded approaches
- State-of-the-art on many biomedical benchmarks

**Challenges for synchrotron data**:
- Extremely large volumes (2048³ = 8 billion voxels) — must process in patches
- Limited labeled training data (manual annotation is expensive)
- Domain shift between different samples/beamlines
- Class imbalance (e.g., small pores in large volume)

### DLSIA (Deep Learning for Scientific Image Analysis)

ALS/LBNL-developed framework specifically for synchrotron data:
- Supports U-Net, mixed-scale dense (MSD) networks
- Designed for scientific (not medical) image analysis
- Integrated with MLExchange platform

## Artifact Removal

### Ring Artifact Correction
- **Cause**: Defective detector pixels create rings in reconstructed images
- **Traditional**: Fourier filtering, sinogram-domain interpolation
- **DL approach**: Train CNN to identify and remove ring patterns
- **Advantage**: DL preserves real circular features while removing artifacts

### Metal Artifact Reduction
- **Cause**: High-Z inclusions cause streak artifacts
- **DL approach**: Inpainting in sinogram domain + CNN post-processing
- **Application**: Environmental samples with metal particles

## Autonomous Experiment Steering

### Real-Time Quality Assessment
- CNN evaluates reconstruction quality during acquisition
- Flags problematic scans (motion blur, misalignment) immediately
- Enables adaptive scan protocols (more projections if quality is low)

### Smart Scanning
- ML predicts optimal scan parameters (energy, exposure, number of projections)
- Bayesian optimization for dose-efficient scanning protocols
- Region-of-interest identification for targeted high-resolution scans

## Benchmarks and Datasets

| Dataset | Description | Size | Access |
|---------|-------------|------|--------|
| TomoBank | APS tomography datasets | 100+ scans | [tomobank.readthedocs.io](https://tomobank.readthedocs.io) |
| FIPS CT | Industrial CT challenge | 50+ volumes | Academic request |
| COVID-19 lung CT | Medical CT (transferable methods) | 1000+ | Multiple sources |

## Current Limitations

1. **Training data scarcity**: Few labeled synchrotron tomography datasets exist
2. **Generalization**: Models trained on one beamline/sample type may not transfer
3. **3D memory constraints**: Full 4096³ volumes cannot fit on single GPU
4. **Hallucination risk**: DL denoising may introduce artificial features
5. **Validation**: How to verify DL-enhanced results are physically accurate?

## Improvement Opportunities

1. **Self-supervised methods**: Reduce dependence on paired training data
2. **Foundation models**: Pre-trained on diverse synchrotron data, fine-tuned per task
3. **Physics-informed networks**: Incorporate CT forward model into network architecture
4. **Federated learning**: Train across beamlines without sharing raw data
5. **Uncertainty quantification**: Provide confidence maps alongside reconstructions
