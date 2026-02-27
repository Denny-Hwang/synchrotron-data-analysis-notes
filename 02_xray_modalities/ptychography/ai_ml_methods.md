# AI/ML Methods for Ptychography

## Overview

AI/ML methods for ptychography focus on accelerating phase retrieval, enabling real-time
reconstruction, handling sparse data, and improving image quality. The combination of
well-defined forward models and large data volumes makes ptychography an ideal candidate
for deep learning approaches.

## ML Problem Classification

| Problem | Type | Input | Output |
|---------|------|-------|--------|
| Phase retrieval | Inverse problem | Diffraction patterns | Complex image |
| Real-time reconstruction | Inference | Streaming patterns | Live image update |
| Sparse reconstruction | Inverse problem | Undersampled data | Full image |
| Resolution enhancement | Image-to-image | Low-resolution | High-resolution |
| Position correction | Regression | Patterns + positions | Corrected positions |
| Probe estimation | Regression | Patterns | Probe function |

## PtychoNet: CNN-Based Phase Retrieval

**Reference**: Guan et al. (2019)

### Architecture
```
Input: Single diffraction pattern (256×256)
    │
    ├─→ Encoder (ResNet-style blocks)
    │       Conv → BN → ReLU → Conv → BN → ReLU + skip
    │       Progressively reduces spatial dimensions
    │
    ├─→ Bottleneck (dense features)
    │
    ├─→ Decoder (transpose convolutions)
    │       Progressively increases spatial dimensions
    │
    └─→ Output: Phase + Amplitude images (256×256 each)
```

### Key Results
- **Speed**: 90% faster than iterative ePIE (single forward pass vs. 100+ iterations)
- **Quality**: Comparable to iterative methods for trained sample types
- **Limitation**: Generalization to unseen sample types requires retraining

### Training Strategy
- Train on pairs: (diffraction pattern, ground-truth reconstruction from iterative method)
- Data augmentation: rotations, translations, noise addition
- Loss: MSE on phase + amplitude, with phase wrapping handling

## AI@Edge: Real-Time Ptychography

**Reference**: Babu et al., Nature Communications (2023)

### Innovation
Deployed ML inference directly at the beamline edge computing hardware for
**real-time ptychographic imaging at 2 kHz frame rate**.

### Architecture
- Lightweight CNN optimized for edge deployment (FPGA or GPU)
- Streaming inference: processes each diffraction pattern as it arrives
- Progressive update: refines reconstruction as more positions are measured

### Pipeline
```
Detector (2 kHz) → Edge GPU/FPGA → CNN Inference → Live Reconstruction
    │                                                      │
    └── 0.5 ms/frame ──────────────────────────── Display update
```

### Significance for APS BER Program
- Enables feedback during experiments (not just post-hoc analysis)
- Experimentalist sees reconstruction quality in real-time
- Can abort or adjust scan parameters immediately
- Critical for APS-U data rates

## Implicit Neural Representations (INR)

### Concept
Represent the ptychographic object as a continuous function parameterized by a neural network:

```
f_θ(x, y) → (amplitude, phase)    for 2D
f_θ(x, y, z) → (amplitude, phase)  for 3D ptycho-tomo
f_θ(x, y, z, t) → (amplitude, phase)  for dynamic 4D
```

### Architecture
- **SIREN**: Sinusoidal activation functions for smooth, high-frequency representation
- **Positional encoding**: Fourier feature mapping for better high-frequency learning

### Advantages
- **Continuous**: Resolution not limited by discrete grid
- **Compact**: Stores model parameters, not full volume
- **Dynamic**: Natural extension to time-varying samples
- **Physics-informed**: Can incorporate forward model as physics constraint

### Challenges
- Training time: Minutes to hours per reconstruction
- Limited to moderate resolution (~512×512) currently
- Hyperparameter sensitivity

## Sparse Ptychography

### Problem
Reducing the number of scan positions to speed up data collection while maintaining
reconstruction quality.

### ML Approaches

1. **Compressed sensing + DL**:
   - Collect fewer overlapping positions (30-50% instead of 60-80%)
   - Use trained CNN to complete missing information
   - 2-3× speedup in data collection

2. **Adaptive scanning**:
   - Start with coarse grid → reconstruct → identify features → refine scan locally
   - ML predicts where additional measurements are most informative
   - Non-uniform scan patterns adapted to sample structure

3. **Transfer learning**:
   - Pre-train on similar sample types
   - Fewer measurements needed for new samples of same class

## Position Correction

### Problem
Ptychography is extremely sensitive to scan position errors (nm precision needed).

### ML Solution
- Train CNN to predict position corrections from diffraction pattern features
- Input: measured pattern + nominal position
- Output: corrected position offset (Δx, Δy)
- Integrated into reconstruction pipeline as preprocessing step

## Probe Reconstruction

### Traditional
ePIE and similar algorithms simultaneously reconstruct probe and object.

### ML-Enhanced
- CNN predicts probe function from first few diffraction patterns
- Provides better initialization → faster convergence of iterative methods
- Can model partial coherence via multi-mode decomposition

## Software Tools

| Tool | Language | GPU | Method |
|------|----------|-----|--------|
| **PtychoShelves** | MATLAB/GPU | Yes | ePIE, DM, LSQ-ML |
| **PyNX** | Python/GPU | Yes | Multiple iterative algorithms |
| **Tike** | Python/CuPy | Yes | APS-developed, ptycho + tomo |
| **PtyPy** | Python | Yes | Flexible, mixed-state support |
| **PtychoNN** | Python/PyTorch | Yes | CNN-based (AI@Edge) |

## Current Limitations

1. **Generalization**: Trained models are sample-type specific
2. **Training data**: Requires iterative reconstructions as ground truth
3. **3D extension**: DL for ptycho-tomography is still early stage
4. **Partial coherence**: ML handling of mixed-state illumination needs work
5. **Quantitative accuracy**: DL reconstructions may not preserve quantitative phase values

## Improvement Opportunities

1. **Self-supervised training**: Use physics-based loss (consistency with measured data)
2. **Foundation models**: Pre-train on large ptychography datasets across facilities
3. **Hybrid approaches**: DL initialization + few iterative refinement steps
4. **Multi-modal**: Joint ptychography + XRF reconstruction
5. **4D ptycho-tomo**: DL-accelerated dynamic 3D imaging
6. **Uncertainty maps**: Provide pixel-wise reconstruction confidence
