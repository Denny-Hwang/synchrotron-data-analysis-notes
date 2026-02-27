# PtychoNet: CNN-Based Ptychographic Phase Retrieval

**Reference**: Guan et al. (2019)

## Overview

PtychoNet replaces iterative ptychographic phase retrieval algorithms (ePIE, DM) with
a trained convolutional neural network that performs single-pass inference — achieving
~90% speedup while maintaining comparable image quality.

## Problem Formulation

### Iterative Phase Retrieval (Traditional)

```
Given: Diffraction patterns I_j = |F{P(r-r_j) × O(r)}|²

Find: Object O(r) and Probe P(r)

Algorithm (ePIE):
  For each iteration (100-1000×):
    For each scan position j:
      1. ψ_j = P(r-r_j) × O(r)          # Exit wave
      2. Ψ_j = F{ψ_j}                    # Propagate to detector
      3. Ψ'_j = √I_j × Ψ_j/|Ψ_j|       # Replace amplitude with measured
      4. ψ'_j = F⁻¹{Ψ'_j}               # Back-propagate
      5. Update O(r) and P(r)             # Gradient update
```

Time: Minutes to hours per reconstruction.

### PtychoNet (Deep Learning)

```
Given: Single diffraction pattern I_j (256×256)

Predict: Local object patch (amplitude + phase) at position j

Time: Milliseconds per pattern
```

## Architecture

### Encoder-Decoder Network

```
Input: Diffraction pattern (1ch, 256×256)
    │
    ▼ [Encoder]
    ResBlock (64) → MaxPool
    ResBlock (128) → MaxPool
    ResBlock (256) → MaxPool
    ResBlock (512) → MaxPool
    │
    ▼ [Bottleneck]
    ResBlock (512)
    │
    ▼ [Decoder — Amplitude branch]          ▼ [Decoder — Phase branch]
    UpConv + skip (256)                      UpConv + skip (256)
    UpConv + skip (128)                      UpConv + skip (128)
    UpConv + skip (64)                       UpConv + skip (64)
    UpConv (1)                               UpConv (1)
    │                                        │
    ▼                                        ▼
    Amplitude image (256×256)                Phase image (256×256)
```

### Key Design Choices
- **Dual output heads**: Separate branches for amplitude and phase
- **Residual blocks**: Better gradient flow, deeper network possible
- **Skip connections**: Preserve high-frequency details from encoder
- **Input**: Square root of diffraction intensity (to handle dynamic range)

## Training

### Data Generation
```
1. Simulate or collect real ptychographic datasets
2. Reconstruct using iterative method (ePIE with 500+ iterations) → Ground truth
3. Training pairs: (diffraction pattern, local reconstruction patch)
4. Data augmentation: rotation, noise addition, probe variation
```

### Loss Function

```python
# Combined loss for amplitude and phase
loss = lambda_amp * MSE(pred_amp, gt_amp) + lambda_phase * MSE(pred_phase, gt_phase)

# With phase wrapping handling:
phase_diff = pred_phase - gt_phase
phase_diff_wrapped = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
phase_loss = MSE(phase_diff_wrapped, 0)
```

### Training Parameters
- **Optimizer**: Adam, lr=1e-4
- **Batch size**: 16-32
- **Epochs**: 50-100
- **GPU**: Single NVIDIA V100/A100
- **Training time**: 4-12 hours
- **Training set**: 10,000-50,000 diffraction patterns

## Performance

### Speed Comparison

| Method | Time per reconstruction | Relative |
|--------|----------------------|----------|
| ePIE (1000 iterations, GPU) | 60 s | 1× |
| ePIE (100 iterations, GPU) | 6 s | 10× |
| DM (500 iterations, GPU) | 30 s | 2× |
| **PtychoNet (inference)** | **0.5 s** | **120×** |

### Quality Comparison

| Metric | ePIE (1000 iter) | PtychoNet |
|--------|-----------------|-----------|
| **SSIM (amplitude)** | Reference | 0.92-0.97 |
| **SSIM (phase)** | Reference | 0.90-0.95 |
| **Phase error (rad)** | 0 (reference) | 0.05-0.15 |

*PtychoNet approaches iterative quality for trained sample types.*

## Strengths

1. **Speed**: 90-120× faster than full iterative reconstruction
2. **Single-pass**: No iterative convergence required
3. **Parallelizable**: Each pattern reconstructed independently
4. **Real-time compatible**: Fast enough for streaming analysis

## Limitations

1. **Generalization**: Trained on specific sample types; new types need retraining
2. **No probe reconstruction**: Assumes known probe (or uses separate estimation)
3. **Stitching artifacts**: Local patches must be stitched for full image
4. **Training cost**: Requires large iterative-reconstructed training set
5. **Quantitative accuracy**: Phase values may not be as quantitatively accurate

## Hybrid Approach: PtychoNet + Iterative Refinement

Best results combine DL speed with iterative accuracy:

```
Diffraction patterns
    │
    ├─→ PtychoNet (single forward pass)
    │       Output: Initial reconstruction (~0.5 s)
    │
    └─→ Iterative refinement (10-50 iterations, starting from PtychoNet output)
            Output: Polished reconstruction (~3 s)
            Quality: Near-identical to 1000-iteration ePIE
```

This hybrid approach achieves:
- 10-20× speedup over full iterative reconstruction
- Quality equivalent to full iterative methods
- Best of both worlds: DL speed + physics-based accuracy

## Relevance to eBERlight

### Beamline Integration (33-ID-C, 2-ID-E)
- Real-time reconstruction feedback during ptychographic scans
- Enables adaptive scanning: reconstruct → assess quality → adjust scan area
- Critical for APS-U data rates (2 kHz frame rates)

### AI@Edge Deployment
- Optimized PtychoNet deployed on edge GPU/FPGA at beamline
- Inference during data acquisition
- Streaming reconstruction pipeline
