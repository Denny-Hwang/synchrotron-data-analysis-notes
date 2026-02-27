# Deep Residual Networks for XRF Resolution Enhancement

**Reference**: npj Computational Materials (2023), DOI: 10.1038/s41524-023-00995-9

## Concept

Rather than simple denoising, this approach uses deep residual networks to **enhance
the effective spatial resolution** of XRF microscopy maps beyond the physical limit
set by the X-ray beam probe size.

### Problem Statement

```
True elemental distribution: f(x, y)     (infinite resolution)
Measured XRF map: g(x, y) = f(x,y) * h(x,y) + n(x,y)

where:
  h(x,y) = probe/beam profile (point spread function)
  n(x,y) = measurement noise
  * = convolution

Goal: Recover f(x,y) from g(x,y) — deconvolution / super-resolution
```

### Why Deep Learning?

Traditional deconvolution (Wiener filter, Richardson-Lucy) amplifies noise.
Deep learning learns to deconvolve while suppressing noise simultaneously.

## Method

### Architecture: Deep Residual Network

```
Input: Low-resolution XRF map (Ny, Nx)
    │
    ├─→ Conv3×3 → ReLU (64 filters)
    │
    ├─→ Residual Block × N (typically 16-20 blocks)
    │       ┌──────────────────────┐
    │       │ Conv3×3 → BN → ReLU │
    │       │ Conv3×3 → BN        │
    │       │ + skip connection    │
    │       └──────────────────────┘
    │
    ├─→ Conv3×3 (64 filters)
    │   + global skip connection (from input)
    │
    └─→ Conv3×3 (1 filter) → Enhanced-resolution map
```

### Residual Learning

The network learns the **difference** between low-res and high-res:
```
output = input + network(input)

The network only needs to learn the residual (high-frequency details),
which is easier than learning the full mapping.
```

### Training Data Generation

```
Approach 1: Paired scans
  - Fine scan (50 nm step) → High-resolution ground truth
  - Coarse scan (200 nm step) → Low-resolution input
  - Train network on patches from paired scans

Approach 2: Simulated degradation
  - Start with highest-resolution scan available
  - Convolve with estimated probe profile → simulated low-res
  - Add realistic noise model
  - Train on (simulated low-res, original high-res) pairs
```

## Results

### Resolution Improvement

| Metric | Before | After Enhancement | Improvement |
|--------|--------|-------------------|-------------|
| **Effective resolution** | 200 nm | 80 nm | 2.5× |
| **PSNR** | Baseline | +6-8 dB | Significant |
| **Feature visibility** | Blurred organelles | Resolved sub-cellular | Qualitative |

### Key Findings
- 2-4× effective resolution improvement demonstrated on XRF elemental maps
- Works across multiple elements simultaneously
- Preserves quantitative accuracy of elemental concentrations
- Enables sub-probe-size feature identification

## Strengths

1. **Beyond physical limit**: Achieves resolution better than beam size
2. **Multi-element**: Single network enhances all elemental channels
3. **Quantitative**: Preserves concentration accuracy
4. **Fast inference**: Real-time capable after training
5. **Residual learning**: Stable training, preserves low-frequency content

## Limitations

1. **Training data**: Needs paired low-res/high-res scans for training
2. **Sample-specific**: May need retraining for very different sample types
3. **Theoretical limit**: Cannot recover information truly below noise floor
4. **Probe model**: Performance depends on accuracy of probe characterization
5. **Validation**: Difficult to validate resolution claims without independent measurement

## Application to APS BER Program

### Potential Impact
- **2-ID-D nanoprobe**: Enhance 100-200 nm maps to sub-100 nm effective resolution
- **2-ID-E microprobe**: Enhance 1 µm maps to sub-µm resolution
- **8-BM-B**: Enhance large-area maps while maintaining throughput

### Workflow Integration
```
Coarse XRF scan (fast, large area)
    │
    ├─→ DL resolution enhancement
    │       Input: coarse maps (1-5 µm step)
    │       Output: enhanced maps (~0.5 µm effective)
    │
    ├─→ ROI-Finder (on enhanced maps)
    │       Better segmentation and clustering
    │
    └─→ Targeted fine scan (only on selected ROIs)
            Full resolution, small area
```

This workflow could significantly reduce beam time while maintaining data quality.

## Code Sketch

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)

class XRFResolutionEnhancer(nn.Module):
    def __init__(self, in_channels=1, n_blocks=16, n_filters=64):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, n_filters, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(n_filters) for _ in range(n_blocks)]
        )
        self.mid_conv = nn.Conv2d(n_filters, n_filters, 3, padding=1)
        self.output_conv = nn.Conv2d(n_filters, in_channels, 3, padding=1)

    def forward(self, x):
        initial = self.input_conv(x)
        residual = self.mid_conv(self.res_blocks(initial))
        enhanced = self.output_conv(initial + residual)
        return x + enhanced  # Global residual connection
```
