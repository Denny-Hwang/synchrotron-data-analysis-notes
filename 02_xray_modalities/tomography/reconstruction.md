# Tomographic Reconstruction Methods

## Overview

Tomographic reconstruction is the mathematical process of computing a 3D volume from
a set of 2D projection images. This is the central computational challenge in X-ray
tomography, and the choice of reconstruction algorithm significantly impacts image
quality, speed, and dose requirements.

## Mathematical Foundation

The forward model relates the object function f(x,y) to measured projections p(θ,t):

```
p(θ, t) = ∫ f(x, y) δ(x cos θ + y sin θ - t) dx dy    (Radon Transform)
```

Reconstruction recovers f(x,y) from p(θ,t) — the **inverse Radon transform**.

## Analytical Methods

### Filtered Back Projection (FBP)

The standard workhorse algorithm for synchrotron tomography.

**Algorithm**:
1. Apply ramp filter to each sinogram row (in Fourier domain)
2. Back-project filtered sinograms onto reconstruction grid
3. Sum contributions from all angles

**Mathematics**:
```
f(x,y) = ∫₀^π p̃(θ, x cos θ + y sin θ) dθ

where p̃ is the filtered projection: p̃ = F⁻¹{|ω| · F{p}}
```

**Properties**:
- Fast: O(N² log N) per slice
- Requires densely sampled angles (Nyquist: N_angles ≈ π/2 × N_pixels)
- Sensitive to noise (ramp filter amplifies high-frequency noise)
- Common filters: Ram-Lak (ramp), Shepp-Logan, Hann, Butterworth

**Implementation**: TomoPy `recon(algorithm='gridrec')`, TomocuPy

### Gridrec

Optimized FBP variant using gridding interpolation in Fourier space:
- 5-10× faster than direct FBP
- Default algorithm in TomoPy
- Nearly identical results to standard FBP

## Iterative Methods

### SIRT (Simultaneous Iterative Reconstruction Technique)

**Algorithm**:
```
f^(k+1) = f^(k) + λ · Aᵀ · C · (p - A · f^(k))

where:
  A = forward projection operator
  Aᵀ = back projection operator
  C = diagonal correction matrix
  λ = relaxation parameter
```

**Properties**:
- Better noise handling than FBP
- Can incorporate prior knowledge (positivity, smoothness)
- 10-100× slower than FBP
- Converges to least-squares solution

### MLEM (Maximum Likelihood Expectation Maximization)

**Algorithm**:
```
f^(k+1) = f^(k) · (Aᵀ · p / (A · f^(k))) / (Aᵀ · 1)
```

**Properties**:
- Multiplicative update → naturally enforces positivity
- Statistically motivated (Poisson noise model)
- Slow convergence, requires regularization (OS-MLEM variant faster)

### CGLS (Conjugate Gradient Least Squares)

- Minimizes ||Af - p||² using conjugate gradient method
- Fast convergence (5-20 iterations typically sufficient)
- No built-in regularization

## Deep Learning Methods

### Neural Network Reconstruction

| Method | Type | Key Innovation |
|--------|------|---------------|
| **FBPConvNet** | Post-processing | CNN denoises FBP reconstruction |
| **LEARN** | Learned iterative | Unrolled optimization with learned parameters |
| **iCT-Net** | Direct inversion | End-to-end sinogram → volume |
| **INR** | Implicit neural | Coordinate MLP represents continuous volume |

### Implicit Neural Representations (INR)

Represent the volume as a continuous function parameterized by a neural network:

```
f_θ(x, y, z) → attenuation value

where θ are network parameters optimized to match measured projections
```

**Advantages**:
- Continuous representation (arbitrary resolution)
- Natural handling of sparse/irregular angular sampling
- Can incorporate temporal dimension for 4D reconstruction: f_θ(x, y, z, t)

**Challenges**:
- Training time per volume (minutes to hours)
- Limited to moderate resolution (~512³) currently
- Requires careful architecture design (positional encoding, SIREN)

## Method Comparison

| Method | Speed | Quality (dense) | Quality (sparse) | Memory | GPU Accel |
|--------|-------|-----------------|-------------------|--------|-----------|
| **FBP/Gridrec** | ★★★★★ | ★★★★ | ★★ | Low | TomocuPy |
| **SIRT** | ★★★ | ★★★★★ | ★★★ | Medium | TomocuPy |
| **MLEM** | ★★ | ★★★★★ | ★★★★ | Medium | Limited |
| **CGLS** | ★★★★ | ★★★★ | ★★★★ | Medium | ASTRA |
| **FBPConvNet** | ★★★★ | ★★★★★ | ★★★★ | High (GPU) | Yes |
| **INR** | ★ | ★★★★ | ★★★★★ | High (GPU) | Yes |

## Software Tools

| Tool | Language | GPU | Key Features |
|------|----------|-----|-------------|
| **TomoPy** | Python/C | No | Standard reference, many algorithms |
| **TomocuPy** | Python/CuPy | Yes | 20-30× speedup, APS-developed |
| **ASTRA Toolbox** | Python/CUDA | Yes | Flexible geometry, GPU iterative |
| **TomOpt** | Python/PyTorch | Yes | Differentiable forward model |
| **SVMBIR** | Python/C | No | Model-based iterative (MBIR) |

## Reconstruction Pipeline at APS

```
Sinograms (preprocessed)
    │
    ├─→ Rotation center finding (automatic: Vo algorithm)
    │
    ├─→ Phase retrieval (optional: Paganin method)
    │
    ├─→ Reconstruction
    │       ├── Real-time: TomocuPy (GPU, seconds)
    │       ├── Standard: TomoPy gridrec (CPU, minutes)
    │       └── High-quality: Iterative (CPU/GPU, hours)
    │
    ├─→ Post-processing
    │       ├── Ring artifact removal
    │       ├── Denoising (DNN or Gaussian/median)
    │       └── Intensity normalization
    │
    └─→ Reconstructed 3D Volume
            └── Ready for segmentation and analysis
```
