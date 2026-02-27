# Implicit Neural Representations for Dynamic Tomography

## Concept

**Implicit Neural Representations (INR)** represent a continuous function (e.g., a 3D
volume or 4D spatiotemporal field) using the weights of a neural network. Instead of
storing discrete voxel values, the network maps coordinates to values:

```
Standard representation: Volume V[i, j, k]  (discrete 3D array)
INR representation:      f_θ(x, y, z) → value  (continuous function)
```

## Architecture

### Coordinate Network (MLP)

```
Input: (x, y, z) coordinates
    │
    ├─→ Positional Encoding
    │       γ(x) = [sin(2⁰πx), cos(2⁰πx), sin(2¹πx), cos(2¹πx), ...,
    │               sin(2^(L-1)πx), cos(2^(L-1)πx)]
    │       Maps low-dimensional input to high-dimensional feature space
    │       Enables learning high-frequency content
    │
    ├─→ MLP (6-8 layers, 256-512 neurons each)
    │       FC → ReLU → FC → ReLU → ... → FC
    │       (or SIREN: FC → sin → FC → sin → ...)
    │
    └─→ Output: attenuation value µ(x, y, z)
```

### SIREN (Sinusoidal Representation Networks)

Uses sinusoidal activation functions instead of ReLU:
```
φ(x) = sin(ω₀ × Wx + b)

where ω₀ is a frequency parameter (typically 30)
```

**Advantage**: Natural representation of continuous, smooth functions with
high-frequency details.

## Application to Dynamic Tomography (4D)

### Problem

In dynamic (4D) tomography, the sample changes during the scan:
- Traditional: collect all angles → single-time reconstruction (assumes static)
- Reality: sample evolves → motion artifacts in reconstruction

### INR Solution

Extend the coordinate network to include time:

```
f_θ(x, y, z, t) → µ(x, y, z, t)

The network represents a 4D spatiotemporal volume as a continuous function.
```

### Training (Physics-Informed)

Instead of supervised training with ground truth, optimize the network
to match the measured projections:

```python
# Forward model:
# Measured projection at angle θ_i, time t_i:
# p_measured(θ_i, t_i) = ∫ f_θ(ray(θ_i, s), t_i) ds
#                         (line integral through the INR volume)

# Loss: consistency with measured projections
loss = Σ_i ||p_measured(θ_i, t_i) - ∫ f_θ(ray(θ_i, s), t_i) ds||²

# + Regularization terms:
# - Temporal smoothness: ||∂f_θ/∂t||² (penalize rapid changes)
# - Spatial smoothness: ||∇f_θ||² (total variation)
# - Sparsity: ||f_θ||₁ (if applicable)
```

### Training Pipeline

```
Projections at different (angle, time) pairs
    │
    ├─→ Sample random batch of projections
    │
    ├─→ For each projection:
    │       ├── Sample points along ray through INR volume
    │       ├── Query f_θ(x, y, z, t) at sampled points
    │       ├── Numerical integration (line integral)
    │       └── Compare with measured projection value
    │
    ├─→ Compute loss (data fidelity + regularization)
    │
    └─→ Backpropagate and update network weights θ
```

## Advantages Over Discrete Methods

| Aspect | Discrete (Voxel Grid) | INR (Neural Network) |
|--------|----------------------|---------------------|
| **Resolution** | Fixed by grid | Continuous (arbitrary) |
| **Memory** | O(N³) for 3D volume | O(parameters) ≈ few MB |
| **Temporal modeling** | Separate volumes per time | Smooth temporal interpolation |
| **Missing angles** | Artifacts | Implicit regularization |
| **Interpolation** | Trilinear (blocky) | Smooth, learned |
| **Super-resolution** | Not possible | Natural querying at any resolution |

## Challenges

### 1. Training Time
- Must optimize per-volume (minutes to hours)
- Not a pre-trained model — training is the reconstruction
- GPU-intensive (multiple CUDA cores needed)

### 2. Limited Resolution
- Current MLP-based INR limited to ~512³ effective resolution
- Higher resolution requires larger networks or hash-based representations
- Instant-NGP (hash grids) partially addresses this

### 3. Spectral Bias
- MLPs naturally learn low-frequency content first
- Positional encoding and SIREN help but don't fully solve
- Progressive training: start with low frequencies, add high over time

### 4. Hyperparameter Sensitivity
- Network architecture (depth, width, activations)
- Positional encoding frequency range
- Regularization weights
- Learning rate schedule

## Instant-NGP for Tomography

**NVIDIA Instant-NGP** uses multi-resolution hash tables instead of positional encoding:

```
Input: (x, y, z)
    │
    ├─→ Multi-resolution hash encoding
    │       For each resolution level:
    │       └── Hash voxel corners → lookup learned features → interpolate
    │
    ├─→ Small MLP (2 layers)
    │
    └─→ Output: value

Training time: seconds to minutes (vs. hours for standard INR)
```

## Example: 4D Soil Wetting Experiment

```
Experiment: Water infiltration into soil column
    - Scan: continuous rotation, 0.5 s per 180° rotation
    - Duration: 30 minutes
    - Challenge: water front moves during each rotation

Traditional approach:
    - 3600 individual reconstructions (one per rotation)
    - Each has motion artifacts from water movement

INR approach:
    - Single continuous 4D representation f_θ(x, y, z, t)
    - Each projection uses its exact time stamp
    - Smooth temporal evolution of water distribution
    - Query at arbitrary time points for visualization
```

## Implementation Sketch

```python
import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, in_dim, n_frequencies=10):
        super().__init__()
        self.n_freq = n_frequencies
        # Frequency bands: 2^0, 2^1, ..., 2^(L-1)
        self.freq_bands = 2.0 ** torch.arange(n_frequencies)

    def forward(self, x):
        # x: (batch, in_dim)
        encodings = [x]
        for freq in self.freq_bands:
            encodings.append(torch.sin(freq * np.pi * x))
            encodings.append(torch.cos(freq * np.pi * x))
        return torch.cat(encodings, dim=-1)
        # output: (batch, in_dim * (2*n_freq + 1))

class DynamicINR(nn.Module):
    """4D INR for dynamic tomography: (x,y,z,t) -> attenuation."""

    def __init__(self, n_frequencies=10, hidden_dim=256, n_layers=6):
        super().__init__()
        in_dim = 4  # x, y, z, t
        encoded_dim = in_dim * (2 * n_frequencies + 1)

        self.pe = PositionalEncoding(in_dim, n_frequencies)

        layers = [nn.Linear(encoded_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, 1), nn.Softplus()]  # positive output

        self.mlp = nn.Sequential(*layers)

    def forward(self, coords):
        # coords: (batch, 4) — (x, y, z, t) normalized to [-1, 1]
        encoded = self.pe(coords)
        return self.mlp(encoded)  # (batch, 1) attenuation values
```

## Relevance to APS BER Program

### Key Applications
- **In-situ tomography**: Track wetting/drying, reaction fronts, root growth
- **Dose reduction**: Reconstruct from fewer projections using INR regularization
- **Arbitrary time query**: Visualize structural changes at any moment
- **Multi-scale**: Query the same INR at different spatial resolutions

### Beamline Integration
- **2-BM-A**: Fast in-situ tomography → 4D INR for temporal dynamics
- **7-BM-B**: Time-resolved studies → continuous temporal representation
- Computational support from ALCF for training large INR models
