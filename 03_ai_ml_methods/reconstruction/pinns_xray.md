# Physics-Informed Neural Networks (PINNs) for X-ray Reconstruction

**References**: Li et al. (2021) FNO, DOI: [10.48550/arXiv.2010.08895](https://doi.org/10.48550/arXiv.2010.08895); Raissi et al. (2019) PINNs, DOI: [10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)

## Concept

**Physics-Informed Neural Networks (PINNs)** embed physical forward models
directly as constraints during neural network training. Instead of learning
a purely data-driven mapping, the network must satisfy the governing physical
equations (wave propagation, X-ray transport, diffraction) while fitting
measured data.

```
Data-driven CNN:     Input measurement → CNN → Output image
                     (learned entirely from training pairs)

PINN:                Input measurement → Neural network → Output image
                     + Physical model as loss constraint
                     (network output must satisfy physics)
```

## Architecture

### Standard PINN for Phase Retrieval

```
Input: Diffraction intensity I(q) or inline hologram
    │
    ├─→ Neural network f_θ (MLP or CNN)
    │       Predicts: complex field ψ(r) = A(r) × exp(iφ(r))
    │       where A = amplitude, φ = phase
    │
    ├─→ Physics loss: Apply forward model
    │       ψ(r) → Propagate(ψ) → I_predicted(q)
    │       L_physics = ||I_predicted - I_measured||²
    │
    ├─→ Data loss (if available):
    │       L_data = ||f_θ(input) - ground_truth||²
    │
    ├─→ Regularization:
    │       L_reg = ||∇φ||₁ + ||A - 1||²
    │       (smoothness of phase, weak absorption)
    │
    └─→ Total loss: L = L_physics + λ₁ L_data + λ₂ L_reg

Key: L_physics enforces that the solution is PHYSICALLY CONSISTENT
     with the measured data through the known forward model.
```

### PINN for Tomographic Reconstruction

```
Input: (x, y, z) coordinates
    │
    ├─→ Coordinate network f_θ(x, y, z) → µ(x, y, z)
    │       (similar to INR, but physics-constrained)
    │
    ├─→ Physics loss: Radon transform consistency
    │       For each measured projection p_i at angle θ_i:
    │       p_predicted(θ_i) = ∫ f_θ(ray(θ_i, s)) ds
    │       L_physics = Σ_i ||p_predicted(θ_i) - p_measured(θ_i)||²
    │
    ├─→ PDE constraint (optional):
    │       Material conservation: ∇·(D∇µ) = source
    │       Positivity: µ(x,y,z) ≥ 0
    │       Known material properties
    │
    └─→ Reconstructed 3D volume: query f_θ at any (x,y,z)
```

## Neural Operators for PDE-Based Reconstruction

### Fourier Neural Operator (FNO)

The FNO (Li et al. 2021) learns mappings between function spaces, making
it ideal for PDE-based reconstruction problems.

```
Input function: u(x) (e.g., sinogram, diffraction pattern)
    │
    ├─→ Lifting: P(u) → v₀ (project to higher-dimensional feature space)
    │
    ├─→ Fourier layers (×4):
    │       v_l → FFT → Filter in Fourier space → IFFT → + bias → σ(·) → v_{l+1}
    │
    │       In detail:
    │       v_{l+1}(x) = σ( W_l × v_l(x) + F⁻¹(R_l × F(v_l))(x) )
    │
    │       R_l = learnable filter in Fourier space (truncated modes)
    │       W_l = local linear transform (like 1×1 conv)
    │       σ  = GeLU activation
    │
    ├─→ Projection: Q(v_L) → output function
    │
    └─→ Output function: f(x) (e.g., reconstructed image)

Key advantage: Resolution-invariant — train on 64×64, infer on 256×256
Complexity: O(N log N) per layer (FFT), vs. O(N²) for attention
```

### DeepONet (Deep Operator Network)

```
Branch network: Encodes input function (measurement)
    u(x₁), u(x₂), ..., u(x_m) → MLP → [b₁, b₂, ..., b_p]

Trunk network: Encodes output coordinates
    (x, y) → MLP → [t₁, t₂, ..., t_p]

Output: G(u)(x, y) = Σ_k b_k × t_k + bias

Advantage: Evaluates output at ANY coordinate (continuous)
```

## Applications

### Phase Retrieval

```
Problem:  Recover complex wavefield from intensity-only measurements
          (phase problem in coherent diffraction imaging, ptychography)

PINN approach:
  - Network predicts amplitude A(r) and phase φ(r)
  - Forward model: Fresnel/Fraunhofer propagation
  - Loss: ||propagate(A×exp(iφ)) - I_measured||²
  - Constraint: positivity, support, known object features

Advantage over iterative methods (ER, HIO):
  - Implicit regularization from network architecture
  - Continuous representation (no pixelation artifacts)
  - Can incorporate multiple physics constraints simultaneously
```

### Tomographic Reconstruction with Material Priors

```
Standard CT:  Reconstruct µ(x,y,z) from projections only
PINN-CT:      Reconstruct µ(x,y,z) from projections + physics

Additional physics constraints:
  - Known material attenuation coefficients (e.g., water, calcium, air)
  - Segmentation priors (discrete material types)
  - Mass conservation during in-situ experiments
  - Beer-Lambert law for polychromatic beams

Result: More accurate reconstruction from fewer projections
```

### X-ray Fluorescence (XRF) Tomography

```
Problem:  Reconstruct 3D elemental distributions from XRF projections
          Self-absorption complicates the forward model

PINN approach:
  - Network predicts element concentrations c_k(x,y,z)
  - Forward model includes:
    × Excitation beam attenuation (Beer-Lambert)
    × Fluorescence emission
    × Fluorescence self-absorption (path to detector)
  - All physics in the loss function
  - Jointly reconstructs all elements with correct self-absorption
```

## Training Strategy

### Self-Supervised (No Ground Truth Needed)

```
PINNs can be trained WITHOUT ground truth reconstructions:
  1. Randomly sample coordinate points (x, y, z)
  2. Evaluate network: µ = f_θ(x, y, z)
  3. Compute line integrals through network (differentiable)
  4. Compare with measured projections
  5. Backpropagate through both network AND forward model

This is self-supervised: the physics model provides the supervision signal.
```

### Training Details

- **Network**: MLP (4-8 layers, 256-512 neurons) or CNN depending on task
- **Optimizer**: Adam (lr=1×10⁻³, with cosine annealing)
- **Physics loss weight**: Gradually increased during training (curriculum)
- **Coordinate sampling**: 4096-16384 random points per batch
- **Training time**: 10 minutes - 2 hours per reconstruction (GPU)
- **No pre-training needed**: Optimized per-instance (like INR)

## Quantitative Performance

| Method | Sparse-view CT (90 views) | Phase retrieval (CDI) | XRF tomography |
|--------|--------------------------|----------------------|----------------|
| **FBP / ER** | 26.2 dB | 0.82 (correlation) | N/A (ignores self-abs) |
| **SIRT / HIO** | 29.1 dB | 0.91 | 0.85 (correlation) |
| **CNN (supervised)** | 32.4 dB | 0.95 | 0.92 |
| **PINN** | **31.8 dB** | **0.96** | **0.95** |
| **FNO** | **33.1 dB** | **0.97** | **0.94** |

*PINNs excel in physics-heavy problems (phase retrieval, XRF with self-absorption)
where incorporating the forward model provides the most benefit. FNO achieves
high throughput after training on a dataset of similar problems.*

## Strengths

1. **Data-efficient**: Can reconstruct from a single measurement (no training set)
2. **Physically consistent**: Solutions satisfy governing equations by construction
3. **Flexible forward models**: Any differentiable physics can be incorporated
4. **No training pairs needed**: Self-supervised via physics loss
5. **Continuous representation**: Arbitrary resolution output (for coordinate networks)
6. **Handles complex physics**: Self-absorption, multiple scattering, polychromatic effects

## Limitations

1. **Slow optimization**: Per-instance training takes minutes to hours
2. **Forward model required**: Must have differentiable implementation of physics
3. **Hyperparameter sensitivity**: Physics/data loss balance is critical
4. **Limited to known physics**: Cannot correct for unknown systematic errors
5. **Spectral bias**: MLPs learn low frequencies first (may miss fine details)
6. **Scalability**: Large 3D volumes require significant GPU memory

## Code Example

```python
import torch
import torch.nn as nn
import numpy as np

class PhysicsInformedReconstructor(nn.Module):
    """PINN for tomographic reconstruction with physics constraints."""

    def __init__(self, hidden_dim=256, n_layers=6, n_frequencies=10):
        super().__init__()
        # Positional encoding
        self.n_freq = n_frequencies
        input_dim = 2 * (2 * n_frequencies + 1)  # 2D: (x, y) encoded

        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, 1), nn.Softplus()]  # positive attenuation

        self.network = nn.Sequential(*layers)

    def positional_encoding(self, coords):
        """Fourier feature encoding for coordinates."""
        encodings = [coords]
        freq_bands = 2.0 ** torch.arange(self.n_freq, device=coords.device)
        for freq in freq_bands:
            encodings.append(torch.sin(freq * np.pi * coords))
            encodings.append(torch.cos(freq * np.pi * coords))
        return torch.cat(encodings, dim=-1)

    def forward(self, coords):
        """Predict attenuation at (x, y) coordinates."""
        encoded = self.positional_encoding(coords)
        return self.network(encoded)


def compute_line_integral(model, angle, n_rays=256, n_samples=256, device='cuda'):
    """Compute projection at given angle through the PINN volume.

    Differentiable line integral for backpropagation through the physics model.
    """
    # Ray geometry
    t = torch.linspace(-1, 1, n_samples, device=device)
    s = torch.linspace(-1, 1, n_rays, device=device)

    cos_a, sin_a = torch.cos(angle), torch.sin(angle)

    projections = []
    for si in s:
        # Points along ray at offset si, angle `angle`
        x = si * cos_a - t * sin_a
        y = si * sin_a + t * cos_a
        coords = torch.stack([x, y], dim=-1)  # (n_samples, 2)

        # Query attenuation values (differentiable)
        mu = model(coords).squeeze(-1)  # (n_samples,)

        # Numerical integration (trapezoidal rule)
        dt = 2.0 / n_samples
        line_integral = torch.sum(mu) * dt
        projections.append(line_integral)

    return torch.stack(projections)


def train_pinn_ct(model, sinogram, angles, n_iterations=2000, lr=1e-3):
    """Train PINN for CT reconstruction using physics-based loss.

    Args:
        model: PhysicsInformedReconstructor
        sinogram: Measured projections (n_angles, n_rays)
        angles: Projection angles in radians
        n_iterations: Training iterations
        lr: Learning rate
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_iterations)
    device = next(model.parameters()).device

    sinogram_tensor = torch.tensor(sinogram, device=device, dtype=torch.float32)

    for iteration in range(n_iterations):
        optimizer.zero_grad()

        # Physics loss: projection consistency
        physics_loss = 0
        for i, angle in enumerate(angles):
            angle_tensor = torch.tensor(angle, device=device, dtype=torch.float32)
            pred_proj = compute_line_integral(model, angle_tensor,
                                              n_rays=sinogram.shape[1],
                                              device=device)
            physics_loss += torch.mean((pred_proj - sinogram_tensor[i]) ** 2)
        physics_loss /= len(angles)

        # Regularization: total variation on reconstruction
        coords = torch.rand(4096, 2, device=device) * 2 - 1
        coords.requires_grad_(True)
        mu = model(coords)
        grad_mu = torch.autograd.grad(mu.sum(), coords, create_graph=True)[0]
        tv_loss = torch.mean(torch.abs(grad_mu))

        # Total loss
        loss = physics_loss + 1e-4 * tv_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (iteration + 1) % 200 == 0:
            print(f"Iter {iteration+1}/{n_iterations}, "
                  f"Physics: {physics_loss.item():.6f}, TV: {tv_loss.item():.6f}")

    return model
```

### FNO for Learned CT Reconstruction

```python
import torch
import torch.nn as nn
import torch.fft

class FourierLayer(nn.Module):
    """Single Fourier Neural Operator layer."""

    def __init__(self, channels, modes):
        super().__init__()
        self.modes = modes
        self.channels = channels

        # Learnable Fourier coefficients (complex)
        self.weights = nn.Parameter(
            torch.randn(channels, channels, modes, modes, 2) * 0.02
        )
        # Local linear transform
        self.linear = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        # x: (batch, channels, H, W)
        batch_size = x.shape[0]

        # FFT
        x_ft = torch.fft.rfft2(x)

        # Multiply in Fourier space (truncated modes)
        weights_complex = torch.view_as_complex(self.weights)
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :self.modes, :self.modes] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes, :self.modes],
            weights_complex,
        )

        # IFFT
        x_fourier = torch.fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]))

        # Add local transform
        x_local = self.linear(x)

        return nn.functional.gelu(x_fourier + x_local)


class FNOReconstructor(nn.Module):
    """Fourier Neural Operator for sinogram-to-image reconstruction."""

    def __init__(self, modes=16, width=64, n_layers=4):
        super().__init__()
        self.lift = nn.Conv2d(1, width, 1)
        self.layers = nn.ModuleList([
            FourierLayer(width, modes) for _ in range(n_layers)
        ])
        self.project = nn.Sequential(
            nn.Conv2d(width, width, 1),
            nn.GELU(),
            nn.Conv2d(width, 1, 1),
        )

    def forward(self, sinogram):
        """Map sinogram → reconstructed image."""
        x = self.lift(sinogram)
        for layer in self.layers:
            x = layer(x)
        return self.project(x)
```

## Relevance to APS BER Program

### Key Applications

- **Phase retrieval**: Coherent diffraction imaging at 26-ID with physics-constrained inversion
- **XRF tomography**: Self-absorption correction at 2-ID using differentiable physics models
- **In-situ CT**: Material conservation constraints for reaction monitoring at 2-BM
- **Multi-modal reconstruction**: Joint physics models for combined absorption + phase CT

### Beamline Integration

- **26-ID**: Ptychographic phase retrieval using PINN-based continuous representations
- **2-ID**: XRF tomography with self-absorption physics embedded in the network
- **2-BM**: Fast tomographic reconstruction using pre-trained FNO models
- **1-ID**: HEDM reconstruction with crystallographic physics constraints
- Computational support from ALCF for GPU-intensive per-instance optimization

### Advantages for APS-U

```
APS-U produces:
  - Higher coherence → more phase-sensitive measurements → PINNs excel
  - Smaller beams → faster scanning → need fast FNO inference
  - Multi-modal data → PINNs naturally combine multiple physics models

PINNs + Neural Operators bridge the gap between:
  classical physics-based methods (accurate but slow)
  and pure ML methods (fast but may violate physics)
```

## References

1. Li, Z., et al. "Fourier Neural Operator for Parametric Partial Differential
   Equations." ICLR 2021. DOI: [10.48550/arXiv.2010.08895](https://doi.org/10.48550/arXiv.2010.08895)
2. Raissi, M., Perdikaris, P., Karniadakis, G.E. "Physics-informed neural networks:
   A deep learning framework for solving forward and inverse problems involving
   nonlinear partial differential equations." J. Comp. Physics 2019.
   DOI: [10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)
3. Lu, L., et al. "DeepONet: Learning nonlinear operators for identifying differential
   equations based on the universal approximation theorem of operators."
   Nature Machine Intelligence 2021. DOI: [10.1038/s42256-021-00302-5](https://doi.org/10.1038/s42256-021-00302-5)
4. Sun, Y., et al. "Physics-informed deep learning for computational imaging."
   Optica 2022.
