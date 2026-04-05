# Diffusion Models for CT Reconstruction

**References**: Song et al. (2021), DOI: [10.48550/arXiv.2011.13456](https://doi.org/10.48550/arXiv.2011.13456); DM4CT Benchmark (OpenReview, 2024)

## Concept

**Score-based diffusion models** learn the gradient of the data distribution
(the "score") and use iterative denoising to generate high-quality images from
noise. Applied to CT reconstruction, they act as powerful learned priors that
can be combined with physics-based data consistency to reconstruct from
severely undersampled measurements.

```
Traditional:   Sparse projections → FBP → Artifact-heavy image
Learned:       Sparse projections → Trained CNN → Single-pass reconstruction
Diffusion:     Sparse projections → Iterative denoise + physics → Clean reconstruction
                                     (100-1000 refinement steps)
```

## Architecture

### Score Network (Noise-Conditional)

```
Input: Noisy image x_t + noise level t
    │
    ├─→ Time embedding
    │       t → sinusoidal encoding → MLP → time features
    │
    ├─→ U-Net with attention
    │       Conv→GN→SiLU (64)  + time ─────────────────────┐
    │       Conv→GN→SiLU (128) + time ──────────────┐       │
    │       Conv→GN→SiLU (256) + time + attn ────┐  │       │
    │       Conv→GN→SiLU (512) + time + attn ─┐  │  │       │
    │       │ [Bottleneck + self-attention]     │  │  │       │
    │       TransConv + skip ──────────────────┘  │  │       │
    │       TransConv + skip + attn ──────────────┘  │       │
    │       TransConv + skip ────────────────────────┘       │
    │       TransConv + skip ────────────────────────────────┘
    │
    └─→ Output: Score estimate ∇_x log p_t(x_t)

GN = Group Normalization
SiLU = Sigmoid Linear Unit (x · sigmoid(x))
```

### Score-Based Generative Modeling

```
Forward process (adding noise):
  x_0 → x_1 → x_2 → ... → x_T ≈ N(0, I)
  x_t = √(α_t) × x_0 + √(1 - α_t) × ε,   ε ~ N(0, I)

Reverse process (denoising):
  x_T → x_{T-1} → ... → x_1 → x_0
  x_{t-1} = (1/√α_t)(x_t - (1-α_t)/√(1-ᾱ_t) × s_θ(x_t, t)) + σ_t × z

where s_θ(x_t, t) is the learned score network
```

## Application to CT Reconstruction

### Problem Setting

```
Sparse-view CT:      Few projection angles (e.g., 60 instead of 720)
Limited-angle CT:    Restricted angular range (e.g., 0°-120° instead of 0°-180°)
Low-dose CT:         Few photons per projection

All cases: Standard FBP produces severe artifacts (streaks, distortions)
```

### Diffusion + Data Consistency

The key innovation is combining the diffusion prior with physics-based
projection constraints:

```
For each reverse diffusion step t = T, T-1, ..., 1:
    1. Denoise step: x̂_{t-1} = denoise(x_t, s_θ)    [learned prior]
    2. Data consistency: x_{t-1} = DC(x̂_{t-1}, y)    [physics constraint]

Data Consistency (DC):
    x_{t-1} = x̂_{t-1} + λ × A^T(y - A × x̂_{t-1})

where:
    A    = forward projection operator (Radon transform)
    A^T  = backprojection operator
    y    = measured sinogram
    λ    = step size for data consistency
```

### Advantage: No Retraining for Different Patterns

```
GAN/CNN approach:   Train for 60 views → retrain for 30 views → retrain for limited-angle
Diffusion approach: Train score model ONCE on CT images
                    → Apply with ANY undersampling pattern at inference time
                    → Data consistency adapts to the measurement geometry

This is because the score model learns the image prior p(x),
NOT the inverse mapping A^{-1}. The measurement model A is
applied separately during inference.
```

## DM4CT Benchmark (2024)

The DM4CT benchmark provides standardized evaluation of diffusion models
for CT reconstruction tasks:

```
Tasks:
  ├── Sparse-view CT (60, 120, 180 views)
  ├── Limited-angle CT (90°, 120°, 150° range)
  ├── Low-dose CT (Poisson noise levels)
  └── Region-of-interest CT (truncated projections)

Key findings:
  - Diffusion models outperform FBP+postprocessing by 3-6 dB PSNR
  - Competitive with or superior to GAN methods
  - More robust to out-of-distribution samples
  - Slower inference (minutes vs. milliseconds for CNNs)
```

## Comparison with GAN-Based Methods

| Aspect | GAN (e.g., TomoGAN) | Diffusion Model |
|--------|-------------------|-----------------|
| **Training stability** | Mode collapse risk | Stable (denoising objective) |
| **Hallucination** | Can generate non-physical features | Less hallucination (iterative refinement) |
| **Diversity** | Single output | Can generate multiple samples |
| **Uncertainty** | No native uncertainty | Sample variance = uncertainty |
| **Speed** | Fast (single forward pass) | Slow (100-1000 steps) |
| **Flexibility** | Retrain for new setup | Same model, different physics |
| **Quality (sparse-view)** | ★★★★ | ★★★★★ |
| **Quality (limited-angle)** | ★★★ | ★★★★ |

## Quantitative Performance

| Method | Sparse-view (60) PSNR | Sparse-view (60) SSIM | Limited-angle (120°) PSNR |
|--------|----------------------|----------------------|--------------------------|
| **FBP** | 22.4 | 0.62 | 18.1 |
| **SIRT (200 iter)** | 26.8 | 0.78 | 22.5 |
| **FBPConvNet** | 31.2 | 0.91 | 27.8 |
| **TomoGAN** | 32.5 | 0.93 | 28.3 |
| **Diffusion + DC** | **34.1** | **0.95** | **30.7** |

*Representative values from DM4CT benchmark; actual performance depends on
dataset and implementation details.*

## Training Strategy

### Training the Score Network

```
Training data: Collection of high-quality CT reconstructions
               (these serve as samples from p(x))

Training procedure:
    1. Sample clean image x_0 from training set
    2. Sample random noise level t ~ Uniform(1, T)
    3. Add noise: x_t = √(ᾱ_t) × x_0 + √(1-ᾱ_t) × ε
    4. Train score network: L = ||s_θ(x_t, t) - ε||²
       (predict the added noise)

No projection/sinogram data needed during training!
The physics model is applied only at inference time.
```

### Training Details

- **Architecture**: U-Net with self-attention at 16×16 and 32×32 resolution
- **Image size**: 256×256 or 512×512 CT slices
- **Training set**: 5,000-50,000 CT slices
- **Batch size**: 8-32
- **Optimizer**: Adam (lr=2×10⁻⁴)
- **Diffusion steps T**: 1000 (training), 50-200 (inference with DDIM)
- **Training time**: 1-3 days on 4 GPUs
- **Inference time**: 30 seconds - 5 minutes per slice (depending on steps)

## Strengths

1. **Flexible measurement model**: Same trained model works with any projection geometry
2. **No mode collapse**: Stable training (unlike GANs)
3. **Uncertainty quantification**: Run multiple times → sample variance as uncertainty
4. **State-of-the-art quality**: Top performance on sparse-view and limited-angle CT
5. **Physics-guided**: Data consistency ensures measurement fidelity
6. **Robust to distribution shift**: Learned prior generalizes across sample types

## Limitations

1. **Slow inference**: 100-1000 denoising steps (minutes per slice vs. milliseconds for CNN)
2. **Memory intensive**: Score network + intermediate states require significant GPU memory
3. **Training data hungry**: Needs thousands of high-quality CT images for prior training
4. **Hyperparameter tuning**: Data consistency weight λ, number of steps, noise schedule
5. **No convergence guarantee**: Iterative process may not converge for all inputs
6. **Limited to 2D slices**: Extension to full 3D volumes is computationally challenging

## Accelerated Inference

```
Standard DDPM:    1000 steps → ~10 minutes per slice
DDIM sampling:    50-100 steps → ~1 minute per slice
DPM-Solver:       20-50 steps → ~20 seconds per slice
Consistency model: 1-4 steps → ~1-5 seconds per slice (quality trade-off)
```

## Code Example

```python
import torch
import torch.nn as nn
import numpy as np

class ScoreUNet(nn.Module):
    """Simplified score network for CT diffusion model."""

    def __init__(self, channels=64, time_emb_dim=128):
        super().__init__()
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # U-Net encoder
        self.enc1 = self._block(1, channels, time_emb_dim)
        self.enc2 = self._block(channels, channels * 2, time_emb_dim)
        self.enc3 = self._block(channels * 2, channels * 4, time_emb_dim)

        # Bottleneck
        self.bottleneck = self._block(channels * 4, channels * 8, time_emb_dim)

        # Decoder
        self.up3 = nn.ConvTranspose2d(channels * 8, channels * 4, 4, 2, 1)
        self.dec3 = self._block(channels * 8, channels * 4, time_emb_dim)
        self.up2 = nn.ConvTranspose2d(channels * 4, channels * 2, 4, 2, 1)
        self.dec2 = self._block(channels * 4, channels * 2, time_emb_dim)
        self.up1 = nn.ConvTranspose2d(channels * 2, channels, 4, 2, 1)
        self.dec1 = self._block(channels * 2, channels, time_emb_dim)

        self.final = nn.Conv2d(channels, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def _block(self, in_ch, out_ch, time_dim):
        return nn.ModuleDict({
            'conv': nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.SiLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.SiLU(),
            ),
            'time_proj': nn.Linear(time_dim, out_ch),
        })

    def _apply_block(self, block, x, t_emb):
        h = block['conv'](x)
        t = block['time_proj'](t_emb)[:, :, None, None]
        return h + t

    def forward(self, x, t):
        t_emb = self.time_mlp(t.unsqueeze(-1).float())

        e1 = self._apply_block(self.enc1, x, t_emb)
        e2 = self._apply_block(self.enc2, self.pool(e1), t_emb)
        e3 = self._apply_block(self.enc3, self.pool(e2), t_emb)

        b = self._apply_block(self.bottleneck, self.pool(e3), t_emb)

        d3 = self._apply_block(self.dec3, torch.cat([self.up3(b), e3], 1), t_emb)
        d2 = self._apply_block(self.dec2, torch.cat([self.up2(d3), e2], 1), t_emb)
        d1 = self._apply_block(self.dec1, torch.cat([self.up1(d2), e1], 1), t_emb)

        return self.final(d1)


def diffusion_ct_reconstruct(score_model, sinogram, angles, n_steps=200,
                              image_size=256, lam=1.0):
    """Reconstruct CT image using diffusion model with data consistency.

    Args:
        score_model: Trained score network
        sinogram: Measured projections (n_angles, n_detectors)
        angles: Projection angles in radians
        n_steps: Number of reverse diffusion steps
        image_size: Output image size
        lam: Data consistency weight
    """
    from skimage.transform import radon, iradon

    device = next(score_model.parameters()).device

    # Initialize from noise
    x = torch.randn(1, 1, image_size, image_size, device=device)

    # Noise schedule (linear)
    betas = torch.linspace(1e-4, 0.02, 1000, device=device)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    # DDIM-style sampling with fewer steps
    step_indices = torch.linspace(999, 0, n_steps, dtype=torch.long, device=device)

    for i in range(len(step_indices)):
        t = step_indices[i]

        # 1. Score prediction (denoise step)
        with torch.no_grad():
            noise_pred = score_model(x, t.unsqueeze(0))

        # Predict x_0
        alpha_bar_t = alpha_bars[t]
        x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)

        # 2. Data consistency step
        x0_np = x0_pred.squeeze().cpu().numpy()
        sino_pred = radon(x0_np, theta=np.degrees(angles))
        sino_residual = sinogram - sino_pred
        correction = iradon(sino_residual, theta=np.degrees(angles),
                           filter_name=None, output_size=image_size)
        correction_tensor = torch.tensor(correction, device=device,
                                         dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        x0_corrected = x0_pred + lam * correction_tensor

        # 3. Add noise for next step (if not last step)
        if i < len(step_indices) - 1:
            t_next = step_indices[i + 1]
            alpha_bar_next = alpha_bars[t_next]
            x = (torch.sqrt(alpha_bar_next) * x0_corrected +
                 torch.sqrt(1 - alpha_bar_next) * torch.randn_like(x))
        else:
            x = x0_corrected

    return x
```

## Relevance to APS BER Program

### Key Applications

- **Sparse-view in-situ CT**: Reduce number of projections for faster time-resolved scans
- **Limited-angle tomography**: Reconstruct from restricted angular ranges (e.g., HEDM)
- **Dose reduction**: Fewer projections = lower total dose for radiation-sensitive samples
- **Uncertainty quantification**: Multiple diffusion samples provide reconstruction uncertainty

### Beamline Integration

- **2-BM**: Sparse-view fast tomography — diffusion models enable 5-10x fewer projections
- **1-ID**: High-energy diffraction microscopy with limited angular coverage
- **32-ID**: Ultrafast imaging where each projection is unique (no repeated angles)
- Computational support from ALCF for GPU-intensive inference

### Future Directions (2025-2026)

- **3D diffusion models**: Operate directly on 3D volumes (vs. slice-by-slice)
- **Conditional diffusion**: Condition on sample type, experimental parameters
- **Real-time inference**: Consistency distillation for single-step reconstruction
- **Multi-modal**: Joint diffusion prior for absorption + phase contrast CT

## References

1. Song, Y., et al. "Score-Based Generative Modeling through Stochastic Differential
   Equations." ICLR 2021. DOI: [10.48550/arXiv.2011.13456](https://doi.org/10.48550/arXiv.2011.13456)
2. Chung, H., et al. "Diffusion Posterior Sampling for General Noisy Inverse Problems."
   ICLR 2023. arXiv: 2209.14687
3. DM4CT: "A Benchmark for Diffusion Models in CT Reconstruction."
   OpenReview, 2024.
4. Song, J., Meng, C., Ermon, S. "Denoising Diffusion Implicit Models."
   ICLR 2021. arXiv: 2010.02502
