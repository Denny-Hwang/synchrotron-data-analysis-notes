# TomoGAN: GAN-Based Denoising for Synchrotron Tomography

**Reference**: Liu et al., J. Opt. Soc. Am. A (2020)

## Architecture

TomoGAN uses a conditional GAN architecture with a U-Net generator and PatchGAN discriminator.

### Generator (U-Net)

```
Noisy input slice (1ch, 256×256)
    │
    ▼ [Encoder]
    Conv→IN→LeakyReLU (64)  ───────────────────────────┐
    Conv→IN→LeakyReLU (128) ────────────────────┐       │
    Conv→IN→LeakyReLU (256) ─────────────┐       │       │
    Conv→IN→LeakyReLU (512) ──────┐       │       │       │
    │                              │       │       │       │
    ▼ [Bottleneck]                 │       │       │       │
    Conv→IN→LeakyReLU (512)       │       │       │       │
    │                              │       │       │       │
    ▼ [Decoder]                    │       │       │       │
    TransConv→IN→ReLU + skip ─────┘       │       │       │
    TransConv→IN→ReLU + skip ─────────────┘       │       │
    TransConv→IN→ReLU + skip ─────────────────────┘       │
    TransConv→IN→ReLU + skip ─────────────────────────────┘
    │
    ▼
    Conv→Tanh → Denoised output (1ch, 256×256)

IN = Instance Normalization
```

### Discriminator (PatchGAN)

```
Input: [real/fake image, noisy input] concatenated (2ch)
    │
    Conv→LeakyReLU (64)
    Conv→IN→LeakyReLU (128)
    Conv→IN→LeakyReLU (256)
    Conv→IN→LeakyReLU (512)
    │
    Conv→Sigmoid → Patch-wise real/fake probabilities (16×16)
```

PatchGAN classifies each 16×16 patch as real or fake, encouraging local texture realism.

## Loss Function

The total generator loss combines three components:

```
L_total = λ₁ × L_L1 + λ₂ × L_adversarial + λ₃ × L_perceptual

where:
  L_L1 = ||G(noisy) - clean||₁               (pixel-wise reconstruction)
  L_adv = -log(D(G(noisy), noisy))            (fool discriminator)
  L_perc = ||VGG(G(noisy)) - VGG(clean)||₂   (feature matching)

Typical weights: λ₁ = 100, λ₂ = 1, λ₃ = 10
```

### Role of Each Loss
- **L1**: Ensures pixel-wise accuracy, prevents color shift
- **Adversarial**: Produces sharp, realistic textures (avoids blurring)
- **Perceptual**: Preserves structural features and high-level similarity

## Training Strategy

### Data Preparation
```
Full-dose scan → Reconstruction (high quality) → Clean target
Low-dose scan  → Reconstruction (noisy)        → Noisy input

OR: Simulate low-dose by subsampling full-dose projections
```

### Training Details
- **Patch size**: 256×256 (randomly cropped from full slices)
- **Batch size**: 4–16
- **Optimizer**: Adam (lr=2×10⁻⁴, β₁=0.5, β₂=0.999)
- **Epochs**: 100–200
- **Augmentation**: Random flip, rotation, intensity jittering
- **Training time**: ~12–24 hours on single GPU

### Alternative: Use Adjacent Slices
```
Input: [slice_{n-1}, slice_n, slice_{n+1}]  (3 channels)
Output: denoised slice_n (1 channel)
```
Uses spatial context from neighboring slices to aid denoising.

## Quantitative Performance

| Metric | Low-dose (noisy) | Gaussian filter | NLM | TomoGAN |
|--------|-----------------|----------------|-----|---------|
| **PSNR (dB)** | 25.3 | 28.1 | 30.2 | **34.5** |
| **SSIM** | 0.72 | 0.81 | 0.88 | **0.95** |
| **NRMSE** | 0.118 | 0.078 | 0.051 | **0.029** |

*Values are representative; actual performance varies by dataset.*

## Strengths

1. **Perceptual quality**: GAN produces sharp, realistic images (not blurred)
2. **Fine detail preservation**: Adversarial loss prevents smoothing of small features
3. **Flexible**: Works for different noise levels and modalities
4. **Dose reduction**: Enables 4-10× dose reduction with maintained image quality

## Limitations

1. **Mode collapse**: GAN training instability can produce repetitive artifacts
2. **Hallucination risk**: Generator may create plausible-looking but non-physical features
3. **Training data dependency**: Needs matched low-dose/full-dose pairs
4. **Sample specificity**: Model trained on one sample type may not generalize
5. **No uncertainty**: Single output with no confidence measure

## Hallucination Mitigation

```python
# Strategy 1: Residual learning
# Train to predict noise, not clean image
denoised = noisy_input - generator(noisy_input)

# Strategy 2: Consistency check
# Run denoised image through forward model (projection)
# Compare with measured projections
# Flag regions with large inconsistency

# Strategy 3: Ensemble uncertainty
# Train multiple GANs, use variance as uncertainty map
```

## Code Example

```python
import torch
import torch.nn as nn

class TomoGANGenerator(nn.Module):
    """Simplified TomoGAN generator (U-Net architecture)."""

    def __init__(self, in_ch=1, out_ch=1, base_filters=64):
        super().__init__()
        # Encoder
        self.enc1 = self._block(in_ch, base_filters)
        self.enc2 = self._block(base_filters, base_filters * 2)
        self.enc3 = self._block(base_filters * 2, base_filters * 4)
        self.enc4 = self._block(base_filters * 4, base_filters * 8)

        # Bottleneck
        self.bottleneck = self._block(base_filters * 8, base_filters * 8)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_filters * 8, base_filters * 8, 4, 2, 1)
        self.dec4 = self._block(base_filters * 16, base_filters * 4)
        self.up3 = nn.ConvTranspose2d(base_filters * 4, base_filters * 4, 4, 2, 1)
        self.dec3 = self._block(base_filters * 8, base_filters * 2)
        self.up2 = nn.ConvTranspose2d(base_filters * 2, base_filters * 2, 4, 2, 1)
        self.dec2 = self._block(base_filters * 4, base_filters)
        self.up1 = nn.ConvTranspose2d(base_filters, base_filters, 4, 2, 1)
        self.dec1 = self._block(base_filters * 2, base_filters)

        self.final = nn.Sequential(nn.Conv2d(base_filters, out_ch, 1), nn.Tanh())
        self.pool = nn.MaxPool2d(2)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x);     p1 = self.pool(e1)
        e2 = self.enc2(p1);    p2 = self.pool(e2)
        e3 = self.enc3(p2);    p3 = self.pool(e3)
        e4 = self.enc4(p3);    p4 = self.pool(e4)

        b = self.bottleneck(p4)

        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))

        return self.final(d1)
```
