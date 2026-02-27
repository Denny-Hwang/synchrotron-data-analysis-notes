# Noise2Noise: Self-Supervised Denoising

**Reference**: Lehtinen et al. (2018), "Noise2Noise: Learning Image Restoration without Clean Data"

## Principle

Noise2Noise demonstrates that a denoising network can be trained using **only noisy
observations** — no clean ground truth is needed. The key insight:

```
E[noisy₁] = E[noisy₂] = clean signal

If we train to map noisy₁ → noisy₂ (instead of noisy → clean),
the optimal network output is the same: the clean signal.
```

### Mathematical Justification

For zero-mean, independent noise:
- Target: y = x + η₂ (noisy observation 2)
- Input: x̂ = x + η₁ (noisy observation 1)
- MSE loss: E[||f(x̂) - y||²] = E[||f(x̂) - x||²] + E[||η₂||²]
- The noise term is constant → minimizing this loss is equivalent to minimizing
  the reconstruction error with respect to the clean signal x

**Requirement**: Noise must be zero-mean and independent between the two observations.

## Application to Synchrotron Data

### Tomography

```
Scan 1 (fast, noisy) → Reconstruction 1 → Input
Scan 2 (fast, noisy) → Reconstruction 2 → Target

Train: f(Recon₁) → Recon₂
Inference: f(noisy_recon) → denoised_recon
```

**Practical implementation**:
- Collect two rapid back-to-back scans of the same sample
- Each scan is independently noisy but captures the same structure
- Network learns to extract the common (signal) component

### XRF Microscopy

```
Scan 1 (short dwell time) → Noisy elemental maps → Input
Scan 2 (short dwell time) → Noisy elemental maps → Target
```

### Spectroscopy

```
Spectrum scan 1 (quick) → Noisy spectrum → Input
Spectrum scan 2 (quick) → Noisy spectrum → Target
```

## Implementation

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class NoisyPairDataset(Dataset):
    """Dataset of paired noisy observations."""

    def __init__(self, noisy_1, noisy_2, patch_size=64):
        self.noisy_1 = noisy_1  # First noisy observation
        self.noisy_2 = noisy_2  # Second noisy observation
        self.patch_size = patch_size

    def __len__(self):
        return self.noisy_1.shape[0] * 100  # oversample with random crops

    def __getitem__(self, idx):
        slice_idx = idx % self.noisy_1.shape[0]
        img1 = self.noisy_1[slice_idx]
        img2 = self.noisy_2[slice_idx]

        # Random crop
        h, w = img1.shape
        y = torch.randint(0, h - self.patch_size, (1,)).item()
        x = torch.randint(0, w - self.patch_size, (1,)).item()

        patch1 = img1[y:y+self.patch_size, x:x+self.patch_size]
        patch2 = img2[y:y+self.patch_size, x:x+self.patch_size]

        return patch1.unsqueeze(0).float(), patch2.unsqueeze(0).float()

# Training loop
model = UNet(in_channels=1, out_channels=1)  # Any denoising architecture
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(100):
    for noisy_input, noisy_target in dataloader:
        pred = model(noisy_input)
        loss = nn.MSELoss()(pred, noisy_target)  # Noisy target is fine!

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Variants

### Noise2Void (N2V)

Requires only a **single noisy image** (no pairs):

- During training, mask the center pixel of each receptive field
- Network predicts the masked pixel from surrounding context
- Exploits spatial noise independence (neighboring pixels have independent noise)
- Lower quality than Noise2Noise but no repeat scans needed

### Noise2Self

Generalization of Noise2Void with theoretical guarantees:
- J-invariant network (output at pixel j doesn't depend on input at pixel j)
- Provably optimal under mild noise assumptions

### Noise2Same

Further extension that works with single noisy images and arbitrary noise distributions.

## Synchrotron-Specific Considerations

### Advantages
1. **No clean reference needed**: Critical when high-dose "clean" scan is impractical
2. **Dose reduction**: Two half-dose scans → train → denoise single half-dose scan
3. **In-situ studies**: Can use temporal neighbors as pairs for slowly-changing samples
4. **Flexible**: Works for any modality producing noisy images

### Challenges
1. **Repeat scans required**: Must acquire two noisy observations of the same scene
2. **Static sample assumption**: Sample must not change between scans
3. **Noise independence**: Systematic artifacts (rings, streaks) violate independence
4. **Lower quality**: Typically 1-2 dB PSNR below supervised methods with clean targets

### When to Use Noise2Noise vs TomoGAN

| Criterion | Noise2Noise | TomoGAN |
|-----------|-------------|---------|
| Clean reference available? | No | Yes |
| Repeat scans feasible? | Yes | Not needed |
| Sample radiation sensitive? | Good (half dose each) | Need full-dose reference |
| Image quality | Good | Best |
| Training stability | Stable | GAN training challenges |
| Generalization | Per-sample | Can generalize across samples |

## Results

Typical performance comparison on synchrotron tomography data:

| Method | PSNR (dB) | SSIM | Notes |
|--------|-----------|------|-------|
| Noisy input | 25.0 | 0.72 | Baseline |
| BM3D | 30.1 | 0.88 | Classical, no training |
| **Noise2Noise** | **32.5** | **0.92** | Self-supervised |
| TomoGAN (supervised) | 34.5 | 0.95 | Needs clean target |

*Values representative; varies by noise level and sample type.*
