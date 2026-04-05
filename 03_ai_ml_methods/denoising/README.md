# Denoising Methods for Synchrotron Data

## Overview

Denoising is critical in synchrotron science because:
1. Reducing radiation dose (protecting biological samples) introduces noise
2. Faster scans (for time-resolved studies) mean fewer photons per pixel
3. Post APS-U data rates demand low-dose, high-throughput acquisition

## Method Classification

| Category | Training Data | Examples |
|----------|--------------|---------|
| **Supervised** | Paired noisy/clean images | TomoGAN, standard denoising CNNs |
| **Self-supervised (paired)** | Noisy pairs (no clean target) | Noise2Noise |
| **Self-supervised (single image)** | Single noisy image only | Noise2Void, Noise2Self, Neighbor2Neighbor |
| **Unsupervised** | No pairs needed | Deep image prior, BM3D |
| **GAN-based** | Paired data + adversarial training | TomoGAN (combines supervised + GAN) |

## Comparison

| Method | Clean Target? | Paired Data? | Quality | Training Cost | Artifacts |
|--------|:------------:|:----------:|:-------:|:------------:|:---------:|
| **Gaussian filter** | No | No | ★★ | None | Blurring |
| **NLM (Non-Local Means)** | No | No | ★★★ | None | Slow |
| **BM3D** | No | No | ★★★★ | None | Slow |
| **Supervised CNN** | Yes | Yes | ★★★★ | Medium | Possible hallucination |
| **TomoGAN** | Yes | Yes | ★★★★★ | High | Mode collapse risk |
| **Noise2Noise** | No | Noisy pairs | ★★★★ | Medium | Needs repeat scans |
| **Noise2Void** | No | No | ★★★ | Medium | Lower quality |
| **Noise2Self** | No | No | ★★★ | Medium | Lower quality |
| **Neighbor2Neighbor** | No | No | ★★★½ | Medium | Subsampling artifacts |

## Directory Contents

| File | Content |
|------|---------|
| [tomogan.md](tomogan.md) | GAN-based denoising for tomography |
| [noise2noise.md](noise2noise.md) | Self-supervised denoising without clean targets |
| [deep_residual_xrf.md](deep_residual_xrf.md) | Resolution enhancement via probe deconvolution |
| [noise2void.md](noise2void.md) | Self-supervised denoising from single noisy images (N2V, N2S, Neighbor2Neighbor) |
