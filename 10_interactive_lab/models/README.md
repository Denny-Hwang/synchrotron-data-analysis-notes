---
doc_id: LAB-MODELS-001
title: Pretrained Models — Lazy-Download Recipes
status: accepted
version: 0.1.0
last_updated: 2026-05-05
supersedes: null
related: [10_interactive_lab/README.md, 10_interactive_lab/docs/external_data_sources.md]
---

# Pretrained Models — Lazy-Download Recipes

## Why no weights are bundled

Pretrained model checkpoints are **not redistributed** in this repository. Reasons:

1. **Size.** Most usable checkpoints are 30–500 MB; the repo would balloon past free GitHub limits.
2. **License diversity.** Some weights are GPL (Topaz), some CC BY-NC (TomoGAN). Bundling them creates a license-compatibility minefield.
3. **Versioning.** Model authors update weights without notice; pinning a stale copy here would mislead users.

Instead, we ship **lazy-download recipes** — see `lazy_download_recipes.yaml`. The Streamlit Lab page calls `pooch.retrieve(...)` on first use, verifies the SHA-256 hash, and caches under the user's OS cache directory.

## Two tiers of models

### Tier 1 — Native synchrotron models (download from authors)

These are domain-specialized and benchmark-relevant, but live on author-hosted GitHub releases or institutional file shares (Box, Google Drive, OneDrive). None are on the Hugging Face Hub.

| Model | Where it lives | Size | License |
|---|---|---|---|
| **TomoGAN** generator | https://github.com/tomography/TomoGAN/releases | ~50 MB | CC BY-NC |
| **TomoGAN** VGG19 loss net | https://github.com/lzhengchun/TomoGAN/blob/master/vgg19_weights_notop.h5 | 80 MB | (Keras default) |
| **Topaz-Denoise** unet, unet-small, fcnet, fcnet2 | https://github.com/tbepler/topaz/tree/master/topaz/pretrained/denoise | 30–60 MB each | GPL-3.0 |
| **Topaz-Denoise** unet-3d, unet-3d-21x21x21 | same path | 100 MB each | GPL-3.0 |
| **CryoDRGN** zoo models | https://github.com/ml-struct-bio/cryodrgn (releases) | varies | GPL |
| **edgePtychoNN** | https://github.com/AnakhaVB/edgePtychoNN (releases) | < 50 MB | MIT |

### Tier 2 — Generic image-restoration baselines on Hugging Face

Use these as **drop-in baselines** to compare against domain-specific models. They are not synchrotron-specific.

| HF model id | Architecture | License | Best for |
|---|---|---|---|
| `mikestealth/nafnet-models` | NAFNet | MIT | General Gaussian denoise / SIDD |
| KAIR SwinIR weights (paper page `huggingface.co/papers/2108.10257`) | Swin Transformer | Apache-2.0 | Gray/color denoise + SR |
| `caidas/swin2SR-classical-sr-x2-64` | Swin2SR | Apache-2.0 | 2× super-resolution |
| `Geonmo/laion-aesthetic-6pls` (text-to-image, only as proxy)| – | – | not relevant – kept here for transparency |

These can be loaded with `transformers` or `diffusers` directly:

```python
from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor
processor = Swin2SRImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
model     = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
```

## What is NOT here

- We do **not** mirror weights ourselves.
- We do **not** mirror datasets that exceed our free-tier budget (see `../docs/external_data_sources.md`).
- We do **not** include any HF Inference Endpoint or paid API; the Lab is meant to run on the user's own CPU/GPU or on a free-tier Streamlit Cloud instance.

## Streaming-download safety

When you implement the actual fetch in the Streamlit page, please:

1. Pin a `known_hash` in `lazy_download_recipes.yaml` (compute on first vetted download).
2. Set a hard timeout and offer a fallback ("network unavailable — see `external_data_sources.md`").
3. Show the user the URL and license **before** the download starts (especially for CC BY-NC weights).
