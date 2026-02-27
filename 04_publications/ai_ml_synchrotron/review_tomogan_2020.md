# Paper Review: TomoGAN -- Low-Dose Synchrotron X-Ray Tomography with Generative Adversarial Networks

## Metadata

| Field              | Value                                                                                  |
|--------------------|----------------------------------------------------------------------------------------|
| **Title**          | TomoGAN: Low-Dose Synchrotron X-Ray Tomography with Generative Adversarial Networks    |
| **Authors**        | Liu, Z.; Bicer, T.; Kettimuthu, R.; Deng, D.; Foster, I.                              |
| **Journal**        | Journal of the Optical Society of America A, 37(3), 422--434                           |
| **Year**           | 2020                                                                                   |
| **DOI**            | [10.1364/JOSAA.375595](https://doi.org/10.1364/JOSAA.375595)                           |
| **Beamline**       | APS 2-BM (synchrotron micro-tomography)                                                |

---

## TL;DR

TomoGAN applies a conditional generative adversarial network to denoise low-dose
synchrotron X-ray tomographic reconstructions, achieving image quality comparable to
full-dose scans while enabling a 4--10x reduction in radiation exposure or acquisition
time.

---

## Background & Motivation

Synchrotron X-ray micro-tomography delivers 3D structural imaging at micron to sub-micron
resolution, but high photon fluence is required for adequate signal-to-noise ratio (SNR),
which can damage radiation-sensitive samples (biological tissue, polymers, batteries) and
limits temporal resolution for in-situ experiments. Reducing dose degrades reconstructed
image quality through amplified Poisson noise and streak artifacts. Classical denoising
filters (median, BM3D, non-local means) trade spatial resolution for noise suppression.
TomoGAN was designed to learn the mapping from low-dose to high-dose tomographic slices
using adversarial training, preserving structural detail that classical filters blur.

---

## Method

1. **Data collection**: Paired low-dose and high-dose sinograms acquired at APS 2-BM by
   varying exposure time per projection (factor of 4x and 10x reduction).
2. **Reconstruction**: Both dose levels reconstructed via filtered back-projection (FBP)
   using TomoPy, yielding paired noisy/clean 2D slice images.
3. **Network architecture**: Generator is a U-Net-style encoder-decoder with skip
   connections (4 downsampling/upsampling blocks). Discriminator is a PatchGAN
   classifier operating on 70x70 pixel patches.
4. **Loss function**: Weighted combination of pixel-wise L1 loss, adversarial loss
   (binary cross-entropy), and perceptual loss (VGG-16 feature matching) to balance
   fidelity, sharpness, and perceptual quality.
5. **Training**: Trained on ~10,000 paired slice images (512x512 pixels). Augmentation
   via random flips and rotations. Adam optimizer, learning rate 2e-4, batch size 16,
   ~200 epochs on 4 NVIDIA V100 GPUs.
6. **Inference**: Single forward pass per slice; ~15 ms per 512x512 image on one GPU.

---

## Key Results

| Metric                              | Value / Finding                                       |
|-------------------------------------|-------------------------------------------------------|
| PSNR improvement (4x dose reduction)| +4.2 dB over noisy input; within 0.8 dB of full dose |
| SSIM (4x dose reduction)            | 0.94 (vs. 0.78 noisy, 0.97 full dose)                |
| PSNR improvement (10x dose)         | +6.1 dB over noisy input                              |
| Edge preservation (Sobel metric)    | 92% edge fidelity retained at 4x dose reduction       |
| Inference time                      | ~15 ms per 512x512 slice (single V100)                |
| Comparison vs. BM3D                 | +1.8 dB PSNR, +0.06 SSIM advantage                   |
| Training time                       | ~8 hours on 4x V100                                   |

---

## Data & Code Availability

| Resource       | Link / Note                                                           |
|----------------|-----------------------------------------------------------------------|
| **Code**       | [github.com/zhengchun/TomoGAN](https://github.com/zhengchun/TomoGAN) |
| **Data**       | Paired datasets hosted at Argonne (Globus endpoint)                   |
| **License**    | MIT                                                                    |
| **Reproducibility Score** | **4 / 5** -- Code and data publicly available; training requires multi-GPU resources but is well-documented. |

---

## Strengths

- Achieves 4--10x dose reduction while maintaining structural fidelity, directly
  enabling faster or less damaging experiments.
- Adversarial + perceptual loss combination preserves fine edges and textures that
  pure L1/L2 losses would smooth away.
- Fast inference (~15 ms/slice) is compatible with real-time reconstruction pipelines.
- Open-source code with publicly accessible paired training data lowers the barrier
  to adoption at other facilities.
- Demonstrated on real synchrotron data (not just simulations), increasing practical
  credibility.

---

## Limitations & Gaps

- Requires matched low-dose/high-dose paired data for training, which is expensive to
  collect and may not generalize across sample types or beamline configurations.
- Evaluated primarily on FBP reconstructions; interaction with iterative reconstruction
  methods (MBIR, SIRT) is not explored.
- GAN training is inherently unstable; no analysis of failure modes, hallucination
  artifacts, or mode collapse is provided.
- No uncertainty quantification: the network provides a point estimate without
  confidence intervals on the denoised output.
- 2D slice-by-slice processing ignores 3D spatial continuity, potentially introducing
  inter-slice inconsistencies.

---

## Relevance to eBERlight

TomoGAN is directly relevant to eBERlight's dose-optimization and throughput goals:

- **Dose-adaptive scanning**: eBERlight's scan planner could use TomoGAN to maintain
  image quality while reducing dose in radiation-sensitive regions, enabling
  heterogeneous dose allocation across a sample.
- **Real-time pipeline integration**: The ~15 ms inference time fits within eBERlight's
  latency budget for streaming reconstruction at APS-U frame rates.
- **Training data generation**: eBERlight could automate collection of paired dose
  datasets during commissioning runs to build facility-specific TomoGAN models.
- **Uncertainty extension**: Coupling TomoGAN with Monte Carlo dropout or ensemble
  methods could provide pixel-wise uncertainty maps for eBERlight's decision engine.

---

## Actionable Takeaways

1. **Deploy for APS-U tomography**: Retrain TomoGAN on APS-U 2-BM commissioning data
   and benchmark against updated detector noise profiles.
2. **Add uncertainty**: Implement MC-dropout or deep ensemble variants to produce
   confidence maps alongside denoised outputs.
3. **3D extension**: Extend the architecture to 3D convolutional blocks or use 2.5D
   (multi-slice) input to enforce inter-slice consistency.
4. **Self-supervised alternative**: Explore Noise2Noise or Noise2Void training
   strategies that eliminate the need for paired high-dose ground truth.
5. **Integrate with Bluesky**: Package TomoGAN inference as a Bluesky callback for
   on-the-fly denoising during live reconstruction at eBERlight beamlines.

---

*Reviewed for the eBERlight Research Archive, 2026-02-27.*
