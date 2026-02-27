# Paper Review: Full-Stack Deep Learning Pipeline for Synchrotron X-Ray Tomography

## Metadata

| Field              | Value                                                                                  |
|--------------------|----------------------------------------------------------------------------------------|
| **Title**          | Deep Learning for Full-Stack Synchrotron X-Ray Tomography: From Preprocessing to Segmentation |
| **Authors**        | Wang, G.; Zhang, T.; Gao, H.; Woollard, M.; Ren, L.; Liu, H.                         |
| **Journal**        | Fundamental Research, 3(6), 1040--1052                                                 |
| **Year**           | 2023                                                                                   |
| **DOI**            | [10.1016/j.fmre.2023.11.003](https://doi.org/10.1016/j.fmre.2023.11.003)              |
| **Beamline**       | Multi-facility review (SSRF, APS, ESRF, Diamond, Swiss Light Source)                   |

---

## TL;DR

This review article presents a unified vision for applying deep learning across the
entire synchrotron tomography pipeline -- from sinogram preprocessing and artifact
correction through reconstruction, denoising, and semantic segmentation -- arguing
that end-to-end integration yields greater quality gains than optimizing any single
stage in isolation.

---

## Background & Motivation

Synchrotron X-ray tomography produces terabyte-scale 3D datasets requiring multi-stage
processing: ring artifact removal, sinogram inpainting for missing angles, tomographic
reconstruction, post-reconstruction denoising, and finally semantic or instance
segmentation of material phases. Traditionally, each stage uses independent, hand-tuned
algorithms (e.g., Fourier filtering for rings, FBP for reconstruction, watershed for
segmentation). Deep learning has been applied to individual stages with demonstrated
success, but these efforts remain siloed. This paper surveys DL approaches for each
stage and makes the case that a full-stack integration -- where the output of one DL
module feeds directly into the next, and gradients can optionally flow end-to-end --
can compound quality improvements and reduce total computational cost.

---

## Method

The paper surveys and categorizes DL methods for four pipeline stages:

1. **Preprocessing (sinogram domain)**:
   - Ring artifact removal via 1D CNNs operating on sinogram columns.
   - Missing angle inpainting via generative models (U-Net, GAN-based).
   - Stripe correction using wavelet-CNN hybrids.

2. **Reconstruction**:
   - Learned post-processing of FBP outputs (residual learning on FBP images).
   - Direct learned reconstruction from sinograms (FBPConvNet, iCT-Net).
   - Physics-informed neural networks embedding the Radon transform as a
     differentiable layer.

3. **Denoising**:
   - Supervised denoising (paired low/high-dose data): U-Net, RED-CNN.
   - Self-supervised: Noise2Noise, Noise2Void for cases without clean references.
   - GAN-based: TomoGAN and variants for perceptually sharp denoising.

4. **Segmentation**:
   - Semantic segmentation: U-Net, DeepLab, SegNet for multi-phase materials.
   - Instance segmentation: Mask R-CNN for particle-level analysis.
   - Weakly supervised and active-learning approaches to reduce annotation cost.

5. **End-to-end integration**:
   - Joint training of reconstruction + denoising modules.
   - Differentiable rendering pipelines that backpropagate segmentation loss
     through reconstruction.
   - Transfer learning across facilities to mitigate domain shift.

---

## Key Results

| Metric / Finding                              | Value / Detail                                    |
|-----------------------------------------------|---------------------------------------------------|
| Ring artifact removal (DL vs. Fourier)        | 3-5 dB PSNR improvement reported across studies   |
| Sparse-angle reconstruction (DL vs. FBP)      | Viable at 1/4 to 1/8 standard projection count    |
| Denoising gain (supervised DL vs. BM3D)       | +2 to +5 dB PSNR advantage across studies         |
| Segmentation (DL vs. Otsu/watershed)          | +10-20% Dice score on multi-phase materials       |
| End-to-end vs. stage-by-stage                 | Additional +1-2 dB PSNR from joint optimization   |
| Throughput improvement                        | 10-100x speedup over iterative reconstruction     |

---

## Data & Code Availability

| Resource       | Link / Note                                                           |
|----------------|-----------------------------------------------------------------------|
| **Code**       | No unified codebase; individual methods reference their own repos     |
| **Data**       | Public benchmarks cited: TomoBank, foam phantom datasets              |
| **License**    | N/A (review article)                                                  |
| **Reproducibility Score** | **2 / 5** -- As a review, it synthesizes others' work; no single unified implementation is provided. Individual cited works vary in reproducibility. |

---

## Strengths

- Provides a comprehensive taxonomy of DL methods across all tomography pipeline
  stages, making it an excellent reference for practitioners entering the field.
- Makes a compelling quantitative argument for end-to-end integration rather than
  stage-wise optimization, supported by cross-study comparisons.
- Identifies key open problems (domain shift across facilities, annotation scarcity,
  uncertainty quantification) that set a research agenda.
- Covers both supervised and self-supervised paradigms, acknowledging the practical
  challenge of obtaining paired training data at synchrotrons.
- Discusses computational requirements and deployment considerations relevant to
  real facility operations.

---

## Limitations & Gaps

- As a review, it does not present novel methods or experiments; all quantitative
  claims are drawn from cited works with varying experimental conditions.
- The end-to-end integration argument is supported more by logical reasoning than
  by rigorous ablation studies on a single, controlled benchmark.
- Limited discussion of failure modes: when does DL introduce artifacts (hallucinated
  features, texture bias) that are worse than classical algorithm artifacts?
- Does not address the regulatory or validation requirements for DL-processed data
  in fields with strict standards (medical imaging, nuclear materials).
- Missing coverage of emerging architectures (vision transformers, diffusion models
  for reconstruction) that have appeared since submission.

---

## Relevance to eBERlight

This review serves as an architectural blueprint for eBERlight's tomography pipeline:

- **Pipeline design**: eBERlight should adopt the full-stack DL philosophy, building
  modular but jointly optimizable stages from sinogram to segmentation.
- **Module selection**: The survey's benchmarking summaries help prioritize which
  DL modules to implement first (denoising and segmentation show the largest gains).
- **Self-supervised priority**: Given the difficulty of collecting paired data at
  APS-U during early commissioning, self-supervised methods (Noise2Void, learned
  reconstruction without clean references) should be prioritized.
- **Benchmark infrastructure**: eBERlight should contribute APS-U datasets to
  TomoBank to enable the cross-facility benchmarking the review advocates.

---

## Actionable Takeaways

1. **Adopt modular full-stack architecture**: Design eBERlight's tomography pipeline
   with standardized tensor interfaces between stages, enabling future joint training.
2. **Start with denoising + segmentation**: These stages show the largest marginal
   gains; deploy U-Net-based modules as initial eBERlight tomography capabilities.
3. **Build APS-U benchmark suite**: Collect and curate paired/unpaired tomographic
   datasets during APS-U commissioning for training and cross-facility benchmarking.
4. **Self-supervised first**: Prioritize Noise2Void and Noise2Noise implementations
   that do not require clean ground truth, as paired data will be scarce initially.
5. **Track emerging architectures**: Monitor vision transformer and diffusion model
   approaches for reconstruction and denoising as potential next-generation upgrades.

---

*Reviewed for the eBERlight Research Archive, 2026-02-27.*
