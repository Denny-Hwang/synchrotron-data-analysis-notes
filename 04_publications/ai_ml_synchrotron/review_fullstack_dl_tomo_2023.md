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
| **Modality**       | Synchrotron X-ray Tomography                                                          |

---

## TL;DR

This review article presents a unified vision for applying deep learning across
the entire synchrotron tomography pipeline -- from sinogram preprocessing and
artifact correction through tomographic reconstruction, post-reconstruction
denoising, and semantic segmentation. The paper argues, with supporting cross-study
evidence, that end-to-end integration of DL modules across the pipeline yields
greater quality gains (additional +1-2 dB PSNR) than optimizing any single stage
in isolation. It provides a comprehensive taxonomy of DL methods per pipeline stage
and identifies open challenges (domain shift, annotation scarcity, uncertainty
quantification) that set the research agenda for the field.

---

## Background & Motivation

Synchrotron X-ray tomography is a workhorse technique producing terabyte-scale 3D
datasets that require multi-stage processing to convert raw detector data into
scientifically usable information:

1. **Sinogram preprocessing**: Ring artifact removal, stripe correction, missing
   angle compensation, and flat/dark field normalization in the projection domain.
2. **Tomographic reconstruction**: Converting sinograms to cross-sectional images
   via filtered back projection (FBP), iterative methods (SIRT, MBIR), or learned
   approaches.
3. **Post-reconstruction denoising**: Suppressing residual noise, streak artifacts,
   and reconstruction errors in the image domain.
4. **Segmentation**: Classifying each voxel into material phases, pore space, or
   structural features for quantitative 3D analysis.

Traditionally, each stage uses independent, hand-tuned algorithms developed by
different research communities. Classical methods (Fourier filtering for rings, FBP
for reconstruction, BM3D for denoising, watershed for segmentation) are mature but
have known performance ceilings. Deep learning has been applied to individual stages
with demonstrated success (TomoGAN for denoising, U-Net for segmentation), but
these efforts remain **siloed** -- each stage is optimized independently, with
error propagation between stages unaccounted for.

This paper surveys DL approaches for each stage and argues for a **full-stack
integration** philosophy where:
- The output of one DL module feeds directly into the next
- Error signals from downstream stages (e.g., segmentation accuracy) can propagate
  backward through upstream stages (e.g., reconstruction quality)
- Joint optimization across stages compounds quality improvements
- Shared feature representations reduce total computational cost

---

## Method

The paper surveys and categorizes DL methods across four pipeline stages:

### Stage 1: Preprocessing (Sinogram Domain)

- **Ring artifact removal**: 1D CNNs operating on sinogram columns to learn the
  ring pattern and subtract it. Outperforms Fourier filtering by 3-5 dB PSNR on
  benchmark datasets.
- **Missing angle inpainting**: Generative models (U-Net, GAN-based) that fill in
  missing angular ranges in sinograms, enabling limited-angle reconstruction at
  1/4 to 1/8 the standard projection count.
- **Stripe correction**: Wavelet-CNN hybrids that decompose sinograms into
  approximation and detail coefficients, process the detail bands with CNNs to
  isolate stripe artifacts, and reconstruct the corrected sinogram.

### Stage 2: Reconstruction

- **Learned post-processing**: Residual networks applied to FBP outputs to correct
  artifacts and enhance quality. These methods keep the FBP as a differentiable
  layer and learn residual corrections, combining physics with data-driven
  refinement.
- **Direct learned reconstruction**: Networks that map sinograms directly to images
  (FBPConvNet, iCT-Net), bypassing explicit FBP. These achieve superior quality at
  sparse angular sampling but require large training datasets.
- **Physics-informed neural networks**: Architectures that embed the Radon
  transform as a differentiable layer, constraining the network to produce outputs
  consistent with the measurement physics. These provide better generalization than
  purely data-driven approaches.

### Stage 3: Denoising

- **Supervised approaches**: U-Net, RED-CNN, and TomoGAN trained on paired
  low-dose/high-dose data. Achieve +2 to +5 dB PSNR over BM3D.
- **Self-supervised approaches**: Noise2Noise (paired noisy images without clean
  reference), Noise2Void (single noisy image, masking-based). Critical for
  synchrotron applications where clean reference data is impractical to acquire.
- **GAN-based approaches**: TomoGAN and variants that preserve perceptual texture
  quality at the risk of hallucinated features.

### Stage 4: Segmentation

- **Semantic segmentation**: U-Net, DeepLab, SegNet for multi-phase material
  classification. Achieves +10-20% Dice score over Otsu/watershed baselines.
- **Instance segmentation**: Mask R-CNN for particle-level analysis (counting,
  sizing, shape characterization of individual features).
- **Weakly supervised and active learning**: Approaches that reduce annotation
  cost by learning from sparse labels, bounding boxes, or interactively refined
  predictions. Critical because manual 3D annotation is extremely expensive.

### Stage 5: End-to-End Integration

- **Joint training**: Reconstruction + denoising modules trained jointly with a
  combined loss function, yielding +1-2 dB additional PSNR over separately trained
  modules.
- **Differentiable rendering**: Pipelines where segmentation loss backpropagates
  through reconstruction, encouraging the reconstructor to produce images that are
  easy to segment correctly.
- **Transfer learning across facilities**: Domain adaptation techniques to mitigate
  the shift between different synchrotron sources, detector types, and sample
  classes.

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

### Key Figures

- **Figure 1**: Full-stack pipeline diagram showing data flow from detector through
  all four stages to segmented volume, with DL modules annotated at each stage.
- **Figure 3**: Cross-study comparison table of DL vs. classical methods at each
  stage, with PSNR, SSIM, and Dice metrics aggregated across cited works.
- **Figure 5**: End-to-end joint training architecture showing differentiable
  connections between reconstruction, denoising, and segmentation modules with
  backpropagation pathways.

---

## Data & Code Availability

| Resource       | Link / Note                                                           |
|----------------|-----------------------------------------------------------------------|
| **Code**       | No unified codebase; individual methods reference their own repos     |
| **Data**       | Public benchmarks cited: TomoBank, foam phantom datasets              |
| **License**    | N/A (review article)                                                  |

**Reproducibility Score**: **2 / 5** -- As a review, it synthesizes others' work
rather than presenting a single reproducible implementation. No unified codebase is
provided. Individual cited works vary significantly in their reproducibility
(ranging from 1/5 to 5/5). The paper's value lies in its synthesis and architectural
vision rather than a specific reproducible result.

---

## Strengths

- **Comprehensive taxonomy**: Covers DL methods across all four tomography pipeline
  stages, making it the definitive reference for practitioners entering the field or
  planning a DL integration effort.
- **Compelling integration argument**: Makes a quantitative case for end-to-end
  pipeline optimization (additional +1-2 dB from joint training) supported by
  cross-study evidence, not just theoretical argument.
- **Identifies open problems**: Domain shift across facilities, annotation scarcity,
  uncertainty quantification, and self-supervised training are identified as key
  research gaps, setting a clear research agenda.
- **Covers both supervised and self-supervised**: Acknowledges the practical
  challenge of obtaining paired training data at synchrotrons and surveys solutions.
- **Multi-facility perspective**: Draws on work from SSRF, APS, ESRF, Diamond, and
  SLS, providing a facility-agnostic view.
- **Deployment considerations**: Discusses computational requirements, latency
  constraints, and infrastructure needs relevant to real facility operations.

---

## Limitations & Gaps

- **No novel methods or experiments**: As a review, all quantitative claims are
  drawn from cited works with varying experimental conditions, metrics, and
  datasets. Direct comparison across studies is approximate.
- **End-to-end integration evidence is thin**: The +1-2 dB joint training benefit
  is supported by few studies, each with different experimental setups. A
  controlled ablation study on a single benchmark would be more convincing.
- **Limited failure mode analysis**: Does not systematically discuss when DL
  introduces artifacts (hallucinated features, texture bias, adversarial
  vulnerability) that are worse than classical algorithm artifacts.
- **No regulatory/validation discussion**: Missing coverage of validation
  requirements for DL-processed data in fields with strict standards (medical
  imaging, nuclear materials, regulatory science).
- **Missing emerging architectures**: Does not cover vision transformers, diffusion
  models for reconstruction, or foundation models for scientific imaging that have
  appeared since the article's submission.

---

## Relevance to eBERlight

This review serves as an architectural blueprint for eBERlight's tomography
pipeline:

- **Pipeline architecture**: eBERlight should adopt the full-stack DL philosophy,
  building modular but jointly optimizable stages from sinogram to segmentation.
  Standardized tensor interfaces between stages enable future joint training.
- **Module prioritization**: The survey's benchmarking summaries help prioritize
  which DL modules to implement first. Denoising and segmentation show the largest
  marginal gains over classical methods and should be deployed first.
- **Self-supervised priority**: Given the difficulty of collecting paired data at
  APS-U during early commissioning, self-supervised methods (Noise2Void, Noise2Noise)
  should be prioritized over supervised approaches.
- **Benchmark contribution**: eBERlight should contribute APS-U datasets to
  TomoBank to enable the cross-facility benchmarking the review advocates.
- **Applicable beamlines**: All APS tomography beamlines (2-BM, 32-ID, etc.).
- **Priority**: **High** -- provides the conceptual framework for eBERlight's
  entire tomography pipeline strategy.

---

## Actionable Takeaways

1. **Adopt modular full-stack architecture**: Design eBERlight's tomography pipeline
   with standardized tensor interfaces between stages, enabling future joint
   optimization and module swapping.
2. **Start with denoising + segmentation**: These show the largest marginal gains;
   deploy U-Net-based modules as initial eBERlight tomography capabilities.
3. **Build APS-U benchmark suite**: Collect and curate paired/unpaired tomographic
   datasets during APS-U commissioning for training and cross-facility benchmarking.
   Contribute to TomoBank.
4. **Self-supervised first**: Prioritize Noise2Void and Noise2Noise implementations
   that do not require clean ground truth for denoising, as paired data will be
   scarce during initial APS-U operations.
5. **Track emerging architectures**: Monitor vision transformer and diffusion model
   approaches for reconstruction and denoising as potential next-generation upgrades
   to the eBERlight pipeline.
6. **Joint training roadmap**: Plan for eventual joint training of reconstruction +
   denoising + segmentation once individual modules are validated and sufficient
   training data is accumulated.

---

## Notes & Discussion

This review is the conceptual counterpart to the systems-level HPC integration
paper reviewed in `review_realtime_uct_hpc_2020.md`. While McClure et al. (2020)
demonstrates a working real-time pipeline using specific tools, this review provides
the broader architectural vision and algorithm-level recommendations. Together, they
define both the "what" (which DL modules) and the "how" (which infrastructure) for
eBERlight's tomography pipeline.

The paper's advocacy for end-to-end training is forward-looking but practically
challenging: joint optimization requires matched training data across all stages and
careful loss weighting. eBERlight should build modular stages first and pursue joint
training as a second-phase optimization once individual modules are validated.

---

## Review Metadata

| Field | Value |
|-------|-------|
| **Reviewed by** | eBERlight AI/ML Team |
| **Review date** | 2025-10-17 |
| **Last updated** | 2025-10-17 |
| **Tags** | tomography, deep-learning, pipeline, review, denoising, reconstruction, segmentation, full-stack |
