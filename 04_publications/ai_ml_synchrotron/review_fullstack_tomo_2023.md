# Paper Review: Full-Stack Deep Learning Pipeline for Synchrotron Tomography

## Metadata

| Field              | Value                                                                                  |
|--------------------|----------------------------------------------------------------------------------------|
| **Title**          | Full-Stack Deep Learning Pipeline for Synchrotron Tomography                           |
| **Authors**        | Wang, G.; Zhang, T.; Gao, H.; Woollard, M.; Ren, L.; Liu, H.                         |
| **Journal**        | Fundamental Research, 3(6), 1040--1052                                                 |
| **Year**           | 2023                                                                                   |
| **DOI**            | [10.1016/j.fmre.2023.11.003](https://doi.org/10.1016/j.fmre.2023.11.003)              |
| **Beamline**       | Multi-facility (APS, SSRF, ESRF, Diamond Light Source, Swiss Light Source)              |
| **Modality**       | X-ray computed tomography (micro-CT, nano-CT)                                           |

---

## TL;DR

This paper presents a unified end-to-end deep learning pipeline for synchrotron X-ray
tomography that integrates multiple DL models across the entire processing chain --
from sinogram preprocessing (ring artifact removal, missing angle inpainting) through
tomographic reconstruction, post-reconstruction denoising, and semantic segmentation
of material phases. The authors argue that joint optimization across pipeline stages
yields 1--2 dB additional PSNR improvement over stage-by-stage processing, and provide
benchmarks on APS tomography datasets with practical deployment considerations for
facility-scale operations.

---

## Background & Motivation

Synchrotron X-ray tomography produces terabyte-scale 3D datasets requiring multi-stage
processing: sinogram preprocessing (ring artifacts, missing angles), reconstruction,
denoising, and segmentation. Traditionally each stage uses independent, hand-tuned
algorithms (Fourier filtering, FBP, BM3D, Otsu thresholding). Deep learning has been
applied to individual stages -- TomoGAN for denoising, FBPConvNet for reconstruction,
U-Net for segmentation -- but these efforts remain isolated, with error propagation
between stages unaccounted for.

This paper makes the case that full-stack integration, where DL modules are trained
with awareness of downstream tasks and gradients can flow end-to-end, compounds
quality improvements and reduces total computational cost. The practical motivation
is the APS-U upgrade, which will increase tomographic data rates by an order of
magnitude, demanding automated pipelines with minimal human intervention.

---

## Method

### Data

| Item | Details |
|------|---------|
| **Data source** | APS 2-BM and 32-ID tomography beamlines; SSRF BL13W; TomoBank public datasets |
| **Sample type** | Metal alloy foams, battery electrodes, geological cores, biological tissue |
| **Data dimensions** | Sinograms: 2048 projections x 2048 detector pixels; volumes: 2048^3 voxels |
| **Preprocessing** | Flat/dark field normalization, phase retrieval (Paganin), center-of-rotation correction |

### Model / Algorithm

The pipeline integrates four DL modules with standardized tensor interfaces between
stages:

1. **Preprocessing (sinogram domain)**: Ring artifact removal via 1D CNN on sinogram
   columns; missing-angle inpainting via U-Net generative model; stripe correction
   using a wavelet-CNN hybrid.

2. **Reconstruction**: Learned post-processing (FBP + residual CNN for artifact
   correction), direct learned reconstruction (FBPConvNet with differentiable Radon
   transform), and physics-informed variants enforcing data consistency.

3. **Denoising**: Supervised U-Net (paired low/high-dose, following TomoGAN paradigm)
   and self-supervised Noise2Void for cases without clean references.

4. **Segmentation**: Semantic segmentation (3D U-Net), instance segmentation (Mask
   R-CNN), and an active learning extension that reduces annotation burden by 60--70%
   by querying humans only on ambiguous slices.

5. **End-to-end integration**: Joint training with combined loss (L_reconstruction +
   alpha * L_denoising + beta * L_perceptual); differentiable pipeline backpropagating
   segmentation loss through reconstruction; transfer learning across facilities.

### Pipeline

```
Raw sinograms --> Ring/stripe correction (1D CNN) --> Missing angle inpainting (U-Net)
    --> Tomographic reconstruction (FBPConvNet / FBP + residual CNN)
    --> Denoising (U-Net / Noise2Void) --> Segmentation (3D U-Net / Mask R-CNN)
    --> Quantitative analysis (phase fractions, porosity, particle statistics)
```

All modules communicate via standardized NumPy/PyTorch tensor interfaces, enabling
individual module replacement without pipeline redesign. An optional end-to-end
training mode fine-tunes the full chain jointly.

---

## Key Results

| Metric / Finding                              | Value / Detail                                    |
|-----------------------------------------------|---------------------------------------------------|
| Ring artifact removal (DL vs. Fourier filter)  | +3.5 dB PSNR improvement on APS 2-BM data        |
| Sparse-angle reconstruction (DL vs. FBP)       | Viable at 1/4 to 1/8 standard projection count    |
| Denoising gain (supervised DL vs. BM3D)        | +3.2 dB PSNR on 4x dose-reduced data              |
| Denoising gain (Noise2Void vs. BM3D)           | +1.8 dB PSNR (no paired data required)             |
| Segmentation (3D U-Net vs. Otsu/watershed)     | +15% mean Dice score on multi-phase alloys         |
| End-to-end vs. stage-by-stage (full pipeline)  | +1.5 dB PSNR from joint optimization               |
| End-to-end segmentation Dice improvement       | +3% Dice score from joint reconstruction-segmentation training |
| Throughput (full pipeline, single A100)         | ~2 seconds per 2048x2048 slice (all stages)        |
| Throughput (FBP only, same hardware)            | ~0.1 seconds per slice                              |
| Comparison: full pipeline vs. classical chain   | 10--50x faster than iterative recon + manual segmentation |

### Key Figures

- **Figure 1**: Full pipeline architecture diagram showing the four DL modules with
  tensor interfaces, loss functions at each stage, and optional end-to-end gradient
  flow paths.
- **Figure 4**: Ring artifact removal comparison on APS 2-BM foam data: classical
  Fourier filtering leaves residual rings while the 1D CNN produces clean sinograms
  with preserved fine structure.
- **Figure 6**: Sparse-angle reconstruction comparison at 1/4 and 1/8 projection
  counts: FBP shows severe streak artifacts while FBPConvNet maintains structural
  fidelity down to 1/4 projections.
- **Figure 9**: End-to-end joint training ablation: segmentation Dice score plotted
  as a function of pipeline configuration (stage-by-stage vs. joint reconstruction-
  denoising vs. full end-to-end), demonstrating cumulative improvement.
- **Table 3**: Cross-facility transfer learning results showing that models trained
  on APS data maintain 85--90% of their performance when applied to SSRF and ESRF
  data without fine-tuning, and reach 95%+ with 100 fine-tuning slices.

---

## Data & Code Availability

| Resource       | Link / Note                                                           |
|----------------|-----------------------------------------------------------------------|
| **Code**       | No unified codebase released; individual modules reference their source repos |
| **Data**       | Public benchmarks cited: TomoBank, APS foam phantom datasets           |
| **License**    | N/A (review-style paper with benchmarking)                             |
| **Reproducibility Score** | **2 / 5** -- Individual module implementations are referenced but no unified pipeline code is released. Benchmark datasets are partially public via TomoBank. Reproducing end-to-end results requires assembling multiple codebases. |

---

## Strengths

- Comprehensive framework treating the entire tomography chain as an integrated DL
  system rather than independent optimization problems.
- Quantitative validation: end-to-end joint training yields +1.5 dB PSNR and +3%
  Dice over independently optimized stages.
- Covers both supervised and self-supervised paradigms, acknowledging the practical
  difficulty of obtaining paired training data at synchrotrons.
- Cross-facility transfer learning shows 85--90% performance retention without
  fine-tuning, demonstrating unexpected model portability.
- Practical deployment considerations (throughput, memory, GPU utilization) make
  the work directly relevant to facility operations.
- Active learning reduces human annotation burden by 60--70%.

---

## Limitations & Gaps

- No unified open-source implementation is provided; the paper describes an
  integrated pipeline conceptually but the community cannot download and run it,
  limiting practical adoption.
- The end-to-end training improvement (+1.5 dB, +3% Dice), while statistically
  significant, is modest compared to the gains from introducing DL at any single
  stage, and may not justify the engineering complexity of joint training in all
  deployment scenarios.
- Limited analysis of failure modes: the paper does not systematically characterize
  when DL-based processing introduces artifacts (hallucinated features, smoothed
  grain boundaries) that are qualitatively worse than classical algorithm artifacts.
- Does not address validation requirements for DL-processed data in fields with
  strict quantitative standards (e.g., NIST-traceable porosity measurements,
  medical imaging regulations).
- Missing coverage of emerging architectures (vision transformers, diffusion models
  for reconstruction) that have appeared since the paper's submission and could
  offer substantial improvements.
- The throughput benchmark (~2 s/slice for the full pipeline) may be insufficient
  for real-time operation at APS-U data rates, where multiple slices per second
  may need to be processed.

---

## Relevance to eBERlight

This paper serves as an architectural blueprint for eBERlight's tomography pipeline:

- **Pipeline architecture**: eBERlight should adopt the full-stack DL philosophy
  with modular but jointly optimizable stages from sinogram to segmentation,
  using the standardized tensor interfaces described in this work.
- **Module prioritization**: The benchmarking results help prioritize implementation
  order: denoising and segmentation show the largest individual-stage gains and
  should be deployed first; end-to-end training can be added as an optimization
  pass later.
- **Self-supervised priority**: Given the difficulty of collecting paired training
  data during early APS-U commissioning, self-supervised methods (Noise2Void,
  active learning for segmentation) should be prioritized over supervised
  approaches that require clean reference data.
- **Cross-facility models**: The transfer learning results suggest eBERlight can
  bootstrap with models trained on existing APS and TomoBank data, then fine-tune
  with APS-U commissioning data rather than training from scratch.
- **Benchmark contribution**: eBERlight should contribute APS-U tomographic datasets
  to TomoBank to enable the cross-facility benchmarking this paper advocates.
- **Applicable beamlines**: APS 2-BM, 32-ID, and other APS-U tomography endstations.
- **Priority**: High -- provides the architectural template for eBERlight's
  tomography processing infrastructure.

---

## Actionable Takeaways

1. **Adopt modular full-stack architecture**: Design eBERlight's tomography pipeline
   with standardized tensor interfaces between DL modules, enabling independent
   module development with future joint-training capability.
2. **Deploy denoising + segmentation first**: These stages show the largest marginal
   quality gains; implement U-Net-based denoising and 3D U-Net segmentation as
   initial eBERlight tomography capabilities.
3. **Build APS-U benchmark suite**: Collect and curate paired and unpaired tomographic
   datasets during APS-U commissioning for model training, validation, and cross-
   facility benchmarking via TomoBank contribution.
4. **Self-supervised first**: Prioritize Noise2Void for denoising and active learning
   for segmentation to minimize dependence on paired ground-truth data during early
   operations.
5. **Track emerging architectures**: Monitor vision transformer and diffusion model
   approaches for tomographic reconstruction and denoising as potential next-
   generation module replacements.
6. **Throughput optimization**: Profile the full pipeline against APS-U data rate
   requirements and identify whether edge compute or GPU cluster deployment is
   needed for real-time operation.

---

## BibTeX Citation

```bibtex
@article{wang2023fullstack_tomo,
  title     = {Full-Stack Deep Learning Pipeline for Synchrotron Tomography},
  author    = {Wang, Ge and Zhang, Tianyuan and Gao, Hao and Woollard, Matthew
               and Ren, Liang and Liu, Hanming},
  journal   = {Fundamental Research},
  volume    = {3},
  number    = {6},
  pages     = {1040--1052},
  year      = {2023},
  publisher = {Elsevier},
  doi       = {10.1016/j.fmre.2023.11.003}
}
```

---

*Reviewed for the eBERlight Research Archive, 2026-02-27.*
