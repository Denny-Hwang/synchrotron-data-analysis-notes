# Paper Review: Real-Time End-to-End AI+HPC Workflow for Micro-CT

## Metadata

| Field              | Value                                                                                  |
|--------------------|----------------------------------------------------------------------------------------|
| **Title**          | Real-Time AI+HPC Workflow for Micro-Computed Tomography: Reconstruction, Denoising, and Segmentation |
| **Authors**        | McClure, M.; Bicer, T.; Kettimuthu, R.; Foster, I.; Schwarz, N.                       |
| **Journal**        | Proceedings of IEEE International Conference on Systems, Man, and Cybernetics (SMC)    |
| **Year**           | 2020                                                                                   |
| **DOI**            | 10.1109/SMC42975.2020.9283269                                                         |
| **Beamline**       | APS 2-BM (micro-tomography)                                                           |

---

## TL;DR

An end-to-end AI+HPC workflow integrates tomographic reconstruction, DL-based
denoising, and automated segmentation into a single real-time pipeline running
on Argonne's HPC infrastructure, processing micro-CT data as it streams from the
beamline detector with sub-minute total latency.

---

## Background & Motivation

Synchrotron micro-computed tomography (micro-CT) at high-flux beamlines like APS 2-BM
generates data at rates exceeding 1 GB/s. Traditionally, data are written to disk,
transferred to a computing cluster, and processed offline -- a workflow that introduces
hours to days of latency between acquisition and usable 3D segmented volumes. This
lag prevents real-time feedback to experimenters, who cannot verify data quality or
adjust experimental parameters until long after the measurement. The authors developed
a streaming pipeline that couples beamline data acquisition directly to Argonne
Leadership Computing Facility (ALCF) resources, executing reconstruction (TomoPy),
DL denoising (TomoGAN variant), and segmentation (U-Net) within the time window of
a single tomographic scan rotation.

---

## Method

1. **Data streaming**: Projection images stream from the detector to ALCF via
   Globus data fabric over a dedicated 100 Gbps network link. A message queue
   (ZeroMQ) coordinates data availability notifications.
2. **Reconstruction stage**: TomoPy performs filtered back-projection (FBP) or
   gridrec on incoming sinograms. Distributed across multiple CPU nodes for
   parallelism, achieving reconstruction of a 2048x2048x2048 volume in < 20 s.
3. **Denoising stage**: A TomoGAN-derived U-Net model runs on NVIDIA V100 GPUs
   to suppress low-dose noise in reconstructed slices. Processing overlaps with
   reconstruction via a producer-consumer queue.
4. **Segmentation stage**: A pre-trained U-Net segments the denoised volume into
   material phases (pore space, solid matrix, inclusions). The model was trained
   on manually annotated micro-CT slices from similar specimen classes.
5. **Orchestration**: An Apache Airflow-based workflow manager coordinates the
   three stages, handles error recovery, and monitors resource utilization.
   Stage-level parallelism ensures the pipeline throughput matches the detector
   frame rate.
6. **Visualization**: Reconstructed, denoised, and segmented slices are streamed
   to a web-based viewer accessible at the beamline for real-time quality control.

---

## Key Results

| Metric                                    | Value / Finding                                   |
|-------------------------------------------|---------------------------------------------------|
| End-to-end latency (acquisition to segmented volume) | < 60 s for a 2048^3 volume                |
| Reconstruction throughput                 | ~100 slices/s (gridrec on 64 CPU nodes)            |
| Denoising throughput                      | ~200 slices/s (U-Net on 4x V100)                   |
| Segmentation throughput                   | ~150 slices/s (U-Net on 4x V100)                   |
| Pipeline vs. offline speedup              | 50--100x reduction in time to usable results       |
| Denoising quality (PSNR improvement)      | +3.5 dB over raw low-dose reconstruction           |
| Segmentation accuracy (Dice score)        | 0.92 on foam phantom, 0.87 on rock core samples   |
| Network bandwidth utilization             | ~40 Gbps sustained (of 100 Gbps available)         |

---

## Data & Code Availability

| Resource       | Link / Note                                                           |
|----------------|-----------------------------------------------------------------------|
| **Code**       | Workflow scripts partially available on GitHub; core TomoPy and TomoGAN repos referenced |
| **Data**       | Foam phantom dataset available via Globus; rock core data on request   |
| **License**    | Various (TomoPy: BSD, TomoGAN: MIT, workflow scripts: not specified)   |
| **Reproducibility Score** | **2 / 5** -- Individual components are open-source, but the orchestration layer and HPC-specific deployment scripts are not fully released. Reproducing requires ALCF-scale resources. |

---

## Strengths

- Demonstrates a complete, end-to-end pipeline from detector to segmented volume,
  proving that real-time synchrotron tomography analysis is technically feasible.
- Sub-minute latency enables intra-experiment decision-making: operators can verify
  data quality and adjust scan parameters before the next sample rotation.
- Producer-consumer architecture allows stages to overlap in time, maximizing
  throughput and GPU/CPU utilization.
- Uses established, community-supported tools (TomoPy, Globus) rather than
  building from scratch, improving maintainability.
- Quantifies network bandwidth requirements, providing practical infrastructure
  guidance for other facilities.

---

## Limitations & Gaps

- Requires dedicated 100 Gbps network and multi-node HPC allocation, which is not
  available at all facilities or at all times.
- The segmentation model is pre-trained on specific material classes; generalization
  to new specimen types requires retraining with annotated data.
- Error handling is described at a high level but not rigorously tested under
  failure conditions (network drops, GPU faults, corrupted projections).
- The denoising module is a simplified TomoGAN variant; comparison with state-of-
  the-art alternatives (self-supervised methods, diffusion models) is absent.
- No discussion of data provenance, versioning, or metadata management for the
  processed outputs, which is critical for scientific reproducibility.

---

## Relevance to eBERlight

This paper provides a systems-level template for eBERlight's real-time analysis
infrastructure:

- **Pipeline architecture**: eBERlight can adopt the producer-consumer, stage-
  overlapped design for its own multi-stage analysis pipelines across modalities
  (not just tomography).
- **ALCF integration**: The Globus-based data fabric and ALCF resource allocation
  patterns demonstrated here directly apply to eBERlight's planned use of ALCF
  for GPU inference and training.
- **Latency benchmarks**: The < 60 s end-to-end latency establishes a target
  for eBERlight's tomography pipeline; APS-U's higher data rates will require
  further optimization.
- **Feedback loop**: The web-based visualization component is a starting point
  for eBERlight's operator dashboard, though it needs to be extended with
  automated decision-making capabilities.

---

## Actionable Takeaways

1. **Adopt producer-consumer staging**: Implement eBERlight's analysis pipelines
   using ZeroMQ or Kafka message queues for stage-level parallelism and decoupling.
2. **Upgrade to Bluesky/Tiled**: Replace the custom data streaming layer with
   Bluesky event model and Tiled for metadata-rich, facility-standard data flow.
3. **Self-supervised denoising**: Replace the supervised TomoGAN with Noise2Void
   or similar self-supervised methods to eliminate the need for paired training data.
4. **Add automated QC**: Integrate automated quality control metrics (PSNR, SSIM,
   edge sharpness) that trigger alerts when pipeline output degrades below thresholds.
5. **Extend to APS-U data rates**: Profile the pipeline at APS-U expected data rates
   (5--10 GB/s) and identify bottlenecks requiring architectural changes.

---

*Reviewed for the eBERlight Research Archive, 2026-02-27.*
