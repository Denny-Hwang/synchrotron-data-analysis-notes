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
| **Modality**       | Synchrotron Micro-Computed Tomography (micro-CT)                                       |

---

## TL;DR

An end-to-end AI+HPC workflow integrates tomographic reconstruction (TomoPy),
DL-based denoising (TomoGAN variant), and automated segmentation (U-Net) into a
single real-time pipeline running on Argonne Leadership Computing Facility (ALCF)
infrastructure. Micro-CT data streams from the 2-BM detector via Globus over a
dedicated 100 Gbps network link, with the complete pipeline -- from raw projections
to segmented 3D volume -- completing in under 60 seconds for a 2048^3 volume. This
represents a 50-100x reduction in time-to-results compared to traditional offline
processing, enabling intra-experiment decision-making.

---

## Background & Motivation

Synchrotron micro-computed tomography (micro-CT) at high-flux beamlines like APS
2-BM generates data at rates exceeding 1 GB/s. A single tomographic scan produces
2,000-5,000 projection images at multiple rotation angles, yielding a 3D volume of
2048^3 or larger voxels after reconstruction. For time-resolved experiments (in-situ
battery cycling, foam collapse, thermal processing), multiple scans are acquired
sequentially, generating terabytes per experiment session.

**The traditional workflow** is entirely offline:

1. Projections are written to local disk at the beamline.
2. Data is transferred to a computing cluster (hours for TB-scale datasets).
3. Reconstruction is run on the cluster (minutes to hours per scan).
4. Denoising and segmentation are performed manually (hours to days).
5. Results are inspected and scientific analysis begins.

**Total time from acquisition to usable results**: typically 24-72 hours.

**The consequence**: Experimenters cannot verify data quality, check for sample
damage, or adjust experimental parameters until long after the measurement. If a
scan failed (wrong sample position, inadequate exposure, sample moved during scan),
precious beamtime is wasted because the problem is only discovered offline.

**The opportunity**: By coupling beamline data acquisition directly to HPC
resources and deploying DL-based analysis, the entire pipeline can run within the
dead time between consecutive scans (typically 1-5 minutes), providing real-time
feedback to the experimenter.

---

## Method

### Data

| Item | Details |
|------|---------|
| **Data source** | Beamline 2-BM, APS; real-time streaming micro-CT data |
| **Sample type** | Foam phantoms, rock core samples |
| **Data dimensions** | 2048x2048 pixel projections, 1500-2000 projections per scan, yielding 2048^3 reconstructed volumes |
| **Preprocessing** | Flat/dark field correction, ring artifact removal (TomoPy built-in) |

### Pipeline Architecture

The workflow uses a **producer-consumer** architecture with stage-level parallelism:

**Stage 1 -- Data Streaming**:
- Projection images stream from the area detector to ALCF via the Globus data
  fabric over a dedicated 100 Gbps network link between APS and ALCF.
- A ZeroMQ message queue coordinates data availability notifications between
  stages, decoupling data production from consumption.
- Sustained data transfer rate: ~40 Gbps (of 100 Gbps available).

**Stage 2 -- Tomographic Reconstruction**:
- TomoPy performs filtered back projection (FBP) or gridrec on incoming sinograms.
- Distributed across 64 CPU nodes at ALCF for parallelism.
- Throughput: ~100 slices/second for 2048x2048 pixel slices.
- Each sinogram (single row across all projections) is reconstructed independently,
  enabling streaming -- reconstruction begins before all projections are available.

**Stage 3 -- DL Denoising**:
- A TomoGAN-derived U-Net model runs on 4x NVIDIA V100 GPUs at ALCF.
- Processes reconstructed slices as they become available from Stage 2 via a shared
  message queue.
- Throughput: ~200 slices/second.
- The producer-consumer design allows denoising to overlap temporally with
  reconstruction, maximizing pipeline throughput.

**Stage 4 -- Semantic Segmentation**:
- A pre-trained U-Net segments denoised slices into material phases (pore space,
  solid matrix, inclusions).
- Runs on 4x V100 GPUs, consuming denoised slices from Stage 3.
- Throughput: ~150 slices/second.
- The model was trained offline on manually annotated slices from similar specimen
  classes.

**Orchestration**: An Apache Airflow-based workflow manager coordinates all four
stages, handles error recovery, monitors resource utilization, and logs pipeline
metadata. Stage-level parallelism ensures that the overall pipeline throughput is
limited by the slowest stage (reconstruction at ~100 slices/s), not by the sum of
all stage latencies.

**Visualization**: Reconstructed, denoised, and segmented slices are streamed to a
web-based viewer accessible at the beamline control station, enabling real-time
quality verification. Experimenters can inspect the current scan's results while
setting up the next scan.

### Pipeline Diagram

```
Area detector (2-BM, 1+ GB/s)
  --> Globus data fabric (100 Gbps link)
  --> ALCF: ZeroMQ queue
  --> TomoPy reconstruction (64 CPU nodes, ~100 slices/s)
  --> ZeroMQ queue
  --> TomoGAN-variant denoising (4x V100, ~200 slices/s)
  --> ZeroMQ queue
  --> U-Net segmentation (4x V100, ~150 slices/s)
  --> Web-based visualization at beamline
  --> Apache Airflow orchestration throughout
```

---

## Key Results

| Metric                                    | Value / Finding                                   |
|-------------------------------------------|---------------------------------------------------|
| End-to-end latency (acquisition to segmented volume) | < 60 s for a 2048^3 volume              |
| Reconstruction throughput                 | ~100 slices/s (gridrec on 64 CPU nodes)            |
| Denoising throughput                      | ~200 slices/s (U-Net on 4x V100)                   |
| Segmentation throughput                   | ~150 slices/s (U-Net on 4x V100)                   |
| Pipeline vs. offline speedup              | 50-100x reduction in time to usable results        |
| Denoising quality (PSNR improvement)      | +3.5 dB over raw low-dose reconstruction           |
| Segmentation accuracy (Dice score)        | 0.92 on foam phantom, 0.87 on rock core samples   |
| Network bandwidth utilization             | ~40 Gbps sustained (of 100 Gbps available)         |

### Key Figures

- **Figure 1**: System architecture diagram showing the four-stage pipeline from
  detector to visualization, with throughput and latency annotations at each stage.
- **Figure 3**: Timeline visualization showing overlapping execution of
  reconstruction, denoising, and segmentation stages for a single scan, with the
  total wall time dominated by reconstruction (slowest stage).
- **Figure 4**: Quality comparison of denoised vs. raw reconstructed slices for
  foam and rock specimens, with segmentation overlays showing material phase
  identification.

---

## Data & Code Availability

| Resource       | Link / Note                                                           |
|----------------|-----------------------------------------------------------------------|
| **Code**       | Workflow scripts partially available on GitHub; core TomoPy and TomoGAN repos referenced |
| **Data**       | Foam phantom dataset available via Globus; rock core data on request   |
| **License**    | Various (TomoPy: BSD, TomoGAN: MIT, workflow scripts: not specified)   |

**Reproducibility Score**: **2 / 5** -- Individual pipeline components (TomoPy,
TomoGAN) are open-source and well-documented. However, the orchestration layer
(Airflow DAGs, ZeroMQ configuration, ALCF-specific deployment scripts) is not
fully released. Reproducing the end-to-end pipeline requires ALCF-scale HPC
resources (64+ CPU nodes, multiple GPU nodes, dedicated 100 Gbps network), which
limits reproducibility to facilities with comparable infrastructure.

---

## Strengths

- **Complete end-to-end demonstration**: Not a single-stage proof of concept but a
  full, working pipeline from detector to segmented volume, proving that real-time
  synchrotron tomography analysis is technically feasible at production scale.
- **Sub-minute latency**: < 60 s for a 2048^3 volume enables intra-experiment
  decision-making. Experimenters can verify results between consecutive scans.
- **Producer-consumer architecture**: Stage-level parallelism with message queues
  allows stages to overlap in time, maximizing GPU/CPU utilization and ensuring
  pipeline throughput matches the detector frame rate.
- **Uses community-supported tools**: TomoPy, Globus, and Apache Airflow are
  established, maintained tools rather than custom research prototypes, improving
  long-term maintainability.
- **Infrastructure quantification**: Provides specific numbers on network bandwidth
  requirements, compute allocation, and throughput per stage, enabling other
  facilities to plan similar deployments.
- **Real beamline validation**: Demonstrated on actual APS 2-BM data, not just
  synthetic benchmarks.

---

## Limitations & Gaps

- **Requires dedicated HPC allocation**: The pipeline needs 64+ CPU nodes and
  multiple GPU nodes on ALCF, plus a dedicated 100 Gbps network link. This level
  of infrastructure is not available at all facilities or at all times, even at
  Argonne. Scheduling contention with other HPC users is not addressed.
- **Pre-trained segmentation model**: The U-Net segmentation is trained on specific
  material classes (foam, rock). New specimen types require retraining with manually
  annotated data -- a significant effort that limits the pipeline's generalizability.
- **Error handling**: Described at a high level but not rigorously tested under
  failure conditions (network drops, GPU faults, corrupted projections, out-of-
  memory conditions). Production deployment requires robust error recovery.
- **Simplified denoising module**: Uses a basic U-Net rather than the full TomoGAN
  with adversarial training. Comparison with state-of-the-art alternatives
  (self-supervised methods, diffusion models) is absent.
- **No data provenance tracking**: Processed outputs lack systematic metadata
  recording which model versions, parameters, and preprocessing steps were applied.
  This is critical for scientific reproducibility.
- **Single beamline demonstration**: Only 2-BM micro-CT; extension to other
  tomography beamlines with different data formats and rates is not demonstrated.

---

## Relevance to eBERlight

This paper provides a systems-level template for eBERlight's real-time analysis
infrastructure:

- **Pipeline architecture**: eBERlight should adopt the producer-consumer,
  stage-overlapped design for its multi-stage analysis pipelines across modalities,
  not just tomography. The ZeroMQ/message queue pattern scales to arbitrary numbers
  of stages.
- **ALCF integration**: The Globus-based data fabric and ALCF resource allocation
  patterns demonstrated here directly apply to eBERlight's planned use of ALCF for
  GPU inference, model training, and large-scale data processing.
- **Latency benchmark**: The < 60 s end-to-end latency establishes a target for
  eBERlight's tomography pipeline. APS-U's higher data rates (5-10 GB/s) will
  require further pipeline optimization or edge preprocessing.
- **Infrastructure planning**: The specific numbers on network bandwidth, compute
  allocation, and per-stage throughput provide concrete infrastructure planning
  data for eBERlight's deployment.
- **Upgrade pathway**: Replace individual stages with improved models (e.g., swap
  the U-Net denoiser for a Noise2Void variant, replace gridrec with a learned
  reconstructor) while maintaining the same pipeline architecture.
- **Priority**: **High** -- provides the systems blueprint for eBERlight's
  tomography and multi-modal analysis infrastructure.

---

## Actionable Takeaways

1. **Adopt producer-consumer staging**: Implement eBERlight's analysis pipelines
   using ZeroMQ or Kafka message queues for stage-level parallelism and decoupling.
2. **Upgrade to Bluesky/Tiled**: Replace the custom data streaming layer with
   Bluesky event model and Tiled for metadata-rich, facility-standard data flow
   with built-in provenance tracking.
3. **Self-supervised denoising**: Replace the supervised U-Net with Noise2Void or
   Noise2Noise to eliminate paired training data requirements.
4. **Add automated QC**: Integrate automated quality control metrics (PSNR, SSIM,
   edge sharpness, ring artifact severity) that trigger alerts when pipeline output
   quality degrades below configurable thresholds.
5. **Extend to APS-U data rates**: Profile the pipeline at APS-U expected data
   rates (5-10 GB/s) and identify bottlenecks requiring edge preprocessing or
   additional compute resources.
6. **Data provenance**: Add systematic metadata tracking recording model versions,
   hyperparameters, and processing timestamps for all pipeline outputs.

---

## Notes & Discussion

This paper is the systems-level complement to the algorithmic review in
`review_fullstack_dl_tomo_2023.md`. While the full-stack review provides the
conceptual framework and algorithm-level recommendations for DL tomography, this
paper demonstrates a working implementation with concrete infrastructure
specifications. Together, they define both the "what" (which DL modules to deploy)
and the "how" (which infrastructure and architecture) for eBERlight's tomography
pipeline.

The Argonne pedigree (APS 2-BM + ALCF) means the infrastructure patterns are
directly applicable to eBERlight without cross-facility translation. The main gap
to address is upgrading from the research-prototype orchestration (custom Airflow
DAGs) to production-quality infrastructure with Bluesky/Tiled integration,
comprehensive error handling, and automated quality monitoring.

---

## Review Metadata

| Field | Value |
|-------|-------|
| **Reviewed by** | eBERlight AI/ML Team |
| **Review date** | 2025-10-18 |
| **Last updated** | 2025-10-18 |
| **Tags** | tomography, micro-CT, HPC, real-time, pipeline, streaming, infrastructure, TomoPy |
