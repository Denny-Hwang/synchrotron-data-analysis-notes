# Paper Review: Real-Time Streaming Ptychographic Imaging on Edge Compute

## Metadata

| Field              | Value                                                                                  |
|--------------------|----------------------------------------------------------------------------------------|
| **Title**          | Understanding Real-Time Streaming Ptychographic Phase Retrieval on Edge GPU and FPGA   |
| **Authors**        | Babu, A. V.; Zhou, T.; Kandel, S.; Bicer, T.; Liu, Z.; Judge, W.; Ching, D. J.; Jiang, Y.; Veseli, S.; Hammer, S.; Schwarz, N.; Cherukara, M. J. |
| **Journal**        | Nature Communications, 14                                                              |
| **Year**           | 2023                                                                                   |
| **DOI**            | [10.1038/s41467-023-41496-z](https://doi.org/10.1038/s41467-023-41496-z)               |
| **Beamline**       | APS 26-ID (CNM nanoprobe), APS 2-ID-D                                                 |
| **Modality**       | Ptychography (coherent diffraction imaging)                                            |

---

## TL;DR

This work demonstrates real-time streaming ptychographic phase retrieval at up to
2 kHz frame rate by deploying deep-learning-based reconstruction on edge GPU
(NVIDIA Jetson) and FPGA (Xilinx Alveo) hardware co-located with the detector.
Data flows directly from the detector to the edge device via RDMA, bypassing
network bottlenecks to remote HPC clusters, and produces reconstructed phase
images with less than 5 ms total end-to-end latency. A hybrid mode combining the
CNN with a small number of iterative refinement steps (5-10 ePIE iterations)
matches the quality of 200+ full iterative reconstructions at 20x lower latency,
establishing the paradigm for adaptive ptychographic experiments at APS-U.

---

## Background & Motivation

Ptychography is a coherent diffraction imaging technique that recovers both
amplitude and phase of a specimen at resolution beyond the focusing optic's limit.
By scanning a coherent beam across overlapping positions and recording far-field
diffraction patterns at each position, ptychography achieves imaging at resolutions
down to ~5-15 nm at hard X-ray energies. The technique is central to the scientific
programs of several APS beamlines (26-ID, 2-ID-D) and will be a primary imaging
modality at APS-U coherent endstations.

**The reconstruction bottleneck**:

Conventional iterative algorithms (ePIE, difference map, RAAR) require hundreds of
iterations per scan position, with computational cost scaling linearly with the
number of positions and quadratically with detector pixel count. At modern detector
frame rates (kHz), iterative reconstruction cannot keep pace with data acquisition.
For a typical ptychographic scan with 10,000 positions acquired at 1 kHz, full
iterative reconstruction takes hours -- long after the experiment has ended.

**The consequence**: No real-time feedback is available during the experiment. The
experimenter cannot see the reconstructed image while scanning and therefore cannot:
- Verify that the scan is producing useful data (correct sample region, adequate
  overlap, proper focus)
- Adjust scan parameters mid-experiment (zoom into an interesting feature, reduce
  overlap in featureless regions, increase exposure in low-contrast areas)
- Make autonomous decisions about where to scan next

**APS-U data rates**: The APS Upgrade will deliver 100-500x more coherent flux,
pushing ptychography detector frame rates to 5-10 kHz. Data rates of 10+ GB/s will
make even data transfer to remote clusters a bottleneck, let alone reconstruction.

**Edge compute as the solution**: By placing the neural network inference hardware
(GPU or FPGA) physically at the beamline, adjacent to the detector, data can be
reconstructed without network transfer. This reduces the reconstruction pipeline
from hours (iterative on HPC) or seconds (iterative on local GPU) to milliseconds.

---

## Method

### Data

| Item | Details |
|------|---------|
| **Data source** | APS 26-ID and 2-ID-D beamlines; simulated and experimental ptychographic data |
| **Sample type** | Siemens star test patterns, integrated circuits, biological specimens |
| **Data dimensions** | 128x128 to 256x256 pixel diffraction patterns; 64x64 to 128x128 pixel reconstructed patches |
| **Preprocessing** | Log-scaling of diffraction intensities, normalization, background subtraction |

### Model / Algorithm

**Neural network architecture**: A compact encoder-decoder CNN (~2M parameters)
takes a single diffraction pattern as input and outputs the complex-valued object
transmission function (amplitude and phase). The architecture uses residual blocks
for feature extraction and is trained to approximate the iterative reconstruction
operator in a single forward pass. Separate output heads for amplitude and phase
avoid phase-wrapping ambiguities.

**Training data generation**: The physics-based forward model simulates the
ptychographic measurement process. Randomly generated complex-valued objects are
illuminated by a measured (or modeled) probe function, propagated to the far field,
and corrupted with Poisson noise matching experimental photon statistics. Transfer
learning from simulation to experiment uses a small amount of experimental data
(~500 patterns) for domain adaptation.

**Edge GPU deployment (NVIDIA Jetson AGX Orin)**:
- Model quantized to FP16 precision using TensorRT
- Compiled and optimized for the Jetson's Ampere GPU architecture
- Inference time: ~0.5 ms per diffraction pattern (2 kHz throughput)
- Power consumption: ~60 W (vs. ~300 W for a server-class GPU)

**FPGA deployment (Xilinx Alveo U250)**:
- Model translated to INT8 fixed-point representation using Vitis AI
- Hardware synthesis onto the FPGA fabric
- Inference time: ~0.3 ms per pattern (3.3 kHz throughput)
- Power consumption: ~45 W
- Deterministic latency (no OS scheduling jitter)

**Streaming pipeline**: Data flows from the area detector (Eiger) via RDMA (Remote
Direct Memory Access) directly to the edge device's memory, bypassing the operating
system network stack and the beamline control workstation entirely. Reconstructed
phase images are streamed to a web-based visualization dashboard for operator
monitoring with < 5 ms total end-to-end latency.

**Hybrid CNN + iterative mode**: The CNN output serves as initialization for a
small number of ePIE iterations (5-10 instead of 200+). This hybrid approach
provides the CNN's speed advantage while converging to iterative-quality results,
reducing total reconstruction time by 10-20x compared to cold-start iterative
methods.

### Pipeline

```
Area detector (Eiger, 1-5 kHz)
  --> RDMA data transfer to edge device (< 0.1 ms)
  --> CNN forward pass (0.3-0.5 ms)
  --> Reconstructed amplitude + phase patches
  --> Overlap-weighted stitching into full image
  --> (Optional) 5-10 ePIE refinement iterations
  --> Web-based visualization dashboard (< 5 ms total)
  --> Feedback signal to Bluesky scan controller
```

---

## Key Results

| Metric                                    | Value / Finding                                   |
|-------------------------------------------|---------------------------------------------------|
| Frame rate (edge GPU, Jetson Orin)        | ~2 kHz (0.5 ms/pattern)                           |
| Frame rate (FPGA, Alveo U250)             | ~3.3 kHz (0.3 ms/pattern)                         |
| Phase retrieval quality (NRMSE)           | 0.08 (CNN alone), 0.03 (CNN + 10 ePIE iterations) |
| Comparison with 200 ePIE iterations       | CNN+10 ePIE matches quality at 20x lower latency  |
| Power consumption (FPGA)                  | ~45 W (vs. ~300 W for server GPU)                  |
| End-to-end latency (detector to display)  | < 5 ms                                             |
| Demonstrated resolution                   | ~15 nm half-pitch on Siemens star test pattern      |
| Data transfer savings                     | >99% reduction vs. streaming to remote HPC          |

### Key Figures

- **Figure 2**: System architecture diagram showing the detector -> RDMA -> edge
  device -> visualization pathway, with latency annotations at each stage.
- **Figure 3**: Side-by-side comparison of CNN-only, CNN + 10 ePIE, and full 200
  ePIE reconstructions on Siemens star and IC test specimens, showing that
  CNN + 10 ePIE is visually indistinguishable from full iterative at 20x lower
  computational cost.
- **Figure 5**: Throughput and power comparison between edge GPU, FPGA, and
  server-class GPU implementations, demonstrating the FPGA's advantage in
  power efficiency.

---

## Data & Code Availability

| Resource       | Link / Note                                                           |
|----------------|-----------------------------------------------------------------------|
| **Code**       | Partial release (inference scripts and model weights on GitHub)        |
| **Data**       | Simulated training sets available; experimental data on request        |
| **License**    | Not explicitly stated in paper                                        |

**Reproducibility Score**: **3 / 5** -- Inference code and pretrained weights are
available for the GPU implementation. FPGA synthesis scripts and the full training
pipeline are not released, limiting reproducibility of the FPGA deployment.
Experimental data requires beamline-specific agreements. The simulation-based
training pipeline is described in sufficient detail for reimplementation.

---

## Strengths

- **Real-time ptychographic reconstruction at kHz rates**: This is a prerequisite
  for adaptive scanning at APS-U data rates and has not been demonstrated before
  this work.
- **Edge deployment eliminates network bottleneck**: Data never leaves the
  beamline hutch for initial reconstruction, fundamentally solving the data
  transfer problem for high-data-rate experiments.
- **FPGA implementation**: Achieves competitive throughput at 6-7x lower power
  than GPU servers, with deterministic latency -- important for time-critical
  feedback loops and sustainable facility operations.
- **Hybrid CNN + iterative approach**: Provides a tunable quality/speed tradeoff
  that acknowledges CNN limitations without abandoning the speed advantage. This
  pragmatic approach is ready for production deployment.
- **Validated on experimental ptychographic data**: Demonstrated on real beamline
  data from APS 26-ID and 2-ID-D, not just simulations.
- **Published in Nature Communications**: High-impact venue reflecting the
  significance of the contribution to the synchrotron community.

---

## Limitations & Gaps

- **Resolution trade-off**: The compact CNN achieves ~15 nm resolution vs. ~10 nm
  from full iterative reconstruction. The 5 nm gap may be acceptable for survey
  mode but not for final publication-quality images.
- **Beamline-specific transfer learning**: The simulation-to-experiment domain
  adaptation requires beamline-specific calibration data (probe function, coherence
  properties, detector characteristics). Generalization across beamlines without
  retraining is not demonstrated.
- **FPGA quantization artifacts**: INT8 quantization introduces systematic errors
  in phase images that are not fully characterized. Quantization-aware training
  could mitigate this but is not explored.
- **2D ptychography only**: The streaming pipeline handles standard 2D
  ptychography. Extension to 3D ptychographic tomography (ptycho-tomo), which
  requires integrating information across rotation angles, is discussed but not
  implemented.
- **Incomplete open-source release**: The FPGA synthesis toolchain and full
  training pipeline are not public, limiting community adoption of the FPGA
  deployment.
- **No uncertainty quantification**: The CNN provides point estimates without
  confidence intervals, and there is no mechanism to detect when the network
  encounters out-of-distribution data that may produce unreliable reconstructions.

---

## Relevance to eBERlight

This paper is a cornerstone reference for eBERlight's real-time feedback
architecture:

- **Applicable beamlines**: APS 26-ID (CNM nanoprobe), 2-ID-D, and future APS-U
  coherent imaging endstations. The edge compute paradigm extends to any
  eBERlight beamline requiring millisecond-scale inference.
- **Latency benchmark**: The < 5 ms end-to-end latency establishes the target for
  eBERlight's real-time decision-making loop at coherent imaging beamlines.
- **Edge compute strategy**: eBERlight should adopt the Jetson Orin / FPGA edge
  deployment model for other modalities where millisecond inference is needed (XRF
  spectral fitting, XANES classification, fast tomographic preview).
- **Adaptive ptychography**: With real-time phase images available, eBERlight's
  scan planner can adjust overlap ratios, skip redundant positions, zoom into
  features of interest, or modify dwell times mid-scan.
- **APS-U readiness**: The 2-3 kHz throughput is well-matched to expected APS-U
  ptychography frame rates.
- **Priority**: **Critical** -- this is foundational infrastructure for eBERlight's
  coherent imaging pipeline.

---

## Actionable Takeaways

1. **Procure edge hardware**: Acquire NVIDIA Jetson Orin modules and Xilinx Alveo
   boards for eBERlight beamline hutches at APS-U ptychography endstations.
2. **Retrain on APS-U probe**: Generate APS-U-specific training data using measured
   probe functions and coherence properties from commissioning.
3. **Streaming integration**: Integrate the detector-to-edge RDMA pipeline with
   eBERlight's Bluesky/Tiled data infrastructure for seamless live reconstruction
   and metadata capture.
4. **Extend to ptycho-tomo**: Develop a 3D extension that reconstructs
   ptychographic tomography volumes in streaming mode by accumulating
   CNN-reconstructed projections.
5. **Quality monitoring**: Implement online NRMSE or Fourier ring correlation
   tracking that triggers fallback to full iterative reconstruction when CNN
   output quality degrades below a user-defined threshold.
6. **Cross-modality edge deployment**: Apply the same edge compute architecture
   to XRF spectral fitting and XANES classification at other eBERlight beamlines.

---

## Notes & Discussion

This paper represents the state of the art in real-time coherent imaging at
synchrotrons. The key insight -- that edge compute eliminates the data transfer
bottleneck that has historically prevented real-time feedback -- is applicable
far beyond ptychography. eBERlight should view this as a template for deploying
edge-based ML inference across all high-data-rate modalities.

The relationship to PtychoNet (`review_ptychonet_2019.md`) is one of direct
lineage: PtychoNet established the CNN-for-ptychography concept, and this work
brings it to production-quality edge deployment. The FPGA implementation is
particularly notable for facility-scale deployment where power efficiency and
deterministic latency matter.

---

## Review Metadata

| Field | Value |
|-------|-------|
| **Reviewed by** | eBERlight AI/ML Team |
| **Review date** | 2025-10-17 |
| **Last updated** | 2025-10-17 |
| **Tags** | ptychography, edge-compute, CNN, FPGA, GPU, real-time, streaming, autonomous |
