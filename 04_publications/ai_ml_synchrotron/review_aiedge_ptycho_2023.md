# Paper Review: Real-Time Streaming Ptychographic Phase Retrieval at Kilohertz Frame Rates

## Metadata

| Field              | Value                                                                                  |
|--------------------|----------------------------------------------------------------------------------------|
| **Title**          | Real-Time Streaming Ptychographic Phase Retrieval at Kilohertz Frame Rates             |
| **Authors**        | Babu, A. V.; Zhou, T.; Kandel, S.; Bicer, T.; Liu, Z.; Judge, W.; Ching, D. J.; Jiang, Y.; Veseli, S.; Hammer, S.; Schwarz, N.; Cherukara, M. J. |
| **Journal**        | Nature Communications, 14, Article 5765                                                |
| **Year**           | 2023                                                                                   |
| **DOI**            | [10.1038/s41467-023-41496-z](https://doi.org/10.1038/s41467-023-41496-z)               |
| **Beamline**       | APS 26-ID (CNM nanoprobe), APS 2-ID-D                                                 |
| **Modality**       | Ptychography (coherent diffraction imaging)                                            |

---

## TL;DR

This paper demonstrates real-time streaming ptychographic phase retrieval at up to
2--3 kHz frame rates by deploying deep-learning-based reconstruction on edge GPU
(NVIDIA Jetson) and FPGA (Xilinx Alveo) hardware co-located with the detector at the
beamline. The system achieves end-to-end latency below 5 ms from detector readout to
reconstructed phase image display, eliminating the data-transfer bottleneck to remote
HPC and enabling real-time adaptive scanning at APS ptychography beamlines. ZMQ-based
streaming integration connects the reconstruction pipeline to the beamline control
system for live feedback.

---

## Background & Motivation

Ptychography recovers both amplitude and phase of a specimen from overlapping coherent
diffraction patterns at resolution beyond the focusing optic limit. Conventional
iterative algorithms (ePIE, difference map, RAAR) require 100--500 iterations per
scan position, creating a severe bottleneck at modern detector frame rates (1--10 kHz).
The APS-U will deliver 100--500x more coherent flux, pushing data rates to 10+ GB/s
and making this gap untenable for experiments requiring real-time feedback.

Prior work by Guan et al. (PtychoNet, 2019) demonstrated CNN-based ptychographic
reconstruction with ~90% speedup, but deployment on conventional GPU servers retained
significant data-transfer latency. This paper takes the critical next step: deploying
optimized neural networks on edge compute hardware co-located with the detector,
connected via RDMA and ZMQ streaming, to achieve true real-time operation.

---

## Method

### Data

| Item | Details |
|------|---------|
| **Data source** | Simulated diffraction patterns (training); experimental data from APS 26-ID and 2-ID-D (validation) |
| **Sample type** | Siemens star test pattern, integrated circuit, biological cell sections |
| **Data dimensions** | 256x256 pixel diffraction patterns; 128x128 pixel reconstructed patches |
| **Preprocessing** | Log-scaling of intensities, background subtraction, per-pattern normalization |

### Model / Algorithm

1. **Neural network architecture**: A compact encoder-decoder CNN with approximately
   2 million parameters, significantly smaller than the original PtychoNet (~8M
   parameters). The encoder uses four convolutional blocks with residual connections
   and strided convolutions (rather than max-pooling) for downsampling. The decoder
   mirrors the encoder with transposed convolutions. The compact design is driven by
   edge hardware constraints: both memory footprint and inference latency must fit
   within the resource envelope of embedded devices.

2. **Training data**: Physics-based forward model generates simulated diffraction
   patterns with realistic Poisson noise, partial coherence effects, and probe
   diversity. Approximately 200,000 pattern-object pairs are generated. Transfer
   learning is then applied using a small corpus (~5,000 patterns) of experimental
   data with iteratively reconstructed ground truth to bridge the sim-to-real
   domain gap.

3. **Edge GPU deployment (NVIDIA Jetson AGX Orin)**: The model is quantized to FP16
   and compiled with TensorRT, achieving ~0.5 ms per diffraction pattern via layer
   fusion and kernel auto-tuning for the ARM-based GPU architecture.

4. **FPGA deployment (Xilinx Alveo U250)**: Further quantized to INT8 and synthesized
   via Xilinx Vitis AI with a pipelined dataflow architecture, achieving ~0.3 ms
   per pattern at significantly lower power consumption than the GPU path.

5. **Streaming pipeline**: Data flows from the Eiger detector via RDMA to the edge
   device, bypassing the control workstation. ZMQ PUB/SUB streams reconstructed
   phase images to a visualization dashboard and Bluesky control system for
   adaptive scan decisions.

6. **Iterative refinement option**: The CNN output can optionally seed 5--10 ePIE
   iterations (instead of 200+) for higher fidelity when latency permits.

### Pipeline

```
Eiger detector --> RDMA --> Edge device (Jetson/Alveo) --> CNN inference
    --> ZMQ PUB --> [Visualization dashboard, Bluesky scan controller]
    --> (Optional) 5-10 ePIE refinement iterations on GPU server
```

The architecture is designed for APS-U data rates, with the edge device handling
real-time reconstruction and a downstream GPU server available for optional
refinement of selected frames.

---

## Key Results

| Metric                                    | Value / Finding                                   |
|-------------------------------------------|---------------------------------------------------|
| Frame rate (edge GPU, Jetson AGX Orin)    | ~2 kHz (0.5 ms/pattern)                           |
| Frame rate (FPGA, Alveo U250)             | ~3.3 kHz (0.3 ms/pattern)                         |
| Phase retrieval quality (NRMSE, CNN only) | 0.08 on experimental test data                    |
| NRMSE (CNN + 10 ePIE iterations)          | 0.03 (matching full 200-iteration ePIE quality)   |
| End-to-end latency (detector to display)  | < 5 ms including data transfer and rendering       |
| Power consumption (FPGA)                  | ~45 W (vs. ~300 W for server-class GPU)            |
| Power consumption (Jetson Orin)           | ~60 W (vs. ~300 W for server-class GPU)            |
| Demonstrated resolution                   | ~15 nm half-pitch on Siemens star test pattern     |
| Resolution (full ePIE, same data)         | ~10 nm half-pitch                                  |
| Speedup vs. full iterative (200 ePIE)    | 20x at matching quality (CNN + 10 ePIE)            |

### Key Figures

- **Figure 1**: System architecture diagram showing the detector-to-edge-to-display
  data path with RDMA and ZMQ connections, illustrating the elimination of the
  network bottleneck by co-locating compute with the detector.
- **Figure 3**: Reconstruction quality comparison across three operating modes:
  CNN-only (fastest, NRMSE 0.08), CNN + 10 ePIE (balanced, NRMSE 0.03), and
  full 200 ePIE (reference, NRMSE 0.02). Visual comparison on Siemens star and
  integrated circuit specimens.
- **Figure 4**: Throughput-vs-quality Pareto curve showing the tunable tradeoff
  between frame rate and reconstruction fidelity across different hardware
  platforms and hybrid configurations.
- **Figure 5**: Power consumption comparison: FPGA (45 W) vs. Jetson (60 W) vs.
  server GPU (300 W), highlighting the sustainability advantage of edge deployment
  for continuous beamline operation.

---

## Data & Code Availability

| Resource       | Link / Note                                                           |
|----------------|-----------------------------------------------------------------------|
| **Code**       | Partial release: inference scripts and pretrained model weights on GitHub |
| **Data**       | Simulated training sets available; experimental data on request        |
| **License**    | Not explicitly stated in paper                                        |
| **Reproducibility Score** | **3 / 5** -- Inference code and pretrained weights publicly available; FPGA synthesis scripts and full training pipeline not released; experimental data available on request but not openly deposited. |

---

## Strengths

- Demonstrates kHz-rate ptychographic reconstruction on edge hardware -- a critical
  milestone and prerequisite for adaptive scanning at APS-U data rates.
- Edge deployment eliminates the network bottleneck: data never leaves the beamline
  hutch, reducing latency by orders of magnitude vs. HPC-based workflows.
- FPGA achieves 5--7x lower power than server GPUs, important for sustainable 24/7
  multi-beamline operations.
- Hybrid CNN + few-iteration ePIE provides continuously tunable quality/speed
  tradeoff suited to different experimental priorities.
- ZMQ streaming provides clean integration with existing Bluesky/Ophyd control
  infrastructure without major redesign.
- Validated on experimental data at two APS beamlines, not just simulations.

---

## Limitations & Gaps

- The compact CNN trades reconstruction quality for speed: achieved resolution is
  ~15 nm half-pitch compared to ~10 nm achievable with full iterative reconstruction
  on the same data, a meaningful gap for experiments requiring the highest resolution.
- Transfer learning from simulation to experiment requires beamline-specific
  calibration data; generalization across beamlines with different probe functions,
  photon energies, or detector types is not demonstrated.
- FPGA INT8 quantization introduces quantization noise whose systematic impact on
  phase image accuracy and spatial resolution is only partially characterized.
- The streaming pipeline is demonstrated for 2D ptychography; extension to 3D
  ptychographic tomography (ptycho-tomo), which requires accumulating and jointly
  processing projections from multiple angles, is discussed but not implemented.
- Full training pipeline and FPGA synthesis toolchain are not open-sourced, limiting
  community reproducibility for the FPGA deployment pathway.
- No uncertainty quantification on the reconstructed phase: users cannot
  distinguish confident reconstruction regions from uncertain ones.

---

## Relevance to APS BER Program

This paper is a cornerstone reference for the BER program's real-time feedback architecture
and edge compute strategy:

- **Latency benchmark**: The < 5 ms end-to-end latency establishes the performance
  target for the BER program's real-time decision-making loop at coherent imaging beamlines,
  demonstrating that millisecond-scale feedback is technically achievable.
- **Edge compute model**: The BER program can adopt the same Jetson Orin / FPGA edge
  deployment architecture for other modalities requiring millisecond inference:
  XRF spectral fitting, XANES classification, diffraction pattern indexing.
- **Adaptive ptychography**: With real-time phase images available, the BER program's
  scan planner can dynamically adjust overlap ratios, skip redundant positions, or
  zoom into features of interest mid-scan, maximizing information gained per unit
  beam time.
- **APS-U readiness**: The demonstrated 2--3 kHz throughput is well-matched to
  expected APS-U ptychography frame rates at 26-ID and 2-ID-D, confirming that
  the approach scales to upgraded source conditions.
- **ZMQ integration**: The ZMQ-based streaming architecture aligns with the BER program's
  planned Bluesky/Tiled data infrastructure, enabling straightforward integration
  of edge-reconstructed images into the experimental control loop.
- **Priority**: Critical -- this is the most directly relevant work for the BER program's
  edge-AI ptychography deployment.

---

## Actionable Takeaways

1. **Procure edge hardware**: Acquire NVIDIA Jetson Orin modules and Xilinx Alveo
   boards for BER program beamline hutches at APS-U ptychography endstations (26-ID,
   2-ID-D), with rack integration and cooling solutions.
2. **Retrain on APS-U probe**: Generate APS-U-specific training data using measured
   probe functions, coherence properties, and detector characteristics from early
   commissioning, then fine-tune the pretrained model via transfer learning.
3. **ZMQ/Bluesky integration**: Integrate the detector-to-edge RDMA pipeline with
   the BER program's Bluesky/Tiled data infrastructure for seamless live reconstruction,
   visualization, and adaptive scan control.
4. **Extend to ptycho-tomo**: Develop a 3D extension that reconstructs ptychographic
   tomography volumes in streaming mode by accumulating CNN-reconstructed projections
   and performing online tomographic synthesis.
5. **Uncertainty monitoring**: Implement online reconstruction quality tracking
   (e.g., running NRMSE against occasional full-iterative reference frames) that
   triggers fallback to full reconstruction when CNN output quality degrades below
   a user-defined threshold.
6. **Cross-modality edge deployment**: Use this work as the template for deploying
   edge inference for XRF, XPCS, and tomographic reconstruction at other BER program
   beamlines.

---

## BibTeX Citation

```bibtex
@article{babu2023realtime_ptycho,
  title     = {Real-Time Streaming Ptychographic Phase Retrieval at
               Kilohertz Frame Rates},
  author    = {Babu, Anakha V. and Zhou, Tao and Kandel, Saugat and
               Bicer, Tekin and Liu, Zhengchun and Judge, William and
               Ching, Daniel J. and Jiang, Yi and Veseli, Sinisa and
               Hammer, Sven and Schwarz, Nicholas and Cherukara, Mathew J.},
  journal   = {Nature Communications},
  volume    = {14},
  pages     = {5765},
  year      = {2023},
  publisher = {Nature Publishing Group},
  doi       = {10.1038/s41467-023-41496-z}
}
```

---

*Reviewed for the Synchrotron Data Analysis Notes, 2026-02-27.*
