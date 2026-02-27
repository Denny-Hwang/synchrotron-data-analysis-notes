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

---

## TL;DR

This work demonstrates real-time streaming ptychographic phase retrieval at up to 2 kHz
frame rate by deploying deep-learning-based reconstruction on edge GPU and FPGA hardware
co-located with the detector, eliminating the latency bottleneck of data transfer to
remote HPC clusters.

---

## Background & Motivation

Ptychography is a coherent diffraction imaging technique that recovers both amplitude
and phase of a specimen at resolution beyond the lens limit. Conventional iterative
algorithms (ePIE, DM, RAAR) require hundreds of iterations per scan position, making
real-time reconstruction infeasible at modern detector frame rates (kHz). Meanwhile,
APS-U will deliver 100--500x more coherent flux, pushing data rates to 10+ GB/s. The
gap between data acquisition speed and reconstruction throughput prevents real-time
feedback that could enable adaptive scanning (adjusting overlap, dwell time, or scan
path based on live-reconstructed images). This paper addresses the gap by deploying
neural-network-based phase retrieval on edge compute hardware (NVIDIA Jetson for GPU,
Xilinx Alveo for FPGA) directly at the beamline.

---

## Method

1. **Neural network architecture**: A compact encoder-decoder CNN (~2M parameters)
   takes a single diffraction pattern as input and outputs the complex-valued
   object transmission function. The architecture uses residual blocks and is
   trained to approximate the iterative reconstruction operator.
2. **Training data**: Simulated diffraction patterns generated from a physics-based
   forward model with realistic noise, probe diversity, and partial coherence.
   Transfer learning is applied using a small amount of experimental data for
   domain adaptation.
3. **Edge GPU deployment**: The trained model is quantized (FP16) and compiled with
   TensorRT for NVIDIA Jetson AGX Orin. Inference achieves ~0.5 ms per diffraction
   pattern.
4. **FPGA deployment**: The model is translated to a fixed-point (INT8) representation
   and synthesized onto a Xilinx Alveo U250 FPGA using Vitis AI. Inference achieves
   ~0.3 ms per pattern with lower power consumption.
5. **Streaming pipeline**: Data flows directly from the detector (Eiger) via RDMA
   to the edge device, bypassing the control workstation. Reconstructed phase images
   are streamed to a visualization dashboard with < 5 ms total latency.
6. **Iterative refinement**: The CNN output can optionally seed a small number of
   ePIE iterations (5--10 instead of 200+) for higher fidelity when time permits.

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
| Demonstrated resolution                   | ~15 nm half-pitch on a Siemens star test pattern    |

---

## Data & Code Availability

| Resource       | Link / Note                                                           |
|----------------|-----------------------------------------------------------------------|
| **Code**       | Partial release (inference scripts and model weights on GitHub)        |
| **Data**       | Simulated training sets available; experimental data on request        |
| **License**    | Not explicitly stated in paper                                        |
| **Reproducibility Score** | **3 / 5** -- Inference code and pretrained weights available; FPGA synthesis scripts and full training pipeline not released. |

---

## Strengths

- Demonstrates real-time ptychographic reconstruction at kHz rates, which is a
  prerequisite for adaptive scanning at APS-U data rates.
- Edge deployment eliminates network bottleneck: data never leaves the beamline
  hutch for initial reconstruction, dramatically reducing latency.
- FPGA implementation achieves competitive throughput at 6--7x lower power than
  GPU servers, important for sustainable facility operations.
- Hybrid CNN + few-iteration ePIE approach provides tunable quality/speed tradeoff
  suited to different experimental priorities.
- Validated on experimental ptychographic data, not just simulations.

---

## Limitations & Gaps

- The compact CNN trades reconstruction quality for speed; final resolution is
  ~15 nm vs. ~10 nm achievable with full iterative reconstruction on the same data.
- Transfer learning from simulation to experiment requires beamline-specific
  calibration data; generalization across beamlines is not demonstrated.
- FPGA INT8 quantization introduces quantization noise; systematic characterization
  of quantization-induced artifacts in phase images is limited.
- The streaming pipeline is demonstrated for 2D ptychography; extension to 3D
  ptychographic tomography (ptycho-tomo) is discussed but not implemented.
- Full training pipeline and FPGA synthesis toolchain are not open-sourced,
  limiting community reproducibility for FPGA deployment.

---

## Relevance to eBERlight

This paper is a cornerstone reference for eBERlight's real-time feedback architecture:

- **Latency requirements**: The < 5 ms end-to-end latency establishes the benchmark
  for eBERlight's real-time decision-making loop at coherent imaging beamlines.
- **Edge compute strategy**: eBERlight can adopt the same Jetson Orin / FPGA edge
  deployment model for other modalities (XRF spectral fitting, XANES classification)
  where millisecond-scale inference is needed.
- **Adaptive ptychography**: With real-time phase images available, eBERlight's
  scan planner can adjust overlap ratios, skip redundant positions, or zoom into
  features of interest mid-scan.
- **APS-U readiness**: The 2--3 kHz throughput is well-matched to expected APS-U
  ptychography frame rates at 26-ID and 2-ID-D.

---

## Actionable Takeaways

1. **Procure edge hardware**: Acquire NVIDIA Jetson Orin modules and Xilinx Alveo
   boards for eBERlight beamline hutches at APS-U ptychography endstations.
2. **Retrain on APS-U probe**: Generate APS-U-specific training data using measured
   probe functions and coherence properties from commissioning.
3. **Streaming integration**: Integrate the detector-to-edge RDMA pipeline with
   eBERlight's Bluesky/Tiled data infrastructure for seamless live reconstruction.
4. **Extend to ptycho-tomo**: Develop a 3D extension that reconstructs ptychographic
   tomography volumes in streaming mode by accumulating CNN-reconstructed projections.
5. **Quality monitoring**: Implement online NRMSE tracking that triggers fallback
   to full iterative reconstruction when the CNN output quality degrades below a
   user-defined threshold.

---

*Reviewed for the eBERlight Research Archive, 2026-02-27.*
