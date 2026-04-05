# Tike

## Overview

Tike is a GPU-accelerated ptychographic reconstruction toolkit developed at the
Advanced Photon Source (APS), Argonne National Laboratory. It provides
high-performance implementations of multiple ptychographic reconstruction
algorithms with near real-time capability for streaming ptychography at modern
synchrotron beamlines.

## Key Features

- **GPU-accelerated reconstruction** -- all core algorithms execute on NVIDIA
  GPUs via CuPy/CUDA, enabling near real-time ptychographic reconstruction.
- **Multiple reconstruction algorithms** -- supports ePIE (extended
  Ptychographic Iterative Engine), DM (Difference Map), and LSQ-ML
  (Least-Squares Maximum Likelihood) solvers.
- **Streaming capability** -- designed for near real-time reconstruction of
  ptychographic data as it is acquired, supporting APS-U data rates.
- **Modular design** -- clean separation between forward models, solvers, and
  I/O for extensibility and testing.
- **Multi-GPU support** -- distributes reconstruction across multiple GPUs for
  large datasets.

## Algorithms Supported

| Algorithm | Description |
|-----------|-------------|
| ePIE | Extended Ptychographic Iterative Engine |
| DM | Difference Map solver |
| LSQ-ML | Least-Squares Maximum Likelihood |
| Position correction | Joint probe position and object refinement |
| Probe recovery | Simultaneous probe and object reconstruction |

## Typical Performance

GPU-accelerated solvers on NVIDIA A100 hardware enable near real-time
ptychographic reconstruction of standard dataset sizes, with throughput
compatible with APS-U streaming data rates.

## Repository

- GitHub: <https://github.com/AdvancedPhotonSource/tike>
- License: BSD-3-Clause
- Language: Python/CUDA
- Primary maintainer: Advanced Photon Source, Argonne National Laboratory

## Related Documents

| Document | Description |
|----------|-------------|
| Ptychography notes | See synchrotron technique documentation |
