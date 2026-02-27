# TomocuPy

## Overview

TomocuPy is a GPU-accelerated tomographic reconstruction package developed at
the Advanced Photon Source (APS), Argonne National Laboratory. It reimplements
the core algorithms found in TomoPy using CuPy, enabling near-real-time
reconstruction of synchrotron micro-CT data on a single NVIDIA GPU.

## Key Features

- **CuPy-based kernels** -- all filtering, back-projection, and ring-removal
  operations execute entirely on the GPU via CuPy, avoiding repeated
  host-device transfers.
- **Streaming chunk pipeline** -- sinograms are processed in overlapping chunks
  so that GPU compute and host-to-device I/O overlap, keeping the GPU
  saturated.
- **Drop-in CLI** -- a command-line interface compatible with the Data
  Management System at APS sector 2-BM and 32-ID beamlines.
- **HDF5 I/O** -- reads and writes the same `exchange` HDF5 format used by
  TomoPy and the APS Data Exchange specification.

## Algorithms Supported

| Algorithm | Description |
|-----------|-------------|
| FBP (gridrec) | Fourier-based filtered back-projection |
| Log-polar FBP | Rotation-center-tolerant variant |
| Phase retrieval | Paganin single-distance method |
| Ring removal | Combined Fourier + Titarenko approach |
| Stripe removal | Wavelet-based vertical stripe correction |

## Typical Performance

On an NVIDIA A100 (40 GB), TomocuPy reconstructs a 2048 x 2048 x 1500
dataset in approximately 25 seconds, compared to several minutes for the
equivalent TomoPy gridrec run on a 64-core CPU node.

## Repository

- GitHub: <https://github.com/tomography/tomocupy>
- License: BSD-3-Clause
- Primary maintainer: Viktor Nikitin (APS)

## Related Documents

| Document | Description |
|----------|-------------|
| [reverse_engineering.md](reverse_engineering.md) | Architecture deep dive |
| [pros_cons.md](pros_cons.md) | Comparison with TomoPy |
| [reproduction_guide.md](reproduction_guide.md) | Setup and benchmarking |
