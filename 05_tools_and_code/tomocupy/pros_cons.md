# TomocuPy vs TomoPy -- Comparison

## Performance Benchmarks

Reconstruction of a 2048 x 2048 x 1500 sinogram dataset:

| Metric | TomocuPy (A100) | TomoPy (64-core CPU) |
|--------|-----------------|----------------------|
| FBP wall time | ~25 s | ~180 s |
| Phase retrieval | ~3 s | ~25 s |
| Ring removal | ~2 s | ~15 s |
| Peak memory (device) | ~18 GB VRAM | ~32 GB RAM |
| I/O overlap | yes (async DMA) | limited (threading) |

## Pros of TomocuPy

- **Speed** -- 5-10x faster than TomoPy for analytical reconstruction methods,
  enabling near-real-time feedback at the beamline.
- **Streaming I/O** -- overlapping compute and data transfer keeps the GPU busy
  and hides I/O latency.
- **Small code base** -- easier to audit and modify than TomoPy's larger C
  extension modules.
- **CLI compatibility** -- integrates directly with APS beamline data
  management workflows.
- **Reproducible benchmarks** -- ships a `bench` subcommand for standardised
  timing.

## Cons of TomocuPy

- **GPU required** -- cannot fall back to CPU; requires NVIDIA hardware and a
  working CUDA toolkit.
- **Single GPU only** -- cannot distribute across multiple GPUs for very large
  datasets.
- **Limited algorithm set** -- only analytical (FBP) methods; no iterative
  solvers (SIRT, CGLS, MLEM).
- **Smaller community** -- fewer contributors and less third-party
  documentation than TomoPy.
- **CuPy dependency** -- CuPy installation can be non-trivial on systems
  without a well-configured CUDA environment.

## Pros of TomoPy (for comparison)

- Mature, well-documented, large user community.
- Supports both analytical and iterative algorithms.
- CPU-only -- runs on any machine without GPU.
- Plugin architecture allows ASTRA and other back-ends.

## Cons of TomoPy (for comparison)

- Slower for analytical methods on large datasets.
- C extensions can be difficult to build from source.
- No native GPU acceleration (requires external plugins).

## Recommendation

Use TomocuPy when fast analytical reconstruction is the priority and an NVIDIA
GPU is available. Fall back to TomoPy for iterative reconstruction, CPU-only
environments, or when broader algorithm selection is needed.
