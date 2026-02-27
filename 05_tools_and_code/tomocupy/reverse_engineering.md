# TomocuPy -- Architecture and Reverse Engineering Notes

## Module Layout

```
tomocupy/
  cli/            # Command-line entry points
  conf.py         # Configuration and parameter defaults
  proc.py         # Top-level processing orchestrator
  rec.py          # Reconstruction loop (chunk scheduler)
  retrieve.py     # Phase retrieval (Paganin)
  remove.py       # Ring / stripe removal
  utils.py        # Padding, centering, logging helpers
  kernels/        # Raw CUDA / CuPy RawKernel sources (.cu)
  tests/
```

## CuPy Kernel Strategy

TomocuPy avoids `cupyx.scipy` where possible and instead ships hand-written
CUDA kernels loaded as `cupy.RawKernel` objects.  Key kernels:

| Kernel file | Purpose |
|-------------|---------|
| `fbp.cu` | Ramp-filtered back-projection (fan/parallel) |
| `log_polar.cu` | Log-polar interpolation for center correction |
| `ring.cu` | Fourier-space ring artifact suppression |

Each kernel is compiled once at import time and cached by CuPy's JIT cache.

## Streaming Pipeline

Reconstruction is organised as a three-stage pipeline:

1. **H2D** -- read a chunk of sinograms from HDF5 into pinned host memory,
   then asynchronously copy to a GPU staging buffer.
2. **Compute** -- apply preprocessing (flat/dark correction, phase retrieval,
   ring removal) and FBP on the GPU.
3. **D2H** -- copy the reconstructed slice chunk back to host memory and write
   to the output HDF5 file.

Stages overlap via two CUDA streams, so the next chunk's H2D transfer
runs concurrently with the current chunk's compute.

### Chunk Sizing

Chunk size is auto-tuned to fill ~60 % of free GPU memory.  The remaining 40 %
is reserved for intermediate FFT buffers and the ramp filter.

## Memory Management

- All GPU arrays are pre-allocated once during `proc.init()`.
- `cupy.get_default_memory_pool().free_all_blocks()` is called between
  reconstruction batches to prevent fragmentation.
- Pinned host memory (`cupy.cuda.alloc_pinned_memory`) is used for all HDF5
  read buffers to enable asynchronous DMA transfers.

## Benchmark Reference Data

| Dataset size | GPU | Wall time | Notes |
|-------------|-----|-----------|-------|
| 2048 x 2048 x 1500 | A100 40 GB | 25 s | FBP gridrec |
| 2048 x 2048 x 1500 | V100 32 GB | 42 s | FBP gridrec |
| 4096 x 4096 x 3000 | A100 40 GB | 210 s | FBP gridrec, chunked |

All timings include I/O.  Measurements performed with `tomocupy bench` CLI.

## Key Design Decisions

1. **Single-GPU only** -- multi-GPU or distributed reconstruction is out of
   scope; the philosophy is "one GPU per beamline workstation."
2. **Minimal dependencies** -- only CuPy, NumPy, h5py, and scipy are required.
3. **No iterative solvers** -- TomocuPy focuses on analytical (FBP) methods.
   For iterative reconstruction, users are directed to ASTRA Toolbox or TomoPy
   with GPU acceleration.
