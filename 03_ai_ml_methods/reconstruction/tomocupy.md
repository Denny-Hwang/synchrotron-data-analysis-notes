# TomocuPy: GPU-Accelerated Tomographic Reconstruction

## Overview

**TomocuPy** is a GPU-accelerated tomographic reconstruction package developed at APS
(Argonne National Laboratory). It reimplements core TomoPy algorithms using **CuPy**
(GPU-accelerated NumPy) to achieve 20-30× speedup over CPU-based TomoPy.

**Repository**: [https://github.com/nikitinvv/tomocupy](https://github.com/nikitinvv/tomocupy)

## Architecture

### Design Philosophy
```
TomoPy (CPU)                    TomocuPy (GPU)
─────────────                   ──────────────
NumPy arrays         →          CuPy arrays
CPU computation      →          CUDA kernels
Sequential slices    →          Parallel slices
Disk I/O bottleneck  →          Streaming pipeline
```

### Key Components

1. **CuPy Kernels**: Core mathematical operations (FFT, back projection) on GPU
2. **Streaming Pipeline**: Overlap data transfer and computation
3. **Chunk Processing**: Handle volumes larger than GPU memory
4. **Wrapper Functions**: TomoPy-compatible API

### Streaming Pipeline Architecture

```
CPU Memory                  GPU Memory               CPU Memory
[Chunk 1] ──Transfer──→ [Process] ──Transfer──→ [Result 1]
[Chunk 2] ──Transfer──→ [Process] ──Transfer──→ [Result 2]
           ↑              ↑                        ↑
           └──────────────┘────────────────────────┘
           Overlapped (concurrent transfer + compute)
```

## Supported Algorithms

| Algorithm | Type | Use Case |
|-----------|------|----------|
| **FBP** | Analytical | Standard reconstruction, fastest |
| **Gridrec** | Analytical (FFT) | Equivalent to FBP, slightly different implementation |
| **SIRT** | Iterative | Better quality from noisy/sparse data |
| **Log-polar FBP** | Analytical | Alternative coordinate system |

### FBP on GPU

```python
# Simplified FBP pipeline on GPU (CuPy)
import cupy as cp
from cupyx.scipy.fft import fft, ifft, fftfreq

def fbp_gpu(sinogram, theta, center):
    """GPU-accelerated filtered back projection for one slice."""
    n_angles, n_det = sinogram.shape

    # 1. Move to GPU
    sino_gpu = cp.asarray(sinogram)

    # 2. Apply ramp filter in Fourier domain
    freq = fftfreq(n_det)
    ramp_filter = cp.abs(freq)
    sino_filtered = cp.real(ifft(fft(sino_gpu, axis=1) * ramp_filter[None, :], axis=1))

    # 3. Back projection
    recon = cp.zeros((n_det, n_det), dtype=cp.float32)
    for i, angle in enumerate(theta):
        # Rotate and accumulate (actual implementation uses custom CUDA kernel)
        recon += back_project_line(sino_filtered[i], angle, center)

    return cp.asnumpy(recon)
```

## Benchmarks

### Speed Comparison (2048×2048 sinograms, 1500 projections)

| Method | Platform | Time per slice | Speedup |
|--------|----------|---------------|---------|
| TomoPy gridrec | CPU (32 cores) | 0.5 s | 1× |
| TomoPy FBP | CPU (32 cores) | 2.0 s | 0.25× |
| **TomocuPy FBP** | **NVIDIA A100** | **0.02 s** | **25×** |
| **TomocuPy gridrec** | **NVIDIA A100** | **0.015 s** | **33×** |

### Full Volume Reconstruction

| Volume Size | TomoPy (CPU) | TomocuPy (GPU) | Speedup |
|------------|-------------|----------------|---------|
| 1024³ | 8 min | 15 s | 32× |
| 2048³ | 65 min | 2 min | 32× |
| 4096³ | 520 min | 20 min | 26× |

*Note: 4096³ requires chunk-based processing due to GPU memory limits.*

### GPU Memory Usage

| Volume | FBP Memory | Iterative (SIRT) |
|--------|-----------|------------------|
| 1024³ | ~4 GB | ~8 GB |
| 2048³ | ~16 GB | ~32 GB |
| 4096³ | Chunked | Chunked |

## Usage Example

```python
import tomocupy as tc
import h5py

# Load data
with h5py.File('scan.h5', 'r') as f:
    proj = f['/exchange/data'][:]
    flat = f['/exchange/data_white'][:]
    dark = f['/exchange/data_dark'][:]
    theta = f['/exchange/theta'][:] * np.pi / 180  # to radians

# Preprocessing (on GPU)
proj_norm = tc.normalize(proj, flat, dark)
proj_log = tc.minus_log(proj_norm)

# Find rotation center
center = tc.find_center(proj_log, theta)

# Reconstruct
recon = tc.recon(proj_log, theta, center=center, algorithm='fbp')
# recon shape: (Nslices, Ny, Nx)
```

## Limitations

1. **NVIDIA GPU required**: CuPy requires CUDA-capable NVIDIA GPU
2. **GPU memory**: Limited to GPU memory per chunk (16-80 GB depending on card)
3. **Geometry**: Primarily parallel-beam geometry (fan/cone beam limited)
4. **Algorithm selection**: Fewer algorithms than TomoPy (focused on most-used)
5. **Dependencies**: CUDA toolkit, CuPy installation can be complex

## Comparison with TomoPy

| Feature | TomoPy | TomocuPy |
|---------|--------|----------|
| **Speed** | Baseline (CPU) | 20-30× faster (GPU) |
| **Algorithms** | Many (FBP, gridrec, SIRT, MLEM, CGLS, ...) | Core subset (FBP, gridrec, SIRT) |
| **Hardware** | CPU only | NVIDIA GPU required |
| **Memory limit** | System RAM (~100s GB) | GPU RAM (16-80 GB) |
| **Large volumes** | Direct processing | Chunk-based streaming |
| **Installation** | Easy (pip) | Requires CUDA setup |
| **Real-time capable** | No | Yes (seconds per scan) |
| **Quality** | Reference implementation | Identical results |

*Detailed comparison: [05_tools_and_code/tomopy/pros_cons.md](../../05_tools_and_code/tomopy/pros_cons.md)*
