# Data Processing Pipeline

## Overview

The processing stage transforms raw detector frames into scientifically
meaningful reconstructions, denoised volumes, and segmented structures. Each
step is GPU-accelerated where possible, targeting near-real-time turnaround
on ALCF resources (Polaris, Aurora).

## Pipeline Stages

```
Raw Frames --> Preprocessing --> Reconstruction --> Denoising --> Segmentation --> Quantification
```

## Step 1: Preprocessing

Dark-field correction subtracts electronic noise; flat-field normalization
removes beam profile and gain variations:

```
normalized = (raw - dark) / (flat - dark)
```

Additional preprocessing operations:

| Operation | Purpose | Implementation |
|---|---|---|
| Zingers removal | Remove cosmic ray spikes | Median filter on projection stack |
| Ring artifact suppression | Correct stripe artifacts | Fourier-Wavelet method |
| Phase retrieval | Extract phase contrast | Paganin single-distance algorithm |
| Rotation center finding | Align rotation axis | Fourier-based cross-correlation |
| Log transform | Transmission to absorption | `-log(normalized)` |

```python
import tomopy
proj, flat, dark, theta = dxchange.read_aps_32id(filename)
proj = tomopy.normalize(proj, flat, dark)
proj = tomopy.minus_log(proj)
proj = tomopy.remove_stripe_fw(proj, level=7, wname='sym16')
rot_center = tomopy.find_center_vo(proj)
```

## Step 2: Reconstruction (TomocuPy)

TomocuPy provides CuPy-based GPU implementations of tomographic algorithms:

| Algorithm | Use Case | Speed (4K x 4K x 2K) |
|---|---|---|
| FBP (Filtered Back-Projection) | Standard reconstruction | ~8 s (A100) |
| Gridrec | Fourier-based gridding | ~12 s (A100) |
| SIRT (iterative) | Noisy / limited-angle data | ~2 min (20 iter) |
| CGLS (iterative) | Regularized reconstruction | ~3 min (30 iter) |

### Multi-GPU Strategy

Workloads are distributed across GPUs by slice. Each GPU processes N/4 slices
of the volume in parallel, with results concatenated into the final output.

```yaml
reconstruction:
  algorithm: "fourierrec"
  filter: "shepp"
  gpu_devices: [0, 1, 2, 3]
  chunk_size: 64
  dtype: "float32"
  output_format: "zarr"
```

## Step 3: Denoising (DNN)

Deep neural network models remove residual noise from reconstructed volumes:

| Model | Architecture | PSNR Gain |
|---|---|---|
| Noise2Noise | U-Net variant | +4-6 dB |
| TomoGAN | GAN with perceptual loss | +5-8 dB |
| BM3D-Net | Block-matching + CNN | +3-5 dB |

```python
import torch
model = torch.load("models/tomogan_v3.pt")
model.eval()
for slab in volume.iter_slabs(size=64):
    slab_gpu = torch.from_numpy(slab).cuda().unsqueeze(0)
    with torch.no_grad():
        denoised = model(slab_gpu)
    output.write_slab(denoised.cpu().numpy())
```

Inference on a 2048^3 volume takes approximately 90 seconds on 4x A100 GPUs.

## Step 4: Segmentation (U-Net)

A 3D U-Net segments denoised volumes into material phases (pore, matrix, etc.):

| Parameter | Value |
|---|---|
| Architecture | 3D U-Net (4 encoder/decoder levels) |
| Input patch size | 128 x 128 x 128 voxels |
| Overlap | 32 voxels (blended stitching) |
| Loss function | Dice + cross-entropy |
| Inference time | ~3 min per 2048^3 volume (A100) |

Post-processing includes connected component labeling, morphological
erosion/dilation, and watershed separation of touching objects.

## Step 5: Quantification

Metrics extracted from segmented volumes:

| Metric | Description | Unit |
|---|---|---|
| Porosity | Volume fraction of pore phase | % |
| Pore size distribution | Equivalent sphere diameter histogram | um |
| Surface area | Marching-cubes mesh area | um^2 |
| Tortuosity | Path length / straight-line distance | dimensionless |
| Connectivity | Euler number of pore network | dimensionless |

Results are stored as JSON sidecar files alongside the HDF5 volumes.

## HPC Job Submission

```bash
qsub -A eBERlight_allocation \
     -q prod \
     -l select=4:ngpus=4 \
     -l walltime=01:00:00 \
     -l filesystems=eagle \
     -- /path/to/pipeline_runner.sh $SCAN_ID
```

The `pipeline_runner.sh` script orchestrates all five steps, writing
intermediate outputs to NVMe scratch and final results to Eagle.

## Output Formats

| Stage | Format | Compression | Typical Size |
|---|---|---|---|
| Preprocessed projections | HDF5 | LZ4 | 20-50 GB |
| Reconstructed volume | Zarr | Blosc-zstd | 30-80 GB |
| Denoised volume | Zarr | Blosc-zstd | 30-80 GB |
| Segmentation labels | HDF5 | gzip | 5-15 GB |
| Quantification | JSON | none | < 1 MB |

Processed data flows into the analysis stage described in
[analysis.md](analysis.md).
