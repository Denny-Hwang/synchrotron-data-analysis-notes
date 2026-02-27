# Data Processing Pipeline

## Overview

The processing stage transforms raw detector frames into scientifically
meaningful reconstructions, denoised volumes, and segmented structures. Each
step is GPU-accelerated where possible, targeting near-real-time turnaround
on ALCF resources (Polaris, Aurora).

## Pipeline Stages

```
Raw Frames
    |
    v
1. Preprocessing (dark / flat correction, normalization)
    |
    v
2. Reconstruction (TomocuPy filtered back-projection / iterative)
    |
    v
3. Denoising (DNN-based noise removal)
    |
    v
4. Segmentation (U-Net semantic labeling)
    |
    v
5. Quantification (morphometric measurements)
    |
    v
Processed Dataset (HDF5 / Zarr)
```

## Step 1: Preprocessing

### Dark-Field Correction

Dark-field images (detector readout with no beam) capture electronic noise and
hot-pixel artifacts. Correction subtracts the averaged dark frame:

```
corrected = raw - mean(dark_frames)
```

Typically 20-50 dark frames are collected and averaged before each scan.

### Flat-Field Normalization

Flat-field images (open beam, no sample) capture beam intensity profile and
detector gain variations:

```
normalized = (raw - dark) / (flat - dark)
```

Flat frames are acquired at regular angular intervals during long scans to
account for beam intensity drift.

### Additional Preprocessing

| Operation | Purpose | Implementation |
|---|---|---|
| Zingers removal | Remove single-pixel spikes from cosmic rays | Median filter on projection stack |
| Ring artifact suppression | Correct stripe artifacts from bad pixels | Fourier-Wavelet method (Munch et al.) |
| Phase retrieval | Extract phase contrast from propagation | Paganin single-distance algorithm |
| Rotation center finding | Align rotation axis for reconstruction | Fourier-based cross-correlation |
| Log transform | Convert transmission to absorption | `-log(normalized)` |

### Implementation

```python
import tomopy

proj, flat, dark, theta = dxchange.read_aps_32id(filename)
proj = tomopy.normalize(proj, flat, dark)
proj = tomopy.minus_log(proj)
proj = tomopy.remove_stripe_fw(proj, level=7, wname='sym16')
rot_center = tomopy.find_center_vo(proj)
```

## Step 2: Reconstruction

### TomocuPy (GPU-Accelerated)

TomocuPy is the primary reconstruction engine, providing CuPy-based GPU
implementations of standard tomographic algorithms.

| Algorithm | Use Case | Speed (4K x 4K x 2K) |
|---|---|---|
| FBP (Filtered Back-Projection) | Standard reconstruction | ~8 seconds (A100) |
| Gridrec | Fourier-based gridding | ~12 seconds (A100) |
| SIRT (iterative) | Noisy / limited-angle data | ~2 minutes (20 iter) |
| CGLS (iterative) | Regularized reconstruction | ~3 minutes (30 iter) |

### Configuration

```yaml
reconstruction:
  algorithm: "fourierrec"        # or "fbp", "sirt", "cgls"
  filter: "shepp"                # Ram-Lak, Shepp-Logan, Butterworth
  num_iterations: 20             # for iterative methods only
  gpu_devices: [0, 1, 2, 3]     # multi-GPU distribution
  chunk_size: 64                 # slices per GPU batch
  dtype: "float32"
  output_format: "zarr"          # or "hdf5", "tiff_stack"
```

### Multi-GPU Strategy

Reconstruction workloads are distributed across GPUs by slice:

```
Volume (N slices)
    |
    +-- GPU 0: slices [0, N/4)
    +-- GPU 1: slices [N/4, N/2)
    +-- GPU 2: slices [N/2, 3N/4)
    +-- GPU 3: slices [3N/4, N)
    |
    v
Concatenated output volume
```

## Step 3: Denoising

### DNN-Based Denoising

Deep neural network models remove residual noise from reconstructed volumes
without degrading structural features.

| Model | Architecture | Training Data | PSNR Gain |
|---|---|---|---|
| Noise2Noise | U-Net variant | Paired noisy acquisitions | +4-6 dB |
| TomoGAN | GAN with perceptual loss | Simulated + real pairs | +5-8 dB |
| BM3D-Net | Block-matching + CNN | Self-supervised | +3-5 dB |

### Inference Pipeline

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

## Step 4: Segmentation

### U-Net Semantic Segmentation

A 3D U-Net segments the denoised volume into material phases (e.g., pore,
matrix, inclusion).

| Parameter | Value |
|---|---|
| Architecture | 3D U-Net (4 encoder/decoder levels) |
| Input patch size | 128 x 128 x 128 voxels |
| Overlap | 32 voxels (blended stitching) |
| Classes | Task-dependent (2-8 phases) |
| Loss function | Dice + cross-entropy |
| Inference time | ~3 minutes per 2048^3 volume (A100) |

### Post-Processing

- **Connected component labeling** -- Identifies individual particles/pores
- **Morphological operations** -- Erosion/dilation to clean boundaries
- **Watershed separation** -- Splits touching objects

## Step 5: Quantification

Quantitative metrics are extracted from segmented volumes:

| Metric | Description | Unit |
|---|---|---|
| Porosity | Volume fraction of pore phase | % |
| Pore size distribution | Equivalent sphere diameter histogram | um |
| Surface area | Marching-cubes mesh area | um^2 |
| Tortuosity | Path length / straight-line distance | dimensionless |
| Connectivity | Euler number of pore network | dimensionless |
| Orientation tensor | Fiber/grain alignment statistics | tensor |

Results are stored as JSON sidecar files alongside the HDF5 volumes.

## HPC Job Submission

Processing jobs are submitted to ALCF via PBS Pro:

```bash
qsub -A eBERlight_allocation \
     -q prod \
     -l select=4:ngpus=4 \
     -l walltime=01:00:00 \
     -l filesystems=eagle \
     -- /path/to/pipeline_runner.sh $SCAN_ID
```

The `pipeline_runner.sh` script orchestrates all five steps sequentially,
writing intermediate outputs to fast NVMe scratch and final results to Eagle.

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
