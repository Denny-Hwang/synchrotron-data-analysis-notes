# TomocuPy -- Setup and Benchmark Reproduction Guide

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| NVIDIA GPU | Pascal (sm_60) | Ampere A100 |
| CUDA Toolkit | 11.2 | 12.x |
| GPU RAM | 16 GB | 40 GB |
| Python | 3.9 | 3.10+ |
| OS | Linux (x86_64) | Rocky Linux 8+ |

## Installation

### 1. Create a Conda Environment

```bash
conda create -n tomocupy python=3.10 -y
conda activate tomocupy
```

### 2. Install CuPy (match your CUDA version)

```bash
# For CUDA 12.x
pip install cupy-cuda12x

# For CUDA 11.x
pip install cupy-cuda11x
```

### 3. Install TomocuPy

```bash
git clone https://github.com/tomography/tomocupy.git
cd tomocupy
pip install -e .
```

### 4. Verify Installation

```bash
python -c "import tomocupy; print(tomocupy.__version__)"
```

## Downloading Test Data

APS provides reference datasets in HDF5 `exchange` format.

```bash
# Small test phantom (256 x 256 x 128)
wget https://tomobank.readthedocs.io/en/latest/_downloads/phantom_00001.h5

# Medium dataset (2048 x 2048 x 1500) -- ~4 GB
wget https://tomobank.readthedocs.io/en/latest/_downloads/tomo_00001.h5
```

Alternatively, generate a Shepp-Logan phantom:

```python
import tomopy
obj = tomopy.shepp3d(size=512)
ang = tomopy.angles(720)
proj = tomopy.project(obj, ang)
tomopy.write_hdf5(proj, fname="shepp_512.h5")
```

## Running a Reconstruction

```bash
tomocupy recon \
    --file-name tomo_00001.h5 \
    --rotation-axis 1024 \
    --reconstruction-type full \
    --nsino-per-chunk 4 \
    --output-dir ./recon_output
```

Key flags:

| Flag | Purpose |
|------|---------|
| `--rotation-axis` | Center of rotation in pixels |
| `--nsino-per-chunk` | Sinograms processed per GPU batch |
| `--retrieve-phase` | Enable Paganin phase retrieval |
| `--remove-stripe` | Enable ring/stripe removal |

## Benchmark Reproduction

TomocuPy ships a built-in benchmarking command.

```bash
tomocupy bench \
    --file-name tomo_00001.h5 \
    --rotation-axis 1024 \
    --nruns 5
```

This will:

1. Warm up the GPU with one untimed run.
2. Execute `--nruns` timed reconstructions.
3. Print mean and standard deviation of wall-clock time.

### Expected Results (A100 40 GB)

| Dataset | Mean time | Std |
|---------|-----------|-----|
| 2048 x 2048 x 1500 | 25.3 s | 0.4 s |
| 4096 x 4096 x 3000 | 212 s | 3.1 s |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `CUDADriverError` | Ensure `nvidia-smi` works and CUDA toolkit version matches CuPy |
| Out of GPU memory | Reduce `--nsino-per-chunk` |
| Slow I/O | Use SSD or NVMe for HDF5 storage; enable `--use-pinned-memory` |
| Incorrect reconstruction | Double-check `--rotation-axis` value |
