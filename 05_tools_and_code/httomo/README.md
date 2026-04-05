# HTTomo

## Overview

HTTomo (High Throughput Tomography) is a GPU-accelerated, modular tomography
processing pipeline developed at Diamond Light Source. It provides a YAML-based
workflow configuration system that chains preprocessing, reconstruction, and
post-processing steps through a plugin architecture, enabling near real-time
processing of TB-scale synchrotron tomography datasets.

## Key Features

- **YAML-based workflow configuration** -- declarative pipeline definition that
  is human-readable, version-controllable, and inherently reproducible.
- **Modular plugin architecture** -- self-contained plugins for preprocessing,
  reconstruction, and post-processing with standardized interfaces. New
  algorithms can be added without modifying the core framework.
- **GPU-accelerated backends** -- leverages CuPy and custom CUDA kernels for
  high-throughput processing compatible with modern detector data rates.
- **Chunk-based processing** -- large datasets are divided into memory-efficient
  chunks, enabling processing of arbitrarily large volumes without
  out-of-memory errors.
- **MPI parallelism** -- distributed processing across multiple GPUs and nodes
  for cluster deployment.
- **HDF5/NeXus I/O** -- reads and writes standard synchrotron data formats.
- **Integration with TomoPy and CuPy backends** -- leverages established
  reconstruction libraries as backends for core algorithms.

## Algorithms Supported

| Algorithm | Description |
|-----------|-------------|
| Normalization | Flat/dark-field correction with outlier handling |
| Ring removal | Multiple ring artifact suppression methods |
| Phase retrieval | Paganin and transport-of-intensity methods |
| Center finding | Automated rotation center detection (Vo method) |
| FBP (CPU/GPU) | Filtered back-projection via TomoPy/CuPy backends |
| Iterative recon | SIRT, CGLS via backend libraries |
| Segmentation | Post-reconstruction segmentation plugins |

## Example Configuration

```yaml
- method: normalize
  module_path: httomo.methods.preprocessing
  parameters:
    cutoff: 10

- method: find_center_vo
  module_path: httomo.methods.misc

- method: recon
  module_path: httomo.methods.reconstruction
  parameters:
    algorithm: FBP_CUDA
    center: auto
```

## Typical Performance

GPU-accelerated processing enables near real-time throughput on standard
synchrotron tomography datasets. Chunk-based memory management allows
processing of TB-scale datasets on hardware with limited GPU memory.

## Repository

- GitHub: <https://github.com/DiamondLightSource/httomo>
- License: Apache-2.0
- Language: Python/CUDA
- Primary maintainer: Diamond Light Source

## Related Documents

| Document | Description |
|----------|-------------|
| [review_httomo_2024.md](../../04_publications/ai_ml_synchrotron/review_httomo_2024.md) | Publication review |
