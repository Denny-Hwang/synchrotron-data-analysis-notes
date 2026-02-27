# TomoPy

## Overview

TomoPy is the standard open-source Python library for tomographic data
processing and image reconstruction, developed and maintained at the Advanced
Photon Source (APS), Argonne National Laboratory. It provides a comprehensive
toolkit covering the full pipeline from raw projection data to reconstructed
3-D volumes.

## Key Features

- **Multi-algorithm support** -- analytical (gridrec, FBP) and iterative
  (ART, SIRT, MLEM, OSEM, TV-regularised) reconstruction methods.
- **Pre-processing suite** -- flat/dark correction, phase retrieval (Paganin,
  Bronnikov), ring removal, stripe correction, zinger removal.
- **Plugin back-ends** -- optional integration with ASTRA Toolbox (GPU) and
  UFO for hardware acceleration.
- **HDF5 Data Exchange I/O** -- native support for the APS Data Exchange
  format (`/exchange/data`, `/exchange/data_white`, `/exchange/data_dark`).
- **Parallel execution** -- multi-threaded C extensions via OpenMP for CPU
  parallelism.

## Typical Use Case

```python
import tomopy

# Read data
proj, flat, dark, theta = tomopy.read_aps_32id("data.h5")

# Preprocessing
proj = tomopy.normalize(proj, flat, dark)
proj = tomopy.minus_log(proj)
rot_center = tomopy.find_center(proj, theta)

# Reconstruction
recon = tomopy.recon(proj, theta, center=rot_center, algorithm="gridrec")

# Write output
tomopy.write_tiff_stack(recon, fname="recon/slice")
```

## Repository

- GitHub: <https://github.com/tomography/tomopy>
- Documentation: <https://tomopy.readthedocs.io>
- License: BSD-3-Clause

## Related Documents

| Document | Description |
|----------|-------------|
| [reverse_engineering.md](reverse_engineering.md) | Module structure and algorithm notes |
| [pros_cons.md](pros_cons.md) | Comparison with TomocuPy |
